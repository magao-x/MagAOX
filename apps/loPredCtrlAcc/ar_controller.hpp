/** \file ar_controller.hpp
 * \brief Modal DDSPC with operational compatibility and explicit offline causal histories.
 */

#ifndef PCARC_HPP
#define PCARC_HPP

#include <cstdint>
#include <Eigen/Dense>

#include "utils.hpp"
#include "recursive_least_squares.hpp"
#include "qrd_rls.hpp"

namespace DDSPC
{

/// Select the existing app convention or the explicitly synchronized offline convention.
enum class HistoryMode
{
    Legacy, ///< Preserve app indexing, startup, scaled regularization, and added feedback, including known defects.
    Causal  ///< Require complete indexed offline records and use the separate reference controller.
};

/// Return the full optimal gain mapping [y_p;u_p;a_p] to newest-first future commands.
/** Solves (C^T C + lambda I) K = -C^T [A B D], without integration or additional feedback/noise.
 * All blocks share a positive output row count; B and D may have zero columns.
 * A strictly positive lambda defines the zero/rank-deficient C case as well.
 * \throws std::invalid_argument for invalid dimensions, nonfinite inputs, or nonpositive lambda.
 * \throws std::runtime_error if the regularized solve fails or returns nonfinite values.
 */
Matrix reference_controller( const Matrix &A /**< [in] Past-WFS prediction block. */,
                             const Matrix &B /**< [in] Past-command prediction block. */,
                             const Matrix &D /**< [in] Past-accelerometer prediction block. */,
                             const Matrix &C /**< [in] Future-command response block. */,
                             realT         lambda /**< [in] Literal command penalty in the reference cost. */ );

/// Coupled modal predictor retaining the existing internal column order [C B D A].
/** For m modes, N WFS history steps, F future steps, and a accelerometer features,
 * Theta is F*m by (F+2*N-1)*m+a. B uses N-1 past command blocks; A uses N WFS blocks.
 * All blocks stack newest time first, then mode/channel within each time.
 *
 * Legacy remains the default app compatibility path, including its known indexing defects.
 * Causal is offline only: append_sample() records y_i, a_i, and actual synthetic u_i held for [i,i+1).
 * The nominal one-step plant produces y_(i+1) from u_i. At t=i+F the matured example uses
 * y_f=[y_t,...,y_(i+1)], u_f=[u_(t-1),...,u_i], and the state available at i.
 * Causal mode never appends calculated proposals implicitly. This is not instrument command readback.
 */
class PredictiveController
{
  private:
    /// Owned conventional learner; the existing two-learner allocation is retained.
    RecursiveLeastSquares *m_rls;

    /// Owned QR information-factor learner.
    QRDRecursiveLeastSquares *m_qrdRls;

    /// Power-of-two sample-ring capacity.
    std::uint64_t m_bufferSize;

    /// WFS record count since reset.
    std::uint64_t m_measurementHead{ 0 };

    /// Command record count since reset.
    std::uint64_t m_commandHead{ 0 };

    /// Accelerometer record count since reset.
    std::uint64_t m_accelHead{ 0 };

    /// Last causal record count learned; prevents duplicate updates.
    std::uint64_t m_lastTrainingHead{ 0 };

    /// WFS samples by time row and mode column.
    Matrix m_measurementBuffer;

    /// Command samples by time row and mode column.
    Matrix m_commandBuffer;

    /// Synchronized accelerometer features by time row and channel column.
    Matrix m_accelBuffer;

    /// Active legacy penalty, pointing to one of the two owned buffers.
    Matrix *m_regularizationMatrix;

    /// Penalty buffer selected at the next switch.
    bool m_useRegularizationMatrix01{ true };

    /// Pending legacy penalty switch.
    bool m_doSwitchRegularizationMatrix{ false };

    /// First legacy diagonal penalty buffer.
    Matrix m_regularizationMatrix01;

    /// Second legacy diagonal penalty buffer.
    Matrix m_regularizationMatrix02;

    /// Legacy next-command gain on [u_p;a_p;y_p].
    Matrix m_controller;

    /// Legacy extra gain on the latest residual, separate from the app's output accumulation.
    Matrix m_integrator;

    /// Residual/command mode count.
    int m_numModes;

    /// Future WFS/control horizon.
    int m_numFuture;

    /// WFS history length; the command state is one block shorter.
    int m_numHistory;

    /// Channels per synchronized accelerometer record.
    int m_numAccelChannels;

    /// Accelerometer history length, independent of WFS history.
    int m_numAccelHistory;

    /// Legacy additional feedback gain.
    realT m_gain;

    /// Retained legacy clip setting; clipping is disabled in the existing path.
    realT m_deltaMax;

    /// Existing legacy metadata scalar; queued penalty changes use the buffers above.
    realT m_regularization;

    /// Target row count F*m.
    int m_numPredictors;

    /// Regression column count, including future commands.
    int m_numFeatures;

    /// Known-state length, excluding future commands.
    int m_numStateFeatures;

    /// Total accelerometer feature count.
    int m_numAccelFeatures;

    /// Future-command count F*m.
    int m_numCorrelations;

    /// Indexing/recording convention selected at construction.
    HistoryMode m_historyMode;

  public:
    /// Existing public learner selector retained for compatibility; changing a trained learner is unsupported.
    bool use_qrd{ false };

    /// Allocate a predictor; operational callers retain Legacy behavior unless explicitly changed.
    PredictiveController( int         num_actuators /**< [in] Residual/command mode count. */,
                          int         num_history /**< [in] WFS history length; command state length is one less. */,
                          int         num_future /**< [in] Prediction/control horizon. */,
                          realT       gain                   = 0.25 /**< [in] Legacy added WFS gain. */,
                          realT       gamma                  = 1.0 /**< [in] Per-update forgetting factor. */,
                          realT       initial_regularization = 0.015 /**< [in] Legacy relative command penalty. */,
                          realT       initial_covariance     = 1e5 /**< [in] Prior information rho; P0=I/rho. */,
                          int         num_accel_channels     = 2 /**< [in] Synchronized accelerometer channel count. */,
                          int         accel_history          = 20 /**< [in] Accelerometer history length. */,
                          HistoryMode history_mode = HistoryMode::Legacy /**< [in] Explicit offline opt-in. */ );

    /// Release the owned learners.
    ~PredictiveController();

    /// Queue an update for the legacy scaled penalty.
    void set_regularization( realT new_regularization /**< [in] Legacy relative command penalty. */ );

    /// Return the selected learner's matrix in internal [C B D A] order.
    Matrix get_prediction_matrix();

    /// Return the same learned matrix in mathematical [A B D C] order.
    Matrix get_reference_prediction_matrix();

    /// Calculate the full reference gain without changing the operational controller.
    Matrix get_reference_controller( realT lambda /**< [in] Positive literal command penalty. */ );

    /// Reset weights, legacy gains, and histories.
    void reset();

    /// Reset histories and causal update tracking, retaining weights.
    void reset_buffers();

    /// Return the newest F WFS blocks as matured targets.
    Matrix get_measurement_future();

    /// Return N WFS blocks ending F steps before the newest record.
    Matrix get_measurement_past();

    /// Return F-skip_cmds command blocks, skipping that many rows in Causal mode.
    /** Legacy intentionally retains its original mode-count-dependent row skip. */
    Matrix get_command_future( int skip_cmds /**< [in] Number of current command records to exclude. */ );

    /// Return N command blocks ending F steps before the latest command; its first block belongs to C.
    Matrix get_command_past();

    /// Return the requested newest command blocks.
    Matrix get_current_command_past( int num_steps /**< [in] Number of command records. */ );

    /// Return the requested newest WFS blocks.
    Matrix get_current_measurement_past( int num_steps /**< [in] Number of WFS records. */ );

    /// Return channel-interleaved accelerometer history ending at the specified offset.
    Matrix get_accelerometer_past( int delay = 0 /**< [in] Steps before the latest accelerometer record. */ );

    /// Legacy-only append of an already synchronized accelerometer sample.
    void push_accelerometer_sample( const Matrix &new_acceleration /**< [in] Channel-count by one column. */ );

    /// Append one finite, contiguous offline record without calculating a command.
    /** Causal only. Invalid data throw before mutation. u_i is the synthetic applied input after y_i,
     * held over [i,i+1), not a candidate proposal. Accelerometers must already be synchronized;
     * no raw-stream ingestion or timestamp alignment is performed here.
     */
    void append_sample( std::uint64_t index /**< [in] Contiguous zero-based decision index; reset restarts at zero. */,
                        const Matrix &measurement /**< [in] Residual y_i, m by one. */,
                        const Matrix &applied_command /**< [in] Synthetic applied input u_i, m by one. */,
                        const Matrix &acceleration /**< [in] Features a_i available at i; empty when disabled. */ );

    /// Copy a matured [u_f;u_p;a_p;y_p] regressor and newest-first future-WFS target.
    /** Returns false until Causal histories are complete, leaving outputs unchanged.
     * Legacy retains its original zero padding and latest-accelerometer behavior.
     */
    bool get_training_example( Matrix &regressor /**< [out] Internal feature column. */,
                               Matrix &target /**< [out] Future-WFS column. */ );

    /// Return the latest Causal [y_p;u_p;a_p] state, excluding u_i from the current record.
    /** Throws until histories exist. This value snapshot cannot change when later records arrive. */
    Matrix get_prediction_state();

    /// Retained legacy declaration; this unused method has no existing definition.
    void get_current_past();

    /// Legacy-only calculation that also appends the WFS value and calculated command.
    Matrix calculate_command( Matrix new_measurement /**< [in] Current modal residual. */,
                              Matrix exploration_noise /**< [in] Additive exploration command. */ );

    /// Learn once per matured Causal record, or on every Legacy call.
    void update_system();

    /// Update the legacy scaled-penalty gain; Causal uses the separate reference law.
    void update_controller();

    /// Save the legacy artifact format; Causal artifacts are unsupported and rejected.
    void save_state( const std::string &filename /**< [in] Legacy artifact basename. */ );

    /// Restore the legacy artifact format; Causal artifacts are unsupported and rejected.
    void load_state( const std::string &filename /**< [in] Legacy artifact basename. */ );
};

} // namespace DDSPC

#endif // PCARC_HPP
