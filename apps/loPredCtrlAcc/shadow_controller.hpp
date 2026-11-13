/** \file shadow_controller.hpp
 * \brief Independent causal DDSPC model management with no command publication interface.
 */

#ifndef PCSHADOW_HPP
#define PCSHADOW_HPP

#include <memory>
#include <vector>

#include "ar_controller.hpp"

namespace DDSPC
{

/// Select an independently identified shadow model; neither option authorizes output.
enum class ShadowMode
{
    Standard,
    AccelAssisted
};

/// Describe why a shadow model cannot yet supply valid diagnostics.
enum class ShadowReason
{
    AwaitingRecord,
    InvalidData,
    Warming,
    Untrained,
    Unexcited,
    NumericalFault,
    Ready
};

/// Return a stable diagnostic description of a model state.
const char *shadow_reason( ShadowReason reason /**< [in] Readiness/fault reason. */ );

/// Configuration fixed for the lifetime of a pair of shadow models.
struct ShadowConfig
{
    /// Residual and applied-input mode count.
    int m_modes{ 1 };

    /// WFS history length; command state length is one less.
    int m_history{ 5 };

    /// Future residual/input horizon.
    int m_future{ 3 };

    /// Already synchronized assisted-model feature channels.
    int m_accelChannels{ 2 };

    /// Assisted history length in decision steps.
    int m_accelHistory{ 20 };

    /// Forgetting factor per actual RLS update.
    realT m_gamma{ 1 };

    /// Initial information strength, P0=I/prior.
    realT m_prior{ 1 };

    /// Literal reference-law command penalty.
    realT m_lambda{ 100 };

    /// Relative rank threshold after feature column normalization.
    realT m_rankTolerance{ 1e-5f };

    /// Minimum learned examples; zero uses each model's feature count.
    std::uint64_t m_minUpdates{ 0 };

    /// Select corrected QR RLS instead of conventional RLS for both independent learners.
    bool m_useQrd{ false };
};

/// One declared synchronized record; validity assertions must come from the caller's input contract.
/** u_i is the applied input for [i,i+1), matching Phase 1. Flags default false deliberately.
 * No timestamps or raw sensor payloads are decoded here. Synthetic tests explicitly label their valid records.
 */
struct ShadowRecord
{
    /// External decision identity; gaps break both histories.
    std::uint64_t m_index{ 0 };

    /// Observed y_i column.
    Matrix m_residual;

    /// Applied u_i column, never a model proposal.
    Matrix m_appliedCommand;

    /// Features a_i known at the decision origin.
    Matrix m_acceleration;

    /// Residual payload shape and values are established; timing is asserted separately.
    bool m_wfsValid{ false };

    /// Applied-command identity/semantics are established.
    bool m_appliedValid{ false };

    /// WFS/input timing satisfies the declared discrete-time contract.
    bool m_timingValid{ false };

    /// Accelerometer features are causally synchronized and fresh.
    bool m_accelValid{ false };
};

/// Snapshot of one predictor's status; readiness is for shadow diagnostics, not physical correction.
struct ShadowStatus
{
    /// Complete, sufficiently trained and excited, finite predictor/proposal.
    bool m_ready{ false };

    /// Requested adaptation state, independent of selection.
    bool m_trainingEnabled{ false };

    /// An RLS update was performed on the latest record.
    bool m_training{ false };

    /// Latest record met this model's input contract.
    bool m_dataValid{ false };

    /// A matured forecast is available in the current uninterrupted segment.
    bool m_predictionValid{ false };

    /// Contiguous records in the current segment.
    std::uint64_t m_samples{ 0 };

    /// Total RLS updates since model reset.
    std::uint64_t m_updates{ 0 };

    /// Scored frozen-origin forecasts in the current segment.
    std::uint64_t m_predictions{ 0 };

    /// Full regressor dimension, including future inputs.
    int m_features{ 0 };

    /// Rank of the current normalized feature window, without an RLS prior.
    int m_rank{ 0 };

    /// Conditional future-residual RMSE; meaningful only with predictionValid.
    double m_rmse{ 0 };

    /// Current readiness/fault description.
    ShadowReason m_reason{ ShadowReason::AwaitingRecord };
};

/// Two independently trained causal models with frozen-origin diagnostics and no output sink.
/** The caller serializes access. Selection never changes histories, training flags, or actuation state.
 * This class has no stream name, publisher, correction-enable operation, or device access.
 */
class ShadowController
{
    /// Offline-only worker copies owned numerical state into bounded snapshots; no application route is added.
    friend class OfflineTrainingWorker;

  private:
    /// Parameters and state frozen at an origin, scored only after its target horizon matures.
    struct Forecast
    {
        /// Canonical [A B D C] prediction matrix at the origin.
        Matrix m_theta;
        /// Canonical [y_p;u_p;a_p] state available at the origin.
        Matrix m_state;
        /// Origin was ready when this forecast was saved.
        bool m_valid{ false };
    };

    /// Owned state of a single independent identification problem.
    struct Model
    {
        /// Solely owned causal learner and histories.
        std::unique_ptr<PredictiveController> m_controller;
        /// Fixed scratch belongs only to this causal learner.
        std::unique_ptr<CausalWorkspace> m_workspace;
        /// Public status snapshot.
        ShadowStatus m_status;
        /// Last p regressor rows used to assess excitation independently of the prior.
        Matrix m_features;
        /// Diagnostic next input, never appended or published.
        Matrix m_proposal;
        /// Fixed forecast scoring and normalized rank workspaces.
        Matrix m_context, m_error, m_scaled;
        /// Empty standard-model sensor argument; both sides of the selection remain lvalues.
        Matrix m_emptyAcceleration;
        /// Reused factorization; the numerical rank rule and threshold are unchanged.
        std::unique_ptr<Eigen::ColPivHouseholderQR<Matrix>> m_rankQR;
        /// Cached rank is reusable only if every feature-window coefficient is bit-identical.
        bool m_rankDirty{ true };
        /// Gain/coefficients are reusable only until the next adaptation, reset or history break.
        bool m_gainCurrent{ false };
        /// Bounded horizon-plus-one origin snapshots.
        std::vector<Forecast> m_forecasts;
        /// Matured rows in the current excitation window.
        std::uint64_t m_featureRows{ 0 };
        /// Adaptation count since the last history break.
        std::uint64_t m_segmentUpdates{ 0 };
        /// Sum of frozen-origin squared prediction errors in this segment.
        double m_squaredError{ 0 };
        /// Number of output values scored in this segment.
        std::uint64_t m_errorValues{ 0 };
        /// Numerical fault latched until an explicit model reset.
        bool m_fault{ false };
    };

    /// Immutable pair configuration.
    ShadowConfig m_config;

    /// WFS/applied-input learner with no accelerometer columns.
    Model m_standard;

    /// Independent WFS/applied-input/accelerometer learner.
    Model m_assisted;

    /// Diagnostic selection only.
    ShadowMode m_selected{ ShadowMode::Standard };

    /// An external decision identity has been established.
    bool m_haveIndex{ false };

    /// Last accepted external decision identity.
    std::uint64_t m_lastIndex{ 0 };

    /// Only validity flags retained from the latest input, without sample matrices.
    ShadowRecord m_validity;

    /// Construct an independent model without changing the other model.
    void initialize( Model &model /**< [out] Model to replace. */,
                     bool   assisted /**< [in] Include accelerometer columns. */ );

    /// Break histories/forecasts and excitation evidence while retaining learned coefficients and training requests.
    void break_history( Model &model /**< [in,out] Model with missing or discontinuous input. */,
                        bool   resetBuffers = true /**< [in] False when a full reset already cleared histories. */ );

    /// Process one model's complete record, score old forecasts, and optionally adapt.
    void step( Model              &model /**< [in,out] Independent model state. */,
               const ShadowRecord &record /**< [in] Validated common record. */,
               bool                assisted /**< [in] Use the supplied acceleration features. */ );

    /// Return the independent state selected by a validated enum value.
    Model &model( ShadowMode mode /**< [in] Standard or assisted. */ );

    /// Return an independent state for read-only diagnostics.
    const Model &model( ShadowMode mode /**< [in] Standard or assisted. */ ) const;

  public:
    /// Allocate the pair and validate fixed configuration; training starts disabled for both models.
    explicit ShadowController( const ShadowConfig &config /**< [in] Fixed model and readiness parameters. */ );

    /// Accept a complete record; invalid common data stop both models, missing acceleration stops only the assisted
    /// one.
    bool process( const ShadowRecord &record /**< [in] Explicit synchronized data and validity assertions. */ );

    /// Record missing/unverified inputs without creating synthetic command or sensor values.
    void invalidate( const ShadowRecord &validity /**< [in] Latest available validity flags. */ );

    /// Enable or pause one learner without changing selection or the other learner.
    void training( ShadowMode mode /**< [in] Learner to configure. */,
                   bool       enabled /**< [in] Requested adaptation state. */ );

    /// Reset one learner, its diagnostics, and histories; selection and the other learner are retained.
    void reset( ShadowMode mode /**< [in] Independent model to reset. */ );

    /// Select a ready diagnostic predictor; idempotent requests are accepted even during warm-up.
    bool select( ShadowMode mode /**< [in] Requested diagnostic model. */ );

    /// Return the selected diagnostic model; this does not identify an active physical controller.
    ShadowMode selected() const;

    /// Return a model's current readiness, training, and prediction diagnostics.
    ShadowStatus status( ShadowMode mode /**< [in] Model to inspect. */ ) const;

    /// Return the last input validity flags; matrices are always empty in this snapshot.
    ShadowRecord validity() const;

    /// Return a copy of canonical [A B D C] coefficients for independent numerical diagnostics.
    Matrix prediction_matrix( ShadowMode mode /**< [in] Independent model to inspect. */ );

    /// Return a diagnostic next input only for a ready model; no history/output mutation is performed.
    Matrix proposal( ShadowMode mode /**< [in] Independent model to inspect. */ ) const;

    /// Evaluate the latest ready validation snapshot without adaptation, rank work or allocation.
    /** Caller serializes access and supplies a causally synchronized canonical state. Readiness refers to the
     * latest completed validation/evidence window, not a hypothetical future window. Training may be paused
     * deliberately; status().m_updates identifies the fit version. Frame age is diagnostic only,
     * not a physical availability/deadline assertion. The opt-in application shadow pipeline uses this API without any
     * publication or actuation route. A false result must be ignored; correctly shaped nonaliasing output is zeroed on
     * failure.
     */
    bool
    predict_cached( ShadowMode    mode /**< [in] Independent diagnostic model. */,
                    std::uint64_t decisionIndex /**< [in] State origin, no earlier than the validation snapshot. */,
                    std::uint64_t maxAgeFrames /**< [in] Explicit snapshot age; zero requires the latest record. */,
                    const Matrix &state /**< [in] Finite [y_p;u_p;a_p], already synchronized. */,
                    Matrix       &output /**< [out] Pre-sized mode-by-one buffer, disjoint from state. */ ) const;
};

} // namespace DDSPC

#endif // PCSHADOW_HPP
