/** \file shadow_controller.hpp
 * \brief Independent causal DDSPC model management with no command publication interface.
 */

#ifndef PCSHADOW_HPP
#define PCSHADOW_HPP

#include <memory>
#include <atomic>
#include <vector>
#include <array>

#include "ar_controller.hpp"

namespace DDSPC
{

/// Select an independently identified shadow model; neither option authorizes output.
enum class ShadowMode
{
    Standard,
    AccelAssisted
};

/// One scalar of a scored frozen-origin forecast; future commands are confirmed later, never known at the origin.
struct ShadowPrediction
{
    ShadowMode    m_mode{ ShadowMode::Standard }; ///< Independent predictor identity.
    std::uint64_t m_origin{ 0 }, m_target{ 0 };   ///< Original external WFS counters, not circular buffer indices.
    unsigned      m_horizon{ 0 }, m_channel{ 0 }; ///< Positive WFS-frame horizon and measured modal channel.
    double        m_prediction{ 0 }, m_measurement{ 0 }; ///< Conditional forecast and identical observed target.
    bool m_late{ false }; ///< Computed after an explicit diagnostic deadline, or without any deadline qualification.
    std::uint64_t m_generation{ 0 }, m_epoch{ 0 }, m_layout{ 0 }; ///< Adapter source/synchronization/calibration tags.
    double   m_originPrediction{ 0 }; ///< Forecast using only origin state and a held-last-confirmed-input assumption.
    bool     m_originValid{ false }; ///< False for older auto-origins whose future input was already known when frozen.
    unsigned m_phase{ 0 };           ///< Upstream counter phase; repeated WFS targets remain real measurements.
    std::uint64_t m_adcCounter{ 0 }, m_version{ 0 }; ///< Individual ADC and immutable coefficient version.
    std::int64_t  m_decisionNs{ -1 }, m_completedNs{ -1 }, m_slotTargetNs{ -1 }; ///< Actual local prediction events.
    double        m_candidate{ 0 };           ///< Private modal candidate; never an observed training command.
    bool          m_conditionalValid{ true }; ///< False when only an origin-only forecast was computed.
    std::int64_t  m_slotCycle{ -1 }, m_slotStep{ -1 }, m_slotEpoch{ 0 }, m_slotGeneration{ 0 },
        m_slotCutoffNs{ -1 }; ///< Actual Request identity.
};

/// Optional diagnostics consumer, serialized with learning; it must not block, throw or change model/input state.
class ShadowPredictionSink
{
  public:
    /// Destroy only the interface, without owning the learner.
    virtual ~ShadowPredictionSink() = default;

    /// Enqueue a fixed scalar or count diagnostic loss; no I/O or learner calls are allowed here.
    virtual void prediction( const ShadowPrediction &sample /**< [in] Scored conditional forecast. */ ) noexcept = 0;
};

/// Bounded SPSC scalar-score transport; diagnostic overflow never changes learner or input state.
class ShadowPredictionQueue
{
    std::vector<ShadowPrediction> m_rows;                   ///< Fixed owned payload slots; empty means disabled.
    std::atomic<std::size_t>      m_head{ 0 }, m_tail{ 0 }; ///< Sole numerical producer and ordinary consumer cursors.
    std::atomic<std::uint64_t>    m_drops{ 0 };             ///< Rejected diagnostic records, never acquisition losses.

  public:
    /// Allocate at most 65536 records before starting either thread; zero disables tracing.
    explicit ShadowPredictionQueue( std::size_t capacity /**< [in] Fixed scalar capacity. */ );

    /// Copy a scalar or count overflow, with no allocation, blocking or I/O.
    bool push( const ShadowPrediction &sample /**< [in] Original score. */ ) noexcept;

    /// Copy one oldest scalar on the sole consumer thread.
    bool pop( ShadowPrediction &sample /**< [out] Original score if present. */ ) noexcept;

    /// Observe approximate active backlog, separate from historical model counters.
    std::size_t queued() const noexcept;

    /// Observe lifetime trace-only losses since reset.
    std::uint64_t drops() const noexcept;

    /// Reset only when producer and consumer are quiescent, including an explicit worker restart.
    void reset() noexcept;
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

    /// Opt-in operational penalty scaling; historical causal replay defaults remain literal.
    bool m_scaledPenalty{ false };

    /// Relative rank threshold after feature column normalization.
    realT m_rankTolerance{ 1e-5f };

    /// Minimum learned examples; zero uses each model's feature count.
    std::uint64_t m_minUpdates{ 0 };

    /// Select corrected QR RLS instead of conventional RLS for both independent learners.
    bool m_useQrd{ false };

    /// Assisted command opportunities per WFS record; one preserves the existing single-rate path.
    int m_assistedCommandSlots{ 1 };

    /// Standard input lift; one retains single-rate compatibility, four gives a matched WFS-only multirate baseline.
    int m_standardCommandSlots{ 1 };

    /// Freeze forecasts only at explicit next-origin calls, not after future commands have been confirmed.
    bool m_externalOrigins{ false };

    /// Adapt/validate every Nth record; every measurement/history row is still retained. One preserves existing work.
    unsigned m_updateStride{ 1 };
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

    /// Optional newest-slot-first confirmed command lift; required when assistedCommandSlots exceeds one.
    Matrix m_assistedAppliedCommand;

    /// Independent complete interval/provenance assertion for the lifted input.
    bool m_assistedAppliedValid{ false };

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

    /// Rank of the normalized feature window at the last adaptation step, without an RLS prior.
    /// Cached for at most updateStride-1 records; input loss invalidates readiness immediately.
    int m_rank{ 0 };

    /// Conditional future-residual RMSE; meaningful only with predictionValid.
    double        m_rmse{ 0 };
    double        m_originRmse{ 0 };        ///< Origin-only RMSE, excluding any actual future input values.
    std::uint64_t m_originPredictions{ 0 }; ///< Scored origin-only forecast vectors in this segment.

    /// Current readiness/fault description.
    ShadowReason m_reason{ ShadowReason::AwaitingRecord };
};

/// Owned immutable numerical snapshot; only the learning thread constructs/factorizes it.
struct ShadowSnapshot
{
    Matrix                m_theta;  ///< Canonical [A B D C] coefficients.
    std::array<Matrix, 5> m_gain;   ///< Cached gains for zero through four fixed oldest quarter-frame slots.
    ShadowStatus          m_status; ///< Excitation/finite/readiness evidence at publication.
    int                   m_stateRows{ 0 }, m_commands{ 0 };     ///< Canonical state size and commands per WFS frame.
    std::uint64_t         m_inputEpoch{ 0 }, m_sensorEpoch{ 0 }; ///< Receiver continuity at publication.
    std::int64_t          m_dataAvailableNs{ -1 }; ///< High-water availability of all evidence used by the fit.
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
        /// Original external counter at which coefficients and state were frozen.
        std::uint64_t m_index{ 0 };
        /// Diagnostic lateness is distinct from data causality and DM slot eligibility.
        bool m_late{ false };
        /// Canonical [A B D C] prediction matrix at the origin.
        Matrix m_theta;
        /// Canonical [y_p;u_p;a_p] state available at the origin.
        Matrix m_state;
        /// Actual origin-only WFS forecast under the explicit held-last-confirmed-input assumption.
        Matrix m_availablePrediction;
        /// Only explicit next-origin calls can establish this forecast without future inputs.
        bool m_availableValid{ false };
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
        /// Pre-subtraction prediction copied only for an enabled scalar trace, so reporting cannot depend on truth.
        Matrix m_scoredPrediction;
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
        /// Separate origin-only diagnostics never feed learning, command history or control.
        double m_originSquaredError{ 0 };
        /// Number of output values scored in this segment.
        std::uint64_t m_errorValues{ 0 };
        /// Number of scalar targets scored without future inputs.
        std::uint64_t m_originErrorValues{ 0 };
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

    /// Borrowed optional diagnostics sink; caller serializes replacement and outlives all process calls.
    ShadowPredictionSink *m_predictionSink{ nullptr };

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

    /// Freeze an origin-only forecast with the last confirmed input repeated, never a shadow-generated input.
    void available_forecast( Model    &model /**< [in,out] Independent ready model. */,
                             Forecast &origin /**< [in,out] Frozen state and coefficients. */ );

    /// Return the independent state selected by a validated enum value.
    Model &model( ShadowMode mode /**< [in] Standard or assisted. */ );

    /// Return an independent state for read-only diagnostics.
    const Model &model( ShadowMode mode /**< [in] Standard or assisted. */ ) const;

  public:
    /// Allocate the pair and validate fixed configuration; training starts disabled for both models.
    explicit ShadowController( const ShadowConfig &config /**< [in] Fixed model and readiness parameters. */ );

    /// Install optional nonblocking diagnostics without changing either numerical model or readiness.
    void prediction_sink( ShadowPredictionSink *sink /**< [in] Borrowed sink or nullptr; no ownership transfer. */ );

    /// Accept a complete record; invalid common data stop both models, missing acceleration stops only the assisted
    /// one.
    bool process( const ShadowRecord &record /**< [in] Explicit synchronized data and validity assertions. */ );

    /// Record missing/unverified inputs without creating synthetic command or sensor values.
    void invalidate( const ShadowRecord &validity /**< [in] Latest available validity flags. */ );

    /// Revoke only assisted histories/forecasts after ADC loss, leaving the WFS-only learner unchanged.
    void invalidate_acceleration();

    /// Cancel next-origin scoring when computation missed its externally established cutoff.
    void discard_next_forecasts();

    /// Preserve a data-causal forecast for diagnostics while explicitly marking missed computation deadlines.
    void mark_next_forecasts_late();

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

    /// Export coefficients and all phase-prefix gains on the learning thread; no measurement history is advanced.
    bool snapshot( ShadowMode      mode /**< [in] Independent model. */,
                   ShadowSnapshot &snapshot /**< [out] Owned immutable copy for fast prediction. */ );

    /// Return a diagnostic next input only for a ready model; no history/output mutation is performed.
    Matrix proposal( ShadowMode mode /**< [in] Independent model to inspect. */ ) const;

    /// Evaluate/freeze a next-origin full command plan using only current measurements and confirmed past inputs.
    bool predict_next( ShadowMode    mode /**< [in] Independent ready model. */,
                       std::uint64_t index /**< [in] Exactly the next external WFS index. */,
                       const Matrix &measurement /**< [in] New real WFS record. */,
                       const Matrix &acceleration /**< [in] Frozen raw-sample lift, empty for standard. */,
                       Matrix       &plan /**< [out] Pre-sized F*m*commandSlots newest-first future plan. */,
                       const Matrix *fixedPrefix =
                           nullptr /**< [in] Independently established oldest command slots, newest first. */ );

    /// Freeze only the DDSPC state/coefficients for conditional scoring, without requiring or solving a DM plan.
    bool freeze_next( ShadowMode    mode /**< [in] Independent ready model. */,
                      std::uint64_t index /**< [in] Next external WFS counter. */,
                      const Matrix &measurement /**< [in] Actual current measurement. */,
                      const Matrix &acceleration /**< [in] Available individual ADC lift. */,
                      bool          late /**< [in] Honest diagnostic computation status, not an output permission. */ );

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
