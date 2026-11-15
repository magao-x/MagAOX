/** \file online_shadow.hpp
 * \brief Budgeted native shadow learning worker with ADC-owner inference from immutable coefficient snapshots.
 */
#ifndef PC_ONLINE_SHADOW_HPP
#define PC_ONLINE_SHADOW_HPP
#include "multirate_shadow.hpp"
#include "native_observer.hpp"
#include "native_pipeline.hpp"

namespace DDSPC
{
/// Separate SCHED_OTHER numerical worker; no stream creation, INDI client, hardware or command-publication API.
class OnlineShadowWorker : public NativeObservationSink
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Sole bounded queues, engine, thread and published scalar/plan snapshots.

  public:
    /// Validate and allocate fixed queue/model bounds, without starting acquisition or any service.
    explicit OnlineShadowWorker( const OnlineShadowConfig &config /**< [in] Fixed resource bounds. */ );

    /// Stop/join only this numerical worker.
    ~OnlineShadowWorker();

    /// Start an ordinary-priority worker; never alter other thread/process priorities or affinities.
    void start();

    /// Join before destroying the source/sink owner; no external process is signalled.
    void stop();

    /// Sole native owner consumes/enqueues without allocation, blocking or learner access; phase mode predicts here.
    void observe( const NativeObservation &record /**< [in] Owned source record. */ ) noexcept override;

    /// Configure the real native receiver before starting either worker; empty calibration remains inhibited.
    void native_inputs( const NativePipelineConfig &config /**< [in] Immutable reviewed native receiver settings. */ );

    /// Consume a complete original request; phase mode performs only its bounded calibrated linear projection.
    void command( const NativeObservation &record /**< [in] Original request metadata. */,
                  std::span<const float>   values /**< [in] Entire coherent payload. */ ) noexcept override;

    /// Forward read-only command coverage; an idle coherent poll is healthy, a busy/racing poll is not a witness.
    void command_poll( bool         coherent /**< [in] Actual poll outcome. */,
                       std::int64_t nowNs /**< [in] Actual post-read time. */ ) noexcept override;

    /// Enqueue an actual local observer sampling opportunity, separate from a measurement.
    void tick( std::int64_t nowNs /**< [in] Actual local observer time. */ ) noexcept override;

    /// Offer the latest existing schedule through a bounded mailbox, never a producer registration.
    void schedule( const MagAOX::app::synco::Record &record /**< [in] Original coherent payload. */,
                   std::int64_t                      receivedNs /**< [in] Actual receipt. */ ) noexcept override;

    /// Offer actual INDI producer status; not a user-facing synchronization enable or trust switch.
    bool sync_status( const NativeSyncStatus &status /**< [in] Checked received status. */ );

    /// Offer an explicit request-domain or physical-evidence contract; not an INDI/config trust bit.
    bool qualify( const MultirateContract &contract /**< [in] Independently verified real source evidence. */ );

    /// Offer a causal WFS origin with upstream status or strict exposure evidence; reject a full queue.
    bool origin( const OnlineOrigin &record /**< [in] Immutable causal origin. */ );

    /// Offer an independently confirmed physical held interval, never a worker-generated proposal.
    bool applied( const OnlineAppliedSlot &record /**< [in] Physical input certificate. */ );

    /// Offer a complete observed request history; physical output/application qualification remains independent.
    bool commands( const OnlineCommandFrame &record /**< [in] Counter-associated combined requests. */ );

    /// Record requested selection atomically; readiness/fallback is resolved on the numerical worker.
    void request( ShadowMode mode /**< [in] Standard or assisted. */ );

    /// Copy the latest bounded status without waiting for RLS/QR work.
    OnlineShadowStatus status() const;

    /// Single ordinary-thread consumer drains an optional bounded score queue; no model or acquisition lock.
    bool read_prediction( ShadowPrediction &sample /**< [out] Original scored forecast when available. */ );

    /// Copy a diagnostic plan only if its source/deadline qualification still holds at readout.
    bool candidate( ShadowMode mode /**< [in] Independent model. */,
                    Matrix    &plan /**< [out] Owned full plan. */ ) const;
};
} // namespace DDSPC
#endif
