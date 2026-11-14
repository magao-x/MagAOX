/** \file online_shadow.hpp
 * \brief Budgeted native shadow learning worker; acquisition only enqueues immutable observations.
 */
#ifndef PC_ONLINE_SHADOW_HPP
#define PC_ONLINE_SHADOW_HPP
#include "multirate_shadow.hpp"
#include "native_observer.hpp"

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

    /// Sole native observation producer enqueues without allocation, blocking or learner access.
    void observe( const NativeObservation &record /**< [in] Owned source record. */ ) noexcept override;

    /// Offer a physical-evidence contract through a bounded ordinary-thread mailbox; not an INDI/config trust bit.
    bool qualify( const MultirateContract &contract /**< [in] Independently verified real source evidence. */ );

    /// Offer a qualified WFS/exposure/schedule packet; reject immediately when its fixed queue is full.
    bool origin( const OnlineOrigin &record /**< [in] Immutable causal origin. */ );

    /// Offer an independently confirmed physical held interval, never a worker-generated proposal.
    bool applied( const OnlineAppliedSlot &record /**< [in] Physical input certificate. */ );

    /// Record requested selection atomically; readiness/fallback is resolved on the numerical worker.
    void request( ShadowMode mode /**< [in] Standard or assisted. */ );

    /// Copy the latest bounded status without waiting for RLS/QR work.
    OnlineShadowStatus status() const;

    /// Copy a diagnostic plan only if its source/deadline qualification still holds at readout.
    bool candidate( ShadowMode mode /**< [in] Independent model. */,
                    Matrix    &plan /**< [out] Owned full plan. */ ) const;
};
} // namespace DDSPC
#endif
