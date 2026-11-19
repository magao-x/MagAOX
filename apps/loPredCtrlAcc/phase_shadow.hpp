/** \file phase_shadow.hpp
 * \brief ADC-driven DDSPC inference and WFS-driven native RLS/QR learning; no clock synchronization or output API.
 */
#ifndef PC_PHASE_SHADOW_HPP
#define PC_PHASE_SHADOW_HPP
#include "multirate_shadow.hpp"
#include "synco_shadow.hpp"

namespace DDSPC
{
struct NativePipelineConfig;
struct NativeSyncStatus;

/// One acquisition owner and one numerical owner, exchanging bounded frames and immutable phase snapshots.
class PhaseShadow
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Fixed histories/queues, independent phase-conditioned native learners.
  public:
    /// Allocate before starting either owner; this component has no stream or hardware access.
    PhaseShadow( const NativePipelineConfig &inputs /**< [in] Reviewed source/calibration contract. */,
                 const OnlineShadowConfig   &config /**< [in] Bounded native numerical settings. */ );
    /// Release only private state after both owners have stopped.
    ~PhaseShadow();
    /// Consume original source records on the sole acquisition owner; only new ADC records initiate fast inference.
    void observe(
        const NativeObservation &record /**< [in] Original immutable source record. */,
        std::int64_t nowNs = -1 /**< [in] Actual fast-owner entry time; replay defaults to source availability. */ );
    /// Project a complete read-only request; no proposed correction can enter this path.
    void command( const NativeObservation &record /**< [in] Original coherent metadata. */,
                  std::span<const float>   values /**< [in] Entire original image. */ );
    /// Witness a coherent read (including idle duplicates) before qualifying historical request cutoffs.
    void command_poll( bool         coherent /**< [in] No busy/racing/missing command metadata. */,
                       std::int64_t nowNs /**< [in] Actual post-read time. */ );
    /// Expire stale inputs and close request samples at actual advertised synco cutoffs, without a local phase timer.
    void tick( std::int64_t nowNs /**< [in] Actual acquisition-owner monotonic time. */ );
    /// Accept the existing scheduler record without attaching to any auxiliary or sidecar stream.
    void schedule( const MagAOX::app::synco::Record &record /**< [in] Original schedule. */,
                   std::int64_t                      receiptNs /**< [in] Actual local receipt. */ );
    /// Stage actual upstream status from INDI; counters are not used to estimate cadence or phase.
    void sync( const NativeSyncStatus &status /**< [in] Actual producer status. */ );
    /// Adapt at most one closed real WFS frame on the numerical owner; never called by ADC acquisition.
    bool learn();
    /// Observe bounded handoff backlog without touching numerical state; safe from either owner.
    std::size_t queued() const;
    /// Drain scores on the sole numerical owner, preserving single-producer recorder ownership.
    bool read_prediction( ShadowPrediction &prediction /**< [out] Matured fast-origin score. */ );
    /// Copy numerical state plus bounded atomic fast-path diagnostics on the numerical owner.
    OnlineShadowStatus
    status( std::int64_t nowNs = -1 /**< [in] Actual local readback time; replay may use explicit ticks. */ ) const;
    /// Select diagnostics atomically, without changing training or granting output authority.
    void request( ShadowMode mode /**< [in] Standard or assisted. */ );
    /// Enable both independent native fits on the numerical owner; histories continue while coefficients are frozen.
    void learning( bool enabled /**< [in] Explicit operator intent. */ );
    /// Inhibit old candidates immediately and reset both fits on their numerical owner.
    void reset_models();
    /// Revoke private payloads and numerical eligibility after worker faults; no operational channel is changed.
    void inhibit();
    /// Inspect an owned coefficient snapshot for offline validation; the returned copy is never a training input.
    Matrix coefficients( ShadowMode mode /**< [in] Independent model. */,
                         unsigned   phase /**< [in] ADC phase zero through three. */ ) const;
    /// Return the existing private Request/payload seam; it cannot publish an image.
    std::shared_ptr<SyncoShadowProducer> producer() const;
};
} // namespace DDSPC
#endif
