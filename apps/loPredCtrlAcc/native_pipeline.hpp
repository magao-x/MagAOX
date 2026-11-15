/** \file native_pipeline.hpp
 * \brief Native read-only request-domain DDSPC assembly; upstream owns acquisition synchronization.
 */
#ifndef PC_NATIVE_PIPELINE_HPP
#define PC_NATIVE_PIPELINE_HPP
#include "synchronized_inputs.hpp"
#include "synco_shadow.hpp"
#include <deque>

namespace DDSPC
{
/// Reviewed calibration file, with explicit identities/units and no guessed DM pixel-to-mode mapping.
struct NativeCalibration
{
    std::uint64_t              m_layout{ 0 }; ///< Calibration revision recorded with every prediction.
    std::array<std::size_t, 2> m_indices{};   ///< Calibrated WFS stream mode indices.
    std::string m_modalUnits, m_commandUnits; ///< Recorded unit convention; neither is inferred from stream names.
    Matrix      m_projection, m_shapes;       ///< Modes by full DM pixels; optional auxiliary pixels by modes.
};

/// Load a bounded text calibration; a missing path leaves native training inhibited, never inserts an identity map.
NativeCalibration load_native_calibration( const std::string &path /**< [in] Reviewed private calibration file. */ );

/// Startup-only receiver contract; anchors must come from the upstream acquisition convention, not rate fitting.
struct NativePipelineConfig
{
    bool  m_enabled{ false };      ///< Explicit opt-in independent of ADC-only/native observation.
    bool  m_sampleDriven{ true };  ///< Native live path; false is retained only for historical replay comparisons.
    bool  m_scaledPenalty{ true }; ///< Match loPredCtrl's model-scaled regularization in both phase model banks.
    realT m_feedback{ 0 }; ///< Same configured additive current-WFS feedback for standard and assisted candidates.
    NativeCalibration m_calibration;           ///< Owned immutable full-image projection.
    std::string   m_statusDevice, m_reference; ///< Actual upstream INDI producer and configured RTC reference identity.
    std::string   m_statusProperty{ "timingDiag" }; ///< Existing producer schema: timingDiag or AD7689 ddspcTiming.
    std::uint64_t m_syncEpoch{ 0 }, m_modalAnchor{ 0 }, m_adcAnchor{ 0 }; ///< Reviewed source-incarnation 4:1 anchors.
    std::int64_t  m_statusMaxAgeNs{ 1000000000 }, m_sampleToleranceNs{ 100000 }; ///< Local heartbeat/sampler limits.
    std::int64_t  m_predictionLeadNs{ 500000 }; ///< Diagnostic compute deadline after modal availability.
    bool          m_syncoAssociation{ false };  ///< Explicit reviewed WFS-counter to scheduler-cycle association.
    std::int64_t  m_syncoCycleOffset{ 0 };      ///< Advertised scheduler cycle plus one, minus selected modal counter.
};

/// Actual producer status receipt; integer counters are checked before constructing this packet.
struct NativeSyncStatus
{
    UpstreamStatus m_status; ///< Actual synchronized mode/lock and local receipt.
    std::uint64_t  m_sequence{ 0 }, m_referenceCounter{ 0 },
        m_gaps{ 0 };                       ///< Upstream diagnostics, not ADC cnt0 anchors.
    bool         m_hasSlotOffset{ false }; ///< Producer exports a nominal-slot versus publication-counter witness.
    std::int64_t m_slotOffset{ 0 }; ///< Explicit counter difference; changes identify producer-side skipped slots.
};

/// Single numerical-thread receiver; all inputs come from bounded queues, never operational output streams.
class NativeInputPipeline
{
    MultirateShadow             &m_engine;      ///< Borrowed sole numerical owner.
    NativePipelineConfig         m_config;      ///< Immutable reviewed calibration/anchor convention.
    ShadowConfig                 m_model;       ///< Actual native dimensions.
    MultirateContract            m_contract;    ///< Bound to observed source incarnations.
    NativeSyncStatus             m_sync;        ///< Most recent actual upstream heartbeat.
    std::deque<NativeSyncStatus> m_syncHistory; ///< Eight received heartbeats for origins queued before a newer status.
    std::array<NativeObservation, 4> m_latest{};      ///< Source identities, not fabricated availability.
    std::deque<ProjectedRequest>     m_requests;      ///< Bounded chronological complete projected snapshots.
    std::uint64_t m_sequence{ 0 }, m_pendingWfs{ 0 }; ///< Independent command packet and open WFS frame.
    std::int64_t  m_openedNs{ -1 }; ///< Actual modal availability defining the local request sampling operator.
    std::array<std::int64_t, 4> m_sampled{};    ///< Actual observer sampling times; never physical application times.
    unsigned                    m_samples{ 0 }; ///< Number of actual cutoffs observed in the current frame.
    bool m_bound{ false }, m_samplerLost{ false }, m_syncLost{ false }; ///< Explicit restart/input gates.
    MagAOX::app::synco::Record m_schedule{};                            ///< Original read-only scheduler record.
    std::int64_t               m_scheduleReceipt{ -1 }; ///< Actual observer availability of that record.
    std::shared_ptr<SyncoShadowProducer>
                m_producer; ///< Worker prepares; acquisition callback copies private images only.
    const char *m_reason{ "native_calibration_unavailable" }; ///< Stable admission/failure reason.

    /// Bind actual sources to reviewed calibration, without requiring exposure or remote clock reconstruction.
    bool bind( const NativeObservation &modal /**< [in] Actual current modal snapshot. */ );

    /// Close one actual command frame before freezing the next measured WFS origin.
    void modal( const NativeObservation &record /**< [in] Original modal snapshot. */,
                std::int64_t             nowNs /**< [in] Numerical processing time. */ );

  public:
    /// Allocate only bounded private model-adapter state; no native stream or hardware access.
    NativeInputPipeline( MultirateShadow            &engine /**< [in,out] Existing native engine. */,
                         const NativePipelineConfig &config /**< [in] Reviewed receiver convention. */,
                         const ShadowConfig         &model /**< [in] Native model dimensions. */ );

    /// Forward one original source snapshot, checking common/ADC restarts independently.
    void observe( const NativeObservation &record /**< [in] Owned source observation. */,
                  std::int64_t             nowNs /**< [in] Actual processing time. */ );

    /// Project a complete immutable request on the numerical thread; no two-pixel surrogate is used.
    void command( const NativeObservation &record /**< [in] Original complete request metadata. */,
                  std::span<const float>   values /**< [in] Entire coherent DM image. */ );

    /// Record an actual observer sampling opportunity, without synthesizing missed ticks.
    void tick( std::int64_t sampledNs /**< [in] Actual local observer time. */ );

    /// Consume observed producer status; backward counters and changed gap counts revoke the lock witness.
    void sync( const NativeSyncStatus &status /**< [in] Received upstream diagnostics. */ );

    /// Retain the original existing schedule; this is not producer registration or an auxiliary write.
    void schedule( const MagAOX::app::synco::Record &record /**< [in] Coherent scheduler payload. */,
                   std::int64_t                      receivedNs /**< [in] Actual local receipt. */,
                   std::int64_t                      nowNs /**< [in] Actual numerical callback time; may be late. */ );

    /// Invalidate pending data and every private payload after queue loss or shutdown.
    void invalidate();

    /// Add compact native-binding and mock-producer diagnostics without granting any output permission.
    void status( OnlineShadowStatus &status /**< [in,out] Numerical snapshot. */ ) const;

    /// Share only the bounded private producer seam; it has no ImageStreamIO or hardware API.
    std::shared_ptr<SyncoShadowProducer> producer() const;
};
} // namespace DDSPC
#endif
