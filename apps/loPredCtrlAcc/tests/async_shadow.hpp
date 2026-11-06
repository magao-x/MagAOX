/** \file async_shadow.hpp
 * \brief Experimental offline worker queues and coherent bounded model publication; no application integration.
 */
#ifndef loPredCtrlAcc_async_shadow_hpp
#define loPredCtrlAcc_async_shadow_hpp

#include "../shadow_controller.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace DDSPC
{
/// Return this process's monotonic nanoseconds; synthetic timestamps must use this same clock domain.
std::uint64_t offline_now_ns();

/// Explicit synthetic timing contract for an already synchronized decision record.
struct OfflineRecord
{
    ShadowRecord  m_data;               ///< Synthetic WFS and actual held-input histories, never calculated proposals.
    std::uint64_t m_streamEpoch{ 1 };   ///< Strictly increasing source incarnation on sequence restart.
    std::uint64_t m_measurementNs{ 0 }; ///< End of the synthetic WFS exposure/decision origin.
    std::uint64_t m_exposureStartNs{ 0 };    ///< Start of the declared synthetic exposure.
    std::uint64_t m_appliedNs{ 0 };          ///< Synthetic u_i hold starts at measurementNs for [i,i+1).
    std::uint64_t m_appliedAvailableNs{ 0 }; ///< This synthetic applied-input identity became known.
    std::uint64_t m_availableNs{ 0 };        ///< All required common inputs became available in the monotonic domain.
    std::uint64_t m_accelNs{ 0 };            ///< Latest sensor sample time, no later than measurementNs.
    std::uint64_t m_accelAvailableNs{ 0 };   ///< Sensor availability is distinct from sample time.

    /// Allocate fixed input shapes before starting a producer or worker.
    explicit OfflineRecord( const ShadowConfig &config /**< [in] Immutable model dimensions. */ );
};

/// Metadata is published atomically with the corresponding matrix payload, never as independent coefficients.
struct OfflineModelInfo
{
    std::uint64_t m_generation{ 0 };  ///< Requested lane generation; any discontinuity invalidates older snapshots.
    std::uint64_t m_reset{ 0 };       ///< Full-reset incarnation disambiguates equal fit versions.
    std::uint64_t m_publication{ 0 }; ///< Monotonic publication identity independent of RLS updates.
    std::uint64_t m_version{ 0 };     ///< RLS update count since full reset.
    std::uint64_t m_sequence{ 0 }; ///< Last completed validation record, also the newest trained target when adapting.
    std::uint64_t m_streamEpoch{ 0 };   ///< Source incarnation.
    std::uint64_t m_measurementNs{ 0 }; ///< Evidence origin; model age is measured from this time.
    std::uint64_t m_availableNs{ 0 };   ///< Input availability time, independent of processing order.
    std::uint64_t m_publishedNs{ 0 };   ///< Time sampled after copying, just before release-publication.
    ShadowStatus  m_status;             ///< Exact owning model diagnostics for this version/window.
};

/// Fixed matrices owned by one publication slot or one diagnostic reader; no references into a learner.
struct OfflineSnapshot
{
    OfflineModelInfo m_info;           ///< Coherent timing, fit and validity metadata.
    Matrix           m_theta;          ///< Canonical coefficients, including when the model is not ready.
    Matrix           m_gain;           ///< Full newest-first reference gain; only valid for a ready snapshot.
    Matrix           m_originState;    ///< Canonical state at the last validation origin, for numerical replay.
    Matrix           m_originProposal; ///< Exact diagnostic proposal at that origin; never actuated.

    /// Reserve all snapshot storage before publication.
    OfflineSnapshot( const ShadowConfig &config /**< [in] Immutable dimensions. */,
                     ShadowMode          mode /**< [in] Independent model. */ );
};

/// Rejection reasons distinguish unavailable publication from invalid, stale or discontinuous data.
enum class OfflinePredictionReason
{
    Valid,
    NoSnapshot,
    Busy,
    InvalidState,
    Invalidated,
    Unready,
    Stale,
    TimeOrder
};

/// Caller-owned result contains no dynamic storage.
struct OfflinePrediction
{
    OfflinePredictionReason m_reason{ OfflinePredictionReason::NoSnapshot }; ///< Valid only on successful prediction.
    OfflineModelInfo        m_model;      ///< Metadata actually used, never a later worker's status.
    std::uint64_t           m_ageNs{ 0 }; ///< Now minus evidence origin, not now minus publication.
};

/// Fixed-capacity worker-only trace, sampled after shutdown or during deterministic manual service.
struct OfflineWorkerEvent
{
    std::uint64_t m_sequence{ 0 }, m_version{ 0 };          ///< Processed identity and resulting fit version.
    std::uint64_t m_measurementNs{ 0 }, m_completedNs{ 0 }; ///< Evidence origin and completed publication attempt.
    std::uint64_t m_trainingNs{ 0 }, m_publicationNs{ 0 }, m_queueWaitNs{ 0 }; ///< Component timings.
    std::size_t   m_eigenRequests{ 0 }, m_cppRequests{ 0 }; ///< Populated only in allocation-instrumented builds.
};

/// Atomic counter snapshot; queue depth includes the record currently owned by its consumer.
struct OfflineWorkerStats
{
    std::uint64_t m_processed{ 0 }, m_discarded{ 0 }, m_publishMisses{ 0 }; ///< Work and invalidated-prefix counts.
    std::uint64_t m_depth{ 0 }, m_highWater{ 0 }, m_traceDrops{ 0 };        ///< Bounded queue/telemetry occupancy.
};

/// Two independent learners with SPSC fanout and three pinned publication slots per lane.
/** One frontend thread owns submit/reset/predict/read_snapshot. One consumer owns each learner, either a
 * spawned thread or manual_step(), never both. No mutex, wait, allocation or retry loop occurs in prediction.
 * start/stop/destruction and trace inspection are non-real-time lifecycle operations. No output sink exists.
 */
class OfflineAsyncShadow
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< All matrices/slots/queues allocated before worker startup.

  public:
    /// Allocate a bounded offline prototype; training starts enabled independently in both workers.
    OfflineAsyncShadow( const ShadowConfig &config /**< [in] Immutable numerical configuration. */,
                        std::size_t         capacity = 32 /**< [in] Records per queue, including in-flight work. */,
                        std::size_t   traceCapacity = 4096 /**< [in] Events per worker; excess telemetry is dropped. */,
                        std::uint64_t sensorMaxAgeNs = 500000 /**< [in] Maximum sample age at the WFS origin. */ );

    /// Stop and join workers outside the prediction path.
    ~OfflineAsyncShadow();

    /// Start exactly one independently owned worker per model; creation is outside frame timing.
    void start();

    /// Stop and join; queued work may remain and is reported, never silently relabeled as processed.
    void stop();

    /// Accept into both queues or drop from both; invalid timing/gaps/overflow invalidate old generations.
    bool submit( const OfflineRecord &record /**< [in] Immutable explicitly timed input. */,
                 std::uint64_t        nowNs /**< [in] Ingress availability check, same monotonic clock. */ );

    /// Request an immediate snapshot invalidation and a worker-owned full reset of one model.
    void reset( ShadowMode mode /**< [in] Model to reset without touching the other learner. */ );

    /// Execute at most one queued record without threads for reproducible scheduling tests.
    bool manual_step( ShadowMode    mode /**< [in] Sole consumer to service. */,
                      std::uint64_t publicationNs /**< [in] Synthetic completion time; zero uses the real clock. */ );

    /// Inject a worker-only stall; the frontend continues without waiting.
    void stall_until( ShadowMode    mode /**< [in] Worker to stall. */,
                      std::uint64_t untilNs /**< [in] Monotonic time to resume. */ );

    /// Pin one coherent snapshot once, predict and release; reject stale/unready/wrong-epoch models.
    bool predict( ShadowMode    mode /**< [in] Model. */,
                  std::uint64_t streamEpoch /**< [in] State's source incarnation. */,
                  std::uint64_t sequence /**< [in] Current decision identity. */,
                  std::uint64_t stateNs /**< [in] State origin; no future features permitted. */,
                  std::uint64_t nowNs /**< [in] Monotonic query time. */,
                  std::uint64_t maxAgeNs /**< [in] Maximum evidence age; publication cannot refresh old evidence. */,
                  const Matrix &state /**< [in] Finite canonical state, synchronized by the caller. */,
                  Matrix       &output /**< [out] Fixed mode-by-one buffer disjoint from state. */,
                  OfflinePrediction &result /**< [out] Exact snapshot metadata and rejection reason. */ );

    /// Copy a coherent snapshot for offline diagnostics; destination must have the configured shapes.
    bool read_snapshot( ShadowMode       mode /**< [in] Model. */,
                        OfflineSnapshot &destination /**< [out] Already allocated diagnostic snapshot. */ );

    /// Read atomic counters without accessing learner state.
    OfflineWorkerStats stats( ShadowMode mode /**< [in] Worker. */ ) const;

    /// Read bounded worker traces only after stop(), or in manual mode with no service in progress.
    const std::vector<OfflineWorkerEvent> &events( ShadowMode mode /**< [in] Worker. */ ) const;

    /// Return the used prefix of the fixed trace storage; same ownership restriction as events().
    std::size_t event_count( ShadowMode mode /**< [in] Worker. */ ) const;

    /// Read frontend-owned common ingress drop count.
    std::uint64_t drops() const;

    /// Return common queue-overflow drops, excluding invalid input rejection.
    std::uint64_t overflows() const;
};
} // namespace DDSPC
#endif
