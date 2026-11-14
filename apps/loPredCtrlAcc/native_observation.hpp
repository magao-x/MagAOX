/** \file native_observation.hpp
 * \brief Bounded observation accounting; no exposure inference, applied-input certification or learner interface.
 */
#ifndef PCNATIVE_OBSERVATION_HPP
#define PCNATIVE_OBSERVATION_HPP
#include "adc_reader.hpp"
#include "../../libMagAOX/common/dmCompletion.hpp"
#include <array>
#include <atomic>
#include <span>
#include <string>
#include <vector>

namespace DDSPC
{
/// Fixed source slots in the private observation journal.
enum class ObservationSource : unsigned
{
    Adc,
    Modal,
    Camera,
    Dm,
    Completion,
    Sdk,
    Count
};
constexpr std::size_t observationSources = std::size_t( ObservationSource::Count );

/// Explicit producer clock declaration; unknown retains raw timestamps without a cross-clock subtraction.
enum class ObservationClock : unsigned
{
    Unknown,
    Realtime,
    Tai,
    Monotonic
};

/// Parse only supported explicit clock declarations; no automatic domain inference.
ObservationClock observation_clock( const std::string &name /**< [in] unknown/realtime/tai/monotonic. */ );

/// Return the stable clock label written to the journal.
const char *observation_clock_name( ObservationClock clock /**< [in] Declared clock. */ );

/// Return a stable source role, independent of stream name and producer identity.
const char *observation_source_name( ObservationSource source /**< [in] Fixed slot. */ );

/// Immutable existing-stream selection; clock names are declarations, not verification.
struct ObservationStream
{
    std::string m_name; ///< Existing-only stream; empty disables an optional source.
    std::string m_acquisitionClock{ "unknown" }, m_publicationClock{ "unknown" }; ///< Raw metadata clock declarations.
};

/// Startup-only private observation settings; no output stream or training configuration exists.
struct NativeObservationConfig
{
    std::array<ObservationStream, observationSources> m_streams; ///< ADC slot name is supplied by the owning app.
    std::string m_directory; ///< Existing private directory when recording; never used by diagnostics-only acquisition.
    double      m_durationSec{ 60 }; ///< Acquisition duration; zero runs continuously only with recording disabled.
    bool        m_record{ true }; ///< Optional finite private journal; installed native entrypoint defaults this off.
    unsigned    m_pollUs{ 50 };   ///< Nominal polling delay; actual coverage/loss is measured.
    std::size_t m_queueCapacity{ 32768 }; ///< Fixed recording queue; overflow counts instead of blocking input.
    std::int64_t m_staleNs{ 10000000 };   ///< Observer silence diagnostic; never a control freshness assertion.
    std::array<std::size_t, 2> m_modalIndices{
        0, 1 };                       ///< Selected calibrated-stream entries, without asserting TT identity.
    bool m_recordDmCommands{ false }; ///< Opt-in full command snapshots; never physical-application certificates.
    std::size_t m_commandQueueCapacity{ 64 },
        m_commandMaxElements{ 4096 };      ///< Independently bounded payload retention; no allocation in source copies.
    std::size_t m_historyCapacity{ 4096 }; ///< Per-source chronological input retention, independent of disk recording.

    /// Set existing source defaults without opening any input or file.
    NativeObservationConfig();
};

/// Validate names, clocks, finite bounds and private output location without opening native streams.
void validate_native_observation( const NativeObservationConfig &config /**< [in] Startup settings. */ );

/// Record kind; gaps and disconnects remain explicit even when no sample can be copied.
enum class ObservationEvent : unsigned
{
    Sample,
    Attached,
    Disconnected,
    AdcBreak
};

/// Per-record diagnostics, never a measurement/application qualification.
enum ObservationFlag : std::uint32_t
{
    ObservationInvalidTime            = 1,
    ObservationUnknownClock           = 2,
    ObservationClockMismatch          = 4,
    ObservationInvalidPayload         = 8,
    ObservationProtocolInvalid        = 16,
    ObservationProtocolIncomplete     = 32,
    ObservationAssociationMissing     = 64,
    ObservationAssociationMatched     = 128,
    ObservationTimestampDiscontinuity = 256,
    ObservationPayloadUnretained      = 512
};

/// Owned fixed-size journal record; no pointer into a producer or model escapes.
struct NativeObservation
{
    ObservationSource m_source{ ObservationSource::Adc };  ///< Input role, never a destination.
    ObservationEvent  m_event{ ObservationEvent::Sample }; ///< Snapshot or lifecycle evidence.
    std::uint32_t     m_flags{ 0 }; ///< Observation problems; zero does not grant training authority.
    std::uint64_t     m_device{ 0 }, m_fileInode{ 0 },
        m_imageInode{ 0 }; ///< Filesystem and ImageStreamIO identities kept distinct.
    std::uint64_t m_generation{ 0 }, m_segment{ 0 }, m_counter{ 0 },
        m_missing{ 0 };            ///< Incarnation and original publication.
    std::uint64_t m_counter1{ 0 }; ///< Uninterpreted native cnt1, not an assumed camera-origin identifier.
    bool m_hasCounter1{ false };   ///< Native snapshot carries cnt1; the ADC cursor does not retain this header field.
    std::uint32_t m_width{ 0 }, m_height{ 0 }, m_depth{ 0 }, m_datatype{ 0 }; ///< Observed original layout.
    std::int64_t  m_acquiredRawNs{ -1 },
        m_publishedRawNs{ -1 }; ///< Unmodified producer metadata, no assumed exposure event.
    std::int64_t m_acquiredMonoNs{ -1 },
        m_publishedMonoNs{ -1 };                          ///< Actual ADC mapping only; other inputs remain unknown.
    std::int64_t m_receivedNs{ -1 }, m_availableNs{ -1 }; ///< Decoder/snapshot receipt and observer availability.
    std::int64_t m_clockNowNs{ -1 }, m_clockBracketNs{ -1 },
        m_publicationAgeNs{ -1 }; ///< Optional declared-clock context.
    std::uint64_t m_elements{ 0 },
        m_payloadHash{ 0 };               ///< Payload extent and diagnostic digest, not an atomic provenance proof.
    std::array<double, 2> m_values{};     ///< ADC acceleration or selected modal/DM values, depending on source role.
    std::array<std::uint16_t, 2> m_raw{}; ///< Original ADC pair; zero for other roles.
    double m_min{ 0 }, m_max{ 0 }, m_rms{ 0 }; ///< Command/vector diagnostics, not projected or applied control input.
    MagAOX::dmCompletion::Wire m_wire{};       ///< Exact optional 16-word software endpoint record.
};

/// Lifetime counters and latest evidence for one source, published only by the private instance.
struct NativeSourceStatus
{
    NativeObservation m_latest; ///< Latest owned event/snapshot, without any validity promotion.
    bool              m_connected{ false }, m_stale{ true }, m_valid{ false },
        m_idle{ false }; ///< Transport validity; a static valid DM request is idle, not stale or physically applied.
    bool          m_busyNow{ false }, m_incompatibleNow{ false }; ///< Last poll state, distinct from lifetime counters.
    std::int64_t  m_busySinceNs{ -1 }; ///< Continuous producer-write observation for stuck-writer diagnostics.
    std::uint64_t m_samples{ 0 }, m_counterGaps{ 0 }, m_restarts{ 0 }, m_duplicates{ 0 }; ///< Publication observations.
    std::uint64_t m_busy{ 0 }, m_races{ 0 }, m_incompatible{ 0 },
        m_invalid{ 0 }; ///< Native copy/layout/payload failures.
    std::uint64_t m_clockUnknown{ 0 },
        m_clockMismatches{ 0 }; ///< Declared-clock coverage, separate from transport health.
    std::uint64_t m_protocolInvalid{ 0 }, m_protocolIncomplete{ 0 },
        m_sequenceGaps{ 0 }; ///< Endpoint protocol evidence.
    std::uint64_t m_publisherRestarts{
        0 }; ///< Endpoint publisher session/epoch changes, separate from backing replacement.
    std::uint64_t m_associationFailures{ 0 },
        m_associations{ 0 };              ///< Metadata matches only, never physical application.
    std::int64_t  m_lastReceiptNs{ -1 };  ///< Observer availability high-water for silence checks.
    std::int64_t  m_firstReceiptNs{ -1 }; ///< Receipt-rate window origin, reset on source/continuity changes.
    std::uint64_t m_epochSamples{ 0 };    ///< Unique snapshots in this receipt-rate epoch, excluding duplicate polls.
    double        m_receivedRateHz{ 0 }; ///< Accepted snapshots per observer second; does not establish a source clock.
    double        m_publicationRateHz{ 0 }; ///< Counter advances per observer second; gaps remain separately counted.
    std::uint64_t m_rateCounter{ 0 };       ///< Counter at the receipt-rate window origin.
    bool          m_rateWindowEstablished{ false }; ///< Once warm, rate changes are published over 250 ms windows.
    std::uint64_t m_timestampDiscontinuities{ 0 },
        m_historyEvictions{ 0 }; ///< Clock regressions and retention eviction.
    std::size_t m_retained{ 0 }; ///< Bounded source-history occupancy; retention eviction is not source loss.
};

/// Thread-safe observer snapshot; every model and physical-application qualification remains false by construction.
struct NativeObservationStatus
{
    AdcStatus m_adcCapture; ///< Independent original decoder counters/rate; never add capture loss to forwarding loss.
    std::array<NativeSourceStatus, observationSources> m_sources; ///< Independent source health and exact IDs.
    std::uint64_t                                      m_adcMissing{ 0 }, m_adcBreaks{ 0 }, m_queueDrops{ 0 },
        m_written{ 0 };                                               ///< Distinct loss populations.
    std::int64_t m_startedNs{ -1 }, m_stoppedNs{ -1 }, m_nowNs{ -1 }; ///< Actual local observation interval.
    bool m_finished{ false }, m_ioError{ false }; ///< Worker completion (including stop) and persistent I/O failure.
    std::uint64_t m_commandsWritten{ 0 }, m_commandDrops{ 0 },
        m_commandOversize{ 0 }; ///< Full-command recording counts, separate from the metadata journal.
    std::size_t   m_queueOccupancy{ 0 }, m_commandQueueOccupancy{ 0 }; ///< Nonblocking recorder backlog snapshots.
    std::size_t   m_dmCommandsRetained{ 0 }; ///< Complete command images retained even with recording disabled.
    std::uint64_t m_dmCommandEvictions{ 0 }; ///< Expected bounded full-command history eviction, not source loss.
    AdcCursor    m_adcCursor; ///< Exact bounded forwarding position, distinct from the reader's current source counter.
    std::size_t  m_adcPending{ 0 }; ///< Unforwarded retained ADC samples, not history occupancy or recorder backlog.
    std::int64_t m_adcCutoffNs{ -1 }, m_adcPendingReceiptNs{ -1 }; ///< Actual last copy deadline and deferred receipt.
};

/// Refresh receipt-based health on a copy so cached worker snapshots cannot conceal silence; no source clock mapping.
void refresh_native_observation(
    NativeObservationStatus &status /**< [in,out] Owned diagnostic snapshot. */,
    std::int64_t             nowNs /**< [in] Current local MONOTONIC time. */,
    std::int64_t             staleNs /**< [in] Observation silence threshold, not a control deadline. */ );

/// Fixed single-producer/single-consumer journal queue; overflow never stalls source observation.
class NativeObservationQueue
{
    std::vector<NativeObservation> m_records; ///< Allocated once before either worker starts.
    std::atomic<std::uint64_t>     m_head{ 0 }, m_tail{ 0 }, m_drops{ 0 }; ///< Independent worker cursors.

  public:
    /// Allocate fixed storage; zero capacity is invalid.
    explicit NativeObservationQueue( std::size_t capacity /**< [in] Maximum retained journal records. */ );

    /// Copy or count overflow; only the input worker may call this method.
    bool push( const NativeObservation &record /**< [in] Owned source evidence. */ );

    /// Copy the next record; only the writer may call this method.
    bool pop( NativeObservation &record /**< [out] Oldest owned evidence, unchanged when empty. */ );

    /// Return observed overflow, separate from ADC transport or source gaps.
    std::uint64_t drops() const;

    /// Return a bounded approximate backlog without blocking either queue worker.
    std::size_t queued() const;
};

/// Separate bounded SPSC payload queue; large DM images never enlarge individual ADC journal records.
class NativeCommandQueue
{
    std::vector<NativeObservation> m_records; ///< Owned command metadata, indexed by the SPSC cursors.
    std::vector<float> m_values; ///< Fixed capacity times maximum elements; only actual payload lengths are copied.
    std::size_t        m_maxElements; ///< Immutable payload bound, validated before allocation.
    std::atomic<std::uint64_t> m_head{ 0 }, m_tail{ 0 }, m_drops{ 0 }; ///< Independent producer/consumer progression.

  public:
    /// Allocate bounded full-command storage before either observer worker starts.
    NativeCommandQueue( std::size_t capacity /**< [in] Number of command snapshots. */,
                        std::size_t maxElements /**< [in] Maximum FLOAT elements per command. */ );

    /// Copy one whole command or count overflow; the sole source worker never waits for disk.
    bool push( const NativeObservation &record /**< [in] Stable command snapshot metadata. */,
               std::span<const float>   values /**< [in] Complete original command image. */ );

    /// Copy the oldest command into writer-owned storage; metadata defines the actual payload length.
    bool pop( NativeObservation &record /**< [out] Original command identity and times. */,
              std::span<float>   values /**< [out] Storage for at least maxElements values. */ );

    /// Return payload queue overflow without combining it with source or metadata losses.
    std::uint64_t drops() const;

    /// Return a bounded approximate complete-command backlog.
    std::size_t queued() const;
};

/// Bounded complete-command history; its owner supplies serialization independently of the recording SPSC queue.
class NativeCommandHistory
{
    std::vector<NativeObservation> m_records; ///< Original identities and timestamps for retained command images.
    std::vector<float>             m_values; ///< Fixed storage for every original FLOAT element in each retained image.
    std::size_t   m_maxElements, m_next{ 0 }, m_count{ 0 }; ///< Capacity, insertion point and retained command count.
    std::uint64_t m_evictions{ 0 }; ///< Lifetime retention evictions, distinct from source/recorder loss.

  public:
    /// Allocate bounded retention before native acquisition starts.
    NativeCommandHistory( std::size_t capacity /**< [in] Number of complete commands retained. */,
                          std::size_t maxElements /**< [in] Maximum elements per command. */ );

    /// Retain one complete stable command; evict the oldest only when bounded history fills.
    void append( const NativeObservation &record /**< [in] Command publication evidence only. */,
                 std::span<const float>   values /**< [in] Complete original producer payload. */ );

    /// Copy an original command and metadata without interpreting it as a physical application certificate.
    bool copy( std::size_t        age /**< [in] Zero selects the newest retained command. */,
               NativeObservation &record /**< [out] Owned original metadata. */,
               std::span<float>   values /**< [out] Caller storage for all elements of this command. */ ) const;

    /// Return the number of complete retained commands.
    std::size_t retained() const;

    /// Return expected retention evictions.
    std::uint64_t evictions() const;

    /// Discard retained commands on source disconnection/replacement without changing lifetime eviction counts.
    void clear();
};

/// Single-owner accounting reused in deterministic tests and the native worker; never calls a learner.
class NativeObservationCapture
{
    NativeObservationConfig            m_config; ///< Immutable clocks/selections.
    NativeObservationStatus            m_status; ///< Owned lifetime evidence.
    std::array<NativeObservation, 512> m_dm;     ///< Bounded command identity history for software endpoint joins.
    std::size_t m_dmNext{ 0 }, m_dmCount{ 0 }; ///< Chronological history cursor; cleared on source incarnation changes.
    std::array<MagAOX::dmCompletion::Record, 2>
        m_endpoints; ///< Per-publisher endpoint progression; never cross-join sequences.
    std::array<std::vector<NativeObservation>, 4> m_history; ///< Individual ADC/WFS/camera/DM records, never resampled.
    std::array<std::size_t, 4> m_historyNext{};              ///< Next chronological source-history insertion slots.

  public:
    /// Construct fixed accounting storage without a stream, clock, file or worker.
    explicit NativeObservationCapture(
        const NativeObservationConfig &config /**< [in] Validated observer settings. */ );

    /// Account one owned event, annotate diagnostics, and retain bounded individual source/command histories.
    NativeObservation accept( NativeObservation record /**< [in] Original copied evidence and clock context. */ );

    /// Account transport polling failures without manufacturing a sample/counter.
    void poll( ObservationSource source /**< [in] Input slot. */,
               bool              busy /**< [in] Producer marked writing. */,
               bool              race /**< [in] Closing snapshot differed. */,
               bool              incompatible /**< [in] Unsupported original allocation/layout. */ );

    /// Update silence diagnostics using actual local monotonic time.
    void tick( std::int64_t nowNs /**< [in] Observation time. */ );

    /// Return owned scalar/snapshot evidence; caller provides serialization.
    NativeObservationStatus status() const;

    /// Copy recent individual source records in chronological order; call only on this capture's owning worker.
    std::size_t
    copy_recent( ObservationSource            source /**< [in] ADC, modal, camera or DM role. */,
                 std::span<NativeObservation> output /**< [out] Caller-owned bounded destination. */ ) const;
};
} // namespace DDSPC
#endif
