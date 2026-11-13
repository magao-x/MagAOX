/** \file native_observation.hpp
 * \brief Bounded observation accounting; no exposure inference, applied-input certification or learner interface.
 */
#ifndef PCNATIVE_OBSERVATION_HPP
#define PCNATIVE_OBSERVATION_HPP
#include "adc_reader.hpp"
#include "../../libMagAOX/common/dmCompletion.hpp"
#include <array>
#include <atomic>
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
    std::string  m_directory;         ///< Existing private directory; exclusive new journal, never append/overwrite.
    double       m_durationSec{ 60 }; ///< Finite observation duration, including absent-source time.
    unsigned     m_pollUs{ 50 };      ///< Nominal polling delay; actual coverage/loss is measured.
    std::size_t  m_queueCapacity{ 32768 }; ///< Fixed recording queue; overflow counts instead of blocking input.
    std::int64_t m_staleNs{ 10000000 };    ///< Observer silence diagnostic; never a control freshness assertion.
    std::array<std::size_t, 2> m_modalIndices{
        0, 1 }; ///< Selected calibrated-stream entries, without asserting TT identity.

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
    ObservationInvalidTime        = 1,
    ObservationUnknownClock       = 2,
    ObservationClockMismatch      = 4,
    ObservationInvalidPayload     = 8,
    ObservationProtocolInvalid    = 16,
    ObservationProtocolIncomplete = 32,
    ObservationAssociationMissing = 64,
    ObservationAssociationMatched = 128
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
    bool              m_connected{ false }, m_stale{ true }; ///< Mapping and local silence state only.
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
        m_associations{ 0 };            ///< Metadata matches only, never physical application.
    std::int64_t m_lastReceiptNs{ -1 }; ///< Observer availability high-water for silence checks.
};

/// Thread-safe observer snapshot; every model and physical-application qualification remains false by construction.
struct NativeObservationStatus
{
    AdcStatus m_adcCapture; ///< Independent original decoder counters/rate; never add capture loss to forwarding loss.
    std::array<NativeSourceStatus, observationSources> m_sources; ///< Independent source health and exact IDs.
    std::uint64_t                                      m_adcMissing{ 0 }, m_adcBreaks{ 0 }, m_queueDrops{ 0 },
        m_written{ 0 };                                               ///< Distinct loss populations.
    std::int64_t m_startedNs{ -1 }, m_stoppedNs{ -1 }, m_nowNs{ -1 }; ///< Actual local observation interval.
    bool m_finished{ false }, m_ioError{ false }; ///< Finite recording completion and persistent recording failure.
};

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

  public:
    /// Construct fixed accounting storage without a stream, clock, file or worker.
    explicit NativeObservationCapture(
        const NativeObservationConfig &config /**< [in] Validated observer settings. */ );

    /// Account one owned event, annotate diagnostics, and retain only bounded command identity history.
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
};
} // namespace DDSPC
#endif
