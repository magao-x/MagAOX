/** \file adc_input.hpp
 * \brief Bounded raw ADC capture and calibration, independent of DDSPC training and native transport.
 */
#ifndef PCADC_INPUT_HPP
#define PCADC_INPUT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace DDSPC
{

/// Startup-only calibration and retention policy; the default profile describes accelSim, not AD7689.
struct AdcConfig
{
    std::string           m_stream{ "acc000" };      ///< Existing input image; never created or written by this reader.
    std::string           m_profile{ "accelSim12" }; ///< Recorded calibration identity, not an inferred hardware type.
    std::array<double, 2> m_zero{ 2048, 2048 };      ///< Per-channel ADC zero in counts.
    std::array<double, 2> m_voltsPerCount{ .0083799, .0083799 }; ///< Positive volts/count calibration.
    std::array<double, 2> m_sensitivity{ 1.02, 1.02 };           ///< Positive sensor sensitivity in volts/(m/s^2).
    std::array<double, 2> m_polarity{ 1, 1 };                    ///< Explicit axis polarity, either -1 or +1.
    std::uint32_t m_minCode{ 0 }, m_maxCode{ 4095 }; ///< Configured UINT16 encoding range; not fixed to 12 bits.
    bool          m_rejectRails{ true };             ///< Rail samples are invalid by default.
    std::size_t   m_capacity{ 4096 };                ///< Retained samples and bounded startup allocation.
    std::int64_t  m_maxAgeNs{ 20000000 };            ///< Maximum acquisition age, including during source silence.
    std::int64_t  m_maxIntervalNs{ 1000000 };    ///< Larger acquisition gaps break retention even if counters advance.
    std::int64_t  m_clockToleranceNs{ 1000000 }; ///< Maximum clock bracket/offset change before invalidation.
};

/// Owned producer metadata used to bracket a payload copy; adapted from accelFFCtrl's input transport.
struct AdcHeader
{
    std::uint32_t m_axes{ 0 }, m_width{ 0 }, m_height{ 0 }, m_depth{ 0 }; ///< Image geometry.
    std::uint64_t m_counter{ 0 }, m_index{ 0 }; ///< Global publication count and newest plane.
    bool          m_valid{ false }, m_uint16{ false }, m_cpu{ false }, m_write{ false }; ///< Layout/write checks.

    /// Compare all metadata surrounding a bounded payload copy.
    bool operator==( const AdcHeader &other /**< [in] Second metadata snapshot. */ ) const = default;
};

/// Raw pair with original per-plane counter and realtime timestamps, copied together.
struct AdcFrame
{
    std::array<std::uint16_t, 2> m_raw{};        ///< Channel 0 then channel 1, without FLOAT reinterpretation.
    std::uint64_t                m_counter{ 0 }; ///< Per-plane publication identity.
    std::int64_t                 m_acquiredRealNs{ -1 }, m_writtenRealNs{ -1 }; ///< Original producer clock values.
};

struct AdcClock;

/// Transport interface for existing-only native mappings and deterministic offline fixtures.
class AdcSource
{
  public:
    /// Release a source implementation without publishing anything.
    virtual ~AdcSource() = default;

    /// Copy current metadata and write state.
    virtual AdcHeader header() const = 0;

    /// Copy an owned raw pair and its metadata within the original allocation bounds.
    virtual AdcFrame frame( std::size_t index /**< [in] Temporal plane. */ ) const = 0;

    /// Read local clocks after a stable copy, so publication/receipt ordering is checked without hindsight.
    virtual AdcClock clock() const = 0;
};

/// Local realtime observation bracketed by monotonic clock reads.
struct AdcClock
{
    std::int64_t m_realNs{ 0 }, m_monoNs{ 0 }, m_uncertaintyNs{ 0 }; ///< Realtime, midpoint and half-span.
};

/// Timestamped physical sample retained for a future WFS adapter; it is never a DDSPC validity assertion.
struct AdcSample
{
    AdcFrame              m_frame;                           ///< Original counts and producer metadata.
    std::array<double, 2> m_acceleration{};                  ///< Calibrated X/Y acceleration in m/s^2.
    std::uint64_t         m_generation{ 0 }, m_segment{ 0 }; ///< Source incarnation and uninterrupted sample segment.
    std::int64_t m_acquiredNs{ 0 }, m_publishedNs{ 0 }, m_receivedNs{ 0 }; ///< Mapped acquisition/write and receipt.
};

/// ADC health only; no state grants prediction, training, or output authority.
enum class AdcState
{
    Disabled,
    Disconnected,
    Waiting,
    Streaming,
    Incompatible,
    Invalid,
    ClockFault,
    Stale
};

/// Return a stable ADC diagnostic string.
const char *adc_state( AdcState state /**< [in] Latest ADC health. */ );

/// Validate startup calibration, stream name, integer bounds and memory limits.
void validate_adc_config( const AdcConfig &config /**< [in] Immutable reader configuration. */ );

/// Latest ADC sample and lifetime accounting, copied at application cadence.
struct AdcStatus
{
    AdcState      m_state{ AdcState::Disconnected }; ///< Input health; latest values are usable only with valid=true.
    bool          m_valid{ false };                  ///< Fresh calibrated data exists in the current segment.
    AdcSample     m_latest; ///< Latest observed pair; acceleration is cleared on rejection and valid gates use.
    AdcHeader     m_source; ///< Last observed native layout and producer counter.
    std::uint64_t m_device{ 0 }, m_inode{ 0 }, m_generation{ 0 }, m_segment{ 0 }; ///< Source/session identity.
    std::uint64_t m_samples{ 0 }, m_dropped{ 0 }, m_invalid{ 0 },
        m_duplicates{ 0 }; ///< Accepted/lost/rejected/poll totals.
    std::uint64_t m_races{ 0 }, m_busy{ 0 }, m_restarts{ 0 }, m_overwrites{ 0 },
        m_timingGaps{ 0 };                    ///< Copy and retention accounting.
    std::size_t m_retained{ 0 };              ///< Current contiguous ring occupancy.
    double      m_rateHz{ 0 }, m_ageUs{ -1 }; ///< Rate from acquisition timestamps, and monotonic age.
};

/// Caller-owned position in one capture source; a new identity never inherits an old counter baseline.
struct AdcCursor
{
    std::uint64_t m_device{ 0 }, m_inode{ 0 }, m_generation{ 0 }; ///< Backing identity and reader source incarnation.
    std::uint64_t m_segment{ 0 }, m_counter{ 0 }; ///< Last observed segment and last copied/established counter.
    bool m_bound{ false }; ///< A retained source counter established the baseline; pre-attachment loss is unknown.
};

/// Coherent bounded-copy evidence; missing counters overlap capture loss and must not be added to it.
struct AdcBatch
{
    AdcStatus     m_status; ///< Health, source layout and lifetime counters observed under the same capture lock.
    AdcCursor     m_next;   ///< Position to retain after processing this owned batch, including an empty discontinuity.
    std::size_t   m_count{ 0 };   ///< Samples copied oldest first into caller-owned storage.
    std::uint64_t m_missing{ 0 }; ///< Unavailable counters since a known baseline in the same source incarnation.
    bool          m_discontinuity{ false }; ///< Source/segment changed or established history was evicted.
    bool          m_more{ false };          ///< More eligible retained samples remain after the bounded output fills.
};

/// Single-owner bounded capture processor; callers serialize access, with no native or hardware operations.
class AdcCapture
{
    AdcConfig              m_config;      ///< Immutable encoding/calibration and age policy.
    AdcStatus              m_status;      ///< Latest sample and lifetime counters.
    std::vector<AdcSample> m_ring;        ///< Preallocated sample retention, oldest entry overwritten only when full.
    std::vector<AdcFrame>  m_batch;       ///< Preallocated bounded stable-copy scratch.
    std::size_t            m_next{ 0 };   ///< Next ring insertion slot.
    std::uint64_t          m_cursor{ 0 }; ///< Last considered source counter, including rejected frames.
    std::uint64_t          m_observedCounter{ 0 }; ///< Newest stable publication, including undrained producer history.
    bool                   m_bound{ false }, m_haveClock{ false }; ///< Cursor and clock anchor validity.
    bool                   m_clockFault{ false }; ///< Clock faults latch until a new source session is established.
    AdcClock               m_anchor;              ///< Stable session realtime-to-monotonic mapping.
    std::int64_t  m_lastTime{ -1 }, m_firstRateTime{ -1 }, m_lastNow{ -1 }; ///< Segment timing/high-water values.
    std::int64_t  m_highTime{ -1 }; ///< Acquisition high-water persists across gaps/invalid samples within one source.
    std::uint64_t m_rateIntervals{ 0 }; ///< Intervals in the current rate estimate.

    /// Break retention and rate evidence without losing lifetime accounting or source identity.
    void break_segment( AdcState state /**< [in] New health state. */ );

    /// Check the local clock bracket and preserve one mapping for a source session.
    bool check_clock( const AdcClock &clock /**< [in] Local observation after payload capture. */ );

  public:
    /// Validate and allocate fixed capture storage before the acquisition worker starts.
    explicit AdcCapture( const AdcConfig &config /**< [in] Startup policy. */ );

    /// Begin a new source incarnation, retaining lifetime loss/restart diagnostics.
    void attached( std::uint64_t device /**< [in] Backing device. */, std::uint64_t inode /**< [in] Backing inode. */ );

    /// Invalidate retained evidence after mapping loss or worker shutdown.
    void disconnected();

    /// Copy and consume one bounded, metadata-bracketed batch; returns accepted sample count.
    std::size_t drain( const AdcSource &source /**< [in] Existing native mapping or offline source. */ );

    /// Expire data during silence and reject monotonic clock reversal.
    void tick( std::int64_t nowNs /**< [in] Current monotonic time. */ );

    /// Copy diagnostics under caller-provided serialization.
    AdcStatus status() const;

    /// Copy retained samples oldest first, outside acquisition; no pointers into the source or ring escape.
    std::vector<AdcSample> samples() const;

    /// Copy only new eligible samples into bounded storage without allocating or changing capture/decoding state.
    AdcBatch
    copy_after( const AdcCursor     &cursor /**< [in] Previous consumer position; default starts at oldest retained. */,
                std::span<AdcSample> output /**< [out] Nonempty caller-owned bounded sample storage. */,
                std::int64_t availableByNs /**< [in] Latest permitted decoder receipt in the common clock. */ ) const;
};

} // namespace DDSPC
#endif // PCADC_INPUT_HPP
