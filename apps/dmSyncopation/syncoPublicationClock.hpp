/** \file syncoPublicationClock.hpp
 * \brief Bounded producer-clock to monotonic mapping for finite counter-publication trials.
 */
#ifndef dmSyncopation_syncoPublicationClock_hpp
#define dmSyncopation_syncoPublicationClock_hpp

#include <cstdint>
#include <limits>

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Qualify paired clock reads without stream access, system calls or callback-arrival phase assumptions.
/** The caller supplies a monotonic/producer-clock/monotonic bracket. The
 * producer clock must be the same CLOCK_ISIO used for stream writetimes; no
 * realtime/TAI offset is assumed. Mapping uses the midpoint of the qualified
 * external-minus-monotonic offset interval. Its bounds describe clock-read
 * uncertainty, not the later distance from a metadata timestamp to cnt0++.
 * A discontinuity begins a new epoch so callers can discard future grid fits.
 */
class PublicationClock
{
  public:
    /// Current mapping, explicit qualification coverage and its source epoch.
    struct Status
    {
        bool     valid{ false };     ///< The latest sample supplies a usable offset; rejected samples never reuse it.
        uint64_t epoch{ 1 };         ///< Changes on reset, malformed/regressing clocks or an offset discontinuity.
        uint64_t attempted{ 0 };     ///< Supplied clock brackets, including rejected captures.
        uint64_t qualified{ 0 };     ///< Positive, narrow and continuous clock brackets accepted.
        uint64_t invalid{ 0 };       ///< Nonpositive or malformed clock captures.
        uint64_t wideBracket{ 0 };   ///< Brackets exceeding the configured capture limit.
        uint64_t discontinuity{ 0 }; ///< Regressing clocks or incompatible successive offset intervals.
        uint64_t mapped{ 0 };        ///< Positive recent publication timestamps mapped successfully.
        uint64_t mapInvalid{ 0 };    ///< Malformed or unrepresentable publication timestamps/mappings.
        uint64_t unavailable{ 0 };   ///< Mapping requested without a qualified current clock sample.
        uint64_t futurePublication{ 0 };     ///< Publication postdates the sampled producer clock.
        uint64_t stalePublication{ 0 };      ///< Publication age exceeds the configured recent-input bound.
        uint64_t projected{ 0 };             ///< Bounded past or future producer times projected into the local clock.
        uint64_t projectionInvalid{ 0 };     ///< Malformed or unrepresentable requested projections.
        uint64_t projectionUnavailable{ 0 }; ///< Projection requested without a qualified current clock sample.
        uint64_t projectionOutside{ 0 };     ///< Requested time lies outside the configured past/future horizon.
        int64_t  monoBeforeNs{ 0 }, externalNs{ 0 }, monoAfterNs{ 0 }; ///< Latest supplied raw clock bracket.
        int64_t  bracketNs{ 0 };                         ///< Latest representable nonnegative bracket width.
        int64_t  offsetLowerNs{ 0 }, offsetUpperNs{ 0 }; ///< Qualified producer-minus-monotonic interval.
        int64_t  offsetNs{ 0 }; ///< Midpoint producer-minus-monotonic offset, rounded down to integer nanoseconds.
        int64_t  publicationLowerNs{ 0 }, publicationUpperNs{ 0 }; ///< Latest mapping/projection's local bounds.
        int64_t  mappedNs{ 0 };          ///< Midpoint local time; zero after any failed mapping/projection/sample.
        int64_t  bracketLimitNs{ 1000 }; ///< Maximum accepted clock-read uncertainty width.
        int64_t  maxAgeNs{ 4000000 };    ///< Maximum producer-clock distance from the pair for mapping/projection.
    };

    /// Select positive bounded capture/age limits and nonnegative continuity tolerance, then reset.
    bool configure( int64_t bracketLimitNs = 1000 /**< [in] Maximum clock-read bracket, ns. */,
                    int64_t continuityNs   = 1000 /**< [in] Offset-interval discontinuity tolerance, ns. */,
                    int64_t maxAgeNs       = 4000000 /**< [in] Maximum recent publication age, ns. */ );

    /// Begin a new source epoch with empty clock history and zero coverage counters.
    void reset();

    /// Qualify one exact-order monotonic/producer-clock/monotonic capture.
    bool sample( int64_t monoBeforeNs /**< [in] Monotonic timestamp immediately before reading the producer clock. */,
                 int64_t externalNs /**< [in] Declared producer-clock timestamp sampled inside the bracket. */,
                 int64_t monoAfterNs /**< [in] Monotonic timestamp immediately after that clock read. */ );

    /// Map a recent positive producer publication; zero denotes explicitly counted unusable evidence.
    int64_t map( int64_t publicationNs /**< [in] Raw publication timestamp in the declared producer clock. */ );

    /// Project a bounded past/future producer time using the current qualified offset and horizon.
    /** Future targets and subsequently observed auxiliary publications may postdate
     * this pair. The caller retains the pair/epoch with the frozen target; no
     * later pair may silently retime that occurrence.
     */
    int64_t
    project( int64_t externalNs /**< [in] Producer-clock target or observation within the configured horizon. */ );

    /// Return qualification counters and current bounds without consuming them.
    const Status &status() const;

  private:
    int64_t m_bracketLimitNs{ 1000 }; ///< Startup clock-read uncertainty limit.
    int64_t m_continuityNs{ 1000 };   ///< Startup slack between successive offset intervals.
    int64_t m_maxAgeNs{ 4000000 };    ///< Startup recent-input bound, including clock-contract mismatch protection.
    Status  m_status;                 ///< Caller-serialized current mapping and coverage.
    bool    m_havePair{ false };      ///< A previous qualified clock bracket is retained for continuity checks.
    int64_t m_previousExternalNs{ 0 }, m_previousAfterNs{ 0 }; ///< Last qualified producer/local capture endpoints.
    int64_t m_previousLowerNs{ 0 }, m_previousUpperNs{ 0 };    ///< Last qualified external-minus-monotonic interval.

    /// Revoke the current mapping and continuity history without discarding accumulated coverage.
    void newEpoch();

    /// Clear the latest mapped publication so stale successful values cannot survive rejection.
    void clearMapping();

    /// Convert a prequalified bounded producer time, allowing one-ppm relative drift in its bounds.
    int64_t convert( int64_t externalNs /**< [in] Positive producer time within the qualified pair's horizon. */ );
};

inline bool PublicationClock::configure( int64_t bracketLimitNs, int64_t continuityNs, int64_t maxAgeNs )
{
    if( bracketLimitNs <= 0 || bracketLimitNs > 1000000000 || continuityNs < 0 || continuityNs > 1000000000 ||
        maxAgeNs <= 0 || maxAgeNs > 1000000000 )
    {
        return false;
    }

    m_bracketLimitNs = bracketLimitNs;
    m_continuityNs   = continuityNs;
    m_maxAgeNs       = maxAgeNs;
    reset();
    return true;
}

inline void PublicationClock::clearMapping()
{
    m_status.publicationLowerNs = m_status.publicationUpperNs = m_status.mappedNs = 0;
}

inline void PublicationClock::newEpoch()
{
    m_status.epoch         = m_status.epoch == UINT64_MAX ? 1 : m_status.epoch + 1;
    m_status.valid         = false;
    m_status.offsetLowerNs = m_status.offsetUpperNs = m_status.offsetNs = 0;
    m_havePair                                                          = false;
    m_previousExternalNs = m_previousAfterNs = m_previousLowerNs = m_previousUpperNs = 0;
    clearMapping();
}

inline void PublicationClock::reset()
{
    const auto epoch        = m_status.epoch;
    m_status                = {};
    m_status.epoch          = epoch;
    m_status.bracketLimitNs = m_bracketLimitNs;
    m_status.maxAgeNs       = m_maxAgeNs;
    newEpoch();
}

inline bool PublicationClock::sample( int64_t monoBeforeNs, int64_t externalNs, int64_t monoAfterNs )
{
    ++m_status.attempted;
    m_status.valid         = false;
    m_status.monoBeforeNs  = monoBeforeNs;
    m_status.externalNs    = externalNs;
    m_status.monoAfterNs   = monoAfterNs;
    m_status.bracketNs     = 0;
    m_status.offsetLowerNs = m_status.offsetUpperNs = m_status.offsetNs = 0;
    clearMapping();

    if( monoBeforeNs <= 0 || externalNs <= 0 || monoAfterNs < monoBeforeNs )
    {
        ++m_status.invalid;
        newEpoch();
        return false;
    }

    m_status.bracketNs = monoAfterNs - monoBeforeNs;

    if( m_status.bracketNs > m_bracketLimitNs )
    {
        ++m_status.wideBracket;
        return false;
    }

    const int64_t lower   = externalNs - monoAfterNs;
    const int64_t upper   = externalNs - monoBeforeNs;
    const auto    elapsed = __int128( monoAfterNs ) - m_previousAfterNs;
    // One ppm allows a small rate difference without accepting a clock step as capture jitter.
    const auto tolerance = __int128( m_continuityNs ) + ( elapsed > 0 ? ( elapsed + 999999 ) / 1000000 : 0 );

    if( m_havePair && ( monoBeforeNs < m_previousAfterNs || externalNs < m_previousExternalNs ||
                        __int128( lower ) > __int128( m_previousUpperNs ) + tolerance ||
                        __int128( upper ) < __int128( m_previousLowerNs ) - tolerance ) )
    {
        ++m_status.discontinuity;
        newEpoch();
        return false;
    }

    m_previousExternalNs   = externalNs;
    m_previousAfterNs      = monoAfterNs;
    m_previousLowerNs      = lower;
    m_previousUpperNs      = upper;
    m_havePair             = true;
    m_status.offsetLowerNs = lower;
    m_status.offsetUpperNs = upper;
    const auto center      = __int128( lower ) + upper;
    m_status.offsetNs      = int64_t( center >= 0 ? center / 2 : -( ( -center + 1 ) / 2 ) );
    m_status.valid         = true;
    ++m_status.qualified;
    return true;
}

inline int64_t PublicationClock::convert( int64_t externalNs )
{
    const auto delta = __int128( externalNs ) - m_status.externalNs;
    const auto drift = ( ( delta < 0 ? -delta : delta ) + 999999 ) / 1000000;
    const auto lower = __int128( externalNs ) - m_status.offsetUpperNs - drift;
    const auto upper = __int128( externalNs ) - m_status.offsetLowerNs + drift;

    if( lower <= 0 || upper < lower || upper > std::numeric_limits<int64_t>::max() )
    {
        return 0;
    }

    m_status.publicationLowerNs = int64_t( lower );
    m_status.publicationUpperNs = int64_t( upper );
    m_status.mappedNs           = int64_t( ( lower + upper + 1 ) / 2 );
    return m_status.mappedNs;
}

inline int64_t PublicationClock::map( int64_t publicationNs )
{
    clearMapping();

    if( !m_status.valid )
    {
        ++m_status.unavailable;
        return 0;
    }

    if( publicationNs <= 0 )
    {
        ++m_status.mapInvalid;
        return 0;
    }

    if( publicationNs > m_status.externalNs )
    {
        ++m_status.futurePublication;
        return 0;
    }

    if( m_status.externalNs - publicationNs > m_maxAgeNs )
    {
        ++m_status.stalePublication;
        return 0;
    }

    const auto result = convert( publicationNs );

    if( !result || m_status.publicationUpperNs > m_status.monoAfterNs )
    {
        ++m_status.mapInvalid;
        clearMapping();
        return 0;
    }
    ++m_status.mapped;
    return result;
}

inline int64_t PublicationClock::project( int64_t externalNs )
{
    clearMapping();

    if( !m_status.valid )
    {
        ++m_status.projectionUnavailable;
        return 0;
    }

    if( externalNs <= 0 )
    {
        ++m_status.projectionInvalid;
        return 0;
    }

    const auto delta = __int128( externalNs ) - m_status.externalNs;

    if( delta < -__int128( m_maxAgeNs ) || delta > m_maxAgeNs )
    {
        ++m_status.projectionOutside;
        return 0;
    }
    const auto result = convert( externalNs );

    if( !result )
    {
        ++m_status.projectionInvalid;
        return 0;
    }
    ++m_status.projected;
    return result;
}

inline const PublicationClock::Status &PublicationClock::status() const
{
    return m_status;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif // dmSyncopation_syncoPublicationClock_hpp
