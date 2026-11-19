/** \file syncoPublicationTiming.hpp
 * \brief Bounded timing populations for interleaved primary and auxiliary counter publications.
 */
#ifndef dmSyncopation_syncoPublicationTiming_hpp
#define dmSyncopation_syncoPublicationTiming_hpp

#include <algorithm>
#include <array>
#include <limits>

#include "syncoCadence.hpp"
#include "syncoZeroTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Measure the interleaved trigger publications independently of subsequent dmcomb acknowledgements.
/** Both streams must use the same CLOCK_ISIO producer clock. Their writetimes
 * are publication proxies immediately preceding cnt0 updates, not exact counter
 * store timestamps. Producer-clock errors never subtract an absolute local
 * monotonic timestamp. The optional paired-clock evidence qualifies separate
 * guard-to-publication and complete update-API durations. Cross-source timestamp
 * ordering violations exclude their current adjacency and main commitment;
 * subsequent main cycles can recover without replacing either source.
 */
class PublicationTiming
{
  public:
    /// Full-rate signed nanosecond populations, indexed by the destination slot.
    struct Summary
    {
        unsigned steps{ 0 }; ///< Configured total slots, including the genuine primary publication.
        std::array<ZeroTiming::Stats, 4>
            interval{}; ///< Consecutive labelled publications; zero is last auxiliary to main.
        std::array<ZeroTiming::Stats, 4> intervalError{}; ///< Adjacent interval minus the preceding frozen period/N.
        std::array<ZeroTiming::Stats, 4> phaseError{};    ///< Auxiliary minus its actual main, less its nominal offset.
        std::array<ZeroTiming::Stats, 4>
            publicationError{};                            ///< Raw publication minus its frozen producer-clock target.
        std::array<ZeroTiming::Stats, 4> preCounterLead{}; ///< Raw publication proxy minus the mapped final guard.
        std::array<ZeroTiming::Stats, 4> apiCost{}; ///< Monotonic final guard through complete update-API return.
        uint64_t                         mainObserved{ 0 }; ///< Unique accepted genuine primary publications.
        uint64_t auxiliaryObserved{ 0 }; ///< Unique accepted auxiliary publications, including unacknowledged ones.
        uint64_t paired{ 0 }; ///< Adjacent publications with consecutive slot/cycle labels and source progression.
        uint64_t phaseMatched{ 0 };   ///< Auxiliary publications paired with their same-cycle genuine main.
        uint64_t sourceGaps{ 0 };     ///< Accepted forward jumps in a stream's own counter, not a lost-command count.
        uint64_t labelGaps{ 0 };      ///< Successive retained publications do not represent consecutive intended slots.
        uint64_t invalid{ 0 };        ///< Malformed, inconsistent, reset or regressing source evidence.
        uint64_t duplicates{ 0 };     ///< Exact repeated source publications ignored without changing adjacency.
        uint64_t timingExcluded{ 0 }; ///< Auxiliary publications lacking usable guard/clock/API evidence.
        int64_t  maxBracketNs{ 0 };   ///< Greatest accepted producer/monotonic mapping uncertainty width.
        bool     sourceFault{ false }; ///< Replacement or regression blocks further observations until reset.
    };

    /// Select one through four total slots and clear all retained identities and populations on success.
    bool configure( unsigned steps /**< [in] Total main-plus-auxiliary slots. */ );

    /// Clear measurements and frozen source identities while retaining the configured slot count.
    void reset();

    /// Observe a coherent genuine primary publication, even when dmcomb linkage is unavailable.
    void main( const ZeroTiming::Key &primary /**< [in] Exact primary-stream publication identity. */,
               GridTime               rawGrid /**< [in] Frozen main grid in the same producer clock. */,
               uint64_t               periodQ32 /**< [in] Frozen cycle period in scaled nanoseconds. */ );

    /// Observe a successful auxiliary counter update before any downstream acknowledgement decision.
    void auxiliary( unsigned               step /**< [in] Positive slot within the current genuine main cycle. */,
                    const ZeroTiming::Key &publication /**< [in] Exact auxiliary-stream publication identity. */,
                    int64_t                rawTargetNs /**< [in] Frozen target in the common producer clock. */,
                    int64_t                monoTargetNs /**< [in] The same target mapped to local monotonic time. */,
                    int64_t                guardNs /**< [in] Local monotonic final admission timestamp. */,
                    int64_t                apiReturnNs /**< [in] Local monotonic update-API return timestamp. */,
                    int64_t                offsetNs /**< [in] Frozen producer-clock minus monotonic mapping offset. */,
                    int64_t bracketNs /**< [in] Nonnegative width of that mapping's uncertainty bracket. */ );

    /// Return fixed-storage measurements without consuming them.
    const Summary &summary() const;

  private:
    /// One accepted source publication and its immutable cycle commitment.
    struct Point
    {
        ZeroTiming::Key key;            ///< Stream-local counter and raw publication stamp.
        uint64_t        primary{ 0 };   ///< Genuine primary counter labelling the cycle.
        unsigned        step{ 0 };      ///< Zero for main, otherwise the auxiliary slot.
        GridTime        grid;           ///< Frozen producer-clock main grid and fractional remainder.
        uint64_t        periodQ32{ 0 }; ///< Period committed with this cycle.
    };

    unsigned             m_steps{ 0 };   ///< Configured slot count; zero prevents collection before configuration.
    Summary              m_summary;      ///< Bounded populations and explicit coverage exclusions.
    std::array<Point, 2> m_sources{};    ///< Last main and auxiliary publications; their inodes freeze until reset.
    std::array<bool, 2>  m_haveSource{}; ///< Each source has established its own counter progression baseline.
    Point                m_previous;     ///< Latest accepted interleaved publication used for adjacency.
    Point                m_main;         ///< Genuine main associated with current auxiliary slots.
    bool                 m_havePrevious{ false }; ///< Adjacency evidence survived the latest observation.
    bool                 m_haveMain{ false };     ///< A complete current main commitment is available.

    /// Convert an already validated producer stamp to nanoseconds.
    static int64_t stamp( const ZeroTiming::Key &key /**< [in] Valid positive producer timestamp. */ );

    /// Add one signed nanosecond observation to a stable online population.
    static void add( ZeroTiming::Stats &stats /**< [in,out] Duration or error population. */,
                     int64_t            value /**< [in] Signed nanosecond value. */ );

    /// Subtract a fractional nominal offset without losing subnanosecond cycle arithmetic.
    static int64_t error( int64_t  elapsed /**< [in] Same-clock producer duration. */,
                          uint64_t periodQ32 /**< [in] Frozen scaled cycle period. */,
                          unsigned step /**< [in] Number of nominal subframe intervals. */,
                          unsigned steps /**< [in] Positive total slots in a cycle. */ );

    /// Reject unusable evidence and optionally latch a source fault until an explicit reset.
    void reject( bool sourceFault /**< [in] Replacement or regression invalidates the fixed source contract. */ );

    /// Validate stream-local progression and retain one new labelled publication.
    bool retain( const Point &point /**< [in] Coherent publication with a checked cycle commitment. */,
                 unsigned     source /**< [in] Zero selects main; one selects auxiliary. */ );
};

inline bool PublicationTiming::configure( unsigned steps )
{
    if( steps == 0 || steps > m_summary.interval.size() )
    {
        return false;
    }

    m_steps = steps;
    reset();
    return true;
}

inline void PublicationTiming::reset()
{
    m_summary       = {};
    m_summary.steps = m_steps;
    m_sources       = {};
    m_haveSource    = {};
    m_previous      = {};
    m_main          = {};
    m_havePrevious = m_haveMain = false;
}

inline int64_t PublicationTiming::stamp( const ZeroTiming::Key &key )
{
    return key.sec * 1000000000LL + key.nsec;
}

inline void PublicationTiming::add( ZeroTiming::Stats &stats, int64_t value )
{
    const double delta = double( value ) - stats.meanNs;
    ++stats.count;
    stats.meanNs += delta / stats.count;
    stats.m2 += delta * ( double( value ) - stats.meanNs );
    stats.minNs = stats.count == 1 ? value : std::min( stats.minNs, value );
    stats.maxNs = stats.count == 1 ? value : std::max( stats.maxNs, value );
}

inline int64_t PublicationTiming::error( int64_t elapsed, uint64_t periodQ32, unsigned step, unsigned steps )
{
    const __int128 scale   = MagAOX::dmSyncoSimulation::periodScale;
    const __int128 value   = __int128( elapsed ) * scale * steps - __int128( periodQ32 ) * step;
    const __int128 divisor = scale * steps;
    return int64_t( value >= 0 ? ( value + divisor / 2 ) / divisor : -( ( -value + divisor / 2 ) / divisor ) );
}

inline void PublicationTiming::reject( bool sourceFault )
{
    ++m_summary.invalid;
    m_summary.sourceFault = m_summary.sourceFault || sourceFault;
    m_havePrevious = m_haveMain = false;
}

inline bool PublicationTiming::retain( const Point &point, unsigned source )
{
    if( m_summary.sourceFault )
    {
        reject( false );
        return false;
    }

    const auto &prior = m_sources[source];
    const auto &other = m_sources[1 - source];

    if( ( m_haveSource[source] && point.key.inode != prior.key.inode ) ||
        ( m_haveSource[1 - source] && point.key.inode == other.key.inode ) )
    {
        reject( true );
        return false;
    }

    bool continuous = true;

    if( m_haveSource[source] )
    {
        if( point.key.counter == prior.key.counter )
        {
            if( ZeroTiming::equal( point.key, prior.key ) && point.primary == prior.primary &&
                point.step == prior.step && point.grid.ns == prior.grid.ns &&
                point.grid.fractionQ32 == prior.grid.fractionQ32 && point.periodQ32 == prior.periodQ32 )
            {
                ++m_summary.duplicates;
            }
            else
            {
                reject( true );
            }

            return false;
        }

        if( point.key.counter < prior.key.counter || stamp( point.key ) <= stamp( prior.key ) )
        {
            reject( true );
            return false;
        }
        continuous = prior.key.counter != UINT64_MAX && point.key.counter == prior.key.counter + 1;
    }

    if( m_havePrevious && stamp( point.key ) <= stamp( m_previous.key ) )
    {
        reject( false );
        return false;
    }

    if( !continuous )
    {
        ++m_summary.sourceGaps;
    }

    if( m_havePrevious )
    {
        const bool labelled = point.step == 0
                                  ? m_previous.step == m_steps - 1 && m_previous.primary != UINT64_MAX &&
                                        point.primary == m_previous.primary + 1
                                  : m_previous.primary == point.primary && m_previous.step + 1 == point.step &&
                                        m_previous.grid.ns == point.grid.ns &&
                                        m_previous.grid.fractionQ32 == point.grid.fractionQ32 &&
                                        m_previous.periodQ32 == point.periodQ32;

        if( !labelled )
        {
            ++m_summary.labelGaps;
        }

        if( continuous && labelled )
        {
            const auto elapsed = stamp( point.key ) - stamp( m_previous.key );
            add( m_summary.interval[point.step], elapsed );
            add( m_summary.intervalError[point.step], error( elapsed, m_previous.periodQ32, 1, m_steps ) );
            ++m_summary.paired;
        }
    }
    m_sources[source]    = point;
    m_haveSource[source] = true;
    m_previous           = point;
    m_havePrevious       = true;

    if( source == 0 )
    {
        ++m_summary.mainObserved;
    }
    else
    {
        ++m_summary.auxiliaryObserved;
    }
    return true;
}

inline void PublicationTiming::main( const ZeroTiming::Key &primary, GridTime rawGrid, uint64_t periodQ32 )
{
    if( !m_steps || !ZeroTiming::valid( primary ) || !stepGrid( rawGrid, periodQ32, 0, m_steps ).ns )
    {
        reject( false );
        return;
    }
    const Point point{ primary, primary.counter, 0, rawGrid, periodQ32 };

    if( !retain( point, 0 ) )
    {
        return;
    }
    m_main     = point;
    m_haveMain = true;
    add( m_summary.publicationError[0], stamp( primary ) - rawGrid.ns );
}

inline void PublicationTiming::auxiliary( unsigned               step,
                                          const ZeroTiming::Key &publication,
                                          int64_t                rawTargetNs,
                                          int64_t                monoTargetNs,
                                          int64_t                guardNs,
                                          int64_t                apiReturnNs,
                                          int64_t                offsetNs,
                                          int64_t                bracketNs )
{
    if( !m_steps || !m_haveMain || step == 0 || step >= m_steps || !ZeroTiming::valid( publication ) ||
        rawTargetNs <= 0 || stepGrid( m_main.grid, m_main.periodQ32, step, m_steps ).ns != rawTargetNs )
    {
        reject( false );
        return;
    }
    const Point point{ publication, m_main.primary, step, m_main.grid, m_main.periodQ32 };

    if( !retain( point, 1 ) )
    {
        return;
    }
    add( m_summary.phaseError[step],
         error( stamp( publication ) - stamp( m_main.key ), m_main.periodQ32, step, m_steps ) );
    add( m_summary.publicationError[step], stamp( publication ) - rawTargetNs );
    ++m_summary.phaseMatched;

    const __int128 mappedGuard = __int128( guardNs ) + offsetNs;
    const __int128 lead        = __int128( stamp( publication ) ) - mappedGuard;
    const __int128 targetError = __int128( monoTargetNs ) + offsetNs - rawTargetNs;
    const __int128 uncertainty = ( __int128( bracketNs ) + 1 ) / 2;

    if( monoTargetNs <= 0 || guardNs <= 0 || apiReturnNs < guardNs || bracketNs < 0 || mappedGuard <= 0 ||
        mappedGuard > std::numeric_limits<int64_t>::max() || lead < std::numeric_limits<int64_t>::min() ||
        lead > std::numeric_limits<int64_t>::max() || targetError < -uncertainty || targetError > uncertainty )
    {
        ++m_summary.timingExcluded;
        return;
    }
    add( m_summary.preCounterLead[step], int64_t( lead ) );
    add( m_summary.apiCost[step], apiReturnNs - guardNs );
    m_summary.maxBracketNs = std::max( m_summary.maxBracketNs, bracketNs );
}

inline const PublicationTiming::Summary &PublicationTiming::summary() const
{
    return m_summary;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif // dmSyncopation_syncoPublicationTiming_hpp
