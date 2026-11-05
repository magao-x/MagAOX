/** \file syncoZeroPhase.hpp
 * \brief Bounded same-clock combined-publication phase measurements for finite zero trials.
 */
#ifndef dmSyncopation_syncoZeroPhase_hpp
#define dmSyncopation_syncoZeroPhase_hpp

#include <array>

#include "../../libMagAOX/common/dmSyncoSimulation.hpp"
#include "syncoZeroTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Measure every acknowledged combined command independently of sampled event logging.
/** Producer timestamps are subtracted only from producer timestamps. Frozen
 * monotonic grids identify a cycle's commitment; they are never subtracted from
 * an absolute producer timestamp. The auxiliary and combined publishers must
 * use the same CLOCK_ISIO clock. These are publication proxies, not timestamps
 * at the selected controller semaphore post or its subsequent wakeup.
 */
class ZeroPhase
{
  public:
    /// Full-rate populations in nanoseconds, indexed by the destination step.
    struct Summary
    {
        unsigned                         steps{ 0 }; ///< Configured total step count, including the genuine main.
        std::array<ZeroTiming::Stats, 4> interval{}; ///< Adjacent combined-publication intervals; zero is last-to-main.
        std::array<ZeroTiming::Stats, 4> intervalError{}; ///< Adjacent interval minus the preceding frozen period/N.
        std::array<ZeroTiming::Stats, 4> phaseError{}; ///< Auxiliary minus its actual main, less nominal step*period/N.
        std::array<ZeroTiming::Stats, 4> transport{};  ///< Auxiliary publication through its acknowledged combination.
        uint64_t observed{ 0 }; ///< Valid labelled combined identities presented to the collector, including repeats.
        uint64_t paired{ 0 };   ///< Exact adjacent publications with consecutive step and genuine-cycle labels.
        uint64_t phaseMatched{ 0 }; ///< Auxiliary publications compared with their retained genuine main.
        uint64_t invalid{ 0 };      ///< Malformed inputs, conflicting duplicate identities or regressing producer time.
        uint64_t duplicates{ 0 };   ///< Exact repeated labelled observations ignored without breaking adjacency.
        uint64_t counterGaps{ 0 };  ///< Successive observations changed inode or omitted/reordered combined counters.
        uint64_t labelGaps{ 0 };    ///< Successive observations do not represent consecutive frozen slots.
        uint64_t phaseExcluded{ 0 };     ///< Auxiliary observations lacking a matching same-cycle main commitment.
        uint64_t transportExcluded{ 0 }; ///< Auxiliary observations lacking usable same-clock publication evidence.
    };

    /// Select one through four total steps and clear all measurements on success.
    bool configure( unsigned steps /**< [in] Total steps, including the genuine main. */ );

    /// Clear statistics and pending identities while retaining the configured step count.
    void reset();

    /// Retain one labelled genuine main or successfully acknowledged auxiliary publication.
    void
    observe( uint64_t               primary /**< [in] Genuine primary counter identifying this cycle. */,
             unsigned               step /**< [in] Zero for the genuine main; otherwise its auxiliary slot. */,
             unsigned               steps /**< [in] Frozen total steps for this occurrence. */,
             const ZeroTiming::Key &key /**< [in] Exact coherent combined-publication identity. */,
             int64_t                referenceGridNs /**< [in] Frozen, unadvanced monotonic main-grid reference. */,
             uint64_t               periodQ32 /**< [in] Frozen period in scaled nanoseconds. */,
             const ZeroTiming::Key &auxiliaryPublication = {} /**< [in] Same-clock auxiliary identity, if any. */ );

    /// Return bounded full-rate measurements without consuming them.
    const Summary &summary() const;

  private:
    /// One labelled publication retained for adjacency or relative-main comparisons.
    struct Point
    {
        ZeroTiming::Key key;            ///< Exact producer identity, including its raw publication timestamp.
        uint64_t        primary{ 0 };   ///< Genuine cycle label.
        unsigned        step{ 0 };      ///< Main or auxiliary slot within the cycle.
        int64_t         gridNs{ 0 };    ///< Immutable monotonic main reference, used only for commitment equality.
        uint64_t        periodQ32{ 0 }; ///< Period committed with this occurrence.
    };

    unsigned m_steps{ 0 };            ///< Configured total steps; zero prevents observations before configuration.
    Summary  m_summary;               ///< Fixed-size populations copied into ordinary telemetry snapshots.
    Point    m_previous;              ///< Latest accepted publication used for exact adjacency.
    Point    m_main;                  ///< Current genuine main used for same-cycle phase differences.
    bool     m_havePrevious{ false }; ///< Previous publication remains available and unambiguous.
    bool     m_haveMain{ false };     ///< A coherent same-cycle main commitment remains available.

    /// Convert a validated raw publication timestamp to nanoseconds.
    static int64_t stamp( const ZeroTiming::Key &key /**< [in] Valid positive producer timestamp. */ );

    /// Add one signed nanosecond value to an online population.
    static void add( ZeroTiming::Stats &stats /**< [in,out] Online population. */,
                     int64_t            value /**< [in] Signed duration or error. */ );

    /// Subtract a fractional frozen subframe offset and round its signed error to the nearest nanosecond.
    static int64_t error( int64_t  elapsed /**< [in] Same-clock producer duration. */,
                          uint64_t periodQ32 /**< [in] Frozen scaled cycle period. */,
                          unsigned step /**< [in] Number of subframe intervals to subtract. */,
                          unsigned steps /**< [in] Positive total steps per cycle. */ );
};

inline bool ZeroPhase::configure( unsigned steps )
{
    if( steps == 0 || steps > m_summary.interval.size() )
    {
        return false;
    }

    m_steps = steps;
    reset();
    return true;
}

inline void ZeroPhase::reset()
{
    m_summary       = {};
    m_summary.steps = m_steps;
    m_previous      = {};
    m_main          = {};
    m_havePrevious  = false;
    m_haveMain      = false;
}

inline int64_t ZeroPhase::stamp( const ZeroTiming::Key &key )
{
    return key.sec * 1000000000LL + key.nsec;
}

inline void ZeroPhase::add( ZeroTiming::Stats &stats, int64_t value )
{
    const double delta = double( value ) - stats.meanNs;
    ++stats.count;
    stats.meanNs += delta / stats.count;
    stats.m2 += delta * ( double( value ) - stats.meanNs );
    stats.minNs = stats.count == 1 ? value : std::min( stats.minNs, value );
    stats.maxNs = stats.count == 1 ? value : std::max( stats.maxNs, value );
}

inline int64_t ZeroPhase::error( int64_t elapsed, uint64_t periodQ32, unsigned step, unsigned steps )
{
    const __int128 scale   = MagAOX::dmSyncoSimulation::periodScale;
    const __int128 value   = __int128( elapsed ) * scale * steps - __int128( periodQ32 ) * step;
    const __int128 divisor = scale * steps;
    return int64_t( value >= 0 ? ( value + divisor / 2 ) / divisor : -( ( -value + divisor / 2 ) / divisor ) );
}

inline void ZeroPhase::observe( uint64_t               primary,
                                unsigned               step,
                                unsigned               steps,
                                const ZeroTiming::Key &key,
                                int64_t                referenceGridNs,
                                uint64_t               periodQ32,
                                const ZeroTiming::Key &auxiliaryPublication )
{
    if( !m_steps || steps != m_steps || step >= steps || !primary || !ZeroTiming::valid( key ) ||
        referenceGridNs <= 0 || !periodQ32 ||
        periodQ32 > uint64_t( 1000000000 ) * MagAOX::dmSyncoSimulation::periodScale )
    {
        ++m_summary.invalid;
        m_havePrevious = false;
        m_haveMain     = false;
        return;
    }

    ++m_summary.observed;
    const Point   current{ key, primary, step, referenceGridNs, periodQ32 };
    const int64_t currentNs = stamp( key );

    if( m_havePrevious && ZeroTiming::equal( key, m_previous.key ) )
    {
        if( primary == m_previous.primary && step == m_previous.step && referenceGridNs == m_previous.gridNs &&
            periodQ32 == m_previous.periodQ32 )
        {
            ++m_summary.duplicates;
        }
        else
        {
            ++m_summary.invalid;
            m_havePrevious = false;
            m_haveMain     = false;
        }

        return;
    }

    if( m_havePrevious )
    {
        const bool adjacent = key.inode == m_previous.key.inode && m_previous.key.counter != UINT64_MAX &&
                              key.counter == m_previous.key.counter + 1;
        const bool labelled = step == 0 ? m_previous.step == steps - 1 && m_previous.primary != UINT64_MAX &&
                                              primary == m_previous.primary + 1
                                        : m_previous.primary == primary && m_previous.step + 1 == step &&
                                              m_previous.gridNs == referenceGridNs && m_previous.periodQ32 == periodQ32;

        if( !adjacent )
        {
            ++m_summary.counterGaps;
        }

        if( !labelled )
        {
            ++m_summary.labelGaps;
        }

        if( currentNs <= stamp( m_previous.key ) )
        {
            ++m_summary.invalid;
            m_havePrevious = false;
            m_haveMain     = false;
            return;
        }

        if( adjacent && labelled )
        {
            const auto elapsed = currentNs - stamp( m_previous.key );
            add( m_summary.interval[step], elapsed );
            add( m_summary.intervalError[step], error( elapsed, m_previous.periodQ32, 1, steps ) );
            ++m_summary.paired;
        }
    }

    if( step == 0 )
    {
        m_main     = current;
        m_haveMain = true;
    }
    else
    {
        if( m_haveMain && m_main.primary == primary && m_main.key.inode == key.inode &&
            m_main.key.counter < key.counter && m_main.gridNs == referenceGridNs && m_main.periodQ32 == periodQ32 &&
            stamp( m_main.key ) < currentNs )
        {
            add( m_summary.phaseError[step], error( currentNs - stamp( m_main.key ), periodQ32, step, steps ) );
            ++m_summary.phaseMatched;
        }
        else
        {
            ++m_summary.phaseExcluded;
        }

        if( ZeroTiming::valid( auxiliaryPublication ) && auxiliaryPublication.inode != key.inode &&
            stamp( auxiliaryPublication ) <= currentNs )
        {
            add( m_summary.transport[step], currentNs - stamp( auxiliaryPublication ) );
        }
        else
        {
            ++m_summary.transportExcluded;
        }
    }

    m_previous     = current;
    m_havePrevious = true;
}

inline const ZeroPhase::Summary &ZeroPhase::summary() const
{
    return m_summary;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif // dmSyncopation_syncoZeroPhase_hpp
