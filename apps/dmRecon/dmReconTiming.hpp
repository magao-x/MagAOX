/** \file dmReconTiming.hpp
 * \brief Diagnostic pairing of dmRecon publications and subsequent loop triggers.
 */

#ifndef dmReconTiming_hpp
#define dmReconTiming_hpp

#include <array>
#include <cstdint>
#include <ctime>
#include <limits>

namespace MagAOX
{
namespace app
{

/// Compare publication completion with the immediately following loop trigger.
/** This observer neither gates publication nor proves exclusion from mfilt reads. The caller must serialize
 * all calls and supply coherent metadata in one clock domain. Increase the generation on every trigger-stream
 * reconnect or counter reset. Missing observations remain unknown; callback receipt time is never substituted
 * for the source write time. Four retained observations permit bounded callback reordering without allocation.
 * Bracketing a delta publication between triggers does not establish which mfilt iteration caused that delta.
 */
class dmReconTiming
{
  public:
    /// Identity and source timestamp of one observed loop trigger.
    struct Trigger
    {
        uint64_t generation{ 0 }; ///< Monotonically increasing, nonzero attachment generation.

        uint64_t inode{ 0 }; ///< Nonzero identity of the attached trigger stream.

        uint64_t counter{ 0 }; ///< Publication counter within this attachment.

        timespec writeTime{}; ///< Source publication time, never observer receipt time.
    };

    /// Newly completed diagnostic classifications from one operation.
    struct Update
    {
        unsigned beforeNextTrigger{ 0 }; ///< Publications completed strictly before the exact next trigger.

        unsigned late{ 0 }; ///< Publications completed at or after the exact next trigger.

        unsigned unknown{ 0 }; ///< Publications whose timing could not be established.

        int64_t marginNs{ 0 }; ///< Next-trigger time minus completion for the classified publication.
    };

  private:
    std::array<Trigger, 4> m_history{}; ///< Recent trigger identities, independent of callback arrival order.

    unsigned m_size{ 0 }; ///< Number of occupied history entries.

    uint64_t m_generation{ 0 }; ///< Most recently accepted attachment generation.

    uint64_t m_inode{ 0 }; ///< Stream identity belonging to the current generation.

    bool m_invalid{ false }; ///< Conflicting metadata invalidates this generation until reset/reconnect.

    bool m_pending{ false }; ///< One publication awaits its exact next trigger.

    Trigger m_baseline{}; ///< Trigger associated with the pending command snapshot.

    int64_t m_completed{ 0 }; ///< Pending publication completion, in nanoseconds.

    int64_t m_sourcePublished{ 0 }; ///< Pending command's source publication time, in the same clock domain.

    /// Convert a nonzero normalized timestamp without integer overflow.
    static bool nanoseconds( int64_t        &value, /**< [out] Timestamp in nanoseconds on success. */
                             const timespec &time   /**< [in] Timestamp to validate. */
    );

    /// Finish the pending publication as unknown, if any.
    void discard( Update &update /**< [in,out] Classifications returned to the caller. */ );

    /// Adopt a new generation or reject obsolete/conflicting stream identities.
    bool attach( const Trigger &trigger, /**< [in] Incoming stream identity. */
                 Update        &update   /**< [in,out] Classifications returned to the caller. */
    );

    /// Retain consistent source metadata, detecting counter resets and clock reversals.
    void remember( const Trigger &trigger, /**< [in] Validated trigger metadata. */
                   int64_t        stamp,   /**< [in] Trigger timestamp in nanoseconds. */
                   Update        &update   /**< [in,out] Classifications returned to the caller. */
    );

    /// Resolve the pending publication if the exact next trigger is retained.
    void resolve( Update &update /**< [in,out] Classifications returned to the caller. */ );

  public:
    /// Retire pending diagnostics and clear observations after a known discontinuity.
    Update reset();

    /// Observe source metadata; delayed or duplicate callbacks do not constitute new triggers.
    Update observe( const Trigger &trigger /**< [in] Coherent trigger identity and source timestamp. */ );

    /// Record completion relative to a captured baseline, using its time as the source boundary.
    Update published( const Trigger  &baseline, /**< [in] Trigger captured at the source boundary. */
                      const timespec &completed /**< [in] Time sampled after publication completes. */
    );

    /// Record completion and require the command publication to lie between exact adjacent triggers.
    /** A second publication retires an unresolved previous publication as unknown. A counter gap alone is
     * never evidence that the intervening trigger occurred after completion. An observer baseline whose next
     * trigger already predates the command is unverified, rather than a falsely reported deadline miss.
     */
    Update published( const Trigger  &baseline,       /**< [in] Trigger captured with the command snapshot. */
                      const timespec &completed,      /**< [in] Time sampled after publication completes. */
                      const timespec &sourcePublished /**< [in] Command publication time in the trigger clock. */
    );
};

inline bool dmReconTiming::nanoseconds( int64_t &value, const timespec &time )
{
    if( time.tv_sec <= 0 || time.tv_nsec < 0 || time.tv_nsec >= 1000000000 ||
        time.tv_sec > ( std::numeric_limits<int64_t>::max() - time.tv_nsec ) / 1000000000 )
    {
        return false;
    }

    value = static_cast<int64_t>( time.tv_sec ) * 1000000000 + time.tv_nsec;
    return true;
}

inline void dmReconTiming::discard( Update &update )
{
    if( m_pending )
    {
        ++update.unknown;
        m_pending = false;
    }
}

inline bool dmReconTiming::attach( const Trigger &trigger, Update &update )
{
    if( trigger.generation == 0 || trigger.inode == 0 || trigger.generation < m_generation )
    {
        return false;
    }

    if( trigger.generation > m_generation )
    {
        discard( update );
        m_size       = 0;
        m_generation = trigger.generation;
        m_inode      = trigger.inode;
        m_invalid    = false;
    }
    else if( trigger.inode != m_inode )
    {
        m_invalid = true;
        discard( update );
    }

    return !m_invalid;
}

inline void dmReconTiming::remember( const Trigger &trigger, int64_t stamp, Update &update )
{
    unsigned oldest = 0;

    for( unsigned n = 0; n < m_size; ++n )
    {
        int64_t previous = 0;
        nanoseconds( previous, m_history[n].writeTime );

        if( ( trigger.counter == m_history[n].counter && stamp != previous ) ||
            ( trigger.counter > m_history[n].counter && stamp <= previous ) ||
            ( trigger.counter < m_history[n].counter && stamp >= previous ) )
        {
            m_invalid = true;
            discard( update );
            return;
        }

        if( trigger.counter == m_history[n].counter )
        {
            return;
        }

        if( m_history[n].counter < m_history[oldest].counter )
        {
            oldest = n;
        }
    }

    if( m_size < m_history.size() )
    {
        m_history[m_size++] = trigger;
    }
    else if( trigger.counter > m_history[oldest].counter )
    {
        m_history[oldest] = trigger;
    }
}

inline void dmReconTiming::resolve( Update &update )
{
    if( !m_pending )
    {
        return;
    }

    for( unsigned n = 0; n < m_size; ++n )
    {
        if( m_history[n].counter == m_baseline.counter + 1 )
        {
            int64_t next = 0;
            nanoseconds( next, m_history[n].writeTime );

            if( next <= m_sourcePublished )
            {
                discard( update );
                return;
            }

            update.marginNs = next - m_completed;

            if( update.marginNs > 0 )
            {
                ++update.beforeNextTrigger;
            }
            else
            {
                ++update.late;
            }

            m_pending = false;
            return;
        }
    }
}

inline dmReconTiming::Update dmReconTiming::reset()
{
    Update update;
    discard( update );
    m_size       = 0;
    m_generation = 0;
    m_inode      = 0;
    m_invalid    = false;
    return update;
}

inline dmReconTiming::Update dmReconTiming::observe( const Trigger &trigger )
{
    Update update;

    if( !attach( trigger, update ) )
    {
        return update;
    }

    int64_t stamp = 0;

    if( !nanoseconds( stamp, trigger.writeTime ) )
    {
        discard( update );
        return update;
    }

    remember( trigger, stamp, update );
    resolve( update );
    return update;
}

inline dmReconTiming::Update dmReconTiming::published( const Trigger &baseline, const timespec &completed )
{
    return published( baseline, completed, baseline.writeTime );
}

inline dmReconTiming::Update
dmReconTiming::published( const Trigger &baseline, const timespec &completed, const timespec &sourcePublished )
{
    Update update;
    discard( update );

    int64_t start  = 0;
    int64_t end    = 0;
    int64_t source = 0;

    if( !attach( baseline, update ) || !nanoseconds( start, baseline.writeTime ) || !nanoseconds( end, completed ) ||
        !nanoseconds( source, sourcePublished ) || source < start || source > end ||
        baseline.counter == std::numeric_limits<uint64_t>::max() )
    {
        ++update.unknown;
        return update;
    }

    remember( baseline, start, update );

    if( m_invalid )
    {
        ++update.unknown;
        return update;
    }

    m_baseline        = baseline;
    m_completed       = end;
    m_sourcePublished = source;
    m_pending         = true;
    resolve( update );
    return update;
}

} // namespace app
} // namespace MagAOX

#endif // dmReconTiming_hpp
