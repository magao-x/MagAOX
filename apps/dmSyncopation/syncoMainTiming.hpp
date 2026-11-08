/** \file syncoMainTiming.hpp
 * \brief Bounded main-identity capture and nonwaiting disabled-mode classification.
 */
#ifndef dmSyncopation_syncoMainTiming_hpp
#define dmSyncopation_syncoMainTiming_hpp

#include <ImageStreamIO/ImageStreamIO.h>

#include "../../libMagAOX/common/dmTimingEvents.hpp"
#include "../../libMagAOX/common/spscQueue.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Capture main identities under the scheduler's existing cycle mutex without publishing shared memory.
/** The ordinary application thread is the sole queue consumer. Disabled-mode
 * snapshots need exact primary/output progression and stable named-primary
 * process timing; uncertain or overwritten observations are discarded. In the
 * supported counter-sum combiner, a main is its first recomputation containing
 * the new primary, including a coalesced last auxiliary update. The trace's
 * primary counter is sampled at output publication, so process start must also
 * follow primary publication; the counter alone cannot identify consumed data.
 */
class MainTiming
{
  public:
    using Key    = dmTimingEvents::Key;
    using Record = dmTimingEvents::Record;

    /// One coherent output identity and its producer's first process-trace row.
    struct Combined
    {
        Key      key;                                    ///< Frozen combined publication.
        uint64_t primaryInode{ 0 }, primaryCounter{ 0 }; ///< Named primary sampled by the output updater.
        int64_t  writer{ 0 };                            ///< Positive process identity reported by the producer.
        int64_t  started{ 0 }, updated{ 0 };             ///< Process timestamps in the image producer's clock.
    };

    /// Allocate the bounded handoff before either owner starts; zero disables capture.
    void configure( size_t capacity /**< [in] Maximum queued immutable records. */ );

    /// Synchronize mode and scheduler epochs, returning the generation frozen for this callback.
    uint64_t synchronize( bool     enabled /**< [in] Requested scheduler state at the cycle boundary. */,
                          uint64_t steps /**< [in] Configured main-plus-auxiliary count. */,
                          uint64_t schedulerGeneration /**< [in] Existing scheduler reset generation. */ );

    /// Break classification continuity after resource replacement, reset or loss.
    void reset();

    /// Queue an already-qualified main with its original callback generation.
    void capture( const Key &key /**< [in] Exact qualified combined publication. */,
                  uint64_t   primaryInode /**< [in] Genuine primary stream incarnation. */,
                  uint64_t   primaryCounter /**< [in] Genuine primary publication counter. */,
                  uint64_t   generation /**< [in] Mode generation frozen before classification. */ );

    /// Inspect one primary/output snapshot without waiting, mutating images or reading clocks.
    void passive( const IMAGE *primary /**< [in] Borrowed primary metadata. */,
                  const IMAGE *combined /**< [in] Borrowed independent output attachment. */ );

    /// Copy one immutable captured record for ordinary-thread publication.
    bool pop( Record &record /**< [out] Captured main or mode record. */ );

    /// Return the currently queued population for a bounded ordinary-thread drain.
    size_t queued() const;

    /// Return producer-side queue losses since configuration.
    uint64_t dropped() const;

    /// Return the current classification epoch; zero means capture is disabled or uninitialized.
    uint64_t generation() const;

    /// Read a single coherent publication identity without retries.
    static bool sample( const IMAGE *image /**< [in] Borrowed stream. */,
                        Key         &key /**< [out] Stable publication identity. */ );

    /// Read an output and its named-primary process trace without retries.
    static bool sample( const IMAGE *image /**< [in] Borrowed combined stream. */,
                        Combined    &combined /**< [out] Stable identity and process trace. */ );

  private:
    bool              m_exhausted{ false }; ///< Sequence/generation exhaustion disables capture until configure.
    spscQueue<Record> m_queue;              ///< Serialized producer and ordinary-thread consumer handoff.
    uint64_t          m_generation{ 0 }, m_sequence{ 0 }; ///< Never reuse a mode epoch or logical capture sequence.
    uint64_t          m_schedulerGeneration{ 0 }, m_steps{ 0 };   ///< Last synchronized scheduler configuration.
    bool              m_enabled{ false }, m_modePending{ false }; ///< Frozen mode and an unqueued mode announcement.
    Key               m_primary, m_combined; ///< Previous passive snapshot; never crosses a mode or continuity change.
    Key               m_lastMain; ///< Most recently captured combined key, for duplicate and incarnation checks.
    uint64_t          m_lastPrimaryInode{ 0 }, m_lastPrimaryCounter{ 0 }; ///< Most recent captured primary identity.

    /// Clear both passive history and accepted-primary continuity without modifying the queue.
    void clearHistory();

    /// Publish a pending mode marker once, retaining it after queue loss.
    void mode();

    /// Assign one logical sequence and enqueue, advancing the epoch on loss.
    bool push( Record record /**< [in] Fixed record populated with the current mode. */ );

    /// Test strict producer-clock order without converting absolute timestamps.
    static bool before( const Key &first /**< [in] Earlier publication. */,
                        const Key &second /**< [in] Later publication. */ );

    /// Require the output to contain the selected primary and stable process timing.
    static bool qualifies( const Key      &primary /**< [in] Exact observed primary. */,
                           const Combined &combined /**< [in] Exact output and process trace. */ );
};

inline void MainTiming::configure( size_t capacity )
{
    m_queue.configure( capacity );
    m_generation = m_sequence = m_schedulerGeneration = m_steps = 0;
    m_enabled = m_modePending = m_exhausted = false;
    clearHistory();
}

inline void MainTiming::clearHistory()
{
    m_primary = m_combined = m_lastMain = {};
    m_lastPrimaryInode = m_lastPrimaryCounter = 0;
}

inline uint64_t MainTiming::synchronize( bool enabled, uint64_t steps, uint64_t schedulerGeneration )
{
    if( !m_queue.capacity() || m_exhausted )
    {
        return 0;
    }

    if( !m_generation || enabled != m_enabled || steps != m_steps || schedulerGeneration != m_schedulerGeneration )
    {
        m_enabled             = enabled;
        m_steps               = steps;
        m_schedulerGeneration = schedulerGeneration;
        reset();
    }

    mode();
    return m_generation;
}

inline void MainTiming::reset()
{
    if( !m_queue.capacity() || m_exhausted )
    {
        return;
    }

    if( m_generation == UINT64_MAX )
    {
        m_exhausted   = true;
        m_modePending = false;
        clearHistory();
        return;
    }

    ++m_generation;
    m_modePending = true;
    clearHistory();
}

inline bool MainTiming::push( Record record )
{
    if( m_exhausted || m_sequence == UINT64_MAX )
    {
        m_exhausted   = true;
        m_modePending = false;
        clearHistory();
        return false;
    }

    record.sequence   = ++m_sequence;
    record.generation = m_generation;
    record.enabled    = m_enabled;
    record.steps      = m_steps;

    if( m_queue.push( record ) )
    {
        return true;
    }

    reset();
    return false;
}

inline void MainTiming::mode()
{
    if( m_modePending && !m_exhausted )
    {
        Record record;
        record.kind   = dmTimingEvents::Kind::mode;
        m_modePending = !push( record );
    }
}

inline void MainTiming::capture( const Key &key, uint64_t primaryInode, uint64_t primaryCounter, uint64_t generation )
{
    if( m_exhausted || !generation || generation != m_generation || !dmTimingEvents::valid( key ) || !primaryInode ||
        !primaryCounter )
    {
        return;
    }

    if( m_lastPrimaryCounter )
    {
        if( primaryInode == m_lastPrimaryInode && primaryCounter == m_lastPrimaryCounter )
        {
            return;
        }

        if( primaryInode != m_lastPrimaryInode || m_lastPrimaryCounter == UINT64_MAX ||
            primaryCounter != m_lastPrimaryCounter + 1 || key.inode != m_lastMain.inode ||
            key.counter <= m_lastMain.counter || !before( m_lastMain, key ) )
        {
            reset();
            return;
        }
    }

    mode();

    if( generation != m_generation )
    {
        return;
    }

    Record record;
    record.kind           = dmTimingEvents::Kind::main;
    record.key            = key;
    record.primaryInode   = primaryInode;
    record.primaryCounter = primaryCounter;

    if( push( record ) )
    {
        m_lastMain           = key;
        m_lastPrimaryInode   = primaryInode;
        m_lastPrimaryCounter = primaryCounter;
    }
}

inline bool MainTiming::before( const Key &first, const Key &second )
{
    return first.sec < second.sec || ( first.sec == second.sec && first.nsec < second.nsec );
}

inline bool MainTiming::qualifies( const Key &primary, const Combined &combined )
{
    const auto publication = primary.sec * 1000000000 + primary.nsec;
    const auto output      = combined.key.sec * 1000000000 + combined.key.nsec;
    return combined.primaryInode == primary.inode && combined.primaryCounter == primary.counter &&
           combined.writer > 0 && combined.started >= publication && combined.updated >= combined.started &&
           output >= combined.updated;
}

inline bool MainTiming::sample( const IMAGE *image, Key &key )
{
    const auto *metadata = image ? image->md : nullptr;

    if( !metadata || metadata->write )
    {
        return false;
    }

    const Key value{ metadata->inode, metadata->cnt0, metadata->writetime.tv_sec, metadata->writetime.tv_nsec };
    std::atomic_thread_fence( std::memory_order_acquire );

    if( metadata->write || metadata->inode != value.inode || metadata->cnt0 != value.counter ||
        metadata->writetime.tv_sec != value.sec || metadata->writetime.tv_nsec != value.nsec ||
        !dmTimingEvents::valid( value ) )
    {
        return false;
    }

    key = value;
    return true;
}

inline bool MainTiming::sample( const IMAGE *image, Combined &combined )
{
    Key key;

    if( !sample( image, key ) || !image->streamproctrace || !image->md->NBproctrace )
    {
        return false;
    }

    const auto    *trace   = image->streamproctrace;
    const auto     rows    = image->md->NBproctrace;
    const auto     trigger = trace[0].trigger_inode;
    const auto     counter = trace[0].cnt0;
    const auto     writer  = trace[0].procwrite_PID;
    const timespec started{ trace[0].ts_procstart.tv_sec, trace[0].ts_procstart.tv_nsec };
    const timespec updated{ trace[0].ts_streamupdate.tv_sec, trace[0].ts_streamupdate.tv_nsec };
    const Key      start{ 1, 1, started.tv_sec, started.tv_nsec }, update{ 1, 1, updated.tv_sec, updated.tv_nsec };
    std::atomic_thread_fence( std::memory_order_acquire );
    Key verified;

    if( !sample( image, verified ) || !dmTimingEvents::equal( key, verified ) || image->md->NBproctrace != rows ||
        trace[0].trigger_inode != trigger || trace[0].cnt0 != counter || trace[0].procwrite_PID != writer ||
        trace[0].ts_procstart.tv_sec != started.tv_sec || trace[0].ts_procstart.tv_nsec != started.tv_nsec ||
        trace[0].ts_streamupdate.tv_sec != updated.tv_sec || trace[0].ts_streamupdate.tv_nsec != updated.tv_nsec ||
        !dmTimingEvents::valid( start ) || !dmTimingEvents::valid( update ) )
    {
        return false;
    }

    combined = {
        key, trigger, counter, writer, start.sec * 1000000000 + start.nsec, update.sec * 1000000000 + update.nsec };
    return true;
}

inline void MainTiming::passive( const IMAGE *primary, const IMAGE *combined )
{
    if( m_exhausted || !m_generation || m_enabled || !m_queue.capacity() )
    {
        return;
    }

    Key      current;
    Combined output;

    if( !sample( primary, current ) || !sample( combined, output ) )
    {
        if( m_primary.inode || m_combined.inode )
        {
            reset();
        }

        return;
    }

    if( dmTimingEvents::equal( current, m_primary ) )
    {
        return;
    }

    const auto generation  = m_generation;
    const bool have        = m_primary.inode && m_combined.inode;
    const bool primaryNext = have && current.inode == m_primary.inode && m_primary.counter != UINT64_MAX &&
                             current.counter == m_primary.counter + 1 && before( m_primary, current );
    const bool sameOutput = have && dmTimingEvents::equal( output.key, m_combined );
    const bool nextOutput = have && output.key.inode == m_combined.inode && m_combined.counter != UINT64_MAX &&
                            output.key.counter == m_combined.counter + 1 && before( m_combined, output.key );

    if( have && ( !primaryNext || ( !sameOutput && !nextOutput ) ) )
    {
        reset();
    }
    else if( primaryNext && nextOutput )
    {
        if( before( m_combined, current ) && qualifies( current, output ) )
        {
            capture( output.key, current.inode, current.counter, generation );
        }
        else if( before( m_combined, m_primary ) && qualifies( m_primary, output ) )
        {
            capture( output.key, m_primary.inode, m_primary.counter, generation );
        }
    }

    // Retain only snapshots from this callback; an output gap never becomes a main label.
    m_primary  = current;
    m_combined = output.key;
}

inline bool MainTiming::pop( Record &record )
{
    return m_queue.pop( record );
}

inline size_t MainTiming::queued() const
{
    return m_queue.queued();
}

inline uint64_t MainTiming::dropped() const
{
    return m_queue.dropped();
}

inline uint64_t MainTiming::generation() const
{
    return m_exhausted ? 0 : m_generation;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
