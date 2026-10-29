/** \file dmReaderTiming.hpp
 * \brief Bounded observational joins of primary, combined and independent controller reader endpoints and receipts.
 */
#ifndef dev_dmReaderTiming_hpp
#define dev_dmReaderTiming_hpp

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <type_traits>
#include <unordered_map>

#include "../../common/dmReaderTiming.hpp"
#include "../../common/spscQueue.hpp"
#include "dmTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Measure the existing shadow reader-protection budget without changing scheduler decisions.
/** Configure only while all producers and the consumer are stopped. Primary,
 * combined and completion capture methods each have one independent owning
 * thread; ordinary app logic owns update/summary and all joining/statistics.
 * Captures copy fixed records into separate bounded SPSC queues. Resets use an
 * atomic join generation, conservatively preventing any old/new attachment join.
 * Pending storage and retained matched samples are independently capacity capped.
 */
class dmReaderTiming
{
  protected:
    /// Per-source capture envelope; generation changes invalidate already queued peers.
    template <class valueT>
    struct Capture
    {
        valueT   value;             ///< Fixed source observation.
        uint64_t generation{ 0 };   ///< Global attachment/join generation at capture.
        bool     coherent{ false }; ///< Source's repeated metadata bracket succeeded.
    };

    /// Efficient pending lookup; exact raw publication identity is checked when joining.
    struct Key
    {
        uint64_t inode{ 0 };   ///< Source inode.
        uint64_t counter{ 0 }; ///< Source counter.

        /// Compare the shared inode/counter lookup key.
        bool operator==( const Key &other /**< [in] Other key to compare. */ ) const;
    };

    /// Hash only the lookup identity; raw timestamps remain part of qualification.
    struct Hash
    {
        /// Return a mixed inode/counter hash for bounded pending storage.
        size_t operator()( const Key &key /**< [in] Lookup key to hash. */ ) const;
    };

    /// Consumer-only progression baseline for one stream observer.
    struct Progress
    {
        MagAOX::dmReaderTiming::Identity identity;         ///< Latest accepted source identity.
        int64_t                          observedNs{ 0 };  ///< Its monotonic capture time.
        bool                             present{ false }; ///< A progression comparison is possible.
    };

    MagAOX::spscQueue<Capture<MagAOX::dmReaderTiming::Primary>>   m_primaryQueue;    ///< Primary thread's handoff.
    MagAOX::spscQueue<Capture<MagAOX::dmReaderTiming::Combined>>  m_combinedQueue;   ///< Combined thread's handoff.
    MagAOX::spscQueue<Capture<MagAOX::dmReaderTiming::Completed>> m_completionQueue; ///< Completion thread's handoff.

    std::atomic<uint64_t> m_generation{ 1 };     ///< Any source reset invalidates queued observations from older joins.
    uint64_t              m_seenGeneration{ 1 }; ///< Generation owned by the current consumer join populations.
    MagAOX::dmCompletion::Kind m_kind{
        MagAOX::dmCompletion::Kind::completion }; ///< Immutable expected endpoint protocol.
    MagAOX::dmTiming::Outcome m_completionOutcome{
        MagAOX::dmTiming::Outcome::unknown };        ///< Last observed result for same-attempt enrichment.
    MagAOX::dmCompletion::Record m_completionRecord; ///< Frozen endpoint evidence for same-attempt result enrichment.
    bool                         m_enabled{ false }; ///< Immutable configuration while producers run.
    int64_t                      m_windowNs{ 0 };    ///< Positive configured trailing retention, or zero when disabled.
    size_t                       m_capacity{ 0 };    ///< Per-source queue/pending and retained matched-sample bound.

    std::unordered_map<Key, MagAOX::dmReaderTiming::Primary, Hash>  m_primaries; ///< Bounded pending primary evidence.
    std::unordered_map<Key, MagAOX::dmReaderTiming::Combined, Hash> m_combined;  ///< Bounded pending combined evidence.
    std::unordered_map<Key, MagAOX::dmReaderTiming::Completed, Hash>
        m_completed; ///< Bounded pending completion evidence.
    std::multimap<int64_t, MagAOX::dmReaderTiming::Sample>
        m_window; ///< Matched samples ordered by completion observation.

    Progress                    m_primaryProgress;         ///< Consumer progression for primary observations.
    Progress                    m_combinedProgress;        ///< Consumer progression for combined observations.
    bool                        m_haveCompletion{ false }; ///< Completion publisher provenance has been observed.
    uint64_t                    m_completionSession{ 0 };  ///< Current completion publisher startup nonce.
    uint64_t                    m_completionEpoch{ 0 };    ///< Current completion attachment/reset epoch.
    uint64_t                    m_completionSequence{ 0 }; ///< Highest observed callback-attempt sequence.
    MagAOX::dmCompletion::State m_completionState{
        MagAOX::dmCompletion::State::unavailable }; ///< Last state in that attempt.
    Progress m_completionProgress; ///< Highest qualified combined identity within the completion epoch.

    MagAOX::dmReaderTiming::Counters m_total;   ///< Lifetime consumer quality counters.
    MagAOX::dmReaderTiming::Summary  m_summary; ///< Coherent ordinary-thread summary cache.

    /// Require a positive representable external timestamp without mixing its clock with local time.
    static bool publication( const MagAOX::dmReaderTiming::Identity &identity /**< [in] Raw producer identity. */,
                             int64_t &ns /**< [out] External-clock nanoseconds when valid. */ );

    /// Compare every raw producer identity field.
    static bool equal( const MagAOX::dmReaderTiming::Identity &first /**< [in] First producer identity. */,
                       const MagAOX::dmReaderTiming::Identity &second /**< [in] Second producer identity. */ );

    /// Discard pending/window state while preserving lifetime counters and extreme evidence.
    void clear();

    /// Synchronize producer resets and account all observed join-generation transitions.
    void synchronize();

    /// Begin a fresh join epoch for an implicit source identity or publisher transition.
    void newGeneration();

    /// Accept forward coherent stream progression; resets conservatively invalidate other queued peers.
    bool progression( Progress &progress /**< [in,out] Source's consumer progression baseline. */,
                      const MagAOX::dmReaderTiming::Identity &identity /**< [in] New producer identity. */,
                      int64_t                                 observedNs /**< [in] New local capture timestamp. */ );

    /// Store a pending observation, evicting the oldest when its independent fixed cap is reached.
    template <class valueT>
    void retain( std::unordered_map<Key, valueT, Hash> &pending /**< [in,out] Source's pending population. */,
                 const valueT                          &value /**< [in] Accepted source observation. */ );

    /// Explain a removed unmatched completion using the evidence still retained.
    void missing( const MagAOX::dmReaderTiming::Completed &value /**< [in] Completion being removed. */ );

    /// Consume one primary capture using only consumer-owned statistics and pending storage.
    void consume( const Capture<MagAOX::dmReaderTiming::Primary> &capture /**< [in] Fixed primary handoff. */,
                  int64_t                                         nowNs /**< [in] Latest allowed local timestamp. */ );

    /// Consume one combined capture using only consumer-owned statistics and pending storage.
    void consume( const Capture<MagAOX::dmReaderTiming::Combined> &capture /**< [in] Fixed combined handoff. */,
                  int64_t                                          nowNs /**< [in] Latest allowed local timestamp. */ );

    /// Consume completion state, qualify optional receipt provenance once and retain current-publisher attempts.
    void consume( const Capture<MagAOX::dmReaderTiming::Completed> &capture /**< [in] Fixed completion handoff. */,
                  int64_t nowNs /**< [in] Latest allowed local timestamp. */ );

    /// Join all currently available three-way identities once, before expiring unmatched evidence.
    void join();

    /// Remove expired pending/window evidence after every captured queue has received its bounded drain.
    void expire( int64_t nowNs /**< [in] Current monotonic evaluation time. */ );

    /// Recompute retained independent duration populations and publish one coherent cache.
    void calculate( int64_t nowNs /**< [in] Current monotonic evaluation time. */ );

  public:
    /// Allocate all handoff storage while producers and consumer are stopped; zero disables collection.
    void configure( double windowSec /**< [in] Positive trailing retention duration in seconds; zero disables. */,
                    size_t capacity /**< [in] Per-source capture/pending and matched-sample cap; zero disables. */,
                    MagAOX::dmCompletion::Kind kind =
                        MagAOX::dmCompletion::Kind::completion /**< [in] Independently measured endpoint protocol. */ );

    /// Capture fixed primary metadata without allocation, joining, waiting or statistics.
    void primary( const MagAOX::dmReaderTiming::Primary &value /**< [in] Primary observation. */,
                  bool coherent = true /**< [in] Producer identity passed the monitor's metadata bracket. */ );

    /// Capture fixed combined metadata without allocation, joining, waiting or statistics.
    void combined( const MagAOX::dmReaderTiming::Combined &value /**< [in] Combined observation and frozen budget. */,
                   bool coherent = true /**< [in] Producer identity and named trace passed consistency checks. */ );

    /// Capture a fixed completion record without allocation, joining, waiting or statistics.
    void completion( const MagAOX::dmReaderTiming::Completed &value /**< [in] Dedicated completion observation. */,
                     bool coherent = true /**< [in] Completion record passed its coherent shared read. */ );

    /// Invalidate joins when the primary observer's stream attachment changes.
    void resetPrimary();

    /// Invalidate joins when the combined observer's stream attachment changes.
    void resetCombined();

    /// Invalidate joins when the dedicated completion observer's attachment changes.
    void resetCompletion();

    /// Drain at most each queue's captured population, join available identities and recompute statistics.
    void update( int64_t nowNs = dmTiming::monotonicNowNs() /**< [in] Monotonic evaluation time. */ );

    /// Return the immutable enabled configuration while producer threads are running.
    bool enabled() const;

    /// Return the coherent consumer-owned cache, never producer-owned state.
    const MagAOX::dmReaderTiming::Summary &summary() const;
};

inline bool dmReaderTiming::Key::operator==( const Key &other ) const
{
    return inode == other.inode && counter == other.counter;
}

inline size_t dmReaderTiming::Hash::operator()( const Key &key ) const
{
    const auto mixed = key.inode ^ ( key.counter + 0x9e3779b97f4a7c15ULL + ( key.inode << 6 ) + ( key.inode >> 2 ) );
    return static_cast<size_t>( mixed );
}

inline bool dmReaderTiming::publication( const MagAOX::dmReaderTiming::Identity &identity, int64_t &ns )
{
    if( identity.inode == 0 || identity.counter == 0 || identity.sec <= 0 || identity.nsec < 0 ||
        identity.nsec >= 1000000000 )
    {
        return false;
    }

    const __int128 value = static_cast<__int128>( identity.sec ) * 1000000000 + identity.nsec;

    if( value > std::numeric_limits<int64_t>::max() )
    {
        return false;
    }

    ns = static_cast<int64_t>( value );
    return true;
}

inline bool dmReaderTiming::equal( const MagAOX::dmReaderTiming::Identity &first,
                                   const MagAOX::dmReaderTiming::Identity &second )
{
    return first.inode == second.inode && first.counter == second.counter && first.sec == second.sec &&
           first.nsec == second.nsec;
}

inline void dmReaderTiming::configure( double windowSec, size_t capacity, MagAOX::dmCompletion::Kind kind )
{
    m_kind              = kind;
    m_completionOutcome = MagAOX::dmTiming::Outcome::unknown;
    m_completionRecord  = {};
    m_enabled           = mx::math::isFinite( windowSec ) && windowSec > 0 && windowSec < 9e9 && capacity > 0;
    m_windowNs          = m_enabled ? static_cast<int64_t>( windowSec * 1e9 ) : 0;
    m_enabled           = m_enabled && m_windowNs > 0;
    m_capacity          = m_enabled ? capacity : 0;
    m_primaryQueue.configure( m_capacity );
    m_combinedQueue.configure( m_capacity );
    m_completionQueue.configure( m_capacity );
    m_generation.store( 1, std::memory_order_relaxed );
    m_seenGeneration = 1;
    clear();
    m_total              = {};
    m_haveCompletion     = false;
    m_completionSession  = 0;
    m_completionEpoch    = 0;
    m_completionSequence = 0;
    m_completionState    = MagAOX::dmCompletion::State::unavailable;
    m_primaries.reserve( m_capacity );
    m_combined.reserve( m_capacity );
    m_completed.reserve( m_capacity );
    m_summary           = {};
    m_summary.enabled   = m_enabled;
    m_summary.kind      = m_kind;
    m_summary.session   = 1;
    m_summary.windowSec = m_enabled ? windowSec : 0;
    m_summary.capacity  = m_capacity;
}

inline void dmReaderTiming::primary( const MagAOX::dmReaderTiming::Primary &value, bool coherent )
{
    if( m_enabled )
    {
        m_primaryQueue.push( { value, m_generation.load( std::memory_order_acquire ), coherent } );
    }
}

inline void dmReaderTiming::combined( const MagAOX::dmReaderTiming::Combined &value, bool coherent )
{
    if( m_enabled )
    {
        m_combinedQueue.push( { value, m_generation.load( std::memory_order_acquire ), coherent } );
    }
}

inline void dmReaderTiming::completion( const MagAOX::dmReaderTiming::Completed &value, bool coherent )
{
    if( m_enabled )
    {
        m_completionQueue.push( { value, m_generation.load( std::memory_order_acquire ), coherent } );
    }
}

inline void dmReaderTiming::resetPrimary()
{
    if( m_enabled )
    {
        m_generation.fetch_add( 1, std::memory_order_acq_rel );
    }
}

inline void dmReaderTiming::resetCombined()
{
    resetPrimary();
}

inline void dmReaderTiming::resetCompletion()
{
    resetPrimary();
}

inline void dmReaderTiming::clear()
{
    for( const auto &entry : m_completed )
    {
        missing( entry.second );
    }

    m_total.evicted += m_primaries.size() + m_combined.size() + m_completed.size();
    m_primaries.clear();
    m_combined.clear();
    m_completed.clear();
    m_window.clear();
    m_primaryProgress    = {};
    m_combinedProgress   = {};
    m_completionProgress = {};
}

inline void dmReaderTiming::synchronize()
{
    const auto generation = m_generation.load( std::memory_order_acquire );

    if( generation != m_seenGeneration )
    {
        const auto changes = generation - m_seenGeneration;
        clear();
        m_seenGeneration = generation;
        m_total.sourceResets += changes;
        m_summary.session += changes;
    }
}

inline void dmReaderTiming::newGeneration()
{
    m_generation.fetch_add( 1, std::memory_order_acq_rel );
    synchronize();
}

inline bool
dmReaderTiming::progression( Progress &progress, const MagAOX::dmReaderTiming::Identity &identity, int64_t observedNs )
{
    if( progress.present &&
        ( identity.inode != progress.identity.inode || identity.counter < progress.identity.counter ) )
    {
        newGeneration();
    }

    if( progress.present && identity.counter == progress.identity.counter )
    {
        ++m_total.duplicates;

        if( !equal( identity, progress.identity ) )
        {
            ++m_total.identityMismatch;
        }

        return false;
    }

    if( progress.present && observedNs < progress.observedNs )
    {
        ++m_total.invalid;
        return false;
    }

    if( progress.present && identity.counter - progress.identity.counter > 1 )
    {
        ++m_total.counterGaps;
    }

    progress.identity   = identity;
    progress.observedNs = observedNs;
    progress.present    = true;
    return true;
}

template <class valueT>
inline void dmReaderTiming::retain( std::unordered_map<Key, valueT, Hash> &pending, const valueT &value )
{
    Key key;

    if constexpr( std::is_same_v<valueT, MagAOX::dmReaderTiming::Completed> )
    {
        key = { value.record.inputInode, value.record.inputCounter };
    }
    else
    {
        key = { value.input.inode, value.input.counter };
    }

    if( pending.size() >= m_capacity )
    {
        const auto oldest = std::min_element( pending.begin(),
                                              pending.end(),
                                              []( const auto &first, const auto &second )
                                              {
                                                  return first.second.observedNs < second.second.observedNs;
                                              } );

        if constexpr( std::is_same_v<valueT, MagAOX::dmReaderTiming::Completed> )
        {
            missing( oldest->second );
        }

        pending.erase( oldest );
        ++m_total.evicted;
    }

    pending.insert_or_assign( key, value );
}

inline void dmReaderTiming::missing( const MagAOX::dmReaderTiming::Completed &value )
{
    const auto combined = m_combined.find( { value.record.inputInode, value.record.inputCounter } );

    if( combined == m_combined.end() )
    {
        ++m_total.missingCombined;
    }
    else if( m_primaries.find( { combined->second.primaryInode, combined->second.primaryCounter } ) ==
             m_primaries.end() )
    {
        ++m_total.missingPrimary;
    }
}

inline void dmReaderTiming::consume( const Capture<MagAOX::dmReaderTiming::Primary> &capture, int64_t nowNs )
{
    ++m_total.events;
    ++m_total.primary;

    if( capture.generation != m_seenGeneration )
    {
        ++m_total.staleEpoch;
        return;
    }

    if( !capture.coherent )
    {
        ++m_total.incoherent;
        return;
    }

    const auto &value     = capture.value;
    int64_t     published = 0;

    if( value.observedNs <= 0 || value.observedNs > nowNs || !publication( value.input, published ) )
    {
        ++m_total.invalid;
        return;
    }

    if( progression( m_primaryProgress, value.input, value.observedNs ) )
    {
        retain( m_primaries, value );
    }
}

inline void dmReaderTiming::consume( const Capture<MagAOX::dmReaderTiming::Combined> &capture, int64_t nowNs )
{
    ++m_total.events;
    ++m_total.combined;

    if( capture.generation != m_seenGeneration )
    {
        ++m_total.staleEpoch;
        return;
    }

    if( !capture.coherent )
    {
        ++m_total.incoherent;
        return;
    }

    const auto &value     = capture.value;
    int64_t     published = 0;

    if( value.observedNs <= 0 || value.observedNs > nowNs || value.primaryInode == 0 || value.primaryCounter == 0 ||
        value.startNs <= 0 || value.budgetNs < 0 || !publication( value.input, published ) ||
        value.startNs > published )
    {
        ++m_total.invalid;
        return;
    }

    if( progression( m_combinedProgress, value.input, value.observedNs ) )
    {
        retain( m_combined, value );
    }
}

inline void dmReaderTiming::consume( const Capture<MagAOX::dmReaderTiming::Completed> &capture, int64_t nowNs )
{
    ++m_total.events;
    ++m_total.completion;

    if( capture.generation != m_seenGeneration )
    {
        ++m_total.staleEpoch;
        return;
    }

    if( !capture.coherent )
    {
        ++m_total.incoherent;
        return;
    }

    const auto &value  = capture.value;
    const auto &record = value.record;

    if( value.observedNs <= 0 || value.observedNs > nowNs || record.protocolVersion != MagAOX::dmCompletion::version ||
        record.session == 0 || record.epoch == 0 || record.kind != m_kind ||
        record.state > MagAOX::dmCompletion::State::released )
    {
        ++m_total.invalid;
        return;
    }

    if( m_haveCompletion && ( record.session < m_completionSession ||
                              ( record.session == m_completionSession && record.epoch < m_completionEpoch ) ) )
    {
        ++m_total.duplicates;
        ++m_total.invalid;
        return;
    }

    const bool newPublisher =
        m_haveCompletion && ( record.session != m_completionSession || record.epoch != m_completionEpoch );

    if( newPublisher )
    {
        newGeneration();
    }

    if( m_haveCompletion && !newPublisher && record.sequence < m_completionSequence )
    {
        ++m_total.duplicates;
        ++m_total.invalid;
        return;
    }

    const bool sameAttempt = m_haveCompletion && !newPublisher && record.sequence == m_completionSequence;
    const bool enrichment  = sameAttempt && m_kind == MagAOX::dmCompletion::Kind::readers &&
                            m_completionState == MagAOX::dmCompletion::State::released &&
                            record.state == MagAOX::dmCompletion::State::released &&
                            m_completionOutcome == MagAOX::dmTiming::Outcome::unknown &&
                            record.outcome != MagAOX::dmTiming::Outcome::unknown;

    if( sameAttempt && !enrichment &&
        ( m_completionState != MagAOX::dmCompletion::State::busy ||
          record.state == MagAOX::dmCompletion::State::busy ) )
    {
        ++m_total.duplicates;
        return;
    }

    if( m_haveCompletion && !newPublisher && record.sequence > m_completionSequence &&
        record.sequence - m_completionSequence > 1 )
    {
        m_total.sequenceGaps += record.sequence - m_completionSequence - 1;
    }

    m_haveCompletion     = true;
    m_completionSession  = record.session;
    m_completionEpoch    = record.epoch;
    m_completionSequence = record.sequence;
    m_completionState    = record.state;

    const auto completedState = m_kind == MagAOX::dmCompletion::Kind::readers ? MagAOX::dmCompletion::State::released
                                                                              : MagAOX::dmCompletion::State::complete;

    if( record.state != completedState )
    {
        switch( record.state )
        {
        case MagAOX::dmCompletion::State::busy:
            ++m_total.busy;
            break;
        case MagAOX::dmCompletion::State::unavailable:
            ++m_total.unavailable;
            break;
        case MagAOX::dmCompletion::State::error:
            ++m_total.error;
            break;
        case MagAOX::dmCompletion::State::unqualified:
            ++m_total.unqualified;
            break;
        default:
            ++m_total.invalid;
            break;
        }

        return;
    }

    if( !( m_kind == MagAOX::dmCompletion::Kind::readers ? MagAOX::dmCompletion::readersQualified( record )
                                                         : MagAOX::dmCompletion::qualified( record ) ) ||
        value.observedNs < record.finishedNs || value.observedNs - record.finishedNs > m_windowNs )
    {
        ++m_total.invalid;
        return;
    }

    if( enrichment &&
        ( record.inputInode != m_completionRecord.inputInode ||
          record.inputCounter != m_completionRecord.inputCounter ||
          record.publicationSec != m_completionRecord.publicationSec ||
          record.publicationNsec != m_completionRecord.publicationNsec ||
          record.startedNs != m_completionRecord.startedNs || record.finishedNs != m_completionRecord.finishedNs ||
          record.callbackResult != m_completionRecord.callbackResult ) )
    {
        ++m_total.invalid;
        return;
    }

    if( m_kind == MagAOX::dmCompletion::Kind::readers && record.outcome != MagAOX::dmTiming::Outcome::unknown )
    {
        if( ( record.outcome != MagAOX::dmTiming::Outcome::success &&
              record.outcome != MagAOX::dmTiming::Outcome::failed ) ||
            ( record.backendResult != MagAOX::dmCompletion::unavailableResult &&
              ( record.outcome == MagAOX::dmTiming::Outcome::success ) != ( record.backendResult == 0 ) ) )
        {
            ++m_total.invalid;
        }
        else
        {
            ++m_total.sdkResults;

            if( record.outcome == MagAOX::dmTiming::Outcome::success )
            {
                ++m_total.sdkSuccess;
            }
            else
            {
                ++m_total.sdkFailed;
            }
        }
    }

    m_completionOutcome = record.outcome;
    m_completionRecord  = record;

    if( enrichment )
    {
        return;
    }

    const MagAOX::dmReaderTiming::Identity identity{
        record.inputInode, record.inputCounter, record.publicationSec, record.publicationNsec };

    if( m_completionProgress.present && ( identity.inode != m_completionProgress.identity.inode ||
                                          identity.counter < m_completionProgress.identity.counter ) )
    {
        ++m_total.invalid;
        ++m_total.identityMismatch;
        return;
    }

    if( progression( m_completionProgress, identity, value.observedNs ) )
    {
        if( value.receiptNs == 0 )
        {
            ++m_total.receiptUnavailable;
        }
        else if( value.receiptNs <= 0 || value.receiptNs < record.finishedNs || value.receiptNs > value.observedNs )
        {
            ++m_total.receiptInvalid;
        }

        if( m_kind == MagAOX::dmCompletion::Kind::readers && record.outcome != MagAOX::dmTiming::Outcome::unknown )
        {
            ++m_total.firstEnriched;
        }

        retain( m_completed, value );
    }
}

inline void dmReaderTiming::join()
{
    for( auto completed = m_completed.begin(); completed != m_completed.end(); )
    {
        const auto combined = m_combined.find( completed->first );

        if( combined == m_combined.end() )
        {
            ++completed;
            continue;
        }

        const auto primary = m_primaries.find( { combined->second.primaryInode, combined->second.primaryCounter } );

        if( primary == m_primaries.end() )
        {
            ++completed;
            continue;
        }

        const auto                            &record = completed->second.record;
        const MagAOX::dmReaderTiming::Identity identity{
            record.inputInode, record.inputCounter, record.publicationSec, record.publicationNsec };
        int64_t primaryPublication = 0;

        if( !equal( combined->second.input, identity ) || !publication( primary->second.input, primaryPublication ) ||
            primaryPublication > combined->second.startNs )
        {
            ++m_total.identityMismatch;
            ++m_total.invalid;
            m_combined.erase( combined );
            completed = m_completed.erase( completed );
            continue;
        }

        MagAOX::dmReaderTiming::Sample sample;
        sample.primary        = primary->second;
        sample.combined       = combined->second;
        sample.completed      = completed->second;
        sample.baseNs         = std::max( sample.primary.observedNs, sample.combined.observedNs );
        sample.offsetNs       = record.finishedNs - sample.baseNs;
        sample.requiredNs     = std::max<int64_t>( 0, sample.offsetNs );
        sample.observedHoldNs = std::max<int64_t>( 0, sample.completed.observedNs - sample.baseNs );
        const __int128 slack  = static_cast<__int128>( sample.combined.budgetNs ) - sample.offsetNs;

        if( slack > std::numeric_limits<int64_t>::max() || slack < std::numeric_limits<int64_t>::min() )
        {
            ++m_total.invalid;
            m_combined.erase( combined );
            completed = m_completed.erase( completed );
            continue;
        }

        sample.slackNs = static_cast<int64_t>( slack );
        ++m_total.matched;

        if( !m_summary.worstValid || sample.requiredNs > m_summary.maxRequiredHoldNs ||
            ( sample.requiredNs == m_summary.maxRequiredHoldNs && sample.offsetNs > m_summary.worst.offsetNs ) )
        {
            m_summary.worstValid = true;
            m_summary.worst      = sample;
        }

        if( !m_summary.worstNotificationValid || sample.completed.observedNs - record.finishedNs >
                                                     m_summary.worstNotification.completed.observedNs -
                                                         m_summary.worstNotification.completed.record.finishedNs )
        {
            m_summary.worstNotificationValid = true;
            m_summary.worstNotification      = sample;
        }

        m_summary.maxRequiredHoldNs = std::max( m_summary.maxRequiredHoldNs, sample.requiredNs );
        m_summary.maxObservedHoldNs = std::max( m_summary.maxObservedHoldNs, sample.observedHoldNs );

        if( sample.slackNs < 0 )
        {
            ++m_summary.overBudget;
        }

        m_window.emplace( sample.completed.observedNs, sample );

        if( m_window.size() > m_capacity )
        {
            m_window.erase( m_window.begin() );
            ++m_total.evicted;
        }

        m_combined.erase( combined );
        completed = m_completed.erase( completed );
    }
}

inline void dmReaderTiming::expire( int64_t nowNs )
{
    const auto cutoff = nowNs > m_windowNs ? nowNs - m_windowNs : 0;

    for( auto entry = m_completed.begin(); entry != m_completed.end(); )
    {
        if( entry->second.observedNs < cutoff )
        {
            missing( entry->second );
            entry = m_completed.erase( entry );
            ++m_total.expired;
        }
        else
        {
            ++entry;
        }
    }

    const auto expireSource = [this, cutoff]( auto &pending )
    {
        for( auto entry = pending.begin(); entry != pending.end(); )
        {
            if( entry->second.observedNs < cutoff )
            {
                entry = pending.erase( entry );
                ++m_total.expired;
            }
            else
            {
                ++entry;
            }
        }
    };
    expireSource( m_primaries );
    expireSource( m_combined );
    m_window.erase( m_window.begin(), m_window.lower_bound( cutoff ) );
}

inline void dmReaderTiming::calculate( int64_t nowNs )
{
    std::array<MagAOX::timing::Accumulator, MagAOX::dmReaderTiming::index( MagAOX::dmReaderTiming::Metric::count )>
        populations;

    for( const auto &entry : m_window )
    {
        const auto   &sample = entry.second;
        const auto   &record = sample.completed.record;
        const int64_t values[]{ sample.offsetNs,
                                sample.requiredNs,
                                sample.observedHoldNs,
                                record.finishedNs - record.startedNs,
                                sample.completed.observedNs - record.finishedNs,
                                sample.slackNs };

        for( size_t metric = 0; metric < MagAOX::dmReaderTiming::index( MagAOX::dmReaderTiming::Metric::wake );
             ++metric )
        {
            populations[metric].add( values[metric] * 1e-9, entry.first, entry.first );
        }

        const auto receipt = sample.completed.receiptNs;

        if( receipt > 0 && receipt >= record.finishedNs && receipt <= sample.completed.observedNs &&
            ( record.kind != MagAOX::dmCompletion::Kind::readers ||
              record.outcome == MagAOX::dmTiming::Outcome::unknown ) )
        {
            populations[MagAOX::dmReaderTiming::index( MagAOX::dmReaderTiming::Metric::wake )].add(
                ( receipt - record.finishedNs ) * 1e-9, entry.first, entry.first );
            populations[MagAOX::dmReaderTiming::index( MagAOX::dmReaderTiming::Metric::capture )].add(
                ( sample.completed.observedNs - receipt ) * 1e-9, entry.first, entry.first );
        }
    }

    for( size_t metric = 0; metric < populations.size(); ++metric )
    {
        m_summary.metrics[metric] = populations[metric].stats();
    }

    ++m_summary.sequence;
    m_summary.nowNs             = nowNs;
    m_summary.span              = m_summary.metrics[0].span;
    m_summary.age               = m_window.empty() ? -1 : ( nowNs - m_window.rbegin()->first ) * 1e-9;
    m_summary.queued            = m_primaryQueue.queued() + m_combinedQueue.queued() + m_completionQueue.queued();
    m_summary.pendingPrimary    = m_primaries.size();
    m_summary.pendingCombined   = m_combined.size();
    m_summary.pendingCompletion = m_completed.size();
    m_total.droppedPrimary      = m_primaryQueue.dropped();
    m_total.droppedCombined     = m_combinedQueue.dropped();
    m_total.droppedCompletion   = m_completionQueue.dropped();
    m_summary.total             = m_total;
}

inline void dmReaderTiming::update( int64_t nowNs )
{
    if( !m_enabled )
    {
        return;
    }

    if( nowNs <= 0 )
    {
        ++m_total.invalid;
        m_summary.total = m_total;
        return;
    }

    if( m_summary.nowNs > nowNs )
    {
        ++m_total.invalid;
        newGeneration();
    }

    synchronize();
    const auto                                 completionCount = m_completionQueue.queued();
    const auto                                 primaryCount    = m_primaryQueue.queued();
    const auto                                 combinedCount   = m_combinedQueue.queued();
    Capture<MagAOX::dmReaderTiming::Completed> completed;

    for( size_t index = 0; index < completionCount && m_completionQueue.pop( completed ); ++index )
    {
        consume( completed, nowNs );
    }

    Capture<MagAOX::dmReaderTiming::Primary> primary;

    for( size_t index = 0; index < primaryCount && m_primaryQueue.pop( primary ); ++index )
    {
        consume( primary, nowNs );
    }

    Capture<MagAOX::dmReaderTiming::Combined> combined;

    for( size_t index = 0; index < combinedCount && m_combinedQueue.pop( combined ); ++index )
    {
        consume( combined, nowNs );
    }

    synchronize();
    join();
    expire( nowNs );
    calculate( nowNs );
}

inline bool dmReaderTiming::enabled() const
{
    return m_enabled;
}

inline const MagAOX::dmReaderTiming::Summary &dmReaderTiming::summary() const
{
    return m_summary;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
