/** \file syncoZeroTiming.hpp
 * \brief Exact command joins for the finite coordinated zero-command trial.
 */
#ifndef dmSyncopation_syncoZeroTiming_hpp
#define dmSyncopation_syncoZeroTiming_hpp

#include <algorithm>
#include <array>
#include <limits>

#include "../../libMagAOX/common/dmCompletion.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Bounded main-to-SDK measurements; the caller serializes all updates.
/** Reader receipt through native entry includes preparation and handoff. It is
 * not the vector-ready-to-entry SDK queue wait. Negative primary-observer-to-SDK
 * values remain signed because a monitor can observe a main after native entry.
 */
class ZeroTiming
{
  public:
    /// Exact combined publication identity; external timestamps are identity only.
    struct Key
    {
        uint64_t inode{ 0 };   ///< Combined stream's metadata inode.
        uint64_t counter{ 0 }; ///< Combined publication counter.
        int64_t  sec{ 0 };     ///< Raw producer writetime seconds.
        int64_t  nsec{ 0 };    ///< Raw producer writetime nanoseconds.
    };

    /// Online signed local-clock duration population, in nanoseconds.
    struct Stats
    {
        uint64_t count{ 0 };  ///< Number of exact matched measurements.
        double   meanNs{ 0 }; ///< Arithmetic mean.
        double   m2{ 0 };     ///< Sum of squared deviations, ns squared.
        int64_t  minNs{ 0 };  ///< Smallest signed duration.
        int64_t  maxNs{ 0 };  ///< Largest signed duration.
    };

    /// Trial-local measurements and explicit match/transport coverage.
    struct Summary
    {
        Stats    primaryToSdk;         ///< Primary monitor receipt through native SDK entry; signed.
        Stats    controllerToSdk;      ///< Controller receipt through native SDK entry, not pure SDK wait.
        Stats    sdkService;           ///< Native SDK entry through return, for classified main commands.
        uint64_t classified{ 0 };      ///< Unique combined commands classified as genuine mains by the caller.
        uint64_t matched{ 0 };         ///< Classified mains matched to successful native SDK records.
        uint64_t readerMatched{ 0 };   ///< SDK-matched mains also joined to exact reader receipt records.
        uint64_t evicted{ 0 };         ///< Main classifications overwritten before all three records joined.
        uint64_t invalid{ 0 };         ///< Rejected identity, protocol or timing evidence.
        uint64_t duplicates{ 0 };      ///< Repeated classifications or endpoint captures.
        uint64_t sdkGaps{ 0 };         ///< SDK attempt sequences missing between observed distinct records.
        uint64_t sourceResets{ 0 };    ///< SDK publisher session/epoch changes after its first record.
        uint64_t failed{ 0 };          ///< Distinct SDK records that do not qualify successful submission.
        uint64_t primaryCounter{ 0 };  ///< Most recent matched main counter.
        uint64_t combinedCounter{ 0 }; ///< Most recent matched combined counter.
        int64_t  primaryToSdkNs{ 0 };  ///< Most recent signed primary-receipt-to-entry duration.
    };

    /// Extract the exact input identity from a decoded endpoint record.
    static Key key( const dmCompletion::Record &record /**< [in] Coherently captured controller endpoint. */ );

    /// Compare all identity components without mixing external and local clocks.
    static bool equal( const Key &a /**< [in] First identity. */, const Key &b /**< [in] Second identity. */ );

    /// Require a nonzero stream identity and a representable positive producer timestamp.
    static bool valid( const Key &value /**< [in] Combined publication identity. */ );

    /// Clear trial measurements and pending joins without any stream access.
    void reset();

    /// Classify one unambiguous combined publication as a genuine main command.
    void main( const Key &identity /**< [in] Exact combined command acknowledged by the scheduler. */,
               uint64_t   primaryCounter /**< [in] Genuine primary counter associated by the caller. */,
               int64_t    primaryReceiptNs /**< [in] Same-host monotonic primary-monitor receipt. */ );

    /// Retain an exact qualified external-reader release, in either arrival order.
    void reader( const dmCompletion::Record &record /**< [in] Coherent reader-kind endpoint. */,
                 bool                        coherent = true /**< [in] False rejects a torn transport capture. */ );

    /// Retain successful native entry/return evidence and explicit source/sequence coverage.
    void sdk( const dmCompletion::Record &record /**< [in] SDK-kind endpoint with native entry/return stamps. */,
              int64_t                     observedNs /**< [in] Same-host monotonic capture time. */,
              bool                        coherent = true /**< [in] False rejects a torn transport capture. */ );

    /// Return current trial-local measurements; no pending records are consumed.
    const Summary &summary() const;

  private:
    /// One modulo-counter slot retains complete identity to prevent aliasing across restarts.
    struct Entry
    {
        Key                  identity;               ///< Exact publication occupying this slot.
        bool                 occupied{ false };      ///< A key has been retained.
        bool                 haveMain{ false };      ///< Caller supplied an unambiguous main classification.
        bool                 haveReader{ false };    ///< Exact reader receipt is present.
        bool                 haveSdk{ false };       ///< Exact native SDK result is present.
        bool                 counted{ false };       ///< Main-to-SDK and service populations were emitted once.
        bool                 readerCounted{ false }; ///< Controller-receipt population was emitted once.
        uint64_t             primaryCounter{ 0 };    ///< Genuine main identity.
        int64_t              primaryReceiptNs{ 0 };  ///< Monotonic primary observer timestamp.
        dmCompletion::Record reader;                 ///< Frozen reader endpoint.
        dmCompletion::Record sdk;                    ///< Frozen native SDK endpoint.
    };

    std::array<Entry, 256> m_entries{}; ///< Fixed bounded out-of-order join history; no event-path allocation.
    Summary                m_summary;   ///< Aggregate measurements and explicit coverage counters.
    dmCompletion::Record   m_lastSdk;   ///< Latest distinct native endpoint used for source/sequence checks.
    int64_t                m_lastObservedNs{ 0 }; ///< Monotonic capture associated with the latest distinct SDK record.

    /// Select one bounded slot and account for an overwritten incomplete main classification.
    Entry &retain( const Key &identity /**< [in] Valid exact command identity. */ );

    /// Emit each available exact measurement at most once.
    void match( Entry &entry /**< [in,out] Pending classification and endpoint evidence. */ );

    /// Retire pending main identities after a native producer restart.
    void clearPending();

    /// Add one signed, overflow-checked local-clock difference to a duration population.
    static bool addDifference( Stats  &stats /**< [in,out] Duration population. */,
                               int64_t end /**< [in] Same-host monotonic endpoint. */,
                               int64_t begin /**< [in] Same-host monotonic origin. */ );
};

inline ZeroTiming::Key ZeroTiming::key( const dmCompletion::Record &record )
{
    return { record.inputInode, record.inputCounter, record.publicationSec, record.publicationNsec };
}

inline bool ZeroTiming::equal( const Key &a, const Key &b )
{
    return a.inode == b.inode && a.counter == b.counter && a.sec == b.sec && a.nsec == b.nsec;
}

inline bool ZeroTiming::valid( const Key &value )
{
    return value.inode != 0 && value.counter != 0 && value.sec > 0 && value.nsec >= 0 && value.nsec < 1000000000 &&
           value.sec <= ( std::numeric_limits<int64_t>::max() - value.nsec ) / 1000000000;
}

inline void ZeroTiming::reset()
{
    m_entries        = {};
    m_summary        = {};
    m_lastSdk        = {};
    m_lastObservedNs = 0;
}

inline void ZeroTiming::clearPending()
{
    for( auto &entry : m_entries )
    {
        if( entry.haveMain && !entry.readerCounted )
        {
            ++m_summary.evicted;
        }

        entry = {};
    }
}

inline ZeroTiming::Entry &ZeroTiming::retain( const Key &identity )
{
    auto &entry = m_entries[identity.counter % m_entries.size()];

    if( !entry.occupied || !equal( entry.identity, identity ) )
    {
        if( entry.haveMain && !entry.readerCounted )
        {
            ++m_summary.evicted;
        }

        entry          = {};
        entry.identity = identity;
        entry.occupied = true;
    }

    return entry;
}

inline bool ZeroTiming::addDifference( Stats &stats, int64_t end, int64_t begin )
{
    const __int128 wide = __int128( end ) - begin;

    if( end <= 0 || begin <= 0 || wide < std::numeric_limits<int64_t>::min() ||
        wide > std::numeric_limits<int64_t>::max() )
    {
        return false;
    }

    const int64_t value = int64_t( wide );
    const double  delta = double( value ) - stats.meanNs;
    ++stats.count;
    stats.meanNs += delta / stats.count;
    stats.m2 += delta * ( double( value ) - stats.meanNs );
    stats.minNs = stats.count == 1 ? value : std::min( stats.minNs, value );
    stats.maxNs = stats.count == 1 ? value : std::max( stats.maxNs, value );
    return true;
}

inline void ZeroTiming::match( Entry &entry )
{
    if( entry.haveMain && entry.haveSdk && !entry.counted )
    {
        if( !addDifference( m_summary.primaryToSdk, entry.sdk.startedNs, entry.primaryReceiptNs ) ||
            !addDifference( m_summary.sdkService, entry.sdk.finishedNs, entry.sdk.startedNs ) )
        {
            ++m_summary.invalid;
            return;
        }

        entry.counted = true;
        ++m_summary.matched;
        m_summary.primaryCounter  = entry.primaryCounter;
        m_summary.combinedCounter = entry.identity.counter;
        m_summary.primaryToSdkNs  = entry.sdk.startedNs - entry.primaryReceiptNs;
    }

    if( entry.counted && entry.haveReader && !entry.readerCounted )
    {
        if( entry.sdk.startedNs < entry.reader.startedNs ||
            !addDifference( m_summary.controllerToSdk, entry.sdk.startedNs, entry.reader.startedNs ) )
        {
            ++m_summary.invalid;
            return;
        }

        entry.readerCounted = true;
        ++m_summary.readerMatched;
    }
}

inline void ZeroTiming::main( const Key &identity, uint64_t primaryCounter, int64_t primaryReceiptNs )
{
    if( !valid( identity ) || primaryReceiptNs <= 0 )
    {
        ++m_summary.invalid;
        return;
    }

    auto &entry = retain( identity );

    if( entry.haveMain )
    {
        if( entry.primaryCounter != primaryCounter || entry.primaryReceiptNs != primaryReceiptNs )
        {
            ++m_summary.invalid;
        }
        else
        {
            ++m_summary.duplicates;
        }

        return;
    }

    entry.haveMain         = true;
    entry.primaryCounter   = primaryCounter;
    entry.primaryReceiptNs = primaryReceiptNs;
    ++m_summary.classified;
    match( entry );
}

inline void ZeroTiming::reader( const dmCompletion::Record &record, bool coherent )
{
    if( !coherent || !dmCompletion::readersQualified( record ) || !valid( key( record ) ) )
    {
        ++m_summary.invalid;
        return;
    }

    auto &entry = retain( key( record ) );

    if( entry.haveReader )
    {
        if( entry.reader.session != record.session || entry.reader.epoch != record.epoch ||
            entry.reader.sequence != record.sequence || entry.reader.startedNs != record.startedNs ||
            entry.reader.finishedNs != record.finishedNs )
        {
            ++m_summary.invalid;
        }
        else
        {
            ++m_summary.duplicates;
        }

        return;
    }

    entry.haveReader = true;
    entry.reader     = record;
    match( entry );
}

inline void ZeroTiming::sdk( const dmCompletion::Record &record, int64_t observedNs, bool coherent )
{
    if( !coherent || record.kind != dmCompletion::Kind::sdk || !dmCompletion::validIdentity( record ) ||
        !valid( key( record ) ) || observedNs <= 0 || record.finishedNs <= 0 || record.finishedNs > observedNs ||
        record.startedNs > record.finishedNs )
    {
        ++m_summary.invalid;
        return;
    }

    if( m_lastSdk.session != 0 &&
        ( record.session < m_lastSdk.session ||
          ( record.session == m_lastSdk.session &&
            ( record.epoch < m_lastSdk.epoch || record.publisherPid != m_lastSdk.publisherPid ) ) ) )
    {
        ++m_summary.invalid;
        return;
    }

    const bool newSource =
        m_lastSdk.session != 0 && ( record.session != m_lastSdk.session || record.epoch != m_lastSdk.epoch );

    if( newSource )
    {
        clearPending();
        ++m_summary.sourceResets;
        m_lastSdk        = {};
        m_lastObservedNs = 0;
    }

    if( m_lastSdk.session != 0 )
    {
        if( record.sequence == m_lastSdk.sequence && equal( key( record ), key( m_lastSdk ) ) )
        {
            if( record.startedNs != m_lastSdk.startedNs || record.finishedNs != m_lastSdk.finishedNs ||
                record.state != m_lastSdk.state || record.outcome != m_lastSdk.outcome ||
                record.callbackResult != m_lastSdk.callbackResult || record.backendResult != m_lastSdk.backendResult )
            {
                ++m_summary.invalid;
            }
            else
            {
                ++m_summary.duplicates;
            }

            return;
        }

        if( record.sequence <= m_lastSdk.sequence || record.inputCounter <= m_lastSdk.inputCounter ||
            record.startedNs <= m_lastSdk.startedNs || record.finishedNs <= m_lastSdk.finishedNs ||
            observedNs < m_lastObservedNs )
        {
            ++m_summary.invalid;
            return;
        }

        m_summary.sdkGaps += record.sequence - m_lastSdk.sequence - 1;
    }

    m_lastSdk        = record;
    m_lastObservedNs = observedNs;

    if( !dmCompletion::sdkQualified( record ) )
    {
        ++m_summary.failed;
        return;
    }

    auto &entry   = retain( key( record ) );
    entry.haveSdk = true;
    entry.sdk     = record;
    match( entry );
}

inline const ZeroTiming::Summary &ZeroTiming::summary() const
{
    return m_summary;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif // dmSyncopation_syncoZeroTiming_hpp
