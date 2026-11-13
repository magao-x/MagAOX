/** \file loopTimingSynco.hpp
 * \brief Bounded, ordered joins of qualified main commands and native SDK timestamps.
 */
#ifndef loopTimingMonitor_loopTimingSynco_hpp
#define loopTimingMonitor_loopTimingSynco_hpp

#include <algorithm>
#include <deque>
#include <vector>

#include "../../libMagAOX/common/dmTimingEvents.hpp"

namespace MagAOX
{
namespace app
{
namespace loopTiming
{

/// Join exact identities in the ordinary consumer; never infer main identity from event position.
/** Each input retains its publisher order, but the two inputs may arrive in either order.
 * BMC's ordered post owner provides a combined-counter watermark: once a later
 * command is observed, a missing earlier SDK record cannot arrive subsequently.
 */
class Synco
{
  public:
    /// One positive main/SDK match, carrying primary continuity rather than all-SDK continuity.
    struct Match
    {
        uint64_t counter{ 0 }; ///< Genuine primary publication counter.
        int64_t  startNs{ 0 }; ///< Original MONOTONIC SDK entry.
    };

    /// Lifetime coverage counters, independent of whether rolling statistics are available.
    struct Quality
    {
        uint64_t sdkMissed{ 0 };  ///< SDK capture sequence records absent before transport.
        uint64_t mainMissed{ 0 }; ///< Main/mode capture sequence records absent before transport.
        uint64_t invalid{ 0 };    ///< Invalid, conflicting or regressing records.
        uint64_t duplicates{ 0 }; ///< Identical repeated producer records.
        uint64_t unmatched{ 0 };  ///< Main labels lacking a usable SDK result.
        uint64_t overflow{ 0 };   ///< Pending main capacity exhausted.
    };

    /// Allocate the SDK cache and select a finite bound on pending main receipt age.
    void configure( size_t  capacity /**< [in] Maximum SDK cache and pending-main populations. */,
                    int64_t ageNs /**< [in] Positive maximum pending-main age. */ );

    /// Forget associations, source incarnations and mode evidence while retaining lifetime quality counters.
    void reset();

    /// Retain one ordered SDK or main/mode record from its independently attached publisher.
    void observe( const dmTimingEvents::Record &record /**< [in] Complete producer record. */,
                  uint64_t                      incarnation /**< [in] Diagnostic stream inode. */,
                  int64_t                       nowNs /**< [in] Positive local MONOTONIC receipt time. */ );

    /// Emit the next ordered exact match, discarding missing or expired mains conservatively.
    bool pop( Match  &match /**< [out] Next qualified main endpoint. */,
              int64_t nowNs /**< [in] Current local MONOTONIC time. */ );

    /// Return the association epoch, advanced whenever prior measurement continuity is revoked.
    uint64_t epoch() const;

    /// Return whether a coherent diagnostic mode has been received.
    bool known() const;

    /// Return the frozen effective enable state of the current diagnostic mode.
    bool enabled() const;

    /// Return the current synco diagnostic mode generation.
    uint64_t generation() const;

    /// Return lifetime record loss and qualification counters.
    const Quality &quality() const;

  private:
    /// Cache one complete native endpoint in a bounded combined-counter slot.
    struct Endpoint
    {
        dmTimingEvents::Key key;          ///< Exact identity, preventing modulo-slot aliasing.
        int64_t             startNs{ 0 }; ///< Original timestamp; zero marks an empty slot.
    };

    /// Keep main labels in producer order even when native endpoints arrive later.
    struct Main
    {
        dmTimingEvents::Record record;          ///< Frozen main identity and mode.
        int64_t                receivedNs{ 0 }; ///< Local receipt time used only to bound waiting.
    };

    std::vector<Endpoint>  m_sdks;                 ///< Fixed SDK cache, also holding unclassified auxiliary records.
    std::deque<Main>       m_mains;                ///< Bounded pending main labels in primary order.
    int64_t                m_ageNs{ 0 };           ///< Maximum pending-main receipt age.
    uint64_t               m_epoch{ 0 };           ///< Consumer association generation.
    uint64_t               m_sdkIncarnation{ 0 };  ///< SDK diagnostic publisher inode.
    uint64_t               m_mainIncarnation{ 0 }; ///< Main diagnostic publisher inode.
    dmTimingEvents::Record m_lastSdk;              ///< Last SDK capture sequence and combined watermark.
    dmTimingEvents::Record m_lastMain;             ///< Last main/mode capture sequence.
    uint64_t               m_generation{ 0 };      ///< Frozen synco mode generation.
    uint64_t               m_steps{ 0 };           ///< Step configuration within the current mode generation.
    uint64_t               m_primaryInode{ 0 };    ///< Genuine primary source incarnation.
    uint64_t               m_primaryCounter{ 0 };  ///< Last queued main counter within the current mode.
    uint64_t               m_emittedCounter{ 0 };  ///< Last delivered main, for rejecting missing-main continuity.
    bool                   m_known{ false };       ///< Mode fields have coherent positive evidence.
    bool                   m_enabled{ false };     ///< Effective enabled state recorded by synco.
    Quality                m_quality;              ///< Cumulative qualification and loss observations.

    /// Clear pending associations and advance the epoch; optionally discard cached native endpoints.
    void discard( bool sdk /**< [in] Also invalidate every cached native identity. */ );

    /// Check producer sequence continuity and distinguish exact repeats from conflicting duplicates.
    bool sequence( const dmTimingEvents::Record &record /**< [in] New producer observation. */,
                   const dmTimingEvents::Record &previous /**< [in] Prior observation from that producer. */,
                   bool sdk /**< [in] Select native rather than main/mode loss accounting. */ );
};

inline void Synco::configure( size_t capacity, int64_t ageNs )
{
    m_sdks.assign( capacity, {} );
    m_ageNs   = ageNs;
    m_quality = {};
    reset();
}

inline void Synco::discard( bool sdk )
{
    m_mains.clear();
    m_primaryCounter = m_emittedCounter = 0;
    ++m_epoch;

    if( !m_epoch )
    {
        ++m_epoch;
    }

    if( sdk )
    {
        std::fill( m_sdks.begin(), m_sdks.end(), Endpoint{} );
    }
}

inline void Synco::reset()
{
    discard( true );
    m_sdkIncarnation = m_mainIncarnation = 0;
    m_lastSdk = m_lastMain = {};
    m_generation = m_steps = m_primaryInode = 0;
    m_known = m_enabled = false;
}

inline bool Synco::sequence( const dmTimingEvents::Record &record, const dmTimingEvents::Record &previous, bool sdk )
{
    if( !previous.sequence )
    {
        return true;
    }

    if( record.sequence == previous.sequence && dmTimingEvents::encode( record ) == dmTimingEvents::encode( previous ) )
    {
        ++m_quality.duplicates;
        return false;
    }

    if( record.sequence <= previous.sequence )
    {
        ++m_quality.invalid;
        discard( true );
        m_known = false;
        return false;
    }

    if( record.sequence - previous.sequence > 1 )
    {
        ( sdk ? m_quality.sdkMissed : m_quality.mainMissed ) += record.sequence - previous.sequence - 1;
        discard( sdk );
    }

    return true;
}

inline void Synco::observe( const dmTimingEvents::Record &record, uint64_t incarnation, int64_t nowNs )
{
    if( m_sdks.empty() || m_ageNs <= 0 || !incarnation || nowNs <= 0 || !dmTimingEvents::valid( record ) )
    {
        ++m_quality.invalid;
        reset();
        return;
    }

    const bool sdk      = record.kind == dmTimingEvents::Kind::sdk;
    auto      &source   = sdk ? m_sdkIncarnation : m_mainIncarnation;
    auto      &previous = sdk ? m_lastSdk : m_lastMain;

    if( source != incarnation )
    {
        if( source )
        {
            discard( sdk );
        }

        source   = incarnation;
        previous = {};

        if( !sdk )
        {
            m_generation = m_steps = m_primaryInode = 0;
            m_known                                 = false;
        }
    }

    if( !sequence( record, previous, sdk ) )
    {
        return;
    }

    if( sdk )
    {
        if( previous.sequence && ( record.key.inode != previous.key.inode ||
                                   record.key.counter <= previous.key.counter || record.startNs <= previous.startNs ) )
        {
            discard( true );

            if( record.key.inode == previous.key.inode )
            {
                ++m_quality.invalid;
                previous = record;
                return;
            }
        }

        previous                                   = record;
        m_sdks[record.key.counter % m_sdks.size()] = { record.key, record.startNs };
        return;
    }

    previous = record;

    if( record.generation < m_generation ||
        ( m_known && record.generation == m_generation && ( record.enabled != m_enabled || record.steps != m_steps ) ) )
    {
        ++m_quality.invalid;
        discard( true );
        m_known = false;
        return;
    }

    if( !m_known || record.generation != m_generation )
    {
        discard( false );
        m_generation   = record.generation;
        m_enabled      = record.enabled;
        m_steps        = record.steps;
        m_primaryInode = 0;
        m_known        = true;
    }

    if( record.kind == dmTimingEvents::Kind::mode )
    {
        return;
    }

    if( m_primaryInode && m_primaryInode != record.primaryInode )
    {
        discard( false );
    }

    m_primaryInode = record.primaryInode;

    if( m_primaryCounter && record.primaryCounter <= m_primaryCounter )
    {
        ++m_quality.invalid;
        discard( true );
        m_known = false;
        return;
    }

    if( m_mains.size() == m_sdks.size() )
    {
        ++m_quality.overflow;
        discard( false );
    }

    m_primaryCounter = record.primaryCounter;
    m_mains.push_back( { record, nowNs } );
}

inline bool Synco::pop( Match &match, int64_t nowNs )
{
    while( m_known && !m_mains.empty() )
    {
        const auto &main     = m_mains.front();
        auto       &endpoint = m_sdks[main.record.key.counter % m_sdks.size()];
        const bool  found    = endpoint.startNs && dmTimingEvents::equal( endpoint.key, main.record.key );

        if( found )
        {
            match    = { main.record.primaryCounter, endpoint.startNs };
            endpoint = {};
            m_mains.pop_front();

            if( m_emittedCounter && match.counter - m_emittedCounter != 1 )
            {
                ++m_epoch;
            }

            m_emittedCounter = match.counter;
            return true;
        }

        const bool passed =
            m_lastSdk.key.inode == main.record.key.inode && m_lastSdk.key.counter >= main.record.key.counter;
        const bool expired = nowNs < main.receivedNs || nowNs - main.receivedNs >= m_ageNs;

        if( !passed && !expired )
        {
            return false;
        }

        ++m_quality.unmatched;
        m_mains.pop_front();
        ++m_epoch;
        m_emittedCounter = 0;
    }

    return false;
}

inline uint64_t Synco::epoch() const
{
    return m_epoch;
}
inline bool Synco::known() const
{
    return m_known;
}
inline bool Synco::enabled() const
{
    return m_enabled;
}
inline uint64_t Synco::generation() const
{
    return m_generation;
}
inline const Synco::Quality &Synco::quality() const
{
    return m_quality;
}

} // namespace loopTiming
} // namespace app
} // namespace MagAOX
#endif // loopTimingMonitor_loopTimingSynco_hpp
