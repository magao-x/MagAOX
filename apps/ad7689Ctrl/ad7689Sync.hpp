/** \file ad7689Sync.hpp
 * \brief Monotonic finite and continuous WFS-referenced ADC scan schedules.
 * \ingroup ad7689Ctrl_files
 */
#ifndef ad7689Ctrl_ad7689Sync_hpp
#define ad7689Ctrl_ad7689Sync_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>

namespace MagAOX
{
namespace app
{
namespace ad7689CtrlDetail
{

/// Finite scan schedule re-anchored by each unique WFS trigger; never free-runs after the source stops.
struct syncSchedule
{
    /** \name Schedule - Data
     * @{ */
    int64_t  m_anchor{ 0 };         ///< Latest accepted WFS arrival on the monotonic clock [ns].
    double   m_wfsPeriod{ 0 };      ///< Measured WFS period [ns].
    double   m_interval{ 0 };       ///< Uniform scan interval within this WFS period [ns].
    double   m_phase{ 0 };          ///< Applied first-slot offset relative to the WFS arrival [ns].
    double   m_service{ 0 };        ///< Complete ADC service time used to reject expired slots [ns].
    double   m_budget{ 0 };         ///< ADC service plus reserved publication/guard time [ns].
    unsigned m_count{ 0 };          ///< Planned complete ADC scans per WFS period.
    unsigned m_slot{ 0 };           ///< Next slot within the finite current period.
    int64_t  m_lastSample{ 0 };     ///< Most recent consumed scan start; retained across WFS updates.
    bool     m_haveSample{ false }; ///< Whether a minimum inter-scan spacing can be enforced.
    uint64_t m_skipped{ 0 };        ///< Slots abandoned after lateness, replacement, or insufficient separation.
    ///@}

    /** \name Schedule
     * @{ */
    /// Abandon pending slots, retaining the last acquisition time and lifetime skipped count.
    void cancel();

    /// Start a new period with the nearest positive integer scan count and a checked service budget.
    bool arm( int64_t anchor /**< [in] unique WFS arrival [monotonic ns] */,
              double  period /**< [in] measured WFS period [ns] */,
              double  fps /**< [in] requested per-channel scan rate [Hz] */,
              double  phase /**< [in] WFS phase-model delay plus signed user offset [ns] */,
              double  service /**< [in] complete ADC service [ns] */,
              double  guard /**< [in] reserved publication and lateness allowance [ns] */ );

    /// Return the next absolute scan deadline, rounding only the accumulated offset.
    int64_t deadline() const;

    /// Return the end of the current WFS period on the monotonic clock.
    int64_t end() const;

    /// Skip slots that cannot finish before the next slot or the next WFS boundary.
    void discardLate( int64_t now /**< [in] current monotonic time [ns] */ );

    /// Consume exactly one due slot; reject early, expired, or unsafe compressed acquisitions.
    bool take( int64_t now /**< [in] actual software scan start [monotonic ns] */ );

    /// Return the scheduled rate, distinct from the user's requested rate.
    double fps() const;
    ///@}
};

inline void syncSchedule::cancel()
{
    m_skipped += m_count - m_slot;
    m_count = m_slot = 0;
    m_wfsPeriod = m_interval = m_phase = m_service = m_budget = 0;
}

inline bool syncSchedule::arm( int64_t anchor, double period, double fps, double phase, double service, double guard )
{
    cancel();

    if( anchor < 0 || !std::isfinite( period ) || period <= 0 || period > 1e9 || !std::isfinite( fps ) || fps <= 0 ||
        fps > 10000 || !std::isfinite( phase ) || !std::isfinite( service ) || service <= 0 ||
        !std::isfinite( guard ) || guard < 0 )
    {
        return false;
    }

    const double   budget   = service + guard;
    const unsigned count    = static_cast<unsigned>( std::max( 1.0, std::floor( fps * period / 1e9 + 0.5 ) ) );
    const double   interval = period / count;

    if( budget >= interval || 1e9 / interval > 10000 ||
        static_cast<double>( anchor ) > static_cast<double>( std::numeric_limits<int64_t>::max() ) - 2e9 )
    {
        return false;
    }

    double wrapped = std::fmod( phase, interval );

    if( wrapped < 0 )
    {
        wrapped += interval;
    }

    m_anchor    = anchor;
    m_wfsPeriod = period;
    m_interval  = interval;
    m_budget    = budget;
    m_service   = service;
    m_phase     = std::min( wrapped, interval - budget );
    m_count     = count;
    return true;
}

inline int64_t syncSchedule::deadline() const
{
    return m_anchor + static_cast<int64_t>( std::llround( m_phase + m_slot * m_interval ) );
}

inline int64_t syncSchedule::end() const
{
    return m_anchor + static_cast<int64_t>( std::llround( m_wfsPeriod ) );
}

inline void syncSchedule::discardLate( int64_t now )
{
    while( m_slot < m_count )
    {
        const int64_t due = deadline();
        const int64_t limit =
            m_slot + 1 < m_count
                ? m_anchor + static_cast<int64_t>( std::llround( m_phase + ( m_slot + 1 ) * m_interval ) )
                : end();
        const bool compressed = m_haveSample && due < m_lastSample + 0.5 * m_interval;

        if( !compressed && static_cast<double>( now ) + m_service <= limit )
        {
            break;
        }

        ++m_slot;
        ++m_skipped;
    }
}

inline bool syncSchedule::take( int64_t now )
{
    discardLate( now );

    if( m_slot >= m_count || now < deadline() )
    {
        return false;
    }

    m_lastSample = now;
    m_haveSample = true;
    ++m_slot;
    return true;
}

inline double syncSchedule::fps() const
{
    return m_count ? 1e9 / m_interval : 0;
}

/// DDSPC phase-model settings, independent of the legacy WFS latency model.
struct ddspcSettings
{
    bool   m_enabled{ false }; ///< Opt in to continuous, cross-boundary DDSPC scheduling.
    double m_process{ 75000 }; ///< Processing after the remaining half exposure [ns].
    double m_latency{ 15000 }; ///< Calibrated complete ACC acquisition-to-publication latency [ns].
    double m_margin{ 10000 };  ///< Desired data-availability lead before DDSPC [ns].
    double m_slew{ 0.05 };     ///< Maximum phase adjustment per future ACC interval, as a fraction.

    /// Validate finite, bounded timing inputs and a positive, conservative phase-slew fraction.
    bool valid() const;

    /// Calculate the unwrapped acquisition delay; a negative result cannot be scheduled causally.
    double delay( double period /**< [in] WFS period [ns] */,
                  double offset /**< [in] existing signed user phase offset [ns] */ = 0 ) const;

    /// Compare staged and acquisition-owner settings without partial updates.
    bool operator==( const ddspcSettings &other /**< [in] settings to compare */ ) const = default;
};

/// Association of an authorized section of the one continuous clock with its WFS reference.
struct scanReference
{
    uint64_t m_id{ 0 };                ///< Source cnt0, independent of the output frame sequence.
    uint64_t m_first{ 0 }, m_end{ 0 }; ///< Half-open range of nominal ACC slot indices authorized by this reference.
    int64_t  m_arrival{ 0 };           ///< Local monotonic semaphore observation [ns].
    double   m_period{ 0 };            ///< Estimated WFS period at this reference [ns].
    double   m_target{ 0 };            ///< DDSPC model plus user offset, without feedback [ns].
    double   m_command{ 0 };           ///< Requested delay including the existing controller correction [ns].
};

/// Immutable timing/association of a single consumed slot, retained through publication.
struct scanSlot
{
    scanReference m_reference;     ///< WFS reference which authorized this slot, even after newer wakes.
    uint64_t      m_index{ 0 };    ///< Nominal slot index within the current continuous-clock epoch.
    uint64_t      m_slot{ 0 };     ///< Zero-based slot within the associated reference's authorized range.
    int64_t       m_deadline{ 0 }; ///< Frozen absolute monotonic acquisition deadline [ns].
};

/// One continuous clock with finite WFS authorization; references never replace the pending deadline.
struct continuousSchedule
{
    /** \name Schedule - Data
     * @{ */
    int64_t     m_epoch{ 0 }, m_deadline{ 0 };             ///< Monotonic origin and frozen next deadline [ns].
    uint64_t    m_next{ 0 }, m_limit{ 0 }, m_segment{ 0 }; ///< Next/authorized-end indices and interval-change origin.
    long double m_offset{ 0 };                             ///< Unrounded segment origin relative to epoch [ns].
    double      m_interval{ 0 }, m_period{ 0 };            ///< Effective ACC interval and latest WFS period [ns].
    double      m_applied{ 0 }, m_target{ 0 };             ///< Applied and requested clock phase correction [ns].
    double   m_service{ 0 }, m_guard{ 0 }, m_slew{ 0 }; ///< Complete scan, lateness/publication guard and slew budgets.
    unsigned m_count{ 0 };                  ///< Generic nearest integer ACC:WFS ratio for the latest reference.
    bool     m_active{ false };             ///< A valid reference/clock epoch has been established.
    uint64_t m_lastId{ 0 };                 ///< Last source ID admitted; duplicates cannot authorize slots.
    int64_t  m_lastSample{ 0 };             ///< Actual last scan start, retained across reacquisition for safe spacing.
    bool     m_haveSample{ false };         ///< Whether a previous scan imposes an inter-scan safety constraint.
    uint64_t m_skipped{ 0 }, m_epochs{ 0 }; ///< Lifetime omitted slots and explicit phase acquisitions.
    std::deque<scanReference> m_references; ///< Bounded association records; these do not own separate schedules.
    ///@}

    /** \name Schedule
     * @{ */
    /// Invalidate the clock and account for all remaining authorized slots.
    void cancel();

    /// Authorize N more slots and update the phase target without moving the pending deadline.
    bool reference( uint64_t id /**< [in] unique source counter */,
                    int64_t  arrival /**< [in] monotonic WFS arrival [ns] */,
                    double   period /**< [in] estimated WFS period [ns] */,
                    double   fps /**< [in] independent requested ACC rate [Hz] */,
                    double   target /**< [in] DDSPC model plus user offset [ns] */,
                    double   command /**< [in] model plus existing feedback correction [ns] */,
                    double   service /**< [in] complete ADC scan service [ns] */,
                    double   guard /**< [in] publication/lateness reserve [ns] */,
                    double   slew /**< [in] maximum phase correction fraction per ACC interval */ );

    /// Return the unrounded nominal offset at an index, excluding controlled phase correction.
    long double offset( uint64_t index /**< [in] nominal ACC index */ ) const;

    /// Advance nominal slots, slew only future deadlines, and retire consumed association records.
    void advance( uint64_t count /**< [in] acquired or skipped nominal slots */ );

    /// Return whether the pending deadline belongs to a received WFS reference.
    bool pending() const;

    /// Return the frozen absolute monotonic deadline of the next slot.
    int64_t deadline() const;

    /// Return the end of the last authorized period, including its full unwrapped delay.
    int64_t end() const;

    /// Skip expired slots in bounded time, preserving the absolute clock and loss accounting.
    void discardLate( int64_t now /**< [in] monotonic time [ns] */ );

    /// Consume one due slot and retain its source association; never emit a catch-up burst.
    bool take( int64_t   now /**< [in] actual software scan start [monotonic ns] */,
               scanSlot &slot /**< [out] immutable scheduled timing and source association */ );

    /// Return the effective ACC rate, including the documented nearest-integer ratio policy.
    double fps() const;
    ///@}
};

inline bool ddspcSettings::valid() const
{
    return std::isfinite( m_process ) && m_process >= 0 && m_process <= 1e9 && std::isfinite( m_latency ) &&
           m_latency >= 0 && m_latency <= 1e9 && std::isfinite( m_margin ) && m_margin >= 0 && m_margin <= 1e9 &&
           std::isfinite( m_slew ) && m_slew > 0 && m_slew <= 0.25;
}

inline double ddspcSettings::delay( double period, double offset ) const
{
    return 0.5 * period + m_process - m_latency - m_margin + offset;
}

inline void continuousSchedule::cancel()
{
    m_skipped += m_limit - m_next;
    m_next = m_limit = m_segment = 0;
    m_offset = m_applied = m_target = 0;
    m_interval = m_period = 0;
    m_count               = 0;
    m_active              = false;
    m_references.clear();
}

inline long double continuousSchedule::offset( uint64_t index ) const
{
    return m_offset + static_cast<long double>( index - m_segment ) * m_interval;
}

inline bool continuousSchedule::reference( uint64_t id,
                                           int64_t  arrival,
                                           double   period,
                                           double   fps,
                                           double   target,
                                           double   command,
                                           double   service,
                                           double   guard,
                                           double   slew )
{
    if( m_active && id == m_lastId )
    {
        return false;
    }

    // Reuse the same physical and nearest-integer admission policy as the legacy schedule.
    syncSchedule candidate;

    if( !candidate.arm( arrival, period, fps, 0, service, guard ) || !std::isfinite( target ) || target < 0 ||
        target > 1e9 || !std::isfinite( command ) || command < 0 || command > 1e9 || !std::isfinite( slew ) ||
        slew <= 0 || slew > 0.25 || m_references.size() >= 16384 )
    {
        cancel();
        return false;
    }

    if( m_active && ( id < m_lastId || id - m_lastId != 1 || std::abs( period - m_period ) > 0.1 * m_period ||
                      candidate.m_count != m_count ) )
    {
        cancel(); // Explicit discontinuity on a gap, source restart, or large cadence/ratio step.
    }

    if( !m_active )
    {
        m_epoch    = arrival;
        m_offset   = command;
        m_interval = candidate.m_interval;
        m_deadline = arrival + static_cast<int64_t>( std::llround( command ) );
        m_active   = true;
        ++m_epochs;
    }
    else
    {
        // Preserve the pending (also unrounded) deadline while changing only subsequent intervals.
        m_offset   = offset( m_next );
        m_segment  = m_next;
        m_interval = candidate.m_interval;
        m_target   = static_cast<double>( static_cast<long double>( arrival - m_epoch ) + command - offset( m_limit ) );
    }

    m_period  = period;
    m_count   = candidate.m_count;
    m_service = service;
    m_guard   = guard;
    m_slew    = slew;
    m_lastId  = id;
    m_references.push_back( { id, m_limit, m_limit + m_count, arrival, period, target, command } );
    m_limit += m_count;
    return true;
}

inline bool continuousSchedule::pending() const
{
    return m_active && m_next < m_limit;
}

inline int64_t continuousSchedule::deadline() const
{
    return m_deadline;
}

inline int64_t continuousSchedule::end() const
{
    return m_epoch + static_cast<int64_t>( std::llround( offset( m_limit ) + m_applied ) );
}

inline void continuousSchedule::advance( uint64_t count )
{
    m_next += count;
    const double step = std::min( m_slew * m_interval, std::max( 0.0, m_interval - m_service - m_guard ) );
    m_applied += std::clamp( m_target - m_applied, -step, step );
    m_deadline = m_epoch + static_cast<int64_t>( std::llround( offset( m_next ) + m_applied ) );

    while( !m_references.empty() && m_next >= m_references.front().m_end )
    {
        m_references.pop_front();
    }
}

inline void continuousSchedule::discardLate( int64_t now )
{
    while( pending() )
    {
        const double spacing    = std::max( 0.5 * m_interval, m_service + m_guard );
        const bool   compressed = m_haveSample && m_deadline < m_lastSample + spacing;
        const double allowance  = std::min( m_guard, 0.25 * m_interval );

        if( !compressed && now <= m_deadline + allowance )
        {
            break;
        }

        const double   expired = std::ceil( ( static_cast<double>( now - m_deadline ) - allowance ) / m_interval );
        const uint64_t skip    = std::min( m_limit - m_next, static_cast<uint64_t>( std::max( 1.0, expired ) ) );
        m_skipped += skip;
        advance( skip );
    }
}

inline bool continuousSchedule::take( int64_t now, scanSlot &slot )
{
    discardLate( now );

    if( !pending() || now < m_deadline )
    {
        return false;
    }

    slot         = { m_references.front(), m_next, m_next - m_references.front().m_first, m_deadline };
    m_lastSample = now;
    m_haveSample = true;
    advance( 1 );
    return true;
}

inline double continuousSchedule::fps() const
{
    return m_active ? 1e9 / m_interval : 0;
}

} // namespace ad7689CtrlDetail
} // namespace app
} // namespace MagAOX
#endif // ad7689Ctrl_ad7689Sync_hpp
