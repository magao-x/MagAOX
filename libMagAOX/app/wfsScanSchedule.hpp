/** \file wfsScanSchedule.hpp
 * \brief Finite, WFS-referenced schedule of ADC scans at an integer multiple of the WFS rate.
 *
 * \ingroup app_files
 */
#ifndef app_wfsScanSchedule_hpp
#define app_wfsScanSchedule_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace MagAOX
{
namespace app
{
namespace wfsScan
{

/// Relative allowance on the scan:WFS ratio, so a WFS slightly faster than nominal keeps its scan count.
/** With a 2 kHz WFS measured at 2000.4 Hz and an 8 kHz limit the ratio is 3.9992, which still gives 4 scans. The
 * scheduled rate can therefore exceed the limit by at most this fraction.
 */
constexpr double c_ratioTolerance = 0.01;

/// Finite scan schedule re-anchored by each unique WFS trigger; never free-runs after the source stops.
/** Each WFS arrival arms `max(1, floor(fps * period))` uniformly spaced scans: the highest integer multiple of the WFS
 * rate not above `fps` (8 kHz at 2 kHz and 500 Hz, 7.7 kHz at 700 Hz for an 8 kHz limit), within
 * `c_ratioTolerance`. A WFS faster than `fps` still gets one scan per WFS frame. The first scan is at the requested
 * phase reduced modulo the scan interval, so a member of the scan grid aligns with the WFS phase model. Originally
 * written for ad7689Ctrl.
 */
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

    /// Start a new period with the highest integer scan count within the rate limit and a checked service budget.
    /**
     * \returns true if the period was armed
     * \returns false if an input is invalid or the scans cannot fit, leaving the schedule cancelled
     */
    bool arm( int64_t anchor /**< [in] unique WFS arrival [monotonic ns] */,
              double  period /**< [in] measured WFS period [ns] */,
              double  fps /**< [in] maximum per-channel scan rate [Hz] */,
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
    /**
     * \returns true if a slot was consumed and a scan should start now
     * \returns false if no slot is due
     */
    bool take( int64_t now /**< [in] actual software scan start [monotonic ns] */ );

    /// Return the scheduled rate, distinct from the maximum rate requested.
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

    const double   budget = service + guard;
    const unsigned count =
        static_cast<unsigned>( std::max( 1.0, std::floor( fps * period / 1e9 * ( 1 + c_ratioTolerance ) ) ) );
    const double interval = period / count;

    if( budget >= interval || 1e9 / interval > 10000 * ( 1 + c_ratioTolerance ) ||
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

} // namespace wfsScan
} // namespace app
} // namespace MagAOX

#endif // app_wfsScanSchedule_hpp
