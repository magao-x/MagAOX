/** \file wfsClock.hpp
 * \brief Numerically centered ten-second regression of WFS semaphore arrivals on source frame IDs.
 *
 * \ingroup app_files
 */
#ifndef app_wfsClock_hpp
#define app_wfsClock_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>
#include <utility>

namespace MagAOX
{
namespace app
{
namespace wfsScan
{

/// One unique source-frame observation; outliers are retained but excluded from regression moments.
struct wfsObservation
{
    uint64_t m_id{ 0 };      ///< Source frame index, including real counter gaps.
    int64_t  m_time{ 0 };    ///< Raw local monotonic semaphore timestamp [ns].
    bool     m_used{ true }; ///< Whether the observation contributes to the fitted clock.
};

/// Sliding amortized O(1) centered least-squares clock, with separate persistent-change acquisition.
/** Fits ten seconds of semaphore arrival times against the source frame counter. The fitted period and per-frame
 * arrival (`reference()`) average out arrival jitter, and missed or repeated notifications are handled through the
 * frame IDs. An arrival outside the jitter gate is rejected; `m_minEvents` mutually consistent rejected arrivals
 * restart the fit at the new cadence. Originally written for ad7689Ctrl.
 */
struct rollingWfsClock
{
    /** \name Clock - Data
     * @{ */
    unsigned m_minEvents{ 16 };  ///< Consistent unique observations required at startup or after a change.
    double   m_tolerance{ 0.1 }; ///< Relative cadence/phase change gate, augmented by measured jitter.
    static constexpr int64_t    c_window = 10000000000LL; ///< Ten-second history, independent of WFS rate.
    std::deque<wfsObservation>  m_history;       ///< Raw accepted/rejected observations inside the time window.
    std::vector<wfsObservation> m_candidate;     ///< Short consistent sequence confirming a new cadence/phase.
    uint64_t                    m_originId{ 0 }; ///< Local frame origin, shifted as the window slides.
    int64_t     m_originTime{ 0 };          ///< Local timestamp origin; absolute timestamps never enter fit moments.
    double      m_seed{ 0 };                ///< Period subtracted from timestamps to keep residual moments small [ns].
    uint64_t    m_count{ 0 };               ///< Number of non-outlier observations in the current regression.
    long double m_meanX{ 0 }, m_meanY{ 0 }; ///< Centered frame and timestamp-residual means.
    long double m_xx{ 0 }, m_xy{ 0 }, m_yy{ 0 }; ///< Centered second moments; add/remove/rebase in constant time.
    double   m_period{ 0 }, m_phase{ 0 }; ///< Fitted period and residual intercept relative to the local origin [ns].
    double   m_jitter{ 0 };               ///< Population RMS fit residual of inlier semaphore timestamps [ns].
    double   m_arrivalError{ 0 };         ///< Latest raw arrival minus the fitted prediction [ns].
    double   m_drift{ 0 }; ///< Fitted phase drift relative to the first full-window clock [ns]; zero while warming.
    uint64_t m_baselineId{ 0 };     ///< Baseline source ID for long-term phase drift.
    int64_t  m_baselineTime{ 0 };   ///< Fitted drift baseline, re-anchored when the history matures [monotonic ns].
    double   m_baselinePeriod{ 0 }; ///< Reference period used only for long-term drift diagnostics [ns].
    int64_t  m_firstTime{ 0 };      ///< First qualified history observation, used to establish a mature drift baseline.
    bool     m_baselineMature{ false };      ///< Ten seconds have elapsed before recording long-term drift.
    uint64_t m_lastId{ 0 };                  ///< Latest unique raw source ID, including rejected events.
    int64_t  m_lastTime{ 0 };                ///< Latest unique monotonic raw arrival, independent of the fitted phase.
    bool     m_haveLast{ false };            ///< Whether source identity and timestamp progression can be checked.
    bool     m_ready{ false };               ///< Enough inliers define a qualified clock.
    bool     m_reacquiring{ false };         ///< Multiple consistent outliers are confirming a new clock.
    uint64_t m_rejected{ 0 }, m_resets{ 0 }; ///< Lifetime outlier count and model invalidations, retained on reset.
    ///@}

    /** \name Clock
     * @{ */
    /// Clear fit and candidate state, preserving settings and lifetime diagnostics.
    void reset();

    /// Add/remove one observation from centered residual moments without rescanning the history.
    void moments( const wfsObservation &sample /**< [in] observation relative to current local origin */,
                  bool                  add /**< [in] true inserts, false removes */ );

    /// Evict observations older than ten seconds and translate the local fit origin exactly.
    void prune( int64_t now /**< [in] latest monotonic observation [ns] */ );

    /// Update period, residual phase, jitter and residual-coordinate conditioning from centered moments.
    void fit();

    /// Predict a source frame's monotonic phase using the fitted clock rather than its raw arrival.
    int64_t reference( uint64_t id /**< [in] source frame index */ ) const;

    /// Return the span of retained raw timing history in seconds.
    double span() const;

    /// Return whether the rolling model is qualified and no cadence change is being confirmed.
    bool valid() const;

    /// Observe a unique monotonic reference; true authorizes scheduling from its fitted phase.
    bool observe( uint64_t id /**< [in] source frame index */,
                  int64_t  time /**< [in] raw monotonic semaphore arrival [ns] */ );
    ///@}
};

inline void rollingWfsClock::reset()
{
    if( m_baselinePeriod > 0 )
    {
        ++m_resets;
    }

    m_history.clear();
    m_candidate.clear();
    m_count = 0;
    m_meanX = m_meanY = m_xx = m_xy = m_yy = 0;
    m_seed = m_period = m_phase = m_jitter = m_arrivalError = m_drift = 0;
    m_baselinePeriod                                                  = 0;
    m_haveLast = m_ready = m_reacquiring = m_baselineMature = false;
}

inline void rollingWfsClock::moments( const wfsObservation &sample, bool add )
{
    const long double x  = static_cast<long double>( sample.m_id - m_originId );
    const long double y  = static_cast<long double>( sample.m_time - m_originTime ) - x * m_seed;
    const long double dx = x - m_meanX, dy = y - m_meanY;

    if( add )
    {
        ++m_count;
        m_meanX += dx / m_count;
        m_meanY += dy / m_count;
        m_xx += dx * ( x - m_meanX );
        m_xy += dx * ( y - m_meanY );
        m_yy += dy * ( y - m_meanY );
    }
    else if( m_count > 1 )
    {
        --m_count;
        m_meanX -= dx / m_count;
        m_meanY -= dy / m_count;
        m_xx -= dx * ( x - m_meanX );
        m_xy -= dx * ( y - m_meanY );
        m_yy -= dy * ( y - m_meanY );
    }
    else
    {
        m_count = 0;
        m_meanX = m_meanY = m_xx = m_xy = m_yy = 0;
    }
}

inline void rollingWfsClock::prune( int64_t now )
{
    while( !m_history.empty() && now - m_history.front().m_time > c_window )
    {
        if( m_history.front().m_used )
        {
            moments( m_history.front(), false );
        }

        m_history.pop_front();
    }

    if( !m_history.empty() )
    {
        const auto       &origin = m_history.front();
        const long double dx     = static_cast<long double>( origin.m_id - m_originId );
        const long double dy     = static_cast<long double>( origin.m_time - m_originTime ) - dx * m_seed;
        if( m_count )
        {
            m_meanX -= dx;
            m_meanY -= dy;
        }
        m_originId   = origin.m_id;
        m_originTime = origin.m_time;
    }
}

inline void rollingWfsClock::fit()
{
    m_ready = m_count >= m_minEvents && m_xx > 0;

    if( !m_ready )
    {
        return;
    }

    const long double slope = m_xy / m_xx;
    m_period                = static_cast<double>( m_seed + slope );

    if( !std::isfinite( m_period ) || m_period <= 0 || m_period > 1e9 )
    {
        m_ready = false;
        return;
    }

    // Shear the residual coordinates onto the current fitted slope. Physical timestamps
    // are unchanged, while long uptime and initial-period error cannot grow the moments.
    const long double delta = static_cast<long double>( m_period ) - m_seed;
    m_meanY -= delta * m_meanX;
    m_yy = std::max( 0.0L, m_yy - 2 * delta * m_xy + delta * delta * m_xx );
    m_xy -= delta * m_xx;
    m_seed   = m_period;
    m_phase  = static_cast<double>( m_meanY - m_xy / m_xx * m_meanX );
    m_jitter = std::sqrt( static_cast<double>( std::max( 0.0L, m_yy - m_xy * m_xy / m_xx ) / m_count ) );
}

inline int64_t rollingWfsClock::reference( uint64_t id ) const
{
    const long double frames =
        id >= m_originId ? static_cast<long double>( id - m_originId ) : -static_cast<long double>( m_originId - id );
    const long double offset  = std::round( m_phase + frames * m_period );
    const int64_t     maximum = std::numeric_limits<int64_t>::max();

    if( offset >= static_cast<long double>( maximum - m_originTime ) )
    {
        return maximum; // Untrusted counter jumps become rejected residuals, never integer overflow.
    }

    if( offset <= -static_cast<long double>( m_originTime ) )
    {
        return 0;
    }

    return m_originTime + static_cast<int64_t>( offset );
}

inline double rollingWfsClock::span() const
{
    return m_history.empty() ? 0 : ( m_history.back().m_time - m_history.front().m_time ) * 1e-9;
}

inline bool rollingWfsClock::valid() const
{
    return m_ready && !m_reacquiring;
}

inline bool rollingWfsClock::observe( uint64_t id, int64_t time )
{
    if( time < 0 || m_minEvents < 2 || m_minEvents > 4096 || !std::isfinite( m_tolerance ) || m_tolerance < 0.01 ||
        m_tolerance > 0.5 )
    {
        return false;
    }

    if( m_haveLast && id == m_lastId )
    {
        return false;
    }

    if( m_haveLast && id < m_lastId )
    {
        reset();
    }

    if( m_haveLast && time <= m_lastTime )
    {
        ++m_rejected;
        return false;
    }

    if( m_history.size() >= 200001 )
    {
        reset();
        return false;
    }

    m_lastId   = id;
    m_lastTime = time;
    m_haveLast = true;

    if( m_ready )
    {
        const double residual = static_cast<double>( time - reference( id ) );
        const double gate     = std::max( { m_tolerance * m_period, 20000.0, 6 * m_jitter } );

        if( std::abs( residual ) <= gate )
        {
            m_candidate.clear();
            m_reacquiring = false;
            m_history.push_back( { id, time, true } );
            moments( m_history.back(), true );
            prune( time );
            fit();
            m_arrivalError = static_cast<double>( time - reference( id ) );

            if( !m_baselineMature && time - m_firstTime >= c_window )
            {
                m_baselineId     = id;
                m_baselineTime   = reference( id );
                m_baselinePeriod = m_period;
                m_baselineMature = true;
            }

            if( m_baselineMature )
            {
                m_drift = static_cast<double>( static_cast<long double>( reference( id ) - m_baselineTime ) -
                                               static_cast<long double>( id - m_baselineId ) * m_baselinePeriod );
            }

            return valid();
        }

        ++m_rejected;
        m_arrivalError = residual;
        m_history.push_back( { id, time, false } );
        prune( time );
        fit();
    }

    if( m_candidate.size() >= 2 )
    {
        const auto  &first   = m_candidate.front();
        const auto  &last    = m_candidate.back();
        const double period  = static_cast<double>( last.m_time - first.m_time ) / ( last.m_id - first.m_id );
        const double instant = static_cast<double>( time - last.m_time ) / ( id - last.m_id );

        if( instant <= 0 || instant > 1e9 ||
            std::abs( instant - period ) > std::max( { m_tolerance * period, 40000.0, 6 * m_jitter } ) )
        {
            m_candidate.clear();
        }
    }

    m_candidate.push_back( { id, time, true } );
    m_reacquiring = m_ready && m_candidate.size() >= 3;

    if( m_candidate.size() < m_minEvents )
    {
        return false;
    }

    auto accepted = std::move( m_candidate );
    reset();
    m_originId   = accepted.front().m_id;
    m_originTime = accepted.front().m_time;
    m_seed       = static_cast<double>( accepted.back().m_time - m_originTime ) / ( accepted.back().m_id - m_originId );

    for( const auto &sample : accepted )
    {
        m_history.push_back( sample );
        moments( sample, true );
    }

    prune( time );
    fit();
    m_lastId         = id;
    m_lastTime       = time;
    m_haveLast       = true;
    m_firstTime      = accepted.front().m_time;
    m_baselineId     = id;
    m_baselineTime   = reference( id );
    m_baselinePeriod = m_period;
    m_arrivalError   = static_cast<double>( time - reference( id ) );
    return valid();
}

} // namespace wfsScan
} // namespace app
} // namespace MagAOX
#endif // app_wfsClock_hpp
