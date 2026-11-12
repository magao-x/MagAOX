/** \file timingStats.hpp
 * \brief Clock-neutral timing populations shared by device diagnostics.
 */
#ifndef common_timingStats_hpp
#define common_timingStats_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace MagAOX
{
namespace timing
{
/// One independently qualified duration or signed-change population, in seconds.
struct Stats
{
    uint64_t count{ 0 };     ///< Retained observations.
    double   mean{ 0 };      ///< Mean seconds; unavailable with no observations.
    double   stddev{ 0 };    ///< Sample deviation; unavailable below two observations.
    double   min{ 0 };       ///< Minimum seconds.
    double   max{ 0 };       ///< Maximum seconds.
    double   span{ 0 };      ///< Coverage of local observation endpoints, in seconds.
    bool     valid{ false }; ///< Sample variance is defined.
};

/// Consumer-owned numerically stable population accumulation.
class Accumulator
{
    Stats   m_stats;                                        ///< Current population, owned by its consumer.
    double  m_square{ 0 };                                  ///< Welford squared-deviation sum.
    int64_t m_first{ std::numeric_limits<int64_t>::max() }; ///< Oldest local coverage endpoint.
    int64_t m_last{ -1 };                                   ///< Newest local coverage endpoint.

  public:
    /// Add seconds with independently supplied local coverage endpoints.
    void add( double  value /**< [in] Duration or signed change in seconds. */,
              int64_t first /**< [in] Earlier local endpoint, ns. */,
              int64_t last /**< [in] Later local endpoint, ns. */ );

    /// Return population statistics without sharing mutable producer state.
    Stats stats() const;
};

inline void Accumulator::add( double value, int64_t first, int64_t last )
{
    ++m_stats.count;
    const double difference = value - m_stats.mean;
    m_stats.mean += difference / m_stats.count;
    m_square += difference * ( value - m_stats.mean );
    m_stats.min = m_stats.count == 1 ? value : std::min( m_stats.min, value );
    m_stats.max = m_stats.count == 1 ? value : std::max( m_stats.max, value );
    m_first     = std::min( m_first, first );
    m_last      = std::max( m_last, last );
}

inline Stats Accumulator::stats() const
{
    Stats result = m_stats;
    result.valid = result.count > 1;

    if( result.valid )
    {
        result.stddev = std::sqrt( std::max( 0.0, m_square / ( result.count - 1 ) ) );
    }

    if( result.count && m_last >= m_first )
    {
        result.span = ( m_last - m_first ) * 1e-9;
    }

    return result;
}
} // namespace timing
} // namespace MagAOX
#endif
