/** \file accelSimTimeline.hpp
 * \brief Bounded, timestamped single vibration realization shared by the accelerometer and DM outputs.
 * \ingroup accelSim_files
 */
#ifndef accelSimTimeline_hpp
#define accelSimTimeline_hpp

#include <mutex>

#include "accelSimModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

/// Return cubic Lagrange weights for nodes k-1, k, k+1 and k+2 at a fractional position after node k.
/** The weights are exact at nodes (mu = 0 returns {0, 1, 0, 0}) and sum to one. At 8 kHz the amplitude error is
 * below 1e-4 for modes up to 300 Hz and below 0.6% at the 900 Hz upper limit of the mode population.
 */
std::array<double, 4> cubicWeights( double mu /**< [in] fractional position in [0,1) after node k */ );

/// One stochastic model realization on a fixed CLOCK_MONOTONIC grid, generated ahead of the acquisition clock.
/** Node k lies at epoch + k/fps, the nominal time of accelerometer tick k. The first advance() after reset() anchors
 * generation at the node preceding that acquisition; acquisition starts at tick one, so node k is then the k-th
 * model::step() and a node-aligned acquisition reproduces the legacy one-step-per-tick sample. Nodes are consecutive
 * steps of one realization until a reset or a bounded-catch-up restart. Physical values between nodes use the same
 * weights for pointing, WFS pointing and noise-free
 * acceleration, so every evaluated time preserves the model's per-mode relation acceleration = factor * pointing.
 *
 * Only the acquisition thread calls reset(), advance() and measurement(); it alone owns the generator and its RNG.
 * physical() may be called from any thread. It never draws random numbers, so DM enable state and polling rate
 * cannot alter the realization. The mutex guards only short ring copies and the grid definition.
 */
class timeline
{
  public:
    static constexpr size_t  c_capacity  = 128;     ///< Ring size, bounding memory independently of run time.
    static constexpr int64_t c_lookahead = 5000000; ///< Horizon generated after each acquisition time [ns].

    static_assert( c_lookahead * 10000 / 1000000000 + maxCatchUp + 8 < c_capacity,
                   "the ring must hold the horizon and bounded catch-up at the 10 kHz maximum rate" );

  protected:
    std::array<sample, c_capacity> m_nodes{}; ///< Generated nodes, stored at node index modulo capacity.

    mutable std::mutex m_mutex; ///< Serializes ring copies and grid changes between acquisition and readers.

    int64_t m_epoch{ 0 }; ///< Monotonic time of node zero [ns], shared with the acquisition schedule.

    double m_period{ 0 }; ///< Node spacing equal to the generator's sampling period [ns].

    uint64_t m_first{ 0 }; ///< Oldest node still held in the ring.

    uint64_t m_end{ 0 }; ///< One past the newest generated node; equal to m_first while empty.

  public:
    /// Clear the ring and define the grid after acquisition (re)starts or changes rate.
    void reset( int64_t      epoch /**< [in] monotonic time of node zero and accelerometer tick zero [ns] */,
                const model &generator /**< [in] generator whose current sampling rate defines the node spacing */ );

    /// Generate nodes through the horizon after an acquisition time; returns true after a stationary restart.
    /** The first call after reset() anchors the ring at the current time. A later gap of more than maxCatchUp
     * ungenerated nodes restarts the generator, as the legacy model did, instead of generating unbounded catch-up.
     */
    bool advance( model  &generator /**< [in,out] sole stochastic realization, owned by the acquisition thread */,
                  int64_t now /**< [in] monotonic acquisition time [ns] */ );

    /// Evaluate noise-free physical pointing, WFS pointing and acceleration at a monotonic time.
    /** Returns false, without changing value, when the time is outside the generated ring. ADC counts are not set.
     */
    bool physical( int64_t time /**< [in] same-host CLOCK_MONOTONIC time [ns] */,
                   sample &value /**< [out] interpolated physical values */ ) const;

    /// Evaluate one accelerometer measurement: physics at its acquisition time plus the tick's own sensor noise.
    bool measurement( int64_t                  time /**< [in] monotonic acquisition time [ns] */,
                      uint64_t                 tick /**< [in] accelerometer tick, selecting one unique noise draw */,
                      const parameters        &config /**< [in] ADC calibration */,
                      sample                  &value /**< [out] measured acceleration and ADC counts */,
                      std::array<uint64_t, 2> &low /**< [in,out] lower ADC clamp counts */,
                      std::array<uint64_t, 2> &high /**< [in,out] upper ADC clamp counts */ ) const;

  protected:
    /// Convert a monotonic time to a fractional node coordinate; caller holds the mutex or owns the grid.
    double position( int64_t time /**< [in] monotonic time [ns] */ ) const;
};

inline std::array<double, 4> cubicWeights( double mu )
{
    return { -mu * ( mu - 1 ) * ( mu - 2 ) / 6,
             ( mu + 1 ) * ( mu - 1 ) * ( mu - 2 ) / 2,
             -( mu + 1 ) * mu * ( mu - 2 ) / 2,
             ( mu + 1 ) * mu * ( mu - 1 ) / 6 };
}

inline void timeline::reset( int64_t epoch, const model &generator )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    m_epoch  = epoch;
    m_period = 1e9 / generator.m_config.m_fps;
    m_first = m_end = 0;
}

inline double timeline::position( int64_t time ) const
{
    return double( time - m_epoch ) / m_period;
}

inline bool timeline::advance( model &generator, int64_t now )
{
    // Only this thread writes the grid and range, so reading them here needs no lock.
    const double   x         = std::floor( position( now ) );
    const uint64_t needed    = x < 1 ? 0 : uint64_t( x ) - 1; // Cubic evaluation at now uses the preceding node.
    const uint64_t last      = uint64_t( std::max( 0.0, std::floor( position( now + c_lookahead ) ) ) ) + 2;
    bool           restarted = false;
    if( m_first == m_end || needed > m_end + maxCatchUp )
    {
        restarted = m_first != m_end;
        if( restarted )
            generator.restart();
        std::lock_guard<std::mutex> lock( m_mutex );
        m_first = m_end = needed;
    }
    while( m_end <= last )
    {
        const sample                next = generator.step(); // The only RNG access; performed outside the lock.
        std::lock_guard<std::mutex> lock( m_mutex );
        m_nodes[m_end % c_capacity] = next;
        ++m_end;
        m_first = std::max( m_first, m_end - std::min<uint64_t>( m_end, c_capacity ) );
    }
    return restarted;
}

inline bool timeline::physical( int64_t time, sample &value ) const
{
    std::array<sample, 4> nodes;
    double                mu;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        const double                x = m_end > m_first ? position( time ) : 0;
        if( !( x >= 1 ) ) // Also rejects an empty ring and nonfinite positions.
            return false;
        const uint64_t k = uint64_t( std::floor( x ) );
        if( k - 1 < m_first || k + 2 >= m_end )
            return false;
        for( size_t j = 0; j < 4; ++j )
            nodes[j] = m_nodes[( k - 1 + j ) % c_capacity];
        mu = x - double( k );
    }
    const auto w = cubicWeights( mu );
    value        = {};
    for( size_t j = 0; j < 4; ++j )
    {
        for( size_t axis = 0; axis < 2; ++axis )
        {
            value.m_pointing[axis] += w[j] * nodes[j].m_pointing[axis];
            value.m_wfsPointing[axis] += w[j] * nodes[j].m_wfsPointing[axis];
            value.m_acceleration[axis] += w[j] * ( nodes[j].m_acceleration[axis] - nodes[j].m_noise[axis] );
        }
    }
    return true;
}

inline bool timeline::measurement( int64_t                  time,
                                   uint64_t                 tick,
                                   const parameters        &config,
                                   sample                  &value,
                                   std::array<uint64_t, 2> &low,
                                   std::array<uint64_t, 2> &high ) const
{
    if( !physical( time, value ) )
        return false;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( tick < m_first || tick >= m_end )
            return false;
        value.m_noise = m_nodes[tick % c_capacity].m_noise;
    }
    for( size_t axis = 0; axis < 2; ++axis )
    {
        value.m_acceleration[axis] += value.m_noise[axis];
        value.m_adc[axis] = adcCounts( value.m_acceleration[axis], config, low[axis], high[axis] );
    }
    return true;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
