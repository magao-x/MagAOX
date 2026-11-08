/** \file po4aoStatistics.hpp
 * \brief Bounded callback sampling with coherent, explicitly identified telemetry windows.
 */
#ifndef po4aoStatistics_hpp
#define po4aoStatistics_hpp

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace MagAOX::app
{
/// One consumer snapshot; durations are callback microseconds, not end-to-end latency.
struct Po4aoStatistics
{
    uint32_t p50{ 0 }, p95{ 0 }, p99{ 0 }, maximum{ 0 }; ///< Rolling-window duration percentiles.
    uint64_t count{ 0 }, sequence{ 0 },
        lastTimeNs{ 0 };                 ///< Valid samples, producer sequence and monotonic sample time.
    uint64_t model{ 0 };                 ///< Identity of the newest included sample.
    uint32_t backend{ 0 };               ///< Backend of the newest included sample.
    bool mixed{ false }, fresh{ false }; ///< Window spans controller identities, or contains newly published samples.
};

/// Single-producer ring; only the non-real-time consumer allocates and sorts.
class Po4aoSamples
{
    struct Sample
    {
        std::atomic<uint64_t> sequence{ 0 }, model{ 0 },
            timeNs{ 0 };                                   ///< Publication stamp and controller identity/time.
        std::atomic<uint32_t> duration{ 0 }, backend{ 0 }; ///< Callback microseconds and fallback/CPU/TensorRT code.
    };
    std::array<Sample, 1024> m_samples;       ///< Fixed storage; zero marks a slot being replaced.
    std::atomic<uint64_t>    m_sequence{ 0 }; ///< Last completely published sample.

  public:
    /// Append one callback sample without allocation or locking.
    void record( uint32_t duration /**< [in] Callback duration in microseconds. */,
                 uint64_t model /**< [in] Effective model ID. */,
                 uint32_t backend /**< [in] 0 fallback, 1 CPU, 2 TensorRT. */,
                 uint64_t timeNs /**< [in] Monotonic callback completion time. */ );

    /// Read a consistent subset of the rolling window, flagging mixed controller identities.
    Po4aoStatistics snapshot( uint64_t previous /**< [in] Sequence in the preceding telemetry record. */ ) const;
};

inline void Po4aoSamples::record( uint32_t duration, uint64_t model, uint32_t backend, uint64_t timeNs )
{
    const uint64_t next   = m_sequence.load() + 1;
    auto          &sample = m_samples[( next - 1 ) % m_samples.size()];
    // Sequentially consistent atomic fields keep wraparound reads free of data races.
    sample.sequence.store( 0 );
    sample.duration.store( duration );
    sample.model.store( model );
    sample.backend.store( backend );
    sample.timeNs.store( timeNs );
    sample.sequence.store( next );
    m_sequence.store( next );
}

inline Po4aoStatistics Po4aoSamples::snapshot( uint64_t previous ) const
{
    Po4aoStatistics result;
    result.sequence = m_sequence.load();
    result.fresh    = result.sequence != previous;
    std::vector<uint32_t> values;
    values.reserve( m_samples.size() );
    const uint64_t first = result.sequence > m_samples.size() ? result.sequence - m_samples.size() + 1 : 1;
    for( uint64_t sequence = first; sequence <= result.sequence; ++sequence )
    {
        const auto &sample = m_samples[( sequence - 1 ) % m_samples.size()];
        if( sample.sequence.load() != sequence )
            continue;
        const auto duration = sample.duration.load();
        const auto identity = sample.model.load();
        const auto kind     = sample.backend.load();
        const auto time     = sample.timeNs.load();
        if( sample.sequence.load() != sequence )
            continue;
        result.mixed |= !values.empty() && ( identity != result.model || kind != result.backend );
        result.model   = identity;
        result.backend = kind;
        values.push_back( duration ); // A sub-microsecond callback is a valid zero sample.
        result.lastTimeNs = std::max( result.lastTimeNs, time );
    }
    std::sort( values.begin(), values.end() );
    result.count = values.size();
    if( !values.empty() )
    {
        result.p50     = values[( values.size() * 50 + 99 ) / 100 - 1];
        result.p95     = values[( values.size() * 95 + 99 ) / 100 - 1];
        result.p99     = values[( values.size() * 99 + 99 ) / 100 - 1];
        result.maximum = values.back();
    }
    return result;
}
} // namespace MagAOX::app
#endif
