/** \file adc_input.cpp
 * \brief Calibrated raw-pair capture with strict source, gap, snapshot and clock handling.
 */
#include "adc_input.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace DDSPC
{
namespace
{
/// Map integer epoch timestamps without subtracting floating-point epoch values or overflowing nanoseconds.
bool map_time( std::int64_t    real /**< [in] Producer realtime ns. */,
               const AdcClock &anchor /**< [in] Session mapping. */,
               std::int64_t   &mono /**< [out] Mapped monotonic ns. */ )
{
    if( real < 0 )
    {
        return false;
    }
    const auto delta = real - anchor.m_realNs; // Both epochs are nonnegative, so this subtraction fits int64_t.
    if( ( delta > 0 && anchor.m_monoNs > std::numeric_limits<std::int64_t>::max() - delta ) ||
        ( delta < 0 && anchor.m_monoNs < -delta ) )
    {
        return false;
    }
    mono = anchor.m_monoNs + delta;
    return true;
}
} // namespace

const char *adc_state( AdcState state )
{
    switch( state )
    {
    case AdcState::Disabled:
        return "disabled";
    case AdcState::Disconnected:
        return "disconnected";
    case AdcState::Waiting:
        return "waiting";
    case AdcState::Streaming:
        return "streaming";
    case AdcState::Incompatible:
        return "incompatible";
    case AdcState::Invalid:
        return "invalid_sample";
    case AdcState::ClockFault:
        return "clock_fault";
    case AdcState::Stale:
        return "stale";
    }
    return "unknown";
}

void validate_adc_config( const AdcConfig &config )
{
    if( config.m_stream.empty() || config.m_stream.size() > 127 || config.m_stream.starts_with( "dm" ) ||
        config.m_stream.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" ) !=
            std::string::npos ||
        config.m_profile.empty() || config.m_minCode >= config.m_maxCode || config.m_maxCode > 65535 ||
        config.m_capacity < 2 || config.m_capacity > 65536 || config.m_maxAgeNs <= 0 || config.m_maxIntervalNs <= 0 ||
        config.m_clockToleranceNs <= 0 )
    {
        throw std::invalid_argument( "Invalid ADC source, encoding, retention or clock policy" );
    }
    for( std::size_t axis = 0; axis < 2; ++axis )
    {
        if( !std::isfinite( config.m_zero[axis] ) || !std::isfinite( config.m_voltsPerCount[axis] ) ||
            config.m_voltsPerCount[axis] <= 0 || !std::isfinite( config.m_sensitivity[axis] ) ||
            config.m_sensitivity[axis] <= 0 || ( config.m_polarity[axis] != 1 && config.m_polarity[axis] != -1 ) ||
            !std::isfinite( config.m_voltsPerCount[axis] / config.m_sensitivity[axis] ) )
        {
            throw std::invalid_argument( "Invalid per-channel ADC calibration" );
        }
    }
}

AdcCapture::AdcCapture( const AdcConfig &config ) : m_config( config )
{
    validate_adc_config( config );
    m_ring.resize( config.m_capacity );
    m_batch.resize( std::min<std::size_t>( config.m_capacity, 256 ) );
}

void AdcCapture::break_segment( AdcState state )
{
    ++m_status.m_segment;
    m_status.m_state    = state;
    m_status.m_valid    = false;
    m_status.m_retained = m_next = 0;
    m_status.m_rateHz            = 0;
    m_status.m_ageUs             = -1;
    m_firstRateTime = m_lastTime = -1;
    m_rateIntervals              = 0;
}

void AdcCapture::attached( std::uint64_t device, std::uint64_t inode )
{
    if( m_status.m_generation )
    {
        ++m_status.m_restarts;
    }
    ++m_status.m_generation;
    m_status.m_device = device;
    m_status.m_inode  = inode;
    m_status.m_latest = {};
    m_status.m_source = {};
    m_cursor = m_observedCounter = 0;
    m_bound = m_haveClock = false;
    m_clockFault          = false;
    m_highTime = m_lastNow = -1;
    break_segment( AdcState::Waiting );
}

void AdcCapture::disconnected()
{
    if( m_status.m_state != AdcState::Disconnected )
    {
        break_segment( AdcState::Disconnected );
    }
    m_status.m_device = m_status.m_inode = 0;
    m_bound = m_haveClock = false;
}

void AdcCapture::tick( std::int64_t nowNs )
{
    if( nowNs < 0 || ( m_lastNow >= 0 && nowNs < m_lastNow ) )
    {
        break_segment( AdcState::ClockFault );
        m_haveClock  = false;
        m_clockFault = true;
        return;
    }
    m_lastNow = nowNs;
    if( m_highTime >= 0 )
    {
        if( m_status.m_valid && nowNs - m_highTime > m_config.m_maxAgeNs )
        {
            break_segment( AdcState::Stale );
        }
        m_status.m_ageUs = ( nowNs - m_highTime ) * 1e-3;
    }
}

bool AdcCapture::check_clock( const AdcClock &clock )
{
    tick( clock.m_monoNs );
    if( m_clockFault || clock.m_realNs <= 0 || clock.m_monoNs < 0 || clock.m_uncertaintyNs < 0 ||
        clock.m_uncertaintyNs > m_config.m_clockToleranceNs || clock.m_monoNs != m_lastNow )
    {
        if( !m_clockFault )
        {
            break_segment( AdcState::ClockFault );
        }
        m_haveClock  = false;
        m_clockFault = true;
        return false;
    }
    if( m_haveClock && std::abs( static_cast<long double>( clock.m_realNs - m_anchor.m_realNs ) -
                                 ( clock.m_monoNs - m_anchor.m_monoNs ) ) > m_config.m_clockToleranceNs )
    {
        break_segment( AdcState::ClockFault );
        m_haveClock  = false;
        m_clockFault = true;
        return false;
    }
    if( !m_haveClock )
    {
        m_anchor    = clock;
        m_haveClock = true;
    }
    return true;
}

std::size_t AdcCapture::drain( const AdcSource &source )
{
    if( m_status.m_state == AdcState::Disconnected )
    {
        return 0;
    }
    const auto before      = source.header();
    m_status.m_source      = before;
    const auto observation = source.clock();
    tick( observation.m_monoNs );
    if( !before.m_valid || !before.m_uint16 || !before.m_cpu || before.m_axes < 2 || before.m_axes > 3 ||
        before.m_width != 2 || before.m_height != 1 || !before.m_depth || before.m_index >= before.m_depth ||
        before.m_counter == std::numeric_limits<std::uint64_t>::max() )
    {
        break_segment( AdcState::Incompatible );
        return 0;
    }
    if( before.m_write )
    {
        ++m_status.m_busy;
        return 0;
    }
    if( m_bound && before.m_counter < std::max( m_cursor, m_observedCounter ) )
    {
        if( source.header() != before )
        {
            ++m_status.m_races;
            return 0;
        }
        attached( m_status.m_device, m_status.m_inode );
        m_status.m_source = before;
    }
    if( !check_clock( observation ) )
    {
        return 0;
    }
    if( !before.m_counter )
    {
        return 0;
    }
    if( !m_bound )
    {
        m_cursor = before.m_counter - 1; // Attach at the newest sample, never import an unknown old history.
        m_bound  = true;
    }
    if( before.m_counter == m_cursor )
    {
        ++m_status.m_duplicates;
        return 0;
    }
    const auto retained = before.m_counter >= before.m_depth ? before.m_counter - before.m_depth + 1 : 1;
    const auto first    = std::max( m_cursor + 1, retained );
    const auto count    = std::min<std::uint64_t>( before.m_counter - first + 1, m_batch.size() );
    for( std::size_t n = 0; n < count; ++n )
    {
        const auto plane = ( before.m_index + before.m_depth - ( ( before.m_counter - first - n ) % before.m_depth ) ) %
                           before.m_depth;
        m_batch[n]       = source.frame( plane );
        if( m_batch[n].m_counter != first + n )
        {
            ++m_status.m_races;
            return 0;
        }
    }
    if( source.header() != before )
    {
        ++m_status.m_races;
        return 0;
    }
    m_observedCounter = before.m_counter;
    const auto clock  = source.clock();
    if( !check_clock( clock ) )
    {
        return 0;
    }
    if( first > m_cursor + 1 )
    {
        m_status.m_dropped += first - m_cursor - 1;
        break_segment( AdcState::Invalid );
    }
    std::size_t accepted = 0;
    for( std::size_t n = 0; n < count; ++n )
    {
        const auto &frame     = m_batch[n];
        m_cursor              = frame.m_counter;
        std::int64_t acquired = 0, published = 0;
        AdcSample    sample;
        bool         rail = false, range = false;
        bool valid = map_time( frame.m_acquiredRealNs, m_anchor, acquired ) &&
                     map_time( frame.m_writtenRealNs, m_anchor, published ) &&
                     frame.m_writtenRealNs >= frame.m_acquiredRealNs && frame.m_writtenRealNs <= clock.m_realNs &&
                     published <= clock.m_monoNs && acquired <= clock.m_monoNs &&
                     clock.m_monoNs - acquired <= m_config.m_maxAgeNs && ( m_highTime < 0 || acquired > m_highTime );
        const bool timeValid = valid;
        for( std::size_t axis = 0; axis < 2; ++axis )
        {
            const auto raw = frame.m_raw[axis];
            rail  = rail || ( m_config.m_rejectRails && ( raw == m_config.m_minCode || raw == m_config.m_maxCode ) );
            range = range || raw < m_config.m_minCode || raw > m_config.m_maxCode;
            valid = valid && raw >= m_config.m_minCode && raw <= m_config.m_maxCode &&
                    ( !m_config.m_rejectRails || ( raw > m_config.m_minCode && raw < m_config.m_maxCode ) );
            sample.m_acceleration[axis] = ( double( raw ) - m_config.m_zero[axis] ) * m_config.m_polarity[axis] *
                                          m_config.m_voltsPerCount[axis] / m_config.m_sensitivity[axis];
            valid                       = valid && std::isfinite( sample.m_acceleration[axis] );
        }
        if( !valid )
        {
            m_status.m_railRejected += rail;
            m_status.m_rangeRejected += range;
            m_status.m_timestampRejected += !timeValid;
            ++m_status.m_invalid;
            ++m_status.m_dropped;
            break_segment( AdcState::Invalid );
            m_status.m_latest              = {};
            m_status.m_latest.m_frame      = frame;
            m_status.m_latest.m_receivedNs = clock.m_monoNs;
            continue;
        }
        if( m_lastTime >= 0 && acquired - m_lastTime > m_config.m_maxIntervalNs )
        {
            ++m_status.m_timingGaps;
            break_segment( AdcState::Invalid );
        }
        sample.m_frame       = frame;
        sample.m_generation  = m_status.m_generation;
        sample.m_segment     = m_status.m_segment;
        sample.m_acquiredNs  = acquired;
        sample.m_publishedNs = published;
        sample.m_receivedNs  = clock.m_monoNs;
        if( m_firstRateTime < 0 )
        {
            m_firstRateTime = sample.m_acquiredNs;
        }
        else
        {
            ++m_rateIntervals;
            m_status.m_rateHz = 1e9 * m_rateIntervals / ( sample.m_acquiredNs - m_firstRateTime );
        }
        m_lastTime        = sample.m_acquiredNs;
        m_highTime        = sample.m_acquiredNs;
        m_status.m_latest = sample;
        m_status.m_valid  = true;
        m_status.m_state  = AdcState::Streaming;
        m_status.m_ageUs  = ( clock.m_monoNs - m_lastTime ) * 1e-3;
        if( m_status.m_retained == m_ring.size() )
        {
            ++m_status.m_overwrites;
        }
        else
        {
            ++m_status.m_retained;
        }
        m_ring[m_next] = sample;
        m_next         = ( m_next + 1 ) % m_ring.size();
        ++m_status.m_samples;
        ++accepted;
    }
    return accepted;
}

AdcStatus AdcCapture::status() const
{
    return m_status;
}

std::vector<AdcSample> AdcCapture::samples() const
{
    std::vector<AdcSample> result;
    result.reserve( m_status.m_retained );
    const auto first = ( m_next + m_ring.size() - m_status.m_retained ) % m_ring.size();
    for( std::size_t n = 0; n < m_status.m_retained; ++n )
    {
        result.push_back( m_ring[( first + n ) % m_ring.size()] );
    }
    return result;
}

AdcBatch
AdcCapture::copy_after( const AdcCursor &cursor, std::span<AdcSample> output, std::int64_t availableByNs ) const
{
    if( output.empty() || availableByNs < 0 )
        throw std::invalid_argument( "Invalid bounded ADC copy storage or time" );
    AdcBatch result;
    result.m_cutoffNs      = availableByNs;
    result.m_status        = m_status;
    const bool same        = cursor.m_generation == m_status.m_generation && cursor.m_device == m_status.m_device &&
                             cursor.m_inode == m_status.m_inode;
    result.m_next          = { m_status.m_device,
                               m_status.m_inode,
                               m_status.m_generation,
                               m_status.m_segment,
                               same ? cursor.m_counter : 0,
                               same && cursor.m_bound };
    result.m_discontinuity = cursor.m_bound && ( !same || cursor.m_segment != m_status.m_segment );
    if( !m_status.m_retained )
        return result;
    const auto head  = ( m_next + m_ring.size() - m_status.m_retained ) % m_ring.size();
    const auto first = m_ring[head].m_frame.m_counter;
    if( !result.m_next.m_bound )
    {
        result.m_next.m_counter = first - 1; // Initial attachment does not claim completeness before retained history.
        result.m_next.m_bound   = true;
    }
    if( result.m_next.m_counter < first - 1 )
    {
        result.m_missing        = first - 1 - result.m_next.m_counter;
        result.m_discontinuity  = true;
        result.m_next.m_counter = first - 1;
    }
    const auto offset = result.m_next.m_counter >= first
                            ? std::min<std::uint64_t>( m_status.m_retained, result.m_next.m_counter - first + 1 )
                            : 0;
    for( std::size_t n = offset; n < m_status.m_retained; ++n )
    {
        const auto &sample = m_ring[( head + n ) % m_ring.size()];
        if( sample.m_receivedNs > availableByNs )
            break; // A later receipt remains pending, never silently consumed at an earlier origin.
        if( result.m_count == output.size() )
        {
            result.m_more = true;
            break;
        }
        output[result.m_count++] = sample;
        result.m_next.m_counter  = sample.m_frame.m_counter;
    }
    result.m_pending = m_status.m_retained - offset - result.m_count;
    if( result.m_pending )
        result.m_pendingReceiptNs = m_ring[( head + offset + result.m_count ) % m_ring.size()].m_receivedNs;
    return result;
}

} // namespace DDSPC
