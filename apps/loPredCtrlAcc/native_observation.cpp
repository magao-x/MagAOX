/** \file native_observation.cpp
 * \brief Observation diagnostics preserve incomplete timing and software-only provenance without qualification.
 */
#include "native_observation.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace DDSPC
{
ObservationClock observation_clock( const std::string &name )
{
    if( name == "unknown" )
        return ObservationClock::Unknown;
    if( name == "realtime" )
        return ObservationClock::Realtime;
    if( name == "tai" )
        return ObservationClock::Tai;
    if( name == "monotonic" )
        return ObservationClock::Monotonic;
    throw std::invalid_argument( "Unknown native observation clock declaration" );
}
const char *observation_clock_name( ObservationClock clock )
{
    switch( clock )
    {
    case ObservationClock::Realtime:
        return "realtime";
    case ObservationClock::Tai:
        return "tai";
    case ObservationClock::Monotonic:
        return "monotonic";
    default:
        return "unknown";
    }
}
const char *observation_source_name( ObservationSource source )
{
    constexpr const char *names[]{ "adc", "modal", "camera", "dm", "completion", "sdk" };
    return std::size_t( source ) < observationSources ? names[std::size_t( source )] : "unknown";
}
NativeObservationConfig::NativeObservationConfig()
{
    m_streams[0]        = { "acc000", "realtime", "realtime" };
    m_streams[1].m_name = "aol1_modevalWFS";
    m_streams[2].m_name = "camwfs";
    m_streams[3].m_name = "dm01disp";
}
void validate_native_observation( const NativeObservationConfig &c )
{
    if( !std::isfinite( c.m_durationSec ) || c.m_durationSec < .01 || c.m_durationSec > 3600 || c.m_pollUs < 10 ||
        c.m_pollUs > 10000 || c.m_queueCapacity < 2 || c.m_queueCapacity > 262144 || c.m_staleNs <= 0 ||
        c.m_staleNs > 1000000000 || c.m_modalIndices[0] >= 65536 || c.m_modalIndices[1] >= 65536 )
        throw std::invalid_argument( "Invalid native observation limits" );
    if( c.m_commandQueueCapacity < 2 || c.m_commandQueueCapacity > 4096 || c.m_commandMaxElements < 1 ||
        c.m_commandMaxElements > 65536 || c.m_commandQueueCapacity * c.m_commandMaxElements > 16777216 )
        throw std::invalid_argument( "Invalid native full-command recording bounds" );
    if( c.m_directory.empty() || !std::filesystem::path( c.m_directory ).is_absolute() )
        throw std::invalid_argument( "Native observation requires an existing absolute private directory" );
    const auto path = std::filesystem::weakly_canonical( c.m_directory ).string();
    if( path == "/opt/MagAOX" || path.starts_with( "/opt/MagAOX/" ) || path == "/milk/shm" ||
        path.starts_with( "/milk/shm/" ) || path == "/dev/shm" || path.starts_with( "/dev/shm/" ) )
        throw std::invalid_argument( "Observation files cannot use instrument installation or stream directories" );
    for( std::size_t i = 0; i < observationSources; ++i )
    {
        const auto &s = c.m_streams[i];
        if( ( i < 4 && s.m_name.empty() ) || s.m_name.size() > 127 ||
            s.m_name.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) !=
                std::string::npos )
            throw std::invalid_argument( "Invalid existing observation stream name" );
        observation_clock( s.m_acquisitionClock );
        observation_clock( s.m_publicationClock );
        for( std::size_t j = 0; j < i; ++j )
            if( !s.m_name.empty() && s.m_name == c.m_streams[j].m_name )
                throw std::invalid_argument( "Native observation source names must be distinct" );
    }
}
NativeObservationCapture::NativeObservationCapture( const NativeObservationConfig &config ) : m_config( config )
{
    validate_native_observation( config );
}
NativeObservation NativeObservationCapture::accept( NativeObservation r )
{
    const auto index = std::size_t( r.m_source );
    if( index >= observationSources )
        throw std::invalid_argument( "Invalid observation source" );
    auto      &s        = m_status.m_sources[index];
    const auto previous = s.m_latest;
    if( r.m_event == ObservationEvent::Disconnected )
    {
        s.m_receivedRateHz = 0;
        s.m_epochSamples   = 0;
        s.m_connected      = false;
        s.m_stale          = true;
        s.m_latest         = r;
        if( r.m_source == ObservationSource::Dm )
            m_dmCount = m_dmNext = 0;
        return r;
    }
    if( r.m_event == ObservationEvent::AdcBreak )
    {
        m_status.m_adcMissing += r.m_missing;
        ++m_status.m_adcBreaks;
        s.m_latest = r;
        return r;
    }
    const bool changed = previous.m_generation && r.m_generation != previous.m_generation;
    if( changed )
        ++s.m_restarts;
    if( r.m_source == ObservationSource::Dm &&
        ( changed || r.m_event == ObservationEvent::Attached || r.m_counter < previous.m_counter ) )
        m_dmCount = m_dmNext = 0;
    s.m_connected = true;
    if( r.m_event == ObservationEvent::Attached )
    {
        s.m_receivedRateHz = 0;
        s.m_epochSamples   = 0;
        s.m_latest         = r;
        return r;
    }
    if( previous.m_event == ObservationEvent::Sample && previous.m_counter && !changed )
    {
        if( r.m_counter == previous.m_counter )
            ++s.m_duplicates;
        else if( r.m_counter > previous.m_counter )
            s.m_counterGaps += r.m_counter - previous.m_counter - 1;
        else
            ++s.m_restarts;
    }
    ++s.m_samples;
    if( !s.m_epochSamples || changed || r.m_counter < previous.m_counter || r.m_availableNs < s.m_lastReceiptNs )
    {
        s.m_epochSamples   = 1;
        s.m_firstReceiptNs = r.m_availableNs;
        s.m_receivedRateHz = 0;
    }
    else if( r.m_counter > previous.m_counter )
    {
        ++s.m_epochSamples;
        const auto elapsed = r.m_availableNs - s.m_firstReceiptNs;
        if( elapsed > 0 )
            s.m_receivedRateHz = double( s.m_epochSamples - 1 ) * 1e9 / double( elapsed );
    }
    s.m_lastReceiptNs   = r.m_availableNs;
    s.m_stale           = false;
    const bool endpoint = r.m_source == ObservationSource::Completion || r.m_source == ObservationSource::Sdk;
    if( !endpoint )
    {
        const auto &cfg   = m_config.m_streams[index];
        const auto  clock = observation_clock( cfg.m_publicationClock );
        if( r.m_acquiredRawNs <= 0 || r.m_publishedRawNs <= 0 )
            r.m_flags |= ObservationInvalidTime;
        if( clock == ObservationClock::Unknown || r.m_clockNowNs < 0 )
        {
            ++s.m_clockUnknown;
            r.m_flags |= ObservationUnknownClock;
        }
        else if( r.m_publishedRawNs >= 0 )
        {
            r.m_publicationAgeNs = r.m_clockNowNs - r.m_publishedRawNs;
            if( r.m_publicationAgeNs < -std::max<std::int64_t>( 0, r.m_clockBracketNs ) )
                r.m_flags |= ObservationClockMismatch;
            if( cfg.m_acquisitionClock == cfg.m_publicationClock && r.m_acquiredRawNs > r.m_publishedRawNs )
                r.m_flags |= ObservationInvalidTime;
        }
        if( r.m_flags & ObservationClockMismatch )
            ++s.m_clockMismatches;
    }
    if( r.m_source == ObservationSource::Modal )
    {
        // No camera-origin/exposure envelope exists here. Equal counters or nearby timestamps cannot supply it.
        r.m_flags |= ObservationAssociationMissing;
        ++s.m_associationFailures;
    }
    if( r.m_source == ObservationSource::Dm )
    {
        m_dm[m_dmNext] = r;
        m_dmNext       = ( m_dmNext + 1 ) % m_dm.size();
        m_dmCount      = std::min( m_dmCount + 1, m_dm.size() );
    }
    if( endpoint )
    {
        MagAOX::dmCompletion::Record decoded;
        const auto                   lane = r.m_source == ObservationSource::Completion ? 0u : 1u;
        const auto expected = lane == 0 ? MagAOX::dmCompletion::Kind::completion : MagAOX::dmCompletion::Kind::sdk;
        if( !MagAOX::dmCompletion::decode( r.m_wire, decoded ) || decoded.kind != expected )
        {
            ++s.m_protocolInvalid;
            r.m_flags |= ObservationProtocolInvalid;
        }
        else
        {
            auto      &last = m_endpoints[lane];
            const bool regression =
                last.session == decoded.session && last.epoch == decoded.epoch && decoded.sequence < last.sequence;
            if( regression )
            {
                ++s.m_protocolInvalid;
                r.m_flags |= ObservationProtocolInvalid;
            }
            if( last.session && last.session == decoded.session && last.epoch == decoded.epoch &&
                decoded.sequence > last.sequence )
                s.m_sequenceGaps += decoded.sequence - last.sequence - 1;
            if( last.session && ( last.session != decoded.session || last.epoch != decoded.epoch ) )
                ++s.m_publisherRestarts;
            if( !regression )
                last = decoded;
            const bool complete = !regression && ( lane == 0 ? MagAOX::dmCompletion::qualified( decoded )
                                                             : MagAOX::dmCompletion::sdkQualified( decoded ) );
            if( !complete )
            {
                ++s.m_protocolIncomplete;
                r.m_flags |= ObservationProtocolIncomplete;
            }
            if( complete )
            {
                bool matched = false;
                for( std::size_t n = 0; n < m_dmCount; ++n )
                {
                    const auto &dm = m_dm[n];
                    matched =
                        matched || ( dm.m_imageInode == decoded.inputInode && dm.m_counter == decoded.inputCounter &&
                                     dm.m_publishedRawNs / 1000000000 == decoded.publicationSec &&
                                     dm.m_publishedRawNs % 1000000000 == decoded.publicationNsec );
                }
                if( matched )
                {
                    ++s.m_associations;
                    r.m_flags |= ObservationAssociationMatched;
                }
                else
                {
                    ++s.m_associationFailures;
                    r.m_flags |= ObservationAssociationMissing;
                }
            }
            // Protocol endpoint clocks are publisher-local MONOTONIC, not metadata publication or mirror application.
        }
    }
    if( r.m_flags & ( ObservationInvalidTime | ObservationInvalidPayload ) )
        ++s.m_invalid;
    s.m_latest = r;
    return r;
}
void NativeObservationCapture::poll( ObservationSource source, bool busy, bool race, bool incompatible )
{
    auto &s = m_status.m_sources.at( std::size_t( source ) );
    s.m_busy += busy;
    s.m_races += race;
    s.m_incompatible += incompatible;
    s.m_duplicates += !busy && !race && !incompatible;
}
void NativeObservationCapture::tick( std::int64_t nowNs )
{
    m_status.m_nowNs = nowNs;
    for( auto &s : m_status.m_sources )
        s.m_stale = !s.m_connected || s.m_lastReceiptNs < 0 || nowNs < s.m_lastReceiptNs ||
                    nowNs - s.m_lastReceiptNs > m_config.m_staleNs;
}
NativeObservationStatus NativeObservationCapture::status() const
{
    return m_status;
}
NativeObservationQueue::NativeObservationQueue( std::size_t capacity ) : m_records( capacity )
{
    if( !capacity )
        throw std::invalid_argument( "Observation queue requires positive capacity" );
}
bool NativeObservationQueue::push( const NativeObservation &record )
{
    const auto head = m_head.load( std::memory_order_relaxed );
    if( head - m_tail.load( std::memory_order_acquire ) >= m_records.size() )
    {
        ++m_drops;
        return false;
    }
    m_records[head % m_records.size()] = record;
    m_head.store( head + 1, std::memory_order_release );
    return true;
}
bool NativeObservationQueue::pop( NativeObservation &record )
{
    const auto tail = m_tail.load( std::memory_order_relaxed );
    if( tail == m_head.load( std::memory_order_acquire ) )
        return false;
    record = m_records[tail % m_records.size()];
    m_tail.store( tail + 1, std::memory_order_release );
    return true;
}
std::uint64_t NativeObservationQueue::drops() const
{
    return m_drops.load();
}
NativeCommandQueue::NativeCommandQueue( std::size_t capacity, std::size_t maxElements ) : m_maxElements( maxElements )
{
    if( capacity < 2 || capacity > 4096 || maxElements < 1 || maxElements > 65536 || capacity * maxElements > 16777216 )
        throw std::invalid_argument( "Invalid full-command queue bounds" );
    m_records.resize( capacity );
    m_values.resize( capacity * maxElements );
}

bool NativeCommandQueue::push( const NativeObservation &record, std::span<const float> values )
{
    if( record.m_source != ObservationSource::Dm || record.m_event != ObservationEvent::Sample || values.empty() ||
        values.size() > m_maxElements || values.size() != record.m_elements )
        throw std::invalid_argument( "Incomplete full-command snapshot" );
    const auto head = m_head.load( std::memory_order_relaxed );
    if( head - m_tail.load( std::memory_order_acquire ) >= m_records.size() )
    {
        ++m_drops;
        return false;
    }
    const auto index = head % m_records.size();
    m_records[index] = record;
    std::copy( values.begin(), values.end(), m_values.begin() + index * m_maxElements );
    m_head.store( head + 1, std::memory_order_release );
    return true;
}

bool NativeCommandQueue::pop( NativeObservation &record, std::span<float> values )
{
    if( values.size() < m_maxElements )
        throw std::invalid_argument( "Incomplete full-command writer workspace" );
    const auto tail = m_tail.load( std::memory_order_relaxed );
    if( tail == m_head.load( std::memory_order_acquire ) )
        return false;
    const auto index = tail % m_records.size();
    record           = m_records[index];
    std::copy_n( m_values.begin() + index * m_maxElements, record.m_elements, values.begin() );
    m_tail.store( tail + 1, std::memory_order_release );
    return true;
}

std::uint64_t NativeCommandQueue::drops() const
{
    return m_drops.load();
}
} // namespace DDSPC
