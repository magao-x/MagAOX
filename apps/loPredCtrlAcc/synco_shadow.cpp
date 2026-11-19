/** \file synco_shadow.cpp
 * \brief Advance preparation and exact-target shadow selection using the native syncopation protocol.
 */
#include "synco_shadow.hpp"
#include "multirate_shadow.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <chrono>

namespace DDSPC
{
namespace synco = MagAOX::app::synco;
SyncoShadowPlans::SyncoShadowPlans( std::size_t elements, unsigned frames ) : m_elements( elements ), m_frames( frames )
{
    if( !elements || elements > 4096 || !frames || frames > 8 )
        throw std::invalid_argument( "Shadow auxiliary payload bounds" );
    m_pixels.resize( elements * 4 * frames );
}
bool SyncoShadowPlans::prepare(
    const Matrix &plan, const Matrix &shapes, const synco::Record &s, std::int64_t grid, std::int64_t ready )
{
    const auto started = std::chrono::steady_clock::now();
    invalidate();
    if( !synco::validProductionRecord( s, ready ) || s[synco::cycle] < 0 ||
        s[synco::cycle] == std::numeric_limits<std::int64_t>::max() || s[synco::steps] != 4 ||
        s[synco::period] != 500000 || grid <= 0 ||
        grid > std::numeric_limits<std::int64_t>::max() - 500000 * m_frames || ready < 0 ||
        shapes.rows() != int( m_elements ) || shapes.cols() < 1 || shapes.cols() > 2 ||
        plan.rows() != int( 4 * m_frames * shapes.cols() ) || plan.cols() != 1 || !plan.allFinite() ||
        !shapes.allFinite() )
        return false;
    for( unsigned slot = 0; slot < 4 * m_frames; ++slot )
        for( std::size_t p = 0; p < m_elements; ++p )
        {
            double value = 0;
            for( int mode = 0; mode < shapes.cols(); ++mode )
                value += double( shapes( p, mode ) ) * plan( ( 4 * m_frames - slot - 1 ) * shapes.cols() + mode );
            if( !std::isfinite( float( value ) ) )
                return false;
            m_pixels[slot * m_elements + p] = float( value );
        }
    m_epoch      = s[synco::epoch];
    m_generation = s[synco::generation];
    m_grid       = grid;
    m_ready =
        ready +
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count();
    m_valid = true;
    m_slots = 4 * m_frames;
    return true;
}
bool SyncoShadowPlans::command( const synco::Request &r,
                                const synco::Record  &s,
                                std::int64_t          now,
                                std::span<float>      pixels ) const
{
    const auto started = std::chrono::steady_clock::now();
    if( !m_valid || !synco::validProductionRecord( s, now ) || s[synco::cycle] < 0 ||
        s[synco::cycle] == std::numeric_limits<std::int64_t>::max() || s[synco::steps] != 4 ||
        ( m_slots == 1 ? ( s[synco::period] < 450000 || s[synco::period] > 550000 ) : s[synco::period] != 500000 ) ||
        r.step < 0 || r.step >= 4 || r.epoch != m_epoch || r.generation != m_generation || now < m_ready ||
        now >= r.cutoffNs || m_ready >= r.cutoffNs || r.targetNs < m_grid || pixels.size() != m_elements )
        return false;
    const auto expected = synco::request( s, r.step );
    if( expected.epoch != r.epoch || expected.generation != r.generation || expected.cycle != r.cycle ||
        expected.targetNs != r.targetNs || expected.cutoffNs != r.cutoffNs )
        return false;
    const auto delta = r.targetNs - m_grid;
    if( m_slots == 1 ? delta != 0 : ( delta % 125000 || delta / 125000 >= m_slots ) )
        return false;
    std::copy_n( m_pixels.data() + ( delta / 125000 ) * m_elements, m_elements, pixels.data() );
    return now + std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started )
                     .count() <
           r.cutoffNs;
}
void SyncoShadowPlans::invalidate()
{
    m_valid = false;
}
bool SyncoShadowPlans::prepare_slot( std::span<const float> modal,
                                     const Matrix          &shapes,
                                     const synco::Request  &r,
                                     const synco::Record   &s,
                                     std::int64_t           now )
{
    invalidate();
    const auto start = std::chrono::steady_clock::now();
    if( !synco::validProductionRecord( s, now ) || r.step < 0 || r.step >= 4 || s[synco::steps] != 4 ||
        s[synco::period] < 450000 || s[synco::period] > 550000 || s[synco::cycle] < 0 ||
        s[synco::cycle] == std::numeric_limits<std::int64_t>::max() || shapes.rows() != int( m_elements ) ||
        shapes.cols() != int( modal.size() ) || modal.empty() || modal.size() > 2 || now >= r.cutoffNs )
        return false;
    const auto expected = synco::request( s, r.step );
    if( expected.epoch != r.epoch || expected.generation != r.generation || expected.cycle != r.cycle ||
        expected.targetNs != r.targetNs || expected.cutoffNs != r.cutoffNs )
        return false;
    for( std::size_t p = 0; p < m_elements; ++p )
    {
        double value = 0;
        for( std::size_t m = 0; m < modal.size(); ++m )
            value += double( shapes( p, m ) ) * modal[m];
        if( !std::isfinite( float( value ) ) )
            return false;
        m_pixels[p] = float( value );
    }
    m_epoch      = r.epoch;
    m_generation = r.generation;
    m_grid       = r.targetNs;
    m_ready =
        now + std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - start ).count();
    m_slots = 1;
    m_valid = m_ready < r.cutoffNs;
    return m_valid;
}
SyncoShadowProducer::SyncoShadowProducer( std::size_t elements, unsigned frames ) : m_copy( elements )
{
    for( unsigned n = 0; n < 2; ++n )
    {
        m_ready[n]    = std::make_unique<SyncoShadowPlans>( elements, frames );
        m_previous[n] = std::make_unique<SyncoShadowPlans>( elements, frames );
        m_work[n]     = std::make_unique<SyncoShadowPlans>( elements, frames );
    }
}
bool SyncoShadowProducer::prepare( ShadowMode           mode,
                                   const Matrix        &plan,
                                   const Matrix        &shapes,
                                   const synco::Record &s,
                                   std::int64_t         grid,
                                   std::int64_t         now,
                                   std::int64_t         origin )
{
    const unsigned n = mode == ShadowMode::Standard ? 0 : 1;
    if( origin < 0 )
        origin = now;
    if( origin <= m_lossNs[n].load() )
        return false;
    if( !m_work[n]->prepare( plan, shapes, s, grid, now ) )
    {
        m_planMask.fetch_and( ~( 1u << n ) );
        return false;
    }
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    if( !lock.owns_lock() )
        return false;
    m_previous[n].swap( m_ready[n] );
    m_work[n].swap( m_ready[n] );
    m_previousOriginNs[n] = m_originNs[n];
    m_originNs[n]         = origin;
    m_planMask.fetch_or( 1u << n );
    if( origin <= m_lossNs[n].load() )
    {
        m_planMask.fetch_and( ~( 1u << n ) );
        return false;
    }
    ++m_prepared;
    return true;
}
bool SyncoShadowProducer::prepare_slot( ShadowMode             mode,
                                        std::span<const float> modal,
                                        const Matrix          &shapes,
                                        const synco::Request  &r,
                                        const synco::Record   &s,
                                        std::int64_t           now,
                                        std::int64_t           origin )
{
    const unsigned n = mode == ShadowMode::AccelAssisted;
    if( origin <= m_lossNs[n].load() || !m_work[n]->prepare_slot( modal, shapes, r, s, now ) )
        return false;
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    if( !lock.owns_lock() )
        return false;
    m_previous[n].swap( m_ready[n] );
    m_work[n].swap( m_ready[n] );
    m_previousOriginNs[n] = m_originNs[n];
    m_originNs[n]         = origin;
    m_planMask.fetch_or( 1u << n );
    if( origin <= m_lossNs[n].load() )
    {
        m_planMask.fetch_and( ~( 1u << n ) );
        return false;
    }
    ++m_prepared;
    return true;
}
void SyncoShadowProducer::models( ShadowMode requested, bool standard, bool assisted )
{
    m_requested.store( requested == ShadowMode::AccelAssisted );
    m_models.store( unsigned( standard ) | ( unsigned( assisted ) << 1 ) );
}
void SyncoShadowProducer::inhibit( bool assisted, std::int64_t time )
{
    for( unsigned n = assisted ? 1 : 0; n < 2; ++n )
    {
        auto previous = m_lossNs[n].load();
        while( time > previous && !m_lossNs[n].compare_exchange_weak( previous, time ) )
        {
        }
    }
    if( assisted )
    {
        m_models.fetch_and( 1 );
        m_planMask.fetch_and( 1 );
    }
    else
    {
        m_models.store( 0 );
        m_planMask.store( 0 );
    }
}
bool SyncoShadowProducer::callback( const synco::Record &s, std::int64_t now )
{
    const auto started = std::chrono::steady_clock::now();
    const int  step    = s[synco::cycle] < 0 || s[synco::cycle] == std::numeric_limits<std::int64_t>::max()
                             ? -1
                             : synco::nextProductionStep( s, now );
    if( step < 0 )
    {
        inhibit( false );
        return false;
    }
    const auto                   request = synco::request( s, step );
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    if( !lock.owns_lock() )
    {
        ++m_busy;
        return false;
    }
    if( request.epoch == m_last.epoch && request.generation == m_last.generation && request.cycle == m_last.cycle &&
        request.step == m_last.step && request.targetNs == m_last.targetNs && request.cutoffNs == m_last.cutoffNs )
        return false;
    m_last = request;
    const auto copyNow =
        now +
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count();
    const bool valid = copy( request, s, copyNow, m_copy );
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count();
    m_callbackNs.store( elapsed );
    m_peakNs.store( std::max<std::int64_t>( m_peakNs.load(), elapsed ) ); // Sole callback writer.
    if( valid )
        ++m_commands;
    else
        ++m_missed;
    return valid; // The private image is deliberately never passed to syncopationPublish.
}
bool SyncoShadowProducer::copy(
    const synco::Request &r, const synco::Record &s, std::int64_t now, std::span<float> pixels, int strictMode )
{
    const auto     started = std::chrono::steady_clock::now();
    const unsigned mask    = m_models.load() & m_planMask.load();
    const unsigned first   = strictMode >= 0 ? unsigned( strictMode ) : ( m_requested.load() ? 1 : 0 );
    for( int mode = int( first ); mode >= ( strictMode >= 0 ? strictMode : 0 ); --mode )
    {
        if( !( mask & ( 1u << mode ) ) )
            continue;
        auto origin = m_originNs[mode];
        auto attempt =
            now +
            std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count();
        bool valid = origin > m_lossNs[mode].load() && m_ready[mode]->command( r, s, attempt, pixels );
        if( !valid )
        {
            origin = m_previousOriginNs[mode];
            attempt =
                now + std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started )
                          .count();
            valid = origin > m_lossNs[mode].load() && m_previous[mode]->command( r, s, attempt, pixels );
        }
        const auto finished =
            now +
            std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now() - started ).count();
        if( valid && finished < r.cutoffNs && ( m_models.load() & m_planMask.load() & ( 1u << mode ) ) &&
            origin > m_lossNs[mode].load() )
            return true;
    }
    return false;
}
bool SyncoShadowProducer::command( const synco::Request &r,
                                   const synco::Record  &s,
                                   std::int64_t          now,
                                   std::span<float>      pixels )
{
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    return lock.owns_lock() && copy( r, s, now, pixels );
}
void SyncoShadowProducer::status( OnlineShadowStatus &s ) const
{
    s.m_preparedPlans  = m_prepared.load();
    s.m_shadowSlots    = m_commands.load();
    s.m_missedSlots    = m_missed.load() + m_busy.load();
    s.m_callbackUs     = m_callbackNs.load() < 0 ? -1 : m_callbackNs.load() / 1000.;
    s.m_callbackPeakUs = m_peakNs.load() < 0 ? -1 : m_peakNs.load() / 1000.;
}
} // namespace DDSPC

namespace DDSPC
{
bool SyncoShadowProducer::command_for(
    ShadowMode mode, const synco::Request &r, const synco::Record &s, std::int64_t now, std::span<float> pixels )
{
    std::unique_lock<std::mutex> lock( m_mutex, std::try_to_lock );
    return lock.owns_lock() && copy( r, s, now, pixels, mode == ShadowMode::AccelAssisted ? 1 : 0 );
}
} //namespace DDSPC
