/** \file dmSyncopationProduction.hpp
 * \brief Continuous independent same-host producer admission on the qualified counter-publication scheduler.
 */
#ifndef dmSyncopation_dmSyncopationProduction_hpp
#define dmSyncopation_dmSyncopationProduction_hpp

#include "dmSyncopation.hpp"

namespace MagAOX
{
namespace app
{

inline bool dmSyncopation::zeroEngine() const
{
    return m_zeroTestEnabled || m_productionEnabled;
}

inline void dmSyncopation::closeProduction()
{
    for( auto &mapping : m_mappings )
    {
        for( auto *image : { &mapping.m_ready, &mapping.m_ack } )
        {
            if( image->md )
            {
                ImageStreamIO_closeIm( image );
            }
            *image = {};
        }
    }
}

inline bool dmSyncopation::openProduction()
{
    if( !openResources() )
    {
        return false;
    }
    for( size_t j = 0; j < m_mappings.size(); ++j )
    {
        auto      &mapping = m_mappings[j];
        const auto source  = m_destinations[j] + "_aux";
        if( !synco::openRecord( mapping.m_ready, source + "_ready", synco::commitWords, true ) ||
            !synco::openRecord( mapping.m_ack, source + "_ack", synco::ackWords, true ) )
        {
            closeResources();
            return false;
        }
        mapping.m_readyInode = inode( source + "_ready" );
        mapping.m_ackInode   = inode( source + "_ack" );
        // On restart, retain the actual destination contribution, never an unaccepted pending source.
        const auto *destination = mapping.m_destination->image()->md;
        const auto  counter     = destination->cnt0;
        if( destination->write )
        {
            closeResources();
            return false;
        }
        std::memcpy( mapping.m_cache.data(), mapping.m_destination->data(), mapping.m_cache.size() * sizeof( float ) );
        std::atomic_thread_fence( std::memory_order_acquire );
        if( destination->write || counter != destination->cnt0 ||
            !std::all_of( mapping.m_cache.begin(),
                          mapping.m_cache.end(),
                          []( float v )
                          {
                              return mx::math::isFinite( v );
                          } ) )
        {
            closeResources();
            return false;
        }
        mapping.m_stage = mapping.m_cache;
    }
    return true;
}

inline void dmSyncopation::publishProductionSchedule( synco::GridTime grid, uint64_t periodQ32, uint64_t primary )
{
    if( !m_timing.md || primary >= INT64_MAX )
    {
        return;
    }
    const auto now = m_now();
    m_record.fill( 0 );
    m_record[synco::version]    = synco::productionVersion;
    m_record[synco::epoch]      = m_epoch;
    m_record[synco::generation] = m_generation;
    m_record[synco::state]      = m_enabled ? synco::armed : synco::invalid;
    m_record[synco::steps]      = m_requested;
    m_record[synco::period]     = int64_t( periodQ32 / MagAOX::dmSyncoSimulation::periodScale );
    m_record[synco::reference]  = now;
    m_record[synco::expires]    = now + 2 * m_record[synco::period];
    m_record[synco::cycle]      = int64_t( primary );
    for( int step = 0; step < m_requested; ++step )
    {
        const auto target = synco::stepGrid( grid, periodQ32, step ? step : m_requested, m_requested ).ns;
        const auto cutoff = target - int64_t( m_productionReadyLeadUs * 1000 );
        if( cutoff > now )
        {
            m_record[synco::cutoffs + step] = cutoff;
            m_record[synco::targets + step] = target;
        }
    }
    m_timing.md->write = 1;
    std::memcpy( m_timing.array.raw, m_record.data(), sizeof( m_record ) );
    ImageStreamIO_UpdateIm( &m_timing );
}

inline bool dmSyncopation::productionSnapshot( const synco::Request &request, int64_t now )
{
    const auto started = m_now();
    for( auto &mapping : m_mappings )
    {
        std::copy( mapping.m_cache.begin(), mapping.m_cache.end(), mapping.m_stage.begin() );
        synco::CommitRecord command{};
        auto               *ready           = mapping.m_ready.md;
        const auto         *source          = mapping.m_source->image()->md;
        const auto         *acknowledgement = mapping.m_ack.md;
        const bool          usable          = mapping.m_inputAvailable && source && source->naxis == 2 &&
                            source->datatype == _DATATYPE_FLOAT && source->location == -1 &&
                            source->size[0] == shmimMonitorT::m_width && source->size[1] == shmimMonitorT::m_height &&
                            source->imdatamemsize == mapping.m_stage.size() * sizeof( float ) &&
                            source->nelement == mapping.m_stage.size() && acknowledgement &&
                            acknowledgement->naxis == 1 && acknowledgement->datatype == _DATATYPE_INT64 &&
                            acknowledgement->location == -1 && acknowledgement->size[0] == synco::ackWords && ready &&
                            ready->naxis == 1 && ready->datatype == _DATATYPE_INT64 && ready->location == -1 &&
                            ready->size[0] == synco::commitWords;
        if( !usable )
        {
            ++mapping.m_rejected;
            continue;
        }
        const auto serial   = ready->cnt0;
        const bool coherent = synco::readRecord( mapping.m_ready, command );
        if( !coherent || !serial || !command[synco::cmdVersion] || serial == mapping.m_readyCounter )
        {
            ++mapping.m_missing;
            continue;
        }
        // Leave a future slot's immutable source outstanding until its own cutoff.
        const auto futureStep = command[synco::cmdStep];
        if( synco::validCommand( command, now ) && synco::validProductionRecord( m_record, now ) && futureStep >= 0 &&
            futureStep < m_record[synco::steps] && command[synco::cmdTarget] > request.targetNs &&
            synco::commandMatches( command, synco::request( m_record, int( futureStep ) ) ) )
        {
            ++mapping.m_missing;
            continue;
        }
        bool accepted = synco::validCommand( command, now ) && synco::commandMatches( command, request ) &&
                        command[synco::cmdReadyNs] <= request.cutoffNs && command[synco::cmdReadyNs] <= now && source &&
                        !source->write && source->cnt0 == uint64_t( command[synco::cmdPayloadCounter] ) &&
                        source->inode == uint64_t( command[synco::cmdPayloadInode] );
        if( accepted )
        {
            std::memcpy( mapping.m_stage.data(), mapping.m_source->data(), mapping.m_stage.size() * sizeof( float ) );
            std::atomic_thread_fence( std::memory_order_acquire );
            accepted = !source->write && source->cnt0 == uint64_t( command[synco::cmdPayloadCounter] ) &&
                       !ready->write && ready->cnt0 == serial &&
                       std::all_of( mapping.m_stage.begin(),
                                    mapping.m_stage.end(),
                                    []( float v )
                                    {
                                        return mx::math::isFinite( v );
                                    } );
        }
        if( accepted )
        {
            ++mapping.m_accepted;
        }
        else
        {
            ++mapping.m_rejected;
            std::copy( mapping.m_cache.begin(), mapping.m_cache.end(), mapping.m_stage.begin() );
        }
        // This releases the producer buffer after a private snapshot or discard, not after DM consumption.
        synco::AckRecord ack{};
        std::copy( command.begin(), command.end(), ack.begin() );
        ack[synco::ackStatus] = accepted ? synco::copied : synco::discarded;
        if( synco::writeRecord( mapping.m_ack, ack ) )
        {
            mapping.m_readyCounter = serial;
        }
    }
    const auto elapsed = m_now() - started;
    m_snapshotMax      = std::max( m_snapshotMax, elapsed );
    if( elapsed > m_bounds.snapshot )
    {
        m_bounds.snapshot = elapsed + 1000;
    }
    return true;
}

inline bool dmSyncopation::productionCopy( uint64_t primary, int64_t cutoff )
{
    const auto started = m_now();
    auto      *input   = shmimMonitorT::m_imageStream.md;
    if( !m_enabled || !input || input->write || input->cnt0 != primary || started + m_bounds.copy > cutoff )
    {
        m_zeroWaitFailure = ZeroOutcome::preparationLate;
        return false;
    }
    for( auto &mapping : m_mappings )
    {
        auto *destination = mapping.m_destination->image()->md;
        if( input->write || input->cnt0 != primary || destination->write || destination->cnt0 != mapping.m_counter )
        {
            m_zeroWaitFailure = ZeroOutcome::primaryChanged;
            return false;
        }
        mapping.m_destination->setWrite( true );
        std::memcpy( mapping.m_destination->data(), mapping.m_stage.data(), mapping.m_stage.size() * sizeof( float ) );
        mapping.m_destination->post();
        std::copy( mapping.m_stage.begin(), mapping.m_stage.end(), mapping.m_cache.begin() );
    }
    const auto finished = m_now();
    m_copyMax           = std::max( m_copyMax, finished - started );
    if( finished - started > m_bounds.copy )
    {
        m_bounds.copy = finished - started + 1000;
    }
    if( finished > cutoff || input->write || input->cnt0 != primary )
    {
        m_zeroWaitFailure = ZeroOutcome::preparationLate;
        return false;
    }
    return true;
}

inline bool dmSyncopation::prepareProductionStep( int step, int64_t target, uint64_t primary )
{
    const synco::Request request{
        m_epoch, m_generation, int64_t( primary ), step, target, target - int64_t( m_productionReadyLeadUs * 1000 ) };
    if( !zeroTestWait( request.cutoffNs, primary ) )
    {
        return false;
    }
    return productionSnapshot( request, m_now() );
}

inline void dmSyncopation::productionStep0( synco::GridTime grid, uint64_t periodQ32, uint64_t primary )
{
    const auto           target = synco::stepGrid( grid, periodQ32, m_requested, m_requested ).ns;
    const synco::Request request{
        m_epoch, m_generation, int64_t( primary + 1 ), 0, target, target - int64_t( m_productionReadyLeadUs * 1000 ) };
    const auto cutoff = target - int64_t( m_productionStep0LeadUs * 1000 );
    bool       ready  = zeroTestWait( request.cutoffNs, primary ) && productionSnapshot( request, m_now() );
    while( ready && m_now() + m_bounds.copy <= cutoff && !zeroTestReady( m_zeroLastCombined ) )
    {
        ready = zeroTestWait( std::min( cutoff, m_now() + 1000 ), primary );
    }
    if( ready && zeroTestReady( m_zeroLastCombined ) && productionCopy( primary, cutoff ) )
    {
        ++m_passiveCopied;
    }
    else
    {
        ++m_passiveSkipped;
    }
}

inline int dmSyncopation::setupProductionINDI()
{
    if( !m_productionEnabled )
    {
        return 0;
    }
    if( createROIndiNumber( m_indiP_production, "production" ) < 0 )
    {
        return -1;
    }
    for( const auto *name : { "enabled", "passiveCopied", "passiveSkipped" } )
    {
        m_indiP_production.add( pcf::IndiElement( name ) );
        m_indiP_production[name] = 0.0;
    }
    for( size_t j = 0; j < m_destinations.size(); ++j )
    {
        for( const auto *prefix : { "accepted", "missing", "rejected" } )
        {
            const auto name = std::string( prefix ) + std::to_string( j );
            m_indiP_production.add( pcf::IndiElement( name ) );
            m_indiP_production[name] = 0.0;
        }
    }
    m_indiP_production["enabled"] = 1.0;
    return registerIndiPropertyReadOnly( m_indiP_production );
}

inline void dmSyncopation::updateProductionINDI()
{
    if( !m_productionEnabled )
    {
        return;
    }
    updateIfChanged( m_indiP_production, "enabled", 1.0 );
    updateIfChanged( m_indiP_production, "passiveCopied", double( m_zeroSnapshot.passiveCopied ) );
    updateIfChanged( m_indiP_production, "passiveSkipped", double( m_zeroSnapshot.passiveSkipped ) );
    for( size_t j = 0; j < m_destinations.size(); ++j )
    {
        updateIfChanged(
            m_indiP_production, "accepted" + std::to_string( j ), double( m_zeroSnapshot.providerAccepted[j] ) );
        updateIfChanged(
            m_indiP_production, "missing" + std::to_string( j ), double( m_zeroSnapshot.providerMissing[j] ) );
        updateIfChanged(
            m_indiP_production, "rejected" + std::to_string( j ), double( m_zeroSnapshot.providerRejected[j] ) );
    }
}

} // namespace app
} // namespace MagAOX
#endif
