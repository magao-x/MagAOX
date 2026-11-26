/** \file dmSyncopationDryRun.hpp
 * \brief Process-local production copies, virtual auxiliary acknowledgements and enable timing capture.
 */
#ifndef dmSyncopationDryRun_hpp
#define dmSyncopationDryRun_hpp

#include "dmSyncopation.hpp"

namespace MagAOX
{
namespace app
{

inline bool dmSyncopation::openDryRunDestinations()
{
    closeResources();
    const auto *primary = shmimMonitorT::m_imageStream.md;

    if( !primary || primary->naxis != 2 || primary->datatype != _DATATYPE_FLOAT || primary->location != -1 ||
        !primary->size[0] || !primary->size[1] || uint64_t( primary->size[0] ) * primary->size[1] > 1048576 ||
        m_destinations.empty() )
    {
        return false;
    }

    const size_t pixels = size_t( primary->size[0] ) * primary->size[1];

    try
    {
        m_mappings.resize( m_destinations.size() );

        for( auto &mapping : m_mappings )
        {
            mapping.m_dryDestination.resize( pixels );
            mapping.m_cache.resize( pixels );
            mapping.m_stage.resize( pixels );
        }
    }
    catch( ... )
    {
        closeResources();
        return false;
    }

    m_resources = true;
    return true;
}

inline int dmSyncopation::postDryRunTrigger()
{
    const auto now     = m_now();
    const auto raw     = m_publicationNow();
    const auto combine = int64_t( std::llround( m_dryCombineUs * 1000 ) );
    const auto release = int64_t( std::llround( m_dryReleaseUs * 1000 ) );

    if( raw <= 0 || raw > INT64_MAX - combine || now <= 0 || now > INT64_MAX - release ||
        m_zeroLastCombined.counter == UINT64_MAX )
    {
        return IMAGESTREAMIO_FAILURE;
    }

    m_dryChannel.writetime = synco::toTimespec( raw );
    ++m_dryChannel.cnt0;

    const auto stamp  = synco::toTimespec( raw + combine );
    m_dryOutput       = { m_dryMain.inode, m_zeroLastCombined.counter + 1, stamp.tv_sec, stamp.tv_nsec };
    m_dryOutputAt     = now + combine;
    m_dryReleaseAt    = now + release;
    m_dryReleaseEpoch = m_zeroReleaseEpoch;
    return IMAGESTREAMIO_SUCCESS;
}

inline void dmSyncopation::captureDryRunCycle( int64_t completed )
{
    if( m_zeroPrimary == 0 || m_zeroPrimary == m_dryCapturedPrimary )
    {
        return;
    }

    m_dryCapturedPrimary = m_zeroPrimary;
    auto &cycle          = m_dryCycle;
    cycle.primary        = m_zeroPrimary;
    cycle.publicationNs =
        simulationTimestamp( { time_t( m_zeroPrimaryPublication.sec ), long( m_zeroPrimaryPublication.nsec ) } );
    cycle.endNs   = completed;
    cycle.posted  = m_zeroStats.posted >= cycle.posted ? m_zeroStats.posted - cycle.posted : 0;
    cycle.skipped = m_zeroStats.skipped >= cycle.skipped ? m_zeroStats.skipped - cycle.skipped : 0;
    cycle.unacknowledged =
        m_zeroStats.unacknowledged >= cycle.unacknowledged ? m_zeroStats.unacknowledged - cycle.unacknowledged : 0;

    auto &capture = m_dryCapture;

    if( capture.enableNs && !capture.firstEnabledNs && cycle.startNs >= capture.enableNs )
    {
        capture.firstEnabledNs = cycle.startNs;
    }

    if( !m_enabled && !capture.enableNs )
    {
        capture.before[capture.beforeNext] = cycle;
        capture.beforeNext                 = ( capture.beforeNext + 1 ) % capture.before.size();
        capture.beforeSize                 = std::min( capture.before.size(), capture.beforeSize + 1 );
    }
    else if( capture.enableNs && ( m_zeroStats.active || capture.afterSize ) &&
             capture.afterSize < capture.after.size() )
    {
        capture.after[capture.afterSize++] = cycle;
    }
}

inline void dmSyncopation::logDryRunCapture()
{
    DryRunCapture capture;
    { //mutex scope
        std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );

        if( !lock.owns_lock() || !m_dryCapture.enableNs ||
            ( m_enabled && m_dryCapture.afterSize < m_dryCapture.after.size() ) )
        {
            return;
        }

        capture      = m_dryCapture;
        m_dryCapture = {};
    }

    std::ostringstream summary;
    summary << "production dry run startup trial=" << capture.trial << " enableNs=" << capture.enableNs
            << " firstEnabledNs=" << capture.firstEnabledNs << " before=" << capture.beforeSize
            << " active=" << capture.afterSize << " combineUs=" << m_dryCombineUs << " releaseUs=" << m_dryReleaseUs;
    log<text_log>( summary.str(), logPrio::LOG_NOTICE );

    const auto logCycle = [this, &capture]( const char *phase, const DryRunCycle &cycle )
    {
        std::ostringstream record;
        record << "production dry run cycle trial=" << capture.trial << " phase=" << phase
               << " primary=" << cycle.primary << " publicationNs=" << cycle.publicationNs
               << " receiptNs=" << cycle.receiptNs << " startNs=" << cycle.startNs << " endNs=" << cycle.endNs
               << " firstCopyNs=" << cycle.firstCopyNs << " maxCopyNs=" << cycle.maxCopyNs << " posted=" << cycle.posted
               << " skipped=" << cycle.skipped << " unacknowledged=" << cycle.unacknowledged;
        log<text_log>( record.str(), logPrio::LOG_NOTICE );
    };

    const auto first = ( capture.beforeNext + capture.before.size() - capture.beforeSize ) % capture.before.size();

    for( size_t index = 0; index < capture.beforeSize; ++index )
    {
        logCycle( "before", capture.before[( first + index ) % capture.before.size()] );
    }

    for( size_t index = 0; index < capture.afterSize; ++index )
    {
        logCycle( "active", capture.after[index] );
    }
}

} // namespace app
} // namespace MagAOX

#endif // dmSyncopationDryRun_hpp
