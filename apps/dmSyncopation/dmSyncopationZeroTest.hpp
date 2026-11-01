/** \file dmSyncopationZeroTest.hpp
 * \brief Finite, genuine-main-coordinated zero auxiliary publications and actual timing evidence.
 */
#ifndef dmSyncopationZeroTest_hpp
#define dmSyncopationZeroTest_hpp

#include "dmSyncopation.hpp"

namespace MagAOX
{
namespace app
{

inline bool dmSyncopation::configureZeroTest()
{
    synco::Simulation::Config settings;
    settings.clockOnly       = true;
    settings.steps           = m_requested;
    settings.window          = m_simulationWindow;
    settings.nominalPeriod   = int64_t( std::llround( 1e9 / m_nominalRate ) );
    settings.minInterval     = int64_t( 1e9 / m_maxRate );
    settings.maxInterval     = int64_t( std::ceil( 1e9 / m_minRate ) );
    settings.phaseGain       = m_simulationPhaseGain;
    settings.phaseSlew       = int64_t( std::ceil( m_simulationPhaseSlewUs * 1000 ) );
    settings.timeout         = int64_t( std::ceil( 2e9 / m_minRate ) );
    settings.cadenceWindowNs = int64_t( std::llround( std::max( 10.0, m_simulationCadenceWindowSec ) * 1e9 ) );
    settings.cadenceUpdateNs = int64_t( std::llround( m_simulationCadenceUpdateSec * 1e9 ) );
    m_zeroClockCounter       = 0;
    m_zeroClockPublication   = 0;
    m_zeroPrimary            = 0;
    m_zeroArmed              = false;
    m_zeroHaveCombined       = false;
    m_zeroHaveRelease        = false;
    m_zeroReaderFault        = false;
    return m_zeroClock.configure( settings );
}

inline void dmSyncopation::closeZeroTest()
{
    for( auto *image : { &m_zeroSdk, &m_zeroCombined, &m_zeroChannel } )
    {
        if( image->md )
        {
            ImageStreamIO_closeIm( image );
        }
        *image = {};
    }
    m_zeroAttached = m_zeroIdle = m_zeroHaveCombined = false;
    m_zeroIdleSince                                  = 0;
    m_zeroHaveRelease                                = false;
}

inline bool dmSyncopation::zeroTestPixels() const
{
    if( !m_zeroChannel.md || !m_zeroChannel.array.F || m_zeroChannel.md->write )
    {
        return false;
    }
    const auto *bytes = reinterpret_cast<const unsigned char *>( m_zeroChannel.array.F );

    for( size_t index = 0; index < m_zeroBytes; index += sizeof( uint32_t ) )
    {
        uint32_t bits;
        std::memcpy( &bits, bytes + index, sizeof( bits ) );

        if( bits & 0x7fffffffU )
        {
            return false;
        }
    }
    return true;
}

inline bool dmSyncopation::openZeroTest()
{
    closeZeroTest();
    const auto *primary       = shmimMonitorT::m_imageStream.md;
    const auto  channelInode  = inode( m_zeroTestChannel );
    const auto  combinedInode = inode( m_combinedName );

    if( !primary || !channelInode || !combinedInode || primary->naxis != 2 || primary->datatype != _DATATYPE_FLOAT ||
        primary->location != -1 || !primary->size[0] || !primary->size[1] ||
        uint64_t( primary->size[0] ) * primary->size[1] > 1048576 ||
        ImageStreamIO_openIm( &m_zeroChannel, m_zeroTestChannel.c_str() ) != 0 ||
        ImageStreamIO_openIm( &m_zeroCombined, m_combinedName.c_str() ) != 0 )
    {
        closeZeroTest();
        return false;
    }
    m_zeroWidth            = primary->size[0];
    m_zeroHeight           = primary->size[1];
    m_zeroBytes            = size_t( m_zeroWidth ) * m_zeroHeight * sizeof( float );
    m_zeroExpected         = m_zeroChannel.md->cnt0;
    m_zeroChannelInode     = channelInode;
    m_zeroCombinedInode    = combinedInode;
    m_zeroChannelIdentity  = m_zeroChannel.md->inode;
    m_zeroCombinedIdentity = m_zeroCombined.md->inode;
    m_zeroPrimaryIdentity  = primary->inode;
    m_zeroAttached         = true;

    if( !zeroTestResources( true ) || !zeroTestPixels() )
    {
        closeZeroTest();
        return false;
    }

    const auto sdkInode = m_zeroSdkStream.empty() ? 0 : inode( m_zeroSdkStream );

    if( sdkInode && ImageStreamIO_openIm( &m_zeroSdk, m_zeroSdkStream.c_str() ) == 0 )
    {
        if( m_zeroSdk.md->naxis != 1 || m_zeroSdk.md->datatype != _DATATYPE_UINT64 || m_zeroSdk.md->location != -1 ||
            m_zeroSdk.md->size[0] != MagAOX::dmCompletion::words || inode( m_zeroSdkStream ) != sdkInode )
        {
            ImageStreamIO_closeIm( &m_zeroSdk );
            m_zeroSdk = {};
        }
        else
        {
            m_zeroSdkInode = sdkInode;
        }
    }
    m_zeroIdleSince     = m_now();
    m_zeroNextFileCheck = m_zeroIdleSince + 1000000000;
    return true;
}

inline bool dmSyncopation::zeroTestResources( bool files )
{
    if( !m_zeroAttached || !m_zeroChannel.md || !m_zeroCombined.md )
    {
        return false;
    }

    for( const auto *image : { &m_zeroChannel, &m_zeroCombined } )
    {
        const auto *metadata = image->md;

        if( metadata->naxis != 2 || metadata->datatype != _DATATYPE_FLOAT || metadata->location != -1 ||
            metadata->shared != 1 || metadata->size[0] != m_zeroWidth || metadata->size[1] != m_zeroHeight ||
            metadata->imdatamemsize != m_zeroBytes || !image->array.F || metadata->sem == 0 )
        {
            return false;
        }
    }

    if( m_zeroChannel.md->write || m_zeroChannel.md->cnt0 != m_zeroExpected || !m_zeroChannelIdentity ||
        !m_zeroCombinedIdentity || !m_zeroPrimaryIdentity || m_zeroChannel.md->inode != m_zeroChannelIdentity ||
        m_zeroCombined.md->inode != m_zeroCombinedIdentity ||
        shmimMonitorT::m_imageStream.md->inode != m_zeroPrimaryIdentity ||
        m_zeroChannel.md->inode == m_zeroCombined.md->inode ||
        m_zeroChannel.md->inode == shmimMonitorT::m_imageStream.md->inode )
    {
        return false;
    }

    return !files ||
           ( inode( m_zeroTestChannel ) == m_zeroChannelInode && inode( m_combinedName ) == m_zeroCombinedInode &&
             inode( shmimMonitorT::m_shmimName ) == shmimMonitorT::m_inode );
}

inline bool dmSyncopation::zeroTestCombined( synco::ZeroTiming::Key &key ) const
{
    const auto *metadata = m_zeroCombined.md;

    if( !metadata || metadata->write )
    {
        return false;
    }
    const synco::ZeroTiming::Key sampled{
        metadata->inode, metadata->cnt0, metadata->writetime.tv_sec, metadata->writetime.tv_nsec };
    std::atomic_thread_fence( std::memory_order_acquire );

    if( metadata->write || metadata->inode != sampled.inode || metadata->cnt0 != sampled.counter ||
        metadata->writetime.tv_sec != sampled.sec || metadata->writetime.tv_nsec != sampled.nsec ||
        !synco::ZeroTiming::valid( sampled ) )
    {
        return false;
    }
    key = sampled;
    return true;
}

inline void dmSyncopation::drainZeroTestReader( int64_t now )
{
    const auto dropped = m_simulationReleaseQueue.dropped();

    if( dropped != m_zeroReleaseDrops )
    {
        m_simulationReleaseEpoch.fetch_add( 1, std::memory_order_acq_rel );
    }
    const auto epoch = m_simulationReleaseEpoch.load( std::memory_order_acquire );

    if( epoch != m_zeroReleaseEpoch || dropped != m_zeroReleaseDrops )
    {
        m_zeroHaveRelease = false;
        m_zeroStats.readerDrops += dropped - std::min( dropped, m_zeroReleaseDrops );
        m_zeroReleaseEpoch = epoch;
        m_zeroReleaseDrops = dropped;
    }
    MagAOX::dmSyncoSimulation::Readiness event;
    const auto                           available = std::min<size_t>( m_simulationReleaseQueue.queued(), 64 );

    for( size_t index = 0; index < available && m_simulationReleaseQueue.pop( event ); ++index )
    {
        const auto sampledNow = std::max( now, m_now() );

        if( event.sourceEpoch != epoch || !event.coherent || event.observedNs <= 0 || event.observedNs > sampledNow )
        {
            m_zeroHaveRelease = false;
            ++m_zeroStats.readerInvalid;
            continue;
        }
        const auto &record = event.record;
        const auto &prior  = m_zeroRelease.record;

        if( prior.session &&
            ( record.session < prior.session || ( record.session == prior.session && record.epoch < prior.epoch ) ) )
        {
            m_zeroHaveRelease = false;
            ++m_zeroStats.readerInvalid;
            continue;
        }
        const bool newSource = prior.session && ( prior.session != record.session || prior.epoch != record.epoch );

        if( newSource )
        {
            m_zeroHaveRelease = false;
            m_zeroReaderFault = false;
        }

        if( !record.session || !record.epoch || !record.sequence || !record.publisherPid ||
            ( prior.session == record.session && prior.epoch == record.epoch && prior.sequence &&
              ( record.sequence < prior.sequence || prior.publisherPid != record.publisherPid ) ) )
        {
            m_zeroHaveRelease = false;
            ++m_zeroStats.readerInvalid;
            continue;
        }
        if( record.kind != MagAOX::dmCompletion::Kind::readers || !MagAOX::dmCompletion::readersQualified( record ) ||
            record.finishedNs > event.observedNs )
        {
            m_zeroHaveRelease = false;
            m_zeroRelease     = event;
            if( record.state != MagAOX::dmCompletion::State::busy )
            {
                ++m_zeroStats.readerInvalid;
            }
            continue;
        }
        m_zeroTiming.reader( record, true );

        if( ( record.outcome != MagAOX::dmTiming::Outcome::unknown &&
              record.outcome != MagAOX::dmTiming::Outcome::success ) ||
            ( record.backendResult != MagAOX::dmCompletion::unavailableResult && record.backendResult != 0 ) )
        {
            m_zeroReaderFault = true;
        }
        m_zeroRelease     = event;
        m_zeroHaveRelease = !m_zeroReaderFault;
    }

    if( epoch != m_simulationReleaseEpoch.load( std::memory_order_acquire ) ||
        dropped != m_simulationReleaseQueue.dropped() )
    {
        m_zeroHaveRelease = false;
    }
}

inline bool dmSyncopation::zeroTestReady( const synco::ZeroTiming::Key &key ) const
{
    return m_zeroHaveRelease && !m_zeroReaderFault &&
           m_zeroReleaseEpoch == m_simulationReleaseEpoch.load( std::memory_order_acquire ) &&
           m_zeroReleaseDrops == m_simulationReleaseQueue.dropped() &&
           synco::ZeroTiming::equal( key, synco::ZeroTiming::key( m_zeroRelease.record ) );
}

inline void dmSyncopation::sampleZeroTestSdk( int64_t now )
{
    if( !m_zeroSdk.md )
    {
        return;
    }
    MagAOX::dmCompletion::Record record;
    const bool                   coherent = dev::dmCompletion::read( &m_zeroSdk, record );
    m_zeroTiming.sdk( record, std::max( now, m_now() ), coherent );

    if( coherent && record.kind == MagAOX::dmCompletion::Kind::sdk &&
        ( record.state == MagAOX::dmCompletion::State::error ||
          ( record.outcome != MagAOX::dmTiming::Outcome::unknown &&
            record.outcome != MagAOX::dmTiming::Outcome::success ) ||
          ( record.backendResult != MagAOX::dmCompletion::unavailableResult && record.backendResult != 0 ) ) )
    {
        m_zeroReaderFault = true;
        m_zeroHaveRelease = false;
    }
}

inline bool dmSyncopation::zeroTestWait( int64_t target, uint64_t primary )
{
    m_zeroWaitFailure = ZeroOutcome::cancelled;
    m_zeroWaitResult  = 0;

    while( m_enabled && !shutdown() )
    {
        const auto now = m_now();
        drainZeroTestReader( now );
        sampleZeroTestSdk( now );
        const auto *metadata = shmimMonitorT::m_imageStream.md;

        if( !metadata )
        {
            m_zeroWaitFailure = ZeroOutcome::resourceUnavailable;
            return false;
        }
        if( metadata->write )
        {
            m_zeroWaitFailure = ZeroOutcome::primaryWriting;
            return false;
        }
        if( metadata->cnt0 != primary )
        {
            m_zeroWaitFailure = ZeroOutcome::primaryChanged;
            return false;
        }
        if( !zeroTestResources( false ) )
        {
            m_zeroWaitFailure = ZeroOutcome::resourceUnavailable;
            return false;
        }
        if( m_zeroStats.startedNs && now - m_zeroStats.startedNs >= int64_t( m_zeroDurationSec * 1e9 ) )
        {
            m_zeroWaitFailure = ZeroOutcome::durationExpired;
            return false;
        }
        if( now >= target )
        {
            m_zeroWaitFailure = ZeroOutcome::none;
            return true;
        }
        const auto wake = synco::toTimespec( std::min( target, now + 10000 ) );
        const auto result =
            m_wait ? m_wait( &wake ) : clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr );

        if( result != 0 && result != EINTR )
        {
            m_zeroWaitFailure = ZeroOutcome::waitError;
            m_zeroWaitResult  = result;
            return false;
        }
    }
    return false;
}

inline const char *dmSyncopation::zeroTestReasonName( ZeroOutcome reason )
{
    switch( reason )
    {
    case ZeroOutcome::none:
        return "none";
    case ZeroOutcome::mainAcknowledged:
        return "main_acknowledged";
    case ZeroOutcome::acknowledged:
        return "acknowledged";
    case ZeroOutcome::late:
        return "late";
    case ZeroOutcome::readerMissing:
        return "reader_missing";
    case ZeroOutcome::primaryChanged:
        return "primary_changed";
    case ZeroOutcome::combinedUnclassified:
        return "combined_unclassified";
    case ZeroOutcome::resourceUnavailable:
        return "resource_unavailable";
    case ZeroOutcome::cancelled:
        return "cancelled";
    case ZeroOutcome::started:
        return "started";
    case ZeroOutcome::primaryWriting:
        return "primary_writing";
    case ZeroOutcome::waitError:
        return "wait_error";
    case ZeroOutcome::unacknowledged:
        return "unacknowledged";
    case ZeroOutcome::durationExpired:
        return "duration_expired";
    case ZeroOutcome::finished:
        return "finished";
    case ZeroOutcome::invalidTarget:
        return "invalid_target";
    case ZeroOutcome::ackTimeout:
        return "ack_timeout";
    case ZeroOutcome::publicationFailed:
        return "publication_failed";
    case ZeroOutcome::ackAmbiguous:
        return "ack_ambiguous";
    case ZeroOutcome::preparationLate:
        return "preparation_late";
    }
    return "unknown";
}

inline const char *dmSyncopation::zeroTestStageName( ZeroStage stage )
{
    switch( stage )
    {
    case ZeroStage::lifecycle:
        return "lifecycle";
    case ZeroStage::mainAcknowledgement:
        return "main_acknowledgement";
    case ZeroStage::wait:
        return "wait";
    case ZeroStage::deadline:
        return "deadline";
    case ZeroStage::combined:
        return "combined";
    case ZeroStage::reader:
        return "reader";
    case ZeroStage::finalGuard:
        return "final_guard";
    case ZeroStage::postCopyGuard:
        return "post_copy_guard";
    case ZeroStage::publication:
        return "publication";
    case ZeroStage::acknowledgement:
        return "acknowledgement";
    case ZeroStage::preparation:
        return "preparation";
    case ZeroStage::publicationGuard:
        return "publication_guard";
    }
    return "unknown";
}

inline const char *dmSyncopation::zeroTestActionName( const ZeroEvent &event )
{
    const auto outcome = static_cast<ZeroOutcome>( event.outcome );

    if( event.step < 0 )
    {
        return zeroTestReasonName( outcome );
    }
    if( event.step == 0 )
    {
        return outcome == ZeroOutcome::mainAcknowledged ? "main_acknowledged" : "main_unmatched";
    }
    if( outcome == ZeroOutcome::acknowledged )
    {
        return "acknowledged";
    }
    if( outcome == ZeroOutcome::unacknowledged )
    {
        return "unacknowledged";
    }
    return "skipped";
}

inline void dmSyncopation::zeroTestEvent( int                           step,
                                          ZeroOutcome                   outcome,
                                          ZeroStage                     stage,
                                          int64_t                       target,
                                          int64_t                       observed,
                                          const synco::ZeroTiming::Key &key,
                                          ZeroOutcome                   reason,
                                          int64_t                       stagedNs )
{
    const auto waitResult = reason == ZeroOutcome::waitError ? m_zeroWaitResult : 0;
    m_zeroEvents.push( { m_zeroTrial,
                         m_zeroPrimary,
                         step,
                         int( outcome ),
                         reason,
                         stage,
                         waitResult,
                         target,
                         observed,
                         key,
                         stagedNs } );
}

inline void dmSyncopation::zeroTestSkip( int                           step,
                                         ZeroOutcome                   reason,
                                         ZeroStage                     stage,
                                         int64_t                       target,
                                         int64_t                       observed,
                                         const synco::ZeroTiming::Key &key,
                                         int64_t                       stagedNs )
{
    ++m_zeroStats.skipped;

    switch( reason )
    {
    case ZeroOutcome::late:
        ++m_zeroStats.late;
        break;
    case ZeroOutcome::preparationLate:
        ++m_zeroStats.preparationLate;
        break;
    case ZeroOutcome::readerMissing:
        ++m_zeroStats.readerMissing;
        break;
    case ZeroOutcome::primaryChanged:
        ++m_zeroStats.primaryChanged;
        break;
    case ZeroOutcome::primaryWriting:
        ++m_zeroStats.primaryWriting;
        break;
    case ZeroOutcome::cancelled:
        ++m_zeroStats.cancelled;
        break;
    case ZeroOutcome::durationExpired:
        ++m_zeroStats.durationExpired;
        break;
    case ZeroOutcome::waitError:
        ++m_zeroStats.waitError;
        break;
    case ZeroOutcome::combinedUnclassified:
        ++m_zeroStats.combinedUnclassified;
        break;
    case ZeroOutcome::invalidTarget:
        ++m_zeroStats.invalidTarget;
        break;
    default:
        ++m_zeroStats.resourceUnavailable;
        break;
    }
    zeroTestEvent( step, reason, stage, target, observed, key, reason, stagedNs );
}

inline int dmSyncopation::processZeroTestPrimary( int64_t observed, int64_t receipt )
{
    const auto *metadata = shmimMonitorT::m_imageStream.md;

    if( !metadata || metadata->write || receipt <= 0 || receipt > observed )
    {
        return 0;
    }
    const uint64_t primary = metadata->cnt0;
    const timespec stamp{ metadata->writetime.tv_sec, metadata->writetime.tv_nsec };
    std::atomic_thread_fence( std::memory_order_acquire );

    if( metadata->write || metadata->cnt0 != primary || metadata->writetime.tv_sec != stamp.tv_sec ||
        metadata->writetime.tv_nsec != stamp.tv_nsec || primary == m_zeroPrimary )
    {
        return 0;
    }
    m_zeroPrimary = primary;

    if( m_zeroClockCounter && ( m_zeroClockCounter == UINT64_MAX || primary != m_zeroClockCounter + 1 ) )
    {
        m_zeroClock.reset();
        m_zeroHaveCombined = false;
        m_zeroHaveRelease  = false;
        m_zeroStats.active = false;
    }
    const auto publication = simulationTimestamp( stamp );

    if( publication <= 0 )
    {
        m_zeroClock.reset();
        m_zeroHaveCombined = false;
        return 0;
    }
    const auto interval = primary == m_zeroClockCounter + 1 && publication > m_zeroClockPublication
                              ? publication - m_zeroClockPublication
                              : 0;
    m_zeroClock.primary( primary, observed, interval );
    m_zeroClockCounter     = primary;
    m_zeroClockPublication = publication;
    drainZeroTestReader( m_now() );
    sampleZeroTestSdk( m_now() );

    if( !m_zeroAttached && observed >= m_zeroNextAttach )
    {
        m_zeroNextAttach = observed + 1000000000;
        openZeroTest();
    }

    if( !m_zeroAttached )
    {
        m_reason = "zero channel unavailable or nonzero";
        return 0;
    }

    if( !zeroTestResources( observed >= m_zeroNextFileCheck ) || ( !m_zeroIdle && !zeroTestPixels() ) )
    {
        ++m_zeroStats.resourceFaults;
        m_enabled             = false;
        m_zeroStats.active    = false;
        m_zeroStats.warming   = false;
        m_zeroStats.stoppedNs = m_now();
        m_reason              = "zero trial resource/ownership changed";
        closeZeroTest();
        return 0;
    }

    if( observed >= m_zeroNextFileCheck )
    {
        m_zeroNextFileCheck = observed + 1000000000;

        if( m_zeroSdk.md && inode( m_zeroSdkStream ) != m_zeroSdkInode )
        {
            ImageStreamIO_closeIm( &m_zeroSdk );
            m_zeroSdk = {};
        }
    }
    m_zeroIdle = m_zeroIdle || observed - m_zeroIdleSince >= 1000000000;
    synco::GridTime grid;
    uint64_t        periodQ32  = 0;
    const bool      clockReady = m_zeroClock.frozenGrid( primary, grid, periodQ32 );
    const auto      period     = periodQ32 / MagAOX::dmSyncoSimulation::periodScale;
    const bool      rateReady  = period && std::abs( 1e9 / double( period ) - 2000.0 ) <= 2000 * m_rateTolerance;

    if( !m_enabled )
    {
        m_zeroArmed        = false;
        m_zeroStats.active = m_zeroStats.warming = false;
        zeroTestCombined( m_zeroLastCombined );
        m_zeroHaveCombined = synco::ZeroTiming::valid( m_zeroLastCombined );
        return 0;
    }

    if( !m_zeroArmed )
    {
        m_zeroStats = {};
        m_zeroTiming.reset();

        if( m_zeroHaveRelease )
        {
            m_zeroTiming.reader( m_zeroRelease.record );
        }
        ++m_zeroTrial;
        m_zeroArmed   = true;
        m_zeroArmedAt = observed;
    }
    m_zeroStats.warming = !clockReady || !rateReady || !m_zeroIdle;

    if( !m_zeroStats.startedNs && m_zeroArmedAt > 0 && observed - m_zeroArmedAt >= 30000000000LL )
    {
        m_enabled          = false;
        m_zeroStats.active = m_zeroStats.warming = false;
        m_zeroStats.finished                     = true;
        m_zeroStats.stoppedNs                    = observed;
        m_reason                                 = "zero trial warmup timed out";
        return 0;
    }

    if( m_zeroStats.startedNs && observed - m_zeroStats.startedNs >= int64_t( m_zeroDurationSec * 1e9 ) )
    {
        m_enabled          = false;
        m_zeroStats.active = m_zeroStats.warming = false;
        m_zeroStats.finished                     = true;
        m_zeroStats.stoppedNs                    = observed;
        m_reason                                 = "zero trial finished";
        zeroTestEvent( -1, ZeroOutcome::finished, ZeroStage::lifecycle, 0, observed, {}, ZeroOutcome::durationExpired );
        return 0;
    }

    if( m_zeroStats.warming )
    {
        m_reason = "zero trial clock/ownership warming";
        zeroTestCombined( m_zeroLastCombined );
        m_zeroHaveCombined = synco::ZeroTiming::valid( m_zeroLastCombined );
        return 0;
    }

    if( !m_zeroStats.startedNs )
    {
        m_zeroStats.startedNs = observed;
        m_zeroStats.active    = true;
        m_reason              = "zero trial active";
        zeroTestEvent( -1, ZeroOutcome::started, ZeroStage::lifecycle, grid.ns, observed );
    }

    m_zeroStats.active = true;
    ++m_zeroStats.mainObserved;
    synco::ZeroTiming::Key main;
    bool                   mainMatched  = false;
    const auto             mainDeadline = std::min( observed + int64_t( m_outputTimeoutUs * 1000 ),
                                        synco::stepGrid( grid, periodQ32, 1, m_requested ).ns );

    while( zeroTestWait( m_now(), primary ) && m_now() <= mainDeadline )
    {
        if( zeroTestCombined( main ) )
        {
            if( m_zeroHaveCombined && main.inode == m_zeroLastCombined.inode &&
                m_zeroLastCombined.counter != UINT64_MAX && main.counter == m_zeroLastCombined.counter + 1 &&
                main.sec * 1000000000LL + main.nsec >= publication )
            {
                mainMatched = true;
                break;
            }

            if( !m_zeroHaveCombined || main.inode != m_zeroLastCombined.inode ||
                ( main.counter != m_zeroLastCombined.counter &&
                  ( m_zeroLastCombined.counter == UINT64_MAX || main.counter != m_zeroLastCombined.counter + 1 ) ) )
            {
                break;
            }
        }

        if( !zeroTestWait( std::min( mainDeadline + 1, m_now() + 1000 ), primary ) )
        {
            break;
        }
    }

    if( !mainMatched )
    {
        ++m_zeroStats.mainUnmatched;
        const auto reason =
            m_zeroWaitFailure == ZeroOutcome::none ? ZeroOutcome::combinedUnclassified : m_zeroWaitFailure;
        const auto rejectedAt = m_now();
        zeroTestEvent(
            0, ZeroOutcome::combinedUnclassified, ZeroStage::mainAcknowledgement, grid.ns, rejectedAt, main, reason );

        for( int step = 1; step < m_requested; ++step )
        {
            const auto target = synco::stepGrid( grid, periodQ32, step, m_requested ).ns;
            zeroTestSkip( step, reason, ZeroStage::mainAcknowledgement, target, rejectedAt, main );
        }
        m_zeroLastCombined = main;
        m_zeroHaveCombined = synco::ZeroTiming::valid( main );
        return 0;
    }
    ++m_zeroStats.mainMatched;
    m_zeroLastCombined           = main;
    bool        impactClassified = false;
    const auto &outputImage      = m_zeroCombined;
    const auto *trace            = outputImage.streamproctrace;

    if( receipt > 0 && trace && outputImage.md->NBproctrace > 0 && !outputImage.md->write )
    {
        const auto     rows          = outputImage.md->NBproctrace;
        const auto     trigger       = trace[0].trigger_inode;
        const auto     tracedCounter = trace[0].cnt0;
        const auto     pid           = trace[0].procwrite_PID;
        const timespec start{ trace[0].ts_procstart.tv_sec, trace[0].ts_procstart.tv_nsec };
        const timespec updated{ trace[0].ts_streamupdate.tv_sec, trace[0].ts_streamupdate.tv_nsec };
        const auto     startNs  = simulationTimestamp( start );
        const auto     updateNs = simulationTimestamp( updated );
        const timespec combinedStamp{ time_t( main.sec ), long( main.nsec ) };
        const auto     combinedNs = simulationTimestamp( combinedStamp );
        std::atomic_thread_fence( std::memory_order_acquire );
        synco::ZeroTiming::Key verified;
        impactClassified =
            trigger == metadata->inode && tracedCounter == primary && pid > 0 && publication > 0 && startNs > 0 &&
            updateNs > 0 && startNs >= publication && updateNs >= startNs && combinedNs >= updateNs &&
            zeroTestCombined( verified ) && synco::ZeroTiming::equal( main, verified ) &&
            outputImage.md->NBproctrace == rows && trace[0].trigger_inode == trigger &&
            trace[0].cnt0 == tracedCounter && trace[0].procwrite_PID == pid &&
            trace[0].ts_procstart.tv_sec == start.tv_sec && trace[0].ts_procstart.tv_nsec == start.tv_nsec &&
            trace[0].ts_streamupdate.tv_sec == updated.tv_sec && trace[0].ts_streamupdate.tv_nsec == updated.tv_nsec;
    }

    if( impactClassified )
    {
        m_zeroTiming.main( main, primary, receipt );
    }
    else
    {
        ++m_zeroStats.mainImpactUnclassified;
    }
    zeroTestEvent( 0, ZeroOutcome::mainAcknowledged, ZeroStage::mainAcknowledgement, grid.ns, observed, main );

    for( int step = 1; step < m_requested; ++step )
    {
        const auto             target = synco::stepGrid( grid, periodQ32, step, m_requested ).ns;
        synco::ZeroTiming::Key combined;
        int64_t                stagedNs = 0;

        if( !target )
        {
            zeroTestSkip( step, ZeroOutcome::invalidTarget, ZeroStage::preparation, target, m_now() );
            continue;
        }
        bool preparationRejected = false;

        // The ideal zero payload is always available. Wait only for actual shared-reader clearance.
        while( true )
        {
            const auto polledAt = m_now();

            if( !zeroTestWait( polledAt, primary ) )
            {
                zeroTestSkip( step, m_zeroWaitFailure, ZeroStage::preparation, target, m_now() );
                preparationRejected = true;
                break;
            }
            if( !zeroTestCombined( combined ) || !synco::ZeroTiming::equal( combined, m_zeroLastCombined ) )
            {
                zeroTestSkip(
                    step, ZeroOutcome::combinedUnclassified, ZeroStage::preparation, target, m_now(), combined );
                preparationRejected = true;
                break;
            }
            if( zeroTestReady( combined ) && m_zeroRelease.observedNs <= target )
            {
                break;
            }
            const auto checkedAt = m_now();

            if( checkedAt >= target )
            {
                zeroTestSkip( step, ZeroOutcome::readerMissing, ZeroStage::preparation, target, checkedAt, combined );
                preparationRejected = true;
                break;
            }
            if( !zeroTestWait( std::min( target, checkedAt + 10000 ), primary ) )
            {
                zeroTestSkip( step, m_zeroWaitFailure, ZeroStage::preparation, target, m_now(), combined );
                preparationRejected = true;
                break;
            }
        }

        if( preparationRejected )
        {
            continue;
        }
        const auto  preparedKey     = combined;
        const auto  preparedCounter = m_zeroExpected;
        ZeroOutcome guard           = ZeroOutcome::none;

        if( !m_enabled )
        {
            guard = ZeroOutcome::cancelled;
        }
        else if( metadata->write )
        {
            guard = ZeroOutcome::primaryWriting;
        }
        else if( metadata->cnt0 != primary )
        {
            guard = ZeroOutcome::primaryChanged;
        }
        else if( !zeroTestResources( false ) || preparedCounter == UINT64_MAX )
        {
            guard = ZeroOutcome::resourceUnavailable;
        }

        if( guard != ZeroOutcome::none )
        {
            zeroTestSkip( step, guard, ZeroStage::preparation, target, m_now(), preparedKey );
            continue;
        }
        const auto preparationStart = m_now();

        if( preparationStart > target )
        {
            zeroTestSkip(
                step, ZeroOutcome::preparationLate, ZeroStage::preparation, target, preparationStart, preparedKey );
            continue;
        }
        m_zeroChannel.md->write = 1;
        std::memset( m_zeroChannel.array.F, 0, m_zeroBytes );
        std::atomic_thread_fence( std::memory_order_release );
        m_zeroChannel.md->write = 0;
        ++m_zeroStats.staged;
        stagedNs = m_now();

        if( stagedNs > target )
        {
            zeroTestSkip(
                step, ZeroOutcome::preparationLate, ZeroStage::preparation, target, stagedNs, preparedKey, stagedNs );
            continue;
        }
        if( !zeroTestResources( false ) || m_zeroExpected != preparedCounter )
        {
            zeroTestSkip( step,
                          ZeroOutcome::resourceUnavailable,
                          ZeroStage::preparation,
                          target,
                          m_now(),
                          preparedKey,
                          stagedNs );
            continue;
        }
        if( !zeroTestWait( target, primary ) )
        {
            zeroTestSkip( step, m_zeroWaitFailure, ZeroStage::wait, target, m_now(), preparedKey, stagedNs );
            continue;
        }
        const auto now = m_now();

        if( now - target > int64_t( m_zeroLateUs * 1000 ) )
        {
            zeroTestSkip( step, ZeroOutcome::late, ZeroStage::deadline, target, now, preparedKey, stagedNs );
            continue;
        }
        if( !zeroTestCombined( combined ) || !synco::ZeroTiming::equal( combined, preparedKey ) )
        {
            zeroTestSkip(
                step, ZeroOutcome::combinedUnclassified, ZeroStage::publicationGuard, target, now, combined, stagedNs );
            continue;
        }
        drainZeroTestReader( now );

        if( !zeroTestReady( preparedKey ) )
        {
            zeroTestSkip(
                step, ZeroOutcome::readerMissing, ZeroStage::publicationGuard, target, now, preparedKey, stagedNs );
            continue;
        }
        int64_t guardAt = 0;

        if( !m_enabled )
        {
            guard = ZeroOutcome::cancelled;
        }
        else if( metadata->write )
        {
            guard = ZeroOutcome::primaryWriting;
        }
        else if( metadata->cnt0 != primary )
        {
            guard = ZeroOutcome::primaryChanged;
        }
        else if( !zeroTestResources( false ) || m_zeroExpected != preparedCounter || preparedCounter == UINT64_MAX )
        {
            guard = ZeroOutcome::resourceUnavailable;
        }
        else if( ( guardAt = m_now() ) - target > int64_t( m_zeroLateUs * 1000 ) )
        {
            guard = ZeroOutcome::late;
        }

        if( guard != ZeroOutcome::none )
        {
            zeroTestSkip( step,
                          guard,
                          ZeroStage::publicationGuard,
                          target,
                          guard == ZeroOutcome::late ? guardAt : m_now(),
                          preparedKey,
                          stagedNs );
            continue;
        }
        m_zeroChannel.md->write = 1;
        const auto published    = ImageStreamIO_UpdateIm( &m_zeroChannel );
        ++m_zeroExpected;
        const auto postedAt = m_now();
        ++m_zeroStats.posted;
        m_zeroHaveRelease        = false;
        const auto  ackDeadline  = postedAt + int64_t( m_outputTimeoutUs * 1000 );
        bool        acknowledged = false;
        ZeroOutcome ackReason    = ZeroOutcome::ackTimeout;
        ZeroStage   ackStage     = ZeroStage::acknowledgement;

        if( published == 0 && m_zeroChannel.md->cnt0 == m_zeroExpected && !m_zeroChannel.md->write )
        {
            while( zeroTestWait( m_now(), primary ) && m_now() <= ackDeadline )
            {
                if( zeroTestCombined( combined ) && combined.counter != m_zeroLastCombined.counter )
                {
                    if( combined.inode != m_zeroLastCombined.inode || m_zeroLastCombined.counter == UINT64_MAX ||
                        combined.counter != m_zeroLastCombined.counter + 1 )
                    {
                        ackReason = ZeroOutcome::ackAmbiguous;
                    }
                    else if( metadata->write )
                    {
                        ackReason = ZeroOutcome::primaryWriting;
                    }
                    else if( metadata->cnt0 != primary )
                    {
                        ackReason = ZeroOutcome::primaryChanged;
                    }
                    else
                    {
                        acknowledged = true;
                    }
                    break;
                }

                if( !zeroTestWait( std::min( ackDeadline + 1, m_now() + 1000 ), primary ) )
                {
                    break;
                }
            }

            if( !acknowledged && m_zeroWaitFailure != ZeroOutcome::none )
            {
                ackReason = m_zeroWaitFailure;
            }
        }
        else
        {
            ackReason = ZeroOutcome::publicationFailed;
            ackStage  = ZeroStage::publication;
        }

        if( acknowledged )
        {
            ++m_zeroStats.acknowledged;
            m_zeroLastCombined = combined;
            zeroTestEvent( step,
                           ZeroOutcome::acknowledged,
                           ZeroStage::acknowledgement,
                           target,
                           postedAt,
                           combined,
                           ZeroOutcome::none,
                           stagedNs );
        }
        else
        {
            ++m_zeroStats.unacknowledged;
            zeroTestEvent( step, ZeroOutcome::unacknowledged, ackStage, target, postedAt, {}, ackReason, stagedNs );
        }
    }
    return 0;
}

inline int dmSyncopation::setupZeroTestINDI()
{
    if( !m_zeroTestEnabled )
    {
        return 0;
    }

    if( createROIndiNumber( m_indiP_zeroTest, "zero_test" ) < 0 )
    {
        return -1;
    }

    for( const auto *name : { "enabled",
                              "labelVersion",
                              "idealPublisher",
                              "steps",
                              "targetHz",
                              "active",
                              "warming",
                              "finished",
                              "elapsedSec",
                              "durationSec",
                              "posted",
                              "staged",
                              "acknowledged",
                              "unacknowledged",
                              "skipped",
                              "late",
                              "preparationLate",
                              "readerMissing",
                              "primaryChanged",
                              "primaryWriting",
                              "resourceUnavailable",
                              "waitError",
                              "durationExpired",
                              "invalidTarget",
                              "cancelled",
                              "combinedUnclassified",
                              "resourceFaults",
                              "readerInvalid",
                              "readerDrops",
                              "eventsDropped",
                              "mainObserved",
                              "mainMatched",
                              "mainUnmatched",
                              "mainImpactUnclassified",
                              "mainSdkMatched",
                              "mainSdkUnmatched",
                              "mainReaderMatched",
                              "mainTimingInvalid",
                              "mainTimingEvicted",
                              "mainTimingSourceResets",
                              "primaryToSdkMeanUs",
                              "primaryToSdkMaxUs",
                              "controllerToSdkMeanUs",
                              "controllerToSdkMaxUs",
                              "mainSdkServiceMeanUs",
                              "mainSdkServiceMaxUs",
                              "sdkGaps",
                              "sdkFailed" } )
    {
        m_indiP_zeroTest.add( pcf::IndiElement( name ) );
        m_indiP_zeroTest[name] = 0.0;
    }
    return registerIndiPropertyReadOnly( m_indiP_zeroTest );
}

inline void dmSyncopation::updateZeroTest()
{
    if( !m_zeroTestEnabled )
    {
        return;
    }
    ZeroStats                  stats;
    synco::ZeroTiming::Summary timing;
    uint64_t                   trial    = 0;
    double                     targetHz = 0;
    int64_t                    elapsed  = 0;
    int                        steps    = 0;
    bool                       copied   = false;

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );

        if( lock.owns_lock() )
        {
            const auto now = m_now();
            drainZeroTestReader( now );
            sampleZeroTestSdk( now );

            if( m_enabled && !m_zeroStats.startedNs && m_zeroArmedAt > 0 && now - m_zeroArmedAt >= 30000000000LL )
            {
                m_enabled          = false;
                m_zeroStats.active = m_zeroStats.warming = false;
                m_zeroStats.finished                     = true;
                m_zeroStats.stoppedNs                    = now;
                m_reason                                 = "zero trial warmup timed out";
            }

            if( m_zeroStats.startedNs && m_enabled &&
                now - m_zeroStats.startedNs >= int64_t( m_zeroDurationSec * 1e9 ) )
            {
                m_enabled          = false;
                m_zeroStats.active = m_zeroStats.warming = false;
                m_zeroStats.finished                     = true;
                m_zeroStats.stoppedNs                    = now;
                m_reason                                 = "zero trial finished";
            }
            stats             = m_zeroStats;
            timing            = m_zeroTiming.summary();
            trial             = m_zeroTrial;
            steps             = m_requested;
            const auto period = m_zeroClock.summary().cadence.periodQ32;
            targetHz = period ? double( m_requested ) * 1e9 * MagAOX::dmSyncoSimulation::periodScale / period : 0;
            elapsed  = stats.startedNs
                           ? std::max<int64_t>( 0, ( stats.stoppedNs ? stats.stoppedNs : now ) - stats.startedNs )
                           : 0;
            copied   = true;
        }
    }

    if( copied )
    {
        const std::vector<std::string> names{ "enabled",
                                              "labelVersion",
                                              "idealPublisher",
                                              "steps",
                                              "targetHz",
                                              "active",
                                              "warming",
                                              "finished",
                                              "elapsedSec",
                                              "durationSec",
                                              "posted",
                                              "staged",
                                              "acknowledged",
                                              "unacknowledged",
                                              "skipped",
                                              "late",
                                              "preparationLate",
                                              "readerMissing",
                                              "primaryChanged",
                                              "primaryWriting",
                                              "resourceUnavailable",
                                              "waitError",
                                              "durationExpired",
                                              "invalidTarget",
                                              "cancelled",
                                              "combinedUnclassified",
                                              "resourceFaults",
                                              "readerInvalid",
                                              "readerDrops",
                                              "eventsDropped",
                                              "mainObserved",
                                              "mainMatched",
                                              "mainUnmatched",
                                              "mainImpactUnclassified",
                                              "mainSdkMatched",
                                              "mainSdkUnmatched",
                                              "mainReaderMatched",
                                              "mainTimingInvalid",
                                              "mainTimingEvicted",
                                              "mainTimingSourceResets",
                                              "primaryToSdkMeanUs",
                                              "primaryToSdkMaxUs",
                                              "controllerToSdkMeanUs",
                                              "controllerToSdkMaxUs",
                                              "mainSdkServiceMeanUs",
                                              "mainSdkServiceMaxUs",
                                              "sdkGaps",
                                              "sdkFailed" };
        const std::vector<double>      values{ 1,
                                          3,
                                          1,
                                          double( steps ),
                                          targetHz,
                                          stats.active ? 1.0 : 0.0,
                                          stats.warming ? 1.0 : 0.0,
                                          stats.finished ? 1.0 : 0.0,
                                          elapsed * 1e-9,
                                          m_zeroDurationSec,
                                          double( stats.posted ),
                                          double( stats.staged ),
                                          double( stats.acknowledged ),
                                          double( stats.unacknowledged ),
                                          double( stats.skipped ),
                                          double( stats.late ),
                                          double( stats.preparationLate ),
                                          double( stats.readerMissing ),
                                          double( stats.primaryChanged ),
                                          double( stats.primaryWriting ),
                                          double( stats.resourceUnavailable ),
                                          double( stats.waitError ),
                                          double( stats.durationExpired ),
                                          double( stats.invalidTarget ),
                                          double( stats.cancelled ),
                                          double( stats.combinedUnclassified ),
                                          double( stats.resourceFaults ),
                                          double( stats.readerInvalid ),
                                          double( stats.readerDrops ),
                                          double( m_zeroEvents.dropped() ),
                                          double( stats.mainObserved ),
                                          double( stats.mainMatched ),
                                          double( stats.mainUnmatched ),
                                          double( stats.mainImpactUnclassified ),
                                          double( timing.matched ),
                                          double( timing.classified - std::min( timing.classified, timing.matched ) ),
                                          double( timing.readerMatched ),
                                          double( timing.invalid ),
                                          double( timing.evicted ),
                                          double( timing.sourceResets ),
                                          timing.primaryToSdk.meanNs * 1e-3,
                                          timing.primaryToSdk.count ? timing.primaryToSdk.maxNs * 1e-3 : -1,
                                          timing.controllerToSdk.meanNs * 1e-3,
                                          timing.controllerToSdk.count ? timing.controllerToSdk.maxNs * 1e-3 : -1,
                                          timing.sdkService.meanNs * 1e-3,
                                          timing.sdkService.count ? timing.sdkService.maxNs * 1e-3 : -1,
                                          double( timing.sdkGaps ),
                                          double( timing.failed ) };
        updateIfChanged( m_indiP_zeroTest, names, values, stats.active ? INDI_BUSY : INDI_IDLE );
        updateSwitchIfChanged( m_indiP_enable,
                               "toggle",
                               m_enabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               m_enabled ? INDI_BUSY : INDI_IDLE );
        std::ostringstream output;
        output << "[zero_test] kind=summary labelVersion=3 idealPublisher=1 trial=" << trial
               << " active=" << stats.active << " warming=" << stats.warming << " finished=" << stats.finished
               << " steps=" << steps << " targetHz=" << targetHz << " elapsedSec=" << elapsed * 1e-9
               << " posted=" << stats.posted << " staged=" << stats.staged << " acknowledged=" << stats.acknowledged
               << " unacknowledged=" << stats.unacknowledged << " skipped=" << stats.skipped << " late=" << stats.late
               << " preparationLate=" << stats.preparationLate << " readerMissing=" << stats.readerMissing
               << " primaryChanged=" << stats.primaryChanged << " primaryWriting=" << stats.primaryWriting
               << " resourceUnavailable=" << stats.resourceUnavailable << " waitError=" << stats.waitError
               << " durationExpired=" << stats.durationExpired << " invalidTarget=" << stats.invalidTarget
               << " cancelled=" << stats.cancelled << " combinedUnclassified=" << stats.combinedUnclassified
               << " resourceFaults=" << stats.resourceFaults << " readerInvalid=" << stats.readerInvalid
               << " readerDrops=" << stats.readerDrops << " eventsDropped=" << m_zeroEvents.dropped()
               << " mainObserved=" << stats.mainObserved << " mainMatched=" << stats.mainMatched
               << " mainUnmatched=" << stats.mainUnmatched << " mainImpactUnclassified=" << stats.mainImpactUnclassified
               << " mainSdkMatched=" << timing.matched
               << " mainSdkUnmatched=" << timing.classified - std::min( timing.classified, timing.matched )
               << " mainReaderMatched=" << timing.readerMatched << " mainTimingInvalid=" << timing.invalid
               << " mainTimingEvicted=" << timing.evicted << " mainTimingSourceResets=" << timing.sourceResets
               << " primaryToSdkMeanUs=" << timing.primaryToSdk.meanNs * 1e-3
               << " primaryToSdkMaxUs=" << ( timing.primaryToSdk.count ? timing.primaryToSdk.maxNs * 1e-3 : -1 )
               << " controllerToSdkMeanUs=" << timing.controllerToSdk.meanNs * 1e-3 << " controllerToSdkMaxUs="
               << ( timing.controllerToSdk.count ? timing.controllerToSdk.maxNs * 1e-3 : -1 )
               << " mainSdkServiceMeanUs=" << timing.sdkService.meanNs * 1e-3
               << " mainSdkServiceMaxUs=" << ( timing.sdkService.count ? timing.sdkService.maxNs * 1e-3 : -1 )
               << " sdkGaps=" << timing.sdkGaps << " sdkFailed=" << timing.failed;
        telemeterT::telem<syncoZeroLog>( { output.str() } );
    }
    ZeroEvent  event;
    const auto available = std::min<size_t>( m_zeroEvents.queued(), m_primaryCadenceMaxEvents );

    for( size_t index = 0; index < available && m_zeroEvents.pop( event ); ++index )
    {
        std::ostringstream output;
        output << "[zero_test] kind=event labelVersion=3 trial=" << event.trial << " primary=" << event.primary
               << " step=" << event.step << " outcome=" << event.outcome << " action=" << zeroTestActionName( event )
               << " reason=" << zeroTestReasonName( event.reason ) << " stage=" << zeroTestStageName( event.stage )
               << " waitResult=" << event.waitResult << " targetNs=" << event.targetNs
               << " observedNs=" << event.observedNs << " stagedNs=" << event.stagedNs
               << " combinedInode=" << event.key.inode << " combinedCounter=" << event.key.counter
               << " publicationSec=" << event.key.sec << " publicationNsec=" << event.key.nsec;
        telemeterT::telem<syncoZeroLog>( { output.str() } );
    }
}

} // namespace app
} // namespace MagAOX
#endif
