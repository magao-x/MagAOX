/** \file dmSyncopationChannelZero.hpp
 * \brief Serialize operator-requested channel clearing with live and idle DM publication.
 */
#ifndef dmSyncopationChannelZero_hpp
#define dmSyncopationChannelZero_hpp

#include "dmSyncopation.hpp"

namespace MagAOX
{
namespace app
{

inline bool dmSyncopation::legacySourceCurrent( const IMAGE_METADATA &metadata ) const
{
    return m_channelZeroSourceCutoff == 0 || simulationTimestamp( metadata.writetime ) > m_channelZeroSourceCutoff;
}

inline bool dmSyncopation::zeroDestinations()
{
    auto closeImage = []( IMAGE *image )
    {
        ImageStreamIO_closeIm( image );
        delete image;
    };
    using ImageHandle    = std::unique_ptr<IMAGE, decltype( closeImage )>;
    const auto openImage = [&closeImage]( const std::string &name, ino_t expected ) -> ImageHandle
    {
        auto image = std::make_unique<IMAGE>();

        if( expected == 0 || ImageStreamIO_openIm( image.get(), name.c_str() ) != IMAGESTREAMIO_SUCCESS )
        {
            return { nullptr, closeImage };
        }

        ImageHandle opened( image.release(), closeImage );
        struct stat status{};

        if( fstat( opened->shmfd, &status ) < 0 || status.st_ino != expected )
        {
            return { nullptr, closeImage };
        }

        return opened;
    };

    const auto &primaryName = shmimMonitorT::m_shmimName;
    const auto  primaryFile = inode( primaryName );
    const auto  outputFile  = inode( m_combinedName );
    auto        primary     = openImage( primaryName, primaryFile );
    auto        combined    = openImage( m_combinedName, outputFile );

    if( ( outputFile && !combined ) || ( primaryFile && !primary ) || m_destinations.empty() ||
        ( outputFile && outputFile == primaryFile ) )
    {
        return false;
    }

    // The numbered channels supply geometry even if the combiner is down.
    const auto firstFile = inode( m_destinations.front() );
    auto       first     = openImage( m_destinations.front(), firstFile );

    if( !first || !first->md )
    {
        return false;
    }

    const auto width       = first->md->size[0];
    const auto height      = first->md->size[1];
    const auto count       = uint64_t( width ) * height;
    const auto bytes       = count * sizeof( float );
    const auto validPixels = [width, height, count, bytes]( const IMAGE &image )
    {
        const auto *md = image.md;
        return md && md->shared == 1 && md->location == -1 && md->naxis == 2 && md->datatype == _DATATYPE_FLOAT &&
               md->size[0] == width && md->size[1] == height && md->nelement == count && md->imdatamemsize == bytes &&
               image.array.F;
    };

    if( !count || count > 1048576 || !validPixels( *first ) || firstFile == primaryFile || firstFile == outputFile ||
        first->md->sem == 0 || ( combined && !validPixels( *combined ) ) || ( primary && !validPixels( *primary ) ) )
    {
        return false;
    }

    std::vector<ImageHandle> destinations;
    std::vector<ino_t>       files{ firstFile };
    std::vector<uint64_t>    counters( m_destinations.size() );
    destinations.push_back( std::move( first ) );

    for( size_t index = 1; index < m_destinations.size(); ++index )
    {
        const auto &name  = m_destinations[index];
        const auto  file  = inode( name );
        auto        image = openImage( name, file );

        if( !image || !validPixels( *image ) || file == primaryFile || file == outputFile ||
            std::find( files.begin(), files.end(), file ) != files.end() )
        {
            return false;
        }

        files.push_back( file );
        destinations.push_back( std::move( image ) );
    }

    const std::vector<float> zeros( count, 0.0F );
    const auto               sameFiles = [&]
    {
        if( inode( primaryName ) != primaryFile || inode( m_combinedName ) != outputFile )
        {
            return false;
        }

        for( size_t index = 0; index < files.size(); ++index )
        {
            if( inode( m_destinations[index] ) != files[index] )
            {
                return false;
            }
        }

        return true;
    };
    const auto sample = []( const IMAGE *image, synco::MainTiming::Key &key )
    {
        if( !image )
        {
            return true;
        }

        const auto &md = *image->md;

        if( !md.write && md.cnt0 == 0 )
        {
            key.inode = md.inode;
            return true;
        }

        return synco::MainTiming::sample( image, key );
    };

    // Check several primary frames per ordinary app iteration. Keep the request queued across attempts.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( 6 );

    do
    {
        { //mutex scope
            std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );
            synco::MainTiming::Key       main;

            if( lock.owns_lock() && sample( primary.get(), main ) )
            {
                auto      &request           = m_channelZero;
                const auto raw               = m_publicationNow();
                const auto mainStamp         = simulationTimestamp( { time_t( main.sec ), long( main.nsec ) } );
                const auto mainAge           = main.counter ? raw - mainStamp : INT64_MAX;
                bool       destinationsReady = true;

                for( size_t index = 0; index < destinations.size(); ++index )
                {
                    const auto &md    = *destinations[index]->md;
                    counters[index]   = md.cnt0;
                    destinationsReady = destinationsReady && !md.write;
                }

                if( request.copiedNs && ( request.destinations != files || request.counters != counters ) )
                {
                    request.copiedNs = 0;
                }

                // A primary publication after the copy supplies the trigger during live operation.
                if( destinationsReady && request.copiedNs && main.counter != request.primary &&
                    mainStamp >= request.copiedNs && mainAge >= 0 && sameFiles() )
                {
                    return true;
                }

                const bool             live = mainAge < int64_t( 2e9 / m_minRate );
                synco::MainTiming::Key output;

                if( destinationsReady && ( !request.copiedNs || !live ) && sample( combined.get(), output ) )
                {
                    const auto outputStamp     = simulationTimestamp( { time_t( output.sec ), long( output.nsec ) } );
                    const auto outputAge       = output.counter ? raw - outputStamp : INT64_MAX;
                    const auto readerHold      = m_bounds.delivery + m_bounds.keepout + m_bounds.guard;
                    const auto latestCopyStart = int64_t( 1e9 / m_maxRate ) - m_bounds.copy - m_bounds.early -
                                                 m_bounds.observation - m_bounds.visibility;
                    bool safe = raw > 0 && mainAge >= 0 && outputAge >= readerHold && !live;

                    if( live && mainAge >= 0 && mainAge < latestCopyStart && output.counter && outputAge >= 0 &&
                        outputStamp >= mainStamp )
                    {
                        if( m_productionEnabled )
                        {
                            drainZeroTestReader( m_now() );
                            safe = zeroTestReady( { output.inode, output.counter, output.sec, output.nsec } );
                        }
                        else
                        {
                            safe = outputAge >= readerHold;
                        }
                    }

                    if( safe && sameFiles() )
                    {
                        synco::MainTiming::Key verifyMain, verifyOutput;

                        for( size_t index = 0; index < destinations.size(); ++index )
                        {
                            const auto &md = *destinations[index]->md;
                            safe           = safe && !md.write && md.cnt0 == counters[index];
                        }

                        const auto age = m_publicationNow() - mainStamp;
                        safe           = safe && sample( primary.get(), verifyMain ) &&
                               dmTimingEvents::equal( main, verifyMain ) && sample( combined.get(), verifyOutput ) &&
                               dmTimingEvents::equal( output, verifyOutput ) &&
                               ( !live || ( age >= 0 && age < latestCopyStart ) );

                        if( safe && !request.copiedNs )
                        {
                            request.destinations = files;
                            request.counters     = counters;

                            for( auto &image : destinations )
                            {
                                image->md->write = 1;
                            }

                            for( auto &image : destinations )
                            {
                                MagAOX::streamCopy( image->array.F, zeros.data(), bytes );
                            }

                            const auto copied = std::max( raw, m_publicationNow() );
                            const auto stamp  = synco::toTimespec( copied );

                            for( auto &image : destinations )
                            {
                                image->md->writetime = image->md->atime = stamp;
                                image->md->write                        = 0;
                            }

                            m_channelZeroSourceCutoff = copied;
                            request.copiedNs          = copied;
                            request.primary           = main.counter;
                        }

                        if( safe && !live && counters.front() != UINT64_MAX && sample( primary.get(), verifyMain ) &&
                            dmTimingEvents::equal( main, verifyMain ) && sample( combined.get(), verifyOutput ) &&
                            dmTimingEvents::equal( output, verifyOutput ) )
                        {
                            auto &trigger            = *destinations.front();
                            trigger.md->write        = 1;
                            const auto result        = ImageStreamIO_UpdateIm( &trigger );
                            request.counters.front() = trigger.md->cnt0;

                            if( result == IMAGESTREAMIO_SUCCESS && request.counters.front() == counters.front() + 1 &&
                                !trigger.md->write )
                            {
                                return true;
                            }

                            trigger.md->write = 0;
                            return false;
                        }
                    }
                }
            }
        }

        std::this_thread::sleep_for( std::chrono::microseconds( 10 ) );
    } while( !shutdown() && std::chrono::steady_clock::now() < deadline );

    return false;
}

inline void dmSyncopation::serviceChannelZero()
{
    std::unique_lock<std::mutex> control( m_channelZeroMutex, std::try_to_lock );

    if( !control.owns_lock() )
    {
        return;
    }

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_cycleMutex, std::try_to_lock );

        if( !lock.owns_lock() || !m_channelZero.pending )
        {
            return;
        }
    }

    try
    {
        if( !zeroDestinations() )
        {
            return;
        }
    }
    catch( const std::exception &error )
    {
        log<text_log>( std::string( "synco channel clear still pending: " ) + error.what(), logPrio::LOG_WARNING );
        return;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_cycleMutex );
        closeResources();
        m_nextAttach          = 0;
        m_lastMain            = 0;
        m_havePrimaryCounter  = false;
        m_previousPrimaryTime = 0;
        m_trackingOutput      = false;

        if( m_productionEnabled )
        {
            closeZeroTest();
            m_zeroNextAttach = 0;
            m_zeroArmedAt    = m_now();
        }

        m_channelZero = {};
        m_reason      = "synco channels cleared";
    }

    updateSwitchIfChanged( m_indiP_zero, "request", pcf::IndiElement::Off, INDI_OK );
    log<text_log>( "synco channels cleared and dmcomb triggered", logPrio::LOG_NOTICE );
}

} // namespace app
} // namespace MagAOX

#endif // dmSyncopationChannelZero_hpp
