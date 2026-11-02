/** \file po4ao_control.hpp
 * \brief Frame processing and replay publication, independently testable at the SDK boundary.
 *
 * Included by po4ao_impl.hpp inside MagAOX::app, after the application declaration.
 * \ingroup po4ao_files
 */
#ifndef po4ao_control_hpp
#define po4ao_control_hpp

inline void po4ao::makeIntegratorCommand( const float *observation )
{
    for( int mode = 0; mode < m_nModesControlled; ++mode )
        m_output[mode] = m_integratorGain * observation[mode];
}

inline int po4ao::invalidateReplay( IMAGE &stream, bool opened, IMAGE &metadataStream, bool metadataOpened )
{
    if( !opened || !metadataOpened )
        return -1;
    stream.md[0].write         = 1;
    metadataStream.md[0].write = 1;
    memset( metadataStream.array.raw, 0, 4 * sizeof( uint32_t ) );
    const uint64_t publication = stream.md[0].cnt0 + 1;
    metadataStream.md[0].cnt0  = publication;
    stream.md[0].cnt0          = publication;
    stream.md[0].cnt1          = 0;
    stream.md[0].cnt2          = 0;
    stream.md[0].write         = 0;
    metadataStream.md[0].write = 0;
    ImageStreamIO_sempost( &stream, -1 );
    ImageStreamIO_sempost( &metadataStream, -1 );
    return 0;
}

inline int po4ao::publishReplay( IMAGE   &stream,
                                 bool     opened,
                                 IMAGE   &metadataStream,
                                 bool     metadataOpened,
                                 uint64_t episodeIdentifier,
                                 uint64_t modelId )
{
    if( !opened || !metadataOpened || !m_publishObservations || !m_publishCommands )
        return -1;
    const size_t valid         = std::min( m_publishObservations->size(), m_publishCommands->size() );
    const size_t planeSize     = static_cast<size_t>( m_nModes ) * m_replayBufferSize;
    stream.md[0].write         = 1;
    metadataStream.md[0].write = 1;
    memset( stream.array.raw, 0, 2 * planeSize * sizeof( float ) );
    auto *destination = static_cast<float *>( stream.array.raw );
    for( size_t sample = 0; sample < valid; ++sample )
    {
        memcpy( destination + sample * m_nModes,
                m_publishObservations->getItem( sample ),
                static_cast<size_t>( m_nModes ) * sizeof( float ) );
        memcpy( destination + planeSize + sample * m_nModes,
                m_publishCommands->getItem( sample ),
                static_cast<size_t>( m_nModes ) * sizeof( float ) );
    }
    const uint64_t publication = stream.md[0].cnt0 + 1;
    auto          *metadata    = static_cast<uint32_t *>( metadataStream.array.raw );
    metadata[0]                = static_cast<uint32_t>( valid );
    metadata[1]                = static_cast<uint32_t>( episodeIdentifier );
    metadata[2]                = static_cast<uint32_t>( modelId & 0xFFFFFFFFULL );
    metadata[3]                = static_cast<uint32_t>( modelId >> 32 );
    stream.md[0].cnt1          = 0;
    stream.md[0].cnt2          = 0;
    metadataStream.md[0].cnt0  = publication;
    stream.md[0].cnt0          = publication;
    stream.md[0].write         = 0;
    metadataStream.md[0].write = 0;
    ImageStreamIO_sempost( &stream, -1 );
    ImageStreamIO_sempost( &metadataStream, -1 );
    return 0;
}

inline void po4ao::finishEpisode()
{
    EpisodeBuffer &completed = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
    completed.episode        = m_episodeCounter.fetch_add( 1, std::memory_order_acq_rel ) + 1;
    completed.generation     = m_activeCollectionGeneration;

    int next = -1;
    for( size_t slot = 0; slot < c_episodeSlots; ++slot )
    {
        int expected = 0;
        if( m_episodeStates[slot].compare_exchange_strong( expected, 1, std::memory_order_acq_rel ) )
        {
            next = static_cast<int>( slot );
            break;
        }
    }
    if( next < 0 )
    {
        m_collection.interrupt( completed.generation );
        m_replayCondition.notify_one();
        completed.reset();
        m_replayDrops.fetch_add( 1, std::memory_order_relaxed );
        return;
    }

    const uint64_t write                 = m_readyWrite.load( std::memory_order_relaxed );
    m_readyQueue[write % c_episodeSlots] = m_activeEpisodeSlot;
    m_episodeStates[static_cast<size_t>( m_activeEpisodeSlot )].store( 2, std::memory_order_release );
    m_readyWrite.store( write + 1, std::memory_order_release );
    m_activeEpisodeSlot = next;
    m_episodeBuffers[static_cast<size_t>( next )].reset();
    m_replayCondition.notify_one();
}

inline void po4ao::replayPublisherLoop()
{
    uint64_t previousEpisode       = 0;
    uint64_t invalidatedGeneration = 0;
    while( true )
    {
        std::unique_lock<std::mutex> publicationLock( m_publicationMutex );
        const uint64_t               token = m_collection.token();
        if( WarmupCollection::phase( token ) == CollectionPhase::interrupted &&
            invalidatedGeneration != WarmupCollection::generation( token ) )
        {
            invalidateReplay( m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened );
            invalidatedGeneration = WarmupCollection::generation( token );
        }
        const uint64_t read  = m_readyRead.load( std::memory_order_relaxed );
        const uint64_t write = m_readyWrite.load( std::memory_order_acquire );
        if( read == write )
        {
            publicationLock.unlock();
            if( m_replayShutdown.load( std::memory_order_acquire ) )
                break;
            std::unique_lock<std::mutex> lock( m_replayMutex );
            m_replayCondition.wait_for( lock,
                                        std::chrono::milliseconds( 100 ),
                                        [this, invalidatedGeneration]()
                                        {
                                            const uint64_t current = m_collection.token();
                                            return m_replayShutdown.load( std::memory_order_acquire ) ||
                                                   m_readyRead.load( std::memory_order_relaxed ) !=
                                                       m_readyWrite.load( std::memory_order_acquire ) ||
                                                   ( WarmupCollection::phase( current ) ==
                                                         CollectionPhase::interrupted &&
                                                     WarmupCollection::generation( current ) != invalidatedGeneration );
                                        } );
            continue;
        }

        const int slot = m_readyQueue[read % c_episodeSlots];
        m_readyRead.store( read + 1, std::memory_order_release );
        m_episodeStates[static_cast<size_t>( slot )].store( 3, std::memory_order_release );
        EpisodeBuffer &episode             = m_episodeBuffers[static_cast<size_t>( slot )];
        const uint64_t requestedGeneration = m_collectionGeneration.load( std::memory_order_acquire );
        if( episode.generation != requestedGeneration )
        {
            episode.reset();
            m_episodeStates[static_cast<size_t>( slot )].store( 0, std::memory_order_release );
            continue;
        }
        if( m_publicationGeneration != requestedGeneration )
        {
            m_publishObservations->reset();
            m_publishCommands->reset();
            m_publicationGeneration = requestedGeneration;
            previousEpisode         = 0;
        }
        if( episode.episode != previousEpisode + 1 )
            m_collection.interrupt( episode.generation );
        previousEpisode = episode.episode;
        for( size_t frame = 0; frame < episode.count; ++frame )
        {
            m_publishObservations->add( episode.observations.data() + frame * m_nModes );
            m_publishCommands->add( episode.commands.data() + frame * m_nModes );
        }
        if( publishReplay( m_replayStream,
                           m_replayOpened,
                           m_replayMetadataStream,
                           m_replayMetadataOpened,
                           episode.episode,
                           episode.modelId ) < 0 )
            m_collection.interrupt( episode.generation );
        if( episode.episode == m_warmupEpisodes &&
            m_collection.token() ==
                ( ( episode.generation << 3 ) | static_cast<uint64_t>( CollectionPhase::publishing ) ) )
        {
            const size_t expectedFrames = static_cast<size_t>( m_warmupEpisodes ) * m_iterationsPerEpisode;
            if( m_publishObservations->size() != expectedFrames || m_publishCommands->size() != expectedFrames ||
                publishReplay( m_warmupStream,
                               m_warmupOpened,
                               m_warmupMetadataStream,
                               m_warmupMetadataOpened,
                               episode.episode,
                               episode.modelId ) < 0 ||
                !m_collection.published( episode.generation ) )
            {
                m_collection.interrupt( episode.generation );
                invalidateReplay( m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened );
            }
        }
        episode.reset();
        m_episodeStates[static_cast<size_t>( slot )].store( 0, std::memory_order_release );
    }
}

inline void po4ao::beginWarmupCollection( uint64_t generation, uint64_t control, uint64_t modelId, uint64_t reset )
{
    m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
    m_episodeCounter.store( 0, std::memory_order_release );
    m_activeCollectionGeneration = generation;
    m_collection.begin(
        generation, static_cast<uint64_t>( m_warmupEpisodes ) * m_iterationsPerEpisode, control, modelId, reset );
    m_initialWarmupPending = false;
}

inline int po4ao::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );
    const auto started       = std::chrono::steady_clock::now();
    auto       recordLatency = [this, started]()
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - started );
        const uint32_t latency =
            static_cast<uint32_t>( std::min<int64_t>( elapsed.count(), std::numeric_limits<uint32_t>::max() ) );
        const uint64_t sample = m_latencyCount.fetch_add( 1, std::memory_order_acq_rel );
        m_latencySamples[sample % c_latencySamples].store( latency, std::memory_order_release );
        if( latency > m_latencyDeadlineUs )
            m_deadlineMisses.fetch_add( 1, std::memory_order_relaxed );
    };
    if( curr_src == nullptr || !m_inputState || !m_output || !m_outputOpened )
        return -1;

    const uint64_t sourceCounter = shmimMonitorT::m_imageStream.md[0].cnt0;
    // Queued notifications can refer to the same latest image after a processing stall.
    if( sourceCounterIsDuplicate( sourceCounter, m_lastInputCounter, m_haveInputCounter ) )
    {
        recordLatency();
        return 0;
    }

    activatePendingEngine();
    const uint64_t resetGeneration = m_modelResetRequest.exchange( 0, std::memory_order_acq_rel );
    if( resetGeneration != 0 )
        resetActiveModel( resetGeneration );
    // Keep one request snapshot throughout this frame; later requests await the next output.
    const uint64_t control         = m_controlRequest.load( std::memory_order_acquire );
    const bool     enabled         = ( control & 1 ) != 0;
    const uint64_t modelGeneration = m_modelGeneration.load( std::memory_order_acquire );
    const auto    *observation     = static_cast<const float *>( curr_src );
    const bool     consecutive = sourceCounterIsConsecutive( sourceCounter, m_lastInputCounter, m_haveInputCounter );
    if( m_haveInputCounter && sourceCounter > m_lastInputCounter + 1 )
        m_missedFrames.fetch_add( sourceCounter - m_lastInputCounter - 1, std::memory_order_relaxed );
    m_lastInputCounter = sourceCounter;
    m_haveInputCounter = true;

    if( !consecutive )
    {
        m_observationHistory->reset();
        m_commandHistory->reset();
        m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
    }
    else
        m_observationHistory->add( observation );

    const bool inferenceReady = consecutive && enabled && m_activeEngine != nullptr &&
                                m_activeEngine->generation == modelGeneration &&
                                m_useNeuralNetwork.load( std::memory_order_acquire ) &&
                                m_observationHistory->size() >= static_cast<size_t>( m_activeEngine->historyLength ) &&
                                m_commandHistory->size() + 1 >= static_cast<size_t>( m_activeEngine->historyLength );
    const uint64_t expectedModel       = inferenceReady ? m_activeEngine->modelId : 0;
    const uint64_t recollectRequest    = m_warmupRecollectRequest.load( std::memory_order_acquire );
    const uint64_t recollectGeneration = recollectRequest > m_activeCollectionGeneration ? recollectRequest : 0;
    if( recollectGeneration != 0 || ( m_initialWarmupPending && enabled ) )
        beginWarmupCollection( recollectGeneration != 0 ? recollectGeneration : m_activeCollectionGeneration,
                               control,
                               expectedModel,
                               modelGeneration );
    m_collection.validate( control, expectedModel, modelGeneration, consecutive );

    bool inferenceOk = inferenceReady && runInference( observation );
    if( inferenceReady && !inferenceOk )
    {
        m_inferenceFailures.fetch_add( 1, std::memory_order_relaxed );
        m_collection.interrupt( m_activeCollectionGeneration );
    }
    if( !inferenceOk )
        makeIntegratorCommand( observation );

    const bool exploring = WarmupCollection::phase( m_collection.token() ) == CollectionPhase::collecting;
    if( exploring && !m_explorationNoise.empty() )
    {
        const uint64_t frame = m_collection.frames();
        const float   *noise = m_explorationNoise.data() + ( frame % m_explorationNoiseRows ) * m_explorationNoiseCols;
        addExploration( m_output.get(),
                        static_cast<size_t>( m_nModesControlled ),
                        noise,
                        frame,
                        static_cast<uint64_t>( m_warmupEpisodes ) * m_iterationsPerEpisode,
                        m_explorationSigmaStart,
                        m_explorationSigmaEnd );
    }
    int effectiveController = inferenceOk ? 2 : 1;
    if( !consecutive && enabled )
    {
        // Preserve the existing source-gap pass-through behavior.
        memcpy( m_output.get(), observation, static_cast<size_t>( m_nModes ) * sizeof( float ) );
        effectiveController = 3;
    }
    else if( !finalizeControlledCommand( m_output.get(),
                                         observation,
                                         static_cast<size_t>( m_nModes ),
                                         static_cast<size_t>( m_nModesControlled ),
                                         m_maxCommandAbs ) )
    {
        m_collection.interrupt( m_activeCollectionGeneration );
        memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
        effectiveController = 0;
        inferenceOk         = false;
    }
    if( sendOutput() < 0 )
    {
        m_collection.interrupt( m_activeCollectionGeneration );
        m_replayCondition.notify_one();
        recordLatency();
        return -1;
    }

    const uint64_t emittedModel = inferenceOk ? expectedModel : 0;
    m_effectiveModelId.store( emittedModel, std::memory_order_release );
    m_effectiveController.store( effectiveController, std::memory_order_release );
    m_appliedControl.store( control, std::memory_order_release );
    if( resetGeneration != 0 )
        m_appliedReset.store( resetGeneration, std::memory_order_release );
    m_collection.emitted();
    if( consecutive )
    {
        m_commandHistory->add( m_output.get() );
        if( enabled || WarmupCollection::phase( m_collection.token() ) != CollectionPhase::idle )
        {
            EpisodeBuffer &episode = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
            episode.modelId        = emittedModel;
            if( !episode.add( observation, m_output.get() ) )
            {
                m_replayDrops.fetch_add( 1, std::memory_order_relaxed );
                m_collection.interrupt( m_activeCollectionGeneration );
            }
            else if( episode.count == episode.capacity )
                finishEpisode();
        }
    }
    if( WarmupCollection::phase( m_collection.token() ) == CollectionPhase::interrupted )
        m_replayCondition.notify_one();
    recordLatency();
    return 0;
}

#endif // po4ao_control_hpp
