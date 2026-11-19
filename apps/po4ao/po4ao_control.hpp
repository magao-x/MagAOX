/** \file po4ao_control.hpp
 * \brief Model commands, acknowledgements, frame processing and replay at the SDK boundary.
 *
 * Included by po4ao_impl.hpp inside MagAOX::app, after the application declaration.
 * \ingroup po4ao_files
 */
#ifndef po4ao_control_hpp
#define po4ao_control_hpp

inline void po4ao::makeIntegratorCommand( const float *observation )
{
    for( int mode = 0; mode < m_nModesControlled; ++mode )
    {
        m_output[mode] = m_integratorGain * observation[mode];
    }
}

inline int po4ao::invalidateReplay( IMAGE &stream, bool opened, IMAGE &metadataStream, bool metadataOpened )
{
    if( !opened || !metadataOpened )
    {
        return -1;
    }
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
    {
        return -1;
    }
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
            {
                break;
            }
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
        {
            m_collection.interrupt( episode.generation );
        }
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
        {
            m_collection.interrupt( episode.generation );
        }
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
    std::lock_guard<std::mutex> callbackLock( m_callbackMutex );
    const auto                  started       = std::chrono::steady_clock::now();
    auto                        recordLatency = [this, started]()
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - started );
        const uint32_t latency =
            static_cast<uint32_t>( std::min<int64_t>( elapsed.count(), std::numeric_limits<uint32_t>::max() ) );
        const auto     model   = m_effectiveModelId.load( std::memory_order_acquire );
        const uint32_t backend = m_effectiveBackend.load( std::memory_order_acquire );
        const auto     time =
            std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
                .count();
        m_runtimeSamples.record( latency, model, backend, static_cast<uint64_t>( time ) );
        if( latency > m_latencyDeadlineUs )
        {
            m_deadlineMisses.fetch_add( 1, std::memory_order_relaxed );
        }
    };
    if( !m_replayOpened )
    {
        if( started >= m_retryOutputsAt )
        {
            // This runs only while collection is unavailable. Keep large replay allocations off healthy frames.
            std::lock_guard<std::mutex> publicationLock( m_publicationMutex );
            openOutputStreams();
            m_retryOutputsAt = std::chrono::steady_clock::now() + std::chrono::seconds( 1 );
        }
        recordLatency();
        return 0;
    }
    if( !m_outputOpened )
    {
        recordLatency();
        return 0;
    }
    if( curr_src == nullptr || !m_inputState || !m_output )
    {
        return -1;
    }

    const uint64_t sourceCounter = shmimMonitorT::m_imageStream.md[0].cnt0;
    // Queued notifications can refer to the same latest image after a processing stall.
    const bool duplicate = sourceCounterIsDuplicate( sourceCounter, m_lastInputCounter, m_haveInputCounter );
    if( duplicate && !m_retryConsecutive.has_value() )
    {
        recordLatency();
        return 0;
    }

    activatePendingEngine();
    const uint64_t resetGeneration = m_modelResetRequest.load( std::memory_order_acquire );
    if( resetGeneration != 0 )
    {
        resetActiveModel( resetGeneration );
    }
    // Keep one request snapshot throughout this frame; later requests await the next output.
    const uint64_t control         = m_controlRequest.load( std::memory_order_acquire );
    const bool     enabled         = ( control & 1 ) != 0;
    const uint64_t modelGeneration = m_modelGeneration.load( std::memory_order_acquire );
    const auto    *observation     = static_cast<const float *>( curr_src );
    const bool     consecutive     = duplicate
                                         ? *m_retryConsecutive
                                         : sourceCounterIsConsecutive( sourceCounter, m_lastInputCounter, m_haveInputCounter );
    if( m_haveInputCounter && sourceCounter > m_lastInputCounter + 1 )
    {
        m_missedFrames.fetch_add( sourceCounter - m_lastInputCounter - 1, std::memory_order_relaxed );
    }
    m_lastInputCounter = sourceCounter;
    m_haveInputCounter = true;
    m_retryConsecutive.reset();

    const volatile auto &md = shmimMonitorT::m_imageStream.md[0];
    bool                 stable = !md.write;
    if( stable )
    {
        if( consecutive )
        {
            m_observationHistory->add( observation );
            observation = m_observationHistory->getItem( 0 );
        }
        else
        {
            memcpy( m_inputState.get(), observation, m_nModes * sizeof( float ) );
            observation = m_inputState.get();
        }
        std::atomic_thread_fence( std::memory_order_seq_cst );
        stable = !md.write && md.cnt0 == sourceCounter;
    }

    if( !consecutive || !stable )
    {
        m_observationHistory->reset();
        m_commandHistory->reset();
        EpisodeBuffer &episode = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
        m_collection.discard( episode.count );
        episode.reset();
        if( !stable )
        {
            m_retryConsecutive = consecutive;
            recordLatency();
            return 0;
        }
    }

    const bool policySelected = enabled && m_activeEngine != nullptr && m_activeEngine->generation == modelGeneration &&
                                m_useNeuralNetwork.load( std::memory_order_acquire );
    const bool inferenceReady = consecutive && policySelected &&
                                m_observationHistory->size() >= static_cast<size_t>( m_activeEngine->historyLength ) &&
                                m_commandHistory->size() + 1 >= static_cast<size_t>( m_activeEngine->historyLength );
    const uint64_t expectedModel       = policySelected ? m_activeEngine->modelId : 0;
    const uint64_t recollectRequest    = m_warmupRecollectRequest.load( std::memory_order_acquire );
    const uint64_t recollectGeneration = recollectRequest > m_activeCollectionGeneration ? recollectRequest : 0;
    if( recollectGeneration != 0 || ( m_initialWarmupPending && enabled ) )
    {
        beginWarmupCollection( recollectGeneration != 0 ? recollectGeneration : m_activeCollectionGeneration,
                               control,
                               expectedModel,
                               modelGeneration );
    }
    m_collection.validate( control, expectedModel, modelGeneration );

    bool inferenceOk = inferenceReady && runInference( observation );
    if( inferenceReady && !inferenceOk )
    {
        m_inferenceFailures.fetch_add( 1, std::memory_order_relaxed );
        m_collection.interrupt( m_activeCollectionGeneration );
    }
    if( !inferenceOk )
    {
        makeIntegratorCommand( observation );
    }

    // Refill prediction history without adding fallback samples or consuming exploration.
    bool       sampleAccepted = consecutive && ( !policySelected || inferenceOk );
    const bool exploring =
        sampleAccepted && WarmupCollection::phase( m_collection.token() ) == CollectionPhase::collecting;
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
    if( !finalizeControlledCommand( m_output.get(),
                                    observation,
                                    static_cast<size_t>( m_nModes ),
                                    static_cast<size_t>( m_nModesControlled ),
                                    m_maxCommandAbs ) )
    {
        m_collection.interrupt( m_activeCollectionGeneration );
        memset( m_output.get(), 0, static_cast<size_t>( m_nModes ) * sizeof( float ) );
        effectiveController = 0;
        inferenceOk         = false;
        sampleAccepted      = false;
    }
    if( !sampleAccepted )
    {
        m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
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
    m_effectiveBackend.store( emittedModel == 0 ? 0 : ( !m_activeEngine->linearWeights.empty() ? 1 : 2 ),
                              std::memory_order_release );
    m_effectiveController.store( effectiveController, std::memory_order_release );
    m_appliedControl.store( control, std::memory_order_release );
    if( resetGeneration != 0 )
    {
        m_appliedReset.store( resetGeneration, std::memory_order_release );
        // Only successful publication consumes this request; preserve a newer reset queued during output.
        uint64_t expected = resetGeneration;
        m_modelResetRequest.compare_exchange_strong( expected, 0, std::memory_order_acq_rel );
    }
    if( consecutive )
    {
        m_commandHistory->add( m_output.get() );
        if( sampleAccepted && ( enabled || WarmupCollection::phase( m_collection.token() ) != CollectionPhase::idle ) )
        {
            EpisodeBuffer &episode = m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )];
            episode.modelId        = emittedModel;
            if( !episode.add( observation, m_output.get() ) )
            {
                m_replayDrops.fetch_add( 1, std::memory_order_relaxed );
                m_collection.interrupt( m_activeCollectionGeneration );
            }
            else
            {
                m_collection.emitted();
                if( episode.count == episode.capacity )
                {
                    finishEpisode();
                }
            }
        }
    }
    if( WarmupCollection::phase( m_collection.token() ) == CollectionPhase::interrupted )
    {
        m_replayCondition.notify_one();
    }
    recordLatency();
    return 0;
}

inline void po4ao::updateControlStatus()
{
    const uint64_t control           = m_controlRequest.load( std::memory_order_acquire );
    const uint64_t applied           = m_appliedControl.load( std::memory_order_acquire );
    const uint64_t reset             = m_modelGeneration.load( std::memory_order_acquire );
    const uint64_t resetApplied      = m_appliedReset.load( std::memory_order_acquire );
    const int      reload            = m_reloadStatus.load( std::memory_order_acquire );
    const uint64_t collection        = m_collection.token();
    const auto     phase             = WarmupCollection::phase( collection );
    const uint64_t generation        = WarmupCollection::generation( collection );
    const uint64_t requested         = m_collectionGeneration.load( std::memory_order_acquire );
    const bool     pendingCollection = m_warmupRecollectRequest.load( std::memory_order_acquire ) > generation;

    auto modelText = []( uint64_t model )
    {
        std::ostringstream value;
        value << std::hex << std::setw( 16 ) << std::setfill( '0' ) << model;
        return value.str();
    };
    static const std::array<std::string, 4> reloadNames{ "idle", "loading", "ready", "failed" };
    static const std::array<std::string, 4> controllers{ "unknown", "integrator", "po4ao", "pass-through" };
    static const std::array<std::string, 5> phases{ "idle", "collecting", "publishing", "ready", "interrupted" };
    const uint64_t                          active = m_activeModelId.load( std::memory_order_acquire );
    std::string                             reloadError;
    { //mutex scope
        std::lock_guard<std::mutex> errorLock( m_reloadErrorMutex );
        reloadError = m_reloadError;
    }
    // Keep lifetime, request generations and heartbeat in one observed property update.
    updateIfChanged<std::string>(
        m_indiP_modelStatus,
        { "active_id",
          "active_backend",
          "effective_controller",
          "effective_id",
          "reload",
          "reload_error",
          "instance",
          "control_target",
          "sequence",
          "control_request",
          "control_applied",
          "reset_request",
          "reset_applied" },
        { modelText( active ),
          active == 0 ? "fallback" : ( m_activeCpuInference.load( std::memory_order_acquire ) ? "cpu" : "tensorrt" ),
          controllers[static_cast<size_t>( m_effectiveController.load( std::memory_order_acquire ) )],
          modelText( m_effectiveModelId.load( std::memory_order_acquire ) ),
          reloadNames[static_cast<size_t>( reload )],
          reloadError,
          m_runtimeInstance,
          ( control & 1 ) ? "on" : "off",
          std::to_string( ++m_statusSequence ),
          std::to_string( control >> 1 ),
          std::to_string( applied >> 1 ),
          std::to_string( reset ),
          std::to_string( resetApplied ) } );
    updateIfChanged( m_indiP_collectionStatus,
                     "state",
                     pendingCollection ? std::string( "pending" ) : phases[static_cast<size_t>( phase )] );
    updateIfChanged( m_indiP_collectionStatus, "requested_generation", std::to_string( requested ) );
    updateIfChanged( m_indiP_collectionStatus, "generation", std::to_string( generation ) );
    updateIfChanged( m_indiP_collectionStatus, "frames", std::to_string( m_collection.frames() ) );
    updateIfChanged( m_indiP_collectionStatus,
                     "target_frames",
                     std::to_string( static_cast<uint64_t>( m_warmupEpisodes ) * m_iterationsPerEpisode ) );

    // Publish observed state before terminal acknowledgements.
    updateSwitchIfChanged( m_indiP_controlToggle,
                           "toggle",
                           ( applied & 1 ) ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           control == applied ? INDI_OK : INDI_BUSY );
    updateSwitchIfChanged(
        m_indiP_resetModel, "toggle", pcf::IndiElement::Off, reset == resetApplied ? INDI_OK : INDI_BUSY );
    updateSwitchIfChanged( m_indiP_reloadToggle,
                           "toggle",
                           pcf::IndiElement::Off,
                           reload == 3 ? INDI_ALERT : ( reload == 0 ? INDI_OK : INDI_BUSY ) );
    updateSwitchIfChanged( m_indiP_recollectWarmup,
                           "toggle",
                           pcf::IndiElement::Off,
                           pendingCollection || phase == CollectionPhase::collecting ||
                                   phase == CollectionPhase::publishing
                               ? INDI_BUSY
                               : ( phase == CollectionPhase::interrupted ? INDI_ALERT : INDI_OK ) );
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_reloadToggle )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( ipRecv.getName() != m_indiP_reloadToggle.getName() || !ipRecv.find( "toggle" ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid reload property" } );
    }
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }
    int expected = m_reloadStatus.load( std::memory_order_acquire );
    if( ( expected != 0 && expected != 3 ) ||
        !m_reloadStatus.compare_exchange_strong( expected, 1, std::memory_order_acq_rel ) )
    {
        return 0;
    }
    { //mutex scope
        std::lock_guard<std::mutex> errorLock( m_reloadErrorMutex );
        m_reloadError.clear();
    }
    m_reloadGeneration.store( m_modelGeneration.load( std::memory_order_acquire ), std::memory_order_release );
    m_reloadRequested.store( true, std::memory_order_release );
    m_loaderCondition.notify_one();
    updateSwitchIfChanged( m_indiP_reloadToggle, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_resetModel )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( ipRecv.getName() != m_indiP_resetModel.getName() || !ipRecv.find( "toggle" ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid model-reset property" } );
    }
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    const uint64_t generation = m_modelGeneration.fetch_add( 1, std::memory_order_acq_rel ) + 1;
    m_modelResetRequest.store( generation, std::memory_order_release );
    updateSwitchIfChanged( m_indiP_resetModel, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    log<text_log>( "PO4AO active-model reset requested", logPrio::LOG_INFO );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_controlToggle )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( ipRecv.getName() != m_indiP_controlToggle.getName() )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid control property" } );
    }
    if( !ipRecv.find( "toggle" ) || ( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On &&
                                      ipRecv["toggle"].getSwitchState() != pcf::IndiElement::Off ) )
    {
        return -1;
    }
    const uint64_t enabled  = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On ? 1 : 0;
    uint64_t       previous = m_controlRequest.load( std::memory_order_acquire );
    if( ( previous & 1 ) == enabled )
    {
        // A delayed client may retry an already-applied target; acknowledge acceptance before the periodic result.
        updateSwitchIfChanged( m_indiP_controlToggle,
                               "toggle",
                               ( m_appliedControl.load( std::memory_order_acquire ) & 1 ) ? pcf::IndiElement::On
                                                                                          : pcf::IndiElement::Off,
                               INDI_BUSY );
        return 0;
    }
    while( !m_controlRequest.compare_exchange_weak(
        previous, ( ( previous & ~uint64_t{ 1 } ) + 2 ) | enabled, std::memory_order_acq_rel ) )
    {
    }
    updateSwitchIfChanged( m_indiP_controlToggle,
                           "toggle",
                           ( m_appliedControl.load( std::memory_order_acquire ) & 1 ) ? pcf::IndiElement::On
                                                                                      : pcf::IndiElement::Off,
                           INDI_BUSY );
    return 0;
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_recollectWarmup )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( ipRecv.getName() != m_indiP_recollectWarmup.getName() || !ipRecv.find( "toggle" ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid warm-up recollection property" } );
    }
    if( ipRecv["toggle"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    { //mutex scope
        std::lock_guard<std::mutex> publicationLock( m_publicationMutex );
        const uint64_t              generation = m_collectionGeneration.fetch_add( 1, std::memory_order_acq_rel ) + 1;
        invalidateReplay( m_replayStream, m_replayOpened, m_replayMetadataStream, m_replayMetadataOpened );
        invalidateReplay( m_warmupStream, m_warmupOpened, m_warmupMetadataStream, m_warmupMetadataOpened );
        m_warmupRecollectRequest.store( generation, std::memory_order_release );
    }
    updateSwitchIfChanged( m_indiP_recollectWarmup, "toggle", pcf::IndiElement::Off, INDI_BUSY );
    log<text_log>( "PO4AO warm-up recollection requested", logPrio::LOG_INFO );
    return 0;
}

inline int po4ao::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    std::unique_lock<std::mutex> lock( m_indiMutex );
    SHMIMMONITOR_UPDATE_INDI;
    updateControlStatus();
    updateRuntimeStatistics();
    TELEMETER_APP_LOGIC;
    return 0;
}

#endif // po4ao_control_hpp
