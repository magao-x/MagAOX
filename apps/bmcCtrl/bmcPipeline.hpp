/** \file bmcPipeline.hpp
 * \brief Bounded preparation, serialized SDK and ordered post-processing implementation for bmcCtrl.
 */
#ifndef bmcPipeline_hpp
#define bmcPipeline_hpp

namespace MagAOX
{
namespace app
{

inline bool bmcCtrl::pipelinedCommand() const
{
    return m_pipelineEnabled;
}

inline void bmcCtrl::shmimMonitorTiming( const dev::shmimT &tag, dev::shmimMonitorEvent event )
{
    if( m_pipelineEnabled )
    {
        if( event == dev::shmimMonitorEvent::connected || event == dev::shmimMonitorEvent::disconnected )
        {
            std::lock_guard<std::mutex> sdk( m_sdkMutex );
            m_pipelineHaveLatest = false;

            for( auto &job : m_pipelineJobs )
            {
                supersedePipelineJob( job );
            }
        }

        dmT::pipelineShmimMonitorTiming( tag, event );
    }
    else
    {
        dmT::shmimMonitorTiming( tag, event );
    }
}

inline int bmcCtrl::allocate( const dev::shmimT &tag )
{
    if( !m_pipelineEnabled )
    {
        return dmT::allocate( tag );
    }

    std::lock_guard<std::mutex> control( m_controlMutex );
    const bool                  reopen = closeAdmission();
    int                         result = dmT::allocate( tag );

    if( result >= 0 )
    {
        try
        {
            allocatePipelineJobs();
        }
        catch( const std::exception &error )
        {
            result = log<software_error, -1>( { std::string( "allocating pipeline storage: " ) + error.what() } );
        }
    }

    openAdmission( reopen && result >= 0 );
    return result;
}

inline void bmcCtrl::allocatePipelineJobs()
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    for( auto &job : m_pipelineJobs )
    {
        allocatePipelineCommand( job.command );
        job.values.assign( m_nbAct, 0 );
        job.identity = {};
        job.result   = {};
        job.order    = 0;
        job.busy = job.staged = job.queued = job.running = job.sdkDone = job.submitted = job.posting = job.cancelled =
            job.timedOut = job.superseded = false;
    }
}

inline int bmcCtrl::processImage( void *source, const dev::shmimT &tag )
{
    return m_pipelineEnabled ? processPipeline( source ) : dmT::processImage( source, tag );
}

inline size_t bmcCtrl::claimPipelineJob()
{
    std::lock_guard<std::mutex> sdk( m_sdkMutex );
    std::lock_guard<std::mutex> command( m_commandMutex );
    const auto                 *metadata = m_imageStream.md;
    const auto                  observed = dev::dmCompletion::captureRecord( &m_imageStream, m_completionReceipt );

    if( m_completionReceipt )
    {
        m_pipelineHaveLatest = observed.state == MagAOX::dmCompletion::State::busy;
        m_pipelineLatest     = observed;
    }

    if( m_admissionClosed || m_sdkFault || !m_dmopen || !m_completionReceipt || !m_pipelineConfigured || !metadata )
    {
        ++m_rejectedInputs;
        return pipelineCapacity;
    }

    const auto inode   = __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE );
    const auto counter = __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE );

    if( !inode || !counter || __atomic_load_n( &metadata->write, __ATOMIC_ACQUIRE ) != 0 ||
        inode != __atomic_load_n( &metadata->inode, __ATOMIC_ACQUIRE ) ||
        counter != __atomic_load_n( &metadata->cnt0, __ATOMIC_ACQUIRE ) ||
        ( m_haveInput && inode == m_inputInode && counter <= m_inputCounter ) )
    {
        if( m_haveInput && inode == m_inputInode && counter < m_inputCounter )
        {
            m_haveInput = false;
        }

        ++m_rejectedInputs;
        return pipelineCapacity;
    }

    size_t slot = pipelineCapacity;

    for( size_t index = 0; index < pipelineCapacity; ++index )
    {
        if( !m_pipelineJobs[index].busy )
        {
            slot = index;
            break;
        }
    }

    if( slot == pipelineCapacity || m_pipelineOrder == UINT64_MAX )
    {
        ++m_pipelineOverloaded;
        ++m_rejectedInputs;
        return pipelineCapacity;
    }

    if( m_haveInput && inode == m_inputInode && counter > m_inputCounter + 1 )
    {
        m_coalescedInputs += counter - m_inputCounter - 1;
    }

    m_inputInode   = inode;
    m_inputCounter = counter;
    m_haveInput    = true;
    auto &job      = m_pipelineJobs[slot];
    job.order      = ++m_pipelineOrder;
    job.busy       = true;
    job.staged = job.queued = job.running = job.sdkDone = job.submitted = job.posting = job.cancelled = job.timedOut =
        job.superseded                                                                                = false;
    job.result                                                                                        = {};
    ++m_pipelineInFlight;
    ++m_pipelineAdmitted;
    m_commandActive = true;
    return slot;
}

inline int bmcCtrl::processPipeline( void *source )
{
    const auto slot = source ? claimPipelineJob() : pipelineCapacity;

    if( slot == pipelineCapacity )
    {
        discardPipelineCommand();
        return 0;
    }

    auto &job    = m_pipelineJobs[slot];
    int   result = -1;

    try
    {
        result       = capturePipelineCommand( job.command, source );
        job.identity = job.command.completion;

        if( result >= 0 )
        {
            pipelineMark( job.command, MagAOX::dmTiming::Marker::actuatorStart );
            const double scale   = m_volume_factor / m_act_gain;
            const float  inverse = 1.0 / scale;
            job.command.shape.setZero();
            std::fill( job.command.saturation.begin(), job.command.saturation.end(), uint8_t( 0 ) );
            job.command.saturatedCount = 0;

            if( job.values.size() != m_nbAct || job.command.saturation.size() != size_t( m_dmWidth * m_dmHeight ) )
            {
                throw std::runtime_error( "pipeline storage dimensions changed without a drained allocation" );
            }

            for( uint32_t index = 0; index < m_nbAct; ++index )
            {
                const int address = m_actuator_mapping[index];

                if( address < 0 )
                {
                    job.values[index] = 0;
                    continue;
                }

                const float  input                = static_cast<float *>( source )[address];
                const double value                = std::clamp( input * scale, 0.0, 1.0 );
                job.values[index]                 = std::sqrt( value );
                job.command.shape.data()[address] = input * scale > 1 ? inverse : ( input * scale < 0 ? 0 : input );
                const bool saturated              = value >= 1 || value <= 0;
                job.command.saturation[address]   = saturated;
                job.command.saturatedCount += saturated;
            }

            m_outputShape = job.command.shape;
            pipelineMark( job.command, MagAOX::dmTiming::Marker::actuatorEnd );
            job.command.dispatchNs = dev::dmTiming::monotonicNowNs();

            { //mutex scope
                std::lock_guard<std::mutex> sdk( m_sdkMutex );
                std::lock_guard<std::mutex> command( m_commandMutex );

                if( m_sdkFault || job.cancelled )
                {
                    job.cancelled = job.sdkDone = true;
                }
                else
                {
                    job.queued = true;
                }
            }

            m_sdkWake.notify_one();
#ifdef BMCCTRL_TEST_PIPELINE_PREP
            BMCCTRL_TEST_PIPELINE_PREP( job.order );
#endif
            result = stagePipelineCommand( job.command );
        }
    }
    catch( const std::exception &error )
    {
        log<software_error>( { std::string( "preparing pipeline command: " ) + error.what() } );
        result = -1;
    }
    catch( ... )
    {
        log<software_error>( { "unknown exception preparing pipeline command" } );
        result = -1;
    }

    job.command.preparedStatus = result;
    releasePipelineCommand( job.command );
    result = job.command.preparedStatus;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_sdkMutex );
        job.staged = true;

        if( result < 0 )
        {
            faultPipeline();
        }

        if( !job.queued && !job.running && !job.sdkDone )
        {
            job.cancelled = job.sdkDone = true;
            ++m_pipelineCancelled;
        }
    }

    m_pipelineWake.notify_one();
    return 0;
}

inline size_t bmcCtrl::nextPipelineSdk() const
{
    size_t selected = pipelineCapacity;

    for( size_t index = 0; index < pipelineCapacity; ++index )
    {
        const auto &job = m_pipelineJobs[index];

        if( job.busy && job.queued && !job.cancelled &&
            ( selected == pipelineCapacity || job.order < m_pipelineJobs[selected].order ) )
        {
            selected = index;
        }
    }

    return selected;
}

inline size_t bmcCtrl::nextPipelinePost() const
{
    size_t selected = pipelineCapacity;

    for( size_t index = 0; index < pipelineCapacity; ++index )
    {
        if( m_pipelineJobs[index].busy &&
            ( selected == pipelineCapacity || m_pipelineJobs[index].order < m_pipelineJobs[selected].order ) )
        {
            selected = index;
        }
    }

    return selected;
}

inline void bmcCtrl::supersedePipelineJob( PipelineJob &job )
{
    if( job.busy && !job.running && !job.submitted && !job.cancelled )
    {
        job.cancelled = job.sdkDone = job.superseded = true;
        job.queued                                   = false;
        ++m_pipelineCancelled;
        ++m_pipelineSuperseded;
        m_pipelineWake.notify_one();
    }
}

inline void bmcCtrl::faultPipeline()
{
    { //mutex scope
        std::lock_guard<std::mutex> command( m_commandMutex );
        m_sdkFault        = true;
        m_admissionClosed = true;
    }

    for( auto &job : m_pipelineJobs )
    {
        if( job.busy && !job.running && !job.submitted && !job.cancelled )
        {
            job.cancelled = job.sdkDone = true;
            job.queued                  = false;
            ++m_pipelineCancelled;
        }
    }

    m_pipelineWake.notify_one();
    m_commandRetired.notify_all();
}

inline int bmcCtrl::startPipelineWorker()
{
    if( m_pipelineThread.joinable() )
    {
        return 0;
    }

    if( m_pipelineResult.open( m_pipelineResultStream, MagAOX::dmCompletion::Kind::sdk ) < 0 )
    {
        return log<software_error, -1>( { errno, "opening pipeline SDK-result stream" } );
    }

    m_pipelineStop       = false;
    m_pipelineConfigured = false;

    if( threadStart( m_pipelineThread,
                     m_pipelineThreadInit,
                     m_pipelineThreadID,
                     m_pipelineThreadProp,
                     m_pipelineThreadPrio,
                     m_pipelineCpuset,
                     "pipeline",
                     this,
                     pipelineThreadStart ) < 0 )
    {
        stopPipelineWorker();
        return -1;
    }

    if( verifyWorkerThread( m_pipelineThread,
                            m_pipelineThreadPrio,
                            m_pipelineCpuset,
                            m_pipelinePolicy,
                            m_pipelineActualPrio,
                            m_pipelineAllowedCpus ) < 0 )
    {
        stopPipelineWorker();
        return -1;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_sdkMutex );
        m_pipelineConfigured = true;
    }

    m_pipelineWake.notify_one();
    return 0;
}

inline void bmcCtrl::stopPipelineWorker() noexcept
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_sdkMutex );
        m_pipelineStop       = true;
        m_pipelineConfigured = false;
    }

    m_pipelineWake.notify_one();

    if( m_pipelineThread.joinable() )
    {
        m_pipelineThread.join();
    }
}

inline void bmcCtrl::pipelineThreadStart( bmcCtrl *app )
{
    app->pipelineThreadExec();
}

inline void bmcCtrl::pipelineThreadExec() noexcept
{
    m_pipelineThreadID = syscall( SYS_gettid );

    for( ;; )
    {
        size_t slot = pipelineCapacity;

        { //mutex scope
            std::unique_lock<std::mutex> lock( m_sdkMutex );
            m_pipelineWake.wait( lock,
                                 [this]()
                                 {
                                     const auto next = nextPipelinePost();
                                     return m_pipelineStop ||
                                            ( m_pipelineConfigured && next < pipelineCapacity &&
                                              m_pipelineJobs[next].staged && m_pipelineJobs[next].sdkDone );
                                 } );

            if( m_pipelineStop )
            {
                return;
            }

            slot                         = nextPipelinePost();
            m_pipelineJobs[slot].posting = true;
        }

        auto &job = m_pipelineJobs[slot];

#ifdef BMCCTRL_TEST_PIPELINE_POST
        BMCCTRL_TEST_PIPELINE_POST( job.order );
#endif

        if( job.cancelled || job.result.timedOut )
        {
            job.command.preparedStatus = -1;
        }

        const bool prepared   = job.command.preparedStatus >= 0;
        const bool successful = job.submitted && job.result.result == NO_ERR && !job.result.exception;

        if( successful && prepared )
        {
            m_nsat.fetch_add( job.command.saturatedCount );
        }

        const auto outcome = !job.submitted ? MagAOX::dmTiming::Outcome::noSend
                                            : ( job.result.result == NO_ERR ? MagAOX::dmTiming::Outcome::success
                                                                            : MagAOX::dmTiming::Outcome::failed );
        int        result  = -1;

        try
        {
            result =
                finishPipelineCommand( job.command,
                                       outcome,
                                       !job.submitted || job.result.exception ? MagAOX::dmCompletion::unavailableResult
                                                                              : int64_t( job.result.result ),
                                       job.result.startNs,
                                       job.result.endNs,
                                       successful );
        }
        catch( ... )
        {
            log<software_critical>( { "exception finishing pipeline command" } );
        }

        const bool postFailed = result < 0 && successful && !job.result.timedOut && prepared && !job.superseded;

        if( postFailed )
        {
            log<text_log>( "DM pipeline post processing failed after successful SDK submission", logPrio::LOG_ERROR );
        }

        { //mutex scope
            std::lock_guard<std::mutex> sdk( m_sdkMutex );

            if( result < 0 && !job.superseded )
            {
                if( postFailed )
                {
                    ++m_pipelinePostFailures;
                }

                faultPipeline();
            }

            ++m_pipelineRetired;
            job.busy = job.posting = false;
            --m_pipelineInFlight;

            { //mutex scope
                std::lock_guard<std::mutex> command( m_commandMutex );
                m_commandActive = m_pipelineInFlight != 0;
            }
        }

        m_commandRetired.notify_all();
        m_sdkDone.notify_all();
        m_pipelineWake.notify_one();
    }
}

inline void bmcCtrl::sdkThreadExec() noexcept
{
    m_sdkThreadID = syscall( SYS_gettid );

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_sdkMutex );
        m_sdkWake.wait( lock,
                        [this]()
                        {
                            return m_sdkConfigured || m_sdkStop;
                        } );
    }

    for( ;; )
    {
        SdkRequest                   request;
        size_t                       slot = pipelineCapacity;
        MagAOX::dmCompletion::Record identity;

        { //mutex scope
            std::unique_lock<std::mutex> lock( m_sdkMutex );
            m_sdkWake.wait( lock,
                            [this]()
                            {
                                return m_sdkStop || ( m_sdkPending && !m_sdkReady ) ||
                                       nextPipelineSdk() < pipelineCapacity;
                            } );

            if( m_sdkStop )
            {
                return;
            }

            if( m_sdkPending && !m_sdkReady )
            {
                request = m_sdkRequest;
            }
            else
            {
                if( m_sdkSequence == UINT64_MAX )
                {
                    faultPipeline();
                    continue;
                }

                slot      = nextPipelineSdk();
                auto &job = m_pipelineJobs[slot];

                if( !m_pipelineHaveLatest || job.identity.inputInode != m_pipelineLatest.inputInode ||
                    job.identity.inputCounter != m_pipelineLatest.inputCounter ||
                    job.identity.publicationSec != m_pipelineLatest.publicationSec ||
                    job.identity.publicationNsec != m_pipelineLatest.publicationNsec )
                {
                    supersedePipelineJob( job );
                    continue;
                }

                job.queued            = false;
                job.running           = true;
                m_pipelineSdkActive   = slot;
                request.operation     = SdkOperation::set;
                request.values        = job.values.data();
                request.sequence      = ++m_sdkSequence;
                request.generation    = m_sdkGeneration;
                request.normalCommand = true;
                request.dispatchNs    = job.command.dispatchNs;
                identity              = job.identity;
            }
        }

        auto result = performSdk( request );

        { //mutex scope
            std::lock_guard<std::mutex> sdk( m_sdkMutex );

            if( slot == pipelineCapacity )
            {
                m_sdkResult = result;
                m_sdkReady  = true;
            }
            else
            {
                auto      &job     = m_pipelineJobs[slot];
                const bool expired = request.dispatchNs > 0 && result.endNs >= request.dispatchNs &&
                                     ( result.endNs - request.dispatchNs ) * 1e-9 >= m_sdkTimeoutSec;
                result.newTimeout   = expired && !job.timedOut;
                result.timedOut     = expired || job.timedOut;
                job.result          = result;
                job.submitted       = true;
                m_pipelineSdkActive = pipelineCapacity;

                { //mutex scope
                    std::lock_guard<std::mutex> command( m_commandMutex );
                    ++m_completedCommands;
                    m_sdkFailures += result.result != NO_ERR;
                    m_sdkTimeouts += result.newTimeout;
                    m_sdkActiveAgeSec = 0;
                }

                if( !job.staged || job.command.preparedStatus < 0 )
                {
                    identity.state = MagAOX::dmCompletion::State::unqualified;
                }

                if( result.result != NO_ERR || result.timedOut )
                {
                    faultPipeline();
                }
            }
        }

        if( slot < pipelineCapacity )
        {
            if( result.result != NO_ERR )
            {
                log<text_log>( result.exception
                                   ? "DM SDK threw during pipeline submission"
                                   : "DM pipeline SDK submission failed with code " + std::to_string( result.result ),
                               logPrio::LOG_ERROR );
            }

            if( result.newTimeout )
            {
                log<text_log>( "DM pipeline SDK command exceeded the diagnostic deadline; actual result was drained",
                               logPrio::LOG_ERROR );
            }

            identity.startedNs = result.startNs;
            identity.backendResult =
                result.exception ? MagAOX::dmCompletion::unavailableResult : int64_t( result.result );
            m_pipelineResult.publishCompleted( identity,
                                               result.result == NO_ERR ? 0 : -1,
                                               result.result == NO_ERR ? MagAOX::dmTiming::Outcome::success
                                                                       : MagAOX::dmTiming::Outcome::failed,
                                               result.endNs );
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_sdkMutex );
                m_pipelineJobs[slot].sdkDone = true;
                m_pipelineJobs[slot].running = false;
                m_pipelineSdkActive          = pipelineCapacity;
            }

            m_pipelineWake.notify_one();
        }

        m_sdkDone.notify_all();
    }
}

} // namespace app
} // namespace MagAOX

#endif // bmcPipeline_hpp
