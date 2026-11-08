/** \file po4ao_run.hpp
 * \brief Serialized whole-app routing, included inside MagAOX::app.
 */
#ifndef po4ao_run_hpp
#define po4ao_run_hpp

inline void po4ao::runPhase( const std::string &phase )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_runPhase = phase;
}

inline void po4ao::waitRun( const std::function<bool()> &ready, Po4aoRouteBackend::Deadline deadline )
{
    for( ;; )
    {
        if( m_runCancel.load() || std::chrono::steady_clock::now() >= deadline )
            throw std::runtime_error( "PO4AO routing cancelled or timed out" );
        if( ready() )
            return;
        std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
    }
}

inline void po4ao::requestWriterHold( const std::string &action )
{
    pcf::IndiProperty request( pcf::IndiProperty::Text );
    request.setDevice( m_nnDevice );
    request.setName( "routing_hold" );
    request.add( pcf::IndiElement( "owner", m_runOwner ) );
    request.add( pcf::IndiElement( "action", action ) );
    if( sendNewProperty( request ) < 0 )
        throw std::runtime_error( "NN routing reservation delivery failed" );
}

inline bool po4ao::writerHeld( bool released )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( std::chrono::duration<double>( std::chrono::steady_clock::now() - m_nnObserved ).count() > 5 )
        return false;
    if( !m_reservedInstance.empty() && m_nnInstance != m_reservedInstance )
        throw std::runtime_error( "NN restarted during the routing reservation" );
    if( m_nnRoutingPhase == "error" )
        throw std::runtime_error( "NN routing reservation requires recovery" );
    return released ? reservationReleased() : ( m_nnRoutingOwner == m_runOwner && m_nnRoutingPhase == "held" );
}

inline bool po4ao::reservationReleased() const
{
    return !m_reservedInstance.empty() && m_nnInstance == m_reservedInstance &&
           std::chrono::duration<double>( std::chrono::steady_clock::now() - m_nnObserved ).count() <= 5 &&
           m_nnRoutingOwner.empty() && m_nnRoutingPhase == "released" && m_nnReleasedOwner == m_runOwner;
}

inline int po4ao::rejectRun( const std::string &reason )
{
    m_runRequestError = reason;
    // Every rejected request needs an acknowledgement, including repeated identical requests.
    m_indiP_run["toggle"].setSwitchState( m_runRunning ? pcf::IndiElement::On : pcf::IndiElement::Off );
    m_indiP_run.setState( INDI_ALERT );
    updateIfChanged( m_indiP_runStatus, "error", m_runError.empty() ? reason : m_runError + "; " + reason, INDI_ALERT );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( m_indiP_run );
    return -1;
}

inline void po4ao::runWorker( bool start )
{
    Po4aoRouteBackend backend;
    const auto        deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 90 );
    backend.capture            = [this]( auto )
    {
        if( m_outputChannel != "po4ao_output" )
            throw std::runtime_error( "Whole-app routing requires output channel po4ao_output" );
        auto                        snapshot = m_routeIO.capture();
        std::lock_guard<std::mutex> lock( m_indiMutex );
        m_runRoute = snapshot.route;
        return snapshot;
    };
    backend.freshOutput = [this]( auto until )
    {
        const auto previous = m_outputSequence.load();
        waitRun( [this, previous] { return m_outputSequence.load() > previous + 1; }, until );
    };
    backend.reserveWriter = [this]( const auto &, auto until )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_indiMutex );
            if( m_nnInstance.empty() ||
                std::chrono::duration<double>( std::chrono::steady_clock::now() - m_nnObserved ).count() > 5 )
                throw std::runtime_error( "NN routing status is unavailable or stale" );
            m_reservedInstance = m_nnInstance;
        }
        requestWriterHold( "hold" );
        waitRun( [this] { return writerHeld( false ); }, until );
        std::lock_guard<std::mutex> lock( m_indiMutex );
        // This is the writer captured atomically when NN accepted the reservation.
        m_runSnapshot.writer = m_nnRoutingSource;
    };
    backend.switchInput = [this]( const auto &target, auto until )
    {
        m_routeIO.switchInput( target, until );
        std::lock_guard<std::mutex> lock( m_indiMutex );
        m_runRoute = target;
    };
    backend.applyDefault  = [this]( auto until ) { m_routeIO.applyDefault( until ); };
    backend.restoreWriter = [this]( const auto &, auto until )
    {
        requestWriterHold( "release" );
        waitRun( [this] { return writerHeld( true ); }, until );
    };
    backend.verifyConsumption = [this]( const auto &target, auto until )
    {
        if( target == "po4ao_output" )
        {
            const auto previous = m_outputSequence.load();
            waitRun( [this, previous] { return m_outputSequence.load() > previous + 1; }, until );
        }
        m_routeIO.verifyConsumption( target, until );
    };
    backend.park = [this]( auto until )
    {
        m_runAdmission.store( false );
        // Drain app work; upstream READY closes its mapping asynchronously.
        waitRun( [this] { return m_callbackMutex.try_lock(); }, until );
        std::unique_lock<std::mutex> callbackLock( m_callbackMutex, std::adopt_lock );
        m_collection.interrupt( m_activeCollectionGeneration );
        m_replayCondition.notify_one();
    };
    try
    {
        if( m_runCheckOnly )
        {
            m_runSnapshot.route      = m_routeIO.route();
            const std::string target = start ? "po4ao_output" : "aol1_modevalWFS";
            if( m_runSnapshot.route != target )
                throw std::runtime_error( "Route changed since the previous request" );
            m_routeIO.verifyConsumption( target, deadline );
            runPhase( start ? "running" : "stopped" );
        }
        else
        {
            m_runAdmission.store( true );
            routePo4ao(
                start,
                backend,
                m_runSnapshot,
                [this, deadline]( const auto &phase )
                {
                    if( m_runCancel.load() || std::chrono::steady_clock::now() >= deadline )
                        throw std::runtime_error( "Routing cancelled or timed out before " + phase );
                    runPhase( phase );
                },
                deadline );
        }
        std::lock_guard<std::mutex> lock( m_indiMutex );
        m_runVerified = true;
        m_runRunning  = start;
    }
    catch( const std::exception &error )
    {
        // If routing may still use PO4AO, no cleanup action can park its monitor.
        std::string actual = "unknown";
        try
        {
            actual = m_routeIO.route();
        }
        catch( const std::exception & )
        {
        }
        std::lock_guard<std::mutex> lock( m_indiMutex );
        m_runError    = m_runPhase + ": " + error.what();
        m_runPhase    = "failed";
        m_runRoute    = actual;
        m_runVerified = false;
    }
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_runBusy = false;
}

inline void po4ao::updateRunControl()
{
    bool launch = false, target = false;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );
        if( m_runPending )
        {
            m_runPending = false;
            launch       = true;
            target       = m_runTarget;
        }
    }
    if( launch )
    {
        if( m_runThread.joinable() )
            m_runThread.join();
        try
        {
            m_runThread = std::thread( [this, target] { runWorker( target ); } );
        }
        catch( const std::exception &error )
        {
            std::lock_guard<std::mutex> lock( m_indiMutex );
            m_runBusy  = false;
            m_runError = error.what();
            m_runPhase = "failed";
        }
    }
    std::lock_guard<std::mutex> lock( m_indiMutex );
    state( m_runAdmission.load() ? stateCodes::OPERATING : stateCodes::READY );
    const auto                  error  = m_runRequestError.empty() ? m_runError
                                         : m_runError.empty()      ? m_runRequestError
                                                                   : m_runError + "; " + m_runRequestError;
    const auto                  status = m_runBusy ? INDI_BUSY : !error.empty() ? INDI_ALERT : INDI_OK;
    updateSwitchIfChanged( m_indiP_run, "toggle", m_runRunning ? pcf::IndiElement::On : pcf::IndiElement::Off, status );
    updateIfChanged( m_indiP_runStatus, "phase", m_runPhase, status );
    updateIfChanged( m_indiP_runStatus, "route", m_runRoute, status );
    updateIfChanged( m_indiP_runStatus, "error", error, status );
    updateIfChanged( m_indiP_runStatus, "owner", m_runOwner, status );
    updateIfChanged( m_indiP_runStatus, "verified", std::string( m_runVerified ? "yes" : "no" ), status );
    // Unchanged routing readbacks also need current NN process status during a long command.
    if( m_indiDriver && ( m_runBusy || !m_reservedInstance.empty() ) )
        m_indiDriver->sendGetProperties( m_indiP_nnRouting );
}

INDI_NEWCALLBACK_DEFN( po4ao, m_indiP_run )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_run, property );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !property.find( "toggle" ) )
        return rejectRun( "Missing Start/Stop target" );
    const auto value = property["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return rejectRun( "Invalid Start/Stop target" );
    const bool start = value == pcf::IndiElement::On;
    if( m_runBusy || m_runCancel.load() || m_modelResetRequest.load() != 0 || m_reloadStatus.load() == 1 ||
        m_reloadStatus.load() == 2 )
        return rejectRun( "Start/Stop conflicts with an active operation or shutdown" );
    if( !m_reservedInstance.empty() && !reservationReleased() )
        return rejectRun( "Previous routing reservation requires verified operator recovery" );
    if( !m_nnRoutingOwner.empty() )
        return rejectRun( "NN has an active routing reservation" );
    m_runCheckOnly = m_runVerified && m_runRunning == start;
    m_runTarget    = start;
    m_runError.clear();
    m_runRequestError.clear();
    m_runPhase    = "queued";
    m_runVerified = false;
    m_reservedInstance.clear();
    m_runOwner = m_runtimeInstance + "-" + std::to_string( ++m_runSequence );
    m_runBusy = m_runPending = true;
    // Publish admission before the worker can finish; clients expect Busy followed by Ok/Alert.
    m_indiP_run.setState( INDI_BUSY );
    if( m_indiDriver )
        m_indiDriver->sendSetProperty( m_indiP_run );
    return 0;
}

INDI_SETCALLBACK_DEFN( po4ao, m_indiP_nnRouting )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nnRouting, property );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( property.find( "owner" ) )
        m_nnRoutingOwner = property["owner"].get<std::string>();
    if( property.find( "released_owner" ) )
        m_nnReleasedOwner = property["released_owner"].get<std::string>();
    if( property.find( "source" ) )
        m_nnRoutingSource = property["source"].get<std::string>();
    if( property.find( "phase" ) )
        m_nnRoutingPhase = property["phase"].get<std::string>();
    if( property.find( "instance" ) )
        m_nnInstance = property["instance"].get<std::string>();
    if( property.find( "sequence" ) && property["sequence"].get<std::string>() != m_nnSequence )
    {
        m_nnSequence = property["sequence"].get<std::string>();
        m_nnObserved = std::chrono::steady_clock::now();
    }
    return 0;
}
#endif
