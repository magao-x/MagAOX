/** \file reconstructorControls.hpp
 * \brief Out-of-class NN/CACAO control definitions; included inside MagAOX::app.
 */
#ifndef nnReconstructor_reconstructorControls_hpp
#define nnReconstructor_reconstructorControls_hpp

inline int nnReconstructor::setupReconstructorControls()
{
    createROIndiText( m_indiP_nnActivate, "nn_activate", "model" );
    m_indiP_nnActivate.setPerm( pcf::IndiProperty::ReadWrite );
    createROIndiText( m_indiP_reconstructor, "reconstructor", "target" );
    m_indiP_reconstructor.add( pcf::IndiElement( "expected" ) );
    m_indiP_reconstructor.setPerm( pcf::IndiProperty::ReadWrite );
    createStandardIndiToggleSw( m_indiP_linkModulation, "link_modulation" );
    createROIndiText( m_indiP_reconstructionStatus, "reconstruction_status", "reconstructor" );
    for( const auto *element : { "phase", "error", "modulation", "heartbeat", "instance" } )
        m_indiP_reconstructionStatus.add( pcf::IndiElement( element ) );
    if( registerIndiPropertyNew( m_indiP_nnActivate, INDI_NEWCALLBACK( m_indiP_nnActivate ) ) < 0 ||
        registerIndiPropertyNew( m_indiP_reconstructor, INDI_NEWCALLBACK( m_indiP_reconstructor ) ) < 0 ||
        registerIndiPropertyNew( m_indiP_linkModulation, INDI_NEWCALLBACK( m_indiP_linkModulation ) ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_reconstructionStatus ) < 0 )
        return -1;
    REG_INDI_SETPROP( m_indiP_modState, m_modDevice, "modState" );
    REG_INDI_SETPROP( m_indiP_modFrequency, m_modDevice, "modFrequency" );
    REG_INDI_SETPROP( m_indiP_modRadius, m_modDevice, "modRadius" );
    createROIndiText( m_indiP_routingHold, "routing_hold", "owner" );
    m_indiP_routingHold.add( pcf::IndiElement( "action" ) );
    m_indiP_routingHold.setPerm( pcf::IndiProperty::ReadWrite );
    createROIndiText( m_indiP_routingStatus, "routing_status", "owner" );
    for( const auto *element : { "source", "phase", "instance", "sequence" } )
        m_indiP_routingStatus.add( pcf::IndiElement( element ) );
    if( registerIndiPropertyNew( m_indiP_routingHold, INDI_NEWCALLBACK( m_indiP_routingHold ) ) < 0 ||
        registerIndiPropertyReadOnly( m_indiP_routingStatus ) < 0 )
        return -1;
    m_switchInstance = std::to_string( std::chrono::system_clock::now().time_since_epoch().count() );
    return 0;
}

inline void nnReconstructor::reconstructorPhase( const std::string &phase )
{
    m_switchPhase   = phase;
    m_switchStarted = std::chrono::steady_clock::now();
}

inline int nnReconstructor::requestReconstructor( const std::string &target, const std::string &expected, int model )
{
    if( m_switchPhase != "idle" && m_switchPhase != "error" )
        return -1;
    m_switchError.clear();
    if( ( target != "nn" && target != "cacao" ) || model < -1 || model >= static_cast<int>( m_models.size() ) ||
        ( !expected.empty() && expected != "nn" && expected != "cacao" && expected != "stopped" ) )
    {
        m_switchError = "Invalid reconstructor target, expected source or catalog selection";
        reconstructorPhase( "error" );
        return -1;
    }
    m_switchTarget   = target;
    m_switchExpected = expected;
    m_switchModel    = model;
    m_switchLinked   = m_linkModulation;
    reconstructorPhase( "queued" );
    return 0;
}

inline void nnReconstructor::startReconstructor()
{
    if( m_switchTarget == "nn" )
    {
        if( m_cacaoWriter.running() )
            throw std::runtime_error( "CACAO must finish stopping before NN activation" );
        if( requestEnabled( true ) < 0 )
            throw std::runtime_error( "NN activation rejected; check model status" );
        reconstructorPhase( "starting_nn" );
    }
    else
    {
        m_cacaoWriter.command( true );
        m_cacaoStartPending = true;
        reconstructorPhase( "starting_cacao" );
    }
}

inline void nnReconstructor::prepareModulation()
{
    if( !m_switchLinked || ( m_switchSource == "nn" && m_switchTarget == "nn" ) ||
        ( m_switchSource == "stopped" && m_switchTarget == "nn" ) )
    {
        startReconstructor();
        return;
    }
    for( size_t i : { 1U, 2U, 0U } ) // Apply frequency/radius before enabling modulation.
    {
        if( m_switchTarget == "nn" && i != 0 )
            continue;
        pcf::IndiProperty request( pcf::IndiProperty::Number );
        request.setDevice( m_modDevice );
        request.setName( i == 0 ? "modState" : i == 1 ? "modFrequency" : "modRadius" );
        const double target = i == 0 ? ( m_switchTarget == "nn" ? 3.0 : 4.0 ) : i == 1 ? 2000.0 : 3.0;
        request.add( pcf::IndiElement( "target", target ) );
        if( sendNewProperty( request ) < 0 )
            throw std::runtime_error( "Modulation request delivery failed" );
    }
    reconstructorPhase( "modulation" );
}

inline void nnReconstructor::updateReconstructorControls()
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    const auto                  now   = std::chrono::steady_clock::now();
    bool                        cacao = false, writerKnown = false, quiescent, active, loading, resident;
    std::string                 loaded, requested, fault;
    try
    {
        cacao       = m_cacaoWriter.running();
        writerKnown = true;
        if( m_switchPhase == "idle" )
            m_switchError.clear();
    }
    catch( const std::exception &error )
    {
        if( m_switchPhase == "idle" )
            m_switchError = error.what();
    }
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_resultMutex );
        quiescent = !m_enabled && !m_enableReadback && !m_disabling && !m_loading && !m_inferencing && !m_publishing;
        active    = m_enabled && m_enableReadback && !m_activeModel.empty() && m_fault.empty();
        loading   = m_loading;
        resident  = m_gpuResident;
        loaded    = m_loadedModel;
        requested = m_requestedModel;
        fault     = m_fault;
    }
    m_reconstructor = !writerKnown            ? "unknown"
                      : !active && !quiescent ? "transition"
                      : active                ? ( cacao ? "conflict" : "nn" )
                      : cacao                 ? "cacao"
                                              : "stopped";
    try
    {
        if( m_switchPhase != "idle" && m_switchPhase != "error" && m_switchPhase != "awaiting_cacao" &&
            std::chrono::duration<double>( now - m_switchStarted ).count() > m_transitionTimeout )
            throw std::runtime_error( "Reconstructor operation timed out in " + m_switchPhase );
        if( m_switchPhase == "awaiting_cacao" && writerKnown && cacao )
        {
            m_cacaoStartPending = false;
            reconstructorPhase( "error" ); // Late completion settles uncertainty, never starts another action.
        }
        else if( m_switchPhase == "queued" )
        {
            if( m_reconstructor != "nn" && m_reconstructor != "cacao" && m_reconstructor != "stopped" )
                throw std::runtime_error( "Reconstructor ownership is not verified" );
            if( !m_switchExpected.empty() && m_switchExpected != m_reconstructor )
                throw std::runtime_error( "Reconstructor changed since the request" );
            m_switchSource = m_reconstructor;
            if( m_switchTarget == m_switchSource &&
                ( m_switchTarget == "cacao" || m_switchModel < 0 || m_models[m_switchModel] == loaded ) )
                reconstructorPhase( "idle" );
            else
            {
                if( requestEnabled( false ) < 0 )
                    throw std::runtime_error( "NN disable rejected" );
                reconstructorPhase( "disabling" );
            }
        }
        else if( m_switchPhase == "disabling" && quiescent )
        {
            if( m_switchTarget == "nn" && m_switchModel >= 0 &&
                ( loaded != m_models[m_switchModel] || requested != loaded || !fault.empty() ) )
            {
                pcf::IndiProperty request( pcf::IndiProperty::Switch );
                request.add( pcf::IndiElement( modelId( m_switchModel ), pcf::IndiElement::On ) );
                if( requestModel( request ) < 0 )
                    throw std::runtime_error( "NN model selection rejected" );
                reconstructorPhase( "loading" );
            }
            else
                reconstructorPhase( "handover" );
        }
        else if( m_switchPhase == "loading" && !loading )
        {
            if( !fault.empty() || loaded != m_models[m_switchModel] || requested != loaded )
                throw std::runtime_error( "NN model load failed: " + fault );
            reconstructorPhase( "handover" );
        }
        else if( m_switchPhase == "handover" && writerKnown && quiescent )
        {
            if( cacao && m_switchSource != "cacao" )
                throw std::runtime_error( "CACAO writer changed during NN operation" );
            if( m_switchTarget == "nn" && !resident )
            {
                if( requestEnabled( true, true ) < 0 )
                    throw std::runtime_error( "NN preparation rejected" );
                reconstructorPhase( "preparing_nn" );
            }
            else if( m_switchTarget == "nn" && cacao )
            {
                m_cacaoWriter.command( false );
                reconstructorPhase( "stopping_cacao" );
            }
            else
                prepareModulation();
        }
        else if( m_switchPhase == "preparing_nn" && !loading )
        {
            if( !fault.empty() || !resident )
                throw std::runtime_error( "NN GPU preparation failed: " + fault );
            reconstructorPhase( "handover" );
        }
        else if( m_switchPhase == "stopping_cacao" && writerKnown && !cacao && quiescent )
            prepareModulation();
        else if( m_switchPhase == "modulation" && writerKnown && quiescent )
        {
            bool applied = true;
            for( size_t i = 0; i < ( m_switchTarget == "nn" ? 1U : 3U ); ++i )
            {
                const double target = i == 0 ? ( m_switchTarget == "nn" ? 3.0 : 4.0 ) : i == 1 ? 2000.0 : 3.0;
                applied &= m_modObserved[i] > m_switchStarted &&
                           std::chrono::duration<double>( now - m_modObserved[i] ).count() < 5 &&
                           std::abs( m_modCurrent[i] - target ) < 0.01;
            }
            if( applied )
                startReconstructor();
        }
        else if( ( m_switchPhase == "starting_nn" && m_reconstructor == "nn" ) ||
                 ( m_switchPhase == "starting_cacao" && m_reconstructor == "cacao" ) )
        {
            m_cacaoStartPending = false;
            reconstructorPhase( "idle" );
        }
    }
    catch( const std::exception &error )
    {
        m_switchError = error.what();
        if( m_switchPhase == "loading" || m_switchPhase == "preparing_nn" )
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_resultMutex );
            m_cancelLoad = true; // A timed-out operation never installs and activates a late candidate.
        }
        if( !active && resident )
            requestEnabled( false );
        reconstructorPhase( m_cacaoStartPending ? "awaiting_cacao" : "error" );
    }
    if( m_routingPhase == "pausing" && quiescent )
        m_routingPhase = "held";
    if( m_routingPhase == "restoring" && m_switchPhase == "idle" && m_reconstructor == m_routingSource )
    {
        m_routingPhase = "released";
        m_routingOwner.clear();
    }
    if( m_routingPhase == "restoring" && m_switchPhase == "error" )
        m_routingPhase = "error";
    const auto routingState = m_routingPhase == "error" ? INDI_ALERT : INDI_OK;
    updateIfChanged( m_indiP_routingStatus, "owner", m_routingOwner, routingState );
    updateIfChanged( m_indiP_routingStatus, "source", m_routingSource, routingState );
    updateIfChanged( m_indiP_routingStatus, "phase", m_routingPhase, routingState );
    updateIfChanged( m_indiP_routingStatus, "instance", m_switchInstance, routingState );
    updateIfChanged( m_indiP_routingStatus, "sequence", std::to_string( m_switchSequence ), routingState );
    if( std::chrono::duration<double>( now - m_modQuery ).count() >= 1 )
    {
        m_modQuery = now;
        try
        {
            if( m_indiDriver )
                for( const auto *property : { &m_indiP_modState, &m_indiP_modFrequency, &m_indiP_modRadius } )
                    m_indiDriver->sendGetProperties( *property );
        }
        catch( const std::exception & )
        {
            m_modObserved = {}; // Lost discovery cannot complete a linked handover.
        }
    }
    const bool busy       = m_switchPhase != "idle" && m_switchPhase != "error";
    const auto state      = m_switchPhase == "awaiting_cacao" ? INDI_ALERT
                            : busy                            ? INDI_BUSY
                            : m_switchPhase == "error"        ? INDI_ALERT
                                                              : INDI_OK;
    const auto modulation = std::chrono::duration<double>( now - m_modObserved[0] ).count() < 5
                                ? std::to_string( m_modCurrent[0] )
                                : "Unknown";
    updateIfChanged( m_indiP_reconstructionStatus, "reconstructor", m_reconstructor, state );
    updateIfChanged( m_indiP_reconstructionStatus, "phase", m_switchPhase, state );
    updateIfChanged( m_indiP_reconstructionStatus, "error", m_switchError, state );
    updateIfChanged( m_indiP_reconstructionStatus, "modulation", modulation, state );
    updateIfChanged( m_indiP_reconstructionStatus, "instance", m_switchInstance, state );
    updateIfChanged( m_indiP_reconstructionStatus, "heartbeat", std::to_string( ++m_switchSequence ), state );
    updateIfChanged( m_indiP_nnActivate, "model", m_indiP_nnActivate["model"].get<std::string>(), state );
    updateIfChanged( m_indiP_reconstructor, "expected", m_switchExpected, state );
    updateIfChanged( m_indiP_reconstructor, "target", m_switchTarget, state );
    updateSwitchIfChanged(
        m_indiP_linkModulation, "toggle", m_linkModulation ? pcf::IndiElement::On : pcf::IndiElement::Off, INDI_OK );
}

INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_nnActivate )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nnActivate, property );
    if( !property.find( "model" ) )
        return -1;
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !m_routingOwner.empty() )
        return -1;
    int model = -2;
    for( size_t i = 0; i < m_models.size(); ++i )
        if( property["model"].get<std::string>() == modelId( i ) )
            model = i;
    const int result = requestReconstructor( "nn", "", model );
    if( result == 0 || m_switchPhase == "error" )
        updateIfChanged(
            m_indiP_nnActivate, "model", property["model"].get<std::string>(), result == 0 ? INDI_BUSY : INDI_ALERT );
    return result;
}

INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_reconstructor )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reconstructor, property );
    if( !property.find( "target" ) || !property.find( "expected" ) || property["expected"].get<std::string>().empty() )
        return -1;
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !m_routingOwner.empty() )
        return -1;
    const int result =
        requestReconstructor( property["target"].get<std::string>(), property["expected"].get<std::string>(), -1 );
    if( result == 0 || m_switchPhase == "error" )
        updateIfChanged( m_indiP_reconstructor,
                         "target",
                         property["target"].get<std::string>(),
                         result == 0 ? INDI_BUSY : INDI_ALERT );
    return result;
}

INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_linkModulation )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_linkModulation, property );
    if( !property.find( "toggle" ) )
        return -1;
    const auto value = property["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( !m_routingOwner.empty() )
        return -1;
    updateSwitchIfChanged( m_indiP_linkModulation, "toggle", value, INDI_BUSY );
    m_linkModulation = value == pcf::IndiElement::On;
    return 0;
}

inline void nnReconstructor::observeModulation( const pcf::IndiProperty &property, size_t index )
{
    if( property.find( "current" ) )
    {
        const double value   = property["current"].get<double>();
        m_modObserved[index] = {};
        if( isFinite( value ) && property.getState() != INDI_ALERT )
        {
            m_modCurrent[index]  = value;
            m_modObserved[index] = std::chrono::steady_clock::now();
        }
    }
}

INDI_SETCALLBACK_DEFN( nnReconstructor, m_indiP_modState )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_modState, property );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    observeModulation( property, 0 );
    return 0;
}

INDI_SETCALLBACK_DEFN( nnReconstructor, m_indiP_modFrequency )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_modFrequency, property );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    observeModulation( property, 1 );
    return 0;
}

INDI_SETCALLBACK_DEFN( nnReconstructor, m_indiP_modRadius )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_modRadius, property );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    observeModulation( property, 2 );
    return 0;
}
INDI_NEWCALLBACK_DEFN( nnReconstructor, m_indiP_routingHold )( const pcf::IndiProperty &property )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_routingHold, property );
    if( !property.find( "owner" ) || !property.find( "action" ) )
        return -1;
    const auto owner  = property["owner"].get<std::string>();
    const auto action = property["action"].get<std::string>();
    if( owner.empty() || owner.size() > 128 )
        return -1;
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( action == "hold" )
    {
        if( !m_routingOwner.empty() )
            return m_routingOwner == owner && m_routingPhase == "held" ? 0 : -1;
        if( m_switchPhase != "idle" )
            return -1;
        bool active, quiet;
        { // mutex scope
            std::lock_guard<std::mutex> resultLock( m_resultMutex );
            active = m_enabled && m_enableReadback && !m_activeModel.empty() && m_fault.empty();
            quiet  = !m_enabled && !m_loading && !m_disabling && !m_gpuReleasing && !m_inferencing && !m_publishing;
        }
        try
        {
            const bool cacao = m_cacaoWriter.running();
            if( ( active && cacao ) || ( !active && !quiet ) || ( !active && !cacao ) )
                return -1;
            m_routingSource = active ? "nn" : "cacao";
            if( requestEnabled( false ) < 0 )
                return -1;
            m_routingOwner = owner;
            m_routingPhase = "pausing";
        }
        catch( const std::exception & )
        {
            return -1;
        }
    }
    else if( action == "release" && m_routingOwner == owner &&
             ( m_routingPhase == "held" || m_routingPhase == "error" ) )
    {
        // Standard handover verifies CACAO has stopped before re-admitting NN.
        if( requestReconstructor( m_routingSource, "", -1 ) < 0 )
            return -1;
        m_switchLinked = false; // Restore writer ownership without changing the operator's modulation.
        m_routingPhase = "restoring";
    }
    else
        return -1;
    return 0;
}

#endif
