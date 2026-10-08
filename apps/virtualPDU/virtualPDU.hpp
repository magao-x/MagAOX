/** \file virtualPDU.hpp
 * \brief Combine named remote power channels through the standard outlet-controller interface.
 * \ingroup virtualPDU_files
 */
#ifndef virtualPDU_hpp
#define virtualPDU_hpp

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

#include <charconv>
#include <set>
#include <string_view>
#include <limits>

namespace MagAOX
{
namespace app
{
/** \defgroup virtualPDU Virtual Power Distribution Unit
 * \brief Ordered command dispatch to named channels on other power controllers.
 * \ingroup apps
 */
/** \defgroup virtualPDU_files Virtual PDU Files
 * \ingroup virtualPDU
 */

/// Virtual outlets observe and command remote outlet-controller channels.
/** \ingroup virtualPDU
 * Each [outletN] section maps one-based N to device/channel. Ordinary channel sections
 * combine these outlet numbers using outletController's order and millisecond-delay language.
 * Commands are dispatched without waiting for confirmation. Only received state elements
 * determine observed power. Unavailable sources block only channels that use them.
 */
class virtualPDU : public MagAOXApp<>, public dev::outletController<virtualPDU>, public dev::telemeter<virtualPDU>
{
protected:
    /// Give the single telemetry helper access to application state.
    friend class dev::telemeter<virtualPDU>;

    /// Application-owned telemetry helper.
    typedef dev::telemeter<virtualPDU> telemeterT;

    /// Common channel implementation and sequence locks.
    typedef dev::outletController<virtualPDU> outletControllerT;

    /// Monotonic clock used to bound source freshness.
    typedef std::chrono::steady_clock clockT;

    /** \name Configurable Parameters - Data
     * @{ */
    /// Seconds between explicit source channel/FSM refreshes.
    double m_pollInterval {5};

    /// Maximum seconds without a valid source observation.
    double m_staleTimeout {15};
    ///@}

    /// One observed remote channel backing a virtual outlet.
    struct endpoint
    {
        /// INDI device owning the channel.
        std::string m_device;

        /// INDI channel property name.
        std::string m_channel;

        /// Stable subscription storage, registered only after configuration completes.
        pcf::IndiProperty m_property;

        /// Index of the unique source-device FSM subscription.
        size_t m_source {0};

        /// Last received observed state, never inferred from a command or target.
        int m_state {OUTLET_STATE_UNKNOWN};

        /// Whether an actual state element has been received since invalidation.
        bool m_haveState {false};

        /// Monotonic receipt time of the last valid state element.
        clockT::time_point m_received {};
    };

    /// Readiness observation shared by all endpoints on one remote device.
    struct source
    {
        /// Stable storage for the remote fsm.state subscription.
        pcf::IndiProperty m_property;

        /// Whether the last received FSM state was READY.
        bool m_ready {false};

        /// Monotonic receipt time of the last valid FSM state.
        clockT::time_point m_received {};
    };

    /// One endpoint per contiguous one-based configuration outlet, immutable after startup.
    std::vector<endpoint> m_endpoints;

    /// One FSM subscription per unique remote device, immutable after startup.
    std::vector<source> m_sources;

    /// Last explicit subscription refresh time.
    clockT::time_point m_lastPoll {};

public:
    /// Construct a service whose own power management is disabled.
    virtualPDU();

    /// Destroy the service and its helper state.
    ~virtualPDU() noexcept;

    /// Register source refresh and helper configuration options.
    void setupConfig() override;

    /// Load configuration and request shutdown on invalid configuration.
    void loadConfig() override;

    /// Parse and validate remote mappings and ordinary channel sections.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] app configuration */ );

    /// Register standard outlet properties and stable remote subscriptions.
    int appStartup() override;

    /// Refresh source definitions, invalidate stale observations, and publish telemetry/INDI.
    int appLogic() override;

    /// Stop application telemetry without issuing remote power commands.
    int appShutdown() override;

    /// Copy an available source's observed state into one virtual outlet.
    int updateOutletState( int outletNum /**< [in] zero-based virtual outlet index */ );

    /// Dispatch On to an available remote channel without changing observed state.
    int turnOutletOn( int outletNum /**< [in] zero-based virtual outlet index */ );

    /// Dispatch Off to an available remote channel without changing observed state.
    int turnOutletOff( int outletNum /**< [in] zero-based virtual outlet index */ );

    /// Validate a channel request and preflight all its sources before normal sequencing.
    int newCallBack_channels( const pcf::IndiProperty &ipRecv /**< [in] virtual channel command */ );

    /// Route a received source Def/Set through the application instance.
    static int st_setCallBack_source( void *app /**< [in] application instance */,
                                      const pcf::IndiProperty &ipRecv /**< [in] source property */ );

    /// Merge source observations; partial/target-only updates do not invent observed state.
    int setCallBack_source( const pcf::IndiProperty &ipRecv /**< [in] source channel or FSM observation */ );

    /// Schedule periodic observed outlet telemetry.
    int checkRecordTimes();

    /// Force an observed outlet telemetry record.
    int recordTelem( const telem_outlet *type /**< [in] unused type selector */ );

protected:
    /// Test endpoint availability from valid state/FSM observations and their receipt times.
    bool available( size_t index /**< [in] zero-based endpoint index */,
                    clockT::time_point now /**< [in] current monotonic time */ ) const;

    /// Send a minimal Text target command to an available endpoint.
    int sendOutlet( int outletNum /**< [in] zero-based endpoint index */,
                    const std::string &target /**< [in] On or Off */ );
};

inline virtualPDU::virtualPDU() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_firstOne = true;
    m_powerMgtEnabled = false;
}

inline virtualPDU::~virtualPDU() noexcept
{
}

inline void virtualPDU::setupConfig()
{
    config.add( "device.pollInterval", "", "device.pollInterval", argType::Required, "device", "pollInterval", false,
                "double", "Seconds between source refreshes (default 5)." );
    config.add( "device.staleTimeout", "", "device.staleTimeout", argType::Required, "device", "staleTimeout", false,
                "double", "Seconds without a valid source observation before invalidation (default 15)." );
    outletControllerT::setupConfig( config );
    TELEMETER_SETUP_CONFIG( config );
}

inline void virtualPDU::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<text_log>( "Invalid virtual PDU configuration", logPrio::LOG_CRITICAL );
        m_shutdown = true;
    }
}

inline int virtualPDU::loadConfigImpl( mx::app::appConfigurator &config )
{
    config( m_pollInterval, "device.pollInterval" );
    config( m_staleTimeout, "device.staleTimeout" );
    if( !(m_pollInterval > 0 && m_staleTimeout > m_pollInterval) ) return -1;
    std::vector<std::string> sections;
    config.unusedSections( sections );
    std::map<size_t, endpoint> mappings;
    std::set<std::pair<std::string, std::string>> identities;
    for( const auto &section : sections )
    {
        if( !section.starts_with( "outlet" ) ) continue;
        size_t number = 0;
        auto suffix = section.substr( 6 );
        auto parsed = std::from_chars( suffix.data(), suffix.data() + suffix.size(), number );
        if( parsed.ec != std::errc() || parsed.ptr != suffix.data() + suffix.size() || number == 0 ||
            section != "outlet" + std::to_string( number ) ) return -1;
        endpoint mapping;
        config.configUnused( mapping.m_device, section, "device" );
        config.configUnused( mapping.m_channel, section, "channel" );
        if( mapping.m_device.empty() || mapping.m_channel.empty() || mapping.m_device == configName() ||
            mapping.m_channel == "fsm" ||
            !identities.emplace( mapping.m_device, mapping.m_channel ).second ) return -1;
        mappings.emplace( number, mapping );
    }
    if( mappings.empty() || mappings.rbegin()->first != mappings.size() ) return -1;
    for( const auto &[number, mapping] : mappings )
    {
        static_cast<void>( number );
        m_endpoints.push_back( mapping );
    }
    // Validate numeric tokens before the base's permissive numeric conversions.
    for( const auto &section : sections )
    {
        for( const auto &keyword : { "outlet", "outlets", "onOrder", "offOrder", "onDelays", "offDelays" } )
        {
            if( !config.isSetUnused( mx::app::iniFile::makeKey( section, keyword ) ) ) continue;
            std::vector<std::string> values;
            config.configUnused( values, section, keyword );
            for( auto value : values )
            {
                size_t first = value.find_first_not_of( " \t" );
                size_t last = value.find_last_not_of( " \t" );
                if( first == std::string::npos ) return -1;
                value = value.substr( first, last-first+1 );
                size_t number;
                auto result = std::from_chars( value.data(), value.data()+value.size(), number );
                if( result.ec != std::errc() || result.ptr != value.data()+value.size() ) return -1;
                std::string_view name = keyword;
                if( (name == "outlet" || name == "outlets") && (number == 0 || number > m_endpoints.size()) ) return -1;
                if( (name == "onDelays" || name == "offDelays") && number > std::numeric_limits<unsigned>::max() ) return -1;
            }
        }
    }
    setNumberOfOutlets( m_endpoints.size() );
    if( outletControllerT::loadConfig( config ) < 0 ) return -1;
    std::set<size_t> assigned;
    const std::set<std::string> reserved { "outlet", "stateTimes", "channelOutlets", "channelOnDelays",
                                         "channelOffDelays", "fsm", "telem_rotate", "telem_maxtime", "fsm_clear_alert", "logs_rotate", "logs_maxtime" };
    for( const auto &[name, channel] : m_channels )
    {
        if( reserved.count( name ) ) return -1;
        for( auto number : channel.m_outlets )
        {
            if( number >= m_endpoints.size() || !assigned.insert( number ).second ) return -1;
        }
        for( const auto &order : { channel.m_onOrder, channel.m_offOrder } )
        {
            if( order.empty() ) continue;
            std::set<size_t> indices( order.begin(), order.end() );
            if( indices.size() != channel.m_outlets.size() || *indices.rbegin() >= channel.m_outlets.size() ) return -1;
        }
    }
    TELEMETER_LOAD_CONFIG( config );
    return 0;
}

inline int virtualPDU::appStartup()
{
    if( outletControllerT::appStartup() < 0 ) return -1;
    // Allocate every subscription before registering pointers to its backing storage.
    std::map<std::string, size_t> sources;
    for( auto &mapping : m_endpoints )
    {
        auto [it, inserted] = sources.emplace( mapping.m_device, sources.size() );
        mapping.m_source = it->second;
        if( inserted )
        {
            source device;
            device.m_property.setDevice( mapping.m_device );
            device.m_property.setName( "fsm" );
            m_sources.push_back( device );
        }
    }
    for( auto &device : m_sources )
    {
        std::string deviceName = device.m_property.getDevice();
        if( registerIndiPropertySet( device.m_property, deviceName, "fsm", st_setCallBack_source ) < 0 )
            return -1;
    }
    for( auto &mapping : m_endpoints )
    {
        if( registerIndiPropertySet( mapping.m_property, mapping.m_device, mapping.m_channel, st_setCallBack_source ) < 0 )
            return -1;
    }
    TELEMETER_APP_STARTUP;
    state( stateCodes::READY );
    return 0;
}

inline bool virtualPDU::available( size_t index, clockT::time_point now ) const
{
    const auto &mapping = m_endpoints[index];
    if( mapping.m_source >= m_sources.size() ) return false;
    const auto &device = m_sources[mapping.m_source];
    return mapping.m_haveState && device.m_ready &&
           std::chrono::duration<double>( now - mapping.m_received ).count() < m_staleTimeout &&
           std::chrono::duration<double>( now - device.m_received ).count() < m_staleTimeout;
}

inline int virtualPDU::appLogic()
{
    std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );
    if( !lock.owns_lock() ) return 0;
    auto now = clockT::now();
    if( m_indiDriver && std::chrono::duration<double>( now - m_lastPoll ).count() >= m_pollInterval )
    {
        sendGetPropertySetList( true );
        m_lastPoll = now;
    }
    if( updateOutletStates() < 0 ) return -1;
    outletControllerT::updateINDI();
    if( recordOutletStates() < 0 ) return -1;
    TELEMETER_APP_LOGIC;
    return 0;
}

inline int virtualPDU::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int virtualPDU::updateOutletState( int outletNum )
{
    if( outletNum < 0 || static_cast<size_t>( outletNum ) >= m_endpoints.size() ) return -1;
    const auto &mapping = m_endpoints[outletNum];
    setOutletState( outletNum, available( outletNum, clockT::now() ) ? mapping.m_state : OUTLET_STATE_UNKNOWN );
    return 0;
}

inline int virtualPDU::sendOutlet( int outletNum, const std::string &target )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( state() != stateCodes::READY || outletNum < 0 || static_cast<size_t>( outletNum ) >= m_endpoints.size() ||
        !available( outletNum, clockT::now() ) )
        return log<software_error, -1>( "Virtual outlet source unavailable" );
    const auto &mapping = m_endpoints[outletNum];
    pcf::IndiProperty command( pcf::IndiProperty::Text, mapping.m_device, mapping.m_channel );
    command.add( pcf::IndiElement( "target", target ) );
    return sendNewProperty( command );
}

inline int virtualPDU::turnOutletOn( int outletNum )
{
    return sendOutlet( outletNum, "On" );
}

inline int virtualPDU::turnOutletOff( int outletNum )
{
    return sendOutlet( outletNum, "Off" );
}

inline int virtualPDU::newCallBack_channels( const pcf::IndiProperty &ipRecv )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_indiMutex );
        if( state() != stateCodes::READY ) return -1;
        auto channel = m_channels.find( ipRecv.getName() );
        if( ipRecv.getDevice() != configName() || ipRecv.getType() != pcf::IndiProperty::Text || channel == m_channels.end() )
            return -1;
        for( auto index : channel->second.m_outlets )
        {
            if( !available( index, clockT::now() ) ) return log<software_error, -1>( "Virtual channel source unavailable" );
        }
    }
    return outletControllerT::newCallBack_channels( ipRecv );
}

inline int virtualPDU::st_setCallBack_source( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<virtualPDU *>( app )->setCallBack_source( ipRecv );
}

inline int virtualPDU::setCallBack_source( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    auto now = clockT::now();
    if( ipRecv.getName() == "fsm" )
    {
        for( size_t index = 0; index < m_sources.size(); ++index )
        {
            auto &device = m_sources[index];
            if( device.m_property.getDevice() != ipRecv.getDevice() ) continue;
            bool malformed = ipRecv.getType() != pcf::IndiProperty::Text;
            if( !malformed && !ipRecv.find( "state" ) ) return -1;
            device.m_ready = !malformed && ipRecv["state"].get<std::string>() == "READY";
            if( !malformed ) device.m_received = now;
            for( size_t n = 0; n < m_endpoints.size(); ++n )
            {
                if( m_endpoints[n].m_source != index ) continue;
                if( !device.m_ready ) m_endpoints[n].m_haveState = false;
                updateOutletState( n );
            }
            outletControllerT::updateINDI();
            int rv = recordOutletStates();
            return malformed ? -1 : rv;
        }
        return -1;
    }
    for( size_t index = 0; index < m_endpoints.size(); ++index )
    {
        auto &mapping = m_endpoints[index];
        if( mapping.m_device != ipRecv.getDevice() || mapping.m_channel != ipRecv.getName() ) continue;
        bool text = ipRecv.getType() == pcf::IndiProperty::Text;
        if( text && !ipRecv.find( "state" ) ) return 0;
        std::string value = text ? ipRecv["state"].get<std::string>() : "";
        mapping.m_state = OUTLET_STATE_UNKNOWN;
        if( ipRecv.getType() == pcf::IndiProperty::Text )
        {
            if( value == "On" ) mapping.m_state = OUTLET_STATE_ON;
            else if( value == "Off" ) mapping.m_state = OUTLET_STATE_OFF;
            else if( value == "Int" ) mapping.m_state = OUTLET_STATE_INTERMEDIATE;
        }
        mapping.m_haveState = ipRecv.getType() == pcf::IndiProperty::Text &&
                              (mapping.m_state != OUTLET_STATE_UNKNOWN || value == "Unk");
        mapping.m_received = now;
        updateOutletState( index );
        outletControllerT::updateINDI();
        return recordOutletStates();
    }
    return -1;
}

inline int virtualPDU::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_outlet() );
}

inline int virtualPDU::recordTelem( const telem_outlet * )
{
    return recordOutletStates( true );
}
} // namespace app
} // namespace MagAOX
#endif // virtualPDU_hpp
