/** \file stateRuleEngine.hpp
 * \brief The MagAO-X stateRuleEngine application header file
 *
 * \ingroup stateRuleEngine_files
 */

#ifndef stateRuleEngine_hpp
#define stateRuleEngine_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "indiCompRuleConfig.hpp"

/** \defgroup stateRuleEngine
 * \brief The MagAO-X stateRuleEngine application
 *
 * <a href="../handbook/operating/software/apps/stateRuleEngine.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup stateRuleEngine_files
 * \ingroup stateRuleEngine
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X stateRuleEngine
/**
 * \ingroup stateRuleEngine
 */
class stateRuleEngine : public MagAOXApp<true>
{

    // Give the test harness access.
    friend class stateRuleEngine_test;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_ruleDir; /**< Directory containing config files containing rules to load. Relative to config
                              directory.  If this is set, then rules in the device config file are ignored*/

    /// Owns the configured rules and subscribed INDI property objects.
    indiRuleMaps m_ruleMaps;

    ///@}

    /// Get the published rule-state property for a reporting priority.
    pcf::IndiProperty *
    ruleStateProperty( const rulePriority &priority /**< [in] the reporting priority for the rule */ );

    /// Get the notification label for a reporting priority.
    static std::string
    notificationLabel( const rulePriority &priority /**< [in] the reporting priority for the rule */ );

    /// Report whether a published rule element is currently `On`.
    static bool ruleIsOn( pcf::IndiProperty &property, /**< [in] the published property to inspect */
                          const std::string &ruleName /**< [in] the rule element name to inspect */ );

    /// Format a notification message for a rule.
    static std::string notificationMessage(
        const std::string &ruleName,        /**< [in] the rule name used as fallback text */
        indiCompRule      &rule,            /**< [in] the rule whose message text is used */
        const std::string &label,           /**< [in] the label prefix, e.g. `INFO` */
        bool               cleared = false, /**< [in] true when formatting a clear notification */
        bool               settime = false /**< [in] true when reading the rule message should set its send time */ );

    /// Send one formatted notification through the INDI driver.
    virtual int sendNotification( const std::string &message /**< [in] the fully formatted notification message */ );

  public:
    /// Default c'tor.
    stateRuleEngine();

    /// D'tor, declared and defined for noexcept.
    ~stateRuleEngine() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for stateRuleEngine.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    /// The static callback function to be registered for rule properties
    /**
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_ruleProp(
        void                    *app,   ///< [in] a pointer to this, will be static_cast-ed to derivedT.
        const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
    );

    /// Callback to process a NEW preset position request
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_ruleProp(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/ );

    /// Published `info`-priority rule states.
    pcf::IndiProperty m_indiP_info;

    /// Published `caution`-priority rule states.
    pcf::IndiProperty m_indiP_caution;

    /// Published `warning`-priority rule states.
    pcf::IndiProperty m_indiP_warning;

    /// Published `alert`-priority rule states.
    pcf::IndiProperty m_indiP_alert;
};

stateRuleEngine::stateRuleEngine() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

pcf::IndiProperty *stateRuleEngine::ruleStateProperty( const rulePriority &priority )
{
    if( priority == rulePriority::info )
    {
        return &m_indiP_info;
    }

    if( priority == rulePriority::caution )
    {
        return &m_indiP_caution;
    }

    if( priority == rulePriority::warning )
    {
        return &m_indiP_warning;
    }

    if( priority == rulePriority::alert )
    {
        return &m_indiP_alert;
    }

    return nullptr;
}

std::string stateRuleEngine::notificationLabel( const rulePriority &priority )
{
    if( priority == rulePriority::info )
    {
        return "INFO";
    }

    if( priority == rulePriority::caution )
    {
        return "CAUTION";
    }

    if( priority == rulePriority::warning )
    {
        return "WARNING";
    }

    if( priority == rulePriority::alert )
    {
        return "ALERT";
    }

    return "INFO";
}

bool stateRuleEngine::ruleIsOn( pcf::IndiProperty &property, const std::string &ruleName )
{
    if( !property.find( ruleName ) )
    {
        return false;
    }

    return property[ruleName].getSwitchState() == pcf::IndiElement::On;
}

std::string stateRuleEngine::notificationMessage(
    const std::string &ruleName, indiCompRule &rule, const std::string &label, bool cleared, bool settime )
{
    std::string detail;
    if( settime )
    {
        detail = rule.message( true );
    }
    else
    {
        detail = rule.message();
    }

    if( detail == "" )
    {
        detail = ruleName;
    }

    if( cleared )
    {
        detail = std::format( "Cleared: {}", detail );
    }

    return std::format( "{}: {}", label, detail );
}

int stateRuleEngine::sendNotification( const std::string &message )
{
    if( m_indiDriver == nullptr )
    {
        return 0;
    }

    pcf::IndiProperty ip;
    ip.setDevice( m_configName );
    ip.setMessage( message );

    try
    {
        m_indiDriver->sendMessage( ip );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( std::format( "exception caught from sendMessage: {}", e.what() ) );
    }

    return 0;
}

void stateRuleEngine::setupConfig()
{
    config.add( "rules.dir",
                "",
                "rules.dir",
                argType::Required,
                "rules",
                "dir",
                false,
                "string",
                "Directory containing config files containing rules to load. Relative to config directory.  If this is "
                "set, then rules in the device config file are ignored" );
}

int stateRuleEngine::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_ruleDir, "rules.dir" );

    std::map<std::string, ruleRuleKeys> rrkMap;

    if( m_ruleDir == "" )
    {
        try
        {
            loadRuleConfig( m_ruleMaps, rrkMap, _config );
            finalizeRuleValRules( m_ruleMaps, rrkMap );
        }
        catch( const std::exception &e )
        {
            return log<software_critical, -1>( std::format( "Rule config exception caught:\n{}", e.what() ) );
        }
    }
    else
    {
        std::vector<std::string> conffiles;
        if( mx::ioutils::getFileNames( conffiles, m_configDir + '/' + m_ruleDir, "", "", ".conf" ) !=
            mx::error_t::noerror )
        {
            return log<software_critical, -1>( "Error reading rules" );
        }

        for( auto &cnf : conffiles )
        {
            // Create a configurator and set it up to log
            mx::app::appConfigurator fcfg;

            fcfg.m_sources = true;
            fcfg.configLog = configLog;

            // now process the config file
            if( fcfg.readConfig( cnf ) < 0 )
            {
                return log<software_critical, -1>( std::format( "error reading rule config file: {}", cnf ) );
            }

            try
            {
                // and finally add to our rule map
                loadRuleConfig( m_ruleMaps, rrkMap, fcfg );
            }
            catch( const std::exception &e )
            {
                return log<software_critical, -1>(
                    std::format( "Rule config exception caught from {}:\n{}", cnf, e.what() ) );
            }
        }

        try
        {
            finalizeRuleValRules( m_ruleMaps, rrkMap );
        }
        catch( const std::exception &e )
        {
            return log<software_critical, -1>( std::format( "Error finalizing rules:\n{}", e.what() ) );
        }
    }

    return 0;
}

void stateRuleEngine::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_critical>( "error in configuration" );
        m_shutdown = true;
    }
}

int stateRuleEngine::appStartup()
{
    for( auto it = m_ruleMaps.rules.begin(); it != m_ruleMaps.rules.end(); ++it )
    {
        if( it->second->priority() == rulePriority::info )
        {
            if( m_indiP_info.getDevice() != m_configName )
            {
                if( registerIndiPropertyNew( m_indiP_info,
                                             "info",
                                             pcf::IndiProperty::Switch,
                                             pcf::IndiProperty::ReadOnly,
                                             pcf::IndiProperty::Idle,
                                             pcf::IndiProperty::AnyOfMany,
                                             nullptr ) < 0 )
                {
                    return log<software_critical, -1>();
                }
            }

            m_indiP_info.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            m_indiP_info[it->first].setLabel( it->second->message() );
        }

        if( it->second->priority() == rulePriority::caution )
        {
            if( m_indiP_caution.getDevice() != m_configName )
            {
                if( registerIndiPropertyNew( m_indiP_caution,
                                             "caution",
                                             pcf::IndiProperty::Switch,
                                             pcf::IndiProperty::ReadOnly,
                                             pcf::IndiProperty::Idle,
                                             pcf::IndiProperty::AnyOfMany,
                                             nullptr ) < 0 )
                {
                    return log<software_critical, -1>( { __FILE__, __LINE__ } );
                }
            }

            m_indiP_caution.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            m_indiP_caution[it->first].setLabel( it->second->message() );
        }

        if( it->second->priority() == rulePriority::warning )
        {
            if( m_indiP_warning.getDevice() != m_configName )
            {
                if( registerIndiPropertyNew( m_indiP_warning,
                                             "warning",
                                             pcf::IndiProperty::Switch,
                                             pcf::IndiProperty::ReadOnly,
                                             pcf::IndiProperty::Idle,
                                             pcf::IndiProperty::AnyOfMany,
                                             nullptr ) < 0 )
                {
                    return log<software_critical, -1>( { __FILE__, __LINE__ } );
                }
            }

            m_indiP_warning.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            m_indiP_warning[it->first].setLabel( it->second->message() );
        }

        if( it->second->priority() == rulePriority::alert )
        {
            if( m_indiP_alert.getDevice() != m_configName )
            {
                if( registerIndiPropertyNew( m_indiP_alert,
                                             "alert",
                                             pcf::IndiProperty::Switch,
                                             pcf::IndiProperty::ReadOnly,
                                             pcf::IndiProperty::Idle,
                                             pcf::IndiProperty::AnyOfMany,
                                             nullptr ) < 0 )
                {
                    return log<software_critical, -1>( { __FILE__, __LINE__ } );
                }
            }

            m_indiP_alert.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            m_indiP_alert[it->first].setLabel( it->second->message() );
        }
    }

    for( auto it = m_ruleMaps.props.begin(); it != m_ruleMaps.props.end(); ++it )
    {
        if( it->second == nullptr )
            continue;

        std::string devName, propName;

        int rv = indi::parseIndiKey( devName, propName, it->first );
        if( rv != 0 )
        {
            log<software_error>( { __FILE__, __LINE__, 0, rv, "error parsing INDI key: " + it->first } );
            return -1;
        }

        registerIndiPropertySet( *it->second, devName, propName, st_newCallBack_ruleProp );
    }

    state( stateCodes::READY );

    return 0;
}

int stateRuleEngine::appLogic()
{
    for( auto it = m_ruleMaps.rules.begin(); it != m_ruleMaps.rules.end(); ++it )
    {
        if( it->second->priority() != rulePriority::none )
        {
            try
            {
                bool        val = it->second->value();
                std::string diagnostic;
                while( it->second->popRuntimeDiagnostic( diagnostic ) )
                {
                    log<software_error>( diagnostic );
                }

                pcf::IndiProperty *ruleState = ruleStateProperty( it->second->priority() );
                if( ruleState == nullptr )
                {
                    continue;
                }

                bool wasOn = ruleIsOn( *ruleState, it->first );

                pcf::IndiElement::SwitchStateType onoff = pcf::IndiElement::Off;

                if( val )
                {
                    onoff = pcf::IndiElement::On;
                }

                updateSwitchIfChanged( *ruleState, it->first, onoff );

                if( val && it->second->timeToSend() )
                {
                    std::string msg = notificationMessage(
                        it->first, *( it->second ), notificationLabel( it->second->priority() ), false, true );

                    it->second->incMessageCount();

                    if( sendNotification( msg ) < 0 )
                    {
                        return -1;
                    }
                }
                else if( !val )
                {
                    if( wasOn )
                    {
                        std::string msg = notificationMessage(
                            it->first, *( it->second ), notificationLabel( rulePriority::info ), true );

                        if( sendNotification( msg ) < 0 )
                        {
                            return -1;
                        }
                    }

                    it->second->messageCount( 0 ); // resets so that next time will get sent
                }
            }
            catch( const std::exception &e )
            {
                std::string diagnostic;
                while( it->second->popRuntimeDiagnostic( diagnostic ) )
                {
                    log<software_error>( diagnostic );
                }

                ///\todo how to handle startup vs misconfiguration

                /*
                if(it->second->priority() == rulePriority::none)
                {
                    updateSwitchIfChanged(m_indiP_info, it->first, pcf::IndiElement::Off);
                }*/
            }
        }
    }

    return 0;
}

int stateRuleEngine::appShutdown()
{
    return 0;
}

int stateRuleEngine::st_newCallBack_ruleProp( void *app, const pcf::IndiProperty &ipRecv )
{
    stateRuleEngine *sre = static_cast<stateRuleEngine *>( app );

    sre->newCallBack_ruleProp( ipRecv );

    return 0;
}

int stateRuleEngine::newCallBack_ruleProp( const pcf::IndiProperty &ipRecv )
{
    std::string key = ipRecv.createUniqueKey();

    if( m_ruleMaps.props.count( key ) == 0 )
    {
        return 0;
    }

    if( m_ruleMaps.props[key] == nullptr ) //
    {
        return 0;
    }

    *m_ruleMaps.props[key] = ipRecv;

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // stateRuleEngine_hpp
