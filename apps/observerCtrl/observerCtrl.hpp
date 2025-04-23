/** \file observerCtrl.hpp
 * \brief The MagAO-X Observer Controller header file
 *
 * \ingroup observerCtrl_files
 */

#ifndef observerCtrl_hpp
#define observerCtrl_hpp

#include <map>
#include <mx/math/geo.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup observerCtrl
 * \brief The MagAO-X Observer Controller application
 *
 * <a href="../handbook/operating/software/apps/observerCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup observerCtrl_files
 * \ingroup observerCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Observer Controller
/**
 * \ingroup observerCtrl
 */
class observerCtrl : public MagAOXApp<true>, public dev::telemeter<observerCtrl>
{

    // Give the test harness access.
    friend class observerCtrl_test;

    friend class dev::telemeter<observerCtrl>;

    typedef dev::telemeter<observerCtrl> telemeterT;

    typedef std::chrono::time_point<std::chrono::steady_clock> timePointT;
    typedef std::string                                        timeStampT;
    typedef std::chrono::duration<double>                      durationT;
    std::string timeStampAsISO8601( const std::chrono::time_point<std::chrono::system_clock> &tp );

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::vector<std::string> m_streamWriters; ///< The stream writers to stop and start

    std::string m_tcsDev{ "tcsi" };
    std::string m_catalogProp{ "catalog" };
    std::string m_objEl{ "object" };
    std::string m_catdataProp{ "catdata" };
    std::string m_raEl{ "ra" };
    std::string m_decEl{ "dec" };
    std::string m_labModeProp{ "labMode" };

    std::string m_teldataProp{ "teldata" };
    std::string m_parangEl{ "pa" };

    std::string m_loopDev{ "holoop" };
    std::string m_loopStateProp{ "loop_state" };

    ///@}

    /// The observer specification
    struct observer
    {
        std::string m_fullName;       ///< Obsever's full name
        std::string m_pfoa;           ///< Observer's preferred forma of address
        std::string m_pronunciation;  ///< Guide for the TTS to pronounced the pfoa (defaults to pfoa)
        std::string m_email;          ///< Observer's email.  Must be unique.
        std::string m_sanitizedEmail; ///< Observer's email sanitized for use in INDI properties
        std::string m_institution;    ///< The observer's institution
    };

    typedef std::map<std::string, observer> observerMapT;

    observerMapT m_observers; ///< The observers from the configuration file

    observer m_currentObserver; ///< The current selected observer

    observer m_currentOperator; ///< The current selected observer

    std::string m_obsName;          ///< The name of the observation.
    double      m_obsDuration{ 0 }; ///< The desired duration of the observation.  If 0 then until stopped.

    bool m_observing{ false }; ///< Flag indicating whether or not we are in an observation

    std::string m_target;

    std::string m_catObj;

    std::string m_catRA;
    std::string m_catDec;

    bool m_loop{ false }; ///< Flag tracking loop state.  true is loop closed.

    bool m_labMode{ false }; ///< Flag tracking whether the TCS interface is in lab mode.

    bool m_newTargetBlock{ true }; /**< Flag to indicate that this is a new target block.  This starts out as true
                                        but becomes false on the first observation.  Then becomes true when the
                                        loop closes for the first time after a target change. */
    bool m_newTarget{ false }; /**< Flag to track when the target changes.  Occurs either automatically on a TCS update
                                    or on a user override.*/

    /// The start time of the current observation
    timePointT m_obsStartTime;

    /// The UTC start time of the current observation
    timeStampT m_obsStartTimeStamp;

    /// The parallactic angle at the start of the observation
    double m_obsStartParang{ 0 };

    /// The start time of the current target
    timePointT m_tgtStartTime;

    /// The UTC start time of the current target
    timeStampT m_tgtStartTimeStamp;

    /// The parallactic angle at the start of observing the current target
    double m_tgtStartParang{ 0 };

    /// The current parallactic angle
    double m_parang;

    /// The current target time.  Only updated while observing.
    durationT m_tgtTime;

    /// The current target angle. Only updated while observing.
    double m_tgtAng{ 0 };

  public:
    /// Default c'tor.
    observerCtrl();

    /// D'tor, declared and defined for noexcept.
    ~observerCtrl() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                                                                    from which to load values*/
    );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for observerCtrl.
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

    void startObserving();

    void stopObserving();

    ///\name INDI
    /** @{
     */
  protected:
    pcf::IndiProperty m_indiP_observers; ///< Selection switch to allow selection of the observer
    pcf::IndiProperty m_indiP_observer;  ///< Text which contains the specifications of the current observer

    pcf::IndiProperty m_indiP_operators; ///< Selection switch to allow selection of the observer
    pcf::IndiProperty m_indiP_operator;  ///< Text which contains the specifications of the current observer

    pcf::IndiProperty m_indiP_obsName;     /**< The current observation name, used to specify the
                                                purpose of the observation*/
    pcf::IndiProperty m_indiP_observing;   ///< Toggle switch to trigger observation
    pcf::IndiProperty m_indiP_obsDuration; ///< Number to set the desired duration of observation
    pcf::IndiProperty m_indiP_obsStart;    ///< String timestamp indicating the start for target/observation
    pcf::IndiProperty m_indiP_obsTime;     ///< Number tracking the elapsed time
    pcf::IndiProperty m_indiP_obsAngle;    ///< Number tracking the change in angle
    pcf::IndiProperty m_indiP_sws;         ///< Selection to switch which stream writers are enabled
    pcf::IndiProperty m_indiP_userlog;     ///< Text to enter a user log

    pcf::IndiProperty m_indiP_resetTarget; ///< Reset the target statistics

    pcf::IndiProperty m_indiP_target; ///< The target name, which can be overridden by the user

    pcf::IndiProperty m_indiP_tcsTarget; ///< Set the target to match TCS catObj

    pcf::IndiProperty m_indiP_catalog; ///< Catalog text data
    pcf::IndiProperty m_indiP_catdata; ///< Catalog numeric data
    pcf::IndiProperty m_indiP_teldata; ///< Telescope data (for parang)

    pcf::IndiProperty m_indiP_labMode; ///< Tracks whether TCS is in lab mode.

    pcf::IndiProperty m_indiP_loop; ///< Tracks the loop state

  public:
    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_observers );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_operators );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_obsName );
    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_observing );
    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_obsDuration );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_sws );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_userlog );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_resetTarget );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_target );

    INDI_NEWCALLBACK_DECL( observerCtrl, m_indiP_tcsTarget );

    INDI_SETCALLBACK_DECL( observerCtrl, m_indiP_catalog );

    INDI_SETCALLBACK_DECL( observerCtrl, m_indiP_catdata );

    INDI_SETCALLBACK_DECL( observerCtrl, m_indiP_teldata );

    INDI_SETCALLBACK_DECL( observerCtrl, m_indiP_labMode );

    INDI_SETCALLBACK_DECL( observerCtrl, m_indiP_loop );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_observer * );

    int recordObserver( bool force = false );

    ///@}
};

observerCtrl::observerCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void observerCtrl::setupConfig()
{
    config.add( "stream.writers",
                "",
                "stream.writers",
                argType::Required,
                "stream",
                "writers",
                false,
                "string",
                "The device names of the stream writers to control." );

    dev::telemeter<observerCtrl>::setupConfig( config );
}

int observerCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_streamWriters, "stream.writers" );

    std::vector<std::string> sections;

    _config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        log<text_log>( "no observers found in config", logPrio::LOG_CRITICAL );
        return -1;
    }

    for( size_t i = 0; i < sections.size(); ++i )
    {
        bool pfoaSet = _config.isSetUnused( mx::app::iniFile::makeKey( sections[i], "pfoa" ) );
        if( !pfoaSet )
            continue;

        std::string email = sections[i];

        std::string pfoa;
        _config.configUnused( pfoa, mx::app::iniFile::makeKey( sections[i], "pfoa" ) );

        std::string pronunciation;
        _config.configUnused( pronunciation, mx::app::iniFile::makeKey( sections[i], "pronunciation" ) );

        if( pronunciation == "" )
        {
            pronunciation = pfoa;
        }

        std::string fullName;
        _config.configUnused( fullName, mx::app::iniFile::makeKey( sections[i], "full_name" ) );

        std::string institution;
        _config.configUnused( institution, mx::app::iniFile::makeKey( sections[i], "institution" ) );

        std::string sanitizedEmail = "";
        for( size_t n = 0; n < email.size(); ++n )
        {
            if( email[n] == '@' )
            {
                sanitizedEmail = sanitizedEmail + "-at-";
            }
            else if( email[n] == '.' )
            {
                sanitizedEmail = sanitizedEmail + "-dot-";
            }
            else
            {
                sanitizedEmail.push_back( email[n] );
            }
        }
        m_observers[email] = observer( { fullName, pfoa, pronunciation, email, sanitizedEmail, institution } );
    }

    return 0;
}

void observerCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
        return;
    }

    if( m_observers.size() < 1 )
    {
        log<text_log>( "no observers found in config", logPrio::LOG_CRITICAL );
        m_shutdown = 1;
        return;
    }

    dev::telemeter<observerCtrl>::loadConfig( config );
}

int observerCtrl::appStartup()
{
    std::vector<std::string> sanitizedEmails;
    std::vector<std::string> emails;
    for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
    {
        sanitizedEmails.push_back( it->second.m_sanitizedEmail );
        emails.push_back( it->second.m_email );
    }

    if( createStandardIndiSelectionSw( m_indiP_observers, "observers", sanitizedEmails, emails ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    // Set to default user of jared
    ///\todo do something else. maybe a default user is specified in the config?
    for( auto &it : m_observers )
    {
        if( it.first.find( "jrmales" ) != std::string::npos )
        {
            m_indiP_observers[it.second.m_sanitizedEmail].setSwitchState( pcf::IndiElement::On );
            m_currentObserver = it.second;
        }
        else
        {
            m_indiP_observers[it.second.m_sanitizedEmail].setSwitchState( pcf::IndiElement::Off );
        }
    }

    REG_INDI_NEWPROP_NOSETUP( m_indiP_observers );

    if( createStandardIndiSelectionSw( m_indiP_operators, "operators", sanitizedEmails, emails ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    // Set to default user of jared
    ///\todo do something else. maybe a default user is specified in the config?
    for( auto &it : m_observers )
    {
        if( it.first.find( "jrmales" ) != std::string::npos )
        {
            m_indiP_operators[it.second.m_sanitizedEmail].setSwitchState( pcf::IndiElement::On );
            m_currentOperator = it.second;
        }
        else
        {
            m_indiP_operators[it.second.m_sanitizedEmail].setSwitchState( pcf::IndiElement::Off );
        }
    }

    REG_INDI_NEWPROP_NOSETUP( m_indiP_operators );

    CREATE_REG_INDI_NEW_TEXT( m_indiP_obsName, "obs_name", "Observation Name", "Observer" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_observing, "obs_on" );

    CREATE_REG_INDI_NEW_NUMBERD( m_indiP_obsDuration, "obs_duration", 0, 300, 0.1, "%0.1f", "Duration", "Observer" );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_obsTime, "obs_time", "Observation Time", "Observer" );
    indi::addNumberElement<double>( m_indiP_obsTime, "observation", 0, 14400, 0.1, "%0.1f", "Current Obs" );
    indi::addNumberElement<double>( m_indiP_obsTime, "target", 0, 14400, 0.1, "%0.1f", "Target" );

    REG_INDI_NEWPROP_NOCB( m_indiP_obsStart, "obs_start", pcf::IndiProperty::Text );
    indi::addTextElement( m_indiP_obsStart, "observation" );
    indi::addTextElement( m_indiP_obsStart, "target" );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_obsAngle, "obs_delta_parang", "Change in Par. Ang.", "Observer" );
    indi::addNumberElement<double>( m_indiP_obsAngle, "observation", 0, 360, 0.1, "%0.1f", "Current Obs" );
    indi::addNumberElement<double>( m_indiP_obsAngle, "target", 0, 360, 0.1, "%0.1f", "Target" );

    REG_INDI_NEWPROP_NOCB( m_indiP_observer, "current_observer", pcf::IndiProperty::Text );
    indi::addTextElement( m_indiP_observer, "full_name" );
    indi::addTextElement( m_indiP_observer, "email" );
    indi::addTextElement( m_indiP_observer, "pfoa" );
    indi::addTextElement( m_indiP_observer, "pronunciation" );
    indi::addTextElement( m_indiP_observer, "institution" );

    REG_INDI_NEWPROP_NOCB( m_indiP_operator, "current_operator", pcf::IndiProperty::Text );
    indi::addTextElement( m_indiP_operator, "full_name" );
    indi::addTextElement( m_indiP_operator, "email" );
    indi::addTextElement( m_indiP_operator, "pfoa" );
    indi::addTextElement( m_indiP_operator, "pronunciation" );
    indi::addTextElement( m_indiP_operator, "institution" );

    m_indiP_sws = pcf::IndiProperty( pcf::IndiProperty::Switch );
    m_indiP_sws.setDevice( configName() );
    m_indiP_sws.setName( "writers" );
    m_indiP_sws.setPerm( pcf::IndiProperty::ReadWrite );
    m_indiP_sws.setState( pcf::IndiProperty::Idle );
    m_indiP_sws.setRule( pcf::IndiProperty::AnyOfMany );

    for( size_t n = 0; n < m_streamWriters.size(); ++n )
    {
        m_indiP_sws.add( pcf::IndiElement( m_streamWriters[n], pcf::IndiElement::Off ) );
    }

    REG_INDI_NEWPROP_NOSETUP( m_indiP_sws );

    m_indiP_userlog = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_userlog.setDevice( configName() );
    m_indiP_userlog.setName( "user_log" );
    m_indiP_userlog.setPerm( pcf::IndiProperty::ReadWrite );
    m_indiP_userlog.setState( pcf::IndiProperty::Idle );
    m_indiP_userlog.add( pcf::IndiElement( "email" ) );
    m_indiP_userlog.add( pcf::IndiElement( "message" ) );
    m_indiP_userlog.add( pcf::IndiElement( "time_s" ) );
    m_indiP_userlog.add( pcf::IndiElement( "time_ns" ) );

    REG_INDI_NEWPROP_NOSETUP( m_indiP_userlog );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_resetTarget, "target_reset" );

    CREATE_REG_INDI_NEW_TEXT( m_indiP_target, "target", "Target", "Observer" );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_tcsTarget, "target_load_from_tcs" );

    REG_INDI_SETPROP( m_indiP_catalog, m_tcsDev, m_catalogProp );
    REG_INDI_SETPROP( m_indiP_catdata, m_tcsDev, m_catdataProp );
    REG_INDI_SETPROP( m_indiP_teldata, m_tcsDev, m_teldataProp );
    REG_INDI_SETPROP( m_indiP_labMode, m_tcsDev, m_labModeProp );

    REG_INDI_SETPROP( m_indiP_loop, m_loopDev, m_loopStateProp );

    TELEMETER_APP_STARTUP;

    state( stateCodes::READY );
    return 0;
}

int observerCtrl::appLogic()
{

    std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

    if( lock.owns_lock() )
    {
        updatesIfChanged<std::string>( m_indiP_observer,
                                       { "full_name", "email", "pfoa", "pronunciation", "institution" },
                                       { m_currentObserver.m_fullName,
                                         m_currentObserver.m_email,
                                         m_currentObserver.m_pfoa,
                                         m_currentObserver.m_pronunciation,
                                         m_currentObserver.m_institution } );

        for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
        {
            if( it->first == m_currentObserver.m_email )
            {
                updateSwitchIfChanged(
                    m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE );
            }
            else
            {
                updateSwitchIfChanged(
                    m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE );
            }
        }

        updatesIfChanged<std::string>( m_indiP_operator,
                                       { "full_name", "email", "pfoa", "pronunciation", "institution" },
                                       { m_currentOperator.m_fullName,
                                         m_currentOperator.m_email,
                                         m_currentOperator.m_pfoa,
                                         m_currentOperator.m_pronunciation,
                                         m_currentOperator.m_institution } );

        for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
        {
            if( it->first == m_currentOperator.m_email )
            {
                updateSwitchIfChanged(
                    m_indiP_operators, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE );
            }
            else
            {
                updateSwitchIfChanged(
                    m_indiP_operators, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE );
            }
        }

        updatesIfChanged<std::string>( m_indiP_obsName, { "current", "target" }, { m_obsName, m_obsName } );

        updatesIfChanged<double>( m_indiP_obsDuration, { "current", "target" }, { m_obsDuration, m_obsDuration } );

        if( m_observing )
        {
            timePointT                          ct      = std::chrono::steady_clock::now();
            const std::chrono::duration<double> obstime = ct - m_obsStartTime;
            m_tgtTime                                   = ct - m_tgtStartTime;

            double obsang = mx::math::angleDiff<mx::math::degreesT<double>>( m_parang, m_obsStartParang );
            m_tgtAng      = mx::math::angleDiff<mx::math::degreesT<double>>( m_parang, m_tgtStartParang );

            updateSwitchIfChanged( m_indiP_observing, "toggle", pcf::IndiElement::On, INDI_OK );
            updatesIfChanged<double>(
                m_indiP_obsTime, { "observation", "target" }, { obstime.count(), m_tgtTime.count() } );

            updatesIfChanged<double>( m_indiP_obsAngle, { "observation", "target" }, { obsang, m_tgtAng } );
            updatesIfChanged<std::string>(
                m_indiP_obsStart, { "observation", "target" }, { m_obsStartTimeStamp, m_tgtStartTimeStamp } );

            if( m_obsDuration > 0.0 && obstime.count() > m_obsDuration )
            {
                stopObserving();
            }
        }
        else
        {
            updateSwitchIfChanged( m_indiP_observing, "toggle", pcf::IndiElement::Off, INDI_IDLE );
            updatesIfChanged<std::string>( m_indiP_obsStart, { "observation", "target" }, { "", m_tgtStartTimeStamp } );
            updatesIfChanged<double>( m_indiP_obsTime, { "observation", "target" }, { 0.0, m_tgtTime.count() } );
            updatesIfChanged<double>( m_indiP_obsAngle, { "observation", "target" }, { 0.0, m_tgtAng } );
        }
    }

    TELEMETER_APP_LOGIC;

    return 0;
}

int observerCtrl::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

void observerCtrl::startObserving()
{
    for( size_t n = 0; n < m_streamWriters.size(); ++n )
    {
        if( m_indiP_sws[m_streamWriters[n]].getSwitchState() == pcf::IndiElement::On )
        {
            pcf::IndiProperty ip( pcf::IndiProperty::Switch );

            ip.setDevice( m_streamWriters[n] + "-sw" );
            ip.setName( "writing" );
            ip.add( pcf::IndiElement( "toggle" ) );
            ip["toggle"].setSwitchState( pcf::IndiElement::On );

            sendNewProperty( ip );
        }
    }

    mx::sys::sleep( 1 );

    m_obsStartTime      = std::chrono::steady_clock::now();
    m_obsStartTimeStamp = timeStampAsISO8601( std::chrono::system_clock::now() );
    m_obsStartParang    = m_parang;

    if( m_newTargetBlock || m_labMode ) // We always reset if in lab mode
    {
        m_tgtStartTime      = m_obsStartTime;
        m_tgtStartTimeStamp = m_obsStartTimeStamp;
        m_tgtStartParang    = m_obsStartParang;

        m_newTargetBlock = false;
    }

    m_observing = true;
    recordObserver( true );
}

void observerCtrl::stopObserving()
{
    m_observing = false;
    recordObserver( true );

    for( size_t n = 0; n < m_streamWriters.size(); ++n )
    {
        if( m_indiP_sws[m_streamWriters[n]].getSwitchState() == pcf::IndiElement::On )
        {
            pcf::IndiProperty ip( pcf::IndiProperty::Switch );

            ip.setDevice( m_streamWriters[n] + "-sw" );
            ip.setName( "writing" );
            ip.add( pcf::IndiElement( "toggle" ) );
            ip["toggle"].setSwitchState( pcf::IndiElement::Off );

            sendNewProperty( ip );
        }
    }
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_observers )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_observers, ipRecv );

    // look for selected mode switch which matches a known mode.  Make sure only one is selected.
    std::string newEmail = "";
    for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
    {
        if( !ipRecv.find( it->second.m_sanitizedEmail ) )
            continue;

        if( ipRecv[it->second.m_sanitizedEmail].getSwitchState() == pcf::IndiElement::On )
        {
            if( newEmail != "" )
            {
                log<text_log>( "More than one observer selected", logPrio::LOG_ERROR );
                return -1;
            }

            newEmail = it->first;
        }
    }

    if( newEmail == "" )
    {
        std::cerr << "nothing\n";
        return 0;
    }

    {
        std::unique_lock<std::mutex> lock( m_indiMutex );

        m_currentObserver = m_observers[newEmail];

        for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
        {
            if( it->first == m_currentObserver.m_sanitizedEmail )
            {
                updateSwitchIfChanged(
                    m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE );
            }
            else
            {
                updateSwitchIfChanged(
                    m_indiP_observers, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE );
            }
        }
    }

    log<logger::observer>( { m_currentObserver.m_fullName,
                             m_currentObserver.m_pfoa,
                             m_currentObserver.m_email,
                             m_currentObserver.m_institution } );

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_operators )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_operators, ipRecv );

    // look for selected mode switch which matches a known mode.  Make sure only one is selected.
    std::string newEmail = "";
    for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
    {
        if( !ipRecv.find( it->second.m_sanitizedEmail ) )
            continue;

        if( ipRecv[it->second.m_sanitizedEmail].getSwitchState() == pcf::IndiElement::On )
        {
            if( newEmail != "" )
            {
                log<text_log>( "More than one operator selected", logPrio::LOG_ERROR );
                return -1;
            }

            newEmail = it->first;
        }
    }

    if( newEmail == "" )
    {
        std::cerr << "nothing\n";
        return 0;
    }

    {
        std::unique_lock<std::mutex> lock( m_indiMutex );

        m_currentOperator = m_observers[newEmail];

        for( auto it = m_observers.begin(); it != m_observers.end(); ++it )
        {
            if( it->first == m_currentOperator.m_sanitizedEmail )
            {
                updateSwitchIfChanged(
                    m_indiP_operators, it->second.m_sanitizedEmail, pcf::IndiElement::On, INDI_IDLE );
            }
            else
            {
                updateSwitchIfChanged(
                    m_indiP_operators, it->second.m_sanitizedEmail, pcf::IndiElement::Off, INDI_IDLE );
            }
        }
    }

    log<logger::ao_operator>( { m_currentOperator.m_fullName,
                                m_currentOperator.m_pfoa,
                                m_currentOperator.m_email,
                                m_currentOperator.m_institution } );

    return 0;
}

std::string observerCtrl::timeStampAsISO8601( const std::chrono::time_point<std::chrono::system_clock> &tp )
{
    // Convert to time_t for whole seconds
    std::time_t t  = std::chrono::system_clock::to_time_t( tp );
    std::tm     tm = *std::gmtime( &t );

    // Get nanoseconds
    auto duration = tp.time_since_epoch();
    auto seconds  = std::chrono::duration_cast<std::chrono::seconds>( duration );
    auto nanos    = std::chrono::duration_cast<std::chrono::nanoseconds>( duration - seconds ).count();

    // Format ISO 8601 with nanoseconds
    std::stringstream ss;
    ss << std::put_time( &tm, "%Y-%m-%dT%H:%M:%S" );
    ss << '.' << std::setw( 9 ) << std::setfill( '0' ) << nanos << "Z"; // Z for UTC

    return ss.str();
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_obsName )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_obsName, ipRecv );

    std::string target;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( indiTargetUpdate( m_indiP_obsName, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_obsName = target;

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_observing )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_observing, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    recordObserver( true );
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex );
        startObserving();
        updateSwitchIfChanged( m_indiP_observing, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        std::unique_lock<std::mutex> lock( m_indiMutex );
        stopObserving();
        updateSwitchIfChanged( m_indiP_observing, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_obsDuration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_obsDuration, ipRecv );

    return indiTargetUpdate( m_indiP_obsDuration, m_obsDuration, ipRecv, false );
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_sws )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sws, ipRecv );

    if( m_observing == true )
    {
        log<text_log>( { "Can't change stream writers while observing" }, logPrio::LOG_WARNING );
        return 0;
    }

    for( size_t n = 0; n < m_streamWriters.size(); ++n )
    {
        if( !ipRecv.find( m_streamWriters[n] ) )
        {
            continue;
        }

        std::unique_lock<std::mutex> lock( m_indiMutex );
        updateSwitchIfChanged( m_indiP_sws, m_streamWriters[n], ipRecv[m_streamWriters[n]].getSwitchState() );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_userlog )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_userlog, ipRecv );

    std::string email;
    std::string message;

    timespecX ts{};

    if( ipRecv.find( "email" ) )
    {
        email = ipRecv["email"].get();
    }

    if( ipRecv.find( "message" ) )
    {
        message = ipRecv["message"].get();
    }

    if( message == "" )
    {
        return 0;
    }

    if( email == "" )
    {
        email = m_currentObserver.m_email;
    }

    if( ipRecv.find( "time_s" ) )
    {
        ts.time_s = ipRecv["time_s"].get<flatlogs::secT>();
    }

    if( ipRecv.find( "time_ns" ) )
    {
        ts.time_ns = ipRecv["time_ns"].get<flatlogs::nanosecT>();
    }

    if( ts.time_s != 0 )
    {
        m_log.template log<user_log>( ts, { email, message }, logPrio::LOG_INFO );
    }
    else
    {
        log<user_log>( { email, message } );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_resetTarget )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_resetTarget, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if( m_observing )
        {
            m_tgtStartTime   = m_obsStartTime;
            m_tgtStartParang = m_obsStartParang;
        }
        else
        {
            m_newTargetBlock = true;
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_target )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_target, ipRecv );

    std::string target;
    if( indiTargetUpdate( m_indiP_target, target, ipRecv ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( target != m_target )
    {
        if( m_observing )
        {
            m_tgtStartTime   = m_obsStartTime;
            m_tgtStartParang = m_obsStartParang;
        }
        else
        {
            m_newTargetBlock = true;
        }

        m_target    = target;
        m_newTarget = true;

        log<text_log>( "Target updated by observer to " + m_target, logPrio::LOG_NOTICE );

        std::unique_lock<std::mutex> lock( m_indiMutex );
        updatesIfChanged<std::string>( m_indiP_target, { "current", "target" }, { m_target, m_target } );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( observerCtrl, m_indiP_tcsTarget )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tcsTarget, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if( m_target != m_catObj )
        {
            if( m_observing )
            {
                m_tgtStartTime   = m_obsStartTime;
                m_tgtStartParang = m_obsStartParang;
            }
            else
            {
                m_newTargetBlock = true;
            }

            m_target    = m_catObj;
            m_newTarget = true;

            log<text_log>( "Target updated by observer to TCS target: " + m_target, logPrio::LOG_NOTICE );
            updatesIfChanged<std::string>( m_indiP_target, { "current", "target" }, { m_target, m_target } );
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( observerCtrl, m_indiP_catalog )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_catalog, ipRecv );

    if( !ipRecv.find( "object" ) )
    {
        return -1;
    }

    std::string object = ipRecv["object"].get();

    if( object != m_catObj )
    {
        m_catObj    = object;
        //m_target    = object;
        //m_newTarget = true;

        // Always log change in name (different from RA and DEC)
        log<text_log>( "TCS target updated to " + m_catObj, logPrio::LOG_NOTICE );

        //std::unique_lock<std::mutex> lock( m_indiMutex );
        //updatesIfChanged<std::string>( m_indiP_target, { "current", "target" }, { m_target, m_target } );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( observerCtrl, m_indiP_catdata )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_catdata, ipRecv );

    bool change = false;
    if( ipRecv.find( "ra" ) )
    {
        std::string ra = ipRecv["ra"].get();

        if( ra != m_catRA )
        {
            m_catRA  = ra;
            //m_target = m_catObj;

            if( !m_newTarget ) // Only log if not already new
            {
                change      = true;
                m_newTarget = true;
            }

            /*std::unique_lock<std::mutex> lock( m_indiMutex );
            updatesIfChanged<std::string>( m_indiP_target, { "current", "target" }, { m_target, m_target } );*/
        }
    }

    if( ipRecv.find( "dec" ) )
    {
        std::string dec = ipRecv["dec"].get();

        if( dec != m_catDec )
        {
            m_catDec = dec;
            //m_target = m_catObj;

            if( !m_newTarget ) // Only log if not already new
            {
                change      = true;
                m_newTarget = true;
            }

            /*std::unique_lock<std::mutex> lock( m_indiMutex );
            updatesIfChanged<std::string>( m_indiP_target, { "current", "target" }, { m_target, m_target } );*/
        }
    }

    if( change ) // Only log if not already new
    {
        log<text_log>( "Pointing change.  Probable target change.", logPrio::LOG_NOTICE );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( observerCtrl, m_indiP_teldata )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_teldata, ipRecv );

    if( ipRecv.find( "pa" ) )
    {
        m_parang = ipRecv["pa"].get<double>();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( observerCtrl, m_indiP_labMode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_labMode, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {
            m_labMode = true;
        }
        else
        {
            m_labMode        = false;
        }
    }

    std::cerr << "got labmode: " << m_labMode << '\n';
    return 0;
}

INDI_SETCALLBACK_DEFN( observerCtrl, m_indiP_loop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loop, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {
            //Now that we're manually synch-ing the target name, we don't need loop closed logic
            //i.e. the target block starts when the observation starts after setting the name.
            /*if( m_newTarget == true && !m_loop )
            {
                m_newTargetBlock = true;
                m_newTarget      = false;
            }*/

            m_loop = true;
        }
        else
        {
            m_loop = false;
        }
    }

    return 0;
}

inline int observerCtrl::checkRecordTimes()
{
    return telemeter<observerCtrl>::checkRecordTimes( telem_observer() );
}

inline int observerCtrl::recordTelem( const telem_observer * )
{
    return recordObserver( true );
}

inline int observerCtrl::recordObserver( bool force )
{
    static std::string last_email;
    static std::string last_obsName;
    static bool        last_observing;
    static std::string last_target;
    static std::string last_operator;

    if( last_email != m_currentObserver.m_email || last_obsName != m_obsName || last_observing != m_observing ||
        last_target != m_target || last_operator != m_currentOperator.m_email || force )
    {
        telem<telem_observer>(
            { m_currentObserver.m_email, m_obsName, m_observing, m_target, m_currentOperator.m_email } );

        last_email     = m_currentObserver.m_email;
        last_obsName   = m_obsName;
        last_observing = m_observing;
        last_target    = m_target;
        last_operator  = m_currentOperator.m_email;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // observerCtrl_hpp
