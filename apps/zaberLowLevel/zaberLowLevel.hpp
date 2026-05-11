/** \file zaberLowLevel.hpp
 * \brief The MagAO-X Low-Level Zaber Controller
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevel_files
 */

#ifndef zaberLowLevel_hpp
#define zaberLowLevel_hpp

#include <iostream>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; // This needs to be before zaberStage.hpp for logging to work.

#include "zaberUtils.hpp"
#include "zaberStage.hpp"
#include "za_serial.h"

#define ZC_CONNECTED ( 0 )
#define ZC_ERROR ( -1 )
#define ZC_NOT_CONNECTED ( 10 )

/** \defgroup zaberLowLevel low-level zaber controller
 * \brief The low-level interface to a set of chained Zaber stages
 *
 * <a href="../handbook/operating/software/apps/zaberLowLevel.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup zaberLowLevel_files zaber low-level files
 * \ingroup zaberLowLevel
 */

namespace MagAOX
{
namespace app
{

/// The low-level ASCII-protocol Zaber controller.
/**
 * This app manages a daisy-chained ASCII Zaber bus and keeps its discovery,
 * recovery, and INDI reporting behavior aligned with the binary-protocol app.
 *
 * \ingroup zaberLowLevel
 */
class zaberLowLevel : public MagAOXAppT, public tty::usbDevice
{

    // Give the test harness access.
    friend class zaberLowLevel_test;

  protected:
    /** \name Stage Mapping - Data
     *
     * @{
     */
    /// Number of configured stages.
    int m_numStages{ 0 };

    /// Connected ASCII protocol port.
    z_port m_port{ 0 };

    /// Stage helpers in configuration order.
    std::vector<zaberStage<zaberLowLevel>> m_stages;

    /// Map from ASCII device address to configured stage index.
    std::unordered_map<int, size_t> m_stageAddress;

    /// Map from configured serial number to configured stage index.
    std::unordered_map<std::string, size_t> m_stageSerial;

    /// Map from configured stage name to configured stage index.
    std::unordered_map<std::string, size_t> m_stageName;

    /// Whether the active connection has completed an initial discovery pass.
    bool m_stageDiscoveryInitialized{ false };
    ///@}

  public:
    /// Default constructor.
    zaberLowLevel();

    /// Destructor, declared and defined for noexcept.
    ~zaberLowLevel() noexcept
    {
    }

    /// Set up application configuration.
    virtual void setupConfig();

    /// Load application configuration.
    virtual void loadConfig();

    /// Connect to the ASCII-protocol stage chain and discover configured devices.
    int connect();

    /// Apply a parsed `system.serial` snapshot to the configured stages.
    int loadStages( std::string &serialRes /**< [in] the raw response to `/ get system.serial` */ );

    /// Refresh discovery on an already-connected ASCII bus.
    int refreshStageDiscovery();

    /// Reset the active ASCII connection bookkeeping.
    int resetConnection();

    /// Recover from an ASCII-transport error without terminating the app.
    int recoverFromError( bool devicePresent /**< [in] True if the USB tty still exists in udev. */ );

    /// Set up the INDI properties and restore retained stage state.
    virtual int appStartup();

    /// Execute the main FSM for `zaberLowLevel`.
    virtual int appLogic();

    /// Handle the transition into the powered-off state.
    virtual int onPowerOff();

    /// Execute the powered-off loop.
    virtual int whilePowerOff();

    /// Perform any shutdown tasks before exit.
    virtual int appShutdown();

  protected:
    /** \name INDI Stage State - Data
     *
     * @{
     */
    /// Current state of the stage.
    pcf::IndiProperty m_indiP_curr_state;

    /// Maximum raw position of the stage.
    pcf::IndiProperty m_indiP_max_pos;

    /// Parked state of the stage.
    pcf::IndiProperty m_indiP_parked;

    /// Time of last homing for the stage.
    pcf::IndiProperty m_indiP_lastHomed;

    /// Current raw position of the stage.
    pcf::IndiProperty m_indiP_curr_pos;

    /// Current temperature of the stage.
    pcf::IndiProperty m_indiP_temp;

    /// Whether the stage has existing warnings.
    pcf::IndiProperty m_indiP_warn;

    /// Target raw position of the stage.
    pcf::IndiProperty m_indiP_tgt_pos;

    /// Command a stage to home.
    pcf::IndiProperty m_indiP_req_home;

    /// Command all stages to home.
    pcf::IndiProperty m_indiP_req_home_all;

    /// Command a stage to safely halt.
    pcf::IndiProperty m_indiP_req_halt;

    /// Command a stage to safely immediately halt.
    pcf::IndiProperty m_indiP_req_ehalt;

    /// Enable or disable a stage's potentiometer.
    pcf::IndiProperty m_indiP_knob_enable;

    /// Enable or disable a stage's LED.
    pcf::IndiProperty m_indiP_led_enable;
    ///@}

  public:
    /** \name INDI Stage State
     *
     * @{
     */
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_tgt_pos );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_req_home );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_req_home_all );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_req_halt );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_req_ehalt );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_knob_enable );
    INDI_NEWCALLBACK_DECL( zaberLowLevel, m_indiP_led_enable );
    ///@}
};

zaberLowLevel::zaberLowLevel() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;

    return;
}

void zaberLowLevel::setupConfig()
{
    tty::usbDevice::setupConfig( config );
}

void zaberLowLevel::loadConfig()
{

    this->m_baudRate = B115200; // default for Zaber stages.  Will be overridden by any config setting.

    int rv = tty::usbDevice::loadConfig( config );

    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND ) // Ignore error if not plugged in
    {
        log<software_error>( { rv, tty::ttyErrorString( rv ) } );
    }

    std::vector<std::string> sections;

    config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        log<software_error>( { "No stages found" } );
        return;
    }

    for( size_t n = 0; n < sections.size(); ++n )
    {
        if( config.isSetUnused( mx::app::iniFile::makeKey( sections[n], "serial" ) ) )
        {
            m_stages.push_back( zaberStage<zaberLowLevel>( this ) );

            size_t idx = m_stages.size() - 1;

            m_stages[idx].name( sections[n] );

            // Get serial number from config.
            std::string tmp = m_stages[idx].serial(); // get default
            config.configUnused( tmp, mx::app::iniFile::makeKey( sections[n], "serial" ) );
            m_stages[idx].serial( tmp );

            m_stageName.insert( { m_stages[idx].name(), idx } );
            m_stageSerial.insert( { m_stages[idx].serial(), idx } );
        }
    }
}

int zaberLowLevel::connect()
{
    if( m_port > 0 )
    {
        int rv = za_disconnect( m_port );
        if( rv < 0 )
        {
            log<text_log>( "Error disconnecting from zaber system.", logPrio::LOG_ERROR );
        }
        m_port = 0;
    }

    if( m_port <= 0 )
    {

        int zrv;

        { // scope for elPriv
            elevatedPrivileges elPriv( this );
            zrv = za_connect( &m_port, m_deviceName.c_str() );
        }

        if( zrv != Z_SUCCESS )
        {
            if( m_port > 0 )
            {
                za_disconnect( m_port );
                m_port = 0;
            }

            if( !stateLogged() )
            {
                log<software_error>( { "can not connect to zaber stage(s)" } );
            }

            return ZC_NOT_CONNECTED; // We aren't connected.
        }
    }

    if( m_port <= 0 )
    {
        // state(stateCodes::ERROR); //Should not get this here.  Probably means no device.
        log<text_log>( "can not connect to zaber stage(s): no port", logPrio::LOG_WARNING );
        return ZC_NOT_CONNECTED; // We aren't connected.
    }

    int rv = za_drain( m_port );

    if( rv != Z_SUCCESS )
    {
        log<software_error>( { rv, "error from za_drain" } );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    char buffer[256];

    //===== First renumber so they are unique.
    std::string renum = "/ renumber";
    int         nwr   = za_send( m_port, renum.c_str(), renum.size() );

    if( nwr == Z_ERROR_SYSTEM_ERROR )
    {
        log<text_log>( "Error sending renumber query to stages", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    //===== Drain the result
    rv = za_drain( m_port );

    if( rv != Z_SUCCESS )
    {
        log<software_error>( { rv, "error from za_drain" } );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    //======= Now find the stages
    std::string gss = "/ get system.serial";
    nwr             = za_send( m_port, gss.c_str(), gss.size() );

    if( nwr == Z_ERROR_SYSTEM_ERROR )
    {
        log<text_log>( "Error sending system.serial query to stages", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    std::string serialRes;
    while( 1 )
    {
        int nrd = za_receive( m_port, buffer, sizeof( buffer ) );
        if( nrd >= 0 )
        {
            buffer[nrd] = '\0';
            log<text_log>( std::string( "Received: " ) + buffer, logPrio::LOG_DEBUG );
            serialRes += buffer;
        }
        else if( nrd != Z_ERROR_TIMEOUT )
        {
            log<text_log>( "Error receiving from stages", logPrio::LOG_ERROR );
            state( stateCodes::ERROR );
            return ZC_ERROR;
        }
        else
        {
            log<text_log>( "TIMEOUT", logPrio::LOG_DEBUG );
            break; // Timeout ok.
        }
    }

    {
        std::vector<int>         addresses;
        std::vector<std::string> serials;

        rv = parseSystemSerial( addresses, serials, serialRes );
        if( rv == ZUTILS_E_BADSERIAL )
        {
            log<text_log>( "Ignoring inconclusive system.serial snapshot during stage activity.", logPrio::LOG_DEBUG );
            return ZC_CONNECTED;
        }
    }

    return loadStages( serialRes );
}

int zaberLowLevel::loadStages( std::string &serialRes )
{
    std::vector<int>         addresses;
    std::vector<std::string> serials;
    std::vector<int>         oldAddresses;
    bool                     firstDiscoveryPass = !m_stageDiscoveryInitialized;
    size_t                   oldPresentCount    = 0;

    oldAddresses.reserve( m_stages.size() );
    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        oldAddresses.push_back( m_stages[n].deviceAddress() );
        if( m_stages[n].deviceAddress() > 0 )
        {
            ++oldPresentCount;
        }
    }

    int rv = parseSystemSerial( addresses, serials, serialRes );

    if( rv < 0 )
    {
        log<software_error>( { errno, rv, "error in parseSystemSerial" } );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }
    else
    {
        if( firstDiscoveryPass || addresses.size() != oldPresentCount )
        {
            log<text_log>( "Found " + std::to_string( addresses.size() ) + " stages." );
        }

        m_stageAddress.clear(); // We clear this map before re-populating.

        for( size_t n = 0; n < m_stages.size(); ++n )
        {
            m_stages[n].deviceAddress( -1 );
        }

        for( size_t n = 0; n < addresses.size(); ++n )
        {
            if( m_stageSerial.count( serials[n] ) == 1 )
            {
                const size_t stageIndex = m_stageSerial[serials[n]];

                m_stages[stageIndex].deviceAddress( addresses[n] );

                m_stageAddress.insert( { addresses[n], stageIndex } );
                if( firstDiscoveryPass || stageIndex >= oldAddresses.size() ||
                    oldAddresses[stageIndex] != addresses[n] )
                {
                    log<text_log>( "stage @" + std::to_string( addresses[n] ) + " with s/n " + serials[n] +
                                   " corresponds to " + m_stages[stageIndex].name() );
                }
            }
            else
            {
                log<text_log>( "Unkown stage @" + std::to_string( addresses[n] ) + " with s/n " + serials[n],
                               logPrio::LOG_WARNING );
            }
        }

        for( size_t n = 0; n < m_stages.size(); ++n )
        {
            if( m_stages[n].deviceAddress() < 1 )
            {
                if( firstDiscoveryPass || n >= oldAddresses.size() || oldAddresses[n] > 0 )
                {
                    log<text_log>( std::format( "stage {} with s/n {} not found in system.",
                                                m_stages[n].name(),
                                                m_stages[n].serial() ),
                                   logPrio::LOG_ERROR );
                    state( state(), true );
                }
            }
        }
    }

    m_stageDiscoveryInitialized = true;

    return ZC_CONNECTED;
}

int zaberLowLevel::refreshStageDiscovery()
{
    if( m_port <= 0 )
    {
        return ZC_NOT_CONNECTED;
    }

    int rv = za_drain( m_port );

    if( rv != Z_SUCCESS )
    {
        log<software_error>( { rv, "error from za_drain" } );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    char        buffer[256];
    std::string gss = "/ get system.serial";
    int         nwr = za_send( m_port, gss.c_str(), gss.size() );

    if( nwr == Z_ERROR_SYSTEM_ERROR )
    {
        log<text_log>( "Error sending system.serial query to stages", logPrio::LOG_ERROR );
        state( stateCodes::ERROR );
        return ZC_ERROR;
    }

    std::string serialRes;
    while( 1 )
    {
        int nrd = za_receive( m_port, buffer, sizeof( buffer ) );
        if( nrd >= 0 )
        {
            buffer[nrd] = '\0';
            log<text_log>( std::string( "Received: " ) + buffer, logPrio::LOG_DEBUG );
            serialRes += buffer;
        }
        else if( nrd != Z_ERROR_TIMEOUT )
        {
            log<text_log>( "Error receiving from stages", logPrio::LOG_ERROR );
            state( stateCodes::ERROR );
            return ZC_ERROR;
        }
        else
        {
            log<text_log>( "TIMEOUT", logPrio::LOG_DEBUG );
            break; // Timeout ok.
        }
    }

    return loadStages( serialRes );
}

int zaberLowLevel::resetConnection()
{
    if( m_port > 0 )
    {
        int rv = za_disconnect( m_port );
        if( rv < 0 )
        {
            log<text_log>( "Error disconnecting from zaber system.", logPrio::LOG_ERROR );
        }
    }

    m_port                      = 0;
    m_stageDiscoveryInitialized = false;

    return 0;
}

int zaberLowLevel::recoverFromError( bool devicePresent )
{
    resetConnection();

    if( devicePresent )
    {
        state( stateCodes::NOTCONNECTED );
    }
    else
    {
        state( stateCodes::NODEVICE );
    }

    for( size_t i = 0; i < m_stages.size(); ++i )
    {
        if( devicePresent && m_stages[i].deviceAddress() > 0 )
        {
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NOTCONNECTED" ) );
        }
        else
        {
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
        }
    }

    return 0;
}

int zaberLowLevel::appStartup()
{
    if( state() == stateCodes::UNINITIALIZED )
    {
        log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( m_stages.size() == 0 )
    {
        log<text_log>( "No stages configured.", logPrio::LOG_CRITICAL );
        return -1;
    }

    REG_INDI_NEWPROP_NOCB( m_indiP_curr_state, "curr_state", pcf::IndiProperty::Text );

    REG_INDI_NEWPROP_NOCB( m_indiP_max_pos, "max_pos", pcf::IndiProperty::Text );

    REG_INDI_NEWPROP_NOCB( m_indiP_parked, "parked", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP_NOCB( m_indiP_lastHomed, "last_homed", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP_NOCB( m_indiP_curr_pos, "curr_pos", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP_NOCB( m_indiP_temp, "temp", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP_NOCB( m_indiP_warn, "warning", pcf::IndiProperty::Switch );
    m_indiP_warn.setRule( pcf::IndiProperty::AnyOfMany );

    REG_INDI_NEWPROP( m_indiP_tgt_pos, "tgt_pos", pcf::IndiProperty::Number );

    REG_INDI_NEWPROP( m_indiP_req_home, "req_home", pcf::IndiProperty::Switch );
    m_indiP_req_home.setRule( pcf::IndiProperty::AtMostOne );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_req_home_all, "home_all" );

    REG_INDI_NEWPROP( m_indiP_req_halt, "req_halt", pcf::IndiProperty::Switch );
    m_indiP_req_halt.setRule( pcf::IndiProperty::AtMostOne );

    REG_INDI_NEWPROP( m_indiP_req_ehalt, "req_ehalt", pcf::IndiProperty::Switch );
    m_indiP_req_ehalt.setRule( pcf::IndiProperty::AtMostOne );

    REG_INDI_NEWPROP( m_indiP_knob_enable, "knob_enable", pcf::IndiProperty::Switch );
    m_indiP_knob_enable.setPerm( pcf::IndiProperty::ReadWrite );
    m_indiP_knob_enable.setState( pcf::IndiProperty::Idle );
    m_indiP_knob_enable.setRule( pcf::IndiProperty::AtMostOne );

    REG_INDI_NEWPROP( m_indiP_led_enable, "led_enable", pcf::IndiProperty::Switch );
    m_indiP_led_enable.setPerm( pcf::IndiProperty::ReadWrite );
    m_indiP_led_enable.setState( pcf::IndiProperty::Idle );
    m_indiP_led_enable.setRule( pcf::IndiProperty::AtMostOne );

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        m_indiP_curr_state.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_max_pos.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_max_pos[m_stages[n].name()] = -1;

        m_indiP_parked.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_lastHomed.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_curr_pos.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_temp.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_warn.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_warn[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        m_indiP_tgt_pos.add( pcf::IndiElement( m_stages[n].name() ) );

        m_indiP_req_home.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_req_home[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        m_indiP_req_halt.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_req_halt[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        m_indiP_req_ehalt.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_req_ehalt[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        m_indiP_knob_enable.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_knob_enable[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        m_indiP_led_enable.add( pcf::IndiElement( m_stages[n].name() ) );
        m_indiP_led_enable[m_stages[n].name()].setSwitchState( pcf::IndiElement::Off );

        // Now load last state from disk
        std::ifstream posIn;
        posIn.open( std::format( "{}/{}/{}", m_sysPath, m_configName, m_stages[n].name() ) );

        if( !posIn )
        {
            continue;
        }

        if( m_stages[n].readStateFile( posIn ) < 0 )
        {
            return log<software_critical, -1>( std::format( "error reading state file for {}", m_stages[n].name() ) );
        }

        m_indiP_curr_pos[m_stages[n].name()].set( m_stages[n].rawPos() );
        m_indiP_tgt_pos[m_stages[n].name()].set( m_stages[n].tgtPos() );
        m_indiP_parked[m_stages[n].name()].set( m_stages[n].parked() );
        m_indiP_lastHomed[m_stages[n].name()].set( m_stages[n].lastHomed() );
        m_indiP_max_pos[m_stages[n].name()].set( m_stages[n].maxPos() );
        m_indiP_knob_enable[m_stages[n].name()].set( m_stages[n].knobEnabled() ? pcf::IndiElement::On
                                                                               : pcf::IndiElement::Off );
        m_indiP_led_enable[m_stages[n].name()].set( m_stages[n].ledEnabled() ? pcf::IndiElement::On
                                                                             : pcf::IndiElement::Off );
    }

    return 0;
}

int zaberLowLevel::appLogic()
{
    if( state() == stateCodes::INITIALIZED )
    {
        log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( state() == stateCodes::POWERON )
    {
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "POWERON" ) );
        }

        state( stateCodes::NODEVICE );

        return 0; // go around once to give POWERON time to propagate
    }

    if( state() == stateCodes::NODEVICE )
    {
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
        }

        int rv = tty::usbDevice::getDeviceName();

        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0; // means we're powering off
            }

            if( !stateLogged() )
            {
                log<software_error>( { rv, tty::ttyErrorString( rv ) } );
            }

            return 0;
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            if( !stateLogged() )
            {
                log<text_log>(
                    std::format( "USB Device {}:{}:{} not found in udev", m_idVendor, m_idProduct, m_serial ) );
            }

            return 0;
        }
        else
        {
            std::stringstream logs;
            log<text_log>( std::format(
                "USB Device {}:{}:{} found in udev as {}", m_idVendor, m_idProduct, m_serial, m_deviceName ) );

            state( stateCodes::NOTCONNECTED );

            for( size_t i = 0; i < m_stages.size(); ++i )
            {
                if( m_stages[i].deviceAddress() < 1 )
                {
                    continue;
                }

                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NOTCONNECTED" ) );
            }

            return 0; // we return to give the stage time to initialize the connection if this is a USB-FTDI power
                      // on/plug-in event.
        }
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        int rv = connect();

        if( rv == ZC_CONNECTED )
        {
            state( stateCodes::CONNECTED );

            for( size_t i = 0; i < m_stages.size(); ++i )
            {
                if( m_stages[i].deviceAddress() < 1 )
                {
                    continue;
                }

                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "CONNECTED" ) );
            }

            if( !stateLogged() )
            {
                log<text_log>( "Connected to stage(s) on " + m_deviceName );
            }
        }
        else if( rv == ZC_NOT_CONNECTED )
        {
            return 0;
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            if( m_stages[i].deviceAddress() < 1 )
            {
                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
                continue; // Skip configured but not found stage
            }

            std::lock_guard<std::mutex> guard( m_indiMutex ); // Inside loop so INDI requests can steal it

            m_stages[i].enableKnob( m_port, false ); // Always disable the knob on startup

            m_stages[i].enableLED( m_port, false ); // Always disable the LEDs on startup

            m_stages[i].getMaxPos( m_port );

            updateIfChanged( m_indiP_max_pos, m_stages[i].name(), m_stages[i].maxPos() );

            // First unpark if possible
            if( m_stages[i].unpark( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            // Get warnings so first pass through has correct state for home/not-homed
            if( m_stages[i].getWarnings( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }
        }

        state( stateCodes::READY );

        return 0;
    }

    if( state() == stateCodes::READY )
    {
        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );

            bool canRefreshDiscovery = true;
            for( size_t i = 0; i < m_stages.size(); ++i )
            {
                if( m_stages[i].deviceAddress() > 0 && m_stages[i].deviceStatus() == 'B' )
                {
                    canRefreshDiscovery = false;
                    break;
                }
            }

            int rv = ZC_CONNECTED;
            if( canRefreshDiscovery )
            {
                rv = refreshStageDiscovery();
            }

            if( rv == ZC_ERROR )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                return 0;
            }
        }

        // Here we check complete stage state.
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            if( m_stages[i].deviceAddress() < 1 )
            {
                continue; // Skip configured but not found stage
            }

            std::lock_guard<std::mutex> guard( m_indiMutex ); // Inside loop so INDI requests can steal it

            if( m_stages[i].getKnob( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }
            updateSwitchIfChanged( m_indiP_knob_enable,
                                   m_stages[i].name(),
                                   ( m_stages[i].knobEnabled() ? pcf::IndiElement::On : pcf::IndiElement::Off ) );

            if( m_stages[i].getLED( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }
            updateSwitchIfChanged( m_indiP_led_enable,
                                   m_stages[i].name(),
                                   ( m_stages[i].ledEnabled() ? pcf::IndiElement::On : pcf::IndiElement::Off ) );

            if( m_stages[i].getParked( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            updateIfChanged( m_indiP_parked, m_stages[i].name(), m_stages[i].parked() );
            updateIfChanged( m_indiP_lastHomed, m_stages[i].name(), m_stages[i].lastHomed() );

            if( m_stages[i].updatePos( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            updateIfChanged( m_indiP_curr_pos, m_stages[i].name(), m_stages[i].rawPos() );
            updateIfChanged( m_indiP_tgt_pos, m_stages[i].name(), m_stages[i].tgtPos() );

            if( m_stages[i].deviceStatus() == 'B' )
            {
                if( m_stages[i].homing() )
                {
                    updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "HOMING" ) );
                }
                else
                {
                    updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "OPERATING" ) );
                }
            }
            else if( m_stages[i].deviceStatus() == 'I' )
            {
                if( m_stages[i].homing() )
                {
                    log<software_error>( std::format( "stage {} idle but in "
                                                      "state homing. bug.",
                                                      m_stages[i].name() ) );
                    return 0;
                }

                if( m_stages[i].warnWR() )
                {
                    updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NOTHOMED" ) );
                }
                else
                {
                    if( !m_stages[i].parked() )
                    {
                        if( m_stages[i].park( m_port ) < 0 )
                        {
                            if( powerState() != 1 || powerStateTarget() != 1 )
                            {
                                return 0; // means we're powering off
                            }

                            log<software_error>();
                            state( stateCodes::ERROR );
                            return 0;
                        }

                        std::ofstream posOut;

                        { // scope for priv
                            elevatedPrivileges ep( this );
                            posOut.open( std::format( "{}/{}/{}", m_sysPath, m_configName, m_stages[i].name() ) );
                        }

                        if( !posOut )
                        {
                            log<software_error>( std::format( "error opening state file for {}", m_stages[i].name() ) );
                        }
                        else if( m_stages[i].writeStateFile( posOut ) < 0 )
                        {
                            log<software_error>( std::format( "error writing state file for {}", m_stages[i].name() ) );
                        }

                        updateIfChanged( m_indiP_parked, m_stages[i].name(), m_stages[i].parked() );
                        updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "READY" ) );
                    }
                }
            }
            else
            {
                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
            }

            if( m_stages[i].warn() )
            {
                updateIfChanged( m_indiP_warn, m_stages[i].name(), pcf::IndiElement::On );
            }
            else
            {
                updateIfChanged( m_indiP_warn, m_stages[i].name(), pcf::IndiElement::Off );
            }

            if( m_stages[i].updateTemp( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }

                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }
            updateIfChanged( m_indiP_temp, m_stages[i].name(), m_stages[i].temp() );

            if( m_stages[i].getWarnings( m_port ) < 0 )
            {
                if( powerState() != 1 || powerStateTarget() != 1 )
                {
                    return 0; // means we're powering off
                }
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }
        }
    }

    if( state() == stateCodes::ERROR )
    {
        int rv = tty::usbDevice::getDeviceName();
        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0; // means we're powering off
            }

            if( !stateLogged() )
            {
                log<software_error>( { rv, tty::ttyErrorString( rv ) } );
            }

            return recoverFromError( false );
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0; // means we're powering off
            }

            if( !stateLogged() )
            {
                log<text_log>(
                    std::format( "USB Device {}:{}:{} not found in udev", m_idVendor, m_idProduct, m_serial ) );
            }

            return recoverFromError( false );
        }

        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return 0; // means we're powering off
        }

        if( !stateLogged() )
        {
            log<text_log>( "Recovering from stage communication error by resetting the connection.",
                           logPrio::LOG_WARNING );
        }

        return recoverFromError( true );
    }

    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0; // means we're powering off
    }

    if( state() == stateCodes::FAILURE )
    {
        return -1;
    }

    return 0;
}

inline int zaberLowLevel::onPowerOff()
{
    resetConnection();

    std::lock_guard<std::mutex> lock( m_indiMutex );

    for( size_t i = 0; i < m_stages.size(); ++i )
    {
        m_stages[i].onPowerOff();

        // Publish the retained stage snapshot before advertising POWEROFF so
        // subscribers can consume the last known parked/position state first.
        updateIfChanged( m_indiP_max_pos, m_stages[i].name(), m_stages[i].maxPos() );
        updateIfChanged( m_indiP_parked, m_stages[i].name(), m_stages[i].parked() );
        updateIfChanged( m_indiP_lastHomed, m_stages[i].name(), m_stages[i].lastHomed() );
        updateIfChanged( m_indiP_curr_pos, m_stages[i].name(), m_stages[i].rawPos() );
        updateIfChanged( m_indiP_tgt_pos, m_stages[i].name(), m_stages[i].tgtPos() );
        updateSwitchIfChanged( m_indiP_knob_enable,
                               m_stages[i].name(),
                               ( m_stages[i].knobEnabled() ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        updateSwitchIfChanged( m_indiP_led_enable,
                               m_stages[i].name(),
                               ( m_stages[i].ledEnabled() ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        updateIfChanged( m_indiP_temp, m_stages[i].name(), std::string( "" ) );
        updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "POWEROFF" ) );
        updateIfChanged( m_indiP_warn, m_stages[i].name(), pcf::IndiElement::Off );
    }

    return 0;
}

inline int zaberLowLevel::whilePowerOff()
{
    return 0;
}

inline int zaberLowLevel::appShutdown()
{
    for( size_t i = 0; i < m_stages.size(); ++i )
    {
        if( m_stages[i].deviceAddress() < 1 )
        {
            continue;
        }

        updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_tgt_pos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tgt_pos, ipRecv );

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            long tgt = ipRecv[m_stages[n].name()].get<long>();
            if( tgt >= 0 )
            {
                if( m_stages[n].deviceAddress() < 1 )
                {
                    return log<software_error, -1>( std::format(
                        "stage {} with with s/n {} not found in system.", m_stages[n].name(), m_stages[n].serial() ) );
                }

                std::lock_guard<std::mutex> guard( m_indiMutex );

                if( m_stages[n].moveAbs( m_port, tgt ) < 0 )
                {
                    return log<software_error, -1>( { "error from moveAbs for " + m_stages[n].name() } );
                }

                updateIfChanged( m_indiP_tgt_pos, m_stages[n].name(), m_stages[n].tgtPos() );
                updateIfChanged( m_indiP_parked, m_stages[n].name(), m_stages[n].parked() );
                updateIfChanged( m_indiP_curr_state, m_stages[n].name(), std::string( "OPERATING" ) );
            }
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_req_home )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_home, ipRecv );

    // Make sure only one request is sent to avoid racing
    size_t stageno = std::numeric_limits<size_t>::max();

    bool found = false;

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( found )
            {
                log<software_error>( { "more than one stage specified in req_home, rejecting request" } );
                return -1;
            }

            if( m_stages[n].deviceAddress() < 1 )
            {
                return log<software_error, -1>( std::format( "stage {} with with "
                                                             "s/n {} not found",
                                                             m_stages[n].name(),
                                                             m_stages[n].serial() ) );
            }

            stageno = n;
            found   = true;
        }
    }

    if( !found || stageno >= m_stages.size() )
    {
        log<software_error>( "no valid stage specified in req_home, rejecting request" );
        return -1;
    }

    if( ipRecv[m_stages[stageno].name()].getSwitchState() != pcf::IndiElement::On )
    {
        return log<software_warning, 0>( std::format( "request off for stage {} "
                                                      "in req_home",
                                                      m_stages[stageno].name() ) );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_stages[stageno].homing() )
    {
        return log<software_warning, 0>( std::format( "stage {} is already "
                                                      "homing in req_home",
                                                      m_stages[stageno].name() ) );
    }

    if( m_stages[stageno].home( m_port ) < 0 )
    {
        return log<software_error, -1>( std::format( "error from home for {}", m_stages[stageno].name() ) );
    }

    updateIfChanged( m_indiP_tgt_pos, m_stages[stageno].name(), 0 );
    updateIfChanged( m_indiP_parked, m_stages[stageno].name(), m_stages[stageno].parked() );
    updateIfChanged( m_indiP_curr_state, m_stages[stageno].name(), std::string( "HOMING" ) );

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_req_home_all )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_home_all, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        for( size_t n = 0; n < m_stages.size(); ++n )
        {
            if( m_stages[n].deviceAddress() < 1 )
            {
                continue;
            }
            std::lock_guard<std::mutex> guard( m_indiMutex );

            if( m_stages[n].homing() )
            {
                continue;
            }

            if( m_stages[n].home( m_port ) < 0 )
            {
                return log<software_error, -1>( { "error from home for " + m_stages[n].name() } );
            }

            updateIfChanged( m_indiP_tgt_pos, m_stages[n].name(), 0 );
            updateIfChanged( m_indiP_parked, m_stages[n].name(), m_stages[n].parked() );
            updateIfChanged( m_indiP_curr_state, m_stages[n].name(), std::string( "HOMING" ) );
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_req_halt )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_halt, ipRecv );

    // Make sure only one request is sent to avoid racing
    size_t stageno = std::numeric_limits<size_t>::max();

    bool found = false;

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( found )
            {
                return log<software_error, -1>( "more than one stage specified in req_halt, rejecting request" );
            }

            if( m_stages[n].deviceAddress() < 1 )
            {
                return log<software_error, -1>( std::format( "stage {} with with "
                                                             "s/n {} not present",
                                                             m_stages[n].name(),
                                                             m_stages[n].serial() ) );
            }

            stageno = n;
            found   = true;
        }
    }

    if( !found || stageno == std::numeric_limits<size_t>::max() )
    {
        return log<software_error, -1>( "no valid stage specified in req_halt, rejecting request" );
    }

    if( ipRecv[m_stages[stageno].name()].getSwitchState() != pcf::IndiElement::On )
    {
        return log<software_warning, 0>( std::format( "request off for stage {} "
                                                      "in req_halt",
                                                      m_stages[stageno].name() ) );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_stages[stageno].stop( m_port ) < 0 )
    {
        return log<software_error, -1>( std::format( "error from stop for {}", m_stages[stageno].name() ) );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_req_ehalt )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_ehalt, ipRecv );

    // Here we accept multiple ehalts all at once just in case.  It's an emergency~
    // and we don't stop for errors
    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( ipRecv[m_stages[n].name()].getSwitchState() == pcf::IndiElement::On )
            {
                if( m_stages[n].deviceAddress() < 1 )
                {
                    log<software_error>( std::format( "stage {} with s/n {} "
                                                      "not present",
                                                      m_stages[n].name(),
                                                      m_stages[n].serial() ) );
                    continue;
                }

                std::lock_guard<std::mutex> guard( m_indiMutex );

                if( m_stages[n].estop( m_port ) < 0 )
                {
                    log<software_error>( { "error from estop for " + m_stages[n].name() } );
                }
            }
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_knob_enable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_knob_enable, ipRecv );

    // Make sure only one request is sent to avoid racing
    size_t stageno = std::numeric_limits<size_t>::max();

    bool found = false;

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( found )
            {
                return log<software_error, -1>( "more than one stage specified in req_halt, rejecting request" );
            }

            if( m_stages[n].deviceAddress() < 1 )
            {
                return log<software_error, -1>( std::format( "stage {} with with "
                                                             "s/n {} not present",
                                                             m_stages[n].name(),
                                                             m_stages[n].serial() ) );
            }

            stageno = n;
            found   = true;
        }
    }

    if( !found || stageno == std::numeric_limits<size_t>::max() )
    {
        return log<software_error, -1>( "no valid stage specified in req_knob, rejecting request" );
    }

    bool enable_knob = ipRecv[m_stages[stageno].name()].getSwitchState() == pcf::IndiElement::On;

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_stages[stageno].enableKnob( m_port, enable_knob ) < 0 )
    {
        return log<software_error, -1>( std::format( "error from enable knob for {}", m_stages[stageno].name() ) );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevel, m_indiP_led_enable )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_led_enable, ipRecv );

    // Make sure only one request is sent to avoid racing
    size_t stageno = std::numeric_limits<size_t>::max();

    bool found = false;

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( found )
            {
                return log<software_error, -1>( "more than one stage specified in req_halt, rejecting request" );
            }

            if( m_stages[n].deviceAddress() < 1 )
            {
                return log<software_error, -1>( std::format( "stage {} with with "
                                                             "s/n {} not present",
                                                             m_stages[n].name(),
                                                             m_stages[n].serial() ) );
            }

            stageno = n;
            found   = true;
        }
    }

    if( !found || stageno == std::numeric_limits<size_t>::max() )
    {
        return log<software_error, -1>( "no valid stage specified in req_led, rejecting request" );
    }

    bool enable_led = ipRecv[m_stages[stageno].name()].getSwitchState() == pcf::IndiElement::On;

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_stages[stageno].enableLED( m_port, enable_led ) < 0 )
    {
        return log<software_error, -1>( std::format( "error from enable led for {}", m_stages[stageno].name() ) );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // zaberLowLevel_hpp
