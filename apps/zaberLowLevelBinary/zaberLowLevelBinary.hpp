/** \file zaberLowLevelBinary.hpp
 * \brief The MagAO-X low-level binary-protocol Zaber controller.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevelBinary_files
 */

#ifndef zaberLowLevelBinary_hpp
#define zaberLowLevelBinary_hpp

#include <iostream>
#include <thread>

#include "../../libMagAOX/libMagAOX.hpp" // Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT;

#include "zaberBinaryStage.hpp"
#include "zb_serial.h"

#define ZBC_CONNECTED ( 0 )
#define ZBC_ERROR ( -1 )
#define ZBC_NOT_CONNECTED ( 10 )

/** \defgroup zaberLowLevelBinary low-level binary zaber controller
 * \brief The low-level binary interface to a set of chained Zaber stages
 *
 * \ingroup apps
 */

/** \defgroup zaberLowLevelBinary_files zaber low-level binary files
 * \ingroup zaberLowLevelBinary
 */

namespace MagAOX
{
namespace app
{

/// The low-level binary-protocol Zaber controller.
/**
 * This app mirrors `zaberLowLevel` as closely as possible while speaking the
 * firmware 5.xx T-series binary protocol.
 *
 * \ingroup zaberLowLevelBinary
 */
class zaberLowLevelBinary : public MagAOXAppT, public tty::usbDevice
{
    friend class zaberLowLevelBinary_test;

  protected:
    /// Number of configured stages.
    int m_numStages{ 0 };

    /// Connected binary protocol port.
    z_port m_port{ 0 };

    /** \name Configurable Parameters
     *
     * @{
     */
    /// Whether to renumber the daisy chain during connect.
    bool m_renumberOnConnect{ true };

    /// Maximum device address to probe while matching configured serial numbers.
    int m_maxDiscoveryAddress{ 16 };

    /// Binary command timeout in milliseconds.
    int m_commandTimeout{ 250 };

    /// Pause in milliseconds after renumbering before further commands are sent.
    int m_renumberPauseMs{ 750 };

    ///@}

    /** \name Stage Mapping - Data
     *
     * @{
     */
    /// Stage helpers in configuration order.
    std::vector<zaberBinaryStage<zaberLowLevelBinary>> m_stages;

    /// Map from binary device address to configured stage index.
    std::unordered_map<int, size_t> m_stageAddress;

    /// Map from configured serial number to configured stage index.
    std::unordered_map<std::string, size_t> m_stageSerial;

    /// Map from configured stage name to configured stage index.
    std::unordered_map<std::string, size_t> m_stageName;

    ///@}

  public:
    /// Default constructor.
    zaberLowLevelBinary();

    /// Destructor.
    ~zaberLowLevelBinary() noexcept
    {
    }

    /// Set up application configuration.
    virtual void setupConfig();

    /// Load application configuration.
    virtual void loadConfig();

    /// Connect to the binary-protocol stage chain and discover configured devices.
    int connect();

    /// Discover configured stages on the binary bus.
    int loadStages();

    /// Query a device directly for discovery-time replies.
    int queryDevice( int32_t &response,      /**< [out] decoded reply data */
                     uint8_t  deviceAddress, /**< [in] device address */
                     uint8_t  commandNumber, /**< [in] command number */
                     int32_t  data,          /**< [in] command data */
                     uint8_t  expectedReply  /**< [in] expected reply command number */
    );

    /// Send a broadcast or address-specific command with no awaited reply.
    int sendCommandNoReply( uint8_t deviceAddress, /**< [in] device address */
                            uint8_t commandNumber, /**< [in] command number */
                            int32_t data           /**< [in] command data */
    );

    /// Startup logic.
    virtual int appStartup();

    /// Main FSM implementation.
    virtual int appLogic();

    /// Power-off transition handler.
    virtual int onPowerOff();

    /// Powered-off loop handler.
    virtual int whilePowerOff();

    /// Shutdown handler.
    virtual int appShutdown();

  protected:
    /** \name INDI Stage State - Data
     *
     * @{
     */
    /// Current stage state reported per configured stage.
    pcf::IndiProperty m_indiP_curr_state;

    /// Maximum raw position reported per configured stage.
    pcf::IndiProperty m_indiP_max_pos;

    /// Parked-state bookkeeping reported per configured stage.
    pcf::IndiProperty m_indiP_parked;

    /// Last-homed timestamps reported per configured stage.
    pcf::IndiProperty m_indiP_lastHomed;

    /// Current raw position reported per configured stage.
    pcf::IndiProperty m_indiP_curr_pos;

    /// Driver temperature reported per configured stage.
    pcf::IndiProperty m_indiP_temp;

    /// Warning-state switch reported per configured stage.
    pcf::IndiProperty m_indiP_warn;

    /// Requested target raw position per configured stage.
    pcf::IndiProperty m_indiP_tgt_pos;

    /// Per-stage home requests.
    pcf::IndiProperty m_indiP_req_home;

    /// Global request to home all configured stages.
    pcf::IndiProperty m_indiP_req_home_all;

    /// Per-stage halt requests.
    pcf::IndiProperty m_indiP_req_halt;

    /// Per-stage emergency-halt requests.
    pcf::IndiProperty m_indiP_req_ehalt;

    /// Enable or disable a stages potentiometer
    pcf::IndiProperty m_indiP_knob_enable;

    ///@}

  public:
    /** \name INDI Stage State
     *
     * @{
     */
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_tgt_pos );
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_req_home );
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_req_home_all );
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_req_halt );
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_req_ehalt );
    INDI_NEWCALLBACK_DECL( zaberLowLevelBinary, m_indiP_knob_enable );
    ///@}
};

zaberLowLevelBinary::zaberLowLevelBinary() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
}

void zaberLowLevelBinary::setupConfig()
{
    tty::usbDevice::setupConfig( config );

    config.add( "stages.renumberOnConnect",
                "",
                "stages.renumberOnConnect",
                argType::Required,
                "stages",
                "renumberOnConnect",
                false,
                "bool",
                "Whether to issue a broadcast renumber on connect. Default is true." );

    config.add( "stages.maxDiscoveryAddress",
                "",
                "stages.maxDiscoveryAddress",
                argType::Required,
                "stages",
                "maxDiscoveryAddress",
                false,
                "int",
                "Maximum device address to scan when matching configured serial numbers." );

    config.add( "stages.commandTimeout",
                "",
                "stages.commandTimeout",
                argType::Required,
                "stages",
                "commandTimeout",
                false,
                "int",
                "Binary command timeout in milliseconds." );

    config.add( "stages.renumberPauseMs",
                "",
                "stages.renumberPauseMs",
                argType::Required,
                "stages",
                "renumberPauseMs",
                false,
                "int",
                "Pause in milliseconds after a renumber command before discovery begins." );
}

void zaberLowLevelBinary::loadConfig()
{
    this->m_baudRate = B9600;

    int rv = tty::usbDevice::loadConfig( config );
    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND )
    {
        log<software_error>( { rv, tty::ttyErrorString( rv ) } );
    }

    config( m_renumberOnConnect, "stages.renumberOnConnect" );
    config( m_maxDiscoveryAddress, "stages.maxDiscoveryAddress" );
    config( m_commandTimeout, "stages.commandTimeout" );
    config( m_renumberPauseMs, "stages.renumberPauseMs" );
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
            m_stages.push_back( zaberBinaryStage<zaberLowLevelBinary>( this ) );

            size_t idx = m_stages.size() - 1;
            m_stages[idx].name( sections[n] );

            std::string tmp = m_stages[idx].serial();
            config.configUnused( tmp, mx::app::iniFile::makeKey( sections[n], "serial" ) );
            m_stages[idx].serial( tmp );

            int32_t targetSpeed = m_stages[idx].targetSpeed();
            config.configUnused( targetSpeed, mx::app::iniFile::makeKey( sections[n], "targetSpeed" ) );
            m_stages[idx].targetSpeed( targetSpeed );

            m_stageName.insert( { m_stages[idx].name(), idx } );
            m_stageSerial.insert( { m_stages[idx].serial(), idx } );
        }
    }
}

int zaberLowLevelBinary::queryDevice(
    int32_t &response, uint8_t deviceAddress, uint8_t commandNumber, int32_t data, uint8_t expectedReply )
{
    uint8_t command[6];
    if( zb_encode( command, deviceAddress, commandNumber, data ) != Z_SUCCESS )
    {
        return log<software_error, -1>( "zb_encode failed" );
    }

    if( zb_send( m_port, command ) != 6 )
    {
        return log<software_error, -1>( "zb_send failed" );
    }

    uint8_t reply[6];
    int     rv = zb_receive( m_port, reply );
    if( rv != 6 )
    {
        return -1;
    }

    if( reply[0] != deviceAddress )
    {
        return -1;
    }

    if( reply[1] == 255 )
    {
        return -1;
    }

    if( reply[1] != expectedReply )
    {
        return -1;
    }

    if( zb_decode( &response, reply ) != Z_SUCCESS )
    {
        return -1;
    }

    return 0;
}

int zaberLowLevelBinary::sendCommandNoReply( uint8_t deviceAddress, uint8_t commandNumber, int32_t data )
{
    uint8_t command[6];
    if( zb_encode( command, deviceAddress, commandNumber, data ) != Z_SUCCESS )
    {
        return log<software_error, -1>( "zb_encode failed" );
    }

    if( zb_send( m_port, command ) != 6 )
    {
        return log<software_error, -1>( "zb_send failed" );
    }

    return 0;
}

int zaberLowLevelBinary::connect()
{
    if( m_port > 0 )
    {
        int rv = zb_disconnect( m_port );
        if( rv < 0 )
        {
            log<text_log>( "Error disconnecting from zaber binary system.", logPrio::LOG_ERROR );
        }
        m_port = 0;
    }

    int zrv;
    { // mutex scope
        elevatedPrivileges elPriv( this );
        zrv = zb_connect( &m_port, m_deviceName.c_str() );
    }

    if( zrv != Z_SUCCESS || m_port <= 0 )
    {
        if( m_port > 0 )
        {
            zb_disconnect( m_port );
            m_port = 0;
        }

        if( !stateLogged() )
        {
            log<software_error>( { "can not connect to zaber binary stage(s)" } );
        }

        return ZBC_NOT_CONNECTED;
    }

    if( zb_set_timeout( m_port, m_commandTimeout ) < 0 )
    {
        log<software_error>( { "error setting binary command timeout" } );
        state( stateCodes::ERROR );
        return ZBC_ERROR;
    }

    if( zb_drain( m_port ) != Z_SUCCESS )
    {
        log<software_error>( { "error draining binary port" } );
        state( stateCodes::ERROR );
        return ZBC_ERROR;
    }

    if( m_renumberOnConnect )
    {
        if( sendCommandNoReply( 0, zaberBinaryStage<zaberLowLevelBinary>::cmdRenumber, 0 ) < 0 )
        {
            log<text_log>( "Error sending renumber query to stages", logPrio::LOG_ERROR );
            state( stateCodes::ERROR );
            return ZBC_ERROR;
        }

        std::this_thread::sleep_for( std::chrono::milliseconds( m_renumberPauseMs ) );
        zb_drain( m_port );
    }

    return loadStages();
}

int zaberLowLevelBinary::loadStages()
{
    m_stageAddress.clear();

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        m_stages[n].deviceAddress( -1 );
    }

    for( int address = 1; address <= m_maxDiscoveryAddress; ++address )
    {
        int32_t serialNumber;
        if( queryDevice( serialNumber,
                         static_cast<uint8_t>( address ),
                         zaberBinaryStage<zaberLowLevelBinary>::cmdReturnSerialNumber,
                         0,
                         zaberBinaryStage<zaberLowLevelBinary>::cmdReturnSerialNumber ) < 0 )
        {
            continue;
        }

        std::string serial = std::to_string( serialNumber );
        if( m_stageSerial.count( serial ) == 1 )
        {
            size_t idx = m_stageSerial[serial];
            m_stages[idx].deviceAddress( address );
            m_stageAddress.insert( { address, idx } );
            log<text_log>( "stage @" + std::to_string( address ) + " with s/n " + serial + " corresponds to " +
                           m_stages[idx].name() );
        }
        else
        {
            log<text_log>( "Unknown stage @" + std::to_string( address ) + " with s/n " + serial,
                           logPrio::LOG_WARNING );
        }
    }

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( m_stages[n].deviceAddress() < 1 )
        {
            log<text_log>(
                std::format( "stage {} with s/n {} not found in system.", m_stages[n].name(), m_stages[n].serial() ),
                logPrio::LOG_ERROR );
            state( state(), true );
        }
    }

    return ZBC_CONNECTED;
}

int zaberLowLevelBinary::appStartup()
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
    }

    return 0;
}

int zaberLowLevelBinary::appLogic()
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
        return 0;
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
                return 0;
            }

            state( stateCodes::FAILURE );
            if( !stateLogged() )
            {
                log<software_critical>( { rv, tty::ttyErrorString( rv ) } );
            }
            return -1;
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

        log<text_log>(
            std::format( "USB Device {}:{}:{} found in udev as {}", m_idVendor, m_idProduct, m_serial, m_deviceName ) );

        state( stateCodes::NOTCONNECTED );
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            if( m_stages[i].deviceAddress() < 1 )
            {
                continue;
            }
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NOTCONNECTED" ) );
        }

        return 0;
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        int rv = connect();
        if( rv == ZBC_CONNECTED )
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
                log<text_log>( "Connected to binary stage(s) on " + m_deviceName );
            }
        }
        else if( rv == ZBC_NOT_CONNECTED )
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
                continue;
            }

            std::lock_guard<std::mutex> guard( m_indiMutex );

            if( m_stages[i].enableKnob( m_port, false ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].getMaxPos( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].setTargetSpeed( m_port, m_stages[i].targetSpeed() ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            updateIfChanged( m_indiP_max_pos, m_stages[i].name(), m_stages[i].maxPos() );

            if( m_stages[i].updatePos( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].recallParkPosition( m_port ) < 0 )
            {
                log<text_log>( "No stored parked position found for " + m_stages[i].name(), logPrio::LOG_INFO );
            }
            else if( m_stages[i].restoreParkedState( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].getWarnings( m_port ) < 0 )
            {
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
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            if( m_stages[i].deviceAddress() < 1 )
            {
                continue;
            }

            std::lock_guard<std::mutex> guard( m_indiMutex );

            if( m_stages[i].updatePos( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].getParked( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            if( m_stages[i].getKnob( m_port ) < 0 )
            {
                log<software_error>();
                state( stateCodes::ERROR );
                return 0;
            }

            updateIfChanged( m_indiP_parked, m_stages[i].name(), m_stages[i].parked() );
            updateIfChanged( m_indiP_lastHomed, m_stages[i].name(), m_stages[i].lastHomed() );
            updateIfChanged( m_indiP_curr_pos, m_stages[i].name(), m_stages[i].rawPos() );
            updateIfChanged( m_indiP_tgt_pos, m_stages[i].name(), m_stages[i].tgtPos() );

            updateSwitchIfChanged( m_indiP_knob_enable,
                                   m_stages[i].name(),
                                   m_stages[i].knobEnabled() ? pcf::IndiElement::On : pcf::IndiElement::Off );

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
                    log<software_error>( std::format( "stage {} idle but in state homing. bug.", m_stages[i].name() ) );
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
                            log<software_error>();
                            state( stateCodes::ERROR );
                            return 0;
                        }

                        std::ofstream posOut;
                        { // mutex scope
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
                    }

                    updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "READY" ) );
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

            m_stages[i].updateTemp( m_port );
            updateIfChanged( m_indiP_temp, m_stages[i].name(), m_stages[i].temp() );
        }
    }

    if( state() == stateCodes::ERROR )
    {
        int rv = tty::usbDevice::getDeviceName();
        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            if( powerState() != 1 || powerStateTarget() != 1 )
            {
                return 0;
            }

            state( stateCodes::FAILURE );
            for( size_t i = 0; i < m_stages.size(); ++i )
            {
                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "FAILURE" ) );
            }
            if( !stateLogged() )
            {
                log<software_critical>( { rv, tty::ttyErrorString( rv ) } );
            }
            return rv;
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            state( stateCodes::NODEVICE );
            for( size_t i = 0; i < m_stages.size(); ++i )
            {
                updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "NODEVICE" ) );
            }
            return 0;
        }

        if( powerState() != 1 || powerStateTarget() != 1 )
        {
            return 0;
        }

        state( stateCodes::FAILURE );
        for( size_t i = 0; i < m_stages.size(); ++i )
        {
            updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "FAILURE" ) );
        }

        log<software_critical>();
        log<text_log>( "Binary-protocol error not due to loss of USB connection.", logPrio::LOG_CRITICAL );
    }

    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        return 0;
    }

    if( state() == stateCodes::FAILURE )
    {
        return -1;
    }

    return 0;
}

inline int zaberLowLevelBinary::onPowerOff()
{
    if( m_port > 0 )
    {
        int rv = zb_disconnect( m_port );
        if( rv < 0 )
        {
            log<text_log>( "Error disconnecting from zaber binary system.", logPrio::LOG_ERROR );
        }
    }

    m_port = 0;

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
        updateIfChanged( m_indiP_temp, m_stages[i].name(), std::string( "" ) );
        updateIfChanged( m_indiP_curr_state, m_stages[i].name(), std::string( "POWEROFF" ) );
        updateIfChanged( m_indiP_warn, m_stages[i].name(), pcf::IndiElement::Off );
    }

    return 0;
}

inline int zaberLowLevelBinary::whilePowerOff()
{
    return 0;
}

inline int zaberLowLevelBinary::appShutdown()
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

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_tgt_pos )( const pcf::IndiProperty &ipRecv )
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
                        "stage {} with s/n {} not found in system.", m_stages[n].name(), m_stages[n].serial() ) );
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

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_req_home )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_home, ipRecv );

    size_t stageno = std::numeric_limits<size_t>::max();
    bool   found   = false;

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) )
        {
            if( found )
            {
                return log<software_error, -1>( "more than one stage specified in req_home, rejecting request" );
            }

            if( m_stages[n].deviceAddress() < 1 )
            {
                return log<software_error, -1>(
                    std::format( "stage {} with s/n {} not found", m_stages[n].name(), m_stages[n].serial() ) );
            }

            stageno = n;
            found   = true;
        }
    }

    if( !found || stageno >= m_stages.size() )
    {
        return log<software_error, -1>( "no valid stage specified in req_home, rejecting request" );
    }

    if( ipRecv[m_stages[stageno].name()].getSwitchState() != pcf::IndiElement::On )
    {
        return log<software_warning, 0>(
            std::format( "request off for stage {} in req_home", m_stages[stageno].name() ) );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );
    if( m_stages[stageno].homing() )
    {
        return log<software_warning, 0>(
            std::format( "stage {} is already homing in req_home", m_stages[stageno].name() ) );
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

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_req_home_all )( const pcf::IndiProperty &ipRecv )
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

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_req_halt )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_halt, ipRecv );

    size_t stageno = std::numeric_limits<size_t>::max();
    bool   found   = false;

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
                return log<software_error, -1>(
                    std::format( "stage {} with s/n {} not present", m_stages[n].name(), m_stages[n].serial() ) );
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
        return log<software_warning, 0>(
            std::format( "request off for stage {} in req_halt", m_stages[stageno].name() ) );
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );
    if( m_stages[stageno].stop( m_port ) < 0 )
    {
        return log<software_error, -1>( std::format( "error from stop for {}", m_stages[stageno].name() ) );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_req_ehalt )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_req_ehalt, ipRecv );

    for( size_t n = 0; n < m_stages.size(); ++n )
    {
        if( ipRecv.find( m_stages[n].name() ) && ipRecv[m_stages[n].name()].getSwitchState() == pcf::IndiElement::On )
        {
            if( m_stages[n].deviceAddress() < 1 )
            {
                log<software_error>(
                    std::format( "stage {} with s/n {} not present", m_stages[n].name(), m_stages[n].serial() ) );
                continue;
            }

            std::lock_guard<std::mutex> guard( m_indiMutex );
            if( m_stages[n].estop( m_port ) < 0 )
            {
                log<software_error>( { "error from estop for " + m_stages[n].name() } );
            }
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( zaberLowLevelBinary, m_indiP_knob_enable )( const pcf::IndiProperty &ipRecv )
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

} // namespace app
} // namespace MagAOX

#endif // zaberLowLevelBinary_hpp
