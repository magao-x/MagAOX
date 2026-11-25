/** \file koolanceCtrl.hpp
 * \brief The MagAO-X Koolance Controller header file
 *
 * \ingroup koolanceCtrl_files
 */

#ifndef koolanceCtrl_hpp
#define koolanceCtrl_hpp

#include <charconv>
#include <chrono>
#include <deque>
#include <utility>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup koolanceCtrl
 * \brief The MagAO-X application to monitor and control a Koolance cooler
 *
 * <a href="../handbook/operating/software/apps/koolanceCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup koolanceCtrl_files
 * \ingroup koolanceCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Koolance Controller
/** This application will monitor a Koolance v1 or v2 protocol controller.  If v2, will also allow
 * changing of pump and fan settings via INDI.
 *
 * The INDI status property publishes the rolling mean in `flow_rate` and the latest measurement
 * in `flow_rate_raw`, both in liters per minute. The mean uses valid replies received within
 * `flow.averageSeconds`, with the available samples used during startup. Telemetry records raw flow.
 *
 * \ingroup koolanceCtrl
 */
class koolanceCtrl : public MagAOXApp<true>, public tty::usbDevice, public dev::telemeter<koolanceCtrl>
{

    //Give the test harness access.
    friend class koolanceCtrl_test;

    friend class dev::telemeter<koolanceCtrl>;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    /// Length in seconds of the rolling INDI flow average. Zero publishes each reading directly.
    int m_flowAverageSeconds{ 60 };

    ///@}

    /// Expected reply length in bytes, 43 for protocol 1 or 51 for protocol 2.
    size_t m_protocolChars{ 0 };

    /// Whether the pump and fan level properties have been registered.
    bool m_indiSetup{ false };

    /// Timestamped raw flow readings in tenths of a liter per minute, ordered oldest first.
    std::deque<std::pair<std::chrono::steady_clock::time_point, int>> m_flowHistory;

    /// Sum of the flow readings currently in the window, in tenths of a liter per minute.
    double m_flowSum{ 0 };

    /// Whether the status property has been registered with its first valid reading.
    bool m_statusRegistered{ false };

    /// Most recent liquid temperature in degrees Celsius.
    float m_liqTemp{ 0 };

    /// Most recent raw flow reading in liters per minute, used for telemetry.
    float m_flowRate{ 0 };

    /// Most recent pump power level, 1 through 10.
    int m_pumpLvl{ 0 };

    /// Most recent pump speed in revolutions per minute.
    int m_pumpRPM{ 0 };

    /// Most recent fan speed in revolutions per minute.
    int m_fanRPM{ 0 };

    /// Most recent fan power level, 0 through 100.
    int m_fanLvl{ 0 };

  public:
    /// Construct the controller application.
    koolanceCtrl();

    /// Destroy the controller application.
    ~koolanceCtrl() noexcept = default;

    /// Register the controller, USB, and telemetry configuration options.
    virtual void setupConfig();

    /// Load the controller and USB settings from the supplied configuration.
    /** This is called by loadConfig().
     * \returns 0 on success
     * \returns -1 on an invalid flow averaging window
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration to load */ );

    /// Load the application and telemetry configuration.
    virtual void loadConfig();

    /// Start telemetry and enter the device discovery state.
    virtual int appStartup();

    /// Implementation of the FSM for koolanceCtrl.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    virtual int appShutdown();

    /// Initial connection to controller
    /** Determine protocol in use based on number of characters in response
     * and sets up INDI appropriately.
     *
     * Calls to this function should be mutexed.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
    int initialConnect();

    /// Read a controller status reply and update telemetry and INDI.
    /**
     *
     * Calls to this function should be mutexed.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
    int getStatus();

    /// Set the pump level
    /**
     * Calls to this function should be mutexed.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
    int setPumpLvl( int lvl /**< [in] the new level */ );

    /// Set the fan level
    /**
     * Calls to this function should be mutexed.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
    int setFanLvl( int lvl /**< [in] the new level */ );

    /** \name INDI - Data
     *
     *@{
     */
  protected:
    /// Read-only liquid temperature, raw and averaged flow rates, and pump and fan speeds.
    pcf::IndiProperty m_indiP_status;

    /// Pump level readback and, on protocol 2, command target.
    pcf::IndiProperty m_indiP_pumplvl;

    /// Fan level readback and, on protocol 2, command target.
    pcf::IndiProperty m_indiP_fanlvl;

    ///@}

    /// Add a valid sample and return the mean flow rate in liters per minute over the current window.
    double updateFlowAverage( int flowTenths /**< [in] flow rate in tenths of a liter per minute */,
                              std::chrono::steady_clock::time_point now /**< [in] receipt time for the sample */ );

    /// Publish the latest cooler status with both raw and averaged flow readings.
    int publishStatus( double rawFlowRate /**< [in] latest flow rate in liters per minute */,
                       double flowRateAverage /**< [in] averaged flow rate in liters per minute */ );

  public:
    /** \name INDI
     *@{
     */

    /// Validate and handle a new pump level request.
    /** \param ipRecv [in] requested pump level property */
    INDI_NEWCALLBACK_DECL( koolanceCtrl, m_indiP_pumplvl );

    /// Validate and handle a new fan level request.
    /** \param ipRecv [in] requested fan level property */
    INDI_NEWCALLBACK_DECL( koolanceCtrl, m_indiP_fanlvl );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether a periodic telemetry record is due.
    int checkRecordTimes();

    /// Record cooler telemetry on request from the telemeter.
    int recordTelem( const telem_cooler * /**< [in] telemetry type tag */ );

    /// Record the current raw cooler readings when they change.
    int recordCooler( bool force = false /**< [in] record even when readings are unchanged */ );

    ///@}
};

koolanceCtrl::koolanceCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void koolanceCtrl::setupConfig()
{
    config.add( "flow.averageSeconds",
                "",
                "flow.averageSeconds",
                argType::Required,
                "flow",
                "averageSeconds",
                false,
                "int",
                "Rolling window for INDI flow averaging in seconds. Default is 60. Use 0 for raw readings." );

    tty::usbDevice::setupConfig( config );

    dev::telemeter<koolanceCtrl>::setupConfig( config );
}

int koolanceCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    if( _config.isSet( "flow.averageSeconds" ) )
    {
        std::string value;
        _config( value, "flow.averageSeconds" );
        const char *end     = value.data() + value.size();
        int         seconds = 0;
        const auto  parsed  = std::from_chars( value.data(), end, seconds );

        if( parsed.ec != std::errc{} || parsed.ptr != end || seconds < 0 )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "flow.averageSeconds must be a nonnegative integer" } );
        }

        m_flowAverageSeconds = seconds;
    }

    this->m_baudRate = B9600; //default for a Koolance controller.  Will be overridden by any config setting.

    int rv = tty::usbDevice::loadConfig( _config );

    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND ) //Ignore error if not plugged in
    {
        log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }

    return 0;
}

void koolanceCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
        return;
    }

    dev::telemeter<koolanceCtrl>::loadConfig( config );
}

int koolanceCtrl::appStartup()
{
    if( state() == stateCodes::UNINITIALIZED )
    {
        log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( dev::telemeter<koolanceCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::NODEVICE );
    return 0;
}

int koolanceCtrl::appLogic()
{

    if( state() == stateCodes::NODEVICE )
    {
        int rv = tty::usbDevice::getDeviceName();
        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            state( stateCodes::FAILURE );
            if( !stateLogged() )
            {
                log<software_critical>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            }
            return -1;
        }

        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            state( stateCodes::NODEVICE );

            if( !stateLogged() )
            {
                std::stringstream logs;
                logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
                log<text_log>( logs.str() );
            }
            return 0;
        }
        else
        {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as "
                 << m_deviceName;
            log<text_log>( logs.str() );

            state( stateCodes::NOTCONNECTED );
        }
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        int rv = 0;
        {
            elevatedPrivileges ep( this );
            rv = connect();
        }

        if( rv < 0 )
        {
            int nrv = tty::usbDevice::getDeviceName();
            if( nrv < 0 && nrv != TTY_E_DEVNOTFOUND && nrv != TTY_E_NODEVNAMES )
            {
                state( stateCodes::FAILURE );
                if( !stateLogged() )
                {
                    log<software_critical>( { __FILE__, __LINE__, nrv, tty::ttyErrorString( nrv ) } );
                }
                return -1;
            }

            if( nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES )
            {
                state( stateCodes::NODEVICE );

                if( !stateLogged() )
                {
                    std::stringstream logs;
                    logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial
                         << " no longer found in udev";
                    log<text_log>( logs.str() );
                }
                return 0;
            }

            //if connect failed, and there is a device, then we have some other problem.
            sleep( 1 ); //wait to see if power state updates
            if( m_powerState == 0 )
            {
                return 0;
            }

            //Ok we can't figure this out, die.
            state( stateCodes::FAILURE );
            if( !stateLogged() )
            {
                log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            }
            return -1;
        }
        else
        {
            std::unique_lock<std::mutex> lock( m_indiMutex );
            if( initialConnect() == 0 )
            {
                state( stateCodes::CONNECTED );
            }
            else
            {
                if( !stateLogged() )
                {
                    log<text_log>( "no response from device" );
                }
                return 0;
            }
        }

        if( state() == stateCodes::CONNECTED )
        {
            std::stringstream logs;
            logs << "Connected to koolance system on " << m_deviceName;
            log<text_log>( logs.str() );
            state( stateCodes::READY );
            return 0;
        }
    }

    if( state() == stateCodes::READY )
    {
        { //mutex scope
            std::unique_lock<std::mutex> lock( m_indiMutex );
            if( getStatus() < 0 )
            {
                log<software_error>( { __FILE__, __LINE__ } );
            }
        }

        if( telemeter<koolanceCtrl>::appLogic() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            return 0;
        }
        return 0;
    }

    return 0;
}

int koolanceCtrl::appShutdown()
{
    dev::telemeter<koolanceCtrl>::appShutdown();
    return 0;
}

int koolanceCtrl::initialConnect()
{
    int rv;

    std::vector<unsigned char> com;
    com.resize( 3, '\0' );
    com[0] = 0xCF;
    com[1] = 0x01;
    com[2] = 0x08;

    rv = write( m_fileDescrip, com.data(), com.size() );

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from write" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    std::vector<unsigned char> resp;
    resp.resize( 51 );

    mx::sys::milliSleep( 1000 ); //Sleep for a long time to make sure device responds
    int readBytes;
    rv = tty::ttyReadRaw( resp, readBytes, m_fileDescrip, 1000 ); ///\todo needs to be iodevice

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from read" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    if( readBytes == 43 )
    {
        log<text_log>( "found protocol 1 device" );
        m_protocolChars = 43;
    }
    else if( readBytes == 51 )
    {
        log<text_log>( "found protocol 2 device" );
        m_protocolChars = 51;
    }
    else
    {
        return -1;
    }

    if( !m_indiSetup )
    {
        if( readBytes == 43 )
        {
            REG_INDI_NEWPROP_NOCB( m_indiP_pumplvl, "pump_level", pcf::IndiProperty::Number );
            m_indiP_pumplvl.add( pcf::IndiElement( "current" ) );

            REG_INDI_NEWPROP_NOCB( m_indiP_fanlvl, "fan_level", pcf::IndiProperty::Number );
            m_indiP_fanlvl.add( pcf::IndiElement( "current" ) );
        }
        else
        {
            createStandardIndiNumber<int>( m_indiP_pumplvl, "pump_level", 1, 10, 1, "%d", "Pump Level", "Lab" );
            registerIndiPropertyNew( m_indiP_pumplvl, INDI_NEWCALLBACK( m_indiP_pumplvl ) );

            createStandardIndiNumber<int>( m_indiP_fanlvl, "fan_level", 0, 100, 1, "%d", "Fan Level", "Lab" );
            registerIndiPropertyNew( m_indiP_fanlvl, INDI_NEWCALLBACK( m_indiP_fanlvl ) );
        }

        m_indiSetup = true;
    }

    return 0;
}

int koolanceCtrl::getStatus()
{
    int rv;

    std::vector<unsigned char> com;
    com.resize( 3, '\0' );
    com[0] = 0xCF;
    com[1] = 0x01;
    com[2] = 0x08;

    rv = write( m_fileDescrip, com.data(), com.size() );

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from write" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    std::string resp;

    rv = tty::ttyRead( resp, m_protocolChars, m_fileDescrip, 1000 ); ///\todo needs to be iodevice

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from read" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    if( resp.size() != m_protocolChars )
    {
        log<software_error>(
            { __FILE__,
              __LINE__,
              std::string( "wrong response size (" ) + std::to_string( resp.size() ) + ") returned" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    const int    flowTenths  = ( static_cast<unsigned char>( resp[12] ) << 8 ) + static_cast<unsigned char>( resp[13] );
    const double rawFlowRate = flowTenths / 10.0;
    const double flowRateAverage = updateFlowAverage( flowTenths, std::chrono::steady_clock::now() );

    m_liqTemp  = ( ( (unsigned char)resp[2] << 8 ) + (unsigned char)resp[3] - 2000 ) / 10.0;
    m_flowRate = rawFlowRate;
    m_fanRPM   = ( (unsigned char)resp[8] << 8 ) + (unsigned char)resp[9];
    m_pumpRPM  = ( (unsigned char)resp[10] << 8 ) + (unsigned char)resp[11];
    m_fanLvl   = (unsigned char)resp[15];
    m_pumpLvl  = (unsigned char)resp[17];

    recordCooler();

    updateIfChanged( m_indiP_pumplvl, "current", m_pumpLvl );
    updateIfChanged( m_indiP_fanlvl, "current", m_fanLvl );

    return publishStatus( rawFlowRate, flowRateAverage );
}

inline double koolanceCtrl::updateFlowAverage( int flowTenths, std::chrono::steady_clock::time_point now )
{
    const auto oldest = now - std::chrono::seconds( m_flowAverageSeconds );

    while( !m_flowHistory.empty() && m_flowHistory.front().first <= oldest )
    {
        m_flowSum -= m_flowHistory.front().second;
        m_flowHistory.pop_front();
    }

    m_flowHistory.emplace_back( now, flowTenths );
    m_flowSum += flowTenths;

    return m_flowSum / ( 10.0 * m_flowHistory.size() );
}

inline int koolanceCtrl::publishStatus( double rawFlowRate, double flowRateAverage )
{
    static const std::vector<std::string> elements{
        "liquid_temp", "flow_rate", "flow_rate_raw", "pump_rpm", "fan_rpm" };
    const std::vector<double> values{
        m_liqTemp, flowRateAverage, rawFlowRate, static_cast<double>( m_pumpRPM ), static_cast<double>( m_fanRPM ) };

    if( !m_statusRegistered )
    {
        // Register only after initializing every element so subscribers get a complete first status.
        createROIndiNumber( m_indiP_status, "status" );
        m_indiP_status.setState( INDI_OK );

        for( size_t n = 0; n < elements.size(); ++n )
        {
            pcf::IndiElement element( elements[n] );
            element.set( values[n] );
            m_indiP_status.add( element );
        }

        if( registerIndiPropertyReadOnly( m_indiP_status ) < 0 )
        {
            return -1;
        }

        m_statusRegistered = true;

        if( m_indiDriver )
        {
            m_indiDriver->sendDefProperty( m_indiP_status );
        }
    }
    else
    {
        updateIfChanged( m_indiP_status, elements, values );
    }

    return 0;
}

int koolanceCtrl::setPumpLvl( int lvl )
{
    if( m_protocolChars == 43 )
    {
        return 0;
    }

    int rv;

    std::vector<unsigned char> com;
    com.resize( m_protocolChars, '\0' );
    com[0] = 0xCF;
    com[1] = 0x04;

    com[15] = m_fanLvl;
    com[17] = lvl;

    //Disable most stuff.
    for( size_t n = 20; n < m_protocolChars - 1; ++n )
    {
        com[n] = 0xAA;
    }

    //Preserve units
    if( m_protocolChars > 43 )
    {
        com[44] = 0;
        com[45] = 0x0001;
        com[46] = 0;
        com[47] = 0;
        com[48] = 0;
        com[49] = 0;
    }

    int checksum = 0;
    for( size_t n = 0; n < m_protocolChars - 1; ++n )
    {
        checksum += com[n];
    }
    com[m_protocolChars - 1] = checksum % 0x64;

    rv = write( m_fileDescrip, com.data(), com.size() );

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from write" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    log<text_log>( "set pump level to " + std::to_string( lvl ) );

    return 0;
}

int koolanceCtrl::setFanLvl( int lvl )
{
    if( m_protocolChars == 43 )
    {
        return 0;
    }

    int rv;

    std::vector<unsigned char> com;
    com.resize( m_protocolChars, '\0' );
    com[0] = 0xCF;
    com[1] = 0x04;

    com[15] = lvl;
    com[17] = m_pumpLvl;

    //Disable most stuff.
    for( size_t n = 20; n < m_protocolChars - 1; ++n )
    {
        com[n] = 0xAA;
    }

    //Preserve units
    if( m_protocolChars > 43 )
    {
        com[44] = 0;
        com[45] = 0x0001;
        com[46] = 0;
        com[47] = 0;
        com[48] = 0;
        com[49] = 0;
    }

    int checksum = 0;
    for( size_t n = 0; n < m_protocolChars - 1; ++n )
    {
        checksum += com[n];
    }
    com[m_protocolChars - 1] = checksum % 0x64;

    rv = write( m_fileDescrip, com.data(), com.size() );

    if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, errno, "error from write" } );
        ::close( m_fileDescrip );
        m_fileDescrip = 0;
        state( stateCodes::NOTCONNECTED );

        return -1;
    }

    log<text_log>( "set fan level to " + std::to_string( lvl ) );

    return 0;
}

INDI_NEWCALLBACK_DEFN( koolanceCtrl, m_indiP_pumplvl )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_pumplvl.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    int lvl = -1;

    if( ipRecv.find( "current" ) )
    {
        lvl = ipRecv["current"].get<int>();
    }

    if( ipRecv.find( "target" ) )
    {
        lvl = ipRecv["target"].get<int>();
    }

    if( lvl < 1 || lvl > 10 )
    {
        log<software_error>( { __FILE__, __LINE__, "Pump level out of range" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    updateIfChanged( m_indiP_pumplvl, "target", lvl );
    return setPumpLvl( lvl );
}

INDI_NEWCALLBACK_DEFN( koolanceCtrl, m_indiP_fanlvl )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fanlvl.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    int lvl = -1;

    if( ipRecv.find( "current" ) )
    {
        lvl = ipRecv["current"].get<int>();
    }

    if( ipRecv.find( "target" ) )
    {
        lvl = ipRecv["target"].get<int>();
    }

    if( lvl < 0 || lvl > 100 )
    {
        log<software_error>( { __FILE__, __LINE__, "Fan level out of range" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    updateIfChanged( m_indiP_fanlvl, "target", lvl );
    return setFanLvl( lvl );

    return 0;
}

inline int koolanceCtrl::checkRecordTimes()
{
    return telemeter<koolanceCtrl>::checkRecordTimes( telem_cooler() );
}

inline int koolanceCtrl::recordTelem( const telem_cooler * )
{
    return recordCooler( true );
}

inline int koolanceCtrl::recordCooler( bool force )
{
    static float last_liqTemp  = std::numeric_limits<float>::max();
    static float last_flowRate = std::numeric_limits<float>::max();
    static int   last_pumpLvl  = std::numeric_limits<int>::max();
    static int   last_pumpRPM  = std::numeric_limits<int>::max();
    static int   last_fanRPM   = std::numeric_limits<int>::max();
    static int   last_fanLvl   = std::numeric_limits<int>::max();

    if( m_liqTemp != last_liqTemp || m_flowRate != last_flowRate || m_pumpLvl != last_pumpLvl ||
        m_pumpRPM != last_pumpRPM || m_fanRPM != last_fanRPM || m_fanLvl != last_fanLvl || force )
    {
        telem<telem_cooler>(
            { m_liqTemp, m_flowRate, (uint8_t)m_pumpLvl, (uint16_t)m_pumpRPM, (uint8_t)m_fanLvl, (uint16_t)m_fanRPM } );

        last_liqTemp  = m_liqTemp;
        last_flowRate = m_flowRate;
        last_pumpLvl  = m_pumpLvl;
        last_pumpRPM  = m_pumpRPM;
        last_fanLvl   = m_fanLvl;
        last_fanRPM   = m_fanRPM;
    }

    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //koolanceCtrl_hpp
