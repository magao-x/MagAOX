/** \file flipperCtrl.hpp
 * \brief Control a two-position MagAO-X flipper with software parking.
 * \author MagAO-X developers
 * \ingroup flipperCtrl_files
 */
#ifndef flipperCtrl_hpp
#define flipperCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" // Included on the command line to trigger the precompiled header.
#include "../../magaox_git_version.h"

#include <chrono>
#include <fstream>

/** \defgroup flipperCtrl
 * \brief Control and retain the position of a two-position filter flipper.
 * \ingroup apps
 */
/** \defgroup flipperCtrl_files
 * \ingroup flipperCtrl
 */
namespace MagAOX
{
namespace app
{
/// Control a flipper and retain confirmed endpoints across power-off and software restarts.
/** Parking records an idle endpoint without commanding hardware parking. The backing record is invalidated before
 * motion; an interrupted move therefore recovers as unknown. The app-specific sys directory contains a `position`
 * file with two integers: the last confirmed physical endpoint (0, 1, or 2) and its parked flag (0 or 1).
 * A parked record requires endpoint 1 or 2. Reversal changes logical names without changing stored endpoints.
 *
 * A fresh settled observation after power-on replaces the inference and warns once if the endpoints disagree.
 * Retention assumes the idle mechanism stays at its endpoint without power and the configuration name continues
 * to identify the same device. Manual movement while powered off cannot be verified by software.
 *
 * After startup, callers must hold m_indiMutex for the position, persistence, publication, and telemetry helpers
 * unless a method explicitly acquires it. appStartup() initializes those helpers before normal execution begins.
 * \ingroup flipperCtrl
 */
class flipperCtrl : public MagAOXApp<true>,
                    public tty::usbDevice,
                    public dev::ioDevice,
                    public dev::telemeter<flipperCtrl>
{
    friend class dev::telemeter<flipperCtrl>;

    typedef dev::telemeter<flipperCtrl> telemeterT;

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    /// Physical endpoint corresponding to the logical in selection.
    int m_inPos{ 1 };

    /// Physical endpoint corresponding to the logical out selection.
    int m_outPos{ 2 };

    ///@}

    /// Last confirmed physical endpoint; zero means none has been observed.
    int m_pos{ 0 };

    /// Requested physical endpoint; zero means no retained target.
    int m_tgt{ 0 };

    /// Whether the last observed endpoint was settled and can be retained without power.
    bool m_parked{ false };

    /// Whether a move has been sent but its target endpoint has not been confirmed.
    bool m_movePending{ false };

    /// Whether the device reports motion or no active endpoint switch.
    bool m_moving{ false };

    /// Whether this connection has supplied a valid live status reply.
    bool m_livePosition{ false };

    /// Retained endpoint to compare with the first settled observation after power-on.
    int m_inferredPos{ 0 };

    /// Whether the last successfully installed state record is known.
    bool m_hasSavedState{ false };

    /// Physical endpoint in the last successfully installed state record.
    int m_savedPos{ 0 };

    /// Parked flag in the last successfully installed state record.
    bool m_savedParked{ false };

    /// Earliest retry of a failed background state write.
    std::chrono::steady_clock::time_point m_nextSave{ std::chrono::steady_clock::time_point::min() };

    /// Position reported by the preceding telemetry record, including the unknown sentinel.
    int m_lastTelemPos{ -1 };

    /// Motion code reported by the preceding telemetry record.
    int m_lastTelemMoving{ -10 };

    /// Logical name reported by the preceding telemetry record.
    std::string m_lastTelemName;

    /// Read-only certification of the retained physical endpoint.
    pcf::IndiProperty m_indiP_parked;

    /// Read the app-specific state record, treating absent or invalid records as unknown.
    /** \returns 0 for a valid or absent file, -1 for an unreadable or invalid file. */
    int readStateFile();

    /// Atomically install and sync an endpoint/parked record.
    /** The temporary file is created beside the record, synced, closed, and renamed before the directory is synced.
     * A failure after rename can leave the replacement installed, but still prevents a hardware command.
     * \returns 0 on success, -1 on a logged filesystem failure.
     */
    int writeStateFile( int  pos /**< [in] last confirmed physical endpoint, or zero */,
                        bool parked /**< [in] whether that endpoint is certified settled */ );

    /// Save changed state, with five-second retries after background failures.
    /** Forced saves bypass both the unchanged-record check and retry delay.
     * An external endpoint change first invalidates any previously parked record. Storage failures are logged;
     * live reporting continues, but recovery can remain stale if that invalidation cannot be written.
     * \returns 0 if saved or unchanged, -1 if failed or waiting to retry.
     */
    int saveState( bool force = false /**< [in] require a fresh durable write immediately */ );

    /// Publish both position switches coherently and update the parked property.
    int publishPosition();

    /// Resolve the physical endpoint currently safe to report.
    /** \returns 1 or 2 for a retained or live pending-move observation, zero for unknown. */
    int reportedPosition();

    /// Decode a complete APT status reply without changing controller state.
    /** Accept channel zero or one and mask the full 32-bit status word. No active endpoint is treated as transit;
     * simultaneous endpoint flags are rejected. Motion flags take precedence over a remaining endpoint flag.
     * \returns 0 on a recognized stationary or moving status, -1 for malformed or contradictory status.
     */
    static int decodePosition( const std::string &response /**< [in] complete 20-byte status reply */,
                               int               &pos /**< [out] settled physical endpoint, or zero in transit */,
                               bool              &moving /**< [out] whether the mechanism is moving */ );

    /// Mark powered-on transport/status failures unknown, retaining a snapshot during a power-off transition.
    /** \returns -1 after logging the failure. */
    int positionError( const std::string &message /**< [in] reason the position query failed */ );

  public:
    /// Construct a power-managed controller with unknown initial position.
    flipperCtrl();

    /// Destroy the controller without throwing.
    ~flipperCtrl() noexcept;

    /// Register USB, serial I/O, reversal, and telemetry configuration.
    void setupConfig() override;

    /// Load configuration while allowing an unplugged USB device.
    /** \returns 0 on success, -1 on configuration failure. */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] configuration from which to load values */ );

    /// Load configuration and request shutdown on failure.
    void loadConfig() override;

    /// Recover retained position, register properties, and start telemetry.
    /** \returns 0 on success, -1 on a fatal startup failure. */
    int appStartup() override;

    /// Discover/connect the device, confirm its position, and save settled endpoints.
    /** \returns 0 on success or a recoverable transport failure, -1 on a fatal failure. */
    int appLogic() override;

    /// Close the connection and publish the retained or unknown OFF snapshot under the state mutex.
    /** The framework sets POWEROFF before calling this hook. No position query or backing-file write is performed.
     * \returns 0 on success.
     */
    int onPowerOff() override;

    /// Maintain OFF publication and telemetry scheduling without hardware or disk I/O.
    /** \returns 0 on success, -1 on telemetry failure. */
    int whilePowerOff() override;

    /// Close the connection and perform telemetry shutdown handling.
    /** \returns 0 on success. */
    int appShutdown() override;

    /// Query and apply a live status; the caller must hold m_indiMutex.
    /** Assemble complete status packets while skipping unsolicited completion notifications. A reply still at the
     * previous endpoint cannot complete a pending move to the other endpoint. The first settled power-on reply
     * consumes the retained comparison, with a WARNING if the live endpoint differs.
     * \returns 0 on a valid status, -1 on transport or decoding failure.
     */
    int getPos();

    /// Acquire the state mutex, durably invalidate parking, and command a physical endpoint.
    /** \returns 0 on success or a settled no-op, -1 for a rejected or failed move. */
    int moveTo( int pos /**< [in] requested physical endpoint, either 1 or 2 */ );

    /// Existing logical in/out selection property exposed as presetName.
    pcf::IndiProperty m_indiP_position;

    /// Validate and process a logical position request.
    /** The callback dispatches to moveTo(), which owns the state lock. */
    int newCallBack_m_indiP_position( const pcf::IndiProperty &ipRecv /**< [in] received position request */ );

    /// Dispatch a position request from the INDI driver.
    static int
    st_newCallBack_m_indiP_position( void                    *app /**< [in] controller receiving the request */,
                                     const pcf::IndiProperty &ipRecv /**< [in] received position request */ );

    /// Check whether stage telemetry has reached its maximum interval.
    int checkRecordTimes();

    /// Force a scheduled stage telemetry record under the caller's state lock.
    int recordTelem( const telem_stage *type /**< [in] unused telemetry type selector */ );

    /// Record changed stage telemetry, using -2 for OFF and -1 for unknown idle position.
    int recordStage( bool force = false /**< [in] record even if values have not changed */ );
};

inline flipperCtrl::flipperCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
}

inline flipperCtrl::~flipperCtrl() noexcept = default;

inline void flipperCtrl::setupConfig()
{
    tty::usbDevice::setupConfig( config );
    dev::ioDevice::setupConfig( config );
    config.add( "flipper.reverse",
                "",
                "flipper.reverse",
                argType::Required,
                "flipper",
                "reverse",
                false,
                "bool",
                "If true, reverse the positions for in and out." );
    TELEMETER_SETUP_CONFIG( config );
}

inline int flipperCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
{
    m_baudRate = B115200;
    int rv     = tty::usbDevice::loadConfig( _config );
    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND )
    {
        log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }
    if( dev::ioDevice::loadConfig( _config ) < 0 )
        return -1;
    bool reverse = false;
    _config( reverse, "flipper.reverse" );
    m_inPos  = reverse ? 2 : 1;
    m_outPos = reverse ? 1 : 2;
    TELEMETER_LOAD_CONFIG( _config );
    return 0;
}

inline void flipperCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        m_shutdown = 1;
    }
}

inline int flipperCtrl::readStateFile()
{
    m_pos = m_tgt = m_inferredPos = 0;
    m_parked = m_movePending = m_moving = m_livePosition = m_hasSavedState = false;
    m_nextSave             = std::chrono::steady_clock::time_point::min();
    const std::string path = m_sysPath + "/" + m_configName + "/position";
    errno                  = 0;
    std::ifstream input( path );
    if( !input )
    {
        if( errno == ENOENT )
            return 0;
        return log<software_error, -1>( { __FILE__, __LINE__, errno, 0, "cannot read flipper state file " + path } );
    }
    int pos = 0, parked = 0;
    if( !( input >> pos >> parked ) || pos < 0 || pos > 2 || parked < 0 || parked > 1 || ( parked && pos == 0 ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid flipper state file " + path } );
    }
    input >> std::ws;
    if( !input.eof() || input.bad() )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "extra or unreadable data in flipper state file " + path } );
    }
    m_pos = m_savedPos = pos;
    m_parked = m_savedParked = ( parked != 0 );
    m_hasSavedState          = true;
    if( m_parked )
        m_tgt = m_inferredPos = pos;
    return 0;
}

inline int flipperCtrl::writeStateFile( int pos, bool parked )
{
    const std::string  directory = m_sysPath + "/" + m_configName;
    const std::string  path      = directory + "/position";
    std::string        temporary = path + ".tmp.XXXXXX";
    const std::string  record    = std::to_string( pos ) + "\n" + ( parked ? "1\n" : "0\n" );
    elevatedPrivileges ep( this );
    int                dir = ::open( directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC );
    if( dir < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, errno, 0, "cannot open flipper state directory " + directory } );
    }
    int fd = ::mkstemp( temporary.data() );
    if( fd < 0 )
    {
        int error = errno;
        ::close( dir );
        return log<software_error, -1>( { __FILE__, __LINE__, error, 0, "cannot create flipper state file " + path } );
    }
    // Files created with elevated privileges must remain readable after privileges are dropped.
    int    error   = ( ::fchmod( fd, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH ) < 0 ) ? errno : 0;
    size_t written = 0;
    while( !error && written < record.size() )
    {
        ssize_t count = ::write( fd, record.data() + written, record.size() - written );
        if( count < 0 && errno == EINTR )
            continue;
        if( count <= 0 )
            error = count < 0 ? errno : EIO;
        else
            written += count;
    }
    if( !error && ::fsync( fd ) < 0 )
        error = errno;
    if( ::close( fd ) < 0 && !error )
        error = errno;
    if( !error && ::rename( temporary.c_str(), path.c_str() ) < 0 )
        error = errno;
    if( !error && ::fsync( dir ) < 0 )
        error = errno;
    if( ::close( dir ) < 0 && !error )
        error = errno;
    if( error )
    {
        ::unlink( temporary.c_str() );
        return log<software_error, -1>(
            { __FILE__, __LINE__, error, 0, "cannot durably store flipper position in " + path } );
    }
    return 0;
}

inline int flipperCtrl::saveState( bool force )
{
    if( !force && m_hasSavedState && m_savedPos == m_pos && m_savedParked == m_parked )
        return 0;
    auto now = std::chrono::steady_clock::now();
    if( !force && now < m_nextSave )
        return -1;
    m_nextSave = now + std::chrono::seconds( 5 );
    // An external endpoint change must invalidate the previous certification before saving the new one.
    if( m_parked && m_hasSavedState && m_savedParked && m_savedPos != m_pos )
    {
        if( writeStateFile( m_pos, false ) < 0 )
            return -1;
        m_savedPos    = m_pos;
        m_savedParked = false;
    }
    if( writeStateFile( m_pos, m_parked ) < 0 )
        return -1;
    m_savedPos      = m_pos;
    m_savedParked   = m_parked;
    m_hasSavedState = true;
    m_nextSave      = std::chrono::steady_clock::time_point::min();
    return 0;
}

inline int flipperCtrl::reportedPosition()
{
    if( m_parked || ( state() != stateCodes::POWEROFF && m_movePending && m_livePosition ) )
        return m_pos;
    return 0;
}

inline int flipperCtrl::publishPosition()
{
    int  pos           = reportedPosition();
    bool off           = state() == stateCodes::POWEROFF;
    auto propertyState = !off && ( m_movePending || m_moving ) ? INDI_BUSY : ( pos ? INDI_IDLE : INDI_ALERT );
    auto in            = pos == m_inPos ? pcf::IndiElement::On : pcf::IndiElement::Off;
    auto out           = pos == m_outPos ? pcf::IndiElement::On : pcf::IndiElement::Off;
    bool changed = m_indiP_position.getState() != propertyState || m_indiP_position["in"].getSwitchState() != in ||
                   m_indiP_position["out"].getSwitchState() != out;
    m_indiP_position["in"].setSwitchState( in );
    m_indiP_position["out"].setSwitchState( out );
    m_indiP_position.setState( propertyState );
    if( changed && m_indiDriver )
    {
        m_indiP_position.setTimeStamp( pcf::TimeStamp() );
        m_indiDriver->sendSetProperty( m_indiP_position );
    }
    bool parkedChanged = m_indiP_parked["current"].get<int>() != static_cast<int>( m_parked );
    m_indiP_parked["current"].set( static_cast<int>( m_parked ) );
    if( parkedChanged && m_indiDriver )
    {
        m_indiP_parked.setTimeStamp( pcf::TimeStamp() );
        m_indiDriver->sendSetProperty( m_indiP_parked );
    }
    return 0;
}

inline int flipperCtrl::appStartup()
{
    readStateFile(); // Bad or absent state is recoverable: a live query will establish the endpoint.
    if( createStandardIndiSelectionSw( m_indiP_position, "presetName", { "in", "out" } ) < 0 ||
        registerIndiPropertyNew( m_indiP_position, st_newCallBack_m_indiP_position ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    createROIndiNumber( m_indiP_parked, "parked", "Parked State" );
    indi::addNumberElement<int>( m_indiP_parked, "current", 0, 1, 1, "%d" );
    m_indiP_parked["current"].set( 0 );
    if( registerIndiPropertyReadOnly( m_indiP_parked ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }
    publishPosition();
    TELEMETER_APP_STARTUP;
    return 0;
}

inline int flipperCtrl::decodePosition( const std::string &response, int &pos, bool &moving )
{
    if( response.size() != 20 )
        return -1;
    const auto *bytes = reinterpret_cast<const unsigned char *>( response.data() );
    if( bytes[0] != 0x81 || bytes[1] != 0x04 || bytes[2] != 14 || bytes[3] != 0 || bytes[4] != 0x81 ||
        bytes[5] != 0x50 || bytes[6] > 1 || bytes[7] != 0 )
        return -1;
    // APT status bits occupy bytes 16..19. Mask endpoints separately from motion and other status flags.
    uint32_t status = 0;
    for( unsigned i = 0; i < 4; ++i )
        status |= static_cast<uint32_t>( bytes[16 + i] ) << ( 8 * i );
    unsigned endpoints = status & 0x03;
    if( endpoints == 3 )
        return -1;
    moving = ( status & 0x02f0 ) != 0 || endpoints == 0;
    pos    = moving ? 0 : static_cast<int>( endpoints );
    return 0;
}

inline int flipperCtrl::positionError( const std::string &message )
{
    if( powerState() == 1 && powerStateTarget() == 1 )
    {
        m_parked = m_livePosition = m_moving = false;
        saveState();
    }
    return log<software_error, -1>( { __FILE__, __LINE__, message } );
}

inline int flipperCtrl::getPos()
{
    if( state() == stateCodes::POWEROFF || powerState() != 1 || powerStateTarget() != 1 )
        return -1;
    if( m_fileDescrip <= 0 )
        return positionError( "flipper position requested without a connection" );
    // Preserve the channel-zero requests used by the installed flippers.
    std::string header( "\x80\x04\x00\x00\x50\x01", 6 );
    if( tty::ttyWrite( header, m_fileDescrip, m_writeTimeout ) < 0 )
    {
        return positionError( "error requesting flipper position" );
    }
    std::string response;
    auto        deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( m_readTimeout );
    size_t      offset   = 0;
    int         pos      = 0;
    bool        moving = false, found = false;
    for( unsigned reads = 0; reads < 8; ++reads )
    {
        int remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>( deadline - std::chrono::steady_clock::now() )
                .count();
        size_t available = response.size() - offset;
        int    needed    = 6 - static_cast<int>( available );
        if( available >= 6 )
        {
            const auto *bytes = reinterpret_cast<const unsigned char *>( response.data() + offset );
            needed            = 6 + bytes[2] + ( static_cast<int>( bytes[3] ) << 8 ) - static_cast<int>( available );
        }
        std::string chunk;
        if( remaining <= 0 || tty::ttyRead( chunk, needed, m_fileDescrip, remaining ) < 0 || chunk.empty() )
        {
            return positionError( "error reading flipper position" );
        }
        response += chunk;
        if( response.size() > 4096 )
            return positionError( "oversized flipper response" );
        while( response.size() - offset >= 6 )
        {
            const auto *bytes  = reinterpret_cast<const unsigned char *>( response.data() + offset );
            size_t      length = ( bytes[4] & 0x80 ) ? 6 + bytes[2] + ( static_cast<size_t>( bytes[3] ) << 8 ) : 6;
            if( length > 1024 )
                return positionError( "invalid flipper response length" );
            if( response.size() - offset < length )
                break;
            unsigned id = bytes[0] | ( static_cast<unsigned>( bytes[1] ) << 8 );
            if( id == 0x0481 )
            {
                if( decodePosition( response.substr( offset, length ), pos, moving ) < 0 )
                {
                    return positionError( "invalid flipper position response" );
                }
                found = true;
            }
            else if( ( id != 0x0464 && id != 0x0466 ) || bytes[4] != 0x81 || bytes[5] != 0x50 )
            {
                return positionError( "unexpected flipper response" );
            }
            offset += length;
        }
        if( found && offset == response.size() )
            break;
    }
    if( !found || offset != response.size() )
        return positionError( "incomplete flipper position response" );
    m_livePosition = true;
    m_moving       = moving;
    if( moving )
    {
        m_parked = false;
    }
    else if( m_movePending && pos != m_tgt )
    {
        // The old limit can remain active briefly after a command; it does not confirm completion.
        m_parked = false;
    }
    else
    {
        if( m_inferredPos )
        {
            if( pos != m_inferredPos )
            {
                log<text_log>( "flipper power-on position " + std::to_string( pos ) +
                                   " differs from inferred parked position " + std::to_string( m_inferredPos ),
                               logPrio::LOG_WARNING );
            }
            m_inferredPos = 0;
        }
        m_pos = m_tgt = pos;
        m_parked      = true;
        m_movePending = false;
    }
    saveState();
    return 0;
}

inline int flipperCtrl::appLogic()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( state() == stateCodes::POWEROFF || powerState() != 1 || powerStateTarget() != 1 )
        return 0;
    if( state() == stateCodes::POWERON )
    {
        m_livePosition = false;
        state( stateCodes::NODEVICE );
    }
    if( state() == stateCodes::NODEVICE )
    {
        int rv = tty::usbDevice::getDeviceName();
        if( rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES )
        {
            state( stateCodes::FAILURE );
            return log<software_critical, -1>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
        }
        if( rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES )
        {
            if( !stateLogged() )
            {
                log<text_log>( "USB Device " + m_idVendor + ":" + m_idProduct + ":" + m_serial + " not found in udev" );
            }
            return 0;
        }
        state( stateCodes::NOTCONNECTED );
        if( !stateLogged() )
        {
            log<text_log>( "USB Device " + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " +
                           m_deviceName );
        }
    }
    if( state() == stateCodes::NOTCONNECTED )
    {
        elevatedPrivileges ep( this );
        int                rv = connect();
        ep.restore();
        if( rv < 0 )
        {
            int nrv = tty::usbDevice::getDeviceName();
            if( nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES )
            {
                state( stateCodes::NODEVICE );
                if( !stateLogged() )
                {
                    log<text_log>( "USB Device " + m_idVendor + ":" + m_idProduct + ":" + m_serial +
                                   " no longer found in udev" );
                }
            }
            else if( nrv < 0 )
            {
                state( stateCodes::FAILURE );
                return log<software_critical, -1>( { __FILE__, __LINE__, nrv, tty::ttyErrorString( nrv ) } );
            }
            return 0;
        }
        state( stateCodes::CONNECTED );
    }
    if( state() == stateCodes::CONNECTED || state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        int rv = getPos();
        if( powerState() != 1 || powerStateTarget() != 1 || state() == stateCodes::POWEROFF )
            return 0;
        if( rv < 0 )
            state( stateCodes::NOTCONNECTED );
        else
            state( m_parked ? stateCodes::READY : stateCodes::OPERATING );
        publishPosition();
        recordStage();
        TELEMETER_APP_LOGIC;
    }
    return 0;
}

inline int flipperCtrl::moveTo( int pos )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( pos != 1 && pos != 2 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid flipper target" } );
    }
    if( ( state() != stateCodes::READY && state() != stateCodes::OPERATING ) || powerState() != 1 ||
        powerStateTarget() != 1 || !m_livePosition || m_fileDescrip <= 0 )
    {
        return log<software_warning, -1>(
            { __FILE__, __LINE__, "flipper move rejected: device is not powered and connected" } );
    }
    if( m_parked && m_pos == pos )
        return 0;
    if( writeStateFile( m_pos, false ) < 0 )
        return -1;
    m_savedPos      = m_pos;
    m_savedParked   = false;
    m_hasSavedState = true;
    m_nextSave      = std::chrono::steady_clock::time_point::min();
    m_parked        = false;
    m_tgt           = pos;
    m_movePending   = true;
    m_inferredPos   = 0;
    if( powerState() != 1 || powerStateTarget() != 1 || state() == stateCodes::POWEROFF )
        return -1;
    std::string header( "\x6a\x04\x00\x00\x50\x01", 6 );
    header[3] = static_cast<char>( pos );
    int rv    = tty::ttyWrite( header, m_fileDescrip, m_writeTimeout );
    if( rv < 0 )
    {
        m_livePosition = false;
        state( stateCodes::NOTCONNECTED );
        publishPosition();
        recordStage( true );
        return log<software_error, -1>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }
    state( stateCodes::OPERATING );
    publishPosition();
    recordStage( true );
    return 0;
}

inline int flipperCtrl::newCallBack_m_indiP_position( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_position, ipRecv );
    bool in  = ipRecv.find( "in" ) && ipRecv["in"].getSwitchState() == pcf::IndiElement::On;
    bool out = ipRecv.find( "out" ) && ipRecv["out"].getSwitchState() == pcf::IndiElement::On;
    if( in && out )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "cannot set flipper position to both in and out" } );
    }
    if( in || out )
        return moveTo( in ? m_inPos : m_outPos );
    return 0;
}

inline int flipperCtrl::st_newCallBack_m_indiP_position( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<flipperCtrl *>( app )->newCallBack_m_indiP_position( ipRecv );
}

inline int flipperCtrl::onPowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    m_movePending = m_moving = m_livePosition = false;
    m_tgt = m_inferredPos = m_parked ? m_pos : 0;
    publishPosition();
    recordStage( true );
    return 0;
}

inline int flipperCtrl::whilePowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    publishPosition();
    recordStage();
    TELEMETER_APP_LOGIC;
    return 0;
}

inline int flipperCtrl::appShutdown()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline int flipperCtrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_stage() );
}

inline int flipperCtrl::recordTelem( const telem_stage * )
{
    return recordStage( true );
}

inline int flipperCtrl::recordStage( bool force )
{
    int    pos       = reportedPosition();
    int8_t moving    = state() == stateCodes::POWEROFF ? -2 : ( m_movePending || m_moving ? 1 : ( m_parked ? 0 : -1 ) );
    std::string name = pos == m_inPos ? "in" : ( pos == m_outPos ? "out" : "" );
    if( force || pos != m_lastTelemPos || moving != m_lastTelemMoving || name != m_lastTelemName )
    {
        telem<telem_stage>( { moving, static_cast<float>( pos ), name } );
        m_lastTelemPos    = pos;
        m_lastTelemMoving = moving;
        m_lastTelemName   = name;
    }
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // flipperCtrl_hpp
