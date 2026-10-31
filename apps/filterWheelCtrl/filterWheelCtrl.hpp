/** \file filterWheelCtrl.hpp
 * \brief The MagAO-X Filter Wheel Controller
 *
 * \ingroup filterWheelCtrl_files
 */

#ifndef filterWheelCtrl_hpp
#define filterWheelCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <atomic>
#include <cstdint>

/** \defgroup filterWheelCtrl Filter Wheel Control
 * \brief Control of MagAO-X MCBL-based f/w.
 *
 * <a href="../handbook/operating/software/apps/filterWheelCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup filterWheelCtrl_files Filter Wheel Control Files
 * \ingroup filterWheelCtrl
 */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a Faulhaber MCBL controlled filter wheel.
 *
 * \todo add temperature monitoring
 * \todo add INDI props to md doc
 * \todo should move in least time direction, rather than always in the same direction.
 * \todo should be an iodevice
 *
 * \ingroup filterWheelCtrl
 */
class filterWheelCtrl : public MagAOXApp<>,
                        public tty::usbDevice,
                        public dev::stdMotionStage<filterWheelCtrl>,
                        public dev::telemeter<filterWheelCtrl>
{

    friend class dev::stdMotionStage<filterWheelCtrl>;

    friend class dev::telemeter<filterWheelCtrl>;

  protected:
    /** \name Non-configurable Parameters - Data
     *@{
     */

    /// Motor type sent to the MCBL controller at each connection.
    int m_motorType{ 2 };

    ///@}

    /** \name Configurable Parameters - Data
     * @{
     */

    /// The timeout for writing to the device [msec].
    int m_writeTimeOut{ 1000 };

    /// The timeout for reading from the device [msec].
    int m_readTimeOut{ 1000 };

    /// Acceleration configured on the motor controller.
    double m_acceleration{ 100 };

    /// Deceleration configured on the motor controller.
    double m_deceleration{ 50 };

    /// Motor speed used for movement and homing.
    double m_motorSpeed{ 3000 };

    /// The number of position counts in one 360-degree revolution.
    long m_circleSteps{ 0 };

    /// The number of position counts to offset from the home position.
    long m_homeOffset{ 0 };

    ///@}

    /** \name Status - Data
     * @{
     */

    /// Last successfully read home switch status.
    bool m_switch{ false };

    /// Last successfully read wheel position in motor counts.
    long m_rawPos{ 0 };

    /// Last successfully read motor speed, used to require exact zero speed for READY recovery.
    long m_rawSpeed{ 0 };

    /// Progress through the multi-step homing sequence; zero means no sequence is active.
    int m_homingState{ 0 };

    /// Permit automatic homing on the first connection after startup or observed power-on.
    bool m_powerOnHomePending{ false };

    /// Raw counts from the last fully validated stationary READY poll.
    long m_readyRawPos{ 0 };

    /// Whether the current READY state has a confirmed stationary position.
    bool m_readyPositionValid{ false };

    /// Preserve a validated READY position across communication recovery attempts.
    bool m_restoreReadyPending{ false };

    /// Generation of the power and command history associated with m_readyRawPos.
    uint64_t m_readyGeneration{ 0 };

    /// Invalidation generation shared with asynchronous power callbacks and immediate stop requests.
    std::atomic<uint64_t> m_readyInvalidation{ 0 };

    /// Latched power interruption prevents READY snapshots until a new homing sequence is started.
    std::atomic<bool> m_readyPowerInterrupted{ false };

    ///@}

  public:
    /// Default c'tor.
    filterWheelCtrl();

    /// D'tor, declared and defined for noexcept.
    ~filterWheelCtrl() noexcept
    {
    }

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Set up the INDI properties and telemetry.
    /**
     *
     * \returns 0 on success
     * \returns -1 on error.
     */
    virtual int appStartup();

    /// Run the filter wheel FSM, including communication recovery.
    /**
     * \returns 0 on success
     * \returns -1 on error.
     */
    virtual int appLogic();

    /// Release the serial connection and run telemetry shutdown.
    /**
     * \returns 0 on success
     * \returns -1 on error.
     */
    virtual int appShutdown();

    /// This method is called when the change to poweroff is detected.
    /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
    virtual int onPowerOff();

    /// This method is called while the power is off, once per FSM loop.
    /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
    virtual int whilePowerOff();

  protected:
    // declare our properties

    /// The position of the wheel in counts
    pcf::IndiProperty m_indiP_counts;

  public:
    /// Handle an absolute raw-count position request.
    int newCallBack_m_indiP_counts( const pcf::IndiProperty &ipRecv /**< [in] Requested INDI property. */ );

    /// Dispatch a raw-count request to its owning application.
    static int st_newCallBack_m_indiP_counts( void                    *app /**< [in] Owning application. */,
                                              const pcf::IndiProperty &ipRecv /**< [in] Requested INDI property. */ );

    /// Invalidate READY recovery on a power interruption and run the base power callback.
    int setCallBack_m_indiP_powerChannel( const pcf::IndiProperty &ipRecv /**< [in] Power channel update. */ );

    /// Dispatch a power update to its owning filter wheel controller.
    static int
    st_setCallBack_m_indiP_powerChannel( void                    *app /**< [in] Owning application. */,
                                         const pcf::IndiProperty &ipRecv /**< [in] Power channel update. */ );

  protected:
    /// Check whether power, FSM state, and the serial descriptor permit communication.
    bool deviceAvailable();

    /// Release the serial port and clear interrupted motion; caller holds m_indiMutex.
    void resetConnection();

    /// Restart USB discovery without automatic homing; caller holds m_indiMutex.
    int recoverFromError();

    /// Write a command and recover on transport failure; caller holds m_indiMutex.
    int writeCommand( const std::string &command /**< [in] Complete controller command, including terminator. */ );

    /// Read an integer without changing its value on failure; caller holds m_indiMutex.
    int readInteger( long              &value /**< [out] Validated reply value on success. */,
                     const std::string &command /**< [in] Complete query command, including terminator. */ );

    // Device methods below require m_indiMutex, except stop(), which acquires it itself.

    /// Set up the MCBL controller, called after each power-on/connection
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int onPowerOnConnect();

    /// Get the home switch status, sets m_switch to true or false.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getSwitch();

    /// Read motion status: m_moving is zero when stopped, one when moving, or two when homing.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getMoving();

    /// Get the current position of the wheel, sets m_rawPos to the current motor counts.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getPos();

    /// Start a high-level homing sequence.
    /** For this device this includes the homing dither.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int startHoming();

    /// Return the nearest filter index, wrapping around the wheel.
    int presetNumber();

    /// Start a low-level homing sequence.
    /** This initiates the device homing sequence.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int home();

    /// Stop the wheel motion immediately.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int stop();

    /// Move to an absolute position in raw counts.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int moveToRaw( const long &counts /**< [in] The new position in absolute motor counts*/ );

    /// Move to a new position relative to current, in raw counts.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int moveToRawRelative( const long &counts_relative /**< [in] The displacement in motor counts. */ );

    /// Move to an absolute position in filter units.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int moveTo( const double &filters /**< [in] The new position in absolute filter units*/ );

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether stage or position telemetry is due.
    int checkRecordTimes();

    /// Force a stage telemetry record.
    int recordTelem( const telem_stage *type /**< [in] Stage telemetry selector. */ );

    /// Force a raw-position telemetry record.
    int recordTelem( const telem_position *type /**< [in] Position telemetry selector. */ );

    /// Record stage telemetry when changed or explicitly requested.
    int recordStage( bool force = false /**< [in] Record even if the stage status is unchanged. */ );

    /// Record raw-position telemetry when changed or explicitly requested.
    int recordPosition( bool force = false /**< [in] Record even if the position is unchanged. */ );

    ///@}
};

inline filterWheelCtrl::filterWheelCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_presetNotation = "filter"; // sets the name of the configs, etc.

    m_powerMgtEnabled = true;

    return;
}

inline void filterWheelCtrl::setupConfig()
{
    tty::usbDevice::setupConfig( config );

    config.add( "timeouts.write",
                "",
                "timeouts.write",
                argType::Required,
                "timeouts",
                "write",
                false,
                "int",
                "The timeout for writing to the device [msec]. Default = 1000" );
    config.add( "timeouts.read",
                "",
                "timeouts.read",
                argType::Required,
                "timeouts",
                "read",
                false,
                "int",
                "The timeout for reading the device [msec]. Default = 1000" );

    config.add( "motor.acceleration",
                "",
                "motor.acceleration",
                argType::Required,
                "motor",
                "acceleration",
                false,
                "real",
                "The motor acceleration parameter. Default=100." );
    config.add( "motor.deceleration",
                "",
                "motor.deceleration",
                argType::Required,
                "motor",
                "deceleration",
                false,
                "real",
                "The motor deceleration parameter. Default=50." );
    config.add( "motor.speed",
                "",
                "motor.speed",
                argType::Required,
                "motor",
                "speed",
                false,
                "real",
                "The motor speed parameter.  Default=3000." );
    config.add( "motor.circleSteps",
                "",
                "motor.circleSteps",
                argType::Required,
                "motor",
                "circleSteps",
                false,
                "long",
                "The number of steps in 1 revolution." );
    config.add( "stage.homeOffset",
                "",
                "stage.homeOffset",
                argType::Required,
                "stage",
                "homeOffset",
                false,
                "long",
                "The homing offset in motor counts." );

    dev::stdMotionStage<filterWheelCtrl>::setupConfig( config );

    dev::telemeter<filterWheelCtrl>::setupConfig( config );
}

inline void filterWheelCtrl::loadConfig()
{
    this->m_baudRate = B9600; // default for MCBL controller.  Will be overridden by any config setting.

    int rv = tty::usbDevice::loadConfig( config );

    if( rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND ) // Ignore error if not plugged in
    {
        log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }

    config( m_writeTimeOut, "timeouts.write" );
    config( m_readTimeOut, "timeouts.read" );

    config( m_acceleration, "motor.acceleration" );
    config( m_deceleration, "motor.deceleration" );
    config( m_motorSpeed, "motor.speed" );
    config( m_circleSteps, "motor.circleSteps" );
    config( m_homeOffset, "stage.homeOffset" );

    dev::stdMotionStage<filterWheelCtrl>::loadConfig( config );

    dev::telemeter<filterWheelCtrl>::loadConfig( config );
}

inline int filterWheelCtrl::appStartup()
{
    if( state() == stateCodes::UNINITIALIZED )
    {
        log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    // Reuse the base subscription before the INDI receive thread starts.
    if( m_powerMgtEnabled )
    {
        std::lock_guard<std::mutex> guard( m_indiCallBackMutex );
        auto                        callback = m_indiSetCallBacks.find( m_indiP_powerChannel.createUniqueKey() );
        if( callback == m_indiSetCallBacks.end() )
            return log<software_error, -1>( { __FILE__, __LINE__, "Power channel callback is not registered." } );
        callback->second.callBack = st_setCallBack_m_indiP_powerChannel;
    }

    // set up the  INDI properties
    createStandardIndiNumber<long>(
        m_indiP_counts, "counts", std::numeric_limits<long>::lowest(), std::numeric_limits<long>::max(), 0.0, "%ld" );
    registerIndiPropertyNew( m_indiP_counts, INDI_NEWCALLBACK( m_indiP_counts ) );

    dev::stdMotionStage<filterWheelCtrl>::appStartup();

    if( dev::telemeter<filterWheelCtrl>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int filterWheelCtrl::appLogic()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    if( powerState() != 1 || powerStateTarget() != 1 || state() == stateCodes::POWEROFF )
    {
        m_powerOnHomePending = false;
        m_readyPositionValid = m_restoreReadyPending = false;
        return 0;
    }

    if( state() == stateCodes::INITIALIZED )
    {
        log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
        return -1;
    }

    if( state() == stateCodes::POWERON )
    {
        resetConnection();
        m_powerOnHomePending = m_powerOnHome;
        state( stateCodes::NODEVICE );
    }

    if( state() == stateCodes::ERROR )
        return recoverFromError();

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
        log<text_log>( "USB Device " + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " +
                       m_deviceName );
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        if( powerState() != 1 || powerStateTarget() != 1 )
            return recoverFromError();

        int rv;
        {
            elevatedPrivileges elPriv( this );
            rv = connect();
        }
        if( rv < 0 )
        {
            log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            return recoverFromError();
        }

        if( getPos() != 0 )
            return 0;

        if( powerState() != 1 || powerStateTarget() != 1 )
            return recoverFromError();

        state( stateCodes::CONNECTED );
        log<text_log>( "Connected to filter wheel on " + m_deviceName );
    }

    if( state() == stateCodes::CONNECTED )
    {
        if( onPowerOnConnect() < 0 )
            return 0;

        std::string landfill;
        // A timeout is expected when draining stale input; other transport errors require recovery.
        int rv = tty::ttyRead( landfill, "\r", m_fileDescrip, m_readTimeOut );
        if( rv != 0 && rv != TTY_E_TIMEOUTONREAD && rv != TTY_E_TIMEOUTONREADPOLL )
        {
            log<software_error>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
            return recoverFromError();
        }
        if( !deviceAvailable() )
            return recoverFromError();

        bool autoHome        = m_powerOnHomePending;
        m_powerOnHomePending = false; // An interrupted automatic home must not be restarted after recovery.
        if( autoHome )
        {
            if( startHoming() < 0 )
                return 0;
        }
        else
        {
            if( !m_restoreReadyPending )
                state( stateCodes::NOTHOMED );
            m_moving = -1;
        }
    }

    if( state() == stateCodes::CONNECTED || state() == stateCodes::NOTHOMED || state() == stateCodes::READY ||
        state() == stateCodes::OPERATING || state() == stateCodes::HOMING )
    {
        uint64_t generation = m_readyInvalidation.load();
        if( getSwitch() != 0 || getMoving() != 0 || getPos() != 0 )
            return 0;
        if( !deviceAvailable() )
            return 0;
        recordPosition();

        if( state() == stateCodes::CONNECTED )
        {
            if( m_restoreReadyPending && m_readyGeneration == generation && generation == m_readyInvalidation.load() &&
                !m_readyPowerInterrupted.load() && m_rawSpeed == 0 && m_rawPos == m_readyRawPos )
            {
                state( stateCodes::READY );
                log<text_log>( "Recovered READY with unchanged stationary position." );
            }
            else
                state( stateCodes::NOTHOMED );
            m_restoreReadyPending = false;
        }

        if( m_moving > 0 )
        {
            if( state() == stateCodes::READY )
                state( stateCodes::OPERATING );
        }
        else if( state() == stateCodes::OPERATING )
        {
            if( m_movingState == 1 )
            {
                if( moveToRawRelative( -20000 ) < 0 )
                    return 0;
                m_movingState = 2;
            }
            else if( m_movingState == 2 )
            {
                if( moveTo( m_preset_target ) < 0 )
                    return 0;
                m_movingState = 3;
            }
            else
            {
                m_movingState = 0;
                state( stateCodes::READY );
            }
        }
        else if( state() == stateCodes::HOMING )
        {
            if( m_homingState == 1 )
            {
                if( moveToRawRelative( -50000 ) < 0 )
                    return 0;
                m_homingState = 2;
            }
            else if( m_homingState == 2 )
            {
                if( home() < 0 )
                    return 0;
                m_homingState = 3;
            }
            else if( m_homingState == 3 )
            {
                if( moveToRaw( m_homeOffset ) < 0 )
                    return 0;
                m_homingState = 4;
            }
            else if( m_homingState == 4 )
            {
                if( m_homePreset >= 0 )
                {
                    m_preset_target = m_presetPositions[m_homePreset];
                    updateIfChanged( m_indiP_preset, "target", m_preset_target, INDI_BUSY );
                    if( moveTo( m_preset_target ) < 0 )
                        return 0;
                }
                m_homingState = 5;
            }
            else
            {
                m_homingState = 0;
                state( stateCodes::READY );
                m_preset_target = ( (double)m_rawPos - m_homeOffset ) / m_circleSteps * m_presetNames.size() + 1.0;
            }
        }

        updateIfChanged( m_indiP_counts, "current", m_rawPos, m_moving > 0 ? INDI_BUSY : INDI_IDLE );
        m_preset             = ( (double)m_rawPos - m_homeOffset ) / m_circleSteps * m_presetNames.size() + 1.0;
        m_readyPositionValid = state() == stateCodes::READY && m_rawSpeed == 0 && m_moving == 0 && m_homingState == 0 &&
                               m_movingState == 0 && generation == m_readyInvalidation.load() &&
                               !m_readyPowerInterrupted.load();
        if( m_readyPositionValid )
        {
            m_readyRawPos     = m_rawPos;
            m_readyGeneration = generation;
        }
        stdMotionStage<filterWheelCtrl>::updateINDI();
        recordStage();

        if( telemeter<filterWheelCtrl>::appLogic() < 0 )
            log<software_error>( { __FILE__, __LINE__ } );
        return 0;
    }

    return log<software_error, -1>( { __FILE__, __LINE__, "appLogic fell through. Terminating." } );
}

inline int filterWheelCtrl::appShutdown()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    resetConnection();
    if( telemeter<filterWheelCtrl>::appShutdown() < 0 )
        log<software_error>( { __FILE__, __LINE__ } );
    return 0;
}

inline int filterWheelCtrl::onPowerOff()
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    m_readyPowerInterrupted = true;
    resetConnection();
    if( stdMotionStage<filterWheelCtrl>::onPowerOff() < 0 )
        log<software_error>( { __FILE__, __LINE__ } );
    recordStage( true );
    recordPosition( true );
    return 0;
}

inline int filterWheelCtrl::whilePowerOff()
{
    if( stdMotionStage<filterWheelCtrl>::whilePowerOff() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    // record telem if it's been longer than 10 sec:
    if( telemeter<filterWheelCtrl>::appLogic() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int filterWheelCtrl::newCallBack_m_indiP_counts( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_counts, ipRecv );

    double counts     = -1;
    double target_abs = -1;
    if( ipRecv.find( "current" ) )
    {
        counts = ipRecv["current"].get<double>();
    }
    if( ipRecv.find( "target" ) )
    {
        target_abs = ipRecv["target"].get<double>();
    }
    if( target_abs == -1 )
        target_abs = counts;

    std::lock_guard<std::mutex> guard( m_indiMutex );
    ++m_readyInvalidation;
    if( !deviceAvailable() )
        return -1;
    clearPresetNameTracking();
    m_movingState   = 0;
    m_preset_target = ( (double)target_abs - m_homeOffset ) / m_circleSteps * m_presetNames.size() + 1.0;
    return moveToRaw( target_abs );
}

inline int filterWheelCtrl::st_newCallBack_m_indiP_counts( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<filterWheelCtrl *>( app )->newCallBack_m_indiP_counts( ipRecv );
}

inline int filterWheelCtrl::setCallBack_m_indiP_powerChannel( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_powerChannel, ipRecv );
    int rv = MagAOXApp<>::setCallBack_m_indiP_powerChannel( ipRecv );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        ++m_readyInvalidation;
        m_readyPowerInterrupted = true;
    }
    return rv;
}

inline int filterWheelCtrl::st_setCallBack_m_indiP_powerChannel( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<filterWheelCtrl *>( app )->setCallBack_m_indiP_powerChannel( ipRecv );
}

inline bool filterWheelCtrl::deviceAvailable()
{
    return powerState() == 1 && powerStateTarget() == 1 && m_fileDescrip > 0 &&
           ( state() == stateCodes::CONNECTED || state() == stateCodes::NOTHOMED || state() == stateCodes::READY ||
             state() == stateCodes::OPERATING || state() == stateCodes::HOMING );
}

inline void filterWheelCtrl::resetConnection()
{
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    m_deviceName.clear();
    m_homingState = m_movingState = 0;
    m_moving                      = -1;
    m_switch                      = false;
    m_powerOnHomePending          = false;
    m_readyPositionValid = m_restoreReadyPending = false;
    clearPresetNameTracking();
}

inline int filterWheelCtrl::recoverFromError()
{
    bool restoreReady = ( m_restoreReadyPending || ( state() == stateCodes::READY && m_readyPositionValid ) ) &&
                        m_readyGeneration == m_readyInvalidation.load() && m_rawSpeed == 0 && m_moving <= 0 &&
                        m_homingState == 0 && m_movingState == 0 && state() != stateCodes::OPERATING &&
                        state() != stateCodes::HOMING && powerState() == 1 && powerStateTarget() == 1 &&
                        !m_readyPowerInterrupted.load();
    resetConnection();
    m_restoreReadyPending = restoreReady;
    if( state() != stateCodes::POWEROFF )
        state( stateCodes::NODEVICE );
    updateIfChanged( m_indiP_counts, "current", m_rawPos, INDI_ALERT );
    updateIfChanged( m_indiP_preset, "current", m_preset, INDI_ALERT );
    recordStage( true );
    return 0;
}

inline int filterWheelCtrl::writeCommand( const std::string &command )
{
    if( !deviceAvailable() )
    {
        recoverFromError();
        return -1;
    }
    int rv = tty::ttyWrite( command, m_fileDescrip, m_writeTimeOut );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        recoverFromError();
        return -1;
    }
    if( rv < 0 )
    {
        recoverFromError();
        return log<software_error, -1>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }
    return 0;
}

inline int filterWheelCtrl::readInteger( long &value, const std::string &command )
{
    if( powerState() != 1 || powerStateTarget() != 1 || m_fileDescrip <= 0 )
    {
        recoverFromError();
        return -1;
    }
    std::string response;
    int rv = tty::ttyWriteRead( response, command, "\r\n", false, m_fileDescrip, m_writeTimeOut, m_readTimeOut );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        recoverFromError();
        return -1;
    }
    if( rv != 0 )
    {
        recoverFromError();
        return log<software_error, -1>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }

    long   parsed;
    size_t consumed;
    try
    {
        parsed = std::stol( response, &consumed );
    }
    catch( const std::exception & )
    {
        recoverFromError();
        return log<software_error, -1>( { __FILE__, __LINE__, "Invalid integer reply to " + command } );
    }
    if( response.find_first_not_of( " \t\r\n", consumed ) != std::string::npos )
    {
        recoverFromError();
        return log<software_error, -1>( { __FILE__, __LINE__, "Invalid integer reply to " + command } );
    }
    value = parsed;
    return 0;
}

inline int filterWheelCtrl::onPowerOnConnect()
{
    if( writeCommand( "ANSW0\r" ) < 0 || writeCommand( "MOTTYP" + std::to_string( m_motorType ) + "\r" ) < 0 ||
        writeCommand( "AC" + std::to_string( m_acceleration ) + "\r" ) < 0 ||
        writeCommand( "DEC" + std::to_string( m_deceleration ) + "\r" ) < 0 ||
        writeCommand( "SP" + std::to_string( m_motorSpeed ) + "\r" ) < 0 )
        return -1;
    return 0;
}

inline int filterWheelCtrl::getSwitch()
{
    if( !deviceAvailable() )
    {
        recoverFromError();
        return -1;
    }
    std::string response;
    int rv = tty::ttyWriteRead( response, "GAST\r", "\r\n", false, m_fileDescrip, m_writeTimeOut, m_readTimeOut );
    if( powerState() != 1 || powerStateTarget() != 1 )
    {
        recoverFromError();
        return -1;
    }
    if( rv != 0 )
    {
        recoverFromError();
        return log<software_error, -1>( { __FILE__, __LINE__, rv, tty::ttyErrorString( rv ) } );
    }
    m_switch = response == "1011";
    return 0;
}

inline int filterWheelCtrl::getMoving()
{
    long speed;
    if( readInteger( speed, "GN\r" ) < 0 )
        return -1;
    m_rawSpeed = speed;
    if( speed != 0 )
        m_readyPositionValid = m_restoreReadyPending = false;
    if( std::fabs( static_cast<double>( speed ) ) > 0.1 * m_motorSpeed )
        m_moving = m_homingState ? 2 : 1;
    else
        m_moving = 0;
    return 0;
}

inline int filterWheelCtrl::getPos()
{
    int rv = readInteger( m_rawPos, "POS\r" );
    if( rv == 0 && m_restoreReadyPending && m_rawPos != m_readyRawPos )
        m_restoreReadyPending = false;
    return rv;
}

inline int filterWheelCtrl::startHoming()
{
    ++m_readyInvalidation;
    bool powerInterrupted = m_readyPowerInterrupted.exchange( false );
    if( !deviceAvailable() )
    {
        if( powerInterrupted )
            m_readyPowerInterrupted = true;
        return -1;
    }
    m_homingState = 1;
    m_moving      = 2;
    recordStage( true );
    recordPosition( true );
    updateSwitchIfChanged( m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE );
    return home();
}

inline int filterWheelCtrl::presetNumber()
{
    // First we calculate current preset name
    int n = floor( m_preset + 0.5 ) - 1;
    if( n < 0 )
    {
        while( n < 0 )
            n += m_presetNames.size();
    }
    if( n > (long)m_presetNames.size() - 1 )
    {
        while( n > (long)m_presetNames.size() - 1 )
            n -= m_presetNames.size();
    }

    if( n < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error calculating " + m_presetNotation + " index, n < 0" } );
        return -1;
    }

    return n;
}

inline int filterWheelCtrl::home()
{
    if( !deviceAvailable() )
        return -1;
    state( stateCodes::HOMING );
    if( writeCommand( "EN\r" ) < 0 || writeCommand( "HA4\r" ) < 0 || writeCommand( "HL4\r" ) < 0 ||
        writeCommand( "CAHOSEQ\r" ) < 0 || writeCommand( "HP0\r" ) < 0 ||
        writeCommand( "HOSP" + std::to_string( m_motorSpeed ) + "\r" ) < 0 || writeCommand( "GOHOSEQ\r" ) < 0 )
        return -1;
    return 0;
}

inline int filterWheelCtrl::stop()
{
    ++m_readyInvalidation;
    // First try without locking, so a stop can interrupt a pending status read.
    if( deviceAvailable() )
        tty::ttyWrite( "DI\r", m_fileDescrip, m_writeTimeOut );

    std::lock_guard<std::mutex> guard( m_indiMutex );
    if( !deviceAvailable() )
        return -1;
    m_homingState = 0;
    int rv        = writeCommand( "DI\r" );
    updateSwitchIfChanged( m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE );
    return rv;
}

inline int filterWheelCtrl::moveToRaw( const long &counts )
{
    ++m_readyInvalidation;
    if( !deviceAvailable() )
        return -1;
    m_moving = 1;
    recordStage( true );
    recordPosition( true );
    if( writeCommand( "EN\r" ) < 0 || writeCommand( "LA" + std::to_string( counts ) + "\r" ) < 0 ||
        writeCommand( "M\r" ) < 0 )
        return -1;
    updateIfChanged( m_indiP_counts, "target", counts, INDI_BUSY );
    return 0;
}

inline int filterWheelCtrl::moveToRawRelative( const long &counts_relative )
{
    ++m_readyInvalidation;
    if( !deviceAvailable() )
        return -1;
    m_moving = 1;
    recordStage( true );
    recordPosition( true );
    if( writeCommand( "EN\r" ) < 0 || writeCommand( "LR" + std::to_string( counts_relative ) + "\r" ) < 0 ||
        writeCommand( "M\r" ) < 0 )
        return -1;
    return 0;
}

inline int filterWheelCtrl::moveTo( const double &filters )
{
    long counts;

    if( m_circleSteps == 0 || m_presetNames.size() == 0 )
        counts = filters;
    else
        counts = m_homeOffset + m_circleSteps / m_presetNames.size() * ( filters - 1 );

    return moveToRaw( counts );
}

inline int filterWheelCtrl::checkRecordTimes()
{
    return dev::telemeter<filterWheelCtrl>::checkRecordTimes( telem_stage(), telem_position() );
}

inline int filterWheelCtrl::recordTelem( const telem_stage * )
{
    return recordStage( true );
}

inline int filterWheelCtrl::recordTelem( const telem_position * )
{
    return recordPosition( true );
}

inline int filterWheelCtrl::recordStage( bool force )
{
    return dev::stdMotionStage<filterWheelCtrl>::recordStage( force );
}

inline int filterWheelCtrl::recordPosition( bool force )
{
    static long last_pos = 999999999999;

    if( m_rawPos != last_pos || force )
    {
        float fpos = m_rawPos;

        telem<telem_position>( fpos );

        last_pos = m_rawPos;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // filterWheelCtrl_hpp
