/** \file kcubeCtrl.hpp
 * \brief Two-axis K-Cube piezo control with power-aware communication recovery.
 * \ingroup kcubeCtrl_files
 */
#ifndef kcubeCtrl_hpp
#define kcubeCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" // Included on the command line to trigger the PCH.
#include "../../magaox_git_version.h"
#include <mx/math/floatUtils.hpp>
#include "tmcController.hpp"

#ifndef TMCC_IO_RECOVERY_VERSION
#error "kcubeCtrl requires tmcController with bounded I/O recovery (see agents/plans/2026-10/kcubeCtrl-power-recovery.md)."
#endif

/** \defgroup kcubeCtrl
 * \brief The K-Cube Controller application.
 * <a href="../handbook/operating/software/apps/kcubeCtrl.html">Application Documentation</a>
 * \ingroup apps
 */
/** \defgroup kcubeCtrl_files
 * \ingroup kcubeCtrl
 */

namespace MagAOX
{
namespace app
{

/// Capture transport diagnostics until the app can classify the power state.
template<class parentT>
class tmcCon : public tmcController
{
protected:
    /// Owning app; never deleted by the transport.
    parentT *m_parent{nullptr};
    /// Most recent transport failure, retained before cleanup changes FTDI diagnostics.
    std::string m_error;

public:
    /// Attach the application supplying cancellation state.
    void owner( parentT *parent /**< [in] Owning app. */ );

    /// Check power and the connection generation before each dependency I/O step.
    bool communicationAllowed() const override;

    /// Read the most recent transport diagnostic.
    const std::string &error() const;

    /// Clear the diagnostic before an operation.
    void clearError();

    /// Capture an FTDI error without prematurely logging an expected power-off failure.
    void ftdiErrmsg( const std::string &src /**< [in] Operation. */,
                     const std::string &msg /**< [in] Failure description. */,
                     int rv /**< [in] FTDI result. */,
                     const std::string &file /**< [in] Source file. */,
                     int line /**< [in] Source line. */ ) override;

    /// Capture a protocol or lifecycle error.
    void otherErrmsg( const std::string &src /**< [in] Operation. */,
                      const std::string &msg /**< [in] Failure description. */,
                      const std::string &file /**< [in] Source file. */,
                      int line /**< [in] Source line. */ ) override;
};

/// Coordinate both piezo controllers without replaying interrupted commands.
/** \ingroup kcubeCtrl */
class kcubeCtrl : public MagAOXApp<true>
{
protected:
    /** \name Connection Configuration - Data
     * @{ */
    /// Minimum interval between connection attempts, in milliseconds.
    unsigned m_retryInterval{1000};
    /// Overall response deadline and individual USB control timeout, in milliseconds.
    unsigned m_ioTimeout{1000};
    ///@}

    /// Owned axis-1 controller; all access is serialized by m_indiMutex.
    tmcCon<kcubeCtrl> m_kAxis1;
    /// Last successfully commanded axis-1 enable state; invalid while disconnected.
    bool m_axis1Enabled{false};
    /// Owned axis-2 controller; all access is serialized by m_indiMutex.
    tmcCon<kcubeCtrl> m_kAxis2;
    /// Last successfully commanded axis-2 enable state; invalid while disconnected.
    bool m_axis2Enabled{false};
    /// True only after every step of the most recent set command succeeds.
    bool m_isSet{false};
    /// Shutdown admission flag, protected by m_indiMutex.
    bool m_stopping{false};
    /// Atomic actual/target snapshot: two bits each encode state+1; upper bits count interruptions.
    std::atomic<uint64_t> m_powerStatus{0};
    /// Power snapshot under which the current connection attempt began, protected by m_indiMutex.
    uint64_t m_connectionPower{0};
    /// Earliest next connection attempt, including configured power-on delay.
    std::chrono::steady_clock::time_point m_nextAttempt{};
    /// Last logged recovery error, suppressing repeated identical failures while retrying.
    std::string m_lastError;

    /// Release both controllers and invalidate published state; caller holds m_indiMutex.
    void resetConnection( pcf::IndiProperty::PropertyStateType status /**< [in] Idle for Off, Alert for uncertainty. */ );

    /// Abandon the active sequence and schedule recovery; returns -1 to stop its caller's sequence.
    int recover( tmcCon<kcubeCtrl> &axis /**< [in] Failed controller. */,
                 const char *operation /**< [in] Failed operation. */,
                 int result /**< [in] Transport result, or cancellation. */ );

    /// Execute one transport step with power checks before and after it; caller holds m_indiMutex.
    template<class operationT>
    int transact( tmcCon<kcubeCtrl> &axis /**< [in] Controller to use. */,
                   const char *name /**< [in] Diagnostic operation name. */,
                   operationT operation /**< [in] Transport operation returning an integer result. */ );

    /// Require initialized controllers and a commandable state after locking.
    bool commandAllowed();

    /// Initialize either axis using the existing disabled-output startup policy.
    int initialize( tmcCon<kcubeCtrl> &axis /**< [in] Controller to configure. */,
                    bool &enabled /**< [out] Last commanded enable state. */ );

    /// Command an axis enable state and update its local state only on success.
    int enable( tmcCon<kcubeCtrl> &axis /**< [in] Controller to command. */,
                bool &enabled /**< [out] Last commanded enable state. */,
                bool value /**< [in] Requested enable state. */ );

    /// Clamp a finite voltage and write it to one axis.
    int voltage( tmcCon<kcubeCtrl> &axis /**< [in] Controller to command. */,
                 float &value /**< [in/out] Requested voltage, clamped to 0..150 V. */ );

    /// Validate and execute either axis's identify callback with the transport lock held.
    int identifyRequest( const pcf::IndiProperty &request /**< [in] Received switch. */,
                         pcf::IndiProperty &property /**< [in/out] Published request. */,
                         tmcCon<kcubeCtrl> &axis /**< [in] Controller. */ );

    /// Validate and execute either axis's enable callback.
    int enableRequest( const pcf::IndiProperty &request /**< [in] Received toggle. */,
                       tmcCon<kcubeCtrl> &axis /**< [in] Controller. */,
                       bool &enabled /**< [in/out] Last commanded enable state. */ );

    /// Validate and execute either axis's voltage callback.
    int voltageRequest( const pcf::IndiProperty &request /**< [in] Received voltage. */,
                        pcf::IndiProperty &property /**< [in/out] Published voltage. */,
                        tmcCon<kcubeCtrl> &axis /**< [in] Controller. */ );

public:
    /// Construct with immediate startup attempts and bounded nonfatal retries.
    kcubeCtrl();

    /// Release controllers through their owning destructors.
    ~kcubeCtrl() noexcept = default;

    /// Register serial numbers and bounded connection timing.
    void setupConfig() override;

    /// Load and validate application configuration.
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in] Configuration source. */ );

    /// Load application configuration and stop startup on invalid settings.
    void loadConfig() override;

    /// Register INDI commands and wrap the existing power subscription.
    int appStartup() override;

    /// Discover, initialize, and poll both axes, recovering nonfatally from device errors.
    int appLogic() override;

    /// Block new commands and release both connections before INDI shutdown.
    int appShutdown() override;

    /// Release connections and clear stale command state after actual power-off.
    int onPowerOff() override;

    /// Keep the app idle while power is off.
    int whilePowerOff() override;

    /// Require confirmed On/On in the synchronized snapshot, or unmanaged power.
    bool powerOnExpected() const;

    /// Test known actual or target Off in the synchronized snapshot.
    bool powerOffRequested() const;

    /// Cancel transport when power changes or shutdown begins; caller holds m_indiMutex.
    bool communicationAllowed() const;

    /// Retain the legacy no-op identify interface.
    int modIdentify();

    /// Initialize the axis with its output disabled. Caller holds m_indiMutex.
    int axis1Initialize();

    /// Enable the axis after admission checks. Caller holds m_indiMutex.
    int axis1Enable();

    /// Disable the axis after admission checks. Caller holds m_indiMutex.
    int axis1Disable();

    /// Set a finite axis voltage, clamping to the supported range. Caller holds m_indiMutex.
    int axis1Voltage(float &value /**< [in/out] Requested volts. */);

    /// Initialize the axis with its output disabled. Caller holds m_indiMutex.
    int axis2Initialize();

    /// Enable the axis after admission checks. Caller holds m_indiMutex.
    int axis2Enable();

    /// Disable the axis after admission checks. Caller holds m_indiMutex.
    int axis2Disable();

    /// Set a finite axis voltage, clamping to the supported range. Caller holds m_indiMutex.
    int axis2Voltage(float &value /**< [in/out] Requested volts. */);

    /// Enable both axes at 75 V only after every step succeeds; caller holds m_indiMutex.
    int set();

    /// Zero and disable both axes without resuming interrupted work; caller holds m_indiMutex.
    int rest();

    /// Apply partial power updates and latch interruptions without taking the transport mutex.
    int setCallBack_m_indiP_powerChannel( const pcf::IndiProperty &request /**< [in] Power update. */ );

    /// Dispatch the power subscription to this application.
    static int st_setCallBack_m_indiP_powerChannel( void *app /**< [in] Owning app. */,
                                                   const pcf::IndiProperty &request /**< [in] Power update. */ );

    /** \name INDI Commands
     * @{ */
    /// Published axis1 identify command and status.
    pcf::IndiProperty m_indiP_axis1_identify;

    /// Validate and dispatch the axis1 identify command.
    int newCallBack_m_indiP_axis1_identify( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis1 identify callback.
    static int st_newCallBack_m_indiP_axis1_identify( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published axis1 enable command and status.
    pcf::IndiProperty m_indiP_axis1_enable;

    /// Validate and dispatch the axis1 enable command.
    int newCallBack_m_indiP_axis1_enable( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis1 enable callback.
    static int st_newCallBack_m_indiP_axis1_enable( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published axis1 voltage command and status.
    pcf::IndiProperty m_indiP_axis1_voltage;

    /// Validate and dispatch the axis1 voltage command.
    int newCallBack_m_indiP_axis1_voltage( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis1 voltage callback.
    static int st_newCallBack_m_indiP_axis1_voltage( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published axis2 identify command and status.
    pcf::IndiProperty m_indiP_axis2_identify;

    /// Validate and dispatch the axis2 identify command.
    int newCallBack_m_indiP_axis2_identify( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis2 identify callback.
    static int st_newCallBack_m_indiP_axis2_identify( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published axis2 enable command and status.
    pcf::IndiProperty m_indiP_axis2_enable;

    /// Validate and dispatch the axis2 enable command.
    int newCallBack_m_indiP_axis2_enable( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis2 enable callback.
    static int st_newCallBack_m_indiP_axis2_enable( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published axis2 voltage command and status.
    pcf::IndiProperty m_indiP_axis2_voltage;

    /// Validate and dispatch the axis2 voltage command.
    int newCallBack_m_indiP_axis2_voltage( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered axis2 voltage callback.
    static int st_newCallBack_m_indiP_axis2_voltage( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Published set command and status.
    pcf::IndiProperty m_indiP_set;

    /// Validate and dispatch the set command.
    int newCallBack_m_indiP_set( const pcf::IndiProperty &request /**< [in] Command update. */ );

    /// Dispatch the registered set callback.
    static int st_newCallBack_m_indiP_set( void *app /**< [in] Owning app. */,
                                             const pcf::IndiProperty &request /**< [in] Command update. */ );

    ///@}
};

template<class parentT>
void tmcCon<parentT>::owner( parentT *parent )
{
    m_parent = parent;
}

template<class parentT>
bool tmcCon<parentT>::communicationAllowed() const
{
    return m_parent && m_parent->communicationAllowed();
}

template<class parentT>
const std::string &tmcCon<parentT>::error() const
{
    return m_error;
}

template<class parentT>
void tmcCon<parentT>::clearError()
{
    m_error.clear();
}

template<class parentT>
void tmcCon<parentT>::ftdiErrmsg( const std::string &src, const std::string &msg, int,
                                const std::string &, int )
{
    m_error = src + ": " + msg + " [libftdi1: " + ( m_ftdi ? ftdi_get_error_string( m_ftdi ) : "no context" ) + "]";
}

template<class parentT>
void tmcCon<parentT>::otherErrmsg( const std::string &src, const std::string &msg, const std::string &, int )
{
    m_error = src + ": " + msg;
}

inline kcubeCtrl::kcubeCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;
    m_powerOnWait = 0; // Preserve immediate startup; late firmware readiness is handled by retries.
}

inline void kcubeCtrl::setupConfig()
{
    config.add( "axis1.serial", "", "axis1.serial", argType::Required, "axis1", "serial", false, "string", "USB serial number" );
    config.add( "axis2.serial", "", "axis2.serial", argType::Required, "axis2", "serial", false, "string", "USB serial number" );
    config.add( "connection.retryInterval", "", "connection.retryInterval", argType::Required,
                "connection", "retryInterval", false, "unsigned", "Connection retry interval in ms (1..60000), default 1000." );
    config.add( "timeouts.io", "", "timeouts.io", argType::Required, "timeouts", "io", false,
                "unsigned", "USB operation/response timeout in ms (1..60000), default 1000." );
}

inline int kcubeCtrl::loadConfigImpl( mx::app::appConfigurator &config )
{
    std::string serial;
    config( serial, "axis1.serial" );
    m_kAxis1.serial( serial );
    serial.clear();
    config( serial, "axis2.serial" );
    m_kAxis2.serial( serial );
    config( m_retryInterval, "connection.retryInterval" );
    config( m_ioTimeout, "timeouts.io" );
    if( m_kAxis1.serial().empty() || m_kAxis2.serial().empty() || m_kAxis1.serial() == m_kAxis2.serial() ||
        m_retryInterval == 0 || m_retryInterval > 60000 || m_ioTimeout == 0 || m_ioTimeout > 60000 )
        return log<software_error, -1>( { __FILE__, __LINE__, "Two distinct axis serials and timeouts in 1..60000 ms are required." } );
    return 0;
}

inline void kcubeCtrl::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = true;
}

inline int kcubeCtrl::appStartup()
{
    if( m_powerMgtEnabled )
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiCallBackMutex );
        auto callback = m_indiSetCallBacks.find( m_indiP_powerChannel.createUniqueKey() );
        if( callback == m_indiSetCallBacks.end() )
            return log<software_error, -1>( { __FILE__, __LINE__, "Power channel callback is not registered." } );
        callback->second.callBack = st_setCallBack_m_indiP_powerChannel;
    }
    // Startup runs before the INDI receive thread; subsequent snapshots are published by its callback.
    m_powerStatus = static_cast<uint64_t>( powerState() + 1 ) |
                    ( static_cast<uint64_t>( powerStateTarget() + 1 ) << 2 );
    m_kAxis1.owner( this );
    m_kAxis2.owner( this );
    m_kAxis1.ioTimeout( m_ioTimeout );
    m_kAxis2.ioTimeout( m_ioTimeout );

    createStandardIndiRequestSw( m_indiP_axis1_identify, "axis1_identify");
    if( registerIndiPropertyNew( m_indiP_axis1_identify, INDI_NEWCALLBACK(m_indiP_axis1_identify)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_axis1_enable, "axis1_enable");
    if( registerIndiPropertyNew( m_indiP_axis1_enable, INDI_NEWCALLBACK(m_indiP_axis1_enable)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    createStandardIndiNumber<float>( m_indiP_axis1_voltage, "axis1_voltage", 0, 150, 1.0/32767, "%0.4f");
    if( registerIndiPropertyNew( m_indiP_axis1_voltage, INDI_NEWCALLBACK(m_indiP_axis1_voltage)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    m_indiP_axis1_voltage["current"]=0;
    m_indiP_axis1_voltage["target"]=0;

    createStandardIndiRequestSw( m_indiP_axis2_identify, "axis2_identify");
    if( registerIndiPropertyNew( m_indiP_axis2_identify, INDI_NEWCALLBACK(m_indiP_axis2_identify)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_axis2_enable, "axis2_enable");
    if( registerIndiPropertyNew( m_indiP_axis2_enable, INDI_NEWCALLBACK(m_indiP_axis2_enable)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_axis2_voltage, "axis2_voltage",0,150,1.0/32767, "%0.4f");
    if( registerIndiPropertyNew( m_indiP_axis2_voltage, INDI_NEWCALLBACK(m_indiP_axis2_voltage)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    m_indiP_axis2_voltage["current"]=0;
    m_indiP_axis2_voltage["target"]=0;

    createStandardIndiToggleSw( m_indiP_set, "set");
    if( registerIndiPropertyNew( m_indiP_set, INDI_NEWCALLBACK(m_indiP_set)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    state(stateCodes::NODEVICE);

    return 0;
}


inline bool kcubeCtrl::powerOnExpected() const
{
    return !m_powerMgtEnabled || ( m_powerStatus.load() & 15 ) == 10;
}

inline bool kcubeCtrl::powerOffRequested() const
{
    auto power = m_powerStatus.load();
    return m_powerMgtEnabled && ( ( power & 3 ) == 1 || ( ( power >> 2 ) & 3 ) == 1 );
}

inline bool kcubeCtrl::communicationAllowed() const
{
    return !m_stopping && ( !m_powerMgtEnabled ||
           ( ( m_connectionPower & 15 ) == 10 && m_powerStatus.load() == m_connectionPower ) );
}

inline bool kcubeCtrl::commandAllowed()
{
    return communicationAllowed() && m_kAxis1.connected() && m_kAxis2.connected() &&
           ( state() == stateCodes::READY || state() == stateCodes::OPERATING );
}

inline int kcubeCtrl::setCallBack_m_indiP_powerChannel( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_powerChannel, request );
    int rv = MagAOXApp<true>::setCallBack_m_indiP_powerChannel( request );
    // Only the receive thread writes snapshots. Base fields are read here, not by the hardware thread.
    uint64_t snapshot = static_cast<uint64_t>( powerState() + 1 ) |
                        ( static_cast<uint64_t>( powerStateTarget() + 1 ) << 2 );
    uint64_t generation = m_powerStatus.load() & ~uint64_t(15);
    if( snapshot != 10 )
        generation += 16;
    m_powerStatus.store( generation | snapshot );
    return rv;
}

inline int kcubeCtrl::st_setCallBack_m_indiP_powerChannel( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->setCallBack_m_indiP_powerChannel( request );
}

inline void kcubeCtrl::resetConnection( pcf::IndiProperty::PropertyStateType status )
{
    m_kAxis1.close( false );
    m_kAxis2.close( false );
    m_axis1Enabled = m_axis2Enabled = m_isSet = false;
    for( auto *property : { &m_indiP_axis1_enable, &m_indiP_axis2_enable, &m_indiP_set } )
        if( property->find( "toggle" ) )
            updateSwitchIfChanged( *property, "toggle", pcf::IndiElement::Off, status );
    for( auto *property : { &m_indiP_axis1_identify, &m_indiP_axis2_identify } )
        if( property->find( "request" ) )
            updateSwitchIfChanged( *property, "request", pcf::IndiElement::Off, status );
    // Retain last measured voltages, but mark them unavailable rather than inventing zero readbacks.
    for( auto *property : { &m_indiP_axis1_voltage, &m_indiP_axis2_voltage } )
        if( property->find( "current" ) )
            updateIfChanged( *property, "current", ( *property )["current"].get<float>(), status );
}

inline int kcubeCtrl::recover( tmcCon<kcubeCtrl> &axis, const char *operation, int result )
{
    if( communicationAllowed() )
    {
        std::string error = std::string( &axis == &m_kAxis1 ? "axis1 " : "axis2 " ) + axis.serial() + " " + operation +
                            " result=" + std::to_string( result ) + " " + axis.error();
        if( error != m_lastError )
        {
            if( std::string( operation ) == "open" && result == -3 )
                log<text_log>( error, logPrio::LOG_INFO );
            else
                log<software_error>( { __FILE__, __LINE__, 0, result,
                                       error + " state=" + std::to_string( state() ) + " power=On target=On" } );
            m_lastError = error;
        }
    }
    resetConnection( powerOffRequested() ? INDI_IDLE : INDI_ALERT );
    if( state() != stateCodes::POWEROFF && !m_stopping )
        state( stateCodes::NODEVICE );
    m_nextAttempt = std::chrono::steady_clock::now() + std::chrono::milliseconds( m_retryInterval );
    return -1;
}

template<class operationT>
int kcubeCtrl::transact( tmcCon<kcubeCtrl> &axis, const char *name, operationT operation )
{
    axis.clearError();
    if( !communicationAllowed() )
        return recover( axis, name, -666 );
    int result = operation();
    if( result != 0 || !communicationAllowed() )
        return recover( axis, name, result );
    return 0;
}

inline int kcubeCtrl::appLogic()
{
    std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );
    if( !lock.owns_lock() || m_stopping )
        return 0;

    auto power = m_powerStatus.load();
    if( !powerOnExpected() )
    {
        resetConnection( powerOffRequested() ? INDI_IDLE : INDI_ALERT );
        if( state() != stateCodes::POWEROFF )
            state( stateCodes::NODEVICE );
        return 0;
    }
    if( ( m_powerMgtEnabled && power != m_connectionPower ) || state() == stateCodes::POWERON )
    {
        resetConnection( INDI_IDLE );
        m_connectionPower = power;
        m_nextAttempt = std::chrono::steady_clock::now() + std::chrono::seconds( m_powerMgtEnabled ? m_powerOnWait : 0 );
        state( stateCodes::NODEVICE );
    }
    if( state() == stateCodes::ERROR )
    {
        recover( m_kAxis1, "connection invalidated", -1 );
        return 0;
    }
    if( state() == stateCodes::NODEVICE )
    {
        if( std::chrono::steady_clock::now() < m_nextAttempt )
            return 0;
        if( transact( m_kAxis1, "open", [&] { elevatedPrivileges privileges( this ); return m_kAxis1.open(); } ) < 0 ||
            transact( m_kAxis2, "open", [&] { elevatedPrivileges privileges( this ); return m_kAxis2.open(); } ) < 0 )
            return 0;
        state( stateCodes::NOTCONNECTED );
    }
    if( state() == stateCodes::NOTCONNECTED )
    {
        if( transact( m_kAxis1, "connect", [&] { elevatedPrivileges privileges( this ); return m_kAxis1.connect(); } ) < 0 ||
            transact( m_kAxis2, "connect", [&] { elevatedPrivileges privileges( this ); return m_kAxis2.connect(); } ) < 0 )
            return 0;
        state( stateCodes::CONNECTED );
    }
    if( state() == stateCodes::CONNECTED )
    {
        if( axis1Initialize() < 0 || axis2Initialize() < 0 )
            return 0;
        state( stateCodes::READY );
        m_lastError.clear();
    }
    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        float volts1 = 0, volts2 = 0;
        if( transact( m_kAxis1, "read voltage", [&] { return m_kAxis1.pz_req_outputvolts( volts1 ); } ) < 0 ||
            transact( m_kAxis2, "read voltage", [&] { return m_kAxis2.pz_req_outputvolts( volts2 ); } ) < 0 )
            return 0;
        updateIfChanged( m_indiP_axis1_voltage, "current", volts1 * 150, m_axis1Enabled ? INDI_OK : INDI_IDLE );
        updateIfChanged( m_indiP_axis2_voltage, "current", volts2 * 150, m_axis2Enabled ? INDI_OK : INDI_IDLE );
        updateSwitchIfChanged( m_indiP_axis1_enable, "toggle", m_axis1Enabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               m_axis1Enabled ? INDI_OK : INDI_IDLE );
        updateSwitchIfChanged( m_indiP_axis2_enable, "toggle", m_axis2Enabled ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               m_axis2Enabled ? INDI_OK : INDI_IDLE );
        updateSwitchIfChanged( m_indiP_set, "toggle", m_isSet ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               m_isSet ? INDI_OK : INDI_IDLE );
        updateSwitchIfChanged( m_indiP_axis1_identify, "request", pcf::IndiElement::Off, INDI_IDLE );
        updateSwitchIfChanged( m_indiP_axis2_identify, "request", pcf::IndiElement::Off, INDI_IDLE );
        state( m_axis1Enabled && m_axis2Enabled ? stateCodes::OPERATING : stateCodes::READY );
    }
    return 0;
}

inline int kcubeCtrl::appShutdown()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_stopping = true;
    resetConnection( INDI_IDLE );
    return 0;
}

inline int kcubeCtrl::onPowerOff()
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    resetConnection( INDI_IDLE );
    m_lastError.clear();
    return 0;
}

inline int kcubeCtrl::whilePowerOff()
{
    return 0;
}

inline int kcubeCtrl::modIdentify()
{
    return 0;
}

inline int kcubeCtrl::initialize( tmcCon<kcubeCtrl> &axis, bool &enabled )
{
    if( state() != stateCodes::CONNECTED || !axis.connected() || !communicationAllowed() )
        return recover( axis, "initialize", -666 );
    tmcController::HWInfo info;
    tmcController::KMMIParams parameters;
    tmcController::TPZIOSettings settings;
    if( transact( axis, "hardware information", [&] { return axis.hw_req_info( info ); } ) < 0 ||
        enable( axis, enabled, false ) < 0 ||
        transact( axis, "stop updates", [&] { return axis.hw_stop_updatemsgs(); } ) < 0 ||
        transact( axis, "read MMI", [&] { return axis.kpz_req_kcubemmiparams( parameters ); } ) < 0 )
        return -1;
    parameters.DispBrightness = 0;
    if( transact( axis, "write MMI", [&] { return axis.kpz_set_kcubemmiparams( parameters ); } ) < 0 ||
        transact( axis, "verify MMI", [&] { return axis.kpz_req_kcubemmiparams( parameters ); } ) < 0 ||
        transact( axis, "read IO settings", [&] { return axis.pz_req_tpz_iosettings( settings ); } ) < 0 )
        return -1;
    settings.VoltageLimit = tmcController::VoltLimit::V150;
    if( transact( axis, "write IO settings", [&] { return axis.pz_set_tpz_iosettings( settings ); } ) < 0 ||
        transact( axis, "verify IO settings", [&] { return axis.pz_req_tpz_iosettings( settings ); } ) < 0 )
        return -1;
    std::stringstream message;
    message << ( &axis == &m_kAxis1 ? "Axis-1 " : "Axis-2 " );
    info.dump( message );
    parameters.dump( message );
    settings.dump( message );
    log<text_log>( message.str() );
    return 0;
}

inline int kcubeCtrl::enable( tmcCon<kcubeCtrl> &axis, bool &enabled, bool value )
{
    if( transact( axis, value ? "enable" : "disable", [&] {
            return axis.mod_set_chanenablestate( 1, value ? tmcController::EnableState::enabled : tmcController::EnableState::disabled );
        } ) < 0 )
        return -1;
    enabled = value;
    if( !value )
        m_isSet = false;
    if( state() != stateCodes::CONNECTED )
        log<text_log>( std::string( value ? "enabled " : "disabled " ) + ( &axis == &m_kAxis1 ? "axis 1 piezo" : "axis 2 piezo" ),
                   logPrio::LOG_NOTICE );
    return 0;
}

inline int kcubeCtrl::voltage( tmcCon<kcubeCtrl> &axis, float &value )
{
    if( !mx::math::isFinite( value ) )
        return -1;
    if( value < 0 || value > 150 )
    {
        log<text_log>( std::string( &axis == &m_kAxis1 ? "axis 1" : "axis 2" ) + " voltage clamped to 0..150 V (" +
                       std::to_string( value ) + ")", logPrio::LOG_WARNING );
        value = std::clamp( value, 0.0f, 150.0f );
    }
    return transact( axis, "write voltage", [&] { return axis.pz_set_outputvolts( value / 150 ); } );
}

inline int kcubeCtrl::axis1Initialize()
{
    return initialize( m_kAxis1, m_axis1Enabled );
}

inline int kcubeCtrl::axis1Enable()
{
    if( !commandAllowed() )
        return -1;
    return enable( m_kAxis1, m_axis1Enabled, true );
}

inline int kcubeCtrl::axis1Disable()
{
    if( !commandAllowed() )
        return -1;
    return enable( m_kAxis1, m_axis1Enabled, false );
}

inline int kcubeCtrl::axis1Voltage(float &value)
{
    if( !commandAllowed() )
        return -1;
    return voltage( m_kAxis1, value );
}

inline int kcubeCtrl::axis2Initialize()
{
    return initialize( m_kAxis2, m_axis2Enabled );
}

inline int kcubeCtrl::axis2Enable()
{
    if( !commandAllowed() )
        return -1;
    return enable( m_kAxis2, m_axis2Enabled, true );
}

inline int kcubeCtrl::axis2Disable()
{
    if( !commandAllowed() )
        return -1;
    return enable( m_kAxis2, m_axis2Enabled, false );
}

inline int kcubeCtrl::axis2Voltage(float &value)
{
    if( !commandAllowed() )
        return -1;
    return voltage( m_kAxis2, value );
}

inline int kcubeCtrl::set()
{
    if( !commandAllowed() )
        return -1;
    if( m_isSet )
        return 0;
    float value = 75;
    if( axis1Enable() < 0 || axis1Voltage( value ) < 0 || axis2Enable() < 0 || axis2Voltage( value ) < 0 )
        return -1;
    m_isSet = true;
    log<text_log>( "set", logPrio::LOG_NOTICE );
    return 0;
}

inline int kcubeCtrl::rest()
{
    if( !commandAllowed() )
        return -1;
    if( !m_isSet )
        return 0;
    float value = 0;
    if( axis1Voltage( value ) < 0 || axis1Disable() < 0 || axis2Voltage( value ) < 0 || axis2Disable() < 0 )
        return -1;
    m_isSet = false;
    log<text_log>( "rested", logPrio::LOG_NOTICE );
    return 0;
}

inline int kcubeCtrl::identifyRequest( const pcf::IndiProperty &request, pcf::IndiProperty &property, tmcCon<kcubeCtrl> &axis )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( request.getType() != pcf::IndiProperty::Switch || !request.find( "request" ) ||
        !commandAllowed() || state() != stateCodes::READY )
        return -1;
    auto value = request["request"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    if( value == pcf::IndiElement::On )
    {
        updateSwitchIfChanged( property, "request", pcf::IndiElement::On, INDI_BUSY );
        return transact( axis, "identify", [&] { return axis.mod_identify(); } );
    }
    return 0;
}

inline int kcubeCtrl::enableRequest( const pcf::IndiProperty &request, tmcCon<kcubeCtrl> &axis, bool &enabled )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( request.getType() != pcf::IndiProperty::Switch || !request.find( "toggle" ) || !commandAllowed() )
        return -1;
    auto value = request["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    return enable( axis, enabled, value == pcf::IndiElement::On );
}

inline int kcubeCtrl::voltageRequest( const pcf::IndiProperty &request, pcf::IndiProperty &property, tmcCon<kcubeCtrl> &axis )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( request.getType() != pcf::IndiProperty::Number || !commandAllowed() ||
        !( request.find( "target" ) || request.find( "current" ) ) )
        return -1;
    float value = 0;
    std::istringstream input( request[request.find( "target" ) ? "target" : "current"].get() );
    if( !( input >> value ) || !( input >> std::ws ).eof() )
        return -1;
    if( !mx::math::isFinite( value ) || voltage( axis, value ) < 0 )
        return -1;
    updateIfChanged( property, "target", value, INDI_BUSY );
    return 0;
}

inline int kcubeCtrl::newCallBack_m_indiP_axis1_identify( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis1_identify, request );
    return identifyRequest( request, m_indiP_axis1_identify, m_kAxis1 );
}

inline int kcubeCtrl::newCallBack_m_indiP_axis1_enable( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis1_enable, request );
    return enableRequest( request, m_kAxis1, m_axis1Enabled );
}

inline int kcubeCtrl::newCallBack_m_indiP_axis1_voltage( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis1_voltage, request );
    return voltageRequest( request, m_indiP_axis1_voltage, m_kAxis1 );
}

inline int kcubeCtrl::newCallBack_m_indiP_axis2_identify( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis2_identify, request );
    return identifyRequest( request, m_indiP_axis2_identify, m_kAxis2 );
}

inline int kcubeCtrl::newCallBack_m_indiP_axis2_enable( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis2_enable, request );
    return enableRequest( request, m_kAxis2, m_axis2Enabled );
}

inline int kcubeCtrl::newCallBack_m_indiP_axis2_voltage( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_axis2_voltage, request );
    return voltageRequest( request, m_indiP_axis2_voltage, m_kAxis2 );
}

inline int kcubeCtrl::newCallBack_m_indiP_set( const pcf::IndiProperty &request )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_set, request );
    std::lock_guard<std::mutex> lock( m_indiMutex );
    if( request.getType() != pcf::IndiProperty::Switch || !request.find( "toggle" ) || !commandAllowed() )
        return -1;
    auto value = request["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    int result = value == pcf::IndiElement::On ? set() : rest();
    if( result == 0 )
        updateSwitchIfChanged( m_indiP_set, "toggle", value, INDI_OK );
    return result;
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis1_identify( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis1_identify( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis1_enable( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis1_enable( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis1_voltage( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis1_voltage( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis2_identify( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis2_identify( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis2_enable( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis2_enable( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_axis2_voltage( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_axis2_voltage( request );
}

inline int kcubeCtrl::st_newCallBack_m_indiP_set( void *app, const pcf::IndiProperty &request )
{
    return static_cast<kcubeCtrl *>( app )->newCallBack_m_indiP_set( request );
}

} // namespace app
} // namespace MagAOX
#endif // kcubeCtrl_hpp
