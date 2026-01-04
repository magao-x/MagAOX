/** \file hwpTracker.hpp
 * \brief The MagAO-X HWP rotation tracker header file
 *
 * \ingroup hwpTracker_files
 */

#ifndef hwpTracker_hpp
#define hwpTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup hwpTracker
 * \brief The MagAO-X application to track pupil rotation with the HWP.
 *
 * <a href="../handbook/operating/software/apps/hwpTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup hwpTracker_files
 * \ingroup hwpTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/**
 * \ingroup hwpTracker
 */
class hwpTracker : public MagAOXApp<true>, public dev::telemeter<hwpTracker>
{

    // Give the test harness access.
    friend class hwpTracker_test;

    friend class dev::telemeter<hwpTracker>;

    typedef dev::telemeter<hwpTracker> telemeterT;

  protected:
    /** \name Configurable Parameters
     *@{
     */
    float m_altitude{ 0 }; ///< Current altitude

    float m_parang{ 0 }; ///< Current parallactic angle

    float m_hwpTrackingOffset{ 0 }; ///< HWP tracking offset

    float m_hwpSetPos{ 0 };

    float m_hwpCurPos{ 0 };

    float m_hwpActualPos{ 0 };

    std::string m_hwpPosName{ "" };

    float m_zero{ 360 }; ///< The zero point of the HWP stage.

    int m_sign{ -1 }; ///< The sign to apply to the calculated HWP angle.

    std::string m_devName{ "stagepolrot" }; ///< The device name of the HWP stage.  Default is 'stagehwprot'

    std::string m_tcsDevName{
        "tcsi" }; ///< The device name of the TCS Interface providing 'teldata.altitude'.  Default is 'tcsi'

    float m_updateInterval{ 10 }; ///< The interval at which to update positions, in seconds.  Default is 10 secs.

    float m_pupilOffset{ 0 }; ///< The pupil offset used for HWP tracking (must match twice the kTracker offset)

    bool m_tracking{ false }; ///< Is the HWP in ADI synchronization mode?. Default is false.

  public:
    /// Default c'tor.
    hwpTracker();

    /// D'tor, declared and defined for noexcept.
    ~hwpTracker() noexcept
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

    /// Implementation of the FSM for hwpTracker.
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

    virtual std::string getHwpStatus();

    virtual void getHwpTrackingOffset();

    virtual void updateHwpPos();

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_tracking;

    pcf::IndiProperty m_indiP_teldata;

    pcf::IndiProperty m_indiP_hwpSetPos;

    pcf::IndiProperty m_indiP_hwpPosName;

    pcf::IndiProperty m_indiP_hwpTrackingOffset;

    pcf::IndiProperty m_indiP_hwpActualPos;

    pcf::IndiProperty m_indiP_hwpStagePos;

    pcf::IndiProperty m_indiP_stagePolRot;

    pcf::IndiProperty m_indiP_stagePolRotFsm;

  public:
    INDI_NEWCALLBACK_DECL( hwpTracker, m_indiP_tracking );

    INDI_NEWCALLBACK_DECL( hwpTracker, m_indiP_hwpSetPos );

    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_teldata );

    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_stagePolRot );

    INDI_SETCALLBACK_DECL( hwpTracker, m_indiP_stagePolRotFsm );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_poltrack * );

    int recordPolTrack( bool force = false );

    ///@}
};

hwpTracker::hwpTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void hwpTracker::setupConfig()
{
    config.add( "hwp.zero",
                "",
                "hwp.zero",
                argType::Required,
                "hwp",
                "zero",
                false,
                "float",
                "The HWP zero position.  Default is 360." );

    config.add( "hwp.sign",
                "",
                "hwp.sign",
                argType::Required,
                "hwp",
                "sign",
                false,
                "int",
                "The HWP rotation sign. Default is -1." );

    config.add( "hwp.devName",
                "",
                "hwp.devName",
                argType::Required,
                "hwp",
                "devName",
                false,
                "string",
                "The device name of the HWPstage.  Default is 'stagepolrot'" );

    config.add( "tcs.devName",
                "",
                "tcs.devName",
                argType::Required,
                "tcs",
                "devName",
                false,
                "string",
                "The device name of the TCS Interface providing 'teldata.zd' and `teldata.pa`.  Default is 'tcsi'" );

    config.add( "tracking.updateInterval",
                "",
                "tracking.updateInterval",
                argType::Required,
                "tracking",
                "updateInterval",
                false,
                "float",
                "The interval at which to update positions, in seconds.  Default is 1 sec." );


    config.add( "tracking.pupilOffset",
                "",
                "tracking.pupilOffset",
                argType::Required,
                "tracking",
                "pupilOffset",
                false,
                "float",
                "The HWP tracking pupil offset. Must match twice the kTracker offset. Default is 0." );


    TELEMETER_SETUP_CONFIG( config );
}

int hwpTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_zero, "hwp.zero" );
    _config( m_sign, "hwp.sign" );
    _config( m_devName, "hwp.devName" );
    _config( m_tcsDevName, "tcs.devName" );
    _config( m_updateInterval, "tracking.updateInterval" );
    _config( m_pupilOffset, "tracking.pupilOffset" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void hwpTracker::loadConfig()
{
    loadConfigImpl( config );
}

int hwpTracker::appStartup()
{

    createStandardIndiToggleSw( m_indiP_tracking, "tracking" );
    registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK( m_indiP_tracking ) );

    REG_INDI_SETPROP( m_indiP_teldata, m_tcsDevName, "teldata" );

    REG_INDI_SETPROP( m_indiP_stagePolRot, m_devName, "position" );

    REG_INDI_SETPROP( m_indiP_stagePolRotFsm, m_devName, "fsm" );

    createStandardIndiNumber<float>(
        m_indiP_hwpSetPos, "hwp_position", -360.0, 360.0, 1e-3, "%.03f", "HWP Set Position", "HWP Status" );
    registerIndiPropertyNew( m_indiP_hwpSetPos, INDI_NEWCALLBACK( m_indiP_hwpSetPos ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpTrackingOffset, "hwp_tracking_offset", pcf::IndiProperty::Number );
    m_indiP_hwpTrackingOffset.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpTrackingOffset["value"].set( 0 );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpPosName, "hwp_position_name", pcf::IndiProperty::Text );
    m_indiP_hwpPosName.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpPosName["value"].set( "" );

    REG_INDI_NEWPROP_NOCB( m_indiP_hwpActualPos, "hwp_position_actual", pcf::IndiProperty::Number );
    m_indiP_hwpActualPos.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpActualPos["value"].set( 0 );

    m_indiP_hwpStagePos = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_hwpStagePos.setDevice( m_devName );
    m_indiP_hwpStagePos.setName( "position" );
    m_indiP_hwpStagePos.add( pcf::IndiElement( "target" ) );

    TELEMETER_APP_STARTUP;

    state( stateCodes::READY );

    return 0;
}

int hwpTracker::appLogic()
{

    static double lastupdate = 0;

    if( m_tracking && mx::sys::get_curr_time() - lastupdate > m_updateInterval )
    {

        getHwpTrackingOffset();

        updatesIfChanged<float>( m_indiP_hwpTrackingOffset, { "value" }, { m_hwpTrackingOffset } );

        updateHwpPos();

        lastupdate = mx::sys::get_curr_time();
    }
    else
    {
        if( !m_tracking )
            lastupdate = 0;
    }

    TELEMETER_APP_LOGIC;

    return 0;
}

int hwpTracker::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

std::string hwpTracker::getHwpStatus()
{
    float tol = 0.5; // degrees
    if( fabs( m_hwpCurPos - 0 ) < tol )
        return "Qplus";
    else if( fabs( m_hwpCurPos - 45 ) < tol )
        return "Qminus";
    else if( fabs( m_hwpCurPos - 22.5 ) < tol )
        return "Uplus";
    else if( fabs( m_hwpCurPos - 67.5 ) < tol )
        return "Uminus";
    else
        return "Unknown";
}

void hwpTracker::getHwpTrackingOffset()
{
    // While on Nasmyth East, the sign is negative
    m_hwpTrackingOffset = -0.5 * m_parang + m_altitude + m_pupilOffset;
}

void hwpTracker::updateHwpPos()
{
    float hwpActualPos = m_hwpSetPos + m_hwpTrackingOffset;

    float hwpStagePos = m_zero + m_sign * hwpActualPos;

    std::cerr << "HWP set to: " << hwpActualPos << "\n";
    std::cerr << "Sending HWP stage to: " << hwpStagePos << "\n";
    log<text_log>( "HWP set to: " + std::to_string( hwpActualPos ) );

    m_indiP_hwpStagePos["target"] = hwpStagePos;
    sendNewProperty( m_indiP_hwpStagePos );

    recordPolTrack();
}

INDI_NEWCALLBACK_DEFN( hwpTracker, m_indiP_hwpSetPos )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_hwpSetPos, ipRecv );

    if( ipRecv.getName() != m_indiP_hwpSetPos.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "target" ) )
        return 0;

    m_hwpSetPos = ipRecv["target"].get<float>();

    updateIfChanged<float>(m_indiP_hwpSetPos, "target", m_hwpSetPos);

    updateHwpPos();

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tracking, ipRecv );

    if( ipRecv.getName() != m_indiP_tracking.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_IDLE );

        m_tracking = true;

        getHwpTrackingOffset();

        updateIfChanged<float>( m_indiP_hwpTrackingOffset, "value", m_hwpTrackingOffset );

        updateHwpPos();

        log<text_log>( "started HWP rotation tracking" );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        m_tracking = false;

        m_hwpTrackingOffset = 0;

        updateIfChanged<float>( m_indiP_hwpTrackingOffset, "value", m_hwpTrackingOffset );

        updateHwpPos();

        log<text_log>( "stopped HWP rotation tracking" );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_teldata )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_teldata, ipRecv );

    if( ipRecv.getName() != m_indiP_teldata.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "zd" ) )
        return 0;

    if( !ipRecv.find( "pa" ) )
        return 0;

    m_altitude = 90 - ipRecv["zd"].get<float>();

    m_parang = ipRecv["pa"].get<float>();

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_stagePolRot )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagePolRot, ipRecv );

    if( ipRecv.getName() != m_indiP_stagePolRot.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "current" ) )
        return 0;

    float hwpStagePos = ipRecv["current"].get<float>();

    m_hwpActualPos = m_sign * (hwpStagePos - m_zero);
    if (fabs(m_hwpActualPos) < 1e-2)
        m_hwpActualPos = 0;
    updateIfChanged<float>(m_indiP_hwpActualPos, "value", m_hwpActualPos);

    m_hwpCurPos = m_hwpActualPos - m_hwpTrackingOffset;
    // round to two decimal points
    m_hwpCurPos = std::round(m_hwpCurPos * 100) / 100;
    updateIfChanged<float>(m_indiP_hwpSetPos, "current", m_hwpCurPos);

    m_hwpPosName = getHwpStatus();

    updateIfChanged<std::string>( m_indiP_hwpPosName, "value", m_hwpPosName );

    recordPolTrack();

    return 0;
}

INDI_SETCALLBACK_DEFN( hwpTracker, m_indiP_stagePolRotFsm )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagePolRotFsm, ipRecv );

    if( ipRecv.getName() != m_indiP_stagePolRotFsm.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "state" ) )
        return 0;

    std::string stagepolrot_state = ipRecv["state"].get<std::string>();

    state(stateCodes::str2CodeFast(stagepolrot_state));

    return 0;
}

int hwpTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_poltrack() );
}

int hwpTracker::recordTelem( const telem_poltrack * )
{
    return recordPolTrack( true );
}

int hwpTracker::recordPolTrack( bool force )
{
    static float hwpSetPos = 0;

    static float hwpActualPos = 0;

    static std::string hwpPosName = "";

    static bool tracking = false;

    if( m_hwpCurPos != hwpSetPos || m_hwpActualPos != hwpActualPos || hwpPosName != m_hwpPosName || tracking != m_tracking || force )
    {
        telem<telem_poltrack>( { m_hwpCurPos, m_hwpActualPos, m_hwpPosName, m_tracking} );

        hwpSetPos    = m_hwpCurPos;
        hwpActualPos = m_hwpActualPos;
        hwpPosName   = m_hwpPosName;
        tracking     = m_tracking;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // hwpTracker_hpp
