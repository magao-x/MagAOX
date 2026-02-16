/** \file qwpTracker.hpp
 * \brief The MagAO-X HWP rotation tracker header file
 *
 * \ingroup qwpTracker_files
 */

#ifndef qwpTracker_hpp
#define qwpTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup qwpTracker
 * \brief The MagAO-X application to track pupil rotation with the HWP.
 *
 * <a href="../handbook/operating/software/apps/qwpTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup qwpTracker_files
 * \ingroup qwpTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/**
 * \ingroup qwpTracker
 */
class qwpTracker : public MagAOXApp<true>, public dev::telemeter<qwpTracker>
{

    // Give the test harness access.
    friend class qwpTracker_test;

    friend class dev::telemeter<qwpTracker>;

    typedef dev::telemeter<qwpTracker> telemeterT;

  protected:
    /** \name Configurable Parameters
     *@{
        */
    // basics: orientation within mount
    float m_qwp1_zero{ 0 };
    int m_qwp1_sign{ 1 };
    float m_qwp2_zero{ 0 };
    int m_qwp2_sign{ -1 };

    // positions
    float m_qwp1_tgtPos{ 0 };
    float m_qwp1_curPos{ 0 };
    float m_qwp2_tgtPos{ 0 };
    float m_qwp2_curPos{ 0 };
    float m_kmirror{ 0 };

    // indi
    std::string m_qwp1_devName{ "stageqwp1rot" }; ///< The device name of the QWP stage.
    std::string m_qwp2_devName{ "stageqwp2rot" }; ///< The device name of the QWP stage.

    std::string m_imrDevName{"ktrack" }; ///< The device name of the TCS Interface providing 'teldata.altitude'.  Default is 'tcsi'

    // tracking
    float m_updateInterval{ 10 }; ///< The interval at which to update positions, in seconds.  Default is 10 secs.

    bool m_tracking{ true }; ///< Are the QWPs tracking?. Default is true.

  public:
    /// Default c'tor.
    qwpTracker();

    /// D'tor, declared and defined for noexcept.
    ~qwpTracker() noexcept
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

    /// Implementation of the FSM for qwpTracker.
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

  protected:
    void getQwp1Angle();

    void getQwp2Angle();

    void updateQwpStages();

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_stagek;

    pcf::IndiProperty m_indiP_stageqwp1rot;
    pcf::IndiProperty m_indiP_stagewp1rotFsm;
    pcf::IndiProperty m_indiP_stageqwp2rot;
    pcf::IndiProperty m_indiP_stageqwp2rotFsm;

    pcf::IndiProperty m_indiP_tracking;
    pcf::IndiProperty m_indiP_qwp1Pos;
    pcf::IndiProperty m_indiP_qwp2Pos;


  public:

    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stagek );

    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stageqwp1rot );
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stagewp1rotFsm );
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stageqwp2rot );
    INDI_SETCALLBACK_DECL( qwpTracker, m_indiP_stageqwp2rotFsm );

    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_tracking );
    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_qwp1Pos );
    INDI_NEWCALLBACK_DECL( qwpTracker, m_indiP_qwp2Pos );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_qwptrack * );

    int recordQwpTrack( bool force = false );

    ///@}
};

qwpTracker::qwpTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void qwpTracker::setupConfig()
{
    config.add( "qwp1.zero",
                "",
                "qwp1.zero",
                argType::Required,
                "qwp1",
                "zero",
                false,
                "float",
                "The QWP1 zero position.  Default is 0." );

    config.add( "qwp1.sign",
                "",
                "qwp1.sign",
                argType::Required,
                "qwp1",
                "sign",
                false,
                "int",
                "The HWP rotation sign. Default is 1." );

    config.add( "qwp1.devName",
                "",
                "qwp1.devName",
                argType::Required,
                "qwp1",
                "devName",
                false,
                "string",
                "The device name of the HWP stage.  Default is 'stageqwp1rot'" );
    config.add( "qwp2.zero",
                "",
                "qwp2.zero",
                argType::Required,
                "qwp2",
                "zero",
                false,
                "float",
                "The QWP2 zero position.  Default is 0." );

    config.add( "qwp2.sign",
                "",
                "qwp2.sign",
                argType::Required,
                "qwp2",
                "sign",
                false,
                "int",
                "The HWP rotation sign. Default is -1." );

    config.add( "qwp2.devName",
                "",
                "qwp2.devName",
                argType::Required,
                "qwp2",
                "devName",
                false,
                "string",
                "The device name of the HWP stage.  Default is 'stageqwp2rot'" );

    config.add( "imr.devName",
                "",
                "imr.devName",
                argType::Required,
                "imr",
                "devName",
                false,
                "string",
                "The device name of the k-mirror image rotator tracker" );

    config.add( "tracking.updateInterval",
                "",
                "tracking.updateInterval",
                argType::Required,
                "tracking",
                "updateInterval",
                false,
                "float",
                "The interval at which to update positions, in seconds.  Default is 1 sec." );

    TELEMETER_SETUP_CONFIG( config );
}

int qwpTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_qwp1_zero, "qwp1.zero" );
    _config( m_qwp1_sign, "qwp1.sign" );
    _config( m_qwp1_devName, "qwp1.devName" );
    _config( m_qwp2_zero, "qwp2.zero" );
    _config( m_qwp2_sign, "qwp2.sign" );
    _config( m_qwp2_devName, "qwp2.devName" );
    _config( m_imrDevName, "imr.devName" );
    _config( m_updateInterval, "tracking.updateInterval" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void qwpTracker::loadConfig()
{
    loadConfigImpl( config );
}

int qwpTracker::appStartup()
{

    REG_INDI_SETPROP( m_indiP_stagek, m_imrDevName, "position" );

    REG_INDI_SETPROP( m_indiP_stageqwp1rot, m_qwp1_devName, "position" );

    REG_INDI_SETPROP( m_indiP_stagewp1rotFsm, m_qwp1_devName, "fsm" );

    REG_INDI_SETPROP( m_indiP_stageqwp2rot, m_qwp2_devName, "position" );

    REG_INDI_SETPROP( m_indiP_stageqwp2rotFsm, m_qwp2_devName, "fsm" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH(m_indiP_tracking, "tracking");

    CREATE_REG_INDI_NEW_NUMBERF(m_indiP_qwp1Pos, "qwp1", -360, 360, 1e-3, "%g", "", "");
    m_indiP_qwp1Pos["current"].setValue(m_qwp1_curPos);
    m_indiP_qwp1Pos["target"].setValue(m_qwp1_tgtPos);
    
    CREATE_REG_INDI_NEW_NUMBERF(m_indiP_qwp2Pos, "qwp2", -360, 360, 1e-3, "%g", "", "");
    m_indiP_qwp2Pos["current"].setValue(m_qwp2_curPos);
    m_indiP_qwp2Pos["target"].setValue(m_qwp2_tgtPos);

    TELEMETER_APP_STARTUP;

    state( stateCodes::READY );

    return 0;
}

int qwpTracker::appLogic()
{

    static double lastupdate = 0;

    if( m_tracking && mx::sys::get_curr_time() - lastupdate > m_updateInterval )
    {

        getQwp1Angle();
        getQwp2Angle();

        updateQwpStages();

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

int qwpTracker::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

void qwpTracker::getQwp1Angle()
{
    m_qwp1_tgtPos = 0.0 * m_kmirror;
}

void qwpTracker::getQwp2Angle()
{
    m_qwp2_tgtPos = 0.0 * m_kmirror;
}

void qwpTracker::updateQwpStages()
{
    /* QWP 2 */

    updateIfChanged<float>(m_indiP_qwp1Pos, "target", m_qwp1_tgtPos);
    float qwp1_stage_angle = m_qwp1_sign * (m_qwp1_tgtPos - m_qwp1_zero);

    std::cerr << "QWP1 set to: " << m_qwp1_tgtPos << "\n";
    std::cerr << "Sending QWP1 stage to: " << qwp1_stage_angle << "\n";
    log<text_log>( "QWP1 set to: " + std::to_string( m_qwp1_tgtPos ) );

    m_indiP_qwp1Pos["target"] = qwp1_stage_angle;
    sendNewProperty( m_indiP_qwp1Pos );


    /* QWP 2 */

    updateIfChanged<float>(m_indiP_qwp2Pos, "target", m_qwp2_tgtPos);
    float qwp2_stage_angle = m_qwp2_sign * (m_qwp2_tgtPos - m_qwp2_zero);

    std::cerr << "QWP2 set to: " << m_qwp2_tgtPos << "\n";
    std::cerr << "Sending QWP2 stage to: " << qwp2_stage_angle << "\n";
    log<text_log>( "QWP2 set to: " + std::to_string( m_qwp2_tgtPos ) );

    m_indiP_qwp2Pos["target"] = qwp2_stage_angle;
    sendNewProperty( m_indiP_qwp2Pos );

    //

    recordQwpTrack();
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_qwp1Pos )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp1Pos, ipRecv );

    if( ipRecv.getName() != m_indiP_qwp1Pos.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "target" ) )
        return 0;

    if (m_tracking) return 0;

    m_qwp1_tgtPos = ipRecv["target"].get<float>();

    updateQwpStages();

    return 0;
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_qwp2Pos )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_qwp2Pos, ipRecv );

    if( ipRecv.getName() != m_indiP_qwp2Pos.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "target" ) )
        return 0;

    if (m_tracking) return 0;

    m_qwp2_tgtPos = ipRecv["target"].get<float>();

    updateQwpStages();

    return 0;
}

INDI_NEWCALLBACK_DEFN( qwpTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
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

        getQwp1Angle();
        getQwp2Angle();

        updateQwpStages();

        log<text_log>( "started QWP rotation tracking" );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        m_tracking = false;

        updateQwpStages();

        log<text_log>( "stopped QWP rotation tracking" );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stagek )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagek, ipRecv );

    if( ipRecv.getName() != m_indiP_stagek.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "current" ) )
        return 0;

    m_kmirror = ipRecv["current"].get<float>();

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stageqwp1rot )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stageqwp1rot, ipRecv );

    if( ipRecv.getName() != m_indiP_stageqwp1rot.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "current" ) )
        return 0;

    float qwp1StagePos = ipRecv["current"].get<float>();
    m_qwp1_curPos = m_qwp1_sign * qwp1StagePos + m_qwp1_zero;

    // round to two decimal points
    m_qwp1_curPos = std::round(m_qwp1_curPos * 100) / 100;

    updateIfChanged<float>(m_indiP_qwp1Pos, "current", m_qwp1_curPos);

    recordQwpTrack();

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stagewp1rotFsm )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stagewp1rotFsm, ipRecv );

    if( ipRecv.getName() != m_indiP_stagewp1rotFsm.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "state" ) )
        return 0;

    std::string stageqwp1rot_state = ipRecv["state"].get<std::string>();

    state(stateCodes::str2CodeFast(stageqwp1rot_state));

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stageqwp2rot )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stageqwp2rot, ipRecv );

    if( ipRecv.getName() != m_indiP_stageqwp2rot.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "current" ) )
        return 0;

    float qwp2StagePos = ipRecv["current"].get<float>();
    m_qwp2_curPos = m_qwp2_sign * qwp2StagePos + m_qwp2_zero;

    // round to two decimal points
    m_qwp2_curPos = std::round(m_qwp2_curPos * 100) / 100;

    updateIfChanged<float>(m_indiP_qwp2Pos, "current", m_qwp2_curPos);

    recordQwpTrack();

    return 0;
}

INDI_SETCALLBACK_DEFN( qwpTracker, m_indiP_stageqwp2rotFsm )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stageqwp2rotFsm, ipRecv );

    if( ipRecv.getName() != m_indiP_stageqwp2rotFsm.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received" } );

        return -1;
    }

    if( !ipRecv.find( "state" ) )
        return 0;

    std::string stageqwp2rot_state = ipRecv["state"].get<std::string>();

    state(stateCodes::str2CodeFast(stageqwp2rot_state));

    return 0;
}

int qwpTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_qwptrack() );
}

int qwpTracker::recordTelem( const telem_qwptrack * )
{
    return recordQwpTrack( true );
}

int qwpTracker::recordQwpTrack( bool force )
{
    static float qwp1Pos = 0;

    static float qwp2Pos = 0;

    static bool tracking = false;

    if( m_qwp1_curPos != qwp1Pos || m_qwp2_curPos != qwp2Pos || m_tracking != tracking || force )
    {
        telem<telem_qwptrack>( { m_qwp1_curPos, m_qwp2_curPos, m_tracking } );

        qwp1Pos    = m_qwp1_curPos;
        qwp2Pos    = m_qwp2_curPos;
        tracking   = m_tracking;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // qwpTracker_hpp
