/** \file orcaCtrl_test.cpp
 * \brief Catch2 tests of orcaCtrl power-on and connection with an offline fake DCAM library.
 *
 * \ingroup orcaCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

#include <dcamapi4.h>
#include <dcamprop.h>

#include <cstring>
#include <algorithm>
#include <map>
#include <set>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace orcaHarness
{

/// Scripted DCAM library state.
struct Fake
{
    /// Camera ID of the single fake camera.
    std::string m_cameraID{ "PHX2" };

    /// Number of initial dcamapi_init calls that fail with DCAMERR_NOCAMERA, as while the camera enumerates.
    unsigned m_noCameraInits{ 0 };

    /// Result of dcamapi_init after the DCAMERR_NOCAMERA calls; a failure code fails every later call.
    DCAMERR m_initResult{ DCAMERR_SUCCESS };

    /// Capture status reported by dcamcap_status.
    int32 m_captureStatus{ DCAMCAP_STATUS_READY };

    /// Property values by ID.
    std::map<int32, double> m_props{ { DCAM_IDPROP_READOUTSPEED, 2 } };

    /// Result of reading DCAM_IDPROP_READOUTSPEED; a failure code makes the read fail.
    DCAMERR m_speedGetResult{ DCAMERR_SUCCESS };

    /// Result of writing DCAM_IDPROP_READOUTSPEED; a failure code makes the write fail.
    DCAMERR m_speedSetResult{ DCAMERR_SUCCESS };

    /// Properties dcamprop_getattr reports as missing. The cooler properties are absent, as in Air mode.
    std::set<int32> m_missing{ DCAM_IDPROP_SENSORCOOLER, DCAM_IDPROP_SENSORCOOLERFAN };

    /// Largest valid DCAM_IDPROP_SENSORCOOLER value; DCAMPROP_SENSORCOOLER__MAX when MAX is available.
    double m_coolerMax{ static_cast<double>( DCAMPROP_SENSORCOOLER__MAX ) };

    /// Result of dcamwait_start; DCAMERR_SUCCESS reports a frame-ready event.
    DCAMERR m_waitResult{ DCAMERR_TIMEOUT };

    /// Number of frames in the ring buffer, from dcambuf_alloc.
    int32 m_ringSize{ 0 };

    /// Total frames transferred, reported by dcamcap_transferinfo. Frame k is in ring slot k % m_ringSize.
    int32 m_transferred{ 0 };

    /// Camera frames lost before frame 0's slot was written; a frame's framestamp is its number plus this.
    int32 m_stampOffset{ 0 };

    /// If positive, dcambuf_lockframe sets m_transferred to this, as if the camera kept writing during the lock.
    int32 m_transferredAfterLock{ 0 };

    /// The single image buffer dcambuf_lockframe returns; like the real DCAM, it only holds the frame locked
    /// by the latest call. Its one "pixel" is that frame's framestamp.
    int32 m_lockBuffer{ -1 };

    /// Call counts by function name.
    std::map<std::string, unsigned> m_calls;
};

/// Current test's fake state.
inline Fake g_fake;

/// Stops idle placeholder threads.
inline std::atomic<bool> g_stopIdle{ false };

/// The fake camera handle.
inline HDCAM camera()
{
    return reinterpret_cast<HDCAM>( 0x1000 );
}

/// The fake wait handle.
inline HDCAMWAIT waiter()
{
    return reinterpret_cast<HDCAMWAIT>( 0x2000 );
}

/// Count a call.
inline DCAMERR call( const std::string &name /**< [in] function name */ )
{
    ++g_fake.m_calls[name];
    return DCAMERR_SUCCESS;
}

} // namespace orcaHarness

extern "C"
{
    DCAMERR dcamapi_init( DCAMAPI_INIT *param )
    {
        auto &f = orcaHarness::g_fake;

        if( ++f.m_calls["dcamapi_init"] <= f.m_noCameraInits )
        {
            return DCAMERR_NOCAMERA;
        }

        if( failed( f.m_initResult ) )
        {
            return f.m_initResult;
        }

        param->iDeviceCount = 1;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamapi_uninit()
    {
        return orcaHarness::call( "dcamapi_uninit" );
    }

    DCAMERR dcamdev_open( DCAMDEV_OPEN *param )
    {
        param->hdcam = orcaHarness::camera();
        return orcaHarness::call( "dcamdev_open" );
    }

    DCAMERR dcamdev_close( HDCAM )
    {
        return orcaHarness::call( "dcamdev_close" );
    }

    DCAMERR dcamdev_getstring( HDCAM, DCAMDEV_STRING *param )
    {
        std::string text = "fake DCAM error";

        if( param->iString == DCAM_IDSTR_CAMERAID )
        {
            text = orcaHarness::g_fake.m_cameraID;
        }
        else if( param->iString == DCAM_IDSTR_VENDOR )
        {
            text = "Hamamatsu";
        }
        else if( param->iString == DCAM_IDSTR_MODEL )
        {
            text = "C15550-22UP";
        }

        snprintf( param->text, param->textbytes, "%s", text.c_str() );
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamprop_getattr( HDCAM, DCAMPROP_ATTR *param )
    {
        auto &f = orcaHarness::g_fake;

        if( f.m_missing.count( param->iProp ) > 0 )
        {
            return DCAMERR_INVALIDPROPERTYID;
        }

        param->attribute = DCAMPROP_ATTR_WRITABLE;
        param->valuemin  = 1e-5;
        param->valuemax  = param->iProp == DCAM_IDPROP_SENSORCOOLER ? f.m_coolerMax : 10;
        param->valuestep = 1e-6;

        // Readout speed 1 is Ultra quiet scan and 2 is Standard scan, which sets the exposure minimum
        if( param->iProp == DCAM_IDPROP_READOUTSPEED )
        {
            param->valuemin  = 1;
            param->valuemax  = 2;
            param->valuestep = 1;
        }
        else if( param->iProp == DCAM_IDPROP_EXPOSURETIME )
        {
            param->valuemin = f.m_props[DCAM_IDPROP_READOUTSPEED] == 1 ? 33.9e-6 : 7.2e-6;
        }

        return orcaHarness::call( "dcamprop_getattr" );
    }

    DCAMERR dcamprop_getvalue( HDCAM, int32 iProp, double *pValue )
    {
        auto &f = orcaHarness::g_fake;

        if( iProp == DCAM_IDPROP_READOUTSPEED && failed( f.m_speedGetResult ) )
        {
            return f.m_speedGetResult;
        }

        *pValue = f.m_props[iProp];
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamprop_setvalue( HDCAM, int32 iProp, double fValue )
    {
        orcaHarness::g_fake.m_props[iProp] = fValue;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamprop_setgetvalue( HDCAM, int32 iProp, double *pValue, int32 )
    {
        auto &f = orcaHarness::g_fake;

        // Like the camera, the readout speed is limited to its valid range [1, 2]
        if( iProp == DCAM_IDPROP_READOUTSPEED )
        {
            ++f.m_calls["set readout speed"];

            if( failed( f.m_speedSetResult ) )
            {
                return f.m_speedSetResult;
            }

            if( *pValue < 1 || *pValue > 2 )
            {
                return DCAMERR_OUTOFRANGE;
            }
        }

        f.m_props[iProp] = *pValue;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcambuf_alloc( HDCAM, int32 framecount )
    {
        orcaHarness::g_fake.m_ringSize = framecount;
        return orcaHarness::call( "dcambuf_alloc" );
    }

    DCAMERR dcambuf_release( HDCAM, int32 )
    {
        return orcaHarness::call( "dcambuf_release" );
    }

    DCAMERR dcambuf_lockframe( HDCAM, DCAMBUF_FRAME *pFrame )
    {
        auto &f = orcaHarness::g_fake;

        if( f.m_transferred <= 0 || f.m_ringSize <= 0 )
        {
            return DCAMERR_TIMEOUT;
        }

        // The newest frame number held in this slot, copied into the one lock buffer
        const int32 newest = f.m_transferred - 1;
        const int32 frame  = newest - ( ( newest - pFrame->iFrame ) % f.m_ringSize + f.m_ringSize ) % f.m_ringSize;

        pFrame->framestamp = frame + f.m_stampOffset;
        f.m_lockBuffer     = pFrame->framestamp;
        pFrame->buf        = &f.m_lockBuffer;

        if( f.m_transferredAfterLock > 0 )
        {
            f.m_transferred          = f.m_transferredAfterLock;
            f.m_transferredAfterLock = 0;
        }

        return orcaHarness::call( "dcambuf_lockframe" );
    }

    DCAMERR dcamcap_start( HDCAM, int32 )
    {
        orcaHarness::g_fake.m_captureStatus = DCAMCAP_STATUS_BUSY;
        return orcaHarness::call( "dcamcap_start" );
    }

    DCAMERR dcamcap_stop( HDCAM )
    {
        orcaHarness::g_fake.m_captureStatus = DCAMCAP_STATUS_READY;
        return orcaHarness::call( "dcamcap_stop" );
    }

    DCAMERR dcamcap_status( HDCAM, int32 *pStatus )
    {
        *pStatus = orcaHarness::g_fake.m_captureStatus;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamcap_transferinfo( HDCAM, DCAMCAP_TRANSFERINFO *param )
    {
        auto &f = orcaHarness::g_fake;

        param->nFrameCount       = f.m_transferred;
        param->nNewestFrameIndex = f.m_transferred > 0 && f.m_ringSize > 0 ? ( f.m_transferred - 1 ) % f.m_ringSize : -1;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamwait_open( DCAMWAIT_OPEN *param )
    {
        param->hwait = orcaHarness::waiter();
        return orcaHarness::call( "dcamwait_open" );
    }

    DCAMERR dcamwait_close( HDCAMWAIT )
    {
        return orcaHarness::call( "dcamwait_close" );
    }

    DCAMERR dcamwait_start( HDCAMWAIT, DCAMWAIT_START *param )
    {
        auto &f = orcaHarness::g_fake;

        param->eventhappened = f.m_waitResult == DCAMERR_SUCCESS ? DCAMWAIT_CAPEVENT_FRAMEREADY : 0;
        ++f.m_calls["dcamwait_start"];
        return f.m_waitResult;
    }

    DCAMERR dcamwait_abort( HDCAMWAIT )
    {
        return orcaHarness::call( "dcamwait_abort" );
    }
}

namespace MagAOX
{
namespace app
{

/// App base with captured logging whose thread starts are replaced by idle placeholders.
template <bool useINDI = true>
class orcaTestApp : public outletTestApp<useINDI>
{
  public:
    using outletTestApp<useINDI>::outletTestApp;

    /// Start an idle placeholder thread that keeps the handle valid instead of the real thread.
    template <class thisPtr, class Function>
    int threadStart( std::thread &thrd,
                     bool &,
                     pid_t &,
                     pcf::IndiProperty &,
                     int,
                     const std::string &,
                     const std::string &,
                     thisPtr *,
                     Function && )
    {
        thrd = std::thread(
            []
            {
                while( !orcaHarness::g_stopIdle )
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                }
            } );

        return 0;
    }
};

} // namespace app
} // namespace MagAOX

#define MagAOXApp orcaTestApp
#define telemeter outletTestTelemeter
#define protected public
#include "../orcaCtrl.hpp"
#undef protected
#undef telemeter
#undef MagAOXApp

namespace orcaHarness
{

/// The real app with an isolated directory, captured logs, and a simulated power loop.
struct Fixture : outletHarness::Controller<MagAOX::app::orcaCtrl>
{
    /// Expose the frameGrabber and MagAOXApp state used by the tests.
    using MagAOX::app::dev::frameGrabber<MagAOX::app::orcaCtrl>::m_reconfig;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::orcaCtrl>::m_fgThread;
    using MagAOX::app::MagAOXApp<true>::m_powerOnWait;

    /// Expose the stdCamera fan and temperature-control state used by the tests.
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_fanSpeedControlEnabled;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_defaultFanSpeed;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_tempControlStatus;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_tempControlStatusSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_tempControlStatusStr;

    /// Expose the stdCamera readout-speed and exposure state used by the tests.
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_readoutSpeedName;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_readoutSpeedNameSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_minExpTime;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_expTimeSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::orcaCtrl>::m_expTime;

    /// Configure one air-cooled camera and reset the fakes.
    Fixture()
    {
        g_fake                  = {};
        outletHarness::g_faults = {};
        m_serialNumber          = "PHX2";
        m_powerOnWait           = 0;
        m_liquidCooling         = false;
    }

    /// Make the fake camera and the app liquid cooled (Cooler Type Water).
    void liquid()
    {
        m_liquidCooling          = true;
        m_fanSpeedControlEnabled = false; // as loadConfigImpl() does
        g_fake.m_missing.clear();
    }

    /// Current DCAM_IDPROP_READOUTSPEED value in the fake camera (1 Ultra quiet, 2 Standard).
    static int32 speed()
    {
        return static_cast<int32>( g_fake.m_props[DCAM_IDPROP_READOUTSPEED] );
    }

    /// Current DCAM_IDPROP_SENSORCOOLER value in the fake camera.
    static int32 cooler()
    {
        return static_cast<int32>( g_fake.m_props[DCAM_IDPROP_SENSORCOOLER] );
    }

    /// Stop and join the placeholder framegrabber thread.
    ~Fixture()
    {
        g_stopIdle = true;

        if( m_fgThread.joinable() )
        {
            m_fgThread.join();
        }

        g_stopIdle = false;
    }

    /// Set observed and target power.
    void power( int observed /**< [in] observed state */, int target /**< [in] target state */ )
    {
        MagAOX::app::MagAOXApp<true>::m_powerState       = observed;
        MagAOX::app::MagAOXApp<true>::m_powerTargetState = target;
    }

    /// Start the app as MagAOXApp::execute does, with the given initial power.
    void start( int on /**< [in] initial observed and target power */ )
    {
        power( on, on );
        state( MagAOX::app::stateCodes::INITIALIZED );
        REQUIRE( appStartup() == 0 );

        if( on > 0 )
        {
            state( MagAOX::app::stateCodes::POWERON );
        }
        else
        {
            m_powerOnCounter = 0;
            state( MagAOX::app::stateCodes::POWEROFF );
            REQUIRE( onPowerOff() == 0 );
        }
    }

    /// Run one iteration of MagAOXApp's main-loop power handling and logic.
    int loop()
    {
        int &powerState = MagAOX::app::MagAOXApp<true>::m_powerState;

        if( state() == MagAOX::app::stateCodes::POWEROFF )
        {
            if( powerState == 1 )
            {
                m_powerOnCounter = 0;
                state( MagAOX::app::stateCodes::POWERON );
            }
        }
        else if( powerState == 0 )
        {
            state( MagAOX::app::stateCodes::POWEROFF );
            REQUIRE( onPowerOff() == 0 );
        }

        if( powerState > 0 )
        {
            return appLogic();
        }

        return whilePowerOff();
    }

    /// Number of captured logs at error priority or worse.
    static size_t errors()
    {
        size_t n = 0;

        for( auto &l : outletHarness::g_faults.m_logs )
        {
            n += l.m_priority <= flatlogs::logPrio::LOG_ERROR;
        }

        return n;
    }

    /// Number of captured logs containing text.
    static size_t count( const std::string &text /**< [in] text */ )
    {
        size_t n = 0;

        for( auto &l : outletHarness::g_faults.m_logs )
        {
            n += l.m_message.find( text ) != std::string::npos;
        }

        return n;
    }
};

} // namespace orcaHarness
/// \endcond

namespace libXWCTest
{

/** \addtogroup orcaCtrl_unit_test
 * \ingroup application_unit_test
 */
namespace orcaCtrlTest
{

using namespace MagAOX::app;
using namespace orcaHarness;

/// A power-on waits powerOnWait, then connects quietly while DCAM reports no camera, and reaches OPERATING.
/** dcamapi_init() fails with DCAMERR_NOCAMERA for a few seconds while the camera enumerates after power-on.
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl powers on without error logs while the camera enumerates", "[orcaCtrl][power]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::connect();
    orcaCtrl::getAcquisitionState();
    orcaCtrl::appLogic();
    #endif
    // clang-format on

    Fixture f;
    f.m_powerOnWait        = 1;
    g_fake.m_noCameraInits = 3;

    f.start( 0 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::POWEROFF );

    // The power-on wait passes with no DCAM calls.
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::POWERON );
    REQUIRE( g_fake.m_calls["dcamapi_init"] == 0 );

    for( int n = 0; n < 10 && f.state() == stateCodes::POWERON; ++n )
    {
        REQUIRE( f.loop() == 0 );
    }

    // While enumerating, the camera is reported once as not available.
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( g_fake.m_calls["dcamapi_init"] == 1 );

    for( int n = 0; n < 2; ++n )
    {
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
    }

    REQUIRE( g_fake.m_calls["dcamapi_init"] == 3 );
    REQUIRE( f.count( "no Hamamatsu available." ) == 1 );

    // Then it connects, and waits for the framegrabber to start capture.
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE( f.count( "Connected to Hamamatsu" ) == 1 );

    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::READY );

    g_fake.m_captureStatus = DCAMCAP_STATUS_BUSY;
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );

    REQUIRE( f.errors() == 0 );
    REQUIRE( f.count( "acquisition stopped" ) == 0 );
}

/// A software restart with the camera already on connects at once without error logs.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl restarts with the camera on without error logs", "[orcaCtrl][power]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::connect();
    orcaCtrl::getAcquisitionState();
    #endif
    // clang-format on

    Fixture f;
    f.m_powerOnWait = 5; // not applied: power was already on at startup

    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE( g_fake.m_calls["dcamapi_init"] == 1 );

    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::READY );

    REQUIRE( f.errors() == 0 );
}

/// Real failures are still reported: capture stopping while operating, and other dcamapi_init errors.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl still reports real startup and acquisition errors", "[orcaCtrl][power]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::connect();
    orcaCtrl::getAcquisitionState();
    #endif
    // clang-format on

    Fixture f;

    SECTION( "capture stopping while operating is logged and restarts acquisition" )
    {
        f.start( 1 );
        g_fake.m_captureStatus = DCAMCAP_STATUS_BUSY;
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );

        f.m_reconfig           = false;
        g_fake.m_captureStatus = DCAMCAP_STATUS_READY;
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::READY );
        REQUIRE( f.count( "acquisition stopped. restarting" ) == 1 );
        REQUIRE( f.m_reconfig );
    }

    SECTION( "dcamapi_init errors other than no camera are logged" )
    {
        g_fake.m_initResult = DCAMERR_NOTSUPPORT;
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::ERROR );
        REQUIRE( f.count( "Failed to initialize DCAM API." ) == 1 );
    }
}

/// `camera.liquidCooling` hides fan control; air cooling leaves it alone.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl liquid-cooling configuration", "[orcaCtrl][cooling]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::setupConfig();
    orcaCtrl::loadConfigImpl();
    #endif
    // clang-format on

    Fixture f;

    SECTION( "liquid cooling hides fan control" )
    {
        f.configText( "[camera]\nliquidCooling=true\nfanSpeedControl=true\n" );
        REQUIRE( f.loadConfigImpl( f.config ) == 0 );
        REQUIRE( f.m_liquidCooling );
        REQUIRE( !f.m_fanSpeedControlEnabled );
        REQUIRE( f.m_defaultFanSpeed == "off" );
    }

    SECTION( "air cooling leaves fan control alone" )
    {
        f.configText( "[camera]\nliquidCooling=false\n" );
        REQUIRE( f.loadConfigImpl( f.config ) == 0 );
        REQUIRE( !f.m_liquidCooling );
        REQUIRE( f.m_fanSpeedControlEnabled );
    }
}

/// A camera whose Cooler Type doesn't match `camera.liquidCooling` puts the app in ERROR, logged once.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl reports a Cooler Type mismatch", "[orcaCtrl][cooling]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::connect();
    #endif
    // clang-format on

    Fixture f;

    SECTION( "liquid config with an Air-mode camera" )
    {
        f.m_liquidCooling = true;
    }

    SECTION( "air config with a Water-mode camera" )
    {
        g_fake.m_missing.clear();
    }

    f.start( 1 );

    for( int n = 0; n < 3; ++n )
    {
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::ERROR );
    }

    REQUIRE( f.count( "Cooler Type mismatch" ) == 1 );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 0 );
}

/// In liquid mode, configureAcquisition() turns the fan off and the sensor cooler to MAX, or ON without MAX.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl applies liquid cooling", "[orcaCtrl][cooling]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::configureAcquisition();
    orcaCtrl::setSensorCooler();
    orcaCtrl::getTemps();
    #endif
    // clang-format on

    Fixture f;
    f.liquid();

    SECTION( "MAX available" )
    {
    }

    SECTION( "MAX not available" )
    {
        g_fake.m_coolerMax = static_cast<double>( DCAMPROP_SENSORCOOLER__ON );
    }

    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::READY );
    REQUIRE( f.count( "cooling mode: liquid" ) == 1 );

    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( g_fake.m_props[DCAM_IDPROP_SENSORCOOLERFAN] == static_cast<double>( DCAMPROP_MODE__OFF ) );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 1 );

    if( g_fake.m_coolerMax == static_cast<double>( DCAMPROP_SENSORCOOLER__MAX ) )
    {
        REQUIRE( f.cooler() == DCAMPROP_SENSORCOOLER__MAX );
        REQUIRE( f.count( "MAX is not available" ) == 0 );
    }
    else
    {
        REQUIRE( f.cooler() == DCAMPROP_SENSORCOOLER__ON );
        REQUIRE( f.count( "MAX is not available" ) == 1 );
    }

    // A cooler reported OFF reads as OFF rather than UNKNOWN
    g_fake.m_props[DCAM_IDPROP_SENSORCOOLERSTATUS] = DCAMPROP_SENSORCOOLERSTATUS__OFF;
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_tempControlStatusStr == "OFF" );
}

/// In liquid mode the `temp_controller` toggle turns the sensor cooler off and on; in air mode it stays on.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl temperature-control toggle", "[orcaCtrl][cooling]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::setTempControl();
    orcaCtrl::sensorCoolerTarget();
    #endif
    // clang-format on

    Fixture f;

    SECTION( "liquid" )
    {
        f.liquid();
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_tempControlStatusSet );

        f.m_tempControlStatusSet = false;
        REQUIRE( f.setTempControl() == 0 );
        REQUIRE( f.cooler() == DCAMPROP_SENSORCOOLER__OFF );

        f.m_tempControlStatusSet = true;
        REQUIRE( f.setTempControl() == 0 );
        REQUIRE( f.cooler() == DCAMPROP_SENSORCOOLER__MAX );
    }

    SECTION( "air" )
    {
        f.start( 1 );
        REQUIRE( f.loop() == 0 );

        f.m_tempControlStatusSet = false;
        REQUIRE( f.setTempControl() == 0 );
        REQUIRE( f.m_tempControlStatus );
        REQUIRE( f.m_tempControlStatusSet );
        REQUIRE( g_fake.m_props.count( DCAM_IDPROP_SENSORCOOLER ) == 0 );
    }

    REQUIRE( f.errors() == 0 );
}

/// Selecting a readout speed and reconfiguring switches the camera between Standard and Ultra quiet scan,
/// restarts capture, and updates the reported speed and the exposure limits.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl switches between Standard and Ultra quiet readout", "[orcaCtrl][readout]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::setReadoutSpeed();
    orcaCtrl::configureAcquisition();
    orcaCtrl::reconfig();
    #endif
    // clang-format on

    Fixture f;
    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_readoutSpeedNameSet == "standard" );

    // The camera powers up in Standard, so nothing is written
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.speed() == 2 );
    REQUIRE( f.m_readoutSpeedName == "standard" );
    REQUIRE( f.m_minExpTime == Approx( 7.2e-6 ) );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 1 );
    REQUIRE( g_fake.m_calls["set readout speed"] == 0 );

    // To Ultra quiet, with an exposure below its minimum
    f.m_expTimeSet          = 10e-6;
    f.m_readoutSpeedNameSet = "ultra_quiet";
    f.m_reconfig            = false;
    REQUIRE( f.setReadoutSpeed() == 0 );
    REQUIRE( f.m_reconfig );

    REQUIRE( f.reconfig() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.speed() == 1 );
    REQUIRE( f.m_readoutSpeedName == "ultra_quiet" );
    REQUIRE( f.m_minExpTime == Approx( 33.9e-6 ) );
    REQUIRE( f.m_expTime == Approx( 33.9e-6 ) );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 2 );
    REQUIRE( g_fake.m_calls["set readout speed"] == 1 );
    REQUIRE( f.count( "readout speed changed from standard to ultra_quiet" ) == 1 );

    // And back to Standard
    f.m_readoutSpeedNameSet = "standard";
    REQUIRE( f.setReadoutSpeed() == 0 );
    REQUIRE( f.reconfig() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.speed() == 2 );
    REQUIRE( f.m_readoutSpeedName == "standard" );
    REQUIRE( f.m_minExpTime == Approx( 7.2e-6 ) );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 3 );
    REQUIRE( f.count( "readout speed changed from ultra_quiet to standard" ) == 1 );

    REQUIRE( f.errors() == 0 );
}

/// `camera.defaultReadoutSpeed` sets the speed applied after power-on.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl starts in the configured default readout speed", "[orcaCtrl][readout]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::powerOnDefaults();
    orcaCtrl::connect();
    orcaCtrl::configureAcquisition();
    #endif
    // clang-format on

    Fixture f;
    f.configText( "[camera]\ndefaultReadoutSpeed=ultra_quiet\n" );
    REQUIRE( f.loadConfigImpl( f.config ) == 0 );
    REQUIRE( !f.m_liquidCooling );

    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_readoutSpeedNameSet == "ultra_quiet" );

    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.speed() == 1 );
    REQUIRE( f.m_readoutSpeedName == "ultra_quiet" );
    REQUIRE( f.errors() == 0 );
}

/// An unknown readout speed is logged and replaced by the current speed, and acquisition still starts.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl rejects an unknown readout speed", "[orcaCtrl][readout]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::configureAcquisition();
    #endif
    // clang-format on

    Fixture f;
    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.speed() == 2 );

    f.m_readoutSpeedNameSet = "bogus";
    REQUIRE( f.reconfig() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.count( "unknown readout speed 'bogus'" ) == 1 );
    REQUIRE( f.m_readoutSpeedNameSet == "standard" );
    REQUIRE( f.m_readoutSpeedName == "standard" );
    REQUIRE( f.speed() == 2 );
    REQUIRE( g_fake.m_calls["dcamcap_start"] == 2 );
    REQUIRE( f.errors() == 0 );
}

/// A readout speed that can't be read or set is a single warning, and acquisition still starts at the
/// camera's current speed instead of failing and being retried every second.
/** \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl keeps acquiring when the readout speed can't be read or set", "[orcaCtrl][readout]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::configureAcquisition();
    #endif
    // clang-format on

    Fixture f;

    SECTION( "read fails" )
    {
        g_fake.m_speedGetResult = DCAMERR_BUSY;

        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.configureAcquisition() == 0 );
        REQUIRE( f.count( "could not read the readout speed" ) == 1 );
        REQUIRE( g_fake.m_calls["set readout speed"] == 0 );
        REQUIRE( f.m_readoutSpeedName == "standard" );
    }

    SECTION( "write fails" )
    {
        g_fake.m_speedSetResult = DCAMERR_BUSY;

        f.start( 1 );
        REQUIRE( f.loop() == 0 );

        f.m_readoutSpeedNameSet = "ultra_quiet";
        REQUIRE( f.configureAcquisition() == 0 );
        REQUIRE( f.count( "could not set the readout speed to ultra_quiet" ) == 1 );
        REQUIRE( f.speed() == 2 );
        REQUIRE( f.m_readoutSpeedName == "standard" );
        REQUIRE( f.m_readoutSpeedNameSet == "standard" );

        // The failed request is not retried on the next reconfigure
        REQUIRE( f.reconfig() == 0 );
        REQUIRE( f.configureAcquisition() == 0 );
        REQUIRE( f.count( "could not set the readout speed" ) == 1 );
    }

    REQUIRE( g_fake.m_calls["dcamcap_start"] >= 1 );
    REQUIRE( f.errors() == 0 );
}

/// Camera frames that never reach the ring buffer are counted from framestamp gaps and logged as
/// `num frames skipped`, a DCAMERR_LOSTFRAME wait result doesn't put the app in ERROR, and a frame
/// DCAM overwrites while it is being locked is dropped rather than published.
/** In build dc780bc9, DCAMERR_LOSTFRAME from dcamwait_start() put camscc in ERROR, and frames lost
 * before the ring buffer were never counted in `acq_stats.dropped`
 * (agents/plans/2026-10/orcaCtrl-latency-new.md).
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl counts skipped and lost frames and keeps acquiring", "[orcaCtrl][frames]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::acquireAndCheckValid();
    orcaCtrl::appLogic();
    #endif
    // clang-format on

    Fixture f;
    f.driver(); // updateIfChanged() only sets property values with an INDI driver attached
    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( g_fake.m_ringSize == 10 );

    // Three frames already in the ring buffer are published in order without waiting
    g_fake.m_transferred = 3;

    for( int n = 0; n < 3; ++n )
    {
        REQUIRE( f.acquireAndCheckValid() == 0 );
    }

    REQUIRE( f.m_statFrames.load() == 3 );
    REQUIRE( g_fake.m_calls["dcamwait_start"] == 0 );

    // The camera loses 5 frames before the next one reaches the ring buffer
    g_fake.m_stampOffset = 5;
    g_fake.m_transferred = 4;
    REQUIRE( f.acquireAndCheckValid() == 0 );
    REQUIRE( f.m_statSkipped.load() == 5 );
    REQUIRE( f.m_statStampErrors.load() == 0 );

    // DCAM reports the lost frames on the next wait; capture continues
    g_fake.m_waitResult = DCAMERR_LOSTFRAME;
    REQUIRE( f.acquireAndCheckValid() == 1 );
    REQUIRE( f.m_statLostEvents.load() == 1 );
    REQUIRE( f.state() != stateCodes::ERROR );

    // DCAM writes 9 more frames while frame 4's slot is first locked, so the slot may hold frame 14
    g_fake.m_waitResult           = DCAMERR_SUCCESS;
    g_fake.m_transferred          = 5;
    g_fake.m_transferredAfterLock = 14;
    REQUIRE( f.acquireAndCheckValid() == 1 );
    REQUIRE( f.m_statDropped.load() == 1 );
    REQUIRE( f.m_statFrames.load() == 4 );

    // Publishing resumes at the oldest safe frame (6), dropping frame 5, with no false skips
    REQUIRE( f.acquireAndCheckValid() == 0 );
    REQUIRE( f.m_statDropped.load() == 2 );
    REQUIRE( f.m_statSkipped.load() == 5 );
    REQUIRE( f.m_statStampErrors.load() == 0 );
    REQUIRE( f.m_statFrames.load() == 5 );

    // appLogic logs the counts once and publishes them
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.count( "num frames skipped: 5 " ) == 1 );
    REQUIRE( f.count( "lost-frame events: 1," ) == 1 );
    REQUIRE( f.count( "dropped: 2 " ) == 1 );
    REQUIRE( f.m_indiP_acqStats["skipped"].get<double>() == 5 );
    REQUIRE( f.m_indiP_acqStats["lost_events"].get<double>() == 1 );
    REQUIRE( f.m_indiP_acqStats["dropped"].get<double>() == 2 );

    REQUIRE( f.loop() == 0 );
    REQUIRE( f.count( "num frames skipped" ) == 1 );
    REQUIRE( f.errors() == 0 );
}

/// Each published frame carries its own image data, not a stale copy of an earlier one.
/** DCAM's dcambuf_lockframe() address only holds the frame locked by that call. Build 690b40c1 cached it
 * per ring slot and locked again only about once a second, so camscc at 12.5 Hz published one image under
 * about 12 consecutive frame counts and DM +/- measurements came out identical
 * (agents/plans/2026-10/orcaCtrl-latency-new.md, 2026-11-23).
 *
 * \ingroup orcaCtrl_unit_test
 */
TEST_CASE( "orcaCtrl publishes fresh image data for every frame", "[orcaCtrl][frames]" )
{
    // clang-format off
    #ifdef ORCACTRL_TEST_DOXYGEN_REF
    orcaCtrl::acquireAndCheckValid();
    #endif
    // clang-format on

    Fixture f;
    g_fake.m_props[DCAM_IDPROP_INTERNALFRAMERATE] = 12.5;
    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( g_fake.m_ringSize == 10 );

    // Three passes through the ring, one new frame at a time
    for( int32 n = 0; n < 30; ++n )
    {
        g_fake.m_transferred = n + 1;
        REQUIRE( f.acquireAndCheckValid() == 0 );
        REQUIRE( *static_cast<int32 *>( f.m_currentFrame.buf ) == n );
    }

    REQUIRE( g_fake.m_calls["dcambuf_lockframe"] == 30 );
    REQUIRE( f.m_statFrames.load() == 30 );
    REQUIRE( f.m_statSkipped.load() == 0 );
    REQUIRE( f.m_statStampErrors.load() == 0 );
    REQUIRE( f.errors() == 0 );
}

} // namespace orcaCtrlTest
} // namespace libXWCTest
