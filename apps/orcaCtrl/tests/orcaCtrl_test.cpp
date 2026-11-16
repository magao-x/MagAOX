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
#include <map>

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
    std::map<int32, double> m_props;

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
        param->attribute = DCAMPROP_ATTR_WRITABLE;
        param->valuemin  = 1e-5;
        param->valuemax  = 10;
        param->valuestep = 1e-6;
        return orcaHarness::call( "dcamprop_getattr" );
    }

    DCAMERR dcamprop_getvalue( HDCAM, int32 iProp, double *pValue )
    {
        *pValue = orcaHarness::g_fake.m_props[iProp];
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamprop_setvalue( HDCAM, int32 iProp, double fValue )
    {
        orcaHarness::g_fake.m_props[iProp] = fValue;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcamprop_setgetvalue( HDCAM, int32 iProp, double *pValue, int32 )
    {
        orcaHarness::g_fake.m_props[iProp] = *pValue;
        return DCAMERR_SUCCESS;
    }

    DCAMERR dcambuf_alloc( HDCAM, int32 )
    {
        return orcaHarness::call( "dcambuf_alloc" );
    }

    DCAMERR dcambuf_release( HDCAM, int32 )
    {
        return orcaHarness::call( "dcambuf_release" );
    }

    DCAMERR dcambuf_lockframe( HDCAM, DCAMBUF_FRAME * )
    {
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
        param->nNewestFrameIndex = -1;
        param->nFrameCount       = 0;
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

    DCAMERR dcamwait_start( HDCAMWAIT, DCAMWAIT_START * )
    {
        return DCAMERR_TIMEOUT;
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

    /// Configure one camera and reset the fakes.
    Fixture()
    {
        g_fake                  = {};
        outletHarness::g_faults = {};
        m_serialNumber          = "PHX2";
        m_powerOnWait           = 0;
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

} // namespace orcaCtrlTest
} // namespace libXWCTest
