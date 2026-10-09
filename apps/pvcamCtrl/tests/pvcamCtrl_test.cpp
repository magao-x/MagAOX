/** \file pvcamCtrl_test.cpp
 * \brief Offline tests of pvcamCtrl configuration, lifecycle, camera control, and acquisition with a fake PVCAM.
 *
 * \ingroup pvcamCtrl_files
 */

#include "pvcamCtrl_harness.hpp"

namespace libXWCTest
{
/** \addtogroup pvcamCtrl_unit_test
 * \ingroup application_unit_test
 */
namespace pvcamCtrlTest
{
using namespace MagAOX::app;
using namespace pvcamHarness;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// An app with a connected camera handle and initialized semaphores, without a PCIe port.
struct CameraFixture : Fixture
{
    /// Start with power on and connect.
    CameraFixture()
    {
        start( 1 );
        REQUIRE( loop() == 0 );
        REQUIRE( state() == stateCodes::OPERATING );
        outletHarness::g_faults.m_logs.clear();
        g_fake.m_calls.clear();
        g_fake.m_sets.clear();
    }

    /// Fail a pl_get_param call.
    static void failGet( uns32    param /**< [in] parameter */,
                         int16    attr /**< [in] attribute */,
                         unsigned n = 0 /**< [in] call, or 0 for every call */ )
    {
        if( n == 0 )
            g_fake.m_failAlways.insert( getKey( param, attr ) );
        else
            g_fake.m_failAt[getKey( param, attr )] = n;
    }

    /// Fail a pl_set_param call.
    static void failSet( uns32 param /**< [in] parameter */ )
    {
        g_fake.m_failAlways.insert( setKey( param ) );
    }

    /// Fail a named PVCAM function.
    static void fail( const std::string &fn /**< [in] function */,
                      unsigned           n = 0 /**< [in] call, or 0 for every call */ )
    {
        if( n == 0 )
            g_fake.m_failAlways.insert( fn );
        else
            g_fake.m_failAt[fn] = n;
    }

    /// Clear injected failures and call counts.
    static void clear()
    {
        g_fake.m_failAlways.clear();
        g_fake.m_failAt.clear();
        g_fake.m_calls.clear();
        g_fake.m_libcFail.clear();
        g_fake.m_libcFailAt.clear();
        outletHarness::g_faults.m_logs.clear();
    }

    /// Set a fake parameter value.
    static void
    value( uns32 param /**< [in] parameter */, int16 attr /**< [in] attribute */, long long v /**< [in] value */ )
    {
        g_fake.m_values[{ param, attr }] = v;
    }
};
/// \endcond

/// The PVCAM error formatter, and configuration loading including the PCIe keys.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl loads its configuration", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamErrMessage(); pvcamCtrl::pvcamCtrl(); pvcamCtrl::setupConfig(); pvcamCtrl::loadConfigImpl();
    pvcamCtrl::loadConfig();
    #endif
    // clang-format on
    REQUIRE( pvcamErrMessage( "pl_x", 3, "" ) == "pl_x failed: fake error 3" );
    REQUIRE( pvcamErrMessage( "pl_x", 3, "more" ) == "pl_x failed: fake error 3 more" );

    SECTION( "valid" )
    {
        Fixture f;
        f.m_serialNumber.clear();
        f.configText( "[camera]\nserialNumber=A22J723004\ncircBuffMaxBytes=1000\n[framegrabber]\nacqSleep=7\n"
                      "[pcie]\ndownstreamPort=0000:42:08.0\nretryInterval=5\n" );
        f.loadConfig();
        REQUIRE_FALSE( f.m_shutdown );
        REQUIRE( f.m_serialNumber == "A22J723004" );
        REQUIRE( f.m_circBuffMaxBytes == 1000 );
        REQUIRE( f.m_acqSleep == 7 );
        REQUIRE( f.m_pcie.port() == "0000:42:08.0" );
        REQUIRE( f.m_pcieRetryInterval == 5 );
    }

    SECTION( "missing serial number" )
    {
        Fixture f;
        f.m_serialNumber.clear();
        f.configText( "" );
        f.loadConfig();
        REQUIRE( f.m_shutdown );
        REQUIRE( Fixture::logged( "camera serial number not provided" ) );
        REQUIRE( Fixture::logged( "error loading config" ) );
    }

    SECTION( "invalid PCIe port" )
    {
        Fixture f;
        f.configText( "[camera]\nserialNumber=A22J723004\n[pcie]\ndownstreamPort=42:08.0\n" );
        f.loadConfig();
        REQUIRE( f.m_shutdown );
        REQUIRE( Fixture::logged( "pcie.downstreamPort: invalid PCI address: '42:08.0'" ) );
    }
}

/// Startup failures, and shutdown that logs and continues past PVCAM and shutter errors.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl starts up and shuts down", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::appStartup(); pvcamCtrl::appShutdown(); pvcamCtrl::releaseCamera();
    #endif
    // clang-format on
    for( unsigned n : { 1u, 2u } )
    {
        Fixture f;
        g_fake.m_libcFailAt["sem_init"] = n;
        REQUIRE( f.appStartup() == -1 );
        REQUIRE( Fixture::logged( n == 1 ? "frame ready semaphore" : "frame done semaphore" ) );
    }

    {
        Fixture f;
        g_fake.m_shutter[0] = -1;
        REQUIRE( f.appStartup() == -1 );
    }

    CameraFixture f;
    CameraFixture::fail( "pl_cam_close" );
    CameraFixture::fail( "pl_pvcam_uninit" );
    g_fake.m_shutter[2] = -1;
    g_stopIdle          = true;
    REQUIRE( f.appShutdown() == 0 );
    REQUIRE( f.m_handle == -1 );
    REQUIRE( Fixture::logged( "pl_cam_close failed: fake error 1 continuing" ) );
    REQUIRE( Fixture::logged( "pl_pvcam_uninit failed: fake error 1 continuing" ) );
    REQUIRE( Fixture::logged( "error from shutterT::appShutdown()" ) );
}

/// The stdCamera setter interface maps names and limits onto PVCAM parameters.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl stdCamera setters", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::powerOnDefaults(); pvcamCtrl::setTempControl(); pvcamCtrl::setTempSetPt();
    pvcamCtrl::setReadoutSpeed(); pvcamCtrl::setVShiftSpeed(); pvcamCtrl::setFanSpeed(); pvcamCtrl::setEMGain();
    pvcamCtrl::setExpTime(); pvcamCtrl::setFPS(); pvcamCtrl::checkNextROI(); pvcamCtrl::setNextROI();
    pvcamCtrl::setShutter();
    #endif
    // clang-format on
    CameraFixture f;

    f.m_fanSpeedControlEnabled = false;
    REQUIRE( f.powerOnDefaults() == 0 );
    REQUIRE( f.m_fanSpeedName.empty() );
    f.m_fanSpeedControlEnabled = true;
    REQUIRE( f.powerOnDefaults() == 0 );
    REQUIRE( f.m_fanSpeedName == "high" );

    REQUIRE( f.setTempControl() == 0 );
    REQUIRE( f.setTempSetPt() == 0 );
    REQUIRE( f.setVShiftSpeed() == 0 );
    REQUIRE( f.setEMGain() == 0 );
    f.m_reconfig = false;
    REQUIRE( f.setReadoutSpeed() == 0 );
    REQUIRE( f.m_reconfig );
    f.m_reconfig = false;
    REQUIRE( f.setNextROI() == 0 );
    REQUIRE( f.m_reconfig );
    REQUIRE( f.setShutter( 1 ) == 0 );
    REQUIRE( g_fake.m_shutterStates == std::vector<int>{ 1 } );

    std::vector<std::pair<std::string, int>> fans{ { "medium", FAN_SPEED_MEDIUM },
                                                   { "low", FAN_SPEED_LOW },
                                                   { "off", FAN_SPEED_OFF },
                                                   { "high", FAN_SPEED_HIGH } };
    for( auto &[name, code] : fans )
    {
        f.m_fanSpeedNameSet = name;
        REQUIRE( f.setFanSpeed() == 0 );
        REQUIRE( g_fake.m_sets.back() == std::pair<uns32, long long>( PARAM_FAN_SPEED_SETPOINT, code ) );
        REQUIRE( f.m_fanSpeedName == name );
    }
    REQUIRE( Fixture::logged( "fan speed changed from 'off' to 'high'" ) );
    REQUIRE( f.setFanSpeed() == 0 );
    REQUIRE( Fixture::logged( "fan speed set to 'high'" ) );
    f.m_fanSpeedNameSet = "fast";
    REQUIRE( f.setFanSpeed() == -1 );
    REQUIRE( Fixture::logged( "invalid fan-speed target: fast" ) );
    f.m_fanSpeedNameSet = "low";
    CameraFixture::failSet( PARAM_FAN_SPEED_SETPOINT );
    REQUIRE( f.setFanSpeed() == -1 );
    REQUIRE( f.m_fanSpeedName == "high" );
    CameraFixture::clear();

    // Exposure time limits are in microseconds.
    CameraFixture::value( PARAM_EXPOSURE_TIME, ATTR_MIN, 2000000 );
    CameraFixture::value( PARAM_EXPOSURE_TIME, ATTR_MAX, 9000000 );
    f.m_expTimeSet = 1;
    REQUIRE( f.setExpTime() == 0 );
    REQUIRE( f.m_expTimeSet == 2 );
    f.m_expTimeSet = 10;
    REQUIRE( f.setExpTime() == 0 );
    REQUIRE( f.m_expTimeSet == 8 );
    f.m_expTimeSet = 5;
    REQUIRE( f.setExpTime() == 0 );
    REQUIRE( f.m_expTimeSet == 5 );
    for( int16 attr : { ATTR_MIN, ATTR_MAX } )
    {
        CameraFixture::failGet( PARAM_EXPOSURE_TIME, attr );
        REQUIRE( f.setExpTime() == -1 );
        CameraFixture::clear();
    }

    f.m_fpsSet = 4;
    REQUIRE( f.setFPS() == 0 );
    REQUIRE( f.m_expTimeSet == Approx( 2 ) );
    REQUIRE( f.m_fpsSetted );

    struct Roi
    {
        float x, y, w, h, ex, ey, ew, eh;
    };
    for( auto r : std::vector<Roi>{ { 1600, 1600, 4000, 4000, 1599.5, 1599.5, 3199, 3199 },
                                    { 100, 100, 400, 400, 0, 0, 300, 300 },
                                    { 3100, 3100, 400, 400, 3049.5, 3049.5, 299, 299 },
                                    { 1600, 1600, 512, 512, 1600, 1600, 512, 512 } } )
    {
        f.m_nextROI.x = r.x;
        f.m_nextROI.y = r.y;
        f.m_nextROI.w = r.w;
        f.m_nextROI.h = r.h;
        REQUIRE( f.checkNextROI() == 0 );
        REQUIRE( f.m_nextROI.x == Approx( r.ex ) );
        REQUIRE( f.m_nextROI.y == Approx( r.ey ) );
        REQUIRE( f.m_nextROI.w == Approx( r.ew ) );
        REQUIRE( f.m_nextROI.h == Approx( r.eh ) );
    }
}

/// Acquisition setup maps readout speeds, exposure, and FPS onto PVCAM and allocates the circular buffer.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl configures acquisition", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::configureAcquisition(); pvcamCtrl::fps();
    #endif
    // clang-format on
    CameraFixture f;
    f.m_circBuffMaxBytes = 64;
    CameraFixture::value( PARAM_EXPOSURE_TIME, ATTR_CURRENT, 20000 );
    CameraFixture::value( PARAM_READOUT_TIME, ATTR_CURRENT, 10000 );
    CameraFixture::value( PARAM_POST_TRIGGER_DELAY, ATTR_CURRENT, 1000 );

    std::vector<std::tuple<std::string, int, bool>> speeds{ { "sensitivity", 0, false },
                                                            { "speed", 1, true },
                                                            { "sub_electron", 3, false },
                                                            { "dynamic_range", 2, false },
                                                            { "bogus", 2, false } };
    for( auto &[name, port, eightBit] : speeds )
    {
        f.m_readoutSpeedNameSet = name;
        f.m_expTimeSet          = 0.02;
        REQUIRE( f.configureAcquisition() == 0 );
        REQUIRE( std::count( g_fake.m_sets.begin(),
                             g_fake.m_sets.end(),
                             std::pair<uns32, long long>( PARAM_READOUT_PORT, port ) ) >= 1 );
        REQUIRE( f.m_8bit == eightBit );
        REQUIRE( f.m_readoutSpeedName == ( name == "bogus" ? "dynamic_range" : name ) );
    }
    REQUIRE( g_fake.m_setupExposure == 19999 ); // 0.02 s truncated to integer microseconds
    REQUIRE( f.m_expTime == Approx( 0.02 ) );
    REQUIRE( f.fps() == Approx( 1.0 / ( 0.02 + 2e-6 ) ) );
    REQUIRE( f.m_circBuffBytes == 64 );

    // Readout limited.
    CameraFixture::value( PARAM_EXPOSURE_TIME, ATTR_CURRENT, 5000 );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( f.fps() == Approx( 1.0 / ( 0.01 + 1e-6 ) ) );

    // FPS requested: readout limited, not limited, and limited after the post-trigger correction.
    for( auto [exposure, fps] :
         std::vector<std::pair<long long, float>>{ { 5000, 50 }, { 20000, 40 }, { 20000, 101 } } )
    {
        CameraFixture::value( PARAM_EXPOSURE_TIME, ATTR_CURRENT, exposure );
        f.m_fpsSetted = true;
        f.m_fpsSet    = fps;
        REQUIRE( f.configureAcquisition() == 0 );
        REQUIRE_FALSE( f.m_fpsSetted );
    }

    // Logged and continued: deregistration and parameter reads.
    CameraFixture::fail( "pl_cam_deregister_callback" );
    for( uns32 p : { PARAM_EXPOSURE_TIME,
                     PARAM_READOUT_TIME,
                     PARAM_PRE_TRIGGER_DELAY,
                     PARAM_CLEARING_TIME,
                     PARAM_POST_TRIGGER_DELAY } )
        CameraFixture::failGet( p, ATTR_CURRENT );
    f.m_fpsSetted = true;
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( Fixture::count( "pl_get_param failed" ) == 7 );
    REQUIRE( Fixture::logged( "pl_cam_deregister_callback failed" ) );
    CameraFixture::clear();

    // Fatal failures.
    CameraFixture::fail( "pl_cam_register_callback_ex3" );
    REQUIRE( f.configureAcquisition() == -1 );
    CameraFixture::clear();
    CameraFixture::failSet( PARAM_READOUT_PORT );
    REQUIRE( f.configureAcquisition() == -1 );
    CameraFixture::clear();
    for( unsigned n : { 1u, 2u } )
    {
        f.m_fpsSetted = true;
        CameraFixture::fail( "pl_exp_setup_cont", n );
        REQUIRE( f.configureAcquisition() == -1 );
        REQUIRE( f.m_shutdown );
        f.m_shutdown = false;
        CameraFixture::clear();
    }
    f.m_fpsSetted = false;

    // Fan speed is restored after setup.
    f.state( stateCodes::READY );
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_CURRENT, FAN_SPEED_LOW );
    REQUIRE( f.configureAcquisition() == 0 );
    REQUIRE( g_fake.m_sets.back() == std::pair<uns32, long long>( PARAM_FAN_SPEED_SETPOINT, FAN_SPEED_HIGH ) );
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_CURRENT, FAN_SPEED_LOW );
    CameraFixture::failSet( PARAM_FAN_SPEED_SETPOINT );
    REQUIRE( f.configureAcquisition() == -1 );
    REQUIRE( Fixture::logged( "could not restore configured fan speed after acquisition setup" ) );
    CameraFixture::clear();
    CameraFixture::failGet( PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL );
    REQUIRE( f.configureAcquisition() == -1 );
    REQUIRE( Fixture::logged( "could not get fan speed after acquisition setup" ) );
    CameraFixture::clear();
    f.m_fanSpeedControlEnabled = false;
    REQUIRE( f.configureAcquisition() == 0 );

    // The circular buffer allocation fails when address space is exhausted.
    rlimit saved;
    REQUIRE( getrlimit( RLIMIT_AS, &saved ) == 0 );
    std::ifstream status( "/proc/self/status" );
    std::string   line;
    rlim_t        vm = 0;
    while( std::getline( status, line ) )
        if( line.starts_with( "VmSize:" ) )
            vm = std::stoull( line.substr( 7 ) ) * 1024;
    rlimit tight{ vm + ( 64u << 20 ), saved.rlim_max };
    f.m_circBuffMaxBytes = 1u << 30;
    REQUIRE( setrlimit( RLIMIT_AS, &tight ) == 0 );
    int rv = f.configureAcquisition();
    REQUIRE( setrlimit( RLIMIT_AS, &saved ) == 0 );
    REQUIRE( rv == -1 );
    REQUIRE( f.state() == stateCodes::FAILURE );
    REQUIRE( Fixture::logged( "failed to allocate acquisition circular buffer." ) );
}

/// Acquisition start, frame-ready polling, frame copies, and reconfiguration.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl acquires and copies frames", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::startAcquisition(); pvcamCtrl::acquireAndCheckValid(); pvcamCtrl::loadImageIntoStream();
    pvcamCtrl::reconfig(); pvcamCtrl::st_endOfFrameCallback(); pvcamCtrl::endOfFrameCallback();
    #endif
    // clang-format on
    CameraFixture f;
    REQUIRE( f.startAcquisition() == 0 );
    CameraFixture::fail( "pl_exp_start_cont" );
    REQUIRE( f.startAcquisition() == -1 );
    CameraFixture::fail( "pl_exp_stop_cont" );
    REQUIRE( f.reconfig() == 0 );
    REQUIRE( Fixture::logged( "pl_exp_stop_cont failed" ) );
    CameraFixture::clear();

    REQUIRE( f.acquireAndCheckValid() == 1 );
    REQUIRE( sem_post( &f.m_frSemaphore ) == 0 );
    REQUIRE( f.acquireAndCheckValid() == 0 );
    g_fake.m_libcFail = { "sem_trywait" };
    REQUIRE( f.acquireAndCheckValid() == -1 );
    CameraFixture::clear();

    f.m_width  = 4;
    f.m_height = 2;
    for( bool eightBit : { true, false } )
    {
        std::vector<uint16_t> dest( 8, 0 );
        f.m_8bit = eightBit;
        REQUIRE( f.loadImageIntoStream( dest.data() ) == 0 );
        REQUIRE( dest[0] == ( eightBit ? 7 : 0x0707 ) );
    }
    std::vector<uint16_t> dest( 8, 0 );
    CameraFixture::fail( "pl_exp_get_latest_frame" );
    REQUIRE( f.loadImageIntoStream( dest.data() ) == -1 );
    CameraFixture::clear();
    g_fake.m_libcFail = { "sem_post" };
    REQUIRE( f.loadImageIntoStream( dest.data() ) == -1 );
    CameraFixture::clear();

    // The callback hands the frame to the writer and waits until it is copied.
    for( bool clockFails : { false, true } )
    {
        while( sem_trywait( &f.m_frSemaphore ) == 0 )
        {
        }
        REQUIRE( sem_post( &f.m_frDoneSemaphore ) == 0 );
        REQUIRE( sem_post( &f.m_frDoneSemaphore ) == 0 );
        if( clockFails )
            g_fake.m_libcFail = { "clock_gettime" };
        FRAME_INFO  info{ 0, 42, 0, 0, 0 };
        std::thread callback( [&] { pvcamCtrl::st_endOfFrameCallback( &info, &f ); } );
        while( f.acquireAndCheckValid() != 0 )
        {
        }
        REQUIRE( f.loadImageIntoStream( dest.data() ) == 0 );
        callback.join();
        REQUIRE( f.m_frameInfo.FrameNr == 42 );
        CameraFixture::clear();
    }
    REQUIRE( Fixture::count( "clock_gettime" ) == 0 );

    g_fake.m_libcFail = { "sem_post" };
    FRAME_INFO info{};
    f.endOfFrameCallback( &info );
    REQUIRE( Fixture::logged( "Error posting to frame ready semaphore" ) );
}

/// Finding and opening the configured camera by serial number, and every enumeration failure.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl connects to its camera by serial number", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::connect();
    #endif
    // clang-format on
    CameraFixture f;
    g_fake.m_cameras = {
        { "pvcamPCIE_0", "A22J723004" }, { "pvcamPCIE_1", "", false }, { "pvcamPCIE_2", "A22J723005" } };

    auto again = [&]( int expected, stateCodes::stateCodeT state )
    {
        f.state( stateCodes::NOTCONNECTED );
        REQUIRE( f.connect() == expected );
        if( expected == 0 )
            REQUIRE( f.state() == state );
        CameraFixture::clear();
    };

    again( 0, stateCodes::CONNECTED );
    REQUIRE( f.m_camName == "pvcamPCIE_2" );
    REQUIRE( f.m_handle == 102 );

    for( auto fn : { "pl_cam_close", "pl_pvcam_init", "pl_cam_get_total", "pl_cam_get_name" } )
    {
        CameraFixture::fail( fn );
        again( -1, stateCodes::NODEVICE );
    }
    f.m_handle = -1;

    // Uninitialization errors other than "not initialized" are logged and ignored.
    g_fake.m_initialized = true;
    CameraFixture::fail( "pl_pvcam_uninit" );
    f.state( stateCodes::NOTCONNECTED );
    REQUIRE( f.connect() == 0 );
    REQUIRE( Fixture::logged( "pl_pvcam_uninit failed: fake error 1 continuing" ) );
    CameraFixture::clear();
    f.m_handle = -1;

    // A camera that cannot be opened is skipped.
    CameraFixture::fail( "pl_cam_open", 3 );
    again( 0, stateCodes::NODEVICE );
    REQUIRE( Fixture::logged( "camera not found" ) == false );

    for( int16 attr : { ATTR_AVAIL, ATTR_CURRENT } )
    {
        for( bool closeFails : { false, true } )
        {
            CameraFixture::failGet( PARAM_HEAD_SER_NUM_ALPHA, attr );
            if( closeFails )
                CameraFixture::fail( "pl_cam_close" );
            again( -1, stateCodes::NODEVICE );
        }
    }

    // Closing a non-matching camera fails.
    CameraFixture::fail( "pl_cam_close", 1 );
    again( -1, stateCodes::NODEVICE );

    g_fake.m_cameras.clear();
    again( 0, stateCodes::NODEVICE );
    g_fake.m_cameras = { { "pvcamPCIE_0", "A22J723004" } };
    again( 0, stateCodes::NODEVICE );

    // Post-open parameter failures are logged; fan failures abort.
    g_fake.m_cameras = { { "pvcamPCIE_0", "A22J723005" } };
    CameraFixture::failSet( PARAM_EXP_RES_INDEX );
    CameraFixture::failGet( PARAM_EXP_RES, ATTR_CURRENT );
    CameraFixture::failGet( PARAM_EXP_RES_INDEX, ATTR_CURRENT );
    f.state( stateCodes::NOTCONNECTED );
    REQUIRE( f.connect() == 0 );
    REQUIRE( Fixture::count( "PARAM_EXP_RES" ) == 3 );
    CameraFixture::clear();
    f.m_handle = -1;
    CameraFixture::failGet( PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL );
    again( -1, stateCodes::CONNECTED );
    f.m_handle = -1;
    CameraFixture::failSet( PARAM_FAN_SPEED_SETPOINT );
    again( -1, stateCodes::CONNECTED );
    f.m_handle                 = -1;
    f.m_fanSpeedControlEnabled = false;
    again( 0, stateCodes::CONNECTED );
}

/// The readout-speed table and enum dump walk PVCAM's enumerations and stop at the first failure.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl enumerates readout speeds", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::fillSpeedTable(); pvcamCtrl::dumpEnum();
    #endif
    // clang-format on
    CameraFixture f;
    REQUIRE( f.fillSpeedTable() == -1 ); // Not CONNECTED or READY.
    f.dumpEnum( PARAM_READOUT_PORT, "PARAM_READOUT_PORT" );

    f.state( stateCodes::CONNECTED );
    CameraFixture::value( PARAM_READOUT_PORT, ATTR_COUNT, 2 );
    CameraFixture::value( PARAM_SPDTAB_INDEX, ATTR_COUNT, 2 );
    CameraFixture::value( PARAM_GAIN_INDEX, ATTR_COUNT, 2 );
    CameraFixture::value( PARAM_GAIN_INDEX, ATTR_MIN, 1 );
    CameraFixture::value( PARAM_GAIN_INDEX, ATTR_MAX, 2 );
    CameraFixture::value( PARAM_PIX_TIME, ATTR_CURRENT, 5 );
    REQUIRE( f.fillSpeedTable() == 0 );
    REQUIRE( f.m_ports.size() == 2 );
    REQUIRE( f.m_ports[1].name == "Speed" );
    REQUIRE( f.m_ports[1].speeds[1].pixTime == 5 );
    REQUIRE( f.m_ports[1].speeds[1].maxG == 2 );
    REQUIRE( f.m_ports[1].speeds[1].gains.size() == 2 );

    for( auto fail :
         std::vector<std::function<void()>>{ [] { CameraFixture::failGet( PARAM_READOUT_PORT, ATTR_COUNT ); },
                                             [] { CameraFixture::fail( "pl_enum_str_length" ); },
                                             [] { CameraFixture::fail( "pl_get_enum_param" ); },
                                             [] { CameraFixture::failSet( PARAM_READOUT_PORT ); },
                                             [] { CameraFixture::failGet( PARAM_SPDTAB_INDEX, ATTR_COUNT ); },
                                             [] { CameraFixture::failSet( PARAM_SPDTAB_INDEX ); },
                                             [] { CameraFixture::failGet( PARAM_PIX_TIME, ATTR_CURRENT ); },
                                             [] { CameraFixture::failGet( PARAM_GAIN_INDEX, ATTR_COUNT ); },
                                             [] { CameraFixture::failGet( PARAM_GAIN_INDEX, ATTR_MIN ); },
                                             [] { CameraFixture::failGet( PARAM_GAIN_INDEX, ATTR_MAX ); },
                                             [] { CameraFixture::failSet( PARAM_GAIN_INDEX ); },
                                             [] { CameraFixture::failGet( PARAM_BIT_DEPTH, ATTR_CURRENT ); } } )
    {
        fail();
        REQUIRE( f.fillSpeedTable() == -1 );
        CameraFixture::clear();
    }

    f.dumpEnum( PARAM_READOUT_PORT, "PARAM_READOUT_PORT" );
    CameraFixture::value( PARAM_READOUT_PORT, ATTR_COUNT, 0 );
    f.dumpEnum( PARAM_READOUT_PORT, "PARAM_READOUT_PORT" );
    CameraFixture::value( PARAM_READOUT_PORT, ATTR_COUNT, 2 );
    for( auto fn : { "get", "pl_enum_str_length", "pl_get_enum_param" } )
    {
        if( std::string( fn ) == "get" )
            CameraFixture::failGet( PARAM_READOUT_PORT, ATTR_COUNT );
        else
            CameraFixture::fail( fn );
        f.dumpEnum( PARAM_READOUT_PORT, "PARAM_READOUT_PORT" );
        REQUIRE( Fixture::logged( "PARAM_READOUT_PORT" ) );
        CameraFixture::clear();
    }
}

/// Temperature and fan-speed reads, with power-loss suppression of read errors.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl reads temperature and fan speed", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::getTemp(); pvcamCtrl::getFanSpeed();
    #endif
    // clang-format on
    CameraFixture f;
    REQUIRE( f.getTemp() == 0 ); // OPERATING
    REQUIRE( f.getFanSpeed() == 0 );

    f.state( stateCodes::READY );
    CameraFixture::value( PARAM_TEMP_SETPOINT, ATTR_AVAIL, 1 );
    CameraFixture::value( PARAM_TEMP_SETPOINT, ATTR_CURRENT, -2000 );
    CameraFixture::value( PARAM_TEMP, ATTR_AVAIL, 1 );
    CameraFixture::value( PARAM_TEMP, ATTR_CURRENT, -2000 );
    REQUIRE( f.getTemp() == 0 );
    REQUIRE( f.m_ccdTemp == Approx( -20 ) );
    REQUIRE( f.m_tempControlStatusStr == "LOCKED" );
    CameraFixture::value( PARAM_TEMP, ATTR_CURRENT, -1500 );
    REQUIRE( f.getTemp() == 0 );
    REQUIRE( f.m_tempControlStatusStr == "UNLOCKED" );

    for( auto [param, attr] : std::vector<std::pair<uns32, int16>>{ { PARAM_TEMP_SETPOINT, ATTR_AVAIL },
                                                                    { PARAM_TEMP_SETPOINT, ATTR_CURRENT },
                                                                    { PARAM_TEMP, ATTR_AVAIL },
                                                                    { PARAM_TEMP, ATTR_CURRENT } } )
    {
        CameraFixture::failGet( param, attr );
        f.power( 1, 0 );
        REQUIRE( f.getTemp() == 0 );
        f.power( 1, 1 );
        REQUIRE( f.getTemp() == -1 );
        REQUIRE( f.state() == stateCodes::ERROR );
        f.state( stateCodes::READY );
        CameraFixture::clear();
    }
    CameraFixture::value( PARAM_TEMP_SETPOINT, ATTR_AVAIL, 0 );
    CameraFixture::value( PARAM_TEMP, ATTR_AVAIL, 0 );
    REQUIRE( f.getTemp() == 0 );

    for( auto [code, name] : std::vector<std::pair<int, std::string>>{ { FAN_SPEED_HIGH, "high" },
                                                                       { FAN_SPEED_MEDIUM, "medium" },
                                                                       { FAN_SPEED_LOW, "low" },
                                                                       { FAN_SPEED_OFF, "off" } } )
    {
        CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_CURRENT, code );
        REQUIRE( f.getFanSpeed() == 0 );
        REQUIRE( f.m_fanSpeedName == name );
    }
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_CURRENT, 99 );
    REQUIRE( f.getFanSpeed() == -1 );
    REQUIRE( Fixture::logged( "unknown PVCAM fan-speed value: 99" ) );
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL, 0 );
    REQUIRE( f.getFanSpeed() == -1 );
    REQUIRE( Fixture::logged( "PARAM_FAN_SPEED_SETPOINT not available while fan control is enabled" ) );
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL, 1 );
    for( int16 attr : { ATTR_AVAIL, ATTR_CURRENT } )
    {
        CameraFixture::failGet( PARAM_FAN_SPEED_SETPOINT, attr );
        f.power( 1, 0 );
        REQUIRE( f.getFanSpeed() == 0 );
        f.power( 1, 1 );
        REQUIRE( f.getFanSpeed() == -1 );
        REQUIRE( f.state() == stateCodes::ERROR );
        f.state( stateCodes::READY );
        CameraFixture::clear();
    }
    f.m_fanSpeedControlEnabled = false;
    REQUIRE( f.getFanSpeed() == 0 );
}

/// The main logic reports shutter, connection, temperature, and fan errors, suppressing them during power loss.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl appLogic handles errors", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::appLogic();
    #endif
    // clang-format on
    CameraFixture f;
    g_fake.m_shutter[1] = -1;
    REQUIRE( f.appLogic() == -1 );
    g_fake.m_shutter[1] = 0;
    CameraFixture::clear();

    // Not connected without power.
    f.state( stateCodes::NOTCONNECTED );
    f.power( 0, 0 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( g_fake.m_calls.empty() );

    // Connection failures are logged only while power is on and targeted on.
    CameraFixture::fail( "pl_pvcam_init" );
    f.power( 1, 0 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( Fixture::errors() == 1 );
    f.power( 1, 1 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( Fixture::errors() == 3 );
    CameraFixture::clear();

    for( uns32 param : { PARAM_TEMP_SETPOINT, PARAM_FAN_SPEED_SETPOINT } )
    {
        CameraFixture::value( PARAM_TEMP_SETPOINT, ATTR_AVAIL, 0 );
        CameraFixture::value( PARAM_TEMP, ATTR_AVAIL, 0 );
        CameraFixture::failGet( param, ATTR_AVAIL );
        f.state( stateCodes::READY );
        f.power( 1, 0 );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( Fixture::errors() == 0 );
        f.state( stateCodes::READY );
        f.power( 1, 1 );
        REQUIRE( f.appLogic() == 0 );
        REQUIRE( Fixture::errors() == 2 );
        CameraFixture::clear();
    }

    // A fan reported unavailable is not logged again by appLogic when power is being turned off.
    CameraFixture::value( PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL, 0 );
    f.state( stateCodes::READY );
    f.power( 1, 0 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( Fixture::errors() == 1 );
    CameraFixture::clear();

    // Power is turned off while a temperature read fails.
    CameraFixture::failGet( PARAM_TEMP_SETPOINT, ATTR_AVAIL );
    g_fake.m_onState = [&]( int s )
    {
        if( s == stateCodes::ERROR )
            f.power( 1, 0 );
    };
    f.state( stateCodes::READY );
    f.power( 1, 1 );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( Fixture::errors() == 1 );
    g_fake.m_onState = nullptr;
}

/// Telemetry wrappers record camera and framegrabber timing state.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl records telemetry", "[pvcamCtrl]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::checkRecordTimes(); pvcamCtrl::recordTelem();
    #endif
    // clang-format on
    CameraFixture f;
    outletHarness::g_faults.m_due = true;
    REQUIRE( f.checkRecordTimes() == 0 );
    REQUIRE( f.recordTelem( static_cast<const MagAOX::logger::telem_fgtimings *>( nullptr ) ) == 0 );
    REQUIRE( outletHarness::g_faults.m_records.size() >= 2 );
}

} // namespace pvcamCtrlTest
} // namespace libXWCTest
