/** \file pvcamCtrl_hotplug_test.cpp
 * \brief Offline power-off/power-on and PCIe hotplug tests for pvcamCtrl, with and without cameras.
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
/// This app's camera on the Dolphin card's 0000:42:09.0 port, with the other camera healthy on 0000:42:08.0.
struct HotplugFixture : Fixture
{
    /// Create both ports and enable hotplug on this camera's port.
    HotplugFixture()
    {
        m_sysfs.port( "0000:42:08.0" );
        m_sysfs.camera( "0000:42:08.0", "0000:43:00.0" );
        m_sysfs.port( "0000:42:09.0" );
        pcie();
    }

    /// Set this camera's sysfs state as seen while its power is off.
    void cameraOff()
    {
        m_sysfs.link( "0000:42:09.0", false );
        if( m_sysfs.has( "0000:42:09.0", "0000:44:00.0" ) )
            m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Unresponsive );
    }

    /// Set this camera's sysfs state as seen after power returns, before any rescan.
    void cameraOn()
    {
        m_sysfs.link( "0000:42:09.0", true );
        if( m_sysfs.has( "0000:42:09.0", "0000:44:00.0" ) )
            m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
    }

    /// Whether any PCIe operation names the other camera's port or device.
    bool sibling()
    {
        for( auto &op : m_pcie.m_ops )
            if( op.find( "42:08.0" ) != std::string::npos || op.find( "43:00.0" ) != std::string::npos )
                return true;
        return !m_sysfs.has( "0000:42:08.0", "0000:43:00.0" ) ||
               readConfig( m_sysfs.m_root + "/0000:42:08.0" )[0x3E] != 0x03;
    }

    /// Whether any PCIe operation contains text.
    bool op( const std::string &text /**< [in] text */ )
    {
        for( auto &o : m_pcie.m_ops )
            if( o.find( text ) != std::string::npos )
                return true;
        return false;
    }
};
/// \endcond

/// Starting with the camera unpowered, then powering on, re-enumerates only this camera and connects.
/** Also verifies the stdCamera power-on block runs (power-on defaults), and that PVCAM is untouched while off.
 * \ingroup pvcamCtrl_unit_test
 */
TEST_CASE( "pvcamCtrl hotplugs its camera at power-on after starting unpowered", "[pvcamCtrl][hotplug]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::appLogic(); pvcamCtrl::onPowerOff(); pvcamCtrl::whilePowerOff(); pvcamCtrl::pcieLogic();
    pvcamCtrl::hotplugCamera(); pvcamCtrl::checkPortDown(); pvcamCtrl::powerOnDefaults();
    #endif
    // clang-format on
    HotplugFixture f;
    f.cameraOff();
    f.start( 0 );
    for( int n = 0; n < 3; ++n )
        REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::POWEROFF );
    REQUIRE( f.m_portSeenDown );
    REQUIRE( f.m_pcie.m_ops.empty() );
    REQUIRE( g_fake.m_calls.empty() );
    REQUIRE( Fixture::errors() == 0 );

    f.m_expTime              = 5;
    f.m_pcie.m_rescanCreates = "0000:44:00.0";
    f.cameraOn();
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.m_expTime == Approx( 0.01 ) );
    REQUIRE( f.m_pcie.m_ops == std::vector<std::string>{ "config 0000:42:09.0 62 0043",
                                                         "pause 1",
                                                         "config 0000:42:09.0 62 0003",
                                                         "pause 2",
                                                         "pause 2",
                                                         "write 0000:42:09.0/rescan",
                                                         "pause 2" } );
    REQUIRE_FALSE( f.m_hotplugPending );
    REQUIRE( f.m_handle == 100 );
    REQUIRE( Fixture::logged( "camera found on PCIe port 0000:42:09.0" ) );
    REQUIRE_FALSE( f.sibling() );
    REQUIRE( Fixture::errors() == 0 );
    REQUIRE( pvcamPcieLock( f.m_pcieLockPath ).locked() );
}

/// A power cycle closes the camera and uninitializes PVCAM before removing and rescanning the stale device.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl re-enumerates its stale camera after a power cycle", "[pvcamCtrl][hotplug]" )
{
    HotplugFixture f;
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
    f.start( 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.m_pcie.m_ops.empty() );

    f.power( 0, 0 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_hotplugPending );
    REQUIRE_FALSE( f.m_portSeenDown ); // The camera has not dropped yet.
    f.cameraOff();
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_portSeenDown );
    REQUIRE( f.m_handle == 100 ); // Not closed while the framegrabber thread may still use it.

    f.cameraOn();
    f.m_pcie.m_rescanCreates = "0000:44:00.0";
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.op( "write 0000:42:09.0/0000:44:00.0/remove" ) );
    REQUIRE_FALSE( f.m_pcie.m_openAtRemove );
    REQUIRE_FALSE( f.sibling() );
    REQUIRE( Fixture::errors() == 0 );
}

/// After a power-on hotplug finds no camera, NODEVICE is reported once and retries are rate limited.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl waits quietly in NODEVICE when no camera appears", "[pvcamCtrl][hotplug]" )
{
    HotplugFixture f;
    f.cameraOff();
    f.start( 0 );
    REQUIRE( f.loop() == 0 );
    f.cameraOn();
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::NODEVICE );
    REQUIRE( f.m_hotplugPending );
    REQUIRE( Fixture::count( "no camera found on PCIe port 0000:42:09.0 after hotplug" ) == 1 );
    REQUIRE( g_fake.m_calls["pl_pvcam_init"] == 0 );

    size_t ops = f.m_pcie.m_ops.size();
    for( int n = 0; n < 3; ++n )
        REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_pcie.m_ops.size() == ops );

    f.m_lastHotplug -= 31;
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_pcie.m_ops.size() == 2 * ops );
    REQUIRE( Fixture::count( "no camera found" ) == 1 );

    f.m_lastHotplug -= 31;
    f.m_pcie.m_rescanCreates = "0000:44:00.0";
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.m_pcieLastLog.empty() );

    // Losing power while waiting stops hotplug attempts without errors.
    f.power( 0, 0 );
    f.cameraOff();
    ops = f.m_pcie.m_ops.size();
    for( int n = 0; n < 3; ++n )
        REQUIRE( f.loop() == 0 );
    REQUIRE( f.m_pcie.m_ops.size() == ops );
    REQUIRE( Fixture::errors() == 0 );
}

/// A port that stays up, or comes back up, while this camera is off belongs to another camera and is never reset.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl refuses to hotplug a port that is not its camera's", "[pvcamCtrl][hotplug]" )
{
    SECTION( "port stays up while power is off" )
    {
        HotplugFixture f;
        f.pcie( "0000:42:08.0" );
        f.start( 0 );
        REQUIRE( f.loop() == 0 );
        REQUIRE_FALSE( f.m_portSeenDown );
        f.power( 1, 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_pcieDisabled );
        REQUIRE( Fixture::count( "stayed up while camera power was off" ) == 1 );
        REQUIRE( f.state() == stateCodes::OPERATING );
        REQUIRE( f.m_pcie.m_ops.empty() );
        REQUIRE_FALSE( f.sibling() );
    }

    SECTION( "port comes up while power is off" )
    {
        HotplugFixture f;
        f.cameraOff();
        f.start( 0 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_portSeenDown );
        f.m_sysfs.link( "0000:42:09.0", true );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_pcieDisabled );
        REQUIRE( Fixture::count( "came up while camera power was off" ) == 1 );
        REQUIRE( f.loop() == 0 );
        f.power( 1, 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_pcie.m_ops.empty() );
    }
}

/// Startup with power on hotplugs a stale camera before PVCAM opens it, and leaves a healthy one alone.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl checks the camera's PCIe state when starting with power on", "[pvcamCtrl][hotplug]" )
{
    SECTION( "stale camera" )
    {
        HotplugFixture f;
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
        f.m_pcie.m_rescanCreates = "0000:44:00.0";
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.op( "remove" ) );
        REQUIRE( f.m_pcie.m_initsAtRemove == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );
    }

    SECTION( "healthy camera" )
    {
        HotplugFixture f;
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.m_pcie.m_ops.empty() );
        REQUIRE( f.state() == stateCodes::OPERATING );
    }

    SECTION( "healthy camera whose serial is not found is not reset" )
    {
        HotplugFixture f;
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
        g_fake.m_cameras[0].m_serial = "A22J723004";
        f.start( 1 );
        for( int n = 0; n < 3; ++n )
            REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE( f.m_pcie.m_ops.empty() );
    }

    SECTION( "no camera, as after booting with the camera unpowered" )
    {
        HotplugFixture f;
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE( Fixture::count( "If the host booted with the camera unpowered, reboot with camera power on." ) == 1 );
        f.m_lastHotplug -= 31;
        REQUIRE( f.loop() == 0 );
        REQUIRE( Fixture::count( "no camera found" ) == 1 );
    }
}

/// The shared lock defers hotplug and enumeration while another instance holds it; lock errors are logged once.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl serializes hotplug and enumeration with other instances", "[pvcamCtrl][hotplug]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamCtrl::connect(); pvcamCtrl::pcieLog();
    #endif
    // clang-format on
    SECTION( "hotplug waits for the lock" )
    {
        HotplugFixture f;
        f.cameraOff();
        f.start( 0 );
        REQUIRE( f.loop() == 0 );
        f.cameraOn();
        f.power( 1, 1 );
        f.m_pcie.m_rescanCreates = "0000:44:00.0";
        {
            pvcamPcieLock other( f.m_pcieLockPath );
            REQUIRE( f.loop() == 0 );
            REQUIRE( f.state() == stateCodes::NOTCONNECTED );
            REQUIRE( f.m_pcie.m_ops.empty() );
            REQUIRE( g_fake.m_calls["pl_pvcam_init"] == 0 );
        }
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );
    }

    SECTION( "enumeration waits for the lock" )
    {
        HotplugFixture f;
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
        f.start( 1 );
        {
            pvcamPcieLock other( f.m_pcieLockPath );
            REQUIRE( f.loop() == 0 );
            REQUIRE( f.state() == stateCodes::NOTCONNECTED );
            REQUIRE( g_fake.m_calls["pl_pvcam_init"] == 0 );
        }
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );
    }

    SECTION( "lock file cannot be opened" )
    {
        HotplugFixture f;
        f.start( 1 );
        f.m_pcieLockPath = f.m_directory.m_path + "/missing/lock";
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE( f.m_pcie.m_ops.empty() );
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );
        REQUIRE( Fixture::count( "locking " + f.m_pcieLockPath ) == 1 );
    }
}

/// sysfs failures during checks and hotplug are logged once, and the app keeps running.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl reports PCIe failures without stopping", "[pvcamCtrl][hotplug]" )
{
    SECTION( "hotplug failure" )
    {
        HotplugFixture f;
        f.start( 1 );
        f.m_pcie.m_failOp = "write 0000:42:09.0/rescan";
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::NODEVICE );
        REQUIRE( Fixture::count( "PCIe hotplug failed: injected" ) == 1 );
    }

    SECTION( "camera check failure falls back to connecting" )
    {
        HotplugFixture f;
        f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
        f.m_pcie.m_failOp = "read 0";
        f.start( 1 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( Fixture::count( "checking PCIe camera: injected" ) == 1 );
        REQUIRE( f.state() == stateCodes::OPERATING );
    }

    SECTION( "port check failure while off" )
    {
        HotplugFixture f;
        f.start( 0 );
        std::filesystem::remove_all( f.m_sysfs.m_root + "/0000:42:09.0" );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.loop() == 0 );
        REQUIRE( Fixture::count( "checking PCIe port: " ) == 1 );
    }

    SECTION( "port fails validation at startup" )
    {
        HotplugFixture f;
        f.pcie( "0000:42:0a.0" );
        f.start( 1 );
        REQUIRE( f.m_pcieDisabled );
        REQUIRE( Fixture::logged( "PCIe port 0000:42:0a.0 not found" ) );
        REQUIRE( f.loop() == 0 );
        REQUIRE( f.state() == stateCodes::OPERATING );
        REQUIRE( f.m_pcie.m_ops.empty() );
    }
}

/// Without a configured port the app connects exactly as before, never touching sysfs.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl without a PCIe port behaves as before", "[pvcamCtrl][hotplug]" )
{
    HotplugFixture f;
    REQUIRE( f.m_pcie.port( "" ) == 0 );
    f.start( 0 );
    REQUIRE( f.loop() == 0 );
    REQUIRE_FALSE( f.m_portSeenDown );
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( f.m_pcie.m_ops.empty() );
    REQUIRE_FALSE( Fixture::logged( "PCIe" ) );
}

/// The power-on wait, and dev helper failures at power-off, are handled by the main-loop hooks.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamCtrl power hooks wait for power-on and report dev helper errors", "[pvcamCtrl][hotplug]" )
{
    HotplugFixture f;
    f.m_powerOnWait = 1;
    f.start( 0 );
    f.power( 1, 1 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( f.state() == stateCodes::POWERON );

    g_fake.m_shutter[3] = -1;
    g_fake.m_shutter[4] = -1;
    f.power( 0, 0 );
    REQUIRE( f.loop() == 0 );
    REQUIRE( Fixture::count( "error from a dev onPowerOff()" ) == 1 );
    REQUIRE( Fixture::count( "error from a dev whilePowerOff()" ) == 1 );
}

} // namespace pvcamCtrlTest
} // namespace libXWCTest
