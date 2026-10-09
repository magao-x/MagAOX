/** \file pvcamPcie_test.cpp
 * \brief Offline tests of single-port PCIe hotplug against a fake sysfs tree.
 *
 * \ingroup pvcamCtrl_files
 */

#include "pvcamCtrl_harness.hpp"

namespace libXWCTest
{
/** \defgroup pvcamCtrl_unit_test pvcamCtrl Unit Tests
 * \ingroup application_unit_test
 */
namespace pvcamCtrlTest
{
using namespace MagAOX::app;
using namespace pvcamHarness;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// A recording helper on a fake tree with both downstream ports of the Dolphin card; 0000:42:09.0 is this camera's.
struct PcieFixture
{
    /// Private directory.
    outletHarness::Directory m_directory;

    /// Fake PCI tree.
    Sysfs m_sysfs{ m_directory.m_path };

    /// Helper under test.
    pvcamTestPcie m_pcie;

    /// Create both ports, with a healthy camera below the sibling port.
    PcieFixture()
    {
        m_sysfs.port( "0000:42:08.0" );
        m_sysfs.port( "0000:42:09.0" );
        m_sysfs.camera( "0000:42:08.0", "0000:43:00.0" );
        m_pcie.sysfsPath( m_sysfs.m_root );
        REQUIRE( m_pcie.port( "0000:42:09.0" ) == 0 );
        m_pcie.m_resetHoldMs = 1;
        m_pcie.m_settleMs    = 2;
    }

    /// Whether any recorded operation mentions text.
    bool touched( const std::string &text /**< [in] text */ )
    {
        for( auto &op : m_pcie.m_ops )
            if( op.find( text ) != std::string::npos )
                return true;
        return false;
    }
};
/// \endcond

/// PCI addresses, port configuration, and port validation against supported switch cards.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcie validates PCI addresses and ports", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcie::validBdf(); pvcamPcie::port(); pvcamPcie::sysfsPath(); pvcamPcie::enabled(); pvcamPcie::error();
    pvcamPcie::validatePort();
    #endif
    // clang-format on
    for( auto bdf : { "0000:42:09.0", "0000:ab:1f.7" } )
        REQUIRE( pvcamPcie::validBdf( bdf ) );
    for( auto bdf :
         { "", "0000:42:09", "0000:42:09.8", "0000:42:0g.0", "0000-42:09.0", "0000:42:09.00", "000A:42:09.0" } )
        REQUIRE_FALSE( pvcamPcie::validBdf( bdf ) );

    PcieFixture f;
    REQUIRE( f.m_pcie.port() == "0000:42:09.0" );
    REQUIRE( f.m_pcie.enabled() );
    REQUIRE( f.m_pcie.sysfsPath() == f.m_sysfs.m_root );
    REQUIRE( f.m_pcie.validatePort() == 0 );

    REQUIRE( f.m_pcie.port( "42:09.0" ) == -1 );
    REQUIRE( f.m_pcie.error() == "invalid PCI address: '42:09.0'" );
    REQUIRE( f.m_pcie.port() == "0000:42:09.0" );

    REQUIRE( f.m_pcie.port( "0000:42:0a.0" ) == 0 );
    REQUIRE( f.m_pcie.validatePort() == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "PCIe port 0000:42:0a.0 not found: reading " ) );

    f.m_sysfs.port( "0000:42:0a.0", true, true, 0x1234 );
    REQUIRE( f.m_pcie.validatePort() == -1 );
    REQUIRE( f.m_pcie.error() ==
             "PCIe port 0000:42:0a.0 has ID 10b5:1234, which is not a supported PVCAM switch card" );

    attribute( f.m_sysfs.m_root + "/0000:42:0a.0/device", "0x87zz" );
    REQUIRE( f.m_pcie.validatePort() == -1 );
    REQUIRE( f.m_pcie.error().find( "invalid value '0x87zz'" ) != std::string::npos );

    attribute( f.m_sysfs.m_root + "/0000:42:0a.0/device", "zz" );
    REQUIRE( f.m_pcie.validatePort() == -1 );
    REQUIRE( f.m_pcie.error().find( "invalid value 'zz'" ) != std::string::npos );

    REQUIRE( f.m_pcie.port( "" ) == 0 );
    REQUIRE_FALSE( f.m_pcie.enabled() );
}

/// Classify the camera below the port from its config space, ignoring the sibling port's camera.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcie classifies the camera below its port", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcie::cameraDevice(); pvcamPcie::cameraState();
    #endif
    // clang-format on
    PcieFixture f;
    std::string camera;

    // Non-camera functions and non-address entries are ignored, as is the sibling port's camera.
    std::filesystem::create_directories( f.m_sysfs.m_root + "/0000:42:09.0/power" );
    std::filesystem::create_directories( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.1" );
    attribute( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.1/vendor", "0x1b6b" );
    attribute( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.1/device", "0x0002" );
    REQUIRE( f.m_pcie.cameraDevice( camera ) == 1 );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::none );

    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
    REQUIRE( f.m_pcie.cameraDevice( camera ) == 0 );
    REQUIRE( camera == "0000:44:00.0" );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::healthy );

    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::stale );

    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Unresponsive );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::unresponsive );

    // Memory decoding enabled, but BAR0 no longer at the kernel-assigned address.
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
    attribute( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0/resource", "0x00000002fc000000" );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::stale );

    // A 32-bit BAR0 ignores BAR1.
    Config c = readConfig( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0" );
    put( c, 0x10, 0xfc000000, 4 );
    writeConfig( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0", c );
    attribute( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0/resource", "0x00000000fc000000" );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::healthy );

    std::filesystem::remove( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0/resource" );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::error );

    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
    for( auto op : { "read 0", "read 4", "read 16", "read 20" } )
    {
        f.m_pcie.m_failOp = op;
        REQUIRE( f.m_pcie.cameraState() == pcieCamera::error );
    }
    f.m_pcie.m_failOp.clear();

    REQUIRE( f.m_pcie.port( "0000:42:0a.0" ) == 0 );
    REQUIRE( f.m_pcie.cameraDevice( camera ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "listing PCIe port 0000:42:0a.0: " ) );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::error );
}

/// Read Data Link Layer Link Active through the PCI Express capability, and decide whether a port is down.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcie reads link activity and port-down evidence", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcie::linkActive(); pvcamPcie::portDown();
    #endif
    // clang-format on
    PcieFixture f;
    std::string port = f.m_sysfs.m_root + "/0000:42:09.0";

    REQUIRE( f.m_pcie.linkActive() == 1 );
    REQUIRE( f.m_pcie.portDown() == 0 );

    f.m_sysfs.link( "0000:42:09.0", false );
    REQUIRE( f.m_pcie.linkActive() == 0 );
    REQUIRE( f.m_pcie.portDown() == 1 );

    // Without link-activity reporting, the camera's condition decides.
    f.m_sysfs.port( "0000:42:09.0", true, false );
    REQUIRE( f.m_pcie.linkActive() == -1 );
    REQUIRE( f.m_pcie.error() == "PCIe port 0000:42:09.0 does not report link activity" );
    REQUIRE( f.m_pcie.portDown() == 1 );
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Unresponsive );
    REQUIRE( f.m_pcie.portDown() == 1 );
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
    REQUIRE( f.m_pcie.portDown() == 0 );
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0" );
    REQUIRE( f.m_pcie.portDown() == 0 );

    // Capability list missing, or without a PCI Express capability.
    Config c = readConfig( port );
    put( c, 0x06, 0, 2 );
    writeConfig( port, c );
    REQUIRE( f.m_pcie.linkActive() == -1 );
    REQUIRE( f.m_pcie.error() == "PCIe port 0000:42:09.0 has no PCI Express capability" );
    put( c, 0x06, 0x10, 2 );
    put( c, 0x68, 0x0d, 1 );
    writeConfig( port, c );
    REQUIRE( f.m_pcie.linkActive() == -1 );

    f.m_sysfs.port( "0000:42:09.0" );
    for( auto op : { "read 6",
                     "read 52",
                     "read 64",
                     "read 65",
                     "read 72",
                     "read 73",
                     "read 104",
                     "read 105",
                     "read 116",
                     "read 122" } )
    {
        f.m_pcie.m_failOp = op;
        REQUIRE( f.m_pcie.linkActive() == -1 );
    }
    f.m_pcie.m_failOp.clear();

    REQUIRE( f.m_pcie.port( "0000:42:0a.0" ) == 0 );
    REQUIRE( f.m_pcie.portDown() == -1 );
}

/// Hotplug touches only the configured port, in the vendor script's order, and reports the result.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcie hotplug resets, removes, and rescans only its own port", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcie::hotplug(); pvcamPcie::secondaryBusReset();
    #endif
    // clang-format on
    PcieFixture f;
    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
    f.m_pcie.m_rescanCreates = "0000:44:00.0";

    REQUIRE( f.m_pcie.hotplug() == 0 );
    REQUIRE( f.m_pcie.m_ops == std::vector<std::string>{ "config 0000:42:09.0 62 0043",
                                                         "pause 1",
                                                         "config 0000:42:09.0 62 0003",
                                                         "pause 2",
                                                         "write 0000:42:09.0/0000:44:00.0/remove",
                                                         "pause 2",
                                                         "write 0000:42:09.0/rescan",
                                                         "pause 2" } );
    REQUIRE( f.m_pcie.cameraState() == pcieCamera::healthy );
    REQUIRE_FALSE( f.touched( "42:08.0" ) );
    REQUIRE_FALSE( f.touched( "43:00.0" ) );
    REQUIRE( f.m_sysfs.has( "0000:42:08.0", "0000:43:00.0" ) );
    REQUIRE( readConfig( f.m_sysfs.m_root + "/0000:42:09.0" )[0x3E] == 0x03 );

    // An empty port is reset and rescanned without a removal; nothing appears.
    std::filesystem::remove_all( f.m_sysfs.m_root + "/0000:42:09.0/0000:44:00.0" );
    f.m_pcie.m_rescanCreates.clear();
    f.m_pcie.m_ops.clear();
    REQUIRE( f.m_pcie.hotplug() == 1 );
    REQUIRE_FALSE( f.touched( "remove" ) );
    REQUIRE( f.touched( "rescan" ) );

    f.m_sysfs.camera( "0000:42:09.0", "0000:44:00.0", Cam::Stale );
    for( auto op : { "read 62",
                     "config 0000:42:09.0 62 0043",
                     "config 0000:42:09.0 62 0003",
                     "write 0000:42:09.0/0000:44:00.0/remove",
                     "write 0000:42:09.0/rescan" } )
    {
        f.m_pcie.m_failOp = op;
        REQUIRE( f.m_pcie.hotplug() == -1 );
        REQUIRE( f.m_pcie.error() == "injected" );
    }
    f.m_pcie.m_failOp.clear();

    // The port cannot be listed after a successful reset.
    std::filesystem::permissions( f.m_sysfs.m_root + "/0000:42:09.0",
                                  std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec );
    REQUIRE( f.m_pcie.hotplug() == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "listing PCIe port" ) );
    std::filesystem::permissions( f.m_sysfs.m_root + "/0000:42:09.0", std::filesystem::perms::owner_all );
}

/// The production sysfs I/O reports open, read, and write failures.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcie sysfs I/O reports failures", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcie::writeAttribute(); pvcamPcie::readConfig(); pvcamPcie::writeConfig(); pvcamPcie::pause();
    #endif
    // clang-format on
    PcieFixture f;
    std::string dir = f.m_directory.m_path;
    uint8_t     bytes[4]{};

    REQUIRE( f.m_pcie.writeAttribute( dir + "/missing/remove", "1" ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "opening " + dir + "/missing/remove: " ) );
    REQUIRE( f.m_pcie.writeAttribute( "/dev/full", "1" ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "writing /dev/full: " ) );

    REQUIRE( f.m_pcie.readConfig( dir + "/missing", 0, bytes, 2 ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "opening " ) );
    REQUIRE( f.m_pcie.readConfig( f.m_sysfs.m_root + "/0000:42:09.0", 255, bytes, 2 ) == -1 );
    REQUIRE( f.m_pcie.error().ends_with( "short read" ) );
    std::filesystem::create_directories( dir + "/dirconfig/config" );
    REQUIRE( f.m_pcie.readConfig( dir + "/dirconfig", 0, bytes, 2 ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "reading " ) );

    REQUIRE( f.m_pcie.writeConfig( dir + "/missing", 0, bytes, 2 ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "opening " ) );
    std::filesystem::create_directories( dir + "/full" );
    std::filesystem::create_symlink( "/dev/full", dir + "/full/config" );
    REQUIRE( f.m_pcie.writeConfig( dir + "/full", 0, bytes, 2 ) == -1 );
    REQUIRE( f.m_pcie.error().starts_with( "writing " ) );

    f.m_pcie.realPause( 0 );
}

/// The shared lock excludes a second holder, reports open failures, and releases on destruction.
/** \ingroup pvcamCtrl_unit_test */
TEST_CASE( "pvcamPcieLock serializes instances", "[pvcamCtrl][pcie]" )
{
    // clang-format off
    #ifdef PVCAMCTRL_TEST_DOXYGEN_REF
    pvcamPcieLock::pvcamPcieLock(); pvcamPcieLock::~pvcamPcieLock(); pvcamPcieLock::locked();
    pvcamPcieLock::busy(); pvcamPcieLock::error();
    #endif
    // clang-format on
    outletHarness::Directory dir;
    std::string              path = dir.m_path + "/pvcamCtrl_pcie.lock";
    {
        pvcamPcieLock first( path );
        REQUIRE( first.locked() );
        REQUIRE( first.error() == 0 );

        pvcamPcieLock second( path );
        REQUIRE_FALSE( second.locked() );
        REQUIRE( second.busy() );
    }

    pvcamPcieLock again( path );
    REQUIRE( again.locked() );

    pvcamPcieLock missing( dir.m_path + "/none/lock" );
    REQUIRE_FALSE( missing.locked() );
    REQUIRE_FALSE( missing.busy() );
    REQUIRE( missing.error() == ENOENT );
}

} // namespace pvcamCtrlTest
} // namespace libXWCTest
