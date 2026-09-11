/** \file ttyUSB_test.cpp
  * \brief Catch2 tests for ttyUSBDevName() and ttyUSBDevNames() in libMagAOX/tty/ttyUSB.cpp.
  *
  * The matching logic in ttyUSB.cpp walks real udev sysfs entries under
  * /sys/class/tty/ttyUSB*. Those entries only exist when USB serial hardware or a kernel
  * USB serial gadget is attached. Neither is available in this build and test environment.
  *
  * libudev is not mocked. Instead ttyUSB.cpp exposes two test-only macros,
  * XWCTEST_TTYUSB_SYSFS_DIR and XWCTEST_TTYUSB_SYSFS_PREFIX. They default to the production
  * values. This file includes ttyUSB.cpp twice more under different namespaces with those
  * macros redefined. The same real getFileNames() and libudev calls then run against real
  * operating system state. A scratch directory covers the branch where a found entry is
  * not a real udev syspath. The always-present non-USB entry /sys/class/tty/tty0 covers the
  * branch where a device is found but has no USB parent.
  *
  * The vendor, product, and serial match and the success path still need real USB serial
  * hardware. They remain untested here.
  */
#include "../../../tests/catch2/catch.hpp"

#include "../ttyUSB.hpp"
#include "../ttyErrors.hpp"

#include <filesystem>
#include <fstream>

namespace libXWCTest
{
namespace ttyUSBTest
{
// Path of the scratch directory used by the XWCTEST_TTYUSB_NOSYSPATH_ns tests. The tests set
// it before they call into that namespace. It is declared here, ahead of the re-include
// below, so its name is visible where the macro substitutes it into the getFileNames()
// call in ttyUSB.cpp.
std::string xwcTestTtyUsbScratchDir;
} // namespace ttyUSBTest
} // namespace libXWCTest

// First re-include. The sysfs directory is redirected to the scratch directory. The default
// ttyUSB prefix is kept, so the plain file ttyUSB0 in that directory is found.
#define XWCTEST_NAMESPACE XWCTEST_TTYUSB_NOSYSPATH_ns
#define XWCTEST_TTYUSB_SYSFS_DIR libXWCTest::ttyUSBTest::xwcTestTtyUsbScratchDir
#include "../ttyUSB.cpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TTYUSB_SYSFS_DIR
#undef XWCTEST_TTYUSB_SYSFS_PREFIX

// Second re-include. The real /sys/class/tty directory is kept. The prefix is changed to
// tty0, which is a real console tty with no USB parent.
#define XWCTEST_NAMESPACE XWCTEST_TTYUSB_NOUSBPARENT_ns
#define XWCTEST_TTYUSB_SYSFS_PREFIX "tty0"
#include "../ttyUSB.cpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_TTYUSB_SYSFS_PREFIX

namespace libXWCTest
{
namespace ttyUSBTest
{

// The production function with the production sysfs path. No ttyUSB entries exist in this
// environment, so it must report that there are no device names and clear the output.
TEST_CASE( "ttyUSBDevName returns TTY_E_NODEVNAMES when no ttyUSB devices exist", "[libMagAOX::tty::ttyUSBDevName]" )
{
    std::string devName = "unset";
    int rv = MagAOX::tty::ttyUSBDevName( devName, "0403", "6001", "" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_NODEVNAMES );
    REQUIRE( devName == "" );
}

// Same as above for the vector returning overload. Stale entries in the output vector must
// be cleared.
TEST_CASE( "ttyUSBDevNames returns TTY_E_NODEVNAMES when no ttyUSB devices exist", "[libMagAOX::tty::ttyUSBDevNames]" )
{
    std::vector<std::string> devNames = { "stale", "entries" };
    int rv = MagAOX::tty::ttyUSBDevNames( devNames, "0403", "6001" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_NODEVNAMES );
    REQUIRE( devNames.size() == 0 );
}

/// A real scratch directory holding one plain file named like a ttyUSB syspath entry.
/** getFileNames() finds the file by reading the directory. The file is not a real kernel
  * sysfs path, so udev_device_new_from_syspath() fails on it. The destructor removes the
  * directory.
  */
struct ScratchSysfsDir
{
    std::filesystem::path dir;

    ScratchSysfsDir()
    {
        std::string dirTemplate = ( std::filesystem::temp_directory_path() / "ttyUSB_test_sysfs_XXXXXX" ).string();
        REQUIRE( mkdtemp( dirTemplate.data() ) != nullptr );
        dir = dirTemplate;
        std::ofstream( dir / "ttyUSB0" ) << "not a real sysfs device\n";
    }

    ~ScratchSysfsDir()
    {
        std::filesystem::remove_all( dir );
    }
};

// An entry is found in the scratch directory but udev rejects it. The function must report
// device not found and clear the output.
TEST_CASE( "ttyUSBDevName reports device-not-found when a syspath entry isn't a real udev syspath",
           "[libMagAOX::tty::ttyUSBDevName]" )
{
    ScratchSysfsDir scratch;
    xwcTestTtyUsbScratchDir = scratch.dir.string() + "/";

    std::string devName = "unset";
    int rv = MagAOX::tty::XWCTEST_TTYUSB_NOSYSPATH_ns::ttyUSBDevName( devName, "0403", "6001", "" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_DEVNOTFOUND );
    REQUIRE( devName == "" );
}

// Same as above for the vector returning overload.
TEST_CASE( "ttyUSBDevNames reports device-not-found when a syspath entry isn't a real udev syspath",
           "[libMagAOX::tty::ttyUSBDevNames]" )
{
    ScratchSysfsDir scratch;
    xwcTestTtyUsbScratchDir = scratch.dir.string() + "/";

    std::vector<std::string> devNames = { "stale" };
    int rv = MagAOX::tty::XWCTEST_TTYUSB_NOSYSPATH_ns::ttyUSBDevNames( devNames, "0403", "6001" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_DEVNOTFOUND );
    REQUIRE( devNames.size() == 0 );
}

// udev accepts the real tty0 entry, but it has no USB parent device. The function must
// report device not found and clear the output.
TEST_CASE( "ttyUSBDevName reports device-not-found for a real tty device with no usb parent",
           "[libMagAOX::tty::ttyUSBDevName]" )
{
    std::string devName = "unset";
    int rv = MagAOX::tty::XWCTEST_TTYUSB_NOUSBPARENT_ns::ttyUSBDevName( devName, "0403", "6001", "" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_DEVNOTFOUND );
    REQUIRE( devName == "" );
}

// Same as above for the vector returning overload.
TEST_CASE( "ttyUSBDevNames reports device-not-found for a real tty device with no usb parent",
           "[libMagAOX::tty::ttyUSBDevNames]" )
{
    std::vector<std::string> devNames = { "stale" };
    int rv = MagAOX::tty::XWCTEST_TTYUSB_NOUSBPARENT_ns::ttyUSBDevNames( devNames, "0403", "6001" ); // FTDI FT232 vendor and product ids

    REQUIRE( rv == TTY_E_DEVNOTFOUND );
    REQUIRE( devNames.size() == 0 );
}

} // namespace ttyUSBTest
} // namespace libXWCTest
