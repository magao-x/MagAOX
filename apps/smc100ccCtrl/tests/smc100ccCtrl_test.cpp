/** \file smc100ccCtrl_test.cpp
 * \brief Offline configuration, preset-assignment, and callback tests for smc100ccCtrl.
 *
 * \ingroup smc100ccCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../../../libMagAOX/libMagAOX.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace MagAOX
{
namespace tty
{
/// Replace configuration-time USB discovery without inspecting or opening hardware.
class smc100ccTestUSBDevice : public usbDevice
{
  public:
    /// Report an absent USB device while testing the controller's real configuration loader.
    int loadConfig( mx::app::appConfigurator &config /**< [in] Unused USB settings. */ );
};

int smc100ccTestUSBDevice::loadConfig( mx::app::appConfigurator & )
{
    return TTY_E_DEVNOTFOUND;
}
} // namespace tty
} // namespace MagAOX
/// \endcond

#define usbDevice smc100ccTestUSBDevice
#include "../smc100ccCtrl.hpp"
#undef usbDevice

using namespace MagAOX::app;

namespace libXWCTest
{

/// Namespace for `smc100ccCtrl` unit tests.
/** \ingroup smc100ccCtrl_unit_test
 */
namespace smc100ccCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Exercise preset assignment and configuration without starting the controller.
class smc100ccCtrl_test : public smc100ccCtrl
{
  public:
    /// Initialize callback-validation properties for the test device.
    smc100ccCtrl_test( const std::string device /**< [in] Test device name. */ );

    /// Configured matching threshold exposed for configuration assertions.
    using smc100ccCtrl::m_presetTolerance;

    /// Observed position supplied directly without polling the stage.
    using smc100ccCtrl::m_position;

    /// Preset coordinates, including the existing index-zero sentinel.
    using smc100ccCtrl::m_presetPositions;

    /// Read a temporary INI file through the real controller configuration methods.
    void configure( const std::string &settings /**< [in] INI configuration text. */ );
};

smc100ccCtrl_test::smc100ccCtrl_test( const std::string device )
{
    m_configName = device;

    XWCTEST_SETUP_INDI_NEW_PROP( position );

    XWCTEST_SETUP_INDI_NEW_PROP( preset );
    XWCTEST_SETUP_INDI_NEW_PROP( presetName );
    XWCTEST_SETUP_INDI_NEW_PROP( home );
    XWCTEST_SETUP_INDI_NEW_PROP( stop );
}

void smc100ccCtrl_test::configure( const std::string &settings )
{
    setupConfig();
    char path[]     = "/tmp/smc100ccCtrl-test-XXXXXX";
    int  descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    close( descriptor );
    std::ofstream( path ) << settings;
    int result = config.readConfig( path );
    std::remove( path );
    REQUIRE( result == 0 );
    loadConfig();
}
/// \endcond

/// Verify the smc100ccCtrl INDI callback validators accept only the expected properties.
/**
 * \ingroup smc100ccCtrl_unit_test
 */
TEST_CASE( "smc100ccCtrl INDI callbacks validate device and property names", "[smc100ccCtrl]" )
{
    // clang-format off
    #ifdef SMC100CCCTRL_TEST_DOXYGEN_REF
    smc100ccCtrl::newCallBack_m_indiP_position( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_preset( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_presetName( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_home( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_stop( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, position );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, preset );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, presetName );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, home );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, stop );
}

/** \brief Omitted tolerance retains the original strict matching boundary and first-preset ordering.
 * \ingroup smc100ccCtrl_unit_test
 */
TEST_CASE( "smc100ccCtrl default preset tolerance preserves assignment", "[smc100ccCtrl]" )
{
    // clang-format off
#ifdef SMC100CCCTRL_TEST_DOXYGEN_REF
    smc100ccCtrl::setupConfig; smc100ccCtrl::loadConfig; smc100ccCtrl::presetNumber;
#endif
    // clang-format on

    smc100ccCtrl_test app( "preset-test" );
    REQUIRE( app.m_presetTolerance == 1e-3 );
    app.configure( "[stage]\n" );
    REQUIRE( app.m_presetTolerance == 1e-3 );
    app.m_presetPositions = { -1, 0, 0.0005, 1 };

    for( double position : { -0.0009, 0.0, 0.0005, 0.0009 } )
    {
        app.m_position = position;
        CHECK( app.presetNumber() == 1 );
    }

    app.m_presetPositions = { -1, 0, 1 };
    for( double position : { -1.0, -0.001, 0.001, 0.5 } )
    {
        app.m_position = position;
        CHECK( app.presetNumber() == 0 );
    }
    app.m_position = 1.0;
    CHECK( app.presetNumber() == 2 );
}

/** \brief Configured wider and tighter tolerances change matching while retaining the strict boundary.
 * \ingroup smc100ccCtrl_unit_test
 */
TEST_CASE( "smc100ccCtrl loads configurable preset tolerance", "[smc100ccCtrl]" )
{
    // clang-format off
#ifdef SMC100CCCTRL_TEST_DOXYGEN_REF
    smc100ccCtrl::setupConfig; smc100ccCtrl::loadConfig; smc100ccCtrl::presetNumber;
#endif
    // clang-format on

    for( const auto &setting : { std::make_pair( "0.01", 0.01 ), std::make_pair( "0.0001", 0.0001 ) } )
    {
        CAPTURE( setting.first );
        smc100ccCtrl_test app( "preset-test" );
        app.configure( "[stage]\npresetTolerance=" + std::string( setting.first ) + "\n" );
        REQUIRE( app.m_presetTolerance == setting.second );
        app.m_presetPositions = { -1, 0 };
        for( double direction : { -1.0, 1.0 } )
        {
            app.m_position = direction * setting.second * 0.5;
            CHECK( app.presetNumber() == 1 );
            app.m_position = direction * setting.second;
            CHECK( app.presetNumber() == 0 );
            app.m_position = direction * setting.second * 2;
            CHECK( app.presetNumber() == 0 );
            app.m_position = direction * 0.0005;
            CHECK( app.presetNumber() == ( setting.second > 1e-3 ? 1 : 0 ) );
        }
    }
}

} // namespace smc100ccCtrlTest

} // namespace libXWCTest
