/** \file accelFFCtrl_test.cpp
 * \brief Offline application configuration, INDI handoff, reset and stale-status tests; no stream access.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrl.hpp"

#include <filesystem>
#include <fstream>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_APP_FIXTURE
/// Expose control/owner snapshots without starting threads, FIFOs, stream readers or instrument connections.
struct observerApp : MagAOX::app::accelFFCtrl
{
    using accelFFCtrl::applyRequested;
    using accelFFCtrl::config;
    using accelFFCtrl::m_allowSingleFrameGap;
    using accelFFCtrl::m_configName;
    using accelFFCtrl::m_device;
    using accelFFCtrl::m_indiP_diagnostics;
    using accelFFCtrl::m_indiP_output;
    using accelFFCtrl::m_indiP_parameters;
    using accelFFCtrl::m_indiP_reset;
    using accelFFCtrl::m_indiP_shadow;
    using accelFFCtrl::m_inode;
    using accelFFCtrl::m_inputName;
    using accelFFCtrl::m_inputPath;
    using accelFFCtrl::m_invalidate;
    using accelFFCtrl::m_loopPause;
    using accelFFCtrl::m_requested;
    using accelFFCtrl::m_revision;
    using accelFFCtrl::m_runtime;
    using accelFFCtrl::m_shadow;
    using accelFFCtrl::m_snapshot;
    using accelFFCtrl::setupProperties;

    /// Configure a local device name without starting the framework or creating an INDI driver.
    observerApp();
};

observerApp::observerApp()
{
    m_configName = "accff-offline-test";
}

/// Build a request using the application's own test device identity.
pcf::IndiProperty scalarRequest( const std::string &name /**< [in] numerical property */,
                                 double             value /**< [in] requested target */ )
{
    pcf::IndiProperty request( pcf::IndiProperty::Number );
    request.setDevice( "accff-offline-test" );
    request.setName( name );
    request.add( pcf::IndiElement( "target" ) );
    request["target"] = value;
    return request;
}
/// \endcond

/** \brief The shipped observation config parses without stream access; invalid source and numerical policies fail cold.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl application configuration", "[accelFFCtrl][app]" )
{
    { // Release the framework singleton before constructing the invalid-configuration fixtures.
        observerApp app;
        app.setupConfig();
        std::string example;
        for( const char *path : { "apps/accelFFCtrl/config/example.conf",
                                  "../apps/accelFFCtrl/config/example.conf",
                                  "../config/example.conf" } )
            if( std::filesystem::exists( path ) )
                example = path;
        REQUIRE_FALSE( example.empty() );
        app.config.readConfig( example );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_inputName == "acc000" );
        REQUIRE( app.m_requested.m_predictionNs == 1000000 );
        REQUIRE( app.m_requested.m_zero == std::array<double, 2>{ 2048, 2048 } );
        REQUIRE( app.m_requested.m_voltsPerBit[0] == .0083799 );
        REQUIRE( app.m_requested.m_sensitivity[1] == 1.02 );
        REQUIRE( app.m_loopPause == 200000000 );
        REQUIRE_FALSE( app.m_allowSingleFrameGap );
        REQUIRE_FALSE( app.m_snapshot.m_allowSingleFrameGap );
        REQUIRE_FALSE( app.m_shadow.m_enabled );
        REQUIRE( app.m_shadow.m_gain == 0 );
        const auto shadowExample = std::filesystem::path( example ).parent_path() / "exao2-shadow-example.conf";
        REQUIRE( std::filesystem::exists( shadowExample ) );
        app.config.readConfig( shadowExample.string() );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE_FALSE( app.m_shadow.m_enabled );
        REQUIRE( app.m_shadow.m_gain == 0 );
    }
    for( const std::string text : { "[input]\nshmimName=dm01disp00\n",
                                    "[input]\nmaxBatch=0\n",
                                    "[input]\nmaxAge_us=0\n",
                                    "[input]\nmaxInterval_us=20000\n",
                                    "[input]\nallowSingleFrameGap=truse\n",
                                    "[input]\nallowSingleFrameGap=2\n",
                                    "[parameters]\nfrequency=0\n",
                                    "[parameters]\nprediction_horizon=51\n",
                                    "[parameters]\nadc_sensitivity_x=0\n",
                                    "[parameters]\nadc_zero_x=nan\n",
                                    "[shadow]\nenabled=truse\n",
                                    "[shadow]\nenabled=true\n",
                                    "[shadow]\ngain=-0.1\n",
                                    "[shadow]\ngain=1.1\n",
                                    "[shadow]\nenabled=true\nwfsShmimName=dm01disp00\nlogFile=/tmp/test.csv\n",
                                    "[shadow]\nenabled=true\nwfsShmimName=acc000\nlogFile=/tmp/test.csv\n",
                                    "[shadow]\nenabled=true\nwfsShmimName=fixture-wfs\nlogFile=relative.csv\n" } )
    {
        char      path[] = "/tmp/accelFFCtrl-config-XXXXXX";
        const int fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );
        { // File must be closed before the parser reads it.
            std::ofstream file( path );
            file << text;
        }
        observerApp invalid;
        invalid.setupConfig();
        invalid.config.readConfig( path );
        unlink( path );
        REQUIRE( invalid.loadConfigImpl( invalid.config ) < 0 );
    }
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::setupConfig;
    MagAOX::app::accelFFCtrl::loadConfigImpl;
#endif
    // clang-format on
}

/** \brief Shadow startup configuration registers read-only model diagnostics without opening files or streams.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl shadow configuration without startup", "[accelFFCtrl][app]" )
{
    observerApp app;
    app.setupConfig();
    char      path[] = "/tmp/accelFFCtrl-shadow-config-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    {
        std::ofstream file( path );
        file << "[shadow]\nenabled=true\nwfsShmimName=fixture-wfs\nlogFile=/tmp/fixture-shadow.csv\ngain=0\n";
    }
    app.config.readConfig( path );
    unlink( path );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_shadow.m_enabled );
    REQUIRE( app.m_shadow.m_gain == 0 );
    REQUIRE( app.setupProperties() == 0 );
    REQUIRE( app.m_indiP_output["mode"].get<std::string>() == "shadow file only" );
    REQUIRE( app.m_indiP_output["dm_output"].get<std::string>() == "unavailable" );
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::loadConfigImpl;
    MagAOX::app::accelFFCtrl::setupProperties;
#endif
    // clang-format on
}

/** \brief The new startup opt-in and read-only gap diagnostics require no stream or INDI connection.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl optional single-gap configuration and diagnostics", "[accelFFCtrl][app]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::loadConfigImpl;
    MagAOX::app::accelFFCtrl::setupProperties;
    MagAOX::app::accelFFCtrl::appLogic;
#endif
    // clang-format on
    observerApp app;
    app.setupConfig();
    char      path[] = "/tmp/accelFFCtrl-gap-config-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    {
        std::ofstream file( path );
        file << "[input]\nallowSingleFrameGap=true\n";
    }
    app.config.readConfig( path );
    unlink( path );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_allowSingleFrameGap );
    app.m_runtime.emplace( app.m_requested, 256, 1000000, app.m_allowSingleFrameGap );
    REQUIRE( app.m_runtime->diagnostics().m_allowSingleFrameGap );
    REQUIRE( app.setupProperties() == 0 );
    app.m_snapshot                = app.m_runtime->diagnostics();
    app.m_snapshot.m_admittedGaps = 3;
    app.m_snapshot.m_rejectedGaps = 2;
    app.m_snapshot.m_resets       = 7;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_diagnostics["gap_enabled"].get<double>() == 1 );
    REQUIRE( app.m_indiP_diagnostics["admitted_gaps"].get<double>() == 3 );
    REQUIRE( app.m_indiP_diagnostics["rejected_gaps"].get<double>() == 2 );
    REQUIRE( app.m_indiP_diagnostics["full_resets"].get<double>() == 7 );
    REQUIRE( app.m_indiP_diagnostics["resets"].get<double>() == 7 );
    REQUIRE( app.m_indiP_output["dm_output"].get<std::string>() == "unavailable" );
    REQUIRE( app.m_indiP_shadow["enabled"].get<double>() == 0 );
    REQUIRE( app.m_indiP_shadow["dm_output_enabled"].get<double>() == 0 );
}

/** \brief INDI changes remain pending until the numerical owner applies one coherent revision; invalid edits preserve
 * it.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl INDI observation control handoff and reset", "[accelFFCtrl][app]" )
{
    observerApp app;
    REQUIRE( app.setupProperties() == 0 );
    app.m_runtime.emplace( app.m_requested );
    REQUIRE( app.newCallBack_m_indiP_parameters( scalarRequest( "frequency", 29 ) ) == 0 );
    REQUIRE( app.m_revision == 1 );
    REQUIRE( app.m_indiP_parameters[0]["current"].get<double>() == 28.5 );
    REQUIRE( app.m_runtime->diagnostics().m_config.m_frequency == 28.5 );
    app.applyRequested();
    app.m_snapshot = app.m_runtime->diagnostics();
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_parameters[0]["current"].get<double>() == 29 );
    REQUIRE( app.newCallBack_m_indiP_parameters( scalarRequest( "frequency", NAN ) ) < 0 );
    REQUIRE( app.m_revision == 1 );
    REQUIRE( app.m_requested.m_frequency == 29 );
    REQUIRE( app.m_indiP_parameters[0].getState() == INDI_ALERT );
    REQUIRE( app.newCallBack_m_indiP_parameters( scalarRequest( "prediction_horizon", 2.5 ) ) == 0 );
    REQUIRE( app.newCallBack_m_indiP_parameters( scalarRequest( "rotation", 90 ) ) == 0 );
    REQUIRE( app.newCallBack_m_indiP_parameters( scalarRequest( "adc_zero_x", 2017 ) ) == 0 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_reset["request"].getSwitchState() == pcf::IndiElement::Off );
    app.applyRequested();
    REQUIRE( app.m_runtime->diagnostics().m_config.m_predictionNs == 2500000 );
    REQUIRE( app.m_runtime->diagnostics().m_config.m_rotationDegrees == 90 );
    REQUIRE( app.m_runtime->diagnostics().m_config.m_zero[0] == 2017 );
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.setDevice( "accff-offline-test" );
    request.setName( "estimator_reset" );
    request.add( pcf::IndiElement( "request" ) );
    request["request"] = pcf::IndiElement::On;
    const auto resets  = app.m_runtime->diagnostics().m_resets;
    REQUIRE( app.newCallBack_m_indiP_reset( request ) == 0 );
    REQUIRE( app.m_indiP_reset["request"].getSwitchState() == pcf::IndiElement::On );
    app.applyRequested();
    REQUIRE( app.m_runtime->diagnostics().m_resets == resets + 1 );
    app.m_snapshot = app.m_runtime->diagnostics();
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_reset["request"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_diagnostics["ready"].get<double>() == 0 );
    REQUIRE( app.m_indiP_diagnostics["tip_x_mas"].get<double>() == 0 );
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::setupProperties;
    MagAOX::app::accelFFCtrl::newCallBack_m_indiP_parameters;
    MagAOX::app::accelFFCtrl::newCallBack_m_indiP_reset;
    MagAOX::app::accelFFCtrl::applyRequested;
    MagAOX::app::accelFFCtrl::appLogic;
#endif
    // clang-format on
}

/** \brief A stalled owner's old ready snapshot is cleared by ordinary supervision before INDI publication.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl stale observation telemetry", "[accelFFCtrl][app]" )
{
    observerApp app;
    REQUIRE( app.setupProperties() == 0 );
    app.m_snapshot.m_prediction.m_state        = estimatorState::ready;
    app.m_snapshot.m_prediction.m_sampleTimeNs = 1;
    app.m_snapshot.m_prediction.m_pointing     = { 1, 2 };
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_diagnostics["ready"].get<double>() == 0 );
    REQUIRE( app.m_indiP_diagnostics["tip_x_mas"].get<double>() == 0 );
    REQUIRE( app.m_indiP_diagnostics["stream_state"].get<double>() == double( inputState::stale ) );
    REQUIRE( app.m_indiP_diagnostics.getState() == INDI_ALERT );
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::appLogic;
#endif
    // clang-format on
}

/** \brief Backing-file supervision requests an owner reconnect after disappearance, without opening an image.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl input backing identity supervision", "[accelFFCtrl][app]" )
{
    observerApp app;
    REQUIRE( app.setupProperties() == 0 );
    char      path[] = "/tmp/accelFFCtrl-identity-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    struct stat identity;
    REQUIRE( stat( path, &identity ) == 0 );
    app.m_inputPath = path;
    app.m_inode     = identity.st_ino;
    app.m_device    = identity.st_dev;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE_FALSE( app.m_invalidate.load() );
    REQUIRE( unlink( path ) == 0 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_invalidate.load() );
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrl::appLogic;
#endif
    // clang-format on
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
