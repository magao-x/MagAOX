/** \file cred2Ctrl_lifecycle_test.cpp
 * \brief Isolated Catch2 lifecycle tests for the cred2Ctrl app.
 * \author OpenAI Codex
 *
 * \ingroup cred2Ctrl_files
 */

#define CRED2CTRL_TEST_SUPPORT_ONLY
#include "cred2Ctrl_test.cpp"
#undef CRED2CTRL_TEST_SUPPORT_ONLY

namespace libXWCTest
{

/** \addtogroup cred2Ctrl_unit_test
 * \brief Additional lifecycle tests for the cred2Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `cred2Ctrl` lifecycle unit tests.
/** \ingroup cred2Ctrl_unit_test
 */
namespace cred2CtrlTest
{

/// Verify lifecycle entrypoints cover startup failure handling and successful startup cleanup.
/**
 * \ingroup cred2Ctrl_unit_test
 */
TEST_CASE( "cred2Ctrl lifecycle entrypoints handle startup failures and success", "[cred2Ctrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef CRED2CTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( cred2Ctrl::appStartup() );
    #endif
    // clang-format on

    SECTION( "appStartup reports failure when no startup mode is available" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "startup_missing_mode" );
        app.m_startupMode.clear();

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup reports failure when the EDT runtime config cannot be read" )
    {
        cred2Ctrl_test app;

        loadDefaultConfig( app, "startup_bad_edt" );
        g_edtStubState.readcfgReturn = -1;

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup succeeds once the runtime mode and telemetry path are configured" )
    {
        cred2Ctrl_test app;
        startupScope   startup( app );

        loadDefaultConfig( app, "startup_success" );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );

        REQUIRE( startupRv == 0 );
        REQUIRE( app.m_pdv != nullptr );
        REQUIRE( app.m_raw_width == 640 );
        REQUIRE( app.m_raw_height == 512 );
        REQUIRE( app.m_raw_depth == 16 );
        REQUIRE( app.m_cameraType == "stub_pdv" );
        REQUIRE( g_edtStubState.readcfgCalls > 0 );
        REQUIRE( g_edtStubState.multibufCalls > 0 );
    }

    SECTION( "appStartup reports failure when the read-only fps property has already been registered" )
    {
        cred2Ctrl_test    app;
        pcf::IndiProperty fpsLimits;

        loadDefaultConfig( app, "startup_duplicate_property" );
        REQUIRE( app.createROIndiNumber( fpsLimits, "fps_limits" ) == 0 );
        REQUIRE( app.registerIndiPropertyReadOnly( fpsLimits ) == 0 );

        REQUIRE( app.appStartup() < 0 );
    }
}

} // namespace cred2CtrlTest
} // namespace libXWCTest
