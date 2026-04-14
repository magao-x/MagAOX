/** \file ocam2KCtrl_lifecycle_test.cpp
 * \brief Isolated Catch2 lifecycle tests for the ocam2KCtrl app.
 * \author OpenAI Codex
 *
 * \ingroup ocam2KCtrl_files
 */

#define OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "ocam2KCtrl_test.cpp"
#undef OCAM2KCTRL_TEST_SUPPORT_ONLY

namespace libXWCTest
{

/** \addtogroup ocam2KCtrl_unit_test
 * \brief Additional lifecycle tests for the ocam2KCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `ocam2KCtrl` lifecycle unit tests.
/** \ingroup ocam2KCtrl_unit_test
 */
namespace ocam2KCtrlTest
{

/// Verify lifecycle entrypoints cover startup failure handling and the POWERON fast-return path.
/**
 * \ingroup ocam2KCtrl_unit_test
 */
TEST_CASE( "ocam2KCtrl lifecycle entrypoints handle startup failures and POWERON logic", "[ocam2KCtrl]" )
{
    resetStubState();

    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::appStartup() );
    XWCTEST_DOXYGEN_REF( ocam2KCtrl::appLogic() );
    #endif
    // clang-format on

    SECTION( "appStartup reports failure when the startup mode cannot be configured" )
    {
        ocam2KCtrl_test app;

        app.m_startupMode = "";

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup reports failure when EDT startup cannot load the configured mode" )
    {
        ocam2KCtrl_test app;

        configureStartupMode( app );
        g_edtStubState.readcfgReturn = -1;

        REQUIRE( app.appStartup() < 0 );
    }

    SECTION( "appStartup succeeds once a startup mode and telemetry path are configured" )
    {
        ocam2KCtrl_test app;
        startupScope    startup( app );

        configureStartupMode( app );

        const int startupRv = app.appStartup();
        startup.markStarted( startupRv == 0 );

        REQUIRE( startupRv == 0 );
        REQUIRE( app.m_temps.CCD == Approx( -999.0f ) );
    }

    SECTION( "appLogic returns immediately while the app is still in POWERON and the wait has not elapsed" )
    {
        ocam2KCtrl_test app;

        static_cast<MagAOXAppT &>( app ).m_powerState = 1;
        app.m_powerTargetState                        = 1;
        app.state( stateCodes::POWERON );
        app.m_powerOnCounter                         = 0;
        static_cast<MagAOXAppT &>( app ).m_loopPause = 0;
        fgThreadScope fgThread( app );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::POWERON );
    }

    SECTION( "appLogic returns an error if the framegrabber thread has already exited" )
    {
        auto *app = new ocam2KCtrl_test;

        static_cast<MagAOXAppT &>( *app ).m_powerState = 1;
        app->m_powerTargetState                        = 1;
        app->state( stateCodes::POWERON );
        startFgThread( *app, 1 );
        std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );

        REQUIRE( app->appLogic() == -1 );

        // `frameGrabber::appLogic()` uses `pthread_tryjoin_np`, which consumes the native thread
        // without clearing the owning `std::thread`. Leak this tiny test instance so its destructor
        // does not terminate the process while exercising the error-return branch.
    }
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
