/** \file ocam2KCtrl_reconfig_test.cpp
 * \brief Offline tests distinguishing OCAM configuration application from new reconfiguration requests.
 * \ingroup ocam2KCtrl_reconfig_unit_test
 */

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
#define OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "ocam2KCtrl_test.cpp"
#undef OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "../../../tests/outletAppTest.hpp"
/// \endcond

#include "../../../tests/testXWC.hpp"
#include <future>

namespace libXWCTest
{
/** \addtogroup ocam2KCtrl_reconfig_unit_test */
namespace ocam2KCtrlTest
{

/// Configuring FPS and synchronization applies settings without requesting another configuration pass.
/** \ingroup ocam2KCtrl_reconfig_unit_test */
TEST_CASE( "ocam2KCtrl configuration does not request itself again", "[ocam2KCtrl][reconfig]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::configureAcquisition();
    MagAOX::app::ocam2KCtrl::setFPS(); MagAOX::app::ocam2KCtrl::setSynchro();
    #endif
    // clang-format on
    for( float fps : { 0.0f, 50.0f } )
    {
        for( bool wasSynchro : { false, true } )
        {
            for( bool synchro : { false, true } )
            {
                for( bool pending : { false, true } )
                {
                    CAPTURE( fps, wasSynchro, synchro, pending );
                    resetStubState();
                    outletHarness::Controller<ocam2KCtrl_test> app;
                    app.driver();
                    setPoweredOn( app );
                    configureStartupMode( app );
                    app.createStandardIndiToggleSw( app.m_indiP_synchro, "synchro" );
                    app.m_indiP_syncFreq.setDevice( app.m_syncDevice );
                    app.m_indiP_syncFreq.setName( app.m_syncFreqProp );
                    app.m_raw_height         = 121;
                    app.m_ocamDescrambleFile = "ocam_descramble_stub.txt";
                    app.m_fpsSet             = fps;
                    app.m_synchro            = wasSynchro;
                    app.m_synchroSet         = synchro;
                    app.m_reconfig           = pending;
                    app.m_nextMode           = "pending_mode";

                    std::vector<std::string> commands{ "mode science" };

                    if( fps > 0 && !wasSynchro )
                    {
                        commands.push_back( "fps " + std::to_string( fps ) );
                    }

                    commands.push_back( "fps 0" );
                    commands.push_back( synchro ? "synchro on" : "synchro off" );

                    if( !synchro )
                    {
                        commands.push_back( "fps " + std::to_string( fps ) );
                    }

                    for( const auto &command : commands )
                    {
                        queueSerialResponse( command.rfind( "fps ", 0 ) == 0
                                                 ? "fps [" + std::to_string( fps > 0 ? fps : 2000.0f ) + "] Hz\n"
                                                 : "ok\n" );
                    }

                    REQUIRE( app.configureAcquisition() == 0 );
                    REQUIRE( g_edtStubState.serialCommands == commands );
                    REQUIRE( app.m_synchro == synchro );
                    REQUIRE( app.m_reconfig == pending );
                    REQUIRE( app.m_nextMode == "pending_mode" );
                    REQUIRE( app.m_syncImageStream != nullptr );
                }
            }
        }
    }
}

/// Ordinary setters request latency-buffer reconfiguration while apply-only calls leave pending requests intact.
/** \ingroup ocam2KCtrl_reconfig_unit_test */
TEST_CASE( "ocam2KCtrl setters distinguish requests from configuration application", "[ocam2KCtrl][reconfig]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::setFPS(); MagAOX::app::ocam2KCtrl::setSynchro();
    #endif
    // clang-format on
    for( bool request : { false, true } )
    {
        for( bool synchro : { false, true } )
        {
            for( bool pending : { false, true } )
            {
                CAPTURE( request, synchro, pending );
                resetStubState();
                outletHarness::Controller<ocam2KCtrl_test> app;
                app.driver();
                setPoweredOn( app );
                app.createStandardIndiToggleSw( app.m_indiP_synchro, "synchro" );
                app.m_indiP_syncFreq.setDevice( app.m_syncDevice );
                app.m_indiP_syncFreq.setName( app.m_syncFreqProp );
                app.m_modeName = "science";
                app.m_nextMode = "pending_mode";
                app.m_fpsSet   = 175.0f;
                app.m_synchro  = synchro;
                app.m_reconfig = pending;

                if( !synchro )
                {
                    queueSerialResponse( "fps [175] Hz\n" );
                }

                REQUIRE( ( request ? app.setFPS() : app.setFPS( false ) ) == 0 );
                REQUIRE( app.m_reconfig == ( pending || ( request && !synchro ) ) );
                REQUIRE( app.m_nextMode == "pending_mode" );
                REQUIRE( g_edtStubState.serialCommands ==
                         ( synchro ? std::vector<std::string>{} : std::vector<std::string>{ "fps 175.000000" } ) );

                resetStubState();
                app.m_nextMode   = "pending_mode";
                app.m_synchro    = !synchro;
                app.m_synchroSet = synchro;
                app.m_reconfig   = pending;
                std::vector<std::string> commands{ "fps 0", synchro ? "synchro on" : "synchro off" };

                if( !synchro )
                {
                    commands.push_back( "fps 175.000000" );
                }

                for( const auto &command : commands )
                {
                    queueSerialResponse( command.rfind( "fps ", 0 ) == 0 ? "fps [175] Hz\n" : "ok\n" );
                }

                REQUIRE( ( request ? app.setSynchro() : app.setSynchro( false ) ) == 0 );
                REQUIRE( g_edtStubState.serialCommands == commands );
                REQUIRE( app.m_synchro == synchro );
                REQUIRE( app.m_reconfig == ( pending || ( request && !synchro ) ) );
                REQUIRE( app.m_nextMode == "pending_mode" );
            }
        }
    }
}

/// Repeated requests reload the current mode after an earlier cleanup has already consumed the requested mode.
/** \ingroup ocam2KCtrl_reconfig_unit_test */
TEST_CASE( "ocam2KCtrl repeated reconfiguration preserves a valid current mode", "[ocam2KCtrl][reconfig]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::reconfig();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    configureStartupMode( app );
    app.m_nextMode = "science";
    app.m_reconfig = true;

    for( int pass = 0; pass < 2; ++pass )
    {
        REQUIRE( app.reconfig() == 0 );
        REQUIRE( app.m_nextMode.empty() );
        REQUIRE( app.m_modeName == "science" );
        REQUIRE( app.m_raw_height == 121 );
        REQUIRE( app.m_reconfig );
    }
}

/// A mode callback arriving during cleanup remains pending until a subsequent cleanup applies it.
/** \ingroup ocam2KCtrl_reconfig_unit_test */
TEST_CASE( "ocam2KCtrl serializes requested modes with reconfiguration", "[ocam2KCtrl][reconfig]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::reconfig();
    MagAOX::app::dev::stdCamera<MagAOX::app::ocam2KCtrl>::newCallBack_mode();
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test app;
    configureStartupMode( app );
    app.m_cameraModes["calibration"] = app.m_cameraModes["science"];
    app.m_nextMode                   = "science";
    app.m_reconfig                   = false;
    app.m_indiP_mode.setName( "mode" );
    pcf::IndiProperty requested( pcf::IndiProperty::Switch, "ocam2KCtrl_test", "mode" );
    requested.add( pcf::IndiElement( "calibration", pcf::IndiElement::On ) );

    bool             locked = false;
    std::future<int> configuring;
    std::future<int> requesting;
    { //mutex scope
        std::unique_lock<std::recursive_mutex> cameraLock( app.m_cameraMutex );
        configuring         = std::async( std::launch::async,
                                  [&app]()
                                  {
                                      return app.reconfig();
                                  } );
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
        do
        {
            if( !app.m_indiMutex.try_lock() )
            {
                locked = true;
                break;
            }

            app.m_indiMutex.unlock();
            std::this_thread::yield();
        } while( std::chrono::steady_clock::now() < deadline );

        if( locked )
        {
            requesting = std::async( std::launch::async,
                                     [&app, &requested]()
                                     {
                                         return app.newCallBack_mode( requested );
                                     } );
        }
    }

    const int configured    = configuring.get();
    const int requestResult = requesting.valid() ? requesting.get() : -1;
    REQUIRE( locked );
    REQUIRE( configured == 0 );
    REQUIRE( requestResult == 0 );
    REQUIRE( app.m_modeName == "science" );
    REQUIRE( app.m_nextMode == "calibration" );
    REQUIRE( app.m_reconfig );
    REQUIRE( app.reconfig() == 0 );
    REQUIRE( app.m_modeName == "calibration" );
    REQUIRE( app.m_nextMode.empty() );
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
