/** \file native_application_test.cpp
 * \brief Installed native entrypoint, rate-limited INDI and persistent output inhibition with transport doubles.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "adc_transport_fixture.hpp"
#include "../loPredCtrlAcc_native.hpp"
#include <chrono>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Exercise the installed entrypoint without invoking its main, a native stream or a service.
class InstalledApplication : public MagAOX::app::loPredCtrlAcc_native
{
  public:
    using loPredCtrlAcc::m_driver;
    using loPredCtrlAcc::m_indiP_adc;
    using loPredCtrlAcc::m_indiP_ddspcMode;
    using loPredCtrlAcc::m_indiP_ddspcTrain;
    using loPredCtrlAcc::m_indiP_nativeSafety;
    using loPredCtrlAcc::m_indiP_nativeSources;
    using loPredCtrlAcc::m_indiP_nativeTiming;
    using loPredCtrlAcc::m_nativeNextDiagnosticNs;
    using loPredCtrlAcc::m_nativeObserver;
    using loPredCtrlAcc::nativeDiagnosticsDue;

    /// Return one registered compact property by its public name.
    const pcf::IndiProperty &property( const std::string &name /**< [in] Registered property identity. */ ) const;

    /// Run the real default loader against test-owned configuration values.
    bool load( const std::map<std::string, std::string> &values /**< [in] Explicit optional overrides. */ );

    /// Verify native acquisition owns no output helper and every output entrypoint refuses commands.
    bool inhibited();
};

bool InstalledApplication::load( const std::map<std::string, std::string> &values )
{
    config.m_values = values;
    setupConfig();
    loPredCtrlAcc_native::loadConfig();
    return !m_shutdown;
}

bool InstalledApplication::inhibited()
{
    float sentinel[2]{ 123, 456 };
    return configureAcquisition() < 0 && startAcquisition() < 0 && acquireAndCheckValid() != 0 &&
           loadImageIntoStream( sentinel ) < 0 && sentinel[0] == 123 && sentinel[1] == 456 && !m_controller &&
           !m_updated && m_fullCommand.size() == 0 && !m_isLearning && !m_isStdLearning && !m_isPredictiveControl &&
           !m_isIntegrating;
}

const pcf::IndiProperty &InstalledApplication::property( const std::string &name ) const
{
    return *m_registered.at( name );
}
} // namespace
/// \endcond

/** \brief Default installed startup is continuous without files or DM output; pending mode requests grant no readiness.
 */
TEST_CASE( "Installed native application defaults to continuous read-only acquisition", "[native-observation][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_native::loadConfig;
    MagAOX::app::loPredCtrlAcc_native::appStartup;
    MagAOX::app::loPredCtrlAcc::updateNativeShadowINDI;
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcMode;
#endif
    // clang-format on
    for( int restart = 0; restart < 2; ++restart )
    {
        InstalledApplication app;
        REQUIRE( app.load( {} ) );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.inhibited() );
        REQUIRE( app.property( "app_health" )["shadow"].get<std::string>() == "true" );
        REQUIRE( app.property( "safety" )["dm_write_enabled"].get<std::string>() == "false" );
        REQUIRE( app.property( "timing" )["phase_offset_us"].get<std::string>() == "unqualified" );
        std::size_t elements = 0;
        for( const auto &[name, property] : app.m_registered )
        {
            elements += property->getNumElements();
            REQUIRE( property->getPerm() ==
                     ( name == "ddspc_mode" ? pcf::IndiProperty::ReadWrite : pcf::IndiProperty::ReadOnly ) );
        }
        REQUIRE( elements == 59 );
        REQUIRE( app.m_registered.size() == 11 );
        REQUIRE_FALSE( app.m_registered.count( "native_adc" ) );
        REQUIRE_FALSE( app.m_registered.count( "learn" ) );
        auto mode = app.m_indiP_ddspcMode;
        mode["standard"].setSwitchState( pcf::IndiElement::Off );
        mode["accel_assisted"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_ddspcMode( mode ) == 0 );
        REQUIRE( app.m_indiP_ddspcMode["accel_assisted"].getSwitchState() == pcf::IndiElement::On );
        REQUIRE( app.property( "ddspc_control" )["requested_mode"].get<std::string>() == "accel_assisted" );
        REQUIRE( app.property( "safety" )["dm_write_enabled"].get<std::string>() == "false" );
        pcf::IndiProperty train( pcf::IndiProperty::Switch );
        train.setDevice( app.m_indiP_ddspcMode.getDevice() );
        train.setName( "ddspc_train" );
        train.add( pcf::IndiElement( "standard", pcf::IndiElement::Off ) );
        train["standard"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_ddspcTrain( train ) < 0 );
        REQUIRE( app.appLogic() == 0 );
        const auto sent = app.m_driver.m_sent.size();
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_driver.m_sent.size() == sent );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( app.inhibited() );
    }
}

/** \brief Configuration and reload cannot disable shadow/input guards or start Legacy/output behavior. */
TEST_CASE( "Installed native configuration refuses unsafe overrides", "[native-observation][app]" )
{
    const auto           key = GENERATE( "parameters.shadow",
                                         "adc.enabled",
                                         "native_shadow.enabled",
                                         "parameters.own_shmim",
                                         "parameters.shadow_pipeline",
                                         "diagnostics.period_ms" );
    InstalledApplication app;
    const bool           output =
        std::string( key ) == "parameters.own_shmim" || std::string( key ) == "parameters.shadow_pipeline";
    REQUIRE_FALSE( app.load( { { key, output ? "1" : "0" } } ) );
    REQUIRE( app.appStartup() < 0 );
    REQUIRE( app.inhibited() );
}

/** \brief Rate admission is independent of acquisition and never exceeds the configured diagnostic cadence. */
TEST_CASE( "Installed diagnostics use a bounded steady-clock update cadence", "[native-observation][app]" )
{
    InstalledApplication app;
    REQUIRE( app.nativeDiagnosticsDue( 1000000000 ) );
    REQUIRE_FALSE( app.nativeDiagnosticsDue( 1000000001 ) );
    REQUIRE_FALSE( app.nativeDiagnosticsDue( 1249999999 ) );
    REQUIRE( app.nativeDiagnosticsDue( 1250000000 ) );
    REQUIRE_FALSE( app.nativeDiagnosticsDue( -1 ) );
}

/** \brief A rejected runtime reload cannot dispatch Legacy helpers during the loop or shutdown. */
TEST_CASE( "Installed native reload fails closed and joins only native resources", "[native-observation][app]" )
{
    InstalledApplication app;
    REQUIRE( app.load( {} ) );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE_FALSE( app.load( { { "parameters.shadow", "0" } } ) );
    REQUIRE( app.appLogic() < 0 );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( app.inhibited() );
}

/** \brief Actual app properties report received ADC/modal values and a healthy idle DM without modifying any input. */
TEST_CASE( "Installed INDI diagnostics match native read-only inputs", "[native-observation][app]" )
{
    IMAGE_METADATA metadata{};
    metadata.naxis   = 2;
    metadata.size[0] = 2;
    metadata.size[1] = metadata.size[2] = 1;
    metadata.datatype                   = _DATATYPE_FLOAT;
    metadata.cnt0                       = 1;
    metadata.inode                      = 123;
    clock_gettime( CLOCK_REALTIME, &metadata.atime );
    metadata.writetime = metadata.atime;
    std::array<float, 2> values{ 1.5f, -2.f };
    IMAGE                image{};
    image.md        = &metadata;
    image.array.raw = values.data();
    AdcTransportFixture  adc( "installed-adc", 16 ), dm( "installed-dm", image ), modal( "installed-modal", image );
    InstalledApplication app;
    REQUIRE( app.load( { { "adc.shmimName", "installed-adc" },
                         { "native_shadow.modal_stream", "installed-modal" },
                         { "native_shadow.camera_stream", "installed-camera" },
                         { "native_shadow.dm_stream", "installed-dm" } } ) );
    REQUIRE( app.appStartup() == 0 );
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( std::chrono::steady_clock::now() < until )
    {
        const auto status = app.m_nativeObserver->status();
        if( status.m_sources[3].m_samples && status.m_sources[3].m_idle && status.m_sources[0].m_samples &&
            status.m_sources[1].m_samples )
            break;
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    app.m_nativeNextDiagnosticNs = 0;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.property( "input_health" )["dm"].get<std::string>() == "idle_valid;restarts=0" );
    REQUIRE( app.property( "input_rates" )["dm_hz"].get<double>() == 0 );
    REQUIRE( app.property( "adc_values" )["raw0"].get<double>() == 2049 );
    DDSPC::NativeObservation command;
    std::array<float, 2>     observed{};
    REQUIRE( app.m_nativeObserver->copy_command( 0, command, observed ) );
    REQUIRE( observed == values );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( metadata.cnt0 == 1 );
    REQUIRE( metadata.write == 0 );
    REQUIRE( values == std::array<float, 2>{ 1.5f, -2.f } );
}
/** \brief Detailed metadata remains explicitly opt-in and cannot change the running INDI schema or output interlocks.
 */
TEST_CASE( "Native detailed INDI requires startup opt-in", "[native-observation][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::setupNativeShadowINDI;
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
#endif
    // clang-format on
    InstalledApplication debug;
    REQUIRE( debug.load( { { "diagnostics.debug_indi", "1" } } ) );
    REQUIRE( debug.appStartup() == 0 );
    REQUIRE( debug.m_registered.count( "native_adc_forwarding" ) == 1 );
    REQUIRE( debug.m_registered.count( "native_adc" ) == 1 );
    REQUIRE_FALSE( debug.m_registered.count( "input_rates" ) );
    REQUIRE( debug.property( "native_modal" ).find( "history_retained" ) );
    REQUIRE_FALSE( debug.property( "native_modal" ).find( "input_queue_occupancy" ) );
    REQUIRE( debug.property( "ddspc_train" ).getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( debug.inhibited() );
    REQUIRE( debug.appShutdown() == 0 );
    InstalledApplication compact;
    REQUIRE( compact.load( {} ) );
    REQUIRE( compact.appStartup() == 0 );
    REQUIRE_FALSE( compact.load( { { "diagnostics.debug_indi", "1" } } ) );
    REQUIRE( compact.appLogic() < 0 );
    REQUIRE( compact.appShutdown() == 0 );
    REQUIRE( compact.inhibited() );
}
} // namespace libXWCTest::loPredCtrlAccTest
