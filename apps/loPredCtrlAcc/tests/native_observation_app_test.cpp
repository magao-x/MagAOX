/** \file native_observation_app_test.cpp
 * \brief Real private application entrypoint guards; lifecycle/INDI doubles and absent private inputs only.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc_native_observe.hpp"
#include <filesystem>
#include <unistd.h>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Expose guarded application admission without running a MagAO-X app, INDI server or native stream library.
class NativeApplication : public MagAOX::app::loPredCtrlAcc_native_observe
{
  public:
    /// Load real startup validation from test-owned values.
    int load( const std::map<std::string, std::string> &values /**< [in] Explicit private config values. */ );

    /// Attempt both existing training switches; native observation must refuse even a direct callback.
    int train();

    /// Confirm all model record-admission seams reject native observation.
    bool records_inhibited();

    /// Check output helpers, original WFS monitor and operational FPS subscription remain absent.
    bool outputs_inhibited();

    /// Return the actual private training property's permission.
    bool training_read_only() const;
};
int NativeApplication::load( const std::map<std::string, std::string> &values )
{
    config.m_values                           = values;
    config.m_values["diagnostics.debug_indi"] = "1"; // Exercise the explicitly retained metadata interface.
    setupConfig();
    return loadConfigImpl( config );
}
int NativeApplication::train()
{
    auto p = m_indiP_ddspcTrain;
    p["standard"].setSwitchState( pcf::IndiElement::On );
    p["accel_assisted"].setSwitchState( pcf::IndiElement::On );
    return newCallBack_m_indiP_ddspcTrain( p );
}
bool NativeApplication::records_inhibited()
{
    DDSPC::ShadowRecord       record;
    DDSPC::TimedWfsRecord     wfs;
    DDSPC::AppliedInputRecord applied;
    DDSPC::ShadowAdcRecord    adc;
    DDSPC::InputConfig        inputs;
    return !processShadowRecord( record ) && !processShadowWfs( wfs, 1 ) && !processShadowApplied( applied, 1 ) &&
           !processShadowAdc( adc, 1 ) && configureShadowInputs( inputs ) < 0 &&
           m_shadowController->status( DDSPC::ShadowMode::Standard ).m_updates == 0 &&
           m_shadowController->status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 &&
           !m_shadowController->status( DDSPC::ShadowMode::Standard ).m_trainingEnabled &&
           !m_shadowController->status( DDSPC::ShadowMode::AccelAssisted ).m_trainingEnabled;
}
bool NativeApplication::outputs_inhibited()
{
    float sentinel[2]{ 123, 456 };
    return configureAcquisition() < 0 && startAcquisition() < 0 && acquireAndCheckValid() != 0 &&
           loadImageIntoStream( sentinel ) < 0 && sentinel[0] == 123 && sentinel[1] == 456 && !m_controller &&
           !m_updated && m_fullCommand.size() == 0 && !m_isLearning && !m_isStdLearning && !m_isPredictiveControl &&
           !m_isIntegrating && MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_starts == 0 &&
           MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc>::m_starts == 0 &&
           m_indiP_fpsSource.getDevice().empty();
}
bool NativeApplication::training_read_only() const
{
    return m_indiP_ddspcTrain.getPerm() == pcf::IndiProperty::ReadOnly;
}
} // namespace
/// \endcond

/** \brief An explicitly selected observer refuses training, all record injection, Legacy output and control
 * subscriptions. */
TEST_CASE( "Private native application remains observation-only through startup and shutdown",
           "[native-observation][app]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
    MagAOX::app::loPredCtrlAcc::appStartup;
    MagAOX::app::loPredCtrlAcc::appShutdown;
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcTrain;
#endif
    // clang-format on
    auto path = ( std::filesystem::temp_directory_path() / "ddspc-native-app-XXXXXX" ).string();
    REQUIRE( mkdtemp( path.data() ) != nullptr );
    NativeApplication app;
    REQUIRE( app.load( { { "parameters.shadow", "1" },
                         { "parameters.shadow_pipeline", "0" },
                         { "parameters.own_shmim", "0" },
                         { "parameters.num_modes", "2" },
                         { "adc.enabled", "1" },
                         { "adc.shmimName", "obs-app-adc" },
                         { "native_shadow.enabled", "1" },
                         { "native_shadow.record_dm_commands", "1" },
                         { "native_shadow.command_queue_capacity", "4" },
                         { "native_shadow.command_max_elements", "4096" },
                         { "native_shadow.directory", path },
                         { "native_shadow.modal_stream", "obs-app-modal" },
                         { "native_shadow.camera_stream", "obs-app-camera" },
                         { "native_shadow.dm_stream", "obs-app-dm" } } ) == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.training_read_only() );
    REQUIRE( app.train() < 0 );
    REQUIRE( app.records_inhibited() );
    REQUIRE( app.outputs_inhibited() );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( std::filesystem::file_size( std::filesystem::path( path ) / "dm_commands.bin" ) == 8 );
    std::filesystem::remove_all( path );
}

/** \brief Unsafe opt-in combinations and the default-disabled private executable fail before any worker starts. */
TEST_CASE( "Private native application refuses invalid observation configurations", "[native-observation][app]" )
{
    const int                          fault = GENERATE( 0, 1, 2, 3, 4, 5, 6 );
    std::map<std::string, std::string> values{
        { "parameters.shadow", "1" },
        { "parameters.shadow_pipeline", "0" },
        { "parameters.own_shmim", "0" },
        { "adc.enabled", "1" },
        { "native_shadow.enabled", "1" },
        { "native_shadow.directory", "/private/tmp/not-created-by-validation" } };
    if( fault == 0 )
        values["native_shadow.enabled"] = "0";
    if( fault == 1 )
        values["parameters.shadow"] = "0";
    if( fault == 2 )
        values["parameters.shadow_pipeline"] = "1";
    if( fault == 3 )
        values["adc.enabled"] = "0";
    if( fault == 4 )
        values["parameters.own_shmim"] = "1";
    if( fault == 5 )
        values["native_shadow.directory"] = "/opt/MagAOX/observation";
    if( fault == 6 )
        values["native_shadow.modal_publication_clock"] = "auto";
    NativeApplication app;
    if( fault == 0 )
    {
        REQUIRE( app.load( values ) == 0 );
        REQUIRE( app.appStartup() < 0 );
    }
    else
    {
        REQUIRE( app.load( values ) < 0 );
        REQUIRE( app.appStartup() < 0 );
    }
}
} // namespace libXWCTest::loPredCtrlAccTest
