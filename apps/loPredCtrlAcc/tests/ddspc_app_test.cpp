/** \file ddspc_app_test.cpp
 * \brief Test actual loPredCtrlAcc lifecycle and INDI callbacks without native stream or driver access.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc.hpp"
#include "causal_input_test_data.hpp"

#include <random>

/// \cond TEST_SUPPORT
namespace MagAOX::app
{
class loPredCtrlAcc_test : public loPredCtrlAcc
{
  public:
    using loPredCtrlAcc::configureShadowInputs;
    using loPredCtrlAcc::expireShadowInputs;
    using loPredCtrlAcc::m_indiP_ddspcAssisted;
    using loPredCtrlAcc::m_indiP_ddspcData;
    using loPredCtrlAcc::m_indiP_ddspcMode;
    using loPredCtrlAcc::m_indiP_ddspcShadow;
    using loPredCtrlAcc::m_indiP_ddspcStandard;
    using loPredCtrlAcc::m_indiP_ddspcStatus;
    using loPredCtrlAcc::m_indiP_ddspcTrain;
    using loPredCtrlAcc::m_shadowController;
    using loPredCtrlAcc::processShadowApplied;
    using loPredCtrlAcc::processShadowRecord;
    using loPredCtrlAcc::processShadowWfs;

    /// Configure a small synthetic model without creating any native app resources.
    void configure( bool shadow /**< [in] Startup mode. */ );

    /// Check that all output-owning helper entrypoints remained unused.
    void requireInhibited();

    /// Exercise the real WFS callback with unknown applied inputs and timing.
    void unverifiedWfs();

    /// Verify each legacy actuation/training control rejects a shadow request.
    void rejectLegacyControls();

    /// Reset through the real callback, independently of the legacy WFS processing path.
    void resetModels();

    /// Verify default compatibility mode retains its existing helper lifecycle dispatch.
    void requireLegacy();
};

inline void loPredCtrlAcc_test::configure( bool shadow /**< [in] Startup mode. */ )
{
    DdspcTestSemaphore::m_calls = 0;
    config.m_values             = { { "parameters.shadow", shadow ? "1" : "0" },
                                    { "parameters.num_modes", "2" },
                                    { "parameters.history", "2" },
                                    { "parameters.future", "2" },
                                    { "parameters.accel_enabled", "1" },
                                    { "parameters.accel_channels", "1" },
                                    { "parameters.accel_history", "2" },
                                    { "parameters.regularization", "0.4" },
                                    { "parameters.shadow_min_updates", "16" },
                                    { "parameters.is_integrating", "1" } };
    if( !shadow )
        config.m_values.erase( "parameters.shadow" );
    setupConfig();
    REQUIRE( loadConfigImpl( config ) == 0 );
}

inline void loPredCtrlAcc_test::requireInhibited()
{
    REQUIRE( DdspcTestSemaphore::m_calls == 0 );
    REQUIRE( frameGrabberT::m_loads == 0 );
    REQUIRE( frameGrabberT::m_starts == 0 );
    REQUIRE( frameGrabberT::m_logic == 0 );
    REQUIRE( frameGrabberT::m_updates == 0 );
    REQUIRE( frameGrabberT::m_stops == 0 );
    REQUIRE( telemeterT::m_loads == 0 );
    REQUIRE( telemeterT::m_starts == 0 );
    REQUIRE( telemeterT::m_logic == 0 );
    REQUIRE( accelShmimMonitorT::m_loads == 0 );
    REQUIRE( accelShmimMonitorT::m_starts == 0 );
    REQUIRE( m_controller == nullptr );
    REQUIRE_FALSE( m_updated );
    REQUIRE( m_fullCommand.size() == 0 );
    REQUIRE_FALSE( m_isLearning );
    REQUIRE_FALSE( m_isStdLearning );
    REQUIRE_FALSE( m_isPredictiveControl );
    REQUIRE_FALSE( m_isIntegrating );
    REQUIRE_FALSE( m_doTriggerLoad );
    REQUIRE_FALSE( m_doTriggerSave );
    REQUIRE_FALSE( m_switchExploration );
}

inline void loPredCtrlAcc_test::unverifiedWfs()
{
    shmimMonitorT::m_width    = 2;
    shmimMonitorT::m_height   = 1;
    shmimMonitorT::m_dataType = _DATATYPE_FLOAT;
    REQUIRE( allocate( dev::shmimT{} ) == 0 );
    float residual[2] = { 1, 2 };
    REQUIRE( processImage( residual, dev::shmimT{} ) == 0 );
    REQUIRE( allocate( accelShmimT{} ) == -1 );
    REQUIRE( processImage( nullptr, accelShmimT{} ) == 0 );
}

inline void loPredCtrlAcc_test::rejectLegacyControls()
{
    for( auto item :
         { std::make_pair( &m_indiP_learningToggle, &loPredCtrlAcc::newCallBack_m_indiP_learningToggle ),
           std::make_pair( &m_indiP_learningStdToggle, &loPredCtrlAcc::newCallBack_m_indiP_learningStdToggle ),
           std::make_pair( &m_indiP_integratingToggle, &loPredCtrlAcc::newCallBack_m_indiP_integratingToggle ),
           std::make_pair( &m_indiP_predictingToggle, &loPredCtrlAcc::newCallBack_m_indiP_predictingToggle ) } )
    {
        auto request = *item.first;
        request["toggle"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( ( this->*item.second )( request ) == -1 );
    }
    auto save = m_indiP_saveToggle;
    save["request"].setSwitchState( pcf::IndiElement::On );
    auto load = m_indiP_loadToggle;
    load["request"].setSwitchState( pcf::IndiElement::On );
    auto exploration = m_indiP_exploration;
    exploration["target"].set( "10,1,1" );
    REQUIRE( newCallBack_m_indiP_saveToggle( save ) == -1 );
    REQUIRE( newCallBack_m_indiP_loadToggle( load ) == -1 );
    REQUIRE( newCallBack_m_indiP_exploration( exploration ) == -1 );
}

inline void loPredCtrlAcc_test::resetModels()
{
    auto request = m_indiP_resetToggle;
    request["request"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( newCallBack_m_indiP_resetToggle( request ) == 0 );
}

inline void loPredCtrlAcc_test::requireLegacy()
{
    REQUIRE_FALSE( m_shadowMode );
    REQUIRE_FALSE( m_shadowController );
    REQUIRE( frameGrabberT::m_loads == 1 );
    REQUIRE( frameGrabberT::m_starts == 1 );
    REQUIRE( telemeterT::m_starts == 1 );
    REQUIRE( accelShmimMonitorT::m_starts == 1 );
    REQUIRE( m_isIntegrating );
}
} // namespace MagAOX::app

namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
/// Build a one-element INDI request retaining actual device/property identity.
pcf::IndiProperty
switch_request( const pcf::IndiProperty          &property /**< [in] Identity. */,
                const std::string                &element /**< [in] Selected element. */,
                pcf::IndiElement::SwitchStateType state = pcf::IndiElement::On /**< [in] Requested state. */ )
{
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.setDevice( property.getDevice() );
    request.setName( property.getName() );
    request.add( pcf::IndiElement( element, state ) );
    return request;
}

/// Make explicitly synthetic applied-command observations, never using proposals as inputs.
DDSPC::ShadowRecord synthetic_app_record( std::uint64_t index /**< [in] Decision identity. */ )
{
    DDSPC::ShadowRecord record;
    record.m_index = index;
    record.m_residual.resize( 2, 1 );
    record.m_appliedCommand.resize( 2, 1 );
    record.m_acceleration.resize( 1, 1 );
    std::mt19937 random( 9173 + index );
    for( int mode = 0; mode < 2; ++mode )
    {
        record.m_residual( mode )       = float( int( random() % 2001 ) - 1000 ) / 1000;
        record.m_appliedCommand( mode ) = float( int( random() % 2001 ) - 1000 ) / 1000;
    }
    record.m_acceleration( 0 ) = float( int( random() % 2001 ) - 1000 ) / 1000;
    record.m_wfsValid = record.m_appliedValid = record.m_timingValid = record.m_accelValid = true;
    return record;
}
} // namespace
/// \endcond

/** \brief Shadow lifecycle and every output entrypoint remain non-actuating.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Application shadow lifecycle never starts the output publisher", "[loPredCtrlAcc][app][shadow]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::appStartup();
    MagAOX::app::loPredCtrlAcc::loadConfigImpl();
    MagAOX::app::loPredCtrlAcc::processImage();
    MagAOX::app::loPredCtrlAcc::configureAcquisition();
    MagAOX::app::loPredCtrlAcc::acquireAndCheckValid();
    MagAOX::app::loPredCtrlAcc::loadImageIntoStream();
    MagAOX::app::loPredCtrlAcc::appShutdown();
#endif
    // clang-format on
    MagAOX::app::loPredCtrlAcc_test app;
    app.configure( true );
    REQUIRE( app.appStartup() == 0 );
    app.requireInhibited();
    REQUIRE( app.m_indiP_ddspcShadow.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_ddspcShadow["toggle"].getSwitchState() == pcf::IndiElement::On );
    REQUIRE( app.configureAcquisition() == -1 );
    REQUIRE( app.startAcquisition() == -1 );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    float destination[2] = { 37, 41 };
    REQUIRE( app.loadImageIntoStream( destination ) == -1 );
    REQUIRE( destination[0] == 37 );
    REQUIRE( destination[1] == 41 );
    app.rejectLegacyControls();
    app.unverifiedWfs();
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcData["applied_valid"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcData["timing_valid"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcStandard["updates"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcAssisted["updates"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcStatus["output"].get<std::string>() == "inhibited" );
    REQUIRE( app.m_indiP_ddspcStatus["standard_reason"].get<std::string>() == "invalid_or_missing_data" );
    app.requireInhibited();
    REQUIRE( app.appShutdown() == 0 );
    app.requireInhibited();
}

/** \brief Real INDI callbacks select ready models without affecting independent training or output.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Application INDI selection and independent training remain shadow only", "[loPredCtrlAcc][app][shadow]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcMode();
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcTrain();
    MagAOX::app::loPredCtrlAcc::processShadowRecord();
    MagAOX::app::loPredCtrlAcc::updateShadowINDI();
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_resetToggle();
#endif
    // clang-format on
    using DDSPC::ShadowMode;
    MagAOX::app::loPredCtrlAcc_test app;
    app.configure( true );
    REQUIRE( app.appStartup() == 0 );
    auto assisted = switch_request( app.m_indiP_ddspcMode, "accel_assisted" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( assisted ) == -1 );
    REQUIRE( app.m_indiP_ddspcMode.getState() == INDI_ALERT );
    REQUIRE( app.m_indiP_ddspcMode["standard"].getSwitchState() == pcf::IndiElement::On );
    auto training = app.m_indiP_ddspcTrain;
    training["standard"].setSwitchState( pcf::IndiElement::On );
    training["accel_assisted"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_ddspcTrain( training ) == 0 );
    for( std::uint64_t i = 0; i < 100; ++i )
        REQUIRE( app.processShadowRecord( synthetic_app_record( i ) ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStandard["ready"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcAssisted["ready"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcStandard["prediction_valid"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcAssisted["prediction_valid"].get<double>() == 1 );
    const auto standard  = app.m_shadowController->prediction_matrix( ShadowMode::Standard );
    const auto augmented = app.m_shadowController->prediction_matrix( ShadowMode::AccelAssisted );
    REQUIRE( standard.cols() == 10 );
    REQUIRE( augmented.cols() == 12 );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( assisted ) == 0 );
    REQUIRE( app.m_indiP_ddspcMode["standard"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_ddspcStatus["selected"].get<std::string>() == "accel_assisted" );
    auto both = app.m_indiP_ddspcMode;
    both["standard"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( both ) == -1 );
    auto off = switch_request( app.m_indiP_ddspcMode, "standard", pcf::IndiElement::Off );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( off ) == -1 );
    auto unknown = switch_request( app.m_indiP_ddspcMode, "unknown" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( unknown ) == -1 );
    auto wrong = assisted;
    wrong.setDevice( "another-device" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( wrong ) == -1 );
    wrong = assisted;
    wrong.setName( "another-property" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( wrong ) == -1 );
    wrong = pcf::IndiProperty( pcf::IndiProperty::Number );
    wrong.setDevice( assisted.getDevice() );
    wrong.setName( assisted.getName() );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( wrong ) == -1 );
    auto invalidTrain = training;
    invalidTrain.add( pcf::IndiElement( "unknown", pcf::IndiElement::On ) );
    invalidTrain["standard"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( app.newCallBack_m_indiP_ddspcTrain( invalidTrain ) == -1 );
    REQUIRE( app.m_shadowController->status( ShadowMode::Standard ).m_trainingEnabled );
    auto pause = switch_request( app.m_indiP_ddspcTrain, "standard", pcf::IndiElement::Off );
    REQUIRE( app.newCallBack_m_indiP_ddspcTrain( pause ) == 0 );
    REQUIRE_FALSE( app.m_shadowController->status( ShadowMode::Standard ).m_trainingEnabled );
    REQUIRE( app.m_shadowController->status( ShadowMode::AccelAssisted ).m_trainingEnabled );
    REQUIRE( app.m_shadowController->prediction_matrix( ShadowMode::Standard ) == standard );
    REQUIRE( app.m_shadowController->prediction_matrix( ShadowMode::AccelAssisted ) == augmented );
    app.rejectLegacyControls();
    app.requireInhibited();
    app.resetModels();
    REQUIRE_FALSE( app.m_shadowController->status( ShadowMode::Standard ).m_ready );
    REQUIRE_FALSE( app.m_shadowController->status( ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( app.m_shadowController->status( ShadowMode::Standard ).m_updates == 0 );
    REQUIRE( app.m_shadowController->status( ShadowMode::AccelAssisted ).m_updates == 0 );
    app.requireInhibited();
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief Startup defaults retain legacy lifecycle dispatch and cannot enable causal output through INDI.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Application legacy compatibility is the startup default", "[loPredCtrlAcc][app][shadow]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl();
    MagAOX::app::loPredCtrlAcc::appStartup();
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcMode();
#endif
    // clang-format on
    MagAOX::app::loPredCtrlAcc_test app;
    app.configure( false );
    REQUIRE( app.appStartup() == 0 );
    app.requireLegacy();
    REQUIRE( app.m_indiP_ddspcMode.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_ddspcTrain.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_ddspcStatus["execution"].get<std::string>() == "legacy" );
    auto mode = switch_request( app.m_indiP_ddspcMode, "accel_assisted" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( mode ) == -1 );
    auto train = switch_request( app.m_indiP_ddspcTrain, "standard" );
    REQUIRE( app.newCallBack_m_indiP_ddspcTrain( train ) == -1 );
    REQUIRE_FALSE( app.processShadowRecord( synthetic_app_record( 0 ) ) );
    REQUIRE( app.configureShadowInputs( input_config() ) == -1 );
    app.requireLegacy();
    REQUIRE( app.appShutdown() == 0 );
}
/** \brief Timed synthetic inputs train both real app models like manually aligned histories without output.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Application timed input integration remains non-actuating", "[loPredCtrlAcc][app][input]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::configureShadowInputs();
    MagAOX::app::loPredCtrlAcc::processShadowWfs();
    MagAOX::app::loPredCtrlAcc::processShadowApplied();
    MagAOX::app::loPredCtrlAcc::expireShadowInputs();
#endif
    // clang-format on
    MagAOX::app::loPredCtrlAcc_test app;
    app.configure( true );
    REQUIRE( app.appStartup() == 0 );
    const auto config = input_config();
    REQUIRE( app.configureShadowInputs( config ) == 0 );
    DDSPC::ShadowConfig settings;
    settings.m_modes         = 2;
    settings.m_history       = 2;
    settings.m_future        = 2;
    settings.m_accelChannels = 1;
    settings.m_accelHistory  = 2;
    settings.m_minUpdates    = 16;
    settings.m_lambda        = 0.4f;
    DDSPC::ShadowController oracle( settings );
    auto                    training = app.m_indiP_ddspcTrain;
    for( const char *name : { "standard", "accel_assisted" } )
        training[name].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_ddspcTrain( training ) == 0 );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        oracle.training( mode, true );
    auto wfs = input_wfs( config, 0 );
    REQUIRE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
    auto command = input_command( config, 0 );
    for( std::uint64_t i = 0; i < 80; ++i )
    {
        wfs = input_wfs( config, i + 1 );
        REQUIRE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
        command = input_command( config, i );
        REQUIRE( app.processShadowApplied( command, command.m_stamp.m_receivedNs ) );
        DDSPC::ShadowRecord truth;
        truth.m_index          = 100 + i;
        truth.m_residual       = input_wfs( config, i ).m_residual;
        truth.m_appliedCommand = input_command( config, i ).m_applied;
        truth.m_acceleration   = input_wfs( config, i ).m_acceleration;
        truth.m_wfsValid = truth.m_appliedValid = truth.m_timingValid = truth.m_accelValid = true;
        REQUIRE( oracle.process( truth ) );
        for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
            REQUIRE( app.m_shadowController->prediction_matrix( mode ) == oracle.prediction_matrix( mode ) );
    }
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStandard["ready"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcAssisted["ready"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcStatus["input_reason"].get<std::string>() == "paired_input_records" );
    REQUIRE( app.m_indiP_ddspcStatus["input_provenance"].get<std::string>() == "synthetic_applied" );
    auto assisted = switch_request( app.m_indiP_ddspcMode, "accel_assisted" );
    REQUIRE( app.newCallBack_m_indiP_ddspcMode( assisted ) == 0 );
    REQUIRE_FALSE( app.processShadowRecord( synthetic_app_record( 800 ) ) ); // No mixing the Phase 2 and timed seams.
    REQUIRE_FALSE( app.expireShadowInputs( command.m_stamp.m_receivedNs + config.m_maxAgeNs + config.m_periodNs ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStatus["input_reason"].get<std::string>() == "stale_input_evidence" );
    REQUIRE( app.m_indiP_ddspcStandard["ready"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcAssisted["ready"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcMode["accel_assisted"].getSwitchState() == pcf::IndiElement::On );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
    {
        REQUIRE( app.m_shadowController->prediction_matrix( mode ) == oracle.prediction_matrix( mode ) );
        REQUIRE( app.m_shadowController->status( mode ).m_trainingEnabled );
        REQUIRE( app.m_shadowController->status( mode ).m_samples == 0 );
    }
    auto changedConfig = config;
    ++changedConfig.m_layout;
    ++changedConfig.m_clockEpoch;
    REQUIRE( app.configureShadowInputs( changedConfig ) == 0 );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
    {
        REQUIRE( app.m_shadowController->prediction_matrix( mode ).isZero() );
        REQUIRE( app.m_shadowController->status( mode ).m_updates == 0 );
        REQUIRE( app.m_shadowController->status( mode ).m_trainingEnabled );
    }
    REQUIRE( app.m_shadowController->selected() == DDSPC::ShadowMode::AccelAssisted );
    app.requireInhibited();
    REQUIRE( app.appShutdown() == 0 );
    app.requireInhibited();
}

/** \brief Raw WFS data, driver-prepared commands, and unverified timing cannot arm live training.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Application live input gates stay closed without application and exposure evidence",
           "[loPredCtrlAcc][app][input]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::configureShadowInputs();
    MagAOX::app::loPredCtrlAcc::processShadowWfs();
    MagAOX::app::loPredCtrlAcc::processShadowApplied();
    MagAOX::app::loPredCtrlAcc::allocate();
    MagAOX::app::loPredCtrlAcc::processImage();
#endif
    // clang-format on
    MagAOX::app::loPredCtrlAcc_test app;
    app.configure( true );
    REQUIRE( app.appStartup() == 0 );
    auto config = input_config();
    auto wfs    = input_wfs( config, 0 );
    REQUIRE_FALSE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
    config.m_timingVerified = false;
    REQUIRE( app.configureShadowInputs( config ) == 0 );
    REQUIRE_FALSE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStatus["input_reason"].get<std::string>() == "unverified_input_contract" );
    config.m_timingVerified = true;
    REQUIRE( app.configureShadowInputs( config ) == 0 );
    REQUIRE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
    wfs = input_wfs( config, 1 );
    REQUIRE( app.processShadowWfs( wfs, wfs.m_decisionNs ) );
    auto command       = input_command( config, 0 );
    command.m_evidence = DDSPC::InputEvidence::DriverPrepared;
    REQUIRE_FALSE( app.processShadowApplied( command, command.m_stamp.m_receivedNs ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStatus["input_reason"].get<std::string>() == "command_application_unverified" );
    app.unverifiedWfs(); // Native allocation/callback cannot reuse a synthetic or previously established session.
    REQUIRE_FALSE(
        app.processShadowApplied( input_command( config, 1 ), command.m_stamp.m_receivedNs + config.m_periodNs ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_ddspcStatus["input_reason"].get<std::string>() == "unverified_input_contract" );
    REQUIRE( app.m_indiP_ddspcData["applied_valid"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcData["timing_valid"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcStandard["updates"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcAssisted["updates"].get<double>() == 0 );
    app.requireInhibited();
    REQUIRE( app.appShutdown() == 0 );
}
} // namespace libXWCTest::loPredCtrlAccTest
