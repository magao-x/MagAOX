/** \file ddspc_shadow_pipeline_test.cpp
 * \brief Actual application replay, causal confirmation, source recovery and shadow-output regression tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "shadow_app_test_support.hpp"
#include <algorithm>
#include <atomic>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Expose only the real selection property for a stream-free application integration check.
class SwitchingReplayApplication : public ReplayApplication
{
  public:
    using loPredCtrlAcc::m_indiP_ddspcMode;
};
} // namespace
/// \endcond

/** \brief INDI selection reaches the actual candidate policy without resetting either qualified fit. */
TEST_CASE( "Application INDI updates the shared pipeline request without rebuilding models",
           "[loPredCtrlAcc][app][pipeline][switching]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::newCallBack_m_indiP_ddspcMode;
    DDSPC::ShadowPipeline::request;
    DDSPC::ShadowPolicy::request;
#endif
    // clang-format on
    SwitchingReplayApplication app;
    ShadowReplayConfig         recording;
    recording.m_frames = 400;
    app.prepare( replay_model_config(), recording.m_inputs );
    for( const bool assistance : { true, false } )
    {
        auto request = app.m_indiP_ddspcMode;
        request["standard"].setSwitchState( assistance ? pcf::IndiElement::Off : pcf::IndiElement::On );
        request["accel_assisted"].setSwitchState( assistance ? pcf::IndiElement::On : pcf::IndiElement::Off );
        REQUIRE( app.newCallBack_m_indiP_ddspcMode( request ) == 0 );
        REQUIRE_FALSE( app.shadowPipelineStatus().m_candidateValid );
        REQUIRE_FALSE( app.model_status( DDSPC::ShadowMode::Standard ).m_ready );
        REQUIRE_FALSE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    }
    app.train();
    replay_shadow( app, recording );
    const auto standard = app.model( DDSPC::ShadowMode::Standard );
    const auto assisted = app.model( DDSPC::ShadowMode::AccelAssisted );
    for( const bool assistance : { true, false, true, false } )
    {
        auto request = app.m_indiP_ddspcMode;
        request["standard"].setSwitchState( assistance ? pcf::IndiElement::Off : pcf::IndiElement::On );
        request["accel_assisted"].setSwitchState( assistance ? pcf::IndiElement::On : pcf::IndiElement::Off );
        REQUIRE( app.newCallBack_m_indiP_ddspcMode( request ) == 0 );
        REQUIRE( app.shadowPipelineStatus().m_requested ==
                 ( assistance ? DDSPC::PolicyMode::Assisted : DDSPC::PolicyMode::Standard ) );
        REQUIRE( app.model( DDSPC::ShadowMode::Standard ) == standard );
        REQUIRE( app.model( DDSPC::ShadowMode::AccelAssisted ) == assisted );
        DDSPC::Matrix candidate = DDSPC::Matrix::Ones( 2, 1 );
        REQUIRE_FALSE( app.shadowCandidate( candidate ) );
        REQUIRE( candidate.isZero( 0 ) );
    }
    DDSPC::PolicyConfig config;
    config.m_recoveryFrames = 2;
    DDSPC::ShadowPolicy    policy( DDSPC::PolicyMode::Standard, config );
    DDSPC::ControlEvidence evidence( 2 );
    evidence.m_commonValid = evidence.m_featuresValid = evidence.m_standardReady = evidence.m_assistedReady = true;
    evidence.m_featureReason = DDSPC::FeatureReason::Ready;
    const DDSPC::Matrix zero = DDSPC::Matrix::Zero( 2, 1 );
    REQUIRE( policy.decide( evidence, zero, zero ).m_effective == DDSPC::PolicyMode::Standard );
    policy.request( DDSPC::PolicyMode::Assisted );
    REQUIRE( policy.decide( evidence, zero, zero ).m_effective == DDSPC::PolicyMode::Standard );
    REQUIRE( policy.decide( evidence, zero, zero ).m_effective == DDSPC::PolicyMode::Assisted );
    evidence.m_featuresValid = false;
    REQUIRE( policy.decide( evidence, zero, zero ).m_effective == DDSPC::PolicyMode::Standard );
    policy.request( DDSPC::PolicyMode::Standard );
    REQUIRE( policy.decide( evidence, zero, zero ).m_effective == DDSPC::PolicyMode::Standard );
    REQUIRE( app.outputs_rejected() );
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief Replay trains the actual app models only after an explicit training request, with no output authority. */
TEST_CASE( "Application shared shadow pipeline produces qualified replay predictions",
           "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::configureShadowInputs;
    MagAOX::app::loPredCtrlAcc::processShadowAdc;
    MagAOX::app::loPredCtrlAcc::processShadowWfs;
    MagAOX::app::loPredCtrlAcc::processShadowApplied;
    MagAOX::app::loPredCtrlAcc::shadowCandidate;
#endif
    // clang-format on
    const bool         training = GENERATE( false, true );
    ReplayApplication  app;
    ShadowReplayConfig recording;
    recording.m_frames = 400;
    app.prepare( replay_model_config(), recording.m_inputs );
    REQUIRE_FALSE( app.model_status( DDSPC::ShadowMode::Standard ).m_trainingEnabled );
    if( training )
        app.train();
    const auto result = replay_shadow( app, recording );
    REQUIRE( result.m_capture.m_rateHz == Approx( 8000 ) );
    REQUIRE( result.m_wfsAccepted == recording.m_frames );
    std::size_t valid = 0, assisted = 0;
    for( const auto &o : app.m_observations )
    {
        const auto &s = o.m_status;
        REQUIRE( s.m_step <= .2 );
        REQUIRE( o.m_candidate.allFinite() );
        if( s.m_candidateValid )
        {
            ++valid;
            REQUIRE( s.m_commonValid );
            REQUIRE( o.m_candidate.cwiseAbs().maxCoeff() <= 3 );
            if( s.m_effective == DDSPC::PolicyMode::Assisted )
            {
                ++assisted;
                REQUIRE( s.m_featureReason == DDSPC::FeatureReason::Ready );
                REQUIRE( s.m_validStreak >= 10 );
                REQUIRE( s.m_availableNs <= s.m_originNs );
                REQUIRE( s.m_originNs - s.m_featureNs <= 500000 );
            }
        }
        else
            REQUIRE( o.m_candidate.isZero( 0 ) );
    }
    if( training )
    {
        REQUIRE( valid > 200 );
        REQUIRE( assisted > 100 );
        REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates > 300 );
        REQUIRE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_updates > 300 );
        for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        {
            REQUIRE( app.model_status( mode ).m_predictionValid );
            REQUIRE( std::isfinite( app.model_status( mode ).m_rmse ) );
        }
    }
    else
    {
        REQUIRE( valid == 0 );
        REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
        REQUIRE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
    }
    REQUIRE( app.outputs_rejected() );
    REQUIRE( app.bad_runtime_outputs_rejected() );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
    DDSPC::Matrix candidate = DDSPC::Matrix::Ones( 2, 1 );
    REQUIRE_FALSE( app.shadowCandidate( candidate ) );
    REQUIRE( candidate.isZero( 0 ) );
}

/** \brief Changing every candidate's selection/slew cannot change coefficients trained on the same actual holds. */
TEST_CASE( "Application learning is independent of shadow-generated commands", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowApplied;
    DDSPC::ShadowController::prediction_matrix;
#endif
    // clang-format on
    ReplayApplication  standard, assisted;
    ShadowReplayConfig c;
    c.m_frames = 400;
    standard.prepare( replay_model_config(), c.m_inputs, true, DDSPC::PolicyMode::Standard, .2 );
    assisted.prepare( replay_model_config(), c.m_inputs, true, DDSPC::PolicyMode::Assisted, .001 );
    standard.train();
    assisted.train();
    replay_shadow( standard, c );
    replay_shadow( assisted, c );
    bool different = false;
    for( std::size_t i = 0; i < c.m_frames; ++i )
        different = different || standard.m_observations[i].m_candidate != assisted.m_observations[i].m_candidate;
    REQUIRE( different );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
    {
        REQUIRE( standard.model( mode ) == assisted.model( mode ) );
        REQUIRE( standard.model_status( mode ).m_updates == assisted.model_status( mode ).m_updates );
    }
    REQUIRE( standard.inhibited() );
    REQUIRE( assisted.inhibited() );
    REQUIRE( standard.appShutdown() == 0 );
    REQUIRE( assisted.appShutdown() == 0 );
}

/** \brief ADC loss causes standard fallback; source restart at a window boundary cannot reuse old training history. */
TEST_CASE( "Application shared pipeline falls back and requalifies after ADC loss or restart",
           "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowAdc;
    DDSPC::ShadowController::process;
#endif
    // clang-format on
    const int          fault = GENERATE( 1, 5, 25, 100, -1 );
    ReplayApplication  app;
    ShadowReplayConfig c;
    c.m_frames = 900;
    if( fault > 0 )
    {
        c.m_adc.m_dropStart = 4 * 300 + 2;
        c.m_adc.m_dropCount = 8 * fault;
    }
    else
        c.m_restartTick = 4 * 300 + 2;
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    replay_shadow( app, c );
    bool before = false, fallback = false, recovered = false;
    for( std::size_t i = 0; i < c.m_frames; ++i )
    {
        const auto &s = app.m_observations[i].m_status;
        if( i < 300 && s.m_effective == DDSPC::PolicyMode::Assisted )
            before = true;
        if( i == 301 )
            REQUIRE( s.m_effective != DDSPC::PolicyMode::Assisted );
        if( i > 300 && s.m_effective == DDSPC::PolicyMode::Standard )
            fallback = true;
        if( i > std::size_t( 350 + 2 * std::max( fault, 0 ) ) && s.m_effective == DDSPC::PolicyMode::Assisted )
        {
            recovered = true;
            REQUIRE( s.m_validStreak >= 10 );
        }
        if( fault < 0 && i == 301 )
        {
            REQUIRE( s.m_adcGeneration == 2 );
            REQUIRE( s.m_featureReason == DDSPC::FeatureReason::Ready );
        }
    }
    REQUIRE( before );
    REQUIRE( fallback );
    REQUIRE( recovered );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief No confirmed inputs means no learning, regardless of valid ADC/WFS data or selectable proposals. */
TEST_CASE( "Application pipeline rejects driver-prepared commands", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowApplied;
#endif
    // clang-format on
    ReplayApplication  app;
    ShadowReplayConfig c;
    c.m_frames             = 200;
    c.m_rejectEveryCommand = true;
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    const auto result = replay_shadow( app, c );
    REQUIRE( result.m_commandsAccepted == 0 );
    REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
    REQUIRE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
    for( const auto &o : app.m_observations )
        REQUIRE_FALSE( o.m_status.m_candidateValid );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief Common-input loss discards candidates; continuing independent external inputs allow passive requalification.
 */
TEST_CASE( "Application pipeline recovers after WFS or applied-input loss", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::expireShadowInputs;
    DDSPC::ShadowController::invalidate;
#endif
    // clang-format on
    const bool         wfsLoss = GENERATE( false, true );
    ReplayApplication  app;
    ShadowReplayConfig c;
    c.m_frames = 700;
    if( wfsLoss )
    {
        c.m_skipWfsStart = 300;
        c.m_skipWfsCount = 20;
    }
    else
    {
        c.m_badAppliedStart = 300;
        c.m_badAppliedCount = 20;
    }
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    replay_shadow( app, c );
    bool unavailable = false, recovered = false;
    for( const auto &o : app.m_observations )
    {
        if( o.m_status.m_index >= 400 && o.m_status.m_index < 450 && !o.m_status.m_candidateValid )
            unavailable = true;
        if( o.m_status.m_index > 550 && o.m_status.m_effective == DDSPC::PolicyMode::Assisted )
            recovered = true;
    }
    REQUIRE( unavailable );
    REQUIRE( recovered );
    const auto final = app.shadowPipelineStatus();
    REQUIRE_FALSE( app.expire( final.m_originNs + c.m_inputs.m_maxAgeNs + 500000 ) );
    DDSPC::Matrix candidate = DDSPC::Matrix::Ones( 2, 1 );
    REQUIRE_FALSE( app.shadowCandidate( candidate ) );
    REQUIRE( candidate.isZero( 0 ) );
    REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_samples == 0 );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief Omitting the new flag preserves existing dispatch, while raw native callbacks cannot inherit replay trust. */
TEST_CASE( "Application pipeline is opt-in and raw callbacks remain unverified", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
    MagAOX::app::loPredCtrlAcc::processImage;
#endif
    // clang-format on
    ReplayApplication  disabled;
    ShadowReplayConfig c;
    c.m_frames = 200;
    disabled.prepare( replay_model_config(), c.m_inputs, false );
    disabled.train();
    const auto result = replay_shadow( disabled, c );
    REQUIRE( result.m_adcAccepted == 0 );
    REQUIRE_FALSE( disabled.shadowPipelineStatus().m_candidateValid );
    REQUIRE( disabled.model_status( DDSPC::ShadowMode::Standard ).m_updates > 100 );
    REQUIRE( disabled.inhibited() );
    REQUIRE( disabled.appShutdown() == 0 );
    ReplayApplication enabled;
    enabled.prepare( replay_model_config(), c.m_inputs );
    enabled.train();
    replay_shadow( enabled, c );
    REQUIRE( enabled.shadowPipelineStatus().m_decisions == c.m_frames );
    enabled.raw_wfs();
    DDSPC::Matrix candidate = DDSPC::Matrix::Ones( 2, 1 );
    REQUIRE_FALSE( enabled.shadowCandidate( candidate ) );
    REQUIRE( enabled.model_status( DDSPC::ShadowMode::Standard ).m_samples == 0 );
    REQUIRE( enabled.outputs_rejected() );
    REQUIRE( enabled.inhibited() );
    REQUIRE( enabled.appShutdown() == 0 );
}
/** \brief Wrong clocks and late origins fail closed; delayed ADC cannot inherit the20-ms transport age allowance. */
TEST_CASE( "Application shared pipeline enforces source clocks freshness and frozen origins",
           "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowAdc;
    MagAOX::app::loPredCtrlAcc::processShadowWfs;
#endif
    // clang-format on
    const int          fault = GENERATE( 0, 1, 2, 3 );
    ShadowReplayConfig c;
    c.m_frames = 600;
    if( fault == 0 )
        c.m_wrongAdcClock = true;
    if( fault == 1 )
        c.m_lateWfsIndex = 300;
    if( fault == 2 )
        c.m_adc.m_latencyNs = 2000000;
    if( fault == 3 )
    {
        c.m_adc.m_jitterNs      = 10000;
        c.m_adc.m_clockDriftPpm = 200;
    }
    ReplayApplication app;
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    replay_shadow( app, c );
    if( fault == 0 || fault == 2 )
    {
        for( const auto &o : app.m_observations )
            REQUIRE( o.m_status.m_effective != DDSPC::PolicyMode::Assisted );
        REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates > 500 );
        REQUIRE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
    }
    if( fault == 1 )
    {
        REQUIRE_FALSE( app.m_observations[300].m_status.m_candidateValid );
        REQUIRE( std::string( app.m_observations[300].m_status.m_reason ) == "missed_origin_deadline" );
    }
    if( fault == 1 || fault == 3 )
        REQUIRE( std::any_of( app.m_observations.begin() + 400,
                              app.m_observations.end(),
                              []( const auto &o ) { return o.m_status.m_effective == DDSPC::PolicyMode::Assisted; } ) );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}

/** \brief Model dimensions are configurable; injected prebuilt WFS features never replace actual decoded ADC values. */
TEST_CASE( "Application replay ignores caller ADC features and supports additional control modes",
           "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowWfs;
    DDSPC::ShadowController::prediction_matrix;
#endif
    // clang-format on
    ShadowReplayConfig c;
    c.m_frames     = 400;
    auto models    = replay_model_config();
    models.m_modes = c.m_inputs.m_modes = 3;
    ReplayApplication clean, forged;
    clean.prepare( models, c.m_inputs );
    forged.prepare( models, c.m_inputs );
    clean.train();
    forged.train();
    forged.m_forgeFeatures = true;
    replay_shadow( clean, c );
    replay_shadow( forged, c );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        REQUIRE( clean.model( mode ) == forged.model( mode ) );
    REQUIRE( std::any_of( clean.m_observations.begin(),
                          clean.m_observations.end(),
                          []( const auto &o ) { return o.m_status.m_candidateValid; } ) );
    REQUIRE( clean.inhibited() );
    REQUIRE( forged.inhibited() );
    REQUIRE( clean.appShutdown() == 0 );
    REQUIRE( forged.appShutdown() == 0 );
}

/** \brief Invalid opt-in configuration fails before any output/helper activity; absence preserves the old path. */
TEST_CASE( "Application shared pipeline startup requires shadow and explicit bounds", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
#endif
    // clang-format on
    for( const auto &invalid :
         { std::map<std::string, std::string>{ { "parameters.shadow", "0" }, { "parameters.shadow_pipeline", "1" } },
           { { "parameters.shadow", "1" }, { "parameters.shadow_pipeline", "1" } },
           { { "parameters.shadow", "1" },
             { "parameters.shadow_pipeline", "1" },
             { "shadow_pipeline.command_limit", "3" },
             { "shadow_pipeline.max_step", ".2" },
             { "shadow_pipeline.max_age_ns", "0" } } } )
    {
        ReplayApplication app;
        REQUIRE( app.load_values( invalid ) < 0 );
        REQUIRE( app.appStartup() < 0 );
        REQUIRE( app.inhibited() );
    }
}

/** \brief Application snapshot/INDI reads are serialized with replay without holding framework locks during processing.
 */
TEST_CASE( "Application replay and diagnostic readers are thread-safe", "[loPredCtrlAcc][app][pipeline][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::shadowPipelineStatus;
    MagAOX::app::loPredCtrlAcc::shadowCandidate;
    MagAOX::app::loPredCtrlAcc::updateShadowINDI;
#endif
    // clang-format on
    ReplayApplication  app;
    ShadowReplayConfig c;
    c.m_frames = 600;
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    app.m_trace = false;
    std::atomic<bool>  done{ false };
    std::exception_ptr failure;
    std::thread        frontend(
        [&]
        {
            try
            {
                replay_shadow( app, c );
            }
            catch( ... )
            {
                failure = std::current_exception();
            }
            done.store( true, std::memory_order_release );
        } );
    DDSPC::Matrix candidate = DDSPC::Matrix::Zero( 2, 1 );
    bool          finite    = true;
    while( !done.load( std::memory_order_acquire ) )
    {
        app.shadowPipelineStatus();
        app.shadowCandidate( candidate );
        finite = finite && candidate.allFinite();
        app.appLogic();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    frontend.join();
    if( failure )
        std::rethrow_exception( failure );
    REQUIRE( finite );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}
/** \brief Changed WFS incarnations/clock mappings and time reversal cannot reuse the previous session's ready models.
 */
TEST_CASE( "Application shared pipeline rejects changed WFS source contracts", "[loPredCtrlAcc][app][pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowWfs;
    DDSPC::CausalInputBuilder::push_wfs;
#endif
    // clang-format on
    const int          fault = GENERATE( 0, 1, 2 );
    ShadowReplayConfig c;
    c.m_frames = 300;
    ReplayApplication app;
    app.prepare( replay_model_config(), c.m_inputs );
    app.train();
    replay_shadow( app, c );
    const auto            before = app.shadowPipelineStatus();
    DDSPC::TimedWfsRecord w;
    w.m_stamp.m_generation    = c.m_inputs.m_wfsGeneration + ( fault == 0 ? 1 : 0 );
    w.m_stamp.m_clockEpoch    = c.m_inputs.m_clockEpoch + ( fault == 1 ? 1 : 0 );
    w.m_stamp.m_layout        = c.m_inputs.m_layout;
    w.m_stamp.m_counter       = before.m_index + 1;
    w.m_decisionNs            = before.m_originNs + ( fault == 2 ? -1 : 500000 );
    w.m_exposureStartNs       = w.m_decisionNs - 250000;
    w.m_exposureEndNs         = w.m_exposureStartNs + 200000;
    w.m_stamp.m_availableNs   = w.m_exposureEndNs + 10000;
    w.m_stamp.m_receivedNs    = w.m_exposureEndNs + 20000;
    w.m_stamp.m_snapshotValid = true;
    w.m_residual              = DDSPC::Matrix::Zero( 2, 1 );
    REQUIRE_FALSE( app.wfs( w, w.m_decisionNs ) );
    DDSPC::Matrix candidate = DDSPC::Matrix::Ones( 2, 1 );
    REQUIRE_FALSE( app.shadowCandidate( candidate ) );
    REQUIRE( candidate.isZero( 0 ) );
    REQUIRE_FALSE( app.model_status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( app.outputs_rejected() );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}
} // namespace libXWCTest::loPredCtrlAccTest
