/** \file ddspc_adc_ddspc_test.cpp
 * \brief Actual two-model DDSPC feedback driven by decoded8-kHz raw ADC data, never a live interface.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "offline_ddspc.hpp"
#include <algorithm>
#include <cmath>

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Actual QR fits converge and produce finite conditional forecasts and simulated residual suppression. */
TEST_CASE( "Offline raw ADC drives independent standard and assisted DDSPC", "[adc-ddspc][offline-simulation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig config;
    const auto     result = adc_ddspc_run( config );
    REQUIRE( result.m_runs[0].m_variance > 0 );
    for( int mode = 1; mode < 3; ++mode )
    {
        const auto &run = result.m_runs[mode];
        INFO( "model=" << mode << " variance=" << run.m_variance << " open=" << result.m_runs[0].m_variance );
        REQUIRE( run.m_firstReady >= 0 );
        REQUIRE( run.m_firstReady < config.m_train );
        REQUIRE( run.m_updates > 1000 );
        REQUIRE( run.m_commands > std::uint64_t( config.m_evaluate / 2 ) );
        REQUIRE( run.m_predictions > 1000 );
        REQUIRE( std::isfinite( run.m_predictionRmse ) );
        REQUIRE( run.m_predictionRmse > 0 );
        REQUIRE( run.m_thetaNorm > 0 );
        REQUIRE( run.m_faults == 0 );
        REQUIRE( run.m_gateViolations == 0 );
        REQUIRE( run.m_maxCommand <= config.m_policy.m_limit );
        REQUIRE( run.m_variance < result.m_runs[0].m_variance );
        REQUIRE( run.m_adcStatus.m_samples == std::uint64_t( 4 * ( config.m_train + config.m_evaluate ) ) );
        REQUIRE( run.m_adcStatus.m_rateHz == Approx( 8000 ) );
        REQUIRE_FALSE( run.m_learning.empty() );
        REQUIRE( run.m_learning.back().m_status.m_updates == run.m_updates );
        REQUIRE( run.m_learning.back().m_index == config.m_train + config.m_evaluate - 1 );
    }
    REQUIRE( result.m_runs[2].m_predictionRmse < result.m_runs[1].m_predictionRmse );
}

/** \brief Missing ADC evidence blocks assisted decisions and both fits recover without a nonfinite output. */
TEST_CASE( "Offline ADC-DDSPC recovers after sensor loss and jitter", "[adc-ddspc][offline-simulation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig config;
    config.m_train             = 1200;
    config.m_evaluate          = 1800;
    config.m_adc.m_jitterNs    = 10000;
    config.m_adc.m_sensorNoise = .015;
    config.m_adc.m_dropStart   = 4 * ( config.m_train + 400 );
    config.m_adc.m_dropCount   = 200;
    const auto result          = adc_ddspc_run( config );
    for( int mode = 1; mode < 3; ++mode )
    {
        const auto &run = result.m_runs[mode];
        REQUIRE( run.m_faults == 0 );
        REQUIRE( run.m_gateViolations == 0 );
        REQUIRE( run.m_adcStatus.m_dropped == 200 );
        REQUIRE( run.m_blocked >= 50 );
        REQUIRE( run.m_lastReady > config.m_train + 600 );
        REQUIRE( run.m_maxCommand <= config.m_policy.m_limit );
        REQUIRE( std::isfinite( run.m_variance ) );
    }
}

/** \brief Uncoupled acceleration does not imply a benefit and calibration/probing transients stay in the metrics. */
TEST_CASE( "Offline uncorrelated ADC does not imply a residual improvement", "[adc-ddspc][offline-simulation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig config;
    config.m_train    = 800;
    config.m_evaluate = 1000;
    config.m_coupling = 0;
    const auto result = adc_ddspc_run( config );
    REQUIRE( result.m_runs[1].m_faults == 0 );
    REQUIRE( result.m_runs[2].m_faults == 0 );
    REQUIRE( result.m_runs[2].m_variance == Approx( result.m_runs[1].m_variance ).epsilon( .05 ) );
    // Declared evaluation probing can itself disturb an otherwise quiet plant; it is not silently subtracted.
    REQUIRE( result.m_runs[1].m_variance > result.m_runs[0].m_variance );
}

/** \brief Persistent loss cannot manufacture a trained assisted model while the standard learner continues. */
TEST_CASE( "Offline repeated ADC gaps keep assisted prediction unavailable", "[adc-ddspc][offline-simulation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig config;
    config.m_train           = 400;
    config.m_evaluate        = 1000;
    config.m_adc.m_dropEvery = 8;
    const auto result        = adc_ddspc_run( config );
    REQUIRE( result.m_runs[1].m_commands > 0 );
    REQUIRE( result.m_runs[2].m_commands > 0 );
    REQUIRE( result.m_runs[2].m_fallbackFrames > 0 );
    REQUIRE( result.m_runs[2].m_updates == 0 );
    REQUIRE( result.m_runs[2].m_lastReady == -1 );
    REQUIRE( result.m_runs[2].m_gateViolations == 0 );
    REQUIRE( result.m_runs[2].m_adcStatus.m_dropped > 0 );
}

/** \brief Horizon, coupling, learner and plant sign are configurable and preserve finite bounded simulation. */
TEST_CASE( "Offline ADC-DDSPC accepts alternate causal model and plant settings", "[adc-ddspc][offline-simulation]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig config;
    config.m_train               = 800;
    config.m_evaluate            = 1000;
    config.m_controller.m_future = 1;
    config.m_controller.m_useQrd = false;
    config.m_coupling            = .3;
    config.m_dmGain              = -.8;
    const auto result            = adc_ddspc_run( config );
    for( const auto &run : result.m_runs )
    {
        REQUIRE( std::isfinite( run.m_variance ) );
        REQUIRE( run.m_maxCommand <= config.m_policy.m_limit );
    }
    REQUIRE( result.m_runs[1].m_updates > 500 );
    REQUIRE( result.m_runs[2].m_updates > 500 );
    config.m_controller.m_future = 0;
    REQUIRE_THROWS_AS( adc_ddspc_run( config ), std::invalid_argument );
}
/** \brief All specified burst lengths trigger immediate fallback and require fresh trained history plus recovery dwell.
 */
TEST_CASE( "Offline DDSPC falls back through 1 5 25 and 100 ms ADC loss", "[adc-ddspc][fallback]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    const int      lossMs = GENERATE( 1, 5, 25, 100 );
    AdcDdspcConfig c;
    c.m_train                   = 1200;
    c.m_evaluate                = 1800;
    c.m_policy.m_recoveryFrames = 15;
    const int start             = c.m_train + 400;
    c.m_adc.m_dropStart         = 4 * start;
    c.m_adc.m_dropCount         = 8 * lossMs;
    const auto  result          = adc_ddspc_run( c );
    const auto &assisted        = result.m_runs[2];
    REQUIRE( assisted.m_fallbackFrames >= unsigned( 2 * lossMs ) );
    REQUIRE( assisted.m_gateViolations == 0 );
    REQUIRE( assisted.m_faults == 0 );
    REQUIRE( assisted.m_adcStatus.m_dropped == unsigned( 8 * lossMs ) );
    REQUIRE( assisted.m_maxSwitchStep <= c.m_policy.m_maxStep );
    REQUIRE( assisted.m_longestFallback >= unsigned( 2 * lossMs ) );
    bool recovered = false;
    for( const auto &d : assisted.m_decisions )
    {
        REQUIRE( d.m_control.m_step <= c.m_policy.m_maxStep );
        if( d.m_index >= start && d.m_index < start + 2 * lossMs )
        {
            REQUIRE( d.m_control.m_effective != OfflineMode::Assisted );
            if( d.m_standardReady )
                REQUIRE( d.m_control.m_effective == OfflineMode::Standard );
        }
        if( d.m_index >= start + 2 * lossMs && d.m_control.m_effective == OfflineMode::Assisted )
        {
            recovered = true;
            REQUIRE( d.m_featuresValid );
            REQUIRE( d.m_assistedReady );
            REQUIRE( d.m_control.m_validStreak >= c.m_policy.m_recoveryFrames );
        }
    }
    REQUIRE( recovered );
}

/** \brief Future losses cannot alter earlier decisions; both feedback plants receive the same exogenous forcing. */
TEST_CASE( "Offline policies share disturbances and never inspect future dropout events", "[adc-ddspc][causality]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig c;
    c.m_train           = 800;
    c.m_evaluate        = 1000;
    const auto normal   = adc_ddspc_run( c );
    c.m_adc.m_dropStart = 4 * ( c.m_train + 400 );
    c.m_adc.m_dropCount = 200;
    const auto loss     = adc_ddspc_run( c );
    REQUIRE( normal.m_runs[0].m_residual == loss.m_runs[0].m_residual );
    REQUIRE( normal.m_runs[1].m_residual == loss.m_runs[1].m_residual );
    for( int i = 0; i < 400; ++i )
    {
        REQUIRE( normal.m_runs[2].m_decisions[i].m_control.m_command ==
                 loss.m_runs[2].m_decisions[i].m_control.m_command );
    }
    for( int mode = 1; mode < 3; ++mode )
    {
        for( int i = 32; i < c.m_evaluate; ++i )
        {
            for( int axis = 0; axis < 2; ++axis )
            {
                const double disturbance =
                    normal.m_runs[mode].m_residual[axis][i] -
                    c.m_dmGain * normal.m_runs[mode].m_decisions[i - 1].m_control.m_command( axis );
                REQUIRE( disturbance == Approx( normal.m_runs[0].m_residual[axis][i] ).margin( 2e-6 ) );
            }
        }
    }
}

/** \brief Independent ADC oscillator drift, jitter and variable FIFO delivery are checked at each WFS deadline. */
TEST_CASE( "Offline timestamp windows tolerate independent cadence and detect latency loss", "[adc-ddspc][timing]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig c;
    c.m_train                 = 800;
    c.m_evaluate              = 1000;
    c.m_adc.m_clockDriftPpm   = GENERATE( -200., 200. );
    c.m_adc.m_jitterNs        = 10000;
    c.m_adc.m_latencyNs       = 100000;
    c.m_adc.m_latencyJitterNs = 80000;
    const auto  result        = adc_ddspc_run( c );
    std::size_t valid         = 0;
    for( const auto &d : result.m_runs[2].m_decisions )
    {
        if( d.m_feature.m_reason == FeatureReason::Ready )
        {
            ++valid;
            REQUIRE( d.m_feature.m_availableNs <= d.m_decisionNs );
            REQUIRE( d.m_feature.m_latestNs < d.m_decisionNs );
            REQUIRE( d.m_decisionNs - d.m_feature.m_effectiveNs <= c.m_features.m_maxAgeNs );
            REQUIRE( d.m_feature.m_value.allFinite() );
        }
        else
            REQUIRE( d.m_control.m_effective != OfflineMode::Assisted );
    }
    REQUIRE( valid > 100 );
    REQUIRE( result.m_runs[2].m_blocked > 0 );
    REQUIRE( result.m_runs[2].m_fallbackFrames > 0 );
    REQUIRE( result.m_runs[2].m_gateViolations == 0 );
}

/** \brief Repeated bursts prevent unsafe re-entry, while loss of common WFS evidence uses the explicit safe ramp. */
TEST_CASE( "Offline intermittent ADC and WFS losses preserve fallback and safe output", "[adc-ddspc][fallback]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig c;
    c.m_train            = 800;
    c.m_evaluate         = 1000;
    c.m_adc.m_dropStart  = 4 * ( c.m_train + 100 );
    c.m_adc.m_dropCount  = 8;
    c.m_adc.m_dropPeriod = 64;
    c.m_wfsInvalidStart  = c.m_train + 300;
    c.m_wfsInvalidCount  = 10;
    const auto result    = adc_ddspc_run( c );
    for( int mode = 1; mode < 3; ++mode )
    {
        const auto &run = result.m_runs[mode];
        REQUIRE( run.m_faults == 0 );
        REQUIRE( run.m_gateViolations == 0 );
        for( const auto &d : run.m_decisions )
        {
            REQUIRE( d.m_control.m_step <= c.m_policy.m_maxStep );
            if( d.m_index >= c.m_wfsInvalidStart && d.m_index < c.m_wfsInvalidStart + c.m_wfsInvalidCount )
                REQUIRE( d.m_control.m_effective == OfflineMode::Safe );
            if( mode == 2 && d.m_index > c.m_train + 120 )
                REQUIRE( d.m_control.m_effective != OfflineMode::Assisted );
        }
    }
}
/** \brief Transport can report healthy20-ms data while the stricter500-us feature contract refuses every window. */
TEST_CASE( "Offline stale ADC falls back despite nominal capture health", "[adc-ddspc][timing]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::AdcCapture::status;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    AdcDdspcConfig c;
    c.m_train           = 800;
    c.m_evaluate        = 1000;
    c.m_adc.m_latencyNs = 2000000;
    const auto result   = adc_ddspc_run( c );
    REQUIRE( result.m_runs[2].m_adcStatus.m_valid );
    REQUIRE( result.m_runs[2].m_adcStatus.m_ageUs < 20000 );
    REQUIRE( result.m_runs[2].m_blocked == unsigned( c.m_evaluate ) );
    REQUIRE( result.m_runs[2].m_residual == result.m_runs[1].m_residual );
    for( const auto &d : result.m_runs[2].m_decisions )
    {
        REQUIRE( d.m_feature.m_reason == FeatureReason::Stale );
        REQUIRE( d.m_control.m_effective != OfflineMode::Assisted );
    }
}
} // namespace libXWCTest::loPredCtrlAccTest
