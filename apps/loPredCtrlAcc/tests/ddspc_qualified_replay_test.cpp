/** \file ddspc_qualified_replay_test.cpp
 * \brief Actual ADC/native-DDSPC matched forecasts, future-target isolation and qualification regression tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "qualified_shadow_replay.hpp"
#include "../online_shadow.hpp"
#include "../synchronized_inputs.hpp"
#include <map>
#include <tuple>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond QUALIFIED_REPLAY_TEST_SUPPORT
namespace
{
/// Compare both predictors on identical measured targets, excluding their unequal warm-up intervals.
std::array<double, 2> rmse( const QualifiedReplayResult &result /**< [in] Actual native traces. */ )
{
    std::map<std::tuple<std::uint64_t, unsigned>, std::array<double, 2>> error;
    std::map<std::tuple<std::uint64_t, unsigned>, unsigned>              seen;
    for( const auto &s : result.m_scores )
        if( s.m_origin > 600 && s.m_horizon == 1 )
        {
            auto key                         = std::make_tuple( s.m_origin, s.m_channel );
            error[key][unsigned( s.m_mode )] = s.m_prediction - s.m_measurement;
            seen[key] |= 1u << unsigned( s.m_mode );
        }
    std::array<double, 2> energy{};
    unsigned              count = 0;
    for( const auto &[key, e] : error )
        if( seen[key] == 3 )
        {
            ++count;
            for( unsigned m = 0; m < 2; ++m )
                energy[m] += e[m] * e[m];
        }
    if( !count )
        throw std::runtime_error( "No common native forecast targets" );
    for( auto &e : energy )
        e = std::sqrt( e / count );
    return energy;
}
} // namespace
/// \endcond

/** \brief Every emitted horizon uses original counters and identical real measured WFS targets. */
TEST_CASE( "Qualified native replay trains both RLS and QR models at all lifted horizons",
           "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::ShadowController::prediction_sink;
    DDSPC::AdcCapture::drain;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames   = 1000;
    c.m_future   = GENERATE( 1u, 2u, 4u, 8u );
    c.m_qrd      = GENERATE( false, true );
    const auto r = replay_qualified( c );
    REQUIRE( r.m_adc.m_samples == 4 * c.m_frames );
    REQUIRE( r.m_adc.m_rateHz == Approx( 8000 ) );
    REQUIRE( r.m_adc.m_invalid == 0 );
    REQUIRE( r.m_adc.m_dropped == 0 );
    REQUIRE( r.m_status.m_standard.m_ready );
    REQUIRE( r.m_status.m_assisted.m_ready );
    REQUIRE( r.m_status.m_standard.m_rank == r.m_status.m_standard.m_features );
    REQUIRE( r.m_status.m_assisted.m_rank == r.m_status.m_assisted.m_features );
    REQUIRE( r.m_standard.allFinite() );
    REQUIRE( r.m_assisted.allFinite() );
    REQUIRE( r.m_status.m_forecasts > 500 );
    REQUIRE( r.m_candidates == 0 );
    REQUIRE_FALSE( r.m_status.m_exposure );
    REQUIRE_FALSE( r.m_status.m_clock );
    REQUIRE( r.m_status.m_inputEvidence == DDSPC::InputEvidence::Requested );
    REQUIRE( r.m_scoreDrops == 0 );
    REQUIRE_FALSE( r.m_scores.empty() );
    for( const auto &s : r.m_scores )
    {
        REQUIRE( s.m_target == s.m_origin + s.m_horizon );
        REQUIRE( s.m_horizon > 0 );
        REQUIRE( s.m_horizon <= c.m_future );
        REQUIRE( s.m_channel < 2 );
        REQUIRE( s.m_originValid );
        REQUIRE( std::isfinite( s.m_originPrediction ) );
        REQUIRE( std::isfinite( s.m_prediction ) );
    }
    if( c.m_future == 1 )
    {
        const auto e = rmse( r );
        REQUIRE( e[1] < .7 * e[0] );
        for( unsigned a = 0; a < 2; ++a )
            for( unsigned s = 0; s < 4; ++s )
                REQUIRE( r.m_assisted( a, 12 + 2 * s + a ) == Approx( .2 ).margin( .06 ) );
    }
}

/** \brief Unrelated ADC cannot modify the standard fit or its targets; duplicate channels cannot qualify rank. */
TEST_CASE( "Qualified replay keeps model independence and rejects deficient ADC excitation",
           "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames           = 1200;
    const auto original  = replay_qualified( c );
    c.m_uncorrelated     = true;
    const auto unrelated = replay_qualified( c );
    REQUIRE( original.m_standard == unrelated.m_standard );
    REQUIRE( original.m_status.m_standard.m_updates == unrelated.m_status.m_standard.m_updates );
    REQUIRE( rmse( original )[1] < .7 * rmse( unrelated )[1] );
    c.m_duplicateChannels = true;
    const auto duplicate  = replay_qualified( c );
    REQUIRE( duplicate.m_status.m_standard.m_ready );
    REQUIRE_FALSE( duplicate.m_status.m_assisted.m_ready );
    REQUIRE( duplicate.m_status.m_assisted.m_rank < duplicate.m_status.m_assisted.m_features );
}

/** \brief Changing only future WFS payloads changes future truth, never a previously frozen conditional forecast. */
TEST_CASE( "Native frozen predictions cannot learn from future target measurements", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames          = 1000;
    c.m_future          = 4;
    const auto baseline = replay_qualified( c );
    c.m_changeWfsAt     = 700;
    const auto changed  = replay_qualified( c );
    using Key           = std::tuple<unsigned, std::uint64_t, unsigned, unsigned>;
    std::map<Key, DDSPC::ShadowPrediction> original;
    for( const auto &s : baseline.m_scores )
        if( s.m_origin < 700 && s.m_target >= 700 )
            original[{ unsigned( s.m_mode ), s.m_origin, s.m_horizon, s.m_channel }] = s;
    unsigned tested = 0;
    for( const auto &s : changed.m_scores )
        if( s.m_origin < 700 && s.m_target >= 700 )
        {
            const auto &before = original.at( { unsigned( s.m_mode ), s.m_origin, s.m_horizon, s.m_channel } );
            REQUIRE( s.m_prediction == before.m_prediction );
            REQUIRE( s.m_measurement - before.m_measurement == Approx( 1.5 ).margin( 1e-5 ) );
            ++tested;
        }
    REQUIRE( tested > 0 );
}

/** \brief Changing future actual inputs changes conditional scoring but cannot change an origin-only forecast. */
TEST_CASE( "Native origin-only forecasts never use future requested commands", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::freeze_next;
    DDSPC::ShadowController::available_forecast;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames          = 1000;
    c.m_future          = 4;
    const auto baseline = replay_qualified( c );
    c.m_changeCommandAt = 700;
    const auto changed  = replay_qualified( c );
    using Key           = std::tuple<unsigned, std::uint64_t, unsigned, unsigned>;
    std::map<Key, DDSPC::ShadowPrediction> original;
    for( const auto &s : baseline.m_scores )
        if( s.m_origin < 700 && s.m_target > 700 )
            original[{ unsigned( s.m_mode ), s.m_origin, s.m_horizon, s.m_channel }] = s;
    unsigned tested = 0, conditionalChanged = 0;
    for( const auto &s : changed.m_scores )
        if( s.m_origin < 700 && s.m_target > 700 )
        {
            const auto &before = original.at( { unsigned( s.m_mode ), s.m_origin, s.m_horizon, s.m_channel } );
            REQUIRE( s.m_originValid );
            REQUIRE( s.m_originPrediction == before.m_originPrediction );
            conditionalChanged += std::abs( s.m_prediction - before.m_prediction ) > 1e-5;
            ++tested;
        }
    REQUIRE( tested > 0 );
    REQUIRE( conditionalChanged > 0 );
}

/** \brief Missing exposure or independent applied-input evidence prevents both models from learning. */
TEST_CASE( "Qualified replay never upgrades missing producer evidence", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames           = 200;
    c.m_upstream         = false;
    c.m_exposureEvidence = GENERATE( true, false );
    c.m_appliedEvidence  = !c.m_exposureEvidence;
    const auto r         = replay_qualified( c );
    REQUIRE( r.m_status.m_standard.m_updates == 0 );
    REQUIRE( r.m_status.m_assisted.m_updates == 0 );
    REQUIRE( r.m_scores.empty() );
    REQUIRE( r.m_candidates == 0 );
}

/** \brief Timer mode, wrong counter phase and future delivery inhibit assistance while preserving standard DDSPC. */
TEST_CASE( "Upstream sync is observed rather than assumed or reconstructed", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames      = 800;
    const int fault = GENERATE( 0, 1, 2, 3 );
    c.m_timerMode   = fault == 0;
    c.m_wrongPhase  = fault == 1;
    c.m_futureAdc   = fault == 2;
    c.m_noAdcAnchor = fault == 3;
    const auto r    = replay_qualified( c );
    REQUIRE( r.m_status.m_standard.m_ready );
    REQUIRE( r.m_status.m_standard.m_predictions > 400 );
    REQUIRE_FALSE( r.m_status.m_assisted.m_ready );
    REQUIRE( r.m_status.m_assisted.m_updates == 0 );
    REQUIRE_FALSE( r.m_status.m_candidateValid );
}

/** \brief Losing upstream lock revokes only assistance, then reacquires consecutive complete groups and fit readiness.
 */
TEST_CASE( "Upstream loss and recovery preserve the independent standard model", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames           = 1000;
    const auto reference = replay_qualified( c );
    c.m_syncLossStart    = 500;
    c.m_syncLossCount    = 50;
    const auto fault     = replay_qualified( c );
    REQUIRE( reference.m_standard == fault.m_standard );
    REQUIRE( fault.m_readiness[480][1] == 1 );
    REQUIRE( fault.m_readiness[510][0] == 1 );
    REQUIRE( fault.m_readiness[510][1] == 0 );
    REQUIRE( fault.m_readiness[554][1] == 0 );
    REQUIRE( fault.m_readiness[700][1] == 1 );
    REQUIRE( fault.m_status.m_assisted.m_ready );
}

/** \brief Optional worker scores are disabled/empty before qualification; all storage bounds are checked first. */
TEST_CASE( "Native diagnostic score transport cannot grant training authority", "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    DDSPC::ShadowPredictionQueue queue( 2 );
    DDSPC::ShadowPrediction      sample;
    sample.m_origin = 10;
    REQUIRE( queue.push( sample ) );
    sample.m_origin = 11;
    REQUIRE( queue.push( sample ) );
    REQUIRE_FALSE( queue.push( sample ) );
    REQUIRE( queue.queued() == 2 );
    REQUIRE( queue.drops() == 1 );
    REQUIRE( queue.pop( sample ) );
    REQUIRE( sample.m_origin == 10 );
    REQUIRE( queue.pop( sample ) );
    REQUIRE( sample.m_origin == 11 );
    REQUIRE_FALSE( queue.pop( sample ) );
    queue.reset();
    REQUIRE( queue.drops() == 0 );
    REQUIRE( queue.queued() == 0 );
    DDSPC::OnlineShadowConfig c;
    c.m_model.m_accelHistory = 1;
    c.m_predictionCapacity   = 4;
    DDSPC::OnlineShadowWorker worker( c );
    DDSPC::ShadowPrediction   s;
    REQUIRE_FALSE( worker.read_prediction( s ) );
    REQUIRE( worker.status().m_predictionDrops == 0 );
    REQUIRE_FALSE( worker.qualify( {} ) );
    c.m_predictionCapacity = 65537;
    REQUIRE_THROWS_AS( DDSPC::OnlineShadowWorker( c ), std::invalid_argument );
}
/** \brief Real adapter functions project whole snapshots, hold idle requests, and construct no physical exposure. */
TEST_CASE( "Synchronized native adapters preserve snapshots and require causal complete histories",
           "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    DDSPC::MultirateContract c;
    c.m_upstream.m_enabled = c.m_upstream.m_combinedRequestModel = true;
    c.m_upstream.m_epoch                                         = 77;
    c.m_upstream.m_modalAnchor                                   = 1;
    c.m_upstream.m_adcAnchor                                     = 10;
    c.m_inputs.m_modes                                           = 2;
    c.m_inputs.m_mappingVerified                                 = true;
    c.m_inputs.m_evidence                                        = DDSPC::InputEvidence::Requested;
    c.m_inputs.m_commandGeneration                               = 3;
    c.m_inputs.m_layout = c.m_inputs.m_clockEpoch = 1;
    c.m_sources[3]                                = { 1, 100, 1000, 1 };
    c.m_sources[1]                                = { 1, 101, 1001, 1 };
    DDSPC::NativeObservation dm;
    dm.m_source     = DDSPC::ObservationSource::Dm;
    dm.m_device     = 1;
    dm.m_fileInode  = 100;
    dm.m_imageInode = 1000;
    dm.m_generation = 1;
    dm.m_counter    = 10;
    dm.m_receivedNs = dm.m_availableNs = 100;
    dm.m_elements                      = 3;
    const std::array<float, 3> pixels{ 1, 2, 3 };
    dm.m_payloadHash = DDSPC::command_payload_hash( pixels );
    DDSPC::Matrix p( 2, 3 );
    p << 1, 0, -1, 0, .5, 0;
    std::array<DDSPC::ProjectedRequest, 2> history;
    REQUIRE( DDSPC::project_request( c, dm, pixels, p, history[0] ) );
    REQUIRE( history[0].m_values[0] == -2 );
    REQUIRE( history[0].m_values[1] == 1 );
    ++dm.m_payloadHash;
    REQUIRE_FALSE( DDSPC::project_request( c, dm, pixels, p, history[1] ) );
    --dm.m_payloadHash;
    dm.m_counter    = 11;
    dm.m_receivedNs = dm.m_availableNs = 200;
    REQUIRE( DDSPC::project_request( c, dm, pixels, p, history[1] ) );
    DDSPC::NativeObservation wfs = dm;
    wfs.m_source                 = DDSPC::ObservationSource::Modal;
    wfs.m_fileInode              = 101;
    wfs.m_imageInode             = 1001;
    wfs.m_counter                = 2;
    wfs.m_availableNs            = 300;
    wfs.m_receivedNs             = 301;
    wfs.m_values                 = { .2, .3 };
    DDSPC::OnlineCommandFrame frame;
    REQUIRE( DDSPC::synchronized_commands( c, 1, 1, history, { 110, 190, 210, 290 }, wfs, frame ) );
    REQUIRE( frame.m_counters == std::array<std::uint64_t, 4>{ 10, 10, 11, 11 } );
    REQUIRE( frame.m_evidence == DDSPC::InputEvidence::Requested );
    REQUIRE_FALSE( DDSPC::synchronized_commands( c, 1, 1, history, { 90, 190, 210, 290 }, wfs, frame ) );
    REQUIRE_FALSE( DDSPC::synchronized_commands( c, 1, 1, history, { 110, 190, 210, 400 }, wfs, frame ) );
    DDSPC::OnlineOrigin origin;
    REQUIRE( DDSPC::synchronized_origin( c, { 77, true, true, 310 }, wfs, 350, -1, origin ) );
    REQUIRE( origin.m_firstAdcCounter == 14 );
    REQUIRE( origin.m_exposureStartNs == -1 );
    REQUIRE( origin.m_exposureEndNs == -1 );
    REQUIRE( origin.m_cameraCounter == 0 );
    REQUIRE( origin.m_stamp.m_availableNs == wfs.m_availableNs );
    REQUIRE_FALSE( DDSPC::synchronized_origin( c, { 77, true, true, 310 }, wfs, 200, -1, origin ) );
}

/** \brief Knowing synchronization never substitutes for independent command inputs in either model. */
TEST_CASE( "Upstream synchronized models still require complete requested command evidence",
           "[multirate][qualified-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::commands;
    DDSPC::ShadowController::freeze_next;
#endif
    // clang-format on
    QualifiedReplayConfig c;
    c.m_frames          = 300;
    c.m_appliedEvidence = false;
    const auto r        = replay_qualified( c );
    REQUIRE( r.m_status.m_standard.m_updates == 0 );
    REQUIRE( r.m_status.m_assisted.m_updates == 0 );
    REQUIRE( r.m_scores.empty() );
}
} // namespace libXWCTest::loPredCtrlAccTest
