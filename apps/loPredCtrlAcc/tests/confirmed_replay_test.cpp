/** \file confirmed_replay_test.cpp
 * \brief Actual accelSim-model replay into the guarded application with independent confirmed inputs.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "confirmed_replay.hpp"
#include "shadow_app_test_support.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

#ifndef DDSPC_SHADOW_ONLY
    #error "Confirmed replay tests must compile the real application with DDSPC_SHADOW_ONLY"
#endif

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
namespace
{
/// Freeze both actual learners at a known exposure, retaining their measured histories and cached predictors.
class FrozenApplication : public ReplayApplication
{
  public:
    std::uint64_t m_freezeCounter{ 500 }; ///< WFS frame 100+400: subsequent records cannot alter coefficients.

    /// Stop training before this origin and enter the real application's WFS integration.
    bool wfs( const DDSPC::TimedWfsRecord &record /**< [in] Immutable WFS input. */,
              std::int64_t                 nowNs /**< [in] Frozen origin. */ ) override;
};
bool FrozenApplication::wfs( const DDSPC::TimedWfsRecord &record, std::int64_t nowNs )
{
    if( record.m_stamp.m_counter == m_freezeCounter )
        train( false, false );
    return ReplayApplication::wfs( record, nowNs );
}

/// Inspect owned ADC evidence before the real app; observations cannot feed the producer or input certificates.
class AuditedApplication : public ReplayApplication
{
  public:
    std::size_t      m_sinceOrigin{ 0 }, m_maxBatch{ 0 }; ///< Actual valid records delivered between WFS origins.
    DDSPC::AdcSample m_last;                ///< Original producer/decoder evidence, preserved across forwarding.
    std::int64_t     m_maxDeliveryAge{ 0 }; ///< Frontend age, separate from original decoder receipt.

    /// Check original timestamps, decoded values and chronological identity before actual application admission.
    bool adc( const DDSPC::ShadowAdcRecord &record /**< [in] Owned decoder output or explicit invalidation. */,
              std::int64_t                  nowNs /**< [in] Frontend availability, never backdated. */ ) override;

    /// Record the dispatch budget and invoke the actual application's WFS integration.
    bool wfs( const DDSPC::TimedWfsRecord &record /**< [in] Independently measured exposure. */,
              std::int64_t                 nowNs /**< [in] Frozen origin time. */ ) override;
};

bool AuditedApplication::adc( const DDSPC::ShadowAdcRecord &record, std::int64_t nowNs )
{
    if( record.m_snapshotValid )
    {
        const auto &s = record.m_sample;
        REQUIRE( s.m_acquiredNs > m_last.m_acquiredNs );
        REQUIRE( s.m_frame.m_acquiredRealNs - 1800000000000000000LL == s.m_acquiredNs );
        REQUIRE( s.m_frame.m_writtenRealNs - 1800000000000000000LL == s.m_publishedNs );
        REQUIRE( s.m_receivedNs <= nowNs );
        REQUIRE( s.m_generation != 0 );
        REQUIRE( s.m_segment != 0 );
        for( std::size_t axis = 0; axis < 2; ++axis )
            REQUIRE( s.m_acceleration[axis] == Approx( ( double( s.m_frame.m_raw[axis] ) - 2048 ) * .0083799 / 1.02 ) );
        if( m_last.m_generation == s.m_generation )
            REQUIRE( s.m_frame.m_counter > m_last.m_frame.m_counter );
        m_last = s;
        ++m_sinceOrigin;
        m_maxDeliveryAge = std::max( m_maxDeliveryAge, nowNs - s.m_acquiredNs );
    }
    return ReplayApplication::adc( record, nowNs );
}

bool AuditedApplication::wfs( const DDSPC::TimedWfsRecord &record, std::int64_t nowNs )
{
    m_maxBatch    = std::max( m_maxBatch, m_sinceOrigin );
    m_sinceOrigin = 0;
    return ReplayApplication::wfs( record, nowNs );
}

/// Assert the actual application's compile-time and runtime output inhibition after every replay.
void shut_down( ReplayApplication &app /**< [in,out] Stream-free actual application fixture. */ )
{
    REQUIRE( app.outputs_rejected() );
    REQUIRE( app.bad_runtime_outputs_rejected() );
    REQUIRE( app.inhibited() );
    REQUIRE( app.appShutdown() == 0 );
}

/// Find an eligible assisted prediction beyond a known recovery interval.
bool assisted_after( const ReplayApplication &app /**< [in] Frozen observations. */,
                     std::size_t              index /**< [in] First observation to inspect. */ )
{
    return std::any_of(
        app.m_observations.begin() + index,
        app.m_observations.end(),
        []( const auto &o )
        { return o.m_status.m_candidateValid && o.m_status.m_effective == DDSPC::PolicyMode::Assisted; } );
}
} // namespace
/// \endcond

/** \brief Both learners have independent switches; actual ADC/WFS/confirmed inputs produce qualified candidates. */
TEST_CASE( "Confirmed accelSim replay exercises independent actual application learners", "[confirmed-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::processShadowAdc;
    MagAOX::app::loPredCtrlAcc::processShadowWfs;
    MagAOX::app::loPredCtrlAcc::processShadowApplied;
    DDSPC::ShadowController::training;
#endif
    // clang-format on
    const int         switches  = GENERATE( 0, 1, 2, 3 );
    const auto        recording = record_accelsim();
    ReplayApplication app;
    app.prepare( replay_model_config(), recording.m_inputs );
    app.train( ( switches & 1 ) != 0, ( switches & 2 ) != 0 );
    const auto result = replay_confirmed( app, recording );
    REQUIRE( result.m_wfsAccepted == recording.m_wfs.size() );
    REQUIRE( result.m_holdsAccepted == recording.m_holds.size() - 1 );
    const auto &f = result.m_forwarding;
    REQUIRE( f.m_source.m_samples == f.m_copied );
    REQUIRE( f.m_accepted == f.m_copied );
    REQUIRE( f.m_source.m_rateHz == Approx( 8000 ) );
    REQUIRE( f.m_source.m_source.m_depth == 128 );
    REQUIRE( f.m_source.m_dropped == 0 );
    REQUIRE( f.m_source.m_invalid == 0 );
    REQUIRE( f.m_source.m_restarts == 0 );
    REQUIRE( f.m_missing == 0 );
    REQUIRE( f.m_breaks == 0 );
    for( const auto &o : app.m_observations )
    {
        REQUIRE( o.m_candidate.allFinite() );
        REQUIRE( o.m_status.m_step <= .2 );
        if( o.m_status.m_effective == DDSPC::PolicyMode::Assisted )
        {
            REQUIRE( o.m_status.m_featureReason == DDSPC::FeatureReason::Ready );
            REQUIRE( o.m_status.m_validStreak >= 10 );
            REQUIRE( o.m_status.m_availableNs <= o.m_status.m_originNs );
            REQUIRE( o.m_status.m_originNs - o.m_status.m_featureNs <= 500000 );
        }
    }
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
    {
        const bool enabled = ( switches & ( mode == DDSPC::ShadowMode::Standard ? 1 : 2 ) ) != 0;
        const auto state   = app.model_status( mode );
        REQUIRE( state.m_trainingEnabled == enabled );
        REQUIRE( state.m_ready == enabled );
        if( enabled )
        {
            REQUIRE( state.m_updates > 700 );
            REQUIRE( state.m_rank == state.m_features );
            REQUIRE( state.m_predictions > 600 );
            REQUIRE( std::isfinite( state.m_rmse ) );
            REQUIRE( app.model( mode ).allFinite() );
        }
        else
            REQUIRE( state.m_updates == 0 );
    }
    REQUIRE( ( app.m_assistedCandidates > 500 ) == ( ( switches & 2 ) != 0 ) );
    REQUIRE( ( app.m_validCandidates > 500 ) == ( switches != 0 ) );
    if( switches == 3 )
    {
        auto times = result.m_frameUs;
        std::sort( times.begin(), times.end() );
        std::cout << "Confirmed replay: ADC=" << f.m_accepted << ", candidates=" << app.m_validCandidates
                  << ", assisted=" << app.m_assistedCandidates
                  << ", frame us mean/p99/max=" << std::accumulate( times.begin(), times.end(), 0. ) / times.size()
                  << "/" << times[std::size_t( .99 * ( times.size() - 1 ) )] << "/" << times.back() << '\n';
        for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        {
            const auto s = app.model_status( mode );
            std::cout << "  " << ( mode == DDSPC::ShadowMode::Standard ? "standard" : "assisted" )
                      << ": updates=" << s.m_updates << ", rank=" << s.m_rank << "/" << s.m_features
                      << ", predictions=" << s.m_predictions << ", training RMSE=" << s.m_rmse << '\n';
        }
    }
    shut_down( app );
}

/** \brief Frozen coefficients isolate actual feature influence; identical WFS/inputs leave the standard fit unchanged.
 */
TEST_CASE( "Decoded ADC influences assisted predictions without changing the standard model", "[confirmed-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::prediction_matrix;
    DDSPC::ShadowController::predict_cached;
#endif
    // clang-format on
    const auto recording = record_accelsim( 700 );
    for( const auto requested : { DDSPC::PolicyMode::Standard, DDSPC::PolicyMode::Assisted } )
    {
        FrozenApplication normal, changed;
        normal.prepare( replay_model_config(), recording.m_inputs, true, requested );
        changed.prepare( replay_model_config(), recording.m_inputs, true, requested );
        normal.train();
        changed.train();
        replay_confirmed( normal, recording );
        ConfirmedReplayOptions o;
        o.m_adcChangeStart  = 4 + 4 * 450;
        o.m_adcOffsetCounts = 3;
        replay_confirmed( changed, recording, o );
        for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        {
            REQUIRE( normal.model( mode ) == changed.model( mode ) ); // Fit was frozen before the sensor perturbation.
            REQUIRE( normal.model_status( mode ).m_updates == changed.model_status( mode ).m_updates );
        }
        REQUIRE( normal.model( DDSPC::ShadowMode::AccelAssisted ).rightCols( 6 ).norm() > 0 );
        double      difference = 0;
        std::size_t compared   = 0;
        for( std::size_t i = 500; i < 700; ++i )
        {
            const auto &a = normal.m_observations[i].m_status, &b = changed.m_observations[i].m_status;
            if( requested == DDSPC::PolicyMode::Standard )
            {
                REQUIRE( a.m_candidateValid == b.m_candidateValid );
                REQUIRE( normal.m_observations[i].m_candidate == changed.m_observations[i].m_candidate );
            }
            if( !a.m_candidateValid || !b.m_candidateValid || a.m_effective != requested || b.m_effective != requested )
                continue; // Actual rank/readiness gates remain active after fitting is paused.
            ++compared;
            difference +=
                ( normal.m_observations[i].m_candidate - changed.m_observations[i].m_candidate ).squaredNorm();
        }
        REQUIRE( compared > 100 );
        if( requested == DDSPC::PolicyMode::Standard )
            REQUIRE( difference == 0 );
        else
        {
            REQUIRE( difference > 1e-8 );
            std::cout << "Frozen-fit ADC influence: summed candidate difference squared=" << difference << '\n';
        }
        shut_down( normal );
        shut_down( changed );
    }
}

/** \brief Candidate selection never feeds learning; pre-limit requests are different from independently consumed
 * inputs. */
TEST_CASE( "Actual-input provenance is independent of replay shadow recommendations", "[confirmed-replay]" )
{
    const auto recording = record_accelsim( 500 );
    bool       limited   = false;
    for( std::size_t i = 0; i < recording.m_holds.size(); ++i )
    {
        limited = limited || recording.m_requested[i] != recording.m_holds[i].m_applied;
        REQUIRE( recording.m_holds[i].m_applied.cwiseAbs().maxCoeff() <= .4f );
        REQUIRE( recording.m_holds[i].m_stamp.m_receivedNs > recording.m_holds[i].m_effectiveEndNs );
    }
    REQUIRE( limited );
    ReplayApplication standard, assisted;
    standard.prepare( replay_model_config(), recording.m_inputs, true, DDSPC::PolicyMode::Standard, .2 );
    assisted.prepare( replay_model_config(), recording.m_inputs, true, DDSPC::PolicyMode::Assisted, .001 );
    standard.train();
    assisted.train();
    replay_confirmed( standard, recording );
    replay_confirmed( assisted, recording );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        REQUIRE( standard.model( mode ) == assisted.model( mode ) );
    ReplayApplication sensorChanged;
    sensorChanged.prepare( replay_model_config(), recording.m_inputs );
    sensorChanged.train();
    ConfirmedReplayOptions sensor;
    sensor.m_adcOffsetCounts = 3;
    replay_confirmed( sensorChanged, recording, sensor );
    REQUIRE( standard.model( DDSPC::ShadowMode::Standard ) == sensorChanged.model( DDSPC::ShadowMode::Standard ) );
    REQUIRE( standard.model( DDSPC::ShadowMode::AccelAssisted ) !=
             sensorChanged.model( DDSPC::ShadowMode::AccelAssisted ) );
    shut_down( sensorChanged );
    bool different = false;
    for( std::size_t i = 0; i < recording.m_wfs.size(); ++i )
        different = different || standard.m_observations[i].m_candidate != assisted.m_observations[i].m_candidate;
    REQUIRE( different );
    shut_down( standard );
    shut_down( assisted );
}

/** \brief Source replacement and bounded-consumer loss affect assistance only; common loss suspends both models. */
TEST_CASE( "Confirmed replay rejects missing late or unverified evidence and recovers", "[confirmed-replay]" )
{
    const int fault = GENERATE( 0, 1, 2, 3, 4, 5, 6, 7, 8 );
    auto recording  = record_accelsim( 900, 721, fault == 0 ? 4 + 4 * 300 : std::numeric_limits<std::size_t>::max() );
    if( fault == 8 )
        recording.m_adc[4 + 4 * 300].m_frame.m_raw = { 0,
                                                       2048 }; // Original rail evidence must reach the actual decoder.
    ConfirmedReplayOptions o;
    if( fault == 1 || fault == 7 )
    {
        o.m_skipForwardStart = 300;
        o.m_skipForwardCount = fault == 1 ? 100 : 10;
    }
    if( fault == 2 )
        o.m_deliveryDelayNs = 2000000;
    if( fault == 3 )
        o.m_wrongClock = true;
    if( fault == 4 )
    {
        o.m_skipWfsStart = 300;
        o.m_skipWfsCount = 20;
    }
    if( fault == 5 || fault == 6 )
    {
        o.m_badHoldStart = fault == 5 ? 0 : 300;
        o.m_badHoldCount = fault == 5 ? 900 : 20;
    }
    ReplayApplication app;
    app.prepare( replay_model_config(), recording.m_inputs );
    app.train();
    const auto result = replay_confirmed( app, recording, o );
    if( fault == 2 || fault == 3 || fault == 5 )
    {
        REQUIRE_FALSE( assisted_after( app, 0 ) );
        REQUIRE( app.model_status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
        if( fault == 5 )
        {
            REQUIRE( result.m_holdsAccepted == 0 );
            REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
            REQUIRE( app.m_validCandidates == 0 );
        }
        else
            REQUIRE( app.model_status( DDSPC::ShadowMode::Standard ).m_updates > 800 );
    }
    else
    {
        REQUIRE( assisted_after( app, 600 ) );
        if( fault == 0 )
        {
            REQUIRE( result.m_forwarding.m_source.m_generation == 2 );
            REQUIRE( result.m_forwarding.m_source.m_inode == 9 );
            REQUIRE( result.m_forwarding.m_breaks >= 1 );
            REQUIRE( result.m_forwarding.m_missing == 0 );
        }
        if( fault == 1 )
        {
            REQUIRE( result.m_forwarding.m_missing > 0 );
            REQUIRE( result.m_forwarding.m_source.m_dropped == 0 ); // Capture succeeded; consumer evicted unread data.
            REQUIRE( result.m_forwarding.m_breaks >= 1 );
        }
        if( fault == 7 )
            REQUIRE( result.m_forwarding.m_missing == 0 ); // Short consumer pause fits retained history.
        if( fault == 8 )
        {
            REQUIRE( result.m_forwarding.m_source.m_invalid == 1 );
            REQUIRE( result.m_forwarding.m_missing >= 1 );
            REQUIRE( result.m_forwarding.m_breaks >= 1 );
        }
        bool unavailable = false;
        for( const auto &v : app.m_observations )
            if( v.m_status.m_index >= 401 && v.m_status.m_index < 430 &&
                v.m_status.m_effective != DDSPC::PolicyMode::Assisted )
            {
                unavailable = true;
                if( fault <= 1 || fault == 7 || fault == 8 )
                    REQUIRE( v.m_status.m_effective == DDSPC::PolicyMode::Standard );
            }
        REQUIRE( unavailable );
    }
    shut_down( app );
}

/** \brief Future source mutations cannot alter previous decisions; no record is delivered before its own availability.
 */
TEST_CASE( "Confirmed replay decisions do not receive future source measurements", "[confirmed-replay]" )
{
    auto normal = record_accelsim( 700 );
    auto future = normal;
    for( std::size_t i = 500; i < future.m_wfs.size(); ++i )
    {
        future.m_wfs[i].m_residual.setConstant( 100 );
        future.m_holds[i].m_applied.setConstant( 2 );
    }
    for( std::size_t i = 4 + 4 * 500; i < future.m_adc.size(); ++i )
        future.m_adc[i].m_frame.m_raw = { 2050, 2046 };
    ReplayApplication a, b;
    a.prepare( replay_model_config(), normal.m_inputs );
    b.prepare( replay_model_config(), normal.m_inputs );
    a.train();
    b.train();
    replay_confirmed( a, normal );
    replay_confirmed( b, future );
    for( std::size_t i = 0; i < 500; ++i )
    {
        REQUIRE( a.m_observations[i].m_candidate == b.m_observations[i].m_candidate );
        REQUIRE( a.m_observations[i].m_status.m_effective == b.m_observations[i].m_status.m_effective );
    }
    shut_down( a );
    shut_down( b );
}

/** \brief A stalled consumer catches up only within its dispatch budget, without rewriting sample availability. */
TEST_CASE( "ADC replay forwarding preserves provenance and a hard per-origin work bound", "[confirmed-replay]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::copy_after;
    MagAOX::app::loPredCtrlAcc::processShadowAdc;
#endif
    // clang-format on
    const auto         recording = record_accelsim( 700 );
    AuditedApplication app;
    app.prepare( replay_model_config(), recording.m_inputs );
    app.train();
    ConfirmedReplayOptions o;
    o.m_skipForwardStart = 300;
    o.m_skipForwardCount = 10; // 40 retained records fit capacity 128, but exceed the 32-sample dispatch budget.
    const auto r         = replay_confirmed( app, recording, o );
    REQUIRE( app.m_maxBatch == 32 );
    REQUIRE( app.m_maxDeliveryAge > 5000000 );
    REQUIRE( r.m_forwarding.m_missing == 0 );
    REQUIRE( r.m_forwarding.m_copied == r.m_forwarding.m_source.m_samples );
    REQUIRE_FALSE( r.m_forwarding.m_pending );
    REQUIRE( assisted_after( app, 500 ) );
    shut_down( app );
}
} // namespace libXWCTest::loPredCtrlAccTest
