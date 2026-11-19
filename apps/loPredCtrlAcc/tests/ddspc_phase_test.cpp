/** \file ddspc_phase_test.cpp
 * \brief Native phase DDSPC causality, independent learning, cached inference and private producer regression.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../phase_shadow.hpp"
#include "../native_pipeline.hpp"
#include "../online_shadow.hpp"
#include "../adc_input.hpp"
#include "adc_transport_fixture.hpp"
#include <random>
#include <chrono>
#include <iostream>
#include <map>
#include <tuple>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond PHASE_TEST_SUPPORT
namespace
{
namespace synco = MagAOX::app::synco;
/// One process-private sample source for the actual ADC decoding/capture implementation.
struct SingleAdc : DDSPC::AdcSource
{
    DDSPC::AdcFrame m_frame; ///< Original raw sample and metadata.
    DDSPC::AdcClock m_clock; ///< Independently declared synthetic clocks.
    /// Return the exact UINT16 two-channel depth-one source geometry.
    DDSPC::AdcHeader header() const override;
    /// Return only the private raw sample.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Sole plane zero. */ ) const override;
    /// Return the declared original receipt bracket.
    DDSPC::AdcClock clock() const override;
};
DDSPC::AdcHeader SingleAdc::header() const
{
    return { 3, 2, 1, 1, m_frame.m_counter, 0, true, true, true, false };
}
DDSPC::AdcFrame SingleAdc::frame( std::size_t ) const
{
    return m_frame;
}
DDSPC::AdcClock SingleAdc::clock() const
{
    return m_clock;
}

/// Build immutable synthetic source identity; no ImageStreamIO producer or namespace is used.
DDSPC::NativeObservation observation( DDSPC::ObservationSource role /**< [in] Source role. */,
                                      std::uint64_t            counter /**< [in] Original publication count. */,
                                      std::int64_t             now /**< [in] Actual synthetic availability. */ )
{
    DDSPC::NativeObservation r;
    r.m_source     = role;
    r.m_counter    = counter;
    r.m_generation = r.m_segment = 1;
    r.m_device                   = 11;
    r.m_fileInode                = 100 + unsigned( role );
    r.m_imageInode               = 200 + unsigned( role );
    r.m_receivedNs = r.m_availableNs = now;
    r.m_elements                     = 2;
    return r;
}
/// Produce a genuine version-two four-slot payload with independently established WFS/cycle association.
synco::Record schedule( std::uint64_t wfs /**< [in] Original modal counter. */,
                        std::int64_t  now /**< [in] Reference time. */ )
{
    synco::Record s{};
    s[synco::version] = synco::productionVersion;
    s[synco::epoch] = s[synco::generation] = 1;
    s[synco::state]                        = synco::armed;
    s[synco::steps]                        = 4;
    s[synco::period]                       = 500000;
    s[synco::cycle]                        = wfs - 1;
    s[synco::reference]                    = now;
    s[synco::expires]                      = now + 1000000;
    for( unsigned p = 0; p < 4; ++p )
    {
        s[synco::cutoffs + p] = now + 100000 + p * 125000;
        s[synco::targets + p] = now + 110000 + p * 125000;
    }
    return s;
}
/// Repeatable causal plant; commands are externally established and never taken from shadow proposals.
struct PhaseReplay
{
    DDSPC::OnlineShadowConfig           m_config;      ///< Bounded actual native learner settings.
    DDSPC::NativePipelineConfig         m_inputs;      ///< Explicit synthetic calibration and upstream source contract.
    std::unique_ptr<DDSPC::PhaseShadow> m_pipeline;    ///< Actual production phase input/prediction path.
    DDSPC::AdcCapture m_decoder{ DDSPC::AdcConfig{} }; ///< Actual native ADC conversion and continuity logic.
    SingleAdc         m_source;                        ///< Process-private raw transport double.
    std::mt19937      m_physical{ 123 }, m_sensor{ 456 }, m_commands{ 789 },
        m_noise{ 987 };                             ///< Independent fixed realizations.
    std::normal_distribution<float> m_draw{ 0, 1 }; ///< Broadband excitation, including negative-control independence.
    std::array<float, 2>            m_y{};          ///< Actual next WFS measurement, never exposed before its frame.
    std::uint64_t                   m_frame{ 0 }, m_counter{ 0 }; ///< Actual independent source counters.
    std::vector<double>   m_times; ///< Full ADC observe-call measurements, including retention and payload rendering.
    std::vector<double>   m_learnTimes;     ///< Whole WFS learning jobs including changed-gain/rank preparation.
    std::array<double, 2> m_error{};        ///< Paired held-out origin-only squared errors.
    std::array<std::uint64_t, 2> m_count{}; ///< Identical target/horizon coverage per model.
    struct Pair
    {
        std::array<double, 2> m_error{};
        unsigned              m_seen{ 0 };
    }; ///< Test-only matched scoring state.
    std::map<std::tuple<std::uint64_t, unsigned, unsigned>, Pair>
         m_pairs;           ///< Held-out origin/phase/channel intersection.
    bool m_related{ true }; ///< Whether the measured ADC has physical predictive information.
    bool m_rotated{ false }, m_probe{ false },
        m_held{ false };   ///< Protocol order, exact targets and held standard input.
    bool m_async{ false }; ///< The test numerical owner may run concurrently; fast tests never read its mutable status.
    std::int64_t m_base{ 1000000000 }; ///< Declared source epoch; paced worker uses the actual local clock.
    std::int64_t m_period{ 500000 };   ///< Advertised nominal WFS period; tests allow real integer-nanosecond rounding.

    /// Construct both models and a reviewed identity projection only for this explicit synthetic plant.
    PhaseReplay( bool     qr /**< [in] Actual covariance or QR RLS. */,
                 bool     related       = true /**< [in] Independent ADC negative control. */,
                 unsigned pixels        = 2 /**< [in] Complete original DM image extent. */,
                 int      horizon       = 2 /**< [in] Actual WFS future targets. */,
                 int      standardSlots = 4 /**< [in] Held baseline or matched lifted baseline. */,
                 unsigned stride        = 4 /**< [in] Real-WFS adaptation stride; one exercises 2-kHz fitting. */ );
    /// Supply one real WFS frame then four individual raw ADC publications; fit only on the numerical side.
    void step( int  fault = 0 /**< [in] Drop ADC, drop WFS, omit schedule, or deliver ADC late. */,
               bool learn = true /**< [in] False proves coefficients are unchanged by ADC inference. */ );
    /// Feed a complete original requested input, never a private proposal.
    void command( std::span<const float> values /**< [in] Complete original request. */,
                  std::int64_t           now /**< [in] Actual availability. */ );
    /// Drain matured actual-WFS scores and retain only the fixed held-out interval.
    void scores();
};
PhaseReplay::PhaseReplay( bool qr, bool related, unsigned pixels, int horizon, int standardSlots, unsigned stride )
    : m_related( related )
{
    auto &m                       = m_config.m_model;
    m.m_modes                     = 2;
    m.m_history                   = 2;
    m.m_future                    = horizon;
    m.m_accelHistory              = 1;
    m.m_gamma                     = 1;
    m.m_prior                     = .001;
    m.m_lambda                    = .1;
    m.m_standardCommandSlots      = standardSlots;
    m.m_useQrd                    = qr;
    m.m_updateStride              = stride;
    m_config.m_predictionCapacity = 32768;
    auto &c                       = m_inputs;
    c.m_enabled                   = true;
    c.m_scaledPenalty             = true;
    c.m_feedback                  = .2;
    c.m_modalAnchor = c.m_adcAnchor = 1;
    c.m_syncEpoch                   = 77;
    c.m_statusDevice                = "synthetic_adc";
    c.m_reference                   = "synthetic_rtc";
    c.m_syncoAssociation            = true;
    c.m_calibration.m_layout        = 17;
    c.m_calibration.m_indices       = { 0, 1 };
    c.m_calibration.m_modalUnits    = "synthetic_modal";
    c.m_calibration.m_commandUnits  = "synthetic_request";
    c.m_calibration.m_projection    = DDSPC::Matrix::Zero( 2, pixels );
    c.m_calibration.m_shapes        = DDSPC::Matrix::Zero( pixels, 2 );
    c.m_calibration.m_projection.leftCols( 2 ).setIdentity();
    c.m_calibration.m_shapes.topRows( 2 ).setIdentity();
    for( unsigned p = 2; p < pixels; ++p )
    {
        c.m_calibration.m_shapes( p, 0 ) = .01;
        c.m_calibration.m_shapes( p, 1 ) = -.01;
    }
    m_pipeline = std::make_unique<DDSPC::PhaseShadow>( c, m_config );
    m_pipeline->learning( true );
    m_decoder.attached( 11, 100 );
}
void PhaseReplay::command( std::span<const float> values, std::int64_t now )
{
    auto r          = observation( DDSPC::ObservationSource::Dm, ++m_counter, now );
    r.m_elements    = values.size();
    r.m_payloadHash = DDSPC::command_payload_hash( values );
    for( unsigned i = 0; i < 2; ++i )
        r.m_values[i] = values[i];
    m_pipeline->command( r, values );
    m_pipeline->observe( r );
}
void PhaseReplay::scores()
{
    DDSPC::ShadowPrediction r;
    while( m_pipeline->read_prediction( r ) )
    {
        REQUIRE( r.m_target > r.m_origin );
        REQUIRE( r.m_horizon >= 1 );
        REQUIRE( r.m_horizon <= unsigned( m_config.m_model.m_future ) );
        REQUIRE( r.m_phase < 4 );
        if( r.m_adcCounter )
            REQUIRE( r.m_adcCounter == 4 * ( r.m_origin - 1 ) + r.m_phase + 1 );
        else
            REQUIRE( r.m_mode == DDSPC::ShadowMode::Standard );
        REQUIRE( r.m_completedNs >= r.m_decisionNs );
        if( r.m_origin >= 1800 && r.m_horizon == 1 )
        {
            const unsigned m    = r.m_mode == DDSPC::ShadowMode::AccelAssisted;
            const double   e    = r.m_originPrediction - r.m_measurement;
            const auto     key  = std::make_tuple( r.m_origin, r.m_phase, r.m_channel );
            auto          &pair = m_pairs[key];
            pair.m_error[m]     = e * e;
            pair.m_seen |= 1u << m;
            if( pair.m_seen == 3 )
            {
                for( unsigned j = 0; j < 2; ++j )
                {
                    m_error[j] += pair.m_error[j];
                    ++m_count[j];
                }
                m_pairs.erase( key );
            }
        }
    }
}
void PhaseReplay::step( int fault, bool learn )
{
    ++m_frame;
    const std::int64_t t = m_base + 500000LL * std::int64_t( m_frame - 1 );
    std::vector<float> image( m_inputs.m_calibration.m_projection.cols(), 0 );
    if( m_frame == 1 )
        command( image, t - 1000 );
    if( fault != 2 )
    {
        auto w     = observation( DDSPC::ObservationSource::Modal, m_frame, t );
        w.m_values = { m_y[0], m_y[1] };
        m_pipeline->observe( w );
    }
    const DDSPC::NativeSyncStatus status{ { 77, true, true, t + 1 }, 4 * m_frame, m_frame, 0 };
    m_pipeline->sync( status );
    auto sched           = schedule( m_frame, t );
    sched[synco::period] = m_period;
    for( unsigned p = 0; p < 4; ++p )
    {
        sched[synco::targets + p] = t + 110000 + p * m_period / 4;
        sched[synco::cutoffs + p] = t + 100000 + p * m_period / 4;
    }
    if( m_rotated )
    {
        const auto original = sched;
        for( unsigned p = 0; p < 4; ++p )
        {
            sched[synco::targets + ( p + 1 ) % 4] = original[synco::targets + p];
            sched[synco::cutoffs + ( p + 1 ) % 4] = original[synco::cutoffs + p];
        }
    }
    if( fault != 3 )
        m_pipeline->schedule( sched, t + 2 );
    if( learn )
        while( true )
        {
            const auto begin = std::chrono::steady_clock::now();
            if( !m_pipeline->learn() )
                break;
            if( m_frame > 1800 )
                m_learnTimes.push_back(
                    std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - begin ).count() );
        }
    std::array<float, 2> next{ .5f * m_y[0], .5f * m_y[1] };
    for( unsigned phase = 0; phase < 4; ++phase )
    {
        for( unsigned m = 0; m < 2; ++m )
            if( !m_held || phase == 0 )
                image[m] = .3f * m_draw( m_commands );
        command( image, t + 5000 + phase * 125000 );
        const auto at                     = t + 20000 + phase * 125000;
        m_source.m_frame.m_counter        = 4 * ( m_frame - 1 ) + phase + 1;
        m_source.m_frame.m_acquiredRealNs = 1000000000000LL + at - 1000;
        m_source.m_frame.m_writtenRealNs  = 1000000000000LL + at - 500;
        m_source.m_clock                  = { 1000000000000LL + at, at, 0 };
        for( unsigned m = 0; m < 2; ++m )
        {
            const double vibration    = 10 * std::sin( 2 * 3.141592653589793 * 28.5 * at * 1e-9 + m ) +
                                        5 * std::sin( 2 * 3.141592653589793 * 56 * at * 1e-9 + m );
            const int    physical     = std::lround( 25 * m_draw( m_physical ) + vibration );
            const int    unrelated    = std::lround( 25 * m_draw( m_sensor ) );
            m_source.m_frame.m_raw[m] = 2048 + ( m_related ? physical : unrelated );
            next[m] += .4f * physical * .0083799f / 1.02f + .08f * image[m];
        }
        REQUIRE( m_decoder.drain( m_source ) == 1 );
        const auto decoded = m_decoder.status().m_latest;
        auto       a       = observation( DDSPC::ObservationSource::Adc, m_source.m_frame.m_counter, at );
        a.m_values         = decoded.m_acceleration;
        a.m_raw            = decoded.m_frame.m_raw;
        a.m_receivedNs     = decoded.m_receivedNs;
        if( fault == 4 )
            a.m_receivedNs -= 300000; // Retained/forwarded late: raw counter is new, availability is stale.
        const auto before = m_async ? DDSPC::OnlineShadowStatus{} : m_pipeline->status();
        const auto begin  = std::chrono::steady_clock::now();
        if( fault != 1 )
            m_pipeline->observe( a );
        const auto end = std::chrono::steady_clock::now();
        if( m_frame > 1800 )
            m_times.push_back( std::chrono::duration<double, std::micro>( end - begin ).count() );
        const auto after = m_async ? DDSPC::OnlineShadowStatus{} : m_pipeline->status();
        if( m_probe && after.m_standard.m_ready && after.m_assisted.m_ready && m_frame > 200 )
        {
            std::array<float, 2> pixels{};
            const auto           request  = synco::request( sched, m_rotated ? ( phase + 1 ) % 4 : phase );
            const auto           copiedAt = at + std::int64_t( std::ceil( after.m_fastUs * 1000 ) ) + 2000;
            if( copiedAt < request.cutoffNs - 5000 )
                REQUIRE( m_pipeline->producer()->command( request, sched, copiedAt, pixels ) );
            REQUIRE_FALSE( m_pipeline->producer()->command( request, sched, request.cutoffNs, pixels ) );
            auto wrong = request;
            wrong.generation++;
            REQUIRE_FALSE( m_pipeline->producer()->command( wrong, sched, copiedAt, pixels ) );
        }
        REQUIRE( after.m_standard.m_updates == before.m_standard.m_updates );
        REQUIRE( after.m_assisted.m_updates == before.m_assisted.m_updates );
        REQUIRE( after.m_wfsReceived == before.m_wfsReceived );
        if( phase && fault == 0 && after.m_standard.m_ready && after.m_assisted.m_ready )
            REQUIRE( after.m_fastPredictions == before.m_fastPredictions + 1 );
        m_pipeline->command_poll( true, t + 105000 + phase * 125000 );
        m_pipeline->tick( t + 105000 + phase * 125000 );
    }
    for( unsigned m = 0; m < 2; ++m )
        next[m] += .02f * m_draw( m_noise );
    m_y = next;
    if( !m_async )
        scores();
}
} // namespace
/// \endcond

/** \brief Individual ADC arrivals predict at four phases while actual WFS frames alone drive both native learners. */
TEST_CASE( "ADC-driven native DDSPC separates phase prediction from WFS learning", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
    DDSPC::PhaseShadow::learn;
    DDSPC::PhaseShadow::coefficients;
#endif
    // clang-format on
    PhaseReplay replay( GENERATE( false, true ) );
    for( unsigned n = 0; n < 2800; ++n )
        replay.step();
    const auto s = replay.m_pipeline->status();
    REQUIRE( s.m_wfsReceived == 2800 );
    REQUIRE( s.m_fastOpportunities == 11200 );
    REQUIRE( s.m_standard.m_ready );
    REQUIRE( s.m_assisted.m_ready );
    for( auto n : s.m_phasePredictions )
        REQUIRE( n > 1000 );
    REQUIRE( replay.m_count[0] == replay.m_count[1] );
    REQUIRE( replay.m_count[0] > 7000 );
    REQUIRE( replay.m_error[1] < .8 * replay.m_error[0] );
    const auto stdWeights = replay.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 );
    const auto accWeights = replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 2 );
    replay.step( 0, false );
    REQUIRE( replay.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 ) == stdWeights );
    REQUIRE( replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 2 ) == accWeights );
    std::cout << "phase positive qr=" << replay.m_config.m_model.m_useQrd
              << " mse_standard=" << replay.m_error[0] / replay.m_count[0]
              << " mse_assisted=" << replay.m_error[1] / replay.m_count[1] << " fast_max_us=" << s.m_fastPeakUs << '\n';
    REQUIRE( ddspc_test_output_calls() == 0 );
}

/** \brief Independent sensor data preserves the standard fit and cannot create a systematic held-out gain. */
TEST_CASE( "Native phase regression rejects unrelated accelerometer benefit", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::coefficients;
    DDSPC::PhaseShadow::read_prediction;
#endif
    // clang-format on
    PhaseReplay related( true, true ), unrelated( true, false );
    for( unsigned n = 0; n < 2800; ++n )
    {
        related.step();
        unrelated.step();
    }
    REQUIRE( related.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 ) ==
             unrelated.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 ) );
    REQUIRE( unrelated.m_pipeline->status().m_assisted.m_ready );
    const auto ratio = unrelated.m_error[1] / unrelated.m_error[0];
    REQUIRE( ratio > .9 );
    REQUIRE( ratio < 1.2 );
    std::cout << "phase negative mse_ratio=" << ratio << '\n';
}

/** \brief Missing samples, delayed delivery, WFS loss and absent command scheduling never grant readiness. */
TEST_CASE( "Native phase inputs lose and recover qualification without synchronization estimation",
           "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
    DDSPC::PhaseShadow::tick;
#endif
    // clang-format on
    const int   fault = GENERATE( 1, 2, 3, 4 );
    PhaseReplay replay( true );
    for( unsigned n = 0; n < 600; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    for( unsigned n = 0; n < 8; ++n )
        replay.step( fault );
    const auto lost = replay.m_pipeline->status();
    REQUIRE_FALSE( lost.m_assisted.m_ready );
    for( unsigned n = 0; n < 600; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_standard.m_ready );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    REQUIRE( ddspc_test_output_calls() == 0 );
}

/** \brief Both models export exactly the same operational penalty convention and cached prefix optimization. */
TEST_CASE( "Phase snapshot gains preserve shared DDSPC control objective", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::snapshot;
    DDSPC::ShadowController::predict_next;
#endif
    // clang-format on
    DDSPC::ShadowConfig c;
    c.m_modes                = 1;
    c.m_history              = 2;
    c.m_future               = 2;
    c.m_accelHistory         = 1;
    c.m_accelChannels        = 2;
    c.m_standardCommandSlots = c.m_assistedCommandSlots = 4;
    c.m_scaledPenalty                                   = true;
    c.m_lambda                                          = .3;
    c.m_prior                                           = .001;
    c.m_externalOrigins                                 = true;
    DDSPC::ShadowController models( c );
    models.training( DDSPC::ShadowMode::Standard, true );
    models.training( DDSPC::ShadowMode::AccelAssisted, true );
    std::mt19937                    rng( 123 );
    std::normal_distribution<float> draw;
    DDSPC::ShadowRecord             r;
    r.m_residual               = DDSPC::Matrix::Zero( 1, 1 );
    r.m_appliedCommand         = DDSPC::Matrix::Zero( 4, 1 );
    r.m_assistedAppliedCommand = r.m_appliedCommand;
    r.m_acceleration           = DDSPC::Matrix::Zero( 2, 1 );
    r.m_wfsValid = r.m_appliedValid = r.m_assistedAppliedValid = r.m_timingValid = r.m_accelValid = true;
    for( unsigned n = 1; n <= 600; ++n )
    {
        r.m_index         = n;
        r.m_residual( 0 ) = draw( rng );
        for( int i = 0; i < 4; ++i )
            r.m_appliedCommand( i ) = draw( rng );
        r.m_assistedAppliedCommand = r.m_appliedCommand;
        for( int i = 0; i < 2; ++i )
            r.m_acceleration( i ) = draw( rng );
        models.process( r );
    }
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
    {
        DDSPC::ShadowSnapshot snapshot;
        REQUIRE( models.snapshot( mode, snapshot ) );
        const DDSPC::Matrix commands = snapshot.m_theta.rightCols( 8 );
        const auto          penalty  = c.m_lambda * ( commands.transpose() * commands ).maxCoeff();
        DDSPC::Matrix       empty( snapshot.m_theta.rows(), 0 );
        const DDSPC::Matrix expected = DDSPC::reference_controller(
            snapshot.m_theta.leftCols( snapshot.m_stateRows ), empty, empty, commands, penalty );
        REQUIRE( ( expected - snapshot.m_gain[0] ).cwiseAbs().maxCoeff() < 1e-6 );
        REQUIRE( snapshot.m_gain[3].rows() == 5 );
        DDSPC::Matrix y = DDSPC::Matrix::Constant( 1, 1, .2 ), prefix = DDSPC::Matrix::Constant( 1, 1, .3 );
        DDSPC::Matrix a =
            mode == DDSPC::ShadowMode::AccelAssisted ? DDSPC::Matrix::Constant( 2, 1, .1 ) : DDSPC::Matrix( 0, 1 );
        DDSPC::Matrix state( snapshot.m_stateRows + 1, 1 ), plan( 8, 1 );
        state << y, r.m_residual, r.m_appliedCommand, a, prefix;
        REQUIRE( models.predict_next( mode, 601, y, a, plan, &prefix ) );
        REQUIRE( ( plan.topRows( 7 ) - snapshot.m_gain[1] * state ).cwiseAbs().maxCoeff() < 1e-6 );
    }
}

/** \brief Full observe-call timings include ADC retention, cached inference and private 2500-pixel preparation. */
TEST_CASE( "Native ADC fast path reports measured execution without a real-time claim", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
    DDSPC::PhaseShadow::learn;
    DDSPC::PhaseShadow::status;
#endif
    // clang-format on
    const unsigned stride = GENERATE( 1u, 4u );
    PhaseReplay    replay( true, true, 2500, 2, 4, stride );
    for( unsigned n = 0; n < 2400; ++n )
        replay.step();
    auto &times = replay.m_times;
    std::sort( times.begin(), times.end() );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    REQUIRE( times.size() > 2000 );
    const auto status = replay.m_pipeline->status();
    REQUIRE( status.m_preparedPlans > 1000 );
    std::cout << "adc_fast stride=" << stride << " pixels=2500 samples=" << times.size()
              << " p50_us=" << times[times.size() / 2] << " p99_us=" << times[times.size() * 99 / 100]
              << " observed_max_us=" << times.back()
              << " target_us=125 deadline_qualified=false publications=" << ddspc_test_output_calls() << '\n';
    auto &jobs = replay.m_learnTimes;
    std::sort( jobs.begin(), jobs.end() );
    REQUIRE( jobs.size() > 500 );
    std::cout << "phase_learning stride=" << stride << " jobs=" << jobs.size() << " p50_us=" << jobs[jobs.size() / 2]
              << " p99_us=" << jobs[jobs.size() * 99 / 100] << " observed_max_us=" << jobs.back()
              << " deadline_qualified=false\n";
    REQUIRE( ddspc_test_output_calls() == 0 );
}

/** \brief Protocol labels may be 1,2,3,0 in chronological order; every cached command retains its exact Request. */
TEST_CASE( "ADC phase candidates use chronological synco targets and safe mode fallback", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::request;
    DDSPC::SyncoShadowProducer::command;
#endif
    // clang-format on
    PhaseReplay replay( true );
    replay.m_rotated = replay.m_probe = true;
    replay.m_pipeline->request( DDSPC::ShadowMode::AccelAssisted );
    for( unsigned n = 0; n < 700; ++n )
        replay.step();
    const auto ready = replay.m_pipeline->status();
    REQUIRE( ready.m_effective == DDSPC::ShadowMode::AccelAssisted );
    REQUIRE( ready.m_preparedPlans > 1000 );
    replay.step( 4 );
    REQUIRE( replay.m_pipeline->status().m_effective == DDSPC::ShadowMode::Standard );
    REQUIRE( replay.m_pipeline->status().m_standard.m_ready );
    for( unsigned n = 0; n < 400; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_effective == DDSPC::ShadowMode::AccelAssisted );
    replay.m_pipeline->request( DDSPC::ShadowMode::Standard );
    replay.step();
    REQUIRE( replay.m_pipeline->status().m_effective == DDSPC::ShadowMode::Standard );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    REQUIRE( ddspc_test_output_calls() == 0 );
}

/** \brief Future local delivery cannot advance a model, and a changed source cannot silently inherit anchors. */
TEST_CASE( "Phase DDSPC rejects future availability and requires restart provenance", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
#endif
    // clang-format on
    PhaseReplay replay( true );
    for( unsigned n = 0; n < 600; ++n )
        replay.step();
    const auto         before = replay.m_pipeline->status();
    const std::int64_t now    = replay.m_base + replay.m_frame * 500000;
    auto               future = observation( DDSPC::ObservationSource::Modal, replay.m_frame + 1, now + 100 );
    replay.m_pipeline->observe( future, now );
    REQUIRE( replay.m_pipeline->status().m_wfsReceived == before.m_wfsReceived );
    REQUIRE_FALSE( replay.m_pipeline->status().m_qualified );
    future.m_availableNs = future.m_receivedNs = now;
    future.m_generation                        = 2;
    replay.m_pipeline->observe( future );
    REQUIRE_FALSE( replay.m_pipeline->status().m_qualified );
    REQUIRE_FALSE( replay.m_pipeline->status().m_predictionValid );
    PhaseReplay restarted( true );
    REQUIRE_FALSE( restarted.m_pipeline->status().m_standard.m_ready );
    for( unsigned n = 0; n < 600; ++n )
        restarted.step();
    REQUIRE( restarted.m_pipeline->status().m_assisted.m_ready );
}

/** \brief The actual production component keeps immutable snapshots coherent between its two owners. */
TEST_CASE( "Native phase inference and WFS learning exchange bounded snapshots concurrently", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
    DDSPC::PhaseShadow::learn;
    DDSPC::PhaseShadow::read_prediction;
#endif
    // clang-format on
    PhaseReplay replay( true );
    replay.m_async = true;
    std::atomic<bool>          finished{ false };
    std::atomic<std::uint64_t> scores{ 0 };
    std::jthread               numerical(
        [&]( std::stop_token stop )
        {
            while( !stop.stop_requested() && ( !finished.load() || replay.m_pipeline->queued() ) )
            {
                replay.m_pipeline->learn();
                DDSPC::ShadowPrediction p;
                while( replay.m_pipeline->read_prediction( p ) )
                    ++scores;
                std::this_thread::yield();
            }
        } );
    for( unsigned n = 0; n < 1400; ++n )
    {
        while( replay.m_pipeline->queued() > 1 )
            std::this_thread::yield();
        replay.step( 0, false );
    }
    finished.store( true );
    numerical.join();
    const auto s = replay.m_pipeline->status();
    REQUIRE( s.m_wfsReceived == 1400 );
    REQUIRE( s.m_fastOpportunities == 5600 );
    REQUIRE( s.m_records > 1300 );
    REQUIRE( s.m_standard.m_updates > 0 );
    REQUIRE( s.m_assisted.m_updates > 0 );
    REQUIRE( s.m_fastPredictions > 0 );
    REQUIRE( s.m_drops == 0 );
    REQUIRE( scores.load() > 0 );
    REQUIRE_FALSE( s.m_candidateValid );
    std::cout << "phase_handoff records=" << s.m_records << " opportunities=" << s.m_fastOpportunities
              << " predictions=" << s.m_fastPredictions << " drops=" << s.m_drops << " deadline_qualified=false\n";
    REQUIRE( ddspc_test_output_calls() == 0 );
}
/** \brief A single-rate baseline must not pretend independently changing quarter-frame requests were held. */
TEST_CASE( "Single-slot native baseline requires a genuinely held command history", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::learn;
#endif
    // clang-format on
    PhaseReplay varying( true, true, 2, 2, 1 ), held( true, true, 2, 2, 1 );
    held.m_held = true;
    for( unsigned n = 0; n < 700; ++n )
    {
        varying.step();
        held.step();
    }
    REQUIRE_FALSE( varying.m_pipeline->status().m_standard.m_ready );
    REQUIRE( held.m_pipeline->status().m_standard.m_ready );
    REQUIRE_FALSE( held.m_pipeline->status().m_assisted.m_ready ); // Four free command columns remain unexcited.
}

/** \brief The fast path retains real WFS target indexing at the minimum and maximum supported lifted horizons. */
TEST_CASE( "Native phase prediction preserves horizon bounds and dimensions", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
#endif
    // clang-format on
    const int   horizon = GENERATE( 1, 4, 8 );
    PhaseReplay replay( true, true, 2, horizon );
    for( unsigned n = 0; n < 900; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_standard.m_ready );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    const auto weights = replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 3 );
    REQUIRE( weights.rows() == 2 * horizon );
    REQUIRE( weights.cols() == 8 * ( horizon + 1 ) + 4 + 8 );
}
/** \brief At stride one, each completed genuine WFS record adapts once; its four ADC events never adapt. */
TEST_CASE( "Native phase models can adapt at every real WFS target", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::learn;
    DDSPC::PhaseShadow::coefficients;
#endif
    // clang-format on
    PhaseReplay replay( GENERATE( false, true ), true, 2, 2, 4, 1 );
    for( unsigned n = 0; n < 400; ++n )
        replay.step();
    const auto before = replay.m_pipeline->status();
    replay.step();
    const auto after = replay.m_pipeline->status();
    REQUIRE( after.m_wfsReceived == before.m_wfsReceived + 1 );
    REQUIRE( after.m_standard.m_updates == before.m_standard.m_updates + 1 );
    REQUIRE( after.m_assisted.m_updates == before.m_assisted.m_updates + 1 );
    REQUIRE( after.m_fastOpportunities == before.m_fastOpportunities + 4 );
}
/** \brief Learning freezes both fitted banks without stopping inference; reset revokes every old coefficient. */
TEST_CASE( "Native learning freeze preserves fitted coefficients and prediction; reset revokes them",
           "[multirate][phase][output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::learning;
    DDSPC::PhaseShadow::reset_models;
#endif
    // clang-format on
    PhaseReplay replay( GENERATE( false, true ) );
    for( unsigned n = 0; n < 700; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_standard.m_ready );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    const auto before   = replay.m_pipeline->status();
    const auto standard = replay.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 );
    const auto assisted = replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 3 );
    replay.m_pipeline->learning( false );
    for( unsigned n = 0; n < 200; ++n )
        replay.step();
    const auto frozen = replay.m_pipeline->status();
    REQUIRE( frozen.m_standard.m_updates == before.m_standard.m_updates );
    REQUIRE( frozen.m_assisted.m_updates == before.m_assisted.m_updates );
    REQUIRE( replay.m_pipeline->coefficients( DDSPC::ShadowMode::Standard, 0 ) == standard );
    REQUIRE( replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 3 ) == assisted );
    REQUIRE( frozen.m_fastPredictions > before.m_fastPredictions );
    REQUIRE( frozen.m_standard.m_ready );
    replay.m_pipeline->reset_models();
    replay.step();
    const auto reset = replay.m_pipeline->status();
    REQUIRE_FALSE( reset.m_standard.m_ready );
    REQUIRE_FALSE( reset.m_assisted.m_ready );
    REQUIRE( reset.m_standard.m_updates == 0 );
    replay.m_pipeline->learning( true );
    for( unsigned n = 0; n < 700; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
}
/** \brief Actual advertised timing is used directly, while a changed WFS-to-slot phase cannot silently reuse a fit. */
TEST_CASE( "Native phase receiver accepts nominal periods and rejects changed slot association", "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::schedule;
#endif
    // clang-format on
    PhaseReplay replay( true );
    replay.m_period = 500004;
    replay.m_probe  = true;
    for( unsigned n = 0; n < 700; ++n )
        replay.step( 5 );
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
    replay.step();
    REQUIRE( replay.m_pipeline->status().m_shadowSlots > 1000 );
    const auto prepared = replay.m_pipeline->status().m_preparedPlans;
    replay.m_rotated    = true;
    replay.m_probe      = false;
    for( unsigned n = 0; n < 8; ++n )
        replay.step( 5 ); // Intentional command-slot fault; model readiness does not certify the current context.
    REQUIRE_FALSE( replay.m_pipeline->status().m_qualified );
    REQUIRE_FALSE( replay.m_pipeline->status().m_assisted.m_ready );
    REQUIRE( replay.m_pipeline->status().m_preparedPlans == prepared );
    replay.m_rotated = false;
    for( unsigned n = 0; n < 500; ++n )
        replay.step();
    REQUIRE( replay.m_pipeline->status().m_assisted.m_ready );
}
/** \brief ADC phase and the next unexpired DM slot differ after delivery latency; use cached actual-prefix gains. */
TEST_CASE( "ADC phase uses the actual next synco opportunity without retagging an expired command",
           "[multirate][phase]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PhaseShadow::observe;
    DDSPC::SyncoShadowProducer::prepare_slot;
#endif
    // clang-format on
    PhaseReplay replay( true );
    for( unsigned n = 0; n < 700; ++n )
        replay.step();
    replay.m_pipeline->request( DDSPC::ShadowMode::AccelAssisted );
    const auto         k = replay.m_frame + 1;
    const std::int64_t t = replay.m_base + 500000LL * std::int64_t( k - 1 );
    auto               w = observation( DDSPC::ObservationSource::Modal, k, t );
    w.m_values           = { replay.m_y[0], replay.m_y[1] };
    replay.m_pipeline->observe( w );
    replay.m_pipeline->sync( { { 77, true, true, t + 1 }, 4 * k, k, 0 } );
    const auto s = schedule( k, t );
    replay.m_pipeline->schedule( s, t + 2 );
    while( replay.m_pipeline->learn() )
    {
    }
    const std::array<float, 2> command{ .1, .2 };
    replay.command( command, t + 5000 );
    replay.m_pipeline->command_poll( true, t + 105000 );
    replay.m_pipeline->tick( t + 105000 );
    auto a             = observation( DDSPC::ObservationSource::Adc, 4 * ( k - 1 ) + 1, t + 150000 );
    a.m_receivedNs     = t + 20000;
    a.m_values         = { .3, -.2 };
    const auto version = replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 0 );
    replay.m_pipeline->observe( a );
    REQUIRE( replay.m_pipeline->coefficients( DDSPC::ShadowMode::AccelAssisted, 0 ) == version );
    std::array<float, 2> pixels{};
    const auto           next = synco::request( s, 1 );
    if( replay.m_pipeline->status().m_fastUs < 55 )
        REQUIRE( replay.m_pipeline->producer()->command( next, s, t + 210000, pixels ) );
    REQUIRE_FALSE( replay.m_pipeline->producer()->command( next, s, next.cutoffNs, pixels ) );
    REQUIRE_FALSE( replay.m_pipeline->producer()->command( synco::request( s, 0 ), s, t + 210000, pixels ) );
    auto target     = observation( DDSPC::ObservationSource::Modal, k + 1, t + 500000 );
    target.m_values = { .1, .2 };
    replay.m_pipeline->observe( target );
    DDSPC::ShadowPrediction score;
    unsigned                checked = 0;
    while( replay.m_pipeline->read_prediction( score ) )
        if( score.m_origin == k && score.m_adcCounter == a.m_counter )
        {
            REQUIRE( score.m_phase == 0 );
            REQUIRE( score.m_slotStep == 1 );
            REQUIRE( score.m_slotCutoffNs == next.cutoffNs );
            ++checked;
        }
    REQUIRE( checked == 4 );
    replay.m_pipeline->inhibit();
    auto producer = replay.m_pipeline->producer();
    REQUIRE_FALSE( producer->prepare_slot( DDSPC::ShadowMode::Standard,
                                           command,
                                           replay.m_inputs.m_calibration.m_shapes,
                                           next,
                                           s,
                                           t + 160000,
                                           t + 150000 ) );
    producer->models( DDSPC::ShadowMode::AccelAssisted, true, true );
    REQUIRE_FALSE( producer->command( next, s, t + 210000, pixels ) );
    REQUIRE( ddspc_test_output_calls() == 0 );
}
} // namespace libXWCTest::loPredCtrlAccTest
