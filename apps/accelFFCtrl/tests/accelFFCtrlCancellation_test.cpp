/** \file accelFFCtrlCancellation_test.cpp
 * \brief Offline calibration, inhibition, delayed response and analytic residual tests; no native interfaces.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "accelFFCtrlCancellation_test.hpp"
#include "accelFFCtrlGap_test.hpp"

#include <complex>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace cancellation;

/** \brief The existing TT projector gives explicit rotated surface/modal units, opposing sign and exact gain.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline correction calibration and sign", "[accelFFCtrl][cancellation]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::tipTiltProjection::response;
    MagAOX::app::accelFFCtrlOffline::feedForward::propose;
#endif
    // clang-format on
    const calibration cal;
    Eigen::Matrix2d   expected;
    expected << std::cos( pi / 6 ), std::sin( pi / 6 ), -std::sin( pi / 6 ), std::cos( pi / 6 );
    REQUIRE( ( cal.m_mapping - 6.5e6 / 8 * expected ).norm() < 1e-9 );
    REQUIRE( ( cal.m_modes.transpose() * cal.m_modes / 8 - Eigen::Matrix2d::Identity() ).norm() < 1e-14 );
    REQUIRE( ( cal.m_plant * cal.m_mapping - Eigen::Matrix2d::Identity() ).norm() < 1e-14 );
    const std::array<prediction, 1> p{ idealPrediction( { 2e-9, -3e-9 }, origin, 1000000 ) };
    for( double gain : { 0., .25, .5, 1. } )
    {
        const feedForward     controller( cal.m_mapping, controls( gain ) );
        const auto            command  = controller.propose( p, origin );
        const Eigen::Vector2d physical = cal.m_plant * Eigen::Vector2d( command.m_modal[0], command.m_modal[1] );
        REQUIRE( physical[0] == Approx( -gain * 2e-9 ).margin( 1e-22 ) );
        REQUIRE( physical[1] == Approx( gain * 3e-9 ).margin( 1e-22 ) );
        REQUIRE( command.m_state == ( gain ? correctionState::active : correctionState::disabled ) );
    }
    correctionParameters disabled;
    const feedForward    zero( cal.m_mapping, disabled );
    REQUIRE( zero.propose( {}, origin ).m_modal == std::array<double, 2>{ 0, 0 } );
    REQUIRE( zero.propose( p, origin ).m_state == correctionState::disabled );
    const feedForward     reversed( -cal.m_mapping, controls( 1 ) );
    const auto            wrong = reversed.propose( p, origin );
    const Eigen::Vector2d residual =
        Eigen::Vector2d( 2e-9, -3e-9 ) + cal.m_plant * Eigen::Vector2d( wrong.m_modal[0], wrong.m_modal[1] );
    REQUIRE( residual.norm() == Approx( 2 * std::hypot( 2e-9, 3e-9 ) ) );
}

/** \brief Bounds preserve modal direction; invalid configuration or prediction always produces a new zero proposal.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline correction bounds and inhibition", "[accelFFCtrl][cancellation]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::feedForward::propose;
#endif
    // clang-format on
    const calibration cal;
    auto              config = controls( 1 );
    config.m_maxModal        = { .0001, .0002 };
    const feedForward         bounded( cal.m_mapping, config );
    std::array<prediction, 2> p{ idealPrediction( { 2e-9, -3e-9 }, origin, 1000000 ),
                                 idealPrediction( { 1e-9, 2e-9 }, origin, 1000000 ) };
    const auto                command = bounded.propose( p, origin );
    REQUIRE( command.m_state == correctionState::limited );
    for( size_t axis = 0; axis < 2; ++axis )
        REQUIRE( std::abs( command.m_modal[axis] ) <= config.m_maxModal[axis] );
    const Eigen::Vector2d unrestricted = -cal.m_mapping * Eigen::Vector2d( 3e-9, -1e-9 );
    REQUIRE( command.m_modal[0] == Approx( command.m_scale * unrestricted[0] ) );
    REQUIRE( command.m_modal[1] == Approx( command.m_scale * unrestricted[1] ) );
    for( unsigned fault = 0; fault < 10; ++fault )
    {
        auto bad = p;
        if( fault == 0 )
            bad[1].m_state = estimatorState::warming;
        if( fault == 1 )
            bad[1].m_pointing[0] = NAN;
        if( fault == 2 )
            bad[1].m_sampleTimeNs = origin - 2000001;
        if( fault == 3 )
            bad[1].m_sampleTimeNs = origin + 1;
        if( fault == 4 )
            ++bad[1].m_targetTimeNs;
        if( fault == 5 )
            ++bad[1].m_sequence;
        if( fault == 6 )
            ++bad[1].m_generation;
        if( fault == 7 )
            bad[1].m_rcond = 0;
        if( fault == 8 )
            bad[1].m_phase[0] = INFINITY;
        if( fault == 9 )
            --bad[1].m_sampleTimeNs;
        const auto inhibited = bounded.propose( bad, origin );
        REQUIRE( inhibited.m_state == correctionState::inhibited );
        REQUIRE( inhibited.m_modal == std::array<double, 2>{ 0, 0 } );
        REQUIRE( inhibited.m_targetNs == 0 );
    }
    REQUIRE( bounded.propose( {}, origin ).m_state == correctionState::inhibited );
    for( double gain : { -1., 1.1, double( NAN ) } )
    {
        auto invalid   = config;
        invalid.m_gain = gain;
        REQUIRE_THROWS_AS( feedForward( cal.m_mapping, invalid ), std::invalid_argument );
    }
    config.m_maxModal[0] = -1;
    REQUIRE_THROWS_AS( feedForward( cal.m_mapping, config ), std::invalid_argument );
    REQUIRE_THROWS_AS( feedForward( Eigen::Matrix2d::Zero() ), std::invalid_argument );
}

/** \brief The plant cannot respond before an issued delayed command and its exact first-order ZOH interval elapses.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline DM response delay and bandwidth", "[accelFFCtrl][cancellation]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::dmResponse::step;
    MagAOX::app::accelFFCtrlOffline::dmResponse::reset;
#endif
    // clang-format on
    const auto identity = Eigen::Matrix2d::Identity();
    dmResponse ideal( identity, period, 3 * period, 0 ), finite( identity, period, 3 * period, 300 );
    for( int n = 0; n < 50; ++n )
    {
        const auto a = ideal.step( { 1, -2 } ), b = finite.step( { 1, -2 } );
        REQUIRE( a[0] == ( n >= 3 ? 1 : 0 ) );
        const double expected = n > 3 ? -std::expm1( -2 * pi * 300 * ( n - 3 ) * period * 1e-9 ) : 0;
        REQUIRE( b[0] == Approx( expected ).margin( 1e-14 ) );
        REQUIRE( b[1] == Approx( -2 * expected ).margin( 1e-14 ) );
    }
    REQUIRE_THROWS_AS( finite.step( { NAN, 0 } ), std::invalid_argument );
    REQUIRE( finite.step( { 0, 0 } ) == std::array<double, 2>{ 0, 0 } );
    REQUIRE_THROWS_AS( dmResponse( identity, period, period + 1, 300 ), std::invalid_argument );
    REQUIRE_THROWS_AS( dmResponse( identity, period, 0, -1 ), std::invalid_argument );
    // Independent steady-state transfer of the causal finite-bandwidth plant, including its extra interval.
    for( double frequency : { 28., 56. } )
    {
        dmResponse           response( identity, period, 8 * period, 300 );
        std::complex<double> measured{};
        for( int n = 0; n < 24000; ++n )
        {
            const double phase = 2 * pi * frequency * n * period * 1e-9;
            const auto   value = response.step( { std::cos( phase ), 0 } );
            if( n >= 8000 )
                measured += value[0] * std::exp( std::complex<double>( 0, -phase ) ) / 8000.;
        }
        const double a        = std::exp( -2 * pi * 300 * period * 1e-9 );
        const auto   z        = std::exp( std::complex<double>( 0, -2 * pi * frequency * period * 1e-9 ) );
        const auto   expected = ( 1 - a ) * std::pow( z, 9 ) / ( 1. - a * z );
        REQUIRE( std::abs( measured - expected ) < 1e-12 );
    }
}

/** \brief Perfect target prediction with a delayed ideal DM matches analytic sinusoidal cancellation and ZOH loss.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline sinusoidal residual oracle", "[accelFFCtrl][cancellation]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::feedForward::propose;
    MagAOX::app::accelFFCtrlOffline::dmResponse::step;
#endif
    // clang-format on
    const calibration cal;
    for( double frequency : { 28., 56. } )
        for( double gain : { 0., .5, 1. } )
        {
            const feedForward     controller( cal.m_mapping, controls( gain, 500000 ) );
            dmResponse            plant( cal.m_plant, period, 500000, 0 );
            std::array<double, 2> held{};
            double                before = 0, after = 0;
            for( int n = 0; n < 24000; ++n )
            {
                const int64_t now = origin + n * period;
                if( n % 4 == 0 )
                {
                    const double theta = 1e-8 * std::cos( 2 * pi * frequency * ( now + 500000 - origin ) * 1e-9 );
                    const std::array<prediction, 1> oracle{ idealPrediction( { theta, -theta }, now, 500000 ) };
                    held = controller.propose( oracle, now ).m_modal;
                }
                const auto   effect = plant.step( held );
                const double truth  = 1e-8 * std::cos( 2 * pi * frequency * n * period * 1e-9 );
                if( n >= 8000 )
                {
                    before += truth * truth;
                    after += std::pow( truth + effect[0], 2 );
                }
            }
            double analytic = 0;
            for( int phase = 0; phase < 4; ++phase )
                analytic += 1 + gain * gain - 2 * gain * std::cos( 2 * pi * frequency * phase * period * 1e-9 );
            REQUIRE( after / before == Approx( analytic / 4 ).margin( 1e-12 ) );
        }
}

/** \brief Raw-only production input, two harmonic fits and the synthetic DM suppress two causal coherent tones.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline causal two-tone cancellation", "[accelFFCtrl][cancellation]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::predict;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::reset;
    MagAOX::app::accelFFCtrlOffline::feedForward::propose;
    MagAOX::app::accelFFCtrlOffline::dmResponse::step;
#endif
    // clang-format on
    const calibration cal;
    parameters        c;
    c.m_predictionNs = 1000000;
    c.m_frequency    = 28;
    c.m_voltsPerBit.fill( 1e-6 );
    c.m_sensitivity.fill( 1 );
    inputProcessor low( c );
    c.m_frequency = 56;
    inputProcessor high( c );
    low.attached();
    high.attached();
    const feedForward       controller( cal.m_mapping, controls( 1 ) );
    dmResponse              plant( cal.m_plant, period, 1000000, 0 );
    gapValidation::snapshot source;
    source.m_header = { 2, 2, 1, 1, 0, 0, true, true, true, false };
    std::array<double, 2> held{};
    double                before = 0, after = 0;
    unsigned              inhibited = 0;
    sim::parameters       adc;
    adc.m_voltsPerBit = 1e-6;
    adc.m_sensitivity = 1;
    uint64_t lowClips = 0, highClips = 0;
    for( int n = 0; n < 16000; ++n )
    {
        const int64_t now = origin + n * period;
        if( n == 8000 )
        {
            low.reset();
            high.reset();
        }
        if( n % 4 == 0 )
        {
            low.predict( now );
            high.predict( now );
            const std::array<prediction, 2> p{ low.diagnostics().m_prediction, high.diagnostics().m_prediction };
            const auto                      command = controller.propose( p, now );
            held                                    = command.m_modal;
            if( command.m_state == correctionState::inhibited )
            {
                ++inhibited;
                REQUIRE( held == std::array<double, 2>{ 0, 0 } );
            }
        }
        const auto effect = plant.step( held );
        double     theta = 0, acceleration = 0;
        for( double f : { 28., 56. } )
        {
            const double value = 1e-9 * std::cos( 2 * pi * f * n * period * 1e-9 );
            theta += value;
            acceleration -= std::pow( 2 * pi * f, 2 ) * 10 * value;
        }
        if( ( n >= 4000 && n < 8000 ) || n >= 12000 )
        {
            before += theta * theta;
            after += std::pow( theta + effect[0], 2 );
        }
        const auto code           = sim::adcCounts( acceleration, adc, lowClips, highClips );
        source.m_header.m_counter = n + 1;
        source.m_frame            = { { code, code },
                                      { ( epoch + now ) / 1000000000, ( epoch + now ) % 1000000000 },
                                      { ( epoch + now ) / 1000000000, ( epoch + now ) % 1000000000 },
                                      uint64_t( n + 1 ) };
        const inputClock clock{ epoch + now, now, 0 };
        low.drain( source, [clock] { return clock; } );
        high.drain( source, [clock] { return clock; } );
    }
    REQUIRE( inhibited > 1000 );
    REQUIRE( after / before < .02 );
    REQUIRE( lowClips + highClips == 0 );
    REQUIRE( low.diagnostics().m_resets == 2 );
}
} // namespace accelFFCtrlTest
} // namespace libXWCTest
