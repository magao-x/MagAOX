/** \file accelFFCtrlShadow_test.cpp
 * \brief Offline absolute proposal, coordinate, provenance and lifecycle tests without any stream API.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "accelFFCtrlCancellation_test.hpp"
#include "accelFFCtrlAO_test.hpp"
#include "../offline/accelFFCtrlShadow.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace cancellation;

/** \brief Explicit coordinate/modal maps give absolute coefficients, including reflections and calibrated order.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl shadow absolute coordinate and modal conventions", "[accelFFCtrl][shadow]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::shadowCommand::propose;
#endif
    // clang-format on
    const calibration cal;
    Eigen::Matrix2d   coordinates;
    coordinates << 0, 1, -1, 0;
    shadowCommand             shadow( coordinates, cal.m_mapping, 42, shadowCalibration::synthetic, controls( .5 ) );
    std::array<prediction, 1> p{ idealPrediction( { 2e-9, -3e-9 }, origin, 1000000 ) };
    const auto                first = shadow.propose( p, origin ), again = shadow.propose( p, origin );
    REQUIRE( first.m_proposal.m_state == correctionState::active );
    REQUIRE( first.m_proposal.m_modal == again.m_proposal.m_modal );
    REQUIRE( again.m_record == first.m_record + 1 );
    REQUIRE( first.m_dmPointing == std::array<double, 2>{ -3e-9, -2e-9 } );
    const Eigen::Vector2d expected = -.5 * cal.m_mapping * coordinates * Eigen::Vector2d( 2e-9, -3e-9 );
    REQUIRE( first.m_proposal.m_modal[0] == Approx( expected[0] ) );
    REQUIRE( first.m_proposal.m_modal[1] == Approx( expected[1] ) );
    REQUIRE( first.m_queryNs == origin );
    REQUIRE( first.m_sampleNs == origin - period );
    REQUIRE( first.m_predictionTargetNs == origin + 1000000 );
    REQUIRE( first.m_requestedTargetNs == first.m_predictionTargetNs );
    REQUIRE( first.m_calibrationId == 42 );
    // An 8-kHz source naturally advances four counters between 2-kHz queries; this is not a lost-frame event.
    p[0]            = idealPrediction( { 2e-9, -3e-9 }, origin + 4 * period, 1000000 );
    const auto next = shadow.propose( p, origin + 4 * period );
    REQUIRE( next.m_proposal.m_modal == first.m_proposal.m_modal );
    REQUIRE( next.m_fault == shadowFault::none );
    Eigen::Matrix2d reflection;
    reflection << 0, 1, 1, 0;
    shadowCommand mirrored( reflection, cal.m_mapping, 43, shadowCalibration::synthetic, controls( .5 ) );
    const auto    reflected = mirrored.propose( p, origin + 4 * period );
    REQUIRE( reflected.m_dmPointing == std::array<double, 2>{ -3e-9, 2e-9 } );
}

/** \brief Default gain is zero; unavailable or malformed predictions replace prior nonzero proposals with zeros.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl shadow inhibition gain and clipping", "[accelFFCtrl][shadow]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::shadowCommand::propose;
#endif
    // clang-format on
    const calibration         cal;
    const auto                identity = Eigen::Matrix2d::Identity();
    shadowCommand             disabled( identity, cal.m_mapping, 1, shadowCalibration::synthetic );
    std::array<prediction, 1> p{ idealPrediction( { 1e-6, -2e-6 }, origin, 1000000 ) };
    REQUIRE( disabled.propose( p, origin ).m_proposal.m_modal == std::array<double, 2>{ 0, 0 } );
    shadowCommand zeroGain( identity, cal.m_mapping, 1, shadowCalibration::synthetic, controls( 0 ) );
    const auto    audited = zeroGain.propose( p, origin );
    REQUIRE( audited.m_validPrediction );
    REQUIRE( audited.m_pointing == p[0].m_pointing );
    REQUIRE( audited.m_proposal.m_state == correctionState::disabled );
    auto c       = controls( 1 );
    c.m_maxModal = { .001, .002 };
    shadowCommand shadow( identity, cal.m_mapping, 2, shadowCalibration::synthetic, c );
    const auto    valid = shadow.propose( p, origin );
    REQUIRE( valid.m_proposal.m_state == correctionState::limited );
    for( size_t axis = 0; axis < 2; ++axis )
        REQUIRE( std::abs( valid.m_proposal.m_modal[axis] ) <= c.m_maxModal[axis] );
    for( unsigned fault = 0; fault < 5; ++fault )
    {
        auto bad = p;
        if( fault == 0 )
            bad[0].m_state = estimatorState::warming;
        if( fault == 1 )
            bad[0].m_pointing[1] = NAN;
        if( fault == 2 )
            bad[0].m_sampleTimeNs = origin - 2000001;
        if( fault == 3 )
            ++bad[0].m_targetTimeNs;
        if( fault == 4 )
            bad[0].m_sampleTimeNs = origin + 1;
        const auto result = shadow.propose( bad, origin );
        REQUIRE( result.m_proposal.m_state == correctionState::inhibited );
        REQUIRE( result.m_proposal.m_modal == std::array<double, 2>{ 0, 0 } );
        REQUIRE( result.m_requestedTargetNs == origin + c.m_horizonNs );
    }
    REQUIRE_THROWS_AS( shadowCommand( Eigen::Matrix2d::Zero(), cal.m_mapping, 1, shadowCalibration::synthetic, c ),
                       std::invalid_argument );
    REQUIRE_THROWS_AS( shadowCommand( identity, cal.m_mapping, 0, shadowCalibration::synthetic, c ),
                       std::invalid_argument );
}

/** \brief Source/query rollback latches; reset requires a fresh post-reset acquisition before a new proposal.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl shadow restart and reset barriers", "[accelFFCtrl][shadow]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::shadowCommand::reset;
    MagAOX::app::accelFFCtrlOffline::shadowCommand::propose;
#endif
    // clang-format on
    const calibration cal;
    for( unsigned fault = 0; fault < 5; ++fault )
    {
        shadowCommand shadow(
            Eigen::Matrix2d::Identity(), cal.m_mapping, 1, shadowCalibration::synthetic, controls( 1 ) );
        std::array<prediction, 1> p{ idealPrediction( { 1e-9, 0 }, origin, 1000000 ) };
        p[0].m_sequence = 100;
        REQUIRE( shadow.propose( p, origin ).m_proposal.m_state == correctionState::active );
        auto bad = p;
        if( fault == 0 )
            ++bad[0].m_generation;
        if( fault == 1 )
            --bad[0].m_sequence;
        if( fault == 3 )
            --bad[0].m_sampleTimeNs;
        if( fault == 4 )
            ++bad[0].m_sequence;
        const auto failed = shadow.propose( bad, origin - ( fault == 2 ) );
        REQUIRE( failed.m_fault == ( fault == 0 ? shadowFault::sourceChange
                                     : fault == 1
                                         ? shadowFault::counterRollback
                                         : ( fault == 2 ? shadowFault::queryRollback : shadowFault::sampleChange ) ) );
        REQUIRE( failed.m_proposal.m_modal == std::array<double, 2>{ 0, 0 } );
        REQUIRE( shadow.propose( p, origin ).m_proposal.m_state == correctionState::inhibited );
        shadow.reset( origin, 7 );
        const auto barrier = shadow.propose( p, origin );
        REQUIRE( barrier.m_fault == shadowFault::resetBarrier );
        REQUIRE( barrier.m_proposal.m_modal == std::array<double, 2>{ 0, 0 } );
        p[0]              = idealPrediction( { 1e-9, 0 }, origin + 4 * period, 1000000 );
        p[0].m_generation = 2;
        const auto fresh  = shadow.propose( p, origin + 4 * period );
        REQUIRE( fresh.m_proposal.m_state == correctionState::active );
        REQUIRE( fresh.m_epoch == 1 );
        REQUIRE( fresh.m_inputRevision == 7 );
        REQUIRE_THROWS_AS( shadow.reset( origin, 8 ), std::invalid_argument );
    }
}

/** \brief Independent archived-FITS fixture verifies numerical DM-pupil sign/scale only, not WFS registration.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl archived pupil shadow calibration fixture", "[.][accelFFCtrl-shadow-calibration]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::shadowCommand::propose;
#endif
    // clang-format on
    const char *path = std::getenv( "ACCELFFCTRL_SHADOW_CALIBRATION" );
    REQUIRE( path != nullptr );
    std::ifstream   input( path );
    Eigen::Matrix2d mapping;
    for( size_t row = 0; row < 2; ++row )
        for( size_t col = 0; col < 2; ++col )
            input >> mapping( row, col );
    size_t count = 0;
    input >> count;
    REQUIRE( count > 0 );
    shadowCommand shadow(
        Eigen::Matrix2d::Identity(), mapping, 2, shadowCalibration::archivedPupilOnly, controls( 1 ) );
    for( size_t n = 0; n < count; ++n )
    {
        std::array<double, 2> theta, expected;
        input >> theta[0] >> theta[1] >> expected[0] >> expected[1];
        REQUIRE( input.good() );
        const std::array<prediction, 1> p{ idealPrediction( theta, origin + n * 4 * period, 1000000 ) };
        const auto                      proposed = shadow.propose( p, origin + n * 4 * period );
        REQUIRE( proposed.m_proposal.m_state == correctionState::active );
        REQUIRE( proposed.m_evidence == shadowCalibration::archivedPupilOnly );
        for( size_t axis = 0; axis < 2; ++axis )
            REQUIRE( proposed.m_proposal.m_modal[axis] == Approx( expected[axis] ).margin( 1e-14 ) );
    }
}

/** \brief A synthetic ideal-plant integrator has the expected causal DC decay; no MagAO-X stability is inferred.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl synthetic AO integrator dynamics", "[accelFFCtrl][shadow]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlOffline::dmResponse::step;
#endif
    // clang-format on
    const calibration cal;
    for( double gain : { 0., .1, .3 } )
    {
        syntheticIntegrator   ao( cal.m_mapping, gain );
        dmResponse            plant( cal.m_plant, period, 0, 0 );
        std::array<double, 2> previous{ 1e-9, -2e-9 };
        for( unsigned n = 0; n < 80; ++n )
        {
            const auto cmd    = ao.update( previous );
            const auto effect = plant.step( { cmd[0], cmd[1] } );
            previous          = { 1e-9 + effect[0], -2e-9 + effect[1] };
            REQUIRE( previous[0] == Approx( 1e-9 * std::pow( 1 - gain, n + 1 ) ).margin( 1e-23 ) );
            REQUIRE( previous[1] == Approx( -2e-9 * std::pow( 1 - gain, n + 1 ) ).margin( 1e-23 ) );
        }
        ao.reset();
        REQUIRE( ao.update( { 0, 0 } ).norm() == 0 );
    }
}

/** \brief Synthetic saturation feedback changes only the test integrator; default gain and invalid-data checks hold.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl synthetic AO state and back calculation", "[accelFFCtrl][shadow]" )
{
    syntheticIntegrator disabled( Eigen::Matrix2d::Identity() );
    REQUIRE( disabled.update( { 1, 2 } ).norm() == 0 );
    syntheticIntegrator ao( Eigen::Matrix2d::Identity(), .5 );
    const auto          raw = ao.update( { 1, -2 } );
    REQUIRE( raw[0] == -.5 );
    REQUIRE( raw[1] == 1 );
    ao.track( raw, .5 );
    const auto next = ao.update( { 0, 0 } );
    REQUIRE( next[0] == -.25 );
    REQUIRE( next[1] == .5 );
    REQUIRE_THROWS_AS( ao.update( { NAN, 0 } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ao.track( Eigen::Vector2d::Zero(), NAN ), std::invalid_argument );
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
