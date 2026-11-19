/** \file accelSimDMOutput_test.cpp
 * \brief Hardware-independent DM synthesis and existing producer acknowledgement interlock regressions.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimDM.hpp"
#include "../../../libMagAOX/app/syncoOutput.hpp"

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app;
using namespace MagAOX::app::accelSimDetail;

/** \brief Scaled same-sample commands and their removal need receipts; recovered timing alone never rearms output.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM scaled output and acknowledged removal", "[accelSim][dm-output]" )
{
    const bool timingFault = GENERATE( false, true );
    // Exercise the actual producer interlock with explicit copy receipts, without native streams or a clock thread.
    SyncoOutputState    output( { true, true, 0, 0, 0, true }, 2500, true );
    dmInjection         dm;
    std::vector<double> x( 2500 ), y( 2500 ), mask( 2500, 1 );
    std::vector<float>  pixels( 2500 ), zero( 2500 );
    for( size_t p = 0; p < 2500; ++p )
    {
        x[p] = double( p % 50 ) - 24.5;
        y[p] = double( p / 50 ) - 24.5;
    }
    dm.prepare( x, y, mask, 1e-6 );
    auto stage = [&]( const std::vector<float> &values )
    {
        pixels = values;
        return true;
    };
    parameters config;
    config.m_fps = 8000;
    model motion;
    motion.configure( config, referenceModes( 123 ) );
    REQUIRE( dm.service( motion.m_sample, 0, 0, false, stage ) == dmState::disabled );
    output.check( 0, false );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE_FALSE( output.status().m_fault );
    REQUIRE( output.status().m_published == 0 );
    REQUIRE( output.status().m_acknowledgements == 0 );

    REQUIRE( output.request( true, true ) );
    output.attached();
    REQUIRE( output.zero_needed() );
    REQUIRE( output.validate( zero, true ) );
    output.submitted( zero, true, 0 );
    REQUIRE_FALSE( output.status().m_zeroConfirmed );
    output.acknowledge( true );
    REQUIRE( output.status().m_zeroConfirmed );
    for( int64_t n = 1; n <= 8000; ++n )
    {
        motion.step();
        REQUIRE( dm.service( motion.m_sample, 3, .25, true, stage ) == dmState::active );
        const pointingDisturbance pointing( motion.m_sample );
        REQUIRE( pixels[0] == Approx( .25 * pointing.surface( -24.5 * 6.5 / 48, -24.5 * 6.5 / 48 ) ).margin( 1e-8 ) );
        REQUIRE( output.validate( pixels, false ) );
        output.submitted( pixels, false, n * 125000 );
        REQUIRE( output.pending() );
        REQUIRE_FALSE( output.validate( pixels, false ) ); // No reuse before the consumer receipt.
        output.acknowledge( true );
        REQUIRE( output.status().m_acknowledged );
    }
    REQUIRE( output.status().m_acknowledgements == 8001 );
    REQUIRE_FALSE( output.status().m_zeroConfirmed );
    if( timingFault )
    {
        REQUIRE( dm.service( motion.m_sample, 3, .25, false, stage ) == dmState::timingFault );
        output.check( 1000125000, false );
        REQUIRE( output.status().m_fault );
    }
    else
    {
        REQUIRE( dm.service( motion.m_sample, 4, .25, true, stage ) == dmState::disabled );
        REQUIRE( output.request( false, true ) );
    }
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( pixels == zero );
    REQUIRE( output.zero_needed() );
    REQUIRE_FALSE( output.status().m_zeroConfirmed );
    REQUIRE( output.validate( pixels, true ) );
    output.submitted( pixels, true, 1000250000 );
    REQUIRE_FALSE( output.status().m_zeroConfirmed );
    output.acknowledge( true );
    REQUIRE( output.status().m_zeroConfirmed );
    REQUIRE_FALSE( output.status().m_acknowledged );
    REQUIRE( output.status().m_acknowledgements == 8002 );
    output.check( 1000375000, true );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( dm.service( motion.m_sample, timingFault ? 3 : 4, 1, true, stage ) ==
             ( timingFault ? dmState::timingFault : dmState::disabled ) );
    REQUIRE( pixels == zero );
    REQUIRE( dm.service( motion.m_sample, 7, .25, true, stage ) == dmState::active );
    REQUIRE( output.request( true, true ) ); // Explicit re-enable after removal and restored qualification.
    REQUIRE_FALSE( output.status().m_fault );
    REQUIRE( output.validate( pixels, false ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::SyncoOutputState::check;
    MagAOX::app::SyncoOutputState::request;
    MagAOX::app::SyncoOutputState::validate;
    MagAOX::app::SyncoOutputState::submitted;
    MagAOX::app::SyncoOutputState::acknowledge;
#endif
    // clang-format on
}

} // namespace accelSimTest
} // namespace libXWCTest
