/** \file ddspc_output_test.cpp
 * \brief Deterministic own-channel enable, bounds, acknowledgement and zero-removal regression.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/app/syncoOutput.hpp"
#include <array>
#include <limits>

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Only explicit qualified requests authorize a bounded own-channel contribution; startup is inert. */
TEST_CASE( "Operational output requires explicit authority and readiness", "[multirate][output]" )
{
    MagAOX::app::SyncoOutputConfig c{ true, true, 1, .5, 1000000 };
    MagAOX::app::SyncoOutputState  output( c, 2, true ), shadow( c, 2, false );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE_FALSE( output.status().m_acknowledged );
    REQUIRE_FALSE( output.request( true, false ) );
    REQUIRE_FALSE( shadow.request( true, true ) );
    REQUIRE( output.request( true, true ) );
    const std::array<float, 2> value{ .2f, -.1f };
    REQUIRE( output.validate( value, false ) );
    output.submitted( value, false, 100 );
    REQUIRE_FALSE( output.status().m_acknowledged );
    output.acknowledge( true );
    REQUIRE( output.status().m_acknowledged );
    REQUIRE( output.request( false, false ) );
    REQUIRE( output.zero_needed() );
    REQUIRE( output.status().m_acknowledged ); // The last copied contribution remains until zero is acknowledged.
    const std::array<float, 2> zero{};
    REQUIRE( output.validate( zero, true ) );
    output.submitted( zero, true, 200 );
    REQUIRE_FALSE( output.status().m_zeroConfirmed );
    output.acknowledge( true );
    REQUIRE( output.status().m_zeroConfirmed );
}
/** \brief Independent owners can remove their own contribution without changing the other provider. */
TEST_CASE( "Disturbance and correction output states remain independent", "[multirate][output]" )
{
    MagAOX::app::SyncoOutputConfig c{ true, true, 1, 1, 1000000 };
    MagAOX::app::SyncoOutputState  disturbance( c, 2, true ), correction( c, 2, true );
    const std::array<float, 2>     a{ .3f, .1f }, b{ -.1f, .2f }, zero{};
    REQUIRE( disturbance.request( true, true ) );
    REQUIRE( correction.request( true, true ) );
    disturbance.submitted( a, false, 100 );
    disturbance.acknowledge( true );
    correction.submitted( b, false, 100 );
    correction.acknowledge( true );
    correction.request( false, false );
    correction.submitted( zero, true, 200 );
    correction.acknowledge( true );
    REQUIRE( disturbance.status().m_acknowledged );
    REQUIRE( correction.status().m_zeroConfirmed );
    disturbance.request( false, false );
    disturbance.submitted( zero, true, 300 );
    disturbance.acknowledge( true );
    REQUIRE( disturbance.status().m_zeroConfirmed );
    REQUIRE_FALSE( correction.status().m_requested );
}
/** \brief Invalid commands, discarded payloads and missing acknowledgements revoke active intent until explicit
 * recovery. */
TEST_CASE( "Operational faults require removal and an explicit new request", "[multirate][output]" )
{
    MagAOX::app::SyncoOutputConfig c{ true, true, 1, .2, 1000 };
    MagAOX::app::SyncoOutputState  output( c, 2, true );
    const std::array<float, 2>     value{ .1f, .1f }, bad{ 2, 0 }, zero{};
    REQUIRE( output.request( true, true ) );
    REQUIRE_FALSE( output.validate( bad, false ) );
    REQUIRE( output.status().m_fault );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE_FALSE( output.request( true, true ) );
    REQUIRE( output.validate( zero, true ) );
    output.submitted( zero, true, 100 );
    output.acknowledge( true );
    REQUIRE( output.request( true, true ) );
    output.submitted( value, false, 200 );
    output.check( 1201, true );
    REQUIRE( output.status().m_fault );
    REQUIRE_FALSE( output.status().m_requested );
    output.acknowledge( false );
    REQUIRE( output.zero_needed() );
    output.submitted( zero, true, 1300 );
    output.acknowledge( true );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( output.clear_fault() );
    REQUIRE( output.request( true, true ) );
    output.check( 1400, false );
    REQUIRE_FALSE( output.status().m_requested );
}
/** \brief Independent absolute, slew and finite checks all reject unsafe candidates before transport. */
TEST_CASE( "Output finite and slew guards apply to every pixel", "[multirate][output]" )
{
    const float invalid =
        GENERATE( 1.1f, .6f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity() );
    MagAOX::app::SyncoOutputState output( { true, true, 1, .5, 1000000 }, 2, true );
    REQUIRE( output.request( true, true ) );
    const std::array<float, 2> pixels{ 0, invalid };
    REQUIRE_FALSE( output.validate( pixels, false ) );
    REQUIRE( output.status().m_fault );
    REQUIRE_FALSE( output.status().m_requested );
}
/** \brief Early INDI-style intent and status reads cannot race startup state/callback installation. */
TEST_CASE( "Output startup remains inert under concurrent early control requests", "[multirate][output]" )
{
    MagAOX::app::SyncoOutput output;
    std::atomic<bool>        finish{ false };
    std::thread              reader(
        [&]
        {
            while( !finish.load() )
            {
                output.status();
                output.request( true ); // No config authorization, in either build profile.
                output.request( false );
            }
        } );
    output.configure( { false, false, 1, 1, 1000000 }, "private_not_attached_aux", "", 50, 2 );
    output.start( [] { return true; },
                  []( const auto &, const auto &, std::int64_t, std::span<float> ) { return false; } );
    finish.store( true );
    reader.join();
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( output.stop() );
}
/** \brief Only the explicit simulator policy accepts absent bounds/watchdog; DDSPC remains strict by default. */
TEST_CASE( "Simulator producer defaults preserve finite values and acknowledgement ownership", "[multirate][output]" )
{
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = true;
    MagAOX::app::SyncoOutputState standard( config, 2, true );
    REQUIRE_FALSE( standard.request( true, true ) );
    REQUIRE_FALSE( standard.initialize() );
    for( unsigned missing = 0; missing < 3; ++missing )
    {
        MagAOX::app::SyncoOutputConfig strict{ true, true, 1, 1, 1000 };
        if( missing == 0 )
            strict.m_maxAbs = 0;
        if( missing == 1 )
            strict.m_maxStep = 0;
        if( missing == 2 )
            strict.m_ackTimeoutNs = 0;
        MagAOX::app::SyncoOutputState controller( strict, 2, true );
        REQUIRE_FALSE( controller.request( true, true ) );
        REQUIRE_FALSE( controller.initialize() );
    }
    config.m_producerDefaults = true;
    MagAOX::app::SyncoOutputState simulator( config, 2, true ), shadow( config, 2, false );
    REQUIRE_FALSE( shadow.request( true, true ) );
    REQUIRE_FALSE( shadow.initialize() );
    REQUIRE_FALSE( simulator.request( true, false ) );
    REQUIRE( simulator.initialize() );
    REQUIRE( simulator.request( true, true ) );
    const std::array<float, 2> zero{}, value{ .2f, -.1f };
    REQUIRE( simulator.validate( zero, true ) );
    simulator.submitted( zero, true, 100 );
    simulator.acknowledge( true );
    REQUIRE( simulator.validate( value, false ) );
    simulator.submitted( value, false, 200 );
    simulator.check( 1000000000, true ); // No extra simulator watchdog; the base still owns the unacknowledged buffer.
    REQUIRE_FALSE( simulator.status().m_fault );
    REQUIRE( simulator.pending() );
    REQUIRE_FALSE( simulator.validate( value, false ) );
    simulator.request( false, false );
    REQUIRE_FALSE( simulator.validate( zero, true ) ); // Disable cannot overwrite a consumer-owned payload.
    simulator.acknowledge( true );
    REQUIRE( simulator.zero_needed() );
    REQUIRE( simulator.validate( zero, true ) );
    simulator.submitted( zero, true, 1000000001 );
    simulator.acknowledge( true );
    REQUIRE( simulator.status().m_zeroConfirmed );
    REQUIRE( simulator.request( true, true ) );
    const std::array<float, 2> invalid{ 0, std::numeric_limits<float>::quiet_NaN() };
    REQUIRE_FALSE( simulator.validate( invalid, false ) );
    REQUIRE( simulator.status().m_fault );
    REQUIRE_FALSE( simulator.status().m_requested );
}
} // namespace libXWCTest::loPredCtrlAccTest
