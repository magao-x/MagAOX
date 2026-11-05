/** \file ddspc_input_test.cpp
 * \brief Deterministic causal WFS/applied-input pairing, timing rejection, and independent-model replay tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "causal_input_test_data.hpp"
#include <limits>

namespace libXWCTest::loPredCtrlAccTest
{

/** \brief Each physical next-exposure input becomes u_i, preserving every multichannel DDSPC history block.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input pairing preserves the physical prediction origin", "[loPredCtrlAcc][input]" )
{
    const auto                  config = input_config();
    DDSPC::CausalInputBuilder   builder( config );
    DDSPC::PredictiveController history( 2, 3, 2, 0, 1, 1, 1, 1, 2, DDSPC::HistoryMode::Causal );
    auto                        first = input_wfs( config, 0 );
    REQUIRE( builder.push_wfs( first, first.m_decisionNs ).m_records.empty() );
    for( std::uint64_t i = 0; i < 28; ++i )
    {
        auto following = input_wfs( config, i + 1 );
        auto command   = input_command( config, i );
        REQUIRE( command.m_requestedNs < command.m_effectiveStartNs );
        REQUIRE( builder.push_wfs( following, following.m_decisionNs ).m_records.empty() );
        const auto output = builder.push_command( command, command.m_stamp.m_receivedNs );
        REQUIRE( output.m_accepted );
        REQUIRE_FALSE( output.m_discontinuity );
        REQUIRE( output.m_records.size() == 1 );
        const auto &pair = output.m_records[0];
        REQUIRE( pair.m_evidence == DDSPC::InputEvidence::SyntheticApplied );
        REQUIRE( pair.m_origin.m_counter == 100 + i );
        REQUIRE( pair.m_target.m_counter == 101 + i );
        REQUIRE( pair.m_command.m_counter == 900 + i );
        REQUIRE( pair.m_inputStartNs == following.m_exposureStartNs - config.m_responseLagNs );
        REQUIRE( pair.m_inputEndNs == following.m_exposureEndNs - config.m_responseLagNs );
        const auto origin = input_wfs( config, i );
        REQUIRE( pair.m_record.m_index == 100 + i );
        REQUIRE( pair.m_record.m_residual == origin.m_residual );
        REQUIRE( pair.m_record.m_appliedCommand == command.m_applied );
        REQUIRE( pair.m_record.m_acceleration == origin.m_acceleration );
        history.append_sample(
            i, pair.m_record.m_residual, pair.m_record.m_appliedCommand, pair.m_record.m_acceleration );
        DDSPC::Matrix phi, target;
        if( i < 4 )
        {
            REQUIRE_FALSE( history.get_training_example( phi, target ) );
            continue;
        }
        REQUIRE( history.get_training_example( phi, target ) );
        // Independent chronological oracle for F=2,N=3:
        // [u_(i-1),u_(i-2),u_(i-3),u_(i-4),a_(i-2),a_(i-3),y_(i-2),y_(i-3),y_(i-4)].
        DDSPC::Matrix expected( 16, 1 ), expectedTarget( 4, 1 );
        expected << input_command( config, i - 1 ).m_applied, input_command( config, i - 2 ).m_applied,
            input_command( config, i - 3 ).m_applied, input_command( config, i - 4 ).m_applied,
            input_wfs( config, i - 2 ).m_acceleration, input_wfs( config, i - 3 ).m_acceleration,
            input_wfs( config, i - 2 ).m_residual, input_wfs( config, i - 3 ).m_residual,
            input_wfs( config, i - 4 ).m_residual;
        expectedTarget << input_wfs( config, i ).m_residual, input_wfs( config, i - 1 ).m_residual;
        REQUIRE( phi == expected );
        REQUIRE( target == expectedTarget );
    }
}

/** \brief Driver preparation/acceptance, guessed timestamps and unverified contracts cannot produce applied history.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input pairing requires explicit applied provenance and timing", "[loPredCtrlAcc][input]" )
{
    const auto config  = input_config();
    auto       command = input_command( config, 0 );
    for( auto evidence : { DDSPC::InputEvidence::Unverified,
                           DDSPC::InputEvidence::Requested,
                           DDSPC::InputEvidence::DriverPrepared,
                           DDSPC::InputEvidence::DriverAccepted,
                           DDSPC::InputEvidence::VerifiedApplied } )
    {
        DDSPC::CausalInputBuilder builder( config );
        command.m_evidence = evidence;
        const auto result  = builder.push_command( command, command.m_stamp.m_receivedNs );
        REQUIRE_FALSE( result.m_accepted );
        REQUIRE( result.m_discontinuity );
        REQUIRE( result.m_reason == DDSPC::InputReason::UnverifiedCommand );
    }
    for( int field = 0; field < 4; ++field )
    {
        auto gated = config;
        if( field == 0 )
            gated.m_timingVerified = false;
        if( field == 1 )
            gated.m_mappingVerified = false;
        if( field == 2 )
            gated.m_clockEpoch = 0;
        if( field == 3 )
            gated.m_evidence = DDSPC::InputEvidence::DriverAccepted;
        DDSPC::CausalInputBuilder builder( gated );
        auto                      wfs = input_wfs( config, 0 );
        REQUIRE( builder.push_wfs( wfs, wfs.m_decisionNs ).m_reason == DDSPC::InputReason::UnverifiedContract );
    }
    // A matching counter and payload cannot repair an effective interval one frame too late.
    DDSPC::CausalInputBuilder builder( config );
    auto                      origin = input_wfs( config, 0 ), target = input_wfs( config, 1 );
    REQUIRE( builder.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( builder.push_wfs( target, target.m_decisionNs ).m_accepted );
    command                   = input_command( config, 1 );
    command.m_stamp.m_counter = 900;
    auto result               = builder.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( result.m_reason == DDSPC::InputReason::MissingCoverage );
    REQUIRE( result.m_records.empty() );

    auto verifiedConfig       = config;
    verifiedConfig.m_evidence = DDSPC::InputEvidence::VerifiedApplied;
    DDSPC::CausalInputBuilder verified( verifiedConfig );
    REQUIRE( verified.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( verified.push_wfs( target, target.m_decisionNs ).m_accepted );
    command            = input_command( verifiedConfig, 0 );
    command.m_evidence = DDSPC::InputEvidence::VerifiedApplied;
    auto paired        = verified.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( paired.m_records.size() == 1 );
    REQUIRE( paired.m_records[0].m_evidence == DDSPC::InputEvidence::VerifiedApplied );
    REQUIRE( std::string( DDSPC::input_evidence( verified.evidence() ) ) == "externally_verified_applied" );
}

/** \brief WFS/command counter gaps and uncounted physical frame gaps break both histories without compression.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input pairing detects frame drops in both sequences", "[loPredCtrlAcc][input]" )
{
    const auto                config = input_config();
    DDSPC::CausalInputBuilder builder( config );
    auto zero = input_wfs( config, 0 ), two = input_wfs( config, 2 ), three = input_wfs( config, 3 );
    REQUIRE( builder.push_wfs( zero, zero.m_decisionNs ).m_accepted );
    auto dropped = builder.push_wfs( two, two.m_decisionNs );
    REQUIRE( dropped.m_accepted );
    REQUIRE( dropped.m_discontinuity );
    REQUIRE( dropped.m_reason == DDSPC::InputReason::CounterGap );
    REQUIRE( builder.push_wfs( three, three.m_decisionNs ).m_records.empty() );
    auto command = input_command( config, 2 );
    auto resumed = builder.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( resumed.m_records.size() == 1 );
    REQUIRE( resumed.m_records[0].m_record.m_index == 102 );
    // A camera tick skipped while published cnt0 advanced by one is detected by physical timestamps.
    DDSPC::CausalInputBuilder tickGap( config );
    REQUIRE( tickGap.push_wfs( zero, zero.m_decisionNs ).m_accepted );
    two.m_stamp.m_counter = 101;
    auto gap              = tickGap.push_wfs( two, two.m_decisionNs );
    REQUIRE( gap.m_discontinuity );
    REQUIRE( gap.m_reason == DDSPC::InputReason::TimingGap );
    // An omitted applied certificate breaks history even when a later interval has a plausible value.
    DDSPC::CausalInputBuilder dmGap( config );
    auto                      firstCommand = input_command( config, 0 );
    REQUIRE( dmGap.push_command( firstCommand, firstCommand.m_stamp.m_receivedNs ).m_accepted );
    command = input_command( config, 2 );
    gap     = dmGap.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( gap.m_discontinuity );
    REQUIRE( gap.m_reason == DDSPC::InputReason::CounterGap );
    REQUIRE( gap.m_records.empty() );
}

/** \brief Silence, late copies, future confirmations, and absent input coverage never become training data.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input pairing rejects stale and impossible timestamps", "[loPredCtrlAcc][input]" )
{
    const auto                config = input_config();
    const auto                origin = input_wfs( config, 0 ), target = input_wfs( config, 1 );
    DDSPC::CausalInputBuilder missing( config );
    REQUIRE( missing.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( missing.push_wfs( target, target.m_decisionNs ).m_records.empty() );
    auto expired = missing.advance( target.m_exposureEndNs + config.m_maxAgeNs + 1 );
    REQUIRE( expired.m_discontinuity );
    REQUIRE( expired.m_reason == DDSPC::InputReason::Stale );
    REQUIRE( expired.m_records.empty() );
    for( int field = 0; field < 7; ++field )
    {
        DDSPC::CausalInputBuilder builder( config );
        auto                      wfs = origin;
        auto                      now = wfs.m_decisionNs;
        if( field == 0 )
            wfs.m_stamp.m_receivedNs = wfs.m_decisionNs + 1;
        if( field == 1 )
            wfs.m_stamp.m_availableNs = wfs.m_exposureEndNs - 1;
        if( field == 2 )
            wfs.m_exposureStartNs = -1;
        if( field == 3 )
            wfs.m_stamp.m_uncertaintyNs = 1;
        if( field == 4 )
            now = wfs.m_exposureEndNs + config.m_maxAgeNs + 1;
        if( field == 5 )
            wfs.m_exposureEndNs = wfs.m_exposureStartNs;
        if( field == 6 )
            wfs.m_decisionNs = now + 1;
        auto result = builder.push_wfs( wfs, now );
        REQUIRE_FALSE( result.m_accepted );
        REQUIRE( result.m_discontinuity );
        REQUIRE( result.m_records.empty() );
    }
    DDSPC::CausalInputBuilder future( config );
    auto                      command = input_command( config, 0 );
    command.m_stamp.m_availableNs     = command.m_effectiveStartNs;
    command.m_stamp.m_receivedNs      = command.m_effectiveStartNs;
    REQUIRE( future.push_command( command, command.m_effectiveStartNs ).m_reason == DDSPC::InputReason::InvalidTime );
}

/** \brief Transitions or uncertainty within an exposure reject single-vector input assignments; latency is explicit.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input pairing rejects ambiguous exposures and committed horizons", "[loPredCtrlAcc][input]" )
{
    for( int scenario = 0; scenario < 4; ++scenario )
    {
        auto config               = input_config();
        config.m_maxUncertaintyNs = 100;
        if( scenario == 2 )
            config.m_minCommandDelayNs = config.m_periodNs;
        if( scenario == 3 )
            config.m_responseLagNs += config.m_periodNs;
        DDSPC::CausalInputBuilder builder( config );
        auto                      origin = input_wfs( config, 0 ), target = input_wfs( config, 1 );
        REQUIRE( builder.push_wfs( origin, origin.m_decisionNs ).m_accepted );
        auto intermediate = builder.push_wfs( target, target.m_decisionNs );
        if( scenario >= 2 )
        {
            REQUIRE( intermediate.m_reason == DDSPC::InputReason::UnsupportedLatency );
            REQUIRE( intermediate.m_discontinuity );
            continue;
        }
        auto command = input_command( config, 0 );
        if( scenario == 0 )
            command.m_effectiveEndNs = command.m_effectiveStartNs + config.m_exposureNs / 2;
        if( scenario == 1 )
            command.m_stamp.m_uncertaintyNs = 100;
        auto result = builder.push_command( command, command.m_stamp.m_receivedNs );
        REQUIRE( result.m_reason == DDSPC::InputReason::AmbiguousExposure );
        REQUIRE( result.m_discontinuity );
        REQUIRE( result.m_records.empty() );
    }
}

/** \brief Generation/layout/clock changes, counter replay, and backward clocks require a fresh session.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input sessions reject restarts and replayed identities", "[loPredCtrlAcc][input]" )
{
    const auto                config = input_config();
    const auto                origin = input_wfs( config, 0 );
    DDSPC::CausalInputBuilder replay( config );
    REQUIRE( replay.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( replay.invalidate().m_discontinuity );
    REQUIRE( replay.push_wfs( origin, origin.m_decisionNs ).m_reason == DDSPC::InputReason::OutOfOrder );
    for( int field = 0; field < 3; ++field )
    {
        DDSPC::CausalInputBuilder builder( config );
        auto                      changed = origin;
        if( field == 0 )
            ++changed.m_stamp.m_generation;
        if( field == 1 )
            ++changed.m_stamp.m_clockEpoch;
        if( field == 2 )
            ++changed.m_stamp.m_layout;
        REQUIRE( builder.push_wfs( changed, changed.m_decisionNs ).m_reason == DDSPC::InputReason::WrongIdentity );
    }
    DDSPC::CausalInputBuilder clock( config );
    REQUIRE( clock.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( clock.advance( origin.m_decisionNs - 1 ).m_reason == DDSPC::InputReason::ClockReversal );
    REQUIRE( clock.advance( origin.m_decisionNs + 1 ).m_reason == DDSPC::InputReason::ClockReversal );
    auto nextConfig = config;
    ++nextConfig.m_wfsGeneration;
    ++nextConfig.m_commandGeneration;
    ++nextConfig.m_clockEpoch;
    DDSPC::CausalInputBuilder restarted( nextConfig );
    REQUIRE( restarted.push_wfs( origin, origin.m_decisionNs ).m_reason == DDSPC::InputReason::WrongIdentity );
    auto next = input_wfs( nextConfig, 0 );
    REQUIRE( restarted.push_wfs( next, next.m_decisionNs ).m_accepted );
    next = input_wfs( nextConfig, 1 );
    REQUIRE( restarted.push_wfs( next, next.m_decisionNs ).m_accepted );
    auto command = input_command( nextConfig, 0 );
    REQUIRE( restarted.push_command( command, command.m_stamp.m_receivedNs ).m_records.size() == 1 );
}

/** \brief Bounded backlog can drain several held-input examples and rejects overflow without replacing evidence.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input queues are bounded and independent of callback ordering", "[loPredCtrlAcc][input]" )
{
    auto config       = input_config();
    config.m_capacity = 4;
    DDSPC::CausalInputBuilder overflow( config );
    for( std::uint64_t i = 0; i < 4; ++i )
    {
        auto wfs = input_wfs( config, i );
        REQUIRE( overflow.push_wfs( wfs, wfs.m_decisionNs ).m_accepted );
    }
    auto wfs = input_wfs( config, 4 );
    REQUIRE( overflow.push_wfs( wfs, wfs.m_decisionNs ).m_reason == DDSPC::InputReason::Overflow );
    DDSPC::CausalInputBuilder batch( config );
    auto                      command = input_command( config, 0 );
    command.m_effectiveEndNs += 2 * config.m_periodNs;
    command.m_stamp.m_availableNs = command.m_effectiveEndNs + 10000;
    command.m_stamp.m_receivedNs  = command.m_effectiveEndNs + 20000;
    for( std::uint64_t i = 0; i < 4; ++i )
    {
        wfs = input_wfs( config, i );
        REQUIRE( batch.push_wfs( wfs, wfs.m_decisionNs ).m_accepted );
    }
    auto output = batch.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( output.m_records.size() == 3 );
    for( std::uint64_t i = 0; i < 3; ++i )
    {
        REQUIRE( output.m_records[i].m_record.m_index == 100 + i );
        REQUIRE( output.m_records[i].m_record.m_appliedCommand == command.m_applied );
    }
    // Cross-source delivery order does not change pairing; already captured origin deadlines stay frozen.
    DDSPC::CausalInputBuilder reverse( config );
    REQUIRE( reverse.push_command( command, command.m_stamp.m_receivedNs ).m_records.empty() );
    std::vector<DDSPC::PairedInputRecord> reversed;
    for( std::uint64_t i = 0; i < 4; ++i )
    {
        wfs               = input_wfs( config, i );
        const auto result = reverse.push_wfs( wfs, command.m_stamp.m_receivedNs );
        reversed.insert( reversed.end(), result.m_records.begin(), result.m_records.end() );
    }
    REQUIRE( reversed.size() == output.m_records.size() );
    for( std::size_t i = 0; i < reversed.size(); ++i )
        REQUIRE( reversed[i].m_record.m_residual == output.m_records[i].m_record.m_residual );
}

/** \brief Integer nanoseconds preserve subframe precision and boundary validation avoids signed overflow.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input timing uses integer boundaries and validates extremes", "[loPredCtrlAcc][input]" )
{
    auto                      config = input_config();
    constexpr std::int64_t    offset = 4000000000000000000LL;
    DDSPC::CausalInputBuilder builder( config );
    for( std::uint64_t i = 0; i < 2; ++i )
    {
        auto wfs = input_wfs( config, i );
        wfs.m_exposureStartNs += offset;
        wfs.m_exposureEndNs += offset;
        wfs.m_decisionNs += offset;
        wfs.m_stamp.m_availableNs += offset;
        wfs.m_stamp.m_receivedNs += offset;
        wfs.m_accelAcquiredNs += offset;
        wfs.m_accelAvailableNs += offset;
        REQUIRE( builder.push_wfs( wfs, wfs.m_decisionNs ).m_accepted );
    }
    auto command = input_command( config, 0 );
    command.m_requestedNs += offset;
    command.m_effectiveStartNs += offset;
    command.m_effectiveEndNs += offset;
    command.m_stamp.m_availableNs += offset;
    command.m_stamp.m_receivedNs += offset;
    const auto output = builder.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( output.m_records.size() == 1 );
    REQUIRE( output.m_records[0].m_inputStartNs ==
             input_wfs( config, 1 ).m_exposureStartNs + offset - config.m_responseLagNs );
    config.m_maxUncertaintyNs = 1;
    DDSPC::CausalInputBuilder extreme( config );
    auto                      wfs = input_wfs( config, 0 );
    wfs.m_exposureEndNs           = std::numeric_limits<std::int64_t>::max();
    wfs.m_stamp.m_uncertaintyNs   = 1;
    REQUIRE( extreme.push_wfs( wfs, wfs.m_decisionNs ).m_reason == DDSPC::InputReason::InvalidTime );
    config.m_periodNs = std::numeric_limits<std::int64_t>::max();
    REQUIRE_NOTHROW( DDSPC::CausalInputBuilder( config ) );
    config.m_maxUncertaintyNs = std::numeric_limits<std::int64_t>::max();
    REQUIRE_THROWS_AS( DDSPC::CausalInputBuilder( config ), std::invalid_argument );
}

/** \brief Malformed, overlapping, stale, replayed, or mismatched applied certificates never advance histories.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal applied certificates reject corrupt and reordered source data", "[loPredCtrlAcc][input]" )
{
    const auto config = input_config();
    for( int scenario = 0; scenario < 8; ++scenario )
    {
        DDSPC::CausalInputBuilder builder( config );
        auto                      command = input_command( config, 0 );
        auto                      now     = command.m_stamp.m_receivedNs;
        if( scenario == 0 )
            command.m_stamp.m_snapshotValid = false;
        if( scenario == 1 )
            command.m_applied( 0 ) = std::numeric_limits<float>::quiet_NaN();
        if( scenario == 2 )
            command.m_applied.resize( 1, 1 );
        if( scenario == 3 )
            ++command.m_stamp.m_generation;
        if( scenario == 4 )
            ++command.m_stamp.m_clockEpoch;
        if( scenario == 5 )
            ++command.m_stamp.m_layout;
        if( scenario == 6 )
            now = command.m_effectiveEndNs + config.m_maxAgeNs + 1;
        if( scenario == 7 )
            command.m_stamp.m_counter = std::numeric_limits<std::uint64_t>::max();
        const auto result = builder.push_command( command, now );
        REQUIRE_FALSE( result.m_accepted );
        REQUIRE( result.m_discontinuity );
        REQUIRE( result.m_records.empty() );
    }
    auto                      first = input_command( config, 0 );
    DDSPC::CausalInputBuilder replay( config );
    REQUIRE( replay.push_command( first, first.m_stamp.m_receivedNs ).m_accepted );
    REQUIRE( replay.invalidate().m_discontinuity );
    REQUIRE( replay.push_command( first, first.m_stamp.m_receivedNs ).m_reason == DDSPC::InputReason::OutOfOrder );
    auto overlap               = input_command( config, 1 );
    overlap.m_effectiveStartNs = first.m_effectiveEndNs - 1;
    REQUIRE( replay.push_command( overlap, overlap.m_stamp.m_receivedNs ).m_reason == DDSPC::InputReason::OutOfOrder );
    auto limited       = config;
    limited.m_capacity = 2;
    DDSPC::CausalInputBuilder bounded( limited );
    for( std::uint64_t i = 0; i < 2; ++i )
    {
        auto command = input_command( limited, i );
        REQUIRE( bounded.push_command( command, command.m_stamp.m_receivedNs ).m_accepted );
    }
    auto command = input_command( limited, 2 );
    REQUIRE( bounded.push_command( command, command.m_stamp.m_receivedNs ).m_reason == DDSPC::InputReason::Overflow );
}

/** \brief Bounded clock/exposure jitter is accepted only when a confirmed hold covers the full uncertainty window.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Causal input uncertainty requires conservative interval coverage", "[loPredCtrlAcc][input]" )
{
    auto config               = input_config();
    config.m_maxUncertaintyNs = 100;
    config.m_jitterNs         = 1;
    DDSPC::CausalInputBuilder builder( config );
    auto                      origin = input_wfs( config, 0 ), target = input_wfs( config, 1 );
    origin.m_stamp.m_uncertaintyNs = 100;
    target.m_stamp.m_uncertaintyNs = 100;
    ++target.m_exposureStartNs; // A real one-nanosecond cadence/duration deviation.
    REQUIRE( builder.push_wfs( origin, origin.m_decisionNs ).m_accepted );
    REQUIRE( builder.push_wfs( target, target.m_decisionNs ).m_accepted );
    auto command = input_command( config, 0 );
    command.m_effectiveStartNs -= 1000;
    command.m_stamp.m_uncertaintyNs = 100;
    const auto result               = builder.push_command( command, command.m_stamp.m_receivedNs );
    REQUIRE( result.m_records.size() == 1 );
    REQUIRE( result.m_records[0].m_inputStartNs == target.m_exposureStartNs - config.m_responseLagNs - 100 );
    REQUIRE( result.m_records[0].m_inputEndNs == target.m_exposureEndNs - config.m_responseLagNs + 100 );
}

/** \brief Both independent fits match manually aligned applied histories; late origin features affect only assistance.
 * \ingroup loPredCtrlAcc_unit_test
 */
TEST_CASE( "Paired records feed identical WFS DM observations to both independent models", "[loPredCtrlAcc][input]" )
{
    const auto config = input_config();
    for( bool qrd : { false, true } )
    {
        DDSPC::ShadowConfig model;
        model.m_modes         = 2;
        model.m_history       = 2;
        model.m_future        = 2;
        model.m_accelChannels = 1;
        model.m_accelHistory  = 2;
        model.m_useQrd        = qrd;
        DDSPC::ShadowController actual( model ), oracle( model );
        for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        {
            actual.training( mode, true );
            oracle.training( mode, true );
        }
        DDSPC::CausalInputBuilder builder( config );
        auto                      origin = input_wfs( config, 0 );
        REQUIRE( builder.push_wfs( origin, origin.m_decisionNs ).m_accepted );
        for( std::uint64_t i = 0; i < 60; ++i )
        {
            auto following = input_wfs( config, i + 1 );
            // A later-arriving acceleration vector is never backfilled into this origin.
            if( i + 1 == 30 )
                following.m_accelAvailableNs = following.m_decisionNs + 1;
            REQUIRE( builder.push_wfs( following, following.m_decisionNs ).m_accepted );
            auto       command = input_command( config, i );
            const auto result  = builder.push_command( command, command.m_stamp.m_receivedNs );
            REQUIRE( result.m_records.size() == 1 );
            REQUIRE( actual.process( result.m_records[0].m_record ) );
            DDSPC::ShadowRecord truth;
            truth.m_index          = 100 + i;
            truth.m_residual       = input_wfs( config, i ).m_residual;
            truth.m_appliedCommand = input_command( config, i ).m_applied;
            truth.m_acceleration   = input_wfs( config, i ).m_acceleration;
            truth.m_wfsValid = truth.m_appliedValid = truth.m_timingValid = true;
            truth.m_accelValid                                            = i != 30;
            REQUIRE( oracle.process( truth ) );
            for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
            {
                REQUIRE( actual.prediction_matrix( mode ) == oracle.prediction_matrix( mode ) );
                REQUIRE( actual.status( mode ).m_samples == oracle.status( mode ).m_samples );
                REQUIRE( actual.status( mode ).m_updates == oracle.status( mode ).m_updates );
            }
        }
        REQUIRE( actual.status( DDSPC::ShadowMode::Standard ).m_ready );
        REQUIRE( actual.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    }
}

} // namespace libXWCTest::loPredCtrlAccTest
