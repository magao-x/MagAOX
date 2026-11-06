/** \file ddspc_async_test.cpp
 * \brief Exact committed replay, deterministic queue/age/reset tests and threaded snapshot coherence.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "async_test_data.hpp"
#include "phase7_replay.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
/// Compare the actual committed manager with a coherent worker publication, including unready fits.
void check_async_snapshot( const DDSPC::OfflineSnapshot &snapshot /**< [in] Pinned copy. */,
                           const Phase6Status           &expected /**< [in] Independent Phase 7 oracle. */ )
{
    const auto               &s = snapshot.m_info.m_status;
    const std::array<bool, 5> flags{ s.m_ready, s.m_trainingEnabled, s.m_training, s.m_dataValid, s.m_predictionValid };
    const std::array<std::uint64_t, 3> counts{ s.m_samples, s.m_updates, s.m_predictions };
    const std::array<int, 3>           codes{ s.m_features, s.m_rank, static_cast<int>( s.m_reason ) };
    REQUIRE( flags == expected.m_flags );
    REQUIRE( counts == expected.m_counts );
    REQUIRE( codes == expected.m_codes );
    REQUIRE( s.m_rmse == expected.m_rmse );
    REQUIRE( snapshot.m_theta.isApprox( expected.m_theta, 0 ) );
    REQUIRE( snapshot.m_info.m_version == s.m_updates );
    if( s.m_ready )
        REQUIRE( snapshot.m_originProposal.isApprox( expected.m_proposal, 0 ) );
}
/// \endcond

/// Independent workers preserve every Phase 7 calculation for identical ordered data and control boundaries.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline independent workers reproduce committed Phase 7 exactly", "[ddspc][async][baseline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
#endif
    // clang-format on
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
            for( float gamma : { 1.f, .99f } )
            {
                DDSPC::ShadowConfig cfg;
                cfg.m_modes  = modes;
                cfg.m_useQrd = qrd;
                cfg.m_gamma  = gamma;
                cfg.m_prior  = .1f;
                cfg.m_lambda = .3f;
                DDSPC::OfflineAsyncShadow async( cfg, 8, 600 );
                auto                      reference = phase7_replay( modes, qrd, gamma );
                reference->training( false, true );
                reference->training( true, true );
                AsyncTestInputs                       input( 8000, 1000000000 );
                DDSPC::OfflineRecord                  record( cfg );
                std::array<DDSPC::OfflineSnapshot, 2> snapshots{
                    DDSPC::OfflineSnapshot( cfg, DDSPC::ShadowMode::Standard ),
                    DDSPC::OfflineSnapshot( cfg, DDSPC::ShadowMode::AccelAssisted ) };
                DDSPC::Matrix output = DDSPC::Matrix::Zero( modes, 1 );
                for( int t = 0; t < 600; ++t )
                {
                    CAPTURE( modes, qrd, gamma, t );
                    input.next( t, record );
                    if( t >= 350 )
                        record.m_data.m_index += 5;
                    record.m_data.m_accelValid  = t != 250;
                    record.m_data.m_timingValid = t != 300;
                    if( t == 420 )
                    {
                        async.reset( DDSPC::ShadowMode::Standard );
                        reference->reset( false );
                    }
                    const bool accepted = async.submit( record, record.m_availableNs );
                    REQUIRE( accepted == reference->process( record.m_data.m_index,
                                                             record.m_data.m_residual,
                                                             record.m_data.m_appliedCommand,
                                                             record.m_data.m_acceleration,
                                                             record.m_data.m_timingValid,
                                                             record.m_data.m_accelValid ) );
                    for( int lane = 0; lane < 2; ++lane )
                    {
                        const auto mode      = lane ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
                        const bool processed = async.manual_step( mode, record.m_availableNs + 500 );
                        REQUIRE( processed == accepted );
                        if( !accepted )
                        {
                            REQUIRE_FALSE( async.read_snapshot( mode, snapshots[lane] ) );
                            continue;
                        }
                        REQUIRE( async.read_snapshot( mode, snapshots[lane] ) );
                        check_async_snapshot( snapshots[lane], reference->inspect( lane ) );
                        const auto              &s = snapshots[lane];
                        DDSPC::OfflinePrediction result;
                        const bool               predicted = async.predict( mode,
                                                                            1,
                                                                            record.m_data.m_index,
                                                                            record.m_measurementNs,
                                                                            record.m_availableNs + 1000,
                                                                            1000000,
                                                                            s.m_originState,
                                                                            output,
                                                                            result );
                        REQUIRE( predicted == s.m_info.m_status.m_ready );
                        if( predicted )
                            REQUIRE( output.isApprox( s.m_originProposal, 0 ) );
                        else
                            REQUIRE( output.isZero( 0 ) );
                    }
                }
                REQUIRE( async.stats( DDSPC::ShadowMode::Standard ).m_processed == 599 );
                REQUIRE( async.stats( DDSPC::ShadowMode::AccelAssisted ).m_processed == 599 );
                REQUIRE( input.sensor_events() == 2400 );
            }
}

/// Queue saturation has common rejection, immediate invalidation, prefix discard and independent reset behavior.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline bounded fanout invalidates dropped histories and resets", "[ddspc][async][queue]" )
{
    using namespace DDSPC;
    ShadowConfig cfg;
    cfg.m_modes   = 2;
    cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
    cfg.m_prior                                       = .1f;
    cfg.m_lambda                                      = .3f;
    OfflineAsyncShadow async( cfg, 2, 256 );
    AsyncTestInputs    input( 4000, 1000000000 );
    OfflineRecord      record( cfg );
    OfflineSnapshot    standard( cfg, ShadowMode::Standard ), assisted( cfg, ShadowMode::AccelAssisted );
    for( int i = 0; i < 80; ++i )
    {
        input.next( i, record );
        REQUIRE( async.submit( record, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
    }
    REQUIRE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE( standard.m_info.m_status.m_ready );
    REQUIRE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
    REQUIRE( assisted.m_info.m_status.m_ready );
    for( int i = 80; i < 82; ++i )
    {
        input.next( i, record );
        REQUIRE( async.submit( record, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
    }
    input.next( 82, record );
    REQUIRE_FALSE( async.submit( record, record.m_availableNs ) );
    REQUIRE( async.drops() == 1 );
    REQUIRE( async.stats( ShadowMode::Standard ).m_depth == 0 );
    REQUIRE( async.stats( ShadowMode::AccelAssisted ).m_depth == 2 );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
    REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
    REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
    REQUIRE( async.stats( ShadowMode::AccelAssisted ).m_discarded == 2 );
    for( int i = 83; i < 110; ++i )
    {
        input.next( i, record );
        REQUIRE( async.submit( record, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
    }
    REQUIRE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
    REQUIRE( assisted.m_info.m_status.m_ready );
    async.reset( ShadowMode::Standard );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
    REQUIRE_FALSE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
    input.next( 110, record );
    record.m_data.m_accelValid = false;
    REQUIRE( async.submit( record, record.m_availableNs ) );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
    REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
    REQUIRE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE( standard.m_info.m_version == 0 );
    REQUIRE( standard.m_info.m_reset > 0 );
    REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
    input.next( 111, record );
    record.m_streamEpoch  = 2;
    record.m_data.m_index = 0;
    REQUIRE( async.submit( record, record.m_availableNs ) );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
    REQUIRE( async.read_snapshot( ShadowMode::Standard, standard ) );
    REQUIRE( standard.m_info.m_streamEpoch == 2 );
    REQUIRE_FALSE( standard.m_info.m_status.m_ready );
    REQUIRE_FALSE( async.submit( record, record.m_availableNs ) ); // Duplicate identity/time never enters either queue.
    input.next( 112, record );
    record.m_availableNs += 1;
    REQUIRE_FALSE(
        async.submit( record, record.m_measurementNs ) ); // Future availability cannot be inferred from callback order.
    REQUIRE( async.stats( ShadowMode::AccelAssisted ).m_highWater <= 2 );
}

/// Delayed publication uses the original fit exactly, and wall-clock evidence age cannot be refreshed by publishing
/// late.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline prediction rejects stale snapshots and measures publication lag", "[ddspc][async][age]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
#endif
    // clang-format on
    using namespace DDSPC;
    ShadowConfig cfg;
    cfg.m_modes   = 2;
    cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
    cfg.m_prior                                       = .1f;
    cfg.m_lambda                                      = .3f;
    OfflineAsyncShadow async( cfg, 32, 256 );
    AsyncTestInputs    input( 8000, 1000000000 );
    OfflineRecord      record( cfg );
    OfflineSnapshot    snapshot( cfg, ShadowMode::AccelAssisted );
    Matrix             output = Matrix::Zero( 2, 1 ), state = Matrix::Zero( 4, 1 );
    OfflinePrediction  result;
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted, 1, 0, 1, 1, 1, state, output, result ) );
    REQUIRE( result.m_reason == OfflinePredictionReason::NoSnapshot );
    auto current = phase7_replay( 2, false, 1, 1, 1, 1 );
    current->training( false, true );
    current->training( true, true );
    double maxPublicationDelta = 0;
    for( int i = 0; i < 120; ++i )
    {
        input.next( i, record );
        REQUIRE( async.submit( record, record.m_availableNs ) );
        REQUIRE( current->process(
            i, record.m_data.m_residual, record.m_data.m_appliedCommand, record.m_data.m_acceleration, true, true ) );
        if( i < 100 )
        {
            REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
            REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs + 1000 ) );
        }
        if( i == 99 )
            REQUIRE( async.read_snapshot( ShadowMode::AccelAssisted, snapshot ) );
        if( i < 100 )
            continue;
        state << record.m_data.m_residual, record.m_data.m_acceleration;
        REQUIRE( async.predict( ShadowMode::AccelAssisted,
                                1,
                                i,
                                record.m_measurementNs,
                                record.m_availableNs,
                                20000000,
                                state,
                                output,
                                result ) );
        Matrix expected = snapshot.m_gain.bottomRows( 2 ) * state;
        REQUIRE( output.isApprox( expected, 0 ) );
        REQUIRE( result.m_model.m_sequence == 99 );
        const auto latest = current->inspect( true );
        maxPublicationDelta =
            std::max( maxPublicationDelta, double( ( output - latest.m_proposal ).cwiseAbs().maxCoeff() ) );
    }
    REQUIRE( maxPublicationDelta > 0 );
    std::cout << "Delayed publication: matched-version prediction difference=0; max current-fit proposal difference="
              << maxPublicationDelta << '\n';
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted,
                                  1,
                                  119,
                                  record.m_measurementNs,
                                  record.m_availableNs,
                                  500000,
                                  state,
                                  output,
                                  result ) );
    REQUIRE( result.m_reason == OfflinePredictionReason::Stale );
    REQUIRE( output.isZero( 0 ) );
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted,
                                  2,
                                  119,
                                  record.m_measurementNs,
                                  record.m_availableNs,
                                  20000000,
                                  state,
                                  output,
                                  result ) );
    REQUIRE( result.m_reason == OfflinePredictionReason::Invalidated );
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted,
                                  1,
                                  98,
                                  record.m_measurementNs,
                                  record.m_availableNs,
                                  20000000,
                                  state,
                                  output,
                                  result ) );
    REQUIRE( result.m_reason == OfflinePredictionReason::TimeOrder );
    state( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted,
                                  1,
                                  119,
                                  record.m_measurementNs,
                                  record.m_availableNs,
                                  20000000,
                                  state,
                                  output,
                                  result ) );
    REQUIRE( result.m_reason == OfflinePredictionReason::InvalidState );
    REQUIRE_FALSE( async.predict( ShadowMode::AccelAssisted,
                                  1,
                                  119,
                                  record.m_measurementNs,
                                  record.m_availableNs,
                                  20000000,
                                  state,
                                  state,
                                  result ) );
    async.reset( ShadowMode::AccelAssisted );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::AccelAssisted, snapshot ) );
}

/// Actual concurrent readers/writers compare every observed complete snapshot to its independent committed version.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline threaded snapshots remain coherent under rapid buffer reuse", "[ddspc][async][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
    DDSPC::ShadowController::reset();
#endif
    // clang-format on
    using namespace DDSPC;
    const int  modes = GENERATE( 2, 10 );
    const bool qrd   = GENERATE( false, true );
    CAPTURE( modes, qrd );
    ShadowConfig cfg;
    cfg.m_modes   = modes;
    cfg.m_useQrd  = qrd;
    cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
    cfg.m_prior                                       = .1f;
    cfg.m_lambda                                      = .3f;
    constexpr int count                               = 512;
    auto          reference                           = phase7_replay( modes, qrd, 1, 1, 1, 1 );
    reference->training( false, true );
    reference->training( true, true );
    std::array<std::vector<Phase6Status>, 2> expected;
    AsyncTestInputs                          inputs( 4000, 1000000000 );
    OfflineRecord                            record( cfg );
    for( int i = 0; i < count; ++i )
    {
        if( i == 128 || i == 256 )
        {
            reference->reset( false );
            reference->reset( false );
        }
        if( i == 192 || i == 320 )
            reference->reset( true );
        inputs.next( i, record );
        reference->process(
            i, record.m_data.m_residual, record.m_data.m_appliedCommand, record.m_data.m_acceleration, true, true );
        expected[0].push_back( reference->inspect( false ) );
        expected[1].push_back( reference->inspect( true ) );
    }
    OfflineAsyncShadow             async( cfg, 8, count );
    std::array<OfflineSnapshot, 2> snapshot{ OfflineSnapshot( cfg, ShadowMode::Standard ),
                                             OfflineSnapshot( cfg, ShadowMode::AccelAssisted ) };
    inputs = AsyncTestInputs( 4000, 1000000000 );
    async.start();
    std::uint64_t observations = 0;
    const auto    deadline     = offline_now_ns() + 10000000000ULL;
    for( int i = 0; i < count; ++i )
    {
        while( ( async.stats( ShadowMode::Standard ).m_depth >= 7 ||
                 async.stats( ShadowMode::AccelAssisted ).m_depth >= 7 ) &&
               offline_now_ns() < deadline )
            std::this_thread::yield(); // Test coordinator only; no waits exist inside submit/predict.
        REQUIRE( offline_now_ns() < deadline );
        if( i == 128 || i == 256 )
        {
            async.reset( ShadowMode::Standard );
            async.reset( ShadowMode::Standard );
        }
        if( i == 192 || i == 320 )
            async.reset( ShadowMode::AccelAssisted );
        inputs.next( i, record );
        REQUIRE( async.submit( record, offline_now_ns() ) );
        for( int j = 0; j < 12; ++j )
            for( int lane = 0; lane < 2; ++lane )
            {
                const auto mode = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
                if( async.read_snapshot( mode, snapshot[lane] ) )
                {
                    REQUIRE( snapshot[lane].m_info.m_sequence <= std::uint64_t( i ) );
                    check_async_snapshot( snapshot[lane], expected[lane][snapshot[lane].m_info.m_sequence] );
                    ++observations;
                }
            }
    }
    while( ( async.stats( ShadowMode::Standard ).m_depth || async.stats( ShadowMode::AccelAssisted ).m_depth ) &&
           offline_now_ns() < deadline )
        std::this_thread::yield();
    async.stop();
    REQUIRE( observations > 0 );
    REQUIRE( async.drops() == 0 );
    for( int lane = 0; lane < 2; ++lane )
    {
        const auto mode = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        REQUIRE( async.stats( mode ).m_processed + async.stats( mode ).m_discarded == count );
        REQUIRE( async.read_snapshot( mode, snapshot[lane] ) );
        check_async_snapshot( snapshot[lane], expected[lane].back() );
    }
}

/// Explicit sensor/DM availability timestamps, rather than callback order, determine which histories are usable.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline input timing rejects late sensors and future applied inputs", "[ddspc][async][timing]" )
{
    using namespace DDSPC;
    ShadowConfig cfg;
    cfg.m_modes   = 2;
    cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
    OfflineAsyncShadow async( cfg, 4, 64 );
    AsyncTestInputs    input( 4000, 1000000000 );
    OfflineRecord      record( cfg );
    OfflineSnapshot    standard( cfg, ShadowMode::Standard ), assisted( cfg, ShadowMode::AccelAssisted );
    for( int i = 0; i < 40; ++i )
    {
        input.next( i, record );
        REQUIRE( async.submit( record, record.m_availableNs ) );
        async.manual_step( ShadowMode::Standard, record.m_availableNs );
        async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs );
    }
    for( int i = 40; i < 43; ++i )
    {
        input.next( i, record );
        if( i == 40 )
            record.m_accelAvailableNs = record.m_availableNs + 1;
        if( i == 41 )
        {
            record.m_accelNs          = record.m_measurementNs - 500001;
            record.m_accelAvailableNs = record.m_accelNs;
        }
        if( i == 42 )
        {
            record.m_accelNs          = record.m_measurementNs + 1;
            record.m_accelAvailableNs = record.m_accelNs;
        }
        REQUIRE( async.submit( record, record.m_availableNs ) );
        REQUIRE( async.read_snapshot( ShadowMode::Standard, standard ) );
        REQUIRE( standard.m_info.m_status.m_ready );
        REQUIRE_FALSE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
        REQUIRE( async.manual_step( ShadowMode::Standard, record.m_availableNs ) );
        REQUIRE( async.manual_step( ShadowMode::AccelAssisted, record.m_availableNs ) );
        REQUIRE( async.read_snapshot( ShadowMode::AccelAssisted, assisted ) );
        REQUIRE_FALSE( assisted.m_info.m_status.m_dataValid );
    }
    input.next( 43, record );
    record.m_appliedNs = record.m_measurementNs + 1;
    REQUIRE_FALSE( async.submit( record, record.m_availableNs ) );
    REQUIRE_FALSE( async.read_snapshot( ShadowMode::Standard, standard ) );
    input.next( 44, record );
    record.m_appliedAvailableNs = record.m_availableNs + 1;
    REQUIRE_FALSE( async.submit( record, record.m_availableNs ) );
    REQUIRE( async.overflows() == 0 );
    REQUIRE( async.drops() == 2 );
}
} // namespace libXWCTest::loPredCtrlAccTest
