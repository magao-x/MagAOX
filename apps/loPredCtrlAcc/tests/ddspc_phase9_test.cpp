/** \file ddspc_phase9_test.cpp
 * \brief Exact committed Phase 8 protocol replay and independent-overflow numerical/provenance tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "async_test_data.hpp"
// Re-open only application guards for the separately namespaced, verbatim fixture; Eigen guards remain intact.
#undef loPredCtrlAcc_async_shadow_hpp
#undef PCSHADOW_HPP
#undef PCARC_HPP
#undef PCRLS_HPP
#undef PC_QRDRLS_HPP
#undef PCUTILS_HPP
#define DDSPC DDSPCPhase8
#include "phase8/tests/async_shadow.hpp"
#undef DDSPC
#include <array>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
/// Copy declared data/timestamps into the independent committed type without sharing sample storage.
void phase8_record( const DDSPC::OfflineRecord &a /**< [in] Declared record. */,
                    DDSPCPhase8::OfflineRecord &b /**< [out] Pinned-type copy. */ )
{
    b.m_streamEpoch           = a.m_streamEpoch;
    b.m_measurementNs         = a.m_measurementNs;
    b.m_availableNs           = a.m_availableNs;
    b.m_exposureStartNs       = a.m_exposureStartNs;
    b.m_appliedNs             = a.m_appliedNs;
    b.m_appliedAvailableNs    = a.m_appliedAvailableNs;
    b.m_accelNs               = a.m_accelNs;
    b.m_accelAvailableNs      = a.m_accelAvailableNs;
    b.m_data.m_index          = a.m_data.m_index;
    b.m_data.m_residual       = a.m_data.m_residual;
    b.m_data.m_appliedCommand = a.m_data.m_appliedCommand;
    b.m_data.m_acceleration   = a.m_data.m_acceleration;
    b.m_data.m_wfsValid       = a.m_data.m_wfsValid;
    b.m_data.m_appliedValid   = a.m_data.m_appliedValid;
    b.m_data.m_timingValid    = a.m_data.m_timingValid;
    b.m_data.m_accelValid     = a.m_data.m_accelValid;
}
/// Compare every numerical diagnostic, including exact gain/state, without a timing-dependent assertion tolerance.
template <class A, class B>
void phase9_status( const A &a /**< [in] Current status. */, const B &b /**< [in] Committed status. */ )
{
    REQUIRE( a.m_ready == b.m_ready );
    REQUIRE( a.m_trainingEnabled == b.m_trainingEnabled );
    REQUIRE( a.m_training == b.m_training );
    REQUIRE( a.m_dataValid == b.m_dataValid );
    REQUIRE( a.m_predictionValid == b.m_predictionValid );
    REQUIRE( a.m_samples == b.m_samples );
    REQUIRE( a.m_updates == b.m_updates );
    REQUIRE( a.m_predictions == b.m_predictions );
    REQUIRE( a.m_features == b.m_features );
    REQUIRE( a.m_rank == b.m_rank );
    REQUIRE( a.m_rmse == b.m_rmse );
    REQUIRE( static_cast<int>( a.m_reason ) == static_cast<int>( b.m_reason ) );
}
/// \endcond

/// Common policy remains exactly compatible with the actual committed Phase 8 asynchronous implementation.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Phase 9 preserves committed Phase 8 ordered snapshots and rejection semantics",
           "[ddspc][phase9][baseline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
    DDSPC::ShadowController::reset();
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
                DDSPCPhase8::ShadowConfig oldCfg;
                oldCfg.m_modes  = modes;
                oldCfg.m_useQrd = qrd;
                oldCfg.m_gamma  = gamma;
                oldCfg.m_prior  = .1f;
                oldCfg.m_lambda = .3f;
                DDSPC::OfflineAsyncShadow       current( cfg, 8, 640 );
                DDSPCPhase8::OfflineAsyncShadow old( oldCfg, 8, 640 );
                DDSPC::OfflineRecord            record( cfg );
                DDSPCPhase8::OfflineRecord      oldRecord( oldCfg );
                AsyncTestInputs                 input( 8000, 1000000000 );
                DDSPC::Matrix                   out = DDSPC::Matrix::Zero( modes, 1 ), oldOut = out;
                for( int t = 0; t < 640; ++t )
                {
                    CAPTURE( modes, qrd, gamma, t );
                    input.next( t, record );
                    record.m_data.m_accelValid  = t != 250;
                    record.m_data.m_timingValid = t != 300;
                    if( t >= 350 )
                        record.m_data.m_index += 3;
                    if( t == 420 )
                    {
                        current.reset( DDSPC::ShadowMode::Standard );
                        old.reset( DDSPCPhase8::ShadowMode::Standard );
                    }
                    if( t == 500 )
                    {
                        current.stall_until( DDSPC::ShadowMode::AccelAssisted, record.m_availableNs + 6001000 );
                        old.stall_until( DDSPCPhase8::ShadowMode::AccelAssisted, record.m_availableNs + 6001000 );
                    }
                    phase8_record( record, oldRecord );
                    const auto now = record.m_availableNs + 1000;
                    REQUIRE( current.submit( record, now ) == old.submit( oldRecord, now ) );
                    for( int lane = 0; lane < 2; ++lane )
                    {
                        const auto mode = lane ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
                        const auto oldMode =
                            lane ? DDSPCPhase8::ShadowMode::AccelAssisted : DDSPCPhase8::ShadowMode::Standard;
                        REQUIRE( current.manual_step( mode, now ) == old.manual_step( oldMode, now ) );
                        DDSPC::OfflineSnapshot       a( cfg, mode );
                        DDSPCPhase8::OfflineSnapshot b( oldCfg, oldMode );
                        const bool                   valid = current.read_snapshot( mode, a );
                        REQUIRE( valid == old.read_snapshot( oldMode, b ) );
                        if( !valid )
                            continue;
                        phase9_status( a.m_info.m_status, b.m_info.m_status );
                        REQUIRE( a.m_theta.isApprox( b.m_theta, 0 ) );
                        REQUIRE( a.m_gain.isApprox( b.m_gain, 0 ) );
                        REQUIRE( a.m_originState.isApprox( b.m_originState, 0 ) );
                        REQUIRE( a.m_originProposal.isApprox( b.m_originProposal, 0 ) );
                        REQUIRE( a.m_info.m_generation == b.m_info.m_generation );
                        REQUIRE( a.m_info.m_reset == b.m_info.m_reset );
                        REQUIRE( a.m_info.m_publication == b.m_info.m_publication );
                        REQUIRE( a.m_info.m_version == b.m_info.m_version );
                        REQUIRE( a.m_info.m_sequence == b.m_info.m_sequence );
                        REQUIRE( a.m_info.m_publishedNs == b.m_info.m_publishedNs );
                        for( std::uint64_t age : { 999, 1000, 20000000 } )
                        {
                            DDSPC::OfflinePrediction       x;
                            DDSPCPhase8::OfflinePrediction y;
                            REQUIRE( current.predict( mode,
                                                      1,
                                                      record.m_data.m_index,
                                                      record.m_measurementNs,
                                                      now,
                                                      age,
                                                      a.m_originState,
                                                      out,
                                                      x ) == old.predict( oldMode,
                                                                          1,
                                                                          record.m_data.m_index,
                                                                          record.m_measurementNs,
                                                                          now,
                                                                          age,
                                                                          b.m_originState,
                                                                          oldOut,
                                                                          y ) );
                            REQUIRE( static_cast<int>( x.m_reason ) == static_cast<int>( y.m_reason ) );
                            REQUIRE( out.isApprox( oldOut, 0 ) );
                        }
                    }
                }
            }
}

/// A stalled assisted lane cannot disturb standard fitting under the explicit independent policy.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Independent overflow preserves healthy standard and exact admitted-record replays",
           "[ddspc][phase9][overflow]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::process();
#endif
    // clang-format on
    for( bool qrd : { false, true } )
    {
        DDSPC::ShadowConfig cfg;
        cfg.m_modes   = 2;
        cfg.m_history = cfg.m_future = cfg.m_accelHistory = 1;
        cfg.m_prior                                       = .1f;
        cfg.m_lambda                                      = .3f;
        cfg.m_useQrd                                      = qrd;
        DDSPCPhase8::ShadowConfig oldCfg;
        oldCfg.m_modes   = 2;
        oldCfg.m_history = oldCfg.m_future = oldCfg.m_accelHistory = 1;
        oldCfg.m_prior                                             = .1f;
        oldCfg.m_lambda                                            = .3f;
        oldCfg.m_useQrd                                            = qrd;
        DDSPC::OfflineAsyncShadow     current( cfg, 4, 256, 500000, DDSPC::OfflineOverflowPolicy::Independent );
        DDSPCPhase8::ShadowController standard( oldCfg ), assisted( oldCfg );
        standard.training( DDSPCPhase8::ShadowMode::Standard, true );
        assisted.training( DDSPCPhase8::ShadowMode::AccelAssisted, true );
        DDSPC::OfflineRecord       record( cfg );
        DDSPCPhase8::OfflineRecord oldRecord( oldCfg );
        AsyncTestInputs            input( 4000, 1000000000 );
        for( int t = 0; t <= 200; ++t )
        {
            input.next( t, record );
            phase8_record( record, oldRecord );
            if( t == 100 )
                current.stall_until( DDSPC::ShadowMode::AccelAssisted, 1000000000 + 141 * 500000 );
            if( t == 140 )
                for( int j = 0; j < 4; ++j )
                    REQUIRE( current.manual_step( DDSPC::ShadowMode::AccelAssisted, record.m_availableNs ) );
            const auto admission = current.submit_record( record, record.m_availableNs );
            REQUIRE( admission.m_standard );
            REQUIRE( admission.m_assisted == ( t < 104 || t >= 140 ) );
            REQUIRE( current.manual_step( DDSPC::ShadowMode::Standard, record.m_availableNs ) );
            standard.process( oldRecord.m_data );
            DDSPC::OfflineSnapshot s( cfg, DDSPC::ShadowMode::Standard );
            REQUIRE( current.read_snapshot( DDSPC::ShadowMode::Standard, s ) );
            phase9_status( s.m_info.m_status, standard.status( DDSPCPhase8::ShadowMode::Standard ) );
            REQUIRE( s.m_theta.isApprox( standard.prediction_matrix( DDSPCPhase8::ShadowMode::Standard ), 0 ) );
            REQUIRE( s.m_info.m_segmentFirstSequence == 0 );
            if( t < 100 || t >= 140 )
            {
                REQUIRE( current.manual_step( DDSPC::ShadowMode::AccelAssisted, record.m_availableNs ) );
                assisted.process( oldRecord.m_data );
                DDSPC::OfflineSnapshot a( cfg, DDSPC::ShadowMode::AccelAssisted );
                REQUIRE( current.read_snapshot( DDSPC::ShadowMode::AccelAssisted, a ) );
                phase9_status( a.m_info.m_status, assisted.status( DDSPCPhase8::ShadowMode::AccelAssisted ) );
                REQUIRE(
                    a.m_theta.isApprox( assisted.prediction_matrix( DDSPCPhase8::ShadowMode::AccelAssisted ), 0 ) );
                REQUIRE( a.m_info.m_segmentFirstSequence == ( t < 100 ? 0 : 140 ) );
            }
        }
        REQUIRE( current.stats( DDSPC::ShadowMode::Standard ).m_overflows == 0 );
        REQUIRE( current.stats( DDSPC::ShadowMode::Standard ).m_accepted == 201 );
        REQUIRE( current.stats( DDSPC::ShadowMode::AccelAssisted ).m_overflows == 36 );
        REQUIRE( current.stats( DDSPC::ShadowMode::AccelAssisted ).m_discarded == 4 );
        input.next( 201, record );
        record.m_appliedAvailableNs = record.m_availableNs + 1;
        const auto invalid          = current.submit_record( record, record.m_availableNs );
        REQUIRE_FALSE( invalid.m_standard );
        REQUIRE_FALSE( invalid.m_assisted );
        DDSPC::OfflineSnapshot s( cfg, DDSPC::ShadowMode::Standard );
        REQUIRE_FALSE( current.read_snapshot( DDSPC::ShadowMode::Standard, s ) );
    }
}
} // namespace libXWCTest::loPredCtrlAccTest
