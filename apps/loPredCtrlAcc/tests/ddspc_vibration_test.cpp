/** \file ddspc_vibration_test.cpp
 * \brief Offline plant oracles, causal prefix replay, bounded corrections and actual worker fault gates.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "vibration.hpp"

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond TEST_SUPPORT
/// Keep deterministic fault tests short while retaining the same model dimensions/rank threshold.
VibrationConfig short_vibration()
{
    VibrationConfig c;
    c.m_train     = 600;
    c.m_evaluate  = 400;
    c.m_broadband = true;
    return c;
}
/// Assert the model gate and physical-input bounds on every evaluated frame, not only averaged metrics.
void vibration_safety( const VibrationConfig &c /**< [in] Explicit limits. */,
                       const VibrationResult &r /**< [in] Actual generated trajectory. */ )
{
    for( int model = 0; model < 5; ++model )
    {
        REQUIRE( r.m_invalidChanges[model] == 0 );
        for( int k = c.m_train; k < c.m_train + c.m_evaluate; ++k )
        {
            const auto &f = r.m_frames[model][k];
            REQUIRE( f.m_residual.allFinite() );
            REQUIRE( f.m_applied.allFinite() );
            REQUIRE( f.m_requested.cwiseAbs().maxCoeff() <= c.m_limit );
            if( model >= 2 && !f.m_valid )
                REQUIRE( f.m_requested.isApprox( r.m_frames[model][k - 1].m_requested, 0 ) );
            if( model >= 3 && f.m_valid )
            {
                REQUIRE( f.m_modelSequence < std::uint64_t( f.m_origin ) );
                REQUIRE( f.m_ageNs <= c.m_maxAgeNs );
            }
        }
    }
}
/// \endcond

/// Check the acceleration integration and full DM inversion against independent equations.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Synthetic acceleration and perfect-prediction DM obey their declared equations", "[ddspc][vibration]" )
{
    auto c     = short_vibration();
    c.m_qrd    = GENERATE( false, true );
    c.m_peak28 = c.m_peak56 = false;
    c.m_accelNoise          = 0;
    const auto    s         = vibration_source( c );
    DDSPC::Matrix mix( 2, 2 );
    mix << 1.f, .2f, -.1f, 1.f;
    for( int i = 1; i < c.m_train; ++i )
        REQUIRE( ( .01f * mix.inverse() * s.m_acceleration[i] -
                   ( s.m_disturbance[i + 1] - 2 * s.m_disturbance[i] + s.m_disturbance[i - 1] ) )
                     .norm() < 2e-7 );
    for( int delay : { 0, 1, 3 } )
    {
        c.m_dmExtraFrames = delay;
        c.m_dmScale       = .7;
        c.m_dmPole        = .2;
        const auto input  = vibration_source( c );
        const auto r      = vibration_run( c, input );
        vibration_safety( c, r );
        for( int i = c.m_train + 20; i < c.m_train + c.m_evaluate; ++i )
        {
            REQUIRE( r.m_frames[1][i].m_residual.norm() < 1e-6 );
            REQUIRE( ( r.m_frames[0][i].m_residual - input.m_disturbance[i] ).norm() < 1e-6 );
        }
    }
}

/// Actual controllers cannot change a prefix when only unavailable future source payloads are changed.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Closed-loop DDSPC receives no future truth and shares common calibration", "[ddspc][vibration]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::calculate_command();
    DDSPC::OfflineAsyncShadow::predict();
    DDSPC::OfflineAsyncShadow::submit_record();
#endif
    // clang-format on
    auto c       = short_vibration();
    c.m_qrd      = GENERATE( false, true );
    auto       s = vibration_source( c );
    const auto a = vibration_run( c, s ), repeat = vibration_run( c, s );
    const int  cut = c.m_train + 200;
    for( std::size_t i = cut; i < s.m_disturbance.size(); ++i )
    {
        s.m_disturbance[i].array() += 10;
        s.m_acceleration[i].array() -= 20;
        s.m_wfsNoise[i].array() += 2;
        s.m_exploration[i].setZero();
    }
    const auto b = vibration_run( c, s );
    for( int model : { 0, 2, 3, 4 } )
        for( int i = 0; i < c.m_train + c.m_evaluate; ++i )
        {
            const auto &f    = a.m_frames[model][i];
            const auto &same = repeat.m_frames[model][i];
            REQUIRE( f.m_requested.isApprox( same.m_requested, 0 ) );
            REQUIRE( f.m_predictionSquared == same.m_predictionSquared );
            REQUIRE( f.m_valid == same.m_valid );
            REQUIRE( f.m_version == same.m_version );
            if( i < cut )
            {
                REQUIRE( f.m_requested.isApprox( b.m_frames[model][i].m_requested, 0 ) );
                REQUIRE( f.m_residual.isApprox( b.m_frames[model][i].m_residual, 0 ) );
            }
            if( i < c.m_train )
            {
                REQUIRE( f.m_measured.isApprox( a.m_frames[0][i].m_measured, 0 ) );
                REQUIRE( f.m_applied.isApprox( a.m_frames[0][i].m_applied, 0 ) );
            }
        }
    vibration_safety( c, a );
    auto invalid              = vibration_source( c );
    invalid.m_sensorIndex[20] = 21;
    REQUIRE_THROWS_AS( vibration_run( c, invalid ), std::invalid_argument );
    invalid                   = vibration_source( c );
    invalid.m_availableNs[20] = invalid.m_availableNs[19];
    REQUIRE_THROWS_AS( vibration_run( c, invalid ), std::invalid_argument );
}

/// Rejections never issue a new correction, including real prototype overflow/reset/staleness gates.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Closed-loop faults hold commands without bypassing model validity", "[ddspc][vibration]" )
{
    const bool qrd = GENERATE( false, true );
    for( int fault = 0; fault < 7; ++fault )
    {
        auto c          = short_vibration();
        c.m_qrd         = qrd;
        const int event = c.m_train + 80;
        if( fault == 0 )
            c.m_controlGain = 0;
        if( fault == 1 )
        {
            c.m_dmScale = -1;
            c.m_limit   = .2;
        }
        if( fault == 2 )
            c.m_publicationFrames = 6;
        if( fault == 3 )
            c.m_resetFrame = event;
        if( fault == 4 )
        {
            c.m_stallFrame  = event;
            c.m_stallFrames = 32;
            c.m_capacity    = 4;
        }
        if( fault == 5 )
        {
            c.m_missingFrame  = event;
            c.m_missingFrames = 20;
        }
        if( fault == 6 )
        {
            c.m_wfsDelayFrames    = 1;
            c.m_jitterNs          = 30000;
            c.m_sensorDelayFrames = 1;
        }
        CAPTURE( fault );
        const auto s = vibration_source( c );
        const auto r = vibration_run( c, s );
        vibration_safety( c, r );
        for( int i = 0; i < c.m_train + c.m_evaluate; ++i )
            if( s.m_sensorIndex[i] >= 0 )
            {
                const auto j = s.m_sensorIndex[i];
                REQUIRE( s.m_sensorNs[j] <= vibration_time( i ) );
                REQUIRE( s.m_sensorAvailableNs[j] >=
                         vibration_time( j ) - 125000 + 1000 + c.m_sensorDelayFrames * 500000ULL );
                REQUIRE( s.m_sensorAvailableNs[j] <= s.m_availableNs[i] );
            }
        if( fault == 0 )
            for( int model = 2; model < 5; ++model )
                for( int i = c.m_train + 20; i < c.m_train + c.m_evaluate; ++i )
                    REQUIRE( r.m_frames[model][i].m_requested.isZero( 0 ) );
        if( fault == 2 )
            for( int model : { 3, 4 } )
                for( int i = c.m_train; i < c.m_train + c.m_evaluate; ++i )
                    REQUIRE_FALSE( r.m_frames[model][i].m_valid );
        if( fault == 3 )
            for( int model : { 3, 4 } )
                REQUIRE_FALSE( r.m_frames[model][event].m_valid );
        if( fault == 4 )
            for( int model : { 3, 4 } )
                REQUIRE( r.m_overflows[model] > 0 );
        if( fault == 5 )
            for( int i = event + 4; i < event + 16; ++i )
                REQUIRE_FALSE( r.m_frames[4][i].m_valid );
    }
}
/// A fixed coherent broadband positive control must learn D and improve the actual closed-loop residual.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Coherent synthetic acceleration improves QR closed-loop residuals", "[ddspc][vibration]" )
{
    VibrationConfig c;
    c.m_qrd             = true;
    c.m_broadband       = true;
    const auto result   = vibration_run( c, vibration_source( c ) );
    double     standard = 0, assisted = 0;
    bool       learnedD = false;
    for( int i = c.m_train + 1000; i < c.m_train + c.m_evaluate; ++i )
    {
        standard += result.m_frames[3][i].m_measured.squaredNorm();
        assisted += result.m_frames[4][i].m_measured.squaredNorm();
        learnedD = learnedD || result.m_frames[4][i].m_dNorm > .01;
    }
    REQUIRE( learnedD );
    REQUIRE( assisted < standard );
    vibration_safety( c, result );
}
} //namespace libXWCTest::loPredCtrlAccTest
