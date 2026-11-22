/** \file counterOffload_test.cpp
 * \brief Catch2 tests for the counterOffload app.
 *
 * \ingroup counterOffload_files
 */

#include "../../../tests/testXWC.hpp"

#include "../counterOffload.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <string>

using namespace MagAOX::app;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace MagAOX
{
namespace app
{
/// Expose the counter-offload state without opening any streams or starting threads.
class counterOffload_test : public counterOffload
{
  public:
    using counterOffload::m_accum;
    using counterOffload::m_accumNonzero;
    using counterOffload::m_bandCenterHz;
    using counterOffload::m_bandHalfWidthHz;
    using counterOffload::m_bandModes;
    using counterOffload::m_bandValid;
    using counterOffload::m_calibLeak;
    using counterOffload::m_calibRunning;
    using counterOffload::m_channel;
    using counterOffload::m_clipped;
    using counterOffload::m_command;
    using counterOffload::m_df;
    using counterOffload::m_durationGridUs;
    using counterOffload::m_enabled;
    using counterOffload::m_events;
    using counterOffload::m_finalZero;
    using counterOffload::m_gain;
    using counterOffload::m_gainGrid;
    using counterOffload::m_leak;
    using counterOffload::m_maxCommand;
    using counterOffload::m_modeMax;
    using counterOffload::m_nAverage;
    using counterOffload::m_overflows;
    using counterOffload::m_psdAfterNs;
    using counterOffload::m_psdNFreq;
    using counterOffload::m_psdNModes;
    using counterOffload::m_psdPowers;
    using counterOffload::m_rampDurationUs;
    using counterOffload::m_ramps;
    using counterOffload::m_rampStartUs;
    using counterOffload::m_settleS;
    using counterOffload::m_startGridUs;
    using counterOffload::m_syncopationTiming;
    using counterOffload::m_timeOffsetUs;
    using counterOffload::m_timeoutS;
    using counterOffload::m_zeroRequested;

    /// Set up 2x2 buffers, a ramp from 120 to 450 us, a gain of 0.5, and a clip level high enough not to act.
    counterOffload_test()
    {
        setupBuffers( 2, 2 );
        m_rampStartUs = 120;
        m_gain        = 0.5;
        m_maxCommand  = 100;
    }

    /// Number of unfinished ramps.
    int active()
    {
        int n = 0;
        for( const auto &r : m_ramps )
        {
            n += r.used;
        }
        return n;
    }
};
} // namespace app
} // namespace MagAOX
/// \endcond

namespace libXWCTest
{

/** \defgroup counterOffload_unit_test counterOffload Unit Tests
 * \brief Unit tests for the counterOffload application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `counterOffload` unit tests.
/** \ingroup counterOffload_unit_test
 */
namespace counterOffloadTest
{

/// 1 us in ns.
constexpr int64_t us = 1000;

/// A 2x2 offload change with every pixel set to a value.
mx::improc::eigenImage<float> delta( float value /**< [in] the pixel value */ )
{
    mx::improc::eigenImage<float> d( 2, 2 );
    d.setConstant( value );
    return d;
}

/// Verify the raised-cosine ramp shape.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload ramp fraction is a raised cosine", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::rampFraction(0,0,0);
    #endif
    // clang-format on

    REQUIRE( counterOffload::rampFraction( 0, 120, 330 ) == 0 );
    REQUIRE( counterOffload::rampFraction( 120, 120, 330 ) == 0 );
    REQUIRE( counterOffload::rampFraction( 285, 120, 330 ) == Approx( 0.5 ) );
    REQUIRE( counterOffload::rampFraction( 450, 120, 330 ) == 1 );
    REQUIRE( counterOffload::rampFraction( 1e6, 120, 330 ) == 1 );

    float prev = 0;
    for( double t = 100; t <= 500; t += 5 )
    {
        float f = counterOffload::rampFraction( t, 120, 330 );
        REQUIRE( f >= prev );
        REQUIRE( std::isfinite( f ) );
        prev = f;
    }
}

/// Verify a single offload ramps to the negative of its change, then hands over to the accumulator.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload commands a single ramped offload", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::addRamp(nullptr,0);
    counterOffload::buildCommand(0);
    counterOffload::needPublish();
    #endif
    // clang-format on

    counterOffload_test app;
    auto                d = delta( 2 );

    REQUIRE_FALSE( app.needPublish() );

    app.addRamp( d.data(), 0 );
    REQUIRE( app.m_events == 1 );
    REQUIRE( app.needPublish() );

    SECTION( "before the ramp starts the command is zero" )
    {
        app.buildCommand( 100 * us );
        REQUIRE( app.m_command.abs().maxCoeff() == 0 );
        REQUIRE( app.active() == 1 );
    }

    SECTION( "at the midpoint the command is half of -gain * delta" )
    {
        app.buildCommand( 285 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -0.5 ) );
        REQUIRE( app.m_command( 1, 1 ) == Approx( -0.5 ) );
    }

    SECTION( "after the ramp ends the change moves to the accumulator" )
    {
        app.buildCommand( 500 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -1 ) );
        REQUIRE( app.active() == 0 );
        REQUIRE( app.m_accumNonzero );
        REQUIRE( app.m_accum( 1, 0 ) == Approx( 1 ) );
    }

    SECTION( "the hand-over to the accumulator is continuous" )
    {
        app.buildCommand( 449 * us );
        float before = app.m_command( 0, 0 );
        app.buildCommand( 451 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( before ).margin( 1e-4 ) );
    }
}

/// Verify the gain is applied when each offload arrives, not to the held contribution.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload applies the gain when an offload arrives", "[counterOffload]" )
{
    counterOffload_test app;
    auto                d = delta( 2 );

    app.addRamp( d.data(), 0 ); // at gain 0.5
    app.buildCommand( 500 * us );
    REQUIRE( app.m_command( 0, 0 ) == Approx( -1 ) );

    SECTION( "a gain change does not change the held contribution" )
    {
        app.m_gain = 1;
        app.buildCommand( 600 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -1 ) );

        app.m_gain = 0;
        app.buildCommand( 700 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -1 ) );
    }

    SECTION( "new offloads use the new gain" )
    {
        app.m_gain = 1;
        app.addRamp( d.data(), 600 * us );
        app.buildCommand( 1100 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -3 ) );

        app.m_gain = 0;
        app.addRamp( d.data(), 1100 * us );
        app.buildCommand( 1600 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -3 ) );
    }
}

/// Verify the command and the held contribution are clipped to maxCommand.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload clips the command and the accumulator", "[counterOffload]" )
{
    counterOffload_test app;
    app.m_maxCommand = 0.6;

    auto d = delta( 2 ); // 1 after the 0.5 gain

    SECTION( "a ramp in progress is clipped in the command only" )
    {
        app.addRamp( d.data(), 0 );
        app.addRamp( d.data(), 0 );
        app.buildCommand( 285 * us ); // both at their midpoints: -1 before clipping
        REQUIRE( app.m_command( 0, 0 ) == Approx( -0.6 ) );
        REQUIRE( app.m_clipped == 1 );
    }

    SECTION( "the accumulator cannot wind up beyond the clip level" )
    {
        for( int n = 0; n < 10; ++n )
        {
            app.addRamp( d.data(), n * 1000 * us );
            app.buildCommand( ( n * 1000 + 500 ) * us );
        }

        REQUIRE( app.m_accum.abs().maxCoeff() == Approx( 0.6 ) );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -0.6 ) );
        REQUIRE( app.m_clipped == 10 );

        // an offload of the opposite sign comes off the clip level at once
        auto neg = delta( -2 );
        app.addRamp( neg.data(), 20000 * us );
        app.buildCommand( 20500 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( 0.4 ) );
    }

    SECTION( "commands within the clip level are not counted" )
    {
        app.m_gain = 0.25;
        app.addRamp( d.data(), 0 );
        app.buildCommand( 500 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -0.5 ) );
        REQUIRE( app.m_clipped == 0 );
    }
}

/// Verify overlapping offloads add.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload sums overlapping ramps", "[counterOffload]" )
{
    counterOffload_test app;
    auto                d1 = delta( 2 );
    auto                d2 = delta( 4 );

    app.addRamp( d1.data(), 0 );
    app.addRamp( d2.data(), 165 * us ); // second ramp starts 165 us later

    app.buildCommand( 450 * us ); // first complete, second at its midpoint
    REQUIRE( app.m_command( 0, 0 ) == Approx( -0.5 * ( 2 + 0.5 * 4 ) ) );
    REQUIRE( app.active() == 1 );

    app.buildCommand( 1000 * us );
    REQUIRE( app.m_command( 0, 0 ) == Approx( -0.5 * 6 ) );
    REQUIRE( app.active() == 0 );
}

/// Verify the leak applied after each publication.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload leaks the completed contribution", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::afterPublish();
    #endif
    // clang-format on

    counterOffload_test app;
    auto                d = delta( 2 );

    app.addRamp( d.data(), 0 );
    app.buildCommand( 500 * us );

    SECTION( "leak 0 holds the contribution and stops publishing" )
    {
        app.m_leak = 0;
        app.afterPublish();
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 1 ) );
        REQUIRE_FALSE( app.needPublish() );
    }

    SECTION( "a partial leak decays geometrically per publication" )
    {
        app.m_leak = 0.5;
        app.afterPublish();
        app.afterPublish();
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 0.25 ) );
        REQUIRE( app.needPublish() );
        app.buildCommand( 600 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( -0.25 ) );
    }

    SECTION( "leak 1 clears at the first publication, then publishes one zero" )
    {
        app.m_leak = 1;
        app.afterPublish();
        REQUIRE( app.m_accum.abs().maxCoeff() == 0 );
        REQUIRE_FALSE( app.m_accumNonzero );
        REQUIRE( app.m_finalZero );
        REQUIRE( app.needPublish() );

        app.buildCommand( 600 * us );
        REQUIRE( app.m_command.abs().maxCoeff() == 0 );
        app.afterPublish();
        REQUIRE_FALSE( app.needPublish() );
    }

    SECTION( "a small leak eventually reaches exactly zero" )
    {
        app.m_leak = 0.1;
        int n      = 0;
        while( app.m_accumNonzero && n < 10000 )
        {
            app.afterPublish();
            ++n;
        }
        REQUIRE( n < 10000 );
        REQUIRE( app.m_accum.abs().maxCoeff() == 0 );
    }
}

/// Verify clearing, pool overflow and disabled input.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload state handling", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::clearState();
    counterOffload::processImage(nullptr, dev::shmimT());
    counterOffload::setEnabled(false);
    #endif
    // clang-format on

    counterOffload_test app;
    auto                d = delta( 2 );

    SECTION( "clearState removes ramps and the accumulator" )
    {
        app.addRamp( d.data(), 0 );
        app.buildCommand( 500 * us );
        app.addRamp( d.data(), 500 * us );
        app.clearState();
        REQUIRE( app.active() == 0 );
        REQUIRE_FALSE( app.m_accumNonzero );
        REQUIRE( app.m_command.abs().maxCoeff() == 0 );
        REQUIRE_FALSE( app.needPublish() );
    }

    SECTION( "a full pool folds the oldest ramp into the accumulator" )
    {
        for( size_t n = 0; n < counterOffload::m_maxRamps + 1; ++n )
        {
            app.addRamp( d.data(), int64_t( n ) * us );
        }
        REQUIRE( app.m_overflows == 1 );
        REQUIRE( app.active() == int( counterOffload::m_maxRamps ) );
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 1 ) );
    }

    SECTION( "offloads are ignored while disabled" )
    {
        app.processImage( d.data(), dev::shmimT() );
        REQUIRE( app.m_events == 0 );

        app.m_enabled = true;
        app.processImage( d.data(), dev::shmimT() );
        REQUIRE( app.m_events == 1 );
        REQUIRE( app.active() == 1 );
    }

    SECTION( "disabling requests a zero, enabling does not" )
    {
        REQUIRE( app.setEnabled( true ) );
        REQUIRE_FALSE( app.m_zeroRequested );
        REQUIRE_FALSE( app.setEnabled( true ) );

        app.addRamp( d.data(), 0 );
        app.buildCommand( 500 * us );
        app.addRamp( d.data(), 500 * us );

        REQUIRE( app.setEnabled( false ) );
        REQUIRE_FALSE( app.m_enabled );
        REQUIRE( app.m_zeroRequested );
        REQUIRE( app.needPublish() ); // the producer thread must not go idle before the zero is published

        REQUIRE_FALSE( app.setEnabled( false ) );
    }
}

/// Verify configuration validation.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload validates its configuration", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::validateConfig();
    #endif
    // clang-format on

    counterOffload_test app;
    app.m_gain = 1;

    REQUIRE( app.validateConfig() < 0 ); // a timing stream is required

    app.m_syncopationTiming = "dmtweeter_synco_timing";
    REQUIRE( app.validateConfig() == 0 );

    SECTION( "the channel must be an auxiliary source" )
    {
        app.m_channel = "dm01disp08";
        REQUIRE( app.validateConfig() < 0 );
    }

    SECTION( "the timing stream cannot alias the channel" )
    {
        app.m_syncopationTiming = app.m_channel;
        REQUIRE( app.validateConfig() < 0 );
    }

    SECTION( "leak must be within 0 to 1 and gain within 0 to 2" )
    {
        app.m_leak = 1.5;
        REQUIRE( app.validateConfig() < 0 );
        app.m_leak = 0.5;
        app.m_gain = -0.1;
        REQUIRE( app.validateConfig() < 0 );
        app.m_gain = 1.5;
        REQUIRE( app.validateConfig() == 0 );
        app.m_gain = 2.5;
        REQUIRE( app.validateConfig() < 0 );
    }

    SECTION( "the clip level must be positive" )
    {
        app.m_maxCommand = 0;
        REQUIRE( app.validateConfig() < 0 );
    }

    SECTION( "nonfinite or nonpositive ramp parameters are rejected" )
    {
        app.m_rampStartUs = std::numeric_limits<double>::quiet_NaN();
        REQUIRE( app.validateConfig() < 0 );
        app.m_rampStartUs    = 120;
        app.m_rampDurationUs = 0;
        REQUIRE( app.validateConfig() < 0 );
    }
}

/// Verify the default ramp and gain are the best measured on the bench.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload operating defaults", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::calibParams();
    #endif
    // clang-format on

    counterOffload app; // only one MagAOXApp may exist at a time, so not alongside a test harness

    REQUIRE( app.calibParams() == counterOffload::calibParamsT{ 25, 330, 0.5 } );
}

/// Verify the calibration setting defaults, and the checks applied when they are set from config or over INDI.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload calibration settings", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::gridValid({},0,0);
    counterOffload::applyCalibSetting(m_nAverage,0,0,0);
    counterOffload::applyCalibGrid(m_gainGrid,{},0,0);
    counterOffload::calibKeys();
    counterOffload::applyCalibKey({},0);
    counterOffload::calibKeyValue({});
    counterOffload::calibSettingsString();
    #endif
    // clang-format on

    counterOffload_test app;

    SECTION( "the defaults wait out the 10.5 s modalPSDs average, take 3 samples, and use valid grids" )
    {
        REQUIRE( app.m_settleS == 11 );
        REQUIRE( app.m_nAverage == 3 );
        REQUIRE( app.m_timeoutS == 5 );
        REQUIRE( app.m_calibLeak > 0 );
        REQUIRE( app.m_startGridUs == std::vector<double>{ 0, 150, 25 } );
        REQUIRE( app.m_durationGridUs == std::vector<double>{ 200, 600, 100 } );
        REQUIRE( app.m_gainGrid == std::vector<double>{ 0.3, 1.0, 0.1 } );
        REQUIRE( counterOffload::gridValid( app.m_startGridUs, 0, 1e6 ) );
        REQUIRE( counterOffload::gridValid( app.m_durationGridUs, 1, 1e6 ) );
        REQUIRE( counterOffload::gridValid( app.m_gainGrid, 0, 2 ) );
    }

    SECTION( "a grid is min, max, step within the limits" )
    {
        REQUIRE_FALSE( counterOffload::gridValid( { 0, 400 }, 0, 1e6 ) );        // step required
        REQUIRE_FALSE( counterOffload::gridValid( { 400, 0, 25 }, 0, 1e6 ) );    // max below min
        REQUIRE_FALSE( counterOffload::gridValid( { 0, 400, 0 }, 0, 1e6 ) );     // step must be positive
        REQUIRE_FALSE( counterOffload::gridValid( { 0.5, 2.5, 0.1 }, 0, 2 ) );   // above the gain limit
        REQUIRE( counterOffload::gridValid( { 0.5, 0.5, 0.1 }, 0, 2 ) );         // a single point
    }

    SECTION( "settings are range checked, and integer settings must be integral" )
    {
        REQUIRE( app.applyCalibSetting( app.m_nAverage, 1, 2, 1000 ) < 0 ); // a single PSD is never used for a score
        REQUIRE( app.applyCalibSetting( app.m_nAverage, 4.5, 2, 1000 ) < 0 );
        REQUIRE( app.m_nAverage == 3 );
        REQUIRE( app.applyCalibSetting( app.m_nAverage, 5, 2, 1000 ) == 0 );
        REQUIRE( app.m_nAverage == 5 );

        REQUIRE( app.applyCalibSetting( app.m_calibLeak, 0, 1e-9, 1 ) < 0 ); // calibration must leak
        REQUIRE( app.applyCalibSetting( app.m_calibLeak, std::numeric_limits<double>::quiet_NaN(), 1e-9, 1 ) < 0 );
        REQUIRE( app.applyCalibSetting( app.m_calibLeak, 0.005, 1e-9, 1 ) == 0 );
        REQUIRE( app.m_calibLeak == 0.005 );
    }

    SECTION( "the config keys cover the 13 numeric settings, and every default is accepted" )
    {
        const auto keys = app.calibKeys();
        REQUIRE( keys.size() == 13 );

        std::set<std::string> names;
        for( const auto &k : keys )
        {
            names.insert( k.key );
            REQUIRE( ( k.real == nullptr ) != ( k.integer == nullptr ) );
            REQUIRE( app.applyCalibKey( k, counterOffload::calibKeyValue( k ) ) == 0 );
        }
        REQUIRE( names.size() == keys.size() );

        const auto nAverage = std::find_if(
            keys.begin(), keys.end(), []( const auto &k ) { return std::string( k.key ) == "nAverage"; } );
        REQUIRE( nAverage != keys.end() );
        REQUIRE( app.applyCalibKey( *nAverage, 4.5 ) < 0 ); // integer settings must be integral
        REQUIRE( app.applyCalibKey( *nAverage, 1 ) < 0 );   // below the limit
        REQUIRE( app.applyCalibKey( *nAverage, 6 ) == 0 );
        REQUIRE( app.m_nAverage == 6 );
    }

    SECTION( "the settings are logged on one line" )
    {
        const std::string s = app.calibSettingsString();
        REQUIRE( s.find( "nAverage=3" ) != std::string::npos );
        REQUIRE( s.find( "leak=0.0025" ) != std::string::npos );
        REQUIRE( s.find( "gainGrid=0.3,1,0.1" ) != std::string::npos );
    }

    SECTION( "grids are checked before they are set" )
    {
        REQUIRE( app.applyCalibGrid( app.m_gainGrid, { 0.5, 2.5, 0.1 }, 0, 2 ) < 0 );
        REQUIRE( app.m_gainGrid == std::vector<double>{ 0.3, 1.0, 0.1 } );
        REQUIRE( app.applyCalibGrid( app.m_gainGrid, { 0.4, 1.2, 0.2 }, 0, 2 ) == 0 );
        REQUIRE( app.m_gainGrid == std::vector<double>{ 0.4, 1.2, 0.2 } );
    }

    SECTION( "nothing changes while a calibration runs" )
    {
        app.m_calibRunning = true;
        REQUIRE( app.applyCalibSetting( app.m_modeMax, 10, 0, 1e6 ) < 0 );
        REQUIRE( app.m_modeMax == 0 );
        REQUIRE( app.applyCalibGrid( app.m_startGridUs, { 0, 100, 25 }, 0, 1e6 ) < 0 );
        REQUIRE( app.m_startGridUs == std::vector<double>{ 0, 150, 25 } );
    }
}

/// Verify the PSD band integral used as the calibration figure of merit.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload integrates the PSD band", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::bandPower(nullptr,0,0,0,0,0,0);
    #endif
    // clang-format on

    // 11 frequencies 0-50 Hz at 5 Hz, 3 modes; mode m has value (m+1) at every frequency, plus 100 at 25 Hz in mode 0.
    constexpr uint32_t nf = 11, nm = 3;
    std::vector<float> psd( nf * nm );
    for( uint32_t m = 0; m < nm; ++m )
    {
        for( uint32_t n = 0; n < nf; ++n )
        {
            psd[n + m * nf] = m + 1;
        }
    }
    psd[5] += 100;

    SECTION( "the band includes its edges and is weighted by df" )
    {
        // 20, 25, 30 Hz in mode 0
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 25, 5, 1 ) == Approx( ( 3 + 100 ) * 5 ) );
    }

    SECTION( "only the first modeMax modes are summed" )
    {
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 25, 0, 2 ) == Approx( ( 101 + 2 ) * 5 ) );
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 25, 0, 10 ) == Approx( ( 101 + 2 + 3 ) * 5 ) );
    }

    SECTION( "a band past the last frequency is clipped" )
    {
        // 45, 50 Hz in mode 0
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 50, 7, 1 ) == Approx( 2 * 5 ) );
    }

    SECTION( "an empty band or mode range returns -1" )
    {
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 80, 5, 1 ) == -1 );
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 22, 1, 1 ) == -1 ); // between bins
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 5, 25, 5, 0 ) == -1 );
        REQUIRE( counterOffload::bandPower( psd.data(), nf, nm, 0, 25, 5, 1 ) == -1 );
    }
}

/// Verify the baseline interpolation used for the ratios.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload interpolates the baseline in time", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::interpBaseline(0,{},{});
    #endif
    // clang-format on

    const std::vector<int64_t> t{ 10, 20, 40 };
    const std::vector<double>  f{ 2, 4, 1 };

    REQUIRE( counterOffload::interpBaseline( 15, t, f ) == Approx( 3 ) );
    REQUIRE( counterOffload::interpBaseline( 30, t, f ) == Approx( 2.5 ) );
    REQUIRE( counterOffload::interpBaseline( 20, t, f ) == Approx( 4 ) );
    REQUIRE( counterOffload::interpBaseline( 0, t, f ) == 2 );  // constant before the first
    REQUIRE( counterOffload::interpBaseline( 99, t, f ) == 1 ); // and after the last
    REQUIRE( counterOffload::interpBaseline( 5, {}, {} ) == 0 );
}

/// Verify the sweep order spreads neighbouring values in time.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload sweep order", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::sweepOrder(0);
    #endif
    // clang-format on

    REQUIRE( counterOffload::sweepOrder( 7 ) == std::vector<size_t>{ 0, 4, 2, 6, 1, 5, 3 } );
    REQUIRE( counterOffload::sweepOrder( 1 ) == std::vector<size_t>{ 0 } );
    REQUIRE( counterOffload::sweepOrder( 0 ).empty() );

    for( size_t n : { 2, 5, 8, 9, 15 } )
    {
        std::vector<size_t> o = counterOffload::sweepOrder( n );
        std::sort( o.begin(), o.end() );

        std::vector<size_t> all( n );
        std::iota( all.begin(), all.end(), 0 );

        REQUIRE( o == all ); // a permutation
    }
}

/// Verify the quadratic fit.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload quadratic fit", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::fitQuadratic({},{});
    #endif
    // clang-format on

    auto q = []( double x )
    {
        return 0.4 + 0.002 * ( x - 60 ) * ( x - 60 ) / 100;
    };

    std::vector<double> xs{ 0, 25, 50, 75, 100, 125, 150 }, rs;
    for( double x : xs )
    {
        rs.push_back( q( x ) );
    }

    SECTION( "an exact quadratic is recovered with no residual" )
    {
        const counterOffload::quadFitT fit = counterOffload::fitQuadratic( xs, rs );

        REQUIRE( fit.ok );
        REQUIRE( fit( 60 ) == Approx( 0.4 ) );
        REQUIRE( fit( 137 ) == Approx( q( 137 ) ) );
        REQUIRE( fit.resSd == Approx( 0 ).margin( 1e-9 ) );
    }

    SECTION( "the residual standard deviation has n - 3 degrees of freedom" )
    {
        rs[3] += 0.1; // one outlier of 0.1

        const counterOffload::quadFitT fit = counterOffload::fitQuadratic( xs, rs );

        REQUIRE( fit.ok );
        REQUIRE( fit.resSd > 0.03 );
        REQUIRE( fit.resSd < 0.1 );
    }

    SECTION( "too few points or a single x value cannot be fitted" )
    {
        REQUIRE_FALSE( counterOffload::fitQuadratic( { 0, 1 }, { 1, 2 } ).ok );
        REQUIRE_FALSE( counterOffload::fitQuadratic( { 1, 1, 1 }, { 1, 2, 3 } ).ok );
    }
}

/// Verify the decision for one parameter at the end of a sweep.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload chooses parameter values beyond the noise", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::chooseValue({},{},0,0,0);
    #endif
    // clang-format on

    const std::vector<double> xs{ 0, 25, 50, 75, 100, 125, 150 };

    auto ratios = [&xs]( double vertex, double curvature )
    {
        std::vector<double> rs;
        for( double x : xs )
        {
            rs.push_back( 0.4 + curvature * ( x - vertex ) * ( x - vertex ) );
        }
        return rs;
    };

    SECTION( "the vertex inside the sweep is the candidate, and a large improvement moves" )
    {
        const counterOffload::choiceT ch = counterOffload::chooseValue( xs, ratios( 60, 2e-5 ), 150, 0.05, 1.5 );

        REQUIRE( ch.candidate == Approx( 60 ) );
        REQUIRE( ch.rCandidate == Approx( 0.4 ) );
        REQUIRE( ch.rCurrent == Approx( 0.4 + 2e-5 * 90 * 90 ) );
        REQUIRE( ch.threshold == Approx( 1.5 * 0.05 * std::sqrt( 6.0 / 7 ) ) );
        REQUIRE( ch.move );
    }

    SECTION( "an improvement within the noise does not move" )
    {
        const counterOffload::choiceT ch = counterOffload::chooseValue( xs, ratios( 60, 2e-5 ), 25, 0.05, 1.5 );

        REQUIRE( ch.candidate == Approx( 60 ) );
        REQUIRE( ch.rCurrent - ch.rCandidate < ch.threshold );
        REQUIRE_FALSE( ch.move );
    }

    SECTION( "a minimum beyond the sweep gives the lower end" )
    {
        REQUIRE( counterOffload::chooseValue( xs, ratios( -100, 2e-5 ), 150, 0.01, 1.5 ).candidate == Approx( 0 ) );
        REQUIRE( counterOffload::chooseValue( xs, ratios( 60, -2e-5 ), 60, 0.01, 1.5 ).candidate == Approx( 150 ) );
    }

    SECTION( "a current value outside the sweep always moves" )
    {
        const counterOffload::choiceT ch = counterOffload::chooseValue( xs, ratios( 60, 2e-5 ), 400, 10, 1.5 );

        REQUIRE( ch.outside );
        REQUIRE( ch.move );
    }

    SECTION( "with fewer than 4 points a running mean is used" )
    {
        const counterOffload::choiceT ch =
            counterOffload::chooseValue( { 0.3, 0.4, 0.5 }, { 0.7, 0.5, 0.45 }, 0.3, 0.01, 1.5 );

        REQUIRE( ch.candidate == Approx( 0.5 ) );
        REQUIRE( ch.rCandidate == Approx( 0.475 ) );
        REQUIRE( ch.rCurrent == Approx( 0.6 ) );
        REQUIRE( ch.move );
    }

    SECTION( "with no points nothing moves" )
    {
        const counterOffload::choiceT ch = counterOffload::chooseValue( {}, {}, 25, 0.05, 1.5 );

        REQUIRE( ch.candidate == 25 );
        REQUIRE_FALSE( ch.move );
    }
}

/// Verify the pooled noise estimate, the run length, the stage names and the clip fraction.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload noise estimate and run length", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::estimatePoints();
    counterOffload::calibStageName(0);
    counterOffload::clipFraction(0, 0);
    #endif
    // clang-format on

    SECTION( "standard deviations pool by degrees of freedom, with a floor" )
    {
        counterOffload::noiseT noise;
        REQUIRE( noise.sigma( 0.02 ) == 0.02 );

        noise.add( 0.06, 2 );
        noise.add( 0.03, 4 );
        REQUIRE( noise.sigma( 0.02 ) == Approx( std::sqrt( ( 0.0036 * 2 + 0.0009 * 4 ) / 6 ) ) );
        REQUIRE( noise.sigma( 0.1 ) == 0.1 );

        noise.add( 1, 0 ); // no degrees of freedom adds nothing
        REQUIRE( noise.dof == 6 );
    }

    SECTION( "a run with the defaults takes well under 20 minutes" )
    {
        counterOffload_test app;

        REQUIRE( app.estimatePoints() == 58 );
        REQUIRE( app.estimatePoints() * ( app.m_settleS + 0.5 * app.m_nAverage ) / 60 < 20 );
    }

    SECTION( "stage names" )
    {
        REQUIRE( std::string( counterOffload::calibStageName( counterOffload::stageClipped ) ) == "clipped" );
        REQUIRE( std::string( counterOffload::calibStageName( counterOffload::stageFitGain ) ) == "fit gain" );
        REQUIRE( std::string( counterOffload::calibStageName( 99 ) ) == "unknown" );
    }

    SECTION( "the clip fraction is clipped over published slots, 0 with none published" )
    {
        REQUIRE( counterOffload::clipFraction( 103, 8000 ) == Approx( 103.0 / 8000 ) );
        REQUIRE( counterOffload::clipFraction( 0, 8000 ) == 0 );
        REQUIRE( counterOffload::clipFraction( 5, 0 ) == 0 );
    }
}

/// Verify that PSDs are collected only during an evaluation and after the settling time.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload collects PSD band powers after a change", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::processImage(nullptr, psdShmimT());
    #endif
    // clang-format on

    counterOffload_test app;

    std::vector<float> psd( 11 * 2, 1.0f );

    app.m_psdNFreq        = 11;
    app.m_psdNModes       = 2;
    app.m_df              = 5;
    app.m_bandCenterHz    = 25;
    app.m_bandHalfWidthHz = 5;
    app.m_bandModes       = 1;

    SECTION( "nothing is collected without a band" )
    {
        app.processImage( psd.data(), psdShmimT() );
        REQUIRE( app.m_psdPowers.empty() );
    }

    SECTION( "PSDs received before the settled time are ignored" )
    {
        app.m_bandValid  = true;
        app.m_psdAfterNs = synco::monotonicNs() + 1000000000000LL;
        app.processImage( psd.data(), psdShmimT() );
        REQUIRE( app.m_psdPowers.empty() );
    }

    SECTION( "PSDs received after the settled time are collected" )
    {
        app.m_bandValid  = true;
        app.m_psdAfterNs = 0;
        app.processImage( psd.data(), psdShmimT() );
        app.processImage( psd.data(), psdShmimT() );
        REQUIRE( app.m_psdPowers.size() == 2 );
        REQUIRE( app.m_psdPowers[0] == Approx( 3 * 5 ) );
    }
}

} // namespace counterOffloadTest

} // namespace libXWCTest
