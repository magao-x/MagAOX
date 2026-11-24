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
    using counterOffload::m_calibLeak;
    using counterOffload::m_calibRunning;
    using counterOffload::m_channel;
    using counterOffload::m_clipped;
    using counterOffload::m_collecting;
    using counterOffload::m_command;
    using counterOffload::m_durationGridUs;
    using counterOffload::m_enabled;
    using counterOffload::m_events;
    using counterOffload::m_finalZero;
    using counterOffload::m_fomWindowMs;
    using counterOffload::m_gain;
    using counterOffload::m_gainGrid;
    using counterOffload::m_latency;
    using counterOffload::m_latencyCount;
    using counterOffload::m_leak;
    using counterOffload::m_maxCommand;
    using counterOffload::m_missed;
    using counterOffload::m_modeMax;
    using counterOffload::m_offloadStamps;
    using counterOffload::m_onsetSigma;
    using counterOffload::m_overflows;
    using counterOffload::m_rampDurationUs;
    using counterOffload::m_ramps;
    using counterOffload::m_rampStartUs;
    using counterOffload::m_residualValues;
    using counterOffload::m_settleS;
    using counterOffload::m_slotNs;
    using counterOffload::m_startGridUs;
    using counterOffload::m_stepAmp;
    using counterOffload::m_stepCount;
    using counterOffload::m_stepData;
    using counterOffload::m_stepMode;
    using counterOffload::m_stepPeriodS;
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

        // with no attached stream there is no woofer stamp, but the latency is still pending
        REQUIRE( app.m_ramps[0].handoffNs == -1 );
        REQUIRE( app.m_ramps[0].latencyPending );
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

    REQUIRE( app.calibParams() == counterOffload::calibParamsT{ 0, 330, 0.5 } );
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
    counterOffload::applyCalibSetting(m_stepCount,0,0,0);
    counterOffload::applyCalibStep(0,0);
    counterOffload::stepElementName(0);
    counterOffload::applyCalibGrid(m_gainGrid,{},0,0);
    counterOffload::calibKeys();
    counterOffload::applyCalibKey({},0);
    counterOffload::calibKeyValue({});
    counterOffload::calibSettingsString();
    #endif
    // clang-format on

    counterOffload_test app;

    SECTION( "the defaults step astig00 by 0.7 ten times 0.5 s apart, and use valid grids" )
    {
        REQUIRE( app.m_settleS == 0.2 );
        REQUIRE( app.m_stepPeriodS == 0.5 );
        REQUIRE( app.m_stepCount == 10 );
        REQUIRE( app.m_onsetSigma == 5 );
        REQUIRE( app.m_fomWindowMs == 50 );
        REQUIRE( app.m_stepMode == 5 );
        REQUIRE( app.m_stepAmp == 0.7 );
        REQUIRE( app.m_timeoutS == 5 );
        REQUIRE( app.m_calibLeak > 0 );
        REQUIRE( app.m_startGridUs == std::vector<double>{ 0, 150, 25 } );
        REQUIRE( app.m_durationGridUs == std::vector<double>{ 200, 600, 100 } );
        REQUIRE( app.m_gainGrid == std::vector<double>{ 0.3, 0.7, 0.1 } );
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
        REQUIRE( app.applyCalibSetting( app.m_stepCount, 1, 2, 1000 ) < 0 ); // the halves need at least 2 steps
        REQUIRE( app.applyCalibSetting( app.m_stepCount, 4.5, 2, 1000 ) < 0 );
        REQUIRE( app.m_stepCount == 10 );
        REQUIRE( app.applyCalibSetting( app.m_stepCount, 6, 2, 1000 ) == 0 );
        REQUIRE( app.m_stepCount == 6 );

        REQUIRE( app.applyCalibSetting( app.m_calibLeak, 0, 1e-9, 1 ) < 0 ); // calibration must leak
        REQUIRE( app.applyCalibSetting( app.m_calibLeak, std::numeric_limits<double>::quiet_NaN(), 1e-9, 1 ) < 0 );
        REQUIRE( app.applyCalibSetting( app.m_calibLeak, 0.005, 1e-9, 1 ) == 0 );
        REQUIRE( app.m_calibLeak == 0.005 );
    }

    SECTION( "the config keys cover the 15 numeric settings, and every default is accepted" )
    {
        const auto keys = app.calibKeys();
        REQUIRE( keys.size() == 15 );

        std::set<std::string> names;
        for( const auto &k : keys )
        {
            names.insert( k.key );
            REQUIRE( ( k.real == nullptr ) != ( k.integer == nullptr ) );
            REQUIRE( app.applyCalibKey( k, counterOffload::calibKeyValue( k ) ) == 0 );
        }
        REQUIRE( names.size() == keys.size() );

        const auto stepCount = std::find_if(
            keys.begin(), keys.end(), []( const auto &k ) { return std::string( k.key ) == "stepCount"; } );
        REQUIRE( stepCount != keys.end() );
        REQUIRE( app.applyCalibKey( *stepCount, 4.5 ) < 0 ); // integer settings must be integral
        REQUIRE( app.applyCalibKey( *stepCount, 1 ) < 0 );   // below the limit
        REQUIRE( app.applyCalibKey( *stepCount, 6 ) == 0 );
        REQUIRE( app.m_stepCount == 6 );
    }

    SECTION( "the settings are logged on one line" )
    {
        const std::string s = app.calibSettingsString();
        REQUIRE( s.find( "stepCount=10" ) != std::string::npos );
        REQUIRE( s.find( "stepMode=5" ) != std::string::npos );
        REQUIRE( s.find( "stepAmp=0.7" ) != std::string::npos );
        REQUIRE( s.find( "leak=0.0025" ) != std::string::npos );
        REQUIRE( s.find( "gainGrid=0.3,0.7,0.1" ) != std::string::npos );
        REQUIRE( s.find( "sigmaFloor=0.002" ) != std::string::npos );
    }

    SECTION( "the step mode must be an integer >= 0 and the amplitude finite and > 0" )
    {
        REQUIRE( app.applyCalibStep( 5.5, 0.7 ) < 0 );
        REQUIRE( app.applyCalibStep( -1, 0.7 ) < 0 );
        REQUIRE( app.applyCalibStep( 6, 0 ) < 0 );
        REQUIRE( app.applyCalibStep( 6, std::numeric_limits<double>::quiet_NaN() ) < 0 );
        REQUIRE( app.m_stepMode == 5 );
        REQUIRE( app.m_stepAmp == 0.7 );
        REQUIRE( app.applyCalibStep( 6, 0.5 ) == 0 );
        REQUIRE( app.m_stepMode == 6 );
        REQUIRE( app.m_stepAmp == 0.5 );
    }

    SECTION( "step modes are named as dmMode names its elements" )
    {
        REQUIRE( counterOffload::stepElementName( 5 ) == "0005" );
        REQUIRE( counterOffload::stepElementName( 12 ) == "0012" );
    }

    SECTION( "grids are checked before they are set" )
    {
        REQUIRE( app.applyCalibGrid( app.m_gainGrid, { 0.5, 2.5, 0.1 }, 0, 2 ) < 0 );
        REQUIRE( app.m_gainGrid == std::vector<double>{ 0.3, 0.7, 0.1 } );
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
        REQUIRE( app.applyCalibStep( 6, 0.5 ) < 0 );
        REQUIRE( app.m_stepMode == 5 );
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

        REQUIRE( app.estimatePoints() == 52 );
        REQUIRE( app.estimatePoints() * ( app.m_settleS + app.m_stepCount * app.m_stepPeriodS ) / 60 < 20 );
    }

    SECTION( "stage names" )
    {
        REQUIRE( std::string( counterOffload::calibStageName( counterOffload::stageClipped ) ) == "clipped" );
        REQUIRE( std::string( counterOffload::calibStageName( counterOffload::stageFitGain ) ) == "fit gain" );
        REQUIRE( std::string( counterOffload::calibStageName( counterOffload::stageStepOnly ) ) == "step only" );
        REQUIRE( std::string( counterOffload::calibStageName( 99 ) ) == "unknown" );
    }

    SECTION( "the clip fraction is clipped over published slots, 0 with none published" )
    {
        REQUIRE( counterOffload::clipFraction( 103, 8000 ) == Approx( 103.0 / 8000 ) );
        REQUIRE( counterOffload::clipFraction( 0, 8000 ) == 0 );
        REQUIRE( counterOffload::clipFraction( 5, 0 ) == 0 );
    }
}

/// Build a series of 3 woofer steps of 2 modes, 200 frames 500 us apart, whose onset is 3 frames after each command.
/** The residual is an offset of 0.1 in mode 0, plus each step's sign times its response: exp(-j/5) in mode 0 and half
 * that in mode 1, for 40 frames after the onset.  Offloads arrive 2 frames after the onset and every 8 frames after
 * that.  With error, the first offload of each step leaves the sign times 0.2 in mode 0 for 2 frames, then 0.1 for 6.
 */
counterOffload::stepDataT stepSeries( bool withError /**< [in] add the counter-offload error */,
                                      bool respond1 = true /**< [in] whether the second step has a response */ )
{
    counterOffload::stepDataT d;

    d.nModes   = 2;
    d.cmdIndex = { 150, 350, 550 };
    d.signs    = { 1, -1, 1 };

    constexpr size_t nFrames = 750;

    d.frames.assign( nFrames * 2, 0 );

    for( size_t i = 0; i < nFrames; ++i )
    {
        d.frames[i * 2] = 0.1f;
        d.frameNs.push_back( 1000000 + static_cast<int64_t>( i ) * 500000 );
    }

    for( size_t k = 0; k < d.cmdIndex.size(); ++k )
    {
        const size_t onset = d.cmdIndex[k] + 3;
        const size_t end   = ( k + 1 < d.cmdIndex.size() ) ? d.cmdIndex[k + 1] : nFrames;

        if( k == 1 && !respond1 )
        {
            continue;
        }

        for( size_t j = 0; j < 40; ++j )
        {
            d.frames[( onset + j ) * 2] += d.signs[k] * std::exp( -double( j ) / 5 );
            d.frames[( onset + j ) * 2 + 1] += d.signs[k] * 0.5 * std::exp( -double( j ) / 5 );
        }

        for( size_t i = onset + 2; i < end; i += 8 )
        {
            d.offloadNs.push_back( d.frameNs[i] - 100 ); // just before the frame it lands in
        }

        if( withError )
        {
            for( size_t l = 0; l < 8; ++l )
            {
                d.frames[( onset + 2 + l ) * 2] += d.signs[k] * ( ( l < 2 ) ? 0.2f : 0.1f );
            }
        }
    }

    return d;
}

/// Verify that each step's onset, offloads, pre-step mean and noise are found in the residual.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload locates woofer steps in the residual", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::findSteps({},0,{});
    #endif
    // clang-format on

    std::vector<counterOffload::stepT> steps;

    SECTION( "the onset is the first frame off the pre-step level, and offloads are placed by their stamps" )
    {
        REQUIRE( counterOffload::findSteps( stepSeries( false ), 5, steps ) == 3 );
        REQUIRE( steps.size() == 3 );

        for( size_t k = 0; k < 3; ++k )
        {
            REQUIRE( steps[k].found );
            REQUIRE( steps[k].cmd == 150 + 200 * k );
            REQUIRE( steps[k].onset == 150 + 200 * k + 3 );
            REQUIRE( steps[k].offload == long( steps[k].onset + 2 ) );
            REQUIRE( steps[k].offloads.size() > 1 );
            REQUIRE( steps[k].preMean[0] == Approx( 0.1 ) );
            REQUIRE( steps[k].preMean[1] == Approx( 0 ).margin( 1e-12 ) );
            REQUIRE( steps[k].noiseVar == Approx( 0 ).margin( 1e-12 ) );
        }

        REQUIRE( steps[0].end == 350 );
        REQUIRE( steps[2].end == 750 );
    }

    SECTION( "a step with no response is not found" )
    {
        REQUIRE( counterOffload::findSteps( stepSeries( false, false ), 5, steps ) == 2 );
        REQUIRE_FALSE( steps[1].found );
    }

    SECTION( "a step with too few frames before its command is not used" )
    {
        counterOffload::stepDataT d = stepSeries( false );
        d.cmdIndex[0]               = 5;

        counterOffload::findSteps( d, 5, steps );
        REQUIRE_FALSE( steps[0].found );
    }
}

/// Verify the step-only response and the step figure of merit with its timing and gain parts.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload computes the step figure of merit", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::stepOnlyResponse({},{},0,{},{});
    counterOffload::stepFom({},{},{},{},0,0);
    #endif
    // clang-format on

    const counterOffload::stepDataT    only = stepSeries( false );
    std::vector<counterOffload::stepT> steps;
    counterOffload::findSteps( only, 5, steps );

    std::vector<double> response, noise;
    REQUIRE( counterOffload::stepOnlyResponse( only, steps, 60, response, noise ) == 3 );
    REQUIRE( response.size() == 60 * 2 );
    REQUIRE( noise.size() == 60 );
    REQUIRE( response[0] == Approx( 1 ) );
    REQUIRE( response[1] == Approx( 0.5 ) );
    REQUIRE( response[5 * 2] == Approx( std::exp( -1.0 ) ) );
    REQUIRE( response[50 * 2] == Approx( 0 ).margin( 1e-12 ) );
    REQUIRE( noise[0] == Approx( 0 ).margin( 1e-12 ) ); // identical steps

    const counterOffload::stepDataT    d = stepSeries( true );
    std::vector<counterOffload::stepT> ws;
    counterOffload::findSteps( d, 5, ws );

    SECTION( "the step-only noise is the scatter of the steps about their mean, over n (n - 1)" )
    {
        counterOffload::stepDataT n3 = stepSeries( false );
        n3.frames[( 153 + 10 ) * 2] += 0.3f; // step 0, lag 10, mode 0

        std::vector<counterOffload::stepT> s3;
        counterOffload::findSteps( n3, 5, s3 );

        std::vector<double> r3, v3;
        counterOffload::stepOnlyResponse( n3, s3, 60, r3, v3 );

        REQUIRE( r3[10 * 2] == Approx( std::exp( -2.0 ) + 0.1 ) );
        REQUIRE( v3[10] == Approx( ( 0.2 * 0.2 + 0.1 * 0.1 + 0.1 * 0.1 ) / 6 ) );
        REQUIRE( v3[11] == Approx( 0 ).margin( 1e-12 ) );
    }

    SECTION( "a step whose response is the step alone has no error" )
    {
        const counterOffload::stepFomT f = counterOffload::stepFom( only, steps, response, noise, 30, 30 );

        REQUIRE( f.used == 3 );
        REQUIRE( f.fom == Approx( 0 ).margin( 1e-12 ) );
    }

    SECTION( "the error after the first offload is split into timing and gain lags" )
    {
        const counterOffload::stepFomT f = counterOffload::stepFom( d, ws, response, noise, 30, 30 );

        REQUIRE( f.used == 3 );
        REQUIRE( f.lags == 30 );
        REQUIRE( f.fom == Approx( 2 * 0.04 + 6 * 0.01 ) );
        REQUIRE( f.fomTiming == Approx( 2 * 0.04 ) ); // lags 0 and 1 after each offload
        REQUIRE( f.fomGain == Approx( 6 * 0.01 ) );
        REQUIRE( f.fomStd == 0 ); // the second half has only 1 step
        REQUIRE( f.onsetDelay == Approx( 3 ) );
        REQUIRE( f.onsetDelaySd == Approx( 0 ).margin( 1e-12 ) );
        REQUIRE( f.phase == Approx( 2 ) );
        REQUIRE( f.phaseSd == Approx( 0 ).margin( 1e-12 ) );
    }

    SECTION( "only the window is summed, and the profile covers all lags" )
    {
        const counterOffload::stepFomT f = counterOffload::stepFom( d, ws, response, noise, 1, 30 );

        REQUIRE( f.lags == 1 );
        REQUIRE( f.fom == Approx( 0.04 ) );
        REQUIRE( f.fomTiming == Approx( 0.04 ) );
        REQUIRE( f.lagEnergy.size() == 30 );
        REQUIRE( f.lagEnergy[2] == Approx( 0.01 ) );
        REQUIRE( f.lagEnergy[9] == Approx( 0 ).margin( 1e-12 ) );
    }

    SECTION( "noise independent between steps is removed" )
    {
        counterOffload::stepDataT dn  = stepSeries( true );
        const float               v[] = { 0.3f, -0.3f, 0 };

        for( size_t k = 0; k < 3; ++k ) // lag 5 after the first offload, mode 1, sign-corrected v
        {
            dn.frames[( 150 + 200 * k + 3 + 2 + 5 ) * 2 + 1] += dn.signs[k] * v[k];
        }

        std::vector<counterOffload::stepT> sn;
        counterOffload::findSteps( dn, 5, sn );

        // The mean of v is 0, and (|sum v|^2 - sum v^2) / (n (n - 1)) = -0.18 / 6 is its unbiased contribution.
        REQUIRE( counterOffload::stepFom( dn, sn, response, noise, 30, 30 ).fom == Approx( 0.14 - 0.03 ) );
    }

    SECTION( "the step-only response's noise is removed at the lags used" )
    {
        const std::vector<double> flat( 60, 0.01 );

        REQUIRE( counterOffload::stepFom( d, ws, response, flat, 30, 30 ).fom == Approx( 0.14 - 30 * 0.01 ) );
    }

    SECTION( "a step whose first offload is too long after its onset is not used" )
    {
        counterOffload::stepDataT dl = stepSeries( true );
        dl.offloadNs.erase( std::remove_if( dl.offloadNs.begin(),
                                            dl.offloadNs.end(),
                                            [&dl]( int64_t t ) { return t > dl.frameNs[353] && t < dl.frameNs[375]; } ),
                            dl.offloadNs.end() );

        std::vector<counterOffload::stepT> sl;
        counterOffload::findSteps( dl, 5, sl );

        REQUIRE( sl[1].offload - long( sl[1].onset ) > long( counterOffload::m_maxPhaseFrames ) );
        REQUIRE( counterOffload::stepFom( dl, sl, response, noise, 30, 30 ).used == 2 );
    }

    SECTION( "nothing is computed with fewer than 2 steps, no lags, or a short noise vector" )
    {
        std::vector<counterOffload::stepT> one = ws;
        one[1].found = one[2].found = false;

        REQUIRE( counterOffload::stepFom( d, one, response, noise, 30, 30 ).used == 0 );
        REQUIRE( counterOffload::stepFom( d, ws, response, noise, 0, 30 ).used == 0 );
        REQUIRE( counterOffload::stepFom( d, ws, response, {}, 30, 30 ).used == 0 );
    }
}

/// Verify that step onsets are moved to where each step best matches the step-only response.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload aligns step onsets on the step-only response", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::alignSteps({},{},{},0,0);
    counterOffload::onsetDelayStats({},0,0);
    #endif
    // clang-format on

    const counterOffload::stepDataT    only = stepSeries( false );
    std::vector<counterOffload::stepT> steps;
    counterOffload::findSteps( only, 5, steps );

    std::vector<double> response, noise;
    counterOffload::stepOnlyResponse( only, steps, 60, response, noise );

    const counterOffload::stepDataT    d = stepSeries( true );
    std::vector<counterOffload::stepT> ws;
    counterOffload::findSteps( d, 5, ws );

    const double exact = counterOffload::stepFom( d, ws, response, noise, 30, 30 ).fom;

    SECTION( "an onset detected 2 frames late is moved back, and the error is as if it were found exactly" )
    {
        ws[1].onset += 2; // the first offload at or after it is still frame 355
        REQUIRE_FALSE( counterOffload::stepFom( d, ws, response, noise, 30, 30 ).fom == Approx( exact ) );

        REQUIRE( counterOffload::alignSteps( d, ws, response, 8, 4 ) == 1 );
        REQUIRE( ws[1].onset == 353 );
        REQUIRE( ws[1].offload == 355 );
        REQUIRE( counterOffload::stepFom( d, ws, response, noise, 30, 30 ).fom == Approx( exact ) );
    }

    SECTION( "an onset detected 2 frames early is moved forward" )
    {
        ws[1].onset -= 2;

        REQUIRE( counterOffload::alignSteps( d, ws, response, 8, 4 ) == 1 );
        REQUIRE( ws[1].onset == 353 );
        REQUIRE( ws[1].offload == 355 );
    }

    SECTION( "correctly detected onsets are not moved" )
    {
        REQUIRE( counterOffload::alignSteps( d, ws, response, 8, 4 ) == 0 );
        REQUIRE( ws[0].onset == 153 );
    }

    SECTION( "an onset is never moved before its command" )
    {
        ws[0].cmd = ws[0].onset; // as if the command came at the detected onset
        ws[0].onset += 2;

        counterOffload::alignSteps( d, ws, response, 8, 4 );
        REQUIRE( ws[0].onset >= ws[0].cmd );
    }

    SECTION( "the onset delay statistics are over the steps found" )
    {
        double mean, sd;

        ws[1].onset += 2;
        counterOffload::onsetDelayStats( ws, mean, sd );
        REQUIRE( mean == Approx( 3 + 2.0 / 3 ) );
        REQUIRE( sd == Approx( std::sqrt( ( 2 * ( 2.0 / 3 ) * ( 2.0 / 3 ) + ( 4.0 / 3 ) * ( 4.0 / 3 ) ) / 2 ) ) );

        ws[0].found = ws[1].found = ws[2].found = false;
        counterOffload::onsetDelayStats( ws, mean, sd );
        REQUIRE( mean == 0 );
        REQUIRE( sd == 0 );
    }
}

/// Verify that residual frames and offload stamps are collected only while a calibration is collecting.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload collects residual frames and offload stamps", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::processImage(nullptr, residualShmimT());
    #endif
    // clang-format on

    counterOffload_test app;

    std::vector<float> frame{ 1, 2, 3, 4 };
    auto               d = delta( 1 );

    app.m_residualValues   = 4;
    app.m_stepData.nModes  = 2;

    SECTION( "nothing is collected while not collecting" )
    {
        app.processImage( frame.data(), residualShmimT() );
        app.processImage( d.data(), dev::shmimT() );
        REQUIRE( app.m_stepData.frames.empty() );
        REQUIRE( app.m_offloadStamps.empty() );
    }

    SECTION( "the first nModes values of each frame are collected, and offloads even while disabled" )
    {
        app.m_collecting = true;
        app.processImage( frame.data(), residualShmimT() );
        app.processImage( frame.data(), residualShmimT() );
        app.processImage( d.data(), dev::shmimT() );

        REQUIRE( app.m_stepData.frames == std::vector<float>{ 1, 2, 1, 2 } );
        REQUIRE( app.m_stepData.frameNs.size() == 2 );
        REQUIRE( app.m_offloadStamps.size() == 1 );
        REQUIRE( app.m_events == 0 ); // disabled: the offload is not ramped
    }
}

/// Verify the woofer-to-receipt handoff taken from the offload frame's timestamps.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload measures the handoff from the woofer stamp", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::stampedHandoffNs(timespec(), timespec(), timespec());
    #endif
    // clang-format on

    const timespec atime{ 100, 999990000 };
    const timespec writetime{ 100, 999995000 };

    SECTION( "a stamped frame gives receipt minus atime, across a second boundary" )
    {
        REQUIRE( counterOffload::stampedHandoffNs( atime, writetime, timespec{ 101, 20000 } ) == 30000 );
    }

    SECTION( "a frame posted without a woofer stamp is rejected" )
    {
        REQUIRE( counterOffload::stampedHandoffNs( writetime, writetime, timespec{ 101, 0 } ) == -1 );
    }

    SECTION( "negative and stale handoffs are rejected" )
    {
        REQUIRE( counterOffload::stampedHandoffNs( atime, writetime, timespec{ 100, 0 } ) == -1 );
        REQUIRE( counterOffload::stampedHandoffNs( atime, writetime, timespec{ 102, 0 } ) == -1 );
    }
}

/// Verify the latency window statistics.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload summarizes the latency window", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::summarizeLatency(nullptr, 0);
    counterOffload::latencyElements();
    #endif
    // clang-format on

    SECTION( "an empty window has no data" )
    {
        const counterOffload::latencySummaryT sum = counterOffload::summarizeLatency( nullptr, 0 );

        REQUIRE( sum.pipelineUs == -1 );
        REQUIRE( sum.slotWaitUs == -1 );
        REQUIRE( sum.computeMaxUs == -1 );
    }

    SECTION( "means, pipeline RMS and largest compute; unstamped offloads leave out handoff and pipeline" )
    {
        // pipelines 10 + 2 + 1 = 13 and 20 + 4 + 1 = 25; the third has no stamp
        const counterOffload::latencySampleT samples[] = {
            { 10, 50, 2, 1 }, { 20, 70, 4, 1 }, { -1, 90, 6, 1 } };

        const counterOffload::latencySummaryT sum = counterOffload::summarizeLatency( samples, 3 );

        REQUIRE( sum.handoffUs == Approx( 15 ) );
        REQUIRE( sum.pipelineUs == Approx( 19 ) );
        REQUIRE( sum.pipelineRmsUs == Approx( 6 ) );
        REQUIRE( sum.slotWaitUs == Approx( 70 ) );
        REQUIRE( sum.computeUs == Approx( 4 ) );
        REQUIRE( sum.writeUs == Approx( 1 ) );
        REQUIRE( sum.computeMaxUs == Approx( 6 ) );
    }

    SECTION( "the INDI element order" )
    {
        REQUIRE( counterOffload::latencyElements().size() == 9 );
        REQUIRE( counterOffload::latencyElements().front() == "pipelineUs" );
        REQUIRE( counterOffload::latencyElements().back() == "missed" );
    }
}

/// Verify latency is recorded once per offload, at the first slot published after it arrived.
/**
 * \ingroup counterOffload_unit_test
 */
TEST_CASE( "counterOffload records each offload's latency at its first slot", "[counterOffload]" )
{
    // clang-format off
    #ifdef COUNTEROFFLOAD_TEST_DOXYGEN_REF
    counterOffload::recordLatency(0, 0, 0, 0);
    counterOffload::updateSlotPeriod(synco::Request());
    #endif
    // clang-format on

    counterOffload_test app;

    const std::vector<float> d( 4, 1.0f );
    const int64_t            us = 1000;

    counterOffload::rampT *early = app.addRamp( d.data(), 0 );
    counterOffload::rampT *late  = app.addRamp( d.data(), 0 );
    REQUIRE( early != nullptr );
    REQUIRE( late != nullptr );

    early->wakeNs         = 1000 * us;
    early->handoffNs      = 12 * us;
    early->latencyPending = true;

    late->wakeNs         = 1200 * us; // after this slot's compute start
    late->latencyPending = true;

    SECTION( "only offloads received before the compute start are recorded, once" )
    {
        app.recordLatency( 1100 * us, 1103 * us, 1150 * us, 1151 * us );

        REQUIRE( app.m_latencyCount == 1 );
        REQUIRE( app.m_latency[0].handoffUs == Approx( 12 ) );
        REQUIRE( app.m_latency[0].slotWaitUs == Approx( 100 ) );
        REQUIRE( app.m_latency[0].computeUs == Approx( 3 ) );
        REQUIRE( app.m_latency[0].writeUs == Approx( 1 ) );
        REQUIRE_FALSE( early->latencyPending );
        REQUIRE( late->latencyPending );

        app.recordLatency( 1225 * us, 1228 * us, 1275 * us, 1276 * us );

        REQUIRE( app.m_latencyCount == 2 );
        REQUIRE( app.m_latency[1].handoffUs == -1 ); // no woofer stamp
        REQUIRE( app.m_latency[1].slotWaitUs == Approx( 25 ) );
    }

    SECTION( "an offload that waited a whole slot or more is missed" )
    {
        app.m_slotNs = 125 * us;
        app.recordLatency( 1100 * us, 1103 * us, 1150 * us, 1151 * us );
        REQUIRE( app.m_missed == 0 ); // the early offload waited 100 us

        app.recordLatency( 1330 * us, 1333 * us, 1380 * us, 1381 * us );
        REQUIRE( app.m_missed == 1 ); // the late offload waited 130 us
    }

    SECTION( "a zero drops pending offloads" )
    {
        app.clearState();
        app.recordLatency( 2000 * us, 2003 * us, 2050 * us, 2051 * us );

        REQUIRE( app.m_latencyCount == 0 );
    }

    SECTION( "the window keeps the latest samples" )
    {
        for( size_t k = 0; k < counterOffload::m_latencyWindow + 5; ++k )
        {
            early->latencyPending = true;
            app.recordLatency( 1100 * us, 1103 * us, 1150 * us, 1151 * us );
        }

        REQUIRE( app.m_latencyCount == counterOffload::m_latencyWindow );
    }

    SECTION( "the slot period comes from consecutive steps of one cycle" )
    {
        synco::Request a{};
        a.cycle    = 7;
        a.step     = 1;
        a.targetNs = 10000 * us;

        synco::Request b = a;
        b.step           = 2;
        b.targetNs       = 10125 * us;

        synco::Request c = b;
        c.cycle          = 8;
        c.step           = 0;
        c.targetNs       = 10500 * us; // a new cycle, not consecutive steps

        app.updateSlotPeriod( a );
        REQUIRE( app.m_slotNs == 0 );

        app.updateSlotPeriod( b );
        REQUIRE( app.m_slotNs == 125 * us );

        app.updateSlotPeriod( c );
        REQUIRE( app.m_slotNs == 125 * us );
    }
}

} // namespace counterOffloadTest

} // namespace libXWCTest
