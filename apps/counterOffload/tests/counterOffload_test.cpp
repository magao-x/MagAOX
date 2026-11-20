/** \file counterOffload_test.cpp
 * \brief Catch2 tests for the counterOffload app.
 *
 * \ingroup counterOffload_files
 */

#include "../../../tests/testXWC.hpp"

#include "../counterOffload.hpp"

#include <limits>

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
    using counterOffload::m_channel;
    using counterOffload::m_command;
    using counterOffload::m_enabled;
    using counterOffload::m_events;
    using counterOffload::m_finalZero;
    using counterOffload::m_gain;
    using counterOffload::m_leak;
    using counterOffload::m_overflows;
    using counterOffload::m_rampDurationUs;
    using counterOffload::m_rampStartUs;
    using counterOffload::m_ramps;
    using counterOffload::m_syncopationTiming;
    using counterOffload::m_timeOffsetUs;
    using counterOffload::m_zeroRequested;

    /// Set up 2x2 buffers and a gain of 0.5.
    counterOffload_test()
    {
        setupBuffers( 2, 2 );
        m_gain = 0.5;
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
        REQUIRE( app.m_accum( 1, 0 ) == Approx( 2 ) );
    }

    SECTION( "the hand-over to the accumulator is continuous" )
    {
        app.buildCommand( 449 * us );
        float before = app.m_command( 0, 0 );
        app.buildCommand( 451 * us );
        REQUIRE( app.m_command( 0, 0 ) == Approx( before ).margin( 1e-4 ) );
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
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 2 ) );
        REQUIRE_FALSE( app.needPublish() );
    }

    SECTION( "a partial leak decays geometrically per publication" )
    {
        app.m_leak = 0.5;
        app.afterPublish();
        app.afterPublish();
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 0.5 ) );
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
        REQUIRE( app.m_accum( 0, 0 ) == Approx( 2 ) );
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

    SECTION( "leak and gain must be within 0 to 1" )
    {
        app.m_leak = 1.5;
        REQUIRE( app.validateConfig() < 0 );
        app.m_leak = 0.5;
        app.m_gain = -0.1;
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

} // namespace counterOffloadTest

} // namespace libXWCTest
