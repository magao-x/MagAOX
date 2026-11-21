/** \file dmSpeckle_test.cpp
 * \brief Deterministic tests for legacy and syncopation-managed dmSpeckle operation.
 * \ingroup dmSpeckle_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmSpeckle.hpp"

#include <limits>
#include <cstdio>

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup dmSpeckle_unit_test dmSpeckle Unit Tests
 * \brief Unit tests for the dmSpeckle application.
 * \ingroup application_unit_test
 */

/// Namespace for dmSpeckle application tests.
/** \ingroup dmSpeckle_unit_test */
namespace dmSpeckleTest
{

/// \cond DMSPECKLE_TEST_HARNESS
/// Expose narrow provider seams without starting any app threads or opening instrument streams.
class dmSpeckleHarness : public dmSpeckle
{
  public:
    using MagAOXApp<true>::config;
    using dmSpeckle::fillTipTilt;
    using dmSpeckle::m_amp;
    using dmSpeckle::m_dmChannelName;
    using dmSpeckle::m_dwell;
    using dmSpeckle::m_frequency;
    using dmSpeckle::m_height;
    using dmSpeckle::m_imageStream;
    using dmSpeckle::m_indiP_loopLock;
    using dmSpeckle::m_indiP_modulating;
    using dmSpeckle::m_indiP_trigger;
    using dmSpeckle::m_indiP_zero;
    using dmSpeckle::m_loopLock;
    using dmSpeckle::m_modulating;
    using dmSpeckle::m_opened;
    using dmSpeckle::m_opMode;
    using dmSpeckle::m_shapes;
    using dmSpeckle::m_syncopationTiming;
    using dmSpeckle::m_syncopationZeroRequested;
    using dmSpeckle::m_trigger;
    using dmSpeckle::m_typeSize;
    using dmSpeckle::m_width;
    using dmSpeckle::m_writeLeadUs;
    using dmSpeckle::opModeName;
    using dmSpeckle::prepareSyncopationCommand;
    using dmSpeckle::syncopationMainOnly;
    using dmSpeckle::validateSyncopationConfig;
};
/// \endcond

/// Check that managed operation is opt-in and cannot select a main destination or legacy trigger.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle provider configuration isolates auxiliary writes", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::validateSyncopationConfig();
    #endif
    // clang-format on
    dmSpeckleHarness app;
    REQUIRE( app.validateSyncopationConfig() == 0 );
    REQUIRE( app.m_trigger );

    app.m_syncopationTiming = "local_test_schedule";
    app.m_dmChannelName     = "dm01disp07";
    REQUIRE( app.validateSyncopationConfig() < 0 );
    app.m_dmChannelName = "dm01disp07_aux";
    REQUIRE( app.validateSyncopationConfig() == 0 );
    REQUIRE_FALSE( app.m_trigger );

    SECTION( "a timing stream cannot alias the writable source by name" )
    {
        app.m_syncopationTiming = app.m_dmChannelName;
        REQUIRE( app.validateSyncopationConfig() < 0 );
    }
    SECTION( "nonfinite or nonpositive lead is rejected" )
    {
        app.m_writeLeadUs = std::numeric_limits<double>::quiet_NaN();
        REQUIRE( app.validateSyncopationConfig() < 0 );
        app.m_writeLeadUs = 0;
        REQUIRE( app.validateSyncopationConfig() < 0 );
    }
    SECTION( "tiptilt requires managed timing" )
    {
        app.m_opMode = TIPTILT;
        app.m_syncopationTiming.clear();
        REQUIRE( app.validateSyncopationConfig() < 0 );
    }
}

/// Load provider configuration through the registered application parser.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle loads managed tiptilt configuration", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::setupConfig();
    dmSpeckle::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    dmSpeckleHarness app;
    app.setupConfig();
    const bool  loopLock = GENERATE( false, true );
    std::string file     = "/tmp/dmSpeckle_provider_" + std::to_string( getpid() ) + ".conf";
    mx::app::writeConfigFile(
        file,
        { "dm", "dm", "dm", "modulator", "syncopation", "syncopation", "syncopation" },
        { "channelName", "frequency", "trigger", "opMode", "timingStream", "writeLeadUs", "loopLock" },
        { "dm01disp07_aux", "200", "on", "tiptilt", "local_test_schedule", "45", loopLock ? "true" : "false" } );
    app.config.readConfig( file );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    std::remove( file.c_str() );
    REQUIRE( app.m_opMode.load() == TIPTILT );
    REQUIRE( app.m_frequency == 200 );
    REQUIRE( app.m_writeLeadUs == 45 );
    CHECK( app.m_loopLock.load() == loopLock );
    CHECK_FALSE( app.syncopationMainOnly() );
    REQUIRE_FALSE( app.m_trigger );
}

/// Verify that the physical vibration is 200 Hz, independently of the 6 kHz source update cadence.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle tiptilt uses target time rather than frame index", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::fillTipTilt(nullptr, 0, 0, 0, 0, 0);
    #endif
    // clang-format on
    std::array<float, 9> image{};
    dmSpeckleHarness::fillTipTilt( image.data(), 3, 3, 0.1, 200, 0 );
    REQUIRE( image[0] == Approx( -0.1 ) );
    REQUIRE( image[2] == Approx( 0.1 ) );
    REQUIRE( image[4] == Approx( 0 ) );
    auto start = image;

    dmSpeckleHarness::fillTipTilt( image.data(), 3, 3, 0.1, 200, 1250000 );
    REQUIRE( image[0] == Approx( -0.1 ) );
    REQUIRE( image[2] == Approx( -0.1 ) );
    REQUIRE( image[6] == Approx( 0.1 ) );
    dmSpeckleHarness::fillTipTilt( image.data(), 3, 3, 0.1, 200, 2500000 );
    REQUIRE( image[0] == Approx( 0.1 ) );
    REQUIRE( image[2] == Approx( -0.1 ) );

    // A 5 ms period contains 30 source samples at 6 kHz.
    dmSpeckleHarness::fillTipTilt( image.data(), 3, 3, 0.1, 200, 5000000 );
    for( size_t i = 0; i < image.size(); ++i )
    {
        REQUIRE( image[i] == Approx( start[i] ).margin( 1e-7 ) );
    }
    dmSpeckleHarness::fillTipTilt( image.data(), 1, 1, 0.1, 200, 5000000 );
    REQUIRE( image[0] == Approx( 0 ) );
}

/// Use the main cycle for sparkle and target time for arbcube and tiptilt.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle provider selects samples from the main cycle or target time", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    MagAOX::app::dmSpeckle::prepareSyncopationCommand;
    #endif
    // clang-format on
    dmSpeckleHarness app;
    app.m_width = app.m_height = 2;
    app.m_frequency            = 2000;
    app.m_dwell                = 2;
    app.m_shapes.resize( 2, 2, 4 );

    for( int plane = 0; plane < 4; ++plane )
    {
        app.m_shapes.image( plane ).setConstant( float( plane + 1 ) );
    }

    mx::improc::eigenImage<float> command( 2, 2 );

    synco::Request slot{ .cycle = 100, .step = 0, .targetNs = 125000 };
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 1 );
    CHECK( command.maxCoeff() == 1 );
    ++slot.cycle;
    slot.targetNs += 500000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 2 );
    ++slot.cycle;
    slot.targetNs += 1000000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 3 );
    slot.cycle += 2;
    slot.targetNs += 2000000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 1 );
    slot.step = 1;
    CHECK( app.prepareSyncopationCommand( command, slot, false ) < 0 );

    app.m_opMode  = ARBCUBE;
    slot.targetNs = 125000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 1 );
    CHECK( command.maxCoeff() == 1 );
    slot.targetNs = 2125000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 3 );
    CHECK( command.maxCoeff() == 3 );
    slot.targetNs = 4125000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 1 );

    app.m_opMode    = TIPTILT;
    app.m_frequency = 200;
    app.m_amp       = 0.1;
    slot.targetNs   = 0;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    REQUIRE( command( 0, 0 ) == Approx( -0.1 ) );
    REQUIRE( command( 1, 0 ) == Approx( 0.1 ) );
    slot.targetNs = 1250000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    REQUIRE( command( 0, 0 ) == Approx( -0.1 ) );
    REQUIRE( command( 1, 0 ) == Approx( -0.1 ) );

    REQUIRE( app.prepareSyncopationCommand( command, slot, true ) == 0 );
    REQUIRE( command.isZero() );
}

/// Switch managed sparkle to auxiliary-slot playback and back while modulation is active.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle can unlock sparkle for higher rates", "[dmSpeckle]" )
{
    dmSpeckleHarness app;
    app.m_syncopationTiming = "local_test_schedule";
    app.m_modulating        = true;
    app.m_frequency         = 8000;
    app.m_shapes.resize( 2, 2, 4 );

    for( int plane = 0; plane < app.m_shapes.planes(); ++plane )
    {
        app.m_shapes.image( plane ).setConstant( float( plane + 1 ) );
    }

    pcf::IndiProperty request( pcf::IndiProperty::Switch, "test_speckle", "loop_lock" );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    app.m_indiP_loopLock = request;

    mx::improc::eigenImage<float> command( 2, 2 );
    // Sample inside each timed plane interval instead of on a floating-point boundary.
    synco::Request slot{ .cycle = 100, .step = 0, .targetNs = 1000100 };
    CHECK( app.syncopationMainOnly() );
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 1 );

    request["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_loopLock( request ) == 0 );
    CHECK_FALSE( app.syncopationMainOnly() );

    for( int step = 1; step < 4; ++step )
    {
        slot.step = step;
        slot.targetNs += 125000;
        REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
        CHECK( command.minCoeff() == step + 1 );
    }

    request["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_loopLock( request ) == 0 );
    CHECK( app.syncopationMainOnly() );
    CHECK( app.prepareSyncopationCommand( command, slot, false ) < 0 );
    slot.step = 0;
    ++slot.cycle;
    slot.targetNs += 125000;
    REQUIRE( app.prepareSyncopationCommand( command, slot, false ) == 0 );
    CHECK( command.minCoeff() == 2 );
    CHECK( app.m_modulating );

    // clang-format off
#ifdef DMSPECKLE_TEST_DOXYGEN_REF
    MagAOX::app::dmSpeckle::newCallBack_m_indiP_loopLock;
    MagAOX::app::dmSpeckle::syncopationMainOnly;
    MagAOX::app::dmSpeckle::prepareSyncopationCommand;
#endif
    // clang-format on
}

/// Queue explicit zero through the helper transport without overwriting its outstanding source.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle managed zero leaves immutable source pixels alone", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::newCallBack_m_indiP_zero(pcf::IndiProperty());
    dmSpeckle::newCallBack_m_indiP_modulating(pcf::IndiProperty());
    #endif
    // clang-format on
    dmSpeckleHarness app;
    IMAGE_METADATA   source{};
    float            pixel  = 0.3;
    app.m_syncopationTiming = "local_test_schedule";
    app.m_dmChannelName     = "dm01disp07_aux";
    app.m_width = app.m_height  = 1;
    app.m_typeSize              = sizeof( float );
    app.m_imageStream.md        = &source;
    app.m_imageStream.array.raw = &pixel;
    app.m_opened                = true;
    source.cnt0                 = 12;
    app.m_indiP_zero.setName( "zero" );
    pcf::IndiProperty request;
    request.setName( "zero" );
    request.add( pcf::IndiElement( "request" ) );
    request["request"] = pcf::IndiElement::On;

    REQUIRE( app.newCallBack_m_indiP_zero( request ) == 0 );
    REQUIRE( app.m_syncopationZeroRequested );
    REQUIRE( pixel == Approx( 0.3 ) );
    REQUIRE( source.cnt0 == 12 );
    REQUIRE( source.write == 0 );

    app.m_syncopationZeroRequested = false;
    app.m_modulating               = true;
    REQUIRE( app.newCallBack_m_indiP_zero( request ) == 0 );
    REQUIRE_FALSE( app.m_syncopationZeroRequested );
    REQUIRE( pixel == Approx( 0.3 ) );

    app.m_modulating               = false;
    app.m_syncopationZeroRequested = true;
    app.m_indiP_modulating.setName( "modulating" );
    pcf::IndiProperty modulating;
    modulating.setName( "modulating" );
    modulating.add( pcf::IndiElement( "toggle" ) );
    modulating["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_modulating( modulating ) == 0 );
    REQUIRE( app.m_modulating );
    REQUIRE_FALSE( app.m_syncopationZeroRequested );
}

/// Preserve direct legacy zero while rejecting a managed producer's independent trigger request.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle preserves legacy zero and isolates managed triggers", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::newCallBack_m_indiP_zero(pcf::IndiProperty());
    dmSpeckle::newCallBack_m_indiP_trigger(pcf::IndiProperty());
    #endif
    // clang-format on
    dmSpeckleHarness app;
    IMAGE_METADATA   source{};
    float            pixel = 0.3;
    app.m_width = app.m_height  = 1;
    app.m_typeSize              = sizeof( float );
    app.m_imageStream.md        = &source;
    app.m_imageStream.array.raw = &pixel;
    app.m_opened                = true;
    source.shared               = 1;
    source.cnt0                 = 12;
    app.m_indiP_zero.setName( "zero" );
    pcf::IndiProperty zero;
    zero.setName( "zero" );
    zero.add( pcf::IndiElement( "request" ) );
    zero["request"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_zero( zero ) == 0 );
    REQUIRE( pixel == 0 );
    REQUIRE( source.cnt0 == 13 );
    REQUIRE( source.write == 0 );
    REQUIRE_FALSE( app.m_syncopationZeroRequested );

    app.m_syncopationTiming = "local_test_schedule";
    app.m_dmChannelName     = "dm01disp07_aux";
    REQUIRE( app.validateSyncopationConfig() == 0 );
    app.m_indiP_trigger.setName( "trigger" );
    pcf::IndiProperty request;
    request.setName( "trigger" );
    request.add( pcf::IndiElement( "toggle" ) );
    request["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_trigger( request ) < 0 );
    REQUIRE_FALSE( app.m_trigger );
}

/// Verify that mode INDI element names match the modulator.opMode configuration values.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle mode names match configuration values", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::opModeName(0);
    #endif
    // clang-format on
    REQUIRE( dmSpeckleHarness::opModeName( SPARKLE ) == "sparkle" );
    REQUIRE( dmSpeckleHarness::opModeName( ARBCUBE ) == "arbcube" );
    REQUIRE( dmSpeckleHarness::opModeName( TIPTILT ) == "tiptilt" );
}

} // namespace dmSpeckleTest
} // namespace libXWCTest
