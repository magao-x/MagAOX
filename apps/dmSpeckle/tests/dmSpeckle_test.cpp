/** \file dmSpeckle_test.cpp
 * \brief Deterministic tests for legacy and syncopation-managed dmSpeckle operation.
 * \ingroup dmSpeckle_files
 */

#include "../../../tests/testXWC.hpp"

#include "../dmSpeckle.hpp"

#include <cstdio>
#include <limits>

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
    using dmSpeckle::handleFileReload;
    using dmSpeckle::m_amp;
    using dmSpeckle::m_arbcube;
    using dmSpeckle::m_dmChannelName;
    using dmSpeckle::m_dwell;
    using dmSpeckle::m_fileName;
    using dmSpeckle::m_frequency;
    using dmSpeckle::m_height;
    using dmSpeckle::m_imageStream;
    using dmSpeckle::m_indiP_amp;
    using dmSpeckle::m_indiP_file;
    using dmSpeckle::m_indiP_frequency;
    using dmSpeckle::m_indiP_loadFile;
    using dmSpeckle::m_indiP_modulating;
    using dmSpeckle::m_indiP_trigger;
    using dmSpeckle::m_indiP_zero;
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
    std::string file = "/tmp/dmSpeckle_provider_" + std::to_string( getpid() ) + ".conf";
    mx::app::writeConfigFile( file,
                              { "dm", "dm", "dm", "modulator", "syncopation", "syncopation" },
                              { "channelName", "frequency", "trigger", "opMode", "timingStream", "writeLeadUs" },
                              { "dm01disp07_aux", "200", "on", "tiptilt", "local_test_schedule", "45" } );
    app.config.readConfig( file );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    std::remove( file.c_str() );
    REQUIRE( app.m_opMode == TIPTILT );
    REQUIRE( app.m_frequency == 200 );
    REQUIRE( app.m_writeLeadUs == 45 );
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

/// Build target-labelled cube samples independently of request cadence, including dwell.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle provider builds the sample for its advertised target", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::prepareSyncopationCommand(mx::improc::eigenImage<float>(), 0, false);
    #endif
    // clang-format on
    dmSpeckleHarness app;
    app.m_width = app.m_height = 2;
    app.m_frequency            = 2000;
    app.m_dwell                = 2;
    app.m_shapes.resize( 2, 2, 4 );
    app.m_arbcube.resize( 2, 2, 4 );
    app.m_amp = 1;

    for( int plane = 0; plane < 4; ++plane )
    {
        app.m_shapes.image( plane ).setConstant( float( plane + 1 ) );
        app.m_arbcube.image( plane ) = app.m_shapes.image( plane );
    }

    mx::improc::eigenImage<float> command( 2, 2 );

    for( int mode : { SPARKLE, ARBCUBE } )
    {
        app.m_opMode = mode;
        REQUIRE( app.prepareSyncopationCommand( command, 125000, false ) == 0 );
        REQUIRE( command.minCoeff() == 1 );
        REQUIRE( command.maxCoeff() == 1 );
        REQUIRE( app.prepareSyncopationCommand( command, 2125000, false ) == 0 );
        REQUIRE( command.minCoeff() == 3 );
        REQUIRE( command.maxCoeff() == 3 );
        REQUIRE( app.prepareSyncopationCommand( command, 4125000, false ) == 0 );
        REQUIRE( command.minCoeff() == 1 );
    }

    app.m_opMode    = TIPTILT;
    app.m_frequency = 200;
    app.m_amp       = 0.1;
    REQUIRE( app.prepareSyncopationCommand( command, 0, false ) == 0 );
    REQUIRE( command( 0, 0 ) == Approx( -0.1 ) );
    REQUIRE( command( 1, 0 ) == Approx( 0.1 ) );
    REQUIRE( app.prepareSyncopationCommand( command, 1250000, false ) == 0 );
    REQUIRE( command( 0, 0 ) == Approx( -0.1 ) );
    REQUIRE( command( 1, 0 ) == Approx( -0.1 ) );

    REQUIRE( app.prepareSyncopationCommand( command, 1250000, true ) == 0 );
    REQUIRE( command.isZero() );
}

/// Load once, then apply new gain and frequency commands entirely from the cached FITS frames.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle reuses a loaded arbitrary cube across start and parameter changes", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::handleFileReload();
    dmSpeckle::loadArbcubeFile(mx::improc::eigenCube<float>(), "");
    dmSpeckle::newCallBack_m_indiP_loadFile(pcf::IndiProperty());
    dmSpeckle::newCallBack_m_indiP_modulating(pcf::IndiProperty());
    dmSpeckle::newCallBack_m_indiP_amp(pcf::IndiProperty());
    dmSpeckle::newCallBack_m_indiP_frequency(pcf::IndiProperty());
    dmSpeckle::prepareSyncopationCommand(mx::improc::eigenImage<float>(), 0, false);
    #endif
    // clang-format on
    dmSpeckleHarness app;
    app.m_width = app.m_height = 2;
    app.m_indiP_file.setName( "file" );
    app.m_indiP_loadFile.setName( "load_file" );
    app.m_indiP_modulating.setName( "modulating" );
    app.m_indiP_amp.setName( "amp" );
    app.m_indiP_frequency.setName( "frequency" );

    mx::improc::eigenCube<float> cube( 2, 2, 40000 );
    cube.setZero();
    cube.image( 1 ).setConstant( 3 );
    cube.image( 2 ).setConstant( 5 );

    char path[]     = "/tmp/dmSpeckle-arbcube-XXXXXX";
    int  descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    close( descriptor );

    mx::fits::fitsFile<float> fits;
    const auto                writeResult = fits.write( path, cube );

    if( writeResult != mx::error_t::noerror )
    {
        std::remove( path );
    }

    REQUIRE( writeResult == mx::error_t::noerror );
    app.m_fileName = path;

    pcf::IndiProperty load;
    load.setName( "load_file" );
    load.add( pcf::IndiElement( "request" ) );
    load["request"]        = pcf::IndiElement::On;
    const int reloadResult = app.newCallBack_m_indiP_loadFile( load );
    app.handleFileReload();
    REQUIRE( std::remove( path ) == 0 );
    REQUIRE( reloadResult == 0 );
    REQUIRE( app.m_indiP_file.getState() == INDI_OK );
    REQUIRE( app.m_arbcube.planes() == 40000 );

    app.m_opMode = ARBCUBE;
    pcf::IndiProperty modulating;
    modulating.setName( "modulating" );
    modulating.add( pcf::IndiElement( "toggle" ) );
    modulating["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_modulating( modulating ) == 0 );
    app.handleFileReload();
    REQUIRE( app.m_modulating );

    pcf::IndiProperty amp;
    amp.setName( "amp" );
    amp.add( pcf::IndiElement( "target" ) );
    amp["target"] = 0.0f;
    REQUIRE( app.newCallBack_m_indiP_amp( amp ) == 0 );

    mx::improc::eigenImage<float> command( 2, 2 );
    REQUIRE( app.prepareSyncopationCommand( command, 1250000, false ) == 0 );
    REQUIRE( command.isZero() );

    amp["target"] = 0.5f;
    REQUIRE( app.newCallBack_m_indiP_amp( amp ) == 0 );
    app.handleFileReload();
    REQUIRE( app.prepareSyncopationCommand( command, 1250000, false ) == 0 );
    REQUIRE( command.minCoeff() == 2.5f );

    pcf::IndiProperty frequency;
    frequency.setName( "frequency" );
    frequency.add( pcf::IndiElement( "target" ) );
    frequency["target"] = 1000.0f;
    REQUIRE( app.newCallBack_m_indiP_frequency( frequency ) == 0 );
    app.handleFileReload();
    REQUIRE( app.prepareSyncopationCommand( command, 1250000, false ) == 0 );
    REQUIRE( command.maxCoeff() == 1.5f );
    REQUIRE( app.m_arbcube.image( 1 ).maxCoeff() == 3 );

    REQUIRE( app.newCallBack_m_indiP_loadFile( load ) == 0 );
    REQUIRE( app.m_indiP_file.getState() == INDI_BUSY );
    app.handleFileReload();
    REQUIRE( app.m_indiP_file.getState() == INDI_ALERT );
    REQUIRE_FALSE( app.m_modulating );
    REQUIRE( app.m_arbcube.planes() == 0 );
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
