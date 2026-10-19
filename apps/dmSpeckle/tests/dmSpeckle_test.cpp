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
    using dmSpeckle::m_frequency;
    using dmSpeckle::m_height;
    using dmSpeckle::m_imageStream;
    using dmSpeckle::m_indiP_trigger;
    using dmSpeckle::m_modulating;
    using dmSpeckle::m_opMode;
    using dmSpeckle::m_syncopationTiming;
    using dmSpeckle::m_trigger;
    using dmSpeckle::m_typeSize;
    using dmSpeckle::m_width;
    using dmSpeckle::m_writeLeadUs;
    using dmSpeckle::publishSyncopation;
    using dmSpeckle::readSyncopationRecord;
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

/// Reject in-progress or incompatible timing records and leave a source untouched after its cutoff.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle obeys record publication and expired writer cutoffs", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::readSyncopationRecord(IMAGE(), synco::Record());
    dmSpeckle::publishSyncopation(nullptr, 0);
    #endif
    // clang-format on
    IMAGE_METADATA metadata{};
    IMAGE          timing{};
    synco::Record  original{}, read{};
    original[synco::version] = synco::layoutVersion;
    timing.md                = &metadata;
    timing.array.raw         = original.data();
    metadata.datatype        = _DATATYPE_INT64;
    metadata.nelement        = synco::words;
    metadata.cnt0            = 4;
    REQUIRE( dmSpeckleHarness::readSyncopationRecord( timing, read ) );
    REQUIRE( read == original );
    metadata.write = 1;
    REQUIRE_FALSE( dmSpeckleHarness::readSyncopationRecord( timing, read ) );
    metadata.write    = 0;
    metadata.nelement = 1;
    REQUIRE_FALSE( dmSpeckleHarness::readSyncopationRecord( timing, read ) );

    dmSpeckleHarness app;
    IMAGE_METADATA   source{};
    float            pixel      = 0.3;
    app.m_imageStream.md        = &source;
    app.m_imageStream.array.raw = &pixel;
    source.cnt0                 = 12;
    REQUIRE_FALSE( app.publishSyncopation( nullptr, 0 ) );
    REQUIRE( pixel == Approx( 0.3 ) );
    REQUIRE( source.cnt0 == 12 );
    REQUIRE( source.write == 0 );
}

/// Verify normal auxiliary publication changes only the attached source and never enables legacy triggering.
/** \ingroup dmSpeckle_unit_test */
TEST_CASE( "dmSpeckle managed publication updates only the source", "[dmSpeckle]" )
{
    // clang-format off
    #ifdef DMSPECKLE_TEST_DOXYGEN_REF
    dmSpeckle::publishSyncopation(nullptr, 0);
    dmSpeckle::newCallBack_m_indiP_trigger(pcf::IndiProperty());
    #endif
    // clang-format on
    dmSpeckleHarness app;
    IMAGE_METADATA   source{};
    float            sourcePixel = 0.3, command = -0.2;
    app.m_width = app.m_height  = 1;
    app.m_typeSize              = sizeof( float );
    app.m_imageStream.md        = &source;
    app.m_imageStream.array.raw = &sourcePixel;
    source.shared               = 1;
    source.cnt0                 = 12;
#ifdef IMAGESTRUCT_WRITEHISTORY
    std::array<FRAMEWRITEMD, IMAGESTRUCT_FRAMEWRITEMDSIZE> history{};
    app.m_imageStream.writehist = history.data();
#endif
    app.m_modulating = true;
    REQUIRE( app.publishSyncopation( &command, synco::monotonicNs() + 1000000000 ) );
    REQUIRE( sourcePixel == Approx( command ) );
    REQUIRE( source.cnt0 == 13 );
    REQUIRE( source.write == 0 );
    app.m_modulating = false;
    REQUIRE_FALSE( app.publishSyncopation( &command, synco::monotonicNs() + 1000000000 ) );
    REQUIRE( source.cnt0 == 13 );

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

} // namespace dmSpeckleTest
} // namespace libXWCTest
