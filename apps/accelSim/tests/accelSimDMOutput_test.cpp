/** \file accelSimDMOutput_test.cpp
 * \brief Actual accelSim operator workflow through the unchanged producer and private API doubles.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include <ImageStreamIO/ImageStreamIO.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

#include "accelSimTestSupport.hpp"

#include "../accelSim.hpp"
#undef MagAOXApp

/// \cond ACCELSIM_OUTPUT_FIXTURE
namespace
{
namespace synco = MagAOX::app::synco;

/// Private image storage; backing files exist only to exercise the real producer identity and flock contract.
struct dmImage
{
    IMAGE_METADATA       m_metadata{}; ///< Metadata owned by this test, never ImageStreamIO runtime storage.
    std::vector<int64_t> m_words;      ///< Schedule or exact commit/receipt words.
    std::vector<float>   m_pixels;     ///< Auxiliary command payload.
};

std::map<std::string, dmImage> g_images; ///< Process-private image registry.
std::filesystem::path g_directory; ///< Fresh ordinary-file namespace; never the instrument shared-memory directory.
int64_t               g_tick{ 1000000000 }; ///< Virtual time for the unchanged producer.
bool                  g_autoAck{ false };   ///< Copy receipts automatically only in shutdown tests.

/// Allocate a named private image without invoking a native image library.
dmImage &makeImage( const std::string &name /**< [in] fixture name */,
                    size_t             count /**< [in] element count */,
                    bool               pixels = false /**< [in] float image rather than protocol record */ )
{
    auto &image = g_images[name];
    std::ofstream( g_directory / name ).close();
    struct stat info{};
    REQUIRE( stat( ( g_directory / name ).c_str(), &info ) == 0 );
    image.m_metadata = {};
    auto &md         = image.m_metadata;
    md.inode         = info.st_ino;
    md.naxis         = pixels ? 2 : 1;
    md.size[0]       = pixels ? 50 : count;
    md.size[1]       = pixels ? 50 : 1;
    md.nelement      = count;
    md.imdatamemsize = count * ( pixels ? sizeof( float ) : sizeof( int64_t ) );
    md.datatype      = pixels ? _DATATYPE_FLOAT : _DATATYPE_INT64;
    md.location      = -1;
    md.sem           = 1;
    std::snprintf( md.name, sizeof( md.name ), "%s", name.c_str() );
    if( pixels )
        image.m_pixels.assign( count, 0 );
    else
        image.m_words.assign( count, 0 );
    return image;
}

/// Advertise the next four standard slots using the production record format.
void schedule()
{
    auto         &words   = g_images.at( "private_timing" ).m_words;
    const int64_t origin  = g_tick - g_tick % 500000;
    words[synco::version] = synco::productionVersion;
    words[synco::epoch] = words[synco::generation] = 1;
    words[synco::state]                            = synco::armed;
    words[synco::steps]                            = 4;
    words[synco::period]                           = 500000;
    words[synco::reference]                        = origin;
    words[synco::cycle]                            = origin / 500000;
    words[synco::expires]                          = origin + 1500000;
    for( int step = 0; step < 4; ++step )
    {
        words[synco::targets + step] = origin + 125000 * ( step + 1 );
        words[synco::cutoffs + step] = words[synco::targets + step] - 10000;
    }
}

/// Echo exactly the latest ready commit with a specified transport disposition.
void acknowledge( int64_t disposition = synco::copied /**< [in] copy or discard */ )
{
    auto       &ack   = g_images.at( "dm01disp10_aux_ack" );
    const auto &ready = g_images.at( "dm01disp10_aux_ready" );
    std::copy( ready.m_words.begin(), ready.m_words.end(), ack.m_words.begin() );
    ack.m_words[synco::ackStatus] = disposition;
    ++ack.m_metadata.cnt0;
}

/// Expose actual application integration points; no framework or acquisition thread is started.
struct dmOutputApp : MagAOX::app::accelSim
{
    using accelSim::config;
    using accelSim::dmThreadExec;
    using accelSim::m_dmActive;
    using accelSim::m_dmChannel;
    using accelSim::m_dmEnabled;
    using accelSim::m_dmInjection;
    using accelSim::m_dmPixels;
    using accelSim::m_dmReady;
    using accelSim::m_dmTransportFault;
    using accelSim::m_dmZeroAcknowledged;
    using accelSim::m_dmZeroPending;
    using accelSim::m_indiP_dmEnable;
    using accelSim::m_indiP_dmScale;
    using accelSim::m_model;
    using accelSim::m_shutdown;
    using accelSim::m_syncopationTiming;
    using accelSim::serviceDM;
    using accelSim::setupProperties;
};

/// Own one isolated registry and synthetic calibrated app for the duration of a test.
struct dmOutputFixture
{
    dmOutputApp m_app; ///< Actual implementation, with only framework lifecycle and stream APIs doubled.

    /// Create ordinary files and calibrated ramps, without native streams or hardware.
    dmOutputFixture();

    /// Release producer ownership before deleting this test's private files.
    ~dmOutputFixture();

    /// Move to another scheduler slot, independently of ADC sample staging.
    void advance();

    /// Set scale and toggle using the real INDI callbacks.
    void enable( double scale /**< [in] requested scale */ );
};

dmOutputFixture::dmOutputFixture()
{
    char path[] = "/tmp/accelSim-producer-XXXXXX";
    REQUIRE( mkdtemp( path ) != nullptr );
    g_directory = path;
    g_images.clear();
    g_tick    = 1000000000;
    g_autoAck = false;
    makeImage( "private_timing", synco::words );
    makeImage( "dm01disp10_aux", 2500, true );
    makeImage( "dm01disp10_aux_ack", synco::ackWords );
    makeImage( "dm10disp10_aux", 2500, true ).m_pixels.assign( 2500, .125f );
    schedule();
    m_app.setupConfig();
    REQUIRE( m_app.loadConfigImpl( m_app.config ) == 0 );
    REQUIRE( m_app.setupProperties() == 0 );
    std::vector<double> x( 2500 ), y( 2500 ), mask( 2500, 1 );
    for( size_t n = 0; n < 2500; ++n )
    {
        x[n] = double( n % 50 ) - 24.5;
        y[n] = double( n / 50 ) - 24.5;
    }
    m_app.m_dmInjection.prepare( x, y, mask, 1e-6 );
    m_app.m_dmReady           = true;
    m_app.m_dmChannel         = "dm01disp10_aux";
    m_app.m_syncopationTiming = "private_timing";
}

dmOutputFixture::~dmOutputFixture()
{
    m_app.appShutdown();
    g_images.clear();
    std::filesystem::remove_all( g_directory );
}

void dmOutputFixture::advance()
{
    g_tick += 125000;
    schedule();
}

void dmOutputFixture::enable( double scale )
{
    auto property      = m_app.m_indiP_dmScale;
    property["target"] = scale;
    REQUIRE( m_app.newCallBack_m_indiP_dmScale( property ) == 0 );
    property           = m_app.m_indiP_dmEnable;
    property["toggle"] = pcf::IndiElement::On;
    REQUIRE( m_app.newCallBack_m_indiP_dmEnable( property ) == 0 );
}
} // namespace

int dmOutputClock( clockid_t clock, timespec *value )
{
    if( clock != CLOCK_MONOTONIC )
        return ::clock_gettime( clock, value );
    value->tv_sec  = g_tick / 1000000000;
    value->tv_nsec = g_tick % 1000000000;
    return 0;
}

int dmOutputSleep( clockid_t, int, const timespec *value, timespec * )
{
    g_tick = std::max( g_tick, int64_t( value->tv_sec ) * 1000000000 + value->tv_nsec );
    return 0;
}

int ImageStreamIO_filename( char *path, size_t length, const char *name )
{
    return std::snprintf( path, length, "%s", ( g_directory / name ).c_str() ) < int( length ) ? 0 : -1;
}

int ImageStreamIO_openIm( IMAGE *image, const char *name )
{
    const auto found = g_images.find( name );
    if( found == g_images.end() )
        return -1;
    image->md        = &found->second.m_metadata;
    image->array.raw = found->second.m_pixels.empty() ? static_cast<void *>( found->second.m_words.data() )
                                                      : static_cast<void *>( found->second.m_pixels.data() );
    return 0;
}

void ImageStreamIO_closeIm( IMAGE *image )
{
    *image = {};
}

int ImageStreamIO_createIm_gpu( IMAGE *image, const char *name, int, uint32_t *size, int, int, int, int, int, int, int )
{
    if( std::string( name ) != "dm01disp10_aux_ready" )
        return -1; // Only the existing base may create its protocol sidecar.
    makeImage( name, size[0] );
    return ImageStreamIO_openIm( image, name );
}

int ImageStreamIO_UpdateIm( IMAGE *image )
{
    ++image->md->cnt0;
    image->md->write = 0;
    if( g_autoAck && std::string( image->md->name ) == "dm01disp10_aux_ready" )
        acknowledge();
    return 0;
}

int ImageStreamIO_getsemwaitindex( IMAGE *, int )
{
    return -1;
}

void ImageStreamIO_semflush( IMAGE *, int )
{
}

#ifdef __APPLE__
int clock_nanosleep( clockid_t, int, const timespec *, timespec * )
{
    return -1; // All producer waits are replaced by dmOutputSleep.
}
#endif
/// \endcond

namespace libXWCTest
{
namespace accelSimTest
{
/** \brief Scale/on/off uses exact acknowledgements and a tagged zero without rate, lock or age qualification.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim direct producer scaled output and acknowledged removal", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &source = g_images.at( "dm01disp10_aux" );
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE_FALSE( g_images.count( "dm01disp10_aux_ready" ) );
    fixture.enable( .25 );
    app.m_model.step();
    const auto ticks = app.m_model.m_ticks;
    REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    fixture.advance(); // The latest physical sample is already older than the deleted 250-us gate.
    fixture.advance();
    fixture.advance();
    REQUIRE( app.serviceDM() );
    REQUIRE( app.m_model.m_ticks == ticks );
    const MagAOX::app::accelSimDetail::pointingDisturbance pointing( app.m_model.m_sample );
    REQUIRE( source.m_pixels[0] ==
             Approx( .25 * pointing.surface( -24.5 * 6.5 / 48, -24.5 * 6.5 / 48 ) ).margin( 1e-8 ) );
    const auto counter = source.m_metadata.cnt0;
    fixture.advance();
    REQUIRE_FALSE( app.serviceDM() ); // The base owns the single buffer until the exact receipt.
    REQUIRE( source.m_metadata.cnt0 == counter );
    acknowledge();
    ++g_images.at( "dm01disp10_aux_ack" ).m_words[synco::cmdSequence];
    REQUIRE_FALSE( app.serviceDM() ); // A mismatched receipt never permits reuse.
    acknowledge();
    auto enable      = app.m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    REQUIRE( app.serviceDM() ); // No subsequent ADC sample is needed for removal.
    REQUIRE( app.m_dmActive );  // Last copied enabled command persists until the zero receipt.
    REQUIRE_FALSE( app.m_dmZeroAcknowledged );
    REQUIRE( std::all_of( source.m_pixels.begin(), source.m_pixels.end(), []( float v ) { return v == 0; } ) );
    acknowledge( synco::discarded );
    fixture.advance();
    REQUIRE( app.serviceDM() ); // A discarded zero is retried, with no latched fault.
    REQUIRE( app.m_dmZeroPending );
    acknowledge();
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE_FALSE( app.m_dmActive );
    REQUIRE( app.m_dmZeroAcknowledged );
    REQUIRE_FALSE( app.m_dmZeroPending );
    REQUIRE( std::all_of( g_images.at( "dm10disp10_aux" ).m_pixels.begin(),
                          g_images.at( "dm10disp10_aux" ).m_pixels.end(),
                          []( float v ) { return v == .125f; } ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::frameGrabberPostPublish;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmEnable;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmScale;
#endif
    // clang-format on
}

/** \brief Eight thousand virtual slots preserve physical scaling and finite checks without advancing the model twice.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim 8000 scaled producer samples and numerical recovery", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &source = g_images.at( "dm01disp10_aux" );
    fixture.enable( .25 );
    app.m_model.setRate( 8000 );
    for( size_t n = 0; n < 8000; ++n )
    {
        app.m_model.step();
        const auto ticks = app.m_model.m_ticks;
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
        fixture.advance();
        REQUIRE( app.serviceDM() );
        const MagAOX::app::accelSimDetail::pointingDisturbance pointing( app.m_model.m_sample );
        for( size_t p : { size_t( 0 ), size_t( 1249 ), size_t( 2499 ) } )
            REQUIRE( source.m_pixels[p] == Approx( .25 * pointing.surface( ( double( p % 50 ) - 24.5 ) * 6.5 / 48,
                                                                           ( double( p / 50 ) - 24.5 ) * 6.5 / 48 ) )
                                               .margin( 1e-8 ) );
        REQUIRE( app.m_model.m_ticks == ticks );
        acknowledge();
    }
    fixture.advance();
    app.m_dmPixels[0]  = NAN; // Verify the unchanged producer's last finite-command check.
    const auto counter = source.m_metadata.cnt0;
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE( source.m_metadata.cnt0 == counter );
    app.m_model.m_sample.m_pointing[0] = NAN;
    REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    fixture.advance();
    REQUIRE( app.serviceDM() );
    REQUIRE( std::all_of( source.m_pixels.begin(), source.m_pixels.end(), []( float v ) { return v == 0; } ) );
    acknowledge();
    app.m_model.step();
    REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    fixture.advance();
    REQUIRE( app.serviceDM() ); // The next finite sample resumes; no off/on or fault reset is required.
    REQUIRE( source.m_pixels[0] != 0 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::frameGrabberPostPublish;
#endif
    // clang-format on
}

/** \brief A valid scheduler generation change releases the old base-owned buffer and permits pending removal.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim removal after scheduler reset", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    fixture.enable( 1 );
    fixture.m_app.m_model.step();
    REQUIRE( fixture.m_app.frameGrabberPostPublish( nullptr ) == 0 );
    REQUIRE( fixture.m_app.serviceDM() );
    auto enable      = fixture.m_app.m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( fixture.m_app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    fixture.advance();
    ++g_images.at( "private_timing" ).m_words[synco::generation];
    REQUIRE( fixture.m_app.serviceDM() );
    REQUIRE( fixture.m_app.m_dmZeroPending );
    acknowledge();
    REQUIRE_FALSE( fixture.m_app.serviceDM() );
    REQUIRE( fixture.m_app.m_dmZeroAcknowledged );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief The unchanged cooperative lock prevents a competing producer from modifying either payload or commit.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim producer ownership and automatic transport retry", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    const auto      lockPath = g_directory / "dm01disp10_aux.producer";
    const int       owner    = open( lockPath.c_str(), O_CREAT | O_RDWR, 0600 );
    REQUIRE( owner >= 0 );
    REQUIRE( flock( owner, LOCK_EX | LOCK_NB ) == 0 );
    fixture.enable( 1 );
    REQUIRE_FALSE( fixture.m_app.serviceDM() );
    REQUIRE( fixture.m_app.m_dmTransportFault );
    REQUIRE( g_images.at( "dm01disp10_aux" ).m_metadata.cnt0 == 0 );
    REQUIRE_FALSE( g_images.count( "dm01disp10_aux_ready" ) );
    close( owner );
    REQUIRE( fixture.m_app.serviceDM() );
    REQUIRE_FALSE( fixture.m_app.m_dmTransportFault );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief Final tagged removal still runs after MagAOXApp requests shutdown, without resetting that request.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim shutdown preserves tagged zero removal", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    fixture.enable( 1 );
    fixture.m_app.m_model.step();
    REQUIRE( fixture.m_app.frameGrabberPostPublish( nullptr ) == 0 );
    REQUIRE( fixture.m_app.serviceDM() );
    acknowledge();
    fixture.advance();
    g_autoAck                = true;
    fixture.m_app.m_shutdown = true;
    fixture.m_app.dmThreadExec();
    REQUIRE( fixture.m_app.m_shutdown );
    REQUIRE( fixture.m_app.m_dmZeroAcknowledged );
    REQUIRE_FALSE( fixture.m_app.m_dmActive );
    REQUIRE( std::all_of( g_images.at( "dm01disp10_aux" ).m_pixels.begin(),
                          g_images.at( "dm01disp10_aux" ).m_pixels.end(),
                          []( float v ) { return v == 0; } ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::dmThreadExec;
    MagAOX::app::accelSim::appShutdown;
#endif
    // clang-format on
}

} // namespace accelSimTest
} // namespace libXWCTest
