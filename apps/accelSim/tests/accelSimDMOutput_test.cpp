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

int64_t g_wakeDelayNs{ 0 };   ///< Injected latency of a sleeping producer after its requested wake time.
bool    g_ackOnPoll{ false }; ///< Deliver a receipt as the producer enters its next poll.

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
    auto         &words  = g_images.at( "private_timing" ).m_words;
    const int64_t origin = g_tick - g_tick % 500000;
    ++g_images.at( "private_timing" ).m_metadata.cnt0;
    words[synco::version] = synco::productionVersion;
    words[synco::epoch] = words[synco::generation] = 1;
    words[synco::state]                            = synco::armed;
    words[synco::steps]                            = 4;
    words[synco::period]                           = 500000;
    words[synco::reference]                        = origin;
    words[synco::cycle]                            = origin / 500000;
    words[synco::expires]                          = origin + 1000000;
    for( int step = 0; step < 4; ++step )
    {
        words[synco::targets + step] = origin + 125000 * ( step ? step : 4 );
        words[synco::cutoffs + step] = words[synco::targets + step] - 70000;
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
    using accelSim::m_acquiredSample;
    using accelSim::m_dmActive;
    using accelSim::m_dmChannel;
    using accelSim::m_dmCommandValid;
    using accelSim::m_dmEnabled;
    using accelSim::m_dmInjection;
    using accelSim::m_dmReady;
    using accelSim::m_dmScale;
    using accelSim::m_dmTransportFault;
    using accelSim::m_dmZeroAcknowledged;
    using accelSim::m_dmZeroPending;
    using accelSim::m_indiP_dmEnable;
    using accelSim::m_indiP_dmScale;
    using accelSim::m_indiP_dmStatus;
    using accelSim::m_model;
    using accelSim::m_shutdown;
    using accelSim::m_syncopationTiming;
    using accelSim::m_timeline;
    using accelSim::measure;
    using accelSim::serviceDM;
    using accelSim::setupProperties;
    using accelSim::updateDMINDI;
};

/// Own one isolated registry and synthetic calibrated app for the duration of a test.
struct dmOutputFixture
{
    dmOutputApp m_app; ///< Actual implementation, with only framework lifecycle and stream APIs doubled.

    int64_t m_epoch{ 0 }; ///< Virtual monotonic time of timeline node zero and accelerometer tick zero.

    uint64_t m_acquisitions{ 0 }; ///< Accelerometer measurements taken through the app's acquisition seam.

    /// Create ordinary files, calibrated ramps and an 8-kHz timeline, without native streams or hardware.
    explicit dmOutputFixture( int writeLeadUs = 50 /**< [in] Configured producer lead before cutoff in us. */ );

    /// Release producer ownership before deleting this test's private files.
    ~dmOutputFixture();

    /// Move to the next 125-us boundary, refresh the 2-kHz schedule there, and acquire that tick.
    void advance();

    /// Measure acc000 at the current virtual monotonic time, as the acquisition thread does.
    void acquire();

    /// Set scale and toggle using the real INDI callbacks.
    void enable( double scale /**< [in] requested scale */ );
};

/// Return the scaled calibrated surface of the shared realization at one monotonic time and pixel.
double expected( dmOutputApp &app /**< [in] app owning the timeline */,
                 int64_t      time /**< [in] same-host monotonic time */,
                 double       scale /**< [in] DM scale */,
                 size_t       pixel /**< [in] native pixel index */ )
{
    MagAOX::app::accelSimDetail::sample physical;
    REQUIRE( app.m_timeline.physical( time, physical ) );
    const MagAOX::app::accelSimDetail::pointingDisturbance pointing( physical );
    return scale *
           pointing.surface( ( double( pixel % 50 ) - 24.5 ) * 6.5 / 48, ( double( pixel / 50 ) - 24.5 ) * 6.5 / 48 );
}

dmOutputFixture::dmOutputFixture( int writeLeadUs )
{
    char path[] = "/tmp/accelSim-producer-XXXXXX";
    REQUIRE( mkdtemp( path ) != nullptr );
    g_directory = path;
    g_images.clear();
    g_tick        = 1000000000;
    g_autoAck     = false;
    g_wakeDelayNs = 0;
    g_ackOnPoll   = false;
    makeImage( "private_timing", synco::words );
    makeImage( "dm01disp10_aux", 2500, true );
    makeImage( "dm01disp10_aux_ack", synco::ackWords );
    makeImage( "dm10disp10_aux", 2500, true ).m_pixels.assign( 2500, .125f );
    schedule();
    m_app.setupConfig();
    m_app.config.m_values["syncopation.writeLeadUs"] = std::to_string( writeLeadUs );
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
    m_app.m_model.setRate( 8000 );
    m_epoch = g_tick - 125000; // The scheduler and accelerometer share the virtual monotonic clock.
    m_app.m_timeline.reset( m_epoch, m_app.m_model );
    acquire();
}

dmOutputFixture::~dmOutputFixture()
{
    m_app.appShutdown();
    g_images.clear();
    std::filesystem::remove_all( g_directory );
}

void dmOutputFixture::advance()
{
    g_tick += 125000 - g_tick % 125000;
    if( g_tick % 500000 == 0 )
        schedule();
    acquire();
}

void dmOutputFixture::acquire()
{
    m_app.measure( g_tick, uint64_t( ( g_tick - m_epoch ) / 125000 ) );
    ++m_acquisitions;
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
    if( g_ackOnPoll )
    {
        g_ackOnPoll = false;
        acknowledge();
    }
    value->tv_sec  = g_tick / 1000000000;
    value->tv_nsec = g_tick % 1000000000;
    return 0;
}

int dmOutputSleep( clockid_t, int, const timespec *value, timespec * )
{
    g_tick = std::max( g_tick, int64_t( value->tv_sec ) * 1000000000 + value->tv_nsec + g_wakeDelayNs );
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
/** \brief Scale/on/off uses target-time surfaces, exact acknowledgements and a tagged zero without output gates.
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
    fixture.advance();
    fixture.advance();
    fixture.advance();
    const auto ticks = app.m_model.m_ticks;
    REQUIRE( app.serviceDM() );
    REQUIRE( app.m_model.m_ticks == ticks ); // DM evaluation reads the realization and draws nothing.
    const auto &ready = g_images.at( "dm01disp10_aux_ready" ).m_words;
    REQUIRE( source.m_pixels[0] == Approx( expected( app, ready[synco::cmdTarget], .25, 0 ) ).margin( 1e-8 ) );
    REQUIRE( std::abs( expected( app, ready[synco::cmdTarget], .25, 0 ) -
                       expected( app, ready[synco::cmdCutoff], .25, 0 ) ) > 1e-7 ); // Time choice is observable.
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
    MagAOX::app::accelSim::measure;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmEnable;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmScale;
#endif
    // clang-format on
}

/** \brief Eight thousand slots carry the target-time surface; rejection and numerical faults recover without latches.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim 8000 scaled producer samples and numerical recovery", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &source = g_images.at( "dm01disp10_aux" );
    fixture.enable( .25 );
    REQUIRE( app.serviceDM() ); // Attachment creates the ready sidecar.
    const auto &ready = g_images.at( "dm01disp10_aux_ready" ).m_words;
    acknowledge();
    for( size_t n = 0; n < 8000; ++n )
    {
        fixture.advance();
        const auto ticks = app.m_model.m_ticks;
        REQUIRE( app.serviceDM() );
        for( size_t p : { size_t( 0 ), size_t( 1249 ), size_t( 2499 ) } )
            REQUIRE( source.m_pixels[p] == Approx( expected( app, ready[synco::cmdTarget], .25, p ) ).margin( 1e-8 ) );
        REQUIRE( app.m_model.m_ticks == ticks );
        acknowledge();
    }
    fixture.advance();
    source.m_metadata.write = 1; // A busy source fails the unchanged producer's final check before cutoff.
    auto counter            = source.m_metadata.cnt0;
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE( source.m_metadata.cnt0 == counter );
    source.m_metadata.write = 0;
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["rejected"].get<int>() == 1 );
    REQUIRE( app.m_indiP_dmStatus["late_returns"].get<int>() == 0 );
    fixture.advance();
    app.m_dmScale = NAN; // Bypass the INDI range check to exercise synthesis rejection.
    REQUIRE( app.serviceDM() );
    REQUIRE( std::all_of( source.m_pixels.begin(), source.m_pixels.end(), []( float v ) { return v == 0; } ) );
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["fault"].get<int>() == 1 );
    acknowledge();
    fixture.advance();
    app.m_dmScale = .25;
    REQUIRE( app.serviceDM() );      // The next valid synthesis resumes; no off/on or fault reset is required.
    REQUIRE_FALSE( app.m_dmActive ); // The copied fault command was a removal, not an active contribution.
    REQUIRE( app.m_dmCommandValid );
    REQUIRE( source.m_pixels[0] == Approx( expected( app, ready[synco::cmdTarget], .25, 0 ) ).margin( 1e-8 ) );
    acknowledge();
    fixture.advance();
    REQUIRE( app.serviceDM() );
    REQUIRE( app.m_dmActive );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::updateDMINDI;
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
    fixture.m_app.updateDMINDI();
    REQUIRE( fixture.m_app.m_indiP_dmStatus["attached"].get<int>() == 0 );
    REQUIRE( g_images.at( "dm01disp10_aux" ).m_metadata.cnt0 == 0 );
    REQUIRE_FALSE( g_images.count( "dm01disp10_aux_ready" ) );
    close( owner );
    REQUIRE( fixture.m_app.serviceDM() );
    REQUIRE_FALSE( fixture.m_app.m_dmTransportFault );
    fixture.m_app.updateDMINDI();
    REQUIRE( fixture.m_app.m_indiP_dmStatus["attached"].get<int>() == 1 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::updateDMINDI;
#endif
    // clang-format on
}

/** \brief Invalid records leave an attached producer waiting; late wakes and timely slots are counted separately.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim distinguishes timing activity from publishable slots", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &timing = g_images.at( "private_timing" );
    auto           &source = g_images.at( "dm01disp10_aux" );
    fixture.enable( .1 );

    for( int frame = 0; frame < 2; ++frame )
    {
        g_tick += 500000;
        fixture.acquire();
        schedule();
        timing.m_words[synco::state] = synco::invalid;
        std::fill( timing.m_words.begin() + synco::reference, timing.m_words.end(), 0 );
        REQUIRE_FALSE( app.serviceDM() );
    }
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["attached"].get<int>() == 1 );
    REQUIRE( app.m_indiP_dmStatus["poll_waits"].get<int>() == 2 );
    REQUIRE( app.m_indiP_dmStatus["requests"].get<int>() == 0 );
    REQUIRE( timing.m_metadata.cnt0 == 3 );
    REQUIRE_FALSE( app.m_dmTransportFault );
    auto &ready = g_images.at( "dm01disp10_aux_ready" );
    REQUIRE( ready.m_metadata.cnt0 == 0 );

    g_tick += 500000;
    fixture.acquire();
    schedule();
    g_wakeDelayNs = 51000; // A wake 51 us late exhausts writeLeadUs=50, even for a valid request.
    for( int slot = 0; slot < 4; ++slot )
    {
        REQUIRE_FALSE( app.serviceDM() );
        fixture.advance();
    }
    REQUIRE( source.m_metadata.cnt0 == 0 );
    REQUIRE( ready.m_metadata.cnt0 == 0 );
    REQUIRE_FALSE( app.m_dmActive );
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["requests"].get<int>() == 4 );
    REQUIRE( app.m_indiP_dmStatus["late_returns"].get<int>() == 4 );
    REQUIRE( app.m_indiP_dmStatus["rejected"].get<int>() == 0 );
    REQUIRE( app.m_indiP_dmStatus["timeline_misses"].get<int>() == 0 );
    REQUIRE( app.m_indiP_dmStatus["published"].get<int>() == 0 );

    g_wakeDelayNs    = 3000;
    const auto cycle = timing.m_words[synco::cycle];
    for( const int step : { 1, 2, 3, 0 } )
    {
        fixture.acquire();
        REQUIRE( app.serviceDM() );
        REQUIRE( ready.m_words[synco::cmdStep] == step );
        REQUIRE( ready.m_words[synco::cmdCycle] == cycle + ( step == 0 ) );
        REQUIRE( ready.m_words[synco::cmdTarget] - ready.m_words[synco::cmdCutoff] == 70000 );
        REQUIRE( ready.m_words[synco::cmdReadyNs] < ready.m_words[synco::cmdCutoff] );
        REQUIRE( source.m_pixels[0] != 0 );
        g_tick = ready.m_words[synco::cmdCutoff];
        acknowledge();
        fixture.advance();
    }
    REQUIRE( app.m_dmActive );
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["published"].get<int>() == 4 );
    REQUIRE( app.m_indiP_dmStatus["copied"].get<int>() == 3 ); // Fourth receipt awaits the next poll.
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::updateDMINDI;
#endif
    // clang-format on
}

/** \brief The deployed 50-us write lead commits all four 8-kHz slots of a 2-kHz grid, each at its own target time.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim commits all four scheduler steps with the deployed write lead", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture( 50 );
    fixture.enable( .1 );
    auto &app     = fixture.m_app;
    auto &source  = g_images.at( "dm01disp10_aux" );
    g_wakeDelayNs = 3000;
    for( int frame = 0; frame < 2; ++frame )
    {
        schedule();
        const auto origin = g_tick;
        for( int slot = 1; slot <= 4; ++slot )
        {
            fixture.acquire();
            REQUIRE( app.serviceDM() );
            const auto &ready  = g_images.at( "dm01disp10_aux_ready" ).m_words;
            const auto  target = ready[synco::cmdTarget];
            REQUIRE( ready[synco::cmdStep] == slot % 4 );
            REQUIRE( target == origin + slot * 125000 );
            REQUIRE( ready[synco::cmdReadyNs] < ready[synco::cmdCutoff] );
            REQUIRE( source.m_pixels[0] == Approx( expected( app, target, .1, 0 ) ).margin( 1e-8 ) );
            // One subframe of timing error would be detected at this tolerance.
            REQUIRE( std::abs( expected( app, target, .1, 0 ) - expected( app, target - 125000, .1, 0 ) ) > 1e-7 );
            // The actual scheduler snapshots at the cutoff; its exact receipt is visible on the next poll.
            g_tick = ready[synco::cmdCutoff] + 3000;
            acknowledge();
        }
        g_tick = origin + 500000;
        fixture.acquire();
    }
    REQUIRE( source.m_metadata.cnt0 == 8 );
    REQUIRE( app.m_dmActive );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief Acquisition continues through disabled output and a scheduler outage; output and removal then recover.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim resumes after a primary timeout", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &words  = g_images.at( "private_timing" ).m_words;
    auto           &source = g_images.at( "dm01disp10_aux" );
    for( int tick = 0; tick < 8000; ++tick )
    {
        fixture.advance();
        REQUIRE_FALSE( app.serviceDM() );
    }
    fixture.enable( .1 );
    fixture.advance();
    REQUIRE( app.serviceDM() );
    acknowledge();
    words[synco::state] = synco::invalid; // Production main timeout invalidates the record.
    const auto counter  = source.m_metadata.cnt0;
    for( int tick = 0; tick < 80; ++tick )
    {
        g_tick += 125000;
        fixture.acquire();
        REQUIRE_FALSE( app.serviceDM() );
    }
    REQUIRE( source.m_metadata.cnt0 == counter );
    REQUIRE( fixture.m_acquisitions == 1 + 8000 + 1 + 80 );
    schedule();
    ++words[synco::generation];
    REQUIRE( app.serviceDM() );
    const auto target = g_images.at( "dm01disp10_aux_ready" ).m_words[synco::cmdTarget];
    REQUIRE( source.m_pixels[0] == Approx( expected( app, target, .1, 0 ) ).margin( 1e-8 ) );
    acknowledge();
    auto enable      = app.m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    fixture.advance();
    REQUIRE( app.serviceDM() );
    acknowledge();
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE( app.m_dmZeroAcknowledged );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::measure;
#endif
    // clang-format on
}

/** \brief A request whose target lies beyond the generated horizon is skipped and counted, never approximated.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim skips requests outside the simulation timeline", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app    = fixture.m_app;
    auto           &source = g_images.at( "dm01disp10_aux" );
    fixture.enable( .1 );
    g_tick += MagAOX::app::accelSimDetail::timeline::c_lookahead; // Acquisition stalled for the whole horizon.
    schedule();
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE( source.m_metadata.cnt0 == 0 );
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["requests"].get<int>() == 1 );
    REQUIRE( app.m_indiP_dmStatus["timeline_misses"].get<int>() == 1 );
    REQUIRE( app.m_indiP_dmStatus["published"].get<int>() == 0 );
    fixture.acquire();
    fixture.advance();
    REQUIRE( app.serviceDM() );
    const auto target = g_images.at( "dm01disp10_aux_ready" ).m_words[synco::cmdTarget];
    REQUIRE( source.m_pixels[0] == Approx( expected( app, target, .1, 0 ) ).margin( 1e-8 ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief acc000 values are bitwise identical with DM output disabled, active through outage/removal, or polled faster.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim accelerometer realization is independent of DM output", "[accelSim][dm-output]" )
{
    using MagAOX::app::accelSimDetail::sample;
    const auto run = []( int mode ) // 0 disabled, 1 enabled, 2 enabled with extra polls and reads.
    {
        std::vector<sample> measured;
        dmOutputFixture     fixture;
        auto               &app   = fixture.m_app;
        auto               &words = g_images.at( "private_timing" ).m_words;
        if( mode > 0 )
            fixture.enable( 1 );
        for( int n = 0; n < 4000; ++n )
        {
            fixture.advance();
            measured.push_back( app.m_acquiredSample );
            if( mode == 0 )
                continue;
            if( n >= 1000 && n < 1200 )
                words[synco::state] = synco::invalid; // Primary outage while acquisition continues.
            if( n == 1200 )
                ++words[synco::generation];
            if( n == 3000 )
            {
                auto enable      = app.m_indiP_dmEnable;
                enable["toggle"] = pcf::IndiElement::Off;
                REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
            }
            for( int poll = 0; poll < ( mode == 2 ? 3 : 1 ); ++poll )
                if( app.serviceDM() )
                    acknowledge();
            sample probe;
            for( int read = 0; mode == 2 && read < 10; ++read )
                app.m_timeline.physical( g_tick + read * 37000, probe );
        }
        app.updateDMINDI();
        if( mode > 0 )
        {
            REQUIRE( app.m_indiP_dmStatus["published"].get<int>() > 2000 );
            REQUIRE( app.m_dmZeroAcknowledged );
        }
        return measured;
    };
    const auto disabled = run( 0 ), enabled = run( 1 ), polled = run( 2 );
    REQUIRE( enabled.size() == disabled.size() );
    REQUIRE( polled.size() == disabled.size() );
    for( size_t n = 0; n < disabled.size(); ++n )
    {
        REQUIRE( enabled[n].m_adc == disabled[n].m_adc );
        REQUIRE( enabled[n].m_acceleration == disabled[n].m_acceleration );
        REQUIRE( enabled[n].m_pointing == disabled[n].m_pointing );
        REQUIRE( polled[n].m_adc == disabled[n].m_adc );
        REQUIRE( polled[n].m_acceleration == disabled[n].m_acceleration );
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::measure;
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief A copied command acknowledged during poll must update status before the buffer is reused.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim recognizes a receipt arriving inside syncopationPoll", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    fixture.enable( .1 );
    auto &app = fixture.m_app;
    REQUIRE( app.serviceDM() );
    REQUIRE_FALSE( app.m_dmActive );
    fixture.advance();
    g_ackOnPoll = true;
    REQUIRE( app.serviceDM() );
    REQUIRE( app.m_dmActive );
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
