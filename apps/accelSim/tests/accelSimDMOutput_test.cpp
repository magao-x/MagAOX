/** \file accelSimDMOutput_test.cpp
 * \brief Actual accelSim operator workflow through the unchanged producer and private API doubles.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include <ImageStreamIO/ImageStreamIO.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
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
    using accelSim::m_controlMutex;
    using accelSim::m_dmActive;
    using accelSim::m_dmChannel;
    using accelSim::m_dmCommandValid;
    using accelSim::m_dmEnabled;
    using accelSim::m_dmInjection;
    using accelSim::m_dmReady;
    using accelSim::m_dmScale;
    using accelSim::m_dmStage;
    using accelSim::m_dmTransportFault;
    using accelSim::m_dmZeroAcknowledged;
    using accelSim::m_dmZeroPending;
    using accelSim::m_indiP_dmEnable;
    using accelSim::m_indiP_dmScale;
    using accelSim::m_indiP_dmStatus;
    using accelSim::m_indiP_dmSteps;
    using accelSim::m_indiP_dmTiming;
    using accelSim::m_indiP_opticalAmplitude;
    using accelSim::m_indiP_opticalCoupling;
    using accelSim::m_model;
    using accelSim::m_parameters;
    using accelSim::m_shutdown;
    using accelSim::m_syncopationTiming;
    using accelSim::m_timeline;
    using accelSim::measure;
    using accelSim::observeSchedule;
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
    explicit dmOutputFixture( int  writeLeadUs = 50 /**< [in] Configured producer lead before cutoff in us. */,
                              bool accelOnly   = false /**< [in] Enable the default 50 Hz ACC-only mode. */ );

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

dmOutputFixture::dmOutputFixture( int writeLeadUs, bool accelOnly )
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
    m_app.config.m_values["accelOnly.enabled"]       = accelOnly ? "1" : "0";
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

/// Publish one production record as the scheduler does when it observes a primary (main) update.
/** Mirrors dmSyncopation::publishProductionSchedule: steps 1..3 target main + k*125 us, step 0 the next main,
 * cutoffs are 70 us (readyLeadUs) earlier, and the record expires two primary periods after publication.
 */
void publishRecord( int64_t main /**< [in] primary grid time [ns] */, int64_t at /**< [in] publication time [ns] */ )
{
    auto &words = g_images.at( "private_timing" ).m_words;
    ++g_images.at( "private_timing" ).m_metadata.cnt0;
    words[synco::version] = synco::productionVersion;
    words[synco::epoch] = words[synco::generation] = 1;
    words[synco::state]                            = synco::armed;
    words[synco::steps]                            = 4;
    words[synco::period]                           = 500000;
    words[synco::reference]                        = at;
    words[synco::cycle]                            = main / 500000;
    words[synco::expires]                          = at + 1000000;
    for( int step = 0; step < 4; ++step )
    {
        words[synco::targets + step] = main + 125000 * ( step ? step : 4 );
        words[synco::cutoffs + step] = words[synco::targets + step] - 70000;
    }
}

/// Copy a ready commit once the scheduler's snapshot at its cutoff has occurred.
void serviceReceipt( int64_t delay /**< [in] receipt delay after the cutoff [ns] */ )
{
    const auto ready = g_images.find( "dm01disp10_aux_ready" );
    if( ready == g_images.end() || !ready->second.m_words[synco::cmdSequence] )
        return;
    const auto &ack = g_images.at( "dm01disp10_aux_ack" ).m_words;
    if( !std::equal( ready->second.m_words.begin(), ready->second.m_words.end(), ack.begin() ) &&
        g_tick >= ready->second.m_words[synco::cmdCutoff] + delay )
        acknowledge();
}

/// Results of a simulated scheduler run, read back through the app's INDI diagnostics.
struct workerRun
{
    std::array<int, 4>    m_requests{}, m_published{}, m_copied{};                     ///< Per-step counts.
    int                   m_late{ 0 }, m_rejected{ 0 }, m_skipped{ 0 }, m_misses{ 0 }; ///< Failure accounting.
    uint64_t              m_acquisitions{ 0 };                                         ///< acc000 measurements taken.
    double                m_minMargin{ 0 }, m_meanMargin{ 0 };                         ///< Cutoff-ready margin [us].
    std::array<double, 4> m_stepMinMargin{ 1e9, 1e9, 1e9, 1e9 }; ///< Smallest margin per step [us].
    int                   m_hits{ 0 }, m_preparedMisses{ 0 };    ///< Staged versus on-demand surfaces.
};

/// Drive the real serviceDM like dmThreadExec against a simulated 2-kHz production scheduler.
/** The worker polls again at once after a commit; otherwise it sleeps pollNs plus a random wake latency in
 * [0, jitterNs]. acc000 is acquired at every 125-us tick. This models scheduling latency; it is not an RTC test.
 */
workerRun simulateWorker( int64_t  pollNs /**< [in] sleep after an unsuccessful service [ns] */,
                          int64_t  jitterNs /**< [in] maximum additional wake latency [ns] */,
                          int      frames /**< [in] simulated primary frames */,
                          uint64_t seed /**< [in] wake-latency seed */,
                          int      writeLeadUs = 50 /**< [in] producer syncopation.writeLeadUs */ )
{
    dmOutputFixture fixture( writeLeadUs );
    auto           &app = fixture.m_app;
    fixture.enable( 1 );
    std::mt19937_64                        rng( seed );
    std::uniform_int_distribution<int64_t> jitter( 0, jitterNs );
    const int64_t                          first           = g_tick + 500000 - g_tick % 500000;
    int64_t                                nextAcquisition = g_tick + 125000;
    g_wakeDelayNs                                          = 3000;
    workerRun run;
    app.updateDMINDI(); // Start the timing interval with the simulation.
    while( g_tick < first + int64_t( frames ) * 500000 )
    {
        const int64_t latest = ( g_tick - first ) / 500000;
        if( g_tick >= first + latest * 500000 + 10000 )
            publishRecord( first + latest * 500000, first + latest * 500000 + 10000 ); // Observed 10 us after main.
        serviceReceipt( 5000 );
        for( ; nextAcquisition <= g_tick; nextAcquisition += 125000 )
        {
            app.measure( nextAcquisition, uint64_t( ( nextAcquisition - fixture.m_epoch ) / 125000 ) );
            ++run.m_acquisitions;
        }
        int64_t previousCutoff = 0;
        if( g_images.count( "dm01disp10_aux_ready" ) )
            previousCutoff = g_images.at( "dm01disp10_aux_ready" ).m_words[synco::cmdCutoff];
        if( !app.serviceDM() )
        {
            g_tick += pollNs + jitter( rng );
            continue;
        }
        // Each commit is exactly its slot's target-time surface, legal, and never ahead of its eligibility.
        const auto &ready  = g_images.at( "dm01disp10_aux_ready" ).m_words;
        const auto  main   = ( ready[synco::cmdCycle] - ( ready[synco::cmdStep] == 0 ) ) * 500000;
        const auto  margin = ( ready[synco::cmdCutoff] - ready[synco::cmdReadyNs] ) / 1e3;
        REQUIRE( g_images.at( "dm01disp10_aux" ).m_pixels[0] ==
                 Approx( expected( app, ready[synco::cmdTarget], 1, 0 ) ).margin( 1e-7 ) );
        REQUIRE( ready[synco::cmdReadyNs] <= ready[synco::cmdCutoff] );
        if( main >= first )
            REQUIRE( ready[synco::cmdReadyNs] >= main + 10000 ); // Not before its record existed.
        REQUIRE( ready[synco::cmdReadyNs] > previousCutoff );    // Not before the previous slot was consumed.
        auto &stepMargin = run.m_stepMinMargin[ready[synco::cmdStep]];
        stepMargin       = std::min( stepMargin, margin );
    }
    app.updateDMINDI();
    for( int step = 0; step < 4; ++step )
    {
        run.m_requests[step]  = app.m_indiP_dmSteps["requests_" + std::to_string( step )].get<int>();
        run.m_published[step] = app.m_indiP_dmSteps["published_" + std::to_string( step )].get<int>();
        run.m_copied[step]    = app.m_indiP_dmSteps["copied_" + std::to_string( step )].get<int>();
    }
    run.m_late           = app.m_indiP_dmStatus["late_returns"].get<int>();
    run.m_rejected       = app.m_indiP_dmStatus["rejected"].get<int>();
    run.m_misses         = app.m_indiP_dmStatus["timeline_misses"].get<int>();
    run.m_skipped        = app.m_indiP_dmSteps["skipped_slots"].get<int>();
    run.m_minMargin      = app.m_indiP_dmTiming["margin_min_us"].get<double>();
    run.m_meanMargin     = app.m_indiP_dmTiming["margin_mean_us"].get<double>();
    run.m_hits           = app.m_indiP_dmSteps["prepared_hits"].get<int>();
    run.m_preparedMisses = app.m_indiP_dmSteps["prepared_misses"].get<int>();
    for( int step = 0; step < 4; ++step )
        REQUIRE( app.m_indiP_dmSteps["advertised_" + std::to_string( step )].get<int>() >= frames - 1 );
    return run;
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
    fixture.enable( 2.5 ); // Scales above one amplify the same target-time surface linearly.
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
            REQUIRE( source.m_pixels[0] == Approx( expected( app, target, 2.5, 0 ) ).margin( 1e-7 ) );
            // One subframe of timing error would be detected at this tolerance.
            REQUIRE( std::abs( expected( app, target, 2.5, 0 ) - expected( app, target - 125000, 2.5, 0 ) ) > 1e-6 );
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

/** \brief INDI optical controls change only later DM pointing; acc000 values and DM publication are unaffected.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim runtime optical controls preserve acquisition and DM timing", "[accelSim][dm-output]" )
{
    using MagAOX::app::accelSimDetail::sample;
    struct record
    {
        std::vector<sample> m_acquired, m_physical; ///< acc000 samples and timeline DM pointing at acquisition.
        int                 m_published{ 0 };       ///< DM publications.
    };
    const auto run = []( bool controlled )
    {
        record          result;
        dmOutputFixture fixture;
        auto           &app = fixture.m_app;
        fixture.enable( 1 );
        const auto set = [&]( double coupling, double amplitude )
        {
            auto common = app.m_indiP_opticalCoupling, independent = app.m_indiP_opticalAmplitude;
            common["target"]      = coupling;
            independent["target"] = amplitude;
            REQUIRE( app.newCallBack_m_indiP_opticalCoupling( common ) == 0 );
            REQUIRE( app.newCallBack_m_indiP_opticalAmplitude( independent ) == 0 );
        };
        for( int n = 0; n < 4000; ++n )
        {
            if( controlled && n == 1000 )
                set( .5, 1 );
            if( controlled && n == 2000 )
                set( 1, 0 );
            fixture.advance();
            sample physical;
            REQUIRE( app.m_timeline.physical( g_tick, physical ) );
            result.m_acquired.push_back( app.m_acquiredSample );
            result.m_physical.push_back( physical );
            if( app.serviceDM() )
                acknowledge();
        }
        app.updateDMINDI();
        result.m_published = app.m_indiP_dmStatus["published"].get<int>();
        return result;
    };
    const auto baseline = run( false ), controlled = run( true );
    REQUIRE( controlled.m_published == baseline.m_published );
    REQUIRE( baseline.m_published > 2000 );
    bool changed = false;
    for( size_t n = 0; n < baseline.m_acquired.size(); ++n )
    {
        REQUIRE( controlled.m_acquired[n].m_adc == baseline.m_acquired[n].m_adc );
        REQUIRE( controlled.m_acquired[n].m_acceleration == baseline.m_acquired[n].m_acceleration );
        // Nodes already generated within the 5-ms (40-tick) lookahead keep their values; later ones follow the
        // controls.
        if( n < 1000 || n >= 2100 )
            REQUIRE( controlled.m_physical[n].m_pointing == baseline.m_physical[n].m_pointing );
        else if( n >= 1100 && n < 2000 )
            changed = changed || controlled.m_physical[n].m_pointing != baseline.m_physical[n].m_pointing;
    }
    REQUIRE( changed );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::measure;
    MagAOX::app::accelSim::newCallBack_m_indiP_opticalCoupling;
    MagAOX::app::accelSim::newCallBack_m_indiP_opticalAmplitude;
#endif
    // clang-format on
}

/** \brief The ACC-only mode changes acc000 but leaves DM pointing, WFS truth and published commands bitwise unchanged,
 * while optical_coupling and optical_amplitude change at runtime through the real callbacks.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim ACC-only mode never reaches the DM", "[accelSim][dm-output][accel-only]" )
{
    using MagAOX::app::accelSimDetail::sample;
    struct record
    {
        std::vector<sample>             m_acquired, m_physical; ///< acc000 samples and timeline values at acquisition.
        std::vector<std::vector<float>> m_commands;             ///< DM source surface after each service.
        int                             m_published{ 0 };       ///< DM publications.
    };
    const auto run = []( bool accelOnly )
    {
        record          result;
        dmOutputFixture fixture( 50, accelOnly );
        auto           &app = fixture.m_app;
        REQUIRE( app.m_parameters.m_accelOnly == accelOnly );
        fixture.enable( 1 );
        const auto set = [&]( double coupling, double amplitude )
        {
            auto common = app.m_indiP_opticalCoupling, independent = app.m_indiP_opticalAmplitude;
            common["target"]      = coupling;
            independent["target"] = amplitude;
            REQUIRE( app.newCallBack_m_indiP_opticalCoupling( common ) == 0 );
            REQUIRE( app.newCallBack_m_indiP_opticalAmplitude( independent ) == 0 );
        };
        for( int n = 0; n < 3000; ++n )
        {
            if( n == 1000 )
                set( .5, 1 );
            if( n == 2000 )
                set( 0, 0 );
            fixture.advance();
            sample physical;
            REQUIRE( app.m_timeline.physical( g_tick, physical ) );
            result.m_acquired.push_back( app.m_acquiredSample );
            result.m_physical.push_back( physical );
            if( app.serviceDM() )
                acknowledge();
            result.m_commands.push_back( g_images.at( "dm01disp10_aux" ).m_pixels );
        }
        app.updateDMINDI();
        result.m_published = app.m_indiP_dmStatus["published"].get<int>();
        return result;
    };
    const auto baseline = run( false ), accelOnly = run( true );
    REQUIRE( accelOnly.m_published == baseline.m_published );
    REQUIRE( baseline.m_published > 2000 );
    REQUIRE( accelOnly.m_commands == baseline.m_commands );
    bool sensed = false;
    for( size_t n = 0; n < baseline.m_acquired.size(); ++n )
    {
        REQUIRE( accelOnly.m_physical[n].m_pointing == baseline.m_physical[n].m_pointing );
        REQUIRE( accelOnly.m_physical[n].m_wfsPointing == baseline.m_physical[n].m_wfsPointing );
        sensed = sensed || accelOnly.m_acquired[n].m_adc != baseline.m_acquired[n].m_adc;
    }
    REQUIRE( sensed );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadConfigImpl;
    MagAOX::app::accelSimDetail::model::step;
#endif
    // clang-format on
}

/** \brief A simulated 2-kHz production scheduler: with 10-us polling every subframe is published, copied and timely.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim sustains four subframes per primary frame with prompt polling", "[accelSim][dm-output]" )
{
    constexpr int frames = 2000; // One simulated second at 2 kHz: 8000 subframes.
    const auto    run    = simulateWorker( 10000, 5000, frames, 3 );
    for( int step = 0; step < 4; ++step )
    {
        INFO( "step " << step );
        REQUIRE( run.m_published[step] >= frames - 1 ); // Edge frames only.
        REQUIRE( run.m_requests[step] == run.m_published[step] );
        REQUIRE( run.m_copied[step] >= run.m_published[step] - 1 ); // The final receipt is not yet polled.
    }
    REQUIRE( run.m_late == 0 );
    REQUIRE( run.m_rejected == 0 );
    REQUIRE( run.m_misses == 0 );
    REQUIRE( run.m_skipped == 0 );
    REQUIRE( run.m_minMargin > 0 ); // Every commit precedes its cutoff.
    REQUIRE( run.m_acquisitions >= uint64_t( 4 * frames ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::updateDMINDI;
#endif
    // clang-format on
}

/** \brief With writeLeadUs covering the slot spacing, commits occur on eligibility with large positive margins.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim advance preparation and early commit keep a consistent readiness margin", "[accelSim][dm-output]" )
{
    constexpr int frames = 2000;
    const auto    early  = simulateWorker( 10000, 20000, frames, 4, 125 );
    const auto    late   = simulateWorker( 10000, 20000, frames, 4, 50 );
    for( int step = 0; step < 4; ++step )
    {
        INFO( "step " << step );
        REQUIRE( early.m_published[step] >= frames - 1 );
        REQUIRE( early.m_stepMinMargin[step] > 0 );
    }
    // Steps 2, 3 and 0 become eligible on the previous receipt, about 120 us before their cutoffs.
    for( int step : { 0, 2, 3 } )
        REQUIRE( early.m_stepMinMargin[step] > 70 );
    // Step 1 is limited by its record (cutoff 45 us after publication), not by preparation.
    REQUIRE( early.m_stepMinMargin[1] > 10 );
    REQUIRE( early.m_meanMargin > late.m_meanMargin + 25 );
    for( int step : { 0, 2, 3 } ) // writeLeadUs=50 waits until cutoff-50 us whenever observed earlier.
        REQUIRE( late.m_stepMinMargin[step] == Approx( 48 ).margin( 3 ) ); // Less the 3-us modeled late wake.
    // Steps 2, 3 and 0 are staged while the previous receipt is awaited; step 1 is synthesized on observation.
    REQUIRE( early.m_hits >= 3 * ( frames - 1 ) );
    REQUIRE( early.m_hits + early.m_preparedMisses ==
             early.m_published[0] + early.m_published[1] + early.m_published[2] + early.m_published[3] );
    REQUIRE( early.m_late == 0 );
    REQUIRE( early.m_skipped == 0 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::observeSchedule;
    MagAOX::app::accelSim::stageStep;
#endif
    // clang-format on
}

/** \brief A staged surface is never reused after its scale, timeline realization or slot identity changes.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim invalidates staged surfaces", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture( 125 );
    auto           &app    = fixture.m_app;
    auto           &source = g_images.at( "dm01disp10_aux" );
    fixture.enable( .1 );
    REQUIRE( app.serviceDM() ); // Step 1, synthesized on observation.
    const auto &ready = g_images.at( "dm01disp10_aux_ready" ).m_words;
    // The scheduler consumes at the cutoff; the next step is eligible only after it.
    const auto consume = [&]
    {
        g_tick = ready[synco::cmdCutoff] + 5000;
        acknowledge();
    };
    REQUIRE_FALSE( app.serviceDM() ); // Awaiting the receipt: steps 2, 3 and 0 are staged now.
    REQUIRE( app.m_dmStage[2].m_valid );
    REQUIRE( app.m_dmStage[3].m_valid );
    REQUIRE( app.m_dmStage[0].m_valid );

    // Scale change: the staged step-2 surface at 0.1 must not be published at 0.2.
    auto scale      = app.m_indiP_dmScale;
    scale["target"] = .2;
    REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) == 0 );
    consume();
    REQUIRE( app.serviceDM() );
    REQUIRE( ready[synco::cmdStep] == 2 );
    REQUIRE( source.m_pixels[0] == Approx( expected( app, ready[synco::cmdTarget], .2, 0 ) ).margin( 1e-8 ) );

    // Timeline reset (rate change or restart): the realization at a target may change, so restage.
    REQUIRE_FALSE( app.serviceDM() ); // Restage steps 3 and 0 at 0.2.
    const auto generation = app.m_timeline.generation();
    app.m_model.restart();
    app.m_timeline.reset( fixture.m_epoch, app.m_model );
    REQUIRE( app.m_timeline.generation() == generation + 1 );
    consume();
    fixture.acquire();
    REQUIRE( app.serviceDM() );
    REQUIRE( ready[synco::cmdStep] == 3 );
    REQUIRE( source.m_pixels[0] == Approx( expected( app, ready[synco::cmdTarget], .2, 0 ) ).margin( 1e-8 ) );

    // Scheduler reacquisition changes the slot identity; disable clears staging and removes the contribution.
    consume();
    ++g_images.at( "private_timing" ).m_words[synco::generation];
    REQUIRE( app.serviceDM() );
    REQUIRE( source.m_pixels[0] == Approx( expected( app, ready[synco::cmdTarget], .2, 0 ) ).margin( 1e-8 ) );
    auto enable      = app.m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    REQUIRE_FALSE( app.m_dmStage[0].m_valid );
    consume();
    fixture.advance();
    REQUIRE( app.serviceDM() );
    REQUIRE( std::all_of( source.m_pixels.begin(), source.m_pixels.end(), []( float v ) { return v == 0; } ) );
    acknowledge();
    REQUIRE_FALSE( app.serviceDM() );
    REQUIRE( app.m_dmZeroAcknowledged );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::stageStep;
    MagAOX::app::accelSimDetail::timeline::generation;
#endif
    // clang-format on
}

/** \brief Slow wakes (the former 50-us SCHED_OTHER sleep) lose subframes, and the per-step counters show which ones.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim per-subframe diagnostics expose wake-latency losses", "[accelSim][dm-output]" )
{
    constexpr int frames = 2000;
    // 50-us sleep, 50-us default timer slack and up to 60 us of further scheduling delay.
    const auto run   = simulateWorker( 100000, 60000, frames, 5 );
    int        total = 0;
    for( int step = 0; step < 4; ++step )
    {
        INFO( "step " << step << " published " << run.m_published[step] );
        total += run.m_published[step];
    }
    // Step 1 exists only from the record's publication (main + 10 us) until its cutoff (main + 55 us).
    for( int step : { 0, 2, 3 } )
        REQUIRE( run.m_published[1] < run.m_published[step] );
    REQUIRE( total < 4 * frames * 0.85 );
    REQUIRE( run.m_skipped >= 4 * frames - total - 8 ); // Losses are never-observed slots, not late commits.
    REQUIRE( run.m_late == 0 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
#endif
    // clang-format on
}

/** \brief Print per-subframe publication versus worker wake latency for the simulated scheduler.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim wake-latency sweep", "[.][dm-benchmark]" )
{
    for( const auto [poll, jitter] : { std::pair<int64_t, int64_t>{ 10000, 5000 },
                                       { 20000, 20000 },
                                       { 50000, 20000 },
                                       { 60000, 60000 },
                                       { 100000, 60000 },
                                       { 100000, 200000 } } )
    {
        for( int lead : { 50, 125 } )
        {
            const auto run = simulateWorker( poll, jitter, 2000, 9, lead );
            std::printf( "sleep %3lld us + wake [0,%3lld] us: published by step 0/1/2/3 = %4d %4d %4d %4d of 2000, "
                         "skipped %5d, late %d, rate %.0f Hz, lead %3d: margin min %5.1f mean %5.1f us, hits %d\n",
                         (long long)poll / 1000,
                         (long long)jitter / 1000,
                         run.m_published[0],
                         run.m_published[1],
                         run.m_published[2],
                         run.m_published[3],
                         run.m_skipped,
                         run.m_late,
                         ( run.m_published[0] + run.m_published[1] + run.m_published[2] + run.m_published[3] ) / 1.0,
                         lead,
                         run.m_minMargin,
                         run.m_meanMargin,
                         run.m_hits );
        }
    }
}

/** \brief Local benchmark of the request-to-commit path; not a native Linux real-time measurement.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM publication path benchmark", "[.][dm-benchmark]" )
{
    using clock   = std::chrono::steady_clock;
    const auto us = []( clock::time_point a, clock::time_point b )
    { return std::chrono::duration<double, std::micro>( b - a ).count(); };
    dmOutputFixture fixture( 125 );
    auto           &app = fixture.m_app;
    fixture.enable( 1 );
    REQUIRE( app.serviceDM() );
    acknowledge();
    std::vector<double> onDemand, staged, stage, evaluate, synthesize;
    for( int n = 0; n < 40000; ++n )
    {
        fixture.advance();
        const bool prepare = n % 2; // Alternate the staged and on-demand publication paths.
        if( prepare )
        {
            const auto t0 = clock::now();
            app.observeSchedule( true ); // What the worker does while awaiting a receipt.
            stage.push_back( us( t0, clock::now() ) );
        }
        else
            for( auto &entry : app.m_dmStage )
                entry.m_valid = false; // Force synthesis on observation.
        const auto start = clock::now();
        REQUIRE( app.serviceDM() ); // Poll, receipt check, surface selection or synthesis, and producer commit.
        ( prepare ? staged : onDemand ).push_back( us( start, clock::now() ) );
        const auto                          target = g_images.at( "dm01disp10_aux_ready" ).m_words[synco::cmdTarget];
        MagAOX::app::accelSimDetail::sample physical;
        const auto                          t0 = clock::now();
        REQUIRE( app.m_timeline.physical( target, physical ) );
        const auto t1 = clock::now();
        app.m_dmInjection.synthesize( physical, 1 );
        evaluate.push_back( us( t0, t1 ) );
        synthesize.push_back( us( t1, clock::now() ) );
        acknowledge();
    }
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmSteps["prepared_hits"].get<int>() == 20000 );
    REQUIRE( app.m_indiP_dmSteps["prepared_misses"].get<int>() == 20001 );
    for( auto &[name, values] : { std::pair{ "publish (on demand)", &onDemand },
                                  std::pair{ "publish (staged)", &staged },
                                  std::pair{ "stage record", &stage },
                                  std::pair{ "timeline", &evaluate },
                                  std::pair{ "synthesis", &synthesize } } )
    {
        std::sort( values->begin(), values->end() );
        double sum = 0;
        for( double v : *values )
            sum += v;
        const auto at = [&]( double q ) { return ( *values )[size_t( q * ( values->size() - 1 ) )]; };
        std::printf( "%-20s mean %7.3f us  p99 %7.3f  p99.9 %7.3f  max %8.3f\n",
                     name,
                     sum / values->size(),
                     at( .99 ),
                     at( .999 ),
                     values->back() );
    }
}

/** \brief The DM worker never waits on the INDI control mutex, which the app loop holds during INDI I/O.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM service is independent of INDI property I/O", "[accelSim][dm-output]" )
{
    dmOutputFixture fixture;
    auto           &app = fixture.m_app;
    fixture.enable( .1 );
    {                                                           // mutex scope
        std::lock_guard<std::mutex> busy( app.m_controlMutex ); // As during updateINDI's sendSetProperty calls.
        REQUIRE( app.serviceDM() );
        acknowledge();
        fixture.advance();
        REQUIRE( app.serviceDM() );
    }
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["published"].get<int>() == 2 );
    REQUIRE( app.m_indiP_dmTiming["samples"].get<int>() == 2 );
    app.updateDMINDI(); // Interval timing is drained by each report; cumulative counts are not.
    REQUIRE( app.m_indiP_dmTiming["samples"].get<int>() == 0 );
    REQUIRE( app.m_indiP_dmStatus["published"].get<int>() == 2 );
    REQUIRE( app.m_indiP_dmSteps["published_1"].get<int>() + app.m_indiP_dmSteps["published_2"].get<int>() == 2 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::serviceDM;
    MagAOX::app::accelSim::updateDMINDI;
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
