/** \file synco_transport_test.cpp
 * \brief Actual producer-base publication into an ordinary-file API double with an exact tagged mock consumer.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include <ImageStreamIO/ImageStreamIO.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

/// Deterministic clock used only in this separate, private output-test executable.
int output_clock( clockid_t clock /**< [in] Requested clock. */, timespec *value /**< [out] Private time. */ );
/// Existing producer wait, advanced in virtual time without touching a service or device.
int output_sleep( clockid_t       clock /**< [in] Ignored. */,
                  int             flags /**< [in] Ignored. */,
                  const timespec *value /**< [in] Virtual absolute deadline. */,
                  timespec       *remaining /**< [out] Ignored. */ );
#define clock_gettime output_clock
#define clock_nanosleep output_sleep
#include "../../../libMagAOX/app/syncoOutput.hpp"
#undef clock_gettime
#undef clock_nanosleep
#include "../shadow_controller.hpp"
#include "../../accelSim/accelSimDM.hpp"

/// \cond PRIVATE_OUTPUT_TRANSPORT
namespace
{
namespace synco = MagAOX::app::synco;
struct Image
{
    IMAGE_METADATA            md;     ///< Process-private metadata with an actual ordinary-file inode.
    std::vector<std::int64_t> words;  ///< Tagged schedule/commit/ack records.
    std::vector<float>        pixels; ///< Complete private own-channel surface.
};
std::map<std::string, Image>  images;
std::map<std::string, double> peaks; ///< Producer-only payload peaks, inspected after join.
std::atomic<std::int64_t>     tick{ 1000000000 };
std::atomic<bool>             accept{ true };
std::mutex                    registry;
std::atomic<bool>     foreignOwned{ false }, interfered{ false }; ///< Explicit competing-provider fault injection.
std::filesystem::path directory;

/// Create only a known private fixture allocation, never an ImageStreamIO namespace.
Image &make_image( const std::string &name /**< [in] Private fixture key. */,
                   unsigned           count /**< [in] Extent. */,
                   bool               pixels /**< [in] FLOAT source rather than INT64 record. */ )
{
    auto &im = images[name];
    std::ofstream( directory / name ).close();
    struct stat st{};
    if( stat( ( directory / name ).c_str(), &st ) != 0 )
        throw std::runtime_error( "Private fixture inode" );
    im.md               = {};
    im.md.inode         = st.st_ino;
    im.md.naxis         = pixels ? 2 : 1;
    im.md.size[0]       = pixels ? ( count == 2500 ? 50 : 2 ) : count;
    im.md.size[1]       = pixels ? ( count == 2500 ? 50 : 2 ) : 1;
    im.md.size[2]       = 1;
    im.md.nelement      = count;
    im.md.imdatamemsize = count * ( pixels ? sizeof( float ) : sizeof( std::int64_t ) );
    im.md.datatype      = pixels ? _DATATYPE_FLOAT : _DATATYPE_INT64;
    im.md.location      = -1;
    im.md.sem           = 1;
    std::snprintf( im.md.name, sizeof( im.md.name ), "%s", name.c_str() );
    if( pixels )
        im.pixels.assign( count, 0 );
    else
        im.words.assign( count, 0 );
    return im;
}
/// Refresh the mock consumer's existing advertised schedule in virtual time only.
void schedule( std::int64_t now /**< [in] Virtual local time. */ )
{
    auto it = images.find( "private_timing" );
    if( it == images.end() )
        return;
    auto      &im     = it->second;
    const auto origin = now - now % 500000;
    if( im.words[synco::reference] == origin )
        return;
    im.md.write              = 1;
    im.words[synco::version] = synco::productionVersion;
    im.words[synco::epoch] = im.words[synco::generation] = 1;
    im.words[synco::state]                               = synco::armed;
    im.words[synco::steps]                               = 4;
    im.words[synco::period]                              = 500000;
    im.words[synco::reference]                           = origin;
    im.words[synco::cycle]                               = origin / 500000;
    im.words[synco::expires]                             = origin + 1500000;
    for( int p = 0; p < 4; ++p )
    {
        im.words[synco::targets + p] = origin + 125000 * ( p + 1 );
        im.words[synco::cutoffs + p] = im.words[synco::targets + p] - 10000;
    }
    ++im.md.cnt0;
    im.md.write = 0;
}
/// Wait only for this private provider; no native MagAO-X application is started.
bool wait_for( const std::function<bool()> &predicate /**< [in] Private completion condition. */ )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( !predicate() && std::chrono::steady_clock::now() < end )
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    return predicate();
}
} // namespace
int output_clock( clockid_t clock, timespec *value )
{
    if( clock != CLOCK_MONOTONIC )
        return ::clock_gettime( clock, value );
    const auto now = tick.fetch_add( 1000 );
    schedule( now );
    value->tv_sec  = now / 1000000000;
    value->tv_nsec = now % 1000000000;
    return 0;
}
int output_sleep( clockid_t, int, const timespec *value, timespec * )
{
    tick.store( std::max( tick.load(), std::int64_t( value->tv_sec ) * 1000000000 + value->tv_nsec ) );
    return 0;
}
int ImageStreamIO_filename( char *out, std::size_t length, const char *name )
{
    return std::snprintf( out, length, "%s", ( directory / name ).c_str() ) < int( length ) ? 0 : -1;
}
int ImageStreamIO_openIm( IMAGE *image, const char *name )
{
    std::lock_guard<std::mutex> lock( registry );
    const auto                  it = images.find( name );
    if( it == images.end() )
        return -1;
    image->md        = &it->second.md;
    image->array.raw = it->second.pixels.empty() ? static_cast<void *>( it->second.words.data() )
                                                 : static_cast<void *>( it->second.pixels.data() );
    return 0;
}
void ImageStreamIO_closeIm( IMAGE *image )
{
    *image = {};
}
int ImageStreamIO_createIm_gpu(
    IMAGE *image, const char *name, int, std::uint32_t *sizes, int, int, int, int, int, int, int )
{
    if( foreignOwned.load() )
        interfered.store( true );
    const std::string key( name );
    if( !key.ends_with( "_aux_ready" ) )
        return -1;
    { //mutex scope
        std::lock_guard<std::mutex> lock( registry );
        make_image( key, sizes[0], false );
    }
    return ImageStreamIO_openIm( image, name );
}
int ImageStreamIO_UpdateIm( IMAGE *image )
{
    if( foreignOwned.load() )
        interfered.store( true );
    ++image->md->cnt0;
    image->md->write = 0;
    const std::string name( image->md->name );
    if( name.ends_with( "_aux" ) )
        for( float value : images.at( name ).pixels )
            peaks[name] = std::max( peaks[name], double( std::abs( value ) ) );
    if( name.ends_with( "_ready" ) )
    {
        auto       &receipt = images.at( name.substr( 0, name.size() - 6 ) + "_ack" );
        const auto *commit  = static_cast<const std::int64_t *>( image->array.raw );
        std::copy_n( commit, synco::commitWords, receipt.words.begin() );
        receipt.words[synco::ackStatus] = accept.load() ? synco::copied : synco::discarded;
        ++receipt.md.cnt0;
    }
    return 0;
}
/// \endcond

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Actual RLS/QR model proposals pass the unchanged synco producer protocol and receive exact copy/zero
 * receipts. */
TEST_CASE( "Real producer publishes independent prepared standard and assisted mock corrections",
           "[operational-transport]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::dev::dmSyncopationProducer<MagAOX::app::SyncoOutput>::syncopationPoll;
    MagAOX::app::dev::dmSyncopationProducer<MagAOX::app::SyncoOutput>::syncopationPublish;
#endif
    // clang-format on
    const bool assisted = GENERATE( false, true );
    const bool qr       = GENERATE( false, true );
    directory = std::filesystem::temp_directory_path() / ( "ddspc-private-output-" + std::to_string( getpid() ) );
    std::filesystem::create_directory( directory );
    images.clear();
    peaks.clear();
    tick.store( 1000000000 );
    accept.store( true );
    make_image( "private_timing", synco::words, false );
    make_image( "dm01disp10_aux", 4, true );
    make_image( "dm01disp10_aux_ack", synco::ackWords, false );
    make_image( "dm10disp10_aux", 4, true );
    make_image( "dm10disp10_aux_ack", synco::ackWords, false );
    schedule( tick.load() );
    DDSPC::ShadowConfig modelConfig;
    modelConfig.m_history       = 2;
    modelConfig.m_future        = 2;
    modelConfig.m_modes         = 1;
    modelConfig.m_accelChannels = 1;
    modelConfig.m_accelHistory  = 1;
    modelConfig.m_prior         = .001;
    modelConfig.m_useQrd        = qr;
    DDSPC::ShadowController models( modelConfig );
    models.training( DDSPC::ShadowMode::Standard, true );
    models.training( DDSPC::ShadowMode::AccelAssisted, true );
    std::mt19937                    rng( 123 );
    std::normal_distribution<float> noise;
    DDSPC::ShadowRecord             row;
    row.m_residual       = DDSPC::Matrix::Zero( 1, 1 );
    row.m_appliedCommand = row.m_acceleration = row.m_residual;
    row.m_wfsValid = row.m_timingValid = row.m_appliedValid = row.m_accelValid = true;
    for( unsigned n = 1; n < 700; ++n )
    {
        row.m_index               = n;
        row.m_residual( 0 )       = noise( rng );
        row.m_appliedCommand( 0 ) = noise( rng );
        row.m_acceleration( 0 )   = noise( rng );
        models.process( row );
    }
    const auto mode = assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
    REQUIRE( models.status( mode ).m_ready );
    const auto candidate = models.proposal( mode )( 0 );
    REQUIRE( std::isfinite( candidate ) );
    const auto               other = images.at( "dm01disp10_aux" ).pixels;
    MagAOX::app::SyncoOutput output;
    output.configure( { true, true, 100, 100, 10000000 }, "dm10disp10_aux", "private_timing", 30, 4 );
    std::atomic<bool>          slotsValid{ true };
    std::atomic<std::uint64_t> copiedCount{ 0 };
    std::atomic<bool>          copiedActive{ false }, copiedZero{ false };
    output.start( [] { return true; },
                  [&]( const auto &request, const auto &record, std::int64_t now, std::span<float> pixels )
                  {
                      if( request.cutoffNs <= now || request.targetNs != record[synco::targets + request.step] )
                          slotsValid.store( false );
                      std::fill( pixels.begin(), pixels.end(), candidate );
                      return true;
                  },
                  [&]( const synco::CommitRecord &commit, std::span<const float> pixels )
                  {
                      if( commit[synco::cmdVersion] != synco::productionVersion )
                          slotsValid.store( false );
                      ++copiedCount;
                      if( pixels.front() == 0 )
                          copiedZero.store( true );
                      else if( pixels.front() == candidate )
                          copiedActive.store( true );
                      else
                          slotsValid.store( false );
                  } );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( output.request( true ) );
    REQUIRE( wait_for( [&] { return output.status().m_acknowledged; } ) );
    REQUIRE( output.request( false ) );
    REQUIRE( wait_for( [&] { return output.status().m_zeroConfirmed; } ) );
    REQUIRE( output.stop() );
    REQUIRE( slotsValid.load() );
    REQUIRE( copiedCount.load() == output.status().m_acknowledgements );
    REQUIRE( copiedActive.load() );
    REQUIRE( copiedZero.load() );
    REQUIRE( images.at( "dm01disp10_aux" ).pixels == other );
    for( auto value : images.at( "dm10disp10_aux" ).pixels )
        REQUIRE( value == 0 );
    REQUIRE( output.status().m_published >= 3 ); // Startup zero, qualified payload, removal zero.
    images.clear();
    std::filesystem::remove_all( directory );
}
/** \brief The unchanged physical simulator and DM scale publish through the same real tagged producer, independently.
 */
TEST_CASE( "Simulator same-sample disturbance uses tagged publication and independent removal",
           "[operational-transport]" )
{
    namespace sim = MagAOX::app::accelSimDetail;
    directory     = std::filesystem::temp_directory_path() / ( "ddspc-private-sim-" + std::to_string( getpid() ) );
    std::filesystem::create_directory( directory );
    images.clear();
    peaks.clear();
    tick.store( 1000000000 );
    accept.store( true );
    make_image( "private_timing", synco::words, false );
    make_image( "dm01disp10_aux", 2500, true );
    make_image( "dm01disp10_aux_ack", synco::ackWords, false );
    make_image( "dm10disp10_aux", 2500, true );
    std::fill( images.at( "dm10disp10_aux" ).pixels.begin(), images.at( "dm10disp10_aux" ).pixels.end(), .123f );
    schedule( tick.load() );
    std::vector<double> x( 2500 ), y( 2500 ), mask( 2500, 1 );
    for( unsigned p = 0; p < 2500; ++p )
    {
        x[p] = ( double( p % 50 ) - 24.5 ) / 12;
        y[p] = ( double( p / 50 ) - 24.5 ) / 12;
    }
    sim::dmInjection injector;
    injector.prepare( x, y, mask, 1e-6 );
    sim::parameters parameters;
    parameters.m_fps = 8000;
    sim::model model;
    model.configure( parameters, sim::referenceModes( 123 ) );
    model.step();
    const auto               sample = model.m_sample;
    const auto               ticks  = model.m_ticks;
    std::atomic<double>      scale{ GENERATE( 0.0, 0.5 ) };
    std::atomic<bool>        allFinite{ true };
    MagAOX::app::SyncoOutput output;
    // Simulator follows dmSpeckle's existing producer policy; all three removed settings are zero.
    output.configure( { true, true, 0, 0, 0, true }, "dm01disp10_aux", "private_timing", 30, 2500 );
    output.start( [] { return true; },
                  [&]( const auto &, const auto &, std::int64_t, std::span<float> pixels )
                  {
                      auto stage = [&]( const std::vector<float> &values )
                      {
                          std::copy( values.begin(), values.end(), pixels.begin() );
                          return true;
                      };
                      const auto state = injector.service( sample, 3, scale.load(), true, stage );
                      if( state != sim::dmState::active )
                          allFinite.store( false );
                      return state == sim::dmState::active;
                  } );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( output.request( true ) );
    const bool activated = wait_for( [&] { return output.status().m_acknowledged; } );
    INFO( output.status().m_reason );
    INFO( output.status().m_published );
    INFO( allFinite.load() );
    REQUIRE( activated );
    scale.store( .5 );
    const auto before = output.status().m_published;
    REQUIRE( wait_for( [&] { return output.status().m_published > before + 3; } ) );
    REQUIRE( output.request( false ) );
    REQUIRE( wait_for( [&] { return output.status().m_zeroConfirmed; } ) );
    REQUIRE( output.stop() );
    REQUIRE( allFinite.load() );
    REQUIRE( peaks.at( "dm01disp10_aux" ) > 0 );
    REQUIRE( model.m_ticks == ticks );
    REQUIRE( model.m_sample.m_adc == sample.m_adc );
    for( float value : images.at( "dm01disp10_aux" ).pixels )
        REQUIRE( value == 0 );
    for( float value : images.at( "dm10disp10_aux" ).pixels )
        REQUIRE( value == .123f );
    images.clear();
    std::filesystem::remove_all( directory );
}

/** \brief An unavailable candidate skips its slot, retaining the last copy, without latching output off. */
TEST_CASE( "Real producer skips slots it cannot prepare in time", "[operational-transport]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::SyncoOutput::work;
    MagAOX::app::SyncoOutputState::skipped;
#endif
    // clang-format on
    directory = std::filesystem::temp_directory_path() / ( "ddspc-private-skip-" + std::to_string( getpid() ) );
    std::filesystem::create_directory( directory );
    images.clear();
    peaks.clear();
    tick.store( 1000000000 );
    accept.store( true );
    make_image( "private_timing", synco::words, false );
    make_image( "dm01disp09_aux", 4, true );
    make_image( "dm01disp09_aux_ack", synco::ackWords, false );
    schedule( tick.load() );
    std::atomic<bool>        ready{ false };
    MagAOX::app::SyncoOutput output;
    output.configure( { true, true, 0, 0, 0, true }, "dm01disp09_aux", "private_timing", 30, 4 );
    output.start( [] { return true; },
                  [&]( const auto &, const auto &, std::int64_t, std::span<float> pixels )
                  {
                      std::fill( pixels.begin(), pixels.end(), .1f );
                      return ready.load();
                  },
                  {},
                  false );
    std::thread worker( [&] { output.work(); } ); // Application-owned thread, as threadStart provides.
    REQUIRE( output.request( true ) );
    REQUIRE( wait_for( [&] { return output.status().m_skipped > 2; } ) );
    REQUIRE_FALSE( output.status().m_fault );
    REQUIRE( output.status().m_requested );
    ready.store( true );
    REQUIRE( wait_for( [&] { return output.status().m_acknowledged; } ) );
    REQUIRE( output.request( false ) );
    REQUIRE( wait_for( [&] { return output.status().m_zeroConfirmed; } ) );
    output.stop();
    worker.join();
    REQUIRE( output.status().m_zeroConfirmed );
    REQUIRE_FALSE( output.status().m_fault );
    for( float value : images.at( "dm01disp09_aux" ).pixels )
        REQUIRE( value == 0 );
    images.clear();
    std::filesystem::remove_all( directory );
}

/** \brief Throwing, discarded or newly invalid candidates latch off and remove only this provider. */
TEST_CASE( "Real producer faults inhibit and remove their own contribution", "[operational-transport]" )
{
    const int failure = GENERATE( 1, 2, 3 );
    directory = std::filesystem::temp_directory_path() / ( "ddspc-private-fault-" + std::to_string( getpid() ) );
    std::filesystem::create_directory( directory );
    images.clear();
    peaks.clear();
    tick.store( 1000000000 );
    accept.store( true );
    make_image( "private_timing", synco::words, false );
    make_image( "dm10disp10_aux", 4, true );
    make_image( "dm10disp10_aux_ack", synco::ackWords, false );
    make_image( "dm01disp10_aux", 4, true );
    std::fill( images.at( "dm01disp10_aux" ).pixels.begin(), images.at( "dm01disp10_aux" ).pixels.end(), .25f );
    schedule( tick.load() );
    std::atomic<bool>        qualified{ true };
    MagAOX::app::SyncoOutput output;
    output.configure( { true, true, 1, 1, 10000000 }, "dm10disp10_aux", "private_timing", 30, 4 );
    output.start( [&] { return qualified.load(); },
                  [&]( const auto &, const auto &, std::int64_t, std::span<float> pixels )
                  {
                      if( failure == 1 )
                          throw std::runtime_error( "Injected private callback failure" );
                      if( failure == 2 )
                          accept.store( false );
                      if( failure == 3 )
                          qualified.store( false );
                      std::fill( pixels.begin(), pixels.end(), .1f );
                      return true;
                  } );
    REQUIRE( output.request( true ) );
    REQUIRE( wait_for( [&] { return output.status().m_fault; } ) );
    accept.store( true );
    REQUIRE( wait_for( [&] { return output.status().m_zeroConfirmed; } ) );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE( output.stop() );
    REQUIRE( output.status().m_fault ); // No automatic nonzero reactivation.
    for( float value : images.at( "dm10disp10_aux" ).pixels )
        REQUIRE( value == 0 );
    for( float value : images.at( "dm01disp10_aux" ).pixels )
        REQUIRE( value == .25f );
    images.clear();
    std::filesystem::remove_all( directory );
}

/** \brief A competing cooperative producer prevents all source and sidecar writes, then only zero may recover. */
TEST_CASE( "Real producer respects an existing exclusive owner", "[operational-transport]" )
{
    directory = std::filesystem::temp_directory_path() / ( "ddspc-private-owner-" + std::to_string( getpid() ) );
    std::filesystem::create_directory( directory );
    images.clear();
    peaks.clear();
    tick.store( 1000000000 );
    accept.store( true );
    make_image( "private_timing", synco::words, false );
    make_image( "dm10disp10_aux", 4, true );
    make_image( "dm10disp10_aux_ack", synco::ackWords, false );
    schedule( tick.load() );
    const int owner = open( ( directory / "dm10disp10_aux.producer" ).c_str(), O_RDWR | O_CREAT, 0600 );
    REQUIRE( owner >= 0 );
    REQUIRE( flock( owner, LOCK_EX | LOCK_NB ) == 0 );
    foreignOwned.store( true );
    interfered.store( false );
    MagAOX::app::SyncoOutput output;
    output.configure( { true, true, 1, 1, 10000000 }, "dm10disp10_aux", "private_timing", 30, 4 );
    output.start( [] { return true; },
                  []( const auto &, const auto &, std::int64_t, std::span<float> pixels )
                  {
                      std::fill( pixels.begin(), pixels.end(), .1f );
                      return true;
                  } );
    REQUIRE( output.request( true ) );
    const bool blocked = wait_for( [&] { return output.status().m_fault; } );
    const auto during  = output.status();
    output.request( false );
    foreignOwned.store( false );
    flock( owner, LOCK_UN );
    close( owner );
    REQUIRE( blocked );
    REQUIRE( during.m_published == 0 );
    REQUIRE_FALSE( interfered.load() );
    REQUIRE( wait_for( [&] { return output.status().m_zeroConfirmed; } ) );
    REQUIRE( output.stop() );
    REQUIRE_FALSE( output.status().m_requested );
    REQUIRE_FALSE( output.status().m_acknowledged );
    for( float value : images.at( "dm10disp10_aux" ).pixels )
        REQUIRE( value == 0 );
    images.clear();
    std::filesystem::remove_all( directory );
}

} // namespace libXWCTest::loPredCtrlAccTest
