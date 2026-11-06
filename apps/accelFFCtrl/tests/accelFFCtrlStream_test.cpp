/** \file accelFFCtrlStream_test.cpp
 * \brief Opt-in native ADC reader tests in a process-private ImageStreamIO namespace.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlStream.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <type_traits>
#include <unistd.h>
#ifdef __linux__
    #include <signal.h>
    #include <sys/wait.h>
#endif

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_NATIVE_FIXTURE
/// Own a synthetic raw ADC stream only after verifying its private, externally selected namespace.
class nativeInput
{
    bool     m_created{ false }; ///< Destruction is permitted only after this fixture created the image.
    uint32_t m_depth, m_axes;    ///< Original fixture geometry, reused only for a controlled private restart.

    /// Verify the private namespace and create this fixture's synthetic image.
    void create();

  public:
    IMAGE       m_image{};           ///< Sole synthetic producer mapping.
    std::string m_name;              ///< Unique non-actuator test stream name.
    bool        m_liveTime{ false }; ///< Use the actual realtime write stamp for the separately paced Linux producer.

    /// Create a small temporal raw-pair ring; MILK_SHM_DIR must already select an empty private directory.
    explicit nativeInput( uint32_t depth /**< [in] retained pair count */,
                          uint32_t axes = 3 /**< [in] temporal cube or depth-one two-dimensional image */ );

    /// Destroy only this fixture's private synthetic image.
    ~nativeInput();

    /// Recreate only this fixture's private stream under the same name, leaving an old reader mapping detached.
    void restart();

    /// Publish a synthetic plane following frameGrabber's ordering and timestamp conventions.
    void publish( uint64_t                counter /**< [in] source identity */,
                  int64_t                 acquired /**< [in] realtime acquisition ns */,
                  std::array<uint16_t, 2> codes /**< [in] raw pair */ );
};

nativeInput::nativeInput( uint32_t depth, uint32_t axes ) : m_depth( depth ), m_axes( axes )
{
    static unsigned serial = 0;
    m_name                 = "accff-private-" + std::to_string( getpid() ) + "-" + std::to_string( ++serial );
    create();
    if( serial == 1 )
        std::cout << "Native ABI: stream version=" << m_image.md->version << ", sizeof(IMAGE)=" << sizeof( IMAGE )
                  << ", sizeof(IMAGE_METADATA)=" << sizeof( IMAGE_METADATA ) << ", UINT16=" << _DATATYPE_UINT16 << '\n';
}

void nativeInput::create()
{
    const char *directory = std::getenv( "MILK_SHM_DIR" );
    REQUIRE( directory != nullptr );
    REQUIRE( std::string( directory ).starts_with( "/tmp/accelFFCtrl-native-" ) );
    char path[1024];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), m_name.c_str() ) == 0 );
    REQUIRE( std::filesystem::path( path ).parent_path() == std::filesystem::path( directory ) );
    REQUIRE_FALSE( std::filesystem::exists( path ) );
    REQUIRE( ( ( m_axes == 3 && m_depth > 0 ) || ( m_axes == 2 && m_depth == 1 ) ) );
    uint32_t size[3]{ 2, 1, m_depth };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image,
                                         m_name.c_str(),
                                         m_axes,
                                         size,
                                         _DATATYPE_UINT16,
                                         -1,
                                         1,
                                         IMAGE_NB_SEMAPHORE,
                                         0,
                                         m_axes == 3 ? CIRCULAR_BUFFER | ZAXIS_TEMPORAL : 0,
                                         0 ) == 0 );
    m_created        = true;
    m_image.md->cnt1 = m_depth - 1;
}

void nativeInput::restart()
{
    REQUIRE( ImageStreamIO_destroyIm( &m_image ) == 0 );
    m_created = false;
    m_image   = {};
    create();
}

nativeInput::~nativeInput()
{
    if( m_created )
        ImageStreamIO_destroyIm( &m_image );
}

void nativeInput::publish( uint64_t counter, int64_t acquired, std::array<uint16_t, 2> codes )
{
    const auto plane  = ( counter - 1 ) % m_depth;
    m_image.md->write = 1;
    std::memcpy( m_image.array.UI16 + 2 * plane, codes.data(), 2 * sizeof( uint16_t ) );
    m_image.md->atime     = { acquired / 1000000000, acquired % 1000000000 };
    m_image.md->writetime = { ( acquired + 10000 ) / 1000000000, ( acquired + 10000 ) % 1000000000 };
    if( m_liveTime && clock_gettime( CLOCK_REALTIME, &m_image.md->writetime ) != 0 )
        m_image.md->writetime = {};
    m_image.md->cnt1 = plane;
    m_image.md->cnt0 = counter;
    if( m_axes == 3 )
    {
        m_image.atimearray[plane]     = m_image.md->atime;
        m_image.writetimearray[plane] = m_image.md->writetime;
        m_image.cntarray[plane]       = counter;
    }
    m_image.md->write = 0;
    ImageStreamIO_sempost( &m_image, -1 );
}

#ifdef __linux__
/// Read the same clock bracket as the application, without a framework dependency.
inputClock nativeClock()
{
    timespec a, real, b;
    int64_t  first, epoch, last;
    if( clock_gettime( CLOCK_MONOTONIC, &a ) || clock_gettime( CLOCK_REALTIME, &real ) ||
        clock_gettime( CLOCK_MONOTONIC, &b ) || !inputProcessor::timeNs( a, first ) ||
        !inputProcessor::timeNs( real, epoch ) || !inputProcessor::timeNs( b, last ) || last < first )
        return {};
    return { epoch, first + ( last - first ) / 2, ( last - first ) / 2 };
}

/// Reap the test's own producer on assertion unwinding; never signal an existing application.
struct nativeChild
{
    pid_t m_pid{ -1 }; ///< Only the child returned by this test's fork, or -1 after waitpid.

    /// Terminate/reap the owned child if a test exits before its normal completion.
    ~nativeChild();
};

nativeChild::~nativeChild()
{
    if( m_pid > 0 )
    {
        kill( m_pid, SIGKILL );
        while( waitpid( m_pid, nullptr, 0 ) < 0 && errno == EINTR )
        {
        }
    }
}
#endif
/// \endcond

/** \brief Native reader copies all retained pairs and leaves producer payload/metadata intact, including depth-one
 * loss.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl private native retained ADC input", "[.][accelFFCtrl-native]" )
{
    constexpr int64_t epoch = 1700000000000000000;
    for( uint32_t depth : { 1, 8, 256 } )
    {
        nativeInput    producer( depth );
        adcStream      reader;
        inputProcessor processor;
        REQUIRE( reader.open( producer.m_name ) );
        REQUIRE( reader.inode() != 0 );
        processor.attached();
        producer.publish( 1, epoch + 1000000000, { 2017, 2031 } );
        auto poll = [&]( int64_t mono )
        { return processor.drain( reader, [mono] { return inputClock{ epoch + mono, mono, 10 }; } ); };
        REQUIRE( poll( 1000050000 ) == 1 );
        REQUIRE( reader.frame( 0 ).m_adc == std::array<uint16_t, 2>{ 2017, 2031 } );
        for( uint64_t n = 2; n <= 5; ++n )
            producer.publish( n, epoch + 1000000000 + ( n - 1 ) * 125000, { uint16_t( 2048 + n ), 2048 } );
        REQUIRE( poll( 1000550000 ) == ( depth == 1 ? 1 : 4 ) );
        REQUIRE( processor.diagnostics().m_dropped == ( depth == 1 ? 3 : 0 ) );
        REQUIRE( producer.m_image.md->cnt0 == 5 );
        REQUIRE( producer.m_image.array.UI16[2 * producer.m_image.md->cnt1] == 2053 );
        REQUIRE( reader.header().m_uint16 );
        producer.m_image.md->write = 1;
        REQUIRE( poll( 1000550000 ) == 0 );
        producer.m_image.md->write = 0;
        timespec now;
        REQUIRE( clock_gettime( CLOCK_MONOTONIC, &now ) == 0 );
        REQUIRE( reader.wait( int64_t( now.tv_sec ) * 1000000000 + now.tv_nsec + 100000 ) );
        reader.close();
        REQUIRE_FALSE( reader.header().m_valid );
        REQUIRE( producer.m_image.md->cnt0 == 5 );
        // Reattachment starts a new generation and binds only the newest publication.
        processor.disconnected();
        REQUIRE( reader.open( producer.m_name ) );
        processor.attached();
        REQUIRE( poll( 1000600000 ) == 1 );
        REQUIRE( processor.diagnostics().m_generation == 2 );
    }
}

/** \brief Missing names and DM-prefixed names cannot attach; incompatible native types cannot feed the estimator.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl native input validation", "[.][accelFFCtrl-native]" )
{
    nativeInput producer( 4 );
    adcStream   reader;
    REQUIRE_FALSE( reader.open( "dm01disp00" ) ); // Rejected before any native filename lookup/open.
    REQUIRE_FALSE( reader.open( "accff-private-missing" ) );
    REQUIRE( reader.open( producer.m_name ) );
    producer.m_image.md->datatype = _DATATYPE_FLOAT;
    REQUIRE_FALSE( reader.header().m_uint16 );
    producer.m_image.md->datatype = _DATATYPE_UINT16;
    producer.m_image.md->size[2]  = 5;
    REQUIRE_FALSE( reader.header().m_valid );
    producer.m_image.md->size[2] = 4;
    REQUIRE( reader.frame( 4 ).m_counter == 0 );
    producer.m_image.md->imagetype = CIRCULAR_BUFFER | ZAXIS_SPACIAL;
    REQUIRE_FALSE( reader.header().m_valid );
    producer.m_image.md->imagetype = CIRCULAR_BUFFER | ZAXIS_TEMPORAL;
    int claimed                    = -1;
    for( unsigned n = 0; n < producer.m_image.md->sem; ++n )
        if( producer.m_image.semReadPID[n] == getpid() )
            claimed = n;
    REQUIRE( claimed >= 0 );
    producer.m_image.semReadPID[claimed] = -1; // Simulate a producer clearing/reassigning the reader slot.
    reader.close();
    REQUIRE( producer.m_image.semReadPID[claimed] == -1 );
}

/** \brief Native temporal ring wrap, ordered batching and prediction use both UINT16 channels and per-plane time.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl native 8k timestamped ring replay", "[.][accelFFCtrl-native]" )
{
    static_assert( !std::is_copy_constructible_v<adcStream> && !std::is_copy_assignable_v<adcStream> );
    constexpr int64_t epoch = 1700000000000000000;
    parameters        config;
    config.m_zero         = { 2017, 2031 };
    config.m_voltsPerBit  = { 1e-6, 1e-6 };
    config.m_sensitivity  = { 1, 1 };
    config.m_predictionNs = 1000000;
    nativeInput    producer( 256 );
    adcStream      reader;
    inputProcessor processor( config );
    REQUIRE( reader.open( producer.m_name ) );
    processor.attached();
    for( uint64_t n = 1; n <= 12000; ++n )
    {
        const int64_t mono  = 1000000000 + n * 125000;
        const double  phase = 6.28318530717958647692 * 28.5 * n / 8000;
        producer.publish( n,
                          epoch + mono,
                          { uint16_t( 2017 + std::round( 1000 * std::cos( phase ) ) ),
                            uint16_t( 2031 + std::round( 700 * std::sin( phase ) ) ) } );
        if( n != 1 && n % 4 )
            continue;
        const int64_t receipt = mono + 50000;
        REQUIRE( processor.drain( reader, [receipt] { return inputClock{ epoch + receipt, receipt, 10 }; } ) > 0 );
        processor.predict( receipt );
        REQUIRE( processor.diagnostics().m_processed == n );
        REQUIRE( processor.diagnostics().m_dropped == 0 );
    }
    const auto &p = processor.diagnostics().m_prediction;
    REQUIRE( p.m_state == estimatorState::ready );
    REQUIRE( p.m_sampleTimeNs == 2500000000 );
    REQUIRE( p.m_targetTimeNs == 2501050000 );
    REQUIRE( p.m_amplitude[0] == Approx( .001 ).margin( 2e-7 ) );
    REQUIRE( p.m_amplitude[1] == Approx( .0007 ).margin( 2e-7 ) );
    const double phase = 6.28318530717958647692 * 28.5 * 1.50105;
    REQUIRE( p.m_acceleration[0] == Approx( .001 * std::cos( phase ) ).margin( 2e-7 ) );
    REQUIRE( p.m_acceleration[1] == Approx( .0007 * std::sin( phase ) ).margin( 2e-7 ) );
    for( uint64_t n = 12001; n <= 12300; ++n )
        producer.publish( n, epoch + 1000000000 + n * 125000, { 2048, 2048 } );
    REQUIRE( processor.drain( reader, [] { return inputClock{ epoch + 2537550000, 2537550000, 10 }; } ) == 256 );
    REQUIRE( processor.diagnostics().m_processed == 12016 );
    REQUIRE( processor.diagnostics().m_dropped == 284 ); // 44 overwritten + 240 retained but older than 2 ms.
    REQUIRE( processor.diagnostics().m_invalid == 240 );
}

/** \brief Two-dimensional depth-one stamps and same-name native recreation bind a new observation generation.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl native two dimensional input and restart", "[.][accelFFCtrl-native]" )
{
    constexpr int64_t epoch = 1700000000000000000;
    nativeInput       producer( 1, 2 );
    adcStream         reader;
    inputProcessor    processor;
    REQUIRE( reader.open( producer.m_name ) );
    REQUIRE( reader.header().m_axes == 2 );
    processor.attached();
    producer.publish( 10, epoch + 1000000000, { 2017, 2031 } );
    REQUIRE( processor.drain( reader, [] { return inputClock{ epoch + 1000050000, 1000050000, 10 }; } ) == 1 );
    const auto inode = reader.inode();
    producer.restart();
    struct stat replacement;
    char        path[1024];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), producer.m_name.c_str() ) == 0 );
    REQUIRE( stat( path, &replacement ) == 0 );
    REQUIRE( replacement.st_ino != inode ); // The still-mapped old inode cannot be reused yet.
    reader.close();
    processor.disconnected();
    REQUIRE( reader.open( producer.m_name ) );
    processor.attached();
    producer.publish( 1, epoch + 1000125000, { 2048, 2048 } );
    REQUIRE( processor.drain( reader, [] { return inputClock{ epoch + 1000175000, 1000175000, 10 }; } ) == 1 );
    REQUIRE( processor.diagnostics().m_generation == 2 );
    REQUIRE( processor.diagnostics().m_counter == 1 );
    producer.publish( 2, epoch + 1000250000, { 2048, 2048 } );
    producer.m_image.md->atime = {};
    REQUIRE( processor.drain( reader, [] { return inputClock{ epoch + 1000300000, 1000300000, 10 }; } ) == 1 );
    REQUIRE( processor.diagnostics().m_state == inputState::invalidTime );
    REQUIRE( processor.diagnostics().m_invalid == 1 );
}

/** \brief Both production policies see identical depth-one native records, gaps and integrity faults at 4/8 kHz.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl native depth-one optional-gap qualification", "[.][accelFFCtrl-native]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::adcStream::frame;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::predict;
#endif
    // clang-format on
    constexpr int64_t epoch = 1700000000000000000;
    for( int rate : { 4000, 8000 } )
        for( uint32_t axes : { 2, 3 } )
            for( const std::string fault :
                 { "single", "cluster", "busy", "delay", "invalid", "stale", "silence", "rollback", "restart" } )
            {
                CAPTURE( rate, axes, fault );
                const int64_t period = 1000000000 / rate;
                parameters    config;
                config.m_sampleRate   = rate;
                config.m_frequency    = 28;
                config.m_predictionNs = 1000000;
                config.m_voltsPerBit.fill( 1e-6 );
                config.m_sensitivity.fill( 1 );
                nativeInput    producer( 1, axes );
                adcStream      reader;
                inputProcessor strict( config ), optional( config, 256, 1000000, true );
                REQUIRE( reader.open( producer.m_name ) );
                strict.attached();
                optional.attached();
                uint64_t counter  = 0;
                int64_t  acquired = 1000000000 - period;
                auto     publish  = [&]
                {
                    acquired += period;
                    const double phase = 6.28318530717958647692 * 28 * ( acquired - 1000000000 ) * 1e-9;
                    producer.publish( ++counter,
                                      epoch + acquired,
                                      { uint16_t( 2048 + std::round( 1000 * std::cos( phase ) ) ),
                                        uint16_t( 2048 + std::round( 700 * std::sin( phase ) ) ) } );
                };
                auto drain = [&]( int64_t now )
                {
                    const inputClock clock{ epoch + now, now, 10 };
                    REQUIRE( strict.drain( reader, [clock] { return clock; } ) == 1 );
                    REQUIRE( optional.drain( reader, [clock] { return clock; } ) == 1 );
                };
                for( int n = 0; n < rate / 2; ++n )
                {
                    publish();
                    drain( acquired + 50000 );
                    if( n % ( rate / 2000 ) == 0 || n == rate / 2 - 1 )
                    {
                        strict.predict( acquired + 50000 );
                        optional.predict( acquired + 50000 );
                    }
                }
                REQUIRE( strict.diagnostics().m_prediction.m_state == estimatorState::ready );
                REQUIRE( optional.diagnostics().m_prediction.m_state == estimatorState::ready );
                const uint64_t warm = counter;
                if( fault == "restart" )
                {
                    const auto inode = reader.inode();
                    producer.restart();
                    reader.close();
                    strict.disconnected();
                    optional.disconnected();
                    REQUIRE( reader.open( producer.m_name ) );
                    REQUIRE( reader.inode() != inode );
                    strict.attached();
                    optional.attached();
                    counter = 0;
                }
                if( fault == "rollback" )
                    counter = 0;
                const unsigned lost = fault == "cluster" || fault == "delay" ? 3
                                      : fault == "single" || fault == "busy" ? 1
                                                                             : 0;
                for( unsigned n = 0; n <= lost; ++n )
                    publish();
                int64_t now = acquired + 50000;
                if( fault == "busy" )
                {
                    producer.m_image.md->write = 1;
                    const inputClock clock{ epoch + now, now, 10 };
                    REQUIRE( strict.drain( reader, [clock] { return clock; } ) == 0 );
                    REQUIRE( optional.drain( reader, [clock] { return clock; } ) == 0 );
                    REQUIRE( optional.diagnostics().m_resets == 1 );
                    REQUIRE( optional.diagnostics().m_busy == 1 );
                    producer.m_image.md->write = 0;
                }
                if( fault == "invalid" )
                    producer.m_image.md->atime = {};
                if( fault == "stale" )
                    now += 3000000;
                if( fault != "silence" )
                    drain( now );
                else
                    now += 3000000;
                strict.predict( now );
                optional.predict( now );
                const auto &a = strict.diagnostics(), &b = optional.diagnostics();
                const bool  admit = fault == "single" || fault == "busy";
                REQUIRE( b.m_admittedGaps == uint64_t( admit ) );
                REQUIRE( a.m_admittedGaps == 0 );
                REQUIRE( ( b.m_prediction.m_state == estimatorState::ready ) == admit );
                REQUIRE( a.m_prediction.m_state != estimatorState::ready );
                REQUIRE( a.m_dropped == b.m_dropped );
                REQUIRE( b.m_dropped == lost + unsigned( fault == "invalid" || fault == "stale" ) );
                REQUIRE( b.m_rejectedGaps == uint64_t( lost > 0 && !admit ) );
                REQUIRE( a.m_rejectedGaps == uint64_t( lost > 0 ) );
                const uint64_t fullResets = fault == "restart" ? 3 : fault == "silence" ? 1 : 2;
                REQUIRE( a.m_resets == fullResets );
                REQUIRE( b.m_resets == ( admit ? 1 : fullResets ) );
                REQUIRE( b.m_invalid == uint64_t( fault == "invalid" || fault == "stale" ) );
                REQUIRE( b.m_rollbacks == uint64_t( fault == "rollback" ) );
                REQUIRE( b.m_restarts == uint64_t( fault == "rollback" || fault == "restart" ) );
                if( fault != "rollback" && fault != "restart" && fault != "silence" )
                    REQUIRE( b.m_processed + b.m_dropped == warm + lost + 1 );
                if( admit )
                {
                    const auto &p = b.m_prediction;
                    REQUIRE( p.m_sequence == counter );
                    REQUIRE( p.m_sampleTimeNs == acquired );
                    REQUIRE( p.m_targetTimeNs == now + 1000000 );
                    REQUIRE( now - p.m_sampleTimeNs <= config.m_maxAgeNs );
                    const double phase = 6.28318530717958647692 * 28 * ( p.m_targetTimeNs - 1000000000 ) * 1e-9;
                    REQUIRE( p.m_acceleration[0] == Approx( .001 * std::cos( phase ) ).margin( 2e-7 ) );
                    REQUIRE( p.m_acceleration[1] == Approx( .0007 * std::sin( phase ) ).margin( 2e-7 ) );
                }
            }
}

#ifdef __linux__
/** \brief A separately paced process exercises native semaphore wakes, wrap, acquisition age, prediction and loss
 * accounting.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl private native 4k and 8k paced input", "[.][accelFFCtrl-native-rate]" )
{
    for( int rate : { 4000, 8000 } )
        for( uint32_t depth : { 1, 256 } )
            for( bool allowGap : { false, true } )
            {
                CAPTURE( rate, depth, allowGap );
                const uint64_t frames = 5 * rate + 1;
                nativeInput    producer( depth );
                producer.m_liveTime = true;
                adcStream  reader;
                parameters config;
                config.m_voltsPerBit  = { 1e-6, 1e-6 };
                config.m_sensitivity  = { 1, 1 };
                config.m_predictionNs = 1000000;
                config.m_sampleRate   = rate;
                inputProcessor processor( config, 256, 1000000, allowGap );
                REQUIRE( reader.open( producer.m_name ) );
                processor.attached();
                const auto start = nativeClock();
                REQUIRE( start.m_mono > 0 );
                producer.publish( 1, start.m_real, { 2048, 2048 } );
                REQUIRE( processor.drain( reader, nativeClock ) == 1 );
                nativeChild child;
                child.m_pid = fork();
                REQUIRE( child.m_pid >= 0 );
                if( child.m_pid == 0 )
                {
                    auto clock = nativeClock();
                    if( clock.m_mono <= 0 )
                        _exit( 2 );
                    int64_t deadline = clock.m_mono;
                    for( uint64_t n = 2; n <= frames; ++n )
                    {
                        deadline += 1000000000 / rate;
                        const timespec target{ deadline / 1000000000, deadline % 1000000000 };
                        int            result;
                        do
                        {
                            result = clock_nanosleep( CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr );
                        } while( result == EINTR );
                        if( result )
                            _exit( 3 );
                        clock = nativeClock();
                        if( clock.m_real <= 0 )
                            _exit( 4 );
                        const double phase = 6.28318530717958647692 * 28.5 * ( ( clock.m_real - start.m_real ) * 1e-9 );
                        producer.publish( n,
                                          clock.m_real,
                                          { uint16_t( 2048 + std::round( 1000 * std::cos( phase ) ) ),
                                            uint16_t( 2048 + std::round( 700 * std::sin( phase ) ) ) } );
                    }
                    _exit( 0 ); // Do not destroy the parent's private fixture/mapping in the producer process.
                }
                int      status         = 0;
                int64_t  nextPrediction = start.m_mono;
                bool     ready          = false;
                uint64_t queries = 0, valid = 0;
                int64_t  serviceTotal = 0, serviceMax = 0;
                while( true )
                {
                    const auto workStart = nativeClock();
                    processor.drain( reader, nativeClock );
                    const auto clock = nativeClock();
                    REQUIRE( clock.m_mono > 0 );
                    REQUIRE( clock.m_mono - start.m_mono < 15000000000 );
                    if( clock.m_mono >= nextPrediction )
                    {
                        processor.predict( clock.m_mono );
                        ready = ready || processor.diagnostics().m_prediction.m_state == estimatorState::ready;
                        ++queries;
                        valid += processor.diagnostics().m_prediction.m_state == estimatorState::ready;
                        nextPrediction = clock.m_mono + 500000;
                    }
                    const auto workNs = nativeClock().m_mono - workStart.m_mono;
                    serviceTotal += workNs;
                    serviceMax       = std::max( serviceMax, workNs );
                    const pid_t done = waitpid( child.m_pid, &status, WNOHANG );
                    if( done == child.m_pid )
                    {
                        child.m_pid = -1;
                        break;
                    }
                    REQUIRE( ( done == 0 || ( done < 0 && errno == EINTR ) ) );
                    REQUIRE( reader.wait( nextPrediction ) );
                }
                // Consume the bounded tail after the producer has stopped, without inventing fresh receipt times.
                for( unsigned n = 0; n < 4 && processor.diagnostics().m_counter != frames; ++n )
                    processor.drain( reader, nativeClock );
                const auto  &stats   = processor.diagnostics();
                const double seconds = ( nativeClock().m_mono - start.m_mono ) * 1e-9;
                std::cout << "Native " << rate << "/2k, depth=" << depth << ", optional=" << allowGap
                          << ": duration=" << seconds << " s, processed=" << stats.m_processed
                          << ", dropped=" << stats.m_dropped << ", invalid=" << stats.m_invalid
                          << ", copy_races=" << stats.m_races << ", resets=" << stats.m_resets
                          << ", input_fps=" << stats.m_rate << ", acquisition_fps=" << stats.m_sampleRate
                          << ", ready_seen=" << ready << ", admitted=" << stats.m_admittedGaps
                          << ", rejected=" << stats.m_rejectedGaps << ", executed_queries=" << queries
                          << ", ready_queries=" << valid
                          << ", executed_query_availability_pct=" << 100. * valid / queries
                          << ", service_total_ns=" << serviceTotal << ", service_max_ns=" << serviceMax << '\n';
                REQUIRE( WIFEXITED( status ) );
                REQUIRE( WEXITSTATUS( status ) == 0 );
                REQUIRE( reader.header().m_counter == frames );
                REQUIRE( stats.m_counter == frames );
                REQUIRE( stats.m_processed + stats.m_dropped == frames );
                REQUIRE( ready );
            }
}
#endif

} // namespace accelFFCtrlTest
} // namespace libXWCTest
