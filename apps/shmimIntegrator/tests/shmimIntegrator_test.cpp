/** \file shmimIntegrator_test.cpp
 * \brief Offline numerical and lifecycle tests for the shmimIntegrator app.
 * \ingroup shmimIntegrator_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include <mx/improc/eigenCube.hpp>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace mx
{
namespace improc
{
/// One-shot failure at the history allocation boundary, leaving the real algorithms intact.
inline bool failIntegratorCubeAllocation{ false };

/// Exercise the application's allocation failure path without exhausting workstation memory.
template <typename T>
class integratorTestCube : public eigenCube<T>
{
  public:
    /// Delegate to mxlib unless this test requested an allocation failure.
    void resize( int rows /**< [in] width */, int columns /**< [in] height */, int planes /**< [in] depth */ );
};

template <typename T>
void integratorTestCube<T>::resize( int rows, int columns, int planes )
{
    if( failIntegratorCubeAllocation )
    {
        failIntegratorCubeAllocation = false;
        throw std::bad_alloc();
    }
    eigenCube<T>::resize( rows, columns, planes );
}
} // namespace improc
} // namespace mx
/// \endcond

// Only the app's history allocation is substituted; all base headers are already included.
#define eigenCube integratorTestCube
#include "../shmimIntegrator.hpp"
#undef eigenCube

#include <filesystem>
#include <future>
#include <atomic>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace MagAOX
{
namespace app
{
/// Expose the real callbacks without starting any instrument or INDI connection.
class shmimIntegrator_test : public shmimIntegrator
{
  public:
    using shmimIntegrator::acquireAndCheckValid;
    using shmimIntegrator::configureAcquisition;
    using shmimIntegrator::findMatchingDark;
    using shmimIntegrator::fps;
    using shmimIntegrator::imageT;
    using shmimIntegrator::installSavedImage;
    using shmimIntegrator::loadImageIntoStream;
    using shmimIntegrator::m_activeAverage;
    using shmimIntegrator::m_activeGeneration;
    using shmimIntegrator::m_avgImage;
    using shmimIntegrator::m_avgTime;
    using shmimIntegrator::m_configuredGeneration;
    using shmimIntegrator::m_continuous;
    using shmimIntegrator::m_dark;
    using shmimIntegrator::m_dark2;
    using shmimIntegrator::m_fileSaveDir;
    using shmimIntegrator::m_fileSaver;
    using shmimIntegrator::m_fps;
    using shmimIntegrator::m_generation;
    using shmimIntegrator::m_imageCondition;
    using shmimIntegrator::m_imageMutex;
    using shmimIntegrator::m_imageValid;
    using shmimIntegrator::m_inFlight;
    using shmimIntegrator::m_nAverage;
    using shmimIntegrator::m_prepared;
    using shmimIntegrator::m_running;
    using shmimIntegrator::m_source;
    using shmimIntegrator::m_stateRevision;
    using shmimIntegrator::m_stateString;
    using shmimIntegrator::m_stateStringChanged;
    using shmimIntegrator::m_stateStringValid;
    using shmimIntegrator::m_stateStringValidOnStart;
    using shmimIntegrator::m_updated;
    using shmimIntegrator::startAcquisition;

    bool             m_workersStarted{ false }; ///< Real worker loops need shutdown.
    struct sigaction m_previousSignal{};        ///< Restored after the worker test.

    /// Launch the real base loops using already-created private input streams.
    void startWorkers( const std::string &sourceName /**< [in] source stream */,
                       const std::string &darkName /**< [in] first dark stream */,
                       const std::string &dark2Name /**< [in] second dark stream */,
                       const std::string &outputName /**< [in] output stream */ );
    /// Join started workers and restore signal handling.
    void stopWorkers();
    /// Wait for a newer acknowledged generation with complete matching darks.
    bool waitReady( uint32_t width /**< [in] desired width */,
                    uint32_t height /**< [in] desired height */,
                    uint64_t after /**< [in] previous generation */ );
    /// Snapshot the current generation.
    uint64_t generation();

    /// Suppress the process logger before any fixture is constructed.
    static bool quietLogs();

    /// Private disk workspace for each fixture.
    std::string m_directory;

    /// Allocate a dark without installing any pixels.
    int darkAllocate( bool     second /**< [in] second dark selector */,
                      uint32_t width /**< [in] width */,
                      uint32_t height /**< [in] height */,
                      uint8_t  type /**< [in] datatype */ = _DATATYPE_FLOAT );
    /// Copy a raw source frame through the real callback.
    int sourcePixels( void *data /**< [in] source pixels */ );
    /// Copy a raw dark frame through the real callback.
    int darkPixels( bool second /**< [in] second dark selector */, void *data /**< [in] dark pixels */ );
    /// Invoke the real numeric callback with a local property.
    int number( const std::string &name /**< [in] property name */, double value /**< [in] requested value */ );
    /// Invoke the start/stop callback.
    int run( bool running /**< [in] start or stop */ );
    /// Invoke the state-source callback, optionally with validity only.
    int stateUpdate( const std::string &value /**< [in] state string */,
                     bool               valid /**< [in] state validity */,
                     bool               validityOnly /**< [in] omit current element */ = false );
    /// Wake a direct callback waiter without invoking any base worker.
    void stop();
    /// Query the output worker's restart request when the worker is not running.
    bool reconfigRequested() const;
    /// Attach a test metadata block for the write-flag failure check.
    void outputStream( IMAGE *image /**< [in] test-owned image */ );

    /// Initialize local properties, disk workspace, and averaging parameters.
    shmimIntegrator_test( unsigned average /**< [in] window length */         = 2,
                          unsigned update /**< [in] moving update interval */ = 0 );
    /// Stop workers and release only this fixture's resources.
    ~shmimIntegrator_test();
    /// Simulate the source monitor importing a descriptor.
    int source( uint32_t width /**< [in] source width */,
                uint32_t height /**< [in] source height */,
                uint8_t  type /**< [in] source datatype */ = _DATATYPE_FLOAT );
    /// Allocate and install one complete dark frame.
    int dark( bool     second /**< [in] select the second dark */,
              uint32_t width /**< [in] dark width */,
              uint32_t height /**< [in] dark height */,
              float    value /**< [in] constant pixels */ );
    /// Feed one constant source frame.
    int feed( float value /**< [in] constant pixels */ );
    /// Complete output configuration like the base frame-grabber.
    int start();
    /// Claim and copy an output with guards on both sides.
    std::vector<float> output();

  private:
    /// Initialize the fixture only after logging has been silenced.
    shmimIntegrator_test( bool     quiet /**< [in] preconstruction logging setup token */,
                          unsigned average /**< [in] window length */,
                          unsigned update /**< [in] update cadence */ );
};

bool shmimIntegrator_test::quietLogs()
{
    m_log.m_logLevel = logPrio::LOG_EMERGENCY;
    return true;
}

shmimIntegrator_test::shmimIntegrator_test( unsigned average, unsigned update )
    : shmimIntegrator_test( quietLogs(), average, update )
{
}

shmimIntegrator_test::shmimIntegrator_test( bool quiet, unsigned average, unsigned update )
{
    static_cast<void>( quiet );
    m_nAverage   = average;
    m_nUpdate    = update;
    m_configName = "integrator_test";
    createStandardIndiNumber<unsigned>( m_indiP_nAverage, "nAverage", 1, 100000, 1, "%u" );
    createStandardIndiNumber<unsigned>( m_indiP_nUpdate, "nUpdate", 0, 100000, 1, "%u" );
    createStandardIndiNumber<float>( m_indiP_avgTime, "avgTime", 0, 100000, 0, "%f" );
    createStandardIndiToggleSw( m_indiP_startAveraging, "start" );
    m_indiP_fpsSource.setDevice( "camera" );
    m_indiP_fpsSource.setName( "fps" );
    m_indiP_stateSource.setDevice( "camera" );
    m_indiP_stateSource.setName( "state_string" );
    char directory[] = "/tmp/shmimIntegrator-test-XXXXXX";
    auto result      = mkdtemp( directory );
    REQUIRE( result != nullptr );
    m_directory   = result;
    m_fileSaveDir = m_directory;
}

shmimIntegrator_test::~shmimIntegrator_test()
{
    stopWorkers();
    stop();
    std::error_code error;
    std::filesystem::remove_all( m_directory, error );
}

int shmimIntegrator_test::source( uint32_t width, uint32_t height, uint8_t type )
{
    shmimMonitorT::m_width    = width;
    shmimMonitorT::m_height   = height;
    shmimMonitorT::m_dataType = type;
    return allocate( dev::shmimT() );
}

int shmimIntegrator_test::dark( bool second, uint32_t width, uint32_t height, float value )
{
    std::vector<float> data( size_t( width ) * height, value );
    int                rv = darkAllocate( second, width, height );
    return rv < 0 ? rv : darkPixels( second, data.data() );
}

int shmimIntegrator_test::darkAllocate( bool second, uint32_t width, uint32_t height, uint8_t type )
{
    if( second )
    {
        dark2MonitorT::m_width    = width;
        dark2MonitorT::m_height   = height;
        dark2MonitorT::m_dataType = type;
        return allocate( dark2ShmimT() );
    }
    darkMonitorT::m_width    = width;
    darkMonitorT::m_height   = height;
    darkMonitorT::m_dataType = type;
    return allocate( darkShmimT() );
}

int shmimIntegrator_test::sourcePixels( void *data )
{
    return processImage( data, dev::shmimT() );
}

int shmimIntegrator_test::darkPixels( bool second, void *data )
{
    return second ? processImage( data, dark2ShmimT() ) : processImage( data, darkShmimT() );
}

int shmimIntegrator_test::number( const std::string &name, double value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number );
    property.setDevice( name == "fps" ? "camera" : m_configName );
    property.setName( name );
    property.add( pcf::IndiElement( name == "fps" ? "current" : "target" ) );
    property[name == "fps" ? "current" : "target"].set( value );
    if( name == "nAverage" )
        return newCallBack_m_indiP_nAverage( property );
    if( name == "nUpdate" )
        return newCallBack_m_indiP_nUpdate( property );
    if( name == "avgTime" )
        return newCallBack_m_indiP_avgTime( property );
    return setCallBack_m_indiP_fpsSource( property );
}

int shmimIntegrator_test::run( bool running )
{
    auto property = m_indiP_startAveraging;
    property["toggle"].setSwitchState( running ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_startAveraging( property );
}

int shmimIntegrator_test::stateUpdate( const std::string &value, bool valid, bool validityOnly )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text );
    property.setDevice( "camera" );
    property.setName( "state_string" );
    property.add( pcf::IndiElement( "valid" ) );
    property["valid"] = valid ? "yes" : "no";
    if( !validityOnly )
    {
        property.add( pcf::IndiElement( "current" ) );
        property["current"] = value;
    }
    return setCallBack_m_indiP_stateSource( property );
}

void shmimIntegrator_test::stop()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    m_stopping = true;
    m_imageCondition.notify_all();
}

bool shmimIntegrator_test::reconfigRequested() const
{
    return frameGrabberT::m_reconfig;
}

void shmimIntegrator_test::outputStream( IMAGE *image )
{
    frameGrabberT::m_imageStream = image;
}

int shmimIntegrator_test::feed( float value )
{
    std::vector<float> data( size_t( shmimMonitorT::m_width ) * shmimMonitorT::m_height, value );
    return processImage( data.data(), dev::shmimT() );
}

int shmimIntegrator_test::start()
{
    int rv = configureAcquisition();
    if( rv != 0 )
        return rv;
    frameGrabberT::m_typeSize = sizeof( float );
    frameGrabberT::m_reconfig = false;
    return startAcquisition();
}

std::vector<float> shmimIntegrator_test::output()
{
    REQUIRE( acquireAndCheckValid() == 0 );
    std::vector<float> data( size_t( frameGrabberT::m_width ) * frameGrabberT::m_height + 2, -999 );
    REQUIRE( loadImageIntoStream( data.data() + 1 ) == 0 );
    REQUIRE( data.front() == -999 );
    REQUIRE( data.back() == -999 );
    return { data.begin() + 1, data.end() - 1 };
}

void shmimIntegrator_test::startWorkers( const std::string &sourceName,
                                         const std::string &darkName,
                                         const std::string &dark2Name,
                                         const std::string &outputName )
{
    struct sigaction action{};
    action.sa_sigaction = sigUsr1Handler;
    action.sa_flags     = SA_SIGINFO;
    sigemptyset( &action.sa_mask );
    REQUIRE( sigaction( SIGUSR1, &action, &m_previousSignal ) == 0 );
    m_workersStarted = true;
    state( stateCodes::OPERATING );
    shmimMonitorT::m_shmimName                = sourceName;
    darkMonitorT::m_shmimName                 = darkName;
    dark2MonitorT::m_shmimName                = dark2Name;
    frameGrabberT::m_shmimName                = outputName;
    frameGrabberT::m_latencyCircBuffMaxLength = 0;
    dark2MonitorT::m_getExistingFirst         = true;
    shmimMonitorT::m_smThreadInit             = false;
    darkMonitorT::m_smThreadInit              = false;
    dark2MonitorT::m_smThreadInit             = false;
    frameGrabberT::m_fgThreadInit             = false;
    shmimMonitorT::m_smThread                 = std::thread( [this]() { shmimMonitorT::smThreadExec(); } );
    darkMonitorT::m_smThread                  = std::thread( [this]() { darkMonitorT::smThreadExec(); } );
    dark2MonitorT::m_smThread                 = std::thread( [this]() { dark2MonitorT::smThreadExec(); } );
    frameGrabberT::m_fgThread                 = std::thread( [this]() { frameGrabberT::fgThreadExec(); } );
}

void shmimIntegrator_test::stopWorkers()
{
    if( !m_workersStarted )
        return;
    m_shutdown = 1; // The real base uses this flag; these tests are excluded from the app-only TSan run.
    appShutdown();
    sigaction( SIGUSR1, &m_previousSignal, nullptr );
    m_workersStarted = false;
}

bool shmimIntegrator_test::waitReady( uint32_t width, uint32_t height, uint64_t after )
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 6 );
    do
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_imageMutex );
            if( m_generation > after && m_activeGeneration == m_generation && m_prepared && m_source.m_width == width &&
                m_source.m_height == height && m_dark.m_info.m_width == width && m_dark.m_info.m_height == height &&
                m_dark.m_set && m_dark2.m_info.m_width == width && m_dark2.m_info.m_height == height && m_dark2.m_set )
                return true;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    } while( std::chrono::steady_clock::now() < deadline );
    return false;
}

uint64_t shmimIntegrator_test::generation()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    return m_generation;
}

/// Stop workers before test-owned input mappings leave scope, including on assertion failure.
struct workerGuard
{
    shmimIntegrator_test &m_app; ///< Fixture whose workers this guard joins.
    /// Join before stream owners are destroyed.
    ~workerGuard();
};

workerGuard::~workerGuard()
{
    m_app.stopWorkers();
}

/// Own one isolated local ImageStreamIO producer or reader.
class localStream
{
  public:
    IMAGE       m_image{};        ///< Mapping owned by this fixture.
    std::string m_name;           ///< Unique stream name inside a private MILK_SHM_DIR.
    bool        m_open{ false };  ///< Mapping needs cleanup.
    bool        m_owner{ false }; ///< Whether cleanup destroys or only closes the stream.

    /// Remember the name; mapping is opened separately.
    explicit localStream( const std::string &name /**< [in] stream name */ );
    /// Close only resources owned by this fixture.
    ~localStream();
    /// Create or replace a constant float image.
    void create( uint32_t width /**< [in] width */,
                 uint32_t height /**< [in] height */,
                 float    value /**< [in] pixel value */ );
    /// Open an existing output mapping and flush its reader semaphore.
    void open();
    /// Close the current mapping.
    void close();
    /// Notify consumers of a complete constant image.
    void post();
    /// Wait for one output and copy its pixels.
    std::vector<float> read();
};

localStream::localStream( const std::string &name ) : m_name( name )
{
}

localStream::~localStream()
{
    close();
}

void localStream::close()
{
    if( !m_open )
        return;
    if( m_owner )
        ImageStreamIO_destroyIm( &m_image );
    else
        ImageStreamIO_closeIm( &m_image );
    m_open = false;
}

void localStream::create( uint32_t width, uint32_t height, float value )
{
    close();
    uint32_t sizes[] = { width, height, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image,
                                         m_name.c_str(),
                                         3,
                                         sizes,
                                         _DATATYPE_FLOAT,
                                         -1,
                                         1,
                                         IMAGE_NB_SEMAPHORE,
                                         0,
                                         CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                         0 ) == 0 );
    m_open           = true;
    m_owner          = true;
    m_image.md->cnt1 = 0;
    for( size_t n = 0; n < size_t( width ) * height; ++n )
        m_image.array.F[n] = value;
}

void localStream::open()
{
    close();
    REQUIRE( ImageStreamIO_openIm( &m_image, m_name.c_str() ) == 0 );
    m_open  = true;
    m_owner = false;
    REQUIRE( ImageStreamIO_semflush( &m_image, 5 ) == 0 );
}

void localStream::post()
{
    ++m_image.md->cnt0;
    REQUIRE( ImageStreamIO_sempost( &m_image, -1 ) == 0 );
}

std::vector<float> localStream::read()
{
    timespec deadline;
    clock_gettime( CLOCK_REALTIME, &deadline );
    deadline.tv_sec += 5;
    REQUIRE( sem_timedwait( m_image.semptr[5], &deadline ) == 0 );
    REQUIRE( m_image.md->write == 0 );
    REQUIRE( m_image.md->datatype == _DATATYPE_FLOAT );
    size_t count = size_t( m_image.md->size[0] ) * m_image.md->size[1];
    return { m_image.array.F, m_image.array.F + count };
}

} // namespace app
} // namespace MagAOX
/// \endcond

namespace libXWCTest
{
/// Integrator regressions using only locally owned arrays and streams.
/** \ingroup shmimIntegrator_unit_test */
namespace shmimIntegratorTest
{
using MagAOX::app::shmimIntegrator_test;

/// Verify block averages subtract both complete dark images.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator block average and two darks", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::allocate;
    MagAOX::app::shmimIntegrator::processImage;
    MagAOX::app::shmimIntegrator::loadImageIntoStream;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    REQUIRE( app.source( 3, 2 ) == 0 );
    REQUIRE( app.dark( false, 3, 2, 1 ) == 0 );
    REQUIRE( app.dark( true, 3, 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 5 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.feed( 9 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 4 ) ) );
}

/// Preserve moving-average burn-in, cadence, and first-dark-only subtraction.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator moving average cadence", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    shmimIntegrator_test app( 3, 2 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.dark( false, 2, 2, 1 ) == 0 );
    REQUIRE( app.dark( true, 2, 2, 100 ) == 0 );
    REQUIRE( app.start() == 0 );
    for( float value : { 1, 2, 3 } )
        REQUIRE( app.feed( value ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.feed( 4 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 2 ) ) );
    REQUIRE( app.feed( 5 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.feed( 6 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 4 ) ) );
}

/// Preserve one-shot completion and a pending frame under producer backpressure.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator one shot and backpressure", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    app.m_continuous = false;
    REQUIRE( app.source( 2, 3 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 2 ) == 0 );
    REQUIRE( app.feed( 6 ) == 0 );
    REQUIRE_FALSE( app.m_running );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 4 ) ) );
}

/// A claimed old frame remains intact while allocations replace the requested shape.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator resize after claim keeps the old destination safe", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::allocate;
    MagAOX::app::shmimIntegrator::acquireAndCheckValid;
    MagAOX::app::shmimIntegrator::loadImageIntoStream;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 7 ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.source( 5, 3 ) == 0 );
    REQUIRE_FALSE( app.m_prepared );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE( app.configureAcquisition() < 0 );
    std::vector<float> old( 6, -999 );
    REQUIRE( app.loadImageIntoStream( old.data() + 1 ) == 0 );
    REQUIRE_THAT( old, Catch::Matchers::Approx( std::vector<float>{ -999, 7, 7, 7, 7, -999 } ) );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.reconfigRequested() );
    for( auto shape : { std::pair{ 5u, 3u }, { 1u, 2u }, { 1u, 2u } } )
    {
        REQUIRE( app.source( shape.first, shape.second ) == 0 );
        REQUIRE( app.start() == 0 );
        REQUIRE_FALSE( app.m_updated );
        REQUIRE( app.feed( 11 ) == 0 );
        REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( shape.first * shape.second, 11 ) ) );
    }
}

/// Setup must reject a generation superseded between configure and start.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator setup acknowledges only the newest allocation", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::configureAcquisition;
    MagAOX::app::shmimIntegrator::startAcquisition;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    REQUIRE( app.configureAcquisition() < 0 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.configureAcquisition() == 0 );
    auto old = app.m_configuredGeneration;
    REQUIRE( app.dark( false, 2, 2, 1 ) == 0 );
    REQUIRE( app.dark( true, 2, 2, 2 ) == 0 );
    REQUIRE( app.m_generation == old + 2 );
    REQUIRE( app.startAcquisition() < 0 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.m_activeGeneration == app.m_generation );
    REQUIRE( app.feed( 10 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 7 ) ) );
}

/// A resize drops partial and completed-but-unclaimed results, including same-size replacements.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator restart discards obsolete work", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::requestReconfigure;
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 2 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.feed( 4 ) == 0 );
    REQUIRE( app.m_updated );
    REQUIRE( app.source( 2, 2, _DATATYPE_INT16 ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.start() == 0 );
    std::vector<int16_t> pixels( 4, 12 );
    REQUIRE( app.sourcePixels( pixels.data() ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.sourcePixels( pixels.data() ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 12 ) ) );
    // A spurious notification is not a frame.
    app.m_imageCondition.notify_all();
    REQUIRE( app.acquireAndCheckValid() == 1 );
}

/// Both axes and a complete post-allocation frame are required for each optional dark.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator dark geometry and readiness", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::allocateDark;
    MagAOX::app::shmimIntegrator::processDark;
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    bool                 second = GENERATE( false, true );
    shmimIntegrator_test app( 1 );
    REQUIRE( app.source( 3, 2 ) == 0 );
    for( auto shape : { std::pair{ 3u, 4u }, { 4u, 2u }, { 3u, 2u } } )
    {
        REQUIRE( app.dark( second, shape.first, shape.second, 3 ) == 0 );
        REQUIRE( app.start() == 0 );
        REQUIRE( app.feed( 10 ) == 0 );
        float expected = shape == std::pair{ 3u, 2u } ? 7 : 10;
        REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, expected ) ) );
    }
    REQUIRE( app.darkAllocate( second, 3, 2 ) == 0 );
    REQUIRE_FALSE( ( second ? app.m_dark2.m_set : app.m_dark.m_set ) );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 10 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 10 ) ) );
    REQUIRE( app.darkAllocate( second, 3, 2, 0 ) < 0 );
    REQUIRE_FALSE( ( second ? app.m_dark2.m_set : app.m_dark.m_set ) );
    REQUIRE( app.configureAcquisition() < 0 );
    REQUIRE( app.darkAllocate( second, 3, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
}

/// Reject unsupported input and overflow before allocation or conversion.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator validates sizes and averaging parameters", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::describeImage;
    MagAOX::app::shmimIntegrator::configureAcquisition;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_nAverage;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_nUpdate;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_avgTime;
    MagAOX::app::shmimIntegrator::setCallBack_m_indiP_fpsSource;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    REQUIRE( app.source( 0, 2 ) < 0 );
    REQUIRE( app.source( UINT32_MAX, UINT32_MAX ) < 0 );
    REQUIRE( app.source( 2, 2, 0 ) < 0 );
    REQUIRE( app.configureAcquisition() < 0 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    for( double invalid : { -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        auto generation = app.m_generation;
        REQUIRE( app.number( "nAverage", invalid ) < 0 );
        REQUIRE( app.number( "nUpdate", invalid ) < 0 );
        REQUIRE( app.number( "avgTime", invalid ) < 0 );
        REQUIRE( app.number( "fps", invalid ) < 0 );
        REQUIRE( app.m_generation == generation );
    }
    REQUIRE( app.number( "nAverage", 0 ) < 0 );
    REQUIRE( app.number( "nAverage", 1.5 ) < 0 );
    REQUIRE( app.number( "nAverage", double( UINT32_MAX ) + 1 ) < 0 );
    REQUIRE( app.number( "nAverage", UINT32_MAX ) == 0 );
    REQUIRE( app.number( "nUpdate", 1 ) == 0 );
    REQUIRE( app.configureAcquisition() < 0 );
    REQUIRE_FALSE( app.m_prepared );
    REQUIRE( app.number( "nAverage", 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.sourcePixels( nullptr ) < 0 );
}

/// Runtime requests gate old buffers until the coherent new mode is acknowledged.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator stages averaging and FPS changes", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_nAverage;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_nUpdate;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_avgTime;
    MagAOX::app::shmimIntegrator::setCallBack_m_indiP_fpsSource;
    MagAOX::app::shmimIntegrator::fps;
    #endif
    // clang-format on
    shmimIntegrator_test app( 2, 1 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE( app.number( "nAverage", 5 ) == 0 );
    REQUIRE( app.m_activeAverage == 2 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.start() == 0 );
    for( unsigned n = 0; n < 5; ++n )
        REQUIRE( app.feed( 3 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 3 ) ) );
    REQUIRE( app.number( "nUpdate", 0 ) == 0 );
    REQUIRE( app.number( "avgTime", 0.5 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.m_activeAverage == 10 ); // Default before an FPS update.
    REQUIRE( app.number( "fps", 8 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.m_activeAverage == 4 );
    REQUIRE( app.fps() == Approx( 2 ) );
    for( unsigned n = 0; n < 4; ++n )
        REQUIRE( app.feed( 6 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 6 ) ) );
    REQUIRE( app.number( "fps", 0.25 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.m_activeAverage == 1 );
    REQUIRE( app.number( "avgTime", 1e30 ) == 0 );
    REQUIRE( app.number( "fps", 1e30 ) == 0 );
    REQUIRE( app.configureAcquisition() < 0 );
}

/// Input invalidation and shutdown wake an output waiter without another source frame.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator wakes idle output for restart and shutdown", "[shmimIntegrator][concurrency]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::acquireAndCheckValid;
    MagAOX::app::shmimIntegrator::appShutdown;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    auto waiter = std::async( std::launch::async, [&app]() { return app.acquireAndCheckValid(); } );
    REQUIRE( app.source( 3, 2 ) == 0 );
    auto result = waiter.wait_for( std::chrono::seconds( 2 ) );
    if( result != std::future_status::ready )
        app.stop();
    REQUIRE( result == std::future_status::ready );
    REQUIRE( waiter.get() == 1 );
    REQUIRE( app.start() == 0 );
    auto stopped = std::async( std::launch::async, [&app]() { return app.acquireAndCheckValid(); } );
    // No monitor threads were started; real shutdown must still wake the waiter.
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( stopped.wait_for( std::chrono::seconds( 2 ) ) == std::future_status::ready );
    REQUIRE( stopped.get() < 0 );
}

/// Dark writes cannot proceed while another callback owns the shared image state.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator serializes complete dark frames", "[shmimIntegrator][concurrency]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::processDark;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    REQUIRE( app.source( 32, 32 ) == 0 );
    REQUIRE( app.dark( false, 32, 32, 1 ) == 0 );
    REQUIRE( app.start() == 0 );
    std::vector<float>           data( 1024, 3 );
    std::promise<void>           entered;
    std::unique_lock<std::mutex> lock( app.m_imageMutex );
    auto                         copied = std::async( std::launch::async,
                              [&]()
                              {
                                  entered.set_value();
                                  return app.darkPixels( false, data.data() );
                              } );
    entered.get_future().wait();
    auto blocked = copied.wait_for( std::chrono::milliseconds( 20 ) );
    lock.unlock(); // Release before any assertion can throw.
    REQUIRE( blocked == std::future_status::timeout );
    REQUIRE( copied.wait_for( std::chrono::seconds( 2 ) ) == std::future_status::ready );
    REQUIRE( copied.get() == 0 );
    REQUIRE( app.feed( 10 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 1024, 7 ) ) );
}

/// Saved output can precede a source connection without fabricating monitor geometry.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator loads saved output before the source connects", "[shmimIntegrator][fits]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::findMatchingDark;
    MagAOX::app::shmimIntegrator::installSavedImage;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    app.m_fileSaver = true;
    REQUIRE( app.run( false ) == 0 );
    REQUIRE( app.stateUpdate( "stateA", true ) == 0 );
    shmimIntegrator_test::imageT old( 3, 2 ), image( 3, 2 );
    old.setConstant( 1 );
    image.setConstant( 8 );
    mx::fits::fitsFile<float> file;
    REQUIRE( file.write( app.m_directory + "/integrator_test_stateA__T20260101000000.fits", old ) ==
             mx::error_t::noerror );
    REQUIRE( file.write( app.m_directory + "/integrator_test_stateA__T20260102000000.fits", image ) ==
             mx::error_t::noerror );
    REQUIRE( app.findMatchingDark() == 0 );
    REQUIRE( app.m_source.m_pixels == 0 );
    REQUIRE( app.m_imageValid );
    REQUIRE( app.start() == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 8 ) ) );
    REQUIRE( app.run( true ) == 0 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 2 ) == 0 );
    REQUIRE( app.feed( 4 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 3 ) ) );
}

/// Reject a disk candidate if shape, state revision, configuration, or run state has changed.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator rejects stale saved candidates without changing storage", "[shmimIntegrator][fits]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::installSavedImage;
    MagAOX::app::shmimIntegrator::setCallBack_m_indiP_stateSource;
    MagAOX::app::shmimIntegrator::newCallBack_m_indiP_startAveraging;
    #endif
    // clang-format on
    shmimIntegrator_test app;
    app.m_fileSaver = true;
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.run( false ) == 0 );
    REQUIRE( app.stateUpdate( "stateA", true ) == 0 );
    auto                         generation = app.m_generation;
    auto                         revision   = app.m_stateRevision;
    const auto                  *storage    = app.m_avgImage.data();
    shmimIntegrator_test::imageT image( 3, 2 );
    image.setConstant( 9 );
    REQUIRE( app.installSavedImage( image, generation, revision ) == 0 );
    REQUIRE( app.m_avgImage.data() == storage );
    REQUIRE( app.m_stateStringChanged );
    REQUIRE_FALSE( app.m_imageValid );
    image.resize( 2, 2 );
    image.setConstant( 9 );
    SECTION( "state changed away and back during read" )
    {
        REQUIRE( app.stateUpdate( "stateB", true ) == 0 );
        REQUIRE( app.stateUpdate( "stateA", true ) == 0 );
    }
    SECTION( "validity-only change during read" )
    {
        REQUIRE( app.stateUpdate( "", false, true ) == 0 );
    }
    SECTION( "start and stop during read" )
    {
        REQUIRE( app.run( true ) == 0 );
        REQUIRE( app.run( false ) == 0 );
    }
    SECTION( "configuration changed during read" )
    {
        REQUIRE( app.number( "nAverage", 3 ) == 0 );
    }
    REQUIRE( app.installSavedImage( image, generation, revision ) == 0 );
    REQUIRE( app.m_avgImage.data() == storage );
    REQUIRE_FALSE( app.m_imageValid );
    REQUIRE_FALSE( app.m_updated );
}

/// One-shot saving uses a completed snapshot and refuses invalid state history.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator saves only valid completed one shots", "[shmimIntegrator][fits]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::saveImage;
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    bool                 changeState = GENERATE( false, true );
    shmimIntegrator_test app;
    app.m_continuous = false;
    app.m_fileSaver  = true;
    REQUIRE( app.stateUpdate( "stateA", true ) == 0 );
    REQUIRE( app.run( true ) == 0 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 4 ) == 0 );
    if( changeState )
        REQUIRE( app.stateUpdate( "stateB", true ) == 0 );
    REQUIRE( app.feed( 8 ) == 0 );
    REQUIRE_FALSE( app.m_running );
    REQUIRE( app.m_imageValid == !changeState );
    std::vector<std::string> files;
    for( auto &entry : std::filesystem::directory_iterator( app.m_directory ) )
        files.push_back( entry.path() );
    REQUIRE( files.size() == ( changeState ? 0 : 1 ) );
    if( !changeState )
    {
        shmimIntegrator_test::imageT image;
        mx::fits::fitsFile<float>    file;
        REQUIRE( file.read( image, files.front() ) == mx::error_t::noerror );
        REQUIRE( image.rows() == 2 );
        REQUIRE( image.cols() == 2 );
        REQUIRE( image.mean() == Approx( 6 ) );
    }
}

/// Unexpected output-copy failures release the base's write flag and the claim.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator copy errors clear output write state", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::loadImageIntoStream;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 1 ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    IMAGE          image{};
    IMAGE_METADATA metadata{};
    image.md = &metadata;
    app.outputStream( &image );
    metadata.write = 1;
    auto result    = app.loadImageIntoStream( nullptr );
    app.outputStream( nullptr );
    REQUIRE( result < 0 );
    REQUIRE( metadata.write == 0 );
    REQUIRE_FALSE( app.m_inFlight );
}

/// Exercise the production monitor and output loops across real local stream replacement.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator workers recover across local stream replacements", "[shmimIntegrator][streams]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::allocate;
    MagAOX::app::shmimIntegrator::processImage;
    MagAOX::app::shmimIntegrator::configureAcquisition;
    MagAOX::app::shmimIntegrator::startAcquisition;
    MagAOX::app::shmimIntegrator::acquireAndCheckValid;
    MagAOX::app::shmimIntegrator::loadImageIntoStream;
    MagAOX::app::shmimIntegrator::reconfig;
    MagAOX::app::shmimIntegrator::appShutdown;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    auto                 shmDirectory = app.m_directory + "/shm";
    std::filesystem::create_directory( shmDirectory );
    REQUIRE( setenv( "MILK_SHM_DIR", shmDirectory.c_str(), 1 ) == 0 );
    using MagAOX::app::localStream;
    std::string suffix = std::to_string( getpid() );
    localStream source( "integrator_source_" + suffix ), dark( "integrator_dark_" + suffix ),
        dark2( "integrator_dark2_" + suffix ), output( "integrator_output_" + suffix );
    source.create( 4, 3, 10 );
    dark.create( 4, 3, 1 );
    dark2.create( 4, 3, 2 );
    app.startWorkers( source.m_name, dark.m_name, dark2.m_name, output.m_name );
    // The worker guard must stop readers before the input mappings are destroyed on failure.
    MagAOX::app::workerGuard guard{ app };
    REQUIRE( app.waitReady( 4, 3, 0 ) );
    output.open();
    source.post();
    REQUIRE_THAT( output.read(), Catch::Matchers::Approx( std::vector<float>( 12, 7 ) ) );
    output.close();
    unsigned index = 0;
    for( auto shape : { std::pair{ 7u, 5u }, { 2u, 2u }, { 2u, 2u }, { 4u, 3u } } )
    {
        auto before = app.generation();
        source.create( shape.first, shape.second, 20 + index );
        dark.create( shape.first, shape.second, 1 );
        dark2.create( shape.first, shape.second, 2 );
        REQUIRE( app.number( "nAverage", 1 + index % 3 ) == 0 );
        // Three monitor allocations plus the numeric request must all have arrived, even at the same size.
        REQUIRE( app.waitReady( shape.first, shape.second, before + 3 ) );
        output.open();
        for( unsigned n = 0; n < 1 + index % 3; ++n )
            source.post();
        REQUIRE_THAT( output.read(),
                      Catch::Matchers::Approx( std::vector<float>( shape.first * shape.second, 17 + index ) ) );
        output.close();
        ++index;
    }
    app.stopWorkers();
}

/// A slow output reader retains one result through both unread and claimed states.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator keeps backpressure through output copy", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::processImage;
    MagAOX::app::shmimIntegrator::acquireAndCheckValid;
    MagAOX::app::shmimIntegrator::loadImageIntoStream;
    #endif
    // clang-format on
    unsigned             update = GENERATE( 0u, 1u );
    shmimIntegrator_test app( 2, update );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 2 ) == 0 );
    REQUIRE( app.feed( 4 ) == 0 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.feed( 200 ) == 0 );
    std::vector<float> pixels( 4 );
    REQUIRE( app.loadImageIntoStream( pixels.data() ) == 0 );
    REQUIRE_THAT( pixels, Catch::Matchers::Approx( std::vector<float>( 4, 3 ) ) );
    REQUIRE( app.source( 2, 2 ) == 0 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 6 ) == 0 );
    REQUIRE( app.feed( 8 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 4, 7 ) ) );
}

/// Concurrent dark updates always produce a uniform, complete correction image.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator concurrent dark updates never tear", "[shmimIntegrator][concurrency]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::processDark;
    MagAOX::app::shmimIntegrator::processImage;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    REQUIRE( app.source( 32, 32 ) == 0 );
    REQUIRE( app.dark( false, 32, 32, 1 ) == 0 );
    REQUIRE( app.start() == 0 );
    std::atomic<bool> failed{ false };
    std::jthread      writer(
        [&]( std::stop_token stop )
        {
            std::vector<float> one( 1024, 1 ), three( 1024, 3 );
            while( !stop.stop_requested() )
            {
                if( app.darkPixels( false, one.data() ) != 0 || app.darkPixels( false, three.data() ) != 0 )
                    failed = true;
                std::this_thread::yield();
            }
        } );
    for( unsigned n = 0; n < 200; ++n )
    {
        REQUIRE( app.feed( 10 ) == 0 );
        auto pixels = app.output();
        REQUIRE( ( pixels.front() == Approx( 9 ) || pixels.front() == Approx( 7 ) ) );
        REQUIRE_THAT( pixels, Catch::Matchers::Approx( std::vector<float>( 1024, pixels.front() ) ) );
    }
    writer.request_stop();
    writer.join();
    REQUIRE_FALSE( failed.load() );
}

/// Simultaneous monitor reallocations serialize without recursive reconnects or lost requests.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator coalesces concurrent monitor allocations", "[shmimIntegrator][concurrency]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::allocate;
    MagAOX::app::shmimIntegrator::allocateDark;
    MagAOX::app::shmimIntegrator::requestReconfigure;
    #endif
    // clang-format on
    shmimIntegrator_test app( 1 );
    std::atomic<bool>    failed{ false };
    auto                 allocateDark = [&]( bool second )
    {
        for( unsigned n = 0; n < 50; ++n )
        {
            if( app.darkAllocate( second, 2 + n % 2, 2 ) != 0 )
                failed = true;
            std::this_thread::yield();
        }
    };
    std::jthread first( allocateDark, false ), second( allocateDark, true );
    for( unsigned n = 0; n < 50; ++n )
        REQUIRE( app.source( 2 + n % 2, 2 ) == 0 );
    first.join();
    second.join();
    REQUIRE_FALSE( failed.load() );
    REQUIRE( app.m_generation == 151 );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.m_activeGeneration == 151 );
    REQUIRE( app.feed( 12 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 12 ) ) );
}

/// An allocation exception leaves the requested generation unavailable and permits a later retry.
/** \ingroup shmimIntegrator_unit_test */
TEST_CASE( "shmimIntegrator recovers from history allocation failure", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    MagAOX::app::shmimIntegrator::configureAcquisition;
    #endif
    // clang-format on
    shmimIntegrator_test app( 2, 1 );
    REQUIRE( app.source( 3, 2 ) == 0 );
    mx::improc::failIntegratorCubeAllocation = true;
    int result                               = app.configureAcquisition();
    mx::improc::failIntegratorCubeAllocation = false;
    REQUIRE( result < 0 );
    REQUIRE_FALSE( app.m_prepared );
    REQUIRE( app.startAcquisition() < 0 );
    REQUIRE( app.feed( 100 ) == 0 );
    REQUIRE_FALSE( app.m_updated );
    REQUIRE( app.start() == 0 );
    REQUIRE( app.feed( 3 ) == 0 );
    REQUIRE( app.feed( 5 ) == 0 );
    REQUIRE_THAT( app.output(), Catch::Matchers::Approx( std::vector<float>( 6, 4 ) ) );
}

} // namespace shmimIntegratorTest
} // namespace libXWCTest
