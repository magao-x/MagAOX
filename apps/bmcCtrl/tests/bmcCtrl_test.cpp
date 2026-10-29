/** \file bmcCtrl_test.cpp
 * \brief Offline BMC SDK outcome and timing endpoint tests with isolated shared memory.
 * \ingroup bmcCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "sdk/BMCApi.h"

#include <condition_variable>
#include <future>
#include <mutex>

/// \cond TEST_HARNESS
namespace bmcTimingTest
{
std::atomic<int>        result{ 0 };        ///< Scripted SDK return value.
std::atomic<unsigned>   calls{ 0 };         ///< SDK submissions made by commandDM.
std::atomic<int64_t>    firstEntry{ 0 };    ///< SDK entry after the start marker.
std::atomic<int64_t>    lastReturn{ 0 };    ///< SDK completion before the end marker.
std::mutex              mutex;              ///< Protects deterministic SDK barriers and recorded calls.
std::condition_variable changed;            ///< Reports entry, permit-return and SDK exit transitions.
bool                    block{ false };     ///< Set calls wait for permission after copying their input.
bool                    entered{ false };   ///< At least one held set call has entered.
bool                    release{ false };   ///< A test explicitly permitted the held set call to return.
bool                    throws{ false };    ///< The held call throws after permission instead of returning its result.
unsigned                active{ 0 };        ///< SDK handle operations currently executing.
unsigned                maximumActive{ 0 }; ///< Maximum simultaneous handle operations.
std::vector<std::pair<std::string, std::thread::id>> operations; ///< Ordered operation names and actual owner threads.
std::vector<double> values; ///< SDK input observed at actual return, after tests mutate shared source pixels.

/// Record all handle accesses, including control operations, under one deterministic mutex.
class Operation
{
  public:
    /// Record entry into one software-only SDK operation.
    explicit Operation( const char *name /**< [in] Operation identity. */ );

    /// Retire an SDK operation even when a scripted exception unwinds it.
    ~Operation();
};

Operation::Operation( const char *name )
{
    std::lock_guard<std::mutex> lock( mutex );
    ++active;
    maximumActive = std::max( maximumActive, active );
    operations.emplace_back( name, std::this_thread::get_id() );
    changed.notify_all();
}

Operation::~Operation()
{
    std::lock_guard<std::mutex> lock( mutex );
    --active;
    changed.notify_all();
}

/// Reset barriers and results only after the preceding fixture joined its worker.
void reset()
{
    std::lock_guard<std::mutex> lock( mutex );
    result        = 0;
    calls         = 0;
    block         = false;
    entered       = false;
    release       = false;
    throws        = false;
    active        = 0;
    maximumActive = 0;
    operations.clear();
    values.clear();
}

/// Wait for actual SDK entry without using a sleep as ordering evidence.
void waitEntered()
{
    std::unique_lock<std::mutex> lock( mutex );
    REQUIRE( changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               []()
                               {
                                   return entered;
                               } ) );
}

/// Permit the held SDK call to return or throw.
void permitReturn()
{
    std::lock_guard<std::mutex> lock( mutex );
    release = true;
    changed.notify_all();
}
/// Drain held SDK work before asynchronous test futures are destroyed during assertion unwinding.
class ReturnGuard
{
    std::function<void()> m_cleanup; ///< Optional retirement for a manually driven callback.
  public:
    /// Retain optional manual callback cleanup, without affecting production code.
    explicit ReturnGuard( std::function<void()> cleanup = {} /**< [in] Extra unwind cleanup after SDK permission. */ );

    /// Permit SDK return and retire a manual callback before its control future is destroyed.
    ~ReturnGuard();
};

ReturnGuard::ReturnGuard( std::function<void()> cleanup ) : m_cleanup( std::move( cleanup ) )
{
}

ReturnGuard::~ReturnGuard()
{
    permitReturn();

    if( m_cleanup )
    {
        m_cleanup();
    }
}

} // namespace bmcTimingTest

/// Complete software-only initialization on the recorded SDK owner.
BMCRC BMCOpen( DM *dm, const char * )
{
    bmcTimingTest::Operation operation( "open" );
    dm->ActCount = 2;
    return NO_ERR;
}

/// Supply the two test actuator addresses on the recorded SDK owner.
BMCRC BMCLoadMap( DM *, const char *, uint32_t *map )
{
    bmcTimingTest::Operation operation( "map" );
    map[0] = 0;
    map[1] = 1;
    return NO_ERR;
}

/// Hold actual submission at a deterministic barrier and read the immutable vector at return.
BMCRC BMCSetArray( DM *dm, const double *values, const void * )
{
    bmcTimingTest::Operation operation( "set" );
    ++bmcTimingTest::calls;
    bmcTimingTest::firstEntry = MagAOX::app::dev::dmTiming::monotonicNowNs();
    std::unique_lock<std::mutex> lock( bmcTimingTest::mutex );
    bmcTimingTest::entered = true;
    bmcTimingTest::changed.notify_all();
    bmcTimingTest::changed.wait( lock,
                                 []()
                                 {
                                     return !bmcTimingTest::block || bmcTimingTest::release;
                                 } );
    bmcTimingTest::values.assign( values, values + dm->ActCount );
    bmcTimingTest::lastReturn = MagAOX::app::dev::dmTiming::monotonicNowNs();

    if( bmcTimingTest::throws )
    {
        throw std::runtime_error( "offline SDK exception" );
    }

    return bmcTimingTest::result;
}

/// Describe a scripted failure without consulting a handle.
const char *BMCErrorString( BMCRC )
{
    return "offline SDK failure";
}

/// Complete a software-only clear on the recorded SDK owner.
BMCRC BMCClearArray( DM * )
{
    bmcTimingTest::Operation operation( "clear" );
    return NO_ERR;
}

/// Complete software-only teardown on the recorded SDK owner.
BMCRC BMCClose( DM * )
{
    bmcTimingTest::Operation operation( "close" );
    return NO_ERR;
}

/// Return the deterministic software SDK version.
const char *BMCVersionString()
{
    return "offline-worker-test";
}

#include "../bmcCtrl.hpp"

namespace MagAOX
{
namespace app
{
/// Prepare the real command implementation with two software actuators and no application threads.
class bmcCtrl_test : public bmcCtrl
{
    IMAGE                m_shape{};    ///< Owned local output-shape stream.
    IMAGE                m_input{};    ///< Owned local combined-command stream.
    std::array<IMAGE, 4> m_products{}; ///< Owned flat/delta-source and delta/difference output streams.
    std::string          m_directory;  ///< Local FITS calibration directory for lifecycle tests.
  public:
    using dev::dm<bmcCtrl, float>::m_dmTiming;
    using dev::dm<bmcCtrl, float>::m_dmCompletion;
    using dev::dm<bmcCtrl, float>::m_dmReaderRelease;
    using bmcCtrl::checkSdkTimeout;
    using bmcCtrl::m_sdkActiveAgeSec;
    using bmcCtrl::m_sdkConfigured;
    using bmcCtrl::m_sdkCpuset;
    using bmcCtrl::m_sdkTimeouts;
    using bmcCtrl::m_sdkTimeoutSec;
    using bmcCtrl::startSdkWorker;
    using bmcCtrl::verifySdkThread;
    using MagAOXApp<true>::config;
    using dev::dm<bmcCtrl, float>::m_completionOutcome;
    using dev::dm<bmcCtrl, float>::m_completionBackendResult;
    using bmcCtrl::m_admissionClosed;
    using bmcCtrl::m_coalescedInputs;
    using bmcCtrl::m_commandActive;
    using bmcCtrl::m_commandJob;
    using bmcCtrl::m_completedCommands;
    using bmcCtrl::m_rejectedInputs;
    using bmcCtrl::m_sdkFailures;
    using bmcCtrl::m_sdkFault;
    using bmcCtrl::m_sdkThread;
    using bmcCtrl::m_sdkThreadPrio;
    using bmcCtrl::m_sdkWorkerEnabled;
    using bmcCtrl::m_stagedSat;
    using bmcCtrl::stopSdkWorker;

    /// Start the real worker loop without requesting workstation realtime privileges or an INDI server.
    void enableWorker();

    /// Publish a genuine coherent input and ask the real admission hook to reserve its callback slot.
    bool admit( uint64_t counter /**< [in] Current combined input counter. */ );

    /// Prepare metadata and receipt evidence without reserving the real generic callback's admission slot.
    void prepareInput( uint64_t counter /**< [in] Current combined input counter. */ );

    /// Allocate private staged delta products and their two input channels.
    void stagedProducts();

    /// Change both live channel inputs after the published early reader release.
    void mutateChannels();

    /// Return one published delta value from the production output stream.
    float delta( size_t index /**< [in] Image-grid actuator address. */ );

    /// Return one published difference value from the production output stream.
    float difference( size_t index /**< [in] Image-grid actuator address. */ );

    /// Return the pending fixed request's dispatch timestamp under its ownership mutex.
    int64_t sdkDispatchStamp();

    /// Backdate only the pending request's watchdog stamp for deterministic finalization checks.
    void ageSdkRequest( int64_t ageNs /**< [in] Nanoseconds to subtract without changing recorded SDK timing. */ );

    /// Wait deterministically for a control caller to close command admission.
    void waitClosed();

    /// Fill the isolated notification semaphore to force a real generic sem_post overflow.
    void fullSaturationSemaphore();

    /// Return the instantaneous saturation flag from production-owned storage.
    uint8_t saturation( size_t index /**< [in] Image-grid actuator address. */ ) const;

    /// Return the owned combined input pixels for a real generic callback.
    float *pixels();

    /// Create the minimal mapping FITS file needed by the real initialization path.
    void mappingFile();
    /// Allocate a unique local output stream and the production command scratch storage.
    bmcCtrl_test();
    /// Remove only the test-owned stream before the production destructor releases scratch storage.
    ~bmcCtrl_test() noexcept;
};

bmcCtrl_test::bmcCtrl_test()
{
    static const std::string directory = []()
    {
        char        path[] = "/tmp/bmc-timing-test-XXXXXX";
        const char *value  = mkdtemp( path );

        if( !value )
        {
            throw std::runtime_error( "cannot create local BMC stream directory" );
        }

        return std::string( value );
    }();
    setenv( "MILK_SHM_DIR", directory.c_str(), 1 );
    uint32_t sizes[]{ 2, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_shape, "bmc_timing_shape", 2, sizes, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) == 0 );
    m_outputShape.open( "bmc_timing_shape" );
    m_imageStream         = {};
    m_nbAct               = 2;
    m_act_gain            = 1;
    m_volume_factor       = 1;
    m_actuator_mapping    = static_cast<int *>( calloc( 2, sizeof( int ) ) );
    m_actuator_mapping[1] = 1;
    m_dminputs            = static_cast<double *>( calloc( 2, sizeof( double ) ) );
    m_instSatMap.resize( 2, 1 );
    m_instSatMap.setZero();
    m_dmTiming.configure( 2.0, 32 );
    m_dmWidth   = 2;
    m_dmHeight  = 1;
    m_directory = directory;
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_input, "bmc_timing_input", 2, sizes, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) == 0 );
    REQUIRE( ImageStreamIO_openIm( &m_imageStream, "bmc_timing_input" ) == 0 );
    REQUIRE( sem_init( &m_satSemaphore, 0, 0 ) == 0 );
    bmcTimingTest::reset();
}

bmcCtrl_test::~bmcCtrl_test() noexcept
{
    bmcTimingTest::permitReturn();
    retireCommand();
    stopSdkWorker();
    m_dmReaderRelease.close();
    m_dmCompletion.close();
    m_outputDelta.close();
    m_outputDiff.close();

    for( auto *channel : m_channels )
    {
        delete channel;
    }

    m_channels.clear();

    for( auto &image : m_products )
    {
        if( image.md != nullptr )
        {
            ImageStreamIO_destroyIm( &image );
        }
    }

    sem_destroy( &m_satSemaphore );
    ImageStreamIO_closeIm( &m_imageStream );
    ImageStreamIO_destroyIm( &m_input );
    m_outputShape.close();
    ImageStreamIO_destroyIm( &m_shape );
}
void bmcCtrl_test::enableWorker()
{
    m_sdkWorkerEnabled = true;
    m_sdkTimeoutSec    = 3600; // Test barriers must not acquire a machine-speed-dependent deadline.
    m_sdkThreadInit    = false;
    m_sdkConfigured    = true;
    m_sdkStop          = false;
    m_sdkThread        = std::thread( sdkThreadStart, this );
    m_dmopen           = true;
    m_sdkValues.assign( m_nbAct, 0 );
    m_stagedSat.assign( 2, 0 );
    m_admissionClosed = false;
}

void bmcCtrl_test::prepareInput( uint64_t counter )
{
    m_imageStream.md->cnt0  = counter;
    m_imageStream.md->write = 0;
    clock_gettime( CLOCK_REALTIME, &m_imageStream.md->writetime );
    m_completionReceipt = true;
}

bool bmcCtrl_test::admit( uint64_t counter )
{
    prepareInput( counter );
    return admitCommand();
}

void bmcCtrl_test::stagedProducts()
{
    uint32_t    sizes[]{ 2, 1 };
    const char *names[]{ "bmc_test_flat", "bmc_test_delta", "bmc_test_delta_out", "bmc_test_diff_out" };

    for( size_t index = 0; index < m_products.size(); ++index )
    {
        REQUIRE( ImageStreamIO_createIm_gpu(
                     &m_products[index], names[index], 2, sizes, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) == 0 );
    }

    for( size_t index = 0; index < 2; ++index )
    {
        auto *channel = new mx::improc::milkImage<float>;
        channel->open( names[index] );
        m_channels.push_back( channel );
    }

    m_products[0].array.F[0] = 0.1;
    m_products[0].array.F[1] = 0.2;
    m_products[1].array.F[0] = 0.15;
    m_products[1].array.F[1] = 1.8;
    m_outputDelta.open( names[2] );
    m_outputDiff.open( names[3] );
    m_deltaChannels = { names[1] };
    m_notDeltas     = { 0 };
    m_deltas        = { 1 };
    m_totalFlat.resize( 2, 1 );
    m_totalDelta.resize( 2, 1 );
    m_stagedDelta.resize( 2, 1 );
    m_stagedDiff.resize( 2, 1 );
}

void bmcCtrl_test::mutateChannels()
{
    for( size_t index = 0; index < 2; ++index )
    {
        m_products[index].array.F[0] = 42;
        m_products[index].array.F[1] = 42;
    }
}

float bmcCtrl_test::delta( size_t index )
{
    return m_outputDelta[index];
}

float bmcCtrl_test::difference( size_t index )
{
    return m_outputDiff[index];
}

int64_t bmcCtrl_test::sdkDispatchStamp()
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    return m_sdkRequest.dispatchNs;
}

void bmcCtrl_test::ageSdkRequest( int64_t ageNs )
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    m_sdkRequest.dispatchNs -= ageNs;
}

void bmcCtrl_test::waitClosed()
{
    std::unique_lock<std::mutex> lock( m_commandMutex );
    REQUIRE( m_commandRetired.wait_for( lock,
                                        std::chrono::seconds( 3 ),
                                        [this]()
                                        {
                                            return m_admissionClosed;
                                        } ) );
}

void bmcCtrl_test::fullSaturationSemaphore()
{
    REQUIRE( sem_destroy( &m_satSemaphore ) == 0 );
    REQUIRE( sem_init( &m_satSemaphore, 0, SEM_VALUE_MAX ) == 0 );
}

uint8_t bmcCtrl_test::saturation( size_t index ) const
{
    return m_instSatMap.data()[index];
}

float *bmcCtrl_test::pixels()
{
    return m_imageStream.array.F;
}

void bmcCtrl_test::mappingFile()
{
    m_calibPath              = m_directory;
    fitsfile         *file   = nullptr;
    int               status = 0;
    long              sizes[]{ 2, 1 };
    long              first[]{ 1, 1 };
    int               addresses[]{ 1, 2 };
    const std::string name = "!" + m_directory + "/bmc_2k_actuator_mapping.fits";
    REQUIRE( fits_create_file( &file, name.c_str(), &status ) == 0 );
    REQUIRE( fits_create_img( file, LONG_IMG, 2, sizes, &status ) == 0 );
    REQUIRE( fits_write_pix( file, TINT, first, 2, addresses, &status ) == 0 );
    REQUIRE( fits_close_file( file, &status ) == 0 );
}

} // namespace app
} // namespace MagAOX

/// \endcond

namespace libXWCTest
{
namespace bmcCtrlTest
{
/** \defgroup bmcCtrl_unit_test bmcCtrl Unit Tests
 * \brief Offline tests of BMC command integration.
 * \ingroup application_unit_test
 */

/// Preserve SDK return behavior and qualify submission before the later saturation-map epilogue.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "bmcCtrl records actual SDK outcomes and phase endpoints", "[bmcCtrl][timing]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::commandDM(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;

    bmcCtrl_test app;
    bmcTimingTest::result = GENERATE( 0, 7 );
    float command[]{ 0.25f, 2.0f };

    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == ( bmcTimingTest::result.load() ? -1 : 0 ) );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();

    const auto &summary = app.dmTimingSummary();

    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( summary.total.submitted == ( bmcTimingTest::result.load() ? 0 : 1 ) );
    REQUIRE( summary.total.failed == ( bmcTimingTest::result.load() ? 1 : 0 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::actuator )].count == 1 );
    REQUIRE( summary.metrics[metricIndex( Metric::saturation )].count == ( bmcTimingTest::result.load() ? 0 : 1 ) );

    const auto &sdk = summary.metrics[metricIndex( Metric::sdk )];

    REQUIRE( sdk.count == 1 );
    REQUIRE( sdk.mean >= ( bmcTimingTest::lastReturn.load() - bmcTimingTest::firstEntry.load() ) * 1e-9 );
}

/// Worker mode freezes its vector and stages saturation before a held SDK call returns.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC worker overlaps private staging and joins actual SDK result", "[bmcCtrl][worker]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::commandDM(nullptr);
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    MagAOX::app::bmcCtrl::retireCommand();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;
    bmcCtrl_test app;
    app.enableWorker();
    bmcTimingTest::block = true;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == 0 );
    bmcTimingTest::waitEntered();
    REQUIRE( app.m_commandJob != 0 );
    REQUIRE( app.m_commandActive );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE( app.m_stagedSat[1] == 1 );
    command[0] = 0;
    command[1] = 0;

    bmcTimingTest::permitReturn();
    REQUIRE( app.finalizeCommand( 0 ) == 0 );
    REQUIRE( app.m_commandActive );
    REQUIRE( app.m_commandJob == 0 );
    REQUIRE( app.saturation( 1 ) == 1 );
    REQUIRE( bmcTimingTest::values == std::vector<double>{ 0.5, 1.0 } );
    REQUIRE( app.m_completionOutcome == Outcome::success );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 1 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
    REQUIRE( app.dmTimingSummary().metrics[metricIndex( Metric::sdk )].count == 1 );
    app.retireCommand();
    REQUIRE_FALSE( app.m_commandActive );
    REQUIRE( app.admit( 2 ) );
    app.retireCommand();
}

/// Backend outcomes remain independent of a failed monitor-side staging operation.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC worker drains SDK failure exception and staging errors", "[bmcCtrl][worker]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;
    bmcCtrl_test app;
    app.enableWorker();
    const int sdkResult        = GENERATE( 0, 7, -1 );
    const int processingResult = GENERATE( 0, -1 );
    bmcTimingTest::result      = sdkResult;
    bmcTimingTest::throws      = sdkResult == -1;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == 0 );
    REQUIRE( app.finalizeCommand( processingResult ) == ( sdkResult || processingResult ? -1 : 0 ) );
    REQUIRE( app.m_commandJob == 0 );
    REQUIRE( app.saturation( 1 ) == ( sdkResult || processingResult ? 0 : 1 ) );
    REQUIRE( app.m_sdkFault == bool( sdkResult || processingResult ) );
    REQUIRE( app.m_completionOutcome == ( sdkResult ? Outcome::failed : Outcome::success ) );
    REQUIRE( app.m_completionBackendResult ==
             ( sdkResult == -1 ? MagAOX::dmCompletion::unavailableResult : sdkResult ) );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == ( sdkResult ? 0 : 1 ) );
    REQUIRE( app.dmTimingSummary().total.failed == ( sdkResult ? 1 : 0 ) );
    REQUIRE( app.m_sdkFailures == ( sdkResult ? 1 : 0 ) );
    app.retireCommand();
    REQUIRE( app.admit( 2 ) == !bool( sdkResult || processingResult ) );
    app.retireCommand();
}

/// Control operations drain the entire callback, then execute on the same sole SDK owner.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC control waits for callback retirement after SDK return", "[bmcCtrl][worker][lifecycle]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::zeroDM();
    MagAOX::app::bmcCtrl::releaseDM();
    MagAOX::app::bmcCtrl::initDM();
    MagAOX::app::bmcCtrl::retireCommand();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enableWorker();
    const auto owner     = app.m_sdkThread.get_id();
    bmcTimingTest::block = true;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    REQUIRE( app.commandDM( command ) == 0 );
    bmcTimingTest::waitEntered();
    const bool releasing = GENERATE( false, true );
    auto       zero      = std::async( std::launch::async,
                            [&app, releasing]()
                            {
                                return releasing ? app.releaseDM() : app.zeroDM();
                            } );

    bmcTimingTest::ReturnGuard guard(
        [&app]()
        {
            if( app.m_commandActive )
            {
                app.finalizeCommand( -1 );
                app.retireCommand();
            }
        } );
    app.waitClosed();
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE_FALSE( app.admit( 2 ) );
    bmcTimingTest::permitReturn();
    REQUIRE( app.finalizeCommand( 0 ) == 0 );
    REQUIRE( zero.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    REQUIRE( bmcTimingTest::calls == 1 );
    app.retireCommand();
    REQUIRE( zero.get() == 0 );
    REQUIRE( bmcTimingTest::calls == 2 );
    REQUIRE( app.m_admissionClosed == releasing );
    REQUIRE( app.releaseDM() == 0 );
    app.mappingFile();
    REQUIRE( app.initDM() == 0 );
    REQUIRE( app.releaseDM() == 0 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
    REQUIRE( bmcTimingTest::operations.size() >= 10 );

    for( const auto &operation : bmcTimingTest::operations )
    {
        REQUIRE( operation.second == owner );
    }

    app.stopSdkWorker();
    REQUIRE_FALSE( app.m_sdkThread.joinable() );
}

/// Queued semaphore notices cannot replay duplicate commands or build an SDK backlog.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC worker rejects duplicate posts and coalesces counter advances", "[bmcCtrl][worker]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::admitCommand();
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    REQUIRE_FALSE( app.m_sdkWorkerEnabled );
    REQUIRE( app.admitCommand() );
    app.enableWorker();
    REQUIRE( app.admit( 10 ) );
    REQUIRE_FALSE( app.admit( 11 ) );
    app.retireCommand();
    REQUIRE_FALSE( app.admit( 10 ) );
    REQUIRE_FALSE( app.admit( 9 ) );
    REQUIRE( app.admit( 14 ) );
    REQUIRE( app.m_coalescedInputs == 0 );
    app.retireCommand();
    REQUIRE_FALSE( app.admit( 14 ) );
    REQUIRE( app.m_rejectedInputs == 4 );
    REQUIRE( app.admit( 18 ) );
    REQUIRE( app.m_coalescedInputs == 3 );
    app.retireCommand();
    REQUIRE_FALSE( app.admit( 0 ) );
    REQUIRE( app.admit( 1 ) );
    app.retireCommand();
    REQUIRE( bmcTimingTest::calls == 0 );
}

/// Real generic processing releases frozen channel reads before SDK return and publishes only afterward.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC generic callback publishes early release and frozen staged products", "[bmcCtrl][worker][completion]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<MagAOX::app::bmcCtrl,float>::processImage(nullptr,MagAOX::app::dev::shmimT());
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enableWorker();
    const bool timeout = GENERATE( false, true );
    app.stagedProducts();
    REQUIRE( app.verifySdkThread() == 0 );
    REQUIRE( app.m_dmCompletion.open( "bmc_test_full_done" ) == 0 );
    REQUIRE( app.m_dmReaderRelease.open( "bmc_test_read_done", Kind::readers ) == 0 );
    IMAGE observer{};
    REQUIRE( ImageStreamIO_openIm( &observer, "bmc_test_read_done" ) == 0 );

    while( sem_trywait( observer.semptr[0] ) == 0 )
    {
    }

    app.pixels()[0] = 0.25;
    app.pixels()[1] = 2;
    app.prepareInput( 1 );
    app.m_dmTiming.beginEvent( true );
    bmcTimingTest::block = true;
    auto callback        = std::async( std::launch::async,
                                [&app]()
                                {
                                    return app.processImage( app.pixels(), dev::shmimT() );
                                } );

    bmcTimingTest::ReturnGuard guard;
    bmcTimingTest::waitEntered();
    timespec deadline{};
    clock_gettime( CLOCK_REALTIME, &deadline );
    deadline.tv_sec += 3;
    REQUIRE( sem_timedwait( observer.semptr[0], &deadline ) == 0 );
    Record release;
    REQUIRE( dev::dmCompletion::read( &observer, release ) );
    REQUIRE( readersQualified( release ) );
    REQUIRE( release.inputCounter == 1 );
    REQUIRE( release.backendResult == unavailableResult );
    REQUIRE( release.outcome == MagAOX::dmTiming::Outcome::unknown );
    REQUIRE( callback.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    REQUIRE( app.delta( 0 ) == 0 );
    REQUIRE( app.saturation( 1 ) == 0 );
    if( timeout )
    {
        app.m_sdkTimeoutSec = 1;
        app.checkSdkTimeout( app.sdkDispatchStamp() + 2000000000LL );
        REQUIRE( app.m_sdkTimeouts == 1 );
    }

    app.mutateChannels();
    app.pixels()[0] = 99;
    app.pixels()[1] = 99;
    app.prepareInput( 2 );
    bmcTimingTest::permitReturn();
    REQUIRE( callback.get() == ( timeout ? -1 : 0 ) );
    REQUIRE( app.delta( 0 ) == Approx( timeout ? 0.0 : 0.15 ) );
    REQUIRE( app.delta( 1 ) == Approx( timeout ? 0.0 : 0.8 ) );
    REQUIRE( app.difference( 0 ) == Approx( 0.0 ).margin( 1e-6 ) );
    REQUIRE( app.difference( 1 ) == Approx( timeout ? 0.0 : 1.0 ) );
    REQUIRE( app.saturation( 1 ) == ( timeout ? 0 : 1 ) );
    REQUIRE( bmcTimingTest::values == std::vector<double>{ 0.5, 1.0 } );
    REQUIRE_FALSE( app.m_commandActive );
    const auto finalRelease = app.m_dmReaderRelease.snapshot();
    REQUIRE( readersQualified( finalRelease ) );
    REQUIRE( finalRelease.sequence == release.sequence );
    REQUIRE( finalRelease.finishedNs == release.finishedNs );
    REQUIRE( finalRelease.backendResult == 0 );
    REQUIRE( finalRelease.outcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( finalRelease.inputCounter == 1 );
    REQUIRE( app.m_dmCompletion.snapshot().state == ( timeout ? State::error : State::unqualified ) );
    REQUIRE( app.m_sdkFault == timeout );
    REQUIRE( finalRelease.finishedNs < bmcTimingTest::lastReturn.load() );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 1 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
    REQUIRE( ImageStreamIO_closeIm( &observer ) == 0 );
}

/// Actual placement mismatch and failed cpuset initialization stop/join the unconfigured worker.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC SDK startup verifies placement and rolls back failures", "[bmcCtrl][worker][lifecycle]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::startSdkWorker();
    MagAOX::app::bmcCtrl::verifySdkThread();
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;

    SECTION( "cpuset attachment failure drains an unconfigured thread" )
    {
        app.m_sdkWorkerEnabled = true;
        app.m_sdkCpuset        = "bmc-offline-missing-cpuset-" + std::to_string( getpid() );
        REQUIRE( app.startSdkWorker() < 0 );
        REQUIRE_FALSE( app.m_sdkThread.joinable() );
        REQUIRE_FALSE( app.m_sdkConfigured );
    }

    SECTION( "requested priority must match actual runtime priority" )
    {
        app.enableWorker();
        sched_param actual{};
        int         policy = -1;
        REQUIRE( pthread_getschedparam( app.m_sdkThread.native_handle(), &policy, &actual ) == 0 );
        app.m_sdkThreadPrio = actual.sched_priority == 0 ? 1 : 0;
        REQUIRE( app.verifySdkThread() < 0 );
        app.stopSdkWorker();
        REQUIRE_FALSE( app.m_sdkThread.joinable() );
    }

    SECTION( "actual allowed CPUs must match the configured cpuset" )
    {
        app.enableWorker();
        app.m_sdkCpuset = "bmc-offline-missing-cpuset-" + std::to_string( getpid() );
        REQUIRE( app.verifySdkThread() < 0 );
        app.stopSdkWorker();
        REQUIRE_FALSE( app.m_sdkThread.joinable() );
    }

    REQUIRE( bmcTimingTest::operations.empty() );
}

/// Startup configuration is opt-in and invalid scheduling priorities are rejected before startup.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC worker configuration defaults off and validates priority", "[bmcCtrl][worker][config]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::setupConfig();
    MagAOX::app::bmcCtrl::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE_FALSE( app.m_sdkWorkerEnabled );
    REQUIRE( app.m_sdkThreadPrio == 0 );
    REQUIRE( app.m_sdkTimeoutSec == 1 );
    const int priority   = GENERATE( -1, 0, 55, 99, 100 );
    char      path[]     = "/tmp/bmc-worker-config-XXXXXX";
    int       descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    close( descriptor );

    { // File lifetime scope.
        std::ofstream file( path );
        file << "[sdk]\nworkerEnabled=true\nthreadPrio=" << priority << "\ncpuset=spare56\n";
    }

    app.config.readConfig( path );
    const bool valid = priority >= 0 && priority <= 99;
    REQUIRE( app.loadConfigImpl( app.config ) == ( valid ? 0 : -1 ) );
    REQUIRE( app.m_sdkWorkerEnabled );
    REQUIRE( app.m_sdkCpuset == "spare56" );
    REQUIRE_FALSE( app.m_sdkThread.joinable() );
    REQUIRE( unlink( path ) == 0 );
}

/// A late callback error closes admission while retaining the already successful actual SDK result.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC callback post failure retains SDK success and faults admission", "[bmcCtrl][worker][completion]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<MagAOX::app::bmcCtrl,float>::processImage(nullptr,MagAOX::app::dev::shmimT());
    MagAOX::app::bmcCtrl::retireCommand();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enableWorker();
    app.fullSaturationSemaphore();
    REQUIRE( app.m_dmReaderRelease.open( "bmc_test_late_read_done", Kind::readers ) == 0 );
    app.pixels()[0] = 0.25;
    app.pixels()[1] = 2;
    app.prepareInput( 1 );
    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.processImage( app.pixels(), dev::shmimT() ) < 0 );
    const auto record = app.m_dmReaderRelease.snapshot();
    REQUIRE( readersQualified( record ) );
    REQUIRE( record.outcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( record.backendResult == 0 );
    REQUIRE( app.m_completionOutcome == MagAOX::dmTiming::Outcome::failed );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_admissionClosed );
    REQUIRE_FALSE( app.m_commandActive );
    REQUIRE( app.m_sdkFailures == 0 );
    REQUIRE_FALSE( app.admit( 2 ) );
    REQUIRE( bmcTimingTest::calls == 1 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 1 );
    REQUIRE( app.dmTimingSummary().total.failed == 1 );
}

/// Early read release remains valid when later SDK return fails, with an exact native code or no-return sentinel.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC generic release retains native SDK failures and exceptions", "[bmcCtrl][worker][completion]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::dev::dm<MagAOX::app::bmcCtrl,float>::processImage(nullptr,MagAOX::app::dev::shmimT());
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enableWorker();
    const bool throwing   = GENERATE( false, true );
    bmcTimingTest::result = 7;
    bmcTimingTest::throws = throwing;
    REQUIRE( app.m_dmReaderRelease.open( "bmc_test_failed_read_done", Kind::readers ) == 0 );
    app.pixels()[0] = 0.25;
    app.pixels()[1] = 2;
    app.prepareInput( 1 );
    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.processImage( app.pixels(), dev::shmimT() ) < 0 );
    const auto record = app.m_dmReaderRelease.snapshot();
    REQUIRE( readersQualified( record ) );
    REQUIRE( record.outcome == MagAOX::dmTiming::Outcome::failed );
    REQUIRE( record.backendResult == ( throwing ? unavailableResult : 7 ) );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_admissionClosed );
    REQUIRE_FALSE( app.m_commandActive );
    REQUIRE( app.m_sdkFailures == 1 );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE_FALSE( app.admit( 2 ) );
    REQUIRE( bmcTimingTest::calls == 1 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 0 );
    REQUIRE( app.dmTimingSummary().total.failed == 1 );
}

/// A polled deadline latches once while the SDK and immutable request remain in flight until actual return.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC diagnostic timeout never cancels or releases the SDK job", "[bmcCtrl][worker][timeout]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::checkSdkTimeout(0);
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    MagAOX::app::bmcCtrl::retireCommand();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enableWorker();
    app.m_sdkTimeoutSec  = 1;
    bmcTimingTest::block = true;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    REQUIRE( app.commandDM( command ) == 0 );
    bmcTimingTest::waitEntered();
    const uint64_t job = app.m_commandJob;
    const int64_t  now = app.sdkDispatchStamp() + 2000000000LL;
    app.checkSdkTimeout( now );
    app.checkSdkTimeout( now + 1 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( app.m_sdkActiveAgeSec >= 2 );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_admissionClosed );
    REQUIRE( app.m_commandJob == job );
    REQUIRE( app.m_commandActive );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE_FALSE( app.admit( 2 ) );
    auto                       zero = std::async( std::launch::async,
                            [&app]()
                            {
                                return app.zeroDM();
                            } );
    bmcTimingTest::ReturnGuard guard(
        [&app]()
        {
            if( app.m_commandActive )
            {
                app.finalizeCommand( -1 );
                app.retireCommand();
            }
        } );
    bmcTimingTest::permitReturn();
    REQUIRE( app.finalizeCommand( 0 ) < 0 );
    REQUIRE( app.m_completionOutcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( app.m_completionBackendResult == 0 );
    REQUIRE( app.m_sdkFailures == 0 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( app.m_sdkActiveAgeSec == 0 );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE( bmcTimingTest::values == std::vector<double>{ 0.5, 1.0 } );
    REQUIRE( zero.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    REQUIRE( bmcTimingTest::calls == 1 );
    app.retireCommand();
    REQUIRE( zero.get() == 0 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( bmcTimingTest::calls == 2 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
    REQUIRE_FALSE( app.admit( 3 ) );
}

/// A slow return is diagnosed even when appLogic did not run while the SDK was occupied.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC finalization detects deadline violations without ordinary polling", "[bmcCtrl][worker][timeout]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enableWorker();
    app.m_sdkTimeoutSec  = 1;
    bmcTimingTest::block = true;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    REQUIRE( app.commandDM( command ) == 0 );
    bmcTimingTest::waitEntered();
    app.ageSdkRequest( 2000000000LL );
    REQUIRE( app.m_sdkTimeouts == 0 );
    bmcTimingTest::permitReturn();
    REQUIRE( app.finalizeCommand( 0 ) < 0 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_completionOutcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( app.m_completionBackendResult == 0 );
    REQUIRE( app.m_sdkFailures == 0 );
    REQUIRE( app.saturation( 1 ) == 0 );
    app.retireCommand();
    REQUIRE_FALSE( app.admit( 2 ) );
}

/// Initialization/zero/clear/close requests are not mistaken for a realtime command deadline.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC diagnostic watchdog exempts serialized control operations", "[bmcCtrl][worker][timeout]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::checkSdkTimeout(0);
    MagAOX::app::bmcCtrl::zeroDM();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enableWorker();
    app.m_sdkTimeoutSec             = 1;
    bmcTimingTest::block            = true;
    auto                       zero = std::async( std::launch::async,
                            [&app]()
                            {
                                return app.zeroDM();
                            } );
    bmcTimingTest::ReturnGuard guard;
    bmcTimingTest::waitEntered();
    app.checkSdkTimeout( INT64_MAX );
    REQUIRE( app.m_sdkTimeouts == 0 );
    REQUIRE( app.m_sdkActiveAgeSec == 0 );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE( zero.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    bmcTimingTest::permitReturn();
    REQUIRE( zero.get() == 0 );
    REQUIRE_FALSE( app.m_admissionClosed );
    REQUIRE( app.m_sdkTimeouts == 0 );
}

/// Timeout validation remains effective in production builds using fast floating-point optimizations.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC SDK diagnostic timeout requires a finite positive value", "[bmcCtrl][worker][config]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.setupConfig();
    const auto candidate = GENERATE( std::pair<double, bool>{ 1, true },
                                     std::pair<double, bool>{ 0.001, true },
                                     std::pair<double, bool>{ 0, false },
                                     std::pair<double, bool>{ -1, false },
                                     std::pair<double, bool>{ std::numeric_limits<double>::infinity(), false },
                                     std::pair<double, bool>{ std::numeric_limits<double>::quiet_NaN(), false } );
    app.m_sdkTimeoutSec  = candidate.first;
    REQUIRE( app.loadConfigImpl( app.config ) == ( candidate.second ? 0 : -1 ) );
}

/// A timed-out command's flag cannot poison control operations or a new handle generation.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC timeout recovers only after drain release and successful reinitialization",
           "[bmcCtrl][worker][timeout][lifecycle]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::checkSdkTimeout(0);
    MagAOX::app::bmcCtrl::releaseDM();
    MagAOX::app::bmcCtrl::initDM();
    MagAOX::app::bmcCtrl::finalizeCommand(0);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enableWorker();
    app.m_sdkTimeoutSec  = 1;
    bmcTimingTest::block = true;
    float command[]{ 0.25f, 2.0f };
    REQUIRE( app.admit( 1 ) );
    REQUIRE( app.commandDM( command ) == 0 );
    bmcTimingTest::waitEntered();
    app.checkSdkTimeout( app.sdkDispatchStamp() + 2000000000LL );
    bmcTimingTest::permitReturn();
    REQUIRE( app.finalizeCommand( 0 ) < 0 );
    app.retireCommand();
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE_FALSE( app.admit( 2 ) );
    REQUIRE( app.releaseDM() == 0 );
    app.mappingFile();
    REQUIRE( app.initDM() == 0 );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE_FALSE( app.m_admissionClosed );
    REQUIRE( app.admit( 3 ) );
    REQUIRE( app.commandDM( command ) == 0 );
    REQUIRE( app.finalizeCommand( 0 ) == 0 );
    REQUIRE( app.saturation( 1 ) == 1 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( app.m_sdkFailures == 0 );
    app.retireCommand();
    REQUIRE( app.releaseDM() == 0 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
}

} // namespace bmcCtrlTest
} // namespace libXWCTest
