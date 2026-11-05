/** \file bmcCtrl_test.cpp
 * \brief Offline BMC SDK outcome and timing endpoint tests with isolated shared memory.
 * \ingroup bmcCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "../../../INDI/libcommon/IndiXmlParser.hpp"
#include "sdk/BMCApi.h"

#include <condition_variable>
#include <future>
#include <mutex>

/// \cond TEST_HARNESS
namespace bmcTimingTest
{
std::atomic<int>        result{ 0 };      ///< Scripted SDK return value.
std::atomic<unsigned>   calls{ 0 };       ///< SDK submissions made by commandDM.
std::atomic<int64_t>    firstEntry{ 0 };  ///< SDK entry after the start marker.
std::atomic<int64_t>    lastReturn{ 0 };  ///< SDK completion before the end marker.
std::mutex              mutex;            ///< Protects deterministic SDK barriers and recorded calls.
std::condition_variable changed;          ///< Reports entry, permit-return and SDK exit transitions.
bool                    block{ false };   ///< Set calls wait for permission after copying their input.
bool                    entered{ false }; ///< At least one held set call has entered.
bool                    release{ false }; ///< A test explicitly permitted the held set call to return.
bool                    perCall{ false }; ///< Permits selected SDK returns independently.
unsigned                permits{ 0 };     ///< Number of native calls explicitly permitted to return.
uint64_t heldVector{ 0 };         ///< Monitor job held after its last combined-pixel read, before source validation.
bool     vectorEntered{ false };  ///< Selected vector conversion reached its validation barrier.
bool     captureHold{ false };    ///< Select source-capture rather than post-conversion barrier.
bool     vectorReleased{ false }; ///< Test permitted source validation and possible SDK dispatch.
uint64_t heldPrep{ 0 };           ///< Monitor job to hold before channel staging.
bool     prepEntered{ false };    ///< Selected monitor job reached its barrier.
bool     prepReleased{ false };   ///< Test permitted monitor channel staging.
uint64_t heldPost{ 0 };           ///< Ordered post job to hold before final publication.
bool     postEntered{ false };    ///< Held post job reached its deterministic barrier.
bool     postReleased{ false };   ///< Test permitted the held post job to finish.
std::vector<std::vector<double>> history; ///< Immutable SDK vectors read at each native return.
bool     throws{ false };                 ///< The held call throws after permission instead of returning its result.
unsigned active{ 0 };                     ///< SDK handle operations currently executing.
unsigned maximumActive{ 0 };              ///< Maximum simultaneous handle operations.
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
    result     = 0;
    calls      = 0;
    firstEntry = lastReturn = 0;
    block                   = false;
    entered                 = false;
    release                 = false;
    throws                  = false;
    perCall                 = false;
    permits                 = 0;
    heldPost                = 0;
    heldPrep                = 0;
    heldVector              = 0;
    captureHold             = false;
    vectorEntered = vectorReleased = false;
    prepEntered = prepReleased = false;
    postEntered = postReleased = false;
    history.clear();
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
    release        = true;
    postReleased   = true;
    prepReleased   = true;
    vectorReleased = true;
    changed.notify_all();
}
/// Permit only the first selected native calls to retire.
void permitCalls( unsigned count /**< [in] Highest native call number permitted to return. */ )
{
    std::lock_guard<std::mutex> lock( mutex );
    permits = count;
    changed.notify_all();
}

/// Wait for a selected native submission to enter without scheduler-delay assertions.
void waitCalls( unsigned count /**< [in] Required native entry count. */ )
{
    std::unique_lock<std::mutex> lock( mutex );
    REQUIRE( changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               [count]()
                               {
                                   return calls >= count;
                               } ) );
}

/// Block one ordered post job independently of the SDK owner.
void beforePost( uint64_t order /**< [in] Private command admission order. */ )
{
    std::unique_lock<std::mutex> lock( mutex );

    if( order == heldPost )
    {
        postEntered = true;
        changed.notify_all();
        changed.wait( lock,
                      []()
                      {
                          return postReleased;
                      } );
    }
}

/// Hold a selected monitor job after vector dispatch but before channel staging.
void beforePrep( uint64_t order /**< [in] Private command admission order. */ )
{
    std::unique_lock<std::mutex> lock( mutex );

    if( order == heldPrep )
    {
        prepEntered = true;
        changed.notify_all();
        changed.wait( lock,
                      []()
                      {
                          return prepReleased;
                      } );
    }
}

/// Hold a selected private vector before its combined-source validation.
void beforeVector( uint64_t order /**< [in] Private command admission order. */,
                   bool     capture = false /**< [in] True selects capture before any source metadata is retained. */ )
{
    std::unique_lock<std::mutex> lock( mutex );

    if( order == heldVector && capture == captureHold )
    {
        vectorEntered = true;
        changed.notify_all();
        changed.wait( lock,
                      []()
                      {
                          return vectorReleased;
                      } );
    }
}

/// Wait for the monitor to finish its combined-pixel reads before changing source metadata.
void waitVector()
{
    std::unique_lock<std::mutex> lock( mutex );
    REQUIRE( changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               []()
                               {
                                   return vectorEntered;
                               } ) );
}

/// Wait for the monitor to reach the preparation barrier.
void waitPrep()
{
    std::unique_lock<std::mutex> lock( mutex );
    REQUIRE( changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               []()
                               {
                                   return prepEntered;
                               } ) );
}

/// Wait until the selected post job owns its immutable payload.
void waitPost()
{
    std::unique_lock<std::mutex> lock( mutex );
    REQUIRE( changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               []()
                               {
                                   return postEntered;
                               } ) );
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
    const auto               call = ++bmcTimingTest::calls;
    bmcTimingTest::firstEntry     = MagAOX::app::dev::dmTiming::monotonicNowNs();
    std::unique_lock<std::mutex> lock( bmcTimingTest::mutex );
    bmcTimingTest::entered = true;
    bmcTimingTest::changed.notify_all();
    bmcTimingTest::changed.wait( lock,
                                 [call]()
                                 {
                                     return !bmcTimingTest::block || bmcTimingTest::release ||
                                            ( bmcTimingTest::perCall && call <= bmcTimingTest::permits );
                                 } );
    bmcTimingTest::values.assign( values, values + dm->ActCount );
    bmcTimingTest::history.push_back( bmcTimingTest::values );
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

#define BMCCTRL_TEST_PIPELINE_CAPTURE( order ) bmcTimingTest::beforeVector( order, true )
#define BMCCTRL_TEST_PIPELINE_VECTOR( order ) bmcTimingTest::beforeVector( order )
#define BMCCTRL_TEST_PIPELINE_PREP( order ) bmcTimingTest::beforePrep( order )
#define BMCCTRL_TEST_PIPELINE_POST( order ) bmcTimingTest::beforePost( order )
#include "../bmcCtrl.hpp"
#undef BMCCTRL_TEST_PIPELINE_POST
#undef BMCCTRL_TEST_PIPELINE_PREP
#undef BMCCTRL_TEST_PIPELINE_VECTOR
#undef BMCCTRL_TEST_PIPELINE_CAPTURE

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
    using bmcCtrl::m_indiP_pipeline;
    using bmcCtrl::m_indiP_sdkWorker;
    using bmcCtrl::m_nsat;
    using bmcCtrl::setupSdkDiagnostics;
    using MagAOXApp<true>::m_configName;
    using bmcCtrl::m_pipelineCpuset;
    using bmcCtrl::m_pipelineEnabled;
    using bmcCtrl::m_pipelineResult;
    using bmcCtrl::m_pipelineResultStream;
    using bmcCtrl::m_pipelineThreadPrio;
    using bmcCtrl::m_rejectedInputs;
    using bmcCtrl::m_sdkFailures;
    using bmcCtrl::m_sdkFault;
    using bmcCtrl::m_sdkThread;
    using bmcCtrl::m_sdkThreadPrio;
    using bmcCtrl::m_sdkWorkerEnabled;
    using bmcCtrl::m_stagedSat;
    using bmcCtrl::stopSdkWorker;

    /// Simulate initial hardware readiness before the monitor has created its output images.
    void deferMonitorAllocation();

    /// Start real pipeline owners with ordinary local scheduling and isolated result publishers.
    void enablePipeline();

    /// Drive one genuine monitor receipt and its private preparation synchronously.
    int pipelineInput( uint64_t counter /**< [in] Current combined command counter. */ );

    /// Wait for all stages of the selected admitted count to retire.
    void waitRetired( uint64_t count /**< [in] Required whole-job retirement count. */ );

    /// Return admitted, retired, overloaded, cancelled, superseded and post-failure counters under ownership lock.
    std::array<uint64_t, 6> pipelineCounts();

    /// Return vector rejection, reader qualification loss and actual preparation failure counts.
    std::array<uint64_t, 3> pipelineQuality();

    /// Snapshot a private job only while a deterministic barrier excludes monitor and post writes.
    std::pair<MagAOX::dmTiming::Event, std::vector<uint8_t>>
    pipelinePreparation( uint64_t counter /**< [in] Input counter retained in an unreused pool slot. */ );

    /// Return the active pipeline request's immutable diagnostic dispatch stamp.
    int64_t pipelineDispatchStamp();

    /// Wait for one submitted command's SDK entry, return and worker-ready endpoints without requiring another input.
    std::array<int64_t, 3>
    waitPipelineSdkReady( uint64_t counter /**< [in] Input counter whose pool slot has not yet been reused. */ );

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

    /// Corrupt staging geometry while the monitor is held before that stage.
    void breakPipelineStaging();

    /// Mark the combined stream busy after admission to reproduce an unstable metadata capture.
    void busyInput();

    /// Return the prepared-shape publication count independently of native result streams.
    uint64_t shapeCounter() const;

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

void bmcCtrl_test::deferMonitorAllocation()
{
    m_outputShape.close();
    m_width      = m_dmWidth;
    m_height     = m_dmHeight;
    m_dataType   = _DATATYPE_FLOAT;
    m_shmimName  = "bmc_timing_";
    m_shmimShape = "bmc_timing_shape";
    m_shmimDelta = "bmc_test_delta_out";
    m_shmimDiff  = "bmc_test_diff_out";

    for( auto *channel : m_channels )
    {
        delete channel;
    }

    m_channels.clear();
    m_notDeltas.clear();
    m_deltas.clear();
    m_deltaChannels.clear();

    for( auto &job : m_pipelineJobs )
    {
        job.values.clear();
        job.command.shape.resize( 0, 0 );
    }
}

void bmcCtrl_test::enablePipeline()
{
    m_pipelineEnabled = true;
    stagedProducts();
    allocatePipelineJobs();
    REQUIRE( m_dmCompletion.open( "bmc_pipeline_full" ) == 0 );
    REQUIRE( m_dmReaderRelease.open( "bmc_pipeline_read", MagAOX::dmCompletion::Kind::readers ) == 0 );
    REQUIRE( m_pipelineResult.open( "bmc_pipeline_sdk", MagAOX::dmCompletion::Kind::sdk ) == 0 );
    m_pipelineThreadInit = false;
    m_pipelineConfigured = true;
    m_pipelineStop       = false;
    m_pipelineThread     = std::thread( pipelineThreadStart, this );
    enableWorker();
}

int bmcCtrl_test::pipelineInput( uint64_t counter )
{
    prepareInput( counter );
    shmimMonitorTiming( dev::shmimT(), dev::shmimMonitorEvent::receipt );
    return processImage( pixels(), dev::shmimT() );
}

void bmcCtrl_test::waitRetired( uint64_t count )
{
    std::unique_lock<std::mutex> lock( m_sdkMutex );
    REQUIRE( m_sdkDone.wait_for( lock,
                                 std::chrono::seconds( 3 ),
                                 [this, count]()
                                 {
                                     return m_pipelineRetired >= count;
                                 } ) );
}

std::array<uint64_t, 6> bmcCtrl_test::pipelineCounts()
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    return { m_pipelineAdmitted,
             m_pipelineRetired,
             m_pipelineOverloaded,
             m_pipelineCancelled,
             m_pipelineSuperseded,
             m_pipelinePostFailures };
}

std::array<uint64_t, 3> bmcCtrl_test::pipelineQuality()
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    return { m_pipelineVectorRejected, m_pipelineReaderUnqualified, m_pipelinePreparationFailures };
}

std::pair<MagAOX::dmTiming::Event, std::vector<uint8_t>> bmcCtrl_test::pipelinePreparation( uint64_t counter )
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );

    for( const auto &job : m_pipelineJobs )
    {
        if( job.identity.inputCounter == counter )
        {
            return { job.command.timing, job.command.saturation };
        }
    }

    throw std::runtime_error( "pipeline test job no longer retained" );
}

int64_t bmcCtrl_test::pipelineDispatchStamp()
{
    std::lock_guard<std::mutex> lock( m_sdkMutex );
    return m_pipelineSdkActive < pipelineCapacity ? m_pipelineJobs[m_pipelineSdkActive].command.dispatchNs : 0;
}

std::array<int64_t, 3> bmcCtrl_test::waitPipelineSdkReady( uint64_t counter )
{
    std::unique_lock<std::mutex> lock( m_sdkMutex );
    std::array<int64_t, 3>       endpoints{};
    REQUIRE( m_sdkDone.wait_for( lock,
                                 std::chrono::seconds( 3 ),
                                 [this, counter, &endpoints]()
                                 {
                                     for( const auto &job : m_pipelineJobs )
                                     {
                                         if( job.submitted && job.identity.inputCounter == counter &&
                                             job.result.readyNs >= 0 )
                                         {
                                             endpoints = { job.result.startNs, job.result.endNs, job.result.readyNs };
                                             return true;
                                         }
                                     }

                                     return false;
                                 } ) );
    return endpoints;
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

void bmcCtrl_test::breakPipelineStaging()
{
    m_totalFlat.resize( 1, 1 );
}

void bmcCtrl_test::busyInput()
{
    m_imageStream.md->write = 1;
}

uint64_t bmcCtrl_test::shapeCounter() const
{
    return m_shape.md->cnt0;
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
    REQUIRE( summary.metrics[metricIndex( Metric::sdkTail )].count == 0 );
    REQUIRE( summary.metrics[metricIndex( Metric::sdkService )].count == 0 );

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
    REQUIRE( app.dmTimingSummary().metrics[metricIndex( Metric::sdkTail )].count == 0 );
    REQUIRE( app.dmTimingSummary().metrics[metricIndex( Metric::sdkService )].count == 0 );
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

/// Private preparation returns while SDK A is held, and SDK B proceeds while ordered post A is held.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline overlaps preparation SDK and ordered post with frozen identities", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::processImage(nullptr,MagAOX::app::dev::shmimT());
    MagAOX::app::bmcCtrl::sdkThreadExec();
    MagAOX::app::bmcCtrl::pipelineThreadExec();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::heldPost                       = 1;
    bmcTimingTest::ReturnGuard guard;
    app.pixels()[0] = 0.25;
    app.pixels()[1] = 2;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    const auto firstRelease = app.m_dmReaderRelease.snapshot();
    REQUIRE( readersQualified( firstRelease ) );
    REQUIRE( firstRelease.inputCounter == 1 );
    REQUIRE( firstRelease.outcome == MagAOX::dmTiming::Outcome::unknown );
    REQUIRE( bmcTimingTest::lastReturn == 0 );
    app.pixels()[0] = 0.49;
    app.pixels()[1] = 0.81;
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    const auto secondRelease = app.m_dmReaderRelease.snapshot();
    REQUIRE( secondRelease.inputCounter == 2 );
    REQUIRE( readersQualified( secondRelease ) );
    app.mutateChannels();
    bmcTimingTest::permitCalls( 1 );
    bmcTimingTest::waitPost();
    bmcTimingTest::waitCalls( 2 );
    const auto firstSdk = app.waitPipelineSdkReady( 1 );
    REQUIRE( firstSdk[2] >= firstSdk[1] );
    REQUIRE( app.delta( 0 ) == 0 );
    REQUIRE( app.pipelineCounts()[1] == 0 );
    REQUIRE( app.m_dmReaderRelease.snapshot().sequence == secondRelease.sequence );
    REQUIRE( app.m_dmReaderRelease.snapshot().outcome == MagAOX::dmTiming::Outcome::unknown );
    bmcTimingTest::permitCalls( 2 );
    const auto secondSdk = app.waitPipelineSdkReady( 2 );
    REQUIRE( secondSdk[0] >= firstSdk[2] );
    REQUIRE( secondSdk[2] >= secondSdk[1] );
    REQUIRE( app.pipelineCounts()[1] == 0 );
    REQUIRE( app.delta( 0 ) == 0 );
    bmcTimingTest::permitReturn();
    app.waitRetired( 2 );
    REQUIRE( app.pipelineCounts() == std::array<uint64_t, 6>{ 2, 2, 0, 0, 0, 0 } );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE_FALSE( app.m_commandActive );
    REQUIRE( app.m_nsat == 1 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
    REQUIRE( bmcTimingTest::history[0] == std::vector<double>{ 0.5, 1 } );
    REQUIRE( bmcTimingTest::history[1][0] == Approx( 0.7 ) );
    REQUIRE( bmcTimingTest::history[1][1] == Approx( 0.9 ) );
    REQUIRE( app.delta( 0 ) == Approx( 0.39 ) );
    REQUIRE( app.delta( 1 ) == Approx( 0.61 ) );
    REQUIRE( app.m_dmCompletion.snapshot().inputCounter == 2 );
    REQUIRE( qualified( app.m_dmCompletion.snapshot() ) );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 2 );
    REQUIRE( sdkQualified( app.m_pipelineResult.snapshot() ) );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 2 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
    REQUIRE( app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkTail )].count ==
             2 );
    REQUIRE(
        app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkService )].count ==
        2 );
    REQUIRE(
        app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::saturation )].count ==
        2 );
}

/// Saturation consumes immutable vectors only in post, while preparation subintervals partition the original timer.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline defers saturation beyond SDK dispatch and measures preparation",
           "[bmcCtrl][pipeline][timing]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::processPipeline(nullptr);
    MagAOX::app::bmcCtrl::pipelineThreadExec();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;
    bmcCtrl_test app;
    app.m_dmTiming.configure( 60, 32 );
    app.enablePipeline();
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::heldPost                       = 1;
    bmcTimingTest::ReturnGuard guard;
    app.pixels()[0] = -0.25;
    app.pixels()[1] = 2;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    const auto [preparation, mask] = app.pipelinePreparation( 1 );
    REQUIRE( mask == std::vector<uint8_t>{ 0, 0 } );
    REQUIRE( app.m_nsat == 0 );
    REQUIRE( ( preparation.valid & ( 1u << markerIndex( Marker::satStart ) ) ) == 0 );
    REQUIRE( ( preparation.valid & ( 1u << markerIndex( Marker::satEnd ) ) ) == 0 );

    for( const auto marker : { Marker::receipt,
                               Marker::actuatorStart,
                               Marker::conversionEnd,
                               Marker::sourceCheckEnd,
                               Marker::actuatorEnd } )
    {
        REQUIRE( preparation.valid & ( 1u << markerIndex( marker ) ) );
    }

    const auto stamp = [&preparation]( Marker marker )
    {
        return preparation.markers[markerIndex( marker )];
    };
    REQUIRE( stamp( Marker::actuatorStart ) >= stamp( Marker::receipt ) );
    REQUIRE( stamp( Marker::conversionEnd ) >= stamp( Marker::actuatorStart ) );
    REQUIRE( stamp( Marker::sourceCheckEnd ) >= stamp( Marker::conversionEnd ) );
    REQUIRE( stamp( Marker::actuatorEnd ) >= stamp( Marker::sourceCheckEnd ) );
    const auto firstSdkEntry = bmcTimingTest::firstEntry.load();
    REQUIRE( firstSdkEntry >= stamp( Marker::actuatorEnd ) );
    app.pixels()[0] = 0.25;
    app.pixels()[1] = 0.81;
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    bmcTimingTest::permitCalls( 1 );
    bmcTimingTest::waitPost();
    bmcTimingTest::waitCalls( 2 );
    REQUIRE( app.pipelinePreparation( 1 ).second == std::vector<uint8_t>{ 0, 0 } );
    REQUIRE( app.m_nsat == 0 );
    const auto firstSdk = app.waitPipelineSdkReady( 1 );
    bmcTimingTest::permitCalls( 2 );
    const auto secondSdk         = app.waitPipelineSdkReady( 2 );
    const auto secondPreparation = app.pipelinePreparation( 2 ).first;
    bmcTimingTest::permitReturn();
    app.waitRetired( 2 );
    REQUIRE( app.m_nsat == 2 );
    REQUIRE( app.saturation( 0 ) == 0 );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE( bmcTimingTest::history[0] == std::vector<double>{ 0, 1 } );
    REQUIRE( bmcTimingTest::history[1][0] == Approx( 0.5 ) );
    REQUIRE( bmcTimingTest::history[1][1] == Approx( 0.9 ) );
    app.m_dmTiming.update();
    const auto &summary = app.dmTimingSummary();
    REQUIRE( summary.total.invalid == 0 );

    for( const auto metric :
         { Metric::conversion, Metric::sourceCheck, Metric::shapePublication, Metric::receiptSdk, Metric::saturation } )
    {
        REQUIRE( summary.metrics[metricIndex( metric )].count == 2 );
    }

    REQUIRE( summary.metrics[metricIndex( Metric::actuator )].mean ==
             Approx( summary.metrics[metricIndex( Metric::conversion )].mean +
                     summary.metrics[metricIndex( Metric::sourceCheck )].mean +
                     summary.metrics[metricIndex( Metric::shapePublication )].mean ) );
    const double prefix = ( firstSdk[0] - stamp( Marker::receipt ) + secondSdk[0] -
                            secondPreparation.markers[markerIndex( Marker::receipt )] ) *
                          0.5e-9;
    REQUIRE( summary.metrics[metricIndex( Metric::receiptSdk )].mean == Approx( prefix ) );
    app.pixels()[0] = 0.16;
    app.pixels()[1] = 0.36;
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    app.waitRetired( 3 );
    REQUIRE( app.m_nsat == 2 );
    REQUIRE( app.saturation( 0 ) == 0 );
    REQUIRE( app.saturation( 1 ) == 0 );
    REQUIRE_FALSE( app.m_sdkFault );
}

/// An isolated command records readiness before waiting, and later pool reuse cannot overwrite its timing sample.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline worker service excludes idle waiting and survives pool reuse", "[bmcCtrl][pipeline][timing]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::sdkThreadExec();
    MagAOX::app::bmcCtrl::pipelineThreadExec();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;
    bmcCtrl_test app;
    app.m_dmTiming.configure( 60, 32 );
    app.enablePipeline();
    int64_t previousReady = -1;
    double  serviceSum    = 0;
    double  tailSum       = 0;

    for( uint64_t counter = 1; counter <= 4; ++counter )
    {
        const auto inputBoundary = dev::dmTiming::monotonicNowNs();
        REQUIRE( previousReady <= inputBoundary );
        REQUIRE( app.pipelineInput( counter ) == 0 );
        app.waitRetired( counter );
        const auto endpoints = app.waitPipelineSdkReady( counter );
        REQUIRE( endpoints[0] >= inputBoundary );
        REQUIRE( endpoints[1] >= endpoints[0] );
        REQUIRE( endpoints[2] >= endpoints[1] );
        REQUIRE( endpoints[2] <= dev::dmTiming::monotonicNowNs() );
        previousReady = endpoints[2];
        serviceSum += ( endpoints[2] - endpoints[0] ) * 1e-9;
        tailSum += ( endpoints[2] - endpoints[1] ) * 1e-9;
        app.m_dmTiming.update();
        const auto &summary = app.dmTimingSummary();
        const auto &service = summary.metrics[metricIndex( Metric::sdkService )];
        const auto &tail    = summary.metrics[metricIndex( Metric::sdkTail )];
        REQUIRE( summary.total.submitted == counter );
        REQUIRE( summary.total.invalid == 0 );
        REQUIRE( service.count == counter );
        REQUIRE( tail.count == counter );
        REQUIRE( service.mean == Approx( serviceSum / counter ) );
        REQUIRE( tail.mean == Approx( tailSum / counter ) );
        REQUIRE( service.valid == ( counter > 1 ) );
        REQUIRE( tail.valid == ( counter > 1 ) );
    }

    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE( app.pipelineCounts() == std::array<uint64_t, 6>{ 4, 4, 0, 0, 0, 0 } );
}

/// A full bounded pool still announces newer input, so old queued vectors never replay after an SDK stall.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline coalesces latest observed input even when all credits are busy", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::claimPipelineJob();
    MagAOX::app::bmcCtrl::supersedePipelineJob(MagAOX::app::bmcCtrl::PipelineJob());
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::ReturnGuard guard;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    REQUIRE( app.pipelineInput( 4 ) == 0 );
    REQUIRE( app.pipelineCounts()[2] == 1 );
    bmcTimingTest::permitReturn();
    app.waitRetired( 3 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.pipelineCounts() == std::array<uint64_t, 6>{ 3, 3, 1, 2, 2, 0 } );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE( app.pipelineInput( 5 ) == 0 );
    app.waitRetired( 4 );
    REQUIRE( bmcTimingTest::calls == 2 );
    REQUIRE( app.m_coalescedInputs == 1 );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 5 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.noSend == 2 );
    REQUIRE( app.dmTimingSummary().total.submitted == 2 );
}

/// SDK failure or exception fences every unsent private command and preserves successful-submission accounting.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline native failures cancel queued jobs without false submitted counts", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::faultPipeline();
    #endif
    // clang-format on
    using namespace MagAOX::dmCompletion;
    MagAOX::app::bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block  = true;
    bmcTimingTest::throws = GENERATE( false, true );
    bmcTimingTest::result = 7;
    bmcTimingTest::ReturnGuard guard;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    bmcTimingTest::permitReturn();
    app.waitRetired( 3 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.pipelineCounts()[3] == 2 );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_admissionClosed );
    REQUIRE( app.m_sdkFailures == 1 );
    REQUIRE_FALSE( app.m_commandActive );
    REQUIRE_FALSE( sdkQualified( app.m_pipelineResult.snapshot() ) );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 1 );
    REQUIRE( app.m_pipelineResult.snapshot().backendResult == ( bmcTimingTest::throws ? unavailableResult : 7 ) );
    REQUIRE( app.delta( 0 ) == 0 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 0 );
    REQUIRE( app.dmTimingSummary().total.failed == 1 );
    REQUIRE( app.dmTimingSummary().total.noSend == 2 );
    REQUIRE( app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkTail )].count ==
             1 );
    REQUIRE(
        app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkService )].count ==
        1 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
}

/// A post error can follow SDK B entry; it fences C without cancelling the actual B call or replaying old products.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline late post failure drains a running SDK and cancels later work", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::pipelineThreadExec();
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::heldPost                       = 1;
    bmcTimingTest::ReturnGuard guard;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    bmcTimingTest::permitCalls( 1 );
    bmcTimingTest::waitPost();
    bmcTimingTest::waitCalls( 2 );
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    app.fullSaturationSemaphore();
    { //mutex scope
        std::lock_guard<std::mutex> lock( bmcTimingTest::mutex );
        bmcTimingTest::postReleased = true;
        bmcTimingTest::changed.notify_all();
    }
    app.waitRetired( 1 );
    REQUIRE( app.m_sdkFault );
    REQUIRE( bmcTimingTest::calls == 2 );
    bmcTimingTest::permitReturn();
    app.waitRetired( 3 );
    REQUIRE( bmcTimingTest::calls == 2 );
    REQUIRE( app.pipelineCounts()[3] == 1 );
    REQUIRE( app.pipelineCounts()[5] == 2 );
    REQUIRE( app.m_sdkFailures == 0 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 2 );
    REQUIRE( app.dmTimingSummary().total.failed == 2 );
    REQUIRE( app.dmTimingSummary().total.noSend == 1 );
}

/// Disconnect invalidates queued old-source commands while a running SDK retains its frozen identity.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline disconnect cancels old queued identities without faulting the backend", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::shmimMonitorTiming(MagAOX::app::dev::shmimT(),MagAOX::app::dev::shmimMonitorEvent::disconnected);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = true;
    bmcTimingTest::ReturnGuard guard;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    app.shmimMonitorTiming( dev::shmimT(), dev::shmimMonitorEvent::disconnected );
    bmcTimingTest::permitReturn();
    app.waitRetired( 2 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.pipelineCounts()[4] == 1 );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 1 );
    REQUIRE( app.m_dmCompletion.snapshot().inputCounter == 2 );
    REQUIRE( app.m_dmCompletion.snapshot().outcome == MagAOX::dmTiming::Outcome::noSend );
}

/// Control operations wait for whole ordered post retirement after native return, and all handle calls stay serialized.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline controls drain post ownership before touching the native handle",
           "[bmcCtrl][pipeline][lifecycle]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::zeroDM();
    MagAOX::app::bmcCtrl::closeAdmission();
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::heldPost                       = 1;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    bmcTimingTest::permitCalls( 1 );
    bmcTimingTest::waitPost();
    auto                       control = std::async( std::launch::async,
                               [&app]()
                               {
                                   return app.zeroDM();
                               } );
    bmcTimingTest::ReturnGuard guard;
    app.waitClosed();
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( control.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    bmcTimingTest::permitReturn();
    REQUIRE( control.get() == 0 );
    app.waitRetired( 1 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
    REQUIRE( bmcTimingTest::calls == 2 );
}

/// A diagnostic deadline cancels unsent jobs but never frees or cancels the actual in-flight native call.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline watchdog counts once and retains actual native ownership", "[bmcCtrl][pipeline][timeout]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::checkSdkTimeout(0);
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block = true;
    bmcTimingTest::ReturnGuard guard;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 1 );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    app.m_sdkTimeoutSec = 1;
    const auto dispatch = app.pipelineDispatchStamp();
    app.checkSdkTimeout( dispatch + 2000000000LL );
    app.checkSdkTimeout( dispatch + 3000000000LL );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( app.m_commandActive );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.pipelineCounts()[1] == 0 );
    bmcTimingTest::permitReturn();
    app.waitRetired( 2 );
    REQUIRE( app.m_sdkTimeouts == 1 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.m_pipelineResult.snapshot().outcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( app.delta( 0 ) == 0 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 1 );
    REQUIRE( app.dmTimingSummary().total.failed == 1 );
    REQUIRE( app.dmTimingSummary().total.noSend == 1 );
    REQUIRE(
        app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkService )].count ==
        1 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
}

/// Validated private vector identity remains independent of later channel-staging and reader qualification.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline late input advance revokes reader provenance without faulting successful native work",
           "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::processPipeline(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::heldPrep            = 1;
    auto                       prepare = std::async( std::launch::async,
                               [&app]()
                               {
                                   return app.pipelineInput( 1 );
                               } );
    bmcTimingTest::ReturnGuard guard;
    bmcTimingTest::waitPrep();
    bmcTimingTest::waitCalls( 1 );
    IMAGE sdk{};
    REQUIRE( ImageStreamIO_openIm( &sdk, "bmc_pipeline_sdk" ) == 0 );
    Record   record;
    timespec deadline{};
    clock_gettime( CLOCK_REALTIME, &deadline );
    deadline.tv_sec += 3;

    do
    {
        REQUIRE( sem_timedwait( sdk.semptr[0], &deadline ) == 0 );
        REQUIRE( dev::dmCompletion::read( &sdk, record ) );
    } while( record.inputCounter != 1 || record.finishedNs == 0 );

    REQUIRE( record.outcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( record.state == State::complete );
    REQUIRE( sdkQualified( record ) );
    const auto ready = app.waitPipelineSdkReady( 1 );
    REQUIRE( ready[1] == record.finishedNs );
    REQUIRE( ready[2] >= ready[1] );
    REQUIRE( ready[2] <= dev::dmTiming::monotonicNowNs() );
    const bool advance = GENERATE( false, true );

    if( advance )
    {
        app.prepareInput( 2 );
    }

    bmcTimingTest::permitReturn();
    REQUIRE( prepare.get() == 0 );
    app.waitRetired( 1 );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE_FALSE( app.m_admissionClosed );
    REQUIRE( qualified( app.m_dmCompletion.snapshot() ) == !advance );
    REQUIRE( sdkQualified( app.m_pipelineResult.snapshot() ) );
    REQUIRE( app.m_dmCompletion.snapshot().outcome == MagAOX::dmTiming::Outcome::success );
    REQUIRE( app.pipelineQuality() == std::array<uint64_t, 3>{ 0, uint64_t( advance ), 0 } );
    REQUIRE( readersQualified( app.m_dmReaderRelease.snapshot() ) == !advance );
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    app.waitRetired( 2 );
    REQUIRE( app.pipelineInput( 4 ) == 0 );
    app.waitRetired( 3 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().fgValid );
    REQUIRE( app.dmTimingSummary().total.submitted == 3 );
    REQUIRE( app.dmTimingSummary().total.failed == 0 );
    REQUIRE( app.dmTimingSummary().total.invalid == 0 );
    REQUIRE( app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkTail )].count ==
             3 );
    REQUIRE(
        app.dmTimingSummary().metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sdkService )].count ==
        3 );
    REQUIRE( ImageStreamIO_closeIm( &sdk ) == 0 );
}

/// Pipeline configuration is startup-only, requires a worker, and cannot publish result records over DM images.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline configuration validates ownership and distinct result names", "[bmcCtrl][pipeline][config]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE_FALSE( app.m_pipelineEnabled );
    REQUIRE( app.m_pipelineThreadPrio == 0 );
    REQUIRE( app.m_pipelineCpuset.empty() );
    REQUIRE( app.m_pipelineResultStream.empty() );
    const auto setting =
        GENERATE( std::string( "workerEnabled=false\npipelineEnabled=true" ),
                  std::string( "workerEnabled=true\npipelineEnabled=false\npipelineResultStream=done" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineThreadPrio=-1" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineThreadPrio=100" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=bmcgrid07" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=read_done" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=flat" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=aux_source" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=bad/name" ),
                  std::string( "workerEnabled=true\npipelineEnabled=true\npipelineResultStream=sdk_done" ) );
    char      path[]     = "/tmp/bmc-pipeline-config-XXXXXX";
    const int descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    close( descriptor );
    {
        std::ofstream file( path );
        file << "[dm]\nshmimName=bmcgrid\nreaderReleaseStream=read_done\nshmimFlat=flat\ndeltaChannels=aux_source\n["
                "sdk]\n"
             << setting << '\n';
    }
    app.config.readConfig( path );
    const bool valid = setting.ends_with( "pipelineResultStream=sdk_done" );
    REQUIRE( app.loadConfigImpl( app.config ) == ( valid ? 0 : -1 ) );
    REQUIRE_FALSE( app.m_sdkThread.joinable() );
    REQUIRE( unlink( path ) == 0 );
}

/// Hardware initialization can precede image allocation, and reconnect allocation waits for all old ownership.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline initializes before monitor images and drains before reallocation",
           "[bmcCtrl][pipeline][lifecycle]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::initDM();
    MagAOX::app::bmcCtrl::allocate(MagAOX::app::dev::shmimT());
    #endif
    // clang-format on
    using namespace MagAOX::app;
    bmcCtrl_test app;
    app.enablePipeline();
    REQUIRE( app.releaseDM() == 0 );
    app.deferMonitorAllocation();
    app.mappingFile();
    REQUIRE( app.initDM() == 0 );
    REQUIRE( app.allocate( dev::shmimT() ) == 0 );
    bmcTimingTest::block = bmcTimingTest::perCall = true;
    bmcTimingTest::heldPost                       = 1;
    REQUIRE( app.pipelineInput( 1 ) == 0 );
    bmcTimingTest::waitCalls( 3 );
    bmcTimingTest::permitCalls( 3 );
    bmcTimingTest::waitPost();
    auto                       allocation = std::async( std::launch::async,
                                  [&app]()
                                  {
                                      return app.allocate( dev::shmimT() );
                                  } );
    bmcTimingTest::ReturnGuard guard;
    app.waitClosed();
    REQUIRE( allocation.wait_for( std::chrono::seconds( 0 ) ) != std::future_status::ready );
    bmcTimingTest::permitReturn();
    REQUIRE( allocation.get() == 0 );
    app.waitRetired( 1 );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE_FALSE( app.m_admissionClosed );
    REQUIRE( app.pipelineInput( 2 ) == 0 );
    app.waitRetired( 2 );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 2 );
    REQUIRE( bmcTimingTest::maximumActive == 1 );
}

/// An unstable combined source at the last-pixel boundary is never sent or published as a prepared shape.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline rejects changed private vector identity before native entry without fault",
           "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::processPipeline(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enablePipeline();
    const bool capture                 = GENERATE( false, true );
    bmcTimingTest::captureHold         = capture;
    bmcTimingTest::heldVector          = 1;
    const auto shapeBefore             = app.shapeCounter();
    app.pixels()[0]                    = 0.25;
    auto                       prepare = std::async( std::launch::async,
                               [&app]()
                               {
                                   return app.pipelineInput( 1 );
                               } );
    bmcTimingTest::ReturnGuard guard;
    bmcTimingTest::waitVector();
    if( capture )
    {
        app.busyInput();
    }
    else
    {
        app.prepareInput( 2 );
    }
    bmcTimingTest::permitReturn();
    REQUIRE( prepare.get() == 0 );
    app.waitRetired( 1 );
    REQUIRE( bmcTimingTest::calls == 0 );
    REQUIRE( app.shapeCounter() == shapeBefore );
    REQUIRE_FALSE( app.m_sdkFault );
    REQUIRE_FALSE( app.m_admissionClosed );
    REQUIRE( app.pipelineCounts()[4] == 1 );
    REQUIRE( app.pipelineQuality() == std::array<uint64_t, 3>{ 1, 1, 0 } );
    REQUIRE_FALSE( readersQualified( app.m_dmReaderRelease.snapshot() ) );
    REQUIRE_FALSE( qualified( app.m_dmCompletion.snapshot() ) );
    REQUIRE( app.m_dmCompletion.snapshot().outcome == MagAOX::dmTiming::Outcome::noSend );
    REQUIRE( app.pipelineInput( 3 ) == 0 );
    app.waitRetired( 2 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( app.m_pipelineResult.snapshot().inputCounter == 3 );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.noSend == 1 );
    REQUIRE( app.dmTimingSummary().total.failed == 0 );
}

/// A true channel-staging geometry failure remains a fault, unlike ordinary late provenance loss.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC pipeline retains fatal preparation errors and diagnoses them separately", "[bmcCtrl][pipeline]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::processPipeline(nullptr);
    MagAOX::app::bmcCtrl::faultPipeline();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmCompletion;
    bmcCtrl_test app;
    app.enablePipeline();
    bmcTimingTest::block               = true;
    bmcTimingTest::heldPrep            = 1;
    auto                       prepare = std::async( std::launch::async,
                               [&app]()
                               {
                                   return app.pipelineInput( 1 );
                               } );
    bmcTimingTest::ReturnGuard guard;
    bmcTimingTest::waitPrep();
    bmcTimingTest::waitCalls( 1 );
    app.breakPipelineStaging();

    { //mutex scope
        std::lock_guard<std::mutex> lock( bmcTimingTest::mutex );
        bmcTimingTest::prepReleased = true;
        bmcTimingTest::changed.notify_all();
    }

    REQUIRE( prepare.get() == 0 );
    REQUIRE( app.m_sdkFault );
    REQUIRE( app.m_admissionClosed );
    REQUIRE( app.pipelineQuality() == std::array<uint64_t, 3>{ 0, 1, 1 } );
    bmcTimingTest::permitReturn();
    app.waitRetired( 1 );
    REQUIRE( app.pipelineCounts()[5] == 0 );
    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( sdkQualified( app.m_pipelineResult.snapshot() ) );
    REQUIRE( app.m_dmCompletion.snapshot().state == State::error );
    app.m_dmTiming.update();
    REQUIRE( app.dmTimingSummary().total.submitted == 1 );
    REQUIRE( app.dmTimingSummary().total.failed == 1 );
}

/// Initial worker definitions must serialize before any appLogic update or hardware initialization.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "BMC worker initial diagnostic definitions are complete valid INDI XML", "[bmcCtrl][pipeline][indi]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::setupSdkDiagnostics();
    #endif
    // clang-format on
    MagAOX::app::bmcCtrl_test app;
    app.m_configName = "bmc-offline";
    app.enablePipeline();
    REQUIRE( app.setupSdkDiagnostics() == 0 );

    for( const auto *property : { &app.m_indiP_sdkWorker, &app.m_indiP_pipeline } )
    {
        std::string xml;
        REQUIRE_NOTHROW( xml = pcf::IndiXmlParser( pcf::IndiMessage( pcf::IndiMessage::Define, *property ), "1.7" )
                                   .createXmlString() );
        REQUIRE( xml.find( "<defNumberVector" ) != std::string::npos );
        REQUIRE( xml.find( "state=\"Idle\"" ) != std::string::npos );
        pcf::IndiXmlParser parser( "1.7" );
        std::string        error;
        parser.parseXml( xml, error );
        REQUIRE( error.empty() );
        REQUIRE( parser.getState() == pcf::IndiXmlParser::CompleteState );
        const auto decoded = parser.createIndiMessage().getProperty();
        REQUIRE( decoded.getNumElements() == property->getNumElements() );

        for( unsigned index = 0; index < decoded.getNumElements(); ++index )
        {
            REQUIRE_NOTHROW( decoded[index].get<double>() );
        }
    }

    REQUIRE( app.m_indiP_pipeline["enabled"].get<double>() == 1 );
    REQUIRE( app.m_indiP_pipeline["capacity"].get<double>() == 3 );
}

} // namespace bmcCtrlTest
} // namespace libXWCTest
