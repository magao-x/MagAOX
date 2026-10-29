/** \file dmSyncopation_test.cpp
 * \brief Deterministic timing/control tests with isolated workstation-local ImageStreamIO streams.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../dmSyncopation.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>

namespace libXWCTest
{
namespace dmSyncopationTest
{
using namespace MagAOX::app;

/// \cond TEST_HARNESS
/// Unactivated production driver whose property XML remains entirely workstation-local.
class SimulationDriver : public indiDriver<MagAOXApp<true>>
{
  public:
    /// Open only the fixture's private FIFOs and enable property serialization.
    SimulationDriver( MagAOXApp<true> *parent /**< [in] App owning private test FIFOs. */ );

    /// Close the unactivated input descriptor; the shared driver closes its output descriptor.
    ~SimulationDriver();

    /// Discard serialized publications without writing a pipe or opening a client connection.
    void sendXml( const std::string &xml /**< [in] Production-formatted local property publication. */ ) const override;

  private:
    int m_input{ -1 }; ///< Private input descriptor not released by an unactivated production driver.
};

SimulationDriver::SimulationDriver( MagAOXApp<true> *parent )
    : indiDriver<MagAOXApp<true>>( parent, "synco-local-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( ::stat( parent->driverInName().c_str(), &input ) == 0 );

    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        int         descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};

        if( ::fstat( descriptor, &candidate ) == 0 && candidate.st_ino == input.st_ino &&
            candidate.st_dev == input.st_dev )
        {
            REQUIRE( m_input == -1 );
            m_input = descriptor;
        }
    }

    REQUIRE( m_input >= 0 );
}

SimulationDriver::~SimulationDriver()
{
    if( m_input >= 0 )
    {
        ::close( m_input );
    }

    setInputFd( -1 );
}

void SimulationDriver::sendXml( const std::string &xml ) const
{
    static_cast<void>( xml );
}

/// Scheduler access and cleanup without starting app, hardware, INDI or monitor threads.
class Harness : public dmSyncopation
{
  public:
    using CombinedMonitor      = dev::shmimMonitor<dmSyncopation, syncoCombinerT>;
    using CameraMonitor        = dev::shmimMonitor<dmSyncopation, syncoCameraT>;
    using ReaderReleaseMonitor = dev::shmimMonitor<dmSyncopation, syncoReaderReleaseT>;
    using CompletionMonitor    = dev::shmimMonitor<dmSyncopation, syncoCompletionT>;

    using dmSyncopation::allocate;
    using dmSyncopation::configureSimulation;
    using dmSyncopation::consume;
    using dmSyncopation::copyBatch;
    using dmSyncopation::executeStep;
    using dmSyncopation::Fault;
    using dmSyncopation::m_advances;
    using dmSyncopation::m_bounds;
    using dmSyncopation::m_cameraCadence;
    using dmSyncopation::m_cameraDevice;
    using dmSyncopation::m_cameraDiagnostics;
    using dmSyncopation::m_cameraElement;
    using dmSyncopation::m_cameraFps;
    using dmSyncopation::m_cameraJitterUs;
    using dmSyncopation::m_cameraMaxUs;
    using dmSyncopation::m_cameraMinUs;
    using dmSyncopation::m_cameraMutex;
    using dmSyncopation::m_cameraProperty;
    using dmSyncopation::m_cameraPublicationCadence;
    using dmSyncopation::m_cameraReport;
    using dmSyncopation::m_cameraTime;
    using dmSyncopation::m_clean;
    using dmSyncopation::m_combined;
    using dmSyncopation::m_combinerCadence;
    using dmSyncopation::m_combinerDiagnostics;
    using dmSyncopation::m_consumed;
    using dmSyncopation::m_copiedLate;
    using dmSyncopation::m_copyMax;
    using dmSyncopation::m_copyStarted;
    using dmSyncopation::m_cycleMutex;
    using dmSyncopation::m_delivered;
    using dmSyncopation::m_destinations;
    using dmSyncopation::m_diagnosticsOnly;
    using dmSyncopation::m_enabled;
    using dmSyncopation::m_epoch;
    using dmSyncopation::m_faultCounts;
    using dmSyncopation::m_faultInterval;
    using dmSyncopation::m_faultPeriod;
    using dmSyncopation::m_faults;
    using dmSyncopation::m_faultTiming;
    using dmSyncopation::m_indiP_camera;
    using dmSyncopation::m_indiP_cameraPairs;
    using dmSyncopation::m_indiP_cameraQuality;
    using dmSyncopation::m_indiP_cameraTiming;
    using dmSyncopation::m_indiP_cameraWritePairs;
    using dmSyncopation::m_indiP_cameraWriteQuality;
    using dmSyncopation::m_indiP_cameraWriteTiming;
    using dmSyncopation::m_indiP_combinerQuality;
    using dmSyncopation::m_indiP_combinerTiming;
    using dmSyncopation::m_indiP_enable;
    using dmSyncopation::m_indiP_pipelineBoundary;
    using dmSyncopation::m_indiP_pipelineQuality;
    using dmSyncopation::m_indiP_pipelineStages;
    using dmSyncopation::m_indiP_primaryPairs;
    using dmSyncopation::m_indiP_primaryTiming;
    using dmSyncopation::m_indiP_primaryTimingQuality;
    using dmSyncopation::m_indiP_reacquire;
    using dmSyncopation::m_indiP_readerQuality;
    using dmSyncopation::m_indiP_readerReleaseQuality;
    using dmSyncopation::m_indiP_readerReleaseTiming;
    using dmSyncopation::m_indiP_readerTiming;
    using dmSyncopation::m_indiP_settings;
    using dmSyncopation::m_indiP_simulation;
    using dmSyncopation::m_indiP_simulationCadence;
    using dmSyncopation::m_indiP_simulationEstimator;
    using dmSyncopation::m_indiP_simulationStatus;
    using dmSyncopation::m_indiP_simulationSteps;
    using dmSyncopation::m_indiP_simulationTrace;
    using dmSyncopation::m_lastFaultNs;
    using dmSyncopation::m_lastFaultReason;
    using dmSyncopation::m_lastMain;
    using dmSyncopation::m_lockFrames;
    using dmSyncopation::m_mappings;
    using dmSyncopation::m_maxCadence;
    using dmSyncopation::m_measuredPeriod;
    using dmSyncopation::m_minRate;
    using dmSyncopation::m_missed;
    using dmSyncopation::m_nextAttach;
    using dmSyncopation::m_notStarted;
    using dmSyncopation::m_now;
    using dmSyncopation::m_period;
    using dmSyncopation::m_periodSource;
    using dmSyncopation::m_pipelineCadence;
    using dmSyncopation::m_pipelineDiagnostics;
    using dmSyncopation::m_previousCallbackEnd;
    using dmSyncopation::m_previousCallbackRuntime;
    using dmSyncopation::m_previousPrimaryPublication;
    using dmSyncopation::m_primaryCadence;
    using dmSyncopation::m_primaryCadenceMaxEvents;
    using dmSyncopation::m_primaryCadenceWindow;
    using dmSyncopation::m_primaryCounter;
    using dmSyncopation::m_primaryExcursionUs;
    using dmSyncopation::m_primaryTiming;
    using dmSyncopation::m_primaryTraceInode;
    using dmSyncopation::m_primed;
    using dmSyncopation::m_rateFrames;
    using dmSyncopation::m_readerReleaseTiming;
    using dmSyncopation::m_readerTiming;
    using dmSyncopation::m_reason;
    using dmSyncopation::m_record;
    using dmSyncopation::m_requested;
    using dmSyncopation::m_resources;
    using dmSyncopation::m_scheduleState;
    using dmSyncopation::m_simulation;
    using dmSyncopation::m_simulationCadenceUpdateSec;
    using dmSyncopation::m_simulationCadenceWindowSec;
    using dmSyncopation::m_simulationCameraTime;
    using dmSyncopation::m_simulationEnabled;
    using dmSyncopation::m_simulationEstimatorLogFailed;
    using dmSyncopation::m_simulationEstimatorStatus;
    using dmSyncopation::m_simulationPhaseGain;
    using dmSyncopation::m_simulationPhaseSlewUs;
    using dmSyncopation::m_simulationReferenceFps;
    using dmSyncopation::m_simulationSnapshot;
    using dmSyncopation::m_simulationTraceLogFailed;
    using dmSyncopation::m_simulationTraceStatus;
    using dmSyncopation::m_simulationWindow;
    using dmSyncopation::m_submission;
    using dmSyncopation::m_timing;
    using dmSyncopation::m_timingName;
    using dmSyncopation::m_wait;
    using dmSyncopation::missedStep;
    using dmSyncopation::readCamera;
    using dmSyncopation::readPrimaryTiming;
    using dmSyncopation::setupCadenceINDI;
    using dmSyncopation::setupCombinerCadenceINDI;
    using dmSyncopation::setupPipelineINDI;
    using dmSyncopation::setupPrimaryCadenceINDI;
    using dmSyncopation::setupReaderTimingINDI;
    using dmSyncopation::setupSimulationINDI;
    using dmSyncopation::snapshot;
    using dmSyncopation::updateCadenceINDI;
    using dmSyncopation::updateCombinerCadenceINDI;
    using dmSyncopation::updatePairsINDI;
    using dmSyncopation::updatePeriod;
    using dmSyncopation::updatePipelineINDI;
    using dmSyncopation::updatePrimaryCadenceINDI;
    using dmSyncopation::updatePrimaryPairsINDI;
    using dmSyncopation::updateReaderTimingINDI;
    using dmSyncopation::updateSimulationCadence;
    using dmSyncopation::updateSimulationINDI;
    using dmSyncopation::waitUntil;
    using MagAOXApp<true>::config;
    using dev::shmimMonitor<dmSyncopation>::m_imageStream;
    using dev::shmimMonitor<dmSyncopation>::m_shmimName;
    using dev::shmimMonitor<dmSyncopation>::m_width;
    using dev::shmimMonitor<dmSyncopation>::m_height;
    using dev::shmimMonitor<dmSyncopation>::m_depth;
    using dev::shmimMonitor<dmSyncopation>::m_dataType;
    using dev::shmimMonitor<dmSyncopation>::m_inode;
    using dev::shmimMonitor<dmSyncopation>::m_semaphoreNumber;
    using dev::shmimMonitor<dmSyncopation>::m_getExistingFirst;
    using dev::shmimMonitor<dmSyncopation>::m_smThreadInit;
    using dev::shmimMonitor<dmSyncopation>::m_smThread;
    using dev::shmimMonitor<dmSyncopation>::m_smThreadPrio;
    using dev::shmimMonitor<dmSyncopation>::m_smCpuset;

    /// Set a stable test identity and an unopened primary handle.
    Harness();

    /// Close only test attachments; the fixture destroys its isolated streams.
    ~Harness() noexcept;

    /// Attach the primary using the same metadata the monitor supplies before allocate.
    void attach();

    /// Attach only the private primary descriptor, leaving all scheduler resources unavailable.
    void attachPrimaryOnly();

    /// Start the real monitor loop against the fixture's existing streams, without normal app startup.
    void startMonitor();

    /// Stop and join the real monitor, clearing its already-closed imported descriptor.
    void stopMonitor();

    /// Import only the combined stream without opening scheduler resources or starting workers.
    void attachCombinedOnly();

    /// Access the combined observer's imported descriptor without involving the scheduler's handle.
    IMAGE &combinedImage();

    /// Start the independent combined monitor against the fixture stream.
    void startCombinedMonitor();

    /// Join the combined worker before clearing its closed import.
    void stopCombinedMonitor();

    /// Return the independent monitor's final configured input name.
    const std::string &combinedMonitorName() const;

    /// Return the independent monitor's requested semaphore index.
    int combinedMonitorSemaphore() const;

    /// Return the independent monitor's configured thread priority.
    int combinedMonitorPriority() const;

    /// Import the fixture's dedicated completion record without starting another monitor.
    void attachCompletionOnly();

    /// Import only the independent early-release observer without starting workers.
    void attachReaderReleaseOnly();

    /// Access the early-release observer fixture attachment.
    IMAGE &readerReleaseImage();

    /// Access the completion observer's imported record for allocation and callback tests.
    IMAGE &completionImage();

    /// Return the optional completion observer's configured stream name.
    const std::string &completionMonitorName() const;

    /// Return the completion observer's scheduling priority.
    int completionMonitorPriority() const;

    /// Return the completion observer's configured CPU set.
    const std::string &completionMonitorCpuset() const;

    /// Return the optional early-release observer's configured stream name.
    const std::string &readerReleaseMonitorName() const;

    /// Return the early-release observer's scheduling priority.
    int readerReleaseMonitorPriority() const;

    /// Return the early-release observer's configured CPU set.
    const std::string &readerReleaseMonitorCpuset() const;

    /// Import an isolated fixture stream for metadata-only acquisition observation.
    void attachCameraOnly();

    /// Return the camera observer's descriptor without opening scheduler resources.
    IMAGE &cameraImage();

    /// Start the real camera monitor using the isolated primary fixture as its input.
    void startCameraMonitor();

    /// Join the camera observer and clear its already-closed imported descriptor.
    void stopCameraMonitor();

    /// Return the camera monitor's configured stream name.
    const std::string &cameraMonitorName() const;

    /// Return the camera observer's configured scheduling priority.
    int cameraMonitorPriority() const;

    /// Initialize callback properties without starting any threads or INDI connections.
    void properties();

    /// Install a threadless INDI driver using private local FIFOs to exercise property publication.
    void localDriver( const std::string &directory /**< [in] Fixture-owned temporary directory. */ );
};

Harness::Harness()
{
    m_configName  = "synco_test";
    m_imageStream = {};
}

Harness::~Harness() noexcept
{
    if( m_indiDriver )
    {
        delete m_indiDriver;
        m_indiDriver = nullptr;

        for( const auto &path : { m_driverInName, m_driverOutName, m_driverCtrlName } )
        {
            ::unlink( path.c_str() );
        }
    }

    stopCameraMonitor();
    stopCombinedMonitor();
    stopMonitor();
    closeResources();
    if( m_imageStream.md )
    {
        ImageStreamIO_closeIm( &m_imageStream );
    }

    if( combinedImage().md )
    {
        ImageStreamIO_closeIm( &combinedImage() );
    }

    if( cameraImage().md )
    {
        ImageStreamIO_closeIm( &cameraImage() );
    }

    if( readerReleaseImage().md )
    {
        ImageStreamIO_closeIm( &readerReleaseImage() );
    }

    if( completionImage().md )
    {
        ImageStreamIO_closeIm( &completionImage() );
    }
}

void Harness::attach()
{
    attachPrimaryOnly();
    REQUIRE( allocate( dev::shmimT{} ) == 0 );
    REQUIRE( m_resources );
}

void Harness::attachPrimaryOnly()
{
    m_shmimName    = "dm97disp03";
    m_combinedName = "dm97disp";
    m_destinations = { "dm97disp07", "dm97disp08" };
    m_timingName   = "synco_test_timing";
    m_epoch        = 12345;
    REQUIRE( ImageStreamIO_openIm( &m_imageStream, m_shmimName.c_str() ) == 0 );
    m_width           = m_imageStream.md->size[0];
    m_height          = m_imageStream.md->size[1];
    m_depth           = 1;
    m_dataType        = _DATATYPE_FLOAT;
    m_inode           = inode( m_shmimName );
    m_semaphoreNumber = 5;
}

void Harness::startMonitor()
{
    m_shmimName        = "dm97disp03";
    m_combinedName     = "dm97disp";
    m_destinations     = { "dm97disp07", "dm97disp08" };
    m_timingName       = "synco_test_timing";
    m_epoch            = 12345;
    m_getExistingFirst = true;
    m_smThreadInit     = false;
    m_semaphoreNumber  = 5;
    state( stateCodes::READY );
    m_smThread = std::thread(
        [this]()
        {
            dev::shmimMonitor<dmSyncopation>::smThreadExec();
        } );
}

void Harness::stopMonitor()
{
    if( m_smThread.joinable() )
    {
        m_shutdown = true;
        m_smThread.join();  // The real monitor's timed wait is bounded to one second.
        m_imageStream = {}; // smThreadExec closed this import without clearing IMAGE.
    }
}

void Harness::attachCombinedOnly()
{
    CombinedMonitor::m_shmimName = "dm97disp";
    REQUIRE( ImageStreamIO_openIm( &combinedImage(), "dm97disp" ) == 0 );
    REQUIRE( allocate( syncoCombinerT{} ) == 0 );
}

IMAGE &Harness::combinedImage()
{
    return CombinedMonitor::m_imageStream;
}

void Harness::startCombinedMonitor()
{
    CombinedMonitor::m_shmimName        = "dm97disp";
    CombinedMonitor::m_getExistingFirst = true;
    CombinedMonitor::m_smThreadInit     = false;
    CombinedMonitor::m_semaphoreNumber  = -1;
    state( stateCodes::READY );
    CombinedMonitor::m_smThread = std::thread(
        [this]()
        {
            CombinedMonitor::smThreadExec();
        } );
}

void Harness::stopCombinedMonitor()
{
    if( CombinedMonitor::m_smThread.joinable() )
    {
        m_shutdown = true;
        CombinedMonitor::m_smThread.join();
        combinedImage() = {}; // The real monitor closes its import without clearing IMAGE.
    }
}

const std::string &Harness::combinedMonitorName() const
{
    return CombinedMonitor::m_shmimName;
}

int Harness::combinedMonitorSemaphore() const
{
    return CombinedMonitor::m_semaphoreNumber;
}

void Harness::attachReaderReleaseOnly()
{
    ReaderReleaseMonitor::m_shmimName = "synco_reader_done";
    REQUIRE( ImageStreamIO_openIm( &readerReleaseImage(), ReaderReleaseMonitor::m_shmimName.c_str() ) == 0 );
    REQUIRE( allocate( syncoReaderReleaseT{} ) == 0 );
    shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::connected );
}

IMAGE &Harness::readerReleaseImage()
{
    return ReaderReleaseMonitor::m_imageStream;
}

void Harness::attachCompletionOnly()
{
    CompletionMonitor::m_shmimName = "synco_reader_done";
    REQUIRE( ImageStreamIO_openIm( &completionImage(), CompletionMonitor::m_shmimName.c_str() ) == 0 );
    REQUIRE( allocate( syncoCompletionT{} ) == 0 );
    shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::connected );
}

IMAGE &Harness::completionImage()
{
    return CompletionMonitor::m_imageStream;
}

const std::string &Harness::completionMonitorName() const
{
    return CompletionMonitor::m_shmimName;
}

int Harness::completionMonitorPriority() const
{
    return CompletionMonitor::m_smThreadPrio;
}

const std::string &Harness::completionMonitorCpuset() const
{
    return CompletionMonitor::m_smCpuset;
}

const std::string &Harness::readerReleaseMonitorName() const
{
    return ReaderReleaseMonitor::m_shmimName;
}

int Harness::readerReleaseMonitorPriority() const
{
    return ReaderReleaseMonitor::m_smThreadPrio;
}

const std::string &Harness::readerReleaseMonitorCpuset() const
{
    return ReaderReleaseMonitor::m_smCpuset;
}

void Harness::attachCameraOnly()
{
    CameraMonitor::m_shmimName = "dm97disp03";
    REQUIRE( ImageStreamIO_openIm( &cameraImage(), "dm97disp03" ) == 0 );
    REQUIRE( allocate( syncoCameraT{} ) == 0 );
}

IMAGE &Harness::cameraImage()
{
    return CameraMonitor::m_imageStream;
}

void Harness::startCameraMonitor()
{
    CameraMonitor::m_shmimName        = "dm97disp03";
    CameraMonitor::m_getExistingFirst = true;
    CameraMonitor::m_smThreadInit     = false;
    CameraMonitor::m_semaphoreNumber  = -1;
    state( stateCodes::READY );
    CameraMonitor::m_smThread = std::thread(
        [this]()
        {
            CameraMonitor::smThreadExec();
        } );
}

void Harness::stopCameraMonitor()
{
    if( CameraMonitor::m_smThread.joinable() )
    {
        m_shutdown = true;
        CameraMonitor::m_smThread.join();
        cameraImage() = {};
    }
}

const std::string &Harness::cameraMonitorName() const
{
    return CameraMonitor::m_shmimName;
}

int Harness::cameraMonitorPriority() const
{
    return CameraMonitor::m_smThreadPrio;
}

int Harness::combinedMonitorPriority() const
{
    return CombinedMonitor::m_smThreadPrio;
}

void Harness::properties()
{
    createStandardIndiToggleSw( m_indiP_enable, "enable" );
    createStandardIndiRequestSw( m_indiP_reacquire, "reacquire" );
    m_indiP_settings = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_settings.setDevice( configName() );
    m_indiP_settings.setName( "settings" );
}

void Harness::localDriver( const std::string &directory )
{
    m_driverInName   = directory + "/synco-input";
    m_driverOutName  = directory + "/synco-output";
    m_driverCtrlName = directory + "/synco-control";

    for( const auto &path : { m_driverInName, m_driverOutName, m_driverCtrlName } )
    {
        REQUIRE( ::mkfifo( path.c_str(), 0600 ) == 0 );
    }

    m_indiDriver = new SimulationDriver( this );
    REQUIRE( m_indiDriver->good() );
}

/// Owned files and images under a unique MILK_SHM_DIR; never attaches an instrument stream.
struct Streams
{
    std::string          directory;         ///< Unique temporary directory.
    std::string          prior;             ///< Saved process stream-directory environment.
    bool                 hadPrior{ false }; ///< Whether the old environment value existed.
    std::array<IMAGE, 4> images{};          ///< Primary, combined and two numbered destinations.

    /// Create local-only float images and initialize their pixels.
    Streams();

    /// Destroy test-owned streams and restore the environment.
    ~Streams();
};

Streams::Streams()
{
    if( const char *value = getenv( "MILK_SHM_DIR" ) )
    {
        prior    = value;
        hadPrior = true;
    }
    // ImageStreamIO caches its directory on first filename lookup. Keep one directory
    // for this test executable; individual fixtures still destroy all of their images.
    static const std::string processDirectory = []()
    {
        char        pattern[] = "/tmp/dm-synco-test-XXXXXX";
        const char *path      = mkdtemp( pattern );
        if( !path )
        {
            throw std::runtime_error( "cannot create isolated ImageStreamIO directory" );
        }
        return std::string( path );
    }();
    directory = processDirectory;
    setenv( "MILK_SHM_DIR", directory.c_str(), 1 );
    const std::array<const char *, 4> names{ "dm97disp03", "dm97disp", "dm97disp07", "dm97disp08" };
    uint32_t                          sizes[]{ 3, 3 };
    for( size_t j = 0; j < images.size(); ++j )
    {
        REQUIRE( ImageStreamIO_createIm_gpu(
                     &images[j], names[j], 2, sizes, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) == 0 );
        std::fill_n( images[j].array.F, 9, float( j ) );
    }
}

Streams::~Streams()
{
    for( const char *name : { "dm97disp07_aux", "dm97disp08_aux", "synco_test_timing" } )
    {
        IMAGE image{};
        if( ImageStreamIO_openIm( &image, name ) == 0 )
        {
            ImageStreamIO_destroyIm( &image );
        }
    }
    for( auto &image : images )
    {
        if( image.md )
        {
            ImageStreamIO_destroyIm( &image );
        }
    }
    if( hadPrior )
    {
        setenv( "MILK_SHM_DIR", prior.c_str(), 1 );
    }
    else
    {
        unsetenv( "MILK_SHM_DIR" );
    }
}

/// Fake monotonic execution clock, with controlled output acknowledgements during waits.
struct Clock
{
    static inline int64_t  time          = 1000000000; ///< Current virtual nanoseconds.
    static inline Harness *app           = nullptr;    ///< Active isolated app fixture.
    static inline bool     acknowledge   = true;       ///< Whether a synthetic combiner consumes auxiliary triggers.
    static inline bool     earlyMain     = false;      ///< Inject a primary event during the next wait.
    static inline bool     lateCopy      = false;      ///< Inject an overrun only after a destination copy starts.
    static inline bool     disableInCopy = false;      ///< Cancel between passive mapping copies.
    static inline int64_t  disableAt     = 0;          ///< Cancel in the final pre-trigger interval.
    static inline int64_t  wakeDelayNs   = 0;          ///< Deterministic scheduling delay after every requested wake.
    static inline bool     finishOutput  = false; ///< Complete an expected output whose counter is already visible.
    static inline uint64_t trigger       = 0;     ///< Last designated destination counter observed.

    /// Read deterministic elapsed work rather than host scheduler latency.
    static int64_t now();

    /// Advance to the absolute target plus wake delay and emulate one combiner publication.
    static int wait( const timespec *target /**< [in] absolute wait target */ );

    /// Connect the fake clock and choose feasible synthetic timing bounds.
    static void connect( Harness &scheduler /**< [in,out] isolated scheduler */ );

    /// Publish synthetic primary/combined events before invoking the monitor callback.
    static void primary( int64_t  observed /**< [in] Absolute virtual observation time. */,
                         uint64_t count = 1 /**< [in] Genuine publications represented by this callback. */ );
};

int64_t Clock::now()
{
    time += 100;
    if( app && ( ( disableInCopy && app->m_copyStarted ) || ( disableAt > 0 && time >= disableAt ) ) )
    {
        app->m_enabled = false;
        disableInCopy  = false;
        disableAt      = 0;
    }
    if( lateCopy && app && app->m_copyStarted )
    {
        time += 50000;
        lateCopy = false;
    }
    return time;
}

int Clock::wait( const timespec *target )
{
    time = std::max( time, int64_t( target->tv_sec ) * 1000000000 + target->tv_nsec ) + wakeDelayNs;
    if( finishOutput )
    {
        app->m_combined->image()->md->write = 0;
        finishOutput                        = false;
    }
    if( earlyMain )
    {
        ++app->m_imageStream.md->cnt0;
        earlyMain = false;
    }
    if( acknowledge && !app->m_mappings.empty() )
    {
        const auto value = app->m_mappings.front().m_destination->image()->md->cnt0;
        if( trigger != value )
        {
            trigger = value;
            ++app->m_combined->image()->md->cnt0;
        }
    }
    return 0;
}

void Clock::connect( Harness &scheduler )
{
    app                            = &scheduler;
    time                           = 1000000000;
    trigger                        = scheduler.m_mappings.front().m_counter;
    acknowledge                    = true;
    earlyMain                      = false;
    lateCopy                       = false;
    disableInCopy                  = false;
    disableAt                      = 0;
    wakeDelayNs                    = 0;
    finishOutput                   = false;
    scheduler.m_now                = now;
    scheduler.m_wait               = wait;
    scheduler.m_enabled            = true;
    scheduler.m_primaryCounter     = scheduler.m_imageStream.md->cnt0;
    scheduler.m_consumed           = scheduler.m_combined->image()->md->cnt0;
    scheduler.m_bounds.guard       = 1000;
    scheduler.m_bounds.delivery    = 1000;
    scheduler.m_bounds.observation = 1000;
    scheduler.m_bounds.early       = 5000;
    scheduler.m_bounds.lead        = 10000;
    scheduler.m_bounds.snapshot    = 3000;
    scheduler.m_bounds.copy        = 3000;
    scheduler.m_bounds.reserve     = 1000;
    scheduler.m_lockFrames         = 2;
}

void Clock::primary( int64_t observed, uint64_t count )
{
    time = observed;
    app->m_imageStream.md->cnt0 += count;
    app->m_combined->image()->md->cnt0 += count;
    REQUIRE( app->processImage( nullptr, dev::shmimT{} ) == 0 );
}

/// Publish independent stream-clock metadata before a monotonic callback without rewinding elapsed execution time.
void primaryObservation( Harness &app /**< [in,out] isolated scheduler */,
                         int64_t  observed /**< [in] monotonic time before callback entry */,
                         int64_t  publication /**< [in] independent stream-clock publication time */,
                         uint64_t count = 1 /**< [in] publications represented by this observation */ )
{
    REQUIRE( observed >= Clock::time );
    auto *metadata      = app.m_imageStream.md;
    metadata->write     = 1;
    metadata->writetime = synco::toTimespec( publication );
    metadata->cnt0 += count;
    if( app.m_combined )
    {
        app.m_combined->image()->md->cnt0 += count;
    }
    metadata->write = 0;
    Clock::time     = observed;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
}

/// Atomic clock and metadata mutation at the existing sampling read, without nondeterministic writer races.
struct PrimaryProbe
{
    static inline std::atomic<int64_t> time{ 1000000000 }; ///< Monotonic execution shared with a lock-wait worker.
    static inline IMAGE *image{ nullptr }; ///< Isolated primary whose publication bracket may be interrupted.
    static inline int    tear{ 0 };        ///< One-shot write-flag or counter change during metadata sampling.
    static inline std::promise<void> *entered{ nullptr }; ///< Optional notification that callback entry was captured.

    /// Read elapsed work, then inject one metadata change or notify the lock-wait coordinator.
    static int64_t now();

    /// Clear all test-only hooks before adopting this clock.
    static void reset();
};

int64_t PrimaryProbe::now()
{
    const int64_t value = time.fetch_add( 100 ) + 100;
    if( image && tear )
    {
        if( tear == 1 )
        {
            image->md->write = 1;
        }
        else
        {
            ++image->md->cnt0;
        }
        tear = 0;
    }
    if( entered )
    {
        auto *signal = entered;
        entered      = nullptr;
        signal->set_value();
    }
    return value;
}

void PrimaryProbe::reset()
{
    time    = 1000000000;
    image   = nullptr;
    tear    = 0;
    entered = nullptr;
}

/// Create a numeric camera acquisition report using seconds for interval statistics.
pcf::IndiProperty
cameraTiming( double fps /**< [in] Measured acquisition frequency in Hz. */,
              bool   statistics = true /**< [in] Include jitter/minimum/maximum acquisition intervals. */ )
{
    pcf::IndiProperty packet( pcf::IndiProperty::Number );
    packet.setDevice( "camwfs" );
    packet.setName( "fg_timing" );
    packet.add( pcf::IndiElement( "acq_fps" ) );
    packet["acq_fps"] = fps;
    if( statistics )
    {
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
        {
            packet.add( pcf::IndiElement( name ) );
        }
        packet["acq_jitter"] = 1.234e-6;
        packet["acq_min"]    = 490e-6;
        packet["acq_max"]    = 510e-6;
    }
    return packet;
}
/// Deterministic local capture clock, independent of external publication epochs.
struct CadenceClock
{
    static inline int64_t  time{ 1000000000 }; ///< Next local timestamp in nanoseconds.
    static inline unsigned waits{ 0 };         ///< Unexpected scheduler waits during read-only observation.

    /// Supply monotonically increasing capture/entry/completion timestamps.
    static int64_t now();

    /// Count an unexpected operational wait without sleeping or reaching any hardware.
    static int wait( const timespec *target /**< [in] unused operational deadline */ );
};

int64_t CadenceClock::now()
{
    return time++;
}

int CadenceClock::wait( const timespec *target )
{
    static_cast<void>( target );
    ++waits;
    return -1;
}

/// Prepare diagnostics without starting a scheduler, monitor, SDK or INDI connection.
void prepareCadence( Harness &app /**< [in,out] Isolated primary fixture. */ )
{
    app.attachPrimaryOnly();
    app.m_nextAttach = std::numeric_limits<int64_t>::max();
    app.m_now        = CadenceClock::now;
    app.m_primaryCadence.configure( 2, 20, app.m_primaryExcursionUs * 1e-6 );
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
}

/// Publish best-effort metadata and feed an actual optional receipt/initial hook followed by the callback.
void cadenceObservation( Harness &app /**< [in,out] Isolated primary fixture. */,
                         int64_t  receipt /**< [in] Monotonic receipt time. */,
                         int64_t  publication /**< [in] External publication nanoseconds. */,
                         uint64_t counter /**< [in] Observed primary counter. */,
                         bool     initial = false /**< [in] Initial callback without a semaphore receipt. */,
                         bool     writing = false /**< [in] Leave the image write flag set during observation. */ )
{
    auto *metadata      = app.m_imageStream.md;
    metadata->write     = 1;
    metadata->writetime = synco::toTimespec( publication );
    metadata->cnt0      = counter;
    metadata->write     = writing ? 1 : 0;
    CadenceClock::time  = receipt;
    app.shmimMonitorTiming( dev::shmimT{},
                            initial ? dev::shmimMonitorEvent::initial : dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
}

/// Seed ordered producer-clock endpoints without modifying any command pixels.
void combinedTrace( IMAGE   &image /**< [in,out] Test-owned combined stream metadata. */,
                    int64_t  primary /**< [in] Upstream publication timestamp, in its producer clock. */,
                    uint64_t counter /**< [in] Combined publication and linked primary counter. */,
                    ino_t    primaryInode /**< [in] Expected primary stream identity. */ )
{
    REQUIRE( image.md->NBproctrace >= 2 );
    REQUIRE( image.streamproctrace != nullptr );
    image.md->write        = 1;
    image.md->cnt0         = counter;
    image.md->writetime    = synco::toTimespec( primary + 18000 );
    auto &trace            = image.streamproctrace[0];
    trace                  = {};
    trace.procwrite_PID    = 4242;
    trace.trigger_inode    = primaryInode;
    trace.cnt0             = counter;
    trace.ts_procstart     = synco::toTimespec( primary + 12000 );
    trace.ts_streamupdate  = synco::toTimespec( primary + 16000 );
    auto &upstream         = image.streamproctrace[1];
    upstream               = {};
    upstream.procwrite_PID = 31337;
    upstream.cnt0          = counter * 3; // This is the primary writer's upstream counter, not its own output counter.
    upstream.ts_procstart  = synco::toTimespec( primary - 1000 );
    upstream.ts_streamupdate = synco::toTimespec( primary );
    image.md->write          = 0;
}

/// Configure a combined observer independently of the primary monitor and all scheduler resources.
void prepareCombined( Harness &app /**< [in,out] Isolated observer fixture. */,
                      Streams &streams /**< [in] Owner of the primary identity and combined stream. */ )
{
    app.attachCombinedOnly();
    app.m_now = CadenceClock::now;
    app.m_combinerCadence.configure( 2, 32 );
    app.m_primaryTraceInode.store( streams.images[0].md->inode );
    REQUIRE( app.m_primaryTraceInode.load() != 0 );
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::connected );
}

/// Observe one stable combined publication through the real app overloads.
void combinedObservation( Harness &app /**< [in,out] Isolated observer fixture. */,
                          IMAGE   &image /**< [in,out] Test-owned combined producer metadata. */,
                          int64_t  receipt /**< [in] Local monotonic receipt timestamp. */,
                          int64_t  primary /**< [in] Independent producer-clock upstream timestamp. */,
                          uint64_t counter /**< [in] Publication counter. */,
                          bool     initial = false /**< [in] Use an initial read without a semaphore receipt. */ )
{
    combinedTrace( image, primary, counter, app.m_primaryTraceInode.load() );
    CadenceClock::time = receipt;
    app.shmimMonitorTiming( syncoCombinerT{},
                            initial ? dev::shmimMonitorEvent::initial : dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoCombinerT{} ) == 0 );
}

/// Acquire a stable simulation through real monitor hooks and coherent producer metadata.
int64_t prepareSimulationTrace( Harness &app /**< [in,out] Isolated simulation fixture. */,
                                Streams &streams /**< [in,out] Owned primary and combined metadata. */ )
{
    app.attachPrimaryOnly();
    app.m_simulationEnabled = true;
    app.m_simulationWindow  = 4;
    app.m_requested         = 2;
    app.m_now               = CadenceClock::now;
    app.m_primaryCadence.configure( 2, 64 );
    app.m_combinerCadence.configure( 2, 64 );
    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::connected );
    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t sample = 0; sample < 40; ++sample )
    {
        const auto observed = 1000000000LL + int64_t( sample ) * 500000;
        cadenceObservation( app, observed, epoch + int64_t( sample ) * 500000, sample + 10 );
        combinedObservation(
            app, streams.images[1], observed + 20000, epoch + int64_t( sample ) * 500000, sample + 10 );
    }

    REQUIRE( app.m_simulation.summary().valid );
    REQUIRE( app.m_simulation.traceStatus().detected == 0 );
    return app.m_simulation.summary().lastMainNs;
}

/// Own a private completion record that never contacts a DM controller or posts command semaphores.
struct CompletionStream
{
    IMAGE m_image{}; ///< Test-owned uint64 transport, destroyed after all observer imports close.

    /// Create the fixed protocol layout inside the already-isolated ImageStreamIO directory.
    CompletionStream();

    /// Destroy only this fixture's private completion stream.
    ~CompletionStream();

    /// Publish a test record without synthesizing any instrument or command-channel semaphore event.
    void store( const MagAOX::dmCompletion::Record &record /**< [in] Exact record for the next local callback. */ );
};

CompletionStream::CompletionStream()
{
    uint32_t size = MagAOX::dmCompletion::words;
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_image, "synco_reader_done", 1, &size, _DATATYPE_UINT64, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) == 0 );
}

CompletionStream::~CompletionStream()
{
    ImageStreamIO_destroyIm( &m_image );
}

void CompletionStream::store( const MagAOX::dmCompletion::Record &record )
{
    const auto wire   = MagAOX::dmCompletion::encode( record );
    m_image.md->write = 1;
    std::copy( wire.begin(), wire.end(), m_image.array.UI64 );
    ++m_image.md->cnt0;
    m_image.md->write = 0;
}

/// Configure real shadow callbacks plus the optional completion observer without starting any worker.
void prepareReaderTiming( Harness &app /**< [in,out] Isolated shadow observer. */ )
{
    app.attachPrimaryOnly();
    app.m_simulationEnabled = true;
    app.m_simulationWindow  = 4;
    app.m_requested         = 2;
    app.m_now               = CadenceClock::now;
    app.m_wait              = CadenceClock::wait;
    app.m_primaryCadence.configure( 2, 64 );
    app.m_combinerCadence.configure( 2, 64 );
    app.m_readerTiming.configure( 2, 64 );
    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::connected );
    app.attachCompletionOnly();
}

/// Construct successful software completion with independent raw and local-clock endpoints.
MagAOX::dmCompletion::Record
readerCompletion( const IMAGE &combined /**< [in] Combined publication whose identity must match. */,
                  uint64_t     sequence /**< [in] Controller callback attempt. */,
                  int64_t      started /**< [in] Local monotonic callback entry. */,
                  int64_t      finished /**< [in] Local monotonic reader completion. */ )
{
    MagAOX::dmCompletion::Record result;
    result.session         = 123456789;
    result.epoch           = 1;
    result.sequence        = sequence;
    result.state           = MagAOX::dmCompletion::State::complete;
    result.inputInode      = combined.md->inode;
    result.inputCounter    = combined.md->cnt0;
    result.publicationSec  = combined.md->writetime.tv_sec;
    result.publicationNsec = combined.md->writetime.tv_nsec;
    result.startedNs       = started;
    result.finishedNs      = finished;
    result.outcome         = MagAOX::dmTiming::Outcome::success;
    result.publisherPid    = getpid();
    return result;
}

/// Exercise the production completion hook and callback on one test-owned serialized record.
void completionObservation(
    Harness                            &app /**< [in,out] Isolated completion observer. */,
    CompletionStream                   &stream /**< [in,out] Private record storage. */,
    const MagAOX::dmCompletion::Record &record /**< [in] Record observed by the real callback. */,
    int64_t                             observed /**< [in] Local monotonic receipt marker. */,
    bool initial = false /**< [in] Use an initial callback without a genuine receipt. */ )
{
    stream.store( record );
    CadenceClock::time = observed;
    app.shmimMonitorTiming( syncoCompletionT{},
                            initial ? dev::shmimMonitorEvent::initial : dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoCompletionT{} ) == 0 );
}

/// Serialize the full public simulation state to detect any completion-observer scheduling mutation.
std::vector<uint8_t> simulationBytes( const MagAOX::dmSyncoSimulation::Summary &summary
                                      /**< [in] Full shadow state before or after observation. */ )
{
    using MagAOX::logger::telem_dmsynco;
    telem_dmsynco::messageT message( 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, false, {}, {}, {}, {}, summary );
    flatlogs::bufferPtrT    buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    const auto *begin = reinterpret_cast<const uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    return { begin, begin + flatlogs::logHeader::msgLen( buffer ) };
}

/// \endcond

/// Verify fixed count feasibility, rounding, passive-only operation and narrow copy-window rejection.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation count remains fixed at the 500 us boundary", "[dmSyncopation]" )
{
    synco::Bounds bounds;
    bounds.delivery = 1000;
    bounds.guard    = 1000;
    bounds.early    = 5000;
    bounds.lead     = 10000;
    for( int64_t period : { 499980, 500000, 500020 } )
    {
        REQUIRE( synco::feasible( period, 3, bounds ) );
        REQUIRE( synco::maximumSteps( period, bounds ) >= 3 );
    }
    REQUIRE( synco::feasible( 500000, 1, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 0, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 17, bounds ) );
    REQUIRE_FALSE( synco::feasible( 500000, 3, synco::Bounds{} ) );
    REQUIRE( synco::copyCutoff( 1000000, synco::Bounds{} ) == 974000 );
    bounds.copy = 100000;
    REQUIRE_FALSE( synco::feasible( 500000, 3, bounds ) );
    REQUIRE( synco::classifyStep( false, 0, 100 ) == synco::StepOutcome::notStarted );
    REQUIRE( synco::classifyStep( true, 100, 100 ) == synco::StepOutcome::copiedOnTime );
    REQUIRE( synco::classifyStep( true, 101, 100 ) == synco::StepOutcome::copiedLate );
}

/// Verify independently clamped feedback, no missed-sample learning and restart/expiry record semantics.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation feedback and future record validity", "[dmSyncopation]" )
{
    int64_t advance = 0;
    for( int j = 0; j < 100; ++j )
    {
        advance = synco::feedback( advance, 2000 - advance, 0.2, 20000, true );
    }
    REQUIRE( advance == Approx( 2000 ).margin( 5 ) );
    REQUIRE( synco::feedback( advance, 1000000, 0.2, 20000, false ) == advance );
    REQUIRE( synco::feedback( 0, 1000000, 1, 20000, true ) == 20000 );
    REQUIRE( synco::feedback( 0, -1000000, 1, 20000, true ) == -20000 );
    REQUIRE( synco::feedback( 17, 1000000, 0, 20000, true ) == 17 );
    synco::Record record{};
    record[synco::version]     = 1;
    record[synco::epoch]       = 123;
    record[synco::generation]  = 1;
    record[synco::state]       = synco::acquiring;
    record[synco::steps]       = 3;
    record[synco::period]      = 500000;
    record[synco::reference]   = 1000000;
    record[synco::expires]     = 2000000;
    record[synco::cutoffs]     = 1400000;
    record[synco::targets]     = 1500000;
    record[synco::cutoffs + 1] = 1100000;
    record[synco::targets + 1] = 1166666;
    REQUIRE( synco::validRecord( record, 1000001 ) );
    REQUIRE( synco::nextStep( record, 1000001 ) == 1 );
    REQUIRE( synco::nextStep( record, 1100001 ) == 0 );
    REQUIRE_FALSE( synco::validRecord( record, 2000000 ) );
    record[synco::epoch] = 0;
    REQUIRE_FALSE( synco::validRecord( record, 1000001 ) );
}

/// Verify the aggregate survives the ordinary log-entry/schema verification path.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation telemetry is registered with log verification", "[dmSyncopation]" )
{
    using namespace MagAOX::logger;
    telem_dmsynco::messageT message(
        synco::armed, 3, 500000, 100, 2, 1, 7, 1000, 2000, { -500, 800, 100 }, { 0, 10, 20 } );
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>(
                 buffer, flatlogs::timespecX( 1, 2 ), message, flatlogs::logPrio::LOG_DEFAULT ) == 0 );
    REQUIRE( verifyLogEntry( telem_dmsynco::eventCode, buffer.get() ) );
    const auto *record = GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    REQUIRE( record->residuals_ns()->Get( 1 ) == 800 );
    REQUIRE( record->advances_ns()->Get( 2 ) == 20 );
    REQUIRE( record->delivered() == 100 );
    REQUIRE_FALSE( verifyLogEntry( telem_dmsynco::eventCode, nullptr ) );
}

/// Exercise registered configuration, invalid ranges and same-DM/alias restrictions without shared-memory writes.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation validates configuration before attaching streams", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig();
    dmSyncopation::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_requested == 3 );
    REQUIRE( app.m_cameraProperty == "fg_timing" );
    REQUIRE( app.m_cameraElement == "acq_fps" );
    REQUIRE_FALSE( app.m_enabled );
    app.m_destinations = { "dm01disp07", "dm02disp08" };
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    app.m_destinations = { "dm01disp07", "dm01disp07" };
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    app.m_destinations = { "dm01disp07", "dm01disp08" };
    app.m_requested    = 17;
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
}

/// Differentiate passive deadline outcomes, dependent cancellations and controller-reader holds.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation classifies misses and preserves reader lifetime", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    dmSyncopation::missedStep();
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_scheduleState = synco::armed;
    app.m_lastMain      = Clock::time;

    SECTION( "reader hold prevents even passive destination replacement" )
    {
        auto &mapping = app.m_mappings.front();
        std::fill_n( mapping.m_destination->data(), 9, 6.0f );
        app.m_submission     = Clock::time + 100000;
        app.m_requested      = 2;
        const int64_t cutoff = Clock::time + 100000;
        REQUIRE_FALSE( app.executeStep( 0, cutoff - 10000, cutoff ) );
        REQUIRE( Clock::time < cutoff ); // Reject the impossible hold before waiting into a future main event.
        REQUIRE( app.m_notStarted == 1 );
        REQUIRE( mapping.m_destination->data()[0] == 6.0f );
        REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
        REQUIRE( app.m_missed == 1 );
        REQUIRE( app.m_faults == 0 );
    }
    SECTION( "a late copy is detected even below a historical maximum" )
    {
        app.m_copyMax   = 1000000;
        Clock::lateCopy = true;
        REQUIRE_FALSE( app.executeStep( 0, Clock::time + 90000, Clock::time + 100000 ) );
        REQUIRE( app.m_copiedLate == 1 );
        REQUIRE( app.m_notStarted == 0 );
        REQUIRE( app.m_missed == 1 );
    }
    SECTION( "three original misses reacquire; a single miss does not count cancelled dependents" )
    {
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::armed );
        REQUIRE( app.m_missed == 1 );
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::armed );
        app.missedStep();
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_missed == 3 );
    }
}

/// Finish an in-flight passive batch on disable, suppress its trigger and wait for complete output publication.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation cancellation preserves complete batches without deadline misses", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    dmSyncopation::consume();
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_scheduleState = synco::armed;
    app.m_lastMain      = Clock::time;
    SECTION( "disable inside a passive batch retains the complete staged values" )
    {
        for( auto &mapping : app.m_mappings )
        {
            std::fill_n( mapping.m_source->data(), 9, 7.0f );
            mapping.m_source->post();
        }
        Clock::disableInCopy = true;
        REQUIRE_FALSE( app.executeStep( 0, Clock::time + 90000, Clock::time + 100000 ) );
        for( auto &mapping : app.m_mappings )
        {
            REQUIRE( mapping.m_destination->data()[0] == 7.0f );
            REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
        }
        REQUIRE( app.m_missed == 0 );
        REQUIRE( app.m_notStarted == 0 );
    }
    SECTION( "disable in the final publication interval suppresses the counter" )
    {
        const int64_t target = Clock::time + 166666;
        Clock::disableAt     = target + 200;
        REQUIRE_FALSE( app.executeStep( 1, target, Clock::time + 300000 ) );
        REQUIRE_FALSE( app.m_enabled );
        REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 0 );
        REQUIRE( app.m_missed == 0 );
    }
    SECTION( "the expected counter can precede write-flag clearing" )
    {
        ++app.m_combined->image()->md->cnt0;
        app.m_combined->image()->md->write = 1;
        Clock::finishOutput                = true;
        REQUIRE( app.consume() );
        REQUIRE( app.m_faults == 0 );
        REQUIRE( app.m_consumed == app.m_combined->image()->md->cnt0 );
    }
}

/// Reject replaced/malformed resources without publishing into a conflicting timing or DM stream.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation invalidates stream replacement and malformed timing", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    SECTION( "timing inode replacement invalidates the old publisher" )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), "synco_test_timing" );
        const std::string backup = std::string( filename ) + ".old";
        REQUIRE( rename( filename, backup.c_str() ) == 0 );
        std::ofstream replacement( filename );
        replacement << "replacement";
        replacement.close();
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_scheduleState == synco::invalid );
        std::remove( filename );
        REQUIRE( rename( backup.c_str(), filename ) == 0 );
    }
    SECTION( "changed timing layout is rejected before a record write" )
    {
        const auto count         = app.m_timing.md->cnt0;
        app.m_timing.md->size[0] = 1;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        // Restore via the original stream handle so cleanup sees the allocated extent.
        IMAGE old{};
        REQUIRE( ImageStreamIO_openIm( &old, "synco_test_timing" ) == 0 );
        REQUIRE( old.md->cnt0 == count );
        old.md->size[0] = synco::words;
        ImageStreamIO_closeIm( &old );
    }
}

/// Verify persistent source fallback and always-copy behavior preserve primary and passive destination counters.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation copies cached and unchanged commands passively", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::snapshot();
    dmSyncopation::copyBatch(0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    const auto primary     = streams.images[0].md->cnt0;
    const auto primaryTime = streams.images[0].md->writetime;
    auto      &mapping     = app.m_mappings.front();
    std::fill_n( mapping.m_source->data(), 9, 7.0f );
    mapping.m_source->post();
    REQUIRE( app.snapshot() );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) );
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    std::fill_n( mapping.m_destination->data(), 9, -1.0f );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) ); // Same staged/counter values still cause every memcpy.
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    mapping.m_source->setWrite( true );
    std::fill_n( mapping.m_source->data(), 9, 99.0f );
    REQUIRE( app.snapshot() );
    REQUIRE( app.copyBatch( Clock::time + 10000 ) );
    REQUIRE( mapping.m_destination->data()[0] == 7.0f );
    mapping.m_source->setWrite( false );
    REQUIRE( mapping.m_destination->image()->md->cnt0 == mapping.m_counter );
    REQUIRE( streams.images[0].md->cnt0 == primary );
    REQUIRE( streams.images[0].md->writetime.tv_sec == primaryTime.tv_sec );
    REQUIRE( streams.images[0].md->writetime.tv_nsec == primaryTime.tv_nsec );
}

/// Drive complete cycles with virtual waits, exact output acknowledgements and a fixed requested count.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primes step zero before issuing later triggers", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::executeStep(0, 0, 0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    for( int cycle = 0; cycle < 6; ++cycle )
    {
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        if( cycle < 2 )
        {
            REQUIRE( app.m_delivered == 0 );
        }
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_requested == 3 );
    REQUIRE( app.m_delivered == 6 );
    REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 6 );
    REQUIRE( app.m_mappings[1].m_destination->image()->md->cnt0 == 0 );
    REQUIRE( streams.images[0].md->cnt0 == 6 );
    REQUIRE( app.m_faults == 0 );
    REQUIRE( synco::validRecord( app.m_record, Clock::time ) );

    SECTION( "an early main cancels before any new counter publication" )
    {
        const auto count = app.m_mappings.front().m_destination->image()->md->cnt0;
        Clock::earlyMain = true;
        REQUIRE_FALSE( app.waitUntil( Clock::time + 20000 ) );
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == count );
    }
    SECTION( "a missing output is an immediate synchronization fault" )
    {
        Clock::acknowledge = false;
        Clock::time        = 1003000000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE( app.m_scheduleState == synco::invalid );
        REQUIRE( app.m_faults > 0 );
    }
    SECTION( "one late main cannot stretch the next safety window" )
    {
        const auto delivered = app.m_delivered;
        Clock::time          = 1004000000; // Last main was at 2.5 ms: a single 1.5 ms interval.
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE( app.m_delivered == delivered );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_scheduleState == synco::invalid );
    }
    SECTION( "unexpected destination writes invalidate and refresh resources" )
    {
        ++streams.images[2].md->cnt0;
        app.processImage( nullptr, dev::shmimT{} );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_scheduleState == synco::invalid );
    }
}

/// Keep conservative copy budgets from moving fast passive copies toward the hard latest start.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation passive feedback retains headroom with a grown copy budget", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::executeStep(0, 0, 0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds.copy   = 21000;
    app.m_requested     = 2;
    app.m_scheduleState = synco::armed;
    for( int cycle = 0; cycle < 30; ++cycle )
    {
        INFO( "passive cycle " << cycle );
        const int64_t observed = 1000000000 + cycle * 500000;
        Clock::time            = observed;
        app.m_lastMain         = observed;
        app.m_submission       = observed;
        const int64_t cutoff   = synco::copyCutoff( observed + 500000, app.m_bounds );
        REQUIRE( app.executeStep( 0, cutoff - app.m_bounds.reserve, cutoff ) );
        // The virtual memcpy is much faster than its conservative budget. Completing
        // before the latest permissible start demonstrates retained scheduling headroom.
        REQUIRE( Clock::time < cutoff - app.m_bounds.copy );
    }
    REQUIRE( app.m_notStarted == 0 );
    REQUIRE( app.m_missed == 0 );
    REQUIRE( app.m_faults == 0 );
    REQUIRE( app.m_mappings.front().m_destination->image()->md->cnt0 == 0 );
}

/// Prime and retain a feasible two-step schedule despite repeatable five-microsecond wake latency.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primes despite deterministic wake overshoot", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds          = synco::Bounds{};
    app.m_bounds.copy     = 21000;
    app.m_bounds.snapshot = 7000;
    app.m_requested       = 2;
    Clock::wakeDelayNs    = 5000;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000, false ) ) == 0 );
    REQUIRE( synco::feasible( 500000, 2, app.m_bounds ) );
    for( int cycle = 0; cycle < 40; ++cycle )
    {
        INFO( "delayed-wake cycle " << cycle );
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        if( cycle >= 10 )
        {
            REQUIRE( app.m_primed );
            REQUIRE( app.m_scheduleState == synco::armed );
        }
    }
    REQUIRE( app.m_requested == 2 );
    REQUIRE( app.m_delivered >= 30 );
    REQUIRE( app.m_notStarted == 0 );
    REQUIRE( app.m_missed == 0 );
    REQUIRE( app.m_faults == 0 );
}

/// Drain duplicate callback notifications without inventing a new main observation during reacquisition.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation ignores queued primary notifications around a counter jump", "[dmSyncopation][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested = 2;
    ++streams.images[0].md->cnt0;
    ++streams.images[1].md->cnt0;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    const int64_t firstObservation = app.m_lastMain;
    REQUIRE( firstObservation > 0 );
    const auto firstRecord = app.m_timing.md->cnt0;
    for( int duplicate = 0; duplicate < 5; ++duplicate )
    {
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE( app.m_lastMain == firstObservation );
        REQUIRE( app.m_clean == 0 );
        REQUIRE( app.m_faults == 0 );
        REQUIRE( app.m_delivered == 0 );
        REQUIRE( app.m_timing.md->cnt0 == firstRecord );
    }

    Clock::time = 1000500000;
    streams.images[0].md->cnt0 += 3;
    streams.images[1].md->cnt0 += 3;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    REQUIRE( app.m_primaryCounter == streams.images[0].md->cnt0 );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_lastMain == 0 );
    REQUIRE_FALSE( app.m_primed );
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::primary )] == 1 );
    const std::string lastFault     = app.m_lastFaultReason;
    const int64_t     lastFaultTime = app.m_lastFaultNs;
    REQUIRE( lastFault != "none" );
    REQUIRE( lastFaultTime > 0 );
    const auto invalidRecord = app.m_timing.md->cnt0;
    for( int duplicate = 0; duplicate < 5; ++duplicate )
    {
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE( app.m_lastMain == 0 );
        REQUIRE( app.m_clean == 0 );
        REQUIRE( app.m_faults == 1 );
        REQUIRE( app.m_delivered == 0 );
        REQUIRE( app.m_timing.md->cnt0 == invalidRecord );
    }

    for( int cycle = 2; cycle < 12; ++cycle )
    {
        Clock::time = 1000000000 + cycle * 500000;
        ++streams.images[0].md->cnt0;
        ++streams.images[1].md->cnt0;
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_delivered > 0 );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( std::string( app.m_reason ) != lastFault );
    REQUIRE( std::string( app.m_lastFaultReason ) == lastFault );
    REQUIRE( app.m_lastFaultNs == lastFaultTime );
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::primary )] == 1 );
}

/// Validate measured acquisition reports, optional seconds-to-microseconds diagnostics and custom property names.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation consumes measured camera acquisition reports", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    dmSyncopation::loadConfigImpl(mx::app::appConfigurator());
    #endif
    // clang-format on
    Harness app;
    Clock::app  = nullptr;
    Clock::time = 1000000000;
    app.m_now   = Clock::now;
    REQUIRE( app.m_cameraJitterUs == -1 );
    REQUIRE( app.m_cameraMinUs == -1 );
    REQUIRE( app.m_cameraMaxUs == -1 );
    auto packet = cameraTiming( 2000.08 );
    REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
    REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
    REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
    REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
    REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
    const int64_t acceptedTime = app.m_cameraTime.load();
    REQUIRE( acceptedTime > 0 );

    SECTION( "a subsequent FPS-only report clears obsolete optional statistics" )
    {
        REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 1900, false ) ) == 0 );
        REQUIRE( app.m_cameraFps.load() == 1900 );
        REQUIRE( app.m_cameraTime.load() > acceptedTime );
        REQUIRE( app.m_cameraJitterUs == -1 );
        REQUIRE( app.m_cameraMinUs == -1 );
        REQUIRE( app.m_cameraMaxUs == -1 );
    }
    SECTION( "the camera reciprocal is rounded once without callback-rate smoothing" )
    {
        app.updatePeriod( Clock::time, 0, true );
        REQUIRE( app.m_period == std::llround( 1e9 / 2000.08 ) );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    SECTION( "numeric acquisition fields allow surrounding whitespace" )
    {
        packet["acq_fps"]    = " 2000.08 \t";
        packet["acq_jitter"] = " 1.234e-6\n";
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
        REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
        REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
        REQUIRE( app.m_cameraTime.load() > acceptedTime );
    }
    SECTION( "invalid required and optional values reject the entire report" )
    {
        for( double bad : { 0.0,
                            -1.0,
                            100001.0,
                            std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity() } )
        {
            packet["acq_fps"] = bad;
            INFO( "acq_fps=" << packet["acq_fps"].getValue() );
            REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
            REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
            REQUIRE( app.m_cameraTime.load() == acceptedTime );
        }
        for( const auto &bad : { "", "invalid", "2000Hz", "nan", "inf", "1e999" } )
        {
            packet["acq_fps"] = bad;
            INFO( "acq_fps=" << bad );
            REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
            REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
            REQUIRE( app.m_cameraTime.load() == acceptedTime );
        }
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
        {
            for( double bad :
                 { -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
            {
                packet       = cameraTiming( 1900 );
                packet[name] = bad;
                INFO( name << '=' << packet[name].getValue() );
                REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
                REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
                REQUIRE( app.m_cameraTime.load() == acceptedTime );
                REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
                REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
                REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
            }
        }
        for( const auto &name : { "acq_jitter", "acq_min", "acq_max" } )
        {
            for( const auto &bad : { "", "invalid", "1e-6 trailing", "nan", "inf", "1e999" } )
            {
                packet       = cameraTiming( 1900 );
                packet[name] = bad;
                INFO( name << '=' << bad );
                REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
                REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
                REQUIRE( app.m_cameraTime.load() == acceptedTime );
                REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
                REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
                REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
            }
        }
        packet            = cameraTiming( 1900 );
        packet["acq_min"] = 1e-3;
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        REQUIRE( app.m_cameraTime.load() == acceptedTime );
    }
    SECTION( "missing acquisition rate and mismatched property identities are rejected" )
    {
        pcf::IndiProperty missing( pcf::IndiProperty::Number );
        missing.setDevice( "camwfs" );
        missing.setName( "fg_timing" );
        missing.add( pcf::IndiElement( "acq_jitter" ) );
        missing["acq_jitter"] = 1e-6;
        REQUIRE( app.setCallBack_m_indiP_camera( missing ) < 0 );
        packet.setDevice( "other_camera" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.setDevice( "camwfs" );
        packet.setName( "fps" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        pcf::IndiProperty text( pcf::IndiProperty::Text );
        text.setDevice( "camwfs" );
        text.setName( "fg_timing" );
        text.add( pcf::IndiElement( "acq_fps" ) );
        text["acq_fps"] = "2000";
        REQUIRE( app.setCallBack_m_indiP_camera( text ) < 0 );
        REQUIRE( app.m_cameraTime.load() == acceptedTime );
    }
    SECTION( "configured camera property and element select the received measurement" )
    {
        app.setupConfig();
        char path[] = "/tmp/dm-synco-camera-XXXXXX";
        int  fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );
        {
            std::ofstream settings( path );
            settings << "[tracking]\ncameraProperty=measurement\ncameraElement=rate\n";
        }
        app.config.readConfig( path );
        std::remove( path );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_cameraProperty == "measurement" );
        REQUIRE( app.m_cameraElement == "rate" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.setName( "measurement" );
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) < 0 );
        packet.add( pcf::IndiElement( "rate" ) );
        packet["rate"] = 1750.0;
        REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
        REQUIRE( app.m_cameraFps.load() == 1750 );
    }
}

/// Accept an INDI acquisition report while a timed cycle owns its scheduling mutex.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera callback does not wait for a timed cycle", "[dmSyncopation][camera][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    #endif
    // clang-format on
    Harness app;
    Clock::app                          = nullptr;
    Clock::time                         = 1000000000;
    app.m_now                           = Clock::now;
    auto                         packet = cameraTiming( 2000 );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         response  = std::async( std::launch::async,
                                [&]()
                                {
                                    return app.setCallBack_m_indiP_camera( packet );
                                } );
    const auto                   completed = response.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock(); // Release even with the old blocking callback before checking or joining the worker.
    const int result = response.get();
    REQUIRE( completed == std::future_status::ready );
    REQUIRE( result == 0 );
    REQUIRE( app.m_cameraFps.load() == 2000 );
    REQUIRE( app.m_cameraTime.load() == 1000000100 );
    REQUIRE( app.m_cameraJitterUs == Approx( 1.234 ) );
    REQUIRE( app.m_cameraMinUs == Approx( 490 ) );
    REQUIRE( app.m_cameraMaxUs == Approx( 510 ) );
}

/// Keep a coherent cached acquisition report if its mailbox is busy and consume the next report without refreshing its
/// age.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation timed cycle never waits for the camera mailbox", "[dmSyncopation][camera][regression]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::readCamera();
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Harness app;
    Clock::app  = nullptr;
    Clock::time = 1000000000;
    app.m_now   = Clock::now;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    app.readCamera();
    const auto cached = app.m_cameraReport;
    REQUIRE( cached.m_fps == 2000 );
    REQUIRE( cached.m_received == 1000000100 );
    REQUIRE( cached.m_jitterUs == Approx( 1.234 ) );
    REQUIRE( cached.m_minUs == Approx( 490 ) );
    REQUIRE( cached.m_maxUs == Approx( 510 ) );

    Clock::time          = 2000000000;
    auto packet          = cameraTiming( 1600 );
    packet["acq_jitter"] = 2e-6;
    packet["acq_min"]    = 600e-6;
    packet["acq_max"]    = 650e-6;
    REQUIRE( app.setCallBack_m_indiP_camera( packet ) == 0 );
    const int64_t                received = app.m_cameraTime.load();
    std::unique_lock<std::mutex> mailbox( app.m_cameraMutex );
    auto                         response  = std::async( std::launch::async,
                                [&]()
                                {
                                    std::lock_guard<std::mutex> cycle( app.m_cycleMutex );
                                    app.readCamera();
                                    app.updatePeriod( 2000000200, 0, true );
                                } );
    const auto                   completed = response.wait_for( std::chrono::milliseconds( 500 ) );
    mailbox.unlock(); // Release before assertions or joining so a blocking implementation cannot hang the suite.
    response.get();
    REQUIRE( completed == std::future_status::ready );
    REQUIRE( app.m_cameraReport.m_fps == cached.m_fps );
    REQUIRE( app.m_cameraReport.m_received == cached.m_received );
    REQUIRE( app.m_cameraReport.m_jitterUs == cached.m_jitterUs );
    REQUIRE( app.m_cameraReport.m_minUs == cached.m_minUs );
    REQUIRE( app.m_cameraReport.m_maxUs == cached.m_maxUs );
    REQUIRE( app.m_period == 500000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );

    Clock::time += 1000000;
    app.readCamera();
    REQUIRE( app.m_cameraReport.m_fps == 1600 );
    REQUIRE( app.m_cameraReport.m_received == received );
    REQUIRE( app.m_cameraReport.m_received < Clock::time );
    REQUIRE( app.m_cameraReport.m_jitterUs == Approx( 2 ) );
    REQUIRE( app.m_cameraReport.m_minUs == Approx( 600 ) );
    REQUIRE( app.m_cameraReport.m_maxUs == Approx( 650 ) );
    app.updatePeriod( Clock::time, 0, true );
    REQUIRE( app.m_period == 625000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
}

/// Use the acquisition period as a stable reference without learning callback-to-callback phase jitter.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera period survives callback jitter", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds    = synco::Bounds{};
    app.m_requested = 2;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    const std::array<int64_t, 4> jitter{ 0, 8000, 0, -8000 };
    for( int cycle = 0; cycle < 80; ++cycle )
    {
        INFO( "jitter cycle " << cycle );
        Clock::primary( 1000000000 + cycle * 500000 + jitter[cycle % jitter.size()] );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    REQUIRE( app.m_faults == 0 );
    REQUIRE( app.m_primed );
    REQUIRE( app.m_delivered > 60 );
    REQUIRE( app.m_bounds.early == 20000 );
    REQUIRE( app.m_bounds.observation == 5000 );
}

/// Reject an isolated late observation without replacing the reference period or creating a follow-on fault.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation isolated phase outlier does not become the next period", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds      = synco::Bounds{};
    app.m_requested   = 2;
    const bool camera = GENERATE( false, true );
    if( camera )
    {
        REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000 ) ) == 0 );
    }
    for( int cycle = 0; cycle < 40; ++cycle )
    {
        Clock::primary( 1000000000 + cycle * 500000 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_faults == 0 );
    const int64_t late = 1000000000 + 39 * 500000 + 525000;
    Clock::primary( late );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_period == 500000 );
    REQUIRE( app.m_faultInterval == 525000 );
    REQUIRE( app.m_faultPeriod == 500000 );
    REQUIRE( app.m_faultInterval - app.m_faultPeriod == 25000 );
    REQUIRE_FALSE( app.m_primed );
    for( int cycle = 1; cycle <= 8; ++cycle )
    {
        Clock::primary( late + cycle * 500000 );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( app.m_faults == 1 );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_scheduleState == synco::armed );
    REQUIRE( app.m_faultInterval == 525000 );
    REQUIRE( app.m_faultPeriod == 500000 );
}

/// Attribute a faulting callback interval to independent publication timing and preserve the same-event diagnostic
/// latch.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation separates primary publication and observation intervals", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::readPrimaryTiming(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested  = 2;
    app.m_lockFrames = 100;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2000, false ) ) == 0 );
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000123456789LL;
    primaryObservation( app, observed, wall );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_primaryTiming.m_gap == -1 );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    const auto previousRuntime = app.m_previousCallbackRuntime;
    const auto previousEnd     = app.m_previousCallbackEnd;

    const int64_t publicationInterval =
        GENERATE( int64_t( 500000 ), int64_t( 555000 ), int64_t( 600000 ), int64_t( 0 ) );
    const int64_t faultPublication = publicationInterval ? wall + 500000 + publicationInterval : 0;
    primaryObservation( app, observed + 1055000, faultPublication );
    REQUIRE( app.m_faults == 1 );
    REQUIRE( app.m_faultInterval == 555000 );
    REQUIRE( app.m_faultPeriod == 500000 );
    REQUIRE( app.m_faultTiming.m_counter == app.m_primaryCounter );
    REQUIRE( app.m_faultTiming.m_publication == ( publicationInterval ? publicationInterval : -1 ) );
    if( publicationInterval )
    {
        REQUIRE( app.m_faultInterval - app.m_faultTiming.m_publication == 555000 - publicationInterval );
    }
    REQUIRE( app.m_faultTiming.m_previousRuntime == previousRuntime );
    REQUIRE( app.m_faultTiming.m_gap == observed + 1055100 - previousEnd );
    REQUIRE( app.m_faultTiming.m_sampleDelay >= 0 );
    REQUIRE( app.m_faultTiming.m_sampleDelay < 10000 );
    const auto fault = app.m_faultTiming;
    for( int miss = 0; miss < 3; ++miss )
    {
        app.missedStep();
    }
    app.m_lockFrames = 2;
    for( int cycle = 1; cycle <= 6; ++cycle )
    {
        primaryObservation( app,
                            observed + 1055000 + cycle * 500000,
                            wall + 500000 + ( publicationInterval ? publicationInterval : 500000 ) + cycle * 500000 );
        REQUIRE( app.m_faultTiming.m_counter == fault.m_counter );
        REQUIRE( app.m_faultTiming.m_publication == fault.m_publication );
        REQUIRE( app.m_faultTiming.m_previousRuntime == fault.m_previousRuntime );
        REQUIRE( app.m_faultTiming.m_gap == fault.m_gap );
        REQUIRE( app.m_faultTiming.m_sampleDelay == fault.m_sampleDelay );
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_faults == 1 );

    streams.images[2].md->write = 1;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    streams.images[2].md->write = 0;
    REQUIRE( app.m_faultCounts[size_t( Harness::Fault::resource )] == 1 );
    REQUIRE( app.m_faultTiming.m_counter == 0 );
    REQUIRE( app.m_faultTiming.m_publication == -1 );
    REQUIRE( app.m_faultTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_faultTiming.m_gap == -1 );
    REQUIRE( app.m_faultTiming.m_sampleDelay == -1 );
}

/// Reject invalid timestamps and publications changed inside the metadata bracket without bridging the invalid sample.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation rejects incoherent primary publication metadata", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::readPrimaryTiming(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );

    PrimaryProbe::reset();
    PrimaryProbe::time  = Clock::time + 10000;
    PrimaryProbe::image = &app.m_imageStream;
    app.m_now           = PrimaryProbe::now;
    auto *metadata      = app.m_imageStream.md;
    metadata->cnt0 += 1;
    metadata->writetime = synco::toTimespec( wall + 1000000 );
    const auto expected = metadata->cnt0;
    SECTION( "a writer is still publishing" )
    {
        metadata->write = 1;
    }
    SECTION( "the supplied observation counter no longer matches" )
    {
        ++metadata->cnt0;
    }
    SECTION( "the timestamp is absent" )
    {
        metadata->writetime = { 0, 0 };
    }
    SECTION( "negative seconds are invalid" )
    {
        metadata->writetime.tv_sec = -1;
    }
    SECTION( "negative nanoseconds are invalid" )
    {
        metadata->writetime.tv_nsec = -1;
    }
    SECTION( "nanoseconds must be normalized" )
    {
        metadata->writetime.tv_nsec = 1000000000;
    }
    SECTION( "nanosecond conversion must fit a signed diagnostic value" )
    {
        metadata->writetime.tv_sec = INT64_MAX / 1000000000 + 1;
    }
    SECTION( "write begins after the timestamp was copied" )
    {
        PrimaryProbe::tear = 1;
    }
    SECTION( "the counter changes after the timestamp was copied" )
    {
        PrimaryProbe::tear = 2;
    }
    app.readPrimaryTiming( PrimaryProbe::time.load(), expected, true );
    REQUIRE( app.m_primaryTiming.m_counter == expected );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    REQUIRE( app.m_previousPrimaryPublication == 0 );
    REQUIRE( app.m_primaryTiming.m_sampleDelay >= 0 );

    metadata->write      = 0;
    app.m_primaryCounter = metadata->cnt0;
    ++metadata->cnt0;
    metadata->writetime = synco::toTimespec( wall + 1500000 );
    app.readPrimaryTiming( PrimaryProbe::time.load(), metadata->cnt0, true );
    REQUIRE( app.m_primaryTiming.m_publication == -1 );
    app.m_primaryCounter = metadata->cnt0;
    ++metadata->cnt0;
    metadata->writetime = synco::toTimespec( wall + 2000000 );
    app.readPrimaryTiming( PrimaryProbe::time.load(), metadata->cnt0, true );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    PrimaryProbe::reset();
}

/// Start new publication baselines after counter discontinuities or a resource epoch change.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation never associates publication intervals across primary discontinuities",
           "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::allocate(dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    primaryObservation( app, observed + 500000, wall + 500000 );
    REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    SECTION( "a skipped counter adopts a new stable publication baseline" )
    {
        primaryObservation( app, observed + 2000000, wall + 2000000, 3 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == wall + 2000000 );
        primaryObservation( app, observed + 2500000, wall + 2500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "a reset counter cannot preserve the former timestamp association" )
    {
        app.m_imageStream.md->cnt0        = 0;
        app.m_combined->image()->md->cnt0 = 0;
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == wall + 1000000 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "unsigned wrap is a discontinuity even though adding one wraps too" )
    {
        app.m_primaryCounter       = UINT64_MAX;
        app.m_imageStream.md->cnt0 = UINT64_MAX;
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_counter == 0 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "equal or backwards timestamps establish a new baseline without a usable interval" )
    {
        const int64_t adjustment = GENERATE( int64_t( 0 ), int64_t( -500000 ) );
        const int64_t restarted  = wall + 500000 + adjustment;
        primaryObservation( app, observed + 1000000, restarted );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_previousPrimaryPublication == restarted );
        primaryObservation( app, observed + 1500000, restarted + 500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
    SECTION( "reallocation discards callback and publication baselines" )
    {
        REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
        REQUIRE( app.m_previousPrimaryPublication == 0 );
        REQUIRE( app.m_previousCallbackRuntime == -1 );
        REQUIRE( app.m_previousCallbackEnd == 0 );
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
        REQUIRE( app.m_primaryTiming.m_gap == -1 );
    }
    SECTION( "a replaced resource breaks the old epoch before reattachment" )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), "synco_test_timing" );
        const std::string backup = std::string( filename ) + ".old";
        REQUIRE( rename( filename, backup.c_str() ) == 0 );
        {
            std::ofstream replacement( filename );
            replacement << "replacement";
        }
        REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
        REQUIRE_FALSE( app.m_resources );
        REQUIRE( app.m_previousPrimaryPublication == 0 );
        REQUIRE( std::remove( filename ) == 0 );
        REQUIRE( rename( backup.c_str(), filename ) == 0 );
        primaryObservation( app, observed + 1000000, wall + 1000000 );
        REQUIRE( app.m_resources );
        REQUIRE( app.m_primaryTiming.m_publication == -1 );
        primaryObservation( app, observed + 1500000, wall + 1500000 );
        REQUIRE( app.m_primaryTiming.m_publication == 500000 );
    }
}

/// Record every completed callback body while ignoring duplicate posts as publication observations.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation records callback completion across early and scheduled returns",
           "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::processPrimary(0);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_requested        = 2;
    app.m_enabled          = false;
    const int64_t observed = Clock::time + 1000000;
    const int64_t wall     = 1700000000000000000LL;
    primaryObservation( app, observed, wall );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - observed - 100 );
    REQUIRE( app.m_previousCallbackRuntime > 0 );
    const auto sample = app.m_primaryTiming;
    const auto stamp  = app.m_previousPrimaryPublication;

    Clock::time += 10000;
    const int64_t duplicateEntry = Clock::time + 100;
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - duplicateEntry );
    REQUIRE( app.m_previousCallbackRuntime > 0 );
    REQUIRE( app.m_primaryTiming.m_counter == sample.m_counter );
    REQUIRE( app.m_primaryTiming.m_publication == sample.m_publication );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == sample.m_previousRuntime );
    REQUIRE( app.m_primaryTiming.m_gap == sample.m_gap );
    REQUIRE( app.m_primaryTiming.m_sampleDelay == sample.m_sampleDelay );
    REQUIRE( app.m_previousPrimaryPublication == stamp );
    auto previousRuntime = app.m_previousCallbackRuntime;
    auto previousEnd     = app.m_previousCallbackEnd;

    app.m_enabled = true;
    for( int cycle = 1; cycle <= 5; ++cycle )
    {
        const int64_t entry = observed + cycle * 500000 + 100;
        primaryObservation( app, entry - 100, wall + cycle * 500000 );
        REQUIRE( app.m_primaryTiming.m_previousRuntime == previousRuntime );
        REQUIRE( app.m_primaryTiming.m_gap == entry - previousEnd );
        REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - entry );
        REQUIRE( app.m_previousCallbackRuntime > 0 );
        if( cycle == 1 )
        {
            REQUIRE( app.m_previousCallbackRuntime >= 50000 );
        }
        if( cycle >= 3 )
        {
            REQUIRE( app.m_previousCallbackRuntime > 400000 );
        }
        previousRuntime = app.m_previousCallbackRuntime;
        previousEnd     = app.m_previousCallbackEnd;
    }
    REQUIRE( app.m_primed );
    REQUIRE( app.m_delivered >= 2 );
    REQUIRE( app.m_faults == 0 );
}

/// Include time waiting for the cycle mutex in both callback runtime and entry-to-metadata delay.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation callback diagnostics include cycle lock delay", "[dmSyncopation][diagnostics]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    app.m_enabled = false;
    PrimaryProbe::reset();
    app.m_now = PrimaryProbe::now;
    ++app.m_imageStream.md->cnt0;
    app.m_imageStream.md->writetime = { 1700000000, 0 };
    std::promise<void> started;
    auto               entered = started.get_future();
    PrimaryProbe::entered      = &started;
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         response = std::async( std::launch::async,
                                [&]()
                                {
                                    return app.processImage( nullptr, dev::shmimT{} );
                                } );
    const auto                   captured = entered.wait_for( std::chrono::milliseconds( 500 ) );
    PrimaryProbe::time.fetch_add( 55000 );
    cycle.unlock(); // Release before checking the entry notification or joining the worker.
    const int result = response.get();
    REQUIRE( captured == std::future_status::ready );
    REQUIRE( result == 0 );
    REQUIRE( app.m_previousCallbackRuntime >= 55000 );
    REQUIRE( app.m_previousCallbackRuntime == app.m_previousCallbackEnd - 1000000100 );
    REQUIRE( app.m_primaryTiming.m_sampleDelay >= 55000 );
    REQUIRE( app.m_primaryTiming.m_previousRuntime == -1 );
    REQUIRE( app.m_primaryTiming.m_gap == -1 );
    PrimaryProbe::reset();
}

/// Qualify real primary-rate changes across phase resets while excluding missed counters and incoherent windows.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation fallback qualifies a coherent primary period", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage(nullptr, dev::shmimT());
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_bounds    = synco::Bounds{};
    app.m_requested = 2;
    SECTION( "a sustained genuine rate change eventually rearms at its qualified mean" )
    {
        for( int cycle = 0; cycle < 40; ++cycle )
        {
            Clock::primary( 1000000000 + cycle * 500000 );
        }
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        const int64_t last = 1000000000 + 39 * 500000;
        for( unsigned cycle = 1; cycle <= app.m_rateFrames + 8; ++cycle )
        {
            Clock::primary( last + cycle * 600000 );
            if( cycle < app.m_rateFrames )
            {
                REQUIRE( app.m_period == 500000 );
            }
        }
        REQUIRE( app.m_period == 600000 );
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        REQUIRE( app.m_primed );
        REQUIRE( app.m_scheduleState == synco::armed );
        REQUIRE( app.m_bounds.early == 20000 );
    }
    SECTION( "counter gaps cannot qualify their longer observation interval as a slow loop" )
    {
        for( unsigned cycle = 0; cycle < app.m_rateFrames + 5; ++cycle )
        {
            Clock::primary( 1000000000 + cycle * 1000000, 2 );
            REQUIRE( app.m_period == 500000 );
            REQUIRE( app.m_measuredPeriod == 0 );
            REQUIRE( std::string( app.m_periodSource ) == "nominal" );
        }
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_delivered == 0 );
    }
    SECTION( "a wide interval range cannot qualify until a complete coherent window follows" )
    {
        app.m_rateFrames = 4;
        int64_t observed = Clock::time;
        for( int interval : { 600000, 650000, 600000, 650000, 600000, 650000 } )
        {
            observed += interval;
            app.updatePeriod( observed, interval, true );
            REQUIRE( app.m_measuredPeriod == 0 );
            REQUIRE( app.m_period == 500000 );
        }
        for( int cycle = 0; cycle < 4; ++cycle )
        {
            observed += 600000;
            app.updatePeriod( observed, 600000, true );
        }
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( app.m_period == 600000 );
    }
}

/// Expire camera timing at five seconds and reacquire with the qualified primary or nominal reference.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation expires camera reports without relaxing timing bounds", "[dmSyncopation][camera]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::updatePeriod(0, 0, true);
    #endif
    // clang-format on
    Streams streams;
    Harness app;
    app.attach();
    Clock::connect( app );
    app.m_rateFrames = 4;
    REQUIRE( app.setCallBack_m_indiP_camera( cameraTiming( 2500 ) ) == 0 );
    const int64_t receipt = app.m_cameraTime.load();
    app.updatePeriod( receipt, 0, true );
    REQUIRE( app.m_period == 400000 );
    REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    const auto bounds = app.m_bounds;
    SECTION( "a future receipt is not a valid reference on this monotonic timeline" )
    {
        Clock::time = receipt - 101;
        app.updatePeriod( receipt - 1, 0, true );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "nominal" );
    }
    SECTION( "a report received after callback entry is still current under the cycle lock" )
    {
        Clock::time = receipt + 1000;
        app.updatePeriod( receipt - 1000, 0, true );
        REQUIRE( app.m_period == 400000 );
        REQUIRE( std::string( app.m_periodSource ) == "camera acquisition" );
    }
    SECTION( "a current camera report wins until expiry then falls back to nominal" )
    {
        Clock::time = receipt + 4999999899;
        app.updatePeriod( receipt + 4999999999, 0, true );
        REQUIRE( app.m_period == 400000 );
        app.m_primed        = true;
        app.m_scheduleState = synco::armed;
        app.m_lastMain      = receipt + 4999999999;
        Clock::time         = receipt + 4999999900;
        app.updatePeriod( receipt + 5000000000, 0, true );
        REQUIRE( app.m_period == 500000 );
        REQUIRE( std::string( app.m_periodSource ) == "nominal" );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_scheduleState != synco::armed );
        REQUIRE( app.m_lastMain == 0 );
        REQUIRE( app.m_faults == 0 );
    }
    SECTION( "an expired report uses an already qualified primary window" )
    {
        for( int cycle = 1; cycle <= 4; ++cycle )
        {
            app.updatePeriod( receipt + cycle * 600000, 600000, true );
        }
        REQUIRE( app.m_measuredPeriod == 600000 );
        REQUIRE( app.m_period == 400000 );
        app.m_primed        = true;
        app.m_scheduleState = synco::armed;
        Clock::time         = receipt + 4999999901;
        app.updatePeriod( receipt + 5000000001, 600000, true );
        REQUIRE( app.m_period == 600000 );
        REQUIRE( std::string( app.m_periodSource ) == "primary window" );
        REQUIRE_FALSE( app.m_primed );
        REQUIRE( app.m_faults == 0 );
    }
    REQUIRE( app.m_bounds.early == bounds.early );
    REQUIRE( app.m_bounds.observation == bounds.observation );
    REQUIRE( app.m_bounds.delivery == bounds.delivery );
    REQUIRE( app.m_bounds.keepout == bounds.keepout );
    REQUIRE( app.m_bounds.guard == bounds.guard );
}

/// Validate actual INDI callback behavior, including atomic updates and disabling before publication.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation INDI callbacks enforce disabled settings", "[dmSyncopation]" )
{
    // clang-format off
    #ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::newCallBack_m_indiP_settings(pcf::IndiProperty());
    dmSyncopation::newCallBack_m_indiP_enable(pcf::IndiProperty());
    dmSyncopation::setCallBack_m_indiP_camera(pcf::IndiProperty());
    #endif
    // clang-format on
    Harness app;
    app.properties();
    pcf::IndiProperty settings( pcf::IndiProperty::Number );
    settings.setDevice( "synco_test" );
    settings.setName( "settings" );
    settings.add( pcf::IndiElement( "steps" ) );
    settings["steps"] = 2.0;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) == 0 );
    REQUIRE( app.m_requested == 2 );
    settings["steps"] = 2.5;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    REQUIRE( app.m_requested == 2 );
    settings["steps"] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    settings["steps"] = 3.0;
    app.m_enabled     = true;
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    REQUIRE( app.m_requested == 2 );
    auto enable      = app.m_indiP_enable;
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_enable( enable ) == 0 );
    REQUIRE_FALSE( app.m_enabled );
    REQUIRE( app.m_scheduleState == synco::invalid );
    settings.setDevice( "different" );
    REQUIRE( app.newCallBack_m_indiP_settings( settings ) < 0 );
    pcf::IndiProperty fps( pcf::IndiProperty::Number );
    fps.setDevice( "camwfs" );
    fps.setName( "fg_timing" );
    fps.add( pcf::IndiElement( "acq_fps" ) );
    fps["acq_fps"] = 2000.08;
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) == 0 );
    REQUIRE( app.m_cameraFps.load() == Approx( 2000.08 ) );
    fps["acq_fps"] = std::numeric_limits<double>::infinity();
    REQUIRE( app.setCallBack_m_indiP_camera( fps ) < 0 );
}

/// Measure primary cadence while synco is disabled and its scheduler resources are unavailable.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary cadence is independent of scheduling resources", "[dmSyncopation][primaryTiming]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10, true );
    cadenceObservation( app, 1000500000, epoch + 500000, 11 );
    cadenceObservation( app, 1000900000, epoch + 1000000, 12 );
    cadenceObservation( app, 1001500000, epoch + 1500000, 13 );
    app.m_primaryCadence.update( CadenceClock::time );

    using namespace MagAOX::dmTiming;
    const auto &summary     = app.m_primaryCadence.summary();
    const auto &receipt     = summary.metrics[metricIndex( Metric::receiptInterval )];
    const auto &publication = summary.metrics[metricIndex( Metric::publicationInterval )];
    REQUIRE( receipt.valid );
    REQUIRE( publication.valid );
    CHECK( receipt.count == 2 );
    CHECK( publication.count == 3 );
    CHECK( receipt.mean == Approx( 0.0005 ) );
    CHECK( receipt.stddev == Approx( std::sqrt( 2.0 ) * 0.0001 ) );
    CHECK( publication.mean == Approx( 0.0005 ) );
    CHECK( publication.stddev == 0 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.receipts == 3 );
    CHECK( summary.total.events == 4 );
    CHECK_FALSE( summary.fgValid ); // No SDK endpoints are invented for the scheduler.
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_faults == 0 );
    CHECK( app.m_lastMain == 0 );

    REQUIRE( app.setupPrimaryCadenceINDI() == 0 );
    CHECK( app.m_indiP_primaryTiming.getElements().size() == 14 );
    CHECK( app.m_indiP_primaryTimingQuality.getElements().size() == 17 );
    CHECK( app.m_indiP_primaryTiming.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_primaryTimingQuality.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_primaryTiming["publication_count"].get<double>() == 0 );
    CHECK( app.m_indiP_primaryTiming["publication_mean"].get<double>() == -1 );
    CHECK( app.m_indiP_primaryTimingQuality["enabled"].get<double>() == 1 );
    app.updatePrimaryCadenceINDI(); // No INDI driver or tunnel is needed for a safe no-op.

    const uint64_t events = summary.total.events;
    app.m_primaryCadence.update( CadenceClock::time + 3000000000LL );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::receiptInterval )].count == 0 );
    CHECK( app.m_primaryCadence.summary().total.events == events );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::shmimMonitorTiming;
    dmSyncopation::readPrimaryCadence;
    dmSyncopation::processImage;
    dmSyncopation::setupPrimaryCadenceINDI;
    dmSyncopation::updatePrimaryCadenceINDI;
#endif
    // clang-format on
}

/// Measure both excursion orderings from real primary metadata without changing scheduler timing or bounds.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary interval pairs retain direction and ordinary successors",
           "[dmSyncopation][primaryTiming][intervalPairs]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    const int64_t     period       = app.m_period;
    const auto        earlyReserve = app.m_bounds.early;
    constexpr int64_t epoch        = 1792000000000000000LL;
    int64_t           offset       = 0;
    uint64_t          counter      = 10;
    cadenceObservation( app, 1000000000, epoch, counter );

    for( const int64_t interval :
         { 500000, 580000, 420000, 500000, 420000, 580000, 500000, 580000, 500000, 500000, 500000 } )
    {
        offset += interval;
        cadenceObservation( app, 1000000000 + offset, epoch + offset, ++counter );
    }

    app.m_primaryCadence.update( CadenceClock::time );
    const auto &pairs = app.m_primaryCadence.summary().intervalPairs;
    REQUIRE( pairs.enabled );
    CHECK( pairs.median == Approx( 0.0005 ) );
    CHECK( pairs.threshold == Approx( 20e-6 ) );
    CHECK( pairs.intervals == 11 );
    CHECK( pairs.pairs == 10 );
    CHECK( pairs.afterLate.count == 3 );
    CHECK( pairs.afterEarly.count == 2 );
    CHECK( pairs.lateEarlySum.count == 1 );
    CHECK( pairs.earlyLateSum.count == 1 );
    CHECK( pairs.afterLate.mean == Approx( ( 420e-6 + 500e-6 + 500e-6 ) / 3 ) );
    CHECK( pairs.afterEarly.mean == Approx( 540e-6 ) );
    CHECK( pairs.lateEarlySum.mean == Approx( 0.001 ) );
    CHECK( pairs.earlyLateSum.mean == Approx( 0.001 ) );
    CHECK_FALSE( pairs.lateEarlySum.valid );
    CHECK( app.m_period == period );
    CHECK( app.m_bounds.early == earlyReserve );
    CHECK( app.m_faults == 0 );
    CHECK_FALSE( app.m_enabled );

    REQUIRE( app.setupPrimaryCadenceINDI() == 0 );
    CHECK( app.m_indiP_primaryPairs.getElements().size() == 29 );
    CHECK( app.m_indiP_primaryPairs.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_primaryPairs["enabled"].get<double>() == 1 );
    CHECK( app.m_indiP_primaryPairs["threshold"].get<double>() == Approx( 20e-6 ) );
    CHECK( app.m_indiP_primaryPairs["lateEarlySum_count"].get<double>() == 0 );
    CHECK( app.m_indiP_primaryPairs["lateEarlySum_jitter"].get<double>() == -1 );
    app.updatePrimaryPairsINDI(); // No INDI driver or tunnel is created for this local test.

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::shmimMonitorTiming;
    dmSyncopation::readPrimaryCadence;
    dmSyncopation::processImage;
    dmSyncopation::setupPrimaryCadenceINDI;
    dmSyncopation::updatePrimaryPairsINDI;
#endif
    // clang-format on
}

/// Reject unusable publication metadata while retaining real receipt observations and loss boundaries.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary cadence tracks metadata quality and attachment epochs",
           "[dmSyncopation][primaryTiming]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    cadenceObservation( app, 1000500000, epoch + 500000, 11 );
    cadenceObservation( app, 1001000000, epoch + 500000, 11 ); // Duplicate notification.
    cadenceObservation( app, 1001500000, epoch + 1000000, 12 );
    cadenceObservation( app, 1002000000, epoch + 2000000, 14 ); // Counter gap.
    cadenceObservation( app, 1002500000, epoch + 2500000, 15 );
    cadenceObservation( app, 1003000000, epoch + 2400000, 16 ); // Backward external stamp.
    cadenceObservation( app, 1003500000, epoch + 2900000, 17 );
    app.m_primaryCadence.update( CadenceClock::time );

    using namespace MagAOX::dmTiming;
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 4 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].mean ==
           Approx( 0.0005 ) );
    CHECK( app.m_primaryCadence.summary().total.duplicates == 1 );
    CHECK( app.m_primaryCadence.summary().total.counterGaps == 1 );
    CHECK( app.m_primaryCadence.summary().total.receipts == 8 );

    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
    cadenceObservation( app, 1004000000, epoch + 3400000, 18 );
    cadenceObservation( app, 1004500000, epoch + 3900000, 19, false, true ); // Write flag prevents metadata acceptance.
    cadenceObservation( app, 1005000000, epoch + 4400000, 20 );
    cadenceObservation( app, 1005500000, epoch + 4900000, 21 );
    app.m_primaryCadence.update( CadenceClock::time );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::receiptInterval )].count == 3 );

    app.m_imageStream.md->writetime.tv_nsec = 1000000000;
    app.m_imageStream.md->cnt0              = 22;
    CadenceClock::time                      = 1006000000;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    app.m_imageStream.md->writetime.tv_sec  = std::numeric_limits<int64_t>::max();
    app.m_imageStream.md->writetime.tv_nsec = 1;
    app.m_imageStream.md->cnt0              = 23;
    CadenceClock::time                      = 1006500000;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, dev::shmimT{} ) == 0 );
    app.m_primaryCadence.update( CadenceClock::time );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::receiptInterval )].count == 5 );

    CadenceClock::time = 1007000000;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::discarded );
    app.m_primaryCadence.update( CadenceClock::time );
    CHECK( app.m_primaryCadence.summary().total.rejected == 1 );
    CHECK_FALSE( app.m_primaryCadence.active() );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::receiptInterval )].count == 6 );

    const auto session = app.m_primaryCadence.summary().session;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
    app.m_primaryCadence.update( CadenceClock::time );
    CHECK( app.m_primaryCadence.summary().session == session + 1 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 0 );
    CHECK( app.m_primaryCadence.summary().metrics[metricIndex( Metric::receiptInterval )].count == 0 );
    CHECK( app.m_faults == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::shmimMonitorTiming;
    dmSyncopation::readPrimaryCadence;
    dmSyncopation::processImage;
#endif
    // clang-format on
}

/// Validate the independent runtime window/cap controls, including both zero-disable modes.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary diagnostics configuration is independently bounded", "[dmSyncopation][primaryTiming]" )
{
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK( app.m_primaryCadenceWindow == 2 );
    CHECK( app.m_primaryCadenceMaxEvents == 20000 );
    CHECK( app.m_primaryExcursionUs == 20 );

    SECTION( "Configured window and capacity" )
    {
        char path[] = "/tmp/dm-synco-diagnostics-XXXXXX";
        int  fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );

        {
            std::ofstream settings( path );
            settings << "[diagnostics]\nwindowSec=0.5\nmaxEvents=64\nexcursionUs=5\n";
        }

        app.config.readConfig( path );
        std::remove( path );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        CHECK( app.m_primaryCadenceWindow == 0.5 );
        CHECK( app.m_primaryCadenceMaxEvents == 64 );
        CHECK( app.m_primaryExcursionUs == 5 );
    }

    SECTION( "Pair threshold is independently validated and zero disables only pairs" )
    {
        for( const double invalid :
             { -1.0, 1000001.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
        {
            app.m_primaryExcursionUs = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        app.m_primaryExcursionUs = 0;
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        app.m_primaryCadence.configure( 2, 20, app.m_primaryExcursionUs * 1e-6 );
        CHECK( app.m_primaryCadence.enabled() );
        CHECK_FALSE( app.m_primaryCadence.summary().intervalPairs.enabled );
    }

    SECTION( "Nonfinite and negative elapsed time are rejected" )
    {
        app.m_primaryCadenceWindow = std::numeric_limits<double>::quiet_NaN();
        CHECK( app.loadConfigImpl( app.config ) < 0 );
        app.m_primaryCadenceWindow = -1;
        CHECK( app.loadConfigImpl( app.config ) < 0 );
    }

    SECTION( "Negative event capacity is rejected before allocation" )
    {
        char path[] = "/tmp/dm-synco-diagnostics-XXXXXX";
        int  fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );

        {
            std::ofstream settings( path );
            settings << "[diagnostics]\nmaxEvents=-1\n";
        }

        app.config.readConfig( path );
        std::remove( path );
        CHECK( app.loadConfigImpl( app.config ) < 0 );
    }

    SECTION( "Either zero control disables capture without consuming the scheduler clock" )
    {
        for( const auto controls : { std::make_pair( 0.0, size_t( 20 ) ), std::make_pair( 2.0, size_t( 0 ) ) } )
        {
            app.m_primaryCadence.configure( controls.first, controls.second );
            app.m_now          = CadenceClock::now;
            CadenceClock::time = 1000000000;
            app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
            CHECK( CadenceClock::time == 1000000000 );
            CHECK_FALSE( app.m_primaryCadence.active() );
            CHECK_FALSE( app.m_primaryCadence.enabled() );
        }
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
    dmSyncopation::shmimMonitorTiming;
#endif
    // clang-format on
}

/// Keep primary telemetry available when the scheduler's complete-cycle lock is busy.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary telemetry does not wait for the scheduler lock", "[dmSyncopation][primaryTiming]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    cadenceObservation( app, 1000500000, epoch + 500000, 11 );
    cadenceObservation( app, 1001000000, epoch + 1000000, 12 );
    app.m_primaryCadence.update( CadenceClock::time );
    std::promise<void> locked;
    std::promise<void> release;
    auto               unlock = release.get_future();
    auto               owner  = std::async( std::launch::async,
                             [&]()
                             {
                                 std::lock_guard<std::mutex> lock( app.m_cycleMutex );
                                 locked.set_value();
                                 unlock.wait();
                             } );
    locked.get_future().wait();
    MagAOX::logger::telem_dmsynco::lastRecord = {};

    auto recording = std::async( std::launch::async,
                                 [&]()
                                 {
                                     return app.recordTelem( nullptr );
                                 } );

    const auto readiness = recording.wait_for( std::chrono::milliseconds( 500 ) );
    release.set_value();
    owner.get();
    const int result = recording.get();

    CHECK( readiness == std::future_status::ready );
    CHECK( result == 0 );
    CHECK( MagAOX::logger::telem_dmsynco::lastRecord.tv_sec > 0 );
    CHECK( app.m_primaryCadence.summary().total.receipts == 3 );
    CHECK( app.m_primaryCadence.summary()
               .metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )]
               .valid );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::recordTelem;
#endif
    // clang-format on
}

/// Exercise the real semaphore loop, inherited hook, callback and collector without a hard scheduling assumption.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation real monitor forwards finite primary cadence events",
           "[dmSyncopation][primaryTiming][regression]" )
{
    Streams           streams;
    Harness           app;
    constexpr int64_t epoch         = 1792000000000000000LL;
    streams.images[0].md->writetime = synco::toTimespec( epoch - 500000 );
    app.m_primaryCadence.configure( 30, 32 ); // Retain every finite post through all bounded waits and worker cleanup.
    app.startMonitor();
    auto waitEvents = [&]( uint64_t count )
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

        do
        {
            app.m_primaryCadence.update();

            if( app.m_primaryCadence.summary().total.events >= count )
            {
                return true;
            }

            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        } while( std::chrono::steady_clock::now() < deadline );

        return false;
    };

    REQUIRE( waitEvents( 1 ) );
    CHECK( app.m_primaryCadence.summary().total.synthetic == 1 );
    CHECK( app.m_primaryCadence.summary().total.receipts == 0 );
    for( uint64_t i = 0; i < 5; ++i )
    {
        auto *metadata      = streams.images[0].md;
        metadata->write     = 1;
        metadata->writetime = synco::toTimespec( epoch + int64_t( i ) * 500000 );
        ++metadata->cnt0;
        metadata->write = 0;
        REQUIRE( ImageStreamIO_sempost( &streams.images[0], -1 ) == 0 );
        REQUIRE( waitEvents( i + 2 ) );
    }

    app.stopMonitor();
    app.m_primaryCadence.update();
    const auto &summary = app.m_primaryCadence.summary();
    CHECK( summary.total.events == 6 );
    CHECK( summary.total.receipts == 5 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.dropped == 0 );
    CHECK( summary.total.rejected == 0 );
    CHECK( summary.total.invalid == 0 );
    CHECK( summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::receiptInterval )].count == 4 );
    const auto &publication =
        summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )];
    CHECK( publication.count == 5 );
    CHECK( publication.mean == Approx( 0.0005 ) );
    CHECK( publication.stddev == 0 );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::shmimMonitor::smThreadExec;
    dmSyncopation::shmimMonitorTiming;
    dmSyncopation::processImage;
#endif
    // clang-format on
}

/// Keep producer-clock trace durations separate from local receipt cadence without writing any DM command.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation observes combined trace timing without scheduler resources", "[dmSyncopation][combinerTiming]" )
{
    Streams streams;
    Harness app;
    prepareCombined( app, streams );
    constexpr int64_t epoch = 1792000000000000000LL;

    combinedObservation( app, streams.images[1], 1000000000, epoch, 10, true );
    combinedObservation( app, streams.images[1], 1000500000, epoch + 500000, 11 );
    combinedObservation( app, streams.images[1], 1000900000, epoch + 1000000, 12 );
    combinedObservation( app, streams.images[1], 1001500000, epoch + 1500000, 13 );
    app.m_combinerCadence.update( CadenceClock::time );

    using namespace MagAOX::dmCombTiming;
    const auto &summary     = app.m_combinerCadence.summary();
    const auto &publication = summary.metrics[index( Metric::publicationInterval )];
    const auto &receipt     = summary.metrics[index( Metric::receiptInterval )];
    CHECK( publication.count == 3 );
    CHECK( publication.mean == Approx( 0.0005 ) );
    CHECK( publication.stddev == 0 );
    CHECK( receipt.count == 2 );
    CHECK( receipt.mean == Approx( 0.0005 ) );
    CHECK( receipt.stddev == Approx( std::sqrt( 2.0 ) * 0.0001 ) );

    for( const auto &expected : { std::make_pair( Metric::execution, 4e-6 ),
                                  std::make_pair( Metric::publication, 2e-6 ),
                                  std::make_pair( Metric::primaryStart, 12e-6 ),
                                  std::make_pair( Metric::primaryPublication, 18e-6 ) } )
    {
        const auto &metric = summary.metrics[index( expected.first )];
        CHECK( metric.count == 3 ); // The first same-writer trace establishes a freshness baseline.
        CHECK( metric.valid );
        CHECK( metric.mean == Approx( expected.second ) );
        CHECK( metric.stddev == 0 );
    }

    CHECK( summary.total.events == 4 );
    CHECK( summary.total.receipts == 3 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.invalidMetadata == 0 );
    CHECK( summary.total.invalidTrace == 0 );
    CHECK( summary.total.unlinkedPrimary == 0 );
    CHECK( summary.total.invalidPrimary == 0 );
    CHECK( summary.writerPid == 4242 );
    CHECK( summary.triggerInode == uint64_t( app.m_primaryTraceInode.load() ) );
    CHECK( summary.primaryCounter == 13 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_combined );
    CHECK( app.m_faults == 0 );

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        for( size_t pixel = 0; pixel < 9; ++pixel )
        {
            CHECK( streams.images[image].array.F[pixel] == float( image ) );
        }
    }

    CHECK( streams.images[0].md->cnt0 == 0 );
    CHECK( streams.images[2].md->cnt0 == 0 );
    CHECK( streams.images[3].md->cnt0 == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::allocate( syncoCombinerT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
#endif
    // clang-format on
}

/// Keep combined observation and property registration independent of the scheduler's cycle mutex.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation combined diagnostics do not wait for the timed cycle", "[dmSyncopation][combinerTiming]" )
{
    Streams streams;
    Harness app;
    prepareCombined( app, streams );
    constexpr int64_t epoch = 1792000000000000000LL;
    combinedTrace( streams.images[1], epoch, 10, app.m_primaryTraceInode.load() );
    CadenceClock::time = 1000000000;
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         observation = std::async( std::launch::async,
                                   [&]()
                                   {
                                       return app.processImage( nullptr, syncoCombinerT{} );
                                   } );
    const auto                   readiness   = observation.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock(); // A blocking regression must finish before assertions or worker destruction.
    const int result = observation.get();
    CHECK( readiness == std::future_status::ready );
    CHECK( result == 0 );
    app.m_combinerCadence.update( CadenceClock::time );
    CHECK( app.m_combinerCadence.summary().total.events == 1 );
    CHECK( app.m_combinerCadence.summary().total.receipts == 1 );

    REQUIRE( app.setupCombinerCadenceINDI() == 0 );
    CHECK( app.m_indiP_combinerTiming.getElements().size() == 36 );
    CHECK( app.m_indiP_combinerQuality.getElements().size() == 25 );
    CHECK( app.m_indiP_combinerTiming.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_combinerQuality.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_combinerTiming["execution_mean"].get<double>() == -1 );
    CHECK( app.m_indiP_combinerTiming["execution_count"].get<double>() == 0 );
    app.updateCombinerCadenceINDI(); // Exercise the no-driver guard; this does not test XML serialization.

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::setupCombinerCadenceINDI;
    dmSyncopation::updateCombinerCadenceINDI;
#endif
    // clang-format on
}

/// Preserve cadence when trace linkage is unavailable and break history only on combined attachment changes.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation combined trace availability does not change scheduling", "[dmSyncopation][combinerTiming]" )
{
    Streams streams;
    Harness app;
    prepareCombined( app, streams );
    constexpr int64_t epoch = 1792000000000000000LL;
    combinedObservation( app, streams.images[1], 1000000000, epoch, 10 );

    combinedTrace( streams.images[1], epoch + 500000, 11, app.m_primaryTraceInode.load() );
    const auto traceCount             = streams.images[1].md->NBproctrace;
    streams.images[1].md->NBproctrace = 0;
    CadenceClock::time                = 1000500000;
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
    const int missingResult           = app.processImage( nullptr, syncoCombinerT{} );
    streams.images[1].md->NBproctrace = traceCount;
    REQUIRE( missingResult == 0 );

    combinedTrace( streams.images[1], epoch + 1000000, 12, app.m_primaryTraceInode.load() + 1 );
    CadenceClock::time = 1001000000;
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoCombinerT{} ) == 0 );
    combinedObservation( app, streams.images[1], 1001500000, epoch + 1500000, 13 );
    combinedObservation( app, streams.images[1], 1002000000, epoch + 2000000, 14 );
    app.m_combinerCadence.update( CadenceClock::time );

    using namespace MagAOX::dmCombTiming;
    const auto &summary = app.m_combinerCadence.summary();
    CHECK( summary.metrics[index( Metric::publicationInterval )].count == 4 );
    CHECK( summary.metrics[index( Metric::receiptInterval )].count == 4 );
    CHECK( summary.metrics[index( Metric::execution )].count == 2 );
    CHECK( summary.metrics[index( Metric::primaryStart )].count == 1 );
    CHECK( summary.total.missingTrace == 1 );
    CHECK( summary.total.unlinkedPrimary == 1 );
    CHECK( app.m_faults == 0 );
    CHECK_FALSE( app.m_resources );

    CadenceClock::time = 1002500000;
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::discarded );
    app.m_combinerCadence.update( CadenceClock::time );
    CHECK( app.m_combinerCadence.summary().total.rejected == 1 );
    CHECK( app.m_combinerCadence.summary().metrics[index( Metric::receiptInterval )].count == 5 );
    const uint64_t session = app.m_combinerCadence.summary().session;
    app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::connected );
    combinedObservation( app, streams.images[1], 1003000000, epoch + 2500000, 15 );
    app.m_combinerCadence.update( CadenceClock::time );
    CHECK( app.m_combinerCadence.summary().session == session + 1 );
    CHECK( app.m_combinerCadence.summary().metrics[index( Metric::publicationInterval )].count == 0 );
    CHECK( app.m_combinerCadence.summary().metrics[index( Metric::receiptInterval )].count == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::connected );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
#endif
    // clang-format on
}

/// Route the second monitor to streams.combined and allow the observer to be disabled without changing the scheduler.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation combined observer configuration preserves stream routing", "[dmSyncopation][combinerTiming]" )
{
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK( app.m_combinerDiagnostics );
    CHECK( app.combinedMonitorName() == "dm01disp" );
    CHECK( app.combinedMonitorSemaphore() == -1 );
    CHECK( app.combinedMonitorPriority() == 0 );
    CHECK( syncoCombinerT::configSection() == "combinerMonitor" );
    CHECK( syncoCombinerT::indiPrefix() == "comb" );
    char      path[] = "/tmp/dm-synco-combiner-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );

    {
        std::ofstream settings( path );
        settings << "[diagnostics]\ncombiner=false\n[combinerMonitor]\nshmimName=unrelated_stream\n";
    }

    app.config.readConfig( path );
    std::remove( path );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK_FALSE( app.m_combinerDiagnostics );
    CHECK( app.combinedMonitorName() == "dm01disp" );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
#endif
    // clang-format on
}

/// Exercise the second real semaphore loop without scheduling or writing any source/destination pixels.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation real combined monitor forwards process traces", "[dmSyncopation][combinerTiming][regression]" )
{
    Streams           streams;
    Harness           app;
    constexpr int64_t epoch = 1792000000000000000LL;
    app.m_primaryTraceInode.store( streams.images[0].md->inode );
    combinedTrace( streams.images[1], epoch, 10, app.m_primaryTraceInode.load() );
    app.m_combinerCadence.configure( 30, 32 );
    app.startCombinedMonitor();
    auto waitEvents = [&]( uint64_t expected )
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

        do
        {
            app.m_combinerCadence.update();

            if( app.m_combinerCadence.summary().total.events >= expected )
            {
                return true;
            }

            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        } while( std::chrono::steady_clock::now() < deadline );

        return false;
    };

    REQUIRE( waitEvents( 1 ) );

    for( uint64_t i = 1; i <= 5; ++i )
    {
        combinedTrace( streams.images[1], epoch + int64_t( i ) * 500000, 10 + i, app.m_primaryTraceInode.load() );
        REQUIRE( ImageStreamIO_sempost( &streams.images[1], -1 ) == 0 );
        REQUIRE( waitEvents( i + 1 ) );
    }

    app.stopCombinedMonitor();
    app.m_combinerCadence.update();
    using namespace MagAOX::dmCombTiming;
    const auto &summary = app.m_combinerCadence.summary();
    CHECK( summary.total.events == 6 );
    CHECK( summary.total.receipts == 5 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.dropped == 0 );
    CHECK( summary.total.rejected == 0 );
    CHECK( summary.metrics[index( Metric::publicationInterval )].count == 5 );
    CHECK( summary.metrics[index( Metric::publicationInterval )].mean == Approx( 0.0005 ) );
    CHECK( summary.metrics[index( Metric::receiptInterval )].count == 4 );
    CHECK( summary.metrics[index( Metric::execution )].count == 5 );
    CHECK( summary.metrics[index( Metric::execution )].mean == Approx( 4e-6 ) );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::shmimMonitor<dmSyncopation, syncoCombinerT>::smThreadExec;
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::receipt );
#endif
    // clang-format on
}

/// Distinguish camera acquisition excursions from publication and local observer wake timing.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera checks acquisition intervals independently", "[dmSyncopation][cameraTiming]" )
{
    Streams streams;
    Harness app;
    app.attachCameraOnly();
    app.m_now = CadenceClock::now;
    app.m_cameraCadence.configure( 2, 32, 20e-6 );
    app.m_cameraPublicationCadence.configure( 2, 32, 20e-6 );
    constexpr int64_t epoch = 1792000000000000000LL;
    const int64_t     offsets[]{ 0, 500000, 1000000, 1580000, 2000000, 2500000, 3000000 };

    for( size_t i = 0; i < std::size( offsets ); ++i )
    {
        auto *metadata      = streams.images[0].md;
        metadata->cnt0      = 10 + i;
        metadata->atime     = synco::toTimespec( epoch + offsets[i] );
        metadata->writetime = synco::toTimespec( epoch + int64_t( i ) * 700000 );
        CadenceClock::time  = 1000000000 + int64_t( i ) * 600000;
        app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::receipt );
        std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
        auto                         observation = std::async( std::launch::async,
                                       [&]()
                                       {
                                           return app.processImage( nullptr, syncoCameraT{} );
                                       } );
        const auto                   ready       = observation.wait_for( std::chrono::milliseconds( 500 ) );
        cycle.unlock();
        CHECK( observation.get() == 0 );
        CHECK( ready == std::future_status::ready );
    }

    app.m_cameraCadence.update( CadenceClock::time );
    app.m_cameraPublicationCadence.update( CadenceClock::time );
    using namespace MagAOX::dmTiming;
    const auto &summary     = app.m_cameraCadence.summary();
    const auto &acquisition = summary.metrics[metricIndex( Metric::publicationInterval )];
    CHECK( acquisition.count == 6 );
    CHECK( acquisition.mean == Approx( 0.0005 ) );
    CHECK( acquisition.min == Approx( 0.00042 ) );
    CHECK( acquisition.max == Approx( 0.00058 ) );
    CHECK( summary.metrics[metricIndex( Metric::receiptInterval )].mean == Approx( 0.0006 ) );
    CHECK( summary.intervalPairs.median == Approx( 0.0005 ) );
    CHECK( summary.intervalPairs.lateEarlySum.count == 1 );
    CHECK( summary.intervalPairs.afterLate.count == 1 );
    CHECK( summary.total.receipts == 7 );
    CHECK( summary.total.rejected == 0 );
    const auto &write = app.m_cameraPublicationCadence.summary();
    CHECK( write.metrics[metricIndex( Metric::publicationInterval )].count == 6 );
    CHECK( write.metrics[metricIndex( Metric::publicationInterval )].mean == Approx( 0.0007 ) );
    CHECK( write.metrics[metricIndex( Metric::receiptInterval )].mean == Approx( 0.0006 ) );
    CHECK( write.intervalPairs.median == Approx( 0.0007 ) );
    CHECK( write.total.rejected == 0 );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_enabled );
    CHECK( app.m_faults == 0 );

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        for( size_t pixel = 0; pixel < 9; ++pixel )
        {
            CHECK( streams.images[image].array.F[pixel] == float( image ) );
        }
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::allocate( syncoCameraT{} );
    dmSyncopation::processImage( nullptr, syncoCameraT{} );
    dmSyncopation::shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::receipt );
#endif
    // clang-format on
}

/// Reject unavailable and nonprogressing acquisition metadata and break associations across source epochs.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera checks expose incomplete observations", "[dmSyncopation][cameraTiming]" )
{
    Streams streams;
    Harness app;
    app.attachCameraOnly();
    app.m_now = CadenceClock::now;
    app.m_cameraCadence.configure( 2, 32, 20e-6 );
    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t i = 0; i < 6; ++i )
    {
        auto *metadata     = streams.images[0].md;
        metadata->cnt0     = 10 + i;
        metadata->atime    = synco::toTimespec( epoch + int64_t( i ) * 500000 );
        CadenceClock::time = 1000000000 + int64_t( i ) * 500000;

        if( i == 2 )
        {
            metadata->atime = synco::toTimespec( epoch + 500000 );
        }
        else if( i == 4 )
        {
            metadata->atime.tv_nsec = 1000000000;
        }

        app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, syncoCameraT{} ) == 0 );
    }

    app.m_cameraCadence.update( CadenceClock::time );
    using namespace MagAOX::dmTiming;
    CHECK( app.m_cameraCadence.summary().total.rejected == 2 );
    CHECK( app.m_cameraCadence.summary().metrics[metricIndex( Metric::publicationInterval )].count == 1 );
    const auto session = app.m_cameraCadence.summary().session;
    app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::connected );
    streams.images[0].md->cnt0  = 0;
    streams.images[0].md->atime = synco::toTimespec( epoch );
    CadenceClock::time += 500000;
    app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoCameraT{} ) == 0 );
    app.m_cameraCadence.update( CadenceClock::time );
    CHECK( app.m_cameraCadence.summary().session == session + 1 );
    CHECK( app.m_cameraCadence.summary().intervalPairs.intervals == 0 );
    CHECK( app.m_faults == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoCameraT{} );
    dmSyncopation::shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::connected );
#endif
    // clang-format on
}

/// Expose optional ordinary-priority camera collection and fixed bounded trace properties without starting workers.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation upstream diagnostic configuration and properties", "[dmSyncopation][pipelineTiming]" )
{
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK( app.m_cameraDiagnostics );
    CHECK_FALSE( app.m_diagnosticsOnly );
    CHECK( app.m_pipelineDiagnostics );
    CHECK( app.cameraMonitorName() == "camwfs" );
    CHECK( app.cameraMonitorPriority() == 0 );
    CHECK( syncoCameraT::configSection() == "cameraMonitor" );
    CHECK( syncoCameraT::indiPrefix() == "cam" );
    app.m_cameraCadence.configure( 2, 32, 20e-6 );
    app.m_pipelineCadence.configure( 2, 32, 20e-6 );
    REQUIRE( app.setupCadenceINDI( app.m_cameraCadence,
                                   app.m_indiP_cameraTiming,
                                   app.m_indiP_cameraQuality,
                                   app.m_indiP_cameraPairs,
                                   "camera",
                                   "acquisition_" ) == 0 );
    REQUIRE( app.setupPipelineINDI() == 0 );
    CHECK( app.m_indiP_cameraTiming.find( "acquisition_min" ) );
    CHECK( app.m_indiP_cameraTiming.find( "receipt_mean" ) );
    CHECK( app.m_indiP_cameraPairs.getElements().size() == 29 );
    CHECK( app.m_indiP_pipelineQuality.find( "invalidTrace" ) );
    CHECK( app.m_indiP_pipelineBoundary.find( "sourceInterval_mean" ) );
    CHECK( app.m_indiP_pipelineStages[0].find( "lateExecutionChange_mean" ) );
    CHECK( app.m_indiP_pipelineStages[9].getPerm() == pcf::IndiProperty::ReadOnly );
    app.updateCadenceINDI( app.m_cameraCadence, app.m_indiP_cameraTiming, app.m_indiP_cameraQuality, "acquisition_" );
    app.updatePairsINDI( app.m_cameraCadence, app.m_indiP_cameraPairs );
    app.updatePipelineINDI();

    char      path[] = "/tmp/dm-synco-upstream-config-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );

    {
        std::ofstream settings( path );
        settings << "[diagnostics]\ncamera=false\npipeline=false\nonly=true\n"
                    "[cameraMonitor]\nshmimName=other_camera\n";
    }

    app.config.readConfig( path );
    std::remove( path );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK_FALSE( app.m_cameraDiagnostics );
    CHECK_FALSE( app.m_pipelineDiagnostics );
    CHECK( app.m_diagnosticsOnly );
    CHECK_FALSE( app.m_combinerDiagnostics );
    CHECK( app.m_semaphoreNumber == -1 );
    CHECK( app.cameraMonitorName() == "other_camera" );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
    dmSyncopation::setupCadenceINDI;
    dmSyncopation::setupPipelineINDI;
    dmSyncopation::updateCadenceINDI;
    dmSyncopation::updatePairsINDI;
    dmSyncopation::updatePipelineINDI;
#endif
    // clang-format on
}

/// Forward primary receipt and session hooks into trace collection even when scheduler resources are absent.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation primary forwards independent upstream trace observations", "[dmSyncopation][pipelineTiming]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    app.m_pipelineCadence.configure( 2, 32, 20e-6 );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    cadenceObservation( app, 1000500000, epoch + 500000, 11 );
    app.m_pipelineCadence.update( CadenceClock::time );
    CHECK( app.m_pipelineCadence.summary().total.events == 2 );
    CHECK( app.m_pipelineCadence.summary().total.receipts == 2 );
    CHECK( app.m_pipelineCadence.summary().intervals == 1 );
    CHECK( app.m_pipelineCadence.summary().stageCount == 0 );
    CHECK( app.m_faults == 0 );
    CHECK_FALSE( app.m_resources );
    CadenceClock::time += 500000;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::discarded );
    app.m_pipelineCadence.update( CadenceClock::time );
    CHECK( app.m_pipelineCadence.summary().total.rejected == 1 );
    const auto session = app.m_pipelineCadence.summary().session;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::connected );
    app.m_pipelineCadence.update( CadenceClock::time );
    CHECK( app.m_pipelineCadence.summary().session == session + 1 );
    CHECK( app.m_pipelineCadence.summary().intervals == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
#endif
    // clang-format on
}

/// Exercise the camera semaphore worker without involving any AO or DM scheduler stream writes.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation real camera monitor retains acquisition timing", "[dmSyncopation][cameraTiming][regression]" )
{
    Streams           streams;
    Harness           app;
    constexpr int64_t epoch         = 1792000000000000000LL;
    streams.images[0].md->atime     = synco::toTimespec( epoch );
    streams.images[0].md->writetime = synco::toTimespec( epoch + 1000000 );
    app.m_cameraCadence.configure( 30, 32, 20e-6 );
    app.m_cameraPublicationCadence.configure( 30, 32, 20e-6 );
    app.startCameraMonitor();
    auto waitEvents = [&]( uint64_t expected )
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );

        do
        {
            app.m_cameraCadence.update();

            if( app.m_cameraCadence.summary().total.events >= expected )
            {
                return true;
            }

            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        } while( std::chrono::steady_clock::now() < deadline );

        return false;
    };
    REQUIRE( waitEvents( 1 ) );

    for( uint64_t i = 1; i <= 5; ++i )
    {
        streams.images[0].md->write     = 1;
        streams.images[0].md->atime     = synco::toTimespec( epoch + int64_t( i ) * 500000 );
        streams.images[0].md->writetime = synco::toTimespec( epoch + 1000000 + int64_t( i ) * 500000 );
        streams.images[0].md->cnt0      = i;
        streams.images[0].md->write     = 0;
        REQUIRE( ImageStreamIO_sempost( &streams.images[0], -1 ) == 0 );
        REQUIRE( waitEvents( i + 1 ) );
    }

    app.stopCameraMonitor();
    app.m_cameraCadence.update();
    app.m_cameraPublicationCadence.update();
    const auto &summary = app.m_cameraCadence.summary();
    CHECK( summary.total.events == 6 );
    CHECK( summary.total.receipts == 5 );
    CHECK( summary.total.synthetic == 1 );
    CHECK( summary.total.rejected == 0 );
    CHECK( summary.total.dropped == 0 );
    CHECK( summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )].count == 5 );
    CHECK( summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )].mean ==
           Approx( 0.0005 ) );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_enabled );

    const auto &write = app.m_cameraPublicationCadence.summary();
    CHECK( write.total.events == 6 );
    CHECK( write.total.receipts == 5 );
    CHECK( write.total.rejected == 0 );
    CHECK( write.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )].count == 5 );
    CHECK( write.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::publicationInterval )].mean ==
           Approx( 0.0005 ) );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::dev::shmimMonitor<dmSyncopation, syncoCameraT>::smThreadExec;
    dmSyncopation::processImage( nullptr, syncoCameraT{} );
#endif
    // clang-format on
}

/// Keep acquisition and write validity independent while sharing one coherent camera metadata capture.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera publication checks retain independent validity", "[dmSyncopation][cameraTiming]" )
{
    Streams streams;
    Harness app;
    app.attachCameraOnly();
    app.m_now = CadenceClock::now;
    app.m_cameraCadence.configure( 2, 32, 20e-6 );
    app.m_cameraPublicationCadence.configure( 2, 32, 20e-6 );
    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t i = 0; i < 6; ++i )
    {
        auto *metadata      = streams.images[0].md;
        metadata->cnt0      = 10 + i;
        metadata->atime     = synco::toTimespec( epoch + int64_t( i ) * 500000 );
        metadata->writetime = synco::toTimespec( epoch + 1000000000 + int64_t( i ) * 500000 );
        CadenceClock::time  = 1000000000 + int64_t( i ) * 600000;

        if( i == 2 )
        {
            metadata->writetime = synco::toTimespec( epoch + 1000500000 );
        }
        else if( i == 5 )
        {
            metadata->atime.tv_nsec = 1000000000;
        }

        app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, syncoCameraT{} ) == 0 );
    }

    app.m_cameraCadence.update( CadenceClock::time );
    app.m_cameraPublicationCadence.update( CadenceClock::time );
    using namespace MagAOX::dmTiming;
    const auto &acquisition = app.m_cameraCadence.summary();
    const auto &publication = app.m_cameraPublicationCadence.summary();
    CHECK( acquisition.total.rejected == 1 );
    CHECK( publication.total.rejected == 1 );
    CHECK( acquisition.metrics[metricIndex( Metric::publicationInterval )].count == 4 );
    CHECK( publication.metrics[metricIndex( Metric::publicationInterval )].count == 3 );
    CHECK( publication.metrics[metricIndex( Metric::publicationInterval )].mean == Approx( 0.0005 ) );
    CHECK( publication.metrics[metricIndex( Metric::receiptInterval )].mean == Approx( 0.0006 ) );
    const auto session = publication.session;
    app.shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::connected );
    app.m_cameraCadence.update( CadenceClock::time );
    app.m_cameraPublicationCadence.update( CadenceClock::time );
    CHECK( app.m_cameraCadence.summary().session == session + 1 );
    CHECK( app.m_cameraPublicationCadence.summary().session == session + 1 );
    CHECK( app.m_cameraPublicationCadence.summary().intervalPairs.intervals == 0 );
    REQUIRE( app.setupCadenceINDI( app.m_cameraPublicationCadence,
                                   app.m_indiP_cameraWriteTiming,
                                   app.m_indiP_cameraWriteQuality,
                                   app.m_indiP_cameraWritePairs,
                                   "camera_write",
                                   "publication_" ) == 0 );
    CHECK( app.m_indiP_cameraWriteTiming.find( "publication_min" ) );
    CHECK( app.m_indiP_cameraWriteTiming.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_cameraWritePairs.getElements().size() == 29 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::cameraStamp;
    dmSyncopation::processImage( nullptr, syncoCameraT{} );
    dmSyncopation::shmimMonitorTiming( syncoCameraT{}, dev::shmimMonitorEvent::connected );
#endif
    // clang-format on
}

/// Verify the shipped isolated profile clears normal scheduler placement and camera INDI defaults.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation isolated profile configures independent ordinary priority observers",
           "[dmSyncopation][isolatedTiming]" )
{
    Harness app;
    app.setupConfig();
    const auto profile =
        std::filesystem::canonical( "/proc/self/exe" ).parent_path().parent_path() / "camera-acquWFS.conf";
    REQUIRE( std::filesystem::exists( profile ) );
    app.config.readConfig( profile.string() );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK( app.m_shmimName == "aol1_imWFS2" );
    CHECK( app.m_smThreadPrio == 0 );
    CHECK( app.m_smCpuset.empty() );
    CHECK( app.m_cameraDevice.empty() );
    CHECK( app.m_diagnosticsOnly );
    CHECK( app.m_semaphoreNumber == -1 );
    CHECK_FALSE( app.m_combinerDiagnostics );
    CHECK( app.m_cameraDiagnostics );
    CHECK( app.m_pipelineDiagnostics );
    CHECK( app.cameraMonitorName() == "camwfs" );
    CHECK( app.cameraMonitorPriority() == 0 );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
#endif
    // clang-format on
}

/// Observe an arbitrary primary without constructing DM mappings, timing streams or a scheduler cycle.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation diagnostic only mode cannot enter the DM scheduler", "[dmSyncopation][isolatedTiming]" )
{
    Streams streams;
    Harness app;
    app.attachPrimaryOnly();
    app.properties();
    app.m_diagnosticsOnly = true;
    app.m_destinations    = { "dm97disp09", "dm97disp10" };
    app.m_timingName      = "synco_isolated_timing";
    app.m_now             = CadenceClock::now;
    app.m_primaryCadence.configure( 2, 32, 20e-6 );
    app.m_pipelineCadence.configure( 2, 32, 20e-6 );
    app.m_semaphoreNumber = 3; // Automatic allocation does not need the normal fixed scheduler reader.
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_mappings.empty() );
    CHECK( app.m_timing.md == nullptr );
    auto enable      = app.m_indiP_enable;
    enable["toggle"] = pcf::IndiElement::On;
    CHECK( app.newCallBack_m_indiP_enable( enable ) < 0 );
    CHECK_FALSE( app.m_enabled );
    constexpr int64_t epoch         = 1792000000000000000LL;
    CadenceClock::time              = 1000000000;
    streams.images[0].md->writetime = synco::toTimespec( epoch );
    streams.images[0].md->cnt0      = 10;
    app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::receipt );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         observation = std::async( std::launch::async,
                                   [&]()
                                   {
                                       return app.processImage( nullptr, dev::shmimT{} );
                                   } );
    const auto                   ready       = observation.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock();
    CHECK( observation.get() == 0 );
    CHECK( ready == std::future_status::ready );
    app.m_primaryCadence.update( CadenceClock::time );
    app.m_pipelineCadence.update( CadenceClock::time );
    CHECK( app.m_primaryCadence.summary().total.receipts == 1 );
    CHECK( app.m_pipelineCadence.summary().total.receipts == 1 );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_faults == 0 );
    CHECK( app.m_mappings.empty() );
    CHECK( app.m_timing.md == nullptr );

    for( const auto *name : { "dm97disp09_aux", "dm97disp10_aux", "synco_isolated_timing" } )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), name );
        CHECK_FALSE( std::filesystem::exists( filename ) );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::allocate( dev::shmimT{} );
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::newCallBack_m_indiP_enable;
#endif
    // clang-format on
}

/// Validate startup-only simulation settings and the shipped primary/count profile before observation.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation simulation configuration validates estimator controls", "[dmSyncopation][simulation]" )
{
    Harness app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK_FALSE( app.m_simulationEnabled );
    CHECK( app.m_simulationWindow == 32 );
    CHECK( app.m_simulationCadenceWindowSec == Approx( 10 ) );
    CHECK( app.m_simulationCadenceUpdateSec == Approx( 1 ) );
    CHECK( app.m_simulationPhaseGain == Approx( 0.1 ) );
    CHECK( app.m_simulationPhaseSlewUs == Approx( 1 ) );

    SECTION( "Shipped non-default profile selects the real primary and two simulated steps" )
    {
        const auto profile =
            std::filesystem::canonical( "/proc/self/exe" ).parent_path().parent_path() / "dmtweeter_synco_sim.conf";
        REQUIRE( std::filesystem::exists( profile ) );
        app.config.readConfig( profile.string() );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        CHECK( app.m_simulationEnabled );
        CHECK( app.m_shmimName == "dm01disp03" );
        CHECK( app.m_requested == 2 );
        CHECK( app.m_combinerDiagnostics );
        CHECK_FALSE( app.m_diagnosticsOnly );
        CHECK_FALSE( app.m_enabled );
    }

    SECTION( "Startup configuration forwards cadence units independently of phase-window size" )
    {
        char path[] = "/tmp/dm-synco-slow-cadence-XXXXXX";
        int  fd     = mkstemp( path );
        REQUIRE( fd >= 0 );
        close( fd );

        {
            std::ofstream settings( path );
            settings << "[simulation]\ncadenceWindowSec=0.1\ncadenceUpdateSec=0.01\nwindow=4\n";
        }

        app.config.readConfig( path );
        std::remove( path );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        CHECK( app.m_simulationCadenceWindowSec == Approx( 0.1 ) );
        CHECK( app.m_simulationCadenceUpdateSec == Approx( 0.01 ) );
        CHECK( app.m_simulationWindow == 4 );
        REQUIRE( app.configureSimulation() );
        CHECK( app.m_simulation.summary().cadence.windowNs == 100000000 );
        CHECK( app.m_simulation.summary().cadence.updateNs == 10000000 );
    }

    SECTION( "Estimator window is bounded before allocating model storage" )
    {
        for( const unsigned invalid : { 0U, 3U, 257U, std::numeric_limits<unsigned>::max() } )
        {
            app.m_simulationWindow = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        for( const unsigned valid : { 4U, 256U } )
        {
            app.m_simulationWindow = valid;
            CHECK( app.loadConfigImpl( app.config ) == 0 );
        }
    }

    SECTION( "Phase gain must be finite and positive without exceeding one" )
    {
        for( const double invalid :
             { -1.0, 0.0, 1.01, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
        {
            app.m_simulationPhaseGain = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        app.m_simulationPhaseGain = 1;
        CHECK( app.loadConfigImpl( app.config ) == 0 );
    }

    SECTION( "Phase slew must be finite and nonnegative" )
    {
        for( const double invalid :
             { -1.0, 1000001.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
        {
            app.m_simulationPhaseSlewUs = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        app.m_simulationPhaseSlewUs = 0;
        CHECK( app.loadConfigImpl( app.config ) == 0 );
    }

    SECTION( "Slow cadence settings must fit fixed storage and a useful refresh interval" )
    {
        for( const double invalid : { -1.0,
                                      0.0,
                                      0.009,
                                      60.001,
                                      std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity() } )
        {
            app.m_simulationCadenceUpdateSec = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        app.m_simulationCadenceUpdateSec = 1;

        for( const double invalid : { -1.0,
                                      0.0,
                                      0.999,
                                      63.001,
                                      1001.0,
                                      std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::infinity() } )
        {
            app.m_simulationCadenceWindowSec = invalid;
            CHECK( app.loadConfigImpl( app.config ) < 0 );
        }

        app.m_simulationCadenceUpdateSec = 0.01;
        app.m_simulationCadenceWindowSec = 0.63;
        CHECK( app.loadConfigImpl( app.config ) == 0 );
        app.m_simulationCadenceUpdateSec = 60;
        app.m_simulationCadenceWindowSec = 1000;
        CHECK( app.loadConfigImpl( app.config ) == 0 );
    }

    SECTION( "Simulation needs combined observation and cannot use the isolated arbitrary-primary mode" )
    {
        app.m_simulationEnabled = true;
        app.m_diagnosticsOnly   = true;
        CHECK( app.loadConfigImpl( app.config ) < 0 );
    }

    SECTION( "Required observation paths cannot be disabled in simulation" )
    {
        app.m_simulationEnabled   = true;
        app.m_combinerDiagnostics = false;
        CHECK( app.loadConfigImpl( app.config ) < 0 );
        app.m_combinerDiagnostics  = true;
        app.m_primaryCadenceWindow = 0;
        CHECK( app.loadConfigImpl( app.config ) < 0 );
        app.m_primaryCadenceWindow    = 2;
        app.m_primaryCadenceMaxEvents = 0;
        CHECK( app.loadConfigImpl( app.config ) < 0 );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
#endif
    // clang-format on
}

/// Keep measured cadence refresh separate from primary freshness and publish its qualification without rounding drift.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation cadence INDI exposes independent qualification and observation ages",
           "[dmSyncopation][simulation][cadence]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Streams streams;
    Harness app;
    app.attachPrimaryOnly();
    app.m_simulationEnabled          = true;
    app.m_simulationWindow           = 4;
    app.m_simulationCadenceWindowSec = 0.1;
    app.m_simulationCadenceUpdateSec = 0.01;
    app.m_requested                  = 2;
    app.m_now                        = CadenceClock::now;
    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    app.localDriver( streams.directory );
    REQUIRE( app.setupSimulationINDI() == 0 );
    app.m_simulationSnapshot = app.m_simulation.summary();
    app.updateSimulationINDI();
    CHECK( app.m_indiP_simulationCadence["state"].get<double>() == 0 );
    CHECK( app.m_indiP_simulationCadence["observationAgeSec"].get<double>() == -1 );
    CHECK( app.m_indiP_simulationCadence["refreshAgeSec"].get<double>() == -1 );

    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t index = 0; index <= 240; ++index )
    {
        const int64_t elapsed  = int64_t( index ) * 500000 + int64_t( ( 3 * index + 2 ) / 4 );
        const int64_t observed = 1000000000 + elapsed;
        cadenceObservation( app, observed, epoch + elapsed, index + 10 );
        combinedObservation( app, streams.images[1], observed + 20000, epoch + elapsed, index + 10 );

        if( index == 4 || index == 50 || index == 240 )
        {
            app.m_simulationSnapshot = app.m_simulation.summary();
            app.updateSimulationINDI();
            const auto &summary = app.m_simulationSnapshot;
            CHECK( summary.valid );
            CHECK( app.m_indiP_simulationCadence["observationAgeSec"].get<double>() ==
                   Approx( ( summary.nowNs - summary.cadence.observationNs ) / 1e9 ) );
            CHECK( app.m_indiP_simulationCadence["windowSec"].get<double>() == Approx( 0.1 ) );
            CHECK( app.m_indiP_simulationCadence["updateSec"].get<double>() == Approx( 0.01 ) );

            if( index == 4 )
            {
                CHECK( summary.cadence.state == CadenceState::nominal );
                CHECK( app.m_indiP_simulationCadence["state"].get<double>() == 0 );
                CHECK( app.m_indiP_simulationCadence["refreshAgeSec"].get<double>() == -1 );
            }
            else
            {
                CHECK( summary.cadence.periodQ32 == uint64_t( 500000 ) * periodScale + 3 * periodScale / 4 );
                CHECK( app.m_indiP_simulationCadence["periodNs"].get<double>() == Approx( 500000.75 ) );
                CHECK( app.m_indiP_simulation["periodUs"].get<double>() == Approx( 500.00075 ) );
                CHECK( app.m_indiP_simulationCadence["spanSec"].get<double>() ==
                       Approx( summary.cadence.spanNs / 1e9 ) );
                CHECK( app.m_indiP_simulationCadence["refreshAgeSec"].get<double>() ==
                       Approx( ( summary.nowNs - summary.cadence.updatedNs ) / 1e9 ) );

                if( index == 50 )
                {
                    CHECK( summary.cadence.state == CadenceState::growing );
                    CHECK( app.m_indiP_simulationCadence["state"].get<double>() == 1 );
                    CHECK( app.m_indiP_simulationCadence["refreshAgeSec"].get<double>() > 0.004 );
                    CHECK( app.m_indiP_simulationCadence["observationAgeSec"].get<double>() < 0.0001 );
                }
                else
                {
                    CHECK( summary.cadence.state == CadenceState::full );
                    CHECK( app.m_indiP_simulationCadence["state"].get<double>() == 2 );
                    CHECK( app.m_indiP_simulationCadence["spanSec"].get<double>() >= 0.1 );
                }
            }
        }
    }

    const auto frozen = app.m_simulation.summary();
    REQUIRE( frozen.cadence.state == CadenceState::full );
    const int64_t early     = frozen.nextGridNs - 33000;
    const int64_t published = epoch + early - 1000000000;
    combinedObservation( app, streams.images[1], early, published, 251 );
    cadenceObservation( app, early + 1000, published, 251 );
    TraceBatch trace;

    while( trace.count < trace.events.size() && app.m_simulation.popOverlap( trace.events[trace.count] ) )
    {
        ++trace.count;
    }

    while( trace.estimatorCount < trace.estimator.size() &&
           app.m_simulation.popEstimator( trace.estimator[trace.estimatorCount] ) )
    {
        ++trace.estimatorCount;
    }

    trace.status          = app.m_simulation.traceStatus();
    trace.estimatorStatus = app.m_simulation.estimatorStatus();
    REQUIRE( trace.count > 0 );
    REQUIRE( trace.estimatorCount > 0 );
    CHECK( trace.events[0].periodQ32 == frozen.cadence.periodQ32 );
    CHECK( trace.events[0].periodNs == 500001 ); // Display rounds; feasibility uses the conservative floor.
    using MagAOX::logger::telem_dmsynco;
    telem_dmsynco::messageT message(
        0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, false, {}, {}, {}, {}, app.m_simulation.summary(), trace );
    flatlogs::bufferPtrT buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    const auto *decoded = MagAOX::logger::GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    REQUIRE( decoded->simulation() != nullptr );
    REQUIRE( decoded->simulation()->cadence() != nullptr );
    CHECK( decoded->simulation()->cadence()->period_q32() == frozen.cadence.periodQ32 );
    const auto *samples = decoded->simulation()->estimator_trace()->samples();
    REQUIRE( samples != nullptr );
    REQUIRE( samples->size() == trace.estimatorCount );
    const auto *last = samples->Get( samples->size() - 1 );
    REQUIRE( last->cadence() != nullptr );
    CHECK( last->cadence()->period_q32() == trace.estimator[trace.estimatorCount - 1].cadence.periodQ32 );
    CHECK( last->base_fraction_q32() == trace.estimator[trace.estimatorCount - 1].baseFractionQ32 );
    CHECK( last->committed_fraction_q32() == trace.estimator[trace.estimatorCount - 1].committedFractionQ32 );

    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::configureSimulation;
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::setupSimulationINDI;
    dmSyncopation::updateSimulationINDI;
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
}

/// Reset only on fresh material camera-rate changes, retaining measured cadence through normal FPS report noise.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation camera reports identify cadence epochs without steering the measured rate",
           "[dmSyncopation][simulation][cadence][camera]" )
{
    Harness app;
    app.m_simulationEnabled          = true;
    app.m_simulationWindow           = 4;
    app.m_simulationCadenceWindowSec = 0.1;
    app.m_simulationCadenceUpdateSec = 0.01;
    app.m_now                        = CadenceClock::now;
    CadenceClock::time               = 2000000000;
    REQUIRE( app.configureSimulation() );

    for( uint64_t index = 0; index <= 240; ++index )
    {
        const int64_t elapsed = int64_t( index ) * 500000 + int64_t( ( 3 * index + 2 ) / 4 );
        app.m_simulation.primary( index + 10, 1000000000 + elapsed, index ? 500000 : 0 );
    }

    REQUIRE( app.m_simulation.summary().valid );
    const auto before = app.m_simulation.summary();
    REQUIRE( before.cadence.state == MagAOX::dmSyncoSimulation::CadenceState::full );

    SECTION( "Normal acquisition measurement noise does not replace or reset a healthy estimator" )
    {
        for( const double rate : { 2000.002, 2001.0, 2005.0, 2010.0, 2019.0 } )
        {
            CadenceClock::time += 1000000;
            app.m_cameraReport = { rate, CadenceClock::time - 1 };
            app.updateSimulationCadence();
            CHECK( app.m_simulationCameraTime == app.m_cameraReport.m_received );
            CHECK( app.m_simulation.summary().cadence.epoch == before.cadence.epoch );
            CHECK( app.m_simulation.summary().sequence == before.sequence );
            CHECK( app.m_simulation.summary().periodNs == before.periodNs );
            CHECK( app.m_simulation.summary().cadence.periodQ32 == before.cadence.periodQ32 );
        }

        CHECK( app.m_simulationReferenceFps == Approx( 2000.002 ) );
        CadenceClock::time += 1000000;
        app.m_cameraReport = { 2021.0, CadenceClock::time - 1 };
        app.updateSimulationCadence();
        CHECK( app.m_simulation.summary().cadence.epoch > before.cadence.epoch );
        CHECK( app.m_simulationReferenceFps == 2021.0 );
    }

    SECTION( "The first materially different report resets phase and seeds the new acquisition rate" )
    {
        app.m_cameraReport = { 1500.0, CadenceClock::time - 1 };
        app.updateSimulationCadence();
        const auto changed = app.m_simulation.summary();
        CHECK( changed.cadence.epoch > before.cadence.epoch );
        CHECK_FALSE( changed.valid );
        CHECK( changed.cadence.state == MagAOX::dmSyncoSimulation::CadenceState::nominal );
        CHECK( changed.cadence.updates == 0 );
        CHECK( changed.phaseSamples == 0 );
        CHECK( changed.periodNs == 666667 );
        CHECK( app.m_simulationReferenceFps == 1500 );
        const auto accepted = app.m_simulationCameraTime;
        app.m_cameraReport  = { 3000.0, accepted }; // A replay cannot establish another camera-rate epoch.
        app.updateSimulationCadence();
        CHECK( app.m_simulation.summary().cadence.epoch == changed.cadence.epoch );
        CHECK( app.m_simulationReferenceFps == 1500 );
        REQUIRE( app.configureSimulation() );
        CHECK( app.m_simulation.summary().periodNs == 666667 );
    }

    SECTION( "A fractional configured minimum accepts a camera seed rounded to the nearest nanosecond" )
    {
        app.m_minRate      = 500.3;
        app.m_cameraReport = { 500.3, CadenceClock::time - 1 };
        app.updateSimulationCadence();
        CHECK( app.m_simulation.summary().cadence.epoch > before.cadence.epoch );
        CHECK( app.m_simulation.summary().periodNs == 1998801 );
        CHECK( app.m_simulationReferenceFps == Approx( 500.3 ) );
        CHECK( app.m_simulationCameraTime == app.m_cameraReport.m_received );
    }

    SECTION( "Stale, future, invalid and out-of-range reports cannot change a cadence epoch" )
    {
        CadenceClock::time = 10000000000;

        for( const auto report : { std::pair<double, int64_t>{ 1500, 0 },
                                   { 1500, 4000000000 },
                                   { 1500, 11000000000 },
                                   { 0, 9990000000 },
                                   { 450, 9990000000 },
                                   { 4500, 9990000000 },
                                   { std::numeric_limits<double>::quiet_NaN(), 9990000000 },
                                   { std::numeric_limits<double>::infinity(), 9990000000 } } )
        {
            app.m_cameraReport = { report.first, report.second };
            app.updateSimulationCadence();
            CHECK( app.m_simulation.summary().cadence.epoch == before.cadence.epoch );
            CHECK( app.m_simulationCameraTime == 0 );
            CHECK( app.m_simulationReferenceFps == 0 );
        }

        app.m_simulationEnabled = false;
        app.m_cameraReport      = { 1500, CadenceClock::time - 1 };
        app.updateSimulationCadence();
        CHECK( app.m_simulation.summary().cadence.epoch == before.cadence.epoch );
        CHECK( app.m_simulationCameraTime == 0 );
    }

    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::configureSimulation;
    dmSyncopation::updateSimulationCadence;
#endif
    // clang-format on
}

/// Reject operational enable requests even though startup-only simulation observes automatically while disabled.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation simulation cannot enable the operational scheduler", "[dmSyncopation][simulation]" )
{
    Harness app;
    app.properties();
    app.m_simulationEnabled = true;
    auto enable             = app.m_indiP_enable;
    enable["toggle"]        = pcf::IndiElement::On;
    CHECK( app.newCallBack_m_indiP_enable( enable ) < 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK( app.m_indiP_enable["toggle"].getSwitchState() == pcf::IndiElement::Off );
    enable["toggle"] = pcf::IndiElement::Off;
    CHECK( app.newCallBack_m_indiP_enable( enable ) == 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_mappings.empty() );
    CHECK( app.m_timing.md == nullptr );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::newCallBack_m_indiP_enable;
#endif
    // clang-format on
}

/// Exercise real monitor allocation/callbacks without destination writes, stream creation, semaphore posts or waits.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation simulation callbacks leave all command channels untouched", "[dmSyncopation][simulation]" )
{
    Streams streams;
    Harness app;
    app.attachPrimaryOnly();
    app.m_destinations      = { "dm97disp15", "dm97disp16" };
    app.m_timingName        = "synco_simulation_test_timing";
    app.m_simulationEnabled = true;
    app.m_simulationWindow  = 4;
    app.m_requested         = 2;
    app.m_now               = CadenceClock::now;
    app.m_wait              = CadenceClock::wait;
    CadenceClock::waits     = 0;

    const std::array<const char *, 3> absentStreams{
        "dm97disp15_aux", "dm97disp16_aux", "synco_simulation_test_timing" };

    for( const auto *name : absentStreams )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), name );
        REQUIRE_FALSE( std::filesystem::exists( filename ) );
    }

    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    CHECK_FALSE( app.m_simulation.summary().valid );

    std::array<std::vector<int>, 4> semaphoreValues;

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        REQUIRE( streams.images[image].semptr != nullptr );
        semaphoreValues[image].resize( streams.images[image].md->sem );

        for( size_t semaphore = 0; semaphore < semaphoreValues[image].size(); ++semaphore )
        {
            REQUIRE( sem_getvalue( streams.images[image].semptr[semaphore], &semaphoreValues[image][semaphore] ) == 0 );
        }
    }

    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t sample = 0; sample < 40; ++sample )
    {
        const auto observed = 1000000000LL + int64_t( sample ) * 500000;
        cadenceObservation( app, observed, epoch + int64_t( sample ) * 500000, sample + 10 );
        combinedObservation(
            app, streams.images[1], observed + 20000, epoch + int64_t( sample ) * 500000, sample + 10 );
    }

    const auto &summary = app.m_simulation.summary();
    CHECK( summary.enabled );
    CHECK( summary.valid );
    CHECK( summary.feasible );
    CHECK( summary.steps == 2 );
    CHECK( summary.sequence == 40 );
    CHECK( summary.periodNs == 500000 );
    CHECK( summary.periodSamples == 4 );
    CHECK( summary.phaseSamples == 4 );
    CHECK( summary.phaseCount > 0 );
    CHECK( summary.associations == 40 );
    CHECK( summary.outcomes[0].considered > 0 );
    CHECK( summary.outcomes[1].wouldRun > 0 );
    const int64_t lastMain = summary.lastMainNs;
    cadenceObservation( app, lastMain + 270000, epoch + 19770000, 50 );
    combinedObservation( app, streams.images[1], lastMain + 290000, epoch + 19770000, 50 );
    CHECK( app.m_simulation.traceStatus().detected == 1 );
    CHECK( app.m_simulation.traceStatus().queued == 1 );
    CHECK( app.m_simulation.estimatorStatus().observed == 41 );
    CHECK( app.m_simulation.estimatorStatus().triggers == 1 );
    CHECK( app.m_simulation.estimatorStatus().queued == 41 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_combined );
    CHECK( app.m_mappings.empty() );
    CHECK( app.m_timing.md == nullptr );
    CHECK( app.m_faults == 0 );
    CHECK( app.m_delivered == 0 );
    CHECK( CadenceClock::waits == 0 );
    CHECK( streams.images[0].md->cnt0 == 50 ); // Only the fixture's primary writes advanced this counter.
    CHECK( streams.images[1].md->cnt0 == 50 ); // Only the fixture's synthetic combined producer advanced it.

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        for( size_t pixel = 0; pixel < 9; ++pixel )
        {
            CHECK( streams.images[image].array.F[pixel] == float( image ) );
        }

        if( image >= 2 )
        {
            CHECK( streams.images[image].md->cnt0 == 0 );
            CHECK( streams.images[image].md->write == 0 );
        }

        for( size_t semaphore = 0; semaphore < semaphoreValues[image].size(); ++semaphore )
        {
            int value = -1;
            REQUIRE( sem_getvalue( streams.images[image].semptr[semaphore], &value ) == 0 );
            CHECK( value == semaphoreValues[image][semaphore] );
        }
    }

    for( const auto *name : absentStreams )
    {
        char filename[PATH_MAX];
        ImageStreamIO_filename( filename, sizeof( filename ), name );
        CHECK_FALSE( std::filesystem::exists( filename ) );
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::allocate( dev::shmimT{} );
    dmSyncopation::allocate( syncoCombinerT{} );
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
#endif
    // clang-format on
}

/// Preserve estimator warmup and expose rejected primary/output metadata without claiming physical delivery.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation simulation qualifies observations and exposes conditional status",
           "[dmSyncopation][simulation][regression]" )
{
    Streams streams;
    Harness app;
    app.attachPrimaryOnly();
    app.properties();
    app.m_simulationEnabled = true;
    app.m_simulationWindow  = 4;
    app.m_requested         = 2;
    app.m_now               = CadenceClock::now;
    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    REQUIRE( app.setupSimulationINDI() == 0 );
    CHECK( app.m_indiP_simulation.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulationCadence.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulationSteps.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulationStatus.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulationTrace.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulationEstimator.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_simulation.getElements().size() == 23 );
    CHECK( app.m_indiP_simulationCadence.getName() == "simulation_cadence" );
    CHECK( app.m_indiP_simulationCadence.getElements().size() == 12 );
    CHECK( app.m_indiP_simulationSteps.getElements().size() == 4 * synco::maxSteps );
    CHECK( app.m_indiP_simulationTrace.getName() == "simulation_trace" );
    CHECK( app.m_indiP_simulationTrace.getElements().size() == 7 );
    CHECK( app.m_indiP_simulationEstimator.getName() == "simulation_estimator" );
    CHECK( app.m_indiP_simulationEstimator.getElements().size() == 12 );

    for( const auto *name : { "enabled", "detected", "drained", "dropped", "queued", "capacity", "logFailed" } )
    {
        CHECK( app.m_indiP_simulationTrace.find( name ) );
    }

    for( const auto *name : { "state",
                              "epoch",
                              "periodNs",
                              "spanSec",
                              "cycles",
                              "anchors",
                              "updates",
                              "rejected",
                              "windowSec",
                              "updateSec",
                              "observationAgeSec",
                              "refreshAgeSec" } )
    {
        CHECK( app.m_indiP_simulationCadence.find( name ) );
    }

    for( const auto *name : { "enabled",
                              "observed",
                              "triggers",
                              "selected",
                              "drained",
                              "dropped",
                              "queued",
                              "capacity",
                              "logFailed",
                              "history",
                              "historyCapacity",
                              "postRemaining" } )
    {
        CHECK( app.m_indiP_simulationEstimator.find( name ) );
    }

    CHECK( app.m_indiP_simulationStatus.find( "state" ) );
    CHECK( app.m_indiP_simulationStatus.find( "policy" ) );
    CHECK( app.m_indiP_simulation.find( "phaseMinUs" ) );
    CHECK( app.m_indiP_simulation.find( "earlyReserveUs" ) );
    CHECK( app.m_indiP_simulationSteps.find( "step0WouldRun" ) );
    CHECK( app.m_indiP_simulationSteps.find( "step1Skipped" ) );
    app.updateSimulationINDI(); // The fixture has no INDI driver or network connection.

    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t sample = 0; sample < 8; ++sample )
    {
        const auto observed = 1000000000LL + int64_t( sample ) * 500000;
        cadenceObservation( app, observed, epoch + int64_t( sample ) * 500000, sample + 10 );
        combinedObservation(
            app, streams.images[1], observed + 20000, epoch + int64_t( sample ) * 500000, sample + 10 );

        if( sample < 3 )
        {
            CHECK_FALSE( app.m_simulation.summary().valid );
        }
    }

    REQUIRE( app.m_simulation.summary().valid );
    const auto before = app.m_simulation.summary();

    SECTION( "Duplicate main notifications do not add cycles or period samples" )
    {
        cadenceObservation( app, 1003600000, epoch + 3500000, 17 );
        CHECK( app.m_simulation.summary().duplicates == before.duplicates + 1 );
        CHECK( app.m_simulation.summary().sequence == before.sequence );
        CHECK( app.m_simulation.summary().periodSamples == before.periodSamples );
    }

    SECTION( "A primary being written cannot add a phase observation" )
    {
        cadenceObservation( app, 1004000000, epoch + 4000000, 18, false, true );
        CHECK( app.m_simulation.summary().invalid == before.invalid + 1 );
        CHECK( app.m_simulation.summary().sequence == before.sequence );
        streams.images[0].md->write = 0;
    }

    SECTION( "An output with another trigger inode cannot establish the current main association" )
    {
        cadenceObservation( app, 1004000000, epoch + 4000000, 18 );
        combinedTrace( streams.images[1], epoch + 4000000, 18, app.m_primaryTraceInode.load() + 1 );
        CadenceClock::time = 1004020000;
        REQUIRE( app.processImage( nullptr, syncoCombinerT{} ) == 0 );
        CHECK( app.m_simulation.summary().associations == before.associations );
        CHECK( app.m_simulation.summary().outputGaps == before.outputGaps + 1 );
    }

    SECTION( "Malformed combined trace endpoints cannot acknowledge a real main" )
    {
        cadenceObservation( app, 1004000000, epoch + 4000000, 18 );
        combinedTrace( streams.images[1], epoch + 4000000, 18, app.m_primaryTraceInode.load() );
        streams.images[1].streamproctrace[0].ts_streamupdate.tv_nsec = 1000000000;
        CadenceClock::time                                           = 1004020000;
        REQUIRE( app.processImage( nullptr, syncoCombinerT{} ) == 0 );
        CHECK( app.m_simulation.summary().associations == before.associations );
        CHECK( app.m_simulation.summary().outputGaps == before.outputGaps + 1 );
    }

    SECTION( "The reserved primary semaphore is rejected without attaching scheduler resources" )
    {
        app.m_semaphoreNumber = 9;
        CHECK( app.allocate( dev::shmimT{} ) < 0 );
        CHECK_FALSE( app.m_resources );
        CHECK_FALSE( app.m_combined );
        CHECK( app.m_mappings.empty() );
        CHECK( app.m_timing.md == nullptr );
    }

    SECTION( "An unrepresentable positive cadence update restores all settings atomically" )
    {
        const int    previousRequested = app.m_requested;
        const double previousCadence   = app.m_maxCadence;
        const auto   previousBounds    = app.m_bounds;
        auto         settings          = app.m_indiP_settings;

        for( const auto *name : { "steps", "maxCadenceHz", "keepoutUs", "guardUs", "step0LeadUs" } )
        {
            settings.add( pcf::IndiElement( name ) );
        }

        settings["steps"]        = 3.0;
        settings["maxCadenceHz"] = 1e-100; // Finite/positive, but its minimum modeled period cannot fit int64.
        settings["keepoutUs"]    = previousBounds.keepout / 1000.0 + 1;
        settings["guardUs"]      = previousBounds.guard / 1000.0 + 1;
        settings["step0LeadUs"]  = previousBounds.lead / 1000.0 + 1;
        CHECK( app.newCallBack_m_indiP_settings( settings ) < 0 );
        CHECK( app.m_requested == previousRequested );
        CHECK( app.m_maxCadence == previousCadence );

        for( const auto bound : { &synco::Bounds::keepout,
                                  &synco::Bounds::guard,
                                  &synco::Bounds::delivery,
                                  &synco::Bounds::observation,
                                  &synco::Bounds::early,
                                  &synco::Bounds::lead,
                                  &synco::Bounds::visibility,
                                  &synco::Bounds::reserve,
                                  &synco::Bounds::snapshot,
                                  &synco::Bounds::copy } )
        {
            CHECK( app.m_bounds.*bound == previousBounds.*bound );
        }
    }

    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::configureSimulation;
    dmSyncopation::allocate( dev::shmimT{} );
    dmSyncopation::newCallBack_m_indiP_settings;
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::setupSimulationINDI;
    dmSyncopation::updateSimulationINDI;
#endif
    // clang-format on
}

/// Preserve real monitor/producer context and correlate the two paths without suppressing overlap evidence.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation overlap evidence preserves monitor and producer clocks",
           "[dmSyncopation][simulation][overlapTrace]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Streams           streams;
    Harness           app;
    const int64_t     last  = prepareSimulationTrace( app, streams );
    constexpr int64_t epoch = 1792000000000000000LL;
    OverlapEvent      event;

    SECTION( "Primary callback identifies the actual active-reader overlap" )
    {
        const int64_t receipt                     = last + 270000;
        const int64_t publication                 = epoch + 19770000;
        MagAOX::logger::telem_dmsynco::lastRecord = {};
        cadenceObservation( app, receipt, publication, 50 );
        REQUIRE( app.m_simulation.popOverlap( event ) );
        CHECK( event.origin == OverlapOrigin::primary );
        CHECK( event.window == OverlapWindow::readerHold );
        CHECK( event.mainCounter == 50 );
        CHECK( event.combinedCounter == 49 );
        CHECK( event.planCounter == 49 );
        CHECK( event.lastMainCounter == 49 );
        CHECK( event.step == 1 );
        CHECK( event.lastMainNs == last );
        CHECK( event.observedNs >= event.copyEndNs );
        CHECK( event.observedNs < event.readerEndNs );
        CHECK( event.observation.streamInode == streams.images[0].md->inode );
        CHECK( event.observation.triggerInode == 0 );
        CHECK( event.observation.receiptNs == receipt );
        CHECK( event.observation.callbackNs == receipt + 1 );
        CHECK( event.observation.publicationNs == publication );
        CHECK( event.observation.primaryPublicationNs == publication );
        CHECK( event.observation.startNs == 0 );
        CHECK( event.observation.updateNs == 0 );
        CHECK( event.observation.publicationNs > event.observedNs );    // Producer epoch never becomes monotonic time.
        CHECK( MagAOX::logger::telem_dmsynco::lastRecord.tv_sec == 0 ); // Callback only queues fixed-size evidence.
    }

    SECTION( "Combined-first and primary-later evidence retain one offline correlation key" )
    {
        const int64_t publication = epoch + 19940000;
        const int64_t receipt     = last + 462000;
        combinedObservation( app, streams.images[1], receipt, publication, 50 );
        REQUIRE( app.m_simulation.popOverlap( event ) );
        CHECK( event.origin == OverlapOrigin::combined );
        CHECK( event.window == OverlapWindow::copy );
        CHECK( event.step == 0 );
        CHECK( event.mainCounter == 50 );
        CHECK( event.combinedCounter == 50 );
        CHECK( event.planCounter == 50 );
        CHECK( event.lastMainCounter == 49 );
        CHECK( event.readerEndNs == 0 );
        CHECK( event.observedNs >= event.copyBeginNs );
        CHECK( event.observedNs < event.copyEndNs );
        CHECK( event.observation.streamInode == streams.images[1].md->inode );
        CHECK( event.observation.triggerInode == streams.images[0].md->inode );
        CHECK( event.observation.receiptNs == receipt );
        CHECK( event.observation.callbackNs == receipt + 1 );
        CHECK( event.observation.publicationNs == publication + 18000 );
        CHECK( event.observation.primaryPublicationNs == 0 ); // Its primary callback has not run yet.
        CHECK( event.observation.startNs == publication + 12000 );
        CHECK( event.observation.updateNs == publication + 16000 );
        const auto combined = event;
        cadenceObservation( app, last + 465000, publication, 50 );
        REQUIRE( app.m_simulation.popOverlap( event ) );
        CHECK( event.origin == OverlapOrigin::primary );
        CHECK( event.id == combined.id + 1 );
        CHECK( event.session == combined.session );
        CHECK( event.mainCounter == combined.mainCounter );
        CHECK( event.planCounter == combined.planCounter );
        CHECK( event.step == combined.step );
        CHECK( event.mainGridNs == combined.mainGridNs );
        CHECK( event.planGridNs == combined.planGridNs );
        CHECK( event.observation.primaryPublicationNs == publication );
        CHECK( app.m_simulation.traceStatus().detected == 2 );

        combinedObservation( app, streams.images[1], last + 466000, publication, 50 );
        cadenceObservation( app, last + 467000, publication, 50 );
        CHECK( app.m_simulation.traceStatus().detected == 2 ); // Repeated notifications add no evidence.
    }

    CHECK( event.id > 0 );
    CHECK( event.session == app.m_simulation.summary().session );
    CHECK( event.periodNs == 500000 );
    CHECK( event.observation.receiptNs <= event.observation.callbackNs );
    CHECK( event.observation.callbackNs <= event.observation.lockRequestNs );
    CHECK( event.observation.lockRequestNs <= event.observation.lockAcquiredNs );
    CHECK( event.observation.lockAcquiredNs <= event.observedNs );
    CHECK_FALSE( app.m_simulation.popOverlap( event ) );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::readSimulationPrimary;
    dmSyncopation::readSimulationCombined;
#endif
    // clang-format on
}

/// Retain the original producer interval and monotonic prediction from real primary callbacks around an overlap.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation estimator history preserves callback clocks and frozen plan provenance",
           "[dmSyncopation][simulation][estimatorTrace]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Streams           streams;
    Harness           app;
    const int64_t     last   = prepareSimulationTrace( app, streams );
    constexpr int64_t epoch  = 1792000000000000000LL;
    const auto        before = app.m_simulation.estimatorStatus();
    REQUIRE( before.observed == 40 );
    CHECK( before.history == 40 );
    CHECK( before.selected == 0 );
    const auto predicted                      = app.m_simulation.summary().nextGridNs;
    MagAOX::logger::telem_dmsynco::lastRecord = {};
    cadenceObservation( app, last + 270000, epoch + 19770000, 50 );
    const auto after = app.m_simulation.estimatorStatus();
    REQUIRE( after.observed == before.observed + 1 );
    CHECK( after.triggers == 1 );
    REQUIRE( after.selected == 41 );
    CHECK( after.queued == after.selected );
    CHECK( after.drained == 0 );
    CHECK( after.dropped == 0 );
    CHECK( after.history == 41 );
    CHECK( after.postRemaining == estimatorPostSamples - 1 );
    CHECK( MagAOX::logger::telem_dmsynco::lastRecord.tv_sec == 0 );

    EstimatorSample sample;
    EstimatorSample previous;
    size_t          retained = 0;

    while( app.m_simulation.popEstimator( sample ) )
    {
        CHECK( sample.id == ++retained );
        CHECK( sample.counter == retained + 9 );
        CHECK( sample.sequence == retained );
        CHECK( sample.session == app.m_simulation.summary().session );
        CHECK( sample.observedNs < epoch ); // Producer timestamps never become monotonic observations.

        if( retained > 1 )
        {
            CHECK( sample.sequential );
            CHECK( sample.intervalAccepted );
            CHECK( sample.publicationIntervalNs == ( retained == 41 ? 270000 : 500000 ) );
            CHECK( sample.observedNs > previous.observedNs );
        }

        if( retained > 5 )
        {
            CHECK( sample.committed );
            CHECK( sample.nextCounter == sample.counter + 1 );
            CHECK( sample.sourcePlanCounter == sample.counter );
            CHECK( sample.sourceEstimatorId == previous.id );
            CHECK( sample.sourceGridNs == previous.committedGridNs );
            CHECK( sample.baseGridNs == sample.sourceGridNs + sample.periodNs );
            CHECK( sample.committedGridNs == sample.baseGridNs + sample.appliedCorrectionNs );
            CHECK( sample.phaseGain == Approx( 0.1 ) );
            CHECK( sample.phaseSlewNs == 1000 );
        }

        previous = sample;
    }

    REQUIRE( retained == 41 );
    CHECK( sample.counter == 50 );
    CHECK( sample.predictedNs == predicted );
    CHECK( sample.observedNs == app.m_simulation.summary().lastMainNs );
    CHECK( sample.periodBeforeNs == 500000 );
    CHECK( sample.periodNs == 500000 );
    OverlapEvent overlap;
    REQUIRE( app.m_simulation.popOverlap( overlap ) );
    CHECK( overlap.lastEstimatorId == sample.id - 1 );
    CHECK( overlap.planEstimatorId > 0 );
    CHECK( overlap.planEstimatorId < sample.id );
    CHECK( overlap.observation.publicationNs == epoch + 19770000 );

    cadenceObservation( app, last + 280000, epoch + 19770000, 50 );
    CHECK( app.m_simulation.estimatorStatus().observed == after.observed );
    cadenceObservation( app, last + 300000, epoch + 19800000, 51, false, true );
    CHECK( app.m_simulation.estimatorStatus().observed == after.observed );
    streams.images[0].md->write = 0;
    CHECK_FALSE( app.m_simulation.popEstimator( sample ) );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::readSimulationPrimary;
#endif
    // clang-format on
}

/// Validate real callback-produced warmup, provenance and bounded corrections through the binary telemetry reader.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation callback estimator history roundtrips through telemetry validation",
           "[dmSyncopation][simulation][estimatorTrace]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    using MagAOX::logger::telem_dmsynco;
    Streams           streams;
    Harness           app;
    const int64_t     last  = prepareSimulationTrace( app, streams );
    constexpr int64_t epoch = 1792000000000000000LL;

    for( uint64_t index = 0; index < 5; ++index )
    {
        const int64_t observed  = last + 270000 + int64_t( index ) * 500000;
        const int64_t published = epoch + 19770000 + int64_t( index ) * 500000;
        cadenceObservation( app, observed, published, index + 50 );
        combinedObservation( app, streams.images[1], observed + 20000, published, index + 50 );
    }

    TraceBatch trace;

    while( trace.count < trace.events.size() && app.m_simulation.popOverlap( trace.events[trace.count] ) )
    {
        ++trace.count;
    }

    while( trace.estimatorCount < trace.estimator.size() &&
           app.m_simulation.popEstimator( trace.estimator[trace.estimatorCount] ) )
    {
        ++trace.estimatorCount;
    }

    trace.status          = app.m_simulation.traceStatus();
    trace.estimatorStatus = app.m_simulation.estimatorStatus();
    REQUIRE( trace.count > 0 );
    REQUIRE( trace.estimatorCount == 45 );
    REQUIRE_FALSE( trace.estimator[0].committed ); // Real estimator warmup is retained with the triggering window.
    REQUIRE( trace.estimator[44].committed );
    REQUIRE( trace.estimator[44].appliedCorrectionNs < 0 );
    const auto              summary = app.m_simulation.summary();
    telem_dmsynco::messageT message( 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, false, {}, {}, {}, {}, summary, trace );
    flatlogs::bufferPtrT    buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_dmsynco>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM ) ==
             0 );
    REQUIRE( telem_dmsynco::verify( buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    const auto *decoded = MagAOX::logger::GetTelem_dmsynco_fb( flatlogs::logHeader::messageBuffer( buffer ) );
    REQUIRE( decoded->simulation() != nullptr );
    REQUIRE( decoded->simulation()->estimator_trace() != nullptr );
    const auto *samples = decoded->simulation()->estimator_trace()->samples();
    REQUIRE( samples != nullptr );
    REQUIRE( samples->size() == trace.estimatorCount );
    CHECK( samples->Get( 0 )->counter() == 10 );
    CHECK( samples->Get( 0 )->predicted_ns() == 0 );
    CHECK( samples->Get( 40 )->publication_interval_ns() == 270000 );
    CHECK( samples->Get( 44 )->observed_ns() == trace.estimator[44].observedNs );
    CHECK( samples->Get( 44 )->source_estimator_id() == trace.estimator[43].id );
    CHECK( samples->Get( 44 )->committed_grid_ns() == trace.estimator[44].committedGridNs );
    CHECK( samples->Get( 44 )->applied_correction_ns() == trace.estimator[44].appliedCorrectionNs );
    CHECK_FALSE( app.m_enabled );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::readSimulationPrimary;
    synco::Simulation::popEstimator;
    MagAOX::logger::telem_dmsynco::messageT;
    MagAOX::logger::telem_dmsynco::verify;
#endif
    // clang-format on
}

/// Keep both unlogged histories queued during cycle-lock contention and drain at most fixed telemetry batches.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation overlap telemetry drains only outside producer callbacks",
           "[dmSyncopation][simulation][overlapTrace]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Streams       streams;
    Harness       app;
    const int64_t last = prepareSimulationTrace( app, streams );

    SECTION( "A busy cycle lock neither blocks telemetry nor drains evidence" )
    {
        cadenceObservation( app, last + 270000, 1792000000019770000LL, 50 );
        REQUIRE( app.m_simulation.traceStatus().queued == 1 );
        const auto estimator = app.m_simulation.estimatorStatus();
        REQUIRE( estimator.queued == 41 );
        std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
        auto                         recording = std::async( std::launch::async,
                                     [&]()
                                     {
                                         return app.recordTelem( nullptr );
                                     } );
        const auto                   ready     = recording.wait_for( std::chrono::milliseconds( 500 ) );
        cycle.unlock();
        CHECK( recording.get() == 0 );
        CHECK( ready == std::future_status::ready );
        CHECK( app.m_simulation.traceStatus().detected == 1 );
        CHECK( app.m_simulation.traceStatus().drained == 0 );
        CHECK( app.m_simulation.traceStatus().queued == 1 );
        CHECK( app.m_simulation.estimatorStatus().drained == 0 );
        CHECK( app.m_simulation.estimatorStatus().queued == estimator.queued );
        REQUIRE( app.recordTelem( nullptr ) == 0 );
        CHECK( app.m_simulation.traceStatus().drained == 1 );
        CHECK( app.m_simulation.traceStatus().queued == 0 );
        CHECK( app.m_simulation.estimatorStatus().drained == estimator.queued );
        CHECK( app.m_simulation.estimatorStatus().queued == 0 );
        CHECK( app.m_simulationEstimatorStatus.selected == estimator.selected );
        CHECK( app.m_simulationEstimatorStatus.drained == estimator.queued );
        CHECK( app.m_simulationEstimatorStatus.queued == 0 );
    }

    SECTION( "One record cannot drain more than the shared batch capacity" )
    {
        constexpr size_t count = traceBatchCapacity + 6;

        for( size_t sample = 0; sample < count; ++sample )
        {
            app.m_simulation.combined( 50 + sample, 50, last + 465000 + int64_t( sample ) );
        }

        REQUIRE( app.m_simulation.traceStatus().queued == count );
        REQUIRE( app.recordTelem( nullptr ) == 0 );
        CHECK( app.m_simulation.traceStatus().drained == traceBatchCapacity );
        CHECK( app.m_simulation.traceStatus().queued == 6 );
        CHECK( app.m_simulationTraceStatus.detected == count );
        CHECK( app.m_simulationTraceStatus.drained == traceBatchCapacity );
        CHECK( app.m_simulationTraceStatus.queued == 6 );
        CHECK( app.m_simulationTraceStatus.capacity == traceCapacity );
        REQUIRE( app.recordTelem( nullptr ) == 0 );
        CHECK( app.m_simulationTraceStatus.drained == count );
        CHECK( app.m_simulationTraceStatus.queued == 0 );
    }

    CHECK( app.m_simulationTraceStatus.enabled );
    CHECK( app.m_simulationTraceStatus.dropped == 0 );
    CHECK( app.m_simulationTraceLogFailed == 0 );
    CHECK( app.m_simulationEstimatorLogFailed == 0 );
    CHECK( app.m_simulationTraceStatus.detected == app.m_simulationTraceStatus.drained +
                                                       app.m_simulationTraceStatus.dropped +
                                                       app.m_simulationTraceStatus.queued );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::recordTelem;
#endif
    // clang-format on
}

/// Preserve queued estimator records across model sessions and bound each ordinary-thread telemetry drain.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation estimator telemetry drains a bounded independent batch",
           "[dmSyncopation][simulation][estimatorTrace]" )
{
    using namespace MagAOX::dmSyncoSimulation;
    Streams          streams;
    Harness          app;
    const int64_t    start         = prepareSimulationTrace( app, streams );
    constexpr size_t samplesPerRun = 41;
    constexpr size_t runs          = estimatorBatchCapacity / samplesPerRun + 1;
    constexpr size_t selected      = runs * samplesPerRun;

    for( size_t run = 0; run < runs; ++run )
    {
        app.m_simulation.reset();
        const int64_t first = start + 1000000 + int64_t( run ) * 21000000;

        for( uint64_t index = 0; index < 40; ++index )
        {
            const int64_t now = first + int64_t( index ) * 500000;
            app.m_simulation.primary( index + 10, now, index ? 500000 : 0 );
            app.m_simulation.combined( index + 10, index + 10, now + 20000 );
        }

        app.m_simulation.primary( 50, first + 19770000, 270000 );
    }

    const auto pending = app.m_simulation.estimatorStatus();
    REQUIRE( pending.selected == selected );
    REQUIRE( pending.queued == selected );
    REQUIRE( pending.queued > estimatorBatchCapacity );
    CHECK( pending.triggers == runs );
    CHECK( pending.dropped == 0 );
    REQUIRE( app.recordTelem( nullptr ) == 0 );
    const auto first = app.m_simulationEstimatorStatus;
    CHECK( first.enabled );
    CHECK( first.selected == selected );
    CHECK( first.drained == estimatorBatchCapacity );
    CHECK( first.queued == selected - estimatorBatchCapacity );
    CHECK( first.capacity == estimatorTraceCapacity );
    CHECK( first.historyCapacity == estimatorHistoryCapacity );
    CHECK( first.history == samplesPerRun );
    CHECK( first.postRemaining == estimatorPostSamples - 1 );
    REQUIRE( app.recordTelem( nullptr ) == 0 );
    const auto second = app.m_simulationEstimatorStatus;
    CHECK( second.selected == selected );
    CHECK( second.drained == selected );
    CHECK( second.queued == 0 );
    CHECK( second.dropped == 0 );
    CHECK( second.logFailed == 0 );
    CHECK( second.selected == second.drained + second.dropped + second.queued );
    CHECK( app.m_simulationEstimatorLogFailed == 0 );
    CHECK( app.m_simulationTraceLogFailed == 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::recordTelem;
#endif
    // clang-format on
}

/// Leave overlap and estimator collection and their INDI interfaces absent from ordinary non-simulation operation.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation default observers do not collect overlap evidence",
           "[dmSyncopation][simulation][overlapTrace]" )
{
    Streams streams;
    Harness app;
    prepareCadence( app );
    prepareCombined( app, streams );
    REQUIRE( app.setupSimulationINDI() == 0 );
    cadenceObservation( app, 1000000000, 1792000000000000000LL, 10 );
    combinedObservation( app, streams.images[1], 1000020000, 1792000000000000000LL, 10 );
    const auto                              status = app.m_simulation.traceStatus();
    MagAOX::dmSyncoSimulation::OverlapEvent event;
    CHECK_FALSE( status.enabled );
    CHECK( status.detected == 0 );
    CHECK( status.queued == 0 );
    CHECK_FALSE( app.m_simulation.popOverlap( event ) );
    CHECK( app.m_indiP_simulationTrace.getElements().empty() );
    const auto                                 estimator = app.m_simulation.estimatorStatus();
    MagAOX::dmSyncoSimulation::EstimatorSample sample;
    CHECK_FALSE( estimator.enabled );
    CHECK( estimator.observed == 0 );
    CHECK( estimator.triggers == 0 );
    CHECK( estimator.selected == 0 );
    CHECK( estimator.queued == 0 );
    CHECK( estimator.history == 0 );
    CHECK_FALSE( app.m_simulation.popEstimator( sample ) );
    CHECK( app.m_indiP_simulationEstimator.getElements().empty() );
    CHECK( app.m_indiP_simulationCadence.getElements().empty() );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::setupSimulationINDI;
#endif
    // clang-format on
}

/// Require explicit shadow selection and an independent stream before enabling completion measurements.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion observation is optional and restricted to shadow configuration",
           "[dmSyncopation][readerTiming]" )
{
    {
        Harness defaults;
        defaults.setupConfig();
        REQUIRE( defaults.loadConfigImpl( defaults.config ) == 0 );
        CHECK( defaults.completionMonitorName().empty() );
        CHECK( defaults.completionMonitorPriority() == 0 );
        CHECK( defaults.completionMonitorCpuset().empty() );
        CHECK( defaults.readerReleaseMonitorName().empty() );
        CHECK( defaults.readerReleaseMonitorPriority() == 0 );
        CHECK( defaults.readerReleaseMonitorCpuset().empty() );
        CHECK_FALSE( defaults.m_readerTiming.enabled() );
        REQUIRE( defaults.setupReaderTimingINDI() == 0 );
        CHECK( defaults.m_indiP_readerTiming.getElements().empty() );
        CHECK( defaults.m_indiP_readerQuality.getElements().empty() );
    }

    for( const auto &entry : { std::make_pair( std::string( "synco_reader_done" ), true ),
                               std::make_pair( std::string( "dm01disp" ), false ),
                               std::make_pair( std::string( "dm01disp07" ), false ),
                               std::make_pair( std::string( "dm01disp07_aux" ), false ),
                               std::make_pair( std::string( "dmC" ), false ),
                               std::make_pair( std::string( "synco_test_timing" ), false ),
                               std::make_pair( std::string( "/tmp/synco_reader_done" ), false ) } )
    {
        for( const bool simulation : { false, true } )
        {
            Harness app;
            app.setupConfig();
            char      path[]     = "/tmp/synco-reader-config-XXXXXX";
            const int descriptor = mkstemp( path );
            REQUIRE( descriptor >= 0 );
            ::close( descriptor );

            {
                std::ofstream config( path );
                config << "[simulation]\nenabled=" << ( simulation ? "true" : "false" )
                       << "\n[completionMonitor]\nshmimName=" << entry.first << '\n';
            }

            app.config.readConfig( path );
            std::remove( path );
            const int result = app.loadConfigImpl( app.config );
            CAPTURE( entry.first, simulation );
            CHECK( ( result == 0 ) == ( simulation && entry.second ) );

            if( result == 0 )
            {
                CHECK( app.completionMonitorName() == entry.first );
                CHECK( app.m_simulationEnabled );
                CHECK_FALSE( app.m_enabled );
                CHECK_FALSE( app.m_resources );
            }
        }
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
    dmSyncopation::setupReaderTimingINDI;
#endif
    // clang-format on
}

/// Match real callback evidence without altering shadow decisions, command pixels or semaphore populations.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation measures controller reader completion without changing shadow execution",
           "[dmSyncopation][readerTiming]" )
{
    using namespace MagAOX::dmReaderTiming;
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );
    app.localDriver( streams.directory );
    REQUIRE( app.setupReaderTimingINDI() == 0 );
    CHECK( app.m_indiP_readerTiming.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_readerQuality.getPerm() == pcf::IndiProperty::ReadOnly );
    CHECK( app.m_indiP_readerTiming.getElements().size() == 48 );
    CHECK( app.m_indiP_readerQuality.getElements().size() == 45 );
    CHECK( app.m_indiP_readerTiming["completionOffset_mean"].get<double>() == -1 );
    CHECK( app.m_indiP_readerTiming["requiredHold_count"].get<double>() == 0 );
    std::array<std::vector<int>, 4> semaphoreValues;

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        semaphoreValues[image].resize( streams.images[image].md->sem );

        for( size_t semaphore = 0; semaphore < semaphoreValues[image].size(); ++semaphore )
        {
            REQUIRE( sem_getvalue( streams.images[image].semptr[semaphore], &semaphoreValues[image][semaphore] ) == 0 );
        }
    }

    constexpr int64_t epoch = 1792000000000000000LL;
    CadenceClock::waits     = 0;

    for( uint64_t sample = 0; sample < 40; ++sample )
    {
        const int64_t observed = 1000000000 + int64_t( sample ) * 500000;
        cadenceObservation( app, observed, epoch + int64_t( sample ) * 500000, sample + 10 );
        combinedObservation( app,
                             streams.images[1],
                             observed + ( sample == 38 ? 180000 : 20000 ),
                             epoch + int64_t( sample ) * 500000,
                             sample + 10 );
        const auto before = simulationBytes( app.m_simulation.summary() );
        const auto record = readerCompletion( streams.images[1], sample + 1, observed + 30000, observed + 145000 );
        completionObservation( app, completion, record, observed + ( sample == 38 ? 190000 : 150000 ) );
        app.m_readerTiming.update( CadenceClock::time );
        app.updateReaderTimingINDI();
        CHECK( simulationBytes( app.m_simulation.summary() ) == before );
    }

    const auto &summary = app.m_readerTiming.summary();
    REQUIRE( summary.total.matched == 40 );
    CHECK( summary.total.primary == 40 );
    CHECK( summary.total.combined == 40 );
    CHECK( summary.total.completion == 40 );
    CHECK( summary.total.identityMismatch == 0 );
    CHECK( summary.total.droppedPrimary == 0 );
    CHECK( summary.total.droppedCombined == 0 );
    CHECK( summary.total.droppedCompletion == 0 );
    CHECK( summary.metrics[index( Metric::completionOffset )].count == 40 );
    CHECK( summary.metrics[index( Metric::completionOffset )].min < 0 );
    CHECK( summary.metrics[index( Metric::requiredHold )].min == 0 );
    CHECK( summary.metrics[index( Metric::processing )].mean == Approx( 115e-6 ) );
    CHECK( summary.metrics[index( Metric::notification )].min == Approx( 5.001e-6 ) );
    CHECK( summary.metrics[index( Metric::notification )].max == Approx( 45.001e-6 ) );
    CHECK( summary.worstValid );
    CHECK( summary.maxRequiredHoldNs > 120000 );
    CHECK( summary.maxRequiredHoldNs < 130000 );
    CHECK( summary.overBudget == 0 );
    CHECK( app.m_indiP_readerTiming["completionOffset_mean"].get<double>() ==
           Approx( summary.metrics[index( Metric::completionOffset )].mean ) );
    CHECK( app.m_indiP_readerTiming["processing_mean"].get<double>() == Approx( 115e-6 ) );
    CHECK( app.m_indiP_readerQuality["matched"].get<double>() == 40 );
    CHECK( app.m_indiP_readerQuality["maxRequiredUs"].get<double>() == Approx( summary.maxRequiredHoldNs * 1e-3 ) );
    CHECK( app.m_indiP_readerQuality["worstBudgetUs"].get<double>() == 155 );
    CHECK( app.m_simulation.summary().valid );
    CHECK( app.m_simulation.summary().sequence == 40 );
    CHECK( app.m_simulation.summary().outcomes[1].readerHold > 0 );
    CHECK( CadenceClock::waits == 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );
    CHECK( app.m_mappings.empty() );
    CHECK( app.m_timing.md == nullptr );

    for( size_t image = 0; image < streams.images.size(); ++image )
    {
        for( size_t pixel = 0; pixel < 9; ++pixel )
        {
            CHECK( streams.images[image].array.F[pixel] == float( image ) );
        }

        if( image >= 2 )
        {
            CHECK( streams.images[image].md->cnt0 == 0 );
            CHECK( streams.images[image].md->write == 0 );
        }

        for( size_t semaphore = 0; semaphore < semaphoreValues[image].size(); ++semaphore )
        {
            int value = -1;
            REQUIRE( sem_getvalue( streams.images[image].semptr[semaphore], &value ) == 0 );
            CHECK( value == semaphoreValues[image][semaphore] );
        }
    }

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, dev::shmimT{} );
    dmSyncopation::processImage( nullptr, syncoCombinerT{} );
    dmSyncopation::processImage( nullptr, syncoCompletionT{} );
    dmSyncopation::shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
    dmSyncopation::setupReaderTimingINDI;
    dmSyncopation::updateReaderTimingINDI;
#endif
    // clang-format on
}

/// Keep the completion monitor's bounded handoff independent of the model's timed-cycle mutex.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion callback does not wait for the simulation cycle", "[dmSyncopation][readerTiming]" )
{
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    combinedObservation( app, streams.images[1], 1000020000, epoch, 10 );
    completion.store( readerCompletion( streams.images[1], 1, 1000030000, 1000145000 ) );
    CadenceClock::time = 1000150000;
    app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
    const auto                   before = simulationBytes( app.m_simulation.summary() );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         callback  = std::async( std::launch::async,
                                [&]()
                                {
                                    return app.processImage( nullptr, syncoCompletionT{} );
                                } );
    const auto                   readiness = callback.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock(); // A locking regression must finish before future destruction or test assertions.
    CHECK( callback.get() == 0 );
    CHECK( readiness == std::future_status::ready );
    app.m_readerTiming.update( CadenceClock::time );
    CHECK( app.m_readerTiming.summary().total.matched == 1 );
    CHECK( simulationBytes( app.m_simulation.summary() ) == before );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoCompletionT{} );
#endif
    // clang-format on
}

/// Keep malformed, synthetic, busy and mismatched completion evidence outside qualified reader statistics.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation rejects completion evidence without a matching genuine observation",
           "[dmSyncopation][readerTiming]" )
{
    using namespace MagAOX::dmReaderTiming;
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    combinedObservation( app, streams.images[1], 1000020000, epoch, 10 );
    auto       record          = readerCompletion( streams.images[1], 1, 1000030000, 1000145000 );
    const auto before          = simulationBytes( app.m_simulation.summary() );
    uint64_t Counters::*reason = &Counters::incoherent;

    SECTION( "An initial callback has no genuine completion notification" )
    {
        completionObservation( app, completion, record, 1000150000, true );
    }

    SECTION( "A malformed wire cannot qualify" )
    {
        completion.store( record );
        completion.m_image.array.UI64[0] = 0;
        CadenceClock::time               = 1000150000;
        app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, syncoCompletionT{} ) == 0 );
    }

    SECTION( "An in-progress transport cannot qualify" )
    {
        completion.store( record );
        completion.m_image.md->write = 1;
        CadenceClock::time           = 1000150000;
        app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
        REQUIRE( app.processImage( nullptr, syncoCompletionT{} ) == 0 );
        completion.m_image.md->write = 0;
    }

    SECTION( "Busy is distinct from completed even when an older post wakes the observer" )
    {
        reason            = &Counters::busy;
        record.state      = MagAOX::dmCompletion::State::busy;
        record.finishedNs = 0;
        completionObservation( app, completion, record, 1000150000 );
    }

    SECTION( "An equal counter with a different raw publication stamp is not the same command" )
    {
        reason = &Counters::identityMismatch;
        ++record.publicationNsec;
        completionObservation( app, completion, record, 1000150000 );
    }

    SECTION( "No-send remains unqualified despite a successful software callback" )
    {
        reason         = &Counters::unqualified;
        record.state   = MagAOX::dmCompletion::State::unqualified;
        record.outcome = MagAOX::dmTiming::Outcome::noSend;
        completionObservation( app, completion, record, 1000150000 );
    }

    SECTION( "A future completion timestamp cannot share the observer's monotonic clock" )
    {
        reason            = &Counters::invalid;
        record.finishedNs = 1000160000;
        completionObservation( app, completion, record, 1000150000 );
    }

    SECTION( "Old unmatched evidence expires instead of authorizing a reader release" )
    {
        reason = &Counters::invalid;
        completionObservation( app, completion, record, 4000150000 );
    }

    app.m_readerTiming.update( CadenceClock::time );
    const auto &summary = app.m_readerTiming.summary();
    CHECK( summary.total.matched == 0 );
    CHECK( summary.total.*reason > 0 );
    CHECK( summary.metrics[index( Metric::requiredHold )].count == 0 );
    CHECK_FALSE( summary.worstValid );
    CHECK( simulationBytes( app.m_simulation.summary() ) == before );
    CHECK_FALSE( app.m_enabled );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoCompletionT{} );
    dmSyncopation::shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::initial );
#endif
    // clang-format on
}

/// Attachment lifecycle changes invalidate pending observations before any three-way join.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion matching respects monitor lifecycle epochs", "[dmSyncopation][readerTiming]" )
{
    using namespace MagAOX::dmReaderTiming;
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );
    constexpr int64_t epoch = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, epoch, 10 );
    combinedObservation( app, streams.images[1], 1000020000, epoch, 10 );
    const auto record = readerCompletion( streams.images[1], 1, 1000030000, 1000145000 );
    completionObservation( app, completion, record, 1000150000 );

    SECTION( "Completion attachment reset" )
    {
        app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::connected );
    }

    SECTION( "Completion attachment disconnect" )
    {
        app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::disconnected );
    }

    SECTION( "Primary attachment disconnect" )
    {
        app.shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::disconnected );
    }

    SECTION( "Combined attachment disconnect" )
    {
        app.shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::disconnected );
    }

    app.m_readerTiming.update( CadenceClock::time );
    CHECK( app.m_readerTiming.summary().total.matched == 0 );
    CHECK( app.m_readerTiming.summary().metrics[index( Metric::requiredHold )].count == 0 );
    CHECK( app.m_readerTiming.summary().total.staleEpoch > 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::shmimMonitorTiming( dev::shmimT{}, dev::shmimMonitorEvent::disconnected );
    dmSyncopation::shmimMonitorTiming( syncoCombinerT{}, dev::shmimMonitorEvent::disconnected );
    dmSyncopation::shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::disconnected );
#endif
    // clang-format on
}

/// Reject a completion monitor attached to pixels or a transport with an incompatible fixed layout.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion allocation requires its fixed CPU transport", "[dmSyncopation][readerTiming]" )
{
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    app.attachCompletionOnly();
    auto *metadata = completion.m_image.md;

    SECTION( "Pixel datatype" )
    {
        const auto original = metadata->datatype;
        metadata->datatype  = _DATATYPE_FLOAT;
        CHECK( app.allocate( syncoCompletionT{} ) < 0 );
        metadata->datatype = original;
    }

    SECTION( "Unexpected word count" )
    {
        const auto original = metadata->size[0];
        --metadata->size[0];
        CHECK( app.allocate( syncoCompletionT{} ) < 0 );
        metadata->size[0] = original;
    }

    SECTION( "Unexpected dimension" )
    {
        const auto original = metadata->naxis;
        metadata->naxis     = 2;
        CHECK( app.allocate( syncoCompletionT{} ) < 0 );
        metadata->naxis = original;
    }

    SECTION( "GPU-backed declaration" )
    {
        const auto original = metadata->location;
        metadata->location  = 0;
        CHECK( app.allocate( syncoCompletionT{} ) < 0 );
        metadata->location = original;
    }

    CHECK( app.allocate( syncoCompletionT{} ) == 0 );
    CHECK_FALSE( app.m_resources );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::allocate( syncoCompletionT{} );
#endif
    // clang-format on
}

/// Observe early channel release without a full-completion peer or any scheduler mutation.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation independently measures early reader release and SDK enrichment",
           "[dmSyncopation][readerTiming]" )
{
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoReaderReleaseT{} );
    dmSyncopation::shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    dmSyncopation::setupReaderTimingINDI;
#endif
    // clang-format on
    using namespace MagAOX::dmReaderTiming;
    Streams          streams;
    CompletionStream release;
    Harness          app;
    app.attachPrimaryOnly();
    app.m_simulationEnabled = true;
    app.m_simulationWindow  = 4;
    app.m_requested         = 2;
    app.m_now               = CadenceClock::now;
    app.m_wait              = CadenceClock::wait;
    app.m_readerReleaseTiming.configure( 2, 64, MagAOX::dmCompletion::Kind::readers );
    REQUIRE( app.configureSimulation() );
    REQUIRE( app.allocate( dev::shmimT{} ) == 0 );
    app.attachCombinedOnly();
    app.attachReaderReleaseOnly();
    app.localDriver( streams.directory );
    REQUIRE( app.setupReaderTimingINDI() == 0 );
    CHECK_FALSE( app.m_readerTiming.enabled() );
    CHECK( app.m_indiP_readerTiming.getElements().empty() );
    CHECK( app.m_indiP_readerReleaseTiming.getElements().size() == 48 );
    CHECK( app.m_indiP_readerReleaseQuality.getElements().size() == 48 );
    constexpr int64_t observed = 1000000000;
    constexpr int64_t external = 1792000000000000000LL;
    cadenceObservation( app, observed, external, 10 );
    combinedObservation( app, streams.images[1], observed + 20000, external, 10 );
    auto record       = readerCompletion( streams.images[1], 1, observed + 1000, observed + 40000 );
    record.kind       = MagAOX::dmCompletion::Kind::readers;
    record.state      = MagAOX::dmCompletion::State::released;
    record.outcome    = MagAOX::dmTiming::Outcome::unknown;
    const auto before = simulationBytes( app.m_simulation.summary() );
    release.store( record );
    CadenceClock::time = observed + 45000;
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoReaderReleaseT{} ) == 0 );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    REQUIRE( app.m_readerReleaseTiming.summary().total.matched == 1 );
    CHECK( app.m_readerReleaseTiming.summary().total.sdkResults == 0 );
    // Metadata-bracket clock reads advance the deterministic clock by a few nanoseconds.
    CHECK( app.m_readerReleaseTiming.summary().metrics[index( Metric::requiredHold )].mean ==
           Approx( 20e-6 ).margin( 5e-9 ) );
    CHECK( app.m_readerReleaseTiming.summary().worst.combined.budgetNs == 155000 );
    CHECK( simulationBytes( app.m_simulation.summary() ) == before );

    record.outcome       = MagAOX::dmTiming::Outcome::failed;
    record.backendResult = 7;
    release.store( record );
    CadenceClock::time = observed + 150000;
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoReaderReleaseT{} ) == 0 );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    app.updateReaderTimingINDI();
    CHECK( app.m_readerReleaseTiming.summary().total.matched == 1 );
    CHECK( app.m_readerReleaseTiming.summary().total.sdkResults == 1 );
    CHECK( app.m_readerReleaseTiming.summary().total.sdkFailed == 1 );
    CHECK( app.m_indiP_readerReleaseQuality["sdkFailed"].get<double>() == 1 );
    CHECK( app.m_indiP_readerReleaseQuality["matched"].get<double>() == 1 );
    CHECK( simulationBytes( app.m_simulation.summary() ) == before );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::disconnected );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    CHECK( app.m_readerReleaseTiming.summary().metrics[index( Metric::requiredHold )].count == 0 );
}

/// Keep the early-release observer opt-in, shadow-only and distinct from every other configured stream.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation early release observer validates its independent configuration",
           "[dmSyncopation][readerTiming]" )
{
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
#endif
    // clang-format on
    const bool        simulation = GENERATE( false, true );
    const std::string name =
        GENERATE( std::string( "early_read_done" ), std::string( "dm01disp" ), std::string( "same_done" ) );
    Harness app;
    app.setupConfig();
    char      path[]     = "/tmp/synco-release-config-XXXXXX";
    const int descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    ::close( descriptor );

    {
        std::ofstream config( path );
        config << "[simulation]\nenabled=" << ( simulation ? "true" : "false" )
               << "\n[completionMonitor]\nshmimName=same_done\n[readerReleaseMonitor]\nshmimName=" << name << '\n';
    }

    app.config.readConfig( path );
    std::remove( path );
    const auto result = app.loadConfigImpl( app.config );
    CHECK( ( result == 0 ) == ( simulation && name == "early_read_done" ) );
}

/// Keep the independent early-release callback out of the realtime simulation mutex.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation early release callback never waits for the simulation cycle", "[dmSyncopation][readerTiming]" )
{
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoReaderReleaseT{} );
#endif
    // clang-format on
    Streams          streams;
    CompletionStream release;
    Harness          app;
    prepareReaderTiming( app );
    app.m_readerReleaseTiming.configure( 2, 64, MagAOX::dmCompletion::Kind::readers );
    app.attachReaderReleaseOnly();
    constexpr int64_t external = 1792000000000000000LL;
    cadenceObservation( app, 1000000000, external, 10 );
    combinedObservation( app, streams.images[1], 1000020000, external, 10 );
    auto record    = readerCompletion( streams.images[1], 1, 1000030000, 1000040000 );
    record.kind    = MagAOX::dmCompletion::Kind::readers;
    record.state   = MagAOX::dmCompletion::State::released;
    record.outcome = MagAOX::dmTiming::Outcome::unknown;
    release.store( record );
    CadenceClock::time = 1000045000;
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    std::unique_lock<std::mutex> cycle( app.m_cycleMutex );
    auto                         callback  = std::async( std::launch::async,
                                [&]()
                                {
                                    return app.processImage( nullptr, syncoReaderReleaseT{} );
                                } );
    const auto                   readiness = callback.wait_for( std::chrono::milliseconds( 500 ) );
    cycle.unlock();
    CHECK( callback.get() == 0 );
    CHECK( readiness == std::future_status::ready );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    CHECK( app.m_readerReleaseTiming.summary().total.matched == 1 );
    CHECK( app.m_readerReleaseTiming.summary().total.sdkResults == 0 );
}

/// Split genuine receipt latency from later coherent capture without rejecting a valid endpoint after an older post.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion observers retain independent receipt and capture endpoints",
           "[dmSyncopation][readerTiming]" )
{
    using namespace MagAOX::dmReaderTiming;
    const bool       early  = GENERATE( false, true );
    const bool       causal = GENERATE( false, true );
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );

    if( early )
    {
        app.m_readerReleaseTiming.configure( 2, 64, MagAOX::dmCompletion::Kind::readers );
        app.attachReaderReleaseOnly();
    }

    app.localDriver( streams.directory );
    REQUIRE( app.setupReaderTimingINDI() == 0 );
    constexpr int64_t observed = 1000000000;
    constexpr int64_t external = 1792000000000000000LL;
    cadenceObservation( app, observed, external, 10 );
    combinedObservation( app, streams.images[1], observed + 20000, external, 10 );
    auto record = readerCompletion( streams.images[1], 1, observed + 30000, observed + 40000 );

    if( early )
    {
        record.kind    = MagAOX::dmCompletion::Kind::readers;
        record.state   = MagAOX::dmCompletion::State::released;
        record.outcome = MagAOX::dmTiming::Outcome::unknown;
    }

    completion.store( record );
    const auto before  = simulationBytes( app.m_simulation.summary() );
    const auto receipt = observed + ( causal ? 45000 : 35000 );
    CadenceClock::time = receipt;

    if( early )
    {
        app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    }
    else
    {
        app.shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
    }

    CadenceClock::time = observed + 55000;

    if( early )
    {
        REQUIRE( app.processImage( nullptr, syncoReaderReleaseT{} ) == 0 );
    }
    else
    {
        REQUIRE( app.processImage( nullptr, syncoCompletionT{} ) == 0 );
    }

    auto &collector = early ? app.m_readerReleaseTiming : app.m_readerTiming;
    collector.update( CadenceClock::time );
    app.updateReaderTimingINDI();
    const auto &summary = collector.summary();
    REQUIRE( summary.total.matched == 1 );
    CHECK( summary.total.receiptUnavailable == 0 );
    CHECK( summary.total.receiptInvalid == ( causal ? 0 : 1 ) );
    CHECK( summary.total.firstEnriched == 0 );
    CHECK( summary.metrics[index( Metric::notification )].mean == Approx( 15e-6 ) );
    CHECK( summary.metrics[index( Metric::requiredHold )].mean == Approx( 20e-6 ).margin( 5e-9 ) );
    REQUIRE( summary.worstValid );
    CHECK( summary.worst.completed.receiptNs == receipt );
    CHECK( summary.worst.completed.observedNs == observed + 55000 );
    REQUIRE( summary.worstNotificationValid );
    CHECK( summary.worstNotification.completed.receiptNs == receipt );
    CHECK( summary.worstNotification.completed.observedNs == observed + 55000 );

    if( causal )
    {
        CHECK( summary.metrics[index( Metric::wake )].count == 1 );
        CHECK( summary.metrics[index( Metric::wake )].mean == Approx( 5e-6 ) );
        CHECK( summary.metrics[index( Metric::capture )].count == 1 );
        CHECK( summary.metrics[index( Metric::capture )].mean == Approx( 10e-6 ) );
        CHECK( summary.metrics[index( Metric::wake )].mean + summary.metrics[index( Metric::capture )].mean ==
               Approx( summary.metrics[index( Metric::notification )].mean ) );
    }
    else
    {
        CHECK( summary.metrics[index( Metric::wake )].count == 0 );
        CHECK( summary.metrics[index( Metric::capture )].count == 0 );
    }

    const auto &quality = early ? app.m_indiP_readerReleaseQuality : app.m_indiP_readerQuality;
    const auto &timing  = early ? app.m_indiP_readerReleaseTiming : app.m_indiP_readerTiming;
    CHECK( quality["receiptInvalid"].get<double>() == ( causal ? 0 : 1 ) );
    CHECK( quality["receiptUnavailable"].get<double>() == 0 );
    CHECK( timing["wake_count"].get<double>() == ( causal ? 1 : 0 ) );
    CHECK( timing["capture_count"].get<double>() == ( causal ? 1 : 0 ) );
    CHECK( simulationBytes( app.m_simulation.summary() ) == before );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );
    CHECK( app.m_delivered == 0 );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoCompletionT{} );
    dmSyncopation::processImage( nullptr, syncoReaderReleaseT{} );
    dmSyncopation::shmimMonitorTiming( syncoCompletionT{}, dev::shmimMonitorEvent::receipt );
    dmSyncopation::shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    dmSyncopation::updateReaderTimingINDI;
#endif
    // clang-format on
}

/// Count a first SDK-enriched release without attributing its result notification to the earlier read-release wake.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation identifies an early release first captured after SDK enrichment",
           "[dmSyncopation][readerTiming]" )
{
    using namespace MagAOX::dmReaderTiming;
    Streams          streams;
    CompletionStream completion;
    Harness          app;
    prepareReaderTiming( app );
    app.m_readerReleaseTiming.configure( 2, 64, MagAOX::dmCompletion::Kind::readers );
    app.attachReaderReleaseOnly();
    constexpr int64_t observed = 1000000000;
    constexpr int64_t external = 1792000000000000000LL;
    cadenceObservation( app, observed, external, 10 );
    combinedObservation( app, streams.images[1], observed + 20000, external, 10 );
    auto record          = readerCompletion( streams.images[1], 1, observed + 30000, observed + 40000 );
    record.kind          = MagAOX::dmCompletion::Kind::readers;
    record.state         = MagAOX::dmCompletion::State::released;
    record.backendResult = 0;
    completion.store( record );
    CadenceClock::time = observed + 150000;
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    CadenceClock::time = observed + 160000;
    REQUIRE( app.processImage( nullptr, syncoReaderReleaseT{} ) == 0 );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    const auto &summary = app.m_readerReleaseTiming.summary();
    REQUIRE( summary.total.matched == 1 );
    CHECK( summary.total.sdkResults == 1 );
    CHECK( summary.total.sdkSuccess == 1 );
    CHECK( summary.total.firstEnriched == 1 );
    CHECK( summary.total.receiptInvalid == 0 );
    CHECK( summary.total.receiptUnavailable == 0 );
    CHECK( summary.metrics[index( Metric::notification )].mean == Approx( 120e-6 ) );
    CHECK( summary.metrics[index( Metric::wake )].count == 0 );
    CHECK( summary.metrics[index( Metric::capture )].count == 0 );
    CHECK( summary.worst.completed.receiptNs == observed + 150000 );

    completion.store( record );
    CadenceClock::time = observed + 170000;
    app.shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
    REQUIRE( app.processImage( nullptr, syncoReaderReleaseT{} ) == 0 );
    app.m_readerReleaseTiming.update( CadenceClock::time );
    CHECK( summary.total.matched == 1 );
    CHECK( summary.total.sdkResults == 1 );
    CHECK( summary.total.firstEnriched == 1 );
    CHECK( summary.metrics[index( Metric::wake )].count == 0 );
    CHECK( summary.metrics[index( Metric::capture )].count == 0 );
    CHECK_FALSE( app.m_enabled );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::processImage( nullptr, syncoReaderReleaseT{} );
    dmSyncopation::shmimMonitorTiming( syncoReaderReleaseT{}, dev::shmimMonitorEvent::receipt );
#endif
    // clang-format on
}

/// Apply observer scheduling overrides only to the selected optional monitor and preserve ordinary defaults.
/** \ingroup dmSyncopation_unit_test */
TEST_CASE( "syncopation completion observer scheduling overrides remain independent", "[dmSyncopation][readerTiming]" )
{
    const bool early = GENERATE( false, true );
    Harness    app;
    app.setupConfig();
    char      path[]     = "/tmp/synco-reader-priority-XXXXXX";
    const int descriptor = mkstemp( path );
    REQUIRE( descriptor >= 0 );
    ::close( descriptor );

    {
        std::ofstream config( path );
        config << "[simulation]\nenabled=true\n[completionMonitor]\nshmimName=synco_done\n";

        if( !early )
        {
            config << "threadPrio=43\ncpuset=reader_test\n";
        }

        config << "[readerReleaseMonitor]\nshmimName=synco_early_done\n";

        if( early )
        {
            config << "threadPrio=43\ncpuset=reader_test\n";
        }
    }

    app.config.readConfig( path );
    std::remove( path );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    CHECK( app.completionMonitorName() == "synco_done" );
    CHECK( app.readerReleaseMonitorName() == "synco_early_done" );
    CHECK( app.completionMonitorPriority() == ( early ? 0 : 43 ) );
    CHECK( app.readerReleaseMonitorPriority() == ( early ? 43 : 0 ) );
    CHECK( app.completionMonitorCpuset() == ( early ? "" : "reader_test" ) );
    CHECK( app.readerReleaseMonitorCpuset() == ( early ? "reader_test" : "" ) );
    CHECK( app.m_smThreadPrio == 30 );
    CHECK( app.m_smCpuset == "dmtweeter-synco" );
    CHECK( app.combinedMonitorPriority() == 0 );
    CHECK( app.cameraMonitorPriority() == 0 );
    CHECK_FALSE( app.m_enabled );
    CHECK_FALSE( app.m_resources );

    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    dmSyncopation::setupConfig;
    dmSyncopation::loadConfigImpl;
#endif
    // clang-format on
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
