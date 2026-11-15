/** \file ad7689Ctrl_test.cpp
 * \brief Catch2 tests for the ad7689Ctrl app.
 *
 * \ingroup ad7689Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include <vector>
#include <filesystem>
#include <functional>
#include <limits>

#define protected public
#include "../ad7689Ctrl.hpp"
#undef protected

/// \cond AD7689_APP_FIXTURE
namespace
{

/// Stub state for AD7689 hardware access during unit tests.
struct adcStubState
{
    std::vector<unsigned short>       m_channelValues;        ///< Values returned for each channel read.
    std::vector<int>                  m_readOrder;            ///< Order in which channels were sampled.
    int                               m_connectCalls{ 0 };    ///< Number of stubbed connect calls.
    bool                              m_failScan{ false };    ///< Fail after partially filling the caller staging area.
    bool                              m_failPrepare{ false }; ///< Reject pipeline initialization.
    std::function<void( timespec & )> m_timestamp;            ///< Optional deterministic actual scan-start timestamp.
    std::function<void()>             m_duringRead; ///< Optional frame-boundary control or shutdown injection.
};

/// Access the shared AD7689 stub state.
adcStubState &stubState()
{
    static adcStubState state;
    return state;
}

/// Reset the shared AD7689 stub state.
void resetStubState()
{
    stubState().m_channelValues.clear();
    stubState().m_readOrder.clear();
    stubState() = {};
}

/// Wrap a delay into `[0, period)` using the same modulo behavior as the app.
double wrapDelay( const double rawDelay_ns /**< [in] unwrapped delay in nanoseconds */,
                  const double period_ns /**< [in] period in nanoseconds */ )
{
    if( period_ns <= 0.0 )
    {
        return 0.0;
    }

    const long long wrapCycles = static_cast<long long>( rawDelay_ns / period_ns );
    double          wrapped    = rawDelay_ns - static_cast<double>( wrapCycles ) * period_ns;

    if( wrapped < 0.0 )
    {
        wrapped += period_ns;
    }
    else if( wrapped >= period_ns )
    {
        wrapped -= period_ns;
    }

    return wrapped;
}

} // namespace

namespace AD7689Lib
{
AD7689::AD7689()           = default;
AD7689::~AD7689() noexcept = default;
int AD7689::connect( const Settings &settings )
{
    m_settings = settings;
    ++stubState().m_connectCalls;
    return 0;
}

int AD7689::disconnect() noexcept
{
    m_ready = false;
    return 0;
}

int AD7689::prepare( const std::vector<unsigned> &channels )
{
    if( stubState().m_failPrepare )
    {
        return -1;
    }

    m_count = channels.size();
    std::copy( channels.begin(), channels.end(), m_channels.begin() );
    m_ready = true;
    return 0;
}

int AD7689::readScan( uint16_t *dest, size_t count, timespec &timestamp )
{
    if( stubState().m_timestamp )
    {
        stubState().m_timestamp( timestamp );
    }
    else
    {
        clock_gettime( CLOCK_REALTIME, &timestamp );
    }

    for( size_t i = 0; i < count; ++i )
    {
        unsigned channel = m_count == count ? m_channels[i] : static_cast<unsigned>( i );
        stubState().m_readOrder.push_back( channel );
        dest[i] = channel < stubState().m_channelValues.size() ? stubState().m_channelValues[channel] : 0;
    }

    if( stubState().m_duringRead )
    {
        stubState().m_duringRead();
    }

    return stubState().m_failScan ? -1 : 0;
}
} // namespace AD7689Lib

/// \endcond

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup ad7689Ctrl_unit_test ad7689Ctrl Unit Tests
 * \brief Unit tests for the ad7689Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `ad7689Ctrl` unit tests.
/** \ingroup ad7689Ctrl_unit_test
 */
namespace ad7689CtrlTest
{

/// \cond AD7689_APP_HARNESS
namespace
{

/// Capture real serialized INDI messages using local files and no input thread or server.
class captureDriver : public MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>
{
    int m_inputFd{ -1 }; ///< Owned unactivated input descriptor, identified by its private file inode.
  public:
    mutable std::vector<std::string> m_messages; ///< XML emitted by the real property update path.

    /// Attach to the fixture's private descriptors.
    captureDriver( MagAOX::app::MagAOXApp<true> *parent /**< [in] local test application */ );

    /// Release the input; the connection destructor releases output.
    ~captureDriver();

    /// Retain an outgoing message without contacting an INDI server.
    void sendXml( const std::string &xml /**< [in] serialized property */ ) const override;
};

captureDriver::captureDriver( MagAOX::app::MagAOXApp<true> *parent ) : indiDriver( parent, "ad7689-test", "0", "1.7" )
{
    REQUIRE( good() );
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( stat( parent->driverInName().c_str(), &input ) == 0 );

    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        const int   descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};

        if( fstat( descriptor, &candidate ) == 0 && candidate.st_dev == input.st_dev &&
            candidate.st_ino == input.st_ino )
        {
            m_inputFd = descriptor;
        }
    }

    REQUIRE( m_inputFd >= 0 );
}

captureDriver::~captureDriver()
{
    if( m_inputFd >= 0 )
    {
        close( m_inputFd );
    }

    setInputFd( -1 );
}

void captureDriver::sendXml( const std::string &xml ) const
{
    m_messages.push_back( xml );
}

/// Test harness exposing small setup helpers for `ad7689Ctrl`.
class ad7689Ctrl_test : public ad7689Ctrl
{
  public:
    bool    m_fakeSync{ true };    ///< Deterministic scheduling clock for tests that use an in-process semaphore.
    int64_t m_clockStep_ns{ 100 }; ///< Simulated clock-call cost; zero for exact nominal-deadline tests.
    std::function<void( int64_t )> m_beforeWait; ///< Optional deterministic source event inserted before a wait.
    int64_t m_fakeNow_ns{ 10000000000 }; ///< Simulated monotonic clock, advanced by waits instead of wall sleeps.

    /// Return simulated time for deterministic scheduler tests, or the real clock for native integration.
    int monotonicNow( int64_t &now /**< [out] monotonic nanoseconds */ ) override;

    /// Advance to an absolute deadline or consume an already queued test notification.
    int waitForSync( int64_t deadline /**< [in] absolute monotonic deadline */ ) override;

    /// Keep source-arrival realtime timestamps coherent with the deterministic scheduling clock.
    int getRealtime( timespec &stamp /**< [out] realtime timestamp */ ) override;

    sem_t          m_testSemaphore{}; ///< Owned raw semaphore for deterministic multi-scan integration tests.
    IMAGE_METADATA m_testMetadata{};  ///< Stable source metadata paired with the raw test semaphore.
    bool           m_testSemaphoreOpen{ false }; ///< Whether the fixture must destroy its raw semaphore.

    /// Configure independent ADC/WFS rates without opening hardware or a named shared-memory source.
    void setupSync( double adcFps /**< [in] requested ADC scan rate */,
                    double wfsFps /**< [in] optional source hint */ );

    /// Prequalify a deterministic two-point clock for existing DDSPC regression fixtures.
    void enableDdspc( uint64_t firstId = 1 /**< [in] first externally supplied source counter */ );

    /// Post synthetic source progress at a precise monotonic reference time.
    void postReference( uint64_t delta /**< [in] source counter advance */,
                        int64_t  period /**< [in] source period [ns] */,
                        unsigned wakes = 1 /**< [in] queued semaphore notifications */ );

    /// Construct the test harness with a stable config name.
    ad7689Ctrl_test();

    /// Release the unactivated INDI fixture and its private input file.
    ~ad7689Ctrl_test();

    /// Install a local recording driver for the real INDI serialization path.
    void captureINDI();

    std::string m_testInput; ///< Private input file removed after driver destruction.

    /// Align direct fixture state with staged controls before an inherited acquisition regression.
    int acquireAndCheckValid();

    /// Initialize the local fps INDI property used by the callback tests.
    void setupFpsProperty();

    /// Initialize the external fps INDI property used by the callback tests.
    void setupFpsSourceProperty();

    /// Build an INDI property update for the local fps callback.
    pcf::IndiProperty makeFpsUpdate( const double target /**< [in] the requested fps */ );

    /// Build an INDI property update for the external fps-source callback.
    pcf::IndiProperty makeFpsSourceUpdate( const double current /**< [in] the reported fps */ );

    /// Initialize the alpha INDI property used by callback tests.
    void setupAlphaProperty();

    /// Build an INDI property update for the alpha callback.
    pcf::IndiProperty makeAlphaUpdate( const double target /**< [in] requested global EMA alpha */ );

    /// Initialize the synchronized-delay INDI property used by callback tests.
    void setupSynchroDelayProperty();

    /// Build an INDI property update for the synchronized-delay callback.
    pcf::IndiProperty makeSynchroDelayUpdate(
        const double target_us /**< [in] requested synchronized signed delay offset in microseconds */ );

    /// Initialize the timing-diagnostics INDI property used by diagnostics tests.
    void setupTimingDiagnosticsProperty();
};

ad7689Ctrl_test::ad7689Ctrl_test()
{
    m_configName      = "ad7689Ctrl_test";
    m_requestedWfsFps = m_wfs_fps = 2000;
}

ad7689Ctrl_test::~ad7689Ctrl_test()
{
    if( m_testSemaphoreOpen )
    {
        sem_destroy( &m_testSemaphore );
    }

    delete m_indiDriver;
    m_indiDriver = nullptr;

    if( !m_testInput.empty() )
    {
        std::filesystem::remove( m_testInput );
    }
}

void ad7689Ctrl_test::setupSync( double adcFps, double wfsFps )
{
    REQUIRE( sem_init( &m_testSemaphore, 0, 0 ) == 0 );
    m_testSemaphoreOpen = true;
    m_synchroShmimName  = "offline-wfs-subscans";
    m_synchroSemaphore  = &m_testSemaphore;
    m_synchroStream.md  = &m_testMetadata;
    m_nextSyncProbe_ns  = std::numeric_limits<int64_t>::max();
    m_requestedWfsFps = m_wfs_fps = wfsFps;
    m_testMetadata.atime          = timespec{ 1700000000, 0 };
    m_values.resize( 2 );
    stubState().m_channelValues = { 65535, 4096 };
    REQUIRE( requestAcquisition( adcFps, 2 ) == 0 );
    applyControls( true );
}

void ad7689Ctrl_test::enableDdspc( uint64_t firstId )
{
    m_requestedDdspc.m_enabled = true;
    applyControls();
    m_wfsClock.m_minEvents = 2;
    const int64_t period   = static_cast<int64_t>( 1e9 / m_wfs_fps );
    REQUIRE_FALSE( m_wfsClock.observe( firstId - 1, m_fakeNow_ns - period ) );
}

void ad7689Ctrl_test::postReference( uint64_t delta, int64_t period, unsigned wakes )
{
    m_testMetadata.cnt0 += delta;
    const int64_t elapsed = static_cast<int64_t>( delta ) * period;
    const int64_t nanos   = m_testMetadata.atime.tv_nsec + elapsed;
    m_testMetadata.atime.tv_sec += nanos / 1000000000;
    m_testMetadata.atime.tv_nsec = nanos % 1000000000;

    if( m_haveSyncArrival && elapsed )
    {
        m_fakeNow_ns = m_lastSyncArrival_ns + elapsed - 2 * m_clockStep_ns;
    }

    for( unsigned i = 0; i < wakes; ++i )
    {
        REQUIRE( sem_post( &m_testSemaphore ) == 0 );
    }
}

void ad7689Ctrl_test::captureINDI()
{
    char      path[] = "/tmp/ad7689-indi-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    m_testInput = m_driverInName = path;
    m_driverOutName = m_driverCtrlName = "/dev/null";
    m_indiDriver                       = new captureDriver( this );
}

int ad7689Ctrl_test::monotonicNow( int64_t &now )
{
    if( !m_fakeSync || m_synchroShmimName.empty() )
    {
        return ad7689Ctrl::monotonicNow( now );
    }

    m_fakeNow_ns += m_clockStep_ns;
    now = m_fakeNow_ns;
    return 0;
}

int ad7689Ctrl_test::waitForSync( int64_t deadline )
{
    if( !m_fakeSync )
    {
        return ad7689Ctrl::waitForSync( deadline );
    }

    if( m_beforeWait )
    {
        m_beforeWait( deadline );
    }

    if( sem_trywait( m_synchroSemaphore ) == 0 )
    {
        return 0;
    }

    m_fakeNow_ns = std::max( m_fakeNow_ns, deadline );
    errno        = ETIMEDOUT;
    return -1;
}

int ad7689Ctrl_test::getRealtime( timespec &stamp )
{
    if( !m_fakeSync || m_synchroShmimName.empty() )
    {
        return ad7689Ctrl::getRealtime( stamp );
    }

    stamp = timespec{ 1700000000 + m_fakeNow_ns / 1000000000, m_fakeNow_ns % 1000000000 };
    return 0;
}

int ad7689Ctrl_test::acquireAndCheckValid()
{
    m_requestedFps      = m_fps;
    m_requestedChannels = m_numChannels;
    m_requestedAlpha    = m_alpha;
    m_requestedOffset   = m_synchroPostDelay;
    m_requestedWfsFps   = m_wfs_fps;

    if( !m_synchroShmimName.empty() && m_fakeSync )
    {
        m_nextSyncProbe_ns = std::numeric_limits<int64_t>::max();
    }

    int result = ad7689Ctrl::acquireAndCheckValid();

    while( result > 0 && !m_shutdown && !m_reconfig && m_syncSchedule.m_slot < m_syncSchedule.m_count )
    {
        result = ad7689Ctrl::acquireAndCheckValid();
    }

    return result;
}

void ad7689Ctrl_test::setupFpsProperty()
{
    m_indiP_fps = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_fps.setName( "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    m_indiP_fps["current"].setValue( m_fps );
    m_indiP_fps.add( pcf::IndiElement( "target" ) );
    m_indiP_fps["target"].setValue( m_fps );
}

void ad7689Ctrl_test::setupFpsSourceProperty()
{
    m_fpsDevice   = "fpsdev";
    m_fpsProperty = "fps";
    m_fpsElement  = "current";

    m_indiP_fpsSource = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_fpsSource.setDevice( m_fpsDevice );
    m_indiP_fpsSource.setName( m_fpsProperty );
}

pcf::IndiProperty ad7689Ctrl_test::makeFpsUpdate( const double target )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setName( "fps" );
    ip.add( pcf::IndiElement( "current" ) );
    ip["current"].setValue( target );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"].setValue( target );
    return ip;
}

pcf::IndiProperty ad7689Ctrl_test::makeFpsSourceUpdate( const double current )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( m_fpsDevice );
    ip.setName( m_fpsProperty );
    ip.add( pcf::IndiElement( m_fpsElement ) );
    ip[m_fpsElement].setValue( current );
    return ip;
}

void ad7689Ctrl_test::setupAlphaProperty()
{
    m_indiP_alpha = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_alpha.setName( "alpha" );
    m_indiP_alpha.add( pcf::IndiElement( "current" ) );
    m_indiP_alpha["current"].setValue( m_alpha );
    m_indiP_alpha.add( pcf::IndiElement( "target" ) );
    m_indiP_alpha["target"].setValue( m_alpha );
}

pcf::IndiProperty ad7689Ctrl_test::makeAlphaUpdate( const double target )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setName( "alpha" );
    ip.add( pcf::IndiElement( "current" ) );
    ip["current"].setValue( target );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"].setValue( target );
    return ip;
}

void ad7689Ctrl_test::setupSynchroDelayProperty()
{
    m_indiP_synchroDelay = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_synchroDelay.setName( "synchroDelay" );
    m_indiP_synchroDelay.add( pcf::IndiElement( "current" ) );
    m_indiP_synchroDelay["current"].setValue( m_synchroPostDelay );
    m_indiP_synchroDelay.add( pcf::IndiElement( "target" ) );
    m_indiP_synchroDelay["target"].setValue( m_synchroPostDelay );
}

pcf::IndiProperty ad7689Ctrl_test::makeSynchroDelayUpdate( const double target_us )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setName( "synchroDelay" );
    ip.add( pcf::IndiElement( "current" ) );
    ip["current"].setValue( target_us );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"].setValue( target_us );
    return ip;
}

void ad7689Ctrl_test::setupTimingDiagnosticsProperty()
{
    if( !m_indiDriver )
    {
        captureINDI();
    }

    m_indiP_timingDiag = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_timingDiag.setName( "timingDiag" );
    m_indiP_timingDiag.setDevice( m_configName );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_read_latency_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "synchro_delay_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "synchro_delay_target_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_applied_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_model_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_phase_error_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_lock" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_budget_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_non_delay_service_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "delay_capped" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "read_latency_error_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "avg_semaphore_period_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "channel_readout_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "trigger_interval_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "trigger_time_us" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "local_frame_seq" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_received" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_written" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frames_dropped" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_frame_id_gap_count" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_producer_frame_id" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "sync_producer_frame_delta" ) );
    m_indiP_timingDiag.add( pcf::IndiElement( "mode_code" ) );
}

} // namespace
/// \endcond

/// Preserve Doxygen links for the real `ad7689Ctrl` APIs exercised by the tests.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl Doxygen references are preserved", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::checkRecordTimes;
    MagAOX::app::ad7689Ctrl::configureAcquisition;
    MagAOX::app::ad7689Ctrl::delayBeforeRead;
    MagAOX::app::ad7689Ctrl::fps;
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_alpha;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_fps;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_synchroDelay;
    MagAOX::app::ad7689Ctrl::nsToTimespec;
    MagAOX::app::ad7689Ctrl::reconfig;
    MagAOX::app::ad7689Ctrl::recordTelem;
    MagAOX::app::ad7689Ctrl::setCallBack_m_indiP_fpsSource;
    MagAOX::app::ad7689Ctrl::startAcquisition;
    MagAOX::app::ad7689Ctrl::synchroStreamStale;
    MagAOX::app::ad7689Ctrl::timespecToNs;
    MagAOX::app::ad7689Ctrl::updateSynchroDelayController;
    MagAOX::app::ad7689Ctrl::updateTimingDiagnosticsIndi;
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupFpsProperty();
    app.setupFpsSourceProperty();
    app.setupAlphaProperty();
    app.setupSynchroDelayProperty();

    XWCTEST_DOXYGEN_REF( app.loadConfigImpl( app.config ) );
    XWCTEST_DOXYGEN_REF( app.configureAcquisition() );
    XWCTEST_DOXYGEN_REF( app.fps() );
    XWCTEST_DOXYGEN_REF( app.startAcquisition() );
    XWCTEST_DOXYGEN_REF( app.acquireAndCheckValid() );
    XWCTEST_DOXYGEN_REF( app.loadImageIntoStream( nullptr ) );
    XWCTEST_DOXYGEN_REF( app.reconfig() );
    XWCTEST_DOXYGEN_REF( app.synchroStreamStale() );
    XWCTEST_DOXYGEN_REF( app.checkRecordTimes() );
    XWCTEST_DOXYGEN_REF( app.recordTelem( nullptr ) );
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 1000.0 ) ) );
    XWCTEST_DOXYGEN_REF( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 1000.0 ) ) );
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( 0.01 ) ) );
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( 1.0 ) ) );
    XWCTEST_DOXYGEN_REF( app.updateTriggerTiming( timespec{} ) );
    XWCTEST_DOXYGEN_REF( app.updateTimingDiagnosticsIndi() );
    XWCTEST_DOXYGEN_REF( app.delayBeforeRead() );
    XWCTEST_DOXYGEN_REF( app.updateSynchroDelayController( 0.0 ) );
    XWCTEST_DOXYGEN_REF( app.timespecToNs( timespec{} ) );
    XWCTEST_DOXYGEN_REF( app.nsToTimespec( 0.0 ) );

    REQUIRE( app.checkRecordTimes() == 0 );
    REQUIRE( app.recordTelem( nullptr ) == 0 );
}

/// Verify synchronized-acquisition defaults load from configuration.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl configuration defaults load synchronized settings", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
    MagAOX::app::ad7689Ctrl::setupConfig;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/ad7689Ctrl_test.conf", { "none" }, { "nada" }, { "0" } );
    app.config.readConfig( "/tmp/ad7689Ctrl_test.conf" );

    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_synchroShmimName.empty() );
    REQUIRE( app.m_synchroPostDelay == 0 );
    REQUIRE( app.m_synchroDtTransfer_ns == Approx( 3000.0 ) );
    REQUIRE( app.m_synchroWfsProcess_ns == Approx( 51500.0 ) );
    REQUIRE( app.m_synchroDtF_ns == Approx( 10000.0 ) );
    REQUIRE( app.m_synchroWfsRead_ns == Approx( 276100.0 ) );
    REQUIRE( app.m_delayLockAbsThreshold_ns == Approx( 50000.0 ) );
    REQUIRE( app.m_delayLockFracThreshold == Approx( 0.1 ) );
    REQUIRE( app.m_cadenceGuard_ns == Approx( 20000.0 ) );
    REQUIRE( app.m_alpha == Approx( 0.01f ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( static_cast<double>( app.m_fps ) ) );
    auto example = std::filesystem::path( "apps/ad7689Ctrl/config/example.conf" );

    if( !std::filesystem::exists( example ) )
    {
        example = "../apps/ad7689Ctrl/config/example.conf";
    }

    REQUIRE( std::filesystem::exists( example ) );
    app.config.readConfig( example.string() );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_shmimName == "accad7689_test" );
    REQUIRE( app.m_numChannels == 2 );
    REQUIRE( app.m_fps == 2000 );
    REQUIRE( app.m_adcSettings.m_bus == 0 );
    REQUIRE( app.m_adcSettings.m_device == 0 );
    REQUIRE( app.m_adcSettings.m_baud == 1000000 );
    REQUIRE( app.m_adcSettings.m_chip == "/dev/gpiochip0" );
    REQUIRE( app.m_adcSettings.m_line == 8 );
    REQUIRE( app.m_adcSettings.m_conversionTime_ns == 10000 );
    REQUIRE( app.m_adcSettings.m_spiSetup_ns == 1000 );
    REQUIRE( app.m_adcSettings.m_quietTime_ns == 1000 );
    REQUIRE( app.m_adcSettings.m_vref == Approx( 3.32 ) );
}

/// Verify synchronized-acquisition overrides load from configuration.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl configuration overrides load synchronized settings", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
    MagAOX::app::ad7689Ctrl::setupConfig;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile(
        "/tmp/ad7689Ctrl_test_override.conf",
        { "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "synchro",
          "accel" },
        { "shmimName",
          "postDelay",
          "dtTransfer_ns",
          "wfsProcess_ns",
          "dtF_ns",
          "wfsRead_ns",
          "delayLockAbsThreshold_ns",
          "delayLockFracThreshold",
          "cadenceGuard_ns",
          "alpha",
          "numChannels" },
        { "camwfs_sync", "17", "4000", "62000", "11000", "290000", "75000", "0.2", "15000", "0.25", "3" } );
    app.config.readConfig( "/tmp/ad7689Ctrl_test_override.conf" );

    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_synchroShmimName == "camwfs_sync" );
    REQUIRE( app.m_synchroPostDelay == 17 );
    REQUIRE( app.m_synchroDtTransfer_ns == Approx( 4000.0 ) );
    REQUIRE( app.m_synchroWfsProcess_ns == Approx( 62000.0 ) );
    REQUIRE( app.m_synchroDtF_ns == Approx( 11000.0 ) );
    REQUIRE( app.m_synchroWfsRead_ns == Approx( 290000.0 ) );
    REQUIRE( app.m_delayLockAbsThreshold_ns == Approx( 75000.0 ) );
    REQUIRE( app.m_delayLockFracThreshold == Approx( 0.2 ) );
    REQUIRE( app.m_cadenceGuard_ns == Approx( 15000.0 ) );
    REQUIRE( app.m_alpha == Approx( 0.25f ) );
    REQUIRE( app.m_numChannels == 3 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 17000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 17000.0f ) );
    REQUIRE( app.m_wfs_fps == 0 );
}

/// Verify the user fps callback still updates cadence metadata.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl fps callback updates trigger metadata", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_fps;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupFpsProperty();

    REQUIRE( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 500.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_fps == Approx( 500.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( 500.0 ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 500.0f ) );
    REQUIRE( app.m_nanoSecTarget == Approx( 1e9f / 500.0f ) );
}

/// Verify the external fps source callback still updates cadence metadata.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl fps source callback updates trigger metadata", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
    MagAOX::app::ad7689Ctrl::setCallBack_m_indiP_fpsSource;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupFpsSourceProperty();

    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 250.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_fps == Approx( 250.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( 250.0 ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 250.0f ) );
    REQUIRE( app.m_nanoSecTarget == Approx( 1e9f / 250.0f ) );
}

/// Verify synchronized alpha configuration is clamped to the valid range.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl configuration clamps synchronized alpha", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
    MagAOX::app::ad7689Ctrl::setupConfig;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/ad7689Ctrl_test_alpha_low.conf", { "synchro" }, { "alpha" }, { "-0.5" } );
    app.config.readConfig( "/tmp/ad7689Ctrl_test_alpha_low.conf" );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_alpha == Approx( 0.0f ) );

    mx::app::writeConfigFile( "/tmp/ad7689Ctrl_test_alpha_high.conf", { "synchro" }, { "alpha" }, { "1.5" } );
    app.config.readConfig( "/tmp/ad7689Ctrl_test_alpha_high.conf" );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_alpha == Approx( 1.0f ) );
}

/// Verify alpha callback updates and clamps the global EMA coefficient.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl alpha callback updates and clamps", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_alpha;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupAlphaProperty();

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( 0.4 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_alpha == Approx( 0.4f ) );

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( -1.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_alpha == Approx( 0.0f ) );

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( 2.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_alpha == Approx( 1.0f ) );
}

/// Verify synchronized-delay callback updates delay state with signed offsets.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchroDelay callback updates signed offsets", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_synchroDelay;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupSynchroDelayProperty();
    app.m_wfsPeriodMeasured_ns = 1000000.0;
    app.m_delayModel_ns        = 120000.0;

    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( 25.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_synchroPostDelay == 25 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 145000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 145000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 145000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );

    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( -5.0 ) ) == 0 );
    app.applyControls( true );
    REQUIRE( app.m_synchroPostDelay == -5 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 115000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 115000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 115000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );
}

/// Verify synchronized-mode timing diagnostics publish loop state and derived error.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timing diagnostics publish synchronized loop metrics", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::captureTimingDiagnostics;
    MagAOX::app::ad7689Ctrl::updateTimingDiagnosticsIndi;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_synchroShmimName       = "camwfs_sync";
    app.m_avgReadLatency_ns      = 125000.0;
    app.m_synchroDelay           = 31000.0f;
    app.m_synchroDelayTarget     = 17000.0f;
    app.m_delayApplied_ns        = 24000.0;
    app.m_delayModel_ns          = 17000.0;
    app.m_avgSemaphorePeriod_ns  = 500000.0;
    app.m_wfsPeriodMeasured_ns   = 500000.0;
    app.m_producerPeriodInst_ns  = 510000.0;
    app.m_avgProducerPeriod_ns   = 500000.0;
    app.m_channelReadoutTime_ns  = 34000.0;
    app.m_triggerInterval_ns     = 600000.0;
    app.m_atime                  = timespec{ 12, 3000000L };
    app.m_triggerTime            = timespec{ 12, 3456789L };
    app.m_localFrameSeq          = 44;
    app.m_syncFramesReceived     = 41;
    app.m_syncFramesWritten      = 40;
    app.m_syncFramesDropped      = 2;
    app.m_syncFrameIdGapCount    = 2;
    app.m_syncProducerFrameId    = 123456;
    app.m_syncProducerFrameDelta = 3;
    app.m_syncProducerFrameValid = true;

    app.m_syncSchedule.m_count    = 1;
    app.m_syncSchedule.m_interval = app.m_wfsPeriodMeasured_ns;
    app.captureTimingDiagnostics();
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["avg_read_latency_us"].get<double>() == Approx( 125.0 ) );
    REQUIRE( app.m_indiP_timingDiag["synchro_delay_us"].get<double>() == Approx( 31.0 ) );
    REQUIRE( app.m_indiP_timingDiag["synchro_delay_target_us"].get<double>() == Approx( 17.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_applied_us"].get<double>() == Approx( 24.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_model_us"].get<double>() == Approx( 17.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 7.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 1.0 ) );
    REQUIRE( app.m_indiP_timingDiag["read_latency_error_us"].get<double>() == Approx( 108.0 ) );
    REQUIRE( app.m_indiP_timingDiag["avg_semaphore_period_us"].get<double>() == Approx( 500.0 ) );
    REQUIRE( app.m_indiP_timingDiag["channel_readout_us"].get<double>() == Approx( 34.0 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_interval_us"].get<double>() == Approx( 600.0 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_time_us"].get<double>() ==
             Approx( 1e-3 * ( ad7689Ctrl::timespecToNs( timespec{ 12, 3456789L } ) -
                              ad7689Ctrl::timespecToNs( timespec{ 12, 3000000L } ) ) ) );
    REQUIRE( app.m_indiP_timingDiag["local_frame_seq"].get<double>() == Approx( 44.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_received"].get<double>() == Approx( 41.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_written"].get<double>() == Approx( 40.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_dropped"].get<double>() == Approx( 2.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frame_id_gap_count"].get<double>() == Approx( 2.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_id"].get<double>() == Approx( 123456.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_delta"].get<double>() == Approx( 3.0 ) );
    REQUIRE( app.m_indiP_timingDiag["mode_code"].get<double>() == Approx( 1.0 ) );
}

/// Verify timing diagnostics report timer mode and update mode code across transitions.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timing diagnostics track mode transitions", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::captureTimingDiagnostics;
    MagAOX::app::ad7689Ctrl::updateTimingDiagnosticsIndi;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_synchroShmimName   = "camwfs_sync";
    app.m_atime              = timespec{ 1, 2 };
    app.m_triggerInterval_ns = 123456.0;
    app.captureTimingDiagnostics();
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["mode_code"].get<double>() == Approx( 1.0 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_interval_us"].get<double>() == Approx( 123.456 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_time_us"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_received"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_written"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_dropped"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frame_id_gap_count"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_id"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_delta"].get<double>() == Approx( 0.0 ) );

    app.m_synchroShmimName.clear();
    app.m_triggerInterval_ns = 456789.0;
    app.captureTimingDiagnostics();
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["mode_code"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_interval_us"].get<double>() == Approx( 456.789 ) );
    REQUIRE( app.m_indiP_timingDiag["trigger_time_us"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_received"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_written"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frames_dropped"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_frame_id_gap_count"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_id"].get<double>() == Approx( 0.0 ) );
    REQUIRE( app.m_indiP_timingDiag["sync_producer_frame_delta"].get<double>() == Approx( 0.0 ) );
}

/// Verify timing diagnostics wrap phase error and require both lock thresholds.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timing diagnostics compute wrapped phase error and lock thresholds", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::captureTimingDiagnostics;
    MagAOX::app::ad7689Ctrl::updateTimingDiagnosticsIndi;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_synchroShmimName         = "camwfs_sync";
    app.m_delayApplied_ns          = 50.0;
    app.m_delayModel_ns            = 900.0;
    app.m_wfsPeriodMeasured_ns     = 1000.0;
    app.m_delayLockAbsThreshold_ns = 200.0;
    app.m_delayLockFracThreshold   = 0.1;

    app.m_syncSchedule.m_count    = 1;
    app.m_syncSchedule.m_interval = app.m_wfsPeriodMeasured_ns;
    app.captureTimingDiagnostics();
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.15 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 0.0 ) );

    app.m_delayLockFracThreshold  = 0.2;
    app.m_syncSchedule.m_count    = 1;
    app.m_syncSchedule.m_interval = app.m_wfsPeriodMeasured_ns;
    app.captureTimingDiagnostics();
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.15 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 1.0 ) );
}

/// Verify nanosecond and timespec conversions preserve normalized values.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timing helpers convert between nanoseconds and timespec", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::timespecToNs;
#endif
    // clang-format on
    const double   ns = -250000000.0;
    const timespec ts = ad7689Ctrl::nsToTimespec( ns );

    REQUIRE( ts.tv_sec == -1 );
    REQUIRE( ts.tv_nsec == 750000000L );
    REQUIRE( ad7689Ctrl::timespecToNs( ts ) == Approx( ns ) );
}

/// Verify synchronized timing uses only the measured semaphore period for delay modeling.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming uses measured semaphore period for delay model", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_alpha                 = 0.2f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 10, 100000000L };
    app.m_avgSemaphorePeriod_ns = 500000.0;
    app.m_wfs_fps               = 1000.0;
    app.m_synchroPostDelay      = 0;

    const timespec secondArrival{ 10, 101000000L };

    app.updateTriggerTiming( secondArrival );

    const double alpha                     = static_cast<double>( app.m_alpha );
    const double expectedAvg_ns            = alpha * 1000000.0 + ( 1.0 - alpha ) * 500000.0;
    const double expectedMeasuredDeltaT_ns = expectedAvg_ns;
    const double expectedBlendedDeltaT_ns  = ( 1.0 - alpha ) * ( 1e9 / 1000.0 ) + alpha * expectedAvg_ns;
    const double rawDelayMeasured_ns      = 0.5 * expectedMeasuredDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double rawDelayBlended_ns       = 0.5 * expectedBlendedDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelayMeasured_ns = wrapDelay( rawDelayMeasured_ns, expectedMeasuredDeltaT_ns );
    const double expectedDelayBlended_ns  = wrapDelay( rawDelayBlended_ns, expectedBlendedDeltaT_ns );
    const double measuredTrigger_ns       = ad7689Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns         = measuredTrigger_ns - ad7689Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedMeasuredDeltaT_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelayMeasured_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelayMeasured_ns ) ) );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelayMeasured_ns ) );
    REQUIRE( expectedDelayMeasured_ns != Approx( expectedDelayBlended_ns ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedMeasuredDeltaT_ns );
}

/// Verify synchronized timing applies a positive user offset to the modeled delay target.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming applies positive synchroDelay offset", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_alpha                 = 1.0f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 0, 0 };
    app.m_avgSemaphorePeriod_ns = 250000.0;
    app.m_synchroPostDelay      = 100;

    const timespec secondArrival{ 0, 1000000L };
    app.updateTriggerTiming( secondArrival );

    const double expectedPeriod_ns  = 1000000.0;
    const double rawModelDelay_ns   = 0.5 * expectedPeriod_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedModel_ns   = wrapDelay( rawModelDelay_ns, expectedPeriod_ns );
    const double expectedTarget_ns  = wrapDelay( expectedModel_ns + 100000.0, expectedPeriod_ns );
    const double measuredTrigger_ns = ad7689Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns   = measuredTrigger_ns - ad7689Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedPeriod_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedModel_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedTarget_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedTarget_ns ) );
}

/// Verify synchronized timing applies a negative user offset with modulo wrap into the current period.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming applies negative synchroDelay offset with wrap", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_alpha                 = 1.0f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 0, 0 };
    app.m_avgSemaphorePeriod_ns = 250000.0;
    app.m_synchroPostDelay      = -300;

    const timespec secondArrival{ 0, 1000000L };
    app.updateTriggerTiming( secondArrival );

    const double expectedPeriod_ns  = 1000000.0;
    const double rawModelDelay_ns   = 0.5 * expectedPeriod_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedModel_ns   = wrapDelay( rawModelDelay_ns, expectedPeriod_ns );
    const double expectedTarget_ns  = wrapDelay( expectedModel_ns - 300000.0, expectedPeriod_ns );
    const double measuredTrigger_ns = ad7689Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns   = measuredTrigger_ns - ad7689Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedPeriod_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedModel_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedTarget_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedTarget_ns ) );
}

/// Verify synchronized timing falls back to EMA period when WFS fps is unavailable.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming falls back to EMA period when fps is invalid", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_alpha                 = 0.2f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 0, 0 };
    app.m_avgSemaphorePeriod_ns = 1000000.0;
    app.m_wfs_fps               = 0.0;

    const timespec nextArrival{ 0, 2000000L };
    app.updateTriggerTiming( nextArrival );

    const double alpha              = static_cast<double>( app.m_alpha );
    const double expectedAvg_ns     = alpha * 2000000.0 + ( 1.0 - alpha ) * 1000000.0;
    const double rawDelay_ns        = 0.5 * expectedAvg_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelay_ns   = wrapDelay( rawDelay_ns, expectedAvg_ns );
    const double expectedTrigger_ns = ad7689Ctrl::timespecToNs( nextArrival ) + expectedDelay_ns;
    const double measuredTrigger_ns = ad7689Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns   = measuredTrigger_ns - ad7689Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( measuredTrigger_ns == Approx( expectedTrigger_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelay_ns ) ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedAvg_ns );
}

/// Verify synchronized timing wraps negative raw delays into the current WFS period.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming wraps delay with modulo period", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 5, 0 };
    app.m_avgSemaphorePeriod_ns = 100000.0;
    app.m_wfs_fps               = 20000.0;

    const timespec nextArrival{ 5, 100000L };
    app.updateTriggerTiming( nextArrival );

    const double expectedAvg_ns    = 100000.0;
    const double expectedDeltaT_ns = expectedAvg_ns;
    const double rawDelay_ns       = 0.5 * expectedDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelay_ns  = wrapDelay( rawDelay_ns, expectedDeltaT_ns );
    const double measuredDelay_ns =
        ad7689Ctrl::timespecToNs( app.m_triggerTime ) - ad7689Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( rawDelay_ns < 0.0 );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelay_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelay_ns ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedDeltaT_ns );
}

/// Verify synchronized timing leaves positive raw delays unchanged by modulo wrapping.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming preserves positive raw delay", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 4, 0 };
    app.m_avgSemaphorePeriod_ns = 2000000.0;
    app.m_wfs_fps               = 0.0;

    const timespec nextArrival{ 4, 2000000L };
    app.updateTriggerTiming( nextArrival );

    const double expectedAvg_ns    = 2000000.0;
    const double expectedDeltaT_ns = expectedAvg_ns;
    const double rawDelay_ns       = 0.5 * expectedDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double measuredDelay_ns =
        ad7689Ctrl::timespecToNs( app.m_triggerTime ) - ad7689Ctrl::timespecToNs( nextArrival );

    REQUIRE( rawDelay_ns > 0.0 );
    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( rawDelay_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( rawDelay_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( rawDelay_ns ) );
}

/// Verify synchronized timing constants directly control the modeled delay target.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming uses configurable timing constants", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 9, 0 };
    app.m_avgSemaphorePeriod_ns = 1200000.0;
    app.m_wfs_fps               = 1000.0;
    app.m_synchroDtTransfer_ns  = 4000.0;
    app.m_synchroWfsProcess_ns  = 62000.0;
    app.m_synchroDtF_ns         = 11000.0;
    app.m_synchroWfsRead_ns     = 290000.0;

    const timespec nextArrival{ 9, 1200000L };
    app.updateTriggerTiming( nextArrival );

    const double expectedAvg_ns         = 1200000.0;
    const double rawDelayCustom_ns      = 0.5 * expectedAvg_ns - ( 4000.0 + 62000.0 + 11000.0 + 290000.0 );
    const double rawDelayDefault_ns     = 0.5 * expectedAvg_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelayCustom_ns = wrapDelay( rawDelayCustom_ns, expectedAvg_ns );
    const double measuredDelay_ns =
        ad7689Ctrl::timespecToNs( app.m_triggerTime ) - ad7689Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelayCustom_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelayCustom_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelayCustom_ns ) );
    REQUIRE( expectedDelayCustom_ns != Approx( wrapDelay( rawDelayDefault_ns, expectedAvg_ns ) ) );
}

/// Verify synchronized timing leaves trigger time unchanged when period estimate is non-positive.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl updateTriggerTiming guards non-positive period", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTriggerTiming;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_triggerTime          = timespec{ 7, 12345L };
    app.m_triggerInterval_ns   = 42.0;
    app.m_synchroDelayTarget   = 12345.0f;
    app.m_delayModel_ns        = 12345.0;
    app.m_wfsPeriodMeasured_ns = 67890.0;
    app.m_firstSemaphore       = true;
    app.m_wfs_fps              = 0.0;

    const timespec nextArrival{ 7, 54321L };
    app.updateTriggerTiming( nextArrival );

    REQUIRE( app.m_firstSemaphore == false );
    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( 0.0 ) );
    REQUIRE( app.m_lastAtime.tv_sec == nextArrival.tv_sec );
    REQUIRE( app.m_lastAtime.tv_nsec == nextArrival.tv_nsec );
    REQUIRE( app.m_triggerTime.tv_sec == 7 );
    REQUIRE( app.m_triggerTime.tv_nsec == 12345L );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 12345.0f ) );
    REQUIRE( app.m_delayModel_ns == Approx( 12345.0 ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 0.0 ) );
}

/// Verify timer-driven acquisition configures the published frame geometry.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timer mode configureAcquisition sizes the output frame", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::configureAcquisition;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_numChannels = 3;

    app.m_requestedChannels = app.m_numChannels;
    REQUIRE( app.configureAcquisition() == 0 );
    REQUIRE( app.m_values.size() == 3 );
    REQUIRE( app.m_width == 3 );
    REQUIRE( app.m_height == 1 );
    REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
}

/// Verify timer-driven acquisition still reads one frame of ADC values.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timer mode reads configured channels", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 11, 22, 33 };

    app.m_numChannels = 3;
    app.m_values.assign( 3, 0 );
    app.m_gain       = 0;
    app.m_trigger    = 0;
    app.m_time_start = std::chrono::steady_clock::now();

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 11, 22, 33 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1, 2 } ) );
}

/// Verify timer-mode trigger interval diagnostics initialize then report measured intervals.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl timer mode trigger interval initializes then measures", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 17 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_gain    = 0.0f;
    app.m_trigger = 1000.0f;

    app.m_time_start = std::chrono::steady_clock::now() - std::chrono::milliseconds( 2 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );

    app.m_time_start = std::chrono::steady_clock::now() - std::chrono::milliseconds( 4 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns > 1000000.0 );
}

/// Verify the current AD7689 values are copied into the output image buffer.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl loadImageIntoStream copies the current values", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
#endif
    // clang-format on
    ad7689Ctrl_test       app;
    std::vector<uint16_t> dest( 3, 0 );

    app.m_values = { 5, 6, 7 };

    app.m_frameValid = true;
    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( dest == std::vector<uint16_t>( { 5, 6, 7 } ) );
}

/// Verify published-frame counters track local and synchronized stream writes.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl loadImageIntoStream updates frame mapping counters", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
#endif
    // clang-format on
    ad7689Ctrl_test       app;
    std::vector<uint16_t> dest( 2, 0 );

    app.m_values = { 9, 8 };

    app.m_frameValid = true;
    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( app.m_localFrameSeq == 1 );
    REQUIRE( app.m_syncFramesWritten == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_frameValid       = true;
    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( app.m_localFrameSeq == 2 );
    REQUIRE( app.m_syncFramesWritten == 1 );
}

/// Verify synchronized acquisition performs one ADC sweep per semaphore wake.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized mode reads on semaphore wake", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();
    stubState().m_channelValues = { 101, 202, 303 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 3;
    app.m_values.assign( 3, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelay       = 0;
    app.m_synchroDelayTarget = 0;
    app.m_gain               = 0;
    app.m_wfs_fps            = 2000.0;

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 101, 202, 303 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1, 2 } ) );
    REQUIRE( app.m_firstSemaphore == false );
    REQUIRE( app.m_lastAtime.tv_sec > 0 );
    REQUIRE( app.m_triggerTime.tv_sec > 0 ); // The fixture supplies an explicit startup WFS cadence.
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    REQUIRE( app.m_firstReadLatency == false );
    REQUIRE( app.m_avgReadLatency_ns ==
             Approx( static_cast<double>( app.m_syncSchedule.m_lastSample - app.m_syncSchedule.m_anchor ) ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized mode derives producer cadence from stream metadata counters and timestamps.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized mode tracks producer cadence from metadata", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;
    IMAGE_METADATA  metadata{};

    resetStubState();
    stubState().m_channelValues = { 44 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroStream.md   = &metadata;
    app.m_synchroDelay       = 0.0f;
    app.m_synchroDelayTarget = 0.0f;
    app.m_gain               = 0.0f;
    app.m_alpha              = 0.5f;
    app.m_wfs_fps            = 2000.0;

    metadata.cnt0  = 100;
    metadata.atime = timespec{ 10, 0 };
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    REQUIRE( app.m_firstProducerSample == false );
    REQUIRE( app.m_producerPeriodInst_ns == Approx( 0.0 ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( 0.0 ) );
    REQUIRE( app.m_syncFramesReceived == 1 );
    REQUIRE( app.m_syncFramesDropped == 0 );
    REQUIRE( app.m_syncFrameIdGapCount == 0 );
    REQUIRE( app.m_syncProducerFrameId == 100 );
    REQUIRE( app.m_syncProducerFrameDelta == 0 );
    REQUIRE( app.m_syncProducerFrameValid == true );

    metadata.cnt0    = 102;
    metadata.atime   = timespec{ 10, 1000000L };
    app.m_fakeNow_ns = app.m_lastSyncArrival_ns + 1000000;
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    REQUIRE( app.m_producerPeriodInst_ns == Approx( 500000.0 ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( 500000.0 ) );
    REQUIRE( app.m_lastProducerCnt0 == 102 );
    REQUIRE( app.m_syncFramesReceived == 2 );
    REQUIRE( app.m_syncFramesDropped == 1 );
    REQUIRE( app.m_syncFrameIdGapCount == 1 );
    REQUIRE( app.m_syncProducerFrameId == 102 );
    REQUIRE( app.m_syncProducerFrameDelta == 2 );

    metadata.cnt0    = 104;
    metadata.atime   = timespec{ 10, 2100000L };
    app.m_fakeNow_ns = app.m_lastSyncArrival_ns + 1000000;
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double expectedPeriod2_ns = 550000.0;
    const double expectedAvg2_ns    = static_cast<double>( app.m_alpha ) * expectedPeriod2_ns +
                                      ( 1.0 - static_cast<double>( app.m_alpha ) ) * 500000.0;
    REQUIRE( app.m_producerPeriodInst_ns == Approx( expectedPeriod2_ns ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( expectedAvg2_ns ) );
    REQUIRE( app.m_syncFramesReceived == 3 );
    REQUIRE( app.m_syncFramesDropped == 2 );
    REQUIRE( app.m_syncFrameIdGapCount == 2 );
    REQUIRE( app.m_syncProducerFrameId == 104 );
    REQUIRE( app.m_syncProducerFrameDelta == 2 );

    const double periodBeforeNoAdvance_ns = app.m_producerPeriodInst_ns;
    const double avgBeforeNoAdvance_ns    = app.m_avgProducerPeriod_ns;

    metadata.cnt0    = 104;
    metadata.atime   = timespec{ 10, 2200000L };
    app.m_fakeNow_ns = app.m_lastSyncArrival_ns + 1000000;
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncDuplicates == 1 );

    REQUIRE( app.m_producerPeriodInst_ns == Approx( periodBeforeNoAdvance_ns ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( avgBeforeNoAdvance_ns ) );
    REQUIRE( app.m_syncFramesReceived == 4 );
    REQUIRE( app.m_syncFramesDropped == 2 );
    REQUIRE( app.m_syncFrameIdGapCount == 2 );
    REQUIRE( app.m_syncProducerFrameId == 104 );
    REQUIRE( app.m_syncProducerFrameDelta == 0 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized read-latency EMA initializes from the first sample and smooths subsequent samples.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized read latency EMA initializes and smooths", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();
    stubState().m_channelValues = { 77 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelayTarget = 0.0f;
    app.m_synchroDelay       = 0.0f;
    app.m_gain               = 0.0f;
    app.m_alpha              = 0.2f;
    app.m_firstReadLatency   = true;
    app.m_avgReadLatency_ns  = 0.0;

    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double readLatency0_ns = static_cast<double>( app.m_syncSchedule.m_lastSample - app.m_syncSchedule.m_anchor );

    REQUIRE( app.m_firstReadLatency == false );
    REQUIRE( app.m_avgReadLatency_ns == Approx( readLatency0_ns ) );

    app.m_fakeNow_ns = app.m_lastSyncArrival_ns + 500000;
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double readLatency1_ns = static_cast<double>( app.m_syncSchedule.m_lastSample - app.m_syncSchedule.m_anchor );
    const double alpha           = static_cast<double>( app.m_alpha );
    const double expectedAvgLatency_ns = alpha * readLatency1_ns + ( 1.0 - alpha ) * readLatency0_ns;

    REQUIRE( app.m_avgReadLatency_ns == Approx( expectedAvgLatency_ns ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized non-delay service EMA uses the configurable global alpha.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized non-delay service EMA uses global alpha", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();
    stubState().m_channelValues = { 55 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore      = &semaphore;
    app.m_synchroDelayTarget    = 0.0f;
    app.m_synchroDelay          = 0.0f;
    app.m_gain                  = 0.0f;
    app.m_alpha                 = 0.2f;
    app.m_firstNonDelayService  = true;
    app.m_avgNonDelayService_ns = 0.0;

    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double nonDelay0_ns = app.m_nonDelayService_ns;
    REQUIRE( app.m_firstNonDelayService == false );
    REQUIRE( app.m_avgNonDelayService_ns == Approx( nonDelay0_ns ) );

    app.m_fakeNow_ns = app.m_lastSyncArrival_ns + 500000;
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double nonDelay1_ns           = app.m_nonDelayService_ns;
    const double alpha                  = static_cast<double>( app.m_alpha );
    const double expectedAvgNonDelay_ns = alpha * nonDelay1_ns + ( 1.0 - alpha ) * nonDelay0_ns;

    REQUIRE( app.m_avgNonDelayService_ns == Approx( expectedAvgNonDelay_ns ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control uses read-latency EMA for the integrator correction.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized delay controller uses read latency EMA", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();
    stubState().m_channelValues = { 99 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelayTarget = 0.0f;
    app.m_synchroDelay       = 0.0f;
    app.m_gain               = 1.0f;
    app.m_alpha              = 0.2f;
    app.m_firstReadLatency   = false;
    app.m_avgReadLatency_ns  = 800000.0;

    app.m_firstSemaphore = false;
    clock_gettime( CLOCK_REALTIME, &app.m_lastAtime );
    app.m_lastAtime             = ad7689Ctrl::nsToTimespec( ad7689Ctrl::timespecToNs( app.m_lastAtime ) - 10000000 );
    app.m_avgSemaphorePeriod_ns = 10000000;
    app.m_requestedWfsFps = app.m_wfs_fps = 100;
    REQUIRE( app.acquireAndCheckValid() == 0 );
    const unsigned referenceSlot = static_cast<unsigned>( std::clamp(
        std::floor( ( app.m_synchroDelayTarget - app.m_syncSchedule.m_phase ) / app.m_syncSchedule.m_interval + 0.5 ),
        0.0,
        static_cast<double>( app.m_syncSchedule.m_count - 1 ) ) );

    while( app.m_syncSchedule.m_slot <= referenceSlot )
    {
        REQUIRE( app.acquireAndCheckValid() == 0 );
    }

    const double readLatency_ns = static_cast<double>( app.m_syncSchedule.m_lastSample - app.m_syncSchedule.m_anchor );
    const double alpha          = static_cast<double>( app.m_alpha );
    const double expectedAvgLatency_ns = alpha * readLatency_ns + ( 1.0 - alpha ) * 800000.0;
    const double expectedDelay_ns =
        std::clamp( 2.0 * app.m_synchroDelayTarget - expectedAvgLatency_ns, 0.0, app.m_delayBudget_ns );

    REQUIRE( app.m_avgReadLatency_ns == Approx( expectedAvgLatency_ns ) );
    REQUIRE( app.m_synchroDelay == Approx( expectedDelay_ns ) );
    REQUIRE( app.m_values[0] == 99 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control clamps at zero when the control step overshoots.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized delay controller clamps to zero", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();
    stubState().m_channelValues = { 11 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelayTarget = 0.0f;
    app.m_synchroDelay       = 1000.0f;
    app.m_gain               = 1.0f;
    app.m_firstReadLatency   = false;
    app.m_avgReadLatency_ns  = 2000000;

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_values[0] == 11 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control blocks integrator windup while cadence capping is active.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized delay controller applies anti-windup at cap", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::updateSynchroDelayController;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_gain                 = 1.0f;
    app.m_synchroDelayTarget   = 300000.0f;
    app.m_wfsPeriodMeasured_ns = 500000.0;
    app.m_delayBudget_ns       = 100000.0;
    app.m_delayApplied_ns      = 100000.0;

    // While capped high, negative error would normally increase command; anti-windup should hold at the cap.
    app.m_avgReadLatency_ns = 100000.0;
    app.updateSynchroDelayController( 500000.0 );
    REQUIRE( app.m_synchroDelay == Approx( 100000.0f ) );

    // If the controller correction reduces delay, allow command to move down below the cap.
    app.m_avgReadLatency_ns = 350000.0;
    app.updateSynchroDelayController( 110000.0 );
    REQUIRE( app.m_synchroDelay == Approx( 60000.0f ) );
}

/// Verify synchronized timeout requests reconfiguration when the trigger stream is stale.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized timeout requests reconfig for a stale stream", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::reconfig;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    sem_t           semaphore;

    resetStubState();

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName  = "camwfs_sync";
    app.m_values            = { 7, 8 };
    app.m_synchroSemaphore  = &semaphore;
    app.m_synchroStreamOpen = false;

    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_reconfig == true );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 7, 8 } ) );
    REQUIRE( stubState().m_readOrder.empty() );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify stale-stream detection notices a missing synchronization stream backing file.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl synchronized stale helper detects missing streams", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::synchroStreamStale;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    IMAGE_METADATA  metadata{};

    app.m_synchroShmimName  = "ad7689Ctrl_unit_test_missing_stream";
    app.m_synchroStreamOpen = true;
    metadata.sem            = SEMAPHORE_MAXVAL;
    app.m_synchroStream.md  = &metadata;

    REQUIRE( app.synchroStreamStale() == true );
}

/// Verify `reconfig()` clears cached synchronization state.
/**
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl reconfig clears cached synchronization state", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::reconfig;
#endif
    // clang-format on
    ad7689Ctrl_test app;

    app.m_synchroSemaphore        = reinterpret_cast<sem_t *>( 0x1 );
    app.m_synchroSemaphoreNumber  = 7;
    app.m_synchroStreamInode      = 1234;
    app.m_synchroStreamOpen       = false;
    app.m_atime                   = timespec{ 1, 1 };
    app.m_lastAtime               = timespec{ 2, 2 };
    app.m_avgSemaphorePeriod_ns   = 42.0;
    app.m_wfsPeriodMeasured_ns    = 21.0;
    app.m_lastProducerAtime       = timespec{ 3, 4 };
    app.m_lastProducerCnt0        = 123;
    app.m_producerPeriodInst_ns   = 500000.0;
    app.m_avgProducerPeriod_ns    = 510000.0;
    app.m_firstProducerSample     = false;
    app.m_localFrameSeq           = 22;
    app.m_syncFramesReceived      = 21;
    app.m_syncFramesWritten       = 20;
    app.m_syncFramesDropped       = 4;
    app.m_syncFrameIdGapCount     = 3;
    app.m_syncProducerFrameId     = 1234567;
    app.m_syncProducerFrameDelta  = 5;
    app.m_lastSyncProducerFrameId = 1234562;
    app.m_syncProducerFrameValid  = true;
    app.m_firstSemaphore          = false;
    app.m_avgReadLatency_ns       = 84.0;
    app.m_firstReadLatency        = false;
    app.m_delayModel_ns           = 900.0;
    app.m_delayApplied_ns         = 875.0;
    app.m_delayBudget_ns          = 450000.0;
    app.m_nonDelayService_ns      = 170000.0;
    app.m_avgNonDelayService_ns   = 160000.0;
    app.m_firstNonDelayService    = false;
    app.m_delayPhaseError_ns      = -25.0;
    app.m_delayLock               = 1.0;
    app.m_delayCapped             = 1.0;
    app.m_triggerTime             = timespec{ 3, 3 };
    app.m_triggerInterval_ns      = 21.0;
    app.m_lastTriggerTime         = timespec{ 4, 4 };
    app.m_firstTriggerTime        = false;
    app.m_firstTimerTrigger       = false;

    REQUIRE( app.reconfig() == 0 );
    REQUIRE( app.m_synchroSemaphore == nullptr );
    REQUIRE( app.m_synchroSemaphoreNumber == 5 );
    REQUIRE( app.m_synchroStreamInode == 0 );
    REQUIRE( app.m_synchroStreamOpen == false );
    REQUIRE( app.m_atime.tv_sec == 0 );
    REQUIRE( app.m_atime.tv_nsec == 0 );
    REQUIRE( app.m_lastAtime.tv_sec == 0 );
    REQUIRE( app.m_lastAtime.tv_nsec == 0 );
    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( 0.0 ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 0.0 ) );
    REQUIRE( app.m_lastProducerAtime.tv_sec == 0 );
    REQUIRE( app.m_lastProducerAtime.tv_nsec == 0 );
    REQUIRE( app.m_lastProducerCnt0 == 0 );
    REQUIRE( app.m_producerPeriodInst_ns == Approx( 0.0 ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( 0.0 ) );
    REQUIRE( app.m_firstProducerSample == true );
    REQUIRE( app.m_localFrameSeq == 0 );
    REQUIRE( app.m_syncFramesReceived == 0 );
    REQUIRE( app.m_syncFramesWritten == 0 );
    REQUIRE( app.m_syncFramesDropped == 0 );
    REQUIRE( app.m_syncFrameIdGapCount == 0 );
    REQUIRE( app.m_syncProducerFrameId == 0 );
    REQUIRE( app.m_syncProducerFrameDelta == 0 );
    REQUIRE( app.m_lastSyncProducerFrameId == 0 );
    REQUIRE( app.m_syncProducerFrameValid == false );
    REQUIRE( app.m_firstSemaphore == true );
    REQUIRE( app.m_avgReadLatency_ns == Approx( 0.0 ) );
    REQUIRE( app.m_firstReadLatency == true );
    REQUIRE( app.m_delayModel_ns == Approx( 0.0 ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 0.0 ) );
    REQUIRE( app.m_delayBudget_ns == Approx( 0.0 ) );
    REQUIRE( app.m_nonDelayService_ns == Approx( 0.0 ) );
    REQUIRE( app.m_avgNonDelayService_ns == Approx( 0.0 ) );
    REQUIRE( app.m_firstNonDelayService == true );
    REQUIRE( app.m_delayPhaseError_ns == Approx( 0.0 ) );
    REQUIRE( app.m_delayLock == Approx( 0.0 ) );
    REQUIRE( app.m_delayCapped == Approx( 0.0 ) );
    REQUIRE( app.m_triggerTime.tv_sec == 0 );
    REQUIRE( app.m_triggerTime.tv_nsec == 0 );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( app.m_lastTriggerTime.tv_sec == 0 );
    REQUIRE( app.m_lastTriggerTime.tv_nsec == 0 );
    REQUIRE( app.m_firstTriggerTime == true );
    REQUIRE( app.m_firstTimerTrigger == true );
}

/** Reject malformed channel maps and impossible rates without accessing hardware.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl validates channel maps and ADC budgets", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
    MagAOX::app::ad7689Ctrl::setupConfig;
#endif
    // clang-format on
    for( const std::string list : { "", "[]", "0,0", "8", "-1", "0.5", "0,", "0,x", "0,1,2,3,4,5,6,7,0" } )
    {
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689Ctrl_invalid.conf", { "accel" }, { "channels" }, { list } );
        app.config.readConfig( "/tmp/ad7689Ctrl_invalid.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == -1 );
    }

    ad7689Ctrl_test app;
    app.setupConfig();
    mx::app::writeConfigFile( "/tmp/ad7689Ctrl_ordered.conf", { "accel" }, { "channels" }, { "[7,0,4]" } );
    app.config.readConfig( "/tmp/ad7689Ctrl_ordered.conf" );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_numChannels == 3 );
    REQUIRE( app.m_channels == std::vector<unsigned>{ 7, 0, 4 } );
    REQUIRE( app.requestAcquisition( 8000, 8 ) == -1 );
    REQUIRE( app.requestAcquisition( 2000, 2 ) == -1 ); // Explicit map fixes frame width.
}

/** Keep validated controls staged until reconfiguration and preserve fractional frame rates.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl stages controls and rejects invalid rates", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_fps;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupFpsProperty();
    const double initial = app.m_fps;
    REQUIRE( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 1234.56789 ) ) == 0 );
    REQUIRE( app.m_fps == initial );
    REQUIRE( app.applyControls() );
    REQUIRE( app.m_fps == initial );
    REQUIRE( app.applyControls( true ) );
    REQUIRE( app.m_fps == Approx( 1234.56789 ) );
    REQUIRE_FALSE( app.applyControls() );

    for( double rate : { 0.0,
                         -1.0,
                         10001.0,
                         1e-40,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN() } )
    {
        REQUIRE( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( rate ) ) == -1 );
        REQUIRE( app.m_indiP_fps.getState() == INDI_ALERT );
        REQUIRE( app.m_fps == Approx( 1234.56789 ) );
    }

    for( const std::string value : { "garbage", "1000 trailing", "1,2", "nan", "inf" } )
    {
        auto update = app.makeFpsUpdate( 1000 );
        update["target"].setValue( value );
        REQUIRE( app.newCallBack_m_indiP_fps( update ) == -1 );
    }

    app.setupAlphaProperty();
    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( std::numeric_limits<double>::quiet_NaN() ) ) == -1 );
    REQUIRE( app.m_indiP_alpha.getState() == INDI_ALERT );
    app.setupSynchroDelayProperty();
    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( 0.5 ) ) == -1 );
    REQUIRE( app.m_indiP_synchroDelay.getState() == INDI_ALERT );
    REQUIRE( app.requestAcquisition( 2000, 1.5 ) == -1 );
    REQUIRE( app.requestAcquisition( 2000, 0 ) == -1 );
    REQUIRE( app.requestAcquisition( 2000, 9 ) == -1 );
}

/** Commit full-range ordered samples once, retaining software scan-start timestamps.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl publishes only complete fresh unsigned frames", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::configureAcquisition;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
    MagAOX::app::ad7689Ctrl::readScan;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.m_explicitChannels  = true;
    app.m_channels          = { 7, 0, 4 };
    app.m_requestedChannels = 3;
    REQUIRE( app.configureAcquisition() == 0 );
    stubState().m_channelValues = { 0, 1, 2, 3, 4096, 5, 6, 65535 };
    REQUIRE( app.readScan() == 0 );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    REQUIRE( stubState().m_readOrder == std::vector<int>{ 7, 0, 4 } );
    std::array<uint16_t, 3> frame{};
    REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
    REQUIRE( frame == std::array<uint16_t, 3>{ 65535, 0, 4096 } );
    REQUIRE( app.loadImageIntoStream( frame.data() ) == -1 );
    REQUIRE( app.m_localFrameSeq == 1 );
    REQUIRE( app.m_usefulConversions == 3 );
    stubState().m_failScan = true;
    REQUIRE( app.readScan() == 1 );
    REQUIRE( app.loadImageIntoStream( frame.data() ) == -1 );
    REQUIRE( app.m_invalidFrames == 1 );
    REQUIRE( frame == std::array<uint16_t, 3>{ 65535, 0, 4096 } );
    resetStubState();
}

/** Discard in-flight frames on shutdown or requested geometry change, and bound recovery attempts.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl stops partial publication and bounds recovery", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::configureAcquisition;
    MagAOX::app::ad7689Ctrl::prepareAdc;
    MagAOX::app::ad7689Ctrl::readScan;
    MagAOX::app::ad7689Ctrl::requestAcquisition;
#endif
    // clang-format on
    for( bool stopping : { false, true } )
    {
        resetStubState();
        ad7689Ctrl_test app;
        REQUIRE( app.configureAcquisition() == 0 );
        stubState().m_duringRead = [&]()
        {
            if( stopping )
            {
                app.m_shutdown = true;
            }
            else
            {
                REQUIRE( app.requestAcquisition( 1000, 1 ) == 0 );
            }
        };
        REQUIRE( app.readScan() == 1 );
        REQUIRE_FALSE( app.m_frameValid );
        REQUIRE( app.m_invalidFrames == 1 );
    }

    resetStubState();
    ad7689Ctrl_test app;
    stubState().m_failPrepare = true;

    for( unsigned attempt = 0; attempt < 3; ++attempt )
    {
        REQUIRE( app.prepareAdc() == -1 );
    }

    REQUIRE( app.m_failureAttempts == 3 );
    REQUIRE( app.prepareAdc() == -1 );
    REQUIRE( app.m_shutdown );
    resetStubState();
}

/** Count late timer periods and explicitly reject a semaphore cadence that cannot fit a scan.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl accounts for missed deadlines and impossible trigger cadence", "[ad7689Ctrl]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireTimerAndCheckValid;
    MagAOX::app::ad7689Ctrl::acquireSynchroAndCheckValid;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    REQUIRE( app.configureAcquisition() == 0 );
    app.m_time_start = std::chrono::steady_clock::now() - std::chrono::milliseconds( 10 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_missedPeriods >= 19 );
    REQUIRE( app.m_trigger >= 0.5f * app.m_nanoSecTarget );
    REQUIRE( app.m_trigger <= app.m_nanoSecTarget );
    REQUIRE( app.m_localFrameSeq == 0 );
    resetStubState();
    sem_t semaphore;
    REQUIRE( sem_init( &semaphore, 0, 1 ) == 0 );
    app.m_synchroShmimName        = "offline-too-fast";
    app.m_synchroSemaphore        = &semaphore;
    app.m_alpha                   = 0;
    app.m_firstSemaphore          = false;
    app.m_avgSemaphorePeriod_ns   = 20000;
    app.m_haveMeasuredCadence     = true;
    app.m_haveSyncArrival         = true;
    app.m_lastSyncArrival_ns      = app.m_fakeNow_ns - 20000;
    app.m_syncProducerFrameValid  = true;
    app.m_lastSyncProducerFrameId = 0;
    clock_gettime( CLOCK_REALTIME, &app.m_lastAtime );
    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_overruns == 1 );
    REQUIRE_FALSE( app.m_frameValid );
    REQUIRE( stubState().m_readOrder.empty() );
    REQUIRE( sem_destroy( &semaphore ) == 0 );
    app.m_synchroSemaphore = nullptr;
}

/** Validate startup WFS hints and silence deadlines without conflating them with ADC rate controls.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl validates WFS scheduler configuration", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
#endif
    // clang-format on
    { // Release the framework singleton before constructing the invalid-config fixtures.
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689-sync-config.conf",
                                  { "synchro", "synchro", "synchro", "accel" },
                                  { "shmimName", "referenceFps", "timeout", "fps" },
                                  { "offline", "2000", "0.05", "8000" } );
        app.config.readConfig( "/tmp/ad7689-sync-config.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_requestedFps == 8000 );
        REQUIRE( app.m_requestedWfsFps == 2000 );
        REQUIRE( app.m_syncTimeout == 0.05 );
    }

    for( const std::string key : { "referenceFps", "timeout" } )
    {
        ad7689Ctrl_test invalid;
        invalid.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689-sync-invalid.conf", { "synchro" }, { key }, { "-1" } );
        invalid.config.readConfig( "/tmp/ad7689-sync-invalid.conf" );
        REQUIRE( invalid.loadConfigImpl( invalid.config ) == -1 );
    }
}

/** One WFS reference publishes four or three complete frames on an independent, evenly spaced scan grid.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl publishes multiple evenly spaced scans per WFS frame", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireSynchroAndCheckValid;
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
#endif
    // clang-format on
    for( double rate : { 8000.0, 6000.0 } )
    {
        resetStubState();
        ad7689Ctrl_test app;
        app.setupSync( rate, 2000 );
        app.postReference( 1, 500000 );
        REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
        const unsigned count = rate == 8000 ? 4 : 3;
        REQUIRE( app.m_syncSchedule.m_count == count );
        REQUIRE( app.m_syncSchedule.fps() == Approx( rate ) );
        REQUIRE( app.m_syncFramesReceived == 1 );
        REQUIRE( app.m_syncFramesWritten == 0 );
        int64_t first = 0;

        for( unsigned slot = 0; slot < count; ++slot )
        {
            REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );

            if( slot == 0 )
            {
                first = app.m_syncSchedule.m_lastSample;
            }

            const double elapsed = static_cast<double>( app.m_syncSchedule.m_lastSample - first );
            REQUIRE( elapsed == Approx( slot * 1e9 / rate ).margin( 1 ) );
            REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
            std::array<uint16_t, 2> frame{};
            REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
            REQUIRE( frame == std::array<uint16_t, 2>{ 65535, 4096 } );
        }

        REQUIRE( app.m_syncFramesReceived == 1 );
        REQUIRE( app.m_syncFramesWritten == count );
        REQUIRE( app.m_localFrameSeq == count );
        REQUIRE( app.m_usefulConversions == 2 * count );
        REQUIRE( app.m_syncSchedule.m_skipped == 0 );
        REQUIRE( app.m_syncSchedule.m_slot == count );
        resetStubState();
    }
}

/** Separate a WFS FPS subscription from the independent ADC control, and wait for measured startup cadence.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl learns WFS cadence without changing requested ADC fps", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::setCallBack_m_indiP_fpsSource;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.setupSync( 8000, 0 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 0 );
    REQUIRE( stubState().m_readOrder.empty() );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_haveMeasuredCadence );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 500000 ) );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    app.setupFpsSourceProperty();
    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 1800 ) ) == 0 );
    REQUIRE( app.m_requestedFps == 8000 );
    REQUIRE_FALSE( app.applyControls() );
    REQUIRE( app.fps() == 8000 );
    REQUIRE( app.m_wfs_fps == 1800 );
    REQUIRE( app.m_syncSchedule.m_count == 4 ); // Metadata alone neither adds nor re-anchors slots.
    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( -1 ) ) == -1 );
    REQUIRE( app.m_requestedFps == 8000 );
    resetStubState();
}

/** A changed WFS cadence re-anchors the finite schedule, reports rate quantization, and preserves ADC admission.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl follows changed WFS cadence and reports actual scan rate", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::captureTimingDiagnostics;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.postReference( 1, 1000000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 8 );
    REQUIRE( app.m_syncSchedule.m_interval == Approx( 125000 ) );
    app.postReference( 1, 555556 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.fps() == Approx( 7200 ).epsilon( 1e-5 ) );
    REQUIRE( app.fps() == 8000 );
    app.captureTimingDiagnostics();
    REQUIRE( app.m_syncSnapshot[0] == 4 );
    REQUIRE( app.m_syncSnapshot[1] == Approx( 7200 ).epsilon( 1e-5 ) );
    REQUIRE( app.m_syncSnapshot[2] == Approx( 138.889 ).margin( 0.001 ) );
    app.postReference( 1, 20000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 0 );
    REQUIRE( app.m_overruns == 1 );
}

/** Coalesce backlogs, ignore duplicate IDs, account for source gaps, and relearn after counter resets.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl rejects duplicate WFS triggers and accounts for source restarts", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::resetSyncSchedule;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.postReference( 1, 500000, 10 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncFramesReceived == 10 );
    REQUIRE( app.m_coalescedWakes == 9 );
    const int64_t anchor = app.m_syncSchedule.m_anchor;
    app.postReference( 0, 0 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncDuplicates == 1 );
    REQUIRE( app.m_syncSchedule.m_anchor == anchor );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    app.postReference( 4, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncFramesDropped == 3 );
    REQUIRE( app.m_syncFrameIdGapCount == 1 );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 500000 ) );
    app.m_testMetadata.cnt0 = 0;
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_cadenceResets == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.m_slot == 0 );
}

/** Source loss ends the finite schedule; delayed scans are counted and never replayed in a burst.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl stops subscans on source loss and skips expired slots", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireSynchroAndCheckValid;
    MagAOX::app::ad7689Ctrl::resetSyncSchedule;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.m_fakeNow_ns = app.m_syncSchedule.m_anchor + 250000;
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    REQUIRE( app.m_syncSchedule.m_skipped >= 1 );
    REQUIRE( stubState().m_readOrder.size() == 2 );
    const int64_t           last = app.m_syncSchedule.m_lastSample;
    std::array<uint16_t, 2> frame{};
    REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    REQUIRE( app.m_syncSchedule.m_lastSample - last >= 100000 );
    REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
    app.m_fakeNow_ns = app.m_syncSchedule.end() + 1;
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 0 );
    REQUIRE_FALSE( app.m_frameValid );
    REQUIRE( app.m_syncFramesWritten == 2 );
    REQUIRE( stubState().m_readOrder.size() == 4 );
    REQUIRE( app.m_cadenceResets == 0 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_cadenceResets == 1 );
    app.captureTimingDiagnostics();
    REQUIRE( app.m_delayLock == 0 );
    resetStubState();
}

/** Reconfiguration cancels old slots before a new ADC rate can publish data.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl changes subscan rate only at a stopped acquisition boundary", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::requestAcquisition;
    MagAOX::app::ad7689Ctrl::reconfig;
    MagAOX::app::ad7689Ctrl::applyControls;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.requestAcquisition( 6000, 2 ) == 0 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_reconfig );
    REQUIRE_FALSE( app.m_frameValid );
    REQUIRE( app.reconfig() == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 0 );
    REQUIRE( app.m_syncSchedule.m_skipped == 4 );
    REQUIRE( app.applyControls( true ) );
    app.m_synchroSemaphore = &app.m_testSemaphore;
    app.m_synchroStream.md = &app.m_testMetadata;
    app.m_nextSyncProbe_ns = std::numeric_limits<int64_t>::max();
    app.m_reconfig         = false;
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 3 );
    REQUIRE( app.m_syncSchedule.fps() == Approx( 6000 ) );
}

/** A producer realtime step cannot steer the monotonic subscan cadence or read-latency controller.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl ignores producer realtime clock steps when scheduling", "[ad7689Ctrl][wfs-subscan]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::acquireSynchroAndCheckValid;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.m_testMetadata.atime.tv_sec += 3600;
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 500000 ) );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.m_interval == Approx( 125000 ) );

    for( unsigned slot = 0; slot < 4; ++slot )
    {
        REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    }

    REQUIRE( app.m_avgReadLatency_ns < 500000 );
    REQUIRE( app.m_avgReadLatency_ns > 0 );
}

/** The app consumes all cross-boundary slots, retaining each original reference and distinct scan timestamp.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC continuous acquisition across WFS boundaries", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireDdspcAndCheckValid;
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    const int64_t epoch         = app.m_fakeNow_ns;
    int64_t       nextReference = epoch + 500000;
    app.m_beforeWait            = [&]( int64_t until )
    {
        if( nextReference <= until )
        {
            app.m_fakeNow_ns = nextReference;
            app.postReference( 1, 500000 );
            nextReference += 500000;
        }
    };
    stubState().m_timestamp  = [&]( timespec &stamp ) { app.getRealtime( stamp ); };
    stubState().m_duringRead = [&]() { app.m_fakeNow_ns += 15000; };
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_ddspcTarget_ns == 300000 );

    for( uint64_t i = 0; i < 32; ++i )
    {
        int result;

        do
        {
            result = app.ad7689Ctrl::acquireAndCheckValid();
        } while( result == 1 && !app.m_reconfig && !app.m_shutdown );

        REQUIRE( result == 0 );
        const int64_t expected = epoch + 300000 + i * 125000;
        REQUIRE( app.m_ddspcSlot.m_deadline == expected );
        REQUIRE( app.m_ddspcStart_ns == expected );
        REQUIRE( app.m_ddspcCompletion_ns == expected + 15000 );
        REQUIRE( app.m_delayApplied_ns == 300000 );
        REQUIRE( app.m_ddspcSlot.m_reference.m_id == 1 + i / 4 );
        REQUIRE( app.m_ddspcSlot.m_slot == i % 4 );
        REQUIRE( app.m_currImageTimestamp.tv_nsec == expected % 1000000000 );
        std::array<uint16_t, 2> frame{};
        REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
        REQUIRE( frame == std::array<uint16_t, 2>{ 65535, 4096 } );
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
        REQUIRE( app.m_ddspcSnapshot[16] == 15 );
    }

    REQUIRE( app.m_localFrameSeq == 32 );
    REQUIRE( app.m_ddspcSchedule.m_epochs == 1 );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 0 );
    REQUIRE( app.m_lateScans == 0 );
    REQUIRE( app.m_syncFramesReceived < app.m_syncFramesWritten );
    resetStubState();
}

/** Cadence changes explicitly reacquire phase while preserving the independent ACC target and scan admission.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC cadence change and effective rate", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::captureTimingDiagnostics;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    app.postReference( 1, 1000000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.postReference( 1, 1000000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_ddspcTarget_ns == 550000 );
    REQUIRE( app.m_ddspcSchedule.m_count == 8 );
    REQUIRE( app.m_ddspcSchedule.m_interval == 125000 );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 3 );
    REQUIRE( app.m_ddspcSchedule.m_epochs == 2 );
    REQUIRE( app.fps() == 8000 );
    app.postReference( 1, 555556 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.postReference( 1, 555556 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSnapshot[0] == 4 );
    REQUIRE( app.m_syncSnapshot[1] == Approx( 7200 ).epsilon( 1e-5 ) );
    app.postReference( 1, 20000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.postReference( 1, 20000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
    REQUIRE( app.m_overruns == 1 );
}

/** Slots can finish after a missing next trigger, then acquisition suspends and reacquires without catch-up.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC source loss duplicate and recovery", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireSynchroAndCheckValid;
    MagAOX::app::ad7689Ctrl::resetSyncSchedule;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    const int64_t frozen = app.m_ddspcSchedule.deadline();
    app.postReference( 0, 0, 4 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncDuplicates == 1 );
    REQUIRE( app.m_ddspcSchedule.deadline() == frozen );
    app.m_fakeNow_ns = frozen + 260000;
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    REQUIRE( app.m_ddspcSlot.m_slot == 2 );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 2 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    REQUIRE( app.m_ddspcSlot.m_slot == 3 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
    REQUIRE( app.m_ddspcSnapshot[1] == 0 );
    app.postReference( 3, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_ddspcSchedule.m_epochs == 2 );
    REQUIRE( app.m_syncFramesDropped == 2 );
    REQUIRE( app.m_ddspcSchedule.m_next == 0 );
    REQUIRE( app.m_ddspcSchedule.deadline() > app.m_fakeNow_ns );
    app.m_syncTimeout = 0.001;
    app.m_fakeNow_ns += 2000000;
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
    REQUIRE( app.m_cadenceResets == 1 );
}

/** Runtime settings validate atomically, affect future phase only, and preserve disabled-mode behavior.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC runtime controls and disabled compatibility", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_ddspcSettings;
    MagAOX::app::ad7689Ctrl::applyControls;
    MagAOX::app::ad7689Ctrl::updateSynchroDelayController;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    app.m_indiP_ddspcSettings = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_ddspcSettings.setDevice( app.configName() );
    app.m_indiP_ddspcSettings.setName( "ddspcSettings" );
    app.m_indiP_ddspcSettings.add( pcf::IndiElement( "enabled" ) );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    const int64_t frozen = app.m_ddspcSchedule.deadline();
    auto          update = app.m_indiP_ddspcSettings;
    update["enabled"]    = 1;
    update.add( pcf::IndiElement( "wfs_process_ns" ) );
    update["wfs_process_ns"] = 125000;
    REQUIRE( app.newCallBack_m_indiP_ddspcSettings( update ) == 0 );
    REQUIRE_FALSE( app.applyControls() );
    REQUIRE( app.m_ddspcSchedule.deadline() == frozen );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_ddspcTarget_ns == 350000 );
    REQUIRE( app.m_ddspcSchedule.deadline() == frozen );
    update["wfs_process_ns"] = -1;
    REQUIRE( app.newCallBack_m_indiP_ddspcSettings( update ) == -1 );
    REQUIRE( app.m_requestedDdspc.m_process == 125000 );
    update["wfs_process_ns"] = 125000;
    update["enabled"]        = 0;
    REQUIRE( app.newCallBack_m_indiP_ddspcSettings( update ) == 0 );
    REQUIRE_FALSE( app.applyControls() );
    REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.m_phase == Approx( 34400 ) );
    REQUIRE( app.m_ddspcSnapshot[0] == 0 );
    app.m_firstReadLatency   = false;
    update["wfs_process_ns"] = 130000;
    REQUIRE( app.newCallBack_m_indiP_ddspcSettings( update ) == 0 );
    REQUIRE_FALSE( app.applyControls() );
    REQUIRE_FALSE( app.m_firstReadLatency ); // Disabled DDSPC-only parameters cannot disturb legacy feedback.
}

/** The DDSPC formula has distinct configuration keys and rejects noncausal timing before acquisition.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC configuration validation", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
#endif
    // clang-format on
    {
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689-ddspc.conf",
                                  { "synchro", "synchro", "synchro", "synchro", "synchro", "synchro", "accel" },
                                  { "shmimName",
                                    "referenceFps",
                                    "ddspcDelayEnabled",
                                    "wfsProcessTime_ns",
                                    "accLatency_ns",
                                    "ddspcMargin_ns",
                                    "fps" },
                                  { "offline", "2000", "true", "75000", "15000", "10000", "8000" } );
        app.config.readConfig( "/tmp/ad7689-ddspc.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        REQUIRE( app.m_requestedDdspc.m_enabled );
        REQUIRE( app.m_requestedDdspc.delay( 500000 ) == 300000 );
        REQUIRE( app.m_requestedFps == 8000 );
        REQUIRE( app.m_requestedWfsFps == 2000 );
    }

    for( const std::string bad : { "-1", "1000000", "1000000001", "nan" } )
    {
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689-ddspc-invalid.conf",
                                  { "synchro", "synchro", "synchro", "synchro" },
                                  { "shmimName", "referenceFps", "ddspcDelayEnabled", "accLatency_ns" },
                                  { "offline", "2000", "true", bad } );
        app.config.readConfig( "/tmp/ad7689-ddspc-invalid.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == -1 );
    }
}

/** Existing feedback corrects the DDSPC target once, without adding the legacy latency formula or wrapping.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC feedback and signed phase offset", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateSynchroDelayController;
    MagAOX::app::ad7689Ctrl::newCallBack_m_indiP_synchroDelay;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    app.m_delayCapped = 1;
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_delayCapped == 0 );
    REQUIRE( app.m_synchroDelayTarget == 300000 );
    app.m_avgReadLatency_ns = 301000;
    app.m_firstReadLatency  = false;
    app.updateSynchroDelayController( 300000 );
    REQUIRE( app.m_synchroDelay == Approx( 299900 ) );
    const int64_t frozen = app.m_ddspcSchedule.deadline();
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_synchroDelayTarget == 300000 );
    REQUIRE( app.m_synchroDelay == Approx( 299900 ) );
    REQUIRE( app.m_ddspcSchedule.deadline() == frozen );
    REQUIRE( app.m_ddspcSchedule.m_target == Approx( -100 ) );
    app.setupSynchroDelayProperty();
    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( -400 ) ) == -1 );
    REQUIRE( app.m_requestedOffset == 0 );
    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( 50 ) ) == 0 );
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_synchroDelayTarget == 350000 );
    REQUIRE( app.m_ddspcTarget_ns == 300000 );
}

/** Stale source identity and source-counter rollback suspend or explicitly reacquire the DDSPC clock.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC source identity and restart", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::synchroStreamStale;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc( 10 );
    app.postReference( 10, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.m_testMetadata.cnt0 = 0;
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    app.m_nextSyncProbe_ns = std::numeric_limits<int64_t>::max(); // Raw fixture has no filesystem source.
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_ddspcSchedule.m_epochs == 2 );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 4 );
    REQUIRE( app.m_cadenceResets == 1 );
    app.m_nextSyncProbe_ns = 0;
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.m_reconfig );
    REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 8 );
    REQUIRE_FALSE( app.m_frameValid );
}

/** INDI publishes distinct deadline/start/completion/availability and original-reference diagnostics.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC timing INDI serialization", "[ad7689Ctrl][ddspc]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::updateTimingDiagnosticsIndi;
    MagAOX::app::ad7689Ctrl::frameGrabberPostPublish;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns = 0;
    app.enableDdspc();
    app.setupTimingDiagnosticsProperty();
    app.m_indiP_ddspcTiming = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_ddspcTiming.setDevice( app.configName() );
    app.m_indiP_ddspcTiming.setName( "ddspcTiming" );

    for( const char *name : { "enabled",
                              "valid",
                              "target_delay_us",
                              "applied_delay_us",
                              "sync_correction_us",
                              "phase_error_us",
                              "scheduled_mono_us",
                              "start_mono_us",
                              "completion_mono_us",
                              "published_mono_us",
                              "reference_mono_us",
                              "reference_id",
                              "reference_slot",
                              "nominal_slot",
                              "frame_seq",
                              "start_lateness_us",
                              "availability_us",
                              "phase_acquisitions" } )
    {
        app.m_indiP_ddspcTiming.add( pcf::IndiElement( name ) );
    }

    app.m_indiP_ddspcSettings = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_ddspcSettings.setDevice( app.configName() );
    app.m_indiP_ddspcSettings.setName( "ddspcSettings" );

    for( const char *name : { "enabled", "wfs_process_ns", "acc_latency_ns", "margin_ns", "slew_fraction" } )
    {
        app.m_indiP_ddspcSettings.add( pcf::IndiElement( name ) );
    }

    app.m_indiP_wfsClock = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_wfsClock.setDevice( app.configName() );
    app.m_indiP_wfsClock.setName( "wfsClock" );

    for( const char *name : { "valid",
                              "samples",
                              "history_s",
                              "fps",
                              "period_us",
                              "phase_mono_us",
                              "phase_frame_id",
                              "jitter_rms_us",
                              "arrival_error_us",
                              "phase_drift_us",
                              "cadence_events",
                              "rejected_events",
                              "fit_resets",
                              "applied_phase_correction_us" } )
    {
        app.m_indiP_wfsClock.add( pcf::IndiElement( name ) );
    }

    stubState().m_duringRead = [&]() { app.m_fakeNow_ns += 15000; };
    app.postReference( 1, 500000 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );
    std::array<uint16_t, 2> frame{};
    REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
    REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    app.updateTimingDiagnosticsIndi();
    REQUIRE( app.m_indiP_ddspcTiming["target_delay_us"].get<double>() == 300 );
    REQUIRE( app.m_indiP_ddspcTiming["applied_delay_us"].get<double>() == 300 );
    REQUIRE( app.m_indiP_ddspcTiming["availability_us"].get<double>() == 15 );
    REQUIRE( app.m_indiP_ddspcTiming["reference_id"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcTiming["reference_slot"].get<double>() == 0 );
    REQUIRE( app.m_indiP_ddspcTiming["frame_seq"].get<double>() == 1 );
    REQUIRE( app.m_indiP_ddspcSettings["enabled"].get<double>() == 1 );
    REQUIRE( app.m_indiP_wfsClock["valid"].get<double>() == 1 );
    REQUIRE( app.m_indiP_wfsClock["period_us"].get<double>() == 500 );
    REQUIRE( app.m_indiP_wfsClock["fps"].get<double>() == 2000 );
    REQUIRE( app.m_indiP_wfsClock["jitter_rms_us"].get<double>() == 0 );
    REQUIRE( app.m_indiP_wfsClock["applied_phase_correction_us"].get<double>() == 0 );
    const auto &messages = static_cast<captureDriver *>( app.m_indiDriver )->m_messages;
    REQUIRE( std::any_of( messages.begin(),
                          messages.end(),
                          []( const std::string &xml ) { return xml.find( "ddspcTiming" ) != std::string::npos; } ) );
    resetStubState();
}

/** The actual app clock rejects raw semaphore jitter while the unchanged controller and ADC path remain active.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl averaged WFS phase isolates ADC cadence from semaphore jitter", "[ad7689Ctrl][wfs-clock]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::acquireDdspcAndCheckValid;
    MagAOX::app::ad7689Ctrl::frameGrabberPostPublish;
#endif
    // clang-format on
    resetStubState();
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns             = 0;
    app.m_fakeNow_ns               = 1000000000000LL;
    app.m_requestedDdspc.m_enabled = true;
    app.applyControls();
    uint32_t   random = 42;
    const auto jitter = [&]() -> int64_t
    {
        random = random * 1664525U + 1013904223U;
        return static_cast<int64_t>( random % 20001 ) - 10000;
    };
    const int64_t origin = app.m_fakeNow_ns - 10000000000LL;

    for( uint64_t id = 1; id <= 20001; ++id )
    {
        app.m_wfsClock.observe( id, origin + ( id - 1 ) * 500000 + jitter() );
    }

    REQUIRE( app.m_wfsClock.valid() );
    uint64_t nextId      = 20002;
    int64_t  nextArrival = origin + ( nextId - 1 ) * 500000 + jitter();
    app.m_beforeWait     = [&]( int64_t until )
    {
        if( nextArrival <= until )
        {
            app.m_fakeNow_ns        = nextArrival;
            app.m_testMetadata.cnt0 = nextId;
            // Deliberately incorrect producer period: only local monotonic arrivals may steer DDSPC.
            app.m_testMetadata.atime = timespec{ 1700000000 + static_cast<time_t>( nextId / 1000 ),
                                                 static_cast<long>( ( nextId % 1000 ) * 1000000 ) };
            REQUIRE( sem_post( &app.m_testSemaphore ) == 0 );
            ++nextId;
            nextArrival = origin + ( nextId - 1 ) * 500000 + jitter();
        }
    };
    stubState().m_timestamp     = [&]( timespec &stamp ) { app.getRealtime( stamp ); };
    stubState().m_duringRead    = [&]() { app.m_fakeNow_ns += 15000; };
    int64_t previous            = 0;
    double  maximumSpacingError = 0;
    bool    valid               = true;

    for( uint64_t i = 0; i < 20000; ++i )
    {
        int result;

        do
        {
            result = app.ad7689Ctrl::acquireAndCheckValid();
        } while( result == 1 && !app.m_reconfig && !app.m_shutdown );

        REQUIRE( result == 0 );
        const int64_t due = app.m_ddspcSlot.m_deadline;

        if( previous )
        {
            maximumSpacingError =
                std::max( maximumSpacingError, std::abs( static_cast<double>( due - previous ) - 125000 ) );
        }

        previous = due;
        valid    = valid && app.m_ddspcSlot.m_reference.m_id == 20002 + i / 4 && app.m_ddspcSlot.m_slot == i % 4;
        valid    = valid && app.m_ddspcStart_ns == due && app.m_ddspcCompletion_ns == due + 15000;
        std::array<uint16_t, 2> frame{};
        REQUIRE( app.loadImageIntoStream( frame.data() ) == 0 );
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
    }

    REQUIRE( valid );
    REQUIRE( maximumSpacingError <= 50 );
    REQUIRE( app.m_wfsClock.m_period == Approx( 500000 ).margin( 0.1 ) );
    REQUIRE( app.m_wfsClock.m_jitter > 5000 );
    REQUIRE( app.m_wfsClock.m_jitter < 6500 );
    REQUIRE( app.m_wfsClock.span() > 9.999 );
    REQUIRE( app.m_ddspcSchedule.m_epochs == 1 );
    REQUIRE( app.m_ddspcSchedule.m_skipped == 0 );
    REQUIRE( app.m_localFrameSeq == 20000 );
    REQUIRE( app.m_lateScans == 0 );
    REQUIRE( app.m_cadenceResets == 0 );
    resetStubState();
}

/** Startup and persistent frequency changes wait for the configured number of consistent monotonic observations.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl qualifies rolling clocks before starting and after cadence changes", "[ad7689Ctrl][wfs-clock]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::processSyncReference;
    MagAOX::app::ad7689Ctrl::resetSyncSchedule;
#endif
    // clang-format on
    ad7689Ctrl_test app;
    app.setupSync( 8000, 2000 );
    app.m_clockStep_ns             = 0;
    app.m_requestedDdspc.m_enabled = true;
    app.m_wfsClock.m_minEvents     = 8;

    for( unsigned i = 0; i < 8; ++i )
    {
        app.postReference( 1, 500000 );
        REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );
        REQUIRE( app.m_ddspcSchedule.m_active == ( i == 7 ) );
    }

    REQUIRE( app.m_ddspcTarget_ns == 300000 );
    REQUIRE( app.m_ddspcSchedule.m_count == 4 );
    REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 0 );

    for( unsigned i = 0; i < 8; ++i )
    {
        app.postReference( 1, 1000000 );
        REQUIRE( app.ad7689Ctrl::acquireAndCheckValid() == 1 );

        if( i >= 2 && i < 7 )
        {
            REQUIRE_FALSE( app.m_ddspcSchedule.m_active );
        }
    }

    REQUIRE( app.m_wfsClock.valid() );
    REQUIRE( app.m_wfsClock.m_period == 1000000 );
    REQUIRE( app.m_ddspcTarget_ns == 550000 );
    REQUIRE( app.m_ddspcSchedule.m_count == 8 );
    REQUIRE( app.m_ddspcSchedule.m_interval == 125000 );
    REQUIRE( app.m_wfsClock.m_history.size() == 8 );
    REQUIRE( app.m_cadenceResets == 1 );
    REQUIRE( app.m_requestedFps == 8000 );
}

/** Fit qualification and change gates reject impossible/nonfinite startup settings without touching hardware.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl rolling WFS configuration validation", "[ad7689Ctrl][wfs-clock]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::loadConfigImpl;
#endif
    // clang-format on
    for( const std::string bad : { "1", "1.5", "4097", "nan" } )
    {
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/ad7689-rolling-invalid.conf", { "synchro" }, { "wfsFitMinEvents" }, { bad } );
        app.config.readConfig( "/tmp/ad7689-rolling-invalid.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == -1 );
    }

    for( const std::string bad : { "0", "0.6", "nan" } )
    {
        ad7689Ctrl_test app;
        app.setupConfig();
        mx::app::writeConfigFile(
            "/tmp/ad7689-rolling-invalid.conf", { "synchro" }, { "wfsCadenceTolerance" }, { bad } );
        app.config.readConfig( "/tmp/ad7689-rolling-invalid.conf" );
        REQUIRE( app.loadConfigImpl( app.config ) == -1 );
    }
}

/** Actual frameGrabber publication follows the deterministic 300-us/8-kHz lattice through 2-kHz WFS wakes.
 * Run alone in a fresh process with a private MILK_SHM_DIR; no ADC or INDI server is used.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl DDSPC native publication across WFS boundaries", "[.][ad7689Ctrl-ddspc-native]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::frameGrabberPostPublish;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
#endif
    // clang-format on
    REQUIRE( std::getenv( "MILK_SHM_DIR" ) != nullptr );
    resetStubState();
    ad7689Ctrl_test app;
    app.m_clockStep_ns             = 0;
    app.m_requestedDdspc.m_enabled = true;
    app.m_requestedFps             = 8000;
    app.m_requestedWfsFps          = 2000;
    app.m_shmimName                = "ad7689-offline-ddspc";
    app.m_synchroShmimName         = "ad7689-offline-ddspc-source";
    app.m_fgThreadInit             = false;
    app.m_circBuffLength           = 4;
    app.m_latencyCircBuffMaxLength = 0;
    app.state( MagAOX::app::stateCodes::OPERATING );
    IMAGE    trigger{}, reader{};
    uint32_t size[2]{ 1, 1 };
    REQUIRE(
        ImageStreamIO_createIm_gpu(
            &trigger, app.m_synchroShmimName.c_str(), 2, size, _DATATYPE_UINT16, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) ==
        0 );
    const int64_t epoch         = app.m_fakeNow_ns;
    int64_t       nextReference = epoch;
    app.m_beforeWait            = [&]( int64_t until )
    {
        if( nextReference <= until )
        {
            app.m_fakeNow_ns  = nextReference;
            trigger.md->write = 1;
            app.getRealtime( trigger.md->atime );
            ++trigger.md->cnt0;
            trigger.md->write = 0;
            ImageStreamIO_sempost( &trigger, -1 );
            nextReference += 500000;
        }
    };
    stubState().m_channelValues = { 65535, 4096 };
    stubState().m_timestamp     = [&]( timespec &stamp ) { app.getRealtime( stamp ); };
    unsigned reads              = 0;
    bool     opened             = false;
    stubState().m_duringRead    = [&]()
    {
        ++reads;
        const int64_t expected =
            epoch + ( app.m_wfsClock.m_minEvents - 1 ) * 500000LL + 300000 + ( reads - 1 ) * 125000;
        CHECK( app.m_ddspcSlot.m_deadline == expected );
        CHECK( app.m_ddspcStart_ns == expected );
        CHECK( app.m_ddspcSlot.m_reference.m_id == app.m_wfsClock.m_minEvents + ( reads - 1 ) / 4 );
        CHECK( app.m_ddspcSlot.m_slot == ( reads - 1 ) % 4 );
        CHECK( app.m_ddspcSchedule.m_skipped == 0 );

        if( !opened )
        {
            opened = ImageStreamIO_openIm( &reader, app.m_shmimName.c_str() ) == 0;
            CHECK( opened );
        }

        if( opened && reads > 1 )
        {
            CHECK( reader.md->write == 0 );
            CHECK( reader.md->cnt0 == reads - 1 );
            CHECK( reader.md->datatype == _DATATYPE_UINT16 );
            const size_t slot = reader.md->cnt1;
            CHECK( slot == ( reads - 2 ) % 4 );
            CHECK( reader.array.UI16[slot * 2] == 65535 );
            CHECK( reader.array.UI16[slot * 2 + 1] == 4096 );
            CHECK( reader.md->atime.tv_nsec == ( expected - 125000 ) % 1000000000 );
            CHECK( reader.atimearray[slot].tv_nsec == reader.md->atime.tv_nsec );
            CHECK( reader.cntarray[slot] == reads - 1 );
        }

        app.m_fakeNow_ns += 15000;

        if( reads == 33 )
        {
            app.m_shutdown = true;
        }
    };
    app.fgThreadExec();
    CHECK( reads == 33 );
    CHECK( app.m_localFrameSeq == 32 );
    CHECK( app.m_syncFramesWritten == 32 );
    CHECK( app.m_ddspcSchedule.m_skipped == 3 ); // Authorized tail explicitly abandoned on shutdown.
    CHECK( app.m_ddspcSchedule.m_epochs == 1 );
    CHECK( app.m_invalidFrames == 1 );

    if( opened )
    {
        ImageStreamIO_closeIm( &reader );
    }

    app.closeSynchroStream();
    ImageStreamIO_destroyIm( &trigger );
    resetStubState();
}

/** Exercise the actual frameGrabber publisher and an independent local shared-memory reader.
 * Run this hidden test alone in a fresh process with a private MILK_SHM_DIR.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "ad7689Ctrl native frameGrabber publication", "[.][ad7689Ctrl-native]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::acquireAndCheckValid;
    MagAOX::app::ad7689Ctrl::closeSynchroStream;
    MagAOX::app::ad7689Ctrl::configureAcquisition;
    MagAOX::app::ad7689Ctrl::loadImageIntoStream;
    MagAOX::app::ad7689Ctrl::reconfig;
    MagAOX::app::ad7689Ctrl::recordTelem;
    MagAOX::app::ad7689Ctrl::startAcquisition;
#endif
    // clang-format on
    REQUIRE( std::getenv( "MILK_SHM_DIR" ) != nullptr );

    for( unsigned samplesPerWfs : { 0U, 4U, 3U } )
    {
        const bool synchronized = samplesPerWfs != 0;
        resetStubState();
        ad7689Ctrl_test app;
        app.m_fakeSync                 = false;
        app.m_shmimName                = synchronized ? "ad7689-offline-sync" : "ad7689-offline-timer";
        app.m_fgThreadInit             = false;
        app.m_circBuffLength           = 4;
        app.m_latencyCircBuffMaxLength = 0;
        app.m_requestedFps             = synchronized ? 500 * samplesPerWfs : 1000;
        app.m_requestedWfsFps = app.m_wfs_fps = 500;
        app.state( MagAOX::app::stateCodes::OPERATING );
        stubState().m_channelValues = { 65535, 4096 };
        IMAGE             reader{}, trigger{};
        unsigned          reads  = 0;
        bool              opened = false;
        std::atomic<bool> running{ true };
        std::thread       producer;

        if( synchronized )
        {
            app.m_synchroShmimName = "ad7689-offline-source";
            uint32_t size[2]{ 1, 1 };
            REQUIRE( ImageStreamIO_createIm_gpu( &trigger,
                                                 app.m_synchroShmimName.c_str(),
                                                 2,
                                                 size,
                                                 _DATATYPE_UINT16,
                                                 -1,
                                                 1,
                                                 IMAGE_NB_SEMAPHORE,
                                                 0,
                                                 0,
                                                 0 ) == 0 );
            producer = std::thread(
                [&]()
                {
                    while( running )
                    {
                        trigger.md->write = 1;
                        clock_gettime( CLOCK_REALTIME, &trigger.md->atime );
                        ++trigger.md->cnt0;
                        trigger.md->write = 0;
                        ImageStreamIO_sempost( &trigger, -1 );
                        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
                    }
                } );
        }

        stubState().m_duringRead = [&]()
        {
            ++reads;

            if( !opened )
            {
                opened = ImageStreamIO_openIm( &reader, app.m_shmimName.c_str() ) == 0;
                CHECK( opened );
            }

            if( opened )
            {
                CHECK( reader.md->datatype == _DATATYPE_UINT16 );
                CHECK( reader.md->size[0] == 2 );
                CHECK( reader.md->size[1] == 1 );
                CHECK( reader.md->size[2] == 4 );
                CHECK( reader.md->write == 0 );
                CHECK( reader.md->cnt0 == reads - 1 );

                if( reads > 1 )
                {
                    const size_t slot = reader.md->cnt1;
                    CHECK( slot == ( reads - 2 ) % 4 );
                    CHECK( reader.array.UI16[slot * 2] == 65535 );
                    CHECK( reader.array.UI16[slot * 2 + 1] == 4096 );
                    CHECK( reader.md->atime.tv_sec > 0 );
                    CHECK( reader.md->writetime.tv_sec >= reader.md->atime.tv_sec );
                    CHECK( reader.atimearray[slot].tv_nsec == reader.md->atime.tv_nsec );
                    CHECK( reader.cntarray[slot] == reader.md->cnt0 );
                }
            }

            if( reads == 9 )
            {
                app.m_shutdown = true;
            }
        };
        app.fgThreadExec();
        running = false;

        if( producer.joinable() )
        {
            producer.join();
        }

        CHECK( reads == 9 );
        CHECK( app.m_localFrameSeq == 8 );
        CHECK( app.m_invalidFrames == 1 );

        if( synchronized )
        {
            CHECK( app.m_syncFramesWritten == 8 );
            CHECK( app.m_syncFramesReceived < app.m_syncFramesWritten );
        }

        if( opened )
        {
            ImageStreamIO_closeIm( &reader );
        }

        app.closeSynchroStream();

        if( synchronized )
        {
            ImageStreamIO_destroyIm( &trigger );
        }

        resetStubState();
    }
}

} // namespace ad7689CtrlTest
} // namespace libXWCTest
