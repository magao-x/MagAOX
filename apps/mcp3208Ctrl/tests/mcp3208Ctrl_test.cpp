/** \file mcp3208Ctrl_test.cpp
 * \brief Catch2 tests for the mcp3208Ctrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mcp3208Ctrl_files
 */

#include "../../../tests/testXWC.hpp"

#include <vector>

#define protected public
#include "../mcp3208Ctrl.hpp"
#undef protected

namespace
{

/// Stub state for MCP3208 hardware access during unit tests.
struct mcp3208StubState
{
    std::vector<unsigned short> m_channelValues;     ///< Values returned for each channel read.
    std::vector<int>            m_readOrder;         ///< Order in which channels were sampled.
    int                         m_connectCalls{ 0 }; ///< Number of stubbed connect calls.
};

/// Access the shared MCP3208 stub state.
mcp3208StubState &stubState()
{
    static mcp3208StubState state;
    return state;
}

/// Reset the shared MCP3208 stub state.
void resetStubState()
{
    stubState().m_channelValues.clear();
    stubState().m_readOrder.clear();
    stubState().m_connectCalls = 0;
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

namespace MCP3208Lib
{

MCP3208::MCP3208( const int dev, const int channel, const int baud, const int flags ) noexcept
    : _handle( -1 ), _dev( dev ), _channel( channel ), _baud( baud ), _flags( flags )
{
}

MCP3208::~MCP3208()
{
}

void MCP3208::connect()
{
    ++stubState().m_connectCalls;
}

void MCP3208::disconnect()
{
}

unsigned short MCP3208::read( const std::uint8_t channel, const Mode ) const
{
    stubState().m_readOrder.push_back( static_cast<int>( channel ) );

    if( static_cast<size_t>( channel ) < stubState().m_channelValues.size() )
    {
        return stubState().m_channelValues[static_cast<size_t>( channel )];
    }

    return 0;
}

} // namespace MCP3208Lib

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup mcp3208Ctrl_unit_test mcp3208Ctrl Unit Tests
 * \brief Unit tests for the mcp3208Ctrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `mcp3208Ctrl` unit tests.
/** \ingroup mcp3208Ctrl_unit_test
 */
namespace mcp3208CtrlTest
{

namespace
{

/// Test harness exposing small setup helpers for `mcp3208Ctrl`.
class mcp3208Ctrl_test : public mcp3208Ctrl
{
  public:
    /// Construct the test harness with a stable config name.
    mcp3208Ctrl_test()
    {
        m_configName = "mcp3208Ctrl_test";
    }

    /// Initialize the local fps INDI property used by the callback tests.
    void setupFpsProperty()
    {
        m_indiP_fps = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_fps.setName( "fps" );
        m_indiP_fps.add( pcf::IndiElement( "current" ) );
        m_indiP_fps["current"].setValue( m_fps );
        m_indiP_fps.add( pcf::IndiElement( "target" ) );
        m_indiP_fps["target"].setValue( m_fps );
    }

    /// Initialize the external fps INDI property used by the callback tests.
    void setupFpsSourceProperty()
    {
        m_fpsDevice   = "fpsdev";
        m_fpsProperty = "fps";
        m_fpsElement  = "current";

        m_indiP_fpsSource = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_fpsSource.setDevice( m_fpsDevice );
        m_indiP_fpsSource.setName( m_fpsProperty );
    }

    /// Build an INDI property update for the local fps callback.
    pcf::IndiProperty makeFpsUpdate( const double target /**< [in] the requested fps */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setName( "fps" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].setValue( target );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target );
        return ip;
    }

    /// Build an INDI property update for the external fps-source callback.
    pcf::IndiProperty makeFpsSourceUpdate( const double current /**< [in] the reported fps */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_fpsDevice );
        ip.setName( m_fpsProperty );
        ip.add( pcf::IndiElement( m_fpsElement ) );
        ip[m_fpsElement].setValue( current );
        return ip;
    }

    /// Initialize the alpha INDI property used by callback tests.
    void setupAlphaProperty()
    {
        m_indiP_alpha = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_alpha.setName( "alpha" );
        m_indiP_alpha.add( pcf::IndiElement( "current" ) );
        m_indiP_alpha["current"].setValue( m_alpha );
        m_indiP_alpha.add( pcf::IndiElement( "target" ) );
        m_indiP_alpha["target"].setValue( m_alpha );
    }

    /// Build an INDI property update for the alpha callback.
    pcf::IndiProperty makeAlphaUpdate( const double target /**< [in] requested global EMA alpha */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setName( "alpha" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].setValue( target );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target );
        return ip;
    }

    /// Initialize the synchronized-delay INDI property used by callback tests.
    void setupSynchroDelayProperty()
    {
        m_indiP_synchroDelay = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_synchroDelay.setName( "synchroDelay" );
        m_indiP_synchroDelay.add( pcf::IndiElement( "current" ) );
        m_indiP_synchroDelay["current"].setValue( m_synchroPostDelay );
        m_indiP_synchroDelay.add( pcf::IndiElement( "target" ) );
        m_indiP_synchroDelay["target"].setValue( m_synchroPostDelay );
    }

    /// Build an INDI property update for the synchronized-delay callback.
    pcf::IndiProperty makeSynchroDelayUpdate( const double target_us /**< [in] requested synchronized signed delay offset in microseconds */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setName( "synchroDelay" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].setValue( target_us );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target_us );
        return ip;
    }

    /// Initialize the timing-source INDI switch property used by callback tests.
    void setupTimingSourceProperty()
    {
        m_indiP_timingSource = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_timingSource.setName( "timingSource" );
        m_indiP_timingSource.setRule( pcf::IndiProperty::OneOfMany );
        m_indiP_timingSource.add( pcf::IndiElement( "semaphore", pcf::IndiElement::Off ) );
        m_indiP_timingSource.add( pcf::IndiElement( "integrator", pcf::IndiElement::On ) );
    }

    /// Build an INDI property update for the timing-source callback.
    pcf::IndiProperty makeTimingSourceUpdate( const bool useSemaphore /**< [in] true selects semaphore timing, false selects integrator timing */ )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setName( "timingSource" );
        ip.add( pcf::IndiElement( "semaphore" ) );
        ip["semaphore"].setSwitchState( useSemaphore ? pcf::IndiElement::On : pcf::IndiElement::Off );
        ip.add( pcf::IndiElement( "integrator" ) );
        ip["integrator"].setSwitchState( useSemaphore ? pcf::IndiElement::Off : pcf::IndiElement::On );
        return ip;
    }

    /// Qualify the rolling WFS clock on 16 references 500 us apart, predicting `nextId` to arrive at `next`.
    void qualifyClock( const int64_t  next /**< [in] predicted arrival of `nextId`, monotonic nanoseconds */,
                       const uint64_t nextId /**< [in] frame ID of the next reference, at least 16 */ )
    {
        for( uint64_t id = nextId - 16; id < nextId; ++id )
        {
            m_wfsClock.observe( id, next - static_cast<int64_t>( nextId - id ) * 500000 );
        }
    }

    /// Initialize the timing-diagnostics INDI property used by diagnostics tests.
    void setupTimingDiagnosticsProperty()
    {
        m_indiP_timingDiag = pcf::IndiProperty( pcf::IndiProperty::Number );
        m_indiP_timingDiag.setName( "timingDiag" );
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
};

} // namespace

/// Preserve Doxygen links for the real `mcp3208Ctrl` APIs exercised by the tests.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl Doxygen references are preserved", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsProperty();
    app.setupFpsSourceProperty();
    app.setupAlphaProperty();
    app.setupSynchroDelayProperty();
    app.setupTimingSourceProperty();

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
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_timingSource( app.makeTimingSourceUpdate( true ) ) );
    XWCTEST_DOXYGEN_REF( app.newCallBack_m_indiP_lateTolerance( pcf::IndiProperty() ) );
    XWCTEST_DOXYGEN_REF( app.updateTriggerTiming( timespec{}, 0.0 ) );
    XWCTEST_DOXYGEN_REF( app.updateTimingDiagnosticsIndi() );
    XWCTEST_DOXYGEN_REF( app.updateTimingSourceIndi() );
    XWCTEST_DOXYGEN_REF( app.updateSyncTimingIndi() );
    XWCTEST_DOXYGEN_REF( app.processSyncReference( 0 ) );
    XWCTEST_DOXYGEN_REF( app.acquireScan( 0 ) );
    XWCTEST_DOXYGEN_REF( app.waitForSync( 0 ) );
    XWCTEST_DOXYGEN_REF( app.monotonicNow() );
    XWCTEST_DOXYGEN_REF( app.timespecToNs( timespec{} ) );
    XWCTEST_DOXYGEN_REF( app.nsToTimespec( 0.0 ) );

    SUCCEED();
}

/// Verify synchronized-acquisition defaults load from configuration.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration defaults load synchronized settings", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test.conf", { "none" }, { "nada" }, { "0" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test.conf" );

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
    REQUIRE( app.m_spinLead_ns == Approx( 50000.0 ) );
    REQUIRE( app.m_lateTolerance_ns == Approx( 50000.0 ) );
    REQUIRE( app.m_completeScans == true );
    REQUIRE( app.m_wfsClock.m_minEvents == 16 );
    REQUIRE( app.m_wfsClock.m_tolerance == Approx( 0.1 ) );
    REQUIRE( app.m_alpha == Approx( 0.01f ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_fps == Approx( 8000.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( static_cast<double>( app.m_fps ) ) );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_semaphoreFallbackActive == false );
}

/// Verify synchronized-acquisition overrides load from configuration.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration overrides load synchronized settings", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test_override.conf",
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
                                "synchro",
                                "synchro",
                                "synchro",
                                "synchro",
                                "synchro",
                                "accel",
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
                                "spinLead_ns",
                                "lateTolerance_ns",
                                "completeScans",
                                "wfsFitMinEvents",
                                "wfsCadenceTolerance",
                                "numChannels",
                                "fps" },
                              { "camwfs_sync",
                                "17",
                                "4000",
                                "62000",
                                "11000",
                                "290000",
                                "75000",
                                "0.2",
                                "15000",
                                "0.25",
                                "30000",
                                "20000",
                                "false",
                                "8",
                                "0.2",
                                "3",
                                "6000" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test_override.conf" );

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
    REQUIRE( app.m_spinLead_ns == Approx( 30000.0 ) );
    REQUIRE( app.m_lateTolerance_ns == Approx( 20000.0 ) );
    REQUIRE( app.m_completeScans == false );
    REQUIRE( app.m_wfsClock.m_minEvents == 8 );
    REQUIRE( app.m_wfsClock.m_tolerance == Approx( 0.2 ) );
    REQUIRE( app.m_alpha == Approx( 0.25f ) );
    REQUIRE( app.m_numChannels == 3 );
    REQUIRE( app.m_fps == Approx( 6000.0f ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 6000.0f ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 17000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 17000.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( static_cast<double>( app.m_fps ) ) );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Semaphore );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Semaphore );
    REQUIRE( app.m_semaphoreFallbackActive == false );
}

/// Verify an out-of-range `accel.fps` keeps the default rate limit.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration rejects an out-of-range fps limit", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test_fps.conf", { "accel" }, { "fps" }, { "20000" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test_fps.conf" );

    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_fps == Approx( 8000.0f ) );
}

/// Verify a synchronized fps source reports the WFS rate without lowering the scan rate limit.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl fps source keeps the synchronized rate limit", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsSourceProperty();
    app.m_timingSource = mcp3208Ctrl::TimingSource::Semaphore;

    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 2000.0 ) ) == 0 );
    REQUIRE( app.m_wfs_fps == Approx( 2000.0 ) );
    REQUIRE( app.m_fps == Approx( 8000.0f ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 8000.0f ) );
}

/// Verify the user fps callback still updates cadence metadata.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl fps callback updates trigger metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsProperty();

    REQUIRE( app.newCallBack_m_indiP_fps( app.makeFpsUpdate( 500.0 ) ) == 0 );
    REQUIRE( app.m_fps == Approx( 500.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( 500.0 ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 500.0f ) );
    REQUIRE( app.nano_sec_target == Approx( 1e9f / 500.0f ) );
}

/// Verify the external fps source callback still updates cadence metadata.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl fps source callback updates trigger metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupFpsSourceProperty();

    REQUIRE( app.setCallBack_m_indiP_fpsSource( app.makeFpsSourceUpdate( 250.0 ) ) == 0 );
    REQUIRE( app.m_fps == Approx( 250.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( 250.0 ) );
    REQUIRE( app.m_trigger == Approx( 1e9f / 250.0f ) );
    REQUIRE( app.nano_sec_target == Approx( 1e9f / 250.0f ) );
}

/// Verify synchronized alpha configuration is clamped to the valid range.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl configuration clamps synchronized alpha", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test_alpha_low.conf", { "synchro" }, { "alpha" }, { "-0.5" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test_alpha_low.conf" );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_alpha == Approx( 0.0f ) );

    mx::app::writeConfigFile( "/tmp/mcp3208Ctrl_test_alpha_high.conf", { "synchro" }, { "alpha" }, { "1.5" } );
    app.config.readConfig( "/tmp/mcp3208Ctrl_test_alpha_high.conf" );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_alpha == Approx( 1.0f ) );
}

/// Verify alpha callback updates and clamps the global EMA coefficient.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl alpha callback updates and clamps", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupAlphaProperty();

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( 0.4 ) ) == 0 );
    REQUIRE( app.m_alpha == Approx( 0.4f ) );

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( -1.0 ) ) == 0 );
    REQUIRE( app.m_alpha == Approx( 0.0f ) );

    REQUIRE( app.newCallBack_m_indiP_alpha( app.makeAlphaUpdate( 2.0 ) ) == 0 );
    REQUIRE( app.m_alpha == Approx( 1.0f ) );
}

/// Verify synchronized-delay callback updates delay state with signed offsets.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchroDelay callback updates signed offsets", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupSynchroDelayProperty();
    app.m_wfsPeriodMeasured_ns = 1000000.0;
    app.m_delayModel_ns        = 120000.0;

    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( 25.0 ) ) == 0 );
    REQUIRE( app.m_synchroPostDelay == 25 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 25000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 25000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 25000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );

    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( -5.0 ) ) == 0 );
    REQUIRE( app.m_synchroPostDelay == -5 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 995000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 995000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 995000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );
}

/// Verify timing-source callback selects runtime mode and requests immediate reconfiguration.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timingSource callback updates mode and reconfig state", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupTimingSourceProperty();
    app.m_synchroShmimName = "";
    app.m_reconfig         = false;

    REQUIRE( app.newCallBack_m_indiP_timingSource( app.makeTimingSourceUpdate( true ) ) == 0 );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Semaphore );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_semaphoreFallbackActive == true );
    REQUIRE( app.m_reconfig == true );

    app.m_reconfig = false;
    REQUIRE( app.newCallBack_m_indiP_timingSource( app.makeTimingSourceUpdate( false ) ) == 0 );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_semaphoreFallbackActive == false );
    REQUIRE( app.m_reconfig == true );
}

/// Verify synchronized-mode timing diagnostics publish loop state and derived error.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timing diagnostics publish synchronized loop metrics", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_timingSource          = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_avgReadLatency_ns     = 125000.0;
    app.m_synchroDelay          = 31000.0f;
    app.m_synchroDelayTarget    = 17000.0f;
    app.m_delayApplied_ns       = 24000.0;
    app.m_delayModel_ns         = 17000.0;
    app.m_avgSemaphorePeriod_ns = 500000.0;
    app.m_wfsPeriodMeasured_ns  = 500000.0;
    app.m_producerPeriodInst_ns = 510000.0;
    app.m_avgProducerPeriod_ns  = 500000.0;
    app.m_channelReadoutTime_ns = 34000.0;
    app.m_triggerInterval_ns    = 600000.0;
    app.m_atime                 = timespec{ 12, 3000000L };
    app.m_triggerTime           = timespec{ 12, 3456789L };
    app.m_localFrameSeq         = 44;
    app.m_syncFramesReceived    = 41;
    app.m_syncFramesWritten     = 40;
    app.m_syncFramesDropped     = 2;
    app.m_syncFrameIdGapCount   = 2;
    app.m_syncProducerFrameId   = 123456;
    app.m_syncProducerFrameDelta = 3;
    app.m_syncProducerFrameValid = true;
    app.qualifyClock( 1000000000, 100 );

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
             Approx( 1e-3 * ( mcp3208Ctrl::timespecToNs( timespec{ 12, 3456789L } ) -
                              mcp3208Ctrl::timespecToNs( timespec{ 12, 3000000L } ) ) ) );
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
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timing diagnostics track mode transitions", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_timingSource        = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_atime              = timespec{ 1, 2 };
    app.m_triggerInterval_ns = 123456.0;
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

    app.m_timingSource          = mcp3208Ctrl::TimingSource::Integrator;
    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Integrator;
    app.m_triggerInterval_ns = 456789.0;
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
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timing diagnostics compute wrapped phase error and lock thresholds", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_timingSource            = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_effectiveTimingSource   = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_delayApplied_ns         = 50.0;
    app.m_synchroDelayTarget       = 900.0f;
    app.m_wfsPeriodMeasured_ns    = 1000.0;
    app.m_delayLockAbsThreshold_ns = 200.0;
    app.m_delayLockFracThreshold   = 0.1;
    app.qualifyClock( 1000000000, 100 );

    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.15 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 0.0 ) );

    app.m_delayLockFracThreshold = 0.2;
    app.updateTimingDiagnosticsIndi();

    REQUIRE( app.m_indiP_timingDiag["delay_phase_error_us"].get<double>() == Approx( 0.15 ) );
    REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 1.0 ) );
}

/// Verify nanosecond and timespec conversions preserve normalized values.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timing helpers convert between nanoseconds and timespec", "[mcp3208Ctrl]" )
{
    const double  ns = -250000000.0;
    const timespec ts = mcp3208Ctrl::nsToTimespec( ns );

    REQUIRE( ts.tv_sec == -1 );
    REQUIRE( ts.tv_nsec == 750000000L );
    REQUIRE( mcp3208Ctrl::timespecToNs( ts ) == Approx( ns ) );
}

/// Verify synchronized timing uses only the measured semaphore period for delay modeling.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming uses measured semaphore period for delay model", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_alpha                = 0.2f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 10, 100000000L };
    app.m_avgSemaphorePeriod_ns = 500000.0;
    app.m_wfs_fps             = 1000.0;
    app.m_synchroPostDelay    = 0;

    const timespec secondArrival{ 10, 101000000L };

    app.updateTriggerTiming( secondArrival );

    const double alpha                     = static_cast<double>( app.m_alpha );
    const double expectedAvg_ns            = alpha * 1000000.0 + ( 1.0 - alpha ) * 500000.0;
    const double expectedMeasuredDeltaT_ns = expectedAvg_ns;
    const double expectedBlendedDeltaT_ns  = ( 1.0 - alpha ) * ( 1e9 / 1000.0 ) + alpha * expectedAvg_ns;
    const double rawDelayMeasured_ns =
        0.5 * expectedMeasuredDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double rawDelayBlended_ns = 0.5 * expectedBlendedDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelayMeasured_ns = wrapDelay( rawDelayMeasured_ns, expectedMeasuredDeltaT_ns );
    const double expectedDelayBlended_ns  = wrapDelay( rawDelayBlended_ns, expectedBlendedDeltaT_ns );
    const double measuredTrigger_ns       = mcp3208Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns         = measuredTrigger_ns - mcp3208Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedMeasuredDeltaT_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelayMeasured_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) ); // No built-in delay: the model is reported only.
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( measuredDelay_ns == Approx( 0.0 ).margin( 1.0 ) );
    REQUIRE( expectedDelayMeasured_ns != Approx( expectedDelayBlended_ns ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedMeasuredDeltaT_ns );
}

/// Verify synchronized timing applies a positive user offset to the modeled delay target.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming applies positive synchroDelay offset", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

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
    const double expectedTarget_ns  = 100000.0;
    const double measuredTrigger_ns = mcp3208Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns   = measuredTrigger_ns - mcp3208Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedPeriod_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedModel_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedTarget_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedTarget_ns ) );
}

/// Verify synchronized timing applies a negative user offset with modulo wrap into the current period.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming applies negative synchroDelay offset with wrap", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

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
    const double expectedTarget_ns  = wrapDelay( -300000.0, expectedPeriod_ns );
    const double measuredTrigger_ns = mcp3208Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns   = measuredTrigger_ns - mcp3208Ctrl::timespecToNs( secondArrival );

    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( expectedPeriod_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedModel_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedTarget_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedTarget_ns ) );
}

/// Verify synchronized timing falls back to EMA period when WFS fps is unavailable.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming falls back to EMA period when fps is invalid", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_alpha                = 0.2f;
    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 0, 0 };
    app.m_avgSemaphorePeriod_ns = 1000000.0;
    app.m_wfs_fps               = 0.0;

    const timespec nextArrival{ 0, 2000000L };
    app.updateTriggerTiming( nextArrival );

    const double alpha                = static_cast<double>( app.m_alpha );
    const double expectedAvg_ns       = alpha * 2000000.0 + ( 1.0 - alpha ) * 1000000.0;
    const double rawDelay_ns          = 0.5 * expectedAvg_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelay_ns     = wrapDelay( rawDelay_ns, expectedAvg_ns );
    const double expectedTrigger_ns   = mcp3208Ctrl::timespecToNs( nextArrival );
    const double measuredTrigger_ns   = mcp3208Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns     = measuredTrigger_ns - mcp3208Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( measuredTrigger_ns == Approx( expectedTrigger_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelay_ns ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedAvg_ns );
}

/// Verify synchronized timing wraps negative raw delays into the current WFS period.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming wraps delay with modulo period", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

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
    const double measuredDelay_ns  = mcp3208Ctrl::timespecToNs( app.m_triggerTime ) - mcp3208Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( rawDelay_ns < 0.0 );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelay_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( measuredDelay_ns == Approx( 0.0 ).margin( 1.0 ) );
    REQUIRE( measuredDelay_ns >= 0.0 );
    REQUIRE( measuredDelay_ns < expectedDeltaT_ns );
}

/// Verify synchronized timing leaves positive raw delays unchanged by modulo wrapping.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming preserves positive raw delay", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_firstSemaphore        = false;
    app.m_lastAtime             = timespec{ 4, 0 };
    app.m_avgSemaphorePeriod_ns = 2000000.0;
    app.m_wfs_fps               = 0.0;

    const timespec nextArrival{ 4, 2000000L };
    app.updateTriggerTiming( nextArrival );

    const double expectedAvg_ns   = 2000000.0;
    const double expectedDeltaT_ns = expectedAvg_ns;
    const double rawDelay_ns      = 0.5 * expectedDeltaT_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double measuredDelay_ns = mcp3208Ctrl::timespecToNs( app.m_triggerTime ) - mcp3208Ctrl::timespecToNs( nextArrival );

    REQUIRE( rawDelay_ns > 0.0 );
    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( rawDelay_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( measuredDelay_ns == Approx( 0.0 ).margin( 1.0 ) );
}

/// Verify synchronized timing constants directly control the modeled delay target.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming uses configurable timing constants", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

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

    const double expectedAvg_ns    = 1200000.0;
    const double rawDelayCustom_ns = 0.5 * expectedAvg_ns - ( 4000.0 + 62000.0 + 11000.0 + 290000.0 );
    const double rawDelayDefault_ns = 0.5 * expectedAvg_ns - ( 3000.0 + 51500.0 + 10000.0 + 276100.0 );
    const double expectedDelayCustom_ns = wrapDelay( rawDelayCustom_ns, expectedAvg_ns );
    const double measuredDelay_ns =
        mcp3208Ctrl::timespecToNs( app.m_triggerTime ) - mcp3208Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( app.m_delayModel_ns == Approx( expectedDelayCustom_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( measuredDelay_ns == Approx( 0.0 ).margin( 1.0 ) );
    REQUIRE( expectedDelayCustom_ns != Approx( wrapDelay( rawDelayDefault_ns, expectedAvg_ns ) ) );
}

/// Verify synchronized timing leaves trigger time unchanged when period estimate is non-positive.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming guards non-positive period", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_triggerTime        = timespec{ 7, 12345L };
    app.m_triggerInterval_ns = 42.0;
    app.m_synchroDelayTarget = 12345.0f;
    app.m_delayModel_ns      = 12345.0;
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

/// Verify the semaphore period starts from the first measured interval and normalizes missed WFS frames.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl updateTriggerTiming starts from the first interval per WFS frame", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_alpha                  = 0.01f;
    app.m_firstSemaphore         = false;
    app.m_lastAtime              = timespec{ 10, 0 };
    app.m_avgSemaphorePeriod_ns  = 0.0;
    app.m_syncProducerFrameValid = true;
    app.m_syncProducerFrameDelta = 2;

    app.updateTriggerTiming( timespec{ 10, 1000000L } );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( 500000.0 ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 500000.0 ) );

    // A fitted period replaces the interval average.
    app.updateTriggerTiming( timespec{ 10, 1300000L }, 714285.0 );
    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( 714285.0 ) );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 714285.0 ) );
}

/// Verify timer-driven acquisition configures the published frame geometry.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode configureAcquisition sizes the output frame", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels = 3;

    REQUIRE( app.configureAcquisition() == 0 );
    REQUIRE( app.m_values.size() == 3 );
    REQUIRE( app.m_width == 3 );
    REQUIRE( app.m_height == 1 );
    REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
}

/// Verify timer-driven acquisition still reads one frame of ADC values.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode reads configured channels", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 11, 22, 33 };

    app.m_numChannels = 3;
    app.m_values.assign( 3, 0 );
    app.m_gain       = 0;
    app.m_trigger    = 0;
    app.m_time_start = std::chrono::high_resolution_clock::now();

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 11, 22, 33 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1, 2 } ) );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
}

/// Verify timer mode dispatch remains in integrator timing even when synchro stream metadata is present.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timingSource dispatch prefers integrator mode over synchro stream name", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 31 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_timingSource       = mcp3208Ctrl::TimingSource::Integrator;
    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_numChannels        = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroStreamOpen  = true;
    app.m_trigger            = 0.0f;
    app.m_gain               = 0.0f;
    app.m_time_start         = std::chrono::high_resolution_clock::now();

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_firstSemaphore == true );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 31 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0 } ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify timer-mode trigger interval diagnostics initialize then report measured intervals.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode trigger interval initializes then measures", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    timespec         firstTimestamp{};

    resetStubState();
    stubState().m_channelValues = { 17 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_gain       = 0.0f;
    app.m_trigger    = 1000.0f;

    app.m_time_start = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds( 2 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    firstTimestamp = app.m_currImageTimestamp;

    app.m_time_start = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds( 4 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns > 1000000.0 );
    REQUIRE( mcp3208Ctrl::timespecToNs( app.m_currImageTimestamp ) > mcp3208Ctrl::timespecToNs( firstTimestamp ) );
}

/// Verify the current MCP3208 values are copied into the output image buffer.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl loadImageIntoStream copies the current values", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test      app;
    std::vector<uint16_t> dest( 3, 0 );

    app.m_values = { 5, 6, 7 };

    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( dest == std::vector<uint16_t>( { 5, 6, 7 } ) );
}

/// Verify published-frame counters track local and synchronized stream writes.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl loadImageIntoStream updates frame mapping counters", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test      app;
    std::vector<uint16_t> dest( 2, 0 );

    app.m_values = { 9, 8 };

    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( app.m_localFrameSeq == 1 );
    REQUIRE( app.m_syncFramesWritten == 0 );

    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Semaphore;
    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( app.m_localFrameSeq == 2 );
    REQUIRE( app.m_syncFramesWritten == 1 );
}

/// Verify a WFS reference arms `floor(fps * period)` scans from the fitted arrival, with no built-in delay.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized reference arms scans at the fitted arrival", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels           = 2;
    app.m_fps                   = 8000;
    app.m_syncFramesReceived    = 100;
    app.m_synchroDelay          = 0.0f;
    app.m_synchroDelayTarget    = 0.0f;

    const int64_t arrival = 1000000000;
    app.qualifyClock( arrival, 101 );

    // No built-in delay: the four 125-us scans start at the fitted arrival. The model (409.4 us) is reported only.
    REQUIRE( app.processSyncReference( arrival ) == 0 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_delayModel_ns == Approx( 409400.0 ) );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.m_interval == Approx( 125000.0 ) );
    REQUIRE( app.m_syncSchedule.m_phase == Approx( 0.0 ) );
    REQUIRE( app.m_syncSchedule.deadline() == arrival );
    REQUIRE( app.m_syncSchedule.fps() == Approx( 8000.0 ) );
    REQUIRE( app.m_delayCapped == Approx( 0.0 ) );
    REQUIRE( app.m_syncFramesReceived == 101 );

    // A synchroDelay offset shifts the scans.
    app.m_synchroPostDelay = 10;
    REQUIRE( app.processSyncReference( arrival + 500000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_phase == Approx( 10000.0 ) );
    REQUIRE( app.m_syncSchedule.deadline() == arrival + 510000 );

    // Requesting the WFS rate keeps one scan per period.
    app.m_synchroPostDelay = 0;
    app.m_fps              = 2000;
    REQUIRE( app.processSyncReference( arrival + 1000000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 1 );
    REQUIRE( app.m_syncSchedule.m_phase == Approx( 0.0 ) );

    // Before the rolling clock qualifies, a reference arms nothing and is not counted as a failure.
    mcp3208Ctrl_test fresh;
    fresh.m_fps = 8000;
    REQUIRE( fresh.processSyncReference( arrival ) == 1 );
    REQUIRE( fresh.m_syncSchedule.m_count == 0 );
    REQUIRE( fresh.m_scheduleFailures == 0 );
}

/// Verify the rolling clock re-fits a WFS rate change and the scan count follows it within the limit.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized scans follow a WFS rate change within the limit", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels        = 2;
    app.m_fps                = 8000;
    app.m_syncFramesReceived = 100;

    int64_t arrival = 1000000000;
    app.qualifyClock( arrival, 101 );
    REQUIRE( app.processSyncReference( arrival ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );

    // 2 kHz to 700 Hz: the arrivals are outliers of the 2 kHz fit, which keeps scheduling them while 16 consistent
    // arrivals confirm the new rate; the last of them re-fits.
    for( int frame = 1; frame <= 16; ++frame )
    {
        arrival += 1428571;
        REQUIRE( app.processSyncReference( arrival ) == 0 );
        REQUIRE( app.m_wfsClock.m_resets == ( frame < 16 ? 0u : 1u ) );
    }

    REQUIRE( app.m_wfsClock.m_resets == 1 );
    REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 1428571.0 ) );
    REQUIRE( app.m_syncSchedule.m_count == 11 );
    REQUIRE( app.m_syncSchedule.fps() == Approx( 7700.0 ).epsilon( 1e-4 ) );
    REQUIRE( app.m_syncSchedule.m_anchor == arrival );

    // A 500 Hz WFS gets 16 scans, 8 kHz.
    REQUIRE( app.m_syncSchedule.arm( 1000000000, 2000000, app.m_fps, 0, 40000, 20000 ) );
    REQUIRE( app.m_syncSchedule.m_count == 16 );

    // Lowering the limit to 6 kHz gives 3 scans at 2 kHz.
    app.m_fps = 6000;
    REQUIRE( app.m_syncSchedule.arm( 1000000000, 500000, app.m_fps, 0, 40000, 20000 ) );
    REQUIRE( app.m_syncSchedule.m_count == 3 );
}

/// Verify a burst of late arrivals (checked for a rate change, but none confirmed) still schedules every frame.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized scans continue through a burst of late arrivals", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels        = 2;
    app.m_fps                = 8000;
    app.m_syncFramesReceived = 100;

    const int64_t start = 1000000000;
    app.qualifyClock( start, 101 );
    app.setupTimingDiagnosticsProperty();
    app.m_timingSource          = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_effectiveTimingSource = mcp3208Ctrl::TimingSource::Semaphore;

    // Frames 3 to 8 arrive 200 us late (as when TCP delivers a burst): rejected, the clock checks them for a rate
    // change, but each frame is still scheduled at its fitted arrival, and the lock holds.
    for( int frame = 0; frame < 12; ++frame )
    {
        const int64_t nominal = start + frame * 500000;
        const int64_t late    = frame >= 3 && frame <= 8 ? 200000 : 0;

        REQUIRE( app.processSyncReference( nominal + late ) == 0 );
        REQUIRE( app.m_syncSchedule.m_anchor == nominal );
        app.m_syncSchedule.m_slot = app.m_syncSchedule.m_count; // Scanned.

        app.m_delayApplied_ns = 0.0;
        app.updateTimingDiagnosticsIndi();
        REQUIRE( app.m_indiP_timingDiag["delay_lock"].get<double>() == Approx( 1.0 ) );
    }

    REQUIRE( app.m_wfsClock.m_rejected == 6 );
    REQUIRE( app.m_wfsClock.m_resets == 0 );
    REQUIRE( app.m_missedFrames == 0 );
    REQUIRE( app.m_syncSchedule.m_skipped == 0 );
}

/// Verify semaphore jitter moves neither the scans' period nor their anchor.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized scans are anchored on the fitted WFS arrival", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels        = 2;
    app.m_fps                = 8000;
    app.m_syncFramesReceived = 100;

    const int64_t start = 1000000000;
    app.qualifyClock( start, 101 );

    // Arrivals alternately 30 us late and 30 us early, and one 200 us late (rejected): every frame still gets 4
    // scans, anchored within 10 us of the nominal 2 kHz frame time, and within 1 us by the last.
    for( int frame = 0; frame < 40; ++frame )
    {
        const int64_t nominal = start + frame * 500000;
        const int64_t error   = frame == 25 ? 200000 : ( frame % 2 ? 30000 : -30000 );

        REQUIRE( app.processSyncReference( nominal + error ) == 0 );
        REQUIRE( app.m_syncSchedule.m_count == 4 );
        REQUIRE( std::abs( app.m_syncSchedule.m_anchor - nominal ) < 10000 );
        REQUIRE( app.m_wfsPeriodMeasured_ns == Approx( 500000.0 ).margin( 1000.0 ) );
    }

    REQUIRE( std::abs( app.m_syncSchedule.m_anchor - ( start + 39 * 500000 ) ) < 1000 );
    REQUIRE( app.m_wfsClock.m_rejected == 1 );
    REQUIRE( app.m_wfsClock.m_resets == 0 );
}

/// Verify INDI `scanLateTolerance` sets the lateness at which scans are still taken, from the next reference.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl scanLateTolerance sets the scan lateness tolerance", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_indiP_lateTolerance = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_lateTolerance.setName( "scanLateTolerance" );
    app.m_indiP_lateTolerance.add( pcf::IndiElement( "current" ) );
    app.m_indiP_lateTolerance.add( pcf::IndiElement( "target" ) );

    pcf::IndiProperty update( pcf::IndiProperty::Number );
    update.setName( "scanLateTolerance" );
    update.add( pcf::IndiElement( "target" ) );
    update["target"].setValue( 12.5 );

    REQUIRE( app.newCallBack_m_indiP_lateTolerance( update ) == 0 );
    REQUIRE( app.m_lateTolerance_ns == Approx( 12500.0 ) );

    app.m_numChannels        = 2;
    app.m_fps                = 8000;
    app.m_syncFramesReceived = 100;
    app.qualifyClock( 1000000000, 101 );

    REQUIRE( app.processSyncReference( 1000000000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_tolerance == Approx( 12500.0 ) );
    REQUIRE( app.m_syncSchedule.m_complete == true );
}

/// Verify that, not complete, a slow ADC read lowers the scans per WFS frame and a read estimate that fits nothing
/// is measured again; complete, the count is kept.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized scans fit the ADC read time", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_completeScans        = false;
    app.m_numChannels          = 2;
    app.m_fps                  = 8000;
    app.m_firstNonDelayService = false;
    app.m_syncFramesReceived   = 100;
    app.qualifyClock( 1000000000, 101 );

    // 106.5 us reads plus the 20 us guard do not fit 125 us: 3 scans, 6 kHz.
    app.m_avgNonDelayService_ns = 106530.0;
    REQUIRE( app.processSyncReference( 1000000000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 3 );
    REQUIRE( app.m_syncSchedule.fps() == Approx( 6000.0 ) );

    // A read estimate longer than the period fits nothing; it restarts from the nominal read time.
    app.m_avgNonDelayService_ns = 600000.0;
    REQUIRE( app.processSyncReference( 1000500000 ) == 1 );
    REQUIRE( app.m_scheduleFailures == 1 );
    REQUIRE( app.m_firstNonDelayService == true );
    REQUIRE( app.processSyncReference( 1001000000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );

    // Complete (the default), slow reads never lower the count.
    app.m_completeScans         = true;
    app.m_avgNonDelayService_ns = 106530.0;
    REQUIRE( app.processSyncReference( 1001500000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
}

/// Verify scans read at their deadlines, record their phase, and are never published as a catch-up burst.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized scans read at deadlines and skip late slots", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 101, 202 };

    app.m_numChannels = 2;
    app.m_values.assign( 2, 0 );
    app.m_gain               = 0.0f;
    app.m_synchroDelayTarget = 409400.0f;

    const int64_t anchor = 1000000000;
    REQUIRE( app.m_syncSchedule.arm( anchor, 500000, 8000, 409400, 40000, 20000 ) );

    REQUIRE( app.acquireScan( anchor + 34399 ) == 1 ); // Early: nothing is read.
    REQUIRE( stubState().m_readOrder.empty() );

    REQUIRE( app.acquireScan( anchor + 34400 ) == 0 );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 101, 202 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1 } ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 34400.0 ) );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( app.m_lateScans == 0 );

    // Waking after slot 3 is due skips slots 1 and 2 and reads slot 3 once, late.
    REQUIRE( app.acquireScan( anchor + 409400 + 5000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_skipped == 2 );
    REQUIRE( app.m_delayApplied_ns == Approx( 409400.0 ) );
    REQUIRE( app.m_triggerInterval_ns == Approx( 380000.0 ) );
    REQUIRE( app.m_lateScans == 1 );
    REQUIRE( app.m_maxLateness_ns == Approx( 5000.0 ) );
    REQUIRE( app.acquireScan( anchor + 500000 ) == 1 ); // The period is exhausted; nothing free-runs.
    REQUIRE( stubState().m_readOrder.size() == 4 );
}

/// Verify one semaphore post publishes four synchronized scans at 8 kHz with a 2 kHz WFS period.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized mode reads four scans per WFS semaphore", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 101, 202, 303 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_timingSource     = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_numChannels      = 3;
    app.m_values.assign( 3, 0 );
    app.m_synchroSemaphore      = &semaphore;
    app.m_synchroStreamOpen     = true;
    app.m_gain                  = 0.0f;
    app.m_fps                   = 8000;
    app.m_syncFramesReceived    = 100;

    app.qualifyClock( mcp3208Ctrl::monotonicNow(), 101 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    for( int scan = 0; scan < 4; ++scan )
    {
        REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Semaphore );
        REQUIRE( app.m_delayApplied_ns == Approx( 125000.0 * scan ).margin( 1.0 ) );
    }

    REQUIRE( app.m_values == std::vector<uint16_t>( { 101, 202, 303 } ) );
    REQUIRE( stubState().m_readOrder.size() == 12 );
    REQUIRE( app.m_syncFramesReceived == 101 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    REQUIRE( app.m_syncSchedule.m_skipped == 0 );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    REQUIRE( app.m_firstReadLatency == false );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized mode derives producer cadence from stream metadata counters and timestamps.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized mode tracks producer cadence from metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    IMAGE_METADATA   metadata{};

    resetStubState();
    stubState().m_channelValues = { 44 };

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_timingSource       = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_numChannels        = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroStreamOpen = true;
    app.m_synchroStream.md = &metadata;
    app.m_synchroDelay     = 0.0f;
    app.m_synchroDelayTarget = 0.0f;
    app.m_gain             = 0.0f;
    app.m_alpha            = 0.5f;
    app.m_wfs_fps          = 2000.0;

    metadata.cnt0  = 100;
    metadata.atime = timespec{ 10, 0 };
    app.processSyncReference( mcp3208Ctrl::monotonicNow() );

    REQUIRE( app.m_firstProducerSample == false );
    REQUIRE( app.m_producerPeriodInst_ns == Approx( 0.0 ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( 0.0 ) );
    REQUIRE( app.m_syncFramesReceived == 1 );
    REQUIRE( app.m_syncFramesDropped == 0 );
    REQUIRE( app.m_syncFrameIdGapCount == 0 );
    REQUIRE( app.m_syncProducerFrameId == 100 );
    REQUIRE( app.m_syncProducerFrameDelta == 0 );
    REQUIRE( app.m_syncProducerFrameValid == true );

    metadata.cnt0  = 102;
    metadata.atime = timespec{ 10, 1000000L };
    app.processSyncReference( mcp3208Ctrl::monotonicNow() );

    REQUIRE( app.m_producerPeriodInst_ns == Approx( 500000.0 ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( 500000.0 ) );
    REQUIRE( app.m_lastProducerCnt0 == 102 );
    REQUIRE( app.m_syncFramesReceived == 2 );
    REQUIRE( app.m_syncFramesDropped == 1 );
    REQUIRE( app.m_syncFrameIdGapCount == 1 );
    REQUIRE( app.m_syncProducerFrameId == 102 );
    REQUIRE( app.m_syncProducerFrameDelta == 2 );

    metadata.cnt0  = 104;
    metadata.atime = timespec{ 10, 2100000L };
    app.processSyncReference( mcp3208Ctrl::monotonicNow() );

    const double expectedPeriod2_ns = 550000.0;
    const double expectedAvg2_ns =
        static_cast<double>( app.m_alpha ) * expectedPeriod2_ns + ( 1.0 - static_cast<double>( app.m_alpha ) ) * 500000.0;
    REQUIRE( app.m_producerPeriodInst_ns == Approx( expectedPeriod2_ns ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( expectedAvg2_ns ) );
    REQUIRE( app.m_syncFramesReceived == 3 );
    REQUIRE( app.m_syncFramesDropped == 2 );
    REQUIRE( app.m_syncFrameIdGapCount == 2 );
    REQUIRE( app.m_syncProducerFrameId == 104 );
    REQUIRE( app.m_syncProducerFrameDelta == 2 );

    const double periodBeforeNoAdvance_ns = app.m_producerPeriodInst_ns;
    const double avgBeforeNoAdvance_ns    = app.m_avgProducerPeriod_ns;

    metadata.cnt0  = 104;
    metadata.atime = timespec{ 10, 2200000L };
    app.processSyncReference( mcp3208Ctrl::monotonicNow() );

    REQUIRE( app.m_producerPeriodInst_ns == Approx( periodBeforeNoAdvance_ns ) );
    REQUIRE( app.m_avgProducerPeriod_ns == Approx( avgBeforeNoAdvance_ns ) );
    REQUIRE( app.m_syncFramesReceived == 4 );
    REQUIRE( app.m_syncFramesDropped == 2 );
    REQUIRE( app.m_syncFrameIdGapCount == 2 );
    REQUIRE( app.m_syncProducerFrameId == 104 );
    REQUIRE( app.m_syncProducerFrameDelta == 0 );

    REQUIRE( app.m_duplicateReferences == 1 );
}

/// Verify complete scanning reads a late slot late, and the previous period's unscanned slot at once.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized complete scans never skip a slot", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 101, 202 };

    app.m_numChannels = 2;
    app.m_values.assign( 2, 0 );
    app.m_gain                    = 0.0f;
    app.m_synchroDelayTarget      = 409400.0f;
    app.m_syncSchedule.m_complete = true;

    const int64_t anchor = 1000000000;
    REQUIRE( app.m_syncSchedule.arm( anchor, 500000, 8000, 409400, 40000, 20000 ) );

    // Slots 0 to 2 are read 300 us late, one after another; nothing is skipped.
    REQUIRE( app.acquireScan( anchor + 334400 ) == 0 );
    REQUIRE( app.acquireScan( anchor + 423400 ) == 0 );
    REQUIRE( app.acquireScan( anchor + 512400 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_slot == 3 );
    REQUIRE( app.m_lateScans == 3 );
    REQUIRE( app.m_delayApplied_ns == Approx( 284400.0 ) );

    // Slot 3 was not read when the next period armed: it is read at once, keeping the last slot's phase.
    REQUIRE( app.m_syncSchedule.arm( anchor + 500000, 500000, 8000, 409400, 40000, 20000 ) );
    REQUIRE( app.acquireScan( anchor + 601400 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_carried == 1 );
    REQUIRE( app.m_syncSchedule.m_backlog == 0 );
    REQUIRE( app.m_delayApplied_ns == Approx( 284400.0 ) );
    REQUIRE( app.m_firstReadLatency == true );
    REQUIRE( app.m_syncSchedule.m_skipped == 0 );
    REQUIRE( stubState().m_readOrder.size() == 8 );
}

/// Verify the scans of a WFS frame that armed nothing (its notification merged or missed) are owed and read.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized complete scans owe missed WFS frames", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_numChannels        = 2;
    app.m_fps                = 8000;
    app.m_syncFramesReceived = 100;
    app.qualifyClock( 1000000000, 101 );

    REQUIRE( app.processSyncReference( 1000000000 ) == 0 );
    REQUIRE( app.m_syncSchedule.m_count == 4 );
    app.m_syncSchedule.m_slot = 4; // Frame 101 fully scanned.

    // Frame 102's notification is lost: frame 103 arms its own 4 scans and owes frame 102's.
    app.m_syncFramesReceived = 102;
    REQUIRE( app.processSyncReference( 1001000000 ) == 0 );
    REQUIRE( app.m_missedFrames == 1 );
    REQUIRE( app.m_syncSchedule.m_backlog == 4 );
    REQUIRE( app.m_syncSchedule.m_carried == 4 );
    REQUIRE( app.m_syncSchedule.m_skipped == 0 );
    REQUIRE( app.m_lastArmedId == 103 );

    // Not complete: the missed frame is counted but its scans are not owed.
    app.m_completeScans      = false;
    app.m_syncSchedule       = {};
    app.m_syncFramesReceived = 104;
    REQUIRE( app.processSyncReference( 1002000000 ) == 0 );
    REQUIRE( app.m_missedFrames == 2 );
    REQUIRE( app.m_syncSchedule.m_backlog == 0 );
}

/// Verify the read-latency EMA updates once per WFS period, at the scan nearest the modelled phase.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized read latency EMA initializes and smooths", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 77 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_gain               = 0.0f;
    app.m_alpha              = 0.2f;
    app.m_synchroDelayTarget = 409400.0f;

    const int64_t anchor = 1000000000;
    REQUIRE( app.m_syncSchedule.arm( anchor, 500000, 8000, 409400, 40000, 20000 ) );

    for( int64_t offset : { 34400LL, 159400LL, 284400LL } )
    {
        REQUIRE( app.acquireScan( anchor + offset ) == 0 );
        REQUIRE( app.m_firstReadLatency == true );
    }

    REQUIRE( app.acquireScan( anchor + 409400 ) == 0 );
    REQUIRE( app.m_firstReadLatency == false );
    REQUIRE( app.m_avgReadLatency_ns == Approx( 409400.0 ) );

    REQUIRE( app.m_syncSchedule.arm( anchor + 500000, 500000, 8000, 409400, 40000, 20000 ) );
    REQUIRE( app.acquireScan( anchor + 500000 + 409400 + 1000 ) == 0 );

    const double alpha = static_cast<double>( app.m_alpha );
    REQUIRE( app.m_avgReadLatency_ns == Approx( alpha * 410400.0 + ( 1.0 - alpha ) * 409400.0 ) );
}

/// Verify the scan service EMA, which bounds the schedule, uses the configurable global alpha.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized non-delay service EMA uses global alpha", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 55 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_gain                  = 0.0f;
    app.m_alpha                 = 0.2f;
    app.m_firstNonDelayService  = true;
    app.m_avgNonDelayService_ns = 0.0;

    const int64_t anchor = 1000000000;
    REQUIRE( app.m_syncSchedule.arm( anchor, 500000, 8000, 0, 40000, 20000 ) );

    REQUIRE( app.acquireScan( anchor ) == 0 );

    const double nonDelay0_ns = app.m_nonDelayService_ns;
    REQUIRE( app.m_firstNonDelayService == false );
    REQUIRE( nonDelay0_ns == Approx( app.m_channelReadoutTime_ns ) );
    REQUIRE( app.m_avgNonDelayService_ns == Approx( nonDelay0_ns ) );

    REQUIRE( app.acquireScan( anchor + 125000 ) == 0 );

    const double nonDelay1_ns = app.m_nonDelayService_ns;
    const double alpha        = static_cast<double>( app.m_alpha );
    REQUIRE( app.m_avgNonDelayService_ns == Approx( alpha * nonDelay1_ns + ( 1.0 - alpha ) * nonDelay0_ns ) );
}

/// Verify the read latency is reported but not fed back: the scan offset stays at the requested one.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized read latency is reported, not fed back", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 99 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_fps                = 8000;
    app.m_alpha              = 0.2f;
    app.m_syncFramesReceived = 100;
    app.qualifyClock( 1000000000, 101 );

    REQUIRE( app.processSyncReference( 1000000000 ) == 0 );

    // Slot 0 starts 10 us late: reported as the read latency, while the scans stay at the fitted arrival.
    REQUIRE( app.acquireScan( 1000010000 ) == 0 );
    REQUIRE( app.m_avgReadLatency_ns == Approx( 10000.0 ) );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_values[0] == 99 );

    app.m_syncSchedule.m_slot = 4;
    REQUIRE( app.processSyncReference( 1000500000 ) == 0 );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_syncSchedule.m_phase == Approx( 0.0 ) );
}

/// Verify unavailable semaphore timing falls back to internal timer acquisition.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl semaphore mode falls back to timer when synchronization is unavailable", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 7, 8 };

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_timingSource       = mcp3208Ctrl::TimingSource::Semaphore;
    app.m_numChannels        = 2;
    app.m_values.assign( 2, 0 );
    app.m_synchroSemaphore   = nullptr;
    app.m_synchroStreamOpen  = false;
    app.m_trigger            = 0.0f;
    app.m_gain               = 0.0f;
    app.m_time_start         = std::chrono::high_resolution_clock::now();

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_timingSource == mcp3208Ctrl::TimingSource::Semaphore );
    REQUIRE( app.m_effectiveTimingSource == mcp3208Ctrl::TimingSource::Integrator );
    REQUIRE( app.m_semaphoreFallbackActive == true );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 7, 8 } ) );
    REQUIRE( stubState().m_readOrder == std::vector<int>( { 0, 1 } ) );
}

/// Verify stale-stream detection notices a missing synchronization stream backing file.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized stale helper detects missing streams", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    IMAGE_METADATA   metadata{};

    app.m_synchroShmimName  = "mcp3208Ctrl_unit_test_missing_stream";
    app.m_synchroStreamOpen = true;
    metadata.sem            = SEMAPHORE_MAXVAL;
    app.m_synchroStream.md  = &metadata;

    REQUIRE( app.synchroStreamStale() == true );
}

/// Verify `reconfig()` clears cached synchronization state.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl reconfig clears cached synchronization state", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_synchroSemaphore       = reinterpret_cast<sem_t *>( 0x1 );
    app.m_synchroSemaphoreNumber = 7;
    app.m_synchroStreamInode     = 1234;
    app.m_synchroStreamOpen      = false;
    app.m_atime                  = timespec{ 1, 1 };
    app.m_lastAtime              = timespec{ 2, 2 };
    app.m_avgSemaphorePeriod_ns  = 42.0;
    app.m_wfsPeriodMeasured_ns   = 21.0;
    app.m_lastProducerAtime      = timespec{ 3, 4 };
    app.m_lastProducerCnt0       = 123;
    app.m_producerPeriodInst_ns  = 500000.0;
    app.m_avgProducerPeriod_ns   = 510000.0;
    app.m_firstProducerSample    = false;
    app.m_localFrameSeq          = 22;
    app.m_syncFramesReceived     = 21;
    app.m_syncFramesWritten      = 20;
    app.m_syncFramesDropped      = 4;
    app.m_syncFrameIdGapCount    = 3;
    app.m_syncProducerFrameId    = 1234567;
    app.m_syncProducerFrameDelta = 5;
    app.m_lastSyncProducerFrameId = 1234562;
    app.m_syncProducerFrameValid  = true;
    app.m_firstSemaphore         = false;
    app.m_avgReadLatency_ns      = 84.0;
    app.m_firstReadLatency       = false;
    app.m_delayModel_ns          = 900.0;
    app.m_delayApplied_ns        = 875.0;
    app.m_delayBudget_ns         = 450000.0;
    app.m_nonDelayService_ns     = 170000.0;
    app.m_avgNonDelayService_ns  = 160000.0;
    app.m_firstNonDelayService   = false;
    app.m_delayPhaseError_ns     = -25.0;
    app.m_delayLock              = 1.0;
    app.m_delayCapped            = 1.0;
    app.m_triggerTime            = timespec{ 3, 3 };
    app.m_triggerInterval_ns     = 21.0;
    app.m_lastTriggerTime        = timespec{ 4, 4 };
    app.m_firstTriggerTime       = false;
    app.m_firstTimerTrigger      = false;
    app.m_lateScans               = 3;
    app.m_maxLateness_ns          = 9000.0;
    app.m_coalescedWakes          = 2;
    app.m_duplicateReferences     = 1;
    app.m_scheduleFailures        = 4;
    app.m_haveArmedId             = true;
    app.m_missedFrames            = 6;
    REQUIRE( app.m_syncSchedule.arm( 1000000000, 500000, 8000, 0, 40000, 20000 ) );
    app.qualifyClock( 2000000000, 100 );
    app.m_wfsClockSpan_s = 0.0075;
    REQUIRE( app.m_wfsClock.valid() );

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
    REQUIRE( app.m_syncSchedule.m_count == 0 );
    REQUIRE( app.m_syncSchedule.m_skipped == 0 );
    REQUIRE( app.m_syncSchedule.m_haveSample == false );
    REQUIRE( app.m_lateScans == 0 );
    REQUIRE( app.m_maxLateness_ns == Approx( 0.0 ) );
    REQUIRE( app.m_coalescedWakes == 0 );
    REQUIRE( app.m_duplicateReferences == 0 );
    REQUIRE( app.m_scheduleFailures == 0 );
    REQUIRE( app.m_haveArmedId == false );
    REQUIRE( app.m_missedFrames == 0 );
    REQUIRE_FALSE( app.m_wfsClock.valid() );
    REQUIRE( app.m_wfsClock.m_count == 0 );
    REQUIRE( app.m_wfsClockSpan_s == Approx( 0.0 ) );
}

} // namespace mcp3208CtrlTest
} // namespace libXWCTest
