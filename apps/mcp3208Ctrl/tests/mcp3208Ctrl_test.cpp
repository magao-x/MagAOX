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
    REQUIRE( app.m_alpha == Approx( 0.01f ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( 0.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( static_cast<double>( app.m_fps ) ) );
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
    REQUIRE( app.m_alpha == Approx( 0.25f ) );
    REQUIRE( app.m_numChannels == 3 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 17000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 17000.0f ) );
    REQUIRE( app.m_wfs_fps == Approx( static_cast<double>( app.m_fps ) ) );
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
    REQUIRE( app.m_synchroDelayTarget == Approx( 145000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 145000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 145000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );

    REQUIRE( app.newCallBack_m_indiP_synchroDelay( app.makeSynchroDelayUpdate( -5.0 ) ) == 0 );
    REQUIRE( app.m_synchroPostDelay == -5 );
    REQUIRE( app.m_synchroDelayTarget == Approx( 115000.0f ) );
    REQUIRE( app.m_synchroDelay == Approx( 115000.0f ) );
    REQUIRE( app.m_delayApplied_ns == Approx( 115000.0 ) );
    REQUIRE( app.m_delayModel_ns == Approx( 120000.0 ) );
}

/// Verify synchronized-mode timing diagnostics publish loop state and derived error.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timing diagnostics publish synchronized loop metrics", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.setupTimingDiagnosticsProperty();
    app.m_synchroShmimName      = "camwfs_sync";
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
    app.m_synchroShmimName = "camwfs_sync";
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

    app.m_synchroShmimName.clear();
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
    app.m_synchroShmimName        = "camwfs_sync";
    app.m_delayApplied_ns         = 50.0;
    app.m_delayModel_ns           = 900.0;
    app.m_wfsPeriodMeasured_ns    = 1000.0;
    app.m_delayLockAbsThreshold_ns = 200.0;
    app.m_delayLockFracThreshold   = 0.1;

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
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelayMeasured_ns ) ) );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelayMeasured_ns ) );
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
    const double expectedTarget_ns  = wrapDelay( expectedModel_ns + 100000.0, expectedPeriod_ns );
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
    const double expectedTarget_ns  = wrapDelay( expectedModel_ns - 300000.0, expectedPeriod_ns );
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
    const double expectedTrigger_ns   = mcp3208Ctrl::timespecToNs( nextArrival ) + expectedDelay_ns;
    const double measuredTrigger_ns   = mcp3208Ctrl::timespecToNs( app.m_triggerTime );
    const double measuredDelay_ns     = measuredTrigger_ns - mcp3208Ctrl::timespecToNs( nextArrival );

    REQUIRE( app.m_avgSemaphorePeriod_ns == Approx( expectedAvg_ns ) );
    REQUIRE( measuredTrigger_ns == Approx( expectedTrigger_ns ) );
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelay_ns ) ) );
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
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelay_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelay_ns ) );
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
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( rawDelay_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( rawDelay_ns ) );
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
    REQUIRE( app.m_synchroDelayTarget == Approx( static_cast<float>( expectedDelayCustom_ns ) ) );
    REQUIRE( measuredDelay_ns == Approx( expectedDelayCustom_ns ) );
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
}

/// Verify timer-mode trigger interval diagnostics initialize then report measured intervals.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl timer mode trigger interval initializes then measures", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    resetStubState();
    stubState().m_channelValues = { 17 };

    app.m_numChannels = 1;
    app.m_values.assign( 1, 0 );
    app.m_gain       = 0.0f;
    app.m_trigger    = 1000.0f;

    app.m_time_start = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds( 2 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns == Approx( 0.0 ) );

    app.m_time_start = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds( 4 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_triggerInterval_ns > 1000000.0 );
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

    app.m_synchroShmimName = "camwfs_sync";
    REQUIRE( app.loadImageIntoStream( dest.data() ) == 0 );
    REQUIRE( app.m_localFrameSeq == 2 );
    REQUIRE( app.m_syncFramesWritten == 1 );
}

/// Verify synchronized acquisition performs one ADC sweep per semaphore wake.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized mode reads on semaphore wake", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

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
    REQUIRE( app.m_triggerTime.tv_sec > 0 );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    REQUIRE( app.m_firstReadLatency == false );
    REQUIRE( app.m_avgReadLatency_ns ==
             Approx( mcp3208Ctrl::timespecToNs( app.m_currImageTimestamp ) - mcp3208Ctrl::timespecToNs( app.m_atime ) ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized mode derives producer cadence from stream metadata counters and timestamps.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized mode tracks producer cadence from metadata", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;
    IMAGE_METADATA   metadata{};

    resetStubState();
    stubState().m_channelValues = { 44 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName = "camwfs_sync";
    app.m_numChannels      = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore = &semaphore;
    app.m_synchroStream.md = &metadata;
    app.m_synchroDelay     = 0.0f;
    app.m_synchroDelayTarget = 0.0f;
    app.m_gain             = 0.0f;
    app.m_alpha            = 0.5f;
    app.m_wfs_fps          = 2000.0;

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

    metadata.cnt0  = 102;
    metadata.atime = timespec{ 10, 1000000L };
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

    metadata.cnt0  = 104;
    metadata.atime = timespec{ 10, 2100000L };
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

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
    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

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
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized read latency EMA initializes and smooths", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 77 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_numChannels        = 1;
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

    const double readLatency0_ns =
        mcp3208Ctrl::timespecToNs( app.m_currImageTimestamp ) - mcp3208Ctrl::timespecToNs( app.m_atime );

    REQUIRE( app.m_firstReadLatency == false );
    REQUIRE( app.m_avgReadLatency_ns == Approx( readLatency0_ns ) );

    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double readLatency1_ns =
        mcp3208Ctrl::timespecToNs( app.m_currImageTimestamp ) - mcp3208Ctrl::timespecToNs( app.m_atime );
    const double alpha = static_cast<double>( app.m_alpha );
    const double expectedAvgLatency_ns = alpha * readLatency1_ns + ( 1.0 - alpha ) * readLatency0_ns;

    REQUIRE( app.m_avgReadLatency_ns == Approx( expectedAvgLatency_ns ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized non-delay service EMA uses the configurable global alpha.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized non-delay service EMA uses global alpha", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 55 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName      = "camwfs_sync";
    app.m_numChannels           = 1;
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

    REQUIRE( sem_post( &semaphore ) == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double nonDelay1_ns = app.m_nonDelayService_ns;
    const double alpha        = static_cast<double>( app.m_alpha );
    const double expectedAvgNonDelay_ns = alpha * nonDelay1_ns + ( 1.0 - alpha ) * nonDelay0_ns;

    REQUIRE( app.m_avgNonDelayService_ns == Approx( expectedAvgNonDelay_ns ) );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control uses read-latency EMA for the integrator correction.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized delay controller uses read latency EMA", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 99 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_numChannels        = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelayTarget = 0.0f;
    app.m_synchroDelay       = 2000000.0f;
    app.m_gain               = 1.0f;
    app.m_alpha              = 0.2f;
    app.m_firstReadLatency   = false;
    app.m_avgReadLatency_ns  = 800000.0;

    REQUIRE( app.acquireAndCheckValid() == 0 );

    const double readLatency_ns =
        mcp3208Ctrl::timespecToNs( app.m_currImageTimestamp ) - mcp3208Ctrl::timespecToNs( app.m_atime );
    const double alpha = static_cast<double>( app.m_alpha );
    const double expectedAvgLatency_ns = alpha * readLatency_ns + ( 1.0 - alpha ) * 800000.0;
    const double expectedDelay_ns =
        ( 2000000.0 - expectedAvgLatency_ns ) > 0.0 ? ( 2000000.0 - expectedAvgLatency_ns ) : 0.0;

    REQUIRE( app.m_avgReadLatency_ns == Approx( expectedAvgLatency_ns ) );
    REQUIRE( app.m_synchroDelay == Approx( expectedDelay_ns ) );
    REQUIRE( app.m_values[0] == 99 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control clamps at zero when the control step overshoots.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized delay controller clamps to zero", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();
    stubState().m_channelValues = { 11 };

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );
    REQUIRE( sem_post( &semaphore ) == 0 );

    app.m_synchroShmimName   = "camwfs_sync";
    app.m_numChannels        = 1;
    app.m_values.assign( 1, 0 );
    app.m_synchroSemaphore   = &semaphore;
    app.m_synchroDelayTarget = 0.0f;
    app.m_synchroDelay       = 1000.0f;
    app.m_gain               = 1.0f;

    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_synchroDelay == Approx( 0.0f ) );
    REQUIRE( app.m_values[0] == 11 );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
}

/// Verify synchronized delay control blocks integrator windup while cadence capping is active.
/**
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized delay controller applies anti-windup at cap", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;

    app.m_gain               = 1.0f;
    app.m_synchroDelayTarget = 300000.0f;
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
 * \ingroup mcp3208Ctrl_unit_test
 */
TEST_CASE( "mcp3208Ctrl synchronized timeout requests reconfig for a stale stream", "[mcp3208Ctrl]" )
{
    mcp3208Ctrl_test app;
    sem_t            semaphore;

    resetStubState();

    REQUIRE( sem_init( &semaphore, 0, 0 ) == 0 );

    app.m_synchroShmimName  = "camwfs_sync";
    app.m_values            = { 7, 8 };
    app.m_synchroSemaphore  = &semaphore;
    app.m_synchroStreamOpen = false;

    REQUIRE( app.acquireAndCheckValid() == 1 );
    REQUIRE( app.m_reconfig == true );
    REQUIRE( app.m_values == std::vector<uint16_t>( { 7, 8 } ) );
    REQUIRE( stubState().m_readOrder.empty() );

    REQUIRE( sem_destroy( &semaphore ) == 0 );
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

} // namespace mcp3208CtrlTest
} // namespace libXWCTest
