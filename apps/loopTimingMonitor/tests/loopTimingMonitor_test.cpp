/** \file loopTimingMonitor_test.cpp
 * \brief Offline configuration, device-gating and fgtimes integration tests.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingMonitor.hpp"

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

/// \cond LOOP_TIMING_MONITOR_TEST_HELPERS
/// Expose only deterministic application operations; no production thread or INDI connection is started.
class Harness : public MagAOX::app::loopTimingMonitor
{
  public:
    using loopTimingMonitor::consume;
    using loopTimingMonitor::m_cameraDevice;
    using loopTimingMonitor::m_capacity;
    using loopTimingMonitor::m_captureEpoch;
    using loopTimingMonitor::m_dmDevice;
    using loopTimingMonitor::m_gateGeneration;
    using loopTimingMonitor::m_lastRecord;
    using loopTimingMonitor::m_loopDevice;
    using loopTimingMonitor::m_maxFPS;
    using loopTimingMonitor::m_monitor;
    using loopTimingMonitor::m_stateTimeoutSec;
    using loopTimingMonitor::m_streamsReady;
    using loopTimingMonitor::m_summarySerial;
    using loopTimingMonitor::monotonicNow;
    using loopTimingMonitor::recordTimings;
    using loopTimingMonitor::timingValues;

    /// Configure isolated storage without starting any service.
    Harness();

    /// Validate the current configured defaults or directly selected test values.
    int validate();

    /// Deliver one configurable source property to the real callback.
    int source( const std::string &device /**< [in] Device name. */,
                const std::string &property /**< [in] Property name. */,
                const std::string &value /**< [in] Text or On/Off value. */ );

    /// Open the known-state gate using the configured device names.
    void enable();

    /// Enqueue one synthetic source observation with the currently observed gate generation.
    void enqueue( bool     camera /**< [in] Camera rather than SDK source. */,
                  uint64_t counter /**< [in] Source counter. */,
                  int64_t  stampNs /**< [in] Monotonic endpoint. */ );

    /// Queue a continuous 300 us cadence with known 100 us camera-to-SDK latency.
    void population();
};

Harness::Harness()
{
    m_configName = "loopTimingMonitor-test";
    setupConfig();
    m_queue.configure( 128 );
    REQUIRE( m_monitor.configure( 3600, 2, 128 ) );
}

int Harness::validate()
{
    return loadConfigImpl( config );
}

int Harness::source( const std::string &device, const std::string &name, const std::string &value )
{
    pcf::IndiProperty property( name == "loop_state" ? pcf::IndiProperty::Switch : pcf::IndiProperty::Text );
    property.setDevice( device );
    property.setName( name );

    if( name == "loop_state" )
    {
        property.add( pcf::IndiElement( "toggle" ) );
        property["toggle"].setSwitchState( value == "On" ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }
    else
    {
        property.add( pcf::IndiElement( "state" ) );
        property["state"] = value;
    }

    return setCallBack_source( property );
}

void Harness::enable()
{
    REQUIRE( source( m_cameraDevice, "fsm", "OPERATING" ) == 0 );
    REQUIRE( source( m_dmDevice, "fsm", "OPERATING" ) == 0 );
    REQUIRE( source( m_loopDevice, "loop_state", "On" ) == 0 );
    m_streamsReady = true;
}

void Harness::enqueue( bool camera, uint64_t counter, int64_t stampNs )
{
    Observation event;
    event.camera  = camera;
    event.counter = counter;
    event.stampNs = event.rawNs = event.lowerNs = event.upperNs = stampNs;
    event.gate                                                  = m_gateGeneration.load();
    event.epoch                                                 = m_captureEpoch.load();
    REQUIRE( m_queue.push( event ) );
}

void Harness::population()
{
    const auto base = monotonicNow() - 10000000;
    enqueue( false, 1, base - 200000 );

    for( uint64_t frame = 0; frame < 7; ++frame )
    {
        enqueue( true, frame + 1, base + int64_t( frame ) * 300000 );
        enqueue( false, frame + 2, base + int64_t( frame ) * 300000 + 100000 );
    }
}
/// \endcond

/** Verify default and configurable limits reject nonfinite/invalid values under production compiler flags.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing configuration validates FPS and device identities", "[loopTimingMonitor]" )
{
    Harness app;
    CHECK( app.m_maxFPS == 3600 );
    CHECK( app.validate() == 0 );

    SECTION( "another fastest cadence" )
    {
        app.m_maxFPS = 6000;
        CHECK( app.validate() == 0 );
    }

    SECTION( "NaN" )
    {
        app.m_maxFPS = std::numeric_limits<double>::quiet_NaN();
        CHECK( app.validate() < 0 );
    }

    SECTION( "infinity" )
    {
        app.m_maxFPS = std::numeric_limits<double>::infinity();
        CHECK( app.validate() < 0 );
    }

    SECTION( "zero" )
    {
        app.m_maxFPS = 0;
        CHECK( app.validate() < 0 );
    }

    SECTION( "empty device" )
    {
        app.m_loopDevice.clear();
        CHECK( app.validate() < 0 );
    }

    SECTION( "duplicate FSM device" )
    {
        app.m_dmDevice = app.m_cameraDevice;
        CHECK( app.validate() < 0 );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::loadConfigImpl;
#endif
    // clang-format on
}

/** Verify all prerequisites are observed and configured names route exact properties.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing accepts only configured operating devices and loop on", "[loopTimingMonitor]" )
{
    Harness app;
    app.m_cameraDevice = "otherCamera";
    app.m_dmDevice     = "otherDM";
    app.m_loopDevice   = "otherLoop";
    CHECK( app.source( "camwfs", "fsm", "OPERATING" ) < 0 );
    app.enable();
    app.population();
    app.consume( app.monotonicNow() );
    std::array<double, 6> values;
    REQUIRE( app.timingValues( values ) );

    SECTION( "camera not operating" )
    {
        REQUIRE( app.source( app.m_cameraDevice, "fsm", "READY" ) == 0 );
    }

    SECTION( "DM not operating" )
    {
        REQUIRE( app.source( app.m_dmDevice, "fsm", "ERROR" ) == 0 );
    }

    SECTION( "loop off" )
    {
        REQUIRE( app.source( app.m_loopDevice, "loop_state", "Off" ) == 0 );
    }

    SECTION( "missing state" )
    {
        pcf::IndiProperty malformed( pcf::IndiProperty::Text );
        malformed.setDevice( app.m_cameraDevice );
        malformed.setName( "fsm" );
        REQUIRE( app.setCallBack_source( malformed ) == 0 );
    }

    SECTION( "state expires" )
    {
        app.m_stateTimeoutSec = 0;
    }

    CHECK_FALSE( app.timingValues( values ) );
    const auto recorded = app.m_lastRecord;
    REQUIRE( app.recordTimings( true ) == 0 );
    CHECK( app.m_lastRecord == recorded );
    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().latency.count == 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::setCallBack_source;
    MagAOX::app::loopTimingMonitor::recordTelem;
    MagAOX::app::loopTimingMonitor::consume;
#endif
    // clang-format on
}

/** Verify gate/attachment epochs reject observations already queued before a transition.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing cannot reuse queued data after gate or source transitions", "[loopTimingMonitor]" )
{
    Harness app;
    app.enable();
    app.population();

    SECTION( "quick off and on" )
    {
        REQUIRE( app.source( app.m_loopDevice, "loop_state", "Off" ) == 0 );
        REQUIRE( app.source( app.m_loopDevice, "loop_state", "On" ) == 0 );
    }

    SECTION( "capture discontinuity" )
    {
        ++app.m_captureEpoch;
    }

    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().latency.count == 0 );
    app.population();
    app.consume( app.monotonicNow() );
    CHECK( app.m_monitor.summary().latency.count > 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::consume;
#endif
    // clang-format on
}

/** Verify exactly the existing persistent fgtimes fields represent the three documented populations.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing serializes camera cadence SDK cadence and latency as fgtimes", "[loopTimingMonitor]" )
{
    Harness               app;
    std::array<double, 6> values;
    CHECK_FALSE( app.timingValues( values ) );
    app.enable();
    app.population();
    app.consume( app.monotonicNow() );
    REQUIRE( app.timingValues( values ) );
    CHECK( values[0] == Approx( 300e-6 ) );
    CHECK( values[1] == Approx( 0 ).margin( 1e-15 ) );
    CHECK( values[2] == Approx( 300e-6 ) );
    CHECK( values[3] == Approx( 0 ).margin( 1e-15 ) );
    CHECK( values[4] == Approx( 100e-6 ) );
    CHECK( values[5] == Approx( 0 ).margin( 1e-15 ) );

    MagAOX::logger::telem_fgtimings::messageT message(
        values[0], values[1], values[2], values[3], values[4], values[5] );
    const auto *record = MagAOX::logger::GetTelem_fgtimings_fb( message.builder.GetBufferPointer() );
    CHECK( record->atime() == values[0] );
    CHECK( record->atime_jitter() == values[1] );
    CHECK( record->wtime() == values[2] );
    CHECK( record->wtime_jitter() == values[3] );
    CHECK( record->wmatime() == values[4] );
    CHECK( record->wmatime_jitter() == values[5] );

    REQUIRE( app.recordTimings() == 0 );
    CHECK( app.m_lastRecord == app.m_summarySerial );
    MagAOX::logger::telem_fgtimings::lastRecord = {};
    REQUIRE( app.recordTimings( true ) == 0 );
    CHECK( MagAOX::logger::telem_fgtimings::lastRecord.tv_sec > 0 );
    app.consume( app.monotonicNow() + 3000000000 );
    CHECK_FALSE( app.timingValues( values ) );
    MagAOX::logger::telem_fgtimings::lastRecord = {};
    REQUIRE( app.recordTimings( true ) == 0 );
    CHECK( MagAOX::logger::telem_fgtimings::lastRecord.tv_sec == 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::timingValues;
#endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
