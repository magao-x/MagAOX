/** \file loopTimingMonitor_test.cpp
 * \brief Offline configuration, device-gating, delayed main batches and fgtimes integration tests.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../dmSyncopation/syncoMainTiming.hpp"
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
    using loopTimingMonitor::m_gateEnabled;
    using loopTimingMonitor::m_gateGeneration;
    using loopTimingMonitor::m_lastRecord;
    using loopTimingMonitor::m_loopDevice;
    using loopTimingMonitor::m_mainStream;
    using loopTimingMonitor::m_maxFPS;
    using loopTimingMonitor::m_monitor;
    using loopTimingMonitor::m_sdkEventStream;
    using loopTimingMonitor::m_streamsReady;
    using loopTimingMonitor::m_summarySerial;
    using loopTimingMonitor::m_syncoDevice;
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
    void population( int64_t baseNs = 0 /**< [in] Monotonic origin; zero selects a recent real timestamp. */ );
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
    const bool        switchProperty = name == "loop_state" || name == "enable";
    pcf::IndiProperty property( switchProperty ? pcf::IndiProperty::Switch : pcf::IndiProperty::Text );
    property.setDevice( device );
    property.setName( name );

    if( switchProperty )
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

void Harness::population( int64_t baseNs )
{
    const auto base = baseNs ? baseNs : monotonicNow() - 10000000;
    enqueue( false, 1, base - 200000 );

    for( uint64_t frame = 0; frame < 7; ++frame )
    {
        enqueue( true, frame + 1, base + int64_t( frame ) * 300000 );
        enqueue( false, frame + 2, base + int64_t( frame ) * 300000 + 100000 );
    }
}

/// Drive the real ordinary consumer with full-rate endpoints and the real delayed main capture helper.
class BatchHarness : public Harness
{
  public:
    /// Select one missing observation halfway through every one-second batch.
    enum class Loss
    {
        none,
        sdk,
        label,
        camera
    };

    /// Configure automatic monitoring without a polling thread or instrument connection.
    explicit BatchHarness( int64_t baseNs /**< [in] Monotonic origin of the synthetic camera timeline. */ );

    /// Enqueue two thousand cameras and eight thousand native endpoints, retaining main labels in the producer.
    void captureBatch( Loss loss /**< [in] Optional single missing observation in this batch. */ );

    /// Publish all pending producer mode/main records as one ordinary-thread burst.
    void publishLabels();

    /// Capture, publish and consume one complete one-second diagnostic batch.
    void batch( Loss loss /**< [in] Optional single missing observation in this batch. */ );

    /// Return a time after all endpoints in the captured batch.
    int64_t now() const;

    /// Return the genuine classification generation, including the real producer's loss-triggered resets.
    uint64_t producerGeneration() const;

    /// Return unmatched positive labels observed by the exact identity join.
    uint64_t unmatched() const;

    /// Return camera evidence not yet released to the temporal matcher.
    size_t pendingCameras() const;

    /// Return camera observations evicted by the independent pending-history capacity bound.
    uint64_t pendingOverflow() const;

  private:
    MagAOX::app::synco::MainTiming m_producer;   ///< Real classification queue, drained only at the end of each batch.
    int64_t                        m_baseNs;     ///< Deterministic MONOTONIC origin of this fixture.
    uint64_t                       m_frame{ 0 }; ///< Continuous primary camera counter across batches.
    uint64_t m_sdkSequence{ 0 }; ///< Ordered successful native observations, independent of combined counter.

    /// Enqueue one complete native or classification record through the application's actual capture handoff.
    void tagged( const MagAOX::dmTimingEvents::Record &record /**< [in] Immutable diagnostic observation. */ );
};

BatchHarness::BatchHarness( int64_t baseNs ) : m_baseNs( baseNs )
{
    m_syncoDevice = "batchSynco";
    m_capacity    = 32768;
    m_queue.configure( m_capacity );
    m_producer.configure( m_capacity );
    m_synco.configure( m_capacity, 2000000000 );
    REQUIRE( m_monitor.configure( 3600, 2, m_capacity ) );
    enable();
    REQUIRE( source( m_syncoDevice, "enable", "On" ) == 0 );
    consume( m_baseNs );
}

void BatchHarness::tagged( const MagAOX::dmTimingEvents::Record &record )
{
    Observation event;
    event.tagged      = true;
    event.event       = record;
    event.incarnation = record.kind == MagAOX::dmTimingEvents::Kind::sdk ? 10 : 20;
    event.gate        = m_gateGeneration.load();
    event.epoch       = m_captureEpoch.load();
    REQUIRE( m_queue.push( event ) );
}

void BatchHarness::captureBatch( Loss loss )
{
    for( uint64_t index = 1; index <= 2000; ++index )
    {
        ++m_frame;
        const auto stamp      = m_baseNs + int64_t( m_frame - 1 ) * 500000;
        const auto generation = m_producer.synchronize( true, 4, 1 );
        const bool missing    = index == 1000;

        if( !missing || loss != Loss::camera )
        {
            enqueue( true, m_frame, stamp );
        }

        for( uint64_t step = 0; step < 4; ++step )
        {
            MagAOX::dmTimingEvents::Record record;
            const auto                     counter = 4 * ( m_frame - 1 ) + step + 1;
            record.key                             = { 42, counter, 10, int64_t( counter ) };
            record.startNs                         = stamp + 166000 + int64_t( step ) * 125000;

            if( !missing || loss != Loss::sdk || step != 0 )
            {
                record.sequence = ++m_sdkSequence;
                tagged( record );
            }

            if( step == 0 && ( !missing || loss != Loss::label ) )
            {
                m_producer.capture( record.key, 7, m_frame, generation );
            }
        }
    }
}

void BatchHarness::publishLabels()
{
    MagAOX::dmTimingEvents::Record record;

    while( m_producer.pop( record ) )
    {
        tagged( record );
    }
}

void BatchHarness::batch( Loss loss )
{
    captureBatch( loss );
    publishLabels();
    consume( now() );
}

int64_t BatchHarness::now() const
{
    return m_baseNs + int64_t( m_frame ) * 500000 + 1000000;
}

uint64_t BatchHarness::producerGeneration() const
{
    return m_producer.generation();
}

uint64_t BatchHarness::unmatched() const
{
    return m_synco.quality().unmatched;
}

size_t BatchHarness::pendingCameras() const
{
    return m_pendingCameras.size();
}

uint64_t BatchHarness::pendingOverflow() const
{
    return m_pendingCameraOverflow;
}
/// \endcond

/** \brief Verify default and configurable limits reject nonfinite/invalid values under production compiler flags.
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

    SECTION( "automatic mode requires all source names" )
    {
        app.m_syncoDevice = "synco";
        CHECK( app.validate() < 0 );
        app.m_mainStream = "mainEvents";
        CHECK( app.validate() < 0 );
        app.m_sdkEventStream = "sdkEvents";
        CHECK( app.validate() == 0 );
        app.m_sdkEventStream = app.m_mainStream;
        CHECK( app.validate() < 0 );
    }

    SECTION( "event streams require a synco device" )
    {
        app.m_mainStream     = "mainEvents";
        app.m_sdkEventStream = "sdkEvents";
        CHECK( app.validate() < 0 );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::loadConfigImpl;
#endif
    // clang-format on
}

/** \brief Verify all prerequisites are observed and configured names route exact properties.
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

/** \brief Retain unchanged subscriptions while independently expiring old timing samples.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing retains subscribed states without refresh traffic", "[loopTimingMonitor]" )
{
    Harness app;
    app.enable();
    const auto generation = app.m_gateGeneration.load();
    const auto now        = app.monotonicNow();
    app.population();
    app.consume( now );
    REQUIRE( app.m_monitor.summary().latency.count > 0 );

    const auto later = now + 3600000000000;
    app.consume( later );
    CHECK( app.m_gateEnabled.load() );
    CHECK( app.m_gateGeneration.load() == generation );
    CHECK( app.m_monitor.summary().latency.count == 0 );

    app.population( later + 10000000 );
    app.consume( later + 20000000 );
    CHECK( app.m_gateEnabled.load() );
    CHECK( app.m_gateGeneration.load() == generation );
    CHECK( app.m_monitor.summary().latency.count > 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::setCallBack_source;
    MagAOX::app::loopTimingMonitor::consume;
#endif
    // clang-format on
}

/** \brief Verify gate/attachment epochs reject observations already queued before a transition.
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

/** \brief Verify exactly the existing persistent fgtimes fields represent the three documented populations.
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

/** \brief One-second classification bursts retain valid post-loss camera segments at two/four-step kilohertz rates.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing delayed main batches survive classification SDK and camera gaps", "[loopTimingMonitor][batch]" )
{
    BatchHarness app( 100000000000 );
    auto         loss = BatchHarness::Loss::none;

    SECTION( "continuous two kHz camera and eight kHz native submissions" )
    {
        loss = BatchHarness::Loss::none;
    }

    SECTION( "a missing successful main SDK record in every burst" )
    {
        loss = BatchHarness::Loss::sdk;
    }

    SECTION( "real producer generation resets after a missing classification in every burst" )
    {
        loss = BatchHarness::Loss::label;
    }

    SECTION( "one missing camera observation in every burst" )
    {
        loss = BatchHarness::Loss::camera;
    }

    for( unsigned second = 1; second <= 3; ++second )
    {
        app.batch( loss );
        const auto &summary = app.m_monitor.summary();
        REQUIRE( summary.cameraIntervals.valid );
        REQUIRE( summary.sdkIntervals.valid );
        REQUIRE( summary.latency.valid );
        CHECK( summary.latency.count > 900 );
        CHECK( summary.cameraIntervals.min == Approx( 500e-6 ) );
        CHECK( summary.cameraIntervals.max == Approx( 500e-6 ) );
        CHECK( summary.sdkIntervals.min == Approx( 500e-6 ) );
        CHECK( summary.sdkIntervals.max == Approx( 500e-6 ) );
        CHECK( summary.latency.min == Approx( 166e-6 ) );
        CHECK( summary.latency.max == Approx( 166e-6 ) );
        CHECK( summary.latency.stddev == Approx( 0 ).margin( 1e-15 ) );

        if( loss == BatchHarness::Loss::label )
        {
            CHECK( app.producerGeneration() == second + 1 );
        }
        else
        {
            CHECK( app.producerGeneration() == 1 );
        }

        CHECK( app.unmatched() == ( loss == BatchHarness::Loss::sdk ? second : 0 ) );
    }

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::consume;
    MagAOX::app::loopTimingMonitor::consumeMain;
    MagAOX::app::synco::MainTiming::synchronize;
    MagAOX::app::synco::MainTiming::capture;
    MagAOX::app::synco::MainTiming::pop;
#endif
    // clang-format on
}

/** \brief Deferred camera evidence still expires and cannot cross a real gate or source invalidation.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing pending cameras obey hard invalidation and retention bounds", "[loopTimingMonitor][batch]" )
{
    BatchHarness app( 100000000000 );
    app.captureBatch( BatchHarness::Loss::none );
    app.consume( app.now() );
    REQUIRE( app.pendingCameras() == 2000 );
    REQUIRE( app.m_monitor.summary().latency.count == 0 );

    SECTION( "camera source incarnation or clock epoch changes" )
    {
        ++app.m_captureEpoch;
        app.consume( app.now() );
    }

    SECTION( "loop gate turns off and back on" )
    {
        REQUIRE( app.source( app.m_loopDevice, "loop_state", "Off" ) == 0 );
        REQUIRE( app.source( app.m_loopDevice, "loop_state", "On" ) == 0 );
        app.consume( app.now() );
    }

    SECTION( "subscribed synco mode changes and returns to the same value" )
    {
        REQUIRE( app.source( app.m_syncoDevice, "enable", "Off" ) == 0 );
        REQUIRE( app.source( app.m_syncoDevice, "enable", "On" ) == 0 );
        app.consume( app.now() );
    }

    SECTION( "classification stops beyond the statistics window" )
    {
        app.consume( app.now() + 2000000000 );
    }

    CHECK( app.pendingCameras() == 0 );
    CHECK( app.m_monitor.summary().cameraIntervals.count == 0 );
    CHECK( app.m_monitor.summary().sdkIntervals.count == 0 );
    CHECK( app.m_monitor.summary().latency.count == 0 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::consume;
    MagAOX::app::loopTimingMonitor::setCallBack_source;
#endif
    // clang-format on
}

/** \brief Pending-camera overflow counts each evicted observation and retains a usable bounded suffix.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing recovers from bounded pending camera history", "[loopTimingMonitor][batch]" )
{
    BatchHarness app( 100000000000 );
    app.m_capacity = 64;
    app.captureBatch( BatchHarness::Loss::none );
    app.consume( app.now() );
    CHECK( app.pendingCameras() == 64 );
    CHECK( app.pendingOverflow() == 1936 );
    CHECK( app.m_monitor.summary().latency.count == 0 );

    app.publishLabels();
    app.consume( app.now() );
    const auto &summary = app.m_monitor.summary();
    REQUIRE( summary.latency.valid );
    CHECK( summary.latency.count == 63 );
    CHECK( summary.latency.min == Approx( 166e-6 ) );
    CHECK( summary.latency.max == Approx( 166e-6 ) );
    CHECK( summary.cameraIntervals.min == Approx( 500e-6 ) );
    CHECK( summary.cameraIntervals.max == Approx( 500e-6 ) );
    CHECK( app.pendingCameras() == 0 );
    CHECK( app.pendingOverflow() == 1936 );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::consume;
    MagAOX::app::loopTimingMonitor::consumeMain;
#endif
    // clang-format on
}

/** \brief A production-rate delayed batch with a real classification reset still exposes current fgtimes values.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing exports telemetry after a loss inside a delayed main batch", "[loopTimingMonitor][batch]" )
{
    BatchHarness app( Harness::monotonicNow() - 1010000000 );
    app.batch( BatchHarness::Loss::label );
    std::array<double, 6> values;
    REQUIRE( app.timingValues( values ) );
    CHECK( values[0] == Approx( 500e-6 ) );
    CHECK( values[1] == Approx( 0 ).margin( 1e-15 ) );
    CHECK( values[2] == Approx( 500e-6 ) );
    CHECK( values[3] == Approx( 0 ).margin( 1e-15 ) );
    CHECK( values[4] == Approx( 166e-6 ) );
    CHECK( values[5] == Approx( 0 ).margin( 1e-15 ) );

    // clang-format off
#ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::consume;
    MagAOX::app::loopTimingMonitor::timingValues;
#endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
