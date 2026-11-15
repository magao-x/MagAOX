/** \file ddspc_native_pipeline_test.cpp
 * \brief Actual native input adapter, independent DDSPC learners and private syncopation payload tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../native_pipeline.hpp"
#include "../online_shadow.hpp"
#include "offline_adc.hpp"
#include "adc_transport_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond NATIVE_PIPELINE_TEST_SUPPORT
namespace
{
namespace synco = MagAOX::app::synco;
// Instrumented replay checks data flow, not throughput. Keep nominal acquisition separate from observer receipt.
#if defined( __SANITIZE_ADDRESS__ )
constexpr std::int64_t workerDeliveryUs = 2000;
#elif defined( __has_feature )
    #if __has_feature( address_sanitizer ) || __has_feature( undefined_behavior_sanitizer )
constexpr std::int64_t workerDeliveryUs = 2000;
    #else
constexpr std::int64_t workerDeliveryUs = 500;
    #endif
#else
constexpr std::int64_t workerDeliveryUs = 500;
#endif
/// Build an original synthetic schedule, without any stream, semaphore or physical actuation.
synco::Record schedule( std::int64_t now /**< [in] Declared local scheduling clock. */ );
/// Own one randomly named private test directory, including cleanup after a failed assertion.
struct PrivateTrace
{
    std::filesystem::path m_directory; ///< Only this test's mkdtemp directory is removed.
    /// Create an empty private directory; the application's exclusive CSV file does not yet exist.
    PrivateTrace();
    /// Remove only this owned private fixture after its worker has been joined.
    ~PrivateTrace();
};
PrivateTrace::PrivateTrace()
{
    auto pattern = ( std::filesystem::temp_directory_path() / "ddspc-native-trace-XXXXXX" ).string();
    if( !mkdtemp( pattern.data() ) )
        throw std::runtime_error( "Cannot create private trace fixture" );
    m_directory = pattern;
}
PrivateTrace::~PrivateTrace()
{
    std::error_code ignored;
    std::filesystem::remove_all( m_directory, ignored );
}

/// Capture only private schedule observations; no protocol record or source is ever written.
class ScheduleSink : public DDSPC::NativeObservationSink
{
    synco::Record m_expected; ///< Immutable fixture record.
  public:
    std::atomic<unsigned> m_valid{ 0 }, m_invalid{ 0 }; ///< Safe cross-thread observations.
    std::atomic<unsigned> m_rejectedDm{ 0 };            ///< Actual incompatible-source qualification notifications.
    /// Copy the immutable expected payload before the observer starts.
    explicit ScheduleSink( const synco::Record &expected /**< [in] Private source record. */ );
    /// Discard unrelated input metadata without changing it.
    void observe( const DDSPC::NativeObservation &record /**< [in] Original input. */ ) noexcept override;
    /// Count exact coherent records versus explicit unavailable markers.
    void schedule( const synco::Record &record /**< [in] Read-only payload. */,
                   std::int64_t         receipt /**< [in] Actual observer receipt. */ ) noexcept override;
};
ScheduleSink::ScheduleSink( const synco::Record &expected ) : m_expected( expected )
{
}
void ScheduleSink::observe( const DDSPC::NativeObservation &r ) noexcept
{
    if( r.m_source == DDSPC::ObservationSource::Dm && r.m_event == DDSPC::ObservationEvent::Invalid )
        ++m_rejectedDm;
}
void ScheduleSink::schedule( const synco::Record &record, std::int64_t ) noexcept
{
    if( record == m_expected )
        ++m_valid;
    else
        ++m_invalid;
}
/// Complete causal native receiver fixture; all inputs/identities/plant responses are explicitly synthetic.
struct NativeReplay
{
    DDSPC::OnlineShadowConfig   m_config;       ///< Same actual native numerical bounds, no hardware/model replacement.
    DDSPC::NativePipelineConfig m_receiver;     ///< Explicit synthetic upstream/calibration oracle.
    DDSPC::OnlineShadowStatus   m_status;       ///< Actual observed native learner state.
    DDSPC::Matrix               m_standard;     ///< Actual final coefficients for bitwise independence checks.
    std::array<bool, 2>         m_duringLoss{}; ///< Model readiness while ADC or WFS input is absent.
    double m_meanUs{ 0 }, m_maxUs{ 0 };         ///< Entire native adapter jobs, not real-time qualification.

    /// Declare independent synthetic source and modal calibration identities.
    NativeReplay();

    /// Feed original sample records, complete request images, actual sampler ticks and producer status.
    void run( bool qrd /**< [in] Use actual QR or covariance RLS. */,
              int  fault = 0 /**< [in] ADC loss, WFS loss, idle DM, missing DM or producer timer mode. */,
              DDSPC::OnlineShadowWorker *worker = nullptr /**< [in] Optional actual paced numerical receiver. */ );
};
NativeReplay::NativeReplay()
{
    auto &m                  = m_config.m_model;
    m.m_modes                = 2;
    m.m_history              = 2;
    m.m_future               = 2;
    m.m_accelHistory         = 1;
    m.m_prior                = .001;
    m.m_lambda               = .1;
    m.m_standardCommandSlots = 4;
    auto &c                  = m_receiver;
    c.m_enabled              = true;
    c.m_sampleDriven         = false; // Historical WFS-origin replay is retained as a reference.
    c.m_syncEpoch            = 77;
    c.m_modalAnchor = c.m_adcAnchor = 1;
    c.m_statusDevice                = "synthetic_adc";
    c.m_reference                   = "synthetic_rtc_sync";
    c.m_calibration.m_layout        = 17;
    c.m_calibration.m_indices       = { 0, 1 };
    c.m_calibration.m_modalUnits    = "synthetic_modal";
    c.m_calibration.m_commandUnits  = "synthetic_request";
    c.m_calibration.m_projection    = DDSPC::Matrix::Identity( 2, 2 );
    c.m_calibration.m_shapes        = DDSPC::Matrix::Identity( 2, 2 );
}
void NativeReplay::run( bool qrd, int fault, DDSPC::OnlineShadowWorker *worker )
{
    m_config.m_model.m_useQrd = qrd;
    DDSPC::MultirateShadow     engine( m_config );
    DDSPC::NativeInputPipeline pipeline( engine, m_receiver, m_config.m_model );
    SyntheticAdcConfig         sensor;
    sensor.m_forceNoise = .1;
    SyntheticAdcSource source( sensor );
    DDSPC::AdcCapture  decoder( sensor.m_calibration );
    decoder.attached( 11, 100 );
    auto raw = []( DDSPC::ObservationSource role, std::uint64_t counter, std::int64_t available )
    {
        DDSPC::NativeObservation r;
        r.m_source     = role;
        r.m_device     = 11;
        r.m_fileInode  = 100 + unsigned( role );
        r.m_imageInode = role == DDSPC::ObservationSource::Adc ? 0 : 1000 + unsigned( role );
        r.m_generation = r.m_segment = 1;
        r.m_counter                  = counter;
        r.m_receivedNs = r.m_availableNs = available;
        r.m_elements                     = 2;
        return r;
    };
    std::mt19937                    rng( 91 );
    std::normal_distribution<float> draw( 0, .2 ), noise( 0, .003 );
    std::array<float, 8>            previousU{}, previousA{};
    std::array<float, 2>            y{};
    std::uint64_t                   commandCounter = 0;
    std::vector<float>              image( m_receiver.m_calibration.m_projection.cols(), 0 );
    const auto                      pace = std::chrono::steady_clock::now();
    timespec                        clock{};
    clock_gettime( CLOCK_MONOTONIC, &clock );
    const std::int64_t base = worker ? std::int64_t( clock.tv_sec ) * 1000000000 + clock.tv_nsec - 500000 : 1000000000;
    auto               observe = [&]( const DDSPC::NativeObservation &r, std::int64_t now )
    {
        pipeline.observe( r, now );
        if( worker && !( fault == 6 && r.m_source == DDSPC::ObservationSource::Dm && r.m_counter == 2404 ) )
            worker->observe( r );
    };
    auto command = [&]( const std::array<float, 2> &values, std::int64_t time )
    {
        auto r          = raw( DDSPC::ObservationSource::Dm, ++commandCounter, time );
        image[0]        = values[0];
        image[1]        = values[1];
        r.m_elements    = image.size();
        r.m_payloadHash = DDSPC::command_payload_hash( image );
        pipeline.command( r, image );
        if( worker )
            worker->command( r, image );
        observe( r, time );
    };
    if( fault != 4 )
        command( { 0, 0 }, base - 1000000 );
    constexpr unsigned frames = 1800;
    for( unsigned f = 0; f < frames; ++f )
    {
        if( worker )
            std::this_thread::sleep_until( pace + std::chrono::microseconds( workerDeliveryUs * f ) );
        const auto         started = std::chrono::steady_clock::now();
        const std::int64_t t       = base + ( worker ? workerDeliveryUs * 1000 : 500000LL ) * f;
        if( fault == 7 && f == 600 )
        {
            auto invalid    = raw( DDSPC::ObservationSource::Dm, 0, t + 98000 );
            invalid.m_event = DDSPC::ObservationEvent::Invalid;
            invalid.m_flags = DDSPC::ObservationInvalidPayload;
            observe( invalid, t + 98000 );
        }
        std::array<float, 8> a{}, u{};
        for( unsigned s = 0; s < 4; ++s )
        {
            source.advance();
            decoder.drain( source );
            const auto sample  = decoder.status().m_latest;
            auto       r       = raw( DDSPC::ObservationSource::Adc, sample.m_frame.m_counter, t + 99000 );
            r.m_receivedNs     = t - 500000 + 125000 * s + 1000;
            r.m_raw            = sample.m_frame.m_raw;
            r.m_values         = sample.m_acceleration;
            r.m_acquiredRawNs  = sample.m_frame.m_acquiredRealNs;
            r.m_publishedRawNs = sample.m_frame.m_writtenRealNs;
            a[2 * s]           = r.m_values[0];
            a[2 * s + 1]       = r.m_values[1];
            if( !( fault == 1 && f >= 600 && f < 650 ) )
                observe( r, t + 99000 );
        }
        for( unsigned m = 0; m < 2; ++m )
        {
            y[m] = .4f * y[m] + noise( rng );
            for( unsigned s = 0; s < 4; ++s )
                y[m] += .2f * previousA[2 * s + m] + ( .1f + .05f * s ) * previousU[2 * s + m];
        }
        const DDSPC::NativeSyncStatus sync{
            { 77, fault != 5, !( fault == 1 && f >= 600 && f < 650 ), t + 99000 }, 4 * ( f + 1 ), f + 1, 0 };
        pipeline.sync( sync );
        if( worker )
            worker->sync_status( sync );
        if( m_receiver.m_syncoAssociation )
        {
            auto s          = schedule( t );
            s[synco::cycle] = f;
            pipeline.schedule( s, t + 150000, t + 150000 ); // Schedule can arrive after the frozen measurement cutoff.
        }
        if( !( fault == 2 && f >= 600 && f < 610 ) )
        {
            auto w            = raw( DDSPC::ObservationSource::Modal, f + 1, t + 100000 );
            w.m_values        = { y[0], y[1] };
            w.m_acquiredRawNs = 1000000000LL + 500000LL * f; // Nominal source cadence, independent of replay delivery.
            const auto processed = t + ( m_receiver.m_syncoAssociation ? 200000 : 100000 );
            observe( w, processed );
            engine.finish_origin( processed + 1000 ); // Explicit virtual completion, not workstation WCET.
        }
        if( m_receiver.m_syncoAssociation )
        {
            DDSPC::OnlineShadowStatus state = engine.status();
            pipeline.status( state );
            auto s          = schedule( t );
            s[synco::cycle] = f;
            pipeline.producer()->callback( s, t + 300000 );
        }
        for( unsigned s = 0; s < 4; ++s )
        {
            for( unsigned m = 0; m < 2; ++m )
                u[2 * s + m] = fault == 3 ? 0.f : std::clamp( draw( rng ), -.5f, .5f );
            if( fault != 3 && fault != 4 && !( fault == 7 && f >= 600 && f < 620 ) )
                command( { u[2 * s], u[2 * s + 1] }, t + 101000 + s * 125000 );
            else if( fault == 7 )
                ++commandCounter; // Original publications were rejected, not held or fabricated.
            pipeline.tick( t + 105000 + s * 125000 );
            if( worker )
                worker->tick( t + 105000 + s * 125000 );
        }
        if( f == 605 )
        {
            const auto s = engine.status();
            m_duringLoss = { s.m_standard.m_ready, s.m_assisted.m_ready };
        }
        previousU = u;
        previousA = a;
        const double us =
            std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - started ).count();
        m_meanUs += us / frames;
        m_maxUs = std::max( m_maxUs, us );
    }
    m_status = engine.status();
    pipeline.status( m_status );
    m_standard = engine.coefficients( DDSPC::ShadowMode::Standard );
    REQUIRE( decoder.status().m_samples == 4 * frames );
    REQUIRE( decoder.status().m_dropped == 0 );
}
/// Construct an original version-2 schedule with four explicit 125-us targets.
synco::Record schedule( std::int64_t now )
{
    synco::Record s{};
    s[synco::version] = synco::productionVersion;
    s[synco::epoch] = s[synco::generation] = 1;
    s[synco::state]                        = synco::armed;
    s[synco::steps]                        = 4;
    s[synco::period]                       = 500000;
    s[synco::reference]                    = now;
    s[synco::expires]                      = now + 1000000;
    s[synco::cycle]                        = 19;
    for( int p = 0; p < 4; ++p )
    {
        s[synco::targets + p] = now + 500000 + 125000 * p;
        s[synco::cutoffs + p] = s[synco::targets + p] - 70000;
    }
    return s;
}
} //namespace
/// \endcond

/** \brief Both native receiver models train and predict from actual packets without exposure/clock reconstruction. */
TEST_CASE( "Native observer packet adapter trains both native DDSPC learners", "[multirate][native-pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::NativeInputPipeline::observe;
    DDSPC::NativeInputPipeline::command;
    DDSPC::NativeInputPipeline::tick;
    DDSPC::NativeInputPipeline::sync;
#endif
    // clang-format on
    NativeReplay r;
    r.run( GENERATE( false, true ) );
    INFO( r.m_status.m_reason );
    REQUIRE( r.m_status.m_records > 1700 );
    REQUIRE( r.m_status.m_standard.m_ready );
    REQUIRE( r.m_status.m_assisted.m_ready );
    REQUIRE( r.m_status.m_standard.m_originPredictions > 1000 );
    REQUIRE( r.m_status.m_assisted.m_originPredictions > 1000 );
    REQUIRE( r.m_status.m_inputEvidence == DDSPC::InputEvidence::Requested );
    REQUIRE_FALSE( r.m_status.m_candidateValid );
    REQUIRE_FALSE( r.m_status.m_exposure );
    REQUIRE_FALSE( r.m_status.m_clock );
    std::cout << "native receiver adapter frame mean_us=" << r.m_meanUs << " max_us=" << r.m_maxUs << '\n';
}

/** \brief ADC loss preserves standard learning; common input loss forces requalification before recovery. */
TEST_CASE( "Native packet loss and idle commands never fabricate readiness", "[multirate][native-pipeline]" )
{
    NativeReplay r;
    const int    fault = GENERATE( 1, 2, 3, 4, 5, 7 );
    CAPTURE( fault );
    r.run( true, fault );
    if( fault == 1 || fault == 2 || fault == 7 )
    {
        REQUIRE( r.m_status.m_standard.m_ready );
        REQUIRE( r.m_status.m_assisted.m_ready );
        REQUIRE_FALSE( r.m_duringLoss[1] );
        if( fault == 7 )
            REQUIRE_FALSE( r.m_duringLoss[0] );
        if( fault == 1 )
        {
            REQUIRE( r.m_duringLoss[0] );
            NativeReplay reference;
            reference.run( true );
            REQUIRE( r.m_standard == reference.m_standard );
        }
    }
    else if( fault == 5 )
    {
        REQUIRE( r.m_status.m_standard.m_ready );
        REQUIRE_FALSE( r.m_status.m_assisted.m_ready );
    }
    else
    {
        REQUIRE_FALSE( r.m_status.m_standard.m_ready );
        REQUIRE_FALSE( r.m_status.m_assisted.m_ready );
        if( fault == 3 )
            REQUIRE( r.m_status.m_records > 1700 ); // Static input transport is healthy, but unexcited.
        if( fault == 4 )
            REQUIRE( r.m_status.m_records == 0 );
    }
}

/** \brief Newest-first native DDSPC rows map to four distinct chronological Request targets, never step ordering. */
TEST_CASE( "Prepared syncopation payloads preserve exact four-slot order and reject late or changed requests",
           "[multirate][synco-shadow]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::SyncoShadowPlans::prepare;
    DDSPC::SyncoShadowPlans::command;
#endif
    // clang-format on
    constexpr std::int64_t  now = 1000000000;
    auto                    s   = schedule( now );
    DDSPC::SyncoShadowPlans p( 2, 1 );
    DDSPC::Matrix           plan( 8, 1 );
    plan << 30, 31, 20, 21, 10, 11, 0, 1;
    REQUIRE( p.prepare( plan, DDSPC::Matrix::Identity( 2, 2 ), s, now + 500000, now + 1000 ) );
    for( int phase = 0; phase < 4; ++phase )
    {
        std::array<float, 2> values{ -1, -1 };
        const auto           request = synco::request( s, phase );
        REQUIRE( p.command( request, s, now + 100000, values ) );
        REQUIRE( values[0] == 10 * phase );
        REQUIRE( values[1] == 10 * phase + 1 );
        REQUIRE_FALSE( p.command( request, s, request.cutoffNs, values ) );
        auto bad = request;
        ++bad.generation;
        REQUIRE_FALSE( p.command( bad, s, now + 100000, values ) );
    }
    std::array<float, 2> values{};
    ++s[synco::generation];
    REQUIRE_FALSE( p.command( synco::request( s, 0 ), s, now + 100000, values ) );
    s         = schedule( now );
    plan( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE( p.prepare( plan, DDSPC::Matrix::Identity( 2, 2 ), s, now + 500000, now + 1000 ) );
    REQUIRE_FALSE( p.command( synco::request( s, 0 ), s, now + 100000, values ) );
}

/** \brief The callback copies precomputed images, falls back by readiness, and never reaches an output API. */
TEST_CASE( "Private native producer callback never publishes or reuses an expired slot", "[multirate][synco-shadow]" )
{
    constexpr std::int64_t     now = 1000000000;
    const auto                 s   = schedule( now );
    DDSPC::SyncoShadowProducer p( 2, 1 );
    const auto                 plan = DDSPC::Matrix::Ones( 8, 1 ).eval();
    REQUIRE(
        p.prepare( DDSPC::ShadowMode::Standard, plan, DDSPC::Matrix::Identity( 2, 2 ), s, now + 500000, now + 1000 ) );
    p.models( DDSPC::ShadowMode::AccelAssisted, true, false );
    REQUIRE( p.callback( s, now + 100000 ) );       // Ready standard fallback, not a fabricated assisted plan.
    REQUIRE_FALSE( p.callback( s, now + 100001 ) ); // The base suppresses duplicate requests too.
    p.inhibit( false );
    REQUIRE_FALSE( p.callback( s, s[synco::cutoffs] + 1 ) );
    DDSPC::OnlineShadowStatus status;
    p.status( status );
    REQUIRE( status.m_shadowSlots == 1 );
    REQUIRE( status.m_missedSlots == 1 );
    REQUIRE( status.m_callbackPeakUs >= 0 );
    auto corrupt          = s;
    corrupt[synco::cycle] = std::numeric_limits<std::int64_t>::max();
    REQUIRE_FALSE(
        p.callback( corrupt, now + 100000 ) ); // Never overflow the unchanged protocol's step-zero cycle label.
}

/** \brief The actual bounded numerical worker and independent recorder train from native packets without a stream. */
TEST_CASE( "Native online worker trains and records origin-only forecasts from qualified synthetic telemetry",
           "[multirate][native-pipeline]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::OnlineShadowWorker::native_inputs;
    DDSPC::OnlineShadowWorker::observe;
    DDSPC::OnlineShadowWorker::command;
    DDSPC::OnlineShadowWorker::sync_status;
    DDSPC::OnlineShadowWorker::tick;
#endif
    // clang-format on
    NativeReplay replay;
    auto         config         = replay.m_config;
    config.m_model.m_useQrd     = true;
    config.m_periodUs           = 20000;
    config.m_budgetUs           = 10000; // Conservative batch test, not workstation real-time qualification.
    config.m_predictionCapacity = 32768;
    const bool limited          = GENERATE( false, true );
    if( limited )
        config.m_predictionMaxRows = 32;
    PrivateTrace temporary;
    const auto   path       = temporary.m_directory / "predictions.csv";
    config.m_predictionFile = path.string();
    DDSPC::OnlineShadowWorker worker( config );
    worker.native_inputs( replay.m_receiver );
    worker.start();
    worker.request( DDSPC::ShadowMode::AccelAssisted );
    replay.run( true, limited ? 6 : 0, &worker ); // Lost metadata must not shift all later full-payload associations.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( worker.status().m_records < ( limited ? 1798u : 1799u ) && std::chrono::steady_clock::now() < deadline )
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    const auto status = worker.status();
    worker.stop();
    INFO( status.m_reason );
    INFO( status.m_adapterReason );
    INFO( status.m_peakUs );
    REQUIRE_FALSE( status.m_overrun );
    REQUIRE( status.m_drops == 0 );
    REQUIRE( status.m_records == ( limited ? 1798u : 1799u ) );
    REQUIRE( status.m_standard.m_ready );
    REQUIRE( status.m_assisted.m_ready );
    REQUIRE( status.m_standard.m_originPredictions > 1000 );
    REQUIRE( status.m_assisted.m_originPredictions > 1000 );
    const auto expectedRows = 4 * ( status.m_standard.m_originPredictions + status.m_assisted.m_originPredictions );
    REQUIRE( expectedRows > 4000 );
    REQUIRE( worker.status().m_traceRows == ( limited ? 32 : expectedRows ) );
    REQUIRE( worker.status().m_predictionDrops == 0 );
    REQUIRE_FALSE( worker.status().m_traceIoError );
    REQUIRE( worker.status().m_traceLimit == limited );
    REQUIRE_FALSE( worker.status().m_candidateValid );
    REQUIRE_THROWS( worker.start() ); // The exclusive recorder never truncates an existing recording.
    std::ifstream file( path );
    std::string   header;
    std::getline( file, header );
    REQUIRE( header.find( "origin_prediction,conditional_prediction" ) != std::string::npos );
    file.close();
    std::filesystem::remove( path );
    std::cout << "native online model_job_us=" << status.m_computeUs << " peak_us=" << status.m_peakUs
              << " records=" << status.m_records << " delivery_period_us=" << workerDeliveryUs
              << " nominal_adc_hz=8000 nominal_wfs_hz=2000 deadline_qualified=false\n";
}

/** \brief Callback timing excludes rendering/RLS and never implies a real-time deadline guarantee. */
TEST_CASE( "Prepared full-image producer callback has bounded copy-only work", "[multirate][synco-shadow]" )
{
    DDSPC::SyncoShadowProducer producer( 2500, 1 );
    DDSPC::Matrix              shapes = DDSPC::Matrix::Zero( 2500, 2 ), plan = DDSPC::Matrix::Ones( 8, 1 );
    shapes( 0, 0 ) = shapes( 1, 1 ) = 1;
    std::vector<double> times;
    times.reserve( 1000 );
    for( unsigned n = 0; n < 1000; ++n )
    {
        const auto now  = 1000000000LL + n * 500000LL;
        auto       s    = schedule( now );
        s[synco::cycle] = n;
        REQUIRE( producer.prepare( DDSPC::ShadowMode::Standard, plan, shapes, s, now + 500000, now + 1000 ) );
        producer.models( DDSPC::ShadowMode::Standard, true, false );
        producer.callback( s, now + 100000 );
        DDSPC::OnlineShadowStatus status;
        producer.status( status );
        times.push_back( status.m_callbackUs );
    }
    std::sort( times.begin(), times.end() );
    std::cout << "private producer callback p50_us=" << times[500] << " p99_us=" << times[990]
              << " max_us=" << times.back() << " pixels=2500 publications=0 deadline_qualified=false\n";
}

/** \brief Native projection timings include the full tweeter-sized payload, without using two pixels as calibration. */
TEST_CASE( "Native full-image projection retains bounded training work", "[multirate][native-pipeline]" )
{
    NativeReplay r;
    auto        &c         = r.m_receiver.m_calibration;
    c.m_projection         = DDSPC::Matrix::Zero( 2, 2500 );
    c.m_projection( 0, 0 ) = c.m_projection( 1, 1 ) = 1;
    c.m_shapes                                      = DDSPC::Matrix::Zero( 2500, 2 );
    c.m_shapes( 0, 0 )                              = c.m_shapes( 1, 1 ) =
        1; // Explicit synthetic calibration; remaining channels are actually zero.
    r.run( true );
    REQUIRE( r.m_status.m_standard.m_ready );
    REQUIRE( r.m_status.m_assisted.m_ready );
    std::cout << "full-image native receiver mean_us=" << r.m_meanUs << " max_us=" << r.m_maxUs
              << " dm_pixels=2500 adc_hz=8000 wfs_hz=2000 deadline_qualified=false\n";
}

/** \brief Real scheduling step numbers are not chronological indices; step zero belongs to the next main cycle. */
TEST_CASE( "Native producer uses target times rather than synco step ordering", "[multirate][synco-shadow]" )
{
    constexpr std::int64_t now = 1000000000;
    auto                   s   = schedule( now );
    for( int phase = 1; phase < 4; ++phase )
    {
        s[synco::targets + phase] = now + phase * 125000;
        s[synco::cutoffs + phase] = s[synco::targets + phase] - 70000;
    }
    DDSPC::SyncoShadowPlans p( 1, 2 );
    DDSPC::Matrix           plan( 8, 1 );
    plan << 7, 6, 5, 4, 3, 2, 1, 0;
    REQUIRE( p.prepare( plan, DDSPC::Matrix::Ones( 1, 1 ), s, now, now + 1000 ) );
    std::array<float, 1> value{};
    for( int phase = 0; phase < 4; ++phase )
    {
        const auto request = synco::request( s, phase );
        REQUIRE( request.cycle == ( phase ? 19 : 20 ) );
        REQUIRE( p.command( request, s, now + 10000, value ) );
        REQUIRE( value[0] == ( phase ? phase : 4 ) );
    }
}

/** \brief An earlier queued model origin cannot rearm private commands after a subsequently observed input loss. */
TEST_CASE( "Native producer loss barrier rejects queued pre-loss preparations", "[multirate][synco-shadow]" )
{
    constexpr std::int64_t     now = 1000000000;
    const auto                 s   = schedule( now );
    DDSPC::SyncoShadowProducer p( 2, 1 );
    DDSPC::Matrix              plan = DDSPC::Matrix::Ones( 8, 1 ), shapes = DDSPC::Matrix::Identity( 2, 2 );
    p.inhibit( false, now + 20000 );
    REQUIRE_FALSE( p.prepare( DDSPC::ShadowMode::Standard, plan, shapes, s, now + 500000, now + 40000, now + 10000 ) );
    p.models( DDSPC::ShadowMode::Standard, true, false );
    REQUIRE_FALSE( p.callback( s, now + 50000 ) );
    REQUIRE( p.prepare( DDSPC::ShadowMode::Standard, plan, shapes, s, now + 500000, now + 60000, now + 30000 ) );
    auto next = s;
    next[synco::cycle]++;
    REQUIRE( p.callback( next, now + 70000 ) );
}

/** \brief Calibration loading checks explicit units/layout/mode indices, exact extent, finite values and inverse
 * shapes. */
TEST_CASE( "Native modal calibration is explicit and never replaced with an inferred identity",
           "[multirate][native-pipeline]" )
{
    REQUIRE( DDSPC::load_native_calibration( "" ).m_layout == 0 );
    const auto path = std::filesystem::temp_directory_path() / ( "ddspc-cal-" + std::to_string( getpid() ) + ".txt" );
    const auto data = GENERATE( std::string( "DDSPC_MODAL_V1 17 2 2 0 1 1 modal request\n1 0 0 1\n1 0 0 1\n" ),
                                std::string( "DDSPC_MODAL_V1 17 2 2 0 1 1 modal request\n1 0 0 1\n0 0 0 0\n" ),
                                std::string( "DDSPC_MODAL_V1 0 2 2 0 1 0 modal request\n1 0 0 1\n" ),
                                std::string( "DDSPC_MODAL_V1 17 2 2 0 1 0 modal request\n1 0 0\n" ) );
    {
        std::ofstream f( path );
        f << data;
    }
    if( data.find( "1 0 0 1\n1 0 0 1" ) != std::string::npos )
    {
        const auto cal = DDSPC::load_native_calibration( path.string() );
        REQUIRE( cal.m_layout == 17 );
        REQUIRE( cal.m_projection == DDSPC::Matrix::Identity( 2, 2 ) );
    }
    else
        REQUIRE_THROWS( DDSPC::load_native_calibration( path.string() ) );
    std::filesystem::remove( path );
}

/** \brief Native RLS/QR plans reach the same private Request seam used after a future base poll. */
TEST_CASE( "Native fitted plans reach the syncopation preparation seam without changing model identification",
           "[multirate][native-pipeline][synco-shadow]" )
{
    NativeReplay r, reference;
    r.m_receiver.m_syncoAssociation = true;
    r.run( true );
    reference.run( true );
    REQUIRE( r.m_status.m_standard.m_ready );
    REQUIRE( r.m_status.m_assisted.m_ready );
    REQUIRE( r.m_status.m_preparedPlans > 1000 );
    REQUIRE( r.m_status.m_shadowSlots > 500 );
    REQUIRE( r.m_standard == reference.m_standard );
    REQUIRE_FALSE( r.m_status.m_candidateValid ); // Hypothetical payloads never become an actuation-qualified plan.
}

/** \brief Replanning does not discard still-pending phases preceding the newest plan's first target. */
TEST_CASE( "Prepared producer retains pending preceding slots and supplies exact Request payloads",
           "[multirate][synco-shadow]" )
{
    constexpr std::int64_t     now   = 1000000000;
    const auto                 first = schedule( now );
    auto                       later = schedule( now + 500000 );
    DDSPC::SyncoShadowProducer p( 1, 2 );
    DDSPC::Matrix              shapes = DDSPC::Matrix::Ones( 1, 1 ), old( 8, 1 ), next( 8, 1 );
    old << 7, 6, 5, 4, 3, 2, 1, 0;
    next << 17, 16, 15, 14, 13, 12, 11, 10;
    REQUIRE( p.prepare( DDSPC::ShadowMode::Standard, old, shapes, first, now + 500000, now + 1000 ) );
    REQUIRE( p.prepare( DDSPC::ShadowMode::Standard, next, shapes, later, now + 1000000, now + 600000 ) );
    p.models( DDSPC::ShadowMode::Standard, true, false );
    auto current              = first;
    current[synco::reference] = now + 500000;
    current[synco::expires]   = now + 1500000;
    std::array<float, 1> pixels{};
    REQUIRE( p.command( synco::request( current, 2 ), current, now + 610000, pixels ) );
    REQUIRE( pixels[0] == 2 ); // The newer origin starts at 1 ms; the 0.75-ms phase still belongs to the prior plan.
    REQUIRE( p.command( synco::request( later, 0 ), later, now + 610000, pixels ) );
    REQUIRE( pixels[0] == 10 );
    REQUIRE( p.prepare( DDSPC::ShadowMode::AccelAssisted, next, shapes, later, now + 1000000, now + 600001 ) );
    p.models( DDSPC::ShadowMode::AccelAssisted, true, true );
    REQUIRE( p.command( synco::request( current, 2 ), current, now + 610002, pixels ) );
    REQUIRE( pixels[0] == 2 ); // A ready assisted model without this target falls back to a valid standard payload.
    p.inhibit( false, now + 620000 );
    REQUIRE_FALSE( p.command( synco::request( current, 2 ), current, now + 630000, pixels ) );
}
/** \brief Schedule observation handles busy/missing/replaced streams without producer or semaphore writes. */
TEST_CASE( "Native schedule observation is read only and recovers without fabricating busy resets",
           "[multirate][native-pipeline][synco-shadow]" )
{
    const int      state  = GENERATE( 0, 1, 2 ); // Coherent, temporarily writing, absent.
    auto           record = schedule( 1000000000 );
    IMAGE_METADATA md{};
    md.naxis   = 1;
    md.size[0] = synco::words;
    md.size[1] = md.size[2] = 1;
    md.datatype             = _DATATYPE_INT64;
    md.nelement             = synco::words;
    md.imdatamemsize        = sizeof( record );
    md.cnt0                 = 9;
    md.write                = state == 1;
    IMAGE image{};
    image.md        = &md;
    image.array.raw = record.data();
    AdcTransportFixture                  adc( "schedule-adc-fixture", 16 );
    std::unique_ptr<AdcTransportFixture> source;
    if( state != 2 )
        source = std::make_unique<AdcTransportFixture>( "schedule-timing-fixture", image );
    DDSPC::AdcConfig adcConfig;
    adcConfig.m_stream = "schedule-adc-fixture";
    DDSPC::AdcReader               reader( adcConfig );
    DDSPC::NativeObservationConfig config;
    config.m_record            = false;
    config.m_durationSec       = 0;
    config.m_streams[0].m_name = "schedule-adc-fixture";
    config.m_timingStream      = "schedule-timing-fixture";
    DDSPC::NativeObserver observer( config );
    ScheduleSink          sink( record );
    reader.start();
    observer.start( reader, &sink );
    const auto wait = []( auto &&predicate )
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds( 500 );
        while( !predicate() && std::chrono::steady_clock::now() < end )
            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        return predicate();
    };
    if( state == 0 )
    {
        CHECK( wait( [&] { return sink.m_valid.load() > 0; } ) );
        source->disconnect();
        CHECK( wait( [&] { return sink.m_invalid.load() > 0; } ) );
        const auto before = sink.m_valid.load();
        source->reconnect();
        CHECK( wait( [&] { return sink.m_valid.load() > before; } ) );
    }
    else if( state == 1 )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
        CHECK( sink.m_valid.load() == 0 );
        CHECK( sink.m_invalid.load() == 0 ); // Busy is no update, not a fabricated scheduler reset.
    }
    else
        CHECK( wait( [&] { return sink.m_invalid.load() > 0; } ) );
    observer.stop();
    reader.stop();
    REQUIRE( md.cnt0 == 9 );
    REQUIRE( md.write == ( state == 1 ) );
    REQUIRE( md.sem == 1 );
    REQUIRE( record == schedule( 1000000000 ) );
    REQUIRE( ddspc_test_output_calls() == 0 );
}
/** \brief Concurrent preparation and copies never expose a partially rendered full image. */
TEST_CASE( "Private producer cache remains coherent across concurrent preparation and readout",
           "[multirate][synco-shadow]" )
{
    constexpr std::int64_t     now = 1000000000;
    const auto                 s   = schedule( now );
    DDSPC::SyncoShadowProducer producer( 256, 1 );
    DDSPC::Matrix              shapes = DDSPC::Matrix::Ones( 256, 1 ), initial = DDSPC::Matrix::Ones( 4, 1 );
    REQUIRE( producer.prepare( DDSPC::ShadowMode::Standard, initial, shapes, s, now + 500000, now + 1000 ) );
    producer.models( DDSPC::ShadowMode::Standard, true, false );
    std::atomic<bool>     coherent{ true };
    std::atomic<unsigned> copies{ 0 };
    std::thread           reader(
        [&]
        {
            std::array<float, 256> pixels{};
            for( unsigned n = 0; n < 3000; ++n )
                if( producer.command( synco::request( s, 0 ), s, now + 400000, pixels ) )
                {
                    ++copies;
                    const auto first = pixels[0];
                    if( ( first != 1 && first != 2 ) ||
                        !std::all_of( pixels.begin(), pixels.end(), [first]( float v ) { return v == first; } ) )
                        coherent.store( false );
                }
        } );
    for( unsigned n = 0; n < 1000; ++n )
    {
        initial.setConstant( n % 2 ? 1 : 2 );
        producer.prepare( DDSPC::ShadowMode::Standard, initial, shapes, s, now + 500000, now + 1000 );
    }
    reader.join();
    REQUIRE( coherent.load() );
    REQUIRE( copies.load() > 0 );
    REQUIRE( ddspc_test_output_calls() == 0 );
}
/** \brief Incompatible DM data is forwarded as unavailable, never retained as a qualified idle input. */
TEST_CASE( "Native observer forwards incompatible command input to the qualification sink",
           "[multirate][native-pipeline]" )
{
    IMAGE_METADATA md{};
    md.naxis   = 2;
    md.size[0] = 2;
    md.size[1] = md.size[2] = 1;
    md.datatype             = _DATATYPE_UINT16;
    md.cnt0                 = 1;
    std::array<std::uint16_t, 2> data{ 2048, 2048 };
    IMAGE                        image{};
    image.md        = &md;
    image.array.raw = data.data();
    AdcTransportFixture adc( "invalid-command-adc", 16 ), dm( "invalid-command-dm", image );
    DDSPC::AdcConfig    ac;
    ac.m_stream = "invalid-command-adc";
    DDSPC::AdcReader               reader( ac );
    DDSPC::NativeObservationConfig c;
    c.m_record            = false;
    c.m_durationSec       = 0;
    c.m_streams[0].m_name = ac.m_stream;
    c.m_streams[3].m_name = "invalid-command-dm";
    DDSPC::NativeObserver observer( c );
    ScheduleSink          sink( {} );
    reader.start();
    observer.start( reader, &sink );
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds( 500 );
    while( !sink.m_rejectedDm.load() && std::chrono::steady_clock::now() < until )
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    observer.stop();
    reader.stop();
    REQUIRE( sink.m_rejectedDm.load() == 1 );
    REQUIRE( observer.status().m_sources[3].m_incompatible > 0 );
    REQUIRE( md.cnt0 == 1 );
    REQUIRE( data == std::array<std::uint16_t, 2>{ 2048, 2048 } );
    REQUIRE( ddspc_test_output_calls() == 0 );
}
} // namespace libXWCTest::loPredCtrlAccTest
