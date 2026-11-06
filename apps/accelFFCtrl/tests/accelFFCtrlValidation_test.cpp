/** \file accelFFCtrlValidation_test.cpp
 * \brief Opt-in offline truth export using the production depth-one ingestion and prediction pipeline.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlInput.hpp"
#include "../../accelSim/accelSimModel.hpp"
#include "accelFFCtrlGap_test.hpp"

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <type_traits>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_VALIDATION_FIXTURES
namespace validation
{
using namespace MagAOX::app::accelFFCtrlDetail;
namespace sim            = MagAOX::app::accelSimDetail;
constexpr int64_t period = 125000, base = 1000000000, epoch = 1700000000000000000;

/// Return an exact synthetic realtime stamp corresponding to a monotonic acquisition.
timespec stamp( int64_t mono /**< [in] monotonic nanoseconds */ )
{
    return { time_t( ( epoch + mono ) / 1000000000 ), long( ( epoch + mono ) % 1000000000 ) };
}

/// A coherent latest-only source; publication and worker events are explicitly serialized offline.
struct source
{
    inputHeader m_header{ 2, 2, 1, 1, 0, 0, true, true, true, false }; ///< Depth-one stream identity.
    inputFrame  m_frame; ///< Latest completed publication, overwritten by the next producer event.

    /// Return current metadata by value.
    inputHeader header() const;

    /// Return the only retained plane by value.
    inputFrame frame( size_t index /**< [in] plane, necessarily zero */ ) const;
};

inputHeader source::header() const
{
    return m_header;
}

inputFrame source::frame( size_t index ) const
{
    REQUIRE( index == 0 );
    return m_frame;
}

/// Flat binary records avoid ASCII rounding and excessive artifact sizes; metadata defines every column.
template <size_t N>
void writeRow( std::ofstream               &out /**< [in,out] binary artifact */,
               const std::array<double, N> &row /**< [in] exact integer nanoseconds and float64 values */ )
{
    out.write( reinterpret_cast<const char *>( row.data() ), sizeof( row ) );
}

/// Scheduling conditions block the worker only; the source continues to generate every sample.
struct schedule
{
    const char *m_name;              ///< Stable artifact identifier.
    int64_t  m_interval, m_duration; ///< Periodic worker blackout spacing/duration in ns, starting after one interval.
    bool     m_queryOnly{ false };   ///< Negative control that incorrectly ingests at the prediction rate.
    int64_t  m_start{ 0 };           ///< First blackout; zero uses one interval, preserving the original study.
    unsigned m_cluster{ 1 };         ///< Number of blackouts separated by 1 ms at each interval boundary.
    bool     m_readerOnly{ false };  ///< Keep prediction deadlines running while ingestion alone is delayed.
    unsigned m_stampJitterPercent{ 0 }; ///< Bounded timestamp-tag error; stochastic physical truth stays on its grid.
};

/// Return elapsed ns within a periodic blackout cluster, or -1 before its first event.
int64_t clusterTime( const schedule &s /**< [in] worker schedule */, int64_t elapsed /**< [in] elapsed ns */ )
{
    const int64_t start = s.m_start ? s.m_start : s.m_interval;
    return s.m_interval && elapsed >= start ? ( elapsed - start ) % s.m_interval : -1;
}

/// Test whether a worker event is suppressed by the selected deterministic blackout.
bool blocked( const schedule &s /**< [in] worker schedule */, int64_t elapsed /**< [in] ns since first sample */ )
{
    const int64_t local = clusterTime( s, elapsed );
    return local >= 0 && ( s.m_cluster == 1 ? local < s.m_duration
                                            : local / 1000000 < s.m_cluster && local % 1000000 < s.m_duration );
}

/// Run the actual input pipeline against a pre-generated recording; no truth enters the controller.
template <typename processorT = inputProcessor>
void replay( const std::filesystem::path    &directory /**< [in] output directory */,
             const std::string              &population /**< [in] recording identifier */,
             const std::vector<sim::sample> &recording /**< [in] source ADC records, truth ignored here */,
             size_t                          count /**< [in] number of producer publications */,
             const schedule                 &s /**< [in] worker scheduling condition */,
             int64_t                         horizon /**< [in] prediction time beyond query [ns] */,
             std::ofstream                  &manifest /**< [in,out] run summary */,
             int64_t                         samplePeriod = period /**< [in] actual acquisition period [ns] */,
             bool extended = false /**< [in] write gap-study accepted-record history and retained-gap count */ )
{
    CAPTURE( s.m_name, samplePeriod );
    parameters config;
    config.m_frequency       = 28;
    config.m_memoryTime      = .1;
    config.m_sampleRate      = 1e9 / samplePeriod;
    config.m_leverArm        = 10;
    config.m_rotationDegrees = 0;
    config.m_zero.fill( 2048 );
    config.m_voltsPerBit.fill( .0083799 );
    config.m_sensitivity.fill( 1.02 );
    config.m_predictionNs = horizon;
    processorT processor( config );
    processor.attached();
    source            input;
    const std::string name = population + "_" + s.m_name + "_" + std::to_string( horizon );
    std::ofstream     output( directory / ( name + ".bin" ), std::ios::binary );
    std::ofstream     events( directory / ( name + "_events.csv" ) );
    std::ofstream     accepted;
    std::ofstream     frames, queries;
    // Small opt-in timing fixtures reuse the exact production replay; historical binary outputs are unchanged.
    const std::string timingScenario = s.m_name;
    const bool        timingTrace = std::getenv( "ACCELFFCTRL_TIMING_TRACE" ) && extended && population == "single28" &&
                                    ( timingScenario == "loss1_1s" || timingScenario == "stamp_jitter2pct_loss1" ||
                                      timingScenario == "stamp_jitter10pct_loss1" ) &&
                                    ( directory.filename() == "reset" || directory.filename() == "optional" );
    if( timingTrace )
    {
        frames.open( directory / ( name + "_frames.csv" ) );
        queries.open( directory / ( name + "_queries.csv" ) );
        frames << "generation,counter,acquired_ns,written_ns,receipt_mono_ns,clock_real_ns,clock_mono_ns,"
                  "clock_uncertainty_ns,depth,accepted,adc_x,adc_y\n";
        queries << "query_ns,target_ns,sample_ns,ready,processed,dropped,resets,admitted_gaps,rejected_gaps\n";
    }
    if( extended )
        accepted.open( directory / ( name + "_accepted.bin" ), std::ios::binary );
    events << "time_ns,dropped,resets\n";
    bool       queryPending = false;
    uint64_t   gapEvents    = 0;
    size_t     ready = 0, scheduledReady = 0, scheduled = 0;
    int64_t    firstReady = -1;
    const auto drain      = [&]( int64_t now )
    {
        const auto before = processor.diagnostics();
        processor.drain( input, [now] { return inputClock{ epoch + now, now, 0 }; } );
        const auto &d = processor.diagnostics();
        if( d.m_dropped != before.m_dropped )
        {
            ++gapEvents;
            events << now << ',' << d.m_dropped - before.m_dropped << ',' << d.m_resets << '\n';
        }
        if( extended && d.m_processed > before.m_processed )
            writeRow<2>( accepted, { double( input.m_header.m_counter - 1 ), double( now ) } );
        if( timingTrace && d.m_counter != before.m_counter )
        {
            int64_t acquired, written;
            REQUIRE( inputProcessor::timeNs( input.m_frame.m_acquired, acquired ) );
            REQUIRE( inputProcessor::timeNs( input.m_frame.m_written, written ) );
            frames << d.m_generation << ',' << input.m_frame.m_counter << ',' << acquired << ',' << written << ','
                   << now << ',' << epoch + now << ',' << now << ",0,1," << ( d.m_processed > before.m_processed )
                   << ',' << input.m_frame.m_adc[0] << ',' << input.m_frame.m_adc[1] << '\n';
        }
    };
    const auto query = [&]( int64_t now, bool nominal, bool produced )
    {
        if( produced )
            processor.predict( now );
        const auto &d     = processor.diagnostics();
        const auto &p     = d.m_prediction;
        const bool  valid = produced && p.m_state == estimatorState::ready;
        if( timingTrace && nominal )
            queries << now << ',' << now + horizon << ',' << p.m_sampleTimeNs << ',' << valid << ',' << d.m_processed
                    << ',' << d.m_dropped << ',' << d.m_resets << ',' << d.m_admittedGaps << ',' << d.m_rejectedGaps
                    << '\n';
        scheduled += nominal;
        scheduledReady += nominal && valid;
        ready += valid;
        if( valid )
        {
            if( firstReady < 0 )
                firstReady = now;
            REQUIRE( p.m_targetTimeNs == now + horizon );
            REQUIRE( ( p.m_targetTimeNs - base ) % samplePeriod == 0 );
            if( s.m_stampJitterPercent )
            {
                const int64_t tagError =
                    ( int64_t( ( p.m_sequence - 1 ) % 3 ) - 1 ) * samplePeriod * s.m_stampJitterPercent / 100;
                REQUIRE( now - p.m_sampleTimeNs == samplePeriod - tagError );
            }
            else if( !s.m_readerOnly )
                REQUIRE( now - p.m_sampleTimeNs == samplePeriod );
            else
                REQUIRE( now - p.m_sampleTimeNs <= config.m_maxAgeNs );
            for( size_t axis = 0; axis < 2; ++axis )
            {
                const double phase = p.m_phase[axis] + 2 * sim::pi * 28 * ( now + horizon - p.m_sampleTimeNs ) * 1e-9;
                CHECK( p.m_acceleration[axis] == Approx( p.m_amplitude[axis] * std::cos( phase ) ).margin( 1e-12 ) );
                CHECK( p.m_pointing[axis] ==
                       Approx( -p.m_acceleration[axis] / ( std::pow( 2 * sim::pi * 28, 2 ) * 10 ) ).margin( 1e-18 ) );
            }
        }
        writeRow<21>( output,
                      { nominal ? double( now ) : -1.,
                        double( now ),
                        double( p.m_sampleTimeNs ),
                        double( now + horizon ),
                        double( valid ),
                        double( produced ),
                        produced ? double( p.m_state ) : -1.,
                        double( d.m_dropped ),
                        double( d.m_resets ),
                        double( d.m_processed ),
                        p.m_acceleration[0],
                        p.m_acceleration[1],
                        p.m_pointing[0],
                        p.m_pointing[1],
                        p.m_amplitude[0],
                        p.m_amplitude[1],
                        p.m_phase[0],
                        p.m_phase[1],
                        p.m_rcond,
                        double( p.m_sequence ),
                        d.m_writeDelayUs } );
    };
    for( size_t n = 0; n < count; ++n )
    {
        const int64_t elapsed = n * samplePeriod, now = base + elapsed;
        const bool    nominal = n % ( 500000 / samplePeriod ) == 1;
        const int64_t local   = clusterTime( s, elapsed );
        const bool    resume =
            local >= 0 && ( s.m_cluster == 1 ? local == s.m_duration
                                             : local / 1000000 < s.m_cluster && local % 1000000 == s.m_duration );
        if( nominal )
        {
            if( !blocked( s, elapsed ) )
                drain( now );
            query( now, true, s.m_readerOnly || !blocked( s, elapsed ) );
            queryPending = !s.m_readerOnly && blocked( s, elapsed );
        }
        else if( resume && ( queryPending || s.m_readerOnly ) )
        {
            drain( now );
            if( queryPending )
                query( now, false, true );
            queryPending = false;
        }
        // Producer completes at acquisition+25 us; the normal worker runs at acquisition+50 us.
        input.m_header.m_counter = n + 1;
        const int64_t tagError   = ( int64_t( n % 3 ) - 1 ) * samplePeriod * s.m_stampJitterPercent / 100;
        input.m_frame            = { recording[n].m_adc, stamp( now + tagError ), stamp( now + 25000 ), n + 1 };
        if( !s.m_queryOnly && !blocked( s, elapsed + 50000 ) )
            drain( now + 50000 );
    }
    drain( base + count * samplePeriod );
    const auto &d = processor.diagnostics();
    REQUIRE( d.m_processed + d.m_dropped == count );
    REQUIRE( d.m_invalid == 0 );
    REQUIRE( d.m_restarts == 0 );
    if( s.m_interval )
    {
        const int64_t  start        = s.m_start ? s.m_start : s.m_interval;
        const uint64_t events       = ( ( count * samplePeriod - 1 - start ) / s.m_interval + 1 ) * s.m_cluster;
        const uint64_t lostPerEvent = s.m_duration / samplePeriod - 1;
        REQUIRE( d.m_dropped == events * lostPerEvent );
        if constexpr( std::is_same_v<processorT, inputProcessor> )
            REQUIRE( d.m_resets == 1 + ( lostPerEvent ? events : 0 ) );
    }
    if( s.m_queryOnly )
    {
        REQUIRE( d.m_processed == count / 4 + 1 );
        REQUIRE( d.m_dropped == 3 * count / 4 - 1 );
        REQUIRE( d.m_resets == count / 4 + 1 );
    }
    if( !s.m_interval && !s.m_queryOnly )
    {
        REQUIRE( d.m_dropped == 0 );
        REQUIRE( d.m_resets == 1 );
        REQUIRE( ready > 0 );
    }
    if( s.m_queryOnly || s.m_interval == 100000000 )
        REQUIRE( ready == 0 );
    output.close();
    events.close();
    REQUIRE( output.good() );
    REQUIRE( events.good() );
    if( extended )
    {
        accepted.close();
        REQUIRE( accepted.good() );
    }
    if( timingTrace )
    {
        frames.close();
        queries.close();
        REQUIRE( frames.good() );
        REQUIRE( queries.good() );
    }
    manifest << population << ',' << s.m_name << ',' << horizon << ',' << count << ',' << d.m_processed << ','
             << d.m_dropped << ',' << d.m_resets << ',' << scheduled << ',' << scheduledReady << ',' << firstReady
             << ',' << name;
    if( extended )
    {
        uint64_t retained = 0;
        if constexpr( requires { processor.retained(); } )
            retained = processor.retained();
        manifest << ',' << retained << ',' << gapEvents - retained;
        if constexpr( std::is_same_v<processorT, inputProcessor> ||
                      std::is_same_v<processorT, gapValidation::productionObserver> )
            REQUIRE( d.m_rejectedGaps == gapEvents - retained );
    }
    manifest << '\n';
}
/// Record truth once, independently of every subsequent raw-only observer replay.
std::vector<sim::sample> record( const std::filesystem::path &directory /**< [in] output directory */,
                                 const std::string           &population /**< [in] recording identifier */,
                                 sim::model &model /**< [in,out] configured source, advanced only here */,
                                 const std::vector<sim::modeSpec> &modes /**< [in] actual source specifications */,
                                 size_t                            count /**< [in] producer publications */,
                                 size_t                            padding /**< [in] future truth samples */,
                                 int64_t                           samplePeriod /**< [in] acquisition period [ns] */ )
{
    std::vector<sim::sample> recording;
    const auto target = std::find_if( modes.begin(), modes.end(), []( const auto &m ) { return m.m_frequency == 28; } );
    REQUIRE( target != modes.end() );
    const size_t  targetIndex = target - modes.begin();
    std::ofstream modeOutput( directory / ( population + "_modes.csv" ) );
    modeOutput << "frequency_hz,accel_rms_ms2,r,coupling_x,coupling_y\n" << std::setprecision( 17 );
    for( const auto &m : modes )
        modeOutput << m.m_frequency << ',' << m.m_accelRms << ',' << m.m_r << ',' << m.m_couplingX << ','
                   << m.m_couplingY << '\n';
    modeOutput.close();
    REQUIRE( modeOutput.good() );
    std::ofstream truth( directory / ( population + "_truth.bin" ), std::ios::binary );
    recording.reserve( count + padding );
    for( size_t n = 0; n < count + padding; ++n )
    {
        recording.push_back( model.step() );
        const auto           &sample = recording.back();
        std::array<double, 2> clean{}, isolated{}, theta{};
        for( size_t axis = 0; axis < 2; ++axis )
        {
            for( const auto &m : model.m_modes )
                clean[axis] += m.m_accelPerArcsec * m.m_axes[axis].m_previous1;
            const auto &m  = model.m_modes[targetIndex];
            theta[axis]    = m.m_axes[axis].m_previous1 * sim::arcsecToRad;
            isolated[axis] = m.m_accelPerArcsec * m.m_axes[axis].m_previous1;
        }
        writeRow<13>( truth,
                      { double( base + n * samplePeriod ),
                        double( sample.m_adc[0] ),
                        double( sample.m_adc[1] ),
                        sample.m_acceleration[0],
                        sample.m_acceleration[1],
                        clean[0],
                        clean[1],
                        sample.m_pointing[0] * sim::arcsecToRad,
                        sample.m_pointing[1] * sim::arcsecToRad,
                        isolated[0],
                        isolated[1],
                        theta[0],
                        theta[1] } );
    }
    truth.close();
    REQUIRE( truth.good() );
    REQUIRE( model.m_clipLow == std::array<uint64_t, 2>{ 0, 0 } );
    REQUIRE( model.m_clipHigh == std::array<uint64_t, 2>{ 0, 0 } );
    return recording;
}
} // namespace validation
/// \endcond

/** \brief Export the supplied 8-kHz calibration, isolated/full truth, horizon sweep and scheduling-loss study.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline end-to-end validation export", "[.][accelFFCtrl-validation-export]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::model::step;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::predict;
#endif
    // clang-format on
    using namespace validation;
    static_assert( sizeof( double ) == 8 && std::numeric_limits<double>::is_iec559 );
    REQUIRE( std::endian::native == std::endian::little );
    const char *path = std::getenv( "ACCELFFCTRL_VALIDATION_DIR" );
    REQUIRE( path != nullptr );
    const std::filesystem::path directory( path );
    std::filesystem::create_directories( directory );
    const char *duration = std::getenv( "ACCELFFCTRL_VALIDATION_SECONDS" );
    const int   seconds  = duration ? std::stoi( duration ) : 60;
    REQUIRE( seconds >= 6 );
    const size_t  count = size_t( seconds ) * 8000, padding = 162;
    std::ofstream metadata( directory / "metadata.txt" );
    metadata << "format=1: little-endian IEEE754 float64 row-major\ncompiler=" << __VERSION__ << "\nseconds=" << seconds
             << "\nfps=8000\nseed=42\nmode_seed=123\nlever_m=10\nnoise_psd=1e-8\n"
                "adc_zero=2048\nvolts_per_bit=0.0083799\nsensitivity=1.02\nfrequency_hz=28\nmemory_s=0.1\n"
                "rotation_deg=0\npublication_delay_ns=25000\nworker_delay_ns=50000\nquery_age_ns=125000\n";
#ifdef _LIBCPP_VERSION
    metadata << "libcxx=" << _LIBCPP_VERSION << '\n';
#elif defined( __GLIBCXX__ )
    metadata << "libstdcxx=" << __GLIBCXX__ << '\n';
#endif
    metadata.close();
    REQUIRE( metadata.good() );
    std::ofstream manifest( directory / "runs.csv" );
    manifest << "population,scenario,horizon_ns,published,processed,dropped,resets,scheduled,scheduled_ready,first_"
                "ready_ns,file\n";
    const std::array<schedule, 7> schedules{ { { "ideal", 0, 0 },
                                               { "delay125us_1s", 1000000000, 125000 },
                                               { "loss1_5s", 5000000000, 250000 },
                                               { "loss1_1s", 1000000000, 250000 },
                                               { "loss1_100ms", 100000000, 250000 },
                                               { "block2ms_5s", 5000000000, 2000000 },
                                               { "query_only_2k", 0, 0, true } } };
    for( bool full : { false, true } )
    {
        const std::string population = full ? "full214" : "single28";
        sim::parameters   config;
        config.m_fps         = 8000;
        config.m_seed        = 42;
        config.m_modeSeed    = 123;
        config.m_leverArm    = 10;
        config.m_noisePSD    = 1e-8;
        config.m_zero        = 2048;
        config.m_voltsPerBit = .0083799;
        config.m_sensitivity = 1.02;
        const auto modes = full ? sim::referenceModes( 123 ) : std::vector<sim::modeSpec>{ { 28, .002, .99999, 1, 1 } };
        sim::model model;
        model.configure( config, modes );
        const auto recording = record( directory, population, model, modes, count, padding, period );
        for( const auto &s : schedules )
            for( int64_t horizon : { 0, 500000, 1000000, 2000000, 5000000, 10000000, 20000000 } )
                if( horizon == 1000000 || std::string( s.m_name ) == "ideal" )
                    replay( directory, population, recording, count, s, horizon, manifest );
        // Controller replays neither step the model nor consume random deviates.
        REQUIRE( model.m_ticks == count + padding );
    }
    manifest.close();
    REQUIRE( manifest.good() );
}

/** \brief Compare production and isolated offline gap policies at 8/4 kHz on the same recorded ADC sequences.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline acquisition-rate and bounded-gap study", "[.][accelFFCtrl-gap-export]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::model::step;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::consume;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::predict;
#endif
    // clang-format on
    using namespace validation;
    const char *path = std::getenv( "ACCELFFCTRL_VALIDATION_DIR" );
    REQUIRE( path != nullptr );
    const char *duration = std::getenv( "ACCELFFCTRL_VALIDATION_SECONDS" );
    const int   seconds  = duration ? std::stoi( duration ) : 60;
    REQUIRE( seconds >= 6 );
    static_assert( sizeof( double ) == 8 && std::numeric_limits<double>::is_iec559 );
    REQUIRE( std::endian::native == std::endian::little );
    for( int rate : { 8000, 4000 } )
    {
        const int64_t               samplePeriod = 1000000000 / rate;
        const size_t                count = size_t( seconds ) * rate, padding = 162;
        const std::filesystem::path directory = std::filesystem::path( path ) / std::to_string( rate );
        std::filesystem::create_directories( directory );
        std::ofstream metadata( directory / "metadata.txt" );
        metadata << "format=1 little-endian float64, existing truth/prediction columns\ncompiler=" << __VERSION__
                 << "\nfps=" << rate << "\nseconds=" << seconds
                 << "\nseed=42\nmode_seed=123\nfrequency_hz=28\nmemory_s=0.1\nlever_m=10\nnoise_psd=1e-8\n"
                    "adc_zero=2048\nvolts_per_bit=0.0083799\nsensitivity=1.02\nhorizon_ns=1000000\n"
                    "pole_radii=unchanged_per_sample\ngap_elapsed_limit_ns=1000000\ngap_spacing_s=0.1\n"
                    "cadence_tolerance=0.05 nominal period\n";
#ifdef _LIBCPP_VERSION
        metadata << "libcxx=" << _LIBCPP_VERSION << '\n';
#elif defined( __GLIBCXX__ )
        metadata << "libstdcxx=" << __GLIBCXX__ << '\n';
#endif
        metadata.close();
        REQUIRE( metadata.good() );
        const std::array<schedule, 15> schedules{
            { { "ideal", 0, 0 },
              { "delay_period_1s", 1000000000, samplePeriod },
              { "block250us_1s", 1000000000, 250000 },
              { "block500us_1s", 1000000000, 500000 },
              { "block1ms_1s", 1000000000, 1000000 },
              { "loss1_1s", 1000000000, 2 * samplePeriod },
              { "loss1_4Hz", 250000000, 2 * samplePeriod, false, 1000000000 },
              { "cluster3_5s", 5000000000, 2 * samplePeriod, false, 0, 3 },
              { "block2ms_5s", 5000000000, 2000000 },
              { "reader3ms_5s", 5000000000, 3000000, false, 0, 1, true },
              { "startup_loss100ms", 100000000, 2 * samplePeriod },
              { "stamp_jitter2pct_ideal", 0, 0, false, 0, 1, false, 2 },
              { "stamp_jitter10pct_ideal", 0, 0, false, 0, 1, false, 10 },
              { "stamp_jitter2pct_loss1", 1000000000, 2 * samplePeriod, false, 0, 1, false, 2 },
              { "stamp_jitter10pct_loss1", 1000000000, 2 * samplePeriod, false, 0, 1, false, 10 } } };
        for( bool full : { false, true } )
        {
            const std::string population = full ? "full214" : "single28";
            sim::parameters   config;
            config.m_fps         = rate;
            config.m_seed        = 42;
            config.m_modeSeed    = 123;
            config.m_noisePSD    = 1e-8;
            config.m_leverArm    = 10;
            config.m_zero        = 2048;
            config.m_voltsPerBit = .0083799;
            config.m_sensitivity = 1.02;
            const auto modes =
                full ? sim::referenceModes( 123 ) : std::vector<sim::modeSpec>{ { 28, .002, .99999, 1, 1 } };
            sim::model model;
            model.configure( config, modes );
            const auto recording = record( directory, population, model, modes, count, padding, samplePeriod );
            for( const std::string policy : { "reset", "gap1", "gap7", "optional" } )
            {
                const auto output = directory / policy;
                std::filesystem::create_directories( output );
                std::ofstream manifest( output / ( population + "_runs.csv" ) );
                manifest << "population,scenario,horizon_ns,published,processed,dropped,resets,scheduled,scheduled_"
                            "ready,first_ready_ns,file,retained,rejected\n";
                for( const auto &s : schedules )
                {
                    if( s.m_stampJitterPercent && ( policy == "gap1" || policy == "gap7" ) )
                        continue; // Historical prototypes also reject ordinary cadence jitter; keep their tests intact.
                    if( policy == "reset" )
                        replay( output, population, recording, count, s, 1000000, manifest, samplePeriod, true );
                    else if( policy == "gap1" )
                        replay<gapValidation::observer<1>>(
                            output, population, recording, count, s, 1000000, manifest, samplePeriod, true );
                    else if( policy == "gap7" )
                        replay<gapValidation::observer<7>>(
                            output, population, recording, count, s, 1000000, manifest, samplePeriod, true );
                    else
                        replay<gapValidation::productionObserver>(
                            output, population, recording, count, s, 1000000, manifest, samplePeriod, true );
                }
                manifest.close();
                REQUIRE( manifest.good() );
            }
            REQUIRE( model.m_ticks == count + padding );
        }
    }
}
} // namespace accelFFCtrlTest
} // namespace libXWCTest
