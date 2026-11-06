/** \file accelFFCtrlValidation_test.cpp
 * \brief Opt-in offline truth export using the production depth-one ingestion and prediction pipeline.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlInput.hpp"
#include "../../accelSim/accelSimModel.hpp"

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>

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
    const char *m_name;             ///< Stable artifact identifier.
    int64_t m_interval, m_duration; ///< Periodic worker blackout spacing/duration in ns, starting after one interval.
    bool    m_queryOnly{ false };   ///< Negative control that incorrectly ingests at the prediction rate.
};

/// Test whether a worker event is suppressed by the selected deterministic blackout.
bool blocked( const schedule &s /**< [in] worker schedule */, int64_t elapsed /**< [in] ns since first sample */ )
{
    return s.m_interval && elapsed >= s.m_interval && elapsed % s.m_interval < s.m_duration;
}

/// Run the actual input pipeline against a pre-generated recording; no truth enters the controller.
void replay( const std::filesystem::path    &directory /**< [in] output directory */,
             const std::string              &population /**< [in] recording identifier */,
             const std::vector<sim::sample> &recording /**< [in] source ADC records, truth ignored here */,
             size_t                          count /**< [in] number of producer publications */,
             const schedule                 &s /**< [in] worker scheduling condition */,
             int64_t                         horizon /**< [in] prediction time beyond query [ns] */,
             std::ofstream                  &manifest /**< [in,out] run summary */ )
{
    parameters config;
    config.m_frequency       = 28;
    config.m_memoryTime      = .1;
    config.m_sampleRate      = 8000;
    config.m_leverArm        = 10;
    config.m_rotationDegrees = 0;
    config.m_zero.fill( 2048 );
    config.m_voltsPerBit.fill( .0083799 );
    config.m_sensitivity.fill( 1.02 );
    config.m_predictionNs = horizon;
    inputProcessor processor( config );
    processor.attached();
    source            input;
    const std::string name = population + "_" + s.m_name + "_" + std::to_string( horizon );
    std::ofstream     output( directory / ( name + ".bin" ), std::ios::binary );
    std::ofstream     events( directory / ( name + "_events.csv" ) );
    events << "time_ns,dropped,resets\n";
    bool       queryPending = false;
    size_t     ready = 0, scheduledReady = 0, scheduled = 0;
    int64_t    firstReady = -1;
    const auto drain      = [&]( int64_t now )
    {
        const auto before = processor.diagnostics().m_dropped;
        processor.drain( input, [now] { return inputClock{ epoch + now, now, 0 }; } );
        const auto &d = processor.diagnostics();
        if( d.m_dropped != before )
            events << now << ',' << d.m_dropped - before << ',' << d.m_resets << '\n';
    };
    const auto query = [&]( int64_t now, bool nominal, bool produced )
    {
        if( produced )
            processor.predict( now );
        const auto &d     = processor.diagnostics();
        const auto &p     = d.m_prediction;
        const bool  valid = produced && p.m_state == estimatorState::ready;
        scheduled += nominal;
        scheduledReady += nominal && valid;
        ready += valid;
        if( valid )
        {
            if( firstReady < 0 )
                firstReady = now;
            REQUIRE( p.m_targetTimeNs == now + horizon );
            REQUIRE( ( p.m_targetTimeNs - base ) % period == 0 );
            REQUIRE( now - p.m_sampleTimeNs == period );
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
        const int64_t elapsed = n * period, now = base + elapsed;
        const bool    nominal = n % 4 == 1;
        const bool    resume  = s.m_interval && elapsed >= s.m_interval && elapsed % s.m_interval == s.m_duration;
        if( nominal )
        {
            if( !blocked( s, elapsed ) )
                drain( now );
            query( now, true, !blocked( s, elapsed ) );
            queryPending = blocked( s, elapsed );
        }
        else if( resume && queryPending )
        {
            drain( now );
            query( now, false, true );
            queryPending = false;
        }
        // Producer completes at acquisition+25 us; the normal worker runs at acquisition+50 us.
        input.m_header.m_counter = n + 1;
        input.m_frame            = { recording[n].m_adc, stamp( now ), stamp( now + 25000 ), n + 1 };
        if( !s.m_queryOnly && !blocked( s, elapsed + 50000 ) )
            drain( now + 50000 );
    }
    drain( base + count * period );
    const auto &d = processor.diagnostics();
    REQUIRE( d.m_processed + d.m_dropped == count );
    REQUIRE( d.m_invalid == 0 );
    REQUIRE( d.m_restarts == 0 );
    if( s.m_interval )
    {
        const uint64_t events       = ( count * period - 1 ) / s.m_interval;
        const uint64_t lostPerEvent = s.m_duration / period - 1;
        REQUIRE( d.m_dropped == events * lostPerEvent );
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
    manifest << population << ',' << s.m_name << ',' << horizon << ',' << count << ',' << d.m_processed << ','
             << d.m_dropped << ',' << d.m_resets << ',' << scheduled << ',' << scheduledReady << ',' << firstReady
             << ',' << name << '\n';
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
        const auto target =
            std::find_if( modes.begin(), modes.end(), []( const auto &m ) { return m.m_frequency == 28; } );
        REQUIRE( target != modes.end() );
        const size_t  targetIndex = target - modes.begin();
        std::ofstream modeOutput( directory / ( population + "_modes.csv" ) );
        modeOutput << "frequency_hz,accel_rms_ms2,r,coupling_x,coupling_y\n" << std::setprecision( 17 );
        for( const auto &m : modes )
            modeOutput << m.m_frequency << ',' << m.m_accelRms << ',' << m.m_r << ',' << m.m_couplingX << ','
                       << m.m_couplingY << '\n';
        modeOutput.close();
        REQUIRE( modeOutput.good() );
        std::ofstream            truth( directory / ( population + "_truth.bin" ), std::ios::binary );
        std::vector<sim::sample> recording;
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
                          { double( base + n * period ),
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
} // namespace accelFFCtrlTest
} // namespace libXWCTest
