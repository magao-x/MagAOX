/** \file accelFFCtrlCancellation_export.cpp
 * \brief Standalone causal offline cancellation study using raw-only production readers and synthetic DM dynamics.
 * \ingroup accelFFCtrl_unit_test
 */
#include "accelFFCtrlCancellation_test.hpp"
#include "accelFFCtrlGap_test.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>

using namespace libXWCTest::accelFFCtrlTest;
using namespace cancellation;

/// \cond ACCELFFCTRL_CANCELLATION_EXPORT
namespace
{
/// A study changes explicit offline controls only; no configuration files are loaded.
struct experiment
{
    std::string m_name, m_fault{ "none" }; ///< Unique run name and deterministic input disturbance.
    double      m_gain{ 0 }, m_bandwidth{ 300 }, m_calibrationGain{ 1 },
        m_calibrationAngle{ 0 };                      ///< Synthetic plant controls.
    int64_t m_horizon{ 1000000 }, m_delay{ 1000000 }; ///< Query prediction horizon and actual plant delay [ns].
    double  m_bound{ .1 }; ///< Modal coefficient bounds, far above nominal baseline commands.
    bool    m_oracle{ false }, m_gap{ false },
        m_trace{ false }; ///< Separate truth oracle, optional policy and binary export.
};

/// One production query, generated before the measurement acquired at that query time is consumed.
struct query
{
    std::array<prediction, 2> m_bands; ///< Raw-only independent harmonic predictions with the same identity/target.
};

/// Cache a raw-only replay once for all gain/plant comparisons sharing its input policy and horizon.
struct replay
{
    std::vector<query> m_queries; ///< Fixed 2-kHz queries; no disturbance truth enters their calculation.
    inputDiagnostics   m_stats;   ///< First band's diagnostics; identical source/drop accounting in both bands.
};

/// Replay only ADC and source timestamps through unchanged production validation and estimation.
replay predict( const std::vector<std::array<uint16_t, 2>> &adc /**< [in] raw-only source recording */,
                size_t                     count /**< [in] input frames excluding independent future scoring padding */,
                const std::vector<double> &frequencies /**< [in] explicitly selected harmonic bands */,
                const sim::parameters     &sensor /**< [in] matching calibration/rate */,
                const experiment          &run /**< [in] horizon and input fault policy */ )
{
    std::vector<inputProcessor> readers;
    readers.reserve( frequencies.size() );
    for( double f : frequencies )
    {
        parameters c;
        c.m_frequency  = f;
        c.m_sampleRate = 8000;
        c.m_leverArm   = sensor.m_leverArm;
        c.m_zero.fill( sensor.m_zero );
        c.m_voltsPerBit.fill( sensor.m_voltsPerBit );
        c.m_sensitivity.fill( sensor.m_sensitivity );
        c.m_predictionNs = run.m_horizon;
        readers.emplace_back( c, 256, 1000000, run.m_gap );
        readers.back().attached();
    }
    replay result;
    result.m_queries.reserve( count / 4 );
    gapValidation::snapshot source;
    source.m_header = { 2, 2, 1, 1, 0, 0, true, true, true, false };
    for( size_t n = 0; n < count; ++n )
    {
        const int64_t now = origin + n * period;
        if( n == 40000 && ( run.m_fault == "reset" || run.m_fault == "restart" ) )
        {
            for( auto &reader : readers )
                if( run.m_fault == "reset" )
                    reader.reset();
                else
                {
                    reader.disconnected();
                    reader.attached();
                }
        }
        if( n % 4 == 0 )
        {
            query q;
            for( size_t band = 0; band < readers.size(); ++band )
            {
                readers[band].predict( now );
                q.m_bands[band] = readers[band].diagnostics().m_prediction;
                const auto &p   = q.m_bands[band];
                if( p.m_state == estimatorState::ready &&
                    ( p.m_sampleTimeNs >= now || p.m_targetTimeNs != now + run.m_horizon ||
                      now - p.m_sampleTimeNs > 2000000 ) )
                    throw std::runtime_error( "noncausal or stale offline query" );
            }
            result.m_queries.push_back( q );
        }
        const bool single  = run.m_fault == "single" && n && n % 8000 == 0;
        const bool cluster = run.m_fault == "cluster" && n >= 40000 && n % 40000 >= 1 && n % 40000 <= 3;
        const bool stale   = run.m_fault == "stale" && n >= 40000 && n < 40080;
        if( single || cluster || stale )
            continue;
        source.m_header.m_counter = n + 1;
        const timespec stamp{ ( epoch + now ) / 1000000000, ( epoch + now ) % 1000000000 };
        source.m_frame = { adc[n], stamp, stamp, n + 1 };
        if( run.m_fault == "invalid" && n == 40000 )
            source.m_frame.m_adc[0] = 4096;
        const inputClock clock{ epoch + now, now, 0 };
        for( auto &reader : readers )
            reader.drain( source, [clock] { return clock; } );
    }
    result.m_stats = readers.front().diagnostics();
    for( const auto &reader : readers )
        if( reader.diagnostics().m_dropped != result.m_stats.m_dropped )
            throw std::runtime_error( "inconsistent band loss accounting" );
    if( result.m_stats.m_processed + result.m_stats.m_dropped != count )
        throw std::runtime_error( "unaccounted source frames" );
    const uint64_t expectedDropped  = run.m_fault == "single"    ? count / 8000 - 1
                                      : run.m_fault == "cluster" ? 3 * ( ( count - 1 ) / 40000 )
                                      : run.m_fault == "invalid" ? 1
                                      : run.m_fault == "stale"   ? 80
                                                                 : 0;
    const uint64_t expectedAdmitted = run.m_fault == "single" && run.m_gap ? expectedDropped : 0;
    if( result.m_stats.m_dropped != expectedDropped || result.m_stats.m_admittedGaps != expectedAdmitted )
        throw std::runtime_error( "incorrect deterministic drop/admission counts" );
    return result;
}

/// Accumulate full 8-kHz statistics, including command holding and inhibited intervals.
struct moments
{
    double m_sum{ 0 }, m_square{ 0 }, m_cos[2]{}, m_sin[2]{}; ///< Mean, RMS and coherent 28/56-Hz projections.

    /// Include one sample in the common, complete scoring interval.
    void add( double value /**< [in] numerical sample */, double seconds /**< [in] time since source origin */ );
};

void moments::add( double value, double seconds )
{
    m_sum += value;
    m_square += value * value;
    for( unsigned band = 0; band < 2; ++band )
    {
        const double phase = 2 * pi * ( band ? 56 : 28 ) * seconds;
        m_cos[band] += value * std::cos( phase );
        m_sin[band] -= value * std::sin( phase );
    }
}

/// Evaluate one plant/gain experiment; only the explicitly labelled oracle reads future truth for commands.
void evaluate( const std::filesystem::path    &directory /**< [in] local artifact directory */,
               const std::string              &population /**< [in] source label */,
               const std::vector<sim::sample> &recording /**< [in] independent source/truth recording */,
               const replay                   &predictions /**< [in] cached raw-only predictions */,
               size_t                          bands /**< [in] number of configured estimator bands */,
               const experiment               &run /**< [in] offline controls */,
               std::ofstream                  &summary /**< [in,out] complete run table */ )
{
    const calibration cal;
    auto              c = controls( run.m_gain, run.m_horizon );
    c.m_maxModal.fill( run.m_bound );
    const feedForward controller( cal.m_mapping, c );
    Eigen::Matrix2d   rotation;
    rotation << std::cos( run.m_calibrationAngle ), -std::sin( run.m_calibrationAngle ),
        std::sin( run.m_calibrationAngle ), std::cos( run.m_calibrationAngle );
    dmResponse        plant( run.m_calibrationGain * rotation * cal.m_plant, period, run.m_delay, run.m_bandwidth );
    const size_t      count = predictions.m_queries.size() * 4;
    const std::string name  = population + "_" + run.m_name;
    std::ofstream     trace;
    if( run.m_trace )
        trace.open( directory / ( name + ".bin" ), std::ios::binary );
    std::array<double, 2>  held{}, target{}, error{};
    correction             proposal;
    std::array<moments, 2> before, after, effect;
    std::array<double, 2>  errorSquare{}, targetSquare{}, modalSquare{}, modalPeak{};
    double                 surfacePeak = 0;
    size_t                 ready = 0, inhibited = 0, limited = 0, errorCount = 0;
    bool                   valid = false;
    for( size_t n = 0; n < count; ++n )
    {
        const int64_t now = origin + n * period;
        if( n % 4 == 0 )
        {
            auto        p           = predictions.m_queries[n / 4].m_bands;
            const auto &truthTarget = recording.at( n + run.m_horizon / period );
            if( run.m_oracle )
                p[0] = idealPrediction(
                    { truthTarget.m_pointing[0] * sim::arcsecToRad, truthTarget.m_pointing[1] * sim::arcsecToRad },
                    now,
                    run.m_horizon );
            const size_t used = run.m_oracle ? 1 : bands;
            valid             = true;
            target            = {};
            error             = {};
            for( size_t band = 0; band < used; ++band )
            {
                valid = valid && p[band].m_state == estimatorState::ready;
                for( size_t axis = 0; axis < 2; ++axis )
                    target[axis] += p[band].m_pointing[axis];
            }
            proposal = controller.propose( std::span<const prediction>( p.data(), used ), now );
            held     = proposal.m_modal; // Every query replaces the contribution, including explicit inhibited zeros.
            if( !valid && run.m_gain > 0 && proposal.m_state != correctionState::inhibited )
                throw std::runtime_error( "unavailable predictor generated a correction" );
            ready += valid;
            inhibited += proposal.m_state == correctionState::inhibited;
            limited += proposal.m_state == correctionState::limited;
            if( ( proposal.m_state == correctionState::inhibited || proposal.m_state == correctionState::disabled ) &&
                held != std::array<double, 2>{ 0, 0 } )
                throw std::runtime_error( "held invalid correction" );
            if( valid )
                for( size_t axis = 0; axis < 2; ++axis )
                {
                    error[axis] = target[axis] - truthTarget.m_pointing[axis] * sim::arcsecToRad;
                    if( n >= 32000 )
                    {
                        errorSquare[axis] += error[axis] * error[axis];
                        targetSquare[axis] += std::pow( truthTarget.m_pointing[axis] * sim::arcsecToRad, 2 );
                    }
                }
            errorCount += valid && n >= 32000;
        }
        const auto            response = plant.step( held );
        std::array<double, 2> truth, residual;
        for( size_t axis = 0; axis < 2; ++axis )
        {
            truth[axis]    = recording[n].m_pointing[axis] * sim::arcsecToRad;
            residual[axis] = truth[axis] + response[axis];
            if( n >= 32000 )
            {
                before[axis].add( truth[axis], n * period * 1e-9 );
                after[axis].add( residual[axis], n * period * 1e-9 );
                effect[axis].add( response[axis], n * period * 1e-9 );
                modalSquare[axis] += held[axis] * held[axis];
                modalPeak[axis] = std::max( modalPeak[axis], std::abs( held[axis] ) );
            }
        }
        if( n >= 32000 )
            surfacePeak =
                std::max( surfacePeak, ( cal.m_modes * Eigen::Vector2d( held[0], held[1] ) ).cwiseAbs().maxCoeff() );
        if( run.m_trace )
        {
            const std::array<double, 16> row{ n * period * 1e-9,
                                              truth[0],
                                              truth[1],
                                              response[0],
                                              response[1],
                                              residual[0],
                                              residual[1],
                                              held[0],
                                              held[1],
                                              double( valid ),
                                              double( proposal.m_state ),
                                              target[0],
                                              target[1],
                                              error[0],
                                              error[1],
                                              double( n % 4 == 0 ) };
            trace.write( reinterpret_cast<const char *>( row.data() ), sizeof( row ) );
        }
    }
    const double samples = count - 32000;
    for( size_t axis = 0; axis < 2; ++axis )
        for( unsigned band = 0; band < 2; ++band )
        {
            const auto  &b = before[axis], &a = after[axis], &e = effect[axis];
            const double bm = std::hypot( b.m_cos[band], b.m_sin[band] );
            const double am = std::hypot( a.m_cos[band], a.m_sin[band] );
            const double em = std::hypot( e.m_cos[band], e.m_sin[band] );
            const double residualPhase =
                std::remainder( std::atan2( a.m_sin[band], a.m_cos[band] ) - std::atan2( b.m_sin[band], b.m_cos[band] ),
                                2 * pi ) *
                180 / pi;
            const double correctionPhase = std::remainder( std::atan2( -e.m_sin[band], -e.m_cos[band] ) -
                                                               std::atan2( b.m_sin[band], b.m_cos[band] ),
                                                           2 * pi ) *
                                           180 / pi;
            const auto  &d               = predictions.m_stats;
            summary << name << ',' << population << ',' << run.m_gain << ',' << run.m_oracle << ',' << axis << ','
                    << ( band ? 56 : 28 ) << ',' << run.m_horizon << ',' << run.m_delay << ',' << run.m_bandwidth << ','
                    << run.m_calibrationGain << ',' << run.m_calibrationAngle << ',' << run.m_bound << ',' << run.m_gap
                    << ',' << d.m_dropped << ',' << d.m_resets << ',' << d.m_admittedGaps << ',' << d.m_rejectedGaps
                    << ',' << 100. * ready / ( count / 4 ) << ',' << inhibited << ',' << limited << ','
                    << std::sqrt( b.m_square / samples ) << ',' << std::sqrt( a.m_square / samples ) << ','
                    << 100 * ( 1 - ( a.m_square / samples - std::pow( a.m_sum / samples, 2 ) ) /
                                       ( b.m_square / samples - std::pow( b.m_sum / samples, 2 ) ) )
                    << ',' << 20 * std::log10( bm / std::max( am, 1e-300 ) ) << ',' << am / bm << ',' << residualPhase
                    << ',' << 100 * ( em / bm - 1 ) << ',' << correctionPhase << ','
                    << ( errorCount ? std::sqrt( errorSquare[axis] / errorCount ) : NAN ) << ','
                    << ( errorCount ? std::sqrt( errorSquare[axis] / targetSquare[axis] ) : NAN ) << ','
                    << std::sqrt( modalSquare[axis] / samples ) << ',' << modalPeak[axis] << ',' << surfacePeak << ','
                    << ( run.m_trace ? name + ".bin" : "-" ) << '\n';
        }
    if( run.m_trace )
    {
        trace.close();
        if( !trace.good() )
            throw std::runtime_error( "failed cancellation trace export" );
    }
}
} // namespace
/// \endcond

/// Export deterministic synthetic cancellation experiments; arguments are output directory and integer seconds.
int main( int argc /**< [in] argument count */, char **argv /**< [in] directory and optional duration */ )
{
    try
    {
        if( argc < 2 || argc > 3 )
            throw std::invalid_argument( "usage: accelFFCtrlCancellation_export directory [seconds>=8]" );
        const int seconds = argc == 3 ? std::stoi( argv[2] ) : 60;
        if( seconds < 8 || seconds > 600 )
            throw std::invalid_argument( "duration must be 8..600 seconds" );
        const size_t                count     = size_t( seconds ) * 8000;
        const std::filesystem::path directory = argv[1];
        std::filesystem::create_directories( directory );
        std::ofstream metadata( directory / "metadata.txt" );
        metadata
            << "Offline synthetic cancellation only\ncompiler=" << __VERSION__ << "\nseconds=" << seconds
            << "\nacquisition_hz=8000\nquery_hz=2000\nscore_start_s=4\nseed=42\nmode_seed=123\nlever_m=10\n"
               "mode_accel_rms_ms2=0.002\nmode_pole_radius=0.99999\ncoupling_xy=1,1\nnominal_noise_psd=1e-8\nnominal_"
               "adc=2048,0.0083799,1.02\nanalytic_adc=2048,0.00001,1.02\n"
               "synthetic_pupil_m=6.5\nsynthetic_mode_rotation_deg=30\nmodal_unit=coefficient;1 gives 1um surface RMS\n"
               "plant=x[n]=a*x[n-1]+(1-a)*u[n-delay-1]; bandwidth0=u[n-delay]\n"
               "prediction_before_current_acquisition;sample_age_ns=125000\n";
#ifdef _LIBCPP_VERSION
        metadata << "libcxx=" << _LIBCPP_VERSION << '\n';
#elif defined( __GLIBCXX__ )
        metadata << "libstdcxx=" << __GLIBCXX__ << '\n';
#endif
        metadata.close();
        std::ofstream summary( directory / "summary.csv" );
        summary
            << std::setprecision( 17 )
            << "run,population,gain,oracle,axis,frequency,horizon_ns,delay_ns,bandwidth_hz,calibration_gain,"
               "calibration_angle_rad,"
               "modal_bound,gap_enabled,dropped,resets,admitted,rejected,availability_pct,inhibited_queries,limited_"
               "queries,"
               "before_rms_rad,after_rms_rad,variance_reduction_pct,coherent_suppression_db,residual_amplitude_ratio,"
               "residual_phase_deg,opposing_amplitude_error_pct,opposing_phase_error_deg,prediction_rmse_rad,"
               "prediction_nrmse,"
               "modal_rms,modal_peak,surface_peak_um,trace\n";
        for( bool analytic : { false, true } )
            for( unsigned population = 0; population < 3; ++population )
                for( unsigned noiseCase = 0; noiseCase < ( !analytic && population == 2 ? 3u : 1u ); ++noiseCase )
                {
                    const std::vector<double> frequencies = population == 0   ? std::vector<double>{ 28 }
                                                            : population == 1 ? std::vector<double>{ 56 }
                                                                              : std::vector<double>{ 28, 56 };
                    const std::string         label       = std::string( analytic ? "analytic_" : "ar_" ) +
                                                            std::to_string( population ) +
                                                            ( noiseCase ? "_noise" + std::to_string( noiseCase ) : "" );
                    sim::parameters           sensor;
                    sensor.m_fps      = 8000;
                    sensor.m_noisePSD = analytic || noiseCase == 2 ? 0 : noiseCase == 1 ? 1e-7 : 1e-8;
                    if( analytic )
                        sensor.m_voltsPerBit = 1e-5;
                    std::vector<sim::modeSpec> modes;
                    for( double f : frequencies )
                        modes.push_back( { f, .002, .99999, 1, 1 } );
                    sim::model model;
                    model.configure( sensor, modes );
                    std::vector<sim::sample> recording( count + 200 );
                    uint64_t                 clippedLow = 0, clippedHigh = 0;
                    for( size_t n = 0; n < recording.size(); ++n )
                        if( !analytic )
                            recording[n] = model.step();
                        else
                            for( size_t axis = 0; axis < 2; ++axis )
                            {
                                double theta = 0, acceleration = 0;
                                for( double f : frequencies )
                                {
                                    const double value = .002 * std::sqrt( 2. ) *
                                                         std::cos( 2 * pi * f * n * period * 1e-9 + .31 - axis * .8 );
                                    acceleration += value;
                                    theta += value / sim::accelerationFactor( f, sensor.m_leverArm );
                                }
                                recording[n].m_pointing[axis] = theta;
                                recording[n].m_adc[axis] =
                                    sim::adcCounts( acceleration, sensor, clippedLow, clippedHigh );
                            }
                    if( model.m_clipLow[0] + model.m_clipLow[1] + model.m_clipHigh[0] + model.m_clipHigh[1] +
                        clippedLow + clippedHigh )
                        throw std::runtime_error( "unexpected source clipping" );
                    std::vector<std::array<uint16_t, 2>> adc( count );
                    for( size_t n = 0; n < count; ++n )
                        adc[n] = recording[n].m_adc;
                    std::vector<experiment> runs;
                    if( !noiseCase )
                        for( int k = 0; k <= 10; ++k )
                            if( !analytic || k == 0 || k == 10 )
                            {
                                experiment r;
                                r.m_gain  = 1;
                                r.m_name  = "gain" + std::to_string( k );
                                r.m_gain  = k / 10.;
                                r.m_trace = k == 10;
                                runs.push_back( r );
                            }
                    if( noiseCase )
                    {
                        experiment r;
                        r.m_gain = 1;
                        r.m_name = "nominal";
                        runs.push_back( r );
                    }
                    if( !noiseCase )
                    {
                        experiment oracle;
                        oracle.m_gain   = 1;
                        oracle.m_name   = "oracle";
                        oracle.m_oracle = true;
                        oracle.m_trace  = true;
                        runs.push_back( oracle );
                        experiment matched;
                        matched.m_gain    = 1;
                        matched.m_name    = "matched";
                        matched.m_horizon = 1750000;
                        runs.push_back( matched );
                        matched.m_name   = "matched_oracle";
                        matched.m_oracle = true;
                        runs.push_back( matched );
                    }
                    if( !analytic && population == 2 && !noiseCase )
                    {
                        for( const std::string fault : { "single", "cluster", "invalid", "stale", "reset", "restart" } )
                        {
                            experiment r;
                            r.m_gain  = 1;
                            r.m_name  = fault;
                            r.m_fault = fault;
                            r.m_trace = fault == "single" || fault == "stale";
                            runs.push_back( r );
                            if( fault == "single" || fault == "cluster" )
                            {
                                r.m_gap = true;
                                r.m_name += "_optional";
                                runs.push_back( r );
                            }
                        }
                        for( const std::string fault : { "wrong_sign",
                                                         "delay5ms",
                                                         "delay20ms",
                                                         "slow50hz",
                                                         "gain_error",
                                                         "rotation_error",
                                                         "limited",
                                                         "horizon0",
                                                         "ideal_dm" } )
                        {
                            experiment r;
                            r.m_gain  = 1;
                            r.m_name  = fault;
                            r.m_trace = fault == "wrong_sign" || fault == "delay5ms";
                            if( fault == "wrong_sign" )
                                r.m_calibrationGain = -1;
                            if( fault == "delay5ms" )
                                r.m_delay = 5000000;
                            if( fault == "delay20ms" )
                                r.m_delay = 20000000;
                            if( fault == "slow50hz" )
                                r.m_bandwidth = 50;
                            if( fault == "gain_error" )
                                r.m_calibrationGain = 1.2;
                            if( fault == "rotation_error" )
                                r.m_calibrationAngle = pi / 12;
                            if( fault == "limited" )
                                r.m_bound = .001;
                            if( fault == "horizon0" )
                                r.m_horizon = 0;
                            if( fault == "ideal_dm" )
                                r.m_bandwidth = 0;
                            runs.push_back( r );
                        }
                    }
                    std::map<std::tuple<int64_t, std::string, bool>, replay> cache;
                    for( const auto &run : runs )
                    {
                        const auto key = std::make_tuple( run.m_horizon, run.m_fault, run.m_gap );
                        auto       it  = cache.find( key );
                        if( it == cache.end() )
                            it = cache.emplace( key, predict( adc, count, frequencies, sensor, run ) ).first;
                        evaluate( directory, label, recording, it->second, frequencies.size(), run, summary );
                        std::cout << label << " " << run.m_name << '\n';
                    }
                }
        summary.close();
        if( !summary.good() || !metadata.good() )
            throw std::runtime_error( "failed cancellation summary export" );
    }
    catch( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
