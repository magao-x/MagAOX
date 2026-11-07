/** \file accelFFCtrlClosedLoop_export.cpp
 * \brief Actual accelSim ADC, harmonic calibration and existing AO feedback through a delayed numerical DM.
 * \ingroup accelFFCtrl_unit_test
 */
#include "accelFFCtrlAO_test.hpp"
#include "accelFFCtrlPhase_test.hpp"
#include "../offline/accelFFCtrlShadow.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace MagAOX::app::accelFFCtrlDetail;
using namespace MagAOX::app::accelFFCtrlOffline;
using namespace libXWCTest::accelFFCtrlTest;
namespace sim = MagAOX::app::accelSimDetail;

/// \cond ACCELFFCTRL_CLOSED_LOOP_FIXTURE
namespace
{
constexpr int64_t origin = 1000000000, acquisitionPeriod = 125000, queryPeriod = 500000, horizon = 1000000;
using frame = std::array<double, 4>; ///< Current WFS disturbance radians and exact integer ADC0/1 codes, as doubles.

/// Use the simulator's existing physical ADC conversion and two independent sensor axes without rescaling counts.
sim::parameters sensorSettings()
{
    sim::parameters result;
    result.m_fps = 8000;
    return result;
}

/// Construct unchanged numerical estimators, including real ADC decoding, cold start and time-validity guards.
std::array<harmonicEstimator, 2> estimators( double lower /**< [in] source-defined lower carrier Hz */ )
{
    const auto source = sensorSettings();
    parameters p;
    p.m_sampleRate   = source.m_fps;
    p.m_frequency    = lower;
    p.m_predictionNs = horizon;
    p.m_leverArm     = source.m_leverArm;
    p.m_zero.fill( source.m_zero );
    p.m_voltsPerBit.fill( source.m_voltsPerBit );
    p.m_sensitivity.fill( source.m_sensitivity );
    harmonicEstimator first( p );
    p.m_frequency = 56;
    return { first, harmonicEstimator( p ) };
}

/// Feed only the current raw pair after the query; neither disturbance truth nor WFS enters the harmonic estimator.
void consume( std::array<harmonicEstimator, 2> &input /**< [in,out] actual fixed-storage estimators */,
              const frame                      &value /**< [in] owned current simulated sample */,
              size_t                            n /**< [in] exact acquisition index */ )
{
    adcSample sample;
    sample.m_adc        = { uint16_t( value[2] ), uint16_t( value[3] ) };
    sample.m_timeNs     = origin + n * acquisitionPeriod;
    sample.m_sequence   = n + 1;
    sample.m_generation = 1;
    for( auto &estimator : input )
        if( !estimator.consume( sample, sample.m_timeNs ) )
            throw std::runtime_error( "unexpected simulated ADC ingestion failure" );
}

/// Generate one immutable forcing record and earlier-only calibration rows with the existing accelSim model.
void prepare( const std::filesystem::path &directory /**< [in] local output directory */,
              int                          seconds /**< [in] total simulated duration */,
              bool     full /**< [in] use the unchanged 214-mode population, otherwise two isolated modes */,
              uint64_t seed /**< [in] deterministic source seed */ )
{
    auto config        = sensorSettings();
    config.m_seed      = seed;
    const double lower = full ? 28 : 28.5;
    const auto   modes = full ? sim::referenceModes( 123 )
                              : std::vector<sim::modeSpec>{ { lower, .002, .99999, 1, 1 }, { 56, .002, .99999, 1, 1 } };
    sim::model   source;
    source.configure( config, modes );
    std::vector<frame> frames( size_t( seconds ) * 8000 );
    for( auto &row : frames )
    {
        const auto &sample = source.step();
        row                = { sample.m_wfsPointing[0] * sim::arcsecToRad,
                               sample.m_wfsPointing[1] * sim::arcsecToRad,
                               double( sample.m_adc[0] ),
                               double( sample.m_adc[1] ) };
    }
    if( source.m_ticks != frames.size() ||
        source.m_clipLow[0] + source.m_clipLow[1] + source.m_clipHigh[0] + source.m_clipHigh[1] )
        throw std::runtime_error( "unexpected simulator advancement or ADC clipping" );
    std::filesystem::create_directories( directory );
    std::ofstream binary( directory / "source.bin", std::ios::binary );
    binary.write( reinterpret_cast<const char *>( frames.data() ), frames.size() * sizeof( frame ) );
    binary.close();
    std::ofstream metadata( directory / "source.txt" );
    metadata << seconds << ' ' << full << ' ' << seed << ' ' << lower << ' ' << modes.size() << '\n';
    metadata.close();
    std::ofstream rows( directory / "training.csv" );
    rows << std::setprecision( 17 ) << "query_ns,sample_ns,target_ns,q0_cos,q0_sin,q1_cos,q1_sin,tip_rad,tilt_rad\n";
    auto                        input = estimators( lower );
    const std::array<double, 2> frequencies{ lower, 56 };
    // Later labels may complete a training pair only before the one-third calibration boundary.
    const size_t fitEnd = frames.size() / 3;
    for( size_t n = 0; n < fitEnd; ++n )
    {
        const int64_t now = origin + n * acquisitionPeriod;
        if( n % 4 == 0 && n + horizon / acquisitionPeriod < fitEnd )
        {
            std::array<prediction, 2> p{ input[0].predict( now ), input[1].predict( now ) };
            if( p[0].m_state == estimatorState::ready && p[1].m_state == estimatorState::ready )
            {
                rows << now << ',' << p[0].m_sampleTimeNs << ',' << now + horizon;
                for( size_t band = 0; band < 2; ++band )
                {
                    const auto q = quadratures( p[band], frequencies[band], band, config.m_leverArm );
                    rows << ',' << q[0] << ',' << q[1];
                }
                const auto &label = frames[n + horizon / acquisitionPeriod];
                rows << ',' << label[0] << ',' << label[1] << '\n';
            }
        }
        consume( input, frames[n], n );
    }
    rows.close();
    if( !binary.good() || !metadata.good() || !rows.good() )
        throw std::runtime_error( "failed source/calibration export" );
}

/// Existing synthetic integrator and feed-forward controls; all scenarios are fixed before held-out scoring.
struct scenario
{
    const char *m_name;                             ///< Stable diagnostic case name.
    double      m_aoGain, m_ffGain;                 ///< WFS feedback gain and bounded opposing feed-forward gain.
    bool        m_off{ false }, m_reverse{ false }; ///< Controlled FF disable or deliberately wrong FF sign.
};

/// Run genuine feedback: current command uses only previous corrected WFS; the next WFS includes delayed commands.
void simulate( const std::vector<frame> &frames /**< [in] common immutable exogenous disturbance and raw ADC */,
               double                    lower /**< [in] lower source carrier Hz */,
               const std::array<phaseCalibration, 2> &calibration /**< [in] frozen simulation-only weights */,
               const std::array<double, 2>           &sigma /**< [in] earlier-only normalization in WFS radians */,
               const scenario                        &run /**< [in] fixed test controls */,
               const std::filesystem::path           &path /**< [in] local CSV output */ )
{
    auto            input   = estimators( lower );
    Eigen::Matrix2d mapping = Eigen::Matrix2d::Zero(), response = mapping;
    for( size_t axis = 0; axis < 2; ++axis )
    {
        mapping( axis, axis )  = 1 / sigma[axis];
        response( axis, axis ) = sigma[axis];
    }
    syntheticIntegrator  ao( mapping, run.m_aoGain );
    correctionParameters controls;
    controls.m_horizonNs = horizon;
    controls.m_maxModal  = { 3, 3 };
    shadowCommand disabled( Eigen::Matrix2d::Identity(), mapping, 16, shadowCalibration::synthetic, controls );
    controls.m_gain = run.m_ffGain;
    shadowCommand active( Eigen::Matrix2d::Identity(), mapping, 16, shadowCalibration::synthetic, controls );
    dmResponse    plant( response, queryPeriod, 2 * queryPeriod, 0 ); // Exact aoSim lag=1: two control frames.
    const int64_t start = origin + ( frames.size() / 3 ) * acquisitionPeriod;
    const int64_t off   = origin + ( 3 * frames.size() / 4 ) * acquisitionPeriod;
    const std::array<double, 2> frequencies{ lower, 56 };
    std::array<double, 2>       previous{};
    int64_t                     previousTime = 0;
    std::ofstream               out( path );
    out << std::setprecision( 17 )
        << "query_ns,prior_wfs_ns,sample_ns,target_ns,counter,ready,ff_enabled,ff_state,ff_scale,total_scale,"
           "disturbance_tip,disturbance_tilt,residual_tip,residual_tilt,effect_tip,effect_tilt,"
           "ao0,ao1,ff0,ff1,total0,total1,prior_tip,prior_tilt\n";
    for( size_t n = 0; n < frames.size(); ++n )
    {
        const int64_t now = origin + n * acquisitionPeriod;
        if( n % 4 == 0 )
        {
            std::array<prediction, 2> p;
            for( size_t band = 0; band < 2; ++band )
            {
                const auto raw = input[band].predict( now );
                if( raw.m_state == estimatorState::ready &&
                    ( raw.m_sampleTimeNs >= now || raw.m_targetTimeNs != now + horizon ) )
                    throw std::runtime_error( "noncausal harmonic query" );
                p[band] = calibrated( raw, frequencies[band], calibration[band], sensorSettings().m_leverArm );
                if( now < start )
                    p[band].m_state =
                        estimatorState::warming; // Parameters are not available before training completes.
            }
            const bool enabled = now >= start && run.m_ffGain > 0 && !( run.m_off && now >= off );
            const auto record  = ( enabled ? active : disabled ).propose( p, now );
            const auto rawAO   = now >= start ? ao.update( previous ) : Eigen::Vector2d::Zero().eval();
            auto       ff      = record.m_proposal.m_modal;
            if( run.m_reverse )
                for( auto &value : ff )
                    value = -value;
            Eigen::Vector2d total = rawAO + Eigen::Vector2d( ff[0], ff[1] );
            double          scale = 1;
            for( size_t axis = 0; axis < 2; ++axis )
                if( std::abs( total[axis] ) > 3 )
                    scale = std::min( scale, 3 / std::abs( total[axis] ) );
            total *= scale; // Existing common-scalar combined-channel bound; no anti-windup is added.
            if( !rawAO.allFinite() || !total.allFinite() )
                throw std::runtime_error( "nonfinite closed-loop command" );
            const auto                  effect = plant.step( { total[0], total[1] } );
            const std::array<double, 2> residual{ frames[n][0] + effect[0], frames[n][1] + effect[1] };
            out << now << ',' << previousTime << ',' << p[0].m_sampleTimeNs << ',' << now + horizon << ','
                << p[0].m_sequence << ',' << record.m_validPrediction << ',' << enabled << ','
                << int( record.m_proposal.m_state ) << ',' << record.m_proposal.m_scale << ',' << scale << ','
                << frames[n][0] << ',' << frames[n][1] << ',' << residual[0] << ',' << residual[1] << ',' << effect[0]
                << ',' << effect[1] << ',' << rawAO[0] << ',' << rawAO[1] << ',' << ff[0] << ',' << ff[1] << ','
                << total[0] << ',' << total[1] << ',' << previous[0] << ',' << previous[1] << '\n';
            previous     = residual; // The corrected measurement becomes the next feedback input, never source truth.
            previousTime = now;
        }
        consume( input, frames[n], n );
    }
    if( input[0].samples() != frames.size() || input[1].samples() != frames.size() )
        throw std::runtime_error( "not every accelerometer frame was consumed" );
    out.close();
    if( !out.good() )
        throw std::runtime_error( "failed closed-loop output" );
}

/// Read a prepared forcing fixture and run predeclared closed-loop comparisons with frozen coefficients.
void run( const std::filesystem::path &directory /**< [in] prepared source and calibration directory */ )
{
    int           seconds;
    bool          full;
    uint64_t      seed;
    double        lower;
    size_t        modes;
    std::ifstream metadata( directory / "source.txt" );
    metadata >> seconds >> full >> seed >> lower >> modes;
    if( !metadata || seconds < 6 || seconds > 180 || lower != ( full ? 28 : 28.5 ) )
        throw std::invalid_argument( "invalid source metadata" );
    std::vector<frame> frames( size_t( seconds ) * 8000 );
    std::ifstream      binary( directory / "source.bin", std::ios::binary );
    binary.read( reinterpret_cast<char *>( frames.data() ), frames.size() * sizeof( frame ) );
    if( !binary || binary.peek() != std::char_traits<char>::eof() )
        throw std::invalid_argument( "invalid source fixture size" );
    for( const auto &f : frames )
        for( size_t col = 0; col < 4; ++col )
            if( !std::isfinite( f[col] ) ||
                ( col >= 2 && ( f[col] < 0 || f[col] > 4095 || f[col] != std::floor( f[col] ) ) ) )
                throw std::invalid_argument( "invalid source sample" );
    std::ifstream         weights( directory / "calibration.txt" );
    std::string           version;
    int64_t               predictionHorizon;
    double                lever;
    std::array<double, 2> frequencies, sigma;
    weights >> version >> predictionHorizon >> lever >> frequencies[0] >> frequencies[1] >> sigma[0] >> sigma[1];
    if( !weights || version != "accelFFCtrl-sim-phase-v1" || predictionHorizon != horizon || lever != 10 ||
        frequencies != std::array<double, 2>{ lower, 56 } || !std::isfinite( sigma[0] ) || !std::isfinite( sigma[1] ) ||
        sigma[0] <= 0 || sigma[1] <= 0 )
        throw std::invalid_argument( "incompatible simulation calibration" );
    std::array<phaseCalibration, 2> calibration;
    for( size_t band = 0; band < 2; ++band )
    {
        calibration[band].m_axis = band;
        for( size_t row = 0; row < 2; ++row )
            for( size_t col = 0; col < 2; ++col )
                weights >> calibration[band].m_weights( row, col );
        if( !weights || !calibration[band].m_weights.allFinite() )
            throw std::invalid_argument( "invalid simulation phase weights" );
    }
    std::string extra;
    if( weights >> extra )
        throw std::invalid_argument( "extra simulation calibration field" );
    const std::array<scenario, 10> cases{ { { "none", 0, 0 },
                                            { "wfs_only", .1, 0 },
                                            { "harmonic_only", 0, 1 },
                                            { "wfs_harmonic", .1, 1 },
                                            { "shadow_disabled", .1, 0 },
                                            { "ff_off", .1, 1, true },
                                            { "wrong_sign", .1, 1, false, true },
                                            { "wfs_only_gain03", .3, 0 },
                                            { "wfs_harmonic_gain03", .3, 1 },
                                            { "high_gain08", .8, 1 } } };
    for( const auto &experiment : cases )
        simulate(
            frames, lower, calibration, sigma, experiment, directory / ( std::string( experiment.m_name ) + ".csv" ) );
}
} // namespace
/// \endcond

/// Prepare immutable simulator data or execute the existing controllers in a local numerical feedback loop.
int main( int    argc /**< [in] argument count */,
          char **argv /**< [in] prepare directory seconds dual|full seed, or run directory */ )
try
{
    if( argc == 6 && std::string( argv[1] ) == "prepare" )
    {
        const int         seconds    = std::stoi( argv[3] );
        const std::string population = argv[4];
        if( seconds < 6 || seconds > 180 || ( population != "dual" && population != "full" ) )
            throw std::invalid_argument( "unsupported source settings" );
        prepare( argv[2], seconds, population == "full", std::stoull( argv[5] ) );
    }
    else if( argc == 3 && std::string( argv[1] ) == "run" )
        run( argv[2] );
    else
        throw std::invalid_argument( "usage: prepare directory seconds dual|full seed, or run directory" );
    return 0;
}
catch( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
