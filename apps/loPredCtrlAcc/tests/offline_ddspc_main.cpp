/** \file offline_ddspc_main.cpp
 * \brief Standalone numerical experiment CLI; optional new-directory CSV artifacts, never a native application.
 */
#include "offline_ddspc.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <type_traits>

namespace
{
using namespace libXWCTest::loPredCtrlAccTest;

/// Parse one complete scalar argument, rejecting trailing characters.
template <typename T>
T number( const std::string &value /**< [in] Complete scalar text. */ )
{
    std::istringstream input( value );
    T                  result{};
    if constexpr( std::is_unsigned_v<T> )
    {
        if( value.find( '-' ) != std::string::npos )
        {
            throw std::invalid_argument( "Negative unsigned argument" );
        }
    }
    if( !( input >> result ) || !( input >> std::ws ).eof() )
    {
        throw std::invalid_argument( "Invalid numeric argument: " + value );
    }
    return result;
}

/// Emit exact scenario parameters and evaluation summaries to the selected output.
void metrics( std::ostream         &out /**< [out] Console or new private file. */,
              const AdcDdspcConfig &config /**< [in] Reproducible configuration. */,
              const AdcDdspcResult &result /**< [in] Synthetic trajectories. */ )
{
    out << std::setprecision( 12 );
    out << "# synthetic only; ADC8000Hz; WFS2000Hz; conditional forecast RMSE; no wall-clock pacing\n";
    out << "# train=" << config.m_train << ",evaluate=" << config.m_evaluate
        << ",history=" << config.m_controller.m_history << ",future=" << config.m_controller.m_future
        << ",accel_history=" << config.m_controller.m_accelHistory << ",qrd=" << config.m_controller.m_useQrd
        << ",gamma=" << config.m_controller.m_gamma << ",prior=" << config.m_controller.m_prior
        << ",lambda=" << config.m_controller.m_lambda << ",coupling=" << config.m_coupling << ",pole=" << config.m_pole
        << ",dm_gain=" << config.m_dmGain << ",limit=" << config.m_policy.m_limit << ",seed=" << config.m_adc.m_seed
        << ",jitter_ns=" << config.m_adc.m_jitterNs << ",drop_every=" << config.m_adc.m_dropEvery
        << ",drop_start=" << config.m_adc.m_dropStart << ",drop_count=" << config.m_adc.m_dropCount
        << ",drop_period=" << config.m_adc.m_dropPeriod << ",adc_drift_ppm=" << config.m_adc.m_clockDriftPpm
        << ",latency_ns=" << config.m_adc.m_latencyNs << ",latency_jitter_ns=" << config.m_adc.m_latencyJitterNs
        << ",uncorrelated_adc=" << config.m_uncorrelatedAdc << ",force_noise=" << config.m_adc.m_forceNoise
        << ",sensor_noise=" << config.m_adc.m_sensorNoise << ",wfs_noise=" << config.m_wfsNoise << '\n';
    out << "# low_hz=" << config.m_adc.m_lowHz << ",high_hz=" << config.m_adc.m_highHz
        << ",low_amplitude_xy=" << config.m_adc.m_lowAmplitude[0] << '/' << config.m_adc.m_lowAmplitude[1]
        << ",high_amplitude_xy=" << config.m_adc.m_highAmplitude[0] << '/' << config.m_adc.m_highAmplitude[1]
        << ",train_probe=" << config.m_trainProbe << ",eval_probe=" << config.m_evalProbe
        << "; accelSim12 zero2048, volts/count.0083799, sensitivity1.02, polarity+1; causal ZOH timestamp windows\n";
    out << "# recovery_frames=" << config.m_policy.m_recoveryFrames << ",max_step=" << config.m_policy.m_maxStep
        << ",window_ns=" << config.m_features.m_windowNs << ",window_lag_ns=" << config.m_features.m_lagNs
        << ",feature_max_age_ns=" << config.m_features.m_maxAgeNs
        << ",feature_max_gap_ns=" << config.m_features.m_maxGapNs << ",wfs_invalid_start=" << config.m_wfsInvalidStart
        << ",wfs_invalid_count=" << config.m_wfsInvalidCount
        << "; safe policy ramps toward zero without exploration; forecast RMSE aggregates requested-model segments\n";
    out << "model,variance,variance_ratio_to_open,conditional_rmse,updates,predictions,commands,blocked_adc_frames,"
           "numerical_fault_frames,gate_violations,first_ready,last_ready,theta_norm,max_command,adc_samples,adc_"
           "dropped,adc_rate_hz,command_rms,power_28_29,power_54_58,fallback_ms,safe_ms,longest_fallback_ms,max_switch_"
           "step\n";
    const char *names[]{ "open", "standard", "assisted" };
    for( int i = 0; i < 3; ++i )
    {
        const auto &run = result.m_runs[i];
        out << names[i] << ',' << run.m_variance << ','
            << ( result.m_runs[0].m_variance ? run.m_variance / result.m_runs[0].m_variance : 0 ) << ',';
        if( run.m_predictions )
        {
            out << run.m_predictionRmse;
        }
        else
        {
            out << "NA";
        }
        out << ',' << run.m_updates << ',' << run.m_predictions << ',' << run.m_commands << ',' << run.m_blocked << ','
            << run.m_faults << ',' << run.m_gateViolations << ',' << run.m_firstReady << ',' << run.m_lastReady << ','
            << run.m_thetaNorm << ',' << run.m_maxCommand << ',' << run.m_adcStatus.m_samples << ','
            << run.m_adcStatus.m_dropped << ',' << run.m_adcStatus.m_rateHz << ',' << run.m_commandRms << ','
            << run.m_power28 << ',' << run.m_power56 << ',' << run.m_fallbackFrames * .5 << ',' << run.m_safeFrames * .5
            << ',' << run.m_longestFallback * .5 << ',' << run.m_maxSwitchStep << '\n';
    }
}
} // namespace

/// Parse test-only parameters, run the actual learners, and optionally export into a new directory.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Test-only options. */ )
{
    try
    {
        AdcDdspcConfig                     config;
        std::string                        output;
        std::map<std::string, std::string> args;
        for( int i = 1; i < argc; ++i )
        {
            const std::string key = argv[i];
            if( key == "--help" )
            {
                std::cout
                    << "Offline ADC-DDSPC: --train N --evaluate N --history N --future N --accel-history N\n"
                    << "--qrd 0|1 --gamma F --prior F --lambda F --coupling F --pole F --dm-gain F --limit F\n"
                    << "--seed N --jitter-ns N --drop-every N --drop-start N --drop-count N\n"
                    << "--force-noise F --sensor-noise F --wfs-noise F --output NEW_DIRECTORY\n"
                    << "--drop-period N --adc-drift-ppm F --latency-ns N --latency-jitter-ns N\n"
                    << "--recovery-frames N --max-step F --window-lag-ns N --feature-max-age-ns N\n"
                    << "--feature-max-gap-ns N --uncorrelated-adc 0|1 --wfs-invalid-start N --wfs-invalid-count N\n";
                return 0;
            }
            if( !key.starts_with( "--" ) || i + 1 >= argc || args.count( key ) )
            {
                throw std::invalid_argument( "Expected unique --option value pairs" );
            }
            args[key] = argv[++i];
        }
        for( const auto &[key, value] : args )
        {
            if( key == "--train" )
            {
                config.m_train = number<int>( value );
            }
            else if( key == "--evaluate" )
            {
                config.m_evaluate = number<int>( value );
            }
            else if( key == "--history" )
            {
                config.m_controller.m_history = number<int>( value );
            }
            else if( key == "--future" )
            {
                config.m_controller.m_future = number<int>( value );
            }
            else if( key == "--accel-history" )
            {
                config.m_controller.m_accelHistory = number<int>( value );
            }
            else if( key == "--qrd" )
            {
                const int qrd = number<int>( value );
                if( qrd != 0 && qrd != 1 )
                {
                    throw std::invalid_argument( "qrd must be0 or1" );
                }
                config.m_controller.m_useQrd = qrd;
            }
            else if( key == "--gamma" )
            {
                config.m_controller.m_gamma = number<float>( value );
            }
            else if( key == "--prior" )
            {
                config.m_controller.m_prior = number<float>( value );
            }
            else if( key == "--lambda" )
            {
                config.m_controller.m_lambda = number<float>( value );
            }
            else if( key == "--coupling" )
            {
                config.m_coupling = number<double>( value );
            }
            else if( key == "--pole" )
            {
                config.m_pole = number<double>( value );
            }
            else if( key == "--dm-gain" )
            {
                config.m_dmGain = number<double>( value );
            }
            else if( key == "--limit" )
            {
                config.m_policy.m_limit = number<double>( value );
            }
            else if( key == "--seed" )
            {
                config.m_adc.m_seed = number<std::uint64_t>( value );
            }
            else if( key == "--jitter-ns" )
            {
                config.m_adc.m_jitterNs = number<std::uint64_t>( value );
            }
            else if( key == "--drop-every" )
            {
                config.m_adc.m_dropEvery = number<std::uint64_t>( value );
            }
            else if( key == "--drop-start" )
            {
                config.m_adc.m_dropStart = number<std::uint64_t>( value );
            }
            else if( key == "--drop-count" )
            {
                config.m_adc.m_dropCount = number<std::uint64_t>( value );
            }
            else if( key == "--drop-period" )
                config.m_adc.m_dropPeriod = number<std::uint64_t>( value );
            else if( key == "--adc-drift-ppm" )
                config.m_adc.m_clockDriftPpm = number<double>( value );
            else if( key == "--latency-ns" )
                config.m_adc.m_latencyNs = number<std::int64_t>( value );
            else if( key == "--latency-jitter-ns" )
                config.m_adc.m_latencyJitterNs = number<std::int64_t>( value );
            else if( key == "--recovery-frames" )
                config.m_policy.m_recoveryFrames = number<unsigned>( value );
            else if( key == "--max-step" )
                config.m_policy.m_maxStep = number<double>( value );
            else if( key == "--window-lag-ns" )
                config.m_features.m_lagNs = number<std::int64_t>( value );
            else if( key == "--feature-max-age-ns" )
                config.m_features.m_maxAgeNs = number<std::int64_t>( value );
            else if( key == "--feature-max-gap-ns" )
                config.m_features.m_maxGapNs = number<std::int64_t>( value );
            else if( key == "--uncorrelated-adc" )
            {
                if( value != "0" && value != "1" )
                    throw std::invalid_argument( "uncorrelated-adc must be0 or1" );
                config.m_uncorrelatedAdc = value == "1";
            }
            else if( key == "--wfs-invalid-start" )
                config.m_wfsInvalidStart = number<int>( value );
            else if( key == "--wfs-invalid-count" )
                config.m_wfsInvalidCount = number<int>( value );
            else if( key == "--force-noise" )
            {
                config.m_adc.m_forceNoise = number<double>( value );
            }
            else if( key == "--sensor-noise" )
            {
                config.m_adc.m_sensorNoise = number<double>( value );
            }
            else if( key == "--wfs-noise" )
            {
                config.m_wfsNoise = number<double>( value );
            }
            else if( key == "--output" )
            {
                output = value;
            }
            else
            {
                throw std::invalid_argument( "Unknown offline option: " + key );
            }
        }
        if( !output.empty() && std::filesystem::exists( output ) )
        {
            throw std::invalid_argument( "Output directory must be new; existing results are never replaced" );
        }
        const auto result = adc_ddspc_run( config );
        metrics( std::cout, config, result );
        if( !output.empty() )
        {
            if( !std::filesystem::create_directories( output ) )
            {
                throw std::runtime_error( "Output directory appeared during simulation; refusing to overwrite" );
            }
            std::ofstream summary( std::filesystem::path( output ) / "metrics.csv" );
            metrics( summary, config, result );
            std::ofstream trace( std::filesystem::path( output ) / "residuals.csv" );
            trace << std::setprecision( 12 ) << "time_s,open_x,open_y,standard_x,standard_y,assisted_x,assisted_y\n";
            for( int i = 0; i < config.m_evaluate; ++i )
            {
                trace << double( i ) / 2000;
                for( const auto &run : result.m_runs )
                {
                    for( const auto &axis : run.m_residual )
                    {
                        trace << ',' << axis[i];
                    }
                }
                trace << '\n';
            }
            std::array<std::vector<double>, 3> spectra;
            for( int mode = 0; mode < 3; ++mode )
            {
                const auto x = periodogram( result.m_runs[mode].m_residual[0], 2000 );
                const auto y = periodogram( result.m_runs[mode].m_residual[1], 2000 );
                spectra[mode].resize( x.size() );
                for( std::size_t k = 0; k < x.size(); ++k )
                {
                    spectra[mode][k] = ( x[k] + y[k] ) / 2;
                }
            }
            std::ofstream psd( std::filesystem::path( output ) / "psd.csv" );
            psd << std::setprecision( 12 ) << "frequency_hz,open,standard,assisted\n";
            for( std::size_t k = 0; k < spectra[0].size(); ++k )
            {
                psd << double( k ) * 2000 / config.m_evaluate;
                for( const auto &spectrum : spectra )
                {
                    psd << ',' << spectrum[k];
                }
                psd << '\n';
            }
            summary.flush();
            trace.flush();
            psd.flush();
            std::ofstream learning( std::filesystem::path( output ) / "learning.csv" );
            learning << std::setprecision( 12 )
                     << "wfs_index,time_s,model,updates,rank,ready,prediction_valid,conditional_rmse,theta_norm\n";
            for( int mode = 1; mode < 3; ++mode )
            {
                for( const auto &point : result.m_runs[mode].m_learning )
                {
                    const auto &status = point.m_status;
                    learning << point.m_index << ',' << double( point.m_index ) / 2000 << ','
                             << ( mode == 1 ? "standard" : "assisted" ) << ',' << status.m_updates << ','
                             << status.m_rank << ',' << status.m_ready << ',' << status.m_predictionValid << ',';
                    if( status.m_predictionValid )
                    {
                        learning << status.m_rmse;
                    }
                    else
                    {
                        learning << "NA";
                    }
                    learning << ',' << point.m_thetaNorm << '\n';
                }
            }
            learning.flush();
            std::ofstream decisions( std::filesystem::path( output ) / "decisions.csv" );
            std::ofstream transitions( std::filesystem::path( output ) / "transitions.csv" );
            const char   *decisionHeader =
                "model,wfs_index,decision_ns,requested,effective,reason,switched,"
                "qualified_streak,common_valid,features_valid,standard_ready,assisted_ready,feature_reason,"
                "effective_ns,latest_ns,available_ns,feature_samples,accel_x,accel_y,command_x,command_y,step\n";
            decisions << std::setprecision( 12 ) << decisionHeader;
            transitions << std::setprecision( 12 ) << decisionHeader;
            for( int mode = 1; mode < 3; ++mode )
            {
                for( const auto &point : result.m_runs[mode].m_decisions )
                {
                    const auto        &d = point.m_control;
                    const auto        &a = point.m_feature;
                    std::ostringstream row;
                    row << std::setprecision( 12 ) << ( mode == 1 ? "standard" : "assisted" ) << ',' << point.m_index
                        << ',' << point.m_decisionNs << ',' << offline_mode( d.m_requested ) << ','
                        << offline_mode( d.m_effective ) << ',' << d.m_reason << ',' << d.m_switched << ','
                        << d.m_validStreak << ',' << point.m_commonValid << ',' << point.m_featuresValid << ','
                        << point.m_standardReady << ',' << point.m_assistedReady << ',' << feature_reason( a.m_reason )
                        << ',' << a.m_effectiveNs << ',' << a.m_latestNs << ',' << a.m_availableNs << ',' << a.m_samples
                        << ',' << a.m_value( 0 ) << ',' << a.m_value( 1 ) << ',' << d.m_command( 0 ) << ','
                        << d.m_command( 1 ) << ',' << d.m_step << '\n';
                    decisions << row.str();
                    if( d.m_switched || point.m_index == config.m_train )
                        transitions << row.str();
                }
            }
            decisions.flush();
            transitions.flush();
            if( !summary || !trace || !psd || !learning || !decisions || !transitions )
            {
                throw std::runtime_error( "Offline artifact write failed" );
            }
        }
    }
    catch( const std::exception &error )
    {
        std::cerr << "Offline ADC-DDSPC: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
