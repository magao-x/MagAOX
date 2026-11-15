/** \file qualified_shadow_main.cpp
 * \brief Standalone synthetic native-DDSPC qualification report; no runtime, stream or instrument access.
 */
#include "qualified_shadow_replay.hpp"
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <tuple>

namespace
{
using namespace libXWCTest::loPredCtrlAccTest;

/// Parse a complete nonnegative command-line integer without accepting truncated or overflowing values.
unsigned number( const std::string &text /**< [in] Complete argument. */ )
{
    unsigned result{};
    auto [end, error] = std::from_chars( text.data(), text.data() + text.size(), result );
    if( error != std::errc() || end != text.data() + text.size() )
        throw std::invalid_argument( "Invalid unsigned integer" );
    return result;
}

/// Open a new artifact within the newly-created output directory; never a native shared-memory mapping.
std::ofstream output( const std::filesystem::path &path /**< [in] Private ordinary file. */ )
{
    std::ofstream file( path );
    file.exceptions( std::ios::badbit | std::ios::failbit );
    file << std::setprecision( 17 );
    return file;
}

/// Emit only matched targets/horizons and gap-aware Hann PSDs, with no fabricated samples or predictions.
void metrics( const QualifiedReplayResult &result /**< [in] Original native scalar scores. */,
              const std::filesystem::path &directory /**< [in] New private output. */,
              unsigned                     warmup /**< [in] Minimum origin counter for evaluation. */ )
{
    struct Pair
    {
        std::array<bool, 2>   present{}; ///< Never fill an absent model result with zero.
        std::array<double, 2> prediction{};
        std::array<double, 2> conditional{}; ///< Original independent forecasts.
        double                target{};      ///< Common measured target, checked exactly.
    };
    using Key = std::tuple<unsigned, unsigned, std::uint64_t, std::uint64_t>; // horizon/channel/origin/target
    std::map<Key, Pair> pairs;
    auto                trace = output( directory / "predictions.csv" );
    trace << "mode,origin,target,horizon,channel,prediction,measurement,late,generation,epoch,layout,origin_prediction,"
             "origin_valid\n";
    for( const auto &s : result.m_scores )
    {
        trace << unsigned( s.m_mode ) << ',' << s.m_origin << ',' << s.m_target << ',' << s.m_horizon << ','
              << s.m_channel << ',' << s.m_prediction << ',' << s.m_measurement << ',' << s.m_late << ','
              << s.m_generation << ',' << s.m_epoch << ',' << s.m_layout << ',' << s.m_originPrediction << ','
              << s.m_originValid << '\n';
        if( s.m_origin < warmup || !s.m_originValid )
            continue;
        auto      &p    = pairs[{ s.m_horizon, s.m_channel, s.m_origin, s.m_target }];
        const auto mode = unsigned( s.m_mode );
        if( mode > 1 || p.present[mode] || ( ( p.present[0] || p.present[1] ) && p.target != s.m_measurement ) )
            throw std::runtime_error( "Duplicate score or mismatched measured target" );
        p.present[mode]     = true;
        p.prediction[mode]  = s.m_originPrediction;
        p.conditional[mode] = s.m_prediction;
        p.target            = s.m_measurement;
    }
    auto table    = output( directory / "metrics.csv" );
    auto spectrum = output( directory / "psd.csv" );
    table << "horizon,channel,count,wfs_rms,standard_rmse,assisted_rmse,welch_windows,nperseg,conditional_standard_"
             "rmse,conditional_assisted_rmse\n";
    spectrum << "horizon,channel,frequency_hz,wfs_psd,standard_error_psd,assisted_error_psd\n";
    for( unsigned h = 1; h <= 8; ++h )
        for( unsigned channel = 0; channel < 2; ++channel )
        {
            std::vector<std::uint64_t>         origins;
            std::array<std::vector<double>, 3> values;
            std::array<double, 3>              energy{};
            std::array<double, 2>              conditionalEnergy{};
            for( const auto &[key, p] : pairs )
                if( std::get<0>( key ) == h && std::get<1>( key ) == channel && p.present[0] && p.present[1] )
                {
                    origins.push_back( std::get<2>( key ) );
                    for( unsigned j = 0; j < 2; ++j )
                        conditionalEnergy[j] += std::pow( p.conditional[j] - p.target, 2 );
                    const std::array<double, 3> v{ p.target, p.prediction[0] - p.target, p.prediction[1] - p.target };
                    for( unsigned j = 0; j < 3; ++j )
                    {
                        values[j].push_back( v[j] );
                        energy[j] += v[j] * v[j];
                    }
                }
            if( origins.empty() )
                continue;
            std::size_t longest = 1, span = 1;
            for( std::size_t n = 1; n < origins.size(); ++n )
            {
                span    = origins[n] == origins[n - 1] + 1 ? span + 1 : 1;
                longest = std::max( longest, span );
            }
            std::size_t size = 2048;
            while( size > longest && size > 256 )
                size /= 2;
            const auto                         step = size / 2;
            std::array<std::vector<double>, 3> power;
            unsigned                           windows = 0;
            for( std::size_t first = 0; first < origins.size(); )
            {
                std::size_t end = first + 1;
                while( end < origins.size() && origins[end] == origins[end - 1] + 1 )
                    ++end;
                for( std::size_t start = first; start + size <= end; start += step )
                {
                    for( unsigned j = 0; j < 3; ++j )
                    {
                        std::vector<double> window( values[j].begin() + start, values[j].begin() + start + size );
                        auto psd = periodogram( window, 2000, 200 ); // Existing Hann/demeaned density estimator.
                        if( power[j].empty() )
                            power[j].resize( psd.size(), 0 );
                        for( std::size_t k = 0; k < psd.size(); ++k )
                            power[j][k] += psd[k];
                    }
                    ++windows;
                }
                first = end; // Never concatenate a telemetry gap into a spectral window.
            }
            table << h << ',' << channel << ',' << origins.size();
            for( auto e : energy )
                table << ',' << std::sqrt( e / origins.size() );
            table << ',' << windows << ',' << size;
            for( auto e : conditionalEnergy )
                table << ',' << std::sqrt( e / origins.size() );
            table << '\n';
            if( windows )
                for( std::size_t k = 0; k < power[0].size(); ++k )
                    spectrum << h << ',' << channel << ',' << k * 2000.0 / size << ',' << power[0][k] / windows << ','
                             << power[1][k] / windows << ',' << power[2][k] / windows << '\n';
        }
}
} // namespace

/// Run an explicitly synthetic offline replay; arguments never name native streams, hosts, outputs or services.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Replay/report options. */ )
{
    try
    {
        QualifiedReplayConfig c;
        std::filesystem::path directory;
        unsigned              warmup = 1000;
        for( int a = 1; a < argc; ++a )
        {
            const std::string key = argv[a];
            if( a + 1 == argc )
                throw std::invalid_argument( "Every argument requires a value" );
            const std::string value = argv[++a];
            if( key == "--output" )
                directory = value;
            else if( key == "--frames" )
                c.m_frames = number( value );
            else if( key == "--seed" )
                c.m_seed = number( value );
            else if( key == "--future" )
                c.m_future = number( value );
            else if( key == "--warmup" )
                warmup = number( value );
            else if( key == "--learner" && ( value == "qr" || value == "rls" ) )
                c.m_qrd = value == "qr";
            else if( key == "--adc" && ( value == "coherent" || value == "unrelated" ) )
                c.m_uncorrelated = value == "unrelated";
            else
                throw std::invalid_argument( "Unknown replay option/value" );
        }
        if( directory.empty() || c.m_frames < 100 || c.m_frames > 20000 || warmup >= c.m_frames || c.m_future < 1 ||
            c.m_future > 8 || !std::filesystem::create_directory( directory ) )
            throw std::invalid_argument( "Specify bounded frames/horizon/warmup and a NEW --output directory" );
        const auto r = replay_qualified( c );
        metrics( r, directory, warmup );
        auto theta = output( directory / "coefficients.csv" );
        theta << "mode,row,column,value\n";
        for( unsigned mode = 0; mode < 2; ++mode )
        {
            const auto &matrix = mode ? r.m_assisted : r.m_standard;
            for( int row = 0; row < matrix.rows(); ++row )
                for( int column = 0; column < matrix.cols(); ++column )
                    theta << mode << ',' << row << ',' << column << ',' << matrix( row, column ) << '\n';
        }
        auto time = output( directory / "compute.csv" );
        time << "frame,whole_frame_us,numerical_us,max_job_us,standard_ready,assisted_ready,standard_rank,assisted_"
                "rank\n";
        for( std::size_t n = 0; n < r.m_computeUs.size(); ++n )
        {
            time << n + 1 << ',' << r.m_computeUs[n] << ',' << r.m_numericalUs[n] << ',' << r.m_maxJobUs[n];
            for( auto state : r.m_readiness[n] )
                time << ',' << state;
            time << '\n';
        }
        auto summary = output( directory / "summary.json" );
        summary << "{\"evidence\":\"synthetic_requested\",\"live_qualified\":false,\"deadline_qualified\":false,"
                << "\"modes\":2,\"history\":2,\"adc_history_frames\":1,\"command_slots\":4,"
                << "\"training_stride\":1,\"gamma\":1,\"prior\":0.001,\"lambda\":0.1,\"rank_tolerance\":0.00001,"
                   "\"warmup_origin\":"
                << warmup << ',' << "\"frames\":" << c.m_frames << ",\"seed\":" << c.m_seed
                << ",\"future\":" << c.m_future << ",\"learner\":\"" << ( c.m_qrd ? "qr" : "rls" ) << "\",\"adc\":\""
                << ( c.m_uncorrelated ? "unrelated" : "coherent" ) << "\",\"adc_samples\":" << r.m_adc.m_samples
                << ",\"adc_rate_hz\":" << r.m_adc.m_rateHz << ",\"adc_invalid\":" << r.m_adc.m_invalid
                << ",\"adc_dropped\":" << r.m_adc.m_dropped << ",\"candidates\":" << r.m_candidates
                << ",\"forecasts\":" << r.m_status.m_forecasts << ",\"score_capacity\":" << r.m_scores.capacity()
                << ",\"score_bytes\":" << r.m_scores.capacity() * sizeof( DDSPC::ShadowPrediction )
                << ",\"score_drops\":" << r.m_scoreDrops << ",\"standard_updates\":" << r.m_status.m_standard.m_updates
                << ",\"assisted_updates\":" << r.m_status.m_assisted.m_updates
                << ",\"standard_rank\":" << r.m_status.m_standard.m_rank
                << ",\"standard_features\":" << r.m_status.m_standard.m_features
                << ",\"assisted_rank\":" << r.m_status.m_assisted.m_rank
                << ",\"assisted_features\":" << r.m_status.m_assisted.m_features
                << ",\"standard_ready\":" << ( r.m_status.m_standard.m_ready ? "true" : "false" )
                << ",\"assisted_ready\":" << ( r.m_status.m_assisted.m_ready ? "true" : "false" ) << "}\n";
        std::cout << "Synthetic native DDSPC report: " << directory << "; physical/deadline qualification=false\n";
        return 0;
    }
    catch( const std::exception &e )
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
