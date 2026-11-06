/** \file accelFFCtrlAO_export.cpp
 * \brief Replay recorded offline feed-forward proposals against a synthetic AO loop and existing DM model.
 * \ingroup accelFFCtrl_unit_test
 */
#include "accelFFCtrlAO_test.hpp"

#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace libXWCTest::accelFFCtrlTest;
using namespace cancellation;

/// \cond ACCELFFCTRL_AO_EXPORT
namespace
{
/// Explicit test scenarios; no values below represent measured MagAO-X loop settings.
struct scenario
{
    const char *m_name; ///< Stable report identity.
    double      m_aoGain{ 0 }, m_ffScale{ 0 }, m_limit{ .03 },
        m_antiWindup{ 0 };      ///< Synthetic gain, duplicate/sign and headroom cases.
    int64_t m_delay{ 1000000 }; ///< Synthetic actual plant delay, independently of the recorded FF prediction horizon.
    bool    m_logShadow{ false }, m_accumulate{ false }, m_disable{ false },
        m_resetAO{ false }; ///< Isolated interaction controls.
};

/// Complete numerical summary; hash compares every AO command and residual, not only aggregate RMS.
struct result
{
    std::array<double, 2> m_before{}, m_after{}, m_sumBefore{}, m_sumAfter{}; ///< Full-rate squared/signed moments.
    double                m_integratorPeak{ 0 }, m_totalPeak{ 0 },
        m_transientPeak{ 0 }; ///< Internal/output/transient coefficient or angle peaks.
    uint64_t m_saturated{ 0 },
        m_hash{ 1469598103934665603ULL }; ///< Saturated query count and exact trajectory fingerprint.
};

/// Run the same prerecorded causal FF commands through a delayed plant and independently specified AO loop.
result
simulate( const std::vector<std::array<double, 16>> &input /**< [in] prior offline cancellation trace */,
          const scenario                            &s /**< [in] synthetic interaction settings */,
          std::ostream                              &out /**< [in,out] host-file decimated WFS/command diagnostic */,
          std::vector<std::array<double, 4>>        *capture /**< [out] optional full-rate AO-only trajectory */,
          const std::vector<std::array<double, 4>>  *reference /**< [in] optional exact no-interference reference */ )
{
    const calibration     cal;
    syntheticIntegrator   ao( cal.m_mapping, s.m_aoGain );
    dmResponse            plant( cal.m_plant, period, s.m_delay, 300 );
    std::array<double, 2> priorResidual{};
    Eigen::Vector2d       ff = Eigen::Vector2d::Zero(), integratedFF = ff, held = ff, aoCommand = ff, shadowModal = ff;
    result                stats;
    out << std::setprecision( 17 )
        << "time_s,truth_x,truth_y,residual_x,residual_y,ao0,ao1,ff0,ff1,total0,total1,saturated,shadow0,shadow1\n";
    for( size_t n = 0; n < input.size(); ++n )
    {
        const auto &frame     = input[n];
        bool        saturated = false;
        if( n % 4 == 0 )
        {
            if( s.m_resetAO && n == input.size() / 2 )
                ao.reset();
            aoCommand =
                ao.update( priorResidual ); // Strictly previous WFS sample: no current/future residual feedback.
            const Eigen::Vector2d proposedFF( frame[7], frame[8] );
            shadowModal = s.m_logShadow ? proposedFF : Eigen::Vector2d::Zero();
            if( s.m_logShadow && !proposedFF.allFinite() )
                throw std::runtime_error( "invalid recorded shadow proposal" );
            ff = s.m_ffScale * proposedFF;
            if( s.m_disable && n >= input.size() / 2 )
                ff.setZero();
            if( s.m_accumulate )
            {
                integratedFF += ff;
                ff = integratedFF;
            }
            const Eigen::Vector2d total = aoCommand + ff;
            double                scale = 1;
            for( size_t axis = 0; axis < 2; ++axis )
                if( std::abs( total[axis] ) > s.m_limit )
                    scale = std::min( scale, s.m_limit / std::abs( total[axis] ) );
            held      = scale * total;
            saturated = scale < 1;
            stats.m_saturated += saturated;
            stats.m_integratorPeak = std::max( stats.m_integratorPeak, aoCommand.cwiseAbs().maxCoeff() );
            stats.m_totalPeak      = std::max( stats.m_totalPeak, held.cwiseAbs().maxCoeff() );
            ao.track( total - held, s.m_antiWindup );
        }
        const auto                  effect = plant.step( { held[0], held[1] } );
        const std::array<double, 2> residual{ frame[1] + effect[0], frame[2] + effect[1] };
        const std::array<double, 4> trajectory{ aoCommand[0], aoCommand[1], residual[0], residual[1] };
        if( capture )
            capture->push_back( trajectory );
        if( reference && reference->at( n ) != trajectory )
            throw std::runtime_error( "shadow changed a full-rate AO sample" );
        for( size_t axis = 0; axis < 2; ++axis )
        {
            if( n >= 32000 )
            {
                stats.m_before[axis] += frame[1 + axis] * frame[1 + axis];
                stats.m_after[axis] += residual[axis] * residual[axis];
                stats.m_sumBefore[axis] += frame[1 + axis];
                stats.m_sumAfter[axis] += residual[axis];
            }
            if( n >= input.size() / 2 && n < input.size() / 2 + 8000 )
                stats.m_transientPeak = std::max( stats.m_transientPeak, std::abs( residual[axis] ) );
            for( double value : { aoCommand[axis], residual[axis] } )
            {
                stats.m_hash ^= std::bit_cast<uint64_t>( value );
                stats.m_hash *= 1099511628211ULL;
            }
        }
        if( n % 4 == 0 )
        {
            // The WFS measurement is available only after the current plant evaluation, for the next query.
            priorResidual = residual;
            out << frame[0] << ',' << frame[1] << ',' << frame[2] << ',' << residual[0] << ',' << residual[1] << ','
                << aoCommand[0] << ',' << aoCommand[1] << ',' << ff[0] << ',' << ff[1] << ',' << held[0] << ','
                << held[1] << ',' << saturated << ',' << shadowModal[0] << ',' << shadowModal[1] << '\n';
        }
    }
    return stats;
}
} // namespace
/// \endcond

/// Read only a prior synthetic offline trace; arguments are input binary and local output directory.
int main( int argc /**< [in] argument count */, char **argv /**< [in] prior trace and output directory */ )
{
    try
    {
        if( argc != 3 )
            throw std::invalid_argument( "usage: accelFFCtrlAO_export prior-offline-trace.bin output-directory" );
        std::ifstream source( argv[1], std::ios::binary | std::ios::ate );
        const auto    bytes = source.tellg();
        if( bytes <= 0 || bytes % std::streamoff( 16 * sizeof( double ) ) )
            throw std::invalid_argument( "incomplete offline trace" );
        std::vector<std::array<double, 16>> frames( size_t( bytes ) / ( 16 * sizeof( double ) ) );
        source.seekg( 0 );
        source.read( reinterpret_cast<char *>( frames.data() ), bytes );
        if( !source || frames.size() < 64000 || frames.size() % 8 )
            throw std::invalid_argument( "expected >=8 seconds of 8k trace" );
        for( size_t n = 0; n < frames.size(); ++n )
        {
            if( frames[n][0] != n * period * 1e-9 )
                throw std::invalid_argument( "wrong offline trace time grid" );
            for( double value : frames[n] )
                if( !std::isfinite( value ) )
                    throw std::invalid_argument( "nonfinite offline trace" );
        }
        const std::filesystem::path directory = argv[2];
        std::filesystem::create_directories( directory );
        std::ofstream summary( directory / "summary.csv" );
        summary
            << std::setprecision( 17 )
            << "case,axis,ao_gain,ff_scale,delay_ns,limit,anti_windup,saturation_queries,integrator_peak,total_peak,"
               "before_rms_rad,after_rms_rad,variance_reduction_pct,transition_peak_rad,trajectory_hash\n";
        const std::array<scenario, 15> scenarios{
            { { "ff_only", 0, 1 },
              { "ao_only", .1, 0 },
              { "shadow_only", .1, 0, .03, 0, 1000000, true },
              { "ao_ff", .1, 1 },
              { "ao_gain03", .3, 0 },
              { "ao_ff_gain03", .3, 1 },
              { "duplicate_ff", .1, 2 },
              { "wrong_sign", .1, -1 },
              { "incremental_error", .1, 1, .03, 0, 1000000, false, true },
              { "saturation", .3, 1, .003 },
              { "back_calculation", .3, 1, .003, .5 },
              { "extra_latency", .3, 1, .03, 0, 3000000 },
              { "high_ao_gain", .8, 1 },
              { "ff_disable", .1, 1, .03, 0, 1000000, false, false, true },
              { "ao_reset", .1, 1, .03, 0, 1000000, false, false, false, true } } };
        uint64_t                           baseline = 0;
        std::vector<std::array<double, 4>> baselineSamples;
        baselineSamples.reserve( frames.size() );
        for( const auto &s : scenarios )
        {
            std::ofstream output( directory / ( std::string( s.m_name ) + ".csv" ) );
            const auto    r = simulate( frames,
                                        s,
                                        output,
                                        std::string( s.m_name ) == "ao_only" ? &baselineSamples : nullptr,
                                        std::string( s.m_name ) == "shadow_only" ? &baselineSamples : nullptr );
            output.close();
            if( !output.good() )
                throw std::runtime_error( "failed AO trace export" );
            if( std::string( s.m_name ) == "ao_only" )
                baseline = r.m_hash;
            if( std::string( s.m_name ) == "shadow_only" && r.m_hash != baseline )
                throw std::runtime_error( "diagnostic shadow affected AO trajectory" );
            const double count = frames.size() - 32000;
            for( size_t axis = 0; axis < 2; ++axis )
                summary << s.m_name << ',' << axis << ',' << s.m_aoGain << ',' << s.m_ffScale << ',' << s.m_delay << ','
                        << s.m_limit << ',' << s.m_antiWindup << ',' << r.m_saturated << ',' << r.m_integratorPeak
                        << ',' << r.m_totalPeak << ',' << std::sqrt( r.m_before[axis] / count ) << ','
                        << std::sqrt( r.m_after[axis] / count ) << ','
                        << 100 * ( 1 - ( r.m_after[axis] / count - std::pow( r.m_sumAfter[axis] / count, 2 ) ) /
                                           ( r.m_before[axis] / count - std::pow( r.m_sumBefore[axis] / count, 2 ) ) )
                        << ',' << r.m_transientPeak << ',' << r.m_hash << '\n';
        }
        summary.close();
        if( !summary.good() )
            throw std::runtime_error( "failed AO summary export" );
    }
    catch( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
