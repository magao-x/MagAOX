/** \file shadow_replay_benchmark.cpp
 * \brief Standalone2-kHz app/8-kHz ADC replay processing and shared-component allocation benchmarks.
 */
#include "shadow_app_test_support.hpp"
#include "allocation_probe.hpp"
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
using namespace libXWCTest::loPredCtrlAccTest;

/// Verify the extracted in-place components require no Eigen or C++ heap requests, including failed qualification.
void shared_allocation_contract()
{
    for( int modes : { 2, 3, 10 } )
    {
        std::array<DDSPC::AdcSample, 4> samples;
        for( std::size_t i = 0; i < samples.size(); ++i )
        {
            auto &s        = samples[i];
            s.m_generation = s.m_segment = 1;
            s.m_frame.m_counter          = i + 1;
            s.m_acquiredNs               = 1000000000 + i * 125000;
            s.m_publishedNs              = s.m_acquiredNs + 1000;
            s.m_receivedNs               = s.m_acquiredNs + 3000;
            s.m_acceleration             = { double( i ), -double( i ) };
        }
        DDSPC::FeatureConfig f;
        DDSPC::AccelFeature  aggregate;
        DDSPC::PolicyConfig  p;
        p.m_modes = modes;
        DDSPC::ShadowPolicy    policy( DDSPC::PolicyMode::Assisted, p );
        DDSPC::ControlEvidence evidence( modes );
        DDSPC::ControlDecision output( modes );
        auto                   previous = DDSPC::Matrix::Zero( modes, 1 ).eval(), zero = previous;
        evidence.m_standard.setConstant( -.1 );
        evidence.m_assisted.setConstant( .1 );
        const auto e = DDSPCBenchmark::eigenRequests, c = DDSPCBenchmark::cppRequests;
        DDSPCBenchmark::enabled = true;
        for( int i = 0; i < 1000; ++i )
        {
            DDSPC::acceleration_window( samples, 1000500000, f, aggregate );
            evidence.m_commonValid   = true;
            evidence.m_featureReason = aggregate.m_reason;
            evidence.m_featuresValid = i % 17 != 0;
            evidence.m_standardReady = i % 31 != 0;
            evidence.m_assistedReady = i % 13 != 0;
            policy.decide( evidence, previous, zero, output );
            previous = output.m_command;
        }
        DDSPCBenchmark::enabled = false;
        if( DDSPCBenchmark::eigenRequests != e || DDSPCBenchmark::cppRequests != c )
            throw std::runtime_error( "Shared component allocation contract failed" );
    }
}

/// Report observed processing percentiles after warmup; these are virtual-cadence workload timings, not scheduler
/// proof.
void report( const char               *profile /**< [in] Model-size identity. */,
             const char               *component /**< [in] ADC, WFS, certificate or combined-frame work. */,
             std::vector<double>       values /**< [in] Owned measured work samples. */,
             std::size_t               warmup /**< [in] Number of initial samples omitted. */,
             const ShadowReplayResult &result /**< [in] Actual ADC and allocation evidence. */,
             ReplayApplication        &app /**< [in] Candidate admission counts. */ )
{
    values.erase( values.begin(), values.begin() + std::min( warmup, values.size() ) );
    const auto misses = std::count_if( values.begin(), values.end(), []( double x ) { return x > 500; } );
    std::sort( values.begin(), values.end() );
    std::cout << profile << ",2000," << result.m_capture.m_rateHz << ',' << component << ',' << values.size();
    for( double q : { .5, .95, .99, .999, 1. } )
        std::cout << ',' << values[std::size_t( q * ( values.size() - 1 ) )];
    std::cout << ',' << misses << ',' << app.m_validCandidates << ',' << app.m_assistedCandidates << ','
              << result.m_eigenRequests << ',' << result.m_cppRequests;
    const auto standard = app.model_status( DDSPC::ShadowMode::Standard );
    const auto assisted = app.model_status( DDSPC::ShadowMode::AccelAssisted );
    if( !standard.m_predictionValid || !assisted.m_predictionValid || !std::isfinite( standard.m_rmse ) ||
        !std::isfinite( assisted.m_rmse ) )
        throw std::runtime_error( "No valid conditional forecast evidence" );
    std::cout << ',' << standard.m_updates << ',' << assisted.m_updates << ',' << standard.m_rmse << ','
              << assisted.m_rmse << '\n';
}
} // namespace

/// Run only stream-free application fixtures and numerical replay, with no install, native app or service entrypoint.
int main( int    argc /**< [in] Optional frame-count argument. */,
          char **argv /**< [in] Standalone benchmark arguments. */ )
{
    try
    {
        const auto frames = argc > 1 ? std::stoul( argv[1] ) : 2000;
        if( frames < 512 || frames > 100000 )
            throw std::invalid_argument( "Use512–100000 replay WFS frames" );
        shared_allocation_contract();
        std::cout << std::setprecision( 10 )
                  << "# Offline virtual cadence; shared in-place components: zero allocation requests at2,3,10 modes\n"
                  << "# Frame work sums actual callbacks between WFS origins; includes synchronous identification.\n"
                  << "profile,wfs_hz,adc_hz,component,n,p50_us,p95_us,p99_us,p999_us,max_us,over_500us,valid_"
                     "candidates,assisted_candidates,eigen_requests,cpp_requests,standard_updates,assisted_updates,"
                     "standard_conditional_rmse,assisted_conditional_rmse\n";
        for( bool operational : { false, true } )
        {
            auto models = replay_model_config();
            if( operational )
            {
                models.m_history      = 15;
                models.m_accelHistory = 5;
                models.m_lambda       = 100;
                models.m_gamma        = 1;
                models.m_prior        = 1e-5;
            }
            ShadowReplayConfig recording;
            recording.m_frames = frames;
            ReplayApplication app;
            app.m_trace = false;
            std::ostringstream startup;
            auto              *saved = std::cout.rdbuf( startup.rdbuf() );
            try
            {
                app.prepare( models, recording.m_inputs );
                app.train();
            }
            catch( ... )
            {
                std::cout.rdbuf( saved );
                throw;
            }
            std::cout.rdbuf( saved );
            const auto  result  = replay_shadow( app, recording );
            const char *profile = operational ? "example_h15_a5" : "small_h4_a3";
            report( profile, "adc", result.m_adcUs, 800, result, app );
            report( profile, "wfs", result.m_wfsUs, 200, result, app );
            report( profile, "confirmed_input", result.m_commandUs, 200, result, app );
            report( profile, "frame_total", result.m_frameUs, 200, result, app );
            if( !app.inhibited() || !app.outputs_rejected() || app.m_validCandidates < frames / 2 )
                throw std::runtime_error( "Replay safety/readiness benchmark failed" );
            app.appShutdown();
        }
    }
    catch( const std::exception &e )
    {
        DDSPCBenchmark::enabled = false;
        std::cerr << "Offline replay benchmark: " << e.what() << '\n';
        return 1;
    }
}
