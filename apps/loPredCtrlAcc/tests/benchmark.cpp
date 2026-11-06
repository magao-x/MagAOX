/** \file benchmark.cpp
 * \brief Offline component latency and allocation-request measurements, without streams or application startup.
 */
#include "../shadow_controller.hpp"
#include "../causal_input.hpp"
#include "causal_input_test_data.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
using DDSPC::Matrix;
volatile double sink = 0; ///< Observe numerical results so measured work cannot be removed.

/// Measure warmed component work; timer-vector allocation and output formatting are outside the measured scope.
template <class Work>
void measure( const char *name /**< [in] CSV component name. */,
              int         iterations /**< [in] Timed records after warmup. */,
              Work        work /**< [in] One component operation, accepting a monotonic local index. */ )
{
    for( int i = 0; i < 512; ++i )
        work( i );
    std::vector<double> times( iterations );
    DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = DDSPCBenchmark::cppBytes = 0;
    for( int i = 0; i < iterations; ++i )
    {
        DDSPCBenchmark::enabled = true;
        const auto start        = Clock::now();
        work( i + 512 );
        const auto end          = Clock::now();
        DDSPCBenchmark::enabled = false;
        times[i]                = std::chrono::duration<double, std::micro>( end - start ).count();
    }
    const auto eigen = DDSPCBenchmark::eigenRequests, cpp = DDSPCBenchmark::cppRequests,
               bytes = DDSPCBenchmark::cppBytes;
    const auto late  = std::count_if( times.begin(), times.end(), []( double t ) { return t > 500; } );
    std::sort( times.begin(), times.end() );
    std::cout << name << ',' << iterations;
    for( double q : { .5, .95, .99, .999, 1. } )
        std::cout << ',' << times[std::size_t( q * ( iterations - 1 ) )];
    std::cout << ',' << double( eigen ) / iterations << ',' << double( cpp ) / iterations << ','
              << double( bytes ) / iterations << ',' << late << '\n';
}

/// Verify that this Eigen version exposes the expected allocation-check seam before reporting any measurements.
bool probe_valid()
{
    DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
    DDSPCBenchmark::enabled                                     = true;
    Matrix allocated                                            = Matrix::Ones( 256, 1 );
    void  *p                                                    = ::operator new( 73 );
    sink                                                        = allocated.sum();
    ::operator delete( p );
    DDSPCBenchmark::enabled = false;
    return DDSPCBenchmark::eigenRequests == 1 && DDSPCBenchmark::cppRequests == 1;
}

/// Assert the critical-path allocation contract, including cold history, reacquisition and prediction failures.
bool check_allocations()
{
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
            for( float gamma : { 1.f, .99f } )
            {
                DDSPC::ShadowConfig cfg;
                cfg.m_modes  = modes;
                cfg.m_useQrd = qrd;
                cfg.m_gamma  = gamma;
                cfg.m_prior  = .1f;
                cfg.m_lambda = .3f;
                DDSPC::ShadowController pair( cfg );
                pair.training( DDSPC::ShadowMode::Standard, true );
                pair.training( DDSPC::ShadowMode::AccelAssisted, true );
                DDSPC::ShadowRecord record;
                record.m_residual       = Matrix::Zero( modes, 1 );
                record.m_appliedCommand = record.m_residual;
                record.m_acceleration   = Matrix::Zero( 2, 1 );
                record.m_wfsValid = record.m_appliedValid = record.m_timingValid = record.m_accelValid = true;
                std::mt19937 rng( 801 );
                Matrix       state = Matrix::Ones( 9 * modes + 40, 1 ), output = Matrix::Zero( modes, 1 );
                DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
                for( int i = 0; i < 768; ++i )
                {
                    for( auto *v : { &record.m_residual, &record.m_appliedCommand, &record.m_acceleration } )
                        for( int j = 0; j < v->size(); ++j )
                            ( *v )( j ) = ( int( rng() % 2001 ) - 1000 ) / 1000.f;
                    record.m_index = i;
                    record.m_accelValid =
                        i != 300; // Missing sensor history and common timing loss must retain storage.
                    record.m_timingValid             = i != 310;
                    const bool noTrainingAllocations = qrd || gamma == 1;
                    if( noTrainingAllocations )
                        Eigen::internal::set_is_malloc_allowed( false );
                    DDSPCBenchmark::enabled = true;
                    const bool accepted     = pair.process( record );
                    DDSPCBenchmark::enabled = false;
                    Eigen::internal::set_is_malloc_allowed( true );
                    if( accepted != record.m_timingValid )
                        return false;
                }
                const auto trainingAllocations = DDSPCBenchmark::eigenRequests;
                if( DDSPCBenchmark::cppRequests || ( ( qrd || gamma == 1 ) && trainingAllocations ) )
                    return false;
                if( !pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready )
                    return false;
                DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
                Eigen::internal::set_is_malloc_allowed( false );
                DDSPCBenchmark::enabled = true;
                bool valid = pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 767, 0, state, output );
                valid      = valid && !pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 766, 1, state, output );
                valid      = valid && !pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 768, 0, state, output );
                state( 0 ) = std::numeric_limits<float>::quiet_NaN();
                valid      = valid && !pair.predict_cached( DDSPC::ShadowMode::AccelAssisted, 767, 0, state, output );
                DDSPCBenchmark::enabled = false;
                Eigen::internal::set_is_malloc_allowed( true );
                if( !valid || DDSPCBenchmark::eigenRequests || DDSPCBenchmark::cppRequests )
                    return false;
                std::cout << "Allocation contract modes=" << modes << " qrd=" << qrd << " gamma=" << gamma
                          << " training_Eigen_requests=" << trainingAllocations << " prediction_requests=0\n";
                pair.training( DDSPC::ShadowMode::Standard, false );
                pair.training( DDSPC::ShadowMode::AccelAssisted, false );
                record.m_index                = 768;
                DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
                Eigen::internal::set_is_malloc_allowed( false );
                DDSPCBenchmark::enabled = true;
                const bool paused       = pair.process( record );
                DDSPCBenchmark::enabled = false;
                Eigen::internal::set_is_malloc_allowed( true );
                if( !paused || DDSPCBenchmark::eigenRequests || DDSPCBenchmark::cppRequests )
                    return false;
                Eigen::internal::set_is_malloc_allowed( false );
                DDSPCBenchmark::enabled = true;
                pair.reset( DDSPC::ShadowMode::Standard );
                pair.reset( DDSPC::ShadowMode::AccelAssisted );
                DDSPCBenchmark::enabled = false;
                Eigen::internal::set_is_malloc_allowed( true );
                if( DDSPCBenchmark::eigenRequests || DDSPCBenchmark::cppRequests ||
                    pair.status( DDSPC::ShadowMode::Standard ).m_ready ||
                    pair.status( DDSPC::ShadowMode::AccelAssisted ).m_ready )
                    return false;
            }
    return true;
}
} // namespace

/// Run synthetic component workloads; optional argument changes sample count, never an instrument endpoint.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional positive timed sample count. */ )
{
    if( argc == 2 && std::string( argv[1] ) == "--check-allocations" )
        return probe_valid() && check_allocations() ? 0 : 4;
    const int iterations = argc == 2 ? std::stoi( argv[1] ) : 2000;
    if( iterations < 100 || iterations > 100000 || !probe_valid() )
        return 2;
    std::cerr << "compiler=" << __VERSION__ << " Eigen=" << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
              << EIGEN_MINOR_VERSION << " float_bytes=" << sizeof( DDSPC::realT )
              << " period_us=500 allocation_counts=Eigen_permission_checks_and_C++new_only\n";
    std::cout << "component,n,p50_us,p95_us,p99_us,p999_us,max_us,eigen_requests_per_record,cpp_new_per_record,cpp_"
                 "bytes_per_record,over_500us\n";
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
        {
            DDSPC::ShadowConfig cfg;
            cfg.m_modes    = modes;
            cfg.m_useQrd   = qrd;
            cfg.m_prior    = 0.1f;
            cfg.m_lambda   = 0.3f;
            Matrix       y = Matrix::Zero( modes, 1 ), u = y, a = Matrix::Zero( 2, 1 );
            std::mt19937 rng( 800 );
            auto         fill = [&]
            {
                for( Matrix *v : { &y, &u, &a } )
                    for( int j = 0; j < v->size(); ++j )
                        ( *v )( j ) = ( int( rng() % 2001 ) - 1000 ) / 1000.f;
            };
            DDSPC::PredictiveController legacy( modes, 5, 3, 0, 1, .3f, .1f, 0, 0 );
            legacy.use_qrd = qrd;
            DDSPC::PredictiveController standard( modes, 5, 3, 0, 1, .3f, .1f, 0, 0, DDSPC::HistoryMode::Causal );
            DDSPC::PredictiveController assisted( modes, 5, 3, 0, 1, .3f, .1f, 2, 20, DDSPC::HistoryMode::Causal );
            standard.use_qrd = assisted.use_qrd = qrd;
            const std::string prefix            = "m" + std::to_string( modes ) + ( qrd ? "_qr_" : "_rls_" );
            // Regressor conditioning is deliberately generic; this benchmark measures work, not physical accuracy.
            measure( ( prefix + "legacy_step_identify_solve" ).c_str(),
                     iterations,
                     [&]( int i )
                     {
                         fill();
                         sink = legacy.calculate_command( y, u ).sum();
                         legacy.update_system();
                         if( i > 30 )
                             legacy.update_controller();
                     } );
            for( auto *model : { &standard, &assisted } )
            {
                const bool acc       = model == &assisted;
                const auto fixedSeed = rng; // New workspace measurements must not advance the original traces.
                measure( ( prefix + ( acc ? "assisted_step_identify_reference" : "standard_step_identify_reference" ) )
                             .c_str(),
                         iterations,
                         [&]( int i )
                         {
                             fill();
                             model->append_sample( i, y, u, acc ? a : Matrix() );
                             model->update_system();
                             if( i > 30 )
                                 sink = model->get_reference_controller( .3f ).sum();
                         } );
                const std::string component = prefix + ( acc ? "assisted_" : "standard_" );
                measure( ( component + "reference_only" ).c_str(),
                         iterations,
                         [&]( int ) { sink = model->get_reference_controller( .3f ).sum(); } );
                measure( ( component + "history_only" ).c_str(),
                         iterations,
                         [&]( int )
                         {
                             Matrix phi, target;
                             if( !model->get_training_example( phi, target ) )
                                 std::abort();
                             sink = phi.sum() + target.sum();
                         } );
                const int    features = ( 3 + 2 * 5 - 1 ) * modes + ( acc ? 2 * 20 : 0 );
                Matrix       featureWindow( features, features );
                std::mt19937 windowRng( 1080 + features ); // Preserve the original full-workload input sequence.
                for( int j = 0; j < featureWindow.size(); ++j )
                    featureWindow.data()[j] = ( int( windowRng() % 2001 ) - 1000 ) / 1000.f;
                // Isolate exactly the normalization + rank operations in ShadowController::step().
                // Frozen seeded windows measure component cost, not every possible pivot/conditioning pattern.
                measure( ( component + "rank_only" ).c_str(),
                         iterations,
                         [&]( int )
                         {
                             Matrix scaled = featureWindow;
                             for( int j = 0; j < scaled.cols(); ++j )
                             {
                                 const auto norm = scaled.col( j ).norm();
                                 if( norm > 0 )
                                     scaled.col( j ) /= norm;
                             }
                             Eigen::ColPivHouseholderQR<Matrix> qr( scaled );
                             qr.setThreshold( cfg.m_rankTolerance );
                             sink = qr.rank();
                         } );
                Matrix                             scaled = featureWindow;
                Eigen::ColPivHouseholderQR<Matrix> reusableRank( features, features );
                measure( ( component + "rank_fixed_only" ).c_str(),
                         iterations,
                         [&]( int )
                         {
                             scaled = featureWindow;
                             for( int j = 0; j < scaled.cols(); ++j )
                             {
                                 const auto norm = scaled.col( j ).norm();
                                 if( norm > 0 )
                                     scaled.col( j ) /= norm;
                             }
                             reusableRank.compute( scaled );
                             reusableRank.setThreshold( cfg.m_rankTolerance );
                             sink = reusableRank.rank();
                         } );
                DDSPC::PredictiveController fixed(
                    modes, 5, 3, 0, 1, .3f, .1f, acc ? 2 : 0, acc ? 20 : 0, DDSPC::HistoryMode::Causal );
                fixed.use_qrd    = qrd;
                auto   workspace = fixed.make_workspace();
                auto   fixedRng  = fixedSeed;
                Matrix fy = Matrix::Zero( modes, 1 ), fu = fy, fa = Matrix::Zero( 2, 1 ), empty;
                measure( ( component + "fixed_step_identify_reference" ).c_str(),
                         iterations,
                         [&]( int i )
                         {
                             for( auto *v : { &fy, &fu, &fa } )
                                 for( int j = 0; j < v->size(); ++j )
                                     ( *v )( j ) = ( int( fixedRng() % 2001 ) - 1000 ) / 1000.f;
                             fixed.append_sample( i, fy, fu, acc ? fa : empty );
                             if( fixed.prepare_training( *workspace ) )
                                 fixed.update_system( *workspace );
                             if( i > 30 )
                                 sink = fixed.get_reference_controller( *workspace, .3f ).sum();
                         } );
                measure( ( component + "reference_fixed_only" ).c_str(),
                         iterations,
                         [&]( int ) { sink = fixed.get_reference_controller( *workspace, .3f ).sum(); } );
                measure( ( component + "history_fixed_only" ).c_str(),
                         iterations,
                         [&]( int )
                         {
                             if( !fixed.prepare_training( *workspace ) )
                                 std::abort();
                             sink = workspace->regressor().sum() + workspace->target().sum();
                         } );
            }
            DDSPC::ShadowController pair( cfg );
            pair.training( DDSPC::ShadowMode::Standard, true );
            pair.training( DDSPC::ShadowMode::AccelAssisted, true );
            DDSPC::ShadowRecord record;
            record.m_wfsValid = record.m_appliedValid = record.m_timingValid = record.m_accelValid = true;
            measure( ( prefix + "managed_pair_training" ).c_str(),
                     iterations,
                     [&]( int i )
                     {
                         fill();
                         record.m_index          = i;
                         record.m_residual       = y;
                         record.m_appliedCommand = u;
                         record.m_acceleration   = a;
                         if( !pair.process( record ) )
                             std::abort();
                     } );
            const auto s = pair.status( DDSPC::ShadowMode::Standard ),
                       b = pair.status( DDSPC::ShadowMode::AccelAssisted );
            if( !s.m_ready || !b.m_ready )
            {
                std::cerr << "Benchmark failed to reach readiness\n";
                return 3;
            }
            std::cerr << prefix << "features=" << s.m_features << ',' << b.m_features << " ready=1,1\n";
            for( bool assistedMode : { false, true } )
            {
                const auto mode      = assistedMode ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
                Matrix     canonical = Matrix::Ones( pair.status( mode ).m_features - 3 * modes, 1 ),
                           output    = Matrix::Zero( modes, 1 );
                measure( ( prefix + ( assistedMode ? "assisted_cached_predict" : "standard_cached_predict" ) ).c_str(),
                         iterations,
                         [&]( int )
                         {
                             if( !pair.predict_cached( mode, record.m_index, 0, canonical, output ) )
                                 std::abort();
                             sink = output.sum();
                         } );
            }
            pair.training( DDSPC::ShadowMode::Standard, false );
            pair.training( DDSPC::ShadowMode::AccelAssisted, false );
            const auto frozenOrigin = record.m_index + 1;
            auto       frozenRng    = rng;
            measure( ( prefix + "managed_pair_frozen" ).c_str(),
                     iterations,
                     [&]( int i )
                     {
                         record.m_index = frozenOrigin + i;
                         for( auto *v : { &record.m_residual, &record.m_appliedCommand, &record.m_acceleration } )
                             for( int j = 0; j < v->size(); ++j )
                                 ( *v )( j ) = ( int( frozenRng() % 2001 ) - 1000 ) / 1000.f;
                         if( !pair.process( record ) )
                             std::abort();
                     } );
            using namespace libXWCTest::loPredCtrlAccTest;
            auto inputCfg            = input_config();
            inputCfg.m_modes         = modes;
            inputCfg.m_accelChannels = 2;
            std::vector<DDSPC::TimedWfsRecord>     wfs;
            std::vector<DDSPC::AppliedInputRecord> commands;
            for( int i = 0; i <= iterations + 512; ++i )
            {
                wfs.push_back( input_wfs( inputCfg, i ) );
                commands.push_back( input_command( inputCfg, i ) );
            }
            for( bool managed : { false, true } )
            {
                DDSPC::CausalInputBuilder builder( inputCfg );
                DDSPC::ShadowController   integrated( cfg );
                integrated.training( DDSPC::ShadowMode::Standard, true );
                integrated.training( DDSPC::ShadowMode::AccelAssisted, true );
                builder.push_wfs( wfs[0], wfs[0].m_decisionNs );
                measure( ( prefix + ( managed ? "builder_and_managed_pair" : "builder_only" ) ).c_str(),
                         iterations,
                         [&]( int i )
                         {
                             const auto first  = builder.push_wfs( wfs[i + 1], wfs[i + 1].m_decisionNs );
                             const auto result = builder.push_command( commands[i], commands[i].m_stamp.m_receivedNs );
                             if( first.m_discontinuity || result.m_discontinuity || result.m_records.size() != 1 )
                                 std::abort();
                             if( managed && !integrated.process( result.m_records[0].m_record ) )
                                 std::abort();
                             sink = result.m_records[0].m_record.m_residual.sum();
                         } );
                if( managed && ( !integrated.status( DDSPC::ShadowMode::Standard ).m_ready ||
                                 !integrated.status( DDSPC::ShadowMode::AccelAssisted ).m_ready ) )
                    return 4;
            }
        }
    return 0;
}
