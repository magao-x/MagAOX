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
#include <new>
#include <random>
#include <vector>

/// Count ordinary C++ allocation requests separately from Eigen's direct malloc path.
void *operator new( std::size_t bytes /**< [in] Requested size. */ )
{
    if( DDSPCBenchmark::enabled )
    {
        ++DDSPCBenchmark::cppRequests;
        DDSPCBenchmark::cppBytes += bytes;
    }
    if( void *p = std::malloc( bytes ? bytes : 1 ) )
        return p;
    throw std::bad_alloc();
}
/// Forward array requests through the same observed allocator.
void *operator new[]( std::size_t bytes /**< [in] Requested size. */ )
{
    return ::operator new( bytes );
}
/// Release a counted allocation.
void operator delete( void *p /**< [in] Allocation. */ ) noexcept
{
    std::free( p );
}
/// Release a counted array allocation.
void operator delete[]( void *p /**< [in] Allocation. */ ) noexcept
{
    std::free( p );
}
/// Support sized-deallocation compilers.
void operator delete( void *p /**< [in] Allocation. */, std::size_t /**< [in] Size. */ ) noexcept
{
    std::free( p );
}
/// Support sized array deallocation.
void operator delete[]( void *p /**< [in] Allocation. */, std::size_t /**< [in] Size. */ ) noexcept
{
    std::free( p );
}

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
} // namespace

/// Run synthetic component workloads; optional argument changes sample count, never an instrument endpoint.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional positive timed sample count. */ )
{
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
                const bool acc = model == &assisted;
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
