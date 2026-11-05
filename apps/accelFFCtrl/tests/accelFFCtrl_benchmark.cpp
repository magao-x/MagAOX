/** \file accelFFCtrl_benchmark.cpp
 * \brief Standalone allocation-instrumented numerical cost at 8-kHz input and selectable prediction cadence.
 * \ingroup accelFFCtrl_files
 */
#include "../accelFFCtrlModel.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

#if !defined( EIGEN_RUNTIME_NO_MALLOC ) || defined( EIGEN_NO_DEBUG )
    #error "Build this benchmark with Eigen runtime malloc checking and assertions enabled."
#endif

/// \cond ACCELFFCTRL_ALLOCATION_PROBE
namespace
{
bool   g_track{ false };   ///< Enable C++ allocation counting only after all fixture and estimator storage is prepared.
size_t g_allocations{ 0 }; ///< Calls to allocating C++ new during the measured loop.
} // namespace

/// Count normal C++ allocations while retaining ordinary throwing-new behavior.
void *operator new( std::size_t size /**< [in] requested bytes */ )
{
    if( g_track )
        ++g_allocations;
    if( void *value = std::malloc( size ? size : 1 ) )
        return value;
    throw std::bad_alloc();
}

/// Release ordinary C++ allocations.
void operator delete( void *value /**< [in] previously allocated storage */ ) noexcept
{
    std::free( value );
}

/// Count array storage through the same allocation probe.
void *operator new[]( std::size_t size /**< [in] requested array bytes */ )
{
    return ::operator new( size );
}

/// Release array storage.
void operator delete[]( void *value /**< [in] previously allocated array */ ) noexcept
{
    std::free( value );
}
/// \endcond

/// Benchmark numerical work only; sample synthesis and output allocation happen before measurement.
int main()
{
    using namespace MagAOX::app::accelFFCtrlDetail;
    constexpr size_t       warmup = 8000, count = 800000;
    std::vector<adcSample> recording( count + warmup );
    for( size_t n = 0; n < recording.size(); ++n )
    {
        auto &value      = recording[n];
        value.m_sequence = n;
        value.m_timeNs   = n * 125000;
        for( size_t axis = 0; axis < 2; ++axis )
            value.m_adc[axis] = static_cast<uint16_t>(
                std::round( 2048 + 12 * std::cos( 6.28318530717958647692 * 28.5 * n / 8000 + axis ) ) );
    }
    for( size_t stride : { 4, 1 } )
    {
        harmonicEstimator estimator;
        for( size_t n = 0; n < warmup; ++n )
            if( !estimator.consume( recording[n], recording[n].m_timeNs ) )
                return 1;
        std::vector<double> ingest( count ), predict( count / stride );
        size_t              predicted = 0;
        double              checksum  = 0;
        g_allocations                 = 0;
#ifdef EIGEN_RUNTIME_NO_MALLOC
        Eigen::internal::set_is_malloc_allowed( false );
#endif
        g_track          = true;
        const auto begin = std::chrono::steady_clock::now();
        for( size_t n = 0; n < count; ++n )
        {
            const auto &value = recording[n + warmup];
            const auto  start = std::chrono::steady_clock::now();
            if( !estimator.consume( value, value.m_timeNs ) )
                return 2;
            const auto ingested = std::chrono::steady_clock::now();
            ingest[n]           = std::chrono::duration<double, std::nano>( ingested - start ).count();
            if( n % stride == stride - 1 )
            {
                const auto result    = estimator.predict( value.m_timeNs, value.m_timeNs + 1000000 );
                const auto completed = std::chrono::steady_clock::now();
                if( result.m_state != estimatorState::ready )
                    return 3;
                predict[predicted++] = std::chrono::duration<double, std::nano>( completed - ingested ).count();
                checksum += result.m_pointing[0];
            }
        }
        const auto end     = std::chrono::steady_clock::now();
        auto       invalid = recording.back();
        ++invalid.m_sequence;
        invalid.m_timeNs += 125000;
        invalid.m_adc[0] = 4096;
        if( estimator.consume( invalid, invalid.m_timeNs ) ||
            estimator.predict( invalid.m_timeNs ).m_state != estimatorState::invalidSample )
            return 5;
        estimator.reset();
        if( !estimator.consume( recording.back(), recording.back().m_timeNs ) ||
            estimator.predict( recording.back().m_timeNs + 2000001 ).m_state != estimatorState::stale )
            return 6;
        g_track = false;
#ifdef EIGEN_RUNTIME_NO_MALLOC
        Eigen::internal::set_is_malloc_allowed( true );
#endif
        std::cout << "8000-Hz input, " << 8000 / stride
                  << "-Hz prediction: " << std::chrono::duration<double, std::nano>( end - begin ).count() / count
                  << " ns/source sample including benchmark clocks; new calls=" << g_allocations
                  << ", checksum=" << checksum << '\n';
        for( auto *timings : { &ingest, &predict } )
        {
            double sum = 0;
            for( double time : *timings )
                sum += time;
            std::sort( timings->begin(), timings->end() );
            std::cout << ( timings == &ingest ? "consume" : "predict" ) << ": mean " << sum / timings->size()
                      << " ns, p50 " << ( *timings )[timings->size() / 2] << " ns, p99 "
                      << ( *timings )[timings->size() * 99 / 100] << " ns, max " << timings->back() << " ns\n";
        }
        if( g_allocations )
            return 4;
    }
}
