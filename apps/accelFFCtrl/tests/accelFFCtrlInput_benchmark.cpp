/** \file accelFFCtrlInput_benchmark.cpp
 * \brief Allocation-instrumented ingestion/prediction cost with a synthetic transport, without OS wait/clock costs.
 * \ingroup accelFFCtrl_files
 */
#include "../accelFFCtrlInput.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>

#if !defined( EIGEN_RUNTIME_NO_MALLOC ) || defined( EIGEN_NO_DEBUG )
    #error "Build with Eigen runtime allocation checks and assertions enabled."
#endif

/// \cond ACCELFFCTRL_INPUT_COST_FIXTURE
namespace
{
bool   tracking{ false }; ///< Count only after fixture and runtime preparation.
size_t allocations{ 0 };  ///< Allocating C++ new calls in the measured region.
using namespace MagAOX::app::accelFFCtrlDetail;

/// Single-plane synthetic source; the worker sees each committed sample before the next one is produced.
struct costSource
{
    inputFrame m_frame; ///< Current committed local pair.

    /// Return a completed one-plane publication snapshot.
    inputHeader header() const;

    /// Return the locally owned synthetic pair.
    inputFrame frame( size_t index /**< [in] unused plane zero */ ) const;
};

inputHeader costSource::header() const
{
    return { 3, 2, 1, 1, m_frame.m_counter, 0, true, true, true, false };
}

inputFrame costSource::frame( size_t ) const
{
    return m_frame;
}
} // namespace

/// Count ordinary allocating new calls inside the prepared loop.
void *operator new( size_t size /**< [in] requested bytes */ )
{
    if( tracking )
        ++allocations;
    if( void *value = std::malloc( size ? size : 1 ) )
        return value;
    throw std::bad_alloc();
}

/// Release ordinary new storage.
void operator delete( void *value /**< [in] allocation */ ) noexcept
{
    std::free( value );
}

/// Count array allocations through the ordinary probe.
void *operator new[]( size_t size /**< [in] requested bytes */ )
{
    return ::operator new( size );
}

/// Release array storage.
void operator delete[]( void *value /**< [in] allocation */ ) noexcept
{
    std::free( value );
}
/// \endcond

/// Measure the bounded input runtime with 8-kHz acquisition and 2-kHz prediction, excluding native waits and syscalls.
int main()
{
    constexpr int64_t       epoch = 1700000000000000000;
    constexpr size_t        count = 240000;
    std::vector<inputFrame> prepared( count );
    std::vector<double>     costs( count );
    for( size_t n = 0; n < count; ++n )
    {
        const int64_t stamp = epoch + 1000000000 + int64_t( n ) * 125000;
        const auto    code  = uint16_t( 2048 + std::round( 20 * std::cos( 6.283185307179586 * 28.5 * n / 8000 ) ) );
        prepared[n]         = { { code, code },
                                { stamp / 1000000000, stamp % 1000000000 },
                                { ( stamp + 10000 ) / 1000000000, ( stamp + 10000 ) % 1000000000 },
                                n + 1 };
    }
    for( bool allowGap : { false, true } )
    {
        inputProcessor processor( parameters{}, 256, 1000000, allowGap );
        std::fill( costs.begin(), costs.end(), 0. );
        allocations = 0;
        processor.attached();
        costSource source;
        Eigen::internal::set_is_malloc_allowed( false );
        tracking        = true;
        double checksum = 0;
        for( size_t n = 0; n < count; ++n )
        {
            if( allowGap && n && n % 8000 == 0 )
                continue;
            const auto    start = std::chrono::steady_clock::now();
            const int64_t now   = 1000050000 + int64_t( n ) * 125000;
            source.m_frame      = prepared[n];
            processor.drain( source, [now] { return inputClock{ epoch + now, now, 10 }; } );
            if( n % 4 == 0 )
            {
                processor.predict( now );
                checksum += processor.diagnostics().m_prediction.m_pointing[0];
            }
            costs[n] = std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - start ).count();
        }
        tracking = false;
        Eigen::internal::set_is_malloc_allowed( true );
        double sum = 0;
        for( double cost : costs )
            sum += cost;
        std::sort( costs.begin(), costs.end() );
        std::cout << ( allowGap ? "Single-gap enabled, one loss/s: " : "Default lossless: " )
                  << "Input runtime (synthetic source/clocks, 8k/2k): mean " << sum / count << " ns, p50 "
                  << costs[count / 2] << " ns, p99 " << costs[99 * count / 100] << " ns, max " << costs.back()
                  << " ns; new calls=" << allocations << ", accepted=" << processor.diagnostics().m_processed
                  << ", dropped=" << processor.diagnostics().m_dropped << ", checksum=" << checksum << '\n';
        if( !allowGap )
        {
            if( allocations || processor.diagnostics().m_processed != count || processor.diagnostics().m_dropped )
                return 1;
        }
        else if( allocations || processor.diagnostics().m_processed != count - 29 ||
                 processor.diagnostics().m_dropped != 29 || processor.diagnostics().m_admittedGaps != 29 ||
                 processor.diagnostics().m_rejectedGaps || processor.diagnostics().m_resets != 1 )
            return 1;
    }
    return 0;
}
