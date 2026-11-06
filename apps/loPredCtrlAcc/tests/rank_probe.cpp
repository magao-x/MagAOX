/** \file rank_probe.cpp
 * \brief Separate assertion-probe overhead from the unchanged normalized pivoted-QR numerical workload.
 */
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{
/// Read a specified clock in nanoseconds; missing thread CPU clocks make qualification fail explicitly.
std::uint64_t stamp( clockid_t clock /**< [in] Monotonic wall or thread CPU clock. */ )
{
    timespec value{};
    if( clock_gettime( clock, &value ) )
        throw std::runtime_error( "Required profiling clock unavailable" );
    return std::uint64_t( value.tv_sec ) * 1000000000 + value.tv_nsec;
}
/// Print original sorted-sample quantiles without allocation in the measured operation.
void quantiles( std::vector<double> values /**< [in] Microsecond samples. */ )
{
    std::sort( values.begin(), values.end() );
    for( double q : { .5, .95, .99, .999, 1. } )
        std::cout << ',' << values[std::size_t( q * ( values.size() - 1 ) )];
}
/// Hash raw factor bytes to verify exactly identical QR results when only the runtime probe switch changes.
std::uint64_t fingerprint( const Eigen::MatrixXf &matrix /**< [in] Computed QR factors. */ )
{
    std::uint64_t value = 14695981039346656037ULL;
    const auto   *bytes = reinterpret_cast<const unsigned char *>( matrix.data() );
    for( std::size_t i = 0; i < std::size_t( matrix.size() ) * sizeof( float ); ++i )
        value = ( value ^ bytes[i] ) * 1099511628211ULL;
    return value;
}
} // namespace

/// Profile identical frozen rank inputs with the same compiled kernel and runtime counting off/on.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional timed sample count. */ )
{
    const int n = argc == 2 ? std::stoi( argv[1] ) : 2000;
    if( n < 100 )
        return 2;
    std::cout << std::setprecision( 9 )
              << "features,probe,n,wall_p50_us,wall_p95_us,wall_p99_us,wall_p999_us,wall_max_us,cpu_p50_us,cpu_p95_us,"
                 "cpu_p99_us,cpu_p999_us,cpu_max_us,eigen_requests,factor_hash\n";
    std::cerr << "compiler=" << __VERSION__ << " wall=CLOCK_MONOTONIC cpu=CLOCK_THREAD_CPUTIME_ID strict_math\n";
    for( int p : { 120, 160 } )
    {
        Eigen::MatrixXf window( p, p ), scaled( p, p );
        std::mt19937    rng( 1080 + p );
        for( int j = 0; j < window.size(); ++j )
            window.data()[j] = ( int( rng() % 2001 ) - 1000 ) / 1000.f;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXf> qr( p, p );
        std::uint64_t                               expected = 0;
        for( bool enabled : { false, true } )
        {
            std::vector<double> wall( n ), cpu( n );
            DDSPCBenchmark::eigenRequests = 0;
            for( int i = -64; i < n; ++i )
            {
                DDSPCBenchmark::enabled = enabled;
                const auto c = stamp( CLOCK_THREAD_CPUTIME_ID ), w = stamp( CLOCK_MONOTONIC );
                scaled = window;
                for( int j = 0; j < p; ++j )
                {
                    const float norm = scaled.col( j ).norm();
                    if( norm > 0 )
                        scaled.col( j ) /= norm;
                }
                qr.compute( scaled );
                qr.setThreshold( 1e-5f );
                if( qr.rank() != p )
                    return 3;
                const auto elapsedWall  = stamp( CLOCK_MONOTONIC ) - w,
                           elapsedCpu   = stamp( CLOCK_THREAD_CPUTIME_ID ) - c;
                DDSPCBenchmark::enabled = false;
                if( i >= 0 )
                {
                    wall[i] = elapsedWall / 1000.;
                    cpu[i]  = elapsedCpu / 1000.;
                }
            }
            const auto hash = fingerprint( qr.matrixQR() );
            if( !enabled )
                expected = hash;
            else if( hash != expected )
                return 4;
            std::cout << p << ',' << enabled << ',' << n;
            quantiles( wall );
            quantiles( cpu );
            std::cout << ',' << DDSPCBenchmark::eigenRequests << ',' << hash << '\n';
        }
    }
}
