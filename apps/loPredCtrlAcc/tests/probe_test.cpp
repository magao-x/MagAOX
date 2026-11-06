/** \file probe_test.cpp
 * \brief Verify compile-time classification, allocation detection, single evaluation and retained assertion failures.
 */
#include <Eigen/Dense>
#include <cstring>
#include <iostream>

/// Run normal instrumentation checks or an intentionally failing child assertion.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional failure injection. */ )
{
    static_assert( DDSPCBenchmark::allocation_check( "is_malloc_allowed() && heap allocation is forbidden" ),
                   "allocation detection" );
    static_assert( !DDSPCBenchmark::allocation_check( "x.rows() == y.rows()" ), "ordinary assertion" );
    static_assert( !DDSPCBenchmark::allocation_check( "heap allocation is forbidde" ), "truncated text" );
    if( argc == 2 && std::strcmp( argv[1], "--fail-assert" ) == 0 )
    {
        eigen_assert( false );
        return 3;
    }
    if( argc == 2 && std::strcmp( argv[1], "--fail-allocation" ) == 0 )
    {
        Eigen::internal::set_is_malloc_allowed( false );
        Eigen::MatrixXf value = Eigen::MatrixXf::Ones( 32, 1 );
        return int( value.sum() );
    }
    int evaluated                 = 0;
    DDSPCBenchmark::enabled       = true;
    DDSPCBenchmark::eigenRequests = 0;
    eigen_assert( ++evaluated == 1 );
    Eigen::MatrixXf value   = Eigen::MatrixXf::Ones( 32, 1 );
    DDSPCBenchmark::enabled = false;
    if( evaluated != 1 || DDSPCBenchmark::eigenRequests != 1 || value.sum() != 32 )
        return 4;
    std::cout << "Probe detects one allocation and preserves single assertion evaluation\n";
}
