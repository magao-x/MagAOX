/** \file allocation_probe.hpp
 * \brief Benchmark-only Eigen allocation-check and C++ allocation counters; never included in operational builds.
 */
#ifndef loPredCtrlAcc_allocation_probe_hpp
#define loPredCtrlAcc_allocation_probe_hpp
#include <cassert>
#include <cstddef>

namespace DDSPCBenchmark
{
inline thread_local bool        enabled = false; ///< Only the numerical measured scope is observed.
inline thread_local std::size_t eigenRequests =
    0;                                           ///< Eigen allocation permission checks, including direct malloc paths.
inline thread_local std::size_t cppRequests = 0; ///< C++ new/new[] calls, separate from Eigen's direct malloc paths.
inline thread_local std::size_t cppBytes    = 0; ///< Bytes requested through C++ new; excludes Eigen's direct malloc.

/// Classify stringized Eigen assertions at compile time; remain compatible with the bundled C++14 INDI unit.
constexpr bool allocation_check( const char *expression /**< [in] String literal from eigen_assert. */ )
{
    constexpr char needle[] = "heap allocation is forbidden";
    for( std::size_t i = 0; expression[i]; ++i )
    {
        std::size_t j = 0;
        while( needle[j] && expression[i + j] == needle[j] )
            ++j;
        if( !needle[j] )
            return true;
    }
    return false;
}

/// Count only allocator permission checks while preserving their original assertion.
inline void check_allocation( bool condition /**< [in] Original assertion result. */ )
{
    if( enabled )
        ++eigenRequests;
    assert( condition );
}
} // namespace DDSPCBenchmark

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert( expression )                                                                                     \
    ( DDSPCBenchmark::allocation_check( #expression ) ? DDSPCBenchmark::check_allocation( bool( expression ) )         \
                                                      : static_cast<void>( assert( expression ) ) )
#endif
