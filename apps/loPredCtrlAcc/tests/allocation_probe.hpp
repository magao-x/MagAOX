/** \file allocation_probe.hpp
 * \brief Benchmark-only Eigen allocation-check and C++ allocation counters; never included in operational builds.
 */
#ifndef loPredCtrlAcc_allocation_probe_hpp
#define loPredCtrlAcc_allocation_probe_hpp
#include <cassert>
#include <cstddef>
#include <cstring>

namespace DDSPCBenchmark
{
inline thread_local bool        enabled = false; ///< Only the numerical measured scope is observed.
inline thread_local std::size_t eigenRequests =
    0;                                           ///< Eigen allocation permission checks, including direct malloc paths.
inline thread_local std::size_t cppRequests = 0; ///< C++ new/new[] calls, separate from Eigen's direct malloc paths.
inline thread_local std::size_t cppBytes    = 0; ///< Bytes requested through C++ new; excludes Eigen's direct malloc.

/// Preserve assertions and observe Eigen allocation permission checks without forbidding allocations.
inline void check( bool        condition /**< [in] Original assertion result. */,
                   const char *expression /**< [in] Original assertion text. */ )
{
    if( enabled && std::strstr( expression, "heap allocation is forbidden" ) )
        ++eigenRequests;
    assert( condition );
}
} // namespace DDSPCBenchmark

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert( expression ) DDSPCBenchmark::check( bool( expression ), #expression )
#endif
