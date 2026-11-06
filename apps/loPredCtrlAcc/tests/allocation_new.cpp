/** \file allocation_new.cpp
 * \brief Standalone benchmark-only ordinary C++ allocation instrumentation, shared by numerical and worker benchmarks.
 */
#include "allocation_probe.hpp"
#include <cstdlib>
#include <new>

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
