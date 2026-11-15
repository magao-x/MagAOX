/** \file floatUtils.hpp
 * \brief Test-only finite predicate required by the unchanged syncopation CRTP header.
 */
#ifndef DDSPC_TEST_MX_FLOAT_UTILS_HPP
#define DDSPC_TEST_MX_FLOAT_UTILS_HPP
#include <cmath>
namespace mx::math
{
/// Preserve the producer's strict finite check in stream-free application tests.
template <class T>
bool isFinite( T value /**< [in] Scalar being checked. */ )
{
    return std::isfinite( value );
}
} //namespace mx::math
#endif
