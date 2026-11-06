/** \file native_assert_probe.hpp
 * \brief Diagnostic baseline using native assertions; allocation counting is intentionally disabled in Eigen.
 */
#ifndef loPredCtrlAcc_native_assert_probe_hpp
#define loPredCtrlAcc_native_assert_probe_hpp
#include "allocation_probe.hpp"
#undef eigen_assert
#define eigen_assert( expression ) assert( expression )
#endif
