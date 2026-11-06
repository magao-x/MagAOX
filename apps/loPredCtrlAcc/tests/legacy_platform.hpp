/** \file legacy_platform.hpp
 * \brief Supply the historical BSD uint spelling for unmodified Darwin GCC replay.
 */
#ifndef loPredCtrlAcc_legacy_platform_hpp
#define loPredCtrlAcc_legacy_platform_hpp
#if defined( __APPLE__ )
using uint = unsigned int;
#endif
#endif
