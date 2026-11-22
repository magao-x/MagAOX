/** \file cacaoWriterTestAbi.hpp
 * \brief Test-only FPS ABI boundary; the installed milk layout is validated on the target.
 */
#ifndef nnReconstructor_cacaoWriterTestAbi_hpp
#define nnReconstructor_cacaoWriterTestAbi_hpp
#include <cstdint>
#include <ctime>
#include <sys/types.h>

/// \cond NNRECONSTRUCTOR_TEST_SUPPORT
/// Only fields consumed by the production metadata reader; deliberately not a production ABI copy.
struct FUNCTION_PARAMETER_STRUCT_MD
{
    char     name[100]{};       ///< FPS name.
    long     NBparamMAX{ 1 };   ///< Parameter count used to check the complete file size.
    pid_t    runpid{ 0 };       ///< Process identity.
    timespec runpidstarttime{}; ///< FPS process lifetime.
    uint32_t status{ 0 };       ///< Requested RUN flag.
};
/// Stand-in parameter storage for file-size checks only.
struct FUNCTION_PARAMETER
{
    char unused[64]{}; ///< Uninterpreted payload.
};
constexpr uint32_t FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN = 0x20;
/// \endcond
#endif
