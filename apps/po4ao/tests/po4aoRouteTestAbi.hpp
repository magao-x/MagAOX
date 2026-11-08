/** \file po4aoRouteTestAbi.hpp
 * \brief Minimal installed-SDK boundaries for offline route-IO logic tests, not ABI validation.
 */
#ifndef po4aoRouteTestAbi_hpp
#define po4aoRouteTestAbi_hpp
#include <cstdint>
#include <sys/types.h>
#include <ctime>
constexpr uint32_t FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN = 1;
constexpr uint32_t FPTYPE_ONOFF                            = 0x8000;
constexpr uint64_t FPFLAG_ONOFF                            = 1;
constexpr int      PROCESSINFO_LOOPSTAT_ACTIVE             = 1;
/// Test metadata containing fields read by the real adapter.
struct FUNCTION_PARAMETER_STRUCT_MD
{
    char     name[80]{};        ///< FPS identity.
    long     NBparamMAX{ 0 };   ///< Parameter count.
    pid_t    runpid{ 0 };       ///< Owner process.
    timespec runpidstarttime{}; ///< FPS start timestamp.
    uint32_t status{ 0 };       ///< RUN request status.
};
/// Test parameter record, deliberately not claiming installed binary layout compatibility.
struct FUNCTION_PARAMETER
{
    char     keywordfull[160]{}; ///< Full parameter key.
    uint32_t type{ 0 };          ///< Parameter type.
    uint64_t fpflag{ 0 };        ///< ON/OFF state.
    union Values
    {
        uint64_t ui64[4];        ///< Numeric values.
        char     string[2][160]; ///< Text values.
        /// Zero-initialize the test union.
        Values() : ui64{}
        {
        }
    } val;
};
/// Test process-info record used to validate counter/inode admission.
struct PROCESSINFO
{
    pid_t    PID{ 0 };                ///< Process identity.
    int      loopstat{ 0 };           ///< Active/run state.
    ino_t    triggerstreaminode{ 0 }; ///< Actually attached trigger inode.
    int      triggerstreamID{ -1 };   ///< Attached trigger indicator.
    uint64_t triggerstreamcnt{ 0 };   ///< Last admitted trigger counter.
    long     loopcnt{ 0 };            ///< Completed loop counter.
};
#endif
