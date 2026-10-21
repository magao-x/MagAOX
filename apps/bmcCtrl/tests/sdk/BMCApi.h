/** \file BMCApi.h
 * \brief Minimal offline declarations used only by BMC controller unit tests; not a vendor SDK replacement.
 */
#ifndef bmcCtrl_test_BMCApi_h
#define bmcCtrl_test_BMCApi_h

/// \cond TEST_SDK_STUB
#include <cstdint>
using BMCRC                    = int;
constexpr BMCRC    NO_ERR      = 0;    ///< Scripted SDK success code.
constexpr unsigned MAX_DM_SIZE = 4096; ///< Capacity needed by the production initialization parser.
/// Software-only subset of the vendor handle fields used by this app.
struct DM
{
    uint32_t ActCount{ 2 }; ///< Software actuator count used by initialization.
};
/// Open a software mirror handle.
BMCRC BMCOpen( DM *dm /**< [out] Stub handle. */, const char *serial /**< [in] Unused serial. */ );
/// Populate a software actuator mapping.
BMCRC BMCLoadMap( DM         *dm /**< [in] Stub handle. */,
                  const char *name /**< [in] Unused mapping name. */,
                  uint32_t   *map /**< [out] Mapping. */ );
/// Record a scripted SDK submission.
BMCRC BMCSetArray( DM           *dm /**< [in] Stub handle. */,
                   const double *values /**< [in] Submitted values. */,
                   const void   *map /**< [in] Unused mapping. */ );
/// Format a scripted SDK error.
const char *BMCErrorString( BMCRC result /**< [in] Scripted result. */ );
/// Complete a software clear operation.
BMCRC BMCClearArray( DM *dm /**< [in] Stub handle. */ );
/// Close a software mirror handle.
BMCRC BMCClose( DM *dm /**< [in] Stub handle. */ );
/// \endcond
#endif
