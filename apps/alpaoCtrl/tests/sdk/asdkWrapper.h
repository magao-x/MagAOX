/** \file asdkWrapper.h
 * \brief Minimal offline declarations used only by ALPAO controller unit tests; not a vendor SDK replacement.
 */
#ifndef alpaoCtrl_test_asdkWrapper_h
#define alpaoCtrl_test_asdkWrapper_h

/// \cond TEST_SDK_STUB
using Scalar     = double;
using UInt       = unsigned;
using COMPL_STAT = int;
namespace acs
{
using UInt = unsigned;
}
/// Software-only opaque ALPAO mirror handle.
struct asdkDM
{
};
/// Open a software mirror handle.
asdkDM *asdkInit( const char *serial /**< [in] Unused serial. */ );
/// Return an empty software error state.
void asdkGetLastError( UInt *code /**< [out] Error code. */,
                       char *text /**< [out] Optional error buffer. */,
                       UInt  size /**< [in] Buffer capacity. */ );
/// Return a software mirror property.
COMPL_STAT asdkGet( asdkDM     *dm /**< [in] Stub handle. */,
                    const char *name /**< [in] Unused property name. */,
                    Scalar     *value /**< [out] Property value. */ );
/// Record a scripted SDK submission.
COMPL_STAT asdkSend( asdkDM *dm /**< [in] Stub handle. */, const Scalar *values /**< [in] Submitted values. */ );
/// Complete a software reset operation.
COMPL_STAT asdkReset( asdkDM *dm /**< [in] Stub handle. */ );
/// Close a software mirror handle.
COMPL_STAT asdkRelease( asdkDM *dm /**< [in] Stub handle. */ );
/// \endcond
#endif
