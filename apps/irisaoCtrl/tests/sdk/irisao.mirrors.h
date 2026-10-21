/** \file irisao.mirrors.h
 * \brief Minimal offline declarations used only by IrisAO controller unit tests; not a vendor SDK replacement.
 */
#ifndef irisaoCtrl_test_mirrors_h
#define irisaoCtrl_test_mirrors_h

/// \cond TEST_SDK_STUB
using SegmentNumber = unsigned;
using MirrorHandle  = void *;
/// Software reachability result for the one test segment.
struct MirrorPosition
{
    bool reachable{ true }; ///< Scripted segment reachability.
};
constexpr int MirrorSendSettings = 1; ///< Final staged-command submission selector.
/// Open a software mirror handle.
MirrorHandle MirrorConnect( const char *mirror /**< [in] Unused mirror identity. */,
                            const char *driver /**< [in] Unused driver identity. */,
                            bool        disabled /**< [in] Unused disable flag. */ );
/// Iterate the one software segment.
bool MirrorIterate( MirrorHandle handle /**< [in] Stub handle. */, SegmentNumber segment /**< [in] Segment index. */ );
/// Stage one software segment.
void SetMirrorPosition( MirrorHandle  handle /**< [in] Stub handle. */,
                        SegmentNumber segment /**< [in] Segment index. */,
                        double        piston /**< [in] Piston. */,
                        double        tip /**< [in] Tip. */,
                        double        tilt /**< [in] Tilt. */ );
/// Return software segment reachability.
void GetMirrorPosition( MirrorHandle    handle /**< [in] Stub handle. */,
                        SegmentNumber   segment /**< [in] Segment index. */,
                        MirrorPosition *position /**< [out] Reachability. */ );
/// Record the staged software command without claiming SDK success semantics.
void MirrorCommand( MirrorHandle handle /**< [in] Stub handle. */, int command /**< [in] Submission selector. */ );
/// Close a software mirror handle.
void MirrorRelease( MirrorHandle handle /**< [in] Stub handle. */ );
/// \endcond
#endif
