/** \file dcamapi4.h
 * \brief Hamamatsu DCAM-API declaration stubs for SDK-less MagAO-X test builds.
 *
 * Declares the subset of the DCAM-SDK 4 API used by orcaCtrl, with the SDK's signatures.  Error codes match the SDK
 * (e.g. DCAMERR_NOCAMERA, logged by orcaCtrl as -2147483130); other constants only need to be distinct.  Tests
 * provide the function definitions.  Only test builds add this directory to the include path, so production builds
 * always use the installed SDK.
 *
 * \ingroup orcaCtrl_unit_test
 */

#ifndef tests_dcam_dcamapi4_h
#define tests_dcam_dcamapi4_h

#include <cstdint>

typedef int32_t int32;

/// Opaque camera handle.
typedef struct tag_dcam *HDCAM;

/// Opaque wait handle.
typedef struct DCAMWAIT *HDCAMWAIT;

/// DCAM error codes, as signed 32-bit values so they convert to int32 without narrowing.
enum DCAMERR : int32
{
    DCAMERR_BUSY              = static_cast<int32>( 0x80000101 ),
    DCAMERR_ABORT             = static_cast<int32>( 0x80000102 ),
    DCAMERR_TIMEOUT           = static_cast<int32>( 0x80000106 ),
    DCAMERR_NOCAMERA          = static_cast<int32>( 0x80000206 ),
    DCAMERR_OUTOFRANGE        = static_cast<int32>( 0x80000822 ),
    DCAMERR_INVALIDPROPERTYID = static_cast<int32>( 0x80000825 ),
    DCAMERR_NOTSUPPORT        = static_cast<int32>( 0x80000f03 ),
    DCAMERR_NONE              = 0,
    DCAMERR_SUCCESS           = 1
};

/// True if a DCAM call failed.
#define failed( err ) ( (int32)( err ) < 0 )

/// Capture status values.
enum DCAMCAP_STATUS
{
    DCAMCAP_STATUS_ERROR    = 0,
    DCAMCAP_STATUS_BUSY     = 1,
    DCAMCAP_STATUS_READY    = 2,
    DCAMCAP_STATUS_STABLE   = 3,
    DCAMCAP_STATUS_UNSTABLE = 4
};

/// Capture start modes.
enum DCAMCAP_START
{
    DCAMCAP_START_SEQUENCE = -1,
    DCAMCAP_START_SNAP     = 0
};

/// Transfer-info kinds.
enum DCAMCAP_TRANSFERKIND
{
    DCAMCAP_TRANSFERKIND_FRAME = 0
};

/// Capture events.
enum DCAMWAIT_EVENT
{
    DCAMWAIT_CAPEVENT_FRAMEREADY = 0x0002,
    DCAMWAIT_CAPEVENT_STOPPED    = 0x0010
};

/// Device identification strings.
enum DCAM_IDSTR
{
    DCAM_IDSTR_CAMERAID = 0x04000102,
    DCAM_IDSTR_VENDOR   = 0x04000103,
    DCAM_IDSTR_MODEL    = 0x04000104
};

/// dcamapi_init() parameters.
struct DCAMAPI_INIT
{
    int32        size;            ///< Size of this structure.
    int32        iDeviceCount;    ///< [out] Number of devices found.
    int32        reserved;        ///< Reserved.
    int32        initoptionbytes; ///< Size of initoption.
    const int32 *initoption;      ///< Initialization options.
    const void  *guid;            ///< API GUID.
};

/// dcamdev_open() parameters.
struct DCAMDEV_OPEN
{
    int32 size;  ///< Size of this structure.
    int32 index; ///< Device index.
    HDCAM hdcam; ///< [out] Camera handle.
};

/// dcamdev_getstring() parameters.
struct DCAMDEV_STRING
{
    int32 size;      ///< Size of this structure.
    int32 iString;   ///< String or error ID.
    char *text;      ///< [out] Text buffer.
    int32 textbytes; ///< Size of the text buffer.
};

/// dcamwait_open() parameters.
struct DCAMWAIT_OPEN
{
    int32     size;         ///< Size of this structure.
    int32     supportevent; ///< [out] Supported events.
    HDCAMWAIT hwait;        ///< [out] Wait handle.
    HDCAM     hdcam;        ///< Camera handle.
};

/// dcamwait_start() parameters.
struct DCAMWAIT_START
{
    int32 size;          ///< Size of this structure.
    int32 eventhappened; ///< [out] Events that happened.
    int32 eventmask;     ///< Events to wait for.
    int32 timeout;       ///< Timeout [ms].
};

/// dcamcap_transferinfo() parameters.
struct DCAMCAP_TRANSFERINFO
{
    int32 size;              ///< Size of this structure.
    int32 iKind;             ///< Transfer kind.
    int32 nNewestFrameIndex; ///< [out] Ring index of the newest frame.
    int32 nFrameCount;       ///< [out] Frames transferred.
};

/// dcambuf_lockframe() parameters.
struct DCAMBUF_FRAME
{
    int32 size;        ///< Size of this structure.
    int32 iKind;       ///< Reserved.
    int32 option;      ///< Reserved.
    int32 iFrame;      ///< Frame index.
    void *buf;         ///< [out] Frame pointer.
    int32 rowbytes;    ///< [out] Row stride.
    int32 type;        ///< [out] Pixel type.
    int32 width;       ///< [out] Width.
    int32 height;      ///< [out] Height.
    int32 left;        ///< [out] Left offset.
    int32 top;         ///< [out] Top offset.
    int32 framestamp;  ///< [out] Framestamp.
    int32 camerastamp; ///< [out] Camera stamp.
};

/// dcamprop_getattr() parameters, defined in dcamprop.h.
struct DCAMPROP_ATTR;

extern "C"
{
    DCAMERR dcamapi_init( DCAMAPI_INIT *param );
    DCAMERR dcamapi_uninit();
    DCAMERR dcamdev_open( DCAMDEV_OPEN *param );
    DCAMERR dcamdev_close( HDCAM h );
    DCAMERR dcamdev_getstring( HDCAM h, DCAMDEV_STRING *param );
    DCAMERR dcamprop_getattr( HDCAM h, DCAMPROP_ATTR *param );
    DCAMERR dcamprop_getvalue( HDCAM h, int32 iProp, double *pValue );
    DCAMERR dcamprop_setvalue( HDCAM h, int32 iProp, double fValue );
    DCAMERR dcamprop_setgetvalue( HDCAM h, int32 iProp, double *pValue, int32 option = 0 );
    DCAMERR dcambuf_alloc( HDCAM h, int32 framecount );
    DCAMERR dcambuf_release( HDCAM h, int32 iKind = 0 );
    DCAMERR dcambuf_lockframe( HDCAM h, DCAMBUF_FRAME *pFrame );
    DCAMERR dcamcap_start( HDCAM h, int32 mode );
    DCAMERR dcamcap_stop( HDCAM h );
    DCAMERR dcamcap_status( HDCAM h, int32 *pStatus );
    DCAMERR dcamcap_transferinfo( HDCAM h, DCAMCAP_TRANSFERINFO *param );
    DCAMERR dcamwait_open( DCAMWAIT_OPEN *param );
    DCAMERR dcamwait_close( HDCAMWAIT hWait );
    DCAMERR dcamwait_start( HDCAMWAIT hWait, DCAMWAIT_START *param );
    DCAMERR dcamwait_abort( HDCAMWAIT hWait );
}

#endif // tests_dcam_dcamapi4_h
