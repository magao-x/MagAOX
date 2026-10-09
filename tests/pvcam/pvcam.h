/** \file pvcam.h
 * \brief PVCAM SDK declaration stubs for SDK-less MagAO-X test builds.
 *
 * Declares the subset of the Teledyne PVCAM API used by pvcamCtrl, with the SDK's signatures and constant values.
 * Tests provide the function definitions.  Only test builds add this directory to the include path.
 *
 * \ingroup pvcamCtrl_unit_test
 */

#ifndef tests_pvcam_pvcam_h
#define tests_pvcam_pvcam_h

#include "master.h"

#define CAM_NAME_LEN          32
#define ERROR_MSG_LEN         255
#define MAX_ALPHA_SER_NUM_LEN 32

/// Frame information delivered with end-of-frame callbacks.
typedef struct _TAG_FRAME_INFO
{
    int16  hCam;         ///< Handle of the camera that sent this structure.
    int32  FrameNr;      ///< Frame number, 1-based.
    long64 TimeStamp;    ///< Frame end-of-frame timestamp.
    int32  ReadoutTime;  ///< Frame readout time.
    long64 TimeStampBOF; ///< Frame beginning-of-frame timestamp.
} FRAME_INFO;

/// Camera open modes.
typedef enum PL_OPEN_MODES
{
    OPEN_EXCLUSIVE
} PL_OPEN_MODES;

/// Parameter attributes.
typedef enum PL_PARAM_ATTRIBUTES
{
    ATTR_CURRENT,
    ATTR_COUNT,
    ATTR_TYPE,
    ATTR_MIN,
    ATTR_MAX,
    ATTR_DEFAULT,
    ATTR_INCREMENT,
    ATTR_ACCESS,
    ATTR_AVAIL,
    ATTR_LIVE
} PL_PARAM_ATTRIBUTES;

/// Exposure modes.
typedef enum PL_EXPOSURE_MODES
{
    TIMED_MODE
} PL_EXPOSURE_MODES;

/// Fan speed set points.
typedef enum PL_FAN_SPEEDS
{
    FAN_SPEED_HIGH,
    FAN_SPEED_MEDIUM,
    FAN_SPEED_LOW,
    FAN_SPEED_OFF
} PL_FAN_SPEEDS;

/// Acquisition abort modes.
typedef enum PL_CCS_ABORT_MODES
{
    CCS_NO_CHANGE = 0,
    CCS_HALT
} PL_CCS_ABORT_MODES;

/// Circular buffer modes.
typedef enum PL_CIRC_MODES
{
    CIRC_NONE = 0,
    CIRC_OVERWRITE
} PL_CIRC_MODES;

/// Callback events.
typedef enum PL_CALLBACK_EVENT
{
    PL_CALLBACK_BOF = 0,
    PL_CALLBACK_EOF
} PL_CALLBACK_EVENT;

/// Sensor region definition.
typedef struct rgn_type
{
    uns16 s1;   ///< First pixel in the serial register.
    uns16 s2;   ///< Last pixel in the serial register.
    uns16 sbin; ///< Serial binning.
    uns16 p1;   ///< First row in the parallel register.
    uns16 p2;   ///< Last row in the parallel register.
    uns16 pbin; ///< Parallel binning.
} rgn_type;

#define TYPE_INT16    1
#define TYPE_FLT64    4
#define TYPE_UNS16    6
#define TYPE_UNS64    8
#define TYPE_ENUM     9
#define TYPE_CHAR_PTR 13
#define TYPE_INT64    16

#define CLASS2 2
#define CLASS3 3

#define PARAM_READOUT_TIME       ( ( CLASS2 << 16 ) + ( TYPE_FLT64 << 24 ) + 179 )
#define PARAM_CLEARING_TIME      ( ( CLASS2 << 16 ) + ( TYPE_INT64 << 24 ) + 180 )
#define PARAM_POST_TRIGGER_DELAY ( ( CLASS2 << 16 ) + ( TYPE_INT64 << 24 ) + 181 )
#define PARAM_PRE_TRIGGER_DELAY  ( ( CLASS2 << 16 ) + ( TYPE_INT64 << 24 ) + 182 )
#define PARAM_TEMP               ( ( CLASS2 << 16 ) + ( TYPE_INT16 << 24 ) + 525 )
#define PARAM_TEMP_SETPOINT      ( ( CLASS2 << 16 ) + ( TYPE_INT16 << 24 ) + 526 )
#define PARAM_HEAD_SER_NUM_ALPHA ( ( CLASS2 << 16 ) + ( TYPE_CHAR_PTR << 24 ) + 533 )
#define PARAM_FAN_SPEED_SETPOINT ( ( CLASS2 << 16 ) + ( TYPE_ENUM << 24 ) + 710 )
#define PARAM_BIT_DEPTH          ( ( CLASS2 << 16 ) + ( TYPE_INT16 << 24 ) + 511 )
#define PARAM_GAIN_INDEX         ( ( CLASS2 << 16 ) + ( TYPE_INT16 << 24 ) + 512 )
#define PARAM_SPDTAB_INDEX       ( ( CLASS2 << 16 ) + ( TYPE_INT16 << 24 ) + 513 )
#define PARAM_READOUT_PORT       ( ( CLASS2 << 16 ) + ( TYPE_ENUM << 24 ) + 247 )
#define PARAM_PIX_TIME           ( ( CLASS2 << 16 ) + ( TYPE_UNS16 << 24 ) + 516 )
#define PARAM_EXP_RES            ( ( CLASS3 << 16 ) + ( TYPE_ENUM << 24 ) + 2 )
#define PARAM_EXP_RES_INDEX      ( ( CLASS3 << 16 ) + ( TYPE_UNS16 << 24 ) + 4 )
#define PARAM_EXPOSURE_TIME      ( ( CLASS3 << 16 ) + ( TYPE_UNS64 << 24 ) + 8 )

#ifdef __cplusplus
extern "C"
{
#endif

    rs_bool PV_DECL pl_pvcam_init( void );
    rs_bool PV_DECL pl_pvcam_uninit( void );
    rs_bool PV_DECL pl_cam_close( int16 hcam );
    rs_bool PV_DECL pl_cam_get_name( int16 cam_num, char *camera_name );
    rs_bool PV_DECL pl_cam_get_total( int16 *totl_cams );
    rs_bool PV_DECL pl_cam_open( char *camera_name, int16 *hcam, int16 o_mode );
    rs_bool PV_DECL pl_cam_register_callback_ex3( int16 hcam, int32 callback_event, void *callback, void *context );
    rs_bool PV_DECL pl_cam_deregister_callback( int16 hcam, int32 callback_event );
    int16 PV_DECL   pl_error_code( void );
    rs_bool PV_DECL pl_error_message( int16 err_code, char *msg );
    rs_bool PV_DECL pl_get_param( int16 hcam, uns32 param_id, int16 param_attribute, void *param_value );
    rs_bool PV_DECL pl_set_param( int16 hcam, uns32 param_id, void *param_value );
    rs_bool PV_DECL pl_get_enum_param( int16 hcam, uns32 param_id, uns32 index, int32 *value, char *desc, uns32 length );
    rs_bool PV_DECL pl_enum_str_length( int16 hcam, uns32 param_id, uns32 index, uns32 *length );
    rs_bool PV_DECL pl_exp_setup_cont( int16           hcam,
                                       uns16           rgn_total,
                                       const rgn_type *rgn_array,
                                       int16           exp_mode,
                                       uns32           exposure_time,
                                       uns32          *exp_bytes,
                                       int16           buffer_mode );
    rs_bool PV_DECL pl_exp_start_cont( int16 hcam, void *pixel_stream, uns32 size );
    rs_bool PV_DECL pl_exp_get_latest_frame( int16 hcam, void **frame );
    rs_bool PV_DECL pl_exp_stop_cont( int16 hcam, int16 cam_state );

#ifdef __cplusplus
}
#endif

#endif // tests_pvcam_pvcam_h
