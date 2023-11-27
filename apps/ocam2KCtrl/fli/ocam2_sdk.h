/** -- FIRSTLIGHT --
  \file    ocam2_sdk.h

  \brief   The purpose of the libocam2sdk library is to provide an easy way to achieve ocam2 specific opération.
           The main feature is ocam2 raw image descrambling.
 
  \author FIRSTLIGHT  
    
  \note    You will find in this header ocam2 camera characteristic constants and functions protoypes.
*/


#ifndef OCAM2_SDK_H
#define OCAM2_SDK_H

/** Amplifier width */
#define OCAM2_AMPLI_WIDTH  60
/** Amplifier width raw */
#define OCAM2_AMPLI_WIDTH_RAW 66
/** Amplifier prescan pixels number */
#define OCAM2_AMPLI_PRESCAN (OCAM2_AMPLI_WIDTH_RAW-OCAM2_AMPLI_WIDTH)

/** Amplifier height */
#define OCAM2_AMPLI_HEIGHT 120

/** 2^14 */
#define OCAM2_PIXEL_MAX_VAL            /*16383*/32767
/** -2^14 */
#define OCAM2_PIXEL_MIN_VAL            /*(-16384)*/(-32768)

/** 16bit witdth  */
#define OCAM2_IMAGE_WIDTH_NORMAL       240
/** 16bit witdth */
#define OCAM2_IMAGE_HEIGHT_NORMAL      240
/** 57600 pixels(16bit witdth) per image area (240x240) */
#define OCAM2_PIXELS_IMAGE_NORMAL      (OCAM2_IMAGE_WIDTH_NORMAL*OCAM2_IMAGE_HEIGHT_NORMAL)
/** 8bit witdth */
#define OCAM2_IMAGE_WIDTH_RAW_NORMAL   1056
/** 8bit witdth */
#define OCAM2_IMAGE_HEIGHT_RAW_NORMAL  121
/** 63888 pixels(16bit witdth) in raw image (including non-image pixels : 66x121x8) */
#define OCAM2_PIXELS_RAW_NORMAL        ((OCAM2_IMAGE_WIDTH_RAW_NORMAL*OCAM2_IMAGE_HEIGHT_RAW_NORMAL)/2)

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_CROPPING240x120   120
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_CROPPING240x120  120
/** 14400 pixels(16bit witdth) per image area (120x120) */
#define OCAM2_PIXELS_IMAGE_CROPPING240x120  (OCAM2_IMAGE_WIDTH_CROPPING240x120*OCAM2_IMAGE_HEIGHT_CROPPING240x120)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_CROPPING240x120   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_CROPPING240x120  61
/** 32208 pixels(16bit witdth) in raw image (including non-image pixels : 66x61x8)*/
#define OCAM2_PIXELS_RAW_CROPPING240x120  ((OCAM2_IMAGE_WIDTH_RAW_CROPPING240x120*OCAM2_IMAGE_HEIGHT_RAW_CROPPING240x120)/2)

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_CROPPING240x128   128
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_CROPPING240x128  128
/** 16384 pixels(16bit witdth) per image area (128x128) */
#define OCAM2_PIXELS_IMAGE_CROPPING240x128  (OCAM2_IMAGE_WIDTH_CROPPING240x128*OCAM2_IMAGE_HEIGHT_CROPPING240x128)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_CROPPING240x128   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_CROPPING240x128  65
/** 34320 pixels(16bit witdth) in raw image (including non-image pixels : 66x65x8)*/
#define OCAM2_PIXELS_RAW_CROPPING240x128  ((OCAM2_IMAGE_WIDTH_RAW_CROPPING240x128*OCAM2_IMAGE_HEIGHT_RAW_CROPPING240x128)/2)

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_BINNING2x2   120
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_BINNING2x2  120
/** 14400 pixels(16bit witdth) per image area (120x120) */
#define OCAM2_PIXELS_IMAGE_BINNING2x2  (OCAM2_IMAGE_WIDTH_BINNING2x2*OCAM2_IMAGE_HEIGHT_BINNING2x2)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_BINNING2x2   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING2x2  62
/** 32736 pixels(16bit witdth) in raw image (including non-image pixels : 66x62x8) */
#define OCAM2_PIXELS_RAW_BINNING2x2  ((OCAM2_IMAGE_WIDTH_RAW_BINNING2x2*OCAM2_IMAGE_HEIGHT_RAW_BINNING2x2)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_BINNING2x2  2
/** Offset for second chunk */
#define OCAM2_BINNING2x2_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_BINNING2x2/2*OCAM2_NB_IDENTICAL_PIXELS_BINNING2x2))

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_BINNING3x3   80
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_BINNING3x3  80
/** 6400 pixels(16bit witdth) per image area (80x80) */
#define OCAM2_PIXELS_IMAGE_BINNING3x3  (OCAM2_IMAGE_WIDTH_BINNING3x3*OCAM2_IMAGE_HEIGHT_BINNING3x3)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_BINNING3x3   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING3x3  41
/** 22176 pixels(16bit witdth) in raw image (including non-image pixels : 66x42x8)*/
#define OCAM2_PIXELS_RAW_BINNING3x3  ((OCAM2_IMAGE_WIDTH_RAW_BINNING3x3*OCAM2_IMAGE_HEIGHT_RAW_BINNING3x3)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_BINNING3x3  3
/** Offset for second chunk */
#define OCAM2_BINNING3x3_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_BINNING3x3/2*OCAM2_NB_IDENTICAL_PIXELS_BINNING3x3))

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_BINNING4x4   60
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_BINNING4x4  60
/** 3600 pixels(16bit witdth) per image area (60x60) */
#define OCAM2_PIXELS_IMAGE_BINNING4x4  (OCAM2_IMAGE_WIDTH_BINNING4x4*OCAM2_IMAGE_HEIGHT_BINNING4x4)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_BINNING4x4   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING4x4  31
/** 16896 pixels(16bit witdth) in raw image (including non-image pixels : 66x32x8) */
#define OCAM2_PIXELS_RAW_BINNING4x4  ((OCAM2_IMAGE_WIDTH_RAW_BINNING4x4*OCAM2_IMAGE_HEIGHT_RAW_BINNING4x4)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_BINNING4x4  4
/** Offset for second chunk */
#define OCAM2_BINNING4x4_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_BINNING4x4/2*OCAM2_NB_IDENTICAL_PIXELS_BINNING4x4))

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_2_TRACK   240
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_2_TRACK  2
/** 480 pixels(16bit witdth) per image area (240x2) */
#define OCAM2_PIXELS_IMAGE_2_TRACK  (OCAM2_IMAGE_WIDTH_2_TRACK*OCAM2_IMAGE_HEIGHT_2_TRACK)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_2_TRACK   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_2_TRACK  2
/** 16896 pixels(16bit witdth) in raw image (including non-image pixels : 66x32x8) */
#define OCAM2_PIXELS_RAW_2_TRACK  ((OCAM2_IMAGE_WIDTH_RAW_2_TRACK*OCAM2_IMAGE_HEIGHT_RAW_2_TRACK)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_2_TRACK  1
/** Offset for second chunk */
#define OCAM2_2_TRACK_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_2_TRACK/2*OCAM2_NB_IDENTICAL_PIXELS_2_TRACK))

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_4_TRACK   240
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_4_TRACK  4
/** 480 pixels(16bit witdth) per image area (240x2) */
#define OCAM2_PIXELS_IMAGE_4_TRACK  (OCAM2_IMAGE_WIDTH_4_TRACK*OCAM2_IMAGE_HEIGHT_4_TRACK)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_4_TRACK   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_4_TRACK  3
/** 16896 pixels(16bit witdth) in raw image (including non-image pixels : 66x32x8) */
#define OCAM2_PIXELS_RAW_4_TRACK  ((OCAM2_IMAGE_WIDTH_RAW_4_TRACK*OCAM2_IMAGE_HEIGHT_RAW_4_TRACK)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_4_TRACK  1
/** Offset for second chunk */
#define OCAM2_4_TRACK_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_4_TRACK/2*OCAM2_NB_IDENTICAL_PIXELS_4_TRACK))


/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_BINNING1x3   240
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_BINNING1x3  80
/** 19200 pixels(16bit witdth) per image area (240x80) */
#define OCAM2_PIXELS_IMAGE_BINNING1x3  (OCAM2_IMAGE_WIDTH_BINNING1x3*OCAM2_IMAGE_HEIGHT_BINNING1x3)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_BINNING1x3   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING1x3  41
/** 22176 pixels(16bit witdth) in raw image (including non-image pixels : 66x42x8)*/
#define OCAM2_PIXELS_RAW_BINNING1x3  ((OCAM2_IMAGE_WIDTH_RAW_BINNING1x3*OCAM2_IMAGE_HEIGHT_RAW_BINNING1x3)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_BINNING1x3  1
/** Offset for second chunk */
#define OCAM2_BINNING1x3_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_BINNING1x3/2*OCAM2_NB_IDENTICAL_PIXELS_BINNING1x3))

/** 16bit witdth*/
#define OCAM2_IMAGE_WIDTH_BINNING1x4   240
/** 16bit witdth*/
#define OCAM2_IMAGE_HEIGHT_BINNING1x4  60
/** 3600 pixels(16bit witdth) per image area (60x60) */
#define OCAM2_PIXELS_IMAGE_BINNING1x4  (OCAM2_IMAGE_WIDTH_BINNING1x4*OCAM2_IMAGE_HEIGHT_BINNING1x4)
/** 8bit witdth*/
#define OCAM2_IMAGE_WIDTH_RAW_BINNING1x4   1056
/** 8bit witdth*/
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING1x4  31
/** 16896 pixels(16bit witdth) in raw image (including non-image pixels : 66x32x8) */
#define OCAM2_PIXELS_RAW_BINNING1x4  ((OCAM2_IMAGE_WIDTH_RAW_BINNING1x4*OCAM2_IMAGE_HEIGHT_RAW_BINNING1x4)/2)
/** Number of identical pixel in raw image */
#define OCAM2_NB_IDENTICAL_PIXELS_BINNING1x4  1
/** Offset for second chunk */
#define OCAM2_BINNING1x4_OFFSET  (OCAM2_PIXELS_IMAGE_NORMAL - (OCAM2_PIXELS_IMAGE_BINNING1x4/2*OCAM2_NB_IDENTICAL_PIXELS_BINNING1x4))


/** Keep for compatibility */
#define OCAM2_IMAGE_WIDTH_BINNING      OCAM2_IMAGE_WIDTH_BINNING2x2
/** Keep for compatibility */
#define OCAM2_IMAGE_HEIGHT_BINNING     OCAM2_IMAGE_HEIGHT_BINNING2x2
/** Keep for compatibility */
#define OCAM2_PIXELS_IMAGE_BINNING     OCAM2_PIXELS_IMAGE_BINNING2x2
/** Keep for compatibility */
#define OCAM2_IMAGE_WIDTH_RAW_BINNING  OCAM2_IMAGE_WIDTH_RAW_BINNING2x2
/** Keep for compatibility */
#define OCAM2_IMAGE_HEIGHT_RAW_BINNING OCAM2_IMAGE_HEIGHT_RAW_BINNING2x2
/** Keep for compatibility */
#define OCAM2_PIXELS_RAW_BINNING       OCAM2_PIXELS_RAW_BINNING2x2
/** Keep for compatibility */
#define OCAM2_BINNING_OFFSET           OCAM2_BINNING2x2_OFFSET

#ifndef OCAM2_IMAGE_NB_OFFSET
/** Offset of image number in raw image(bytes)*/
#define OCAM2_IMAGE_NB_OFFSET 8
#endif
/** Image number field width(bytes)*/
#define OCAM2_IMAGE_NB_SZ 4

/**
  \typedef ocam2_mode
  \brief typedef of ocam2 camera mode
*/
/**
  \enum workMode
  \brief Enum of ocam2 camera work mode
*/
typedef enum workMode
{
    /** Invalid */
    OCAM2_UNKNOWN=0,
    /** Default mode */
    OCAM2_NORMAL          = 1,
    /** Cropping 120 */
    OCAM2_CROPPING240x120 = 2,
    /** Binning 2x2 */
    OCAM2_BINNING2x2      = 3,
    /** Binning 3x3 */
    OCAM2_BINNING3x3      = 4,
    /** Binning 4x4 */
    OCAM2_BINNING4x4      = 5,
    /** For compatibility(= Binning 2x2) */
    OCAM2_BINNING = OCAM2_BINNING2x2,
    /** Cropping 128 */
    OCAM2_CROPPING240x128 = 6,
	/** Binning 2 lignes */
	OCAM2_2_TRACK = 7,
	/** Binning 4 lignes */
	OCAM2_4_TRACK = 8,
    /** Binning 1x3 */
    OCAM2_BINNING1x3 = 9,
    /** Binning 1x4 */
    OCAM2_BINNING1x4 = 10,
    /** Number of mode */
    OCAM2_NB_WORK_MODE = 10 /** Number of mode */

} ocam2_mode;


/** 
  \typedef ocam2_rc
  \brief Enum of ocam2 library return code
*/
typedef enum
{
   OCAM2_ERROR=-1,
   OCAM2_OK=0

} ocam2_rc;

/** 
  \typedef ocam2_id
  \brief Library camera identifier
*/
typedef int ocam2_id;

/**
  \typedef ocam2_camFirmVer
  \brief Camera firmware release
  For firmware build date before 18/03/2015, use
  OCAM2_FIRM_V1 else OCAM2_FIRM_V2
  You can know the camera firmware build using the command:
  "firmware read"
*/
typedef enum
{
    OCAM2_FIRM_V1=1,
    OCAM2_FIRM_V2=2

} ocam2_camFirmVer;

/** Max number of camera managed by the sdk */
#define OCAM2_SDK_MAX_CAMERA 10

#ifdef __cplusplus
extern "C" {
#endif

/**
  \fn const char * ocam2_sdkVersion()
  \brief Return sdk version
  \return sdk version as a string
*/
const char * ocam2_sdkVersion();

/**
  \fn const char * ocam2_sdkBuild()
  \brief Return sdk build
  \return sdk build as a string
*/
const char * ocam2_sdkBuild();

/**
  \fn ocam2_rc ocam2_init(ocam2_mode mode, const char *descrbFile, ocam2_id *id)
  \brief Create a camera instance with the provided mode
  \param[in]  mode  camera mode 
  \param[in]  descrbFile  descrambling file to use
  \param[out] id  camera identifier

  \return Return OK or Error
*/
ocam2_rc ocam2_init(ocam2_mode mode, const char *descrbFile, ocam2_id *id);

/**
  \fn void ocam2_descramble(int id, unsigned int *number, short *image, const short *imageRaw)
  \brief Create a camera instance with the provided mode
  \param[in]  id      camera identifier 
  \param[out] number  Image number
  \param[out] image   Descrambled image
  \param[in] imageRaw Raw image
  
  \return none
*/
void ocam2_descramble(ocam2_id id, unsigned int *number, short *image, const short *imageRaw);

/**
  \fn ocam2_rc ocam2_exit(ocam2_id id)
  \brief Clear a camera instance
  \param[in]  id      camera identifier 
  
  \return Return OK or Error
*/
ocam2_rc ocam2_exit(ocam2_id id);

/**
  \fn ocam2_mode ocam2_getMode(ocam2_id id)
  \brief Return the camera mode
  \param[in]  id      camera identifier 

  \return camera mode as ocam2_mode
*/
ocam2_mode ocam2_getMode(ocam2_id id);

/**
  \fn const char *ocam2_modeStr(ocam2_mode mode)
  \brief Return a description text for camera mode
  \param[in]  mode    camera mode 
  
  \return camera mode description as a string
*/
const char *ocam2_modeStr(ocam2_mode mode);

/* --- New flat bias --- */

/**
  \typedef ocam2_serialOut_func_t
  \brief Callback used to send characters to the serial port
  \param  p       private data
  \param  buffer  characters to send to serial port
  \param  number  number of character to send to serial port
*/
typedef void (*ocam2_serialOut_func_t)(void *p, const char *buffer, int number);


/**
  \fn ocam2_rc ocam2_sendBias(ocam2_id id, const char *biasFile, int offset, ocam2_serialOut_func_t cb, void *p, ocam2_camFirmVer camVer)
  \brief Send bias image to the camera link serial port
  \param[in]  id       camera identifier
  \param[in]  biasFile bias file
  \param[in]  offset   offset to reach pixel in bias file(can be used to skip a header)
  \param[in]  cb       callback, it must send the received characters to the camera link serial port
  \param[in]  p        private data
  \param[in]  camVer   camera firmware release

  \return Return OK or Error
*/
ocam2_rc ocam2_sendBias(ocam2_id id, const char *biasFile, int offset, ocam2_serialOut_func_t cb, void *p, ocam2_camFirmVer camVer);

/**
  \fn ocam2_rc ocam2_sendFlat(ocam2_id id, const char *flatFile, int offset, ocam2_serialOut_func_t cb, void *p, ocam2_camFirmVer camVer)
  \brief Send flat image to the camera link serial port
  \param[in]  id       camera identifier
  \param[in]  flatFile flat file
  \param[in]  offset   offset to reach pixel in bias file(can be used to skip a header)
  \param[in]  cb       callback, it must send the received characters to the camera link serial port
  \param[in]  p        private data
  \param[in]  camVer   camera firmware release

  \return Return OK or Error
*/
ocam2_rc ocam2_sendFlat(ocam2_id id, const char *flatFile, int offset, ocam2_serialOut_func_t cb, void *p, ocam2_camFirmVer camVer);


#ifdef __cplusplus
}
#endif

#endif // OCAM2_SDK_H



