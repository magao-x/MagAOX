/** -- FIRSTLIGHT --
  \file    ocam2_pvt.h

  \brief   Private definition for libocam2sdk

  \author FIRSTLIGHT

*/


#ifndef OCAM2_PVT_H
#define OCAM2_PVT_H

#define BIASFLAT_BUFFER_SIZE  (128*1024) /* 128KBytes */
#define BIASFLAT_TRANSMISSION_BUFFER_SIZE (BIASFLAT_BUFFER_SIZE+4) /* bias/flat buffer + checksum(4) */
/* The buffer size for transmitting the Bias/Flat image, 16-bit pixels (=65538) */
#define BIASFLAT_TRANSMISSION_BUFFER_SIZE_USHORT (BIASFLAT_TRANSMISSION_BUFFER_SIZE/sizeof(unsigned short))

#define NB_AMPLIFIER 8
/* Pixels per amplifier, including overscan. (==7986=66*121) */
#define PIXELS_PER_AMPLIFIER_NORMAL (OCAM2_PIXELS_RAW_NORMAL/NB_AMPLIFIER)
/* Pixels per amplifier, including overscan. (==4092=66*62) */
#define PIXELS_PER_AMPLIFIER_BINNING (OCAM2_PIXELS_RAW_BINNING/NB_AMPLIFIER)

typedef enum
{
   OCAM2_SEV_INFO=0,
   OCAM2_SEV_WARNING=1,
   OCAM2_SEV_ERROR=2

} ocam2_printf_sev;

typedef enum
{
   OCAM2_BIAS=0,
   OCAM2_FLAT=1

} ocam2_ImgType;

/* SDK per camera information */
typedef void (*ocam2_descramble_func_t)(unsigned int *number, short *image, const short *imageRaw);

typedef struct ocam2_camInfo {
    ocam2_camFirmVer camVer;
    ocam2_mode mode;
    ocam2_descramble_func_t pfuncDescrbl;
    short *imgBiasFlat;
    short *imgBiasFlatRaw;
    unsigned short *biasFlatBufferTx;
	ocam2_serialOut_func_t cbSerialOut;
	void *p; /* User private data */
	
	
} ocam2_camInfo;



void ocam2_printf(ocam2_printf_sev severity, const char * format, ...);
int ocam2_isIdValid(ocam2_id id);
int ocam2_fsize(FILE *fp);

#endif // OCAM2_PVT_H
