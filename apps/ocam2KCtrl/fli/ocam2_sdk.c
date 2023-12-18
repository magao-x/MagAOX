/** -- FIRSTLIGHT --
  \file    ocam2_sdk.c

  \brief   Main library file
  
  \author FIRSTLIGHT    
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>

#include "ocam2_sdk.h"
#include "ocam2_pvt.h"

#if defined(_WIN32) || defined(_WIN64)
/* We are on Windows */
#define strtok_r strtok_s
#define __restrict__
#endif

/* SDK global information */
int *g_descrblTab=NULL;
static int  g_nbElem=0;
static int  g_nbCamera=0;


ocam2_camInfo g_camInfoTab[OCAM2_SDK_MAX_CAMERA] =
{ {0, OCAM2_UNKNOWN,NULL}, /* 0 */
  {0, OCAM2_UNKNOWN,NULL}, /* 1 */
  {0, OCAM2_UNKNOWN,NULL}, /* 2 */
  {0, OCAM2_UNKNOWN,NULL}, /* 3 */
  {0, OCAM2_UNKNOWN,NULL}, /* 4 */
  {0, OCAM2_UNKNOWN,NULL}, /* 5 */
  {0, OCAM2_UNKNOWN,NULL}, /* 6 */
  {0, OCAM2_UNKNOWN,NULL}, /* 7 */
  {0, OCAM2_UNKNOWN,NULL}, /* 8 */
  {0, OCAM2_UNKNOWN,NULL}  /* 9 */
};


void ocam2_printf(ocam2_printf_sev severity, const char * format, ...)
{
    if ( (OCAM2_SEV_INFO==severity) || (OCAM2_SEV_WARNING==severity) || (OCAM2_SEV_ERROR==severity) )
    {
        static const char *severityStr[]={"INFO","WARNING","ERROR"};

        va_list arguments;
        va_start(arguments, format);

        fprintf(stderr, "OCAM2SDK:%s: ", severityStr[severity]);
        vfprintf(stderr, format, arguments);

        va_end(arguments);
    }
}

int ocam2_isIdValid(ocam2_id id)
{
    if ((0<=id) && (id<OCAM2_SDK_MAX_CAMERA))
        return 1;/*true==valid*/
    else
        return 0;/*false==invalid*/
}

/* API function */
ocam2_mode ocam2_getMode(ocam2_id id)
{
    assert(ocam2_isIdValid(id));

    return g_camInfoTab[id].mode;
}

/* API function */
const char *ocam2_modeStr(ocam2_mode mode)
{
    static const char *ocam2_modeText[] =
    {"Uknown",
     "Standard Mode(240x240@2060Hz)",
     "Cropping Mode(240x120@3680Hz)",
     "Binning 2x2 Mode(120x120@3620Hz)",
     "Binning 3x3 Mode(80x80@4950Hz)",
     "Binning 4x4 Mode(60x60@5900Hz)",
     "Cropping Mode(240x128@3500Hz)",
     "Mode 7",
     "Mode 8",
     "Binning 1x3 Mode(240x80@XXXXHz)",
     "Binning 1x4 Mode(240x50@XXXXHz)"};
    
    switch(mode)
    {
        case OCAM2_UNKNOWN:
        case OCAM2_NORMAL:
        case OCAM2_CROPPING240x120:
        case OCAM2_BINNING2x2:
        case OCAM2_BINNING3x3:
        case OCAM2_BINNING4x4:
        case OCAM2_CROPPING240x128:
        case OCAM2_BINNING1x3:
        case OCAM2_BINNING1x4:
        
        return ocam2_modeText[mode];
        default:
            return NULL;
    }
}

/* API function */
const char * ocam2_sdkVersion()
{
    return OCAMSDK_VERSION;
}

/* API function */
const char * ocam2_sdkBuild()
{
    return OCAMSDK_BUILD;
}

int ocam2_fsize(FILE *fp)
{
    int current, sz;

    current=ftell(fp);
    fseek(fp, 0L, SEEK_END);
    sz=ftell(fp);
    fseek(fp,current,SEEK_SET);
    return sz;
}

static int ocam2_fnbElem(char *buff, int bsz, char c)
{
    int nb=0;
    int i=0;

    while(i<bsz)
    {
        if (buff[i]==c) nb++;
        i++;
    }
    return nb;
}


static int ocam2_fillDescrblTab(const char *descrbFile)
{
    FILE *fp;
    int ret =-1;

    if (NULL!=g_descrblTab)
    {
       /* Descramble table already filled, return without error */
       /* WARNING: if different table exist one day, this code will have to be modified */
        return 0;
    }

    /* Ouverture du fichier de descrambling */
    if (NULL != (fp=fopen(descrbFile, "rb")))
    {
        int nbElem=0;
        int sz=ocam2_fsize(fp);

        if (0!=sz)
        {
            char *pfileContent;
            pfileContent=malloc(sz);

            if (NULL != pfileContent)
            {
                if (1==fread(pfileContent, sz, 1, fp))
                {
                    nbElem=ocam2_fnbElem(pfileContent, sz, ',');

                    if (0!=nbElem)
                    {
                        g_descrblTab = malloc(nbElem*sizeof(int));

                        if (NULL != g_descrblTab)
                        {
                            char *arg;
                            char delim[] = ",\n\r";
                            char *saveptr;

                            char *pfc=pfileContent;
                            int i=0;

                            while (i<nbElem)
                            {
                                arg = strtok_r(pfc, delim, &saveptr);
                                if (NULL != arg)
                                    g_descrblTab[i] = atoi(arg);

                                i++;
                                /* For next token pfc should be NULL */
                                if (NULL!=pfc) pfc=NULL;
                            }
                            ret=0;
                        }
                    }
                }
                free(pfileContent);
            }
        }
        g_nbElem = nbElem;
        fclose(fp);
        if (g_nbElem != OCAM2_PIXELS_IMAGE_NORMAL)
            ocam2_printf(OCAM2_SEV_WARNING, "Nb element found in descrambling file(=%d) different from normal pixel number(=%d) !!!\n",
                         g_nbElem, OCAM2_PIXELS_IMAGE_NORMAL);
    }
    else
    {
        ocam2_printf(OCAM2_SEV_ERROR, "Invalid File\n");
    }
    return ret;
}


static void ocam2_descramble_normal(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    int i;
    for (i=0; i< OCAM2_PIXELS_IMAGE_NORMAL; i++)
    {
         image[i] = imageRaw[g_descrblTab[i]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_cropping240x120(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    int chunk=OCAM2_PIXELS_IMAGE_CROPPING240x120/2;
    int i;
    for (i=0; i< chunk; i++)
    {
        int x=i%OCAM2_IMAGE_WIDTH_CROPPING240x120;
        int y=i/OCAM2_IMAGE_WIDTH_CROPPING240x120;

        image[i] = imageRaw[g_descrblTab[OCAM2_AMPLI_WIDTH+x+y*240]];
        image[chunk+i] = imageRaw[g_descrblTab[OCAM2_AMPLI_WIDTH+x+y*240+(OCAM2_PIXELS_IMAGE_NORMAL-(240*OCAM2_IMAGE_HEIGHT_CROPPING240x120/2))]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_cropping240x128(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    int chunk=OCAM2_PIXELS_IMAGE_CROPPING240x128/2;
    int i;
    for (i=0; i< chunk; i++)
    {
        int x=i%OCAM2_IMAGE_WIDTH_CROPPING240x128;
        int y=i/OCAM2_IMAGE_WIDTH_CROPPING240x128;

        image[i] = imageRaw[g_descrblTab[OCAM2_AMPLI_WIDTH-4+x+y*240]];
        image[chunk+i] = imageRaw[g_descrblTab[OCAM2_AMPLI_WIDTH-4+x+y*240+(OCAM2_PIXELS_IMAGE_NORMAL-(240*OCAM2_IMAGE_HEIGHT_CROPPING240x128/2))]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_binning2x2(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    const int chunk=OCAM2_PIXELS_IMAGE_BINNING2x2/2;
    int i;
    for (i=0; i < chunk; i++)
    {
         image[i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_BINNING2x2]];
         image[chunk+i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_BINNING2x2+OCAM2_BINNING2x2_OFFSET]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_binning3x3(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    /* In this mode,per ampli, the 17 first pixels are duplicated 3 times,
       then the 9 last pixels are sent without duplication.
       17+9=26 beacuse the 6 prescan pixels are sent duplicated 3 times.
       So we get , 6(x3) + [11(x3)+9] = 60 = the amplifier width in normal mode
    */
    const int chunk=OCAM2_PIXELS_IMAGE_BINNING3x3/2;
    int i;
    /* image[]:line size 80, the g_descrblTab[]:line size 240 */
    for (i=0; i < chunk; i++)
    {
         int x=i%OCAM2_IMAGE_WIDTH_BINNING3x3;
         int xa=x%20; /* xa = x related to one ampli, range[0-19] */
         int na=x/20; /* na = ampli number */
         int y=i/OCAM2_IMAGE_WIDTH_BINNING3x3;

         if (!(na%2)) /* even ampli */
         {
             if (xa<9)
             {
                 image[i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+xa+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+xa+y*240+OCAM2_BINNING3x3_OFFSET]];
             }
             else
             {
                 image[i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+9+3*(xa-9)+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+9+3*(xa-9)+y*240+OCAM2_BINNING3x3_OFFSET]];
             }
         }
         else /* odd ampli */
         {

             if (xa<11)
             {
                 image[i] = imageRaw[g_descrblTab[3*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+3*xa+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[3*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+3*xa+y*240+OCAM2_BINNING3x3_OFFSET]];
             }
             else
             {
                 image[i] = imageRaw[g_descrblTab[3*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+3*11+xa-11+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[3*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+3*11+xa-11+y*240+OCAM2_BINNING3x3_OFFSET]];
             }
         }
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_binning4x4(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    /* In this mode,per ampli, the first pixel is duplicated 3 times, then 12 pixels are duplicated 4 times,
       and at last the 9 last pixels are sent without duplication.
       1+12+9=22 because the 6 prescan pixels are sent too.
       So we get, [1(x3)+6(x4)] + [6(x4)+9] = 60 = the amplifier width in normal mode.
       We notice we have an extra pixel in addition to prescan pixel because the end image is 15 pixels per ampli only.
    */
    const int chunk=OCAM2_PIXELS_IMAGE_BINNING4x4/2;
    int i;
    /* image[]:line size 80, the g_descrblTab[]:line size 240 */
    for (i=0; i < chunk; i++)
    {
         int x=i%OCAM2_IMAGE_WIDTH_BINNING4x4;
         int xa=x%15; /* xa = x related to one ampli, range[0-19] */
         int na=x/15;
         int y=i/OCAM2_IMAGE_WIDTH_BINNING4x4;

         if (!(na%2)) /* even ampli */
         {
             if (xa<9)
             {
                 image[i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+xa+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+xa+y*240+OCAM2_BINNING4x4_OFFSET]];
             }
             else
             {
                 image[i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+9+4*(xa-9)+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[na*OCAM2_AMPLI_WIDTH+9+4*(xa-9)+y*240+OCAM2_BINNING4x4_OFFSET]];
             }
         }
         else /* odd ampli */
         {
             if (xa<6)
             {
                 image[i] = imageRaw[g_descrblTab[3+4*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+4*xa+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[3+4*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+4*xa+y*240+OCAM2_BINNING4x4_OFFSET]];
             }
             else
             {
                 image[i] = imageRaw[g_descrblTab[3+4*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+4*6+xa-6+y*240]];
                 image[chunk+i] = imageRaw[g_descrblTab[3+4*OCAM2_AMPLI_PRESCAN+na*OCAM2_AMPLI_WIDTH+4*6+xa-6+y*240+OCAM2_BINNING4x4_OFFSET]];
             }
         }
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_2_track(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    const int chunk=OCAM2_PIXELS_IMAGE_2_TRACK/2;
    int i;
    for (i=0; i < chunk; i++)
    {
         image[i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_2_TRACK]];
         image[chunk+i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_2_TRACK+OCAM2_2_TRACK_OFFSET]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}

static void ocam2_descramble_4_track(unsigned int *__restrict__ number, short *__restrict__ image, const short *__restrict__ imageRaw)
{
    const int chunk=OCAM2_PIXELS_IMAGE_4_TRACK/2;
    int i;
    for (i=0; i < chunk; i++)
    {
         image[i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_4_TRACK]];
         image[chunk+i] = imageRaw[g_descrblTab[i*OCAM2_NB_IDENTICAL_PIXELS_4_TRACK+OCAM2_4_TRACK_OFFSET]];
    }
    *number = ((int *)imageRaw)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
}


static void ocam2_descramble_binning1x3(unsigned int* __restrict__ number, short* __restrict__ image, const short* __restrict__ imageRaw)
{
    const int chunk = OCAM2_PIXELS_IMAGE_BINNING1x3 / 2;
    int i;
    for (i = 0; i < chunk; i++)
    {
        image[i] = imageRaw[g_descrblTab[i * OCAM2_NB_IDENTICAL_PIXELS_BINNING1x3]];
        image[chunk + i] = imageRaw[g_descrblTab[i * OCAM2_NB_IDENTICAL_PIXELS_BINNING1x3 + OCAM2_BINNING1x3_OFFSET]];
    }
    *number = ((int*)imageRaw)[OCAM2_IMAGE_NB_OFFSET / 4]; /* int offset */
}

static void ocam2_descramble_binning1x4(unsigned int* __restrict__ number, short* __restrict__ image, const short* __restrict__ imageRaw)
{
    const int chunk = OCAM2_PIXELS_IMAGE_BINNING1x4 / 2;
    int i;
    for (i = 0; i < chunk; i++)
    {
        image[i] = imageRaw[g_descrblTab[i * OCAM2_NB_IDENTICAL_PIXELS_BINNING1x4]];
        image[chunk + i] = imageRaw[g_descrblTab[i * OCAM2_NB_IDENTICAL_PIXELS_BINNING1x4 + OCAM2_BINNING1x4_OFFSET]];
    }
    *number = ((int*)imageRaw)[OCAM2_IMAGE_NB_OFFSET / 4]; /* int offset */
}

/* API function */
void ocam2_descramble(ocam2_id id, unsigned int *number, short *image, const short *imageRaw)
{
    assert(ocam2_isIdValid(id));
    assert(NULL!=number);
    assert(NULL!=image);
    assert(NULL!=imageRaw);
    assert(NULL!=g_descrblTab);
    assert(OCAM2_UNKNOWN !=g_camInfoTab[id].mode);

    g_camInfoTab[id].pfuncDescrbl(number, image, imageRaw);
}

/* API function */
ocam2_rc ocam2_init(ocam2_mode mode, const char *descrbFile, ocam2_id *id)
{
    if ( (mode!=OCAM2_NORMAL) &&
         (mode!=OCAM2_CROPPING240x120) &&
         (mode!=OCAM2_CROPPING240x128) &&
         (mode!=OCAM2_BINNING2x2) &&
         (mode!=OCAM2_BINNING3x3) &&
         (mode!=OCAM2_BINNING4x4) &&
         (mode!=OCAM2_2_TRACK) &&
         (mode!=OCAM2_4_TRACK) &&
         (mode != OCAM2_BINNING1x3) &&
         (mode != OCAM2_BINNING1x4) )
        return OCAM2_ERROR;

    if ( (NULL == descrbFile) && (0==g_nbCamera) )
        return OCAM2_ERROR;

    if (NULL == id)
        return OCAM2_ERROR;

    if (g_nbCamera>=OCAM2_SDK_MAX_CAMERA)
        return OCAM2_ERROR;

    if (0==g_nbCamera)
    {
        if (0!=ocam2_fillDescrblTab(descrbFile))
            return OCAM2_ERROR;
    }

    *id=g_nbCamera;
    g_nbCamera++;

    g_camInfoTab[*id].mode=mode;

    if (OCAM2_CROPPING240x120==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_cropping240x120;
    else if (OCAM2_CROPPING240x128==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_cropping240x128;
    else if (OCAM2_BINNING2x2==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_binning2x2;
    else if (OCAM2_BINNING3x3==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_binning3x3;
    else if (OCAM2_BINNING4x4==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_binning4x4;
	else if (OCAM2_2_TRACK==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_2_track;
	else if (OCAM2_4_TRACK==mode)
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_4_track;
    else if (OCAM2_BINNING1x3 == mode)
        g_camInfoTab[*id].pfuncDescrbl = ocam2_descramble_binning1x3;
    else if (OCAM2_BINNING1x4 == mode)
        g_camInfoTab[*id].pfuncDescrbl = ocam2_descramble_binning1x4;
    else
        g_camInfoTab[*id].pfuncDescrbl=ocam2_descramble_normal;

    return OCAM2_OK;
}

/* API function */
ocam2_rc ocam2_exit(ocam2_id id)
{
    if (!ocam2_isIdValid(id))
        return OCAM2_ERROR;

    g_camInfoTab[id].mode=OCAM2_UNKNOWN;
    g_camInfoTab[id].pfuncDescrbl=NULL;
    g_nbCamera--;

    if (0==g_nbCamera)
    {
        free(g_descrblTab);
        g_descrblTab=NULL;
    }
    return OCAM2_OK;
}


