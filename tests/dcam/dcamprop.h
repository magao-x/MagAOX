/** \file dcamprop.h
 * \brief Hamamatsu DCAM property stubs for SDK-less MagAO-X test builds.
 *
 * Declares the subset of the DCAM-SDK 4 property IDs and values used by orcaCtrl.  Values only need to be distinct.
 * Only test builds add this directory to the include path.
 *
 * \ingroup orcaCtrl_unit_test
 */

#ifndef tests_dcam_dcamprop_h
#define tests_dcam_dcamprop_h

/// Property IDs.
enum DCAMIDPROP
{
    DCAM_IDPROP_EXPOSURETIME            = 0x001F0110,
    DCAM_IDPROP_SENSORTEMPERATURE       = 0x00200310,
    DCAM_IDPROP_SENSORCOOLER            = 0x00200320,
    DCAM_IDPROP_SENSORTEMPERATURETARGET = 0x00200330,
    DCAM_IDPROP_SENSORCOOLERSTATUS      = 0x00200340,
    DCAM_IDPROP_SENSORCOOLERFAN         = 0x00200350,
    DCAM_IDPROP_READOUTSPEED            = 0x00400110,
    DCAM_IDPROP_BINNING                 = 0x00401110,
    DCAM_IDPROP_SUBARRAYHPOS            = 0x00402110,
    DCAM_IDPROP_SUBARRAYHSIZE           = 0x00402120,
    DCAM_IDPROP_SUBARRAYVPOS            = 0x00402130,
    DCAM_IDPROP_SUBARRAYVSIZE           = 0x00402140,
    DCAM_IDPROP_SUBARRAYMODE            = 0x00402150,
    DCAM_IDPROP_TIMING_READOUTTIME      = 0x00403010,
    DCAM_IDPROP_INTERNALFRAMERATE       = 0x00403810,
    DCAM_IDPROP_IMAGE_WIDTH             = 0x00420210,
    DCAM_IDPROP_IMAGE_HEIGHT            = 0x00420220,
    DCAM_IDPROP_IMAGE_ROWBYTES          = 0x00420230,
    DCAM_IDPROP_IMAGE_FRAMEBYTES        = 0x00420240,
    DCAM_IDPROP_BITSPERCHANNEL          = 0x00420530,
    DCAM_IDPROP_FRAMEBUNDLE_NUMBER      = 0x00421020
};

/// Property attribute flags.
enum DCAMPROPATTRIBUTE
{
    DCAMPROP_ATTR_WRITABLE = 0x00020000
};

/// Property values.
enum DCAMPROPMODEVALUE
{
    DCAMPROP_MODE__OFF                   = 1,
    DCAMPROP_MODE__ON                    = 2,
    DCAMPROP_SENSORCOOLER__OFF           = 1,
    DCAMPROP_SENSORCOOLER__ON            = 2,
    DCAMPROP_SENSORCOOLER__MAX           = 4,
    DCAMPROP_SENSORCOOLERSTATUS__OFF     = 1,
    DCAMPROP_SENSORCOOLERSTATUS__READY   = 2,
    DCAMPROP_SENSORCOOLERSTATUS__BUSY    = 3,
    DCAMPROP_SENSORCOOLERSTATUS__WARNING = 5
};

/// dcamprop_getattr() parameters.
struct DCAMPROP_ATTR
{
    int32  cbSize;       ///< Size of this structure.
    int32  iProp;        ///< Property ID.
    int32  option;       ///< Option.
    int32  iReserved1;   ///< Reserved.
    int32  attribute;    ///< [out] Attribute flags.
    int32  iGroup;       ///< [out] Group.
    int32  iUnit;        ///< [out] Unit.
    int32  attribute2;   ///< [out] More attribute flags.
    double valuemin;     ///< [out] Minimum value.
    double valuemax;     ///< [out] Maximum value.
    double valuestep;    ///< [out] Step.
    double valuedefault; ///< [out] Default value.
};

#endif // tests_dcam_dcamprop_h
