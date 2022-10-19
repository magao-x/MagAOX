/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aMTMUSBStem.h                                            //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: Definition of the Acroname 40-pin module object.   //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// Copyright (c) 2018 Acroname Inc. - All Rights Reserved          //
//                                                                 //
// This file is part of the BrainStem release. See the license.txt //
// file included with this package or go to                        //
// https://acroname.com/software/brainstem-development-kit         //
// for full license details.                                       //
/////////////////////////////////////////////////////////////////////

#ifndef __aMTMUSBStem_H__
#define __aMTMUSBStem_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aMTM_USBSTEM_MODULE_BASE_ADDRESS aMTM_STEM_MODULE_BASE_ADDRESS

#define aMTM_USBSTEM_NUM_STORES                   aMTM_STEM_NUM_STORES
#define   aMTM_USBSTEM_NUM_INTERNAL_SLOTS aMTM_STEM_NUM_INTERNAL_SLOTS
#define   aMTM_USBSTEM_NUM_RAM_SLOTS           aMTM_STEM_NUM_RAM_SLOTS
#define   aMTM_USBSTEM_NUM_SD_SLOTS             aMTM_STEM_NUM_SD_SLOTS

#define aMTM_USBSTEM_NUM_A2D                         aMTM_STEM_NUM_A2D
#define aMTM_USBSTEM_NUM_APPS                       aMTM_STEM_NUM_APPS
#define aMTM_USBSTEM_NUM_CLOCK                     aMTM_STEM_NUM_CLOCK
#define aMTM_USBSTEM_NUM_DIG                         aMTM_STEM_NUM_DIG
#define aMTM_USBSTEM_NUM_I2C                         aMTM_STEM_NUM_I2C
#define aMTM_USBSTEM_NUM_POINTERS               aMTM_STEM_NUM_POINTERS
#define aMTM_USBSTEM_NUM_SERVOS                   aMTM_STEM_NUM_SERVOS
#define aMTM_USBSTEM_NUM_TIMERS                   aMTM_STEM_NUM_TIMERS


class aMTMUSBStem : public aMTMStemModule
{
public:

    aMTMUSBStem(const uint8_t module = aMTM_USBSTEM_MODULE_BASE_ADDRESS,
                bool bAutoNetworking = true,
                const uint8_t model = aMODULE_TYPE_MTM_USBStem) :
    aMTMStemModule(module, bAutoNetworking, model)
    {

    }

    
};

#endif
