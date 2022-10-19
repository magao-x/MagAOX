/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aUSBStem.h                                            //
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

#ifndef __aUSBStem_H__
#define __aUSBStem_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aUSBSTEM_MODULE                             a40PINSTEM_MODULE

#define aUSBSTEM_NUM_STORES                     a40PINSTEM_NUM_STORES
#define   aUSBSTEM_NUM_INTERNAL_SLOTS   a40PINSTEM_NUM_INTERNAL_SLOTS
#define   aUSBSTEM_NUM_RAM_SLOTS             a40PINSTEM_NUM_RAM_SLOTS
#define   aUSBSTEM_NUM_SD_SLOTS               a40PINSTEM_NUM_SD_SLOTS

#define aUSBSTEM_NUM_A2D                           a40PINSTEM_NUM_A2D
#define aUSBSTEM_NUM_APPS                         a40PINSTEM_NUM_APPS
#define aUSBSTEM_NUM_CLOCK                       a40PINSTEM_NUM_CLOCK
#define aUSBSTEM_NUM_DIG                           a40PINSTEM_NUM_DIG
#define aUSBSTEM_NUM_I2C                           a40PINSTEM_NUM_I2C
#define aUSBSTEM_NUM_POINTERS                 a40PINSTEM_NUM_POINTERS
#define aUSBSTEM_NUM_SERVOS                     a40PINSTEM_NUM_SERVOS
#define aUSBSTEM_NUM_TIMERS                     a40PINSTEM_NUM_TIMERS


class aUSBStem : public a40PinModule
{
public:

    aUSBStem(const uint8_t module = aUSBSTEM_MODULE,
             bool bAutoNetworking = true,
             const uint8_t model = aMODULE_TYPE_USBStem_1) :
    a40PinModule(module, bAutoNetworking, model)
    {
        
    }
    
};

#endif
