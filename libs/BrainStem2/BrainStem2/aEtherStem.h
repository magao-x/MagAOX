/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aEtherStem.h                                              //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: Definition of the Acroname 40-pin module object.   //
//                                                                 //


#ifndef __aEtherStem_H__
#define __aEtherStem_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aETHERSTEM_MODULE_ADDRESS                   a40PINSTEM_MODULE

#define aETHERSTEM_NUM_STORES                   a40PINSTEM_NUM_STORES
#define aETHERSTEM_NUM_INTERNAL_SLOTS   a40PINSTEM_NUM_INTERNAL_SLOTS
#define aETHERSTEM_NUM_RAM_SLOTS             a40PINSTEM_NUM_RAM_SLOTS
#define aETHERSTEM_NUM_SD_SLOTS               a40PINSTEM_NUM_SD_SLOTS

#define aETHERSTEM_NUM_A2D                         a40PINSTEM_NUM_A2D
#define aETHERSTEM_NUM_APPS                       a40PINSTEM_NUM_APPS
#define aETHERSTEM_NUM_CLOCK                     a40PINSTEM_NUM_CLOCK
#define aETHERSTEM_NUM_DIG                         a40PINSTEM_NUM_DIG
#define aETHERSTEM_NUM_I2C                         a40PINSTEM_NUM_I2C
#define aETHERSTEM_NUM_POINTERS               a40PINSTEM_NUM_POINTERS
#define aETHERSTEM_NUM_SERVOS                   a40PINSTEM_NUM_SERVOS
#define aETHERSTEM_NUM_TIMERS                   a40PINSTEM_NUM_TIMERS



class aEtherStem : public a40PinModule
{
public:
    
    aEtherStem(const uint8_t module = aETHERSTEM_MODULE_ADDRESS,
               bool bAutoNetworking = true,
               const uint8_t model = aMODULE_TYPE_EtherStem_1) :
    a40PinModule(module, bAutoNetworking, model)
    {
        
    }
    
};

#endif
