/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aUSBHub2x4.h	 	  	                           //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: USBHub2x4 C++ Module object.                       //
//                                                                 //
// build number: source                                            //
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

#ifndef __aUSBHub2x4_H__
#define __aUSBHub2x4_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aUSBHUB2X4_MODULE                                         6

#define aUSBHUB2X4_NUM_APPS                                       4
#define aUSBHUB2X4_NUM_POINTERS                                   4
#define aUSBHUB2X4_NUM_STORES                                     2
#define   aUSBHUB2X4_NUM_INTERNAL_SLOTS                          12
#define   aUSBHUB2X4_NUM_RAM_SLOTS                                1

#define aUSBHUB2X4_NUM_TIMERS                                     8
#define aUSBHUB2X4_NUM_USB                                        1
#define aUSBHUB2x4_NUM_USB_PORTS                                  4

// Bit defines for port state UInt32
// use _BIT(X) from aDefs.h to get bit value.
// i.e if (state & _BIT(aUSBHUB2X4_USB_VBUS_ENABLED))
#define aUSBHUB2X4_USB_VBUS_ENABLED                                0
#define aUSBHUB2X4_USB2_DATA_ENABLED                               1
#define aUSBHUB2X4_USB_ERROR_FLAG                                 19
#define aUSBHUB2X4_USB2_BOOST_ENABLED                             20
#define aUSBHUB2X4_DEVICE_ATTACHED                                23
#define aUSBHUB2X4_CONSTANT_CURRENT                               24

// Bit defines for port error UInt32
// use _BIT(X) from aDefs.h to get bit value.
// i.e if (error & _BIT(aUSBHUB2X4_ERROR_VBUS_OVERCURRENT))
#define aUSBHUB2X4_ERROR_VBUS_OVERCURRENT                          0
#define aUSBHUB2X4_ERROR_OVER_TEMPERATURE                          3
#define aUSBHub2X4_ERROR_DISCHARGE                                 4


using Acroname::BrainStem::Module;
using Acroname::BrainStem::Link;
using Acroname::BrainStem::AppClass;
using Acroname::BrainStem::PointerClass;
using Acroname::BrainStem::StoreClass;
using Acroname::BrainStem::SystemClass;
using Acroname::BrainStem::TemperatureClass;
using Acroname::BrainStem::TimerClass;
using Acroname::BrainStem::USBClass;

class aUSBHub2x4 : public Module
{
public:

    aUSBHub2x4(const uint8_t module = aUSBHUB2X4_MODULE,
               bool bAutoNetworking = true,
               const uint8_t model = aMODULE_TYPE_USBHub2x4) :
    Module(module, bAutoNetworking, model)
    {
        app[0].init(this, 0);
        app[1].init(this, 1);
        app[2].init(this, 2);
        app[3].init(this, 3);
        
        pointer[0].init(this, 0);
        pointer[1].init(this, 1);
        pointer[2].init(this, 2);
        pointer[3].init(this, 3);
        
        store[storeInternalStore].init(this, storeInternalStore);
        store[storeRAMStore].init(this, storeRAMStore);
        
        system.init(this, 0);
        
        temperature.init(this, 0);
        
        timer[0].init(this, 0);
        timer[1].init(this, 1);
        timer[2].init(this, 2);
        timer[3].init(this, 3);
        timer[4].init(this, 4);
        timer[5].init(this, 5);
        timer[6].init(this, 6);
        timer[7].init(this, 7);
        
        usb.init(this, 0);
    }
    
    AppClass app[aUSBHUB2X4_NUM_APPS];
    PointerClass pointer[aUSBHUB2X4_NUM_POINTERS];
    StoreClass store[aUSBHUB2X4_NUM_STORES];
    SystemClass system;
    TemperatureClass temperature;
    TimerClass timer[aUSBHUB2X4_NUM_TIMERS];
    USBClass usb;
};

#endif /* __aUSBHub2x4_H__ */
