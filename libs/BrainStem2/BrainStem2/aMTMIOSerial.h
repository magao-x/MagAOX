/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aMTMIOSerial.h	 	  	                           //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: BrainStem MTM IO Serial module object.             //
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

#ifndef __aMTMIOSerial_H__
#define __aMTMIOSerial_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aMTMIOSERIAL_MODULE_BASE_ADDRESS                           8

#define aMTMIOSERIAL_NUM_APPS                                      4
#define aMTMIOSERIAL_NUM_DIGITALS                                  8
#define aMTMIOSERIAL_NUM_I2C                                       1

#define aMTMIOSERIAL_NUM_UART                                      4

#define aMTMIOSERIAL_NUM_POINTERS                                  4

#define aMTMIOSERIAL_NUM_RAILS                                     3
#define   aMTMIOSERIAL_5VRAIL                                      0
#define   aMTMIOSERIAL_ADJRAIL1                                    1
#define   aMTMIOSERIAL_ADJRAIL2                                    2
#define   aMTMIOSERIAL_MAX_MICROVOLTAGE                      5000000
#define   aMTMIOSERIAL_MIN_MICROVOLTAGE                      1800000

#define aMTMIOSERIAL_NUM_SERVOS                                    8

#define aMTMIOSERIAL_NUM_STORES                                    2
#define   aMTMIOSERIAL_NUM_INTERNAL_SLOTS                         12
#define   aMTMIOSERIAL_NUM_RAM_SLOTS                               1

#define aMTMIOSERIAL_NUM_TIMERS                                    8

#define aMTMIOSERIAL_NUM_USB                                       1
/* cmdUSB number of Channels */
#define   aMTMIOSERIAL_USB_NUM_CHANNELS                            4
/* Upstream Modes */
#define   aUSB_UPSTREAM_CONFIG_AUTO                                0 /* (Default) */
#define   aUSB_UPSTREAM_CONFIG_ONBOARD                             1
#define   aUSB_UPSTREAM_CONFIG_EDGE                                2
/* Upstream states */
#define   aUSB_UPSTREAM_ONBOARD                                    0
#define   aUSB_UPSTREAM_EDGE                                       1

// Bit defines for port state UInt32
// use _BIT(X) from aDefs.h to get bit value.
// i.e if (state & _BIT(aMTMIOSERIAL_USB_VBUS_ENABLED))
#define aMTMIOSERIAL_USB_VBUS_ENABLED                              0
#define aMTMIOSERIAL_USB2_DATA_ENABLED                             1
#define aMTMIOSERIAL_USB_ERROR_FLAG                               19
#define aMTMIOSERIAL_USB2_BOOST_ENABLED                           20

// Bit defines for port error UInt32
// use _BIT(X) from aDefs.h to get bit value.
// i.e if (error & _BIT(aMTMIOSERIAL_ERROR_VBUS_OVERCURRENT))
#define aMTMIOSERIAL_ERROR_VBUS_OVERCURRENT                        0


using Acroname::BrainStem::Module;
using Acroname::BrainStem::Link;
using Acroname::BrainStem::AppClass;
using Acroname::BrainStem::DigitalClass;
using Acroname::BrainStem::I2CClass;
using Acroname::BrainStem::PointerClass;
using Acroname::BrainStem::RailClass;
using Acroname::BrainStem::StoreClass;
using Acroname::BrainStem::SystemClass;
using Acroname::BrainStem::TemperatureClass;
using Acroname::BrainStem::TimerClass;
using Acroname::BrainStem::USBClass;
using Acroname::BrainStem::RCServoClass;
using Acroname::BrainStem::UARTClass;


class aMTMIOSerial : public Module
{
public:

    aMTMIOSerial(const uint8_t module = aMTMIOSERIAL_MODULE_BASE_ADDRESS,
                 bool bAutoNetworking = true,
                 const uint8_t model = aMODULE_TYPE_MTMIOSerial_1) :
    Module(module, bAutoNetworking, model)
    {

        digital[0].init(this, 0);
        digital[1].init(this, 1);
        digital[2].init(this, 2);
        digital[3].init(this, 3);
        digital[4].init(this, 4);
        digital[5].init(this, 5);
        digital[6].init(this, 6);
        digital[7].init(this, 7);

        app[0].init(this, 0);
        app[1].init(this, 1);
        app[2].init(this, 2);
        app[3].init(this, 3);

        i2c[0].init(this, 0);
        
        uart[0].init(this, 0);
        uart[1].init(this, 1);
        uart[2].init(this, 2);
        uart[3].init(this, 3);
        
        pointer[0].init(this, 0);
        pointer[1].init(this, 1);
        pointer[2].init(this, 2);
        pointer[3].init(this, 3);

        rail[aMTMIOSERIAL_5VRAIL].init(this, 0);
        rail[aMTMIOSERIAL_ADJRAIL1].init(this, 1);
        rail[aMTMIOSERIAL_ADJRAIL2].init(this, 2);

        servo[0].init(this, 0);
        servo[1].init(this, 1);
        servo[2].init(this, 2);
        servo[3].init(this, 3);
        servo[4].init(this, 4);
        servo[5].init(this, 5);
        servo[6].init(this, 6);
        servo[7].init(this, 7);

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

    AppClass app[aMTMIOSERIAL_NUM_APPS];
    DigitalClass digital[aMTMIOSERIAL_NUM_DIGITALS];
    I2CClass i2c[aMTMIOSERIAL_NUM_I2C];
    UARTClass uart[aMTMIOSERIAL_NUM_UART];
    PointerClass pointer[aMTMIOSERIAL_NUM_POINTERS];
    RailClass rail[aMTMIOSERIAL_NUM_RAILS];
    RCServoClass servo[aMTM_STEM_NUM_SERVOS];
    StoreClass store[aMTMIOSERIAL_NUM_STORES];
    SystemClass system;
    TemperatureClass temperature;
    TimerClass timer[aMTMIOSERIAL_NUM_TIMERS];
    USBClass usb;


};

#endif /* __aMTMIOSerial_H__ */
