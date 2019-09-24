/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aUSBCSwitch.h	 	  	                                   //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: USBCSwitch C++ Module object.                      //
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

#ifndef __aUSBCSwitch_H__
#define __aUSBCSwitch_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aUSBCSWITCH_MODULE                                      6

#define aUSBCSWITCH_NUM_APPS                                    4
#define aUSBCSWITCH_NUM_POINTERS                                4
#define aUSBCSWITCH_NUM_STORES                                  2
#define   aUSBCSWITCH_NUM_INTERNAL_SLOTS                        12
#define   aUSBCSWITCH_NUM_RAM_SLOTS                             1
#define aUSBCSWITCH_NUM_TIMERS                                  8
#define aUSBCSWITCH_NUM_USB                                     1
#define aUSBCSWITCH_NUM_MUX                                     1
#define aUSBCSWITCH_NUM_EQ                                      2
#define   aUSBCSWITCH_NUM_MUX_CHANNELS                          4

//////////////////////////////////////////////////////////////////////////////

// Bit defines for port state UInt32
// use _BIT(usbPortStateXX) from aDefs.h to get bit value. i.e if (state & _BIT(usbPortState))
#define usbPortStateVBUS                               0
#define usbPortStateUSB2A                              1
#define usbPortStateUSB2B                              2
#define usbPortStateSBU                                3
#define usbPortStateSS1                                4
#define usbPortStateSS2                                5
#define usbPortStateCC1                                6
#define usbPortStateCC2                                7
#define set_usbPortStateCOM_ORIENT_STATUS(var, state)  ((var & ~(3 << 8 )) | (state << 8))
#define get_usbPortStateCOM_ORIENT_STATUS(var)         ((var &  (3 << 8 )) >> 8)
#define set_usbPortStateMUX_ORIENT_STATUS(var, state)  ((var & ~(3 << 10 )) | (state << 10))
#define get_usbPortStateMUX_ORIENT_STATUS(var)         ((var &  (3 << 10 )) >> 10)
#define set_usbPortStateSPEED_STATUS(var, state)       ((var & ~(3 << 12)) | (state << 12))
#define get_usbPortStateSPEED_STATUS(var)              ((var &  (3 << 12)) >> 12)
#define usbPortStateCCFlip                             14
#define usbPortStateSSFlip                             15
#define usbPortStateSBUFlip                            16
#define usbPortStateUSB2Flip                           17
#define set_usbPortStateDaughterCard(var, state)       ((var & ~(3 << 18)) | (state << 18))
#define get_usbPortStateDaughterCard(var)              ((var & (3 << 18)) >> 18)
#define usbPortStateErrorFlag                          20
#define usbPortStateUSB2Boost                          21
#define usbPortStateUSB3Boost                          22
#define usbPortStateConnectionEstablished              23
#define usbPortStateCC1Inject                          26
#define usbPortStateCC2Inject                          27
#define usbPortStateCC1Detect                          28
#define usbPortStateCC2Detect                          29
#define usbPortStateCC1LogicState                      30
#define usbPortStateCC2LogicState                      31

// State defines for 2 bit orientation state elements.
#define usbPortStateOff                                0
#define usbPortStateSideA                              1
#define usbPortStateSideB                              2
#define usbPortStateSideUndefined                      3

using Acroname::BrainStem::Module;
using Acroname::BrainStem::Link;
using Acroname::BrainStem::AppClass;
using Acroname::BrainStem::MuxClass;
using Acroname::BrainStem::PointerClass;
using Acroname::BrainStem::StoreClass;
using Acroname::BrainStem::SystemClass;
using Acroname::BrainStem::TimerClass;
using Acroname::BrainStem::USBClass;
using Acroname::BrainStem::EqualizerClass;


class aUSBCSwitch : public Module
{
public:
    
    aUSBCSwitch(const uint8_t module = aUSBCSWITCH_MODULE,
                bool bAutoNetworking = true,
                const uint8_t model = aMODULE_TYPE_USBC_Switch) :
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
        
        mux.init(this, 0);
        
        timer[0].init(this, 0);
        timer[1].init(this, 1);
        timer[2].init(this, 2);
        timer[3].init(this, 3);
        timer[4].init(this, 4);
        timer[5].init(this, 5);
        timer[6].init(this, 6);
        timer[7].init(this, 7);
        
        usb.init(this, 0);
        
        equalizer[equalizer2p0].init(this, equalizer2p0);
        equalizer[equalizer3p0].init(this, equalizer3p0);
    }
    
    AppClass app[aUSBCSWITCH_NUM_APPS];
    MuxClass mux;
    PointerClass pointer[aUSBCSWITCH_NUM_POINTERS];
    StoreClass store[aUSBCSWITCH_NUM_STORES];
    SystemClass system;
    TimerClass timer[aUSBCSWITCH_NUM_TIMERS];
    USBClass usb;
    EqualizerClass equalizer[aUSBCSWITCH_NUM_EQ];
    
    enum EQUALIZER_3P0_TRANSMITTER_CONFIGS {
        MUX_1db_COM_0db_900mV = 0,
        MUX_0db_COM_1db_900mV,
        MUX_1db_COM_1db_900mV,
        MUX_0db_COM_0db_900mV,
        MUX_0db_COM_0db_1100mV,
        MUX_1db_COM_0db_1100mV,
        MUX_0db_COM_1db_1100mV,
        MUX_2db_COM_2db_1100mV,
        MUX_0db_COM_0db_1300mV,
    };
    
    enum EQUALIZER_3P0_RECEIVER_CONFIGS {
        LEVEL_1_3P0 = 0,
        LEVEL_2_3P0,
        LEVEL_3_3P0,
        LEVEL_4_3P0,
        LEVEL_5_3P0,
        LEVEL_6_3P0,
        LEVEL_7_3P0,
        LEVEL_8_3P0,
        LEVEL_9_3P0,
        LEVEL_10_3P0,
        LEVEL_11_3P0,
        LEVEL_12_3P0,
        LEVEL_13_3P0,
        LEVEL_14_3P0,
        LEVEL_15_3P0,
        LEVEL_16_3P0,
    };
    
    enum EQUALIZER_2P0_TRANSMITTER_CONFIGS {
        TRANSMITTER_2P0_40mV = 0,
        TRANSMITTER_2P0_60mV,
        TRANSMITTER_2P0_80mV,
    };
    
    enum EQUALIZER_2P0_RECEIVER_CONFIGS {
        LEVEL_1_2P0 = 0,
        LEVEL_2_2P0,
    };
    
    enum EQUALIZER_CHANNELS {
        BOTH = 0,
        MUX,
        COMMON
    };
    
    enum daughtercard_type {
        NO_DAUGHTERCARD = 0,
        PASSIVE_DAUGHTERCARD,
        REDRIVER_DAUGHTERCARD,
        UNKNOWN_DAUGHTERCARD
    };
    
};



#endif /* aUSBCSwitch_h */
