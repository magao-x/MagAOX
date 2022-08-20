/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aMTMPM1.h	 	  	                           //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: BrainStem MTM-PM1 module object.                   //
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

#ifndef __aMTMPM1_H__
#define __aMTMPM1_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aMTMPM1_MODULE_BASE_ADDRESS                                 6

#define aMTMPM1_NUM_APPS                                            4
#define aMTMPM1_NUM_DIGITALS                                        2
#define aMTMPM1_NUM_I2C                                             1
#define aMTMPM1_NUM_POINTERS                                        4
#define aMTMPM1_NUM_RAILS                                           2
#define   aMTMPM1_RAIL0                                             0
#define   aMTMPM1_RAIL1                                             1
#define   aMTMPM1_MAX_MICROVOLTAGE                            5000000
#define   aMTMPM1_MIN_MICROVOLTAGE                            1800000
#define   aMTMPM1_MAX_CURRENT_LIMIT_MICROAMPS                 3000000
#define   aMTMPM1_MIN_CURRENT_LIMIT_MICROAMPS                       0

#define aMTMPM1_NUM_STORES                                          2
#define   aMTMPM1_NUM_INTERNAL_SLOTS                               12
#define   aMTMPM1_NUM_RAM_SLOTS                                     1

#define aMTMPM1_NUM_TEMPERATURES                                    1
#define aMTMPM1_NUM_TIMERS                                          8



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



class aMTMPM1 : public Module
{
public:

    aMTMPM1(const uint8_t module = aMTMPM1_MODULE_BASE_ADDRESS,
            bool bAutoNetworking = true,
            const uint8_t model = aMODULE_TYPE_MTM_PM_1) :
    Module(module, bAutoNetworking, model)
    {
        app[0].init(this, 0);
        app[1].init(this, 1);
        app[2].init(this, 2);
        app[3].init(this, 3);
        
        digital[0].init(this, 0);
        digital[1].init(this, 1);

        i2c[0].init(this, 0);
        
        pointer[0].init(this, 0);
        pointer[1].init(this, 1);
        pointer[2].init(this, 2);
        pointer[3].init(this, 3);
        
        rail[0].init(this, 0);
        rail[1].init(this, 1);
        
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
        


    }
    
    AppClass app[aMTMPM1_NUM_APPS];
    DigitalClass digital[aMTMPM1_NUM_DIGITALS];
    I2CClass i2c[aMTMPM1_NUM_I2C];
    PointerClass pointer[aMTMPM1_NUM_POINTERS];
    RailClass rail[aMTMPM1_NUM_RAILS];
    StoreClass store[aMTMPM1_NUM_STORES];
    SystemClass system;
    TemperatureClass temperature;
    TimerClass timer[aMTMPM1_NUM_TIMERS];
    
    
};

#endif /* __aMTMPM1_H__ */
