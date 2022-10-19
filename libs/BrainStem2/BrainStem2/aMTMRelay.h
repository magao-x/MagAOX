/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aMTMRelay.h	 	  	                                   //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: BrainStem MTM-RELAY module object.                 //
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

#ifndef __aMTMRelay_H__
#define __aMTMRelay_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aMTMRELAY_MODULE_BASE_ADDRESS                              12

#define aMTMRELAY_NUM_APPS                                          4
#define aMTMRELAY_NUM_DIGITALS                                      4
#define aMTMRELAY_NUM_I2C                                           1
#define aMTMRELAY_NUM_POINTERS                                      4
#define aMTMRELAY_NUM_RELAYS                                        4
#define aMTMRELAY_NUM_STORES                                        2
#define   aMTMRELAY_NUM_INTERNAL_SLOTS                             12
#define   aMTMRELAY_NUM_RAM_SLOTS                                   1
#define aMTMRELAY_NUM_TIMERS                                        8


using Acroname::BrainStem::Module;
using Acroname::BrainStem::Link;
using Acroname::BrainStem::AppClass;
using Acroname::BrainStem::DigitalClass;
using Acroname::BrainStem::I2CClass;
using Acroname::BrainStem::PointerClass;
using Acroname::BrainStem::RelayClass;
using Acroname::BrainStem::StoreClass;
using Acroname::BrainStem::SystemClass;
using Acroname::BrainStem::TimerClass;


class aMTMRelay : public Module
{
public:
    aMTMRelay(const uint8_t module = aMTMRELAY_MODULE_BASE_ADDRESS,
              bool bAutoNetworking = true,
              const uint8_t model = aMODULE_TYPE_MTM_Relay) :
    Module(module, bAutoNetworking, model)
    {
        
        app[0].init(this, 0);
        app[1].init(this, 1);
        app[2].init(this, 2);
        app[3].init(this, 3);

        digital[0].init(this, 0);
        digital[1].init(this, 1);
        digital[2].init(this, 2);
        digital[3].init(this, 3);

        i2c[0].init(this, 0);
        
        pointer[0].init(this, 0);
        pointer[1].init(this, 1);
        pointer[2].init(this, 2);
        pointer[3].init(this, 3);

        relay[0].init(this, 0);
        relay[1].init(this, 1);
        relay[2].init(this, 2);
        relay[3].init(this, 3);

        store[storeInternalStore].init(this, storeInternalStore);
        store[storeRAMStore].init(this, storeRAMStore);

        system.init(this, 0);

        timer[0].init(this, 0);
        timer[1].init(this, 1);
        timer[2].init(this, 2);
        timer[3].init(this, 3);
        timer[4].init(this, 4);
        timer[5].init(this, 5);
        timer[6].init(this, 6);
        timer[7].init(this, 7);

    }

    AppClass app[aMTMRELAY_NUM_APPS];
    DigitalClass digital[aMTMRELAY_NUM_DIGITALS];
    I2CClass i2c[aMTMRELAY_NUM_I2C];
    PointerClass pointer[aMTMRELAY_NUM_POINTERS];
    RelayClass relay[aMTMRELAY_NUM_RELAYS];
    StoreClass store[aMTMRELAY_NUM_STORES];
    SystemClass system;
    TimerClass timer[aMTMRELAY_NUM_TIMERS];

};

#endif /* __aMTMRelay_H__ */
