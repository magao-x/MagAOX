/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aMTMDAQ1.h	 	  	                                       //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: MTMDAQ1 C++ Module object.                         //
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

#ifndef __aMTMDAQ1_H__
#define __aMTMDAQ1_H__

#include "BrainStem-all.h"
#include "aProtocoldefs.h"

#define aMTMDAQ1_MODULE_BASE_ADDRESS                             10


#define aMTMDAQ1_NUM_APPS                                        4
#define aMTMDAQ1_NUM_POINTERS                                    4
#define aMTMDAQ1_NUM_STORES                                      2
#define   aMTMDAQ1_NUM_INTERNAL_SLOTS                           12
#define   aMTMDAQ1_NUM_RAM_SLOTS                                 1
#define aMTMDAQ1_NUM_TIMERS                                      8
#define aMTMDAQ1_NUM_DIGITALS                                    2
#define aMTMDAQ1_NUM_ANALOGS                                    18
#define   aMTMDAQ1_NUM_ANALOG_INPUTS                            16
#define   aMTMDAQ1_NUM_ANALOG_OUTPUTS                            2
#define aMTMDAQ1_NUM_I2C                                         1


using Acroname::BrainStem::Module;
using Acroname::BrainStem::Link;
using Acroname::BrainStem::AnalogClass;
using Acroname::BrainStem::AppClass;
using Acroname::BrainStem::DigitalClass;
using Acroname::BrainStem::I2CClass;
using Acroname::BrainStem::PointerClass;
using Acroname::BrainStem::AnalogClass;
using Acroname::BrainStem::StoreClass;
using Acroname::BrainStem::SystemClass;
using Acroname::BrainStem::TimerClass;

class aMTMDAQ1 : public Module
{
public:

    aMTMDAQ1(const uint8_t module = aMTMDAQ1_MODULE_BASE_ADDRESS,
             bool bAutoNetworking = true,
             const uint8_t model = aMODULE_TYPE_MTM_DAQ_1) :
    Module(module, bAutoNetworking, model)
    {
        analog[0].init(this, 0);
        analog[1].init(this, 1);
        analog[2].init(this, 2);
        analog[3].init(this, 3);
        analog[4].init(this, 4);
        analog[5].init(this, 5);
        analog[6].init(this, 6);
        analog[7].init(this, 7);
        analog[8].init(this, 8);
        analog[9].init(this, 9);
        analog[10].init(this, 10);
        analog[11].init(this, 11);
        analog[12].init(this, 12);
        analog[13].init(this, 13);
        analog[14].init(this, 14);
        analog[15].init(this, 15);
        analog[16].init(this, 16);
        analog[17].init(this, 17);
        analog[18].init(this, 18);
        analog[19].init(this, 19);

        app[0].init(this, 0);
        app[1].init(this, 1);
        app[2].init(this, 2);
        app[3].init(this, 3);

        digital[0].init(this, 0);
        digital[1].init(this, 1);

        i2c[0].init(this,0);

        pointer[0].init(this, 0);
        pointer[1].init(this, 1);
        pointer[2].init(this, 2);
        pointer[3].init(this, 3);

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

    AnalogClass analog[aMTMDAQ1_NUM_ANALOGS];
    AppClass app[aMTMDAQ1_NUM_APPS];
    DigitalClass digital[aMTMDAQ1_NUM_DIGITALS];
    I2CClass i2c[aMTMDAQ1_NUM_I2C];
    PointerClass pointer[aMTMDAQ1_NUM_POINTERS];
    StoreClass store[aMTMDAQ1_NUM_STORES];
    SystemClass system;
    TimerClass timer[aMTMDAQ1_NUM_TIMERS];

    static const std::list<uint8_t>& getSingleEndedInputRanges(void) {
        static const uint8_t arr[] = {
            analogRange_P10V24N10V24,
            analogRange_P5V12N5V12,
            analogRange_P2V56N2V56,
            analogRange_P1V28N1V28,
            analogRange_P0V64N0V64,
            analogRange_P10V24N0V0,
            analogRange_P5V12N0V0,
            analogRange_P2V56N0V0,
            analogRange_P1V28N0V0
        };
        static const std::list<uint8_t> singleEndedInputRanges(arr, arr + sizeof(arr)/sizeof(uint8_t));
        return singleEndedInputRanges;
    };

    static const std::list<uint8_t>& getDifferentialInputRanges(void) {
        static const uint8_t arr[] = {
            analogRange_P0V064N0V064,
            analogRange_P0V64N0V64,
            analogRange_P0V128N0V128,
            analogRange_P1V28N1V28,
            analogRange_P0V256N0V256,
            analogRange_P2V56N2V56,
            analogRange_P0V512N0V512,
            analogRange_P5V12N5V12,
            analogRange_P1V024N1V024,
            analogRange_P10V24N10V24
        };
        static const std::list<uint8_t> differentialInputRanges(arr, arr + sizeof(arr)/sizeof(uint8_t));
        return differentialInputRanges;
    };

    static const std::list<uint8_t>& getOutputRanges(void) {
        static const uint8_t arr[] = {
            analogRange_P10V24N10V24,
            analogRange_P4V096N0V0,
            analogRange_P2V048N0V0
        };
        static const std::list<uint8_t> outputRanges(arr, arr + sizeof(arr)/sizeof(uint8_t));
        return outputRanges;
    };
};

#endif /* __aMTMDAQ1_H__ */
