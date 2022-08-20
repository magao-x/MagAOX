/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: BrainStem-all.h                                           //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: BrainStem API's and support.                       //
//                                                                 //
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

/////////////////////////////////////////////////////////////////////
/// All inclusive Header for C++ API access.

/**
 * This file can be included by users of the C++ API and includes all of
 * the necessary header files which represent the C++ API.
 */

#ifndef __BrainStem_all_H__
#define __BrainStem_all_H__

/////////////////////////////////////////////////////////////////////
/// Core Classes and type definitions for the BrainStem2 library. This
/// includes Link, Module, and the base Entity class.
#ifndef __BrainStem_core_H__
#include "BrainStem-core.h"
#endif //__BrainStem_core_H__

/////////////////////////////////////////////////////////////////////
/// Contains concrete classes for each of the BrainStem Entity Types. This
/// includes System, Store, Analog, etc.
#ifndef __BrainStem_entity_H__
#include "BrainStem-entity.h"
#endif //__BrainStem_entity_H__

#ifndef __a40PinModule_H__
#include "a40PinModule.h"
#endif

#ifndef __aMTMStemModule_H__
#include "aMTMStemModule.h"
#endif

#ifndef __aEtherStem_H__
#include "aEtherStem.h"
#endif

#ifndef __aMTMDAQ1_H__
#include "aMTMDAQ1.h"
#endif

#ifndef __aMTMEtherStem_H__
#include "aMTMEtherStem.h"
#endif

#ifndef __aMTMIOSerial_H__
#include "aMTMIOSerial.h"
#endif

#ifndef __aMTMPM1_H__
#include "aMTMPM1.h"
#endif

#ifndef __aMTMUSBStem_H__
#include "aMTMUSBStem.h"
#endif

#ifndef __aUSBStem_H__
#include "aUSBStem.h"
#endif

#ifndef __aUSBHub2x4_H__
#include "aUSBHub2x4.h"
#endif

#ifndef __aMTMRelay_H__
#include "aMTMRelay.h"
#endif

#ifndef __aUSBHub3p_H__
#include "aUSBHub3p.h"
#endif

#ifndef __aUSBCSwitch_H__
#include "aUSBCSwitch.h"
#endif

#ifndef __aMTMDAQ2_H__
#include "aMTMDAQ2.h"
#endif

#endif //__BrainStem_all_H__
