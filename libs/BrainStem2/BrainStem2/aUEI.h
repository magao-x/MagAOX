/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aUEI.h                                                    */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: UEI processing utilities.                          */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* Copyright (c) 2018 Acroname Inc. - All Rights Reserved          */
/*                                                                 */
/* This file is part of the BrainStem release. See the license.txt */
/* file included with this package or go to                        */
/* https://acroname.com/software/brainstem-development-kit         */
/* for full license details.                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _aUEI_H_
#define _aUEI_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif

/////////////////////////////////////////////////////////////////////
/// UEI Utilities.

/** \defgroup aUEI UEI Utilities
 * \ref aUEI "aUEI.h" Provides structs and utilities for working with UEIs.
 */

// Mark: -UEI data struct

/// Typedef Enum #dataType 

/**
 * UEI datatype
 */
typedef enum {
    aUEI_VOID = 0, ///< Void datatype.
    aUEI_BYTE = 1, ///< Char datatype.
    aUEI_SHORT = 2, ///< Short datatype.
    aUEI_INT = 4, ///< Int datatype.
} dataType;

/// Typedef Struct #uei

/**
 * UEI data struct.
 */
typedef struct {
    uint8_t module; ///< Module address.
    uint8_t command; ///< Command code.
    uint8_t option; ///< option code & UEI operation.
    uint8_t specifier; ///< Entity index & response specifier.
    union { 
        uint8_t byteVal; ///< Char value union member.
        uint16_t shortVal; ///< Short value union member.
        uint32_t intVal; ///< Int value union member.
    } v;
    
    dataType type; ///< Union dataType.
} uei;


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

    /// Retreive a short from a UEI.

    /**
     * \param p Pointer to byte array containing short.
     * \returns uint16_t The short value.
     */
    aLIBEXPORT uint16_t aUEI_RetrieveShort(const uint8_t* p);

    /// Store a short in a UEI.

    /**
     * \param p Pointer to uei shortVal.
     * \param v Short value to store.
     */
    void aUEI_StoreShort(uint8_t* p, uint16_t v);
    
    /// Retreive an Int from a UEI.

    /**
     * \param p Pointer to byte array containing the Int.
     * \returns uint32_t The integer value.
     */
    aLIBEXPORT uint32_t aUEI_RetrieveInt(const uint8_t* p);
    
    /// Store an Int in a UEI.

    /** 
     * \param p Pointer to the IntVal of a UEI.
     * \param v The value to store.
     */
    void aUEI_StoreInt(uint8_t* p, uint32_t v);
    
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // _aReflex_H_
