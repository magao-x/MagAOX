
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aError.h						                           */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Common error codes.                      	       */
/*                                                                 */
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

#ifndef _aError_H_
#define _aError_H_

/////////////////////////////////////////////////////////////////////
/// Unified list of Error codes for BrainStem module Interation.

/** \defgroup aErrors Error Interface
 * \ref aErrors "aError.h" provides a unified list of error codes. These
 * error codes apply accross all API's. Library functions will return one
 * of these error codes when appropriate.
 */


/////////////////////////////////////////////////////////////////////
/// The #aErr enum lists the possible error codes for library calls.
/// BrainStem commands generally return a set of unified Error codes.
/// The API tries to be consistent and return these errors from every
/// interaction with the stem.
///
typedef enum {
    aErrNone,                     ///< 0 - Success, no error.
    aErrMemory,                   ///< 1 - Memory allocation.
    aErrParam,                    ///< 2 - Invalid parameter.
    aErrNotFound,                 ///< 3 - Not found.
    aErrFileNameLength,           ///< 4 - File name too long.
    aErrBusy,                     ///< 5 - Resource busy.
    aErrIO,                       ///< 6 - Input/Output error.
    aErrMode,                     ///< 7 - Invalid Mode.
    aErrWrite,                    ///< 8 - Write error.
    aErrRead,                     ///< 9 - Read error.
    aErrEOF,                      ///< 10 - End of file.
    aErrNotReady,                 ///< 11 - Not ready, no bytes available.
    aErrPermission,               ///< 12 - Insufficient permissions.
    aErrRange,                    ///< 13 - Value out of range.
    aErrSize,                     ///< 14 - Invalid Size.
    aErrOverrun,                  ///< 15 - Buffer/queue overrun.
    aErrParse,                    ///< 16 - Parse error.
    aErrConfiguration,            ///< 17 - Configuration error.
    aErrTimeout,                  ///< 18 - Timeout occurred.
    aErrInitialization,           ///< 19 - Initialization error.
    aErrVersion,                  ///< 20 - Invalid version.
    aErrUnimplemented,            ///< 21 - Functionality unimplemented.
    aErrDuplicate,                ///< 22 - Duplicate request.
    aErrCancel,                   ///< 23 - Cancelation occured, or did not complete.
    aErrPacket,                   ///< 24 - Packet byte invalid.
    aErrConnection,               ///< 25 - Connection error.
    aErrIndexRange,               ///< 26 - Index out of range.
    aErrShortCommand,             ///< 27 - BrainStem command to short.
    aErrInvalidEntity,            ///< 28 - Invalid entity error.
    aErrInvalidOption,            ///< 29 - Invalid option code.
    aErrResource,                 ///< 30 - Resource unavailable.
    aErrMedia,		              ///< 31 - Media error.
    aErrAsyncReturn,              ///< 32 - Asynchronous return.
    /* must be last in list */
    aErrUnknown                   ///< 33 - Unknown error.
} aErr;


#ifndef __reflex

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
    
#ifdef _WIN32
#define aLIBEXPORT                             __declspec(dllexport)
#endif
    
    // Defined with an ifndef here because arc does not understand #else.
#ifndef _WIN32 // Not Windows.
#define aLIBEXPORT           __attribute__((visibility ("default")))
#endif
    
    /////////////////////////////////////////////////////////////////////
    /// Returns a printable error string.
    aLIBEXPORT const char* aError_GetErrorText(aErr err);
    
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !__reflex

#endif /* _aError_H_ */
