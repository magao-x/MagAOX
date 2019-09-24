/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aDefs.h                        						   */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Definition of platform-independent types    	   */
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

#ifndef _aDefs_H_
#define _aDefs_H_

/////////////////////////////////////////////////////////////////////
/// Acroname Specific Universal Defines and includes.

/** \defgroup aDefs (Acroname Include Defs)
 * The C-Interface requires some specific defines for cross platform compatibility.
 * The \ref aDefs "aDefs.h" file contains those defines and includes that are necessary at a global
 * level across platforms.
 *
 * Things like a cross platform way to specify line endings, safe c-string copy
 * and concatenation operations, and boolean typedefs when they are not defined
 * by default.
 *
 * We rely on the following std headers:
 *  - assert.h
 *  - stddef.h
 *  - stdint.h
 *  - stdbool.h
 *  - string.h
 *  - stdio.h
 *  - stdlib.h
 */

/** \def aSHOWERR(msg,error)
 * \brief A macro that will emit an error on stdout including file and line number when compiled without NDEBUG
 *        flag. When NDEBUG is defined it emits nothing.
 */

/** \def OS_NEW_LN
 * \brief A macro containing the appropriate line ending characters for a given platform \\r\\n on windows and \\n
 * elsewhere.
 *
 */

/** \def aStringCopySafe(d, l, s)
 * \brief A macro that maps to platform specific safe string copy. Parameters are;
 *       - d: destination
 *       - l: length
 *       - s: source
 */

/** \def aStringCatSafe(d, l, s)
 * \brief A macro that maps to platform specific safe string concatination. Parameters are;
 *        - d: destination
 *        - l: length
 *        - s: source
 */

/** \def aSNPRINTF
 * \brief A macro that maps to the platform specific safe printf output.
 *
 */

/** \def aLIBEXPORT
 * \brief A macro that expands for dynamic library linking on a given platform.
 */

/** \def aMemPtr
 * \brief An acroname specific semantic define for a pointer to a chunk of memory.
 */

#ifndef __include__assert_h
#define __include__assert_h
#include <assert.h>

#ifndef NDEBUG
#define aSHOWERR(msg, error)  (printf("Error in File; %s on line; %d, %s: %d\n", __FILE__, __LINE__, msg, error))
#define aPERROR(msg)    (perror(msg))
#else // Else Ignore these in Release versions.
#define aSHOWERR(ignore, ign) ((void) 0)
#define aPERROR(ignore) ((void) 0)
#endif // End if Debug target or not.

#endif // __include__assert_h

#ifndef __include__stddef_h
#define __include__stddef_h
#include <stddef.h>
#endif // __include__stddef_h

#ifndef __include__stdint_h
#define __include__stdint_h
#include <stdint.h>
#endif // __include__stdint_h

#ifndef _WIN32
#ifndef __include_stdbool_h
#define __include_stdbool_h
#include <stdbool.h>
#endif // __include_stdbool_h
#endif // not _WIN32

#ifndef __include__string_h
#define __include__string_h
#include <string.h>
#endif // __include__string_h

#ifndef __include__stdio_h
#define __include__stdio_h
#include <stdio.h>
#endif // __include_stdio_h

#ifndef __include__stdlib_h
#define __include__stdlib_h
#include <stdlib.h>
#endif // __include_stdlib_h

#ifdef _WIN32
#ifndef __cplusplus
#define bool int
#define true 1
#define false 0
#endif // __cplusplus

#define OS_NEW_LN                                         "\x0D\x0A"
#define aSNPRINTF                                          sprintf_s
#define aStringCopySafe(d, l, s)             strcpy_s((d), (l), (s))
#define aStringCatSafe(d, l, s)              strcat_s((d), (l), (s))
#define aLIBEXPORT                             __declspec(dllexport)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef SSIZE_T ssize_t;

#else // Else Not MSWindows.

#define OS_NEW_LN                                               "\n"
#define aSNPRINTF                                           snprintf
#define aStringCopySafe(d, l, s)              strncpy((d), (s), (l))
#define aStringCatSafe(d, l, s)	              strncat((d), (s), (l))
#define aLIBEXPORT           __attribute__((visibility ("default")))

#endif // end platform defines.

/* Semantic name for a block of heap allocated memory */
#define aMemPtr                                               void *
#define _BIT(x)     (1 << x)

#ifdef __cplusplus
extern "C" {
#endif
    /// Returns a printable model string.
    aLIBEXPORT const char* aDefs_GetModelName(const int modelNum);
#ifdef __cplusplus
}
#endif

#endif /* _aDefs_H_ */
