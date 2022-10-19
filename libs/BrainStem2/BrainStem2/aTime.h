/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aTime.h                                                   */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Definition of platform-independent time utils	   */
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

#ifndef _aTime_H_
#define _aTime_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDefs_H_

#include "aError.h"

/////////////////////////////////////////////////////////////////////
/// Basic Time procedures Sleep and Get process tics.

/** \defgroup aTime Time Interface
 * \ref aTime "aTime.h" provides a platform independent interface for
 * millisecond sleep, and for getting process tics.
 */

#ifdef __cplusplus
extern "C" {
#endif
    
    /////////////////////////////////////////////////////////////////////
    /// Get the current tick count in milliseconds.
    
    /**
     * This call returns a number of milliseconds. Depending on the platform,
	 * this can be the number of milliseconds since the last boot, or from the
	 * epoc start. As such, this call should not be used as an external reference
	 * clock. It is accurate when used as a differential, i.e. internal, measurement only.
     *
     * \return unsigned long number of milliseconds elapsed. 
     */
    aLIBEXPORT unsigned long aTime_GetMSTicks(void);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Sleep the current process for msTime milliseconds.
    
    /**
     * Sleeps the current process. This is not an active sleep, there are no
     * signals which will "wake" the process.
     *
     * \param msTime Milliseconds to sleep.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone The call returned successfully.
     * \retval aErrUnknown Um unknown what went wrong.
     */
    aLIBEXPORT aErr aTime_MSSleep(const unsigned long msTime);
    
#ifdef __cplusplus
}
#endif

#endif /* _aTime_H_ */
