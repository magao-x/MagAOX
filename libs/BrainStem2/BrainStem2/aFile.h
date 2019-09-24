/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aFile.h                                                   */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Definition of a platform-independent file utility. */
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

#ifndef _aFile_H_
#define _aFile_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDefs_H_

#include "aError.h"

/////////////////////////////////////////////////////////////////////
/// Platform Independent File Access Interface

/** \defgroup aFile File Interface
 * \ref aFile "aFile.h" provides a platform independent interface for opening, reading, 
 * and writing files. 
 */

/////////////////////////////////////////////////////////////////////
/// Typedef #aFileRef Opaque reference to a file handle.
typedef void* aFileRef;


/////////////////////////////////////////////////////////////////////
/// Enum #aFileMode.

/**
 * Represents whether the file is to be opened in read or write mode.
 */
typedef enum aFileMode {
    aFileModeReadOnly, /**< File read mode. */
    aFileModeWriteOnly, /**< File write mode. */
    aFileModeAppend, /**< File write mode from end of current file. */
    aFileModeUnknown /**< File in unknown mode. */
} aFileMode;


/////////////////////////////////////////////////////////////////////
/// Enum #aFileSeekMode.

/**
 * Represents the seek start location.
 */
typedef enum aFileSeekMode {
    aSeekStart, /**< Perform a seek from the beginning of the file. */
    aSeekCurrent, /**< Perform a seek from the current location. */
    aSeekEnd /**< Perform a seek from the end of the file. */
} aFileSeekMode;


#ifdef __cplusplus
extern "C" {
#endif
    
    /////////////////////////////////////////////////////////////////////
    /// Does the File Exist.
    
    /**
     * Checks for the existence of a file at filename.
     * \param pFilename path to file.
     * \returns bool True if file exists, false otherwise.
     */
    aLIBEXPORT bool aFile_Exists(const char* pFilename);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Open a File.
    
    /**
     * Opens the file Given in pFilename with the given fileMode eMode.
     * \param pFilename path to file.
     * \param eMode Open the file for Reading or Writing.
     * \returns aFileRef on success or NULL on failure.
     */
    aLIBEXPORT aFileRef aFile_Open(const char* pFilename, const aFileMode eMode);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Close an open file.
    
    /**
     * Close an open file. The fileRef is set to NULL on success.
     * \param fileRef Pointer to the handle to the open file.
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrParam invalid file reference.
     */
    aLIBEXPORT aErr aFile_Close(aFileRef* fileRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read from an open file.
    
    /**
     * Read from an open file.
     *
     * \param fileRef The handle to the open file.
     * \param pBuffer The data buffer to read into.
     * \param nLength The length of the read buffer.
     * \param pActuallyRead The Number of bytes actually read from the file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrMode The file is not readable.
     * \retval aErrIO An error occured reading from the file.
     * \retval aErrEOF Read reached the end of the file.
     */
    aLIBEXPORT aErr aFile_Read(aFileRef fileRef,
                               uint8_t* pBuffer,
                               const size_t nLength,
                               size_t* pActuallyRead);
    
    /////////////////////////////////////////////////////////////////////
    /// Write to an open file.
    
    /**
     * Write to an open file.
     *
     * \param fileRef The handle to the open file.
     * \param pBuffer The data to write.
     * \param nLength The length of the data to write.
     * \param pActuallyWritten The Number of bytes actually written to the file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrMode The file is not writable.
     * \retval aErrIO An error occured writing to the file.
     */
    aLIBEXPORT aErr aFile_Write(aFileRef fileRef,
                                const uint8_t* pBuffer,
                                const size_t nLength,
                                size_t* pActuallyWritten);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Seek within an open file.
    
    /**
     * Seek within an open file.
     *
     * \param fileRef The handle to the open file.
     * \param nOffset The number of bytes to move within the file.
     * \param seekFrom The location to begin the seek from.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrEOF Seek would run off the end of the file.
     * \retval aErrRange Seek would run off the beginning of the file.
     * \retval aErrIO An error occured moving the file pointer.
     */
    aLIBEXPORT aErr aFile_Seek(aFileRef fileRef,
                               const long nOffset,
                               aFileSeekMode  seekFrom);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Get the size of an open file.
    
    /**
     * Get the size of an open file.
     *
     * \param fileRef The handle to the open file.
     * \param pulSize Out param filled with the size of the open file.
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrParam the fileRef is invalid.
     * \retval aErrIO an error occured calculating the size.
     */
    aLIBEXPORT aErr aFile_GetSize(aFileRef fileRef, size_t* pulSize);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Delete a File.
    
    /**
     * Deletes the given file pFilename.
     * \param pFilename Path to file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrPermission user has insufficient priviledges.
     * \retval aErrNotFound if the file cannot be located.
     */
    aLIBEXPORT aErr aFile_Delete(const char *pFilename);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read a line from an open file.
    /**
     * Read a line from an open file.
     *
     * This function will treat the file as an ASCII file,
     * even though the file is opened for binary operations.
     *
     * \param fileRef The handle to the open file.
     * \param pBuffer The data buffer to read into.
     * \param nLength The length of the read buffer.
     * \param pActuallyRead The Number of bytes actually read from the file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrMode The file is not readable.
     * \retval aErrIO An error occured reading from the file.
     * \retval aErrEOF Read reached the end of the file.
     */
    aLIBEXPORT aErr aFile_ReadLine(aFileRef fileRef,
                                   uint8_t* pBuffer,
                                   const size_t nLength,
                                   size_t* pActuallyRead);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Set the I/O for this file to be unbuffered.
    /**
     * Set the I/O for this file to be unbuffered. This function
     * call must be performed before any other operations are made on 
     * an open file.
     *
     * \param fileRef The handle to the open file.
     *
     * \returns Function returns aErrMode on error.
     * \retval aErrNone Success.
     */
    aLIBEXPORT aErr aFile_SetUnbuffered(aFileRef fileRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Get the current position of this file.
    /**
     * Get the current position of this file.
     *
     * \param fileRef The handle to the open file.
     * \param pPos The current position of the file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     */
    aLIBEXPORT aErr aFile_GetPosition(aFileRef fileRef, size_t* pPos);
    
    /////////////////////////////////////////////////////////////////////
    /// Write formatted string to an open file.
    /**
     * Write formatted string to an open file.
     *
     * This function will treat the file as an ASCII file,
     * even though the file is opened for binary operations.
     *
     * \param fileRef The handle to the open file.
     * \param pFormat The data buffer to read into.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone Success.
     * \retval aErrMode The file is not writable.
     * \retval aErrIO An error occured writing to the file.
     */
    aLIBEXPORT aErr aFile_Printf(aFileRef fileRef,
                                 const char* pFormat,
                                 ...);


#ifdef __cplusplus
}
#endif

#endif /* _aFile_H_ */
