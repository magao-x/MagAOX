
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: Stream.h                                                  */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Cross-Platform definition of stream I/O 		   */
/*              routines.                                          */
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

#ifndef _Stream_H_
#define _Stream_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDefs_H_

#include "aError.h"
#include "aFile.h"

/////////////////////////////////////////////////////////////////////
/// Platform Independent Stream Abstraction.

/** \defgroup aStream Stream Interface
 * \ref aStream "aStream.h" provides a platform independent stream abstraction for 
 * common I/O streams. Provides facilities for creating and destroying
 * as well as writing and reading from streams.
 */


/////////////////////////////////////////////////////////////////////
/// Typedef #aStreamRef Opaque reference to stream primitive.
typedef void* aStreamRef;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Low level operation proc defs for I/O streams.  These need to
 * be implimented for new stream types.
 */


/////////////////////////////////////////////////////////////////////
/// \defgroup StreamCallbacks (Stream Implementation Callbacks)
/// @{

/////////////////////////////////////////////////////////////////////
/// Typedef #aStreamGetProc.

/**
 * This callback is defined to read one byte from the concrete stream implementation.
 * \param pData The data Buffer to fill.
 * \param ref Opaque reference to concrete stream implementation.
 *
 * \returns Function returns aErr values.
 * \retval aErrNone Successfully read the byte.
 * \retval aErrNotReady No bytes in stream to read.
 * \retval aErrEOF Reached the end of the stream.
 * \retval aErrIO An error encountered reading from stream.
 */
typedef aErr (*aStreamGetProc)(uint8_t* pData, void* ref);


/////////////////////////////////////////////////////////////////////
/// Typedef #aStreamPutProc.

/**
 * This callback is defined to write one byte to the concrete stream implementation.
 * \param pData The data Buffer to write.
 * \param ref opaque reference to concrete stream implementation.
 *
 * \returns Function returns aErr values.
 * \retval aErrNone Successfully wrote the byte.
 * \retval aErrIO An error encountered reading from stream.
 */
typedef aErr (*aStreamPutProc)(const uint8_t* pData, void* ref);

/////////////////////////////////////////////////////////////////////
/// Typedef #aStreamDeleteProc.

/**
 * This callback is defined to destroy the concrete stream implementation.
 * \param ref opaque reference to concrete stream implementation.
 *
 * \returns Function returns aErr values.
 * \retval aErrNone Successfully destroyed.
 * \retval aErrParam Invalid ref.
 */
typedef aErr (*aStreamDeleteProc)(void* ref);


/////////////////////////////////////////////////////////////////////
/// Typedef #aStreamWriteProc. (Optional)

/**
 * Optional multi-byte write for efficiency, not required..
 * \param ref Opaque reference to concrete stream implementation.
 *
 * \returns Function returns aErr values.
 * \retval aErrNone Successfully destroyed.
 * \retval aErrIO An error encountered reading from stream.
 */
typedef aErr (*aStreamWriteProc)(const uint8_t* pData,
                                 const size_t nSize,
                                 void* ref);

////////////////////////////////////////////////////////////////////
/// @}


/////////////////////////////////////////////////////////////////////
/// Enum #aBaudRate.

/**
 * Accepted serial stream baudrates.
 */
typedef enum {
    aBAUD_2400, ///< 2400 baud
    aBAUD_4800, ///< 4800 baud
    aBAUD_9600, ///< 9600 baud
    aBAUD_19200, ///< 19,200 baud
    aBAUD_38400, ///< 38,400 baud
    aBAUD_57600, ///< 57,600 baud
    aBAUD_115200, ///< 115,200 baud
    aBAUD_230400 ///< 230,400 buad
} aBaudRate;

/////////////////////////////////////////////////////////////////////
/// Enum #aSerial_Bits.

/**
 * The accepted number of serial bits per byte.
 */
typedef enum {
    aBITS_8, ///< 8 bits
    aBITS_7 ///< 7 bits
} aSerial_Bits;


/////////////////////////////////////////////////////////////////////
/// Enum #aSerial_Stop_bits.

/**
 * The accepted number of serial stop bits.
 */
typedef enum {
    aSTOP_BITS_1, ///< 1 stop bit
    aSTOP_BITS_2 ///< 2 stop bits
} aSerial_Stop_bits;

#ifdef __cplusplus
extern "C" {
#endif
    
    
    /////////////////////////////////////////////////////////////////////
    /// Base Stream creation procedure.
    
    /**
     * Creates a Stream Reference.
     *
     * \param getProc - Callback for reading bytes from the underlying stream.
     * \param putProc - Callback for writing bytes to the underlying stream.
     * \param writeProc - Optional callback for optimized writing of multiple 
     *                    bytes.
     * \param deleteProc - Callback for safe destruction of underlying resource.
     * \param procRef - opaque reference to the underlying resource,
     *
     * \returns Function returns aStreamRef on success and NULL on error.
     */
    aLIBEXPORT aStreamRef aStream_Create(aStreamGetProc getProc,
                                         aStreamPutProc putProc,
                                         aStreamWriteProc writeProc,
                                         aStreamDeleteProc deleteProc,
                                         const void* procRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a file input stream.
    
    /**
     * Creates a file input stream.
     *
     * \param pFilename - The filename and path of the file to read from.
     * \param pStreamRef - The resulting stream accessor for the input file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrNotFound - The file to read was not found.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateFileInput(const char *pFilename,
                                            aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a file output stream.
    
    /**
     * Creates a file output stream.
     *
     * \param pFilename - The filename and path of the file to write to.
     * \param pStreamRef - The resulting stream accessor for the output file.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateFileOutput(const char *pFilename,
                                             aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a serial communication stream.
    
    /**
     * Creates a serial stream.
     *
     * \param pPortName - The portname of the serial device.
     * \param nBaudRate - The baudrate to connect to the device at.
     * \param parity - Whether serial parity is enabled.
     * \param bits - The number of bits per serial byte.
     * \param stop - The number of stop bits per byte.
     * \param pStreamRef - The resulting stream accessor for the serial device.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrConnection - The connection was unsuccessful.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateSerial(const char *pPortName,
                                         const aBaudRate nBaudRate,
                                         const bool parity,
                                         const aSerial_Bits bits,
                                         const aSerial_Stop_bits stop,
                                         aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a TCP/IP socket stream.
    
    /**
     * Creates a TCP/IP socket stream.
     *
     * \param address - The IP4 address of the connection.
     * \param port - The TCP port to connect to.
     * \param pStreamRef - The resulting stream accessor for the TCP connection.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrConnection - The connection was unsuccessful.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateSocket(const uint32_t address,
                                         const uint16_t port,
                                         aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a stream accessor for a block of memory.
    
    /**
     * Creates a stream accessor for a block of allocated memory. Reads and
     * Writes like any other stream. The memory stream does not make a copy 
     * of the memory and doesn't free it but rather provides a stream layer to
     * access it.
     *
     * \param pMemory - a pointer to a block of memory.
     * \param size - The size of the block in bytes.
     * \param pStreamRef - The resulting stream accessor for the memory block.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrParam - The memory block is invalid.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateMemory(const aMemPtr pMemory,
                                         const size_t size,
                                         aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a stream to a USB device.

    /**
     * Creates a BrainStem link stream to a USB based module.
     *
     * \param serialNum - The BrainStem serial number.
     * \param pStreamRef - The resulting stream accessor for the BrainStem 
     *                     module.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrNotFound - The brainstem device was not found.
     * \retval aErrIO - A communication error occured.
     */
    aLIBEXPORT aErr aStream_CreateUSB(const uint32_t serialNum,
                                      aStreamRef* pStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a stream buffer.
    
    /**
     * Creates a stream buffer.
     *
     * StreamBuffers are typically used to agregate a bunch of output
     * into a single pile of bytes.  This pile can then be checked for
     * size or accessed as a single block of bytes using the
     * aStreamBuffer_Get call.  Finally, these bytes can then be read
     * back out of the buffer until it is empty when it will report an
     * error of aErrEOF.  While this stream is thread-safe for different
     * threads doing reads and writes, it is not the best candidate for
     * managing a pipe between threads.  Use the aStream_CreatePipe in
     * that scenario as it can be filled and emptied over and over which
     * is typically the use case for cross-thread pipes. 
     *
     * \param nIncSize - The Increment size to expand the buffer by when it 
     *                   becomes full.
     * \param pBufferStreamRef - The buffer stream resulting from the call.
     * 
     * \returns Function returns aErr values.
     * \retval aErrNone - The buffer was successfully created.
     * \retval aErrResource - The resources were not available to create the buffer.
     */
    aLIBEXPORT aErr aStreamBuffer_Create(const size_t nIncSize,
                                         aStreamRef* pBufferStreamRef);
    
    
    
    /////////////////////////////////////////////////////////////////////
    /// Get the contents of the buffer.
    
    /**
     * Get the contents of the buffer.
     *
     * StreamBuffers are typically used to agregate a bunch of output
     * into a single pile of bytes.  This pile can then be checked for
     * size or accessed as a single block of bytes using the
     * aStreamBuffer_Get call.  Finally, these bytes can then be read
     * back out of the buffer until it is empty when it will report an
     * error of aErrEOF.  While this stream is thread-safe for different
     * threads doing reads and writes, it is not the best candidate for
     * managing a pipe between threads.  Use the aStream_CreatePipe in
     * that scenario as it can be filled and emptied over and over which
     * is typically the use case for cross-thread pipes.
     *
     *
     * \param bufferStreamRef - The buffer stream resulting from the call.
     * \param aSize - The size of the buffered data in bytes.
     * \param ppData - The resulting buffer of the bytes.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - The buffer was successfully created.
     * \retval aErrParam - An invalid stream ref was given.
     */
    aLIBEXPORT aErr aStreamBuffer_Get(aStreamRef bufferStreamRef,
                                      size_t* aSize,
                                      uint8_t** ppData);
    
    /////////////////////////////////////////////////////////////////////
    /// Create a pipe buffered stream.
    
    /**
     * Get the contents of the buffer. Offers a pipe that is thread-safe 
     * for reading and writing between two different contexts.  Returns
     * aErrNotReady when data is not available on reads.  Expands a buffer 
     * internally to hold data when written to until it is read out (FIFO). 
     *
     * \param pBufferStreamRef - The buffered stream to create the pipe out of.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrParam - The bufferStream is invalid.
     */
    aLIBEXPORT aErr aStream_CreatePipe(aStreamRef* pBufferStreamRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Flush the cotents of the buffer.
    
    /**
     * Flushes the content of the buffer into the flushStream.
     *
     * \param bufferStreamRef - The buffered stream to flush.
     * \param flushStream - the stream to flush the buffer into.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - The flush succeeded.
     * \retval aErrParam - The bufferStream is invalid.
     * \retval aErrIO - IO error writing to flushStream.
     */
    aLIBEXPORT aErr aStreamBuffer_Flush(aStreamRef bufferStreamRef,
                                        aStreamRef flushStream);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a Logging stream.
    
    /**
     * Creates a stream which contains an upstream log stream and a downstream
     * log stream. The logging stream logs reads to the upstream log and writes
     * to the downstream log, while passing all data to and from the
     * pLogStreamRef.
     *
     * \param streamToLog - The reference to the stream to log.
     * \param upStreamLog - Log stream for reads.
     * \param downStreamLog - Log stream for writes.
     * \param pLogStreamRef - The logged stream reference.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful creation.
     * \retval aErrParam - The stream to log is invalid.
     * \return aErrIO - A communication error occured creating the logging stream.
     */
    aLIBEXPORT aErr aStream_CreateLogStream(const aStreamRef streamToLog,
                                            const aStreamRef upStreamLog,
                                            const aStreamRef downStreamLog,
                                            aStreamRef* pLogStreamRef);
    
    
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     * Stream Operations... READ and Write FLUSH....                           *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read a byte array record from a stream.
    
    /**
     * Read a byte array record from a stream.
     *
     * \param streamRef - The reference to the stream to read from.
     * \param pBuffer - byte array buffer to read into.
     * \param length - the length of the read buffer.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful read.
     * \retval aErrMode - The streamRef is not readable.
     * \retval aErrIO - An error occured reading the data.
     */
    aLIBEXPORT aErr aStream_Read(aStreamRef streamRef,
                                 uint8_t* pBuffer,
                                 const size_t length);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Write a byte array to a Stream.
    
    /**
     * Write a byte array to a Stream.
     *
     * \param streamRef - The reference to the stream to write to.
     * \param pBuffer - byte array to write out to the stream.
     * \param length - the byte array length
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful write.
     * \retval aErrMode - The streamRef is not writable.
     * \retval aErrIO - An error occured writing the data.
     */
    aLIBEXPORT aErr aStream_Write(aStreamRef streamRef,
                                  const uint8_t* pBuffer,
                                  const size_t length);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read a byte array record from a stream with a record terminator.
    
    /**
     * Read a byte array record from a stream with a record terminator.
     *
     * \param streamRef - The reference to the stream to read from.
     * \param pBuffer - Byte array buffer to read into.
     * \param lengthRead - The length of the read buffer.
     * \param maxLength - The Maximum record length.
     * \param recordTerminator - The byte array representing the record terminator.
     * \param terminatorLength - The length of the record terminator.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful read.
     * \return aErrMode - The streamRef is not readable.
     * \return aErrIO - An error occured reading the data.
     */
    aLIBEXPORT aErr aStream_ReadRecord(aStreamRef streamRef,
                                       uint8_t* pBuffer,
                                       size_t* lengthRead,
                                       const size_t maxLength,
                                       const uint8_t* recordTerminator,
                                       const size_t terminatorLength);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Write a byte array with a record terminator to a Stream.
    
    /**
     * Write a byte array with a record terminator to a Stream.
     *
     * \param streamRef - The reference to the stream to write to.
     * \param pBuffer - byte array to write out to the stream.
     * \param bufferLength - the byte array length
     * \param recordTerminator - the byte array representing the record terminator
     * \param terminatorLength - the length of the record terminator.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful write.
     * \retval aErrMode - The streamRef is not writable.
     * \retval aErrIO - An error occured writing the data.
     */
    aLIBEXPORT aErr aStream_WriteRecord(aStreamRef streamRef,
                                        const uint8_t* pBuffer,
                                        const size_t bufferLength,
                                        const uint8_t* recordTerminator,
                                        const size_t terminatorLength);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read a null terminated string from Stream.
    
    /**
     * Read a null terminated string from Stream.
     *
     * \param streamRef - The reference to the stream to read from.
     * \param pBuffer - Character array buffer to read into.
     * \param maxLength - The maximum length of the string.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful read.
     * \retval aErrMode - The streamRef is not readable.
     * \retval aErrIO - An error occured reading the data.
     */
    aLIBEXPORT aErr aStream_ReadCString(aStreamRef streamRef,
                                        char* pBuffer,
                                        const size_t maxLength);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Write a null terminated string.
    
    /**
     * Write a null terminated string.
     *
     * \param streamRef - The reference to the stream to write to.
     * \param pBuffer - character array to write.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful write.
     * \retval aErrMode - The streamRef is not writable.
     * \retval aErrIO - An error occured writing the data.
     */
    aLIBEXPORT aErr aStream_WriteCString(aStreamRef streamRef,
                                         const char* pBuffer);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read a null terminated string with a record terminator to pBuffer.
    
    /**
     * Read a null terminated string with a record terminator to pBuffer.
     *
     * \param streamRef - The reference to the stream to read to.
     * \param pBuffer - character array to read to.
     * \param maxLength - The maximum number of characters to read.
     * \param recordTerminator - The record terminator to read to.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful read.
     * \retval aErrMode - The streamRef is not readable.
     * \retval aErrIO - An error occured reading the data.
     */
    aLIBEXPORT aErr aStream_ReadCStringRecord(aStreamRef streamRef,
                                              char* pBuffer,
                                              const size_t maxLength,
                                              const char* recordTerminator);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Write a null terminated string with a record terminator to the stream.
    
    /**
     * Write a null terminated string with a record terminator to the stream.
     *
     * \param streamRef - The reference to the stream to be written to.
     * \param pBuffer - Null terminated string to write to the stream.
     * \param recordTerminator - The record terminator to write after the contents.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful write.
     * \retval aErrMode - The streamRef is not writable.
     * \retval aErrIO - An error occured writing the data.
     */
    aLIBEXPORT aErr aStream_WriteCStringRecord(aStreamRef streamRef,
                                               const char* pBuffer,
                                               const char* recordTerminator);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Flush contents of inStream into outStream.
    
    /**
     * Flush the entire current content of the instream into the outstream.
     *
     * \param inStreamRef - The reference to the stream to be flushed into the outstream.
     * \param outStreamRef - The reference to the stream instream is flushed into.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Successful Flush.
     * \retval aErrMode - The outstream is not writable or instream is not readable.
     * \retval aErrIO - An error occured flushing the data.
     */
    aLIBEXPORT aErr aStream_Flush(aStreamRef inStreamRef,
                                  aStreamRef outStreamRef);
    
    /////////////////////////////////////////////////////////////////////
    /// Destroy a Stream.
    
    /**
     * Safely destroy a stream and deallocate the associated resources.
     *
     * \param pStreamRef - A pointer to a pointer of a valid streamRef. The StreamRef
     *                     will be set to NULL on successful destruction of the stream.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - The stream was successfully destroyed.
     * \retval aErrParam - If the streamRef is invalid.
     */
    aLIBEXPORT aErr aStream_Destroy(aStreamRef* pStreamRef);
    
#ifdef __cplusplus
}
#endif

#endif /* _Stream_H_ */
