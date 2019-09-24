/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aPacket.h                                                 */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Definition for BrainStem packet.                   */
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

#ifndef _aPacket_H_
#define _aPacket_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDefs_H_

#include "aError.h"

#define MAX_PACKET_BYTES  	28

/////////////////////////////////////////////////////////////////////
/// BrainStem Packet

/** \defgroup aPacket Packet Interface
 * \ref aPacket "aPacket.h" Provides and interface for creating and destroying
 * BrainStem Protocol packets.
 */


/////////////////////////////////////////////////////////////////////
/// Const value used to check packet validity.
extern const uint16_t VALIDPACKET;

/////////////////////////////////////////////////////////////////////
/// Struct for BrainStem packets.

/**
 * the check member is for checking the validity of the packet structure in memory.
 * Current size is used during link stream processing. Address, dataSize and data
 * fulfill the requirements of the BrainStem protocol.
 */
typedef struct aPacket {
    const uint16_t              check;
    uint8_t		        curSize;  /* current number of data input */
    uint8_t		        address;  /* module address */
    uint8_t		        dataSize; /* length of following data */
    uint8_t		        data[MAX_PACKET_BYTES];
} aPacket;

/////////////////////////////////////////////////////////////////////
/// Typedef #aPacketMatchPacketProc.

/**
 * This callback is used to match against packets in the FIFO.
 * \param p - the que packet to match against.
 * \return - uint8_t non-zero on a match and zero on a mismatch.
 */
typedef uint8_t (*aPacketMatchPacketProc) (const aPacket* p, const void* vpRef);


#ifdef __cplusplus
extern "C" {
#endif
    
    
    /////////////////////////////////////////////////////////////////////
    /// Check packet pointer for validity.
    
    /**
     * Checks to make sure a packet was allocated using aPacket_Create.
     *
     * \param packet - valid packet pointer.
     * \returns bool - True for valid false otherwise.
     */
    aLIBEXPORT bool aVALIDPACKET(const aPacket* packet);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Create a BrainStem packet.
    
    /**
     * Create a BrainStem packet.
     *
     * \returns aPacket - Pointer or NULL on error.
     */
    aLIBEXPORT aPacket* aPacket_Create(void);
    
    /////////////////////////////////////////////////////////////////////
    /// Create a BrainStem packet, containing the given data.
    
    /**
     * Create a BrainStem packet with data.
     *
     * \param address - Module address of the BrainStem module.
     * \param dataLength - The length of the data array.
     * \param data - Pointer to the beginning of the packet data.
     * \returns aPacket - Pointer or NULL on error.
     */
    aLIBEXPORT aPacket* aPacket_CreateWithData(const uint8_t address,
                                               const uint8_t dataLength,
                                               const uint8_t* data);
    
    /////////////////////////////////////////////////////////////////////
    /// Reset an existing packet.
    
    /**
     * Zero out any data the packet contains.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - If the reset was successful.
     * \retval aErrParam - If the packet is not valid.
     */
    aLIBEXPORT aErr aPacket_Reset(aPacket* packet);
    
    /////////////////////////////////////////////////////////////////////
    /// Accumulate a Byte into a packet.
    
    /**
     * A packet can be constructed byte by byte. the first byte added will
     * be the BrainStem module address, the second byte the data length, and
     * subsequent bytes will be data payload. This call will fail if more than
     * datalength bytes are added, or if address is an invalid module address (i.e. an odd number).
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - Adding the byte was successful.
     * \retval aErrParam - The packet was invalid.
     * \retval aErrPacket - The byte added violates the BrainStem protocol.
     */
    aLIBEXPORT aErr aPacket_AddByte(aPacket* packet, const uint8_t byte);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Determine whether a packet is complete.
    
    /**
     * A packet can be constructed byte by byte. This call determines whether such a
     * packet has been completed. It checks that dataSize is equal to the currentSize
     * minus the Address and dataSize bytes.
     *
     * \returns bool - True if complete false if not complete.
     */
    aLIBEXPORT bool aPacket_IsComplete(const aPacket* packet);
    
    /////////////////////////////////////////////////////////////////////
    /// Destroy a BrainStem packet.
    
    /**
     * Safely destroy a brainstem packet and deallocate the associated resources.
     *
     * \param packet - A pointer to a pointer of a valid packet. The packet pointer
     *                 will be set to NULL on successful destruction of the packet.
     *
     * \returns Function returns aErr values.
     * \retval aErrNone - The packet was successfully destoryed.
     * \retval aErrParam - The packetRef is invalid.
     */
    aLIBEXPORT aErr aPacket_Destroy(aPacket** packet);
    
#ifdef __cplusplus
}
#endif

#endif /* _aPacket_H_ */
