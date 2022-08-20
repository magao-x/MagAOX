/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: aLink.h                                                   */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Definition for BrainStem packet FIFO queue.        */
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

#ifndef _aLink_H_
#define _aLink_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDeft_H_

#include "aError.h"
#include "aPacket.h"
#include "aStream.h"

/////////////////////////////////////////////////////////////////////
/// BrainStem Link Interface

/** \defgroup aLink Link Interface
 * \ref aLink "aLink.h" provides the interface for creating and maintaining
 * the link to a BrainStem module, and the BrainStem network. It includes
 * facilities for starting and stopping links, as well as sending and receiving
 * BrainStem protocol packets.
 */


/////////////////////////////////////////////////////////////////////
/// Typedef #aLinkRef Opaque reference to a BrainStem link.
//typedef void* aLinkRef;

/// Typedef for aLinkRef for an opaque reference to BrainStem Link.
typedef uint32_t aLinkRef;

/////////////////////////////////////////////////////////////////////
/// Enum #linkStatus.

/**
 * Represents the current state of the BrainStem link.
 */
typedef enum {
    STOPPED, /**< Link currently stopped. */
    INITIALIZING, /**< Starting communication with module. */
    RUNNING, /**< Link running. */
    STOPPING, /**< Link is in the process of stopping. */
    SYNCING, /**< Packet framing lost re-syncing. */
    INVALID_LINK_STREAM, /**< Link stream provided is not valid. */
    IO_ERROR, /**< Communication error occured on link, could not resync. */
    RESETTING, /**< Resetting the link connection */
    UNKNOWN_ERROR /**< Something really bad happened, but we couldn't determin what. */
} linkStatus;


#ifdef __cplusplus
extern "C" {
#endif

    /////////////////////////////////////////////////////////////////////
    /// Create a BrainStem link reference.
    
    /**
     * Creates a reference to a BrainStem link. The linkStream is now
     * maintained by the BrainStem link.  If the link already exists, the
     * use count for that link will be incremented and the linkRef for
     * that entry will be returned.
     *
     * Links created with this procedure must use aLink_Destroy to properly
     * dispose of the link reference and associated connections.
     *
     * \param serialNumber  the TCPIP address
     *
     * \returns aLinkRef identifier if successful or 0 otherwise.
     */
    aLIBEXPORT aLinkRef aLink_CreateUSB(const uint32_t serialNumber);

    /**
     * Creates a reference to a BrainStem link. The linkStream is now
     * maintained by the BrainStem link.  If the link already exists, the 
     * use count for that link will be incremented and the linkRef for 
     * that entry will be returned.
     *
     * Links created with this procedure must use aLink_Destroy to properly
     * dispose of the link reference and associated connections.
     *
     * \param address  the TCPIP address
     * \param port     the TCPIP port
     *
     * \returns aLinkRef identifier if successful or 0 otherwise.
     */
    aLIBEXPORT aLinkRef aLink_CreateTCPIP(const uint32_t address,
                                          const uint16_t port);

    /**
     * Creates a reference to a BrainStem link. -- *** DEPRECATED *** --
     * The linkStream is now maintained by the BrainStem link.
     * The aLink_Create(aStreamRef linkStream) is maintained for backward
     * compatability, but linkStream CANNOT be NULL.  Once the linkStream
     * has been used in the create, caller must not use or delete the stream.
     * Each call to this routine will create an new link entry,
     * independant of any other link to the same device.
     *
     * Links created with this procedure must use aLink_Destroy to properly
     * dispose of the link reference and associated connections.
     *
     * \param linkStream  a StreamRef the transport stream (NON-NULL).
     *
     * \returns aLinkRef identifier if successful or 0 otherwise.
     */
    aLIBEXPORT aLinkRef aLink_Create(aStreamRef linkStream);

    
    /////////////////////////////////////////////////////////////////////
    /// Create a BrainStem link reference.
    
    /**
     * Destroys a Link reference. deallocating associated resources cleanly.
     *
     * Links created with aLink_Create must use aLink_Destroy to clean up
     * resources used by the link Ref.
     *
     * \param linkRef  a Pointer to a valid LinkRef. The linkRef will be set to
     *                  NULL on succesful completion of the Destroy call.
     * \returns aStreamRef  Return value will always be NULL. The return
     *                       value has been left for backwards compatability.
     */
    aLIBEXPORT aErr aLink_Destroy(aLinkRef* linkRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Reset a connection to a BrainStem module.
    
    /**
     * Stop the active connection to the BrainStem if the Link contains a valid
     * stream Reference, and clear out the communication buffers, and restart
     * the link.
     *
     * \param linkRef  A valid LinkRef.
     * \returns Function returns aErr values.
     * \retval aErrNone  the call completed successfully, a subsequent
     *                    call to aLink_GetStatus should return the current
     *                    state of the link.
     * \retval aErrParam  No valid LinkRef provided.
     */
    aLIBEXPORT aErr aLink_Reset(const aLinkRef linkRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Return the current status of the BrainStem link.
    
    /**
     * Return the current status of the BrainStem link.
     *
     * \param linkRef  A valid LinkRef.
     * \return linkStatus  See the possible linkStatus values.
     */
    aLIBEXPORT linkStatus aLink_GetStatus(const aLinkRef linkRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Return the first packet in the Link incomming FIFO.
    
    /**
     * Return the first packet in the Link incomming FIFO. This call is non blocking,
     * and will return immediately.
     *
     * \param linkRef  A valid LinkRef.
     * \return aPacket  Returns a BrainStem packet on success or NULL.
     */
    aLIBEXPORT aPacket* aLink_GetPacket(const aLinkRef linkRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Return the first packet in the Link incomming FIFO.
    
    /**
     * Return the first packet in the Link incomming FIFO. This call blocks waiting
     * for msTimeout milliseconds.
     *
     * \param linkRef  A valid LinkRef.
     * \param msTimeout  The maximum amount of time in milliseconds to wait for a packet.
     * \return aPacket  Returns a BrainStem packet on success or NULL.
     */
    aLIBEXPORT aPacket* aLink_AwaitPacket(const aLinkRef linkRef,
                                          const unsigned long msTimeout);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Return the first packet matched by proc in the Link incomming FIFO.
    
    /**
     * Return the first packet matched by proc in the Link incomming FIFO. This call
     * is non blocking and returns immediatly.
     *
     * \param linkRef  A valid LinkRef.
     * \param proc  The callback used for determining a matching packet.
     * \param vpRef  A resource passed to the callback proc.
     * \return aPacket  Returns the first packet that is matched by proc or NULL.
     */
    aLIBEXPORT aPacket* aLink_GetFirst(const aLinkRef linkRef,
                                       aPacketMatchPacketProc proc,
                                       const void* vpRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Return the first packet matched by proc in the Link incomming FIFO.
    
    /**
     * Return the first packet matched by proc in the Link incomming FIFO. This call
     * blocks for up to msTimeout milliseconds waiting for a matching packet.
     *
     * \param linkRef  A valid LinkRef.
     * \param proc  The callback used for determining a matching packet.
     * \param vpRef  A resource passed to the callback proc.
     * \param msTimeout  The maximum amount of time in milliseconds to wait for a
     *                    matching packet.
     * \return aPacket  Returns the first packet that is matched by proc or NULL.
     */
    aLIBEXPORT aPacket* aLink_AwaitFirst(const aLinkRef linkRef,
                                         aPacketMatchPacketProc proc,
                                         const void* vpRef,
                                         const unsigned long msTimeout);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Drain all matching packets from the incomming FIFO.
    
    /**
     * Drain all matching packets from the incomming FIFO. This call does not block.
     *
     * \param linkRef  A valid LinkRef.
     * \param proc  The callback used for determining a matching packet.
     * \param vpRef  A resource passed to the callback proc.
     * \return aPacket  Returns the first packet that is matched by proc or NULL.
     */
    aLIBEXPORT size_t aLink_DrainPackets(const aLinkRef linkRef,
                                         aPacketMatchPacketProc proc,
                                         const void* vpRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Put a packet into the outgoing link FIFO.
    
    /**
     * Put a packet into the outgoing link FIFO.
     *
     * \param linkRef  A valid LinkRef.
     * \param packet  A BrainStem packet.
     * \returns Function returns aErr values.
     * \retval aErrNone  Call successfully added the packet.
     * \retval aErrParam Invalid LinkRef or packet.
     * \retval aErrResource Unable to create memory for packet in FIFO.
     */
    aLIBEXPORT aErr aLink_PutPacket(const aLinkRef linkRef, const aPacket* packet);
    
    
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    /// Enum type of debug packet entry.
    typedef enum {
        PACKET_LOG_OUTBOUND     = 0x01, // packet sent to device
        PACKET_LOG_INBOUND      = 0x02, // packet from device
        PACKET_LOG_NOTE         = 0x03,
        PACKET_LOG_UNKNOWN_TYPE = 0x80, // unknown type of data
    } packetLogType;
    

    /////////////////////////////////////////////////////////////////////
    /// Packet debug routines

    /////////////////////////////////////////////////////////////////////
    /// Enable logging of packets to/from the device.
    
    /**
     * Enable logging of packets to/from the device.
     *
     * \param linkRef  A valid LinkRef.
     * \param log_size  Size of buffer to use for logging packets.
     * \returns Function returns aErr values.
     * \retval aErrNone  Call successfully set log size.
     * \retval aErrParam Invalid LinkRef.
     * \retval aErrResource Unable to create memory for logging.
     */
    aLIBEXPORT aErr aLink_PacketDebug_SetLogSize(const aLinkRef linkRef, int32_t log_size);

    /////////////////////////////////////////////////////////////////////
    /// Enable logging of packets to/from the device.
    
    /**
     * Enable logging of packets to/from the device.
     *
     * \param linkRef  A valid LinkRef.
     * \returns Function returns aErr values.
     * \retval aErrNone  Call successfully enabled logging.
     * \retval aErrParam Invalid LinkRef.
     * \retval aErrResource Unable to create memory for logging.
     */
    aLIBEXPORT aErr aLink_PacketDebug_Enable(const aLinkRef linkRef);

    
    /////////////////////////////////////////////////////////////////////
    /// Disable logging of packets to/from the device.
    
    /**
     * Disable logging of packets to/from the device.
     *
     * \param linkRef  A valid LinkRef.
     * \returns Function returns aErr values.
     * \retval aErrNone  Call successfully disabled logging.
     * \retval aErrParam Invalid LinkRef.
     * \retval aErrNotReady Logging was never enabled.
     */
    aLIBEXPORT aErr aLink_PacketDebug_Disable(const aLinkRef linkRef);
    
    
    /////////////////////////////////////////////////////////////////////
    /// Read next entry from the log.
    
    /**
     * Read next entry from the log.
     *
     * \param linkRef  A valid LinkRef.
     * \param type  The type of data stored in the buffer upon return.
     * \param data_size  Data buffer size.
     * \param data  Buffer for data to be stored in upon return.
     * \returns Function returns aErr values.
     * \retval aErrNone  Call successfully read next entry.
     * \retval aErrParam Invalid LinkRef.
     * \retval aErrNotReady Logging was never enabled.
     * \retval aErrOverrun Data in entry is larger than buffer supplied
     */
    aLIBEXPORT aErr aLink_PacketDebug_Read(const aLinkRef linkRef, packetLogType *type, uint32_t *data_size, uint8_t *data);

    aLIBEXPORT aErr aLink_PacketDebug_Write(const aLinkRef linkRef, packetLogType type, const uint32_t data_size, const uint8_t *data);
    
#ifdef __cplusplus
}
#endif

#endif /* _aLink_H_ */

