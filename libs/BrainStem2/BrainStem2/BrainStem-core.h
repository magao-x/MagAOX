/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aBrainStem-core.h                                         //
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

#ifndef __BrainStem_core_H__
#define __BrainStem_core_H__

#include <stdint.h>
#include <string>
#include <list>
#include <memory>

#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef in_addr_t
#define in_addr_t uint32_t
#endif

#else // Linux

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#endif

#include "BrainStem-C.h"

#ifdef __cplusplus
using std::list;
#endif // __cplusplus


namespace Acroname {
    namespace BrainStem {

        // MARK: -Link Class
        /////////////////////////////////////////////////////////////////////
        /// The Link class provides an interface to a BrainStem link.
        /// The link is used to create interfaces to modules on a BrainStem network.
        /// The link represents a connection to the BrainStem network from a host computer.
        /// The link is always associated with a transport (e.g.: USB, Ethernet, etc.) and a
        /// link module, but there are several ways to make this association.\n
        ///    -# The link can be fully specified with a transport and module serial number
        ///    -# The link can be created by searching a transport and connecting to the first
        ///    module found.
        ///
        /// Calling connect on a link will start a connection to the module based on
        /// The link specification. Calling disconnect will disconnect the link from
        /// the the current connection.
        class aLIBEXPORT Link
        {
        public:

            /// Discover is called with a specified transport to search for
            /// link modules on that transport. The callback is called with
            /// a fully filled in specifier for any link module found. The sDiscover returns
            /// aErrNone if the discovery process is successful, regardless of if any links
            /// are found. An error is only returned if the link discovery
            /// process fails. Discovery can take some time.
            /// The callback will occur in the same thread context as this routine call.
            /// \param type Transport to search for available BrainStem link modules
            /// on. See the \ref linkType "transport" enum for supported transports.
            /// \param cbLinkFound Process that is called when a module is discovered.
            /// \param vpCBRef This is passed to cbLinkFound when a module is discovered.
            /// \return ::aErrNotFound if no devices were found.
            /// \return ::aErrNone on success.
            static aErr sDiscover(const linkType type,
                                  aDiscoveryModuleFoundProc cbLinkFound,
                                  void* vpCBRef)
            {
                aErr err = aErrNone;
                uint8_t count = 0;

                count = aDiscovery_EnumerateModules(type, cbLinkFound, vpCBRef);
                if (count == 0) {
                    err = aErrNotFound;
                }
                return err;

            };

            /// sFindAll is a callback function which matches any found stem.
            /// SFindAll is used by sDiscover(const linkType, list<linkSpec>*) to
            /// fill the list provided with any found modules on the specified link
            /// type.
            /// \param spec The linkspec pointer for the device currently being
            ///             evaluated.
            /// \param bSuccess a returned value indicating whether the search has
            ///                 succeeded.
            /// \param vpCBRef Reference pointer to the std::list that was passed in.
            /// \return true To continue processing, or false to stop processing.
            static bContinueSearch sFindAll(const linkSpec* spec, bool* bSuccess, void* vpCBRef) {

                list<linkSpec>* pSpecs = (std::list<linkSpec>*)vpCBRef;
                pSpecs->push_back(*spec);
                *bSuccess = true;
                return true;
            }

            /// Discover is called with a specified transport to search for
            /// link modules on that transport. The devices list is filled with device
            /// scecifiers. sDiscover returns aErrNone if the discovery process is
            /// successful, regardless of whether any links
            /// are found. An error is only returned if the link discovery
            /// process fails. Discovery can take some time.
            /// \param type Transport to search for available BrainStem link modules
            /// on. See the \ref linkType "transport" enum for supported transports.
            /// \param devices an empty list of specifiers that will be filled in.
            /// \return ::aErrNotFound if no devices were found.
            /// \return ::aErrNone on success.
            static aErr sDiscover(const linkType type,
                                  list<linkSpec>* devices) 
            {
                return sDiscover(type, sFindAll, devices);
            };


            /// Link Constructor. Takes a fully specified linkSpec pointer and creates
            /// a link instance with this specifier information.
            /// \param linkSpecifier The connection details for a specific module.
            /// \param name A name for the link to be created.
            /// This name can be used to reference the link during later interactions.
            Link(const linkSpec linkSpecifier, const char* name = "Link");

            /// Link constructor without a specifier will most likely use the
            /// discoverAndConnect call to create a connection to a link module.
            /// \param name A name for the link to be created.
            Link(const char* name = "Link");

            /// Destructor.
            ~Link(void);

            /// A discovery-based connect.
            /// This member function will connect to the first available BrainStem
            /// found on the given transport. If the
            /// serial number is passed, it will only connect to the module with that serial number.
            /// Passing 0 as the serial number will create a link to the first link module found
            /// on the specified transport.
            /// If a link module is found on the specified transport, a connection will be made.
            /// \param type Transport on which to search for available BrainStem link
            /// modules. See the \ref linkType "transport" enum for supported transports.
            /// \param serialNumber Specify a serial number to connect to a specific
            /// link module. Use 0 to connect to the first link module found.
            /// \param model Acroname model number for the device.
            /// \return ::aErrBusy - if the module is already in use.
            /// \return ::aErrParam - if the transport type is undefined.
            /// \return ::aErrNotFound - if the module cannot be found or if no modules found.
            /// \return ::aErrNone - If the connect was successful.
            aErr discoverAndConnect(const linkType type,
                                    const uint32_t serialNumber = 0,
                                    const uint8_t model = 0);

            /// Connect to a link with a fully defined specifier.
            /// \return ::aErrBusy - if the module is running, starting or stopping. Try again in a bit.
            /// \return ::aErrDuplicate - If the module is aleady connected and running.
            /// \return ::aErrConnection - If there was an error with the connection. User needs to disconnect, then reconnect.
            /// \return ::aErrConfiguration - If the link has an invalid linkSpec.
            /// \return ::aErrNotFound - if the module cannot be found.
            /// \return ::aErrNone If the connect was successful.
            aErr connect(void);

            
            /// Check to see if a module is connected.
            /// isConnected looks for a connection to an active module.
            /// \return true: connected, false: not connected.
            bool isConnected(void);
            
            /// Check the status of the module connection.
            /// \return linkStatus (see aLink.h for status values)
            linkStatus getStatus(void);
            

            /// Disconnect from the BrainStem module.
            /// \return ::aErrResource - If the there is no valid connection.
            /// \return ::aErrConnection - If the disconnect failed, due to a
            /// communication issue.
            /// \return ::aErrNone If the disconnect was successful.
            aErr disconnect(void);
            
            /// Reset The underlying link stream.
            /// \return ::aErrResource - If the there is no valid connection.
            /// \return ::aErrConnection - If the reset failed, due to a
            /// communication issue.
            /// \return ::aErrNone If the reset was successful.
            aErr reset(void);

            /////////////////////////////////////////////////////////////////////
            //
            // access routines
            //
            /////////////////////////////////////////////////////////////////////
            
            /// Accessor for link Name.
            /// Returns a pointer to the string representing the link. This string
            /// is part of the link, and will be destroyed with it. If you need access
            /// to the link name beyond the life of the link, then copy the char* returned.

            /// \return Pointer to character array containing the name of the link.
            const char* getName(void);

            /// Acessor for current link specificaiton.
            /// \param spec - an allocated empty link spec reference.
            /// \return aErrNotFound - If no linkSpec set for current link.
            aErr getLinkSpecifier(linkSpec* spec);

            /// Accessor Set current link specification.
            /// \param linkSpecifier - The specifier that will replace the current spec.
            /// \return aErrBusy - If link is currently connected.
            aErr setLinkSpecifier(const linkSpec linkSpecifier);

            /// Gets the module address of the module the link is connected too.
            /// A zero is returned if no module can not be determined or
            /// if the link is not connected.
            aErr getModuleAddress(uint8_t * address);


            /////////////////////////////////////////////////////////////////////
            //
            // Send/Receive Packets to/from the Brainstem
            //
            /////////////////////////////////////////////////////////////////////
            
            /// Sends a BrainStem protocol UEI packet on the link.
            /// This is an advanced interface, please see the relevant section of the
            ///  reference manual for more information about UEIs.
            /// \param packet The command UEI packet to send.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - data too long or short.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrNone - success.
            aErr sendUEI(const uei packet);

            /// Sends a BrainStem protocol UEI packet on the link where the packet contains
            /// a subindex.
            /// This is an advanced interface, please see the relevant section of the
            ///  reference manual for more information about UEIs.
            /// \param packet The command UEI packet to send.
            /// \param subindex The subindex of the command option.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - data too long or short.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrNone - success.
            aErr sendUEI(const uei packet, const uint8_t subindex);

            /// Awaits receipt of the first available matching UEI packet.
            /// from the link. The first four arguments decribe the packet to wait for.
            /// When successful, the supplied uei ref is filled with the received UEI.
            /// This is an advanced interface, please see the relevant section of the
            ///  reference manual for more information about UEIs.
            /// \param module The module address.
            /// \param command The command.
            /// \param option The uei option.
            /// \param index The index of the uei entity.
            /// \param packet The uei packet reference to be filled on success.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrTimeout - no packet available.
            /// \return ::aErrNone - success.
            aErr receiveUEI(const uint8_t module,
                            const uint8_t command,
                            const uint8_t option,
                            const uint8_t index,
                            uei* packet);

            /// Drops all existing queued packets that match.
            /// from the link.
            /// The arguments decribe the packets to be matched
            /// This is an advanced interface, please see the relevant section of the
            ///  reference manual for more information about UEIs.
            /// \param module The module address.
            /// \param command The command.
            /// \param option The uei option.
            /// \param index The index of the uei entity.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrNone - success.
            aErr dropMatchingUEIPackets(const uint8_t module,
                                        const uint8_t command,
                                        const uint8_t option,
                                        const uint8_t index);

            /// Sends a raw BrainStem protocol packet on the link.
            /// where the length does not include the module or the command.
            /// address byte and can be 0 to aBRAINSTEM_MAXPACKETBYTES - 1.
            /// This is an advanced interface, please see the relevant section of the
            /// reference manual for more information about BrainStem Packet protocol.
            /// \param module The address of the destination module.
            /// \param command The length of the data being sent.
            /// \param length The length of the data being sent.
            /// \param data The data to send.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - data too long or short.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrNone - success.
            aErr sendPacket(const uint8_t module,
                            const uint8_t command,
                            const uint8_t length,
                            const uint8_t* data);



            /// Awaits receipt of the first available matching raw BrainStem protocol packet
            /// from the link where the length does not include the module or command bytes and can be zero.
            /// The provided module and match array are compared to packets available and the first match is
            /// returned.  The supplied data pointer must point to at least aBRAINSTEM_MAXPACKETBYTES - 1 bytes.
            /// When successful, the data is filled in with the packet data
            /// not including the module and command and the length pointer
            /// is updated with the length of the returned data.
            ///
            /// This is an advanced interface, please see the relevant section of the
            /// reference manual for more information about BrainStem Packet protocol.
            /// \param module The module address.
            /// \param match A byte array of the values to match for received packets.
            /// \param length The length of the match data on entry and lenth of the returned data filled on success.
            /// \param data The data filled on success.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrPacket - invalid module address.
            /// \return ::aErrTimeout - no packet available.
            /// \return ::aErrNone - success.
            aErr receivePacket(const uint8_t module,
                               const uint8_t* match,
                               uint8_t* length,
                               uint8_t* data);

            
            /////////////////////////////////////////////////////////////////////
            //
            // Routines for handling slots at the module
            //
            /////////////////////////////////////////////////////////////////////
            
            /// Loads data into a BrainStem Slot. See the relevant section of the BrainStem
            /// reference for information about BrainStem Slots and Stores.
            /// \param module - Module address.
            /// \param store - BrainStem store to access, possibilities include Internal, RAM, and SD.
            /// \param slot - The Slot within the Brainstem store to place the data.
            /// \param pData - Pointer to a buffer containing the data to load.
            /// \param length - The length in bytes of the data buffer to write.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - invalid module address.
            /// \return ::aErrCancel - The write process is closing and this call
            /// was unable to successfully complete.
            /// \return ::aErrNone - success.
            aErr loadStoreSlot(const uint8_t module,
                               const uint8_t store,
                               const uint8_t slot,
                               const uint8_t* pData,
                               const size_t length);

            /// Unloads data from a BrainStem Slot. If there are no read
            // errors but the dataLength supplied is less than the actual slot size,
            // an error of aErrOverrun is returned. See the relevant section of the BrainStem
            /// reference for information about BrainStem Slots and Stores.
            /// \param module - Module address.
            /// \param store - BrainStem store to access, possibilities include Internal, RAM, and SD.
            /// \param slot - The Slot within the Brainstem store to place the data.
            /// \param pData - Pointer to a buffer with dataLength space in bytes that will be filled by the call.
            /// \param dataLength - Expected length of the data, and at most the size of the pData buffer.
            /// \param pNRead - The number of bytes actually read.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - invalid module address.
            /// \return ::aErrCancel - The write process is closing and this call
            /// was unable to successfully complete.
            /// \return ::aErrOverrun - The read would overrun the buffer, i.e there is more data in the slot than the buffer can handle.
            /// \return ::aErrNone - success.
            aErr unloadStoreSlot(const uint8_t module,
                                 const uint8_t store,
                                 const uint8_t slot,
                                 uint8_t* pData,
                                 const size_t dataLength,
                                 size_t* pNRead);


            /// Returns the current size of the data loaded in the slot specified.
            /// \param module - Module address.
            /// \param store - BrainStem store to access, possibilities include Internal, RAM, and SD.
            /// \param slot - The Slot within the Brainstem store to place the data.
            /// \param size - size in bytes of the data stored in the slot.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - invalid module address.
            /// \return ::aErrCancel - The write process is closing and this request
            /// was unable to successfully complete.
            /// \return ::aErrNone - success.
            aErr storeSlotSize(const uint8_t module,
                               const uint8_t store,
                               const uint8_t slot,
                               size_t* size);

            /// Returns the maximum data capacity of the slot specified.
            /// \param module - Module address.
            /// \param store - BrainStem store to access, possibilities include Internal, RAM, and SD.
            /// \param slot - The Slot within the Brainstem store to place the data.
            /// \param capacity - size in bytes of the data stored in the slot.
            /// \return ::aErrConnection - link not connected.
            /// \return ::aErrParam - invalid module address.
            /// \return ::aErrCancel - The write process is closing and this request
            /// was unable to successfully complete.
            /// \return ::aErrNone - success.
            aErr storeSlotCapacity(const uint8_t module,
                                   const uint8_t store,
                                   const uint8_t slot,
                                   size_t* capacity);
            
            /////////////////////////////////////////////////////////////////////
            //
            // Routines to handle packet logging
            //
            /////////////////////////////////////////////////////////////////////

            /// Enable Packet logging.
            ///
            /// Enable packet logging for this link. Enables the packet logging buffer, and writes
            /// packet traffic out to the file specified by logname.
            /// \param logname the path and filename indicating where to write the packet log.
            /// \return aErr returns appropriate errors if it fails to enable the packet log.
            aErr enablePacketLog(const char* logname);
            
            /// Disable Packet logging.
            ///
            /// disable packet logging for this link. Disables the packet log.
            /// \return aErr returns appropriate errors if it fails to disable the debug log.
            aErr disablePacketLog(void);
            
            
        private:

            class impl; impl* zit;

        };


        // MARK: -Module Class
        /////////////////////////////////////////////////////////////////////
        /// The Module class provides a generic interface to a BrainStem hardware module.
        /// The Module class is the parent class for all BrainStem modules. Each module
        /// inherits from Module and implements its hardware specific features.
        class aLIBEXPORT Module
        {
        public:
			
            /// Constructor. Implicitly creates a link object with no specifier. Most often objects
            /// created with this constructor will use linkDiscoverAndConnect to find and connect to a module.
            /// \param address The BrainStem network address of the module. The default address
            /// (or base address for modules that support address offsets) is defined in each
            /// module's "Defs.h" header.
            /// \param model Acroname model number.
            Module(const uint8_t address, const uint8_t model = 0);
            
            Module(const uint8_t address, bool bAutoNetworking, const uint8_t model = 0);
            
            /// Destructor.
            virtual ~Module(void);
            
            
            /// Connect using the current link specifier.
            /// \param type - Transport on which to search for available BrainStem link
            /// modules. See the \ref linkType "transport" enum for supported transports.
            /// \param serialNum - Specify a serial number to connect to a specific
            /// link module. Use 0 to connect to the first link module found.
            /// \return ::aErrBusy - if the module is already in use.
            /// \return ::aErrParam - if the type is incorrect or serialNum is not specified
            /// \return ::aErrNotFound - if the module cannot be found.
            /// \return ::aErrNone If the connect was successful.
            aErr connect(const linkType type,
                         const uint32_t serialNum);
            
            /// Connect to a link with a fully defined specifier.
            /// \param linkSpecifier - Connect to module with specifier.
            /// \return ::aErrInitialization - If there is currently no link object.
            /// \return ::aErrBusy - If the link is currently connected.
            /// \return ::aErrParam - if the specifier is incorrect.
            /// \return ::aErrNotFound - if the module cannot be found.
            /// \return ::aErrNone If the connect was successful.
            aErr connectFromSpec(const linkSpec linkSpecifier);
            
            /// A discovery-based connect.
            /// This member function will connect to the first available BrainStem
            /// found on the given transport. If the
            /// serial number is passed, it will only connect to the module with that serial number.
            /// Passing 0 as the serial number will create a link to the first link module found
            /// on the specified transport.
            /// If a link module is found on the specified transport, a connection will
            /// \param type - Transport on which to search for available BrainStem link
            /// modules. See the \ref linkType "transport" enum for supported transports.
            /// \param serialNum - Specify a serial number to connect to a specific
            /// link module. Use 0 to connect to the first link module found.
            /// \return ::aErrBusy - if the module is already in use.
            /// \return ::aErrParam - if the transport type is undefined.
            /// \return ::aErrNotFound - if the module cannot be found.
            /// \return ::aErrNone If the connect was successful.
            aErr discoverAndConnect(linkType type,
                                      const uint32_t serialNum = 0);
            
            /// Connect using link from another Module.
            /// This member function will connect to the same BrainStem used by given Module.
            /// If a link module is found on the specified transport, a connection will
            /// \param pModule - Pointer to a valid Module class object.
            /// \return ::aErrParam - if the module is undefined.
            /// \return ::aErrNone  - if the connect was successful.
            aErr connectThroughLinkModule(Module* pModule);
            
            
            /// Is the link connected to the BrainStem Module.
            bool isConnected(void);
            
            /// Check the status of the BrainStem module connection.
            /// \return linkStatus (see aLink.h for status values)
            linkStatus getStatus(void);
            

            /// Disconnect from the BrainStem module.
            /// \return ::aErrResource - If the there is no valid connection.
            /// \return ::aErrConnection - If the disconnect failed, due to a
            /// communication issue.
            /// \return ::aErrNone If the disconnect was successful.
            aErr disconnect(void);
            
            /// Reconnect using the current link specifier.
            /// \return ::aErrBusy - if the module is already in use.
            /// \return ::aErrParam - if the specifier is incorrect.
            /// \return ::aErrNotFound - if the module cannot be found.
            /// \return ::aErrNone If the connect was successful.
            aErr reconnect();

            /// Get the current link object.
            /// \return The link associated with the module.
            Link* getLink(void) const;
            

            /// Accessor to get the address of the BrainStem module associated with the instance
            /// on the host machine.  (Not to be confused with the System entity which effects the
            /// device hardware.)
            /// \return The module address.
            uint8_t getModuleAddress(void) const;
            
            /// Accessor to set the address of the BrainStem module associated with the instance
            /// on the host machine.  (Not to be confused with the System entity which effects the
            /// device hardware.)
            /// \param address The module address.
            void setModuleAddress(const uint8_t address);
            
            /// Get linkSpecifier
            /// \param spec - allocated linkspec struct will be filled with spec.
            /// \return aErrNone - If the module does not have a spec.
            aErr getLinkSpecifier(linkSpec* spec);
            
            /// Queries the module to determine if it implements a UEI. Each
            /// UEI has a command, option or variant, index and flag. The hasUEI method
            /// queries for a fully specified UEI.
            /// Returns aErrNone if the variation is supported and an appropriate error
            /// if not. This call is blocking for up to the nMSTimeout period.
            /// \param command One of the UEI commands (cmdXXX).
            /// \param option The option or variant of the command.
            /// \param index The entity index.
            /// \param flags The flags (ueiOPTION_SET or ueiOPTION_GET).
            /// \return ::aErrNone - The module supports this command and access flags.
            /// \return ::aErrMode - The module supports the command but not the access
            ///                    flag.
            /// \return ::aErrNotFound - The module does not support the command, option,
            ///                        or index.
            /// \return ::aErrTimeout - The request timed out without a response.
            /// \return ::aErrConnection - There is no active link
            aErr hasUEI(const uint8_t command,
                        const uint8_t option,
                        const uint8_t index,
                        const uint8_t flags);

            /// Queries the module to determine how many entities of the specified
            /// class are implemented by the module. Zero is a valid return value.
            /// For example, calling classQuantity with the command parameter of
            /// cmdANALOG would return the number of analog entities implemented by the module.
            /// \param command One of UEI commands (cmdXXX).
            /// \param count When the request is successful count
            ///              is updated with the number of entities found.
            /// \return ::aErrNone - Success.
            /// \return ::aErrTimeout - The request timed out without a response.
            /// \return ::aErrConnection - There is no active link.
            aErr classQuantity(const uint8_t command,
                               uint8_t* count);

            /// Queries the module to determine how many subclass entities of the specified
            /// class are implemented by the module for a given entity index. This is used
            /// for entities which may be 2-dimensional. E.g. cmdMUX subclasses are the number
            /// of channels supported by a particular mux type (index); as a specific example,
            /// a module may support 4 UART channels, so subClassQuantity(cmdMUX, aMUX_UART...)
            /// could return 4.
            /// Zero is a valid return value.
            /// \param command One of the UEI commands (cmdXXX).
            /// \param index The entity index.
            /// \param count The number of subclasses found.
            /// \return ::aErrNone - Success.
            /// \return ::aErrTimeout - The request timed out waiting for response.
            /// \return ::aErrConnection - There is no active link.
            aErr subClassQuantity(const uint8_t command,
                                  const uint8_t index,
                                  uint8_t* count);

            /// Queries the module the group assigned to an entity and index. Entities groups
            /// are used to specify when certain hardware features are fundamentally related. E.g.
            /// certain hardware modules may have some digital pins associated with an adjustable
            /// voltage rail; these digitals would be in the same group as the rail.
            /// Zero is the default group.
            /// \param command One of the UEI commands (cmdXXX).
            /// \param index The entity index.
            /// \param group Upon success, group is filled with the entities group value.
            /// \return ::aErrNone - Success.
            /// \return ::aErrTimeout - The request timed out without response.
            /// \return ::aErrConnection - There is no active link.
            aErr entityGroup(const uint8_t command,
                             const uint8_t index,
                             uint8_t* group);

            
            /// Sends a debug packet to the module containing the provided data.
            /// Modules receiving debug packets simply echo the packet back to the sender.
            /// If the round-trip is successful, the reply data will match the data sent.
            /// This method returns aErrNone when successful, if not successful,
            /// an appropriate error is returned.
            /// \param pData A pointer to an array of data to be sent in the debug packet.
            /// \param length The length of the data array.
            /// \return ::aErrNone - Success.
            /// \return ::aErrTimeout - Timeout occured without response.
            /// \return ::aErrConnection - No active link exists.
            aErr debug(const uint8_t* pData,
                       const uint8_t length);
            
            
            /// Sets the networking mode of the module object.
            /// By default the module object is configure to automatically adjust
            /// its address based on the devices current module address. So that,
            /// if the device has a software or hardware offset it will still be
            /// able to communication with the device. If advanced networking is required
            /// the auto networking mode can be turned off.
            /// \param mode True/1 for Auto Networking, False/0 for manual networking
            void setNetworkingMode(const bool mode);
            
            
            aErr enablePacketLog(const char* packetLogName = "StemDebug");
            
            aErr disablePacketLog(void);
            

        private:
            Module();
            Link* m_pLink;
            uint8_t m_address;
            bool m_bAutoNetworking;
            uint8_t m_model;
            
            void _autoNetwork(void);
        };

        // MARK: -Entity Class
        /////////////////////////////////////////////////////////////////////
        /// The EntityClass is the base class for interacting with BrainStem UEI entities.
        /// All BrainStem UEI classes inherit from EntityClass. Advanced users may use
        /// EntityClass to extend BrainStem functionality specific to their needs.
        /// \defgroup EntityReturnValues Common EntityClass Return Values
        /// Common return values for aEntityClass UEI get and set operations
        /// \{
        ///     ::aErrNone - Action completed successfully.
        ///     ::aErrTimeout - Request timed out without response.
        ///     ::aErrConnection - No active link.
        /// \}
        class aLIBEXPORT EntityClass
        {
        public:

            /// Constructor.
            EntityClass(void);

            /// Destructor.
            virtual ~EntityClass(void);

            /// init.
            ///
            /// Initialize the entity class.
            /// \param pModule The BrainStem module object.
            /// \param command The command of the UEI.
            /// \param index The index of the UEI entity.
            void init(Module* pModule,
                      const uint8_t command,
                      const uint8_t index);

            /// A callUEI is a setUEI that has no data length.
            /// \param option An option for the UEI.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr callUEI(const uint8_t option);


            /// Set a byte value.
            /// \param option The option for the UEI.
            /// \param byteValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUEI8(const uint8_t option,
                         const uint8_t byteValue);

            /// Set a byte value with a subindex.
            /// \param option The option for the UEI.
            /// \param param of the option.
            /// \param byteValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUEI8(const uint8_t option,
                         const uint8_t param,
                         const uint8_t byteValue);

            /// Get a byte value.
            /// \param option The option for the UEI.
            /// \param byteValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI8(const uint8_t option,
                         uint8_t* byteValue);

            /// Get a byte value with a parameter.
            /// \param option The option for the UEI.
            /// \param param The parameter.
            /// \param byteValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI8(const uint8_t option,
                         const uint8_t param,
                         uint8_t* byteValue);

            /// Set a 2-byte value.
            /// \param option The option for the UEI.
            /// \param shortValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUEI16(const uint8_t option,
                          const uint16_t shortValue);

            /// Get a 2-byte value.
            /// \param option The option for the UEI.
            /// \param shortValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI16(const uint8_t option,
                          uint16_t* shortValue);

            /// Get a 2-byte value with a parameter.
            /// \param option The option for the UEI.
            /// \param param The parameter.
            /// \param shortValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI16(const uint8_t option,
                          const uint8_t param,
                          uint16_t* shortValue);

            /// Set a 4-byte value.
            /// \param option The option for the UEI.
            /// \param intValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUEI32(const uint8_t option,
                          const uint32_t intValue);

            /// Set a 4-byte value, with a subindex parameter.
            /// \param option The option for the UEI.
            /// \param subIndex The subindex to set.
            /// \param intValue The value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUEI32(const uint8_t option,
                          const uint8_t subIndex,
                          const uint32_t intValue);

            /// Get a 4-byte value.
            /// \param option The option for the UEI.
            /// \param intValue The 4 byte value
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI32(const uint8_t option,
                          uint32_t* intValue);

            /// Get a 4-byte value with parameter.
            /// \param option The option for the UEI.
            /// \param param The parameter.
            /// \param intValue The 4 byte value
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUEI32(const uint8_t option,
                          const uint8_t param,
                          uint32_t* intValue);


            /// Get the UEI entity index.
            /// \return The 1 byte index of the UEI entity.
            uint8_t getIndex(void) const;
            
            /// Drain all packets matching this UEI from the packet fifo.
            ///
            /// This functionality is useful in rare cases where packet
            /// synchronization is lost and a valid return packet is not
            /// accessible.
            aErr drainUEI(const uint8_t option);

        protected:

            aErr sendPacket(const uint8_t command,
                            const uint8_t length,
                            const uint8_t* data);

            aErr receivePacket(const uint8_t* match,
                               uint8_t* length,
                               uint8_t* data);

            aErr getUEI(const uint8_t option,
                        uei* d);

            aErr setUEI(const uint8_t option,
                        uei* d);

            aErr setUEI(const uint8_t option,
                        const uint8_t subIndex,
                        uei* d);

            aErr awaitUEI32Val(const uint32_t option,
                               uint32_t* intValue,
                               const uint32_t msTimeout);

            class impl; impl* zit;

        };

    } // namespace BrainStem
} // namespace Acroname

#endif // __BrainStem_core_H__
