/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: Discover.h                                                */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Discover header file                    		   */
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

#ifndef _aDiscover_H_
#define _aDiscover_H_

#ifndef _aDefs_H_
#include "aDefs.h"
#endif // _aDefs_H_

#include "aError.h"

/////////////////////////////////////////////////////////////////////
/// Link Discovery Interface.

/** \defgroup aDiscovery Link Discovery Interface
 * \ref aDiscovery "aDiscovery.h" provides an interface for locating BrainStem modules
 * accross multiple transports. It provides a way to find all modules
 * for a give transport as well as specific modules by serial number,
 * or first found.
 */


/////////////////////////////////////////////////////////////////////
/// Enum #linkType.

/**
 * The linkType enum specifies the connection transport type.
 */
typedef enum {
    INVALID, ///< - Undefined link type.
    USB, ///< - USB link type.
    TCPIP ///< - TCP/IP link type.
} linkType;


/////////////////////////////////////////////////////////////////////
/// Struct #linkSpec.

/**
 * The linkSpec contains the necessary information for connecting to a BrainStem
 * module.
 */
typedef struct linkSpec {
    /**
     * \name Module Specifics
     * @{
     */
    linkType type; /**< The transport type of this spec. */
    uint32_t serial_num; /**< The serial number of the module */
    uint32_t module; /**< The module address */
    uint32_t router; /**< The BrainStem network router address */
    uint32_t router_serial_num; /**< The BrainStem network router serial number */
    uint32_t model; /**< The model type */
    /** @} */

    
    /** \name Transport Specifics
     *
     * The transport specifics are contained in a union named \ref t.
     * The union contains either of two structs usb or ip.
     *
     * The USB struct contains a single element:
     *   - **usb_id** - *uint32_t* the usb_id of the BrainStem module.
     *   .
     *
     * The TCP/IP struct contains two elements:
     *   - **ip_address** - *uint32_t* the IP4 address of the module.
     *   - **ip_port** - *uint32_t* the TCP port for socket connection on the module.
     *   .
     *
     * Address this member like ``spec.t.usb`` or ``spec.t.ip``
     * @{
     */
    union {
        struct {
            uint32_t usb_id;
        } usb;
        struct {
            uint32_t ip_address;
            uint32_t ip_port;
        } ip;
    } t; /**< transport union member. */
    /** @} */
} linkSpec;

/// Typedef #bContinueSearch

/**
* Semantic typdef for continuing the search for modules.
*/
typedef bool bContinueSearch;


/////////////////////////////////////////////////////////////////////
/// Typedef #aDiscoveryModuleFoundProc.

/**
 * This procedure is the callback to determine whether modules match the ones
 * we are looking for.
 *
 *  - **spec** - *linkSpec* passed into the continueSearch callback.
 *  - **bSuccess** - *bool* Filled with true if a module was found. false otherwise
 *  - **vpRef** - *void\* A reference to environment, or other element needed within the
 *              callback.
 *  - **return** *bContinueSearch* - Return true to continue, false to stop the search.
 */
typedef bContinueSearch (*aDiscoveryModuleFoundProc)(const linkSpec* spec,
                                                      bool* bSuccess,
                                                      void* vpRef);

#ifdef __cplusplus
extern "C" {
#endif

    /// Function #aDiscovery_EnumerateModules

    /**
     * Enumerates the discoverable modules for the given link type. Takes
     * a #aDiscoveryModuleFoundProc which will determine when to stop the
     * enumeration.
     *
     * \param type The trasport type on which search for devices. Valid #linkType "linktypes" are accepted
     * \param cbFound The #aDiscoveryModuleFoundProc to call for each module found.
     * \param vpCBRef The vpRef passed into the callback.
     *
     * \return Returns the number of modules found.
     */
    aLIBEXPORT uint8_t aDiscovery_EnumerateModules(const linkType type,
                                                   aDiscoveryModuleFoundProc cbFound,
                                                   void* vpCBRef);

    /// Function #aDiscovery_FindModule

    /**
     * Finds the module with the given serial number on the given transport type.
     *
     * \param type The trasport type on which search for devices. Valid #linkType "linktypes" are accepted
     * \param serialNum The serial number of the Module to find.
     *
     * \return A pointer to the linkSpec for the requested module if found or NULL otherwise. This call
     *         Allocates memory that must be freed by a call to #aLinkSpec_Destroy.
     */
    aLIBEXPORT linkSpec* aDiscovery_FindModule(const linkType type,
                                               uint32_t serialNum);

    /// Function #aDiscovery_FindFirstModule

    /**
     * Finds the first module found on the given transport.
     *
     * \param type The transport type on which search for devices. Valid #linkType "linktypes" are accepted
     *
     * \return A pointer to the linkSpec for the requested module if found or NULL otherwise. This call
     *         Allocates memory that must be freed by a call to #aLinkSpec_Destroy.
     */
    aLIBEXPORT linkSpec* aDiscovery_FindFirstModule(const linkType type);

    /// Function #aLinkSpec_Create

    /**
     * Creates a linkSpec object with transport set to the given type.
     *
     * \param type The transport type on which search for devices. Valid #linkType "linktypes" are accepted
     *
     * \return A pointer to the linkSpec for the requested module or NULL if there was an error allocating memory.
     *         This call Allocates memory that must be freed by a call to #aLinkSpec_Destroy.
     */
    aLIBEXPORT linkSpec* aLinkSpec_Create(const linkType type);

    /// Function #aLinkSpec_Destroy

    /**
     * Destroys and clears the referenced linkSpec.
     *
     * \param spec A pointer to the linkSpec pointer previously allocated.
     *
     * \return aErrNone on success or an error if there was an error encountered deallocating the linkSpec.
     *
     */
    aLIBEXPORT aErr aLinkSpec_Destroy(linkSpec** spec);

#ifdef __cplusplus
}
#endif
#endif //_aDiscover_H_
