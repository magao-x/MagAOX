/////////////////////////////////////////////////////////////////////

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* file: Version.h                                                 */
/*                                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                 */
/* description: Version header file                                */
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

#ifndef _aVersion_H_
#define _aVersion_H_

/////////////////////////////////////////////////////////////////////
/// Library version interface.

/** \defgroup aVersion Version Interface
 * \ref aVersion "aVersion.h" Provides version information for the BrainStem2
 * library.
 */

/// Major revision level of library.

/**
 * Major revision bumps will break compatibility with existing
 * versions and may introduce protocol changes or other fundamental
 * differences.
 */
#define aVERSION_MAJOR                                             2

/// Minor revision level of library.
/**
 * Minor revisions should largly be compatible, however new features
 * may be added with a minor revision change.
 */
#define aVERSION_MINOR                                             7

/// Patch revision level of library.
/**
 * Patch revisions are bug fixes and small performance changes. They
 * add no significant new features or interfaces.
 */
#define aVERSION_PATCH                                             7

/// The maximum string length of the human readable version string.
#define MAX_VERSION_STRING                                       15U

#ifndef _aDefs_H_
#include "aDefs.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /** \defgroup Firmware_version_parsing Firmware Version Parsing
     *
     * @{
     */

    /// Parse out the major revision number.

    /**
     * Parses the major revision level from the given uint32.
     * \param build The packed version number returned from the
     *              system.getVersion call.
     * \returns The major revision number.
     */
    aLIBEXPORT uint8_t aVersion_ParseMajor(uint32_t build);

    /// Parse out the minor revision number.

    /**
     * Parses the minor revision level from the given uint32.
     * \param build The packed version number returned from the
     *              system.getVersion call.
     * \returns The minor revision number.
     */
    aLIBEXPORT uint8_t aVersion_ParseMinor(uint32_t build);


    /// Parse out the revision patch number.

    /**
     * Parses the revision patch level from the given uint32.
     * \param build The packed version number returned from the
     *              system.getVersion call.
     * \returns The revision patch number.
     */
    aLIBEXPORT uint32_t aVersion_ParsePatch(uint32_t build);

    /// Parse the Version number into a human readable format.

    /**
     * Fills the string parameter with a human readable formated version number.
     * \param build The packed version number returned from the
     *              system.getVersion call.
     * \param string The string to fill with the version string.
     * \param len The length of the filled string, not longer than
     *            MAX_VERSION_STRING.
     */
    aLIBEXPORT void aVersion_ParseString(uint32_t build, char* string, size_t len);
    /** @} */

    /// Return the major revision number.

    /**
     * \returns The major revision number.
     */
    aLIBEXPORT uint8_t aVersion_GetMajor(void);


    /// Return the minor revision number.

    /**
     * \returns The minor revision number.
     */
    aLIBEXPORT uint8_t aVersion_GetMinor(void);

    /// Return the revision patch number.

    /**
     * \returns The revision patch number.
     */
    aLIBEXPORT uint32_t aVersion_GetPatch(void);


    /// Return a human readable version string.

    /**
     * \returns char* human readable version string.
     */
    aLIBEXPORT const char* aVersion_GetString(void);

    /// Check that the current version is at least major.minor.patch

    /**
     * \param major The major revision level.
     * \param minor The minor revision.
     * \param patch The patch level.
     *
     * \returns True when current version is at least what is given, false
     *          otherwise
     */
    aLIBEXPORT bool aVersion_IsAtLeast(const uint8_t major,
                                       const uint8_t minor,
                                       const uint8_t patch);
    /// Check that the library supports a feature
    /**
     * \param feature The feature to check.
     * \returns True if the current library supports the feature or false if not.
     */
    aLIBEXPORT bool aVersion_HasFeature(const char* feature);

    /// Get an array of the features the library supports.
    /**
     * \returns an array of c strings describing the features the library
     * supports.
     */
    aLIBEXPORT char* aVersion_GetFeatureList(void);

    /// Destroy the feature list.
    /**
     * \param featureList pointer to featurelist.
     */
    aLIBEXPORT void aVersion_DestroyFeatureList(char** featureList);

#ifdef __cplusplus
}
#endif

#endif //_aVersion_H_
