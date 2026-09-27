/** \file logDefs.hpp
  * \brief Type definitions for the flatlogs format.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  *
  * History:
  * - 2018-08-17 created by JRM
  */
#ifndef flatlogs_logDefs_hpp
#define flatlogs_logDefs_hpp

#include <cstdint>

namespace flatlogs
{

/// The type of the log priority code.
/** \ingroup logDefs
  */
typedef int8_t logPrioT;

/// The type used for seconds.
/** Rationale: unsigned 32 bits gives us enough to last 168 yrs from the UNIX epoch, which is more than enough.
  * 24 bits is just 1/2 year, so we would have to use a partial byte to be more optimum.
  *
  * \ingroup logDefs
  */
typedef uint32_t secT;

/// The type used for nanoseconds.
/** Rationale: unsigned 32 bits gives >4x10^9 nanoseconds, so enough for 1x10^9 nanoseconds.
  */
typedef uint32_t nanosecT;

/// The type of an event code (16-bit unsigned int).
/** Rationale: gives us 65,536 individual events.
  * \ingroup logDefs
  */
typedef uint16_t eventCodeT;

/// The type used for the short message length
/** Rationale: most flatlog entries are short, so we use minimum space for this.
  *
  * \ingroup logDefs
  */
typedef uint8_t  msgLen0T;

/// The type used for intermediate message length
/** Rationale: using 1+2 =3 bytes for a 256 byte message is 1.2%. 1+4 = 2.0%, and 1+8 = 3.5%.
  *
  * \ingroup logDefs
  */
typedef uint16_t msgLen1T;

/// The type used for long message length
/** Rationale: once messages are 65536 or longer, the length field is negligible.  This
  * admits messages of huge sizes.
  *
  * \ingroup logDefs
  */
typedef uint64_t msgLen2T;

/// The type used to refer to the message length, regardless of length.
/** This is not necessarily what is written to the buffer.  Should always be msgLen2T so it is big enough
  * to handle any possible length.
  * \ingroup logDefs
  */
typedef msgLen2T msgLenT;

/// The type of the version field.
/** A single byte at the front of a version-2+ entry identifies the layout so a reader can select the 
  * correct parsing rules.
  * \ingroup logDefs
  */
typedef uint8_t logVersionT;

/// The type of the per-entry checksum field.
/** 32-bit CRC Castagnoli; see logCRC.hpp for implementation.
  *
  * \ingroup logDefs
  */
typedef uint32_t crcT;

/// The format version written by the original (unversioned) flatlogs specification.
/** Version-1 entries have no version field and no checksum. This is used only to select v1 
  * parsing rules but is never written to a v1 buffer.
  *
  * \ingroup logDefs
  */
constexpr logVersionT logVersion1 = 1;

/// The new version that adds a leading version byte and a trailing CRC-32C to each entry.
/**
  * \ingroup logDefs
  */
constexpr logVersionT logVersion2 = 2;

/// The high nibble identifying a byte as a flatlogs version marker.
/** Only used for V2+ entries
  * V1's header doesn't have a version byte and instead starts with a priority with every value defined
  * in the range 0-127. V2 encodes the version byte as 0xD*. The version number is in the low
  * nibble, \ref logVersionMarker. The high nibble is a marker such that the full byte is placed outside
  * the range of defined priority values.
  * 
  * Note that this convention assumes that no defined priority is negative and none will be introduced.
  * 
  * \ingroup logDefs
  */
constexpr logVersionT logVersionMarkerBase = 0xD0;

/// The full marker byte stored at the from of a V2+ entry.
/** V1's header doesn't have a version byte and instead starts with a priority with every value defined
  * in the range 0-127. V2 encodes the version byte as 0xD*. The logical version number is in the 
  * low nibble; the high nibble is \ref logVersionMarkerBase. The high nibble is a marker such that the 
  * full byte is placed outside the range of defined priority values.
  * 
  * Out-of-range versions are rejected rather than masked into range, so that no two versions can
  * ever share a marker. Version 1 has no marker at all, so it is out of range here.
  *
  * \returns the marker byte for logical version \p v, if 2 <= v <= 15.
  * \returns 0, which is never a valid marker, otherwise.
  *
  * \ingroup logDefs
  */
constexpr logVersionT logVersionMarker( logVersionT v /**< [in] the logical format version */ )
{
   if( v < logVersion2 || v > 0x0F ) return 0;

   return logVersionMarkerBase | v;
}

/// Test whether a stored byte is the version marker for a given logical version.
/** 
  * \returns true if \p storedByte is the marker for logical version \p v, and \p v has a valid marker.
  * \returns false otherwise.
  *
  * \ingroup logDefs
  */
constexpr bool isVersionMarker( logVersionT storedByte, /**< [in] the byte read from the entry */
                                logVersionT v           /**< [in] the logical format version */
                              )
{
   const logVersionT marker = logVersionMarker( v );

   return ( marker != 0 ) && ( storedByte == marker );
}

/// The default format version.
/** For now default to \ref logVersion1.
  *
  * \ingroup logDefs
  */
constexpr logVersionT logVersionDefault = logVersion1;

}//namespace flatlogs

#endif //flatlogs_logDefs_hpp
