/** \file logHeader.hpp
  * \brief The flatlogs buffer header format.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  *
  * History:
  * - 2017-08-29 created by JRM
  * - 2018-08-17 moved to flatlogs
  *
  */
#ifndef flatlogs_logHeader_hpp
#define flatlogs_logHeader_hpp

#include <memory>
#include <type_traits>
#include <cstring>
#include "logDefs.hpp"
#include "timespecX.hpp"
#include "logPriority.hpp"
#include "logCRC.hpp"

namespace flatlogs
{

/** The log entry is a binary buffer with the following format:
  * \addtogroup flatlogs
  * \verbatim
    Version 1 (the original, unversioned format):
                         1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8
    |p|evt|time_s |time_ns|l| [message (0-253)]
    |p|evt|time_s |time_ns|E|len| [message (254-65534)]
    |p|evt|time_s |time_ns|F|len          | [message (>=65535)]

    Version 2 (adds a leading version byte and a trailing CRC-32C):
    |v|p|evt|time_s |time_ns|l| [message (0-253)] |crc|
    |v|p|evt|time_s |time_ns|E|len| [message (254-65534)] |crc|
    |v|p|evt|time_s |time_ns|F|len          | [message (>=65535)] |crc|
    \endverbatim
  * where the header consists of everything up to the [message]. Table notations are:
  * - v (uint8_t) is the format version, only in version 2+.
  * - p (uint8_t) denotes the log priority.
  * - evt (uint16_t) denotes the event code.
  * - time_s (uint32_t) is the time in seconds since the epoch
  * - time_ns (uint32_t) denotes the nanoseconds past that second.
  * - l/len is the message length field, which  is variable in length
  *   depending on the size of the message:
  *   - Message length 0-253: uint8_t in a single byte.
  *   - Message length 254-65534: uint8_t = 0xFF-1 (254) in the first byte, uint16_t in the next two bytes.
  *   - Message length 65535 and up: uint8_t = 0xFF (255) in the first byte, uint64_t in the next 8 bytes.
  * - crc (uint32_t) is the CRC-32C of every byte preceding it. Only in version 2+; See logCRC.hpp.
  *
  * Rationale for variable length: this keeps the space used for 0 length and short messages
  * to a minimum, while allowing for progressively larger messages to be saved.
  * The savings will be large for small messages, while the the cost is proportionally small
  * for larger messages.  For larger messages, this requires a progressive read to
  * sniff out the entire length.  Likewise, the cost is small relative to the cost of reading
  * the larger message.
  *
  * If a non-zero message exists, it is usually a flatlogs serialized buffer.
  *
  * \note Byte order: all multi-byte fields, including the CRC, are stored in the host's native byte
  * order, so the format is not portable across endianness.
  *
  * \note Telling the versions apart: the file extension indicates the version (.binlog/.bintel are v1,
  * .binlog2/.bintel2 are v2), and the first byte is an additional check. In v2+, the version byte
  * starts with a marker which is intended to not match any defined v1 priority (defined priorities are 
  * 0-127, markers are >= 0xD0), so a v2 reader always rejects v1 data outright.
  */

///The log entry buffer smart pointer.
/** \ingroup logbuff
  */
typedef std::shared_ptr<char> bufferPtrT;

/// The outcome of parsing one log entry.
/** Status mapping to readers' report:
  * - checksumOk: Verified.
  * - checksumMismatch: Failed (checksum).
  * - noChecksum: passed on to the v1 plausibility checks in libMagAOX, then NotChecked or Failed.
  * - overrun, markerMismatch: Failed (structure).
  *
  * \ingroup logbuff
  */
enum class entryStatus : uint8_t
{
   checksumOk,       ///< v2+: complete entry; the version marker and the checksum both match.
   checksumMismatch, ///< v2+: complete entry, but the checksum doesn't match.
   markerMismatch,   ///< v2+: the first byte is not expected v2 marker.
   noChecksum,       ///< v1: complete entry.
   overrun,          ///< The entry as described doesn't fit in the available bytes: the data is cut short, or its length is damaged.
   unknownVersion    ///< Only from parseEntry(): no matching layout for version.
};

/// The decoded contents of one log entry (and their validity).
/** Owns nothing: \ref msg points into the buffer that was parsed, and is only valid while that buffer is.
  *
  * Which fields are valid depends on \ref status:
  * - checksumOk, noChecksum: all fields are valid.
  * - checksumMismatch: all fields are decoded but untrusted, since any of them may be the damaged part.
  * - overrun: \ref version is set; \ref headerSize and \ref msgLen are set if the full header fitted.
  * Everything else is zero.
  * - markerMismatch, unknownVersion: only \ref version is set. Everything else is zero.
  *
  * \ingroup logbuff
  */
struct parsedEntry
{
   entryStatus status{ entryStatus::overrun }; ///< What the checks found.
   logVersionT version{ 0 };                   ///< The entry's version.
   logPrioT prio{ 0 };                         ///< The entry's priority.
   eventCodeT eventCode{ 0 };                  ///< The entry's event code.
   timespecX ts;                               ///< The entry's timestamp.
   msgLenT msgLen{ 0 };                        ///< The message length claimed by the header.
   size_t headerSize{ 0 };                     ///< The size of the header, including the version byte for v2+.
   size_t entrySize{ 0 };                      ///< The whole entry's size, including the checksum for v2+. 0 unless the whole entry fits.
   const char* msg{ nullptr };                 ///< The start of the message, or nullptr unless the whole entry fits.
};


/// The log entry header
/**
  * This class is designed to work with the log header only as a shared pointer to it,
  * not directly on the members.
  *
  * The class is templatized on the header \p version:
  * - v1 is the original, unversioned format (no version byte, no checksum).
  * - v2 prepends a one-byte version field and appends a four-byte CRC-32C to each entry.
  *
  * \tparam version the flatlogs format version
  *
  * \ingroup logbuff
  */
template<logVersionT version = logVersionDefault>
class logHeaderT
{

public:

   ///Only versions with a defined layout may be instantiated.
   static_assert( version == logVersion1 || version == logVersion2,
                  "logHeaderT: unsupported flatlogs format version" );

   ///The max value in the msgLen0 field.
   constexpr static size_t MAX_LEN0 = std::numeric_limits<msgLen0T>::max() ;

   ///The max value in the msgLen1 field.
   constexpr static size_t MAX_LEN1 = std::numeric_limits<msgLen1T>::max();

   ///The size of the leading version field: sizeof(logVersionT) for v2+, otherwise 0.
   constexpr static size_t versionSize = (version >= logVersion2) ? sizeof(logVersionT) : 0;

   ///The size of the trailing checksum field: sizeof(crcT) for v2+, otherwise 0.
   constexpr static size_t crcSize = (version >= logVersion2) ? sizeof(crcT) : 0;

   ///The size of the fixed-width part of the header, i.e. everything before the length field.
   /** Defined once here as a single source of truth for both allocating entries and reading.
     */
   constexpr static size_t fixedHeadSize = versionSize + sizeof(logPrioT) + sizeof(eventCodeT) +
                                           sizeof(timespecX);

   ///The minimum header size
   /** A buffer must be allocated to at least this size. Includes the version byte for v2+.
     */
   constexpr static int minHeadSize = fixedHeadSize + sizeof(msgLen0T);

   ///The maximum header size
   /** The header component could be as big as this.
     */
   constexpr static int maxHeadSize = minHeadSize + sizeof(msgLen2T);

private:

   /// On-disk header layout for v1 (unversioned).
   struct internal_logHeaderV1
   {
      logPrioT m_logLevel;
      eventCodeT m_eventCode;
      timespecX m_timespecX;
      msgLen0T msgLen0; ///< The short message length.  Always present.
      union
      {
         msgLen1T msgLen1; ///< The intermediate message length.  Only present if msgLen0 == MAX_LEN0-1 (i.e. 254).
         msgLen2T msgLen2; ///< The long message length.  Only present if msgLen0 == MAX_LEN0 (i.e. 255).
      };
   } __attribute__((packed));

   /// On-disk header layout for v2: a leading version byte, then the version-1 fields.
   struct internal_logHeaderV2
   {
      logVersionT m_version;
      logPrioT m_logLevel;
      eventCodeT m_eventCode;
      timespecX m_timespecX;
      msgLen0T msgLen0; ///< The short message length.  Always present.
      union
      {
         msgLen1T msgLen1; ///< The intermediate message length. Only present if msgLen0 == MAX_LEN0-1 (i.e. 254).
         msgLen2T msgLen2; ///< The long message length. Only present if msgLen0 == MAX_LEN0 (i.e. 255).
      };
   } __attribute__((packed));

   /// The layout is selected at compile time according to the version.
   using internal_logHeader = std::conditional_t<( version >= logVersion2 ), internal_logHeaderV2, internal_logHeaderV1>;

public:

   ///Get the format version of this instance.
   /** \returns the \p version template parameter.
     */
   constexpr static logVersionT formatVersion();

   ///Set the version field of a log entry (v2+ only; no-op for v1).
   /**
     * \returns 0 on success
     *
     * \ingroup logbuff
     */
   static int setVersion( bufferPtrT & logBuffer /**< [in,out] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Set the version field of a log entry (v2+ only; no-op for v1).
   /** Writes \ref logVersionMarker for this version, not the logical version number.
     *
     * \returns 0 on success
     *
     * \ingroup logbuff
     */
   static int setVersion( char * logBuffer /**< [in,out] a pointer to a raw log entry buffer.*/);

   ///Extract the logical format version of a log entry.
   /** 
     * \returns \ref logVersion1 for v1, which stores no version field.
     * \returns the logical version for v2+.
     * \returns 0 if the stored byte is not a valid \ref logVersionMarker.
     *
     * \ingroup logbuff
     */
   static logVersionT getVersion( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the logical format version of a log entry.
   /** 
     * \returns \ref logVersion1 for v1, which stores no version field.
     * \returns the logical version for v2+.
     * \returns 0 if the stored byte is not a valid \ref logVersionMarker.
     *
     * \ingroup logbuff
     */
   static logVersionT getVersion( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Set the level of a log entry in a logBuffer header
   /**
     * \returns 0 on success
     * \returns -1 on error
     *
     * \ingroup logbuff
     */
   static int logLevel( bufferPtrT & logBuffer, ///< [in/out] a shared_ptr\<char\> containing a raw log entry buffer.
                        const logPrioT & lvl   ///< [in] the new log level.
                      );

   ///Extract the level of a log entry
   /**
     * \returns the level
     *
     * \ingroup logbuff
     */
   static logPrioT logLevel( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the level of a log entry
   /**
     * \returns the level
     *
     * \ingroup logbuff
     */
   static logPrioT logLevel( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Set the event code of a log entry
   /**
     * \returns 0 on success
     * \returns -1 on error
     *
     * \ingroup logbuff
     */
   static int eventCode( bufferPtrT & logBuffer, ///< [in,out] a shared_ptr\<char\> containing a raw log entry buffer.
                         const eventCodeT & ec   ///< [in] the new event code.
                       );

   ///Extract the event code of a log entry
   /**
     * \returns the event code
     *
     * \ingroup logbuff
     */
   static eventCodeT eventCode( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the event code of a log entry
   /**
     * \returns the event code
     *
     * \ingroup logbuff
     */
   static eventCodeT eventCode( char * logBuffer /**< [in] a pointer a raw log entry buffer.*/);

   ///Set the timespec of a log entry
   /**
     * \returns 0 on success
     * \returns -1 on error
     *
     * \ingroup logbuff
     */
   static int timespec( bufferPtrT & logBuffer,    ///< [in, out] a shared_ptr\<char\> containing a raw log entry buffer.*/
                        const timespecX & ts ///< [in] the new timespec
                      );

   ///Extract the timespec of a log entry
   /**
     * \returns the timespec
     *
     * \ingroup logbuff
     */
   static timespecX timespec( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the timespec of a log entry
   /**
     * \returns the timespec
     *
     * \ingroup logbuff
     */
   static timespecX timespec( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the size in bytes of the length field for an existing logBuffer.
   /**
     * \returns the number of bytes in the length field.
     *
     * \ingroup logbuff
     */
   static size_t lenSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the size in bytes of the length field for an existing logBuffer.
   /**
     * \returns the number of bytes in the length field.
     *
     * \ingroup logbuff
     */
   static size_t lenSize( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the size in bytes of the length field for a logBuffer given the intended message length
   /**
     * \returns the number of bytes to be put in the length field.
     *
     * \ingroup logbuff
     */
   static size_t lenSize( msgLenT & msgSz /**< [in] the size of the intended message.*/);

   ///Set the message length of a log entry message
   /**
     * \note The logBuffer must already be allocated with a header large enough for this message size.
     *
     * \returns 0 on success
     * \returns -1 on error
     *
     * \ingroup logbuff
     */
   static int msgLen( bufferPtrT & logBuffer, ///< [out] a shared_ptr\<char\> containing a raw log entry buffer allocated with large enough header for this message length.
                      const msgLenT & msgLen  ///< [in] the message length to set.
                    );

   ///Extract the short message length of a log entry message
   /** This is always safe on a minimally allocated logBuffer, can be used to test for progressive reading.
     *
     * \returns the short message length field
     *
     * \ingroup logbuff
     */
   static msgLen0T msgLen0(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the short message length of a log entry message
   /** This is always safe on a minimally allocated logBuffer, can be used to test for progressive reading.
     *
     * \returns the short message length field
     *
     * \ingroup logbuff
     */
   static msgLen0T msgLen0( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Extract the medium message length of a log entry message
   /** This is NOT always safe, and should only be caled if msgLen0 is 0xFE. Can be used to test for progressive reading.
     *
     * \returns the medium message length field
     *
     * \ingroup logbuff
     */
   static msgLen1T msgLen1(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the medium message length of a log entry message
   /** This is NOT always safe, and should only be caled if msgLen0 is 0xFE. Can be used to test for progressive reading.
     *
     * \returns the medium message length field
     *
     * \ingroup logbuff
     */
   static msgLen1T msgLen1( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Extract the message length of a log entry message
   /**
     * \returns the message length
     *
     * \ingroup logbuff
     */
   static msgLenT msgLen(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Extract the message length of a log entry message
   /**
     * \returns the message length
     *
     * \ingroup logbuff
     */
   static msgLenT msgLen( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the size of the header, including the variable size length field, for an existing logBuffer.
   /**
     * \returns the size of the header in this log entry.
     *
     * \ingroup logbuff
     */
   static size_t headerSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the size of the header, including the variable size length field, for an existing logBuffer.
   /**
     * \returns the size of the header in this log entry.
     *
     * \ingroup logbuff
     */
   static size_t headerSize( char* logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the size of the header, including the variable size length field, given a message size.
   /**
     * \returns the size to be used for the header.
     *
     * \ingroup logbuff
     */
   static size_t headerSize(  msgLenT & msgSz /**< [in] the size of the intended message.*/);

   ///Get the total size of the log entry, including the message buffer.
   /** Includes the trailing CRC field for v2+.
     *
     * \returns the total size of this log entry.
     *
     * \ingroup logbuff
     */
   static size_t totalSize(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the total size of the log entry, including the message buffer.
   /** Includes the trailing CRC field for v2+.
     *
     * \returns the total size of this log entry.
     *
     * \ingroup logbuff
     */
   static size_t totalSize( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Get the total size of a log entry, given the message buffer size.
   /** Includes the trailing CRC field for v2+.
     *
     * \returns the total size to be used for a log entry.
     *
     * \ingroup logbuff
     */
   static size_t totalSize(  msgLenT & msgSz /**< [in] the intended size of the message buffer.*/);

   ///Get the message buffer address.
   /**
     * \returns the address of the message buffer.
     *
     * \ingroup logbuff
     */
   static void * messageBuffer(  bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the message buffer address.
   /**
     * \returns the address of the message buffer.
     *
     * \ingroup logbuff
     */
   static void * messageBuffer( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Compute the CRC-32C over the header and message of a v2+ entry
   /** The message length must already be set.
     *
     * \returns the CRC-32C of the entry excluding the CRC field.
     *
     * \ingroup logbuff
     */
   static crcT computeCRC( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Compute the CRC-32C over the header and message of a v2+ entry
   /** The message length must already be set.
     *
     * \returns the CRC-32C of the entry excluding the CRC field.
     *
     * \ingroup logbuff
     */
   static crcT computeCRC( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Get the byte offset of the CRC field (same as the number of bytes it's computed over).
   /** The message length must already be set.
     *
     * \warning This reads the length field without validating it, so it is only safe on a buffer
     * already known to hold a complete entry.
     *
     * \returns the offset of the CRC field from the start of the entry.
     *
     * \ingroup logbuff
     */
   static size_t crcOffset( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Extract the CRC field of a v2+ log entry.
   /** \returns the stored CRC-32C for v2+, otherwise 0.
     *
     * \ingroup logbuff
     */
   static crcT getCRC( char * logBuffer /**< [in] a pointer to a raw log entry buffer.*/);

   ///Extract the CRC field of a v2+ log entry.
   /** \returns the stored CRC-32C for v2+, otherwise 0.
     *
     * \ingroup logbuff
     */
   static crcT getCRC( bufferPtrT & logBuffer /**< [in] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Compute and write the CRC field of a v2+ log entry (no-op for v1).
   /** The message length must already be set and the message already written.
     *
     * \returns 0 on success
     *
     * \ingroup logbuff
     */
   static int setCRC( bufferPtrT & logBuffer /**< [in,out] a shared_ptr\<char\> containing a raw log entry buffer.*/);

   ///Compute and write the CRC field of a v2+ log entry (no-op for v1).
   /** The message length must already be set and the message already written.
     *
     * \returns 0 on success
     *
     * \ingroup logbuff
     */
   static int setCRC( char * logBuffer /**< [in,out] a pointer to a raw log entry buffer.*/);

   ///Parse one log entry, reporting which check it failed and decoding its fields, without reading out of bounds.
   /** \ref verify is a wrapper over this.
     *
     * If \p avail is smaller than the minimal header the result is overrun, never markerMismatch, because
     * the marker byte may not even be there.
     *
     * See \ref parsedEntry for which fields are valid for each status.
     *
     * \returns the parsed entry.
     *
     * \ingroup logbuff
     */
   static parsedEntry parse( const char * logBuffer, ///< [in] a pointer to a raw log entry buffer.
                             size_t avail            ///< [in] the number of readable bytes at logBuffer.
                           );

   ///Verify that a buffer holds a well-formed, uncorrupted log entry, without reading out of bounds.
   /** Implemented with \ref parse; true when parse returns checksumOk (v2+) or noChecksum (v1).
     *
     * On success \p entrySize is set to the total number of bytes the entry occupies, so the caller can
     * advance to the next entry.
     * On failure \p entrySize is set to 0.
     * 
     * \returns true if the entry is well-formed (and for v2+ its checksum matches).
     * \returns false otherwise, including when \p avail is too small.
     *
     * \ingroup logbuff
     */
   static bool verify( const char * logBuffer, ///< [in] a pointer to a raw log entry buffer.
                       size_t avail,           ///< [in] the number of readable bytes at logBuffer.
                       size_t & entrySize      ///< [out] the total size of the entry on success, 0 on failure.
                     );

   /// Create a formatted log entry, filling in a buffer.
   /** This version has the timestamp provided.
     * For v2+ it also sets the version byte and, after the message is written, computes and stores
     * the CRC-32C.
     *
     * \tparam logT is a log entry type
     *
     * \returns 0 on success, -1 on error.
     */
   template<typename logT>
   static int createLog( bufferPtrT & logBuffer,              ///< [out] a shared_ptr\<logBuffer\>, which will be allocated and populated with the log entry
                         const timespecX & ts,          ///< [in] the timestamp of this log entry.
                         const typename logT::messageT & msg, ///< [in] the message to log (could be of type emptyMessage)
                         const logPrioT & level              ///< [in] the level (verbosity) of this log
                       );

   ///Extract the basic details of a log entry
   /** Convenience wrapper for the other extraction functions.
     *
     * \returns 0 on success, -1 on error.
     *
     * \ingroup logbuff
     */
   static int extractBasicLog( logPrioT & lvl,        ///< [out] The log level
                               eventCodeT & ec,       ///< [out] the event code
                               timespecX & ts,        ///< [out] the timestamp of the log entry
                               msgLenT & len,         ///< [out] the message length
                               bufferPtrT & logBuffer ///< [in] a shared_ptr\<char\> containing a raw log entry buffer.
                             );

   ///Extract the basic details of a log entry
   /** Convenience wrapper for the other extraction functions.
     *
     * \returns 0 on success, -1 on error.
     *
     * \ingroup logbuff
     */
   static int extractBasicLog( logPrioT & lvl,  ///< [out] The log level
                               eventCodeT & ec, ///< [out] the event code
                               timespecX & ts,  ///< [out] the timestamp of the log entry
                               msgLenT & len,   ///< [out] the message length
                               char* logBuffer  ///< [in] a pointer to a raw log entry buffer.
                             );

};

template<logVersionT version>
constexpr logVersionT logHeaderT<version>::formatVersion()
{
   return version;
}

template<logVersionT version>
int logHeaderT<version>::setVersion( bufferPtrT & logBuffer )
{
   return setVersion(logBuffer.get());
}

template<logVersionT version>
int logHeaderT<version>::setVersion( char * logBuffer )
{
   if constexpr( version >= logVersion2 )
   {
      reinterpret_cast<internal_logHeader *>(logBuffer)->m_version = logVersionMarker(version);
   }

   return 0;
}

template<logVersionT version>
logVersionT logHeaderT<version>::getVersion( bufferPtrT & logBuffer )
{
   return getVersion(logBuffer.get());
}

template<logVersionT version>
logVersionT logHeaderT<version>::getVersion( char * logBuffer )
{
   if constexpr( version >= logVersion2 )
   {
      logVersionT stored = reinterpret_cast<internal_logHeader *>(logBuffer)->m_version;

      logVersionT v = stored & 0x0F;

      return isVersionMarker( stored, v ) ? v : 0;
   }
   else
   {
      return logVersion1;
   }
}

template<logVersionT version>
int logHeaderT<version>::logLevel( bufferPtrT & logBuffer,
                          const logPrioT & lvl
                        )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_logLevel = lvl;

   return 0;
}

template<logVersionT version>
logPrioT logHeaderT<version>::logLevel( bufferPtrT & logBuffer)
{
   return logLevel( logBuffer.get() );

   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_logLevel;
}

template<logVersionT version>
logPrioT logHeaderT<version>::logLevel( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_logLevel;
}

template<logVersionT version>
int logHeaderT<version>::eventCode( bufferPtrT & logBuffer,
                           const eventCodeT & ec
                         )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_eventCode = ec;

   return 0;
}

template<logVersionT version>
eventCodeT logHeaderT<version>::eventCode( bufferPtrT & logBuffer)
{
   return eventCode(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_eventCode;
}

template<logVersionT version>
eventCodeT logHeaderT<version>::eventCode( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_eventCode;
}

template<logVersionT version>
int logHeaderT<version>::timespec( bufferPtrT & logBuffer,
                          const timespecX & ts
                        )
{
   reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_timespecX = ts;

   return 0;
}

template<logVersionT version>
timespecX logHeaderT<version>::timespec( bufferPtrT & logBuffer)
{
   return timespec(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->m_timespecX;
}

template<logVersionT version>
timespecX logHeaderT<version>::timespec( char * logBuffer)
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->m_timespecX;
}

template<logVersionT version>
size_t logHeaderT<version>::lenSize( bufferPtrT & logBuffer )
{
   return lenSize(logBuffer.get());
}

template<logVersionT version>
size_t logHeaderT<version>::lenSize( char* logBuffer )
{
   msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;

   if(len0 < MAX_LEN0-1) return sizeof(msgLen0T);

   if(len0 == MAX_LEN0-1)
   {
      return sizeof(msgLen0T) + sizeof(msgLen1T);
   }

   return sizeof(msgLen0T) + sizeof(msgLen2T);
}

template<logVersionT version>
size_t logHeaderT<version>::lenSize(  msgLenT & msgSz /**< [in] the size of the intended message.*/)
{
   if(msgSz < MAX_LEN0-1) return sizeof(msgLen0T);

   if(msgSz < MAX_LEN1) return sizeof(msgLen0T) + sizeof(msgLen1T);

   return sizeof(msgLen0T) + sizeof(msgLen2T);
}

template<logVersionT version>
size_t logHeaderT<version>::headerSize(  bufferPtrT & logBuffer )
{
   return headerSize(logBuffer.get());
}

template<logVersionT version>
size_t logHeaderT<version>::headerSize(  char* logBuffer )
{
   return fixedHeadSize + lenSize(logBuffer);
}

template<logVersionT version>
size_t logHeaderT<version>::headerSize(  msgLenT & msgSz )
{
   return fixedHeadSize + lenSize(msgSz);
}

template<logVersionT version>
int logHeaderT<version>::msgLen( bufferPtrT & logBuffer,
                        const msgLenT & msgLen
                      )
{
   internal_logHeader * lh = reinterpret_cast<internal_logHeader *>(logBuffer.get());

   if( msgLen < MAX_LEN0-1 ) //254 for uint8_t
   {
      lh->msgLen0 = msgLen;
      return 0;
   }

   if(msgLen < MAX_LEN1) //65535 for uint16_t
   {
      lh->msgLen0 = MAX_LEN0-1; //254 for uint8_t
      lh->msgLen1 = msgLen;
      return 0;
   }

   lh->msgLen0 = MAX_LEN0; //255 for  uint8_t
   lh->msgLen2 = msgLen;
   return 0;
}

template<logVersionT version>
msgLen0T logHeaderT<version>::msgLen0( bufferPtrT & logBuffer )
{
   return msgLen0(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen0;
}

template<logVersionT version>
msgLen0T logHeaderT<version>::msgLen0( char * logBuffer )
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;
}

template<logVersionT version>
msgLen1T logHeaderT<version>::msgLen1( bufferPtrT & logBuffer )
{
   return msgLen1(logBuffer.get());
   //return reinterpret_cast<internal_logHeader *>(logBuffer.get())->msgLen1;
}

template<logVersionT version>
msgLen1T logHeaderT<version>::msgLen1( char* logBuffer )
{
   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen1;
}

template<logVersionT version>
msgLenT logHeaderT<version>::msgLen(  bufferPtrT & logBuffer )
{
   return msgLen(logBuffer.get());
}

template<logVersionT version>
msgLenT logHeaderT<version>::msgLen(  char* logBuffer )
{
   msgLen0T len0 = reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen0;

   //std::cerr << "len0: " << (int) len0 << "\n";

   if(len0 < MAX_LEN0-1) return len0;

   if(len0 == MAX_LEN0-1)
   {
      return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen1;
   }

   return reinterpret_cast<internal_logHeader *>(logBuffer)->msgLen2;
}

template<logVersionT version>
size_t logHeaderT<version>::totalSize(  bufferPtrT & logBuffer )
{
   return totalSize(logBuffer.get());
   //return headerSize(logBuffer) + msgLen(logBuffer);
}

template<logVersionT version>
size_t logHeaderT<version>::totalSize( char* logBuffer )
{
   return headerSize(logBuffer) + msgLen(logBuffer) + crcSize;
}

template<logVersionT version>
size_t logHeaderT<version>::totalSize(  msgLenT & msgSz )
{
   return headerSize(msgSz) + msgSz + crcSize;
}

template<logVersionT version>
void * logHeaderT<version>::messageBuffer(  bufferPtrT & logBuffer )
{
   return messageBuffer(logBuffer.get());
   //return logBuffer.get() + headerSize(logBuffer);
}

template<logVersionT version>
void * logHeaderT<version>::messageBuffer( char* logBuffer )
{
   return logBuffer + headerSize(logBuffer);
}

template<logVersionT version>
size_t logHeaderT<version>::crcOffset( char * logBuffer )
{
   //The CRC is the last field of an entry and covers everything before it.
   return headerSize(logBuffer) + msgLen(logBuffer);
}

template<logVersionT version>
crcT logHeaderT<version>::computeCRC( bufferPtrT & logBuffer )
{
   return computeCRC(logBuffer.get());
}

template<logVersionT version>
crcT logHeaderT<version>::computeCRC( char * logBuffer )
{
   return crc32c(logBuffer, crcOffset(logBuffer));
}

template<logVersionT version>
crcT logHeaderT<version>::getCRC( bufferPtrT & logBuffer )
{
   return getCRC(logBuffer.get());
}

template<logVersionT version>
crcT logHeaderT<version>::getCRC( char * logBuffer )
{
   if constexpr( version >= logVersion2 )
   {
      crcT stored;
      std::memcpy( &stored, logBuffer + crcOffset(logBuffer), sizeof(crcT) );
      return stored;
   }
   else
   {
      return 0;
   }
}

template<logVersionT version>
int logHeaderT<version>::setCRC( bufferPtrT & logBuffer )
{
   return setCRC(logBuffer.get());
}

template<logVersionT version>
int logHeaderT<version>::setCRC( char * logBuffer )
{
   if constexpr( version >= logVersion2 )
   {
      crcT computed = computeCRC(logBuffer);
      std::memcpy( logBuffer + crcOffset(logBuffer), &computed, sizeof(crcT) );
   }

   return 0;
}

template<logVersionT version>
parsedEntry logHeaderT<version>::parse( const char * logBuffer,
                                        size_t avail
                                      )
{
   parsedEntry pe;
   pe.version = version;
   pe.status = entryStatus::overrun;

   // No buffer means no bytes available.
   if( logBuffer == nullptr ) return pe;

   // Check there are enough bytes for the minimal header.
   if( avail < static_cast<size_t>(minHeadSize) ) return pe;

   // Parsing doesn't modify the buffer, but the accessors take char* not const char*.
   char* buf = const_cast<char*>(logBuffer);

   // For v2+ validate the leading marker byte.
   if constexpr( version >= logVersion2 )
   {
      if( !isVersionMarker( reinterpret_cast<internal_logHeader *>(buf)->m_version, version ) )
      {
         pe.status = entryStatus::markerMismatch;
         return pe;
      }
   }

   // Need the full header before reading the length
   size_t hSz = headerSize(buf);
   if( avail < hSz ) return pe;

   msgLenT mLen = msgLen(buf);

   // Report what the header claims to check for overrun.
   pe.headerSize = hSz;
   pe.msgLen = mLen;

   // Check length is not corrupted.
   if( mLen > avail ) return pe;

   size_t tSz = hSz + mLen + crcSize;
   if( tSz < hSz ) return pe;
   if( avail < tSz ) return pe; //not enough bytes for the whole entry (including CRC)

   // The whole entry fits, so its fields can be decoded. The status records whether they are valid.
   pe.prio = logLevel(buf);
   pe.eventCode = eventCode(buf);
   pe.ts = timespec(buf);
   pe.entrySize = tSz;
   pe.msg = logBuffer + hSz;

   // For v2+, verify the checksum.
   // hSz and mLen have already been checked against avail, so crcOffset() shouldn't be used here,
   // since it re-reads the length field without validating it.
   if constexpr( version >= logVersion2 )
   {
      crcT stored;
      std::memcpy( &stored, buf + hSz + mLen, sizeof(crcT) );
      crcT computed = crc32c(buf, hSz + mLen);
      pe.status = ( stored == computed ) ? entryStatus::checksumOk : entryStatus::checksumMismatch;
   }
   else
   {
      pe.status = entryStatus::noChecksum;
   }

   return pe;
}

template<logVersionT version>
bool logHeaderT<version>::verify( const char * logBuffer,
                                  size_t avail,
                                  size_t & entrySize
                                )
{
   parsedEntry pe = parse( logBuffer, avail );

   if( pe.status == entryStatus::checksumOk || pe.status == entryStatus::noChecksum )
   {
      entrySize = pe.entrySize;
      return true;
   }

   entrySize = 0;
   return false;
}

template<logVersionT version>
template<typename logT>
int logHeaderT<version>::createLog( bufferPtrT & logBuffer,
                          const timespecX & ts,
                          const typename logT::messageT & msg,
                          const logPrioT & level
                        )
{
   logPrioT lvl;
   if(level == logPrio::LOG_DEFAULT)
   {
      lvl = logT::defaultLevel;
   }
   else lvl = level;

   //We first allocate the buffer.
   msgLenT len = logT::length(msg);
   logBuffer = bufferPtrT( (char*) ::operator new(totalSize(len)*sizeof(char)) );

   //For v2+ set the version byte first
   setVersion(logBuffer);

   //Now load the basics.
   logLevel(logBuffer, lvl);
   eventCode(logBuffer, +logT::eventCode); //The + fixes an issue with undefined references
   timespec(logBuffer, ts);

   msgLen( logBuffer, len);


   //Each log-type is responsible for loading its message
   logT::format( messageBuffer(logBuffer), msg);

   //For v2+, compute and store the trailing CRC over the completed entry.
   setCRC(logBuffer);

   return 0;

}

template<logVersionT version>
int logHeaderT<version>::extractBasicLog( logPrioT & lvl,
                                eventCodeT & ec,
                                timespecX & ts,
                                msgLenT & len,
                                bufferPtrT & logBuffer
                              )
{
   return extractBasicLog(lvl, ec, ts, len, logBuffer.get());
}

template<logVersionT version>
int logHeaderT<version>::extractBasicLog( logPrioT & lvl,
                                eventCodeT & ec,
                                timespecX & ts,
                                msgLenT & len,
                                char* logBuffer
                              )
{
   lvl = logLevel(logBuffer);

   ec = eventCode(logBuffer);

   ts = timespec(logBuffer);

   len = logHeaderT<version>::msgLen(logBuffer);

   return 0;
}

/// The original flatlogs format (no version byte, no checksum).
/** \ingroup logbuff */
using logHeaderV1 = logHeaderT<logVersion1>;

/// The versioned flatlogs format with a leading version byte and trailing CRC-32C.
/** \ingroup logbuff */
using logHeaderV2 = logHeaderT<logVersion2>;

/// The default flatlogs format when not version is set explicitly.
/** Remains \ref logHeaderV1 for now.
  * \ingroup logbuff */
using logHeader = logHeaderT<logVersionDefault>;

} //namespace flatlogs

#endif //flatlogs_logHeader_hpp
