/** \file logEntry.hpp
  * \brief Parses log entries with version known only at runtime and finds the next good v2 entry after damage.
  *
  * \ingroup flatlogs_files
  */
#ifndef flatlogs_logEntry_hpp
#define flatlogs_logEntry_hpp

#include <cstddef>
#include <limits>
#include "logDefs.hpp"
#include "logHeader.hpp"

namespace flatlogs
{

/// Returned by \ref findNextEntryV2 when no further verified entry is found.
/** \ingroup logbuff
  */
constexpr size_t entryNotFound = std::numeric_limits<size_t>::max();

/// Parse one log entry with version known only at runtime.
/** For readers that learn a file's version from its extension. Dispatches to \ref logHeaderT::parse for
  * the given version.
  *
  * \returns the parsed entry, with status unknownVersion if \p version has no defined layout.
  *
  * \ingroup logbuff
  */
inline parsedEntry parseEntry( const char *bytes,  ///< [in] a pointer to a raw log entry buffer.
                               size_t avail,       ///< [in] the number of readable bytes at bytes.
                               logVersionT version ///< [in] the format version to parse the entry as.
)
{
   switch( version )
   {
   case logVersion1:
      return logHeaderV1::parse( bytes, avail );
   case logVersion2:
      return logHeaderV2::parse( bytes, avail );
   default:
   {
      parsedEntry pe;
      pe.version = version;
      pe.status  = entryStatus::unknownVersion;
      return pe;
   }
   }
}

/// Get the smallest header size for version known only at runtime.
/** Minimum number of bytes a reader must have before it can work out the full header size of an entry.
  *
  * \returns the minimal header size, or 0 if \p version has no defined layout.
  *
  * \ingroup logbuff
  */
inline size_t entryMinHeadSize( logVersionT version /**< [in] the format version */ )
{
   switch( version )
   {
   case logVersion1:
      return logHeaderV1::minHeadSize;
   case logVersion2:
      return logHeaderV2::minHeadSize;
   default:
      return 0;
   }
}

/// Get entry's header size, with version known only at runtime.
/** \p bytes must hold at least \ref entryMinHeadSize.
  *
  * \returns the header size, or 0 if \p version has no defined layout.
  *
  * \ingroup logbuff
  */
inline size_t entryHeaderSize( const char *bytes,  ///< [in] the start of a log entry.
                               logVersionT version ///< [in] the format version.
                             )
{
   // Not modified, but the size accessors take char*
   char *buf = const_cast<char *>( bytes );

   switch( version )
   {
   case logVersion1:
      return logHeaderV1::headerSize( buf );
   case logVersion2:
      return logHeaderV2::headerSize( buf );
   default:
      return 0;
   }
}

/// Get an entry's claimed size, with version known only at runtime.
/** \p bytes must hold \ref entryHeaderSize.
  * For v2, includes the checksum.
  * The result comes from the entry's length field, which may be damaged: a huge length can
  * wrap the result, so a value smaller than the header size means the length is impossible.
  *
  * \returns the claimed entry size, or 0 if \p version has no defined layout.
  *
  * \ingroup logbuff
  */
inline size_t entryTotalSize( const char *bytes,  ///< [in] the start of a log entry.
                              logVersionT version ///< [in] the format version.
                            )
{
   // Not modified, but the size accessors take char*
   char *buf = const_cast<char *>( bytes );

   switch( version )
   {
   case logVersion1:
      return logHeaderV1::totalSize( buf );
   case logVersion2:
      return logHeaderV2::totalSize( buf );
   default:
      return 0;
   }
}

/// Find the next entry that passes its checksum, after a damaged v2 entry.
/** Used by readers to carry on past damage. Starting from the damaged entry at \p failedAt:
  * - If \p failed is a checksum mismatch, first try where the damaged entry's own length says the next
  *   entry starts.
  * - Otherwise, or if that fails, step forward one byte at a time.
  *
  * Never reads at or beyond \p avail. It's the caller's responsibility to set \p avail to end
  * at the current file's end.
  *
  * Only for reading. Don't use for writing.
  *
  * \returns the offset from \p bytes of the next entry whose status is checksumOk.
  * \returns \ref entryNotFound if there is none before \p avail.
  *
  * \ingroup logbuff
  */
inline size_t findNextEntryV2( const char *bytes,           ///< [in] the start of the readable bytes.
                               size_t avail,                ///< [in] the number of readable bytes at bytes.
                               size_t failedAt,             ///< [in] the offset of the damaged entry from bytes.
                               const parsedEntry &failed    ///< [in] the result of parsing the damaged entry.
)
{
   if( bytes == nullptr || failedAt >= avail )
   {
      return entryNotFound;
   }

   // After a checksum mismatch, try where the damaged entry's own length points.
   if( failed.status == entryStatus::checksumMismatch && failed.entrySize > 0 )
   {
      size_t candidate = failedAt + failed.entrySize;

      // Guard against wrap-around
      if( candidate > failedAt && candidate < avail &&
          logHeaderV2::parse( bytes + candidate, avail - candidate ).status == entryStatus::checksumOk )
      {
         return candidate;
      }
   }

   // Otherwise step forward one byte at a time.
   for( size_t pos = failedAt + 1; pos < avail; ++pos )
   {
      if( logHeaderV2::parse( bytes + pos, avail - pos ).status == entryStatus::checksumOk )
      {
         return pos;
      }
   }

   return entryNotFound;
}

} // namespace flatlogs

#endif // flatlogs_logEntry_hpp
