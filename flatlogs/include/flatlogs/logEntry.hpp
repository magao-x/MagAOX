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
