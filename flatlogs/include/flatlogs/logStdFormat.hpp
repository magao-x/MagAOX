/** \file logStdFormat.hpp
  * \brief Standard formating of log entries for readable output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  *
  * History:
  * - 2017-12-24 created by JRM
  * - 2018-08-18 moved to flatlogs
  */

#ifndef flatlogs_logStdFormat_hpp
#define flatlogs_logStdFormat_hpp

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"

#include "logHeader.hpp"
#include "logPriority.hpp"

namespace flatlogs
{

/// Get a decoded entry's message, to hand to log type.
/**
  * \returns a pointer to the entry's message.
  *
  * \ingroup logformat
  */
inline void * entryMessage( const parsedEntry & entry /**< [in] the decoded entry */ )
{
   // Not modified, but the log type functions don't take const
   return const_cast<char *>( entry.msg );
}

/// Get an entry's message as text.
/** An entry handed to a formatter should decode in full.
  * If it doesn't, a placeholder is returned instead.
  *
  * \returns the log type's text for the message, or "<undecodable entry>".
  *
  * \ingroup logformat
  */
template<typename logT>
std::string entryMessageString( const parsedEntry & entry /**< [in] the decoded entry */ )
{
   if( entry.msg == nullptr ) return "<undecodable entry>";

   return logT::msgString(entryMessage(entry), entry.msgLen);
}

/// Decode a raw log entry with the default version.
/** Used only by raw-entry formatters (for entries this build has just written).
  *
  * \returns the decoded entry.
  *
  * \ingroup logformat
  */
inline parsedEntry decodeWithDefaultVersion( bufferPtrT & logBuffer /**< [in] a raw log entry */ )
{
   return logHeader::parse( logBuffer.get(), logHeader::totalSize( logBuffer.get() ) );
}

/// Generate a JSON-formatted text representation of a flatlog record.
/**
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & jsonFormat( iosT & ios,                           ///< [out] the iostream to output the log to
                   const parsedEntry & entry,            ///< [in] the log to output
                   const std::string & eventCodeName,    ///< [in] the name of the entry's log type
                   const uint8_t * binarySchema,         ///< [in] the binary schema of the log type, or nullptr
                   const unsigned int binarySchemaLength ///< [in] the length of the binary schema
                 )
{
   timespecX ts = entry.ts;
   logPrioT prio = entry.prio;

   ios << "{";
   ios << "\"ts\": \"" << ts.ISO8601DateTimeStrX() << "\", ";
   ios << "\"prio\": \"" << priorityString(prio) << "\", ";
   ios << "\"ec\": \"" << eventCodeName << "\", ";
   if( entry.msg == nullptr )
   {
      ios << "\"msg\": null";
   }
   else
   {
      ios << "\"msg\": " << logT::msgJSON(entryMessage(entry), entry.msgLen, binarySchema, binarySchemaLength);
   }
   ios << "}";
   return ios;
}

/// Generate a JSON-formatted text representation of a flatlog record
/** For entries this build has just written; decodes the entry with the default version.
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & jsonFormat( iosT & ios,                            ///< [out] the iostream to output the log to
                  bufferPtrT & logBuffer,                 ///< [in] the binary log buffer to output
                  const std::string & eventCodeName,      ///< [in] the name of the entry's log type
                  const uint8_t * binarySchema,           ///< [in] the binary schema of the log type, or nullptr
                  const unsigned int binarySchemaLength   ///< [in] the length of the binary schema
                )
{
   return jsonFormat<logT>( ios, decodeWithDefaultVersion( logBuffer ), eventCodeName, binarySchema, binarySchemaLength );
}

/// Worker function that formats a log into the standard text representation.
/**
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & stdFormat( iosT & ios,               ///< [out] the iostream to output the log to
                  const parsedEntry & entry ///< [in] the log to output
                )
{
   timespecX ts = entry.ts;
   logPrioT prio = entry.prio;

   ios << ts.ISO8601DateTimeStrX() << " " << priorityString(prio) << " " << entryMessageString<logT>(entry);

   return ios;
}

/// Worker function that formats a log into the standard text representation.
/** For entries this build has just written; decodes the entry with the default version.
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & stdFormat( iosT & ios,               ///< [out] the iostream to output the log to
                  bufferPtrT & logBuffer    ///< [in] the binary log buffer to output
                )
{
   return stdFormat<logT>( ios, decodeWithDefaultVersion( logBuffer ) );
}

/// Worker function that formats a log into the standard text representation with short timespec.
/**
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & stdShortFormat( iosT & ios,                  ///< [out] the iostream to output the log to
                       const std::string & appName, ///< [in] the name of the app that wrote the entry
                       const parsedEntry & entry    ///< [in] the log to output
                     )
{
   timespecX ts = entry.ts;
   logPrioT prio = entry.prio;

   std::string outApp;

   if(appName.size() > 15)
   {
      outApp.resize(17, ' ');
      outApp[16] = ':';
      for(int n=15; n >= 0; --n)
      {
         if( 15-n + 1 > appName.size()) break;
         outApp[n] = appName[appName.size()-1 - (15-n)];
      }
   }
   else
   {
      outApp = appName;
      outApp += ":";
      outApp += std::string( 15-appName.size(), ' ');
   }


   ios << outApp << " " << ts.secondStrX() << " " << priorityString(prio) << " " << entryMessageString<logT>(entry);

   return ios;
}

/// Worker function that formats a log into the standard text representation with short timespec.
/** For entries this build has just written; decodes the entry with the default version.
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & stdShortFormat( iosT & ios,                    ///< [out] the iostream to output the log to
                       const std::string & appName,   ///< [in] the name of the app that wrote the entry
                       bufferPtrT & logBuffer         ///< [in] the binary log buffer to output
                     )
{
   return stdShortFormat<logT>( ios, appName, decodeWithDefaultVersion( logBuffer ) );
}

/// Worker function that formats a log into the minimal text representation.
/**
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & minFormat( iosT & ios,               ///< [out] the iostream to output the log to
                  const parsedEntry & entry ///< [in] the log to output
                )
{
   logPrioT prio = entry.prio;

   ios << priorityString(prio) << " " << entryMessageString<logT>(entry);

   return ios;
}

/// Worker function that formats a log into the minimal text representation.
/** For entries this build has just written; decodes the entry with the default version.
  *
  * \ingroup logformat
  */
template<typename logT, typename iosT>
iosT & minFormat( iosT & ios,            ///< [out] the iostream to output the log to
                  bufferPtrT & logBuffer ///< [in] the binary log buffer to output
                )
{
   return minFormat<logT>( ios, decodeWithDefaultVersion( logBuffer ) );
}

} //namespace flatlogs

#endif //flatlogs_logStdFormat_hpp
