/** \file logMap.hpp
  * \brief Declares and defines the logMap class and related classes.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2020-01-02 created by JRM
  */

#ifndef logger_logMap_hpp
#define logger_logMap_hpp

#include <mx/sys/timeUtils.hpp>
using namespace mx::sys::tscomp;

#include <mx/ioutils/fileUtils.hpp>

#include <vector>
#include <map>

#include <flatlogs/flatlogs.hpp>
#include "logFileName.hpp"

namespace MagAOX
{
namespace logger
{

/// Structure to hold a log file in memory, tracking when a new file needs to be opened.
struct logInMemory
{
   std::vector<char> m_memory; ///< The buffer holding the log.
   
   flatlogs::timespecX m_startTime {0,0};
   flatlogs::timespecX m_endTime{0,0};
   
   int loadFile( logFileName const& lfn);

};

/// Map of log entries by application name, mapping both to files and to loaded buffers.
struct logMap
{
   /// The app-name to file-name map type, for sorting the input files by application
   typedef std::map< std::string, std::set<logFileName, compLogFileName>> appToFileMapT;
   
   /// The app-name to buffer map type, for looking up the currently loaded logs for a given app.
   typedef std::map< std::string, logInMemory> appToBufferMapT;
   
   appToFileMapT m_appToFileMap;
   
   appToBufferMapT m_appToBufferMap;
   
   ///Get log file names in a directory and distribute them into the map by app-name
   int loadAppToFileMap( const std::string & dir, ///< [in] the directory to search for files
                         const std::string & ext  ///< [in] the extension to search for
                       );

   ///Get the log for an event code which is the first prior to the supplied time
   int getPriorLog( char * &logBefore,           ///< [out] pointer to the first byte of the prior log entry
                    const std::string & appName, ///< [in] the name of the app specifying which log to search
                    const flatlogs::eventCodeT & ev,       ///< [in] the event code to search for
                    const flatlogs::timespecX & ts,        ///< [in] the timestamp to be prior to
                    char * hint = 0              ///< [in] [optional] a hint specifying where to start searching.  If null search starts at beginning. 
                  );
   
   ///Get the next log with the same event code which is after the supplied time
   int getNextLog( char * &logAfter,            ///< [out] pointer to the first byte of the prior log entry
                   char * logCurrent,           ///< [in] The log to start from
                   const std::string & appName  ///< [in] the name of the app specifying which log to search
                 );
   
   int getNearestLogs( flatlogs::bufferPtrT & logBefore,
                       flatlogs::bufferPtrT & logAfter,
                       const std::string & appName
                     );
                       
   int loadFiles( const std::string & appName, ///< MagAO-X app name for which to load files
                  const flatlogs::timespecX & startTime  ///<
                );

   
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logMap_hpp
