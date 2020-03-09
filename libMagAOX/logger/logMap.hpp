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

#include <mx/timeUtils.hpp>
using namespace mx::tscomp;

#include <mx/ioutils/fileUtils.hpp>

#include <vector>
#include <map>

#include "logFileName.hpp"

/// Structure to hold a log file in memory, tracking when a new file needs to be opened.
struct logInMemory
{
   std::vector<char> m_memory; ///< The buffer holding the log.
   
   timespecX m_startTime {0,0};
   timespecX m_endTime{0,0};
   
   int loadFile( logFileName const& lfn)
   {
      int fd = open(lfn.fullName().c_str(), O_RDONLY );
      
      off_t fsz = mx::ioutils::fileSize(fd);
      
      std::vector<char> memory(fsz);
      
      ssize_t nrd = read(fd, memory.data(), memory.size());
      
      close(fd);
      
      if(nrd != fsz)
      {
         std::cerr << "logInMemory::loadFile(" << lfn.fullName() << ") did not read all bytes\n";
         return -1;
      }
      
      timespecX startTime = logHeader::timespec(memory.data());
      
      size_t st=0;
      size_t ed = logHeader::totalSize(memory.data());
      st = ed;
      
      while(st < memory.size())
      {
         ed = logHeader::totalSize(memory.data()+st);
         st = st + ed;
      }
      
      if(st != memory.size())
      {
         std::cerr << "Possibly corrupt logfile.\n";
         return -1;
      }
      
      st -= ed;
      
      timespecX endTime = logHeader::timespec(memory.data()+st);
      
      if(m_memory.size() == 0)
      {
         m_memory.swap(memory);
         m_startTime = startTime;
         m_endTime = endTime;
         return 0;
      }
      
      if(startTime < m_startTime)
      {
         
         if(endTime >= m_startTime)
         {
            std::cerr << "overlapping log files!\n";
            return -1;
         }
         
         m_memory.insert(m_memory.begin(), memory.begin(), memory.end());
         m_startTime = startTime;
         std::cerr << "added before!\n";
         return 0;
      }
      
      if(startTime > m_endTime)
      {
         std::cerr << "gonna append\n";
         m_memory.insert(m_memory.end(), memory.begin(), memory.end());
         m_endTime = endTime;
         std::cerr << "added after!\n";
         return 0;
      }
      
      std::cerr << "Need to implement insert in the middle!\n";
      std::cerr << m_startTime.time_s << " " << m_startTime.time_ns << "\n";
      std::cerr << startTime.time_s << " " << startTime.time_ns << "\n";
      
      return -1;
   }
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
                    const eventCodeT & ev,       ///< [in] the event code to search for
                    const timespecX & ts,        ///< [in] the timestamp to be prior to
                    char * hint = 0              ///< [in] [optional] a hint specifying where to start searching.  If null search starts at beginning. 
                  );
   
   ///Get the next log with the same event code which is after the supplied time
   int getNextLog( char * &logAfter,            ///< [out] pointer to the first byte of the prior log entry
                   char * logCurrent,           ///< [in] The log to start from
                   const std::string & appName  ///< [in] the name of the app specifying which log to search
                 );
   
   int getNearestLogs( bufferPtrT & logBefore,
                       bufferPtrT & logAfter,
                       const std::string & appName
                     );
                       
   int loadFiles( const std::string & appName, ///< MagAO-X app name for which to load files
                  const timespecX & startTime  ///<
                );

   
};

inline
int logMap::loadAppToFileMap( const std::string & dir,
                              const std::string & ext
                            )
{
   std::vector<std::string> flist = mx::ioutils::getFileNames(dir, "", "", ext);

   for(size_t n=0;n<flist.size(); ++n)
   {
      logFileName lfn(flist[n]);
   
      m_appToFileMap[lfn.appName()].insert(lfn);
   }

   return 0;
}

inline
int logMap::getPriorLog( char * &logBefore,
                         const std::string & appName,
                         const eventCodeT & ev,
                         const timespecX & ts,
                         char * hint
                       )
{
   eventCodeT evL;
   
   if(m_appToFileMap[appName].size() == 0)
   {
      return -1;
   }
   
   logInMemory & lim = m_appToBufferMap[appName];
   
   if(lim.m_startTime > ts || lim.m_endTime < ts)
   {
      if(loadFiles(appName, ts) < 0)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " error returned from loadfiles\n";
         return -1;
      }
   }
   
   char* buffer, *priorBuffer;
   
   if(hint) 
   {
      if(logHeader::timespec(hint) <= ts) buffer = hint;
      else buffer = lim.m_memory.data();
   }
      
   else buffer = lim.m_memory.data();
   
   priorBuffer = buffer;
   evL = logHeader::eventCode(buffer);
   
   while(evL != ev)
   {
      priorBuffer = buffer;
      buffer += logHeader::totalSize(buffer);
      if(buffer >=lim.m_memory.data() + lim.m_memory.size()) break;
      evL = logHeader::eventCode(buffer);
   }
   
   if(evL != ev)
   {
      std::cerr << "Event code not found.\n";
      return -1;
   }
   
   if( logHeader::timespec(buffer) < ts )
   {
      while( logHeader::timespec(buffer) < ts ) //Loop until buffer is after the timestamp we want
      {
         if(buffer >lim.m_memory.data() +lim.m_memory.size()) 
         {
            std::cerr << __FILE__ << " " << __LINE__ << " attempt to read too mach data, possible log corruption.\n";
            return -1;
         }
         
         if(buffer ==lim.m_memory.data() +lim.m_memory.size()) 
         {
            std::cerr << __FILE__ << " " << __LINE__ << " did not find following log -- need to load more data.\n";
            //Proper action here is to load the next file if possible...
            return 1;
         }
         
         priorBuffer = buffer;
         
         buffer += logHeader::totalSize(buffer);
         
         evL = logHeader::eventCode(buffer);
         
         while(evL != ev) //Find the next log with the event code we want.
         {
            if(buffer >lim.m_memory.data() + lim.m_memory.size()) 
            {
               std::cerr << __FILE__ << " " << __LINE__ << " attempt to read too mach data, possible log corruption.\n";
               return -1;
            }
            
            if(buffer ==lim.m_memory.data() + lim.m_memory.size()) 
            {
               std::cerr << __FILE__ << " " << __LINE__ << " did not find following log -- need to load more data.\n";
               //Proper action here is to load the next file if possible...
               return 1;
            }
            
            buffer += logHeader::totalSize(buffer);
            evL = logHeader::eventCode(buffer);
         }
      }
      
   }
   
   logBefore = priorBuffer;
   
   return 0;
}

inline
int logMap::getNextLog( char * &logAfter,            
                        char * logCurrent,           
                        const std::string & appName
                      )
{
   eventCodeT ev, evL;
   
   logInMemory & lim = m_appToBufferMap[appName];
   
   char* buffer;
   
   ev = logHeader::eventCode(logCurrent);
   
   buffer = logCurrent;
   
   buffer += logHeader::totalSize(buffer);
   if(buffer >= lim.m_memory.data() + lim.m_memory.size())
   {
      std::cerr << "Reached end of data -- need to load more data\n";
      //propoer action is to load the next file if possible.
      return 1;
   }
      
   evL = logHeader::eventCode(buffer);
   
   while(evL != ev)
   {
      buffer += logHeader::totalSize(buffer);
      if(buffer >= lim.m_memory.data() + lim.m_memory.size())
      {
         std::cerr << "Reached end of data -- need to load more data\n";
         //propoer action is to load the next file if possible.
         return 1;
      }
      evL = logHeader::eventCode(buffer);
   }
   
   if(evL != ev)
   {
      std::cerr << "Event code not found.\n";
      return -1;
   }
   
   logAfter = buffer;
   
   return 0;
}

inline
int logMap::loadFiles( const std::string & appName,
                       const timespecX & startTime
                     )
{
   if(m_appToFileMap[appName].size() == 0)
   {
      std::cerr << "*************************************\n\n";
      std::cerr << "No files for " << appName << "\n";
      std::cerr << "*************************************\n\n";
      return -1;
   }
 
   //First check if already loaded files cover this time
   if(m_appToBufferMap[appName].m_memory.size() > 0)
   {
      if( m_appToBufferMap[appName].m_startTime <= startTime && m_appToBufferMap[appName].m_endTime >= startTime) 
      {
         std::cerr << "good!\n";
         return 0;
      }
      
      if( m_appToBufferMap[appName].m_startTime > startTime ) // Files don't go back far enough
      {
         auto last = m_appToFileMap[appName].begin();
         while( last->timestamp() < m_appToBufferMap[appName].m_startTime)
         {
            ++last;
            if(last == m_appToFileMap[appName].end()) break;
         }
         //Now last is the last file to open in the for loop sense.
         auto first = last;
         
         while( first->timestamp() > startTime)
         {
            --first;
            if(first == m_appToFileMap[appName].begin()) break;
         }
         
         //Now open each of these files, in reverse
         std::cerr << "open earlier files!\n";
         --last;
         --first;
         for(auto it=last; it != first; --it)
         {
            m_appToBufferMap[appName].loadFile(*it);
         }
         
         return 0;
      }
      else
      {
         auto first = m_appToFileMap[appName].end();
         --first;
         
         while( first->timestamp() > m_appToBufferMap[appName].m_endTime)
         {
            --first;
            if(first == m_appToFileMap[appName].begin()) break;
         }
         ++first;
         auto last = first;
         while( last->timestamp() < startTime)
         {
            ++last;
            if(last == m_appToFileMap[appName].end()) break;
         }
         
         //Now open each of these files
         std::cerr << "open later file!\n";
         for(auto it=first; it != last; ++it)
         {
            m_appToBufferMap[appName].loadFile(*it);
         }
         return 0;
      }
      
   }
 
   auto before = m_appToFileMap[appName].begin();
   
   for(; before != m_appToFileMap[appName].end(); ++before)
   {
      if( !(before->timestamp() < startTime) )
      {
         break;
      }
   }

   if(before == m_appToFileMap[appName].begin())
   {
      std::cerr << "No files in range for " << appName << "\n";
   }
   --before;
   
   m_appToBufferMap.emplace(std::pair<std::string, logInMemory>(appName, logInMemory()));
   m_appToBufferMap[appName].loadFile(*before);
   
   return 0;
}

#endif //logger_logMap_hpp
