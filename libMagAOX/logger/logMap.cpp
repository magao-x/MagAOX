/** \file logMap.cpp
  * \brief Declares and defines the logMap class and related classes.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  */

#include "logMap.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace flatlogs;

namespace MagAOX
{
namespace logger
{
   
int logInMemory::loadFile( logFileName const& lfn)
{
   int fd = open(lfn.fullName().c_str(), O_RDONLY );

   off_t fsz = mx::ioutils::fileSize(fd);
   
   std::vector<char> memory(fsz);
   
   ssize_t nrd = read(fd, memory.data(), memory.size());
   
   close(fd);
   
   if(nrd != fsz)
   {
      std::cerr << __FILE__ << " " << __LINE__ << " logInMemory::loadFile(" << lfn.fullName() << ") did not read all bytes\n";
      return -1;
   }
   
   flatlogs::timespecX startTime = logHeader::timespec(memory.data());
   
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
      std::cerr <<  __FILE__ << " " << __LINE__ << " Possibly corrupt logfile.\n";
      return -1;
   }
   
   st -= ed;
   
   flatlogs::timespecX endTime = logHeader::timespec(memory.data()+st);
   
   if(m_memory.size() == 0)
   {
      m_memory.swap(memory);
      m_startTime = startTime;
      m_endTime = endTime;

      std::string timestamp;   
      timespec ts{ endTime.time_s, endTime.time_ns};
      mx::sys::timeStamp(timestamp, ts);

      #ifdef DEBUG
      std::cerr << __FILE__ << " " << __LINE__ << " loading: " << lfn.fullName() << " " << timestamp << "\n";
      #endif

      return 0;
   }
   
   if(startTime < m_startTime)
   {
      
      if(endTime >= m_startTime)
      {
         std::cerr <<  __FILE__ << " " << __LINE__ << " overlapping log files!\n";
         return -1;
      }
      
      m_memory.insert(m_memory.begin(), memory.begin(), memory.end());
      m_startTime = startTime;
      std::cerr <<  __FILE__ << " " << __LINE__ << " added before!\n";
      return 0;
   }
   
   if(startTime > m_endTime)
   {
      #ifdef DEBUG
      std::cerr <<  __FILE__ << " " << __LINE__ << " gonna append\n";
      #endif

      m_memory.insert(m_memory.end(), memory.begin(), memory.end());
      m_endTime = endTime;
      
      #ifdef DEBUG
      std::cerr <<  __FILE__ << " " << __LINE__ << " added after!\n";
      #endif
      
      return 0;
   }
   
   std::cerr <<  __FILE__ << " " << __LINE__ << " Need to implement insert in the middle!\n";
   std::cerr << m_startTime.time_s << " " << m_startTime.time_ns << "\n";
   std::cerr << startTime.time_s << " " << startTime.time_ns << "\n";
   
   return -1;
}

int logMap::loadAppToFileMap( const std::string & dir,
                              const std::string & ext
                            )
{
   std::vector<std::string> flist = mx::ioutils::getFileNames(dir, "", "", ext);

   for(size_t n=0;n<flist.size(); ++n)
   {
      //std::cerr << "loading: " << flist[n] << "\n";

      logFileName lfn(flist[n]);
   
      m_appToFileMap[lfn.appName()].insert(lfn);
   }

   return 0;
}

int logMap::getPriorLog( char * &logBefore,
                         const std::string & appName,
                         const flatlogs::eventCodeT & ev,
                         const flatlogs::timespecX & ts,
                         char * hint
                       )
{
   flatlogs::eventCodeT evL;
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   if(m_appToFileMap[appName].size() == 0)
   {
      std::cerr << __FILE__ << " " << __LINE__ << " getPriorLog empty map\n";
      return -1;
   }
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   logInMemory & lim = m_appToBufferMap[appName];
   
   flatlogs::timespecX et = lim.m_endTime;
   et.time_s += 30;
   if(lim.m_startTime > ts || et < ts)
   {
      #ifdef DEBUG
      std::cerr << __FILE__ << " " << __LINE__ << "\n";
      #endif

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
      std::cerr <<  __FILE__ << " " << __LINE__ << " Event code not found.\n";
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
            std::cerr << __FILE__ << " " << __LINE__ << " did not find following log for " << appName << " -- need to load more data.\n";
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
               std::cerr << __FILE__ << " " << __LINE__ << " did not find following log for " << appName << " -- need to load more data.\n";
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
}//getPriorLog

int logMap::getNextLog( char * &logAfter,            
                        char * logCurrent,           
                        const std::string & appName
                      )
{
   flatlogs::eventCodeT ev, evL;
   
   logInMemory & lim = m_appToBufferMap[appName];
   
   char* buffer;
   
   ev = logHeader::eventCode(logCurrent);
   
   buffer = logCurrent;
   
   buffer += logHeader::totalSize(buffer);
   if(buffer >= lim.m_memory.data() + lim.m_memory.size())
   {
      std::cerr << __FILE__ << " " << __LINE__ << " Reached end of data for " << appName << " -- need to load more data\n";
      //propoer action is to load the next file if possible.
      return 1;
   }
      
   evL = logHeader::eventCode(buffer);
   
   while(evL != ev)
   {
      buffer += logHeader::totalSize(buffer);
      if(buffer >= lim.m_memory.data() + lim.m_memory.size())
      {
         std::cerr << __FILE__ << " " << __LINE__ << " Reached end of data for " << appName << "-- need to load more data\n";
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

int logMap::loadFiles( const std::string & appName,
                       const flatlogs::timespecX & startTime
                     )
{
   if(m_appToFileMap[appName].size() == 0)
   {
      std::cerr << "*************************************\n\n";
      std::cerr << "No files for " << appName << "\n";
      std::cerr << "*************************************\n\n";
      return -1;
   }
 
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   //First check if already loaded files cover this time
   if(m_appToBufferMap[appName].m_memory.size() > 0)
   {
      if( m_appToBufferMap[appName].m_startTime <= startTime && m_appToBufferMap[appName].m_endTime >= startTime) 
      {
         std::cerr << "good!\n";
         return 0;
      }
      
      #ifdef DEBUG
      std::cerr << __FILE__ << " " << __LINE__ << "\n";
      #endif

      if( m_appToBufferMap[appName].m_startTime > startTime ) // Files don't go back far enough
      {
         #ifdef DEBUG
         std::cerr << __FILE__ << " " << __LINE__ << "\n";
         #endif

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
         std::cerr << "open later file for " << appName << "!\n";
         for(auto it=first; it != last; ++it)
         {
            m_appToBufferMap[appName].loadFile(*it);
         }
         return 0;
      }
      
   }
 
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   auto before = m_appToFileMap[appName].begin();
   
   for(; before != m_appToFileMap[appName].end(); ++before)
   {
      if( !(before->timestamp() < startTime) )
      {
         break;
      }
   }

   #ifdef debug
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   if(before == m_appToFileMap[appName].begin())
   {
      std::cerr << "No files in range for " << appName << "\n";
   }
   --before;
   
   #ifdef DEBUG 
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   m_appToBufferMap.emplace(std::pair<std::string, logInMemory>(appName, logInMemory()));
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   m_appToBufferMap[appName].loadFile(*before);
   if(++before != m_appToFileMap[appName].end()) 
   {
      m_appToBufferMap[appName].loadFile(*before);
   }
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   return 0;
}

} //namespace logger
} //namespace MagAOX
