/** \file logstream.hpp
  * \brief A simple utility to stream MagAO-X binary logs to stdout.
  */

#ifndef logstream_hpp
#define logstream_hpp

#include <iostream>
#include <string>
#include <set>
#include <map>

#include <mx/ioutils/fileUtils.hpp>

//#include "../../libMagAOX/libMagAOX.hpp"
//using namespace MagAOX::logger;

//using namespace flatlogs;


class logstream //: public mx::app::application
{

public:
   
   std::string m_dir {"/opt/MagAOX/logs/"};
   std::string m_ext {".binlog"};
      
   unsigned long m_pauseTime {250};
   int m_fileCheckInterval {4}; ///When following, number of loops to wait before checking for a new file.  Default is 4.
   
   logPrioT m_level {logPrio::LOG_DEFAULT};
   
   double m_startTime {0};
   
   bool m_shutdown {false};
   
   ///Mutex for locking stream access
   std::mutex m_streamMutex;
   
   struct s_logThread
   {
      std::string m_appName;
      
      std::shared_ptr<std::thread>  m_thread; ///< Thread for monitoring a single log 
      
      logstream * m_lstr {nullptr};            ///< a pointer to a logstream instance (normally this)
      
      ///C'tor to create the thread object
      s_logThread() : m_thread {std::shared_ptr<std::thread>(new std::thread)}
      {
      }      
      
      s_logThread( const s_logThread & cplt ) : m_appName{cplt.m_appName}, m_thread {cplt.m_thread}, m_lstr {cplt.m_lstr}
      {
      }

   };
   
   std::vector<s_logThread> m_logThreads; 
   
   
   struct s_logEntry
   {
      std::string m_appName;
      
      bufferPtrT logBuff;
      
      explicit s_logEntry( const std::string & appName ) : m_appName{appName}
      {
      }
      
   };
   
   std::multimap<double, s_logEntry> m_logStream;
   
public: 
   
   logstream();
   
   int getAppsWithLogs( std::set<std::string> & appNames );
   
   void printLogBuff( const std::string & appName,
                      bufferPtrT & logBuff
                    );
   
   private:
   
   ///Log thread starter, called by logThreadStart on thread construction.  Calls logThreadExec.
   static void internal_logThreadStart( s_logThread* lt /**< [in] a pointer to an s_logThread structure */);

public:
   /// Start the log thread.
   int logThreadStart( size_t thno /**< [in] the thread to start */);

   /// Execute the log thread.
   void logThreadExec( const std::string & appName /**< [in] the application name to monitor */ );
};

inline 
logstream::logstream()
{
   m_startTime = mx::get_curr_time();
}

inline
int logstream::getAppsWithLogs( std::set<std::string> & appNames )
{
   std::vector<std::string> allfiles = mx::ioutils::getFileNames( m_dir, m_ext);
   
   std::cerr << "Found " << allfiles.size() << " files\n";
   
   for(size_t i=0; i< allfiles.size(); ++i)
   {
      std::string fullPath = allfiles[i].substr(0, allfiles[i].size()-31);
      size_t spos = fullPath.rfind('/');
      if(spos == std::string::npos) spos = 0;
      else ++spos;
      
      std::string appName = fullPath.substr(spos);
      
      appNames.insert(appName);
      
   }
   
   m_logThreads.resize( appNames.size() );
   
   size_t n = 0;
   for(auto it = appNames.begin(); it != appNames.end();it++)
   {
      //std::cerr << *it << "\n";
      m_logThreads[n].m_appName = *it;
      m_logThreads[n].m_lstr = this;
      
      logThreadStart(n);
      
      ++n;
   }
   
   //m_shutdown = true;
   
   while(!m_shutdown)
   {
      if( m_logStream.size() > 0)
      {
         auto it=m_logStream.begin();
         
         while(it != m_logStream.end())
         {
            printLogBuff(it->second.m_appName, it->second.logBuff);
            
            m_logStream.erase(it);
            it = m_logStream.begin();
         }
      }
      else
      {
         std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::milli>(m_pauseTime));
      }
   }
   
   return 0;
}  

   
inline
void logstream::printLogBuff( const std::string & appName, 
                              bufferPtrT & logBuff
                            )
{
   
   logPrioT lvl = logHeader::logLevel( logBuff);
   eventCodeT ec = logHeader::eventCode( logBuff);

   if(ec == eventCodes::GIT_STATE)
   {
      if(git_state::repoName(logHeader::messageBuffer(logBuff)) == "MagAOX")
      {
         for(int i=0;i<80;++i) std::cout << '-';
         std::cout << "\n\t\t\t\t SOFTWARE RESTART\n";
         for(int i=0;i<80;++i) std::cout << '-';
         std::cout << '\n';
      }            
   }

   if(lvl < logPrio::LOG_INFO)
   {
      if(lvl == logPrio::LOG_EMERGENCY)
      {
         std::cout << "\033[104m\033[91m\033[5m\033[1m";
      }

      if(lvl == logPrio::LOG_ALERT)
      {
         std::cout << "\033[101m\033[5m";
      }

      if(lvl == logPrio::LOG_CRITICAL)
      {
         std::cout << "\033[41m\033[1m";
      }

      if(lvl == logPrio::LOG_ERROR)
      {
         std::cout << "\033[91m\033[1m";
      }

      if(lvl == logPrio::LOG_WARNING)
      {
         std::cout << "\033[93m\033[1m";
      }

      if(lvl == logPrio::LOG_NOTICE)
      {
         std::cout << "\033[1m";
      }

   }

   //std::cout << appName << " ";
   
   logShortStdFormat( std::cout, appName, logBuff);

   std::cout << "\033[0m";
   std::cout << "\n";
}

void logstream::internal_logThreadStart( s_logThread* lt )
{
   lt->m_lstr->logThreadExec( lt->m_appName );
}

int logstream::logThreadStart( size_t thno )
{
   try
   {
      *m_logThreads[thno].m_thread = std::thread( internal_logThreadStart, &m_logThreads[thno]);
   }
   catch( const std::exception & e )
   {
      //log<software_error>({__FILE__, __LINE__, std::string("exception in log thread startup: ") +e.what()});
      return -1;
   }
   catch( ... )
   {
      //log<software_error>({__FILE__, __LINE__, "unknown exception in log thread startup"});
      return -1;
   }
   
   if(!m_logThreads[thno].m_thread->joinable())
   {
      //log<sofware_error>({__FILE__, __LINE__, "log thread did not start"});
      return -1;
   }
   
   
   return 0;
}

void logstream::logThreadExec( const std::string & appName )
{
   while(!m_shutdown)
   {
      
      std::vector<std::string> logs = mx::ioutils::getFileNames( m_dir, appName, "", m_ext);

      std::string fname = logs[logs.size()-1];

      FILE * fin;

      bufferPtrT head(new char[logHeader::maxHeadSize]);

      bufferPtrT logBuff;

      fin = fopen(fname.c_str(), "rb");

      size_t buffSz = 0;
      while(!feof(fin) && !m_shutdown) //<--This should be an exit condition controlled by loop logic, not feof.
      {
         int nrd;

         ///\todo check for errors on all reads . . .
         nrd = fread( head.get(), sizeof(char), logHeader::minHeadSize, fin);
         if(nrd == 0)
         {
            int check = 0;
            while(nrd == 0 && !m_shutdown)
            {
               std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::milli>(m_pauseTime));
               clearerr(fin);
               nrd = fread( head.get(), sizeof(char), logHeader::minHeadSize, fin);
               if(nrd > 0) break;

               ++check;
               if(check >= m_fileCheckInterval)
               {
                  //Check if a new file exists now.
                  size_t oldsz = logs.size();
                  logs = mx::ioutils::getFileNames( m_dir, appName, "", m_ext);
                  if(logs.size() > oldsz)
                  {
                     //new file(s) detected;
                     break;
                  }
                  check = 0;
               }
            }
            
            if(m_shutdown) break;
         
         }

         //We got here without any data, probably means time to get a new file.
         if(nrd == 0) break;


         if( logHeader::msgLen0(head) == logHeader::MAX_LEN0-1)
         {
            //Intermediate size message, read two more bytes
            nrd = fread( head.get() + logHeader::minHeadSize, sizeof(char), sizeof(msgLen1T), fin);
         }
         else if( logHeader::msgLen0(head) == logHeader::MAX_LEN0)
         {
            //Large size message: read 8 more bytes
            nrd = fread( head.get() + logHeader::minHeadSize, sizeof(char), sizeof(msgLen2T), fin);
         }


         logPrioT lvl = logHeader::logLevel(head);
         eventCodeT ec = logHeader::eventCode(head);
         msgLenT len = logHeader::msgLen(head);

         //Here: check if lvl, eventCode, etc, match what we want.
         //If not, fseek and loop.
         if(lvl > m_level)
         {
            fseek(fin, len, SEEK_CUR);
            continue;
         }

         
         size_t hSz = logHeader::headerSize(head);

         if( (size_t) hSz + (size_t) len > buffSz )
         {
            logBuff = bufferPtrT(new char[hSz + len]);
         }

         memcpy( logBuff.get(), head.get(), hSz);

         ///\todo what do we do if nrd not equal to expected size?
         nrd = fread( logBuff.get() + hSz, sizeof(char), len, fin);
         // If not following, exit loop without printing the incomplete log entry (go on to next file).
         // If following, wait for it, but also be checking for new log file in case of crash

         //printLogBuff(lvl, ec, len, logBuff);

         timespecX ts = logHeader::timespec(logBuff);
         double dts = ((double) ts.time_s) + ((double) ts.time_ns)/1e9;
         
         if(m_startTime - dts > 10.0) continue;
         
         {
            std::unique_lock<std::mutex> lock(m_streamMutex);
            auto it = m_logStream.insert( std::pair<double,s_logEntry>(dts, s_logEntry(appName)));
         
            it->second.logBuff = logBuff;
         }
      }

      fclose(fin);
   }

}


#endif
