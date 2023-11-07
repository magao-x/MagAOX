/** \file logdump.hpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  *
  * \ingroup logdump_files
  */

#ifndef logdump_hpp
#define logdump_hpp

#include <iostream>
#include <cstring>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
using namespace MagAOX::logger;

using namespace flatlogs;

/** \defgroup logdump logdump: MagAO-X Log Reader
  * \brief Read a MagAO-X binary log file.
  *
  * <a href="../handbook/utils/logdump.html">Utility Documentation</a>
  *
  * \ingroup utils
  *
  */

/** \defgroup logdump_files logdump Files
  * \ingroup logdump
  */

/// An application to dump MagAo-X binary logs to the terminal.
/** \todo document this
  * \todo add config for colors, both on/off and options to change.
  *
  * \ingroup logdump
  */
class logdump : public mx::app::application
{
protected:

   std::string m_dir;
   std::string m_ext;
   std::string m_file;

   bool m_time {false};
   bool m_jsonMode {false};

   unsigned long m_pauseTime {250}; ///When following, pause time to check for new data. msec. Default is 250 msec.
   int m_fileCheckInterval {4}; ///When following, number of loops to wait before checking for a new file.  Default is 4.

   std::vector<std::string> m_prefixes;

   size_t m_nfiles {0}; ///Number of files to dump.  Default is 0, unless following then the default is 1.

   bool m_follow {false};

   logPrioT m_level {logPrio::LOG_DEFAULT};

   std::vector<eventCodeT> m_codes;

   void printLogBuff( const logPrioT & lvl,
                      const eventCodeT & ec,
                      const msgLenT & len,
                      bufferPtrT & logBuff
                    );
   void printLogJson( const msgLenT & len,
                      bufferPtrT & logBuff
                    );

public:
   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();

   virtual int gettimes(std::vector<std::string> & logs);

};

void logdump::setupConfig()
{
   config.add("pauseTime","p", "pauseTime" , argType::Required, "", "pauseTime", false,  "int", "When following, time in milliseconds to pause before checking for new entries.");
   config.add("fileCheckInterval","", "fileCheckInterval" , argType::Required, "", "fileCheckInterval", false,  "int", "When following, number of pause intervals between checks for new files.");

   config.add("dir","d", "dir" , argType::Required, "", "dir", false,  "string", "Directory to search for logs. MagAO-X default is normally used.");
   config.add("ext","e", "ext" , argType::Required, "", "ext", false,  "string", "The file extension of log files.  MagAO-X default is normally used.");
   config.add("nfiles","n", "nfiles" , argType::Required, "", "nfiles", false,  "int", "Number of log files to dump.  If 0, then all matching files dumped.  Default: 0, 1 if following.");
   config.add("follow","f", "follow" , argType::True, "", "follow", false,  "bool", "Follow the log, printing new entries as they appear.");
   config.add("level","L", "level" , argType::Required, "", "level", false,  "int/string", "Minimum log level to dump, either an integer or a string. -1/TELEMETRY [the default], 0/DEFAULT, 1/D1/DBG1/DEBUG2, 2/D2/DBG2/DEBUG1,3/INFO,4/WARNING,5/ERROR,6/CRITICAL,7/FATAL.  Note that only the mininum unique string is required.");
   config.add("code","C", "code" , argType::Required, "", "code", false,  "int", "The event code, or vector of codes, to dump.  If not specified, all codes are dumped.  See logCodes.hpp for a complete list of codes.");
   config.add("file","F", "file" , argType::Required, "", "file", false,  "string", "A single file to process.  If no / are found in name it will look in the specified directory (or MagAO-X default).");
   config.add("time","T", "time" , argType::True, "", "time", false,  "bool", "time span mode: prints the ISO 8601 UTC timestamps of the first and last entry, the elapsed time in seconds, and the number of records in the file as a space-delimited string");
   config.add("json","J", "json" , argType::True, "", "json", false,  "bool", "JSON mode: emits one JSON document per line for each record in the log");


}

void logdump::loadConfig()
{
   config(m_pauseTime, "pauseTime");
   config(m_fileCheckInterval, "fileCheckInterval");

   //Get default log dir
   std::string tmpstr = mx::sys::getEnv(MAGAOX_env_path);
   if(tmpstr == "")
   {
      tmpstr = MAGAOX_path;
   }
   m_dir = tmpstr +  "/" + MAGAOX_logRelPath;;

   //Now check for config option for dir
   config(m_dir, "dir");

   m_ext = ".";
   m_ext += MAGAOX_default_logExt;
   config(m_ext, "ext");
   ///\todo need to check for lack of "." and error or fix

   config(m_file, "file");

   if(m_file == "" && config.nonOptions.size() < 1)
   {
      std::cerr << "logdump: need application name. Try logdump -h for help.\n";
   }

   if(m_file == "" && config.nonOptions.size() > 1)
   {
      std::cerr << "logdump: only one application at a time supported. Try logdump -h for help.\n";
   }

   m_prefixes.resize(config.nonOptions.size());
   for(size_t i=0;i<config.nonOptions.size(); ++i)
   {
      m_prefixes[i] = config.nonOptions[i];
   }

   if(config.isSet("time")) m_time = true;
   if(config.isSet("json")) m_jsonMode = true;

   config(m_follow, "follow");

   if(m_follow) m_nfiles = 1; //default to 1 if follow is set.
   config(m_nfiles, "nfiles");

   tmpstr = "";
   config(tmpstr, "level");
   if(tmpstr != "")
   {
      m_level = logLevelFromString(tmpstr);
   }

   config(m_codes, "code");

   std::cerr << m_codes.size() << "\n";
}

int logdump::execute()
{

   if(m_file == "" && m_prefixes.size() !=1 ) return -1; //error message will have been printed in loadConfig.

   std::vector<std::string> logs;

   if(m_file != "")
   {
      if(m_file.find('/') == std::string::npos)
      {
         m_file = m_dir + '/' + m_file;
      }
      std::cerr << "m_file: " << m_file << "\n";

      logs.push_back(m_file);
   }
   else
   {
      logs = mx::ioutils::getFileNames( m_dir, m_prefixes[0], "", m_ext);
   }

   ///\todo if follow is set, then should nfiles default to 1 unless explicitly set?
   if(m_nfiles == 0)
   {
      m_nfiles = logs.size();
   }

   if(m_nfiles > logs.size()) m_nfiles = logs.size();

   if(m_time)
   {
      return gettimes(logs);
   }

   bool firstRun = true; //for only showing latest entries on first run when following.
   
   for(size_t i=logs.size() - m_nfiles; i < logs.size(); ++i)
   {
      std::string fname = logs[i];
      FILE * fin;

      bufferPtrT head(new char[logHeader::maxHeadSize]);

      bufferPtrT logBuff;

      fin = fopen(fname.c_str(), "rb");

      //--> get size here!!
      off_t finSize = mx::ioutils::fileSize( fileno(fin) );
      
      std::cerr << fname << "\n";

      off_t totNrd = 0;
      
      size_t buffSz = 0;
      while(!feof(fin)) //<--This should be an exit condition controlled by loop logic, not feof.
      {
         int nrd;

         ///\todo check for errors on all reads . . .
         
         //Read next header
         nrd = fread( head.get(), sizeof(char), logHeader::minHeadSize, fin);
         if(nrd == 0)
         {
            //If we're following and on the last log file, wait for more to show up.
            if( m_follow == true  && i == logs.size()-1)
            {
               int check = 0;
               firstRun = false; //from now on we show all logs
               while(nrd == 0)
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
                     logs = mx::ioutils::getFileNames( m_dir, m_prefixes[0], "", m_ext);
                     if(logs.size() > oldsz)
                     {
                        //new file(s) detected;
                        break;
                     }
                     check = 0;
                  }
               }
            }
            else
            {
               break;
            }
         }

         //We got here without any data, probably means time to get a new file.
         if(nrd == 0) break;

         totNrd += nrd;
         
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


         if(m_codes.size() > 0)
         {
            bool found = false;
            for(size_t c = 0; c< m_codes.size(); ++c)
            {
               if( m_codes[c] == ec )
               {
                  found = true;
                  break;
               }
            }

            if(!found)
            {
               fseek(fin, len, SEEK_CUR);
               continue;
            }
         }

         size_t hSz = logHeader::headerSize(head);

         if( (size_t) hSz + (size_t) len > buffSz )
         {
            logBuff = bufferPtrT(new char[hSz + len]);
         }
         memcpy( logBuff.get(), head.get(), hSz);

         ///\todo what do we do if nrd not equal to expected size?
         nrd = fread( logBuff.get() + hSz, sizeof(char), len, fin);
         // If not following, exit loop without printing the incomplete log entry (go on to next file).cd
         // If following, wait for it, but also be checking for new log file in case of crash

         totNrd += nrd;
         
         if(m_follow && firstRun && finSize > 512 && totNrd < finSize-512) 
         {
            //firstRun = false;
            continue;
         }

         if (!logVerify(ec, logBuff, len))
         {
            std::cerr << "Log " << fname << " failed verification on code=" << ec <<  " at byte=" << totNrd-len-hSz <<". File possibly corrupt.  Exiting." << std::endl;
            return -1;
         }

         if (m_jsonMode) {
            printLogJson(len, logBuff);
         } else {
            printLogBuff(lvl, ec, len, logBuff);
         }

      }

      fclose(fin);
   }

   return 0;
}

inline
void logdump::printLogBuff( const logPrioT & lvl,
                            const eventCodeT & ec,
                            const msgLenT & len,
                            bufferPtrT & logBuff
                          )
{
   static_cast<void>(len); //be unused

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

   logStdFormat( std::cout, logBuff);

   std::cout << "\033[0m";
   std::cout << std::endl;
}


inline
void logdump::printLogJson( const msgLenT & len,
                            bufferPtrT & logBuff
                          )
{
   static_cast<void>(len); //be unused
   logJsonFormat(std::cout, logBuff);
   std::cout << std::endl;

}


int logdump::gettimes(std::vector<std::string> & logs)
{
   for(size_t i=logs.size() - m_nfiles; i < logs.size(); ++i)
   {
      std::string fname = logs[i];
      FILE * fin;

      bufferPtrT head(new char[logHeader::maxHeadSize]);

      fin = fopen(fname.c_str(), "rb");

      //--> get size here!!
      //off_t finSize = mx::ioutils::fileSize( fileno(fin) );
      
      
      off_t totNrd = 0;
      
      //size_t buffSz = 0;

      //Read firs header

      int nrd;

      ///\todo check for errors on all reads . . .
         
      //Read next header
      nrd = fread( head.get(), sizeof(char), logHeader::minHeadSize, fin);
      if(nrd == 0)
      {
         std::cerr << "got no header\n";
         return 0;
      }

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

      //logPrioT lvl = logHeader::logLevel(head);
      //eventCodeT ec = logHeader::eventCode(head);
      msgLenT len = logHeader::msgLen(head);
      timespecX ts0 = logHeader::timespec(head);
      //size_t hSz = logHeader::headerSize(head);

      uint32_t nRecords = 1;
      fseek(fin, len, SEEK_CUR);

      timespecX ts;

      while(!feof(fin)) //<--This should be an exit condition controlled by loop logic, not feof.
      {
         int nrd;

         //Read next header
         nrd = fread( head.get(), sizeof(char), logHeader::minHeadSize, fin);
         if(nrd == 0)
         {
            break;
         }
         nRecords += 1;

         //We got here without any data, probably means time to get a new file.
         if(nrd == 0) break;

         totNrd += nrd;
         
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

         //lvl = logHeader::logLevel(head);
         //ec = logHeader::eventCode(head);
         len = logHeader::msgLen(head);
         ts = logHeader::timespec(head);
         //hSz = logHeader::headerSize(head);
         
         fseek(fin, len, SEEK_CUR);


      }

      fclose(fin);

      double t0 = ts0.time_s + ts0.time_ns/1e9;
      double t = ts.time_s + ts.time_ns/1e9;

      std::cout << fname << " " << ts0.ISO8601DateTimeStrX() << "Z " << ts.ISO8601DateTimeStrX() << "Z " << t-t0 << " " << nRecords << std::endl;
   }

   return 0;
}

#endif //logdump_hpp
