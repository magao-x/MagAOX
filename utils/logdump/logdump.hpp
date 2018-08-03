/** \file logdump.hpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  */

#ifndef logdump_hpp
#define logdump_hpp

#include <iostream>
#include <cstring>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
using namespace MagAOX::logger;

class logdump : public mx::application
{
protected:

   std::string m_dir;
   std::string m_ext;

   unsigned long m_pauseTime {250}; ///When following, pause time to check for new data. msec. Default is 250 msec.
   int m_fileCheckInterval {4}; ///When following, number of loops to wait before checking for a new file.  Default is 4.

   std::vector<std::string> m_prefixes;

   int m_nfiles {0}; ///Number of files to dump.  Default is 0, unless following then the default is 1.

   bool m_follow {false};

   logLevelT m_level {logLevels::TELEMETRY};


   void printLogBuff( const logLevelT & lvl,
                      const eventCodeT & ec,
                      const msgLenT & len,
                      bufferPtrT & logBuff
                    );

public:
   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();

};

void logdump::setupConfig()
{
   config.add("pauseTime","p", "pauseTime" , mx::argType::Required, "", "pauseTime", false,  "int", "When following, time in milliseconds to pause before checking for new entries.");
   config.add("fileCheckInterval","F", "fileCheckInterval" , mx::argType::Required, "", "fileCheckInterval", false,  "int", "When following, number of pause intervals between checks for new files.");

   config.add("dir","d", "dir" , mx::argType::Required, "", "dir", false,  "string", "Directory to search for logs. MagAO-X default is normally used.");
   config.add("ext","e", "ext" , mx::argType::Required, "", "ext", false,  "string", "The file extension of log files.  MagAO-X default is normally used.");
   config.add("nfiles","n", "nfiles" , mx::argType::Required, "", "nfiles", false,  "int", "Number of log files to dump.  If 0, then all matching files dumped.  Default: 0, 1 if following.");
   config.add("follow","f", "follow" , mx::argType::True, "", "follow", false,  "bool", "Follow the log, printing new entries as they appear.");
   config.add("level","l", "level" , mx::argType::Required, "", "level", false,  "int/string", "Minimum log level to dump, either an integer or a string. -1/TELEMETRY [the default], 0/DEFAULT, 1/D1/DBG1/DEBUG2, 2/D2/DBG2/DEBUG1,3/INFO,4/WARNING,5/ERROR,6/CRITICAL,7/FATAL.  Note that only the mininum unique string is required.");

}

void logdump::loadConfig()
{
   config(m_pauseTime, "pauseTime");
   config(m_fileCheckInterval, "fileCheckInterval");

   //Get default log dir
   std::string tmpstr = mx::getEnv(MAGAOX_env_path);
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



   if(config.nonOptions.size() < 1)
   {
      std::cerr << "logdump: need application name. Try logdump -h for help.\n";
   }

   if(config.nonOptions.size() > 1)
   {
      std::cerr << "logdump: only one application at a time supported. Try logdump -h for help.\n";
   }

   m_prefixes.resize(config.nonOptions.size());
   for(int i=0;i<config.nonOptions.size(); ++i)
   {
      m_prefixes[i] = config.nonOptions[i];
   }

   config(m_follow, "follow");

   if(m_follow) m_nfiles = 1; //default to 1 if follow is set.
   config(m_nfiles, "nfiles");

   tmpstr = "";
   config(tmpstr, "level");
   if(tmpstr != "")
   {
      m_level = logLevelFromString(tmpstr);
   }
}

int logdump::execute()
{

   if(m_prefixes.size() !=1 ) return -1; //error message will have been printed in loadConfig.


   std::vector<std::string> logs = mx::ioutils::getFileNames( m_dir, m_prefixes[0], "", m_ext);

   ///\todo if follow is set, then should nfiles default to 1 unless explicitly set?
   if(m_nfiles <= 0)
   {
      m_nfiles = logs.size();
   }

   if(m_nfiles > logs.size()) m_nfiles = logs.size();

   for(int i=logs.size() - m_nfiles; i < logs.size(); ++i)
   {
      std::string fname = logs[i];
      FILE * fin;

      bufferPtrT head(new char[headerSize]);

      bufferPtrT logBuff;

      fin = fopen(fname.c_str(), "rb");

      size_t buffSz = 0;
      while(!feof(fin)) //<--This should be an exit condition controlled by loop logic, not feof.
      {
         int nrd;

         nrd = fread( head.get(), sizeof(char), headerSize, fin);
         if(nrd == 0)
         {
            //If we're following and on the last log file, wait for more to show up.
            if( m_follow == true  && i == logs.size()-1)
            {
               int check = 0;
               while(nrd == 0)
               {
                  std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::milli>(m_pauseTime));
                  clearerr(fin);
                  nrd = fread( head.get(), sizeof(char), headerSize, fin);
                  if(nrd > 0) break;

                  ++check;
                  if(check >= m_fileCheckInterval)
                  {
                     //Check if a new file exists now.
                     size_t oldsz = logs.size();
                     logs = mx::ioutils::getFileNames( m_dir, m_prefixes[0], "", m_ext);
                     if(logs.size() > oldsz) break;

                     check = 0;
                  }
               }
            }
            else
            {
               break;
            }
         }

         logLevelT lvl = logLevel(head);
         eventCodeT ec = eventCode(head);
         msgLenT len = msgLen(head);

         //Here: check if lvl, eventCode, etc, match what we want.
         //If not, fseek and loop.
         if(lvl < m_level)
         {
            fseek(fin, len, SEEK_CUR);
            continue;
         }

         if( headerSize + len > buffSz )
         {
            logBuff = bufferPtrT(new char[headerSize + len]);
         }

         memcpy( logBuff.get(), head.get(), headerSize);

         ///\todo what do we do if nrd not equal to expected size?
         nrd = fread( logBuff.get() + headerSize, sizeof(char), len, fin);
         // If not following, exit loop without printing the incomplete log entry (go on to next file).
         // If following, wait for it, but also be checking for new log file in case of crash

         /*if(lvl < m_level)
         {
            continue;
         }*/

         printLogBuff(lvl, ec, len, logBuff);


      }

      fclose(fin);
   }

   return 0;
}

inline
void logdump::printLogBuff( const logLevelT & lvl,
                            const eventCodeT & ec,
                            const msgLenT & len,
                            bufferPtrT & logBuff
                          )
{
   if(ec == eventCodes::GIT_STATE)
   {
      typename git_state::messageT msg;
      git_state::extract(msg, logBuff.get()+messageOffset, len);

      if(msg.m_repoName == "MagAOX")
      {
         for(int i=0;i<80;++i) std::cout << '-';
         std::cout << "\n\t\t\t\t SOFTWARE RESTART\n";
         for(int i=0;i<80;++i) std::cout << '-';
         std::cout << '\n';
      }
   }

   if(lvl > logLevels::INFO)
   {
      std::cout << "\033[";

      if(lvl == logLevels::WARNING) std::cout << "33";
      else std::cout << "31";
      std::cout << "m";
   }

   logStdFormat(logBuff);

   if(lvl > logLevels::INFO)
   {
      std::cout << "\033[0m";
   }
}

#endif //logdump_hpp
