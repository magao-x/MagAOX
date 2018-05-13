/** \file logdump.cpp
  * \brief A simple utility to dump MagAO-X binary logs to stdout.
  */

#include "logdump.hpp"

// argv[1] (required) = prefix of logs
// argv[2] (optional) = number of logs.  1 just shows latest, 2 last two, etc.

/** \todo document this
  * \todo make this an application with robust config 
  */ 
int main(int argc, char **argv)
{
   std::string dir;
   std::string prefix;
   int n;
   
   dir = "./";
   
   if(argc < 2) 
   {
      std::cerr << "Need application name\n";
      return -1;
   }
   
   prefix = argv[1];
   
   std::vector<std::string> logs = mx::getFileNames( dir, prefix, "", ".binlog");
   
   if(logs.size() == 0)
   {
      std::cerr << "No logs found.\n";
      return -1;
   }
   
   
   if(argc < 3)
   {
      n = logs.size();
   }
   else
   {
      n = atoi(argv[2]);
   }
   
   if(n < 0) n = 0;
   if(n > logs.size()) n = logs.size();
   
   using namespace MagAOX::logger;
   
   for(int i=logs.size() - n; i < logs.size(); ++i)
   {
      
      std::string fname = logs[i]; 
      FILE * fin;
   
      bufferPtrT head(new char[headerSize]);
   
      bufferPtrT logBuff;
   
      fin = fopen(fname.c_str(), "r");
      
      size_t buffSz = 0;
      while(!feof(fin))
      {
         int nrd;
   
         nrd = fread( head.get(), sizeof(char), headerSize, fin);
         if(nrd == 0) break;

         logLevelT lvl = logLevel(head);
         eventCodeT ec = eventCode(head);
         msgLenT len = msgLen(head);
         
         if( headerSize + len > buffSz )
         {
            logBuff = bufferPtrT(new char[headerSize + len]);
         }
      
         memcpy( logBuff.get(), head.get(), headerSize);
      
         nrd = fread( logBuff.get() + headerSize, sizeof(char), len, fin);

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
   
      fclose(fin);
   }
   
   return 0;
}

