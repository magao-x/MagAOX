/** \file runCommand.hpp 
  * \brief Run a command get the output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup sys_files
  * History:
  * - 2020-09-11 created by JRM
  */

#ifndef sys_runCommand_hpp
#define sys_runCommand_hpp

#include <unistd.h>
#include <sys/wait.h>

namespace MagAOX 
{
namespace sys 
{
   
/// Runs a command (with parameters) passed in using fork/exec
/** New process is made with fork(), and child runs execvp with command provided.
  * 
  * Original code by C. Bohlman for sysMonitor, then promoted to libMagAOX for general use.
  * 
  * \returns 0 on success
  * \returns -1 on error
  * 
  * \ingroup sys
  */
inline
int runCommand( std::vector<std::string> & commandOutput, /// [out] the output, line by line.  If an error, first entry contains the message.
                std::vector<std::string> & commandList    ///< [in] command to be run, with one entry per command line word
              )
{
   int link[2];
   pid_t pid;
   
   if (pipe(link)==-1) 
   {
      commandOutput.push_back(std::string("Pipe error: ") + strerror(errno));
      return -1;
   }

   if ((pid = fork()) == -1) 
   {
      commandOutput.push_back(std::string("Fork error: ") + strerror(errno));
      return -1;
   }

   if(pid == 0) 
   {
      dup2 (link[1], STDOUT_FILENO);
      close(link[0]);
      close(link[1]);
      std::vector<const char *>charCommandList( commandList.size()+1, NULL);
      for(int index = 0; index < (int) commandList.size(); ++index)
      {
         charCommandList[index]=commandList[index].c_str();
      }
      execvp( charCommandList[0], const_cast<char**>(charCommandList.data()));
      commandOutput.push_back(std::string("execvp returned: ") + strerror(errno));
      return -1;
   }
   else 
   {
      char commandOutput_c[4096];
         
      wait(NULL);
      close(link[1]);
      
      int rd;
      if ( (rd = read(link[0], commandOutput_c, sizeof(commandOutput_c))) < 0) 
      {
         commandOutput.push_back(std::string("Read error: ") + strerror(errno));  
         close(link[0]);
         return -1;
      }
      close(link[0]);
      
      std::string line{};
      
      commandOutput_c[rd] = '\0';
      std::string commandOutputString(commandOutput_c);
      
      std::istringstream iss(commandOutputString);
      
      while (getline(iss, line)) 
      {
         commandOutput.push_back(line);
      }
      wait(NULL);
      return 0;
   }
}
   

  


} //namespace sys 
} //namespace MagAOX 

#endif //sys_thSetuid_hpp
 
