/** \file runCommand.cpp 
  * \brief Run a command get the output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup sys_files
  */

#include "runCommand.hpp"

#include <cstring>
#include <sstream>

#include <unistd.h>
#include <sys/wait.h>
#include <iostream>

namespace MagAOX 
{
namespace sys 
{

int runCommand( std::vector<std::string> & commandOutput, // [out] the output, line by line.  If an error, first entry contains the message.
                std::vector<std::string> & commandStderr, // [out] the output of stderr.
                std::vector<std::string> & commandList    // [in] command to be run, with one entry per command line word
              )
{
   int link[2];
   int errlink[2];
   
   pid_t pid;
   
   if (pipe(link)==-1) 
   {
      commandOutput.push_back(std::string("Pipe error stdout: ") + strerror(errno));
      return -1;
   }

   if (pipe(errlink)==-1) 
   {
      commandOutput.push_back(std::string("Pipe error stderr: ") + strerror(errno));
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
      
      dup2 (errlink[1], STDERR_FILENO);
      close(errlink[0]);
      close(errlink[1]);
      
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
      close(errlink[1]);
      
      int rd;
      if ( (rd = read(link[0], commandOutput_c, sizeof(commandOutput_c))) < 0) 
      {
         commandOutput.push_back(std::string("Read error: ") + strerror(errno));  
         close(link[0]);
         return -1;
      }
      close(link[0]);
      
      std::string line;
      
      commandOutput_c[rd] = '\0';
      std::string commandOutputString(commandOutput_c);
      
      std::istringstream iss(commandOutputString);
      
      while (getline(iss, line)) 
      {
         commandOutput.push_back(line);
      }
      
      //----stderr
      if ( (rd = read(errlink[0], commandOutput_c, sizeof(commandOutput_c))) < 0) 
      {
         commandStderr.push_back(std::string("Read error on stderr: ") + strerror(errno));  
         close(errlink[0]);
         return -1;
      }
      close(errlink[0]);
      
      commandOutput_c[rd] = '\0';
      commandOutputString = commandOutput_c;
      
      std::istringstream iss2(commandOutputString);
      
      while (getline(iss2, line)) 
      {
         commandStderr.push_back(line);
      }
      
      wait(NULL);
      return 0;
   }
}
   

  


} //namespace sys 
} //namespace MagAOX 

 
