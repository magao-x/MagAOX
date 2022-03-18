/** \file runCommand.hpp 
  * \brief Run a command get the output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup sys_files
  */

#ifndef sys_runCommand_hpp
#define sys_runCommand_hpp

#include <string>
#include <vector>

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
int runCommand( std::vector<std::string> & commandOutput, ///< [out] the output, line by line.  If an error, first entry contains the message.
                std::vector<std::string> & commandStderr, ///< [out] the output of stderr.
                std::vector<std::string> & commandList    ///< [in] command to be run, with one entry per command line word
              );

} //namespace sys 
} //namespace MagAOX 

#endif //sys_thSetuid_hpp
 
