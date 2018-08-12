/** \file sshDigger.hpp
  * \brief The MagAO-X SSH tunnel manager
  *
  * \ingroup sshDigger_files
  */

#ifndef sshDigger_hpp
#define sshDigger_hpp

#include <iostream>


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"



namespace MagAOX
{
namespace app
{
   
class sshDigger : public MagAOXApp
{

   //Give the test harness access.
   friend class sshDigger_test;

      
protected:

   std::vector<std::string> m_hostSpecs;
   
   std::vector<std::string> m_hostNames;
   std::vector<std::string> m_tunnelSpecs;
   std::vector<int> m_tunnelPIDs;
   
public:
   /// Default c'tor.
   sshDigger();

   /// D'tor, declared and defined for noexcept.
   ~sshDigger() noexcept
   {}
   
   virtual void setupConfig();

   virtual void loadConfig();

   int forkTunnel(size_t tunnelNo);
   
   /// Startup functions
   /** Setsup the INDI vars.
     * 
     */
   virtual int appStartup();

   /// Implementation of the FSM for sshDigger.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   

};

sshDigger::sshDigger() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void sshDigger::setupConfig()
{
   config.add("hosts", "H", "hosts", mx::argType::Required, "", "hosts", false,  "string vector", "comma separated list of remote hosts, in name[:remotePort]:localPort format");

   ///\todo add autossh config settings
}

void sshDigger::loadConfig()
{
   config(m_hostSpecs, "hosts");
}


int sshDigger::appStartup()
{

   for(size_t i=0; i < m_hostSpecs.size(); ++i)
   {
      netcom::tunneledHost th;
      
      int rv = th.parse( m_hostSpecs[i] );
      
      if(rv < 0)
      {
         log<software_fatal>({__FILE__, __LINE__, 0, "Error parsing host specification: " + m_hostSpecs[i]});         
         return -1;
      }
      
      std::string ts = mx::ioutils::convertToString(th.localPort()) + ":" + th.name() + ":" 
                                 + mx::ioutils::convertToString(th.remotePort());
      m_tunnelSpecs.push_back(ts);
      m_hostNames.push_back(th.name());
   }
   
   m_hostSpecs.clear();
   
   m_tunnelPIDs.resize(m_tunnelSpecs.size(), 0);
    
   for(size_t i=0; i<m_tunnelSpecs.size(); ++i)
   {
      if(forkTunnel(i) < 0)
      {
         log<software_trace_fatal>({__FILE__,__LINE__});
         return -1;
      }
   }
   
   return 0;
}

int sshDigger::forkTunnel( size_t tunnelNo )
{
   
   
   std::vector<std::string> argsV = {"autossh", "-M0", "-nNTL", m_tunnelSpecs[tunnelNo], m_hostNames[tunnelNo]};
   
   if(m_log.logLevel() <= logLevels::INFO)
   {
      std::string coml = "Starting autossh with command: ";
      for(size_t i=0;i<argsV.size();++i)
      {
         coml += argsV[i];
         coml += " ";
      }
      log<text_log>(coml);
   }
      
   m_tunnelPIDs[tunnelNo] = fork();
   
   if(m_tunnelPIDs[tunnelNo] < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, std::string("fork failed: ") + strerror(errno)});
      return -1;
   }
   
   if(m_tunnelPIDs[tunnelNo] == 0)
   {
      const char ** args = new const char*[argsV.size() + 1];
      for(size_t i=0; i< argsV.size();++i) args[i] = argsV[i].data();
      args[argsV.size()] = NULL;
   
      ///\todo need to set environment vars and capture autossh logs...
      
      execvp("autossh", (char * const*) args);

      log<software_error>({__FILE__, __LINE__, errno, std::string("execvp returned: ") + strerror(errno)});
      
      delete[] args;
      
      return -1;
   }
   
   return 0;
}

int sshDigger::appLogic()
{
   state(stateCodes::CONNECTED);
   
   ///\todo sshDigger: Here need to cycle through each PID and make sure it is still alive, restart if needed.
   
   return 0;
}

int sshDigger::appShutdown()
{
   for(size_t i=0; i<m_tunnelPIDs.size(); ++i)
   {
      kill(m_tunnelPIDs[i], SIGTERM);
   }
   
   return 0;
}

} //namespace app 
} //namespace MagAOX

#endif //sshDigger_hpp
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
