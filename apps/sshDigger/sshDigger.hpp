/** \file sshDigger.hpp
  * \brief The MagAO-X SSH tunnel manager
  *
  * \ingroup sshDigger_files
  */

#ifndef sshDigger_hpp
#define sshDigger_hpp

#include <sys/wait.h>

#include <iostream>

#include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


//Return codes, these are primarily for testing purposes
#define SSHDIGGER_E_NOTUNNELS (-10)
#define SSHDIGGER_E_NOTUNNELFOUND (-11)
#define SSHDIGGER_E_NOHOSTNAME (-12)
#define SSHDIGGER_E_NOLOCALPORT (-13)
#define SSHDIGGER_E_NOREMOTEPORT (-14)

namespace MagAOX
{
namespace app
{
   
/// The MagAO-X SSH tunnel manager
/** Each instance of this app manages one SSH tunnel to another computer.
  *
  * \todo add options for verboseness of ssh and autossh (loglevel)
  * \todo add options for ssh and autossh log thread priorities
  */ 
class sshDigger : public MagAOXApp<false>
{

   //Give the test harness access.
   friend class sshDigger_test;

protected:

   /** \name Configurable Parameters
     *@{
     */ 
   std::string m_remoteHost; ///< The name of the remote host
   int m_localPort {0}; ///< The local port to forward from
   int m_remotePort {0}; ///< The remote port to forward to
   
   ///@}
   
   int m_tunnelPID; ///< The PID of the autossh process
   
   int m_sshSTDERR {-1}; ///< The output of stderr of the ssh process
   int m_sshSTDERR_input {-1}; ///< The input end of stderr, used to wake up the log thread on shutdown.
   
   int m_sshLogThreadPrio {0}; ///< Priority of the ssh log capture thread, should normally be 0.
   
   std::thread m_sshLogThread; ///< A separate thread for capturing ssh logs
   
   std::string m_autosshLogFile; ///< Name of the autossh logfile.
   int m_autosshLogFD; ///< File descriptor of the autossh logfile.
   
   int m_autosshLogThreadPrio {0}; ///< Priority of the autossh log capture thread, should normally be 0.
   
   std::thread m_autosshLogThread; ///< A separate thread for capturing autossh logs
   
   
   bool m_sshError {false}; ///< Flag to signal when ssh logs an error, and should be restarted via SIGUSR1 to autossh.
   
public:
   /// Default c'tor.
   sshDigger();

   /// D'tor, declared and defined for noexcept.
   ~sshDigger() noexcept
   {}
   
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);
   
   virtual void loadConfig();

   /// Create the tunnel specification string, [localPort]:localhost:[remotePort].
   /**
     * \returns a string containing the tunnel specification.
     */  
   std::string tunnelSpec();
   
   /// Generate the argv vector for the exec of autossh.
   void genArgsV( std::vector<std::string> & argsV /**< [out] will contain the argv vector for an autssh exec call */);
   
   /// Generate the envp vector for the exec of autossh.
   void genEnvp( std::vector<std::string> & envp /**< [out] will contain the envp vector for an autssh exec call */);
   
   ///  Creates the tunnel in a child process using exec.
   int execTunnel();
   
   ///Thread starter, called by sshLogThreadStart on thread construction.  Calls sshLogThreadExec.
   static void _sshLogThreadStart( sshDigger * s /**< [in] a pointer to an sshDigger instance (normally this) */);

   /// Start the log capture.
   int sshLogThreadStart();

   /// Execute the log capture.
   void sshLogThreadExec();
   
   /// Process a log entry from indiserver, putting it into MagAO-X standard form 
   int processSSHLog( std::string logs );
   
   ///Thread starter, called by sshLogThreadStart on thread construction.  Calls sshLogThreadExec.
   static void _autosshLogThreadStart( sshDigger * s /**< [in] a pointer to an sshDigger instance (normally this) */);

   /// Start the log capture.
   int autosshLogThreadStart();

   /// Execute the log capture.
   void autosshLogThreadExec();
   
   /// Process a log entry from indiserver, putting it into MagAO-X standard form 
   int processAutoSSHLog( std::string logs );
   
   /// Startup functions
   /** 
     * 
     */
   virtual int appStartup();

   /// Implementation of the FSM for sshDigger.
   virtual int appLogic();

   /// SIGTERM the tunnel process.
   virtual int appShutdown();
   

};

sshDigger::sshDigger() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   //Use the sshTunnels.conf config file
   m_configBase = "sshTunnels";
   
   //set mx::app::application flag to not report lack of config file for this app.
   m_requireConfigPathLocal = false;
   
   return;
}

void sshDigger::setupConfig()
{
}

int sshDigger::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   std::vector<std::string> sections;

   _config.unusedSections(sections);

   if( sections.size() == 0 ) 
   {
      log<text_log>("No tunnels found in config.", logPrio::LOG_CRITICAL);
      
      return SSHDIGGER_E_NOTUNNELS;
   }

   //Now see if any sections match our m_configName

   bool found =false;
   for(size_t i=0;i<sections.size(); ++i)
   {
      if( sections[i] == m_configName )
      {
         found = true;
      }
   }
   
   if( found == false ) 
   {
      log<text_log>("No matching tunnel configuration found.", logPrio::LOG_CRITICAL);      
      return SSHDIGGER_E_NOTUNNELFOUND;
   }
  
   //Now we configure the tunnel.

   _config.configUnused( m_remoteHost, mx::app::iniFile::makeKey(m_configName, "remoteHost" ) );
   if( m_remoteHost == "" ) 
   {
      log<text_log>("No remote host specified.", logPrio::LOG_CRITICAL);
      return SSHDIGGER_E_NOHOSTNAME;
   }
   
   _config.configUnused( m_localPort, mx::app::iniFile::makeKey(m_configName, "localPort" ) );
   if( m_localPort == 0 ) 
   {
      log<text_log>("No local port specified.", logPrio::LOG_CRITICAL);
      
      return SSHDIGGER_E_NOLOCALPORT;
   }
   
   _config.configUnused( m_remotePort, mx::app::iniFile::makeKey(m_configName, "remotePort" ) );
   if( m_remotePort == 0 ) 
   {
      log<text_log>("No remote port specified.", logPrio::LOG_CRITICAL);
      
      return SSHDIGGER_E_NOREMOTEPORT;
   }
   
   return 0;
}

void sshDigger::loadConfig()
{
   loadConfigImpl(config);
}

std::string sshDigger::tunnelSpec()
{
   std::string ts = mx::ioutils::convertToString(m_localPort) + ":localhost:" + mx::ioutils::convertToString(m_remotePort);
   
   return ts;
}

void sshDigger::genArgsV( std::vector<std::string> & argsV )
{
   ///\todo make monitor port a config variable.
   argsV = {"autossh", "-M0", "-nNTL", tunnelSpec(), m_remoteHost};   
}

void sshDigger::genEnvp( std::vector<std::string> & envp )
{
   std::string logenv="AUTOSSH_LOGFILE=" + m_autosshLogFile;
   
   envp = {logenv};   
}

int sshDigger::execTunnel()
{
   std::vector<std::string> argsV;
   genArgsV(argsV);
   
   std::vector<std::string> envps;
   genEnvp(envps);
   
   if(m_log.logLevel() <= logPrio::LOG_INFO)
   {
      std::string coml = "Starting autossh with command: ";
      for(size_t i=0;i<argsV.size();++i)
      {
         coml += argsV[i];
         coml += " ";
      }
      log<text_log>(coml);
   }
      
   int filedes[2];
   if (pipe(filedes) == -1) 
   {
      log<software_error>({__FILE__, __LINE__, errno});
      return -1;
   }
   
   m_tunnelPID  = fork();
   
   if(m_tunnelPID < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, std::string("fork failed: ") + strerror(errno)});
      return -1;
   }
   
   if(m_tunnelPID == 0)
   {
      //Route STDERR of child to pipe input.
      while ((dup2(filedes[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}
      close(filedes[1]);
      close(filedes[0]);
      
      const char ** args = new const char*[argsV.size() + 1];
      for(size_t i=0; i< argsV.size();++i) args[i] = argsV[i].data();
      args[argsV.size()] = NULL;
   
      
      ///\todo need to set environment vars and capture autossh logs...
      const char ** envp = new const char*[envps.size() + 1];
      for(size_t i=0; i< envps.size();++i) envp[i] = envps[i].data();
      envp[envps.size()] = NULL;
      
      execvpe("autossh", (char * const*) args, (char * const*) envp);

      log<software_error>({__FILE__, __LINE__, errno, std::string("execvp returned: ") + strerror(errno)});
      
      delete[] args;
      
      return -1;
   }
   
   
   
   m_sshSTDERR = filedes[0];
   m_sshSTDERR_input = filedes[1];
   
   if(m_log.logLevel() <= logPrio::LOG_INFO)
   {
      std::string coml = "autossh tunnel started with PID " + mx::ioutils::convertToString(m_tunnelPID);   
      log<text_log>(coml);
   }
   return 0;
}

inline
void sshDigger::_sshLogThreadStart( sshDigger * s)
{
   s->sshLogThreadExec();
}

inline
int sshDigger::sshLogThreadStart()
{
   try
   {
      m_sshLogThread  = std::thread( _sshLogThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on ssh log thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on ssh log thread start"});
      return -1;
   }
   
   if(!m_sshLogThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "ssh log thread did not start"});
      return -1;
   }
   
   sched_param sp;
   sp.sched_priority = m_sshLogThreadPrio;

   int rv = pthread_setschedparam( m_sshLogThread.native_handle(), SCHED_OTHER, &sp);
   
   if(rv != 0)
   {
      log<software_error>({__FILE__, __LINE__, rv, "Error setting thread params."});
      return -1;
   }
   
   return 0;

}

inline
void sshDigger::sshLogThreadExec()
{
   char buffer[4096];

   std::string logs;
   while(m_shutdown == 0 && m_sshSTDERR > 0)
   {
      ssize_t count = read(m_sshSTDERR, buffer, sizeof(buffer));
      if (count <= 0) 
      {
         continue;
      }
      else 
      {
         buffer[count] = '\0';
         
         logs += buffer;
         
         //Keep reading until \n found, then process.
         if(logs.back() == '\n')
         {
            size_t bol = 0;
            while(bol < logs.size())
            {
               size_t eol = logs.find('\n', bol);
               if(eol == std::string::npos) break;
               
               processSSHLog(logs.substr(bol, eol-bol));               
               bol = eol + 1;
            }
            logs = "";
         }
      }      
   }

}

inline
int sshDigger::processSSHLog( std::string logs )
{
   ///\todo interpret logs, giving errors vs info vs debug
   
   logPrioT lp = logPrio::LOG_INFO;
   
   if(logs.find("bind: Address already in use") != std::string::npos)
   {
      lp = logPrio::LOG_ERROR;
      m_sshError = true;
   }
   
   if(logs.find("channel_setup_fwd_listener_tcpip: cannot listen to port:") != std::string::npos)
   {
      lp = logPrio::LOG_ERROR;
      m_sshError = true;
   }
   
   m_log.log<text_log>({"SSH: " + logs}, lp);

   return 0;
}


inline
void sshDigger::_autosshLogThreadStart( sshDigger * s)
{
   s->autosshLogThreadExec();
}

inline
int sshDigger::autosshLogThreadStart()
{
   try
   {
      m_autosshLogThread  = std::thread( _autosshLogThreadStart, this);
   }
   
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on autossh log thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on autossh log thread start"});
      return -1;
   }
   
   if(!m_autosshLogThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "autossh log thread did not start"});
      return -1;
   }
   
   sched_param sp;
   sp.sched_priority = m_autosshLogThreadPrio;

   int rv = pthread_setschedparam( m_autosshLogThread.native_handle(), SCHED_OTHER, &sp);
   
   if(rv != 0)
   {
      log<software_error>({__FILE__, __LINE__, rv, "Error setting thread params."});
      return -1;
   }
   
   return 0;

}

inline
void sshDigger::autosshLogThreadExec()
{
   char buffer[4096];

   m_autosshLogFD = 0;
   
   m_autosshLogFD = open(m_autosshLogFile.c_str(), O_RDONLY );
   
   std::string logs;
   while(m_shutdown == 0)
   {
      ssize_t count = read(m_autosshLogFD, buffer, sizeof(buffer));
      if (count <= 0) 
      {
         mx::milliSleep(100); //read doesn't block, probably because autossh holds it open 
         continue;
      }
      else 
      {
         buffer[count] = '\0';
         
         logs += buffer;
         
         //Keep reading until \n found, then process.
         if(logs.back() == '\n')
         {
            size_t bol = 0;
            while(bol < logs.size())
            {
               size_t eol = logs.find('\n', bol);
               if(eol == std::string::npos) break;
               
               processAutoSSHLog(logs.substr(bol, eol-bol));               
               bol = eol + 1;
            }
            logs = "";
         }
      }      
   }

}

inline
int sshDigger::processAutoSSHLog( std::string logs )
{
   ///\todo interpret logs, giving errors vs info vs debug, strip timestamps, etc.
   
   m_log.log<text_log>({"AUTOSSH: " + logs});

   return 0;
}

int sshDigger::appStartup()
{
   m_tunnelPID = 0;

   m_autosshLogFile = "/dev/shm/sshDigger_autossh_" + m_configName + "_" + std::to_string(m_pid);
   m_autosshLogFD = open(m_autosshLogFile.c_str(), O_CREAT , S_IRUSR | S_IWUSR );
   close(m_autosshLogFD);
   
   
   if(execTunnel() < 0)
   {
      log<software_critical>({__FILE__,__LINE__});
      return -1;
   }

   if(autosshLogThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }

   if(sshLogThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   
   return 0;
}

int sshDigger::appLogic()
{
   state(stateCodes::CONNECTED);
   
   if(m_sshError)
   {
      log<text_log>("sending SIGUSR1 to restart ssh");
      kill(m_tunnelPID, SIGUSR1);
      m_sshError = false;
   }
   
   //Check if the tunnelPID is still in proc and not a zombie
   bool needrestart = false; //will be true if autossh is no longer alive
   std::string procDir = "/proc/" + std::to_string(m_tunnelPID) + "/status";
   if(access( procDir.c_str(), F_OK) != 0)
   {
      log<text_log>("autossh died");
      needrestart = true;
   }
   else
   {
      std::ifstream statin;
      statin.open(procDir.c_str());
      
      //Find the state entry and check for zombie-ness
      std::string line;
      bool statefound = false;
      while(statin.good() && !statefound)
      {
         getline( statin, line);
         if(line.find("State:") != std::string::npos) statefound = true;
      }
      
      if(line.find("Z") != std::string::npos)
      {
         log<text_log>("autossh has gone zombie");
         needrestart = true;
      }
   }
   
   if(needrestart)
   {
      int status;
      waitpid(m_tunnelPID, &status, 0);

      m_sshSTDERR = -1; //This tells the sshLogThread to exit
      char w = '\0';
      ssize_t nwr = write(m_sshSTDERR_input, &w, 1); //And this wakes it up from the blocking read
   
      if(m_sshLogThread.joinable()) m_sshLogThread.join();
      if(nwr != 1)
      {
         log<software_error>({__FILE__, __LINE__, errno });
         log<software_error>({__FILE__, __LINE__, "Error on write to ssh log thread. restart failed."});
         return -1;
      }
   
      //And now we can restart autossh
      if(execTunnel() < 0)
      {
         log<software_critical>({__FILE__,__LINE__,"restart of tunnel failed."});
         return -1;
      }
   
      //And the log thread.
      if(sshLogThreadStart() < 0)
      {
         log<software_critical>({__FILE__, __LINE__, "restart of ssh log thread failed."});
         return -1;
      }
   
   }
   
   
   return 0;
}

int sshDigger::appShutdown()
{
   kill(m_tunnelPID, SIGTERM);
   
   //Write the the ssh stderr to wake up the ssh log thread, which will then see shutdown is set.
   char w = '\0';
   ssize_t nwr = write(m_sshSTDERR_input, &w, 1);
   if(nwr != 1)
   {
      log<software_error>({__FILE__, __LINE__, errno });
      log<software_error>({__FILE__, __LINE__, "Error on write to ssh log thread. Sending SIGTERM."});
      pthread_kill(m_sshLogThread.native_handle(), SIGTERM);
      
   }
   
   if(m_sshLogThread.joinable()) m_sshLogThread.join();
   
   //Close the autossh FD to get that thread to shutdown.
   close(m_autosshLogFD);
   if(m_autosshLogThread.joinable()) m_autosshLogThread.join();
   
   return 0;
}

} //namespace app 
} //namespace MagAOX

#endif //sshDigger_hpp
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
