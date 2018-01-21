/** \file magAOXApp.hpp 
  * \brief The basic MagAO-X Application
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-12-24 created by JRM
  */ 

#ifndef app_MagAOXApp_hpp
#define app_MagAOXApp_hpp


#include <signal.h>
#include <sys/stat.h>

#include <fstream>

#include <boost/filesystem.hpp>

#include <mx/app/application.hpp>

#include "../common/environment.hpp"
#include "../common/defaults.hpp"
#include "../common/config.hpp"

#include "../logger/logManager.hpp"
#include "../logger/logTypes.hpp"
#include "../logger/logFileRaw.hpp"

#include "stateCodes.hpp"

using namespace MagAOX::logger;

namespace MagAOX 
{
namespace app 
{

///The base-class for MagAO-X applications.
/**
  * You can define a base configuration file for this class by writing 
  * \code
  *  #define MAGAOX_configBase "relative/path/from/configDir.conf"
  * \endcode
  * before including MagAOXApp.hpp.  This would be used, for instance to have a config common to 
  * all filter wheels.
  *  
  * \todo add INDI!
  * \todo need plan and default handling for magaox status file system. (see lockPID())
  * \todo Need to dump log to log/something..something on failed startup, that is before we're sure PID isn't locked.  Maybe a check in d'tor.
  * \todo do we need libMagAOX error handling? (a stack?)
  * \todo config logger thread priority
  * \todo define-default and config for log extension
  * \todo define-default and config for max log size
  * \todo define-default for writePause
  * \todo define-default for loopPause
  */ 
class MagAOXApp : public mx::application
{
   
protected:
   std::string configDir; ///< The configuration directory.  
   std::string configName; ///< The name of the configuration file (minus .conf).

   unsigned long loopPause {1000000000}; ///< The time in nanoseconds to pause the main loop.  The appLogic() function of the derived class is called every loopPause nanoseconds.  Default is 1,000,000,000 ns.  Config with loopPause=X.
   
   MagAOX::logger::logManager<logFileRaw> log;
   
   int m_shutdown {0}; ///< Flag to signal it's time to shutdown.  When not 0, the main loop exits.
   
   
   
public:
   
   /// Default c'tor.  Handles uid and initializes static members.
   /** 
     * Only one MagAOXApp can be instantiated per program.  Hence this c'tor will issue exit(-1) 
     * if the static self-pointer m_self is already initialized.
     * 
     * euid is set to 'real' to ensure that the application has normal privileges unless
     * explicitly needed.
     * 
     * Reference: http://man7.org/linux/man-pages/man2/getresuid.2.html
     * 
     */
   MagAOXApp();
  
   /// Set the paths for config files
   /** Replaces the mx::application defaults with the MagAO-X config system.
     *
     * This function parses the CL for "-n" or "--name".  
     * 
     * 
     * Do not override this unless you intend to depart from the MagAO-X standard.
     */
   virtual void setDefaults( int argc,    ///< [in] standard command line result specifying number of arguments in argv 
                             char ** argv ///< [in] standard command line result containing the arguments.
                           );
   
   /// The basic MagAO-X configuration setup method.  Should not normally be overridden.
   /** This method sets up the config system with the standard MagAO-X key=value pairs.
     *
     * Though it is virtual, it should not normally be overridden unless you need
     * to depart from the MagAO-X standard.
     * 
     * Setting up app specific config goes in setupConfig() implemented in the derived class.
     */
   virtual void setupBasicConfig();
      
   /// The basic MagAO-X configuration processing method.  Should not normally be overridden.
   /** This method processes the standard MagAO-X key=value pairs.
     * 
     * Though it is virtual, it should not normally be overridden unless you need
     * to depart from the MagAO-X standard.
     * 
     * Processing of app specific config goes in loadConfig() implemented by the derived class.
     */
   virtual void loadBasicConfig();
      
   /// The execute method implementing the standard main loop.  Should not normally be overridden.
   /**
     *
     */ 
   virtual int execute();
   
  
   /** \name Pure Virtual Functions
     * Derived applications must implement these.
     * @{
     */
   
   /// Any tasks to perform prior to the main event loop go here.
   /** This is called after signal handling is installed.
     *
     * Set m_shutdown = 1 on any fatal errors here.
     */ 
   virtual int appStartup() = 0;
   
   /// This is where derived applications implement their main FSM logic.
   /** This will be called again after loopPause nanoseconds until the application terminates.
     * 
     * Should return -1 on an any uncrecoverable errors which will caues app to terminate.  Could also set m_shutdown=1.
     * Return 0 on success, or at least intent to continue.
     *
     */   
   virtual int appLogic() = 0;
   
   /// Any tasks to perform after main loop exit go here.
   /** Should be able to handle case where appStartup and/or appLogic have not run.
     */
   virtual int appShutdown() = 0;
   
   ///@} -- Pure Virtual Functions


   /** \name Signal Handling 
     * @{
     */
private:
   
   static MagAOXApp * m_self; ///< Static pointer to this (set in constructor).  Mainly for getting out of static signal handlers.
   
   ///Sets the handler for SIGTERM, SIGQUIT, and SIGINT.
   int setSigTermHandler();
   
   ///The handler called when SIGTERM, SIGQUIT, or SIGINT is received.  Just a wrapper for handlerSigTerm.
   static void _handlerSigTerm( int signum,        ///< [in] specifies the signal.
                                siginfo_t *siginf, ///< [in] ignored by MagAOXApp
                                void *ucont        ///< [in] ignored by MagAOXApp
                              ); 
   
   ///Handles SIGTERM, SIGQUIT, and SIGINT.  Sets m_shutdown to 1 and logs the signal.
   void handlerSigTerm( int signum,         ///< [in] specifies the signal.
                        siginfo_t *siginf,  ///< [in] ignored by MagAOXApp
                        void *ucont         ///< [in] ignored by MagAOXApp
                      );

   ///@} -- Signal Handling
   
   
   /** \name RT Priority
     * @{
     */
private:
   uid_t m_euidReal;     ///< The real user id of the proces (i.e. the lower priveleged id of the user)
   uid_t m_euidCalled;   ///< The user id of the process as called (i.e. the higher priveleged id of the owner, root if setuid).
   uid_t m_suid;         ///< The save-set user id of the process
   
   int m_RTPriority {0}; ///< The real-time scheduling priority.  Default is 0.
   
protected:
   
   /// Set the effective user ID to the called value, i.e. the highest possible.
   /** If setuid is set on the file, this will be super-user privileges.
     *
     * Reference: http://pubs.opengroup.org/onlinepubs/009695399/functions/seteuid.html 
     * 
     * \returns 0 on success
     * \returns -1 on error from setuid().
     */
   int euidCalled();
   
   /// Set the effective user ID to the real value, i.e. the file owner.
   /** 
     * Reference: http://pubs.opengroup.org/onlinepubs/009695399/functions/seteuid.html
     * 
     * \returns 0 on success
     * \returns -1 on error from setuid().
     */ 
   int euidReal();
   
   /// Set the real-time priority of this process.
   /** This method attempts to set euid to 'called' with \ref euidCalled.  It then sets the priority
     * but will fail if it does not have sufficient privileges.  Regardless, it will then restore
     * privileges with \ref euidReal.
     * 
     * If prio is > 99, then it is changed to 99.
     * 
     * \returns 0 on success.
     * \returns -1 on an error.  In this case priority will not have been changed.
     */
   int RTPriority( unsigned prio /**< [in] the desired new RT priority */ );
   
   ///@} -- RT Priority
   
   /** \name PID Locking
     *
     * @{
     */
   
   std::string pidFileName; ///<The name of the PID file
   
   pid_t m_pid {0}; ///< This process's PID
   
   /// Attempt to lock the PID by writing it to a file. Fails if a process is already running with the same config name.
   /** First checks the PID file for an existing PID.  If found, interrogates /proc to determine if that process is
     * running and if so if the command line matches.  If a matching process is currently running, then this returns an error.
     * 
     * Will not fail if a PID file exists but the stored PID does not correspond to a running process with the same command line name.
     *
     * Reference: https://linux.die.net/man/3/getpid
     * 
     * \returns 0 on success.
     * \returns -1 on any error, including creating the PID file or if this app is already running.
     */ 
   int lockPID();
   
   /// Remove the PID file.
   int unlockPID();
   
   ///@} -- PID Locking
   
   /** \name Application State
     *
     * @{
     */
private:
   stateCodeT m_state {stateCodes::UNINITIALIZED}; ///< The application's state.  Never ever set this directly, use state(const stateCodeT & s).
   int m_stateLogged {0} ;///< Counter and flag for use to log errors just once.  Never ever access directly, use stateLogged().
   
public:
   /// Get the current state code 
   /** \returns m_state
     */
   stateCodeT state();
   
   /// Set the current state code 
   /** If no change, returns immediately with no actions.
     * 
     * If it is a change, the state change is logged.  Also resets m_stateLogged to 0.
     */ 
   void state(const stateCodeT & s /**< [in] The new application state */);
   
   /// Updates and returns the value of m_stateLogged.  Will be 0 on first call after a state change, \>0 afterwords.
   /** This method exists to facilitate logging the reason for a state change once, but not 
     * logging it on subsequent event loops.  Returns the current value upon entry, but updates 
     * before returning so that the next call returns the incremented value.  Example usage:
     * \code
       if( connection_failed ) //some condition set this to true
       {
          state( stateCodes::NOTCONNECTED );
          if(!stateLogged()) log.log<text_log>("Not connected");
       }
       \endcode
     * In this example, the log entry is made the first time the state changes.  If there are no changes to a 
     * different state in the mean time, then when the event loop gets here again and decides it is not connected,
     * the log entry will not be made.  
     *
     * \returns current value of m_stateLogged, that is the value before it is incremented.
     */ 
   int stateLogged();
   
   ///@} --Application State

};

MagAOXApp * MagAOXApp::m_self = nullptr;

inline
MagAOXApp::MagAOXApp()
{
   if( m_self != nullptr )
   {
      std::cerr << "Attempt to instantiate 2nd MagAOXApp.  Exiting immediately.\n";
      exit(-1);
   }
   
   m_self = this;
   
   //Get the uids of this process.
   getresuid(&m_euidReal, &m_euidCalled, &m_suid);
   euidReal(); //immediately step down to unpriveleged uid.   
   
}

inline
void MagAOXApp::setDefaults( int argc, 
                             char ** argv
                           )   //virtual
{
   char *tmpstr;
   std::string tmp;
   

   tmpstr = getenv(MAGAOX_configDir);
   if(tmpstr != 0) 
   {
      configDir = tmpstr;
   }
   else
   {
      configDir = MAGAOX_default_path;
      configDir += "/config";
   }
   
   tmpstr = getenv(MAGAOX_globalConfig);
   if(tmpstr != 0) 
   {
      tmp = tmpstr;
   }
   else
   {
      tmp = MAGAOX_default_global_config;
   }
   
   configPathGlobal = configDir + "/" + tmp;
   
   #ifdef MAGAOX_configBase
      //We use mx::application's configPathUser for this components base config file
      configPathUser = configDir + "/" + MX_APP_DEFAULT_configPathUser;
   #endif
   
   //Parse CL just to get the "name".      
   config.add("name","n", "name",mx::argType::Required, "", "name", false, "string", "The name of the application, specifies config.");
   
   config.parseCommandLine(argc, argv, "name");
   config(configName, "name");
   
   if(configName == "")
   {
      boost::filesystem::path p(invokedName);
      configName = p.stem().string();
      
      log.log<text_log>("Application name (-n --name) not set.  Using argv[0].");
   }
   
   //We use mx::application's configPathLocal for this components config file 
   configPathLocal = configDir + "/" + configName + ".conf";

   return;
   
}

inline
void MagAOXApp::setupBasicConfig() //virtual
{
   //App stuff   
   config.add("loopPause", "p", "loopPause", mx::argType::Required, "", "loopPause", false, "unsigned long", "The main loop pause time in ns");
   config.add("RTPriority", "P", "RTPriority", mx::argType::Required, "", "RTPriority", false, "unsigned", "The real-time priority (0-99)");   
   
   //Logger Stuff
   config.add("logger.logDir","L", "logDir",mx::argType::Required, "logger", "logDir", false, "string", "The directory for log files");
   config.add("logger.writePause","", "writePause",mx::argType::Required, "logger", "writePause", false, "unsigned long", "The log thread pause time in ns");
   config.add("logger.logLevel","l", "logLevel",mx::argType::Required, "logger", "logLevel", false, "string", "The log level");
   

  
}

inline
void MagAOXApp::loadBasicConfig() //virtual
{
   //---------- Setup the logger ----------//
   
   //-- logDir
   std::string tmp;
   config(tmp, "logger.logDir");
   
   log.m_logFile.path(tmp);
   log.m_logFile.name(configName);
      
   //-- logLevel
   tmp = "";
   config(tmp, "logger.logLevel");
   if(tmp != "")
   {
      logLevelT lev;

      lev = logLevelFromString(tmp);
   
      if(  lev == logLevels::DEFAULT ) lev = logLevels::INFO;
      if( lev == logLevels::UNKNOWN )
      {
         std::cerr << "Unkown log level specified.  Using default (INFO)\n";
         lev = logLevels::INFO;
      }
      
      log.m_logLevel = lev;
   }

   //-- writePause
   unsigned long wp = 0;
   config(wp, "logger.writePause");
   if(wp != 0)
   {
      log.writePause(wp);
   }
   
   //--------- Loop Pause Time --------//
   config(loopPause, "loopPause");
   
   //--------- RT Priority ------------//
   int prio;
   config(prio, "RTPriority");
   if(prio != m_RTPriority)
   {
      RTPriority(prio);
   }
}


inline
int MagAOXApp::execute() //virtual
{
   if( lockPID() < 0 )
   {
      log.log<text_log>("Failed to lock PID.", logLevels::FATAL);
      m_shutdown = 1; //This lets logger run, etc, but appLogic never goes.
   }
   
   //Begin the logger   
   log.logThreadStart();
   
   setSigTermHandler();
   
   if( m_shutdown == 0 )
   {
      state(stateCodes::INITIALIZED);
      if(appStartup() < 0) m_shutdown = 1;
   }
   
   while( m_shutdown == 0)
   {
      /** \todo Add a mutex to lock every time appLogic is called.  
        * This would allow other threads to run appLogic, making it more responsive to status queries, etc.
        *  
        */
      if( appLogic() < 0) m_shutdown = 1;

      /** \todo Need a heartbeat update here.
        */

      //Pause loop unless shutdown is set
      if( m_shutdown == 0)
      {
         std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(loopPause));
      }
   }
   
   appShutdown();
   
   state(stateCodes::SHUTDOWN);
   
   unlockPID();
   
   return 0;
}

inline 
int MagAOXApp::setSigTermHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &MagAOXApp::_handlerSigTerm;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGTERM, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGTERM failed. Errno says: ";
      logss += strerror(errno);
      
      log.log<software_error>({__FILE__, __LINE__, errno, logss});
      
      return -1;
   }
   
   errno = 0;
   if( sigaction(SIGQUIT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGQUIT failed. Errno says: ";
      logss += strerror(errno);
      
      log.log<software_error>({__FILE__, __LINE__, errno, logss});
      
      return -1;
   }
   
   errno = 0;
   if( sigaction(SIGINT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGINT failed. Errno says: ";
      logss += strerror(errno);
      
      log.log<software_error>({__FILE__, __LINE__, errno, logss});
      
      return -1;
   }
   
   log.log<text_log>("Installed SIGTERM/SIGQUIT/SIGINT signal handler.");
   
   return 0;
}
   
inline 
void MagAOXApp::_handlerSigTerm( int signum, 
                                 siginfo_t *siginf, 
                                 void *ucont
                               )
{
   m_self->handlerSigTerm(signum, siginf, ucont);
}

inline 
void MagAOXApp::handlerSigTerm( int signum, 
                                siginfo_t *siginf __attribute__((unused)), 
                                void *ucont __attribute__((unused))
                              ) 
{
   m_shutdown = 1;
   
   std::string signame;
   switch(signum)
   {
      case SIGTERM:
         signame = "SIGTERM";
         break;
      case SIGINT:
         signame = "SIGINT";
         break;
      case SIGQUIT:
         signame = "SIGQUIT";
         break;
      default:
         signame = "OTHER";
   }

   std::string logss = "Caught signal ";
   logss += signame;
   logss += ". Shutting down.";
   
   log.log<text_log>(logss);
}

inline 
int MagAOXApp::euidCalled()
{
   errno = 0;
   if(seteuid(m_euidCalled) < 0)
   {
      std::string logss = "Setting effective user id to euidCalled (";
      logss += mx::convertToString<int>(m_euidCalled);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);
      
      log.log<software_error>({__FILE__, __LINE__, errno, logss});
      
      return -1;
   }
   
   return 0;
}

inline 
int MagAOXApp::euidReal()
{
   errno = 0;
   if(seteuid(m_euidReal) < 0)
   {
      std::string logss = "Setting effective user id to euidReal (";
      logss += mx::convertToString<int>(m_euidReal);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);
      
      log.log<software_error>({__FILE__, __LINE__, errno, logss});
      
      return -1;
   }
   
   return 0;
   
}

inline
int MagAOXApp::RTPriority( unsigned prio)
{
   struct sched_param schedpar;
   
   if(prio > 99) prio = 99;
   schedpar.sched_priority = prio;
      
   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log.log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to called failed."});
      return -1;
   }
   
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = sched_setscheduler(0, MAGAOX_RT_SCHED_POLICY, &schedpar);
   else rv = sched_setscheduler(0, SCHED_OTHER, &schedpar);

   if(rv < 0)
   {
      std::stringstream logss;
      logss << "Setting scheduler priority to " << prio <<" failed.  Errno says: " << strerror(errno) << ".  ";
      log.log<software_error>({__FILE__, __LINE__, errno, logss.str()});
   }
   else
   {
      m_RTPriority = prio;

      std::stringstream logss;
      logss << "Scheduler priority (RT_priority) set to " << m_RTPriority << ".";
      log.log<text_log>(logss.str());
   }

   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log.log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to real failed."});
      return -1;
   }
   
   return rv;
}

inline
int MagAOXApp::lockPID()
{
   m_pid = getpid();
   
   
   std::string statusDir = "/tmp/magaox/";
   
   // Create statusDir root with read/write/search permissions for owner and group, and with read/search permissions for others.
   errno = 0;
   if( mkdir(statusDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
   {
      if( errno != EEXIST)
      {
         std::stringstream logss;
         logss << "Failed to create root of statusDir (" << statusDir << ").  Errno says: " << strerror(errno);
         log.log<software_critical>({__FILE__, __LINE__, errno, logss.str()});
         return -1;
      }
      
   }

   statusDir += configName;
   
   pidFileName = statusDir + "/pid";
   
   // Create statusDir with read/write/search permissions for owner and group, and with read/search permissions for others.
   errno = 0;
   if( mkdir(statusDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
   {
      if( errno != EEXIST)
      {
         std::stringstream logss;
         logss << "Failed to create statusDir (" << statusDir << ").  Errno says: " << strerror(errno);
         log.log<software_critical>({__FILE__, __LINE__, errno, logss.str()});
         
         return -1;
      }
      
      //If here, then we need to check the pid file.
      
      std::ifstream pidIn;
      pidIn.open( pidFileName );
      
      if(pidIn.good()) //PID file exists, now read its contents and compare to proc/<pid>/cmdline
      {
         //Read PID from file
         pid_t testPid;   
         pidIn >> testPid;
         pidIn.close();

         //Get command line used to start this process from /proc
         std::stringstream procN;
         procN << "/proc/" << testPid << "/cmdline";
         
         std::ifstream procIn;
         std::string pidCmdLine;

         ///\todo what happens if /proc/pid/cmdline doesn't exist?  Need error handling here?
         procIn.open(procN.str());         
         procIn >> pidCmdLine;
         procIn.close();

         ///\todo This needs to also check for configName in case same invokedName but different configName is the one who is re-using the PID (pathological)
         //Search for invokedName in command line.
         if( pidCmdLine.find( invokedName ) != std::string::npos )
         {
            //This means that this app already exists for this config, and we need to die.
            std::stringstream logss;
            logss << "PID already locked (" << testPid  << ").  Time to die.";
            log.log<text_log>(logss.str(), logLevels::CRITICAL);            
            return -1;
         }
      }
      else
      {
         //No PID File so we should just go on.
         pidIn.close();
      }
   }
   
   //Now write current PID to file and go on with life.
   std::ofstream pidOut;
   pidOut.open(pidFileName);

   /// \todo need some error checking here.
   
   pidOut << m_pid;
   
   pidOut.close();
   
   std::stringstream logss;
   logss << "PID (" << m_pid << ") locked.";
   log.log<text_log>(logss.str());
}

inline
int MagAOXApp::unlockPID()
{
   /// \todo need error handling here.
   remove(pidFileName.c_str());
   
   std::stringstream logss;
   logss << "PID (" << m_pid << ") unlocked.";
   log.log<text_log>(logss.str());
   
   return 0;
}

stateCodeT MagAOXApp::state()
{
   return m_state;
}

void MagAOXApp::state(const stateCodeT & s)
{
   if(m_state == s) return;
   
   
   log.log<state_change>( {m_state, s} );
   
   m_state = s;
   m_stateLogged = 0;
}

int MagAOXApp::stateLogged()
{
   if(m_stateLogged > 0)
   {
      ++m_stateLogged;
      return m_stateLogged - 1;
   }
   else
   {
      m_stateLogged = 1;
      return 0;
   }
}
   
} //namespace app 
} //namespace MagAOX 

#endif //app_MagAOXApp_hpp

