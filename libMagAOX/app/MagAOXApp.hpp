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

#include <cstdlib>
#include <unistd.h>
//#include <fstream>

#include <unordered_map>

#include <boost/filesystem.hpp>

#include <mx/mxlib.hpp>
#include <mx/app/application.hpp>
#include <mx/environment.hpp>



#include "../common/environment.hpp"
#include "../common/paths.hpp"
#include "../common/defaults.hpp"
#include "../common/config.hpp"

#include "../logger/logManager.hpp"
#include "../logger/logTypes.hpp"
#include "../logger/logFileRaw.hpp"

#include "stateCodes.hpp"
#include "indiDriver.hpp"
#include "indiMacros.hpp"

#include "../../INDI/libcommon/Config.hpp"
#include "../../INDI/libcommon/System.hpp"

using namespace MagAOX::logger;

namespace MagAOX
{
namespace app
{

/// The base-class for MagAO-X applications.
/**
  * You can define a base configuration file for this class by writing
  * \code
  *  #define MAGAOX_configBase "relative/path/from/configDir.conf"
  * \endcode
  * before including MagAOXApp.hpp.  This would be used, for instance to have a config common to
  * all filter wheels.
  *
  *
  * \todo do we need libMagAOX error handling? (a stack?)
  *
  * \ingroup magaoxapp
  */
class MagAOXApp : public mx::application
{

protected:

   std::string MagAOXPath; ///< The base path of the MagAO-X system.

   std::string m_configName; ///< The name of the configuration file (minus .conf).

   std::string sysPath;  ///< The path to the system directory, for PID file, etc.

   std::string secretsPath; ///< Path to the secrets directory, where passwords, etc, are stored.

   unsigned long loopPause {MAGAOX_default_loopPause}; ///< The time in nanoseconds to pause the main loop.  The appLogic() function of the derived class is called every loopPause nanoseconds.  Default is 1,000,000,000 ns.  Config with loopPause=X.

   int m_shutdown {0}; ///< Flag to signal it's time to shutdown.  When not 0, the main loop exits.

private:

   ///Default c'tor is deleted.
   MagAOXApp() = delete;

public:

   /// Public c'tor.  Handles uid, logs git repo status, and initializes static members.
   /**
     * Only one MagAOXApp can be instantiated per program.  Hence this c'tor will issue exit(-1)
     * if the static self-pointer m_self is already initialized.
     *
     * euid is set to 'real' to ensure that the application has normal privileges unless
     * explicitly needed.
     *
     * Reference: http://man7.org/linux/man-pages/man2/getresuid.2.html
     *
     * The git repository status is required to create a MagAOXApp.  Derived classes
     * should include the results of running `gengithead.sh` and pass the defined
     * sha1 and modified flags.
     *
     */
   MagAOXApp( const std::string & git_sha1, ///< [in] The current SHA1 hash of the git repository
              const bool git_modified,       ///< [in] Whether or not the repo is modified.
              const bool useINDI = true ///< [in] [optional] Whether or not this app uses INDI
            );

   ~MagAOXApp() noexcept(true);

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
   /** This will be called every loopPause nanoseconds until the application terminates.
     *
     * Should return -1 on an any unrecoverable errors which will caues app to terminate.  Could also set m_shutdown=1.
     * Return 0 on success, or at least intent to continue.
     *
     */
   virtual int appLogic() = 0;

   /// Any tasks to perform after main loop exit go here.
   /** Should be able to handle case where appStartup and/or appLogic have not run.
     */
   virtual int appShutdown() = 0;

   ///@} -- Pure Virtual Functions

   /** \name Logging
     * @{
     */
protected:
   logger::logManager<logFileRaw> m_log;

public:

   /// Make a log entry
   /** Wrapper for logManager::log
     */
   template<typename logT>
   void log( const typename logT::messageT & msg, ///< [in] the message to log
             logLevelT level = logLevels::DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );

   /// Make a log entry
   /** Wrapper for logManager::log
     */
   template<typename logT>
   void log( logLevelT level = logLevels::DEFAULT /**< [in] [optional] the log level.  The default is used if not specified.*/);

   ///@} -- logging

   /** \name Signal Handling
     * @{
     */
private:

   static MagAOXApp * m_self; ///< Static pointer to this (set in constructor).  Used to test whether a a MagAOXApp is already instatiated (a fatal error) and used for getting out of static signal handlers.

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

   /** \name Privilege Management
     * @{
     */
private:
   uid_t m_euidReal;     ///< The real user id of the proces (i.e. the lower privileged id of the user)
   uid_t m_euidCalled;   ///< The user id of the process as called (i.e. the higher privileged id of the owner, root if setuid).
   uid_t m_suid;         ///< The save-set user id of the process

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


   ///@} -- Privilege Management

   /** \name RT Priority
     * @{
     */
private:
   int m_RTPriority {0}; ///< The real-time scheduling priority.  Default is 0.

protected:
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
     * Each MagAOXApp has a PID lock file in the system directory.  The app will not 
     * startup if it detects that the PID is already locked, preventing duplicates.  This is
     * based on the configured name, not the invoked name (argv[0]).
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
   stateCodes::stateCodeT m_state {stateCodes::UNINITIALIZED}; ///< The application's state.  Never ever set this directly, use state(const stateCodeT & s).

   int m_stateLogged {0} ;///< Counter and flag for use to log errors just once.  Never ever access directly, use stateLogged().

public:
   /// Get the current state code
   /** \returns m_state
     */
   stateCodes::stateCodeT state();

   /// Set the current state code
   /** If no change, returns immediately with no actions.
     *
     * If it is a change, the state change is logged.  Also resets m_stateLogged to 0.
     */
   void state(const stateCodes::stateCodeT & s /**< [in] The new application state */);

   /// Updates and returns the value of m_stateLogged.  Will be 0 on first call after a state change, \>0 afterwords.
   /** This method exists to facilitate logging the reason for a state change once, but not
     * logging it on subsequent event loops.  Returns the current value upon entry, but updates
     * before returning so that the next call returns the incremented value.  Example usage:
     * \code
       if( connection_failed ) //some condition set this to true
       {
          state( stateCodes::NOTCONNECTED );
          if(!stateLogged()) log<text_log>("Not connected");
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

   /** \name INDI Interface
     *
     * @{
     */
protected:
   
   ///Flag controlling whether INDI is used.  If false, then no INDI code executes.
   bool m_useINDI {true};
   
   ///The INDI driver wrapper.  Constructed and initialized by execute, which starts and stops communications.
   indiDriver<MagAOXApp> * m_indiDriver {nullptr};

   ///Mutex for locking INDI communications.
   std::mutex m_indiMutex;

   ///Structure to hold the call-back details for handling INDI communications.
   struct indiCallBack
   {
      pcf::IndiProperty * property {0}; ///< A pointer to an INDI property.
      int (*newCallBack)( void *, const pcf::IndiProperty &) {0}; ///< The function to call for a new property request.
   };

   ///Map to hold the indiCallBacks for this App, with fast lookup by property name.
   std::unordered_map< std::string, indiCallBack> m_indiCallBacks;

   ///Value type of the indiCallBack map.
   typedef std::pair<std::string, indiCallBack> callBackValueType;

   ///Iterator type of the indiCallBack map.
   typedef std::unordered_map<std::string, indiCallBack>::iterator callBackIterator;

   ///Return type of insert on the indiCallBack map.
   typedef std::pair<callBackIterator,bool> callBackInsertResult;

   ///Full path name of the INDI driver input FIFO.
   std::string m_driverInName;

   ///Full path name of the INDI driver output FIFO.
   std::string m_driverOutName;

   /// Register an INDI property.
   /**
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     * \todo needs error logging
     * \todo needs exception handling
     * \todo is a failure to register a FATAL error?
     */
   int registerIndiProperty( pcf::IndiProperty & prop,                               ///< [out] the property to register
                             const std::string & propName,                           ///< [in] the name of the property
                             const pcf::IndiProperty::Type & propType,               ///< [in] the type of the property
                             const pcf::IndiProperty::PropertyPermType & propPerm,   ///< [in] the permissions of the property
                             const pcf::IndiProperty::PropertyStateType & propState, ///< [in] the state of the property
                             int (*)( void *, const pcf::IndiProperty &)             ///< [in] the callback for changing the property
                           );

   /// Create the INDI FIFOs
   /** Changes permissions to max available and creates the
     * FIFOs at the configured path.
     */
   int createINDIFIFOS();

   /// Start INDI Communications
   /**
     * \returns 0 on success
     * \returns -1 on error.  This is fatal.
     */
   int startINDI();

public:
   /// Handler for the get INDI properties request
   /** Uses the properties registered in m_indiCallBacks to respond to the request.  This is called by
     * m_indiDriver's indiDriver::handleGetProperties.
     */
   void handleGetProperties( const pcf::IndiProperty &ipRecv /**< [in] The property being requested. */ );

   /// Handler for the new INDI property request
   /** Uses the properties registered in m_indiCallBacks to respond to the request, looking up the callback for this
     * property and calling it.
     *
     * This is called by m_indiDriver's indiDriver::handleGetProperties.
     *
     * \todo handle errors, are they FATAL?
     */
   void handleNewProperty( const pcf::IndiProperty &ipRecv /**< [in] The property being changed. */);

protected:

   ///indi Property to report the application state.
   pcf::IndiProperty indiP_state;

   ///@} --INDI Interface

public:

   /** \name Member Accessors
     *
     * @{
     */

   ///Get the config name
   /**
     * \returns the current value of m_configName
     */
   std::string configName();

   ///Get the INDI input FIFO file name
   /**
     * \returns the current value of m_driverInName
     */
   std::string driverInName();

   ///Get the INDI output FIFO file name
   /**
     * \returns the current value of m_driverOutName
     */
   std::string driverOutName();

   ///@} --Member Accessors
};

//Set self pointer to null so app starts up uninitialized.
MagAOXApp * MagAOXApp::m_self = nullptr;

inline
MagAOXApp::MagAOXApp( const std::string & git_sha1,
                      const bool git_modified,
                      const bool useINDI
                    )
{
   if( m_self != nullptr )
   {
      std::cerr << "Attempt to instantiate 2nd MagAOXApp.  Exiting immediately.\n";
      exit(-1);
   }

   m_self = this;

   //We log the current GIT status.
   log<git_state>(git_state::messageT("MagAOX", git_sha1, git_modified));
   log<git_state>(git_state::messageT("mxlib", MXLIB_UNCOMP_CURRENT_SHA1, MXLIB_UNCOMP_REPO_MODIFIED));

   //Get the uids of this process.
   getresuid(&m_euidReal, &m_euidCalled, &m_suid);
   euidReal(); //immediately step down to unpriveleged uid.


   m_useINDI = useINDI;
}

inline
MagAOXApp::~MagAOXApp() noexcept(true)
{
   if(m_indiDriver) delete m_indiDriver;
   
   MagAOXApp::m_self = nullptr;
}
   
inline
void MagAOXApp::setDefaults( int argc,
                             char ** argv
                           )   //virtual
{
   std::string tmpstr;
   std::string configDir;

   tmpstr = mx::getEnv(MAGAOX_env_path);
   if(tmpstr != "")
   {
      MagAOXPath = tmpstr;
   }
   else
   {
      MagAOXPath = MAGAOX_path;
   }

   //Set the config path relative to MagAOXPath
   tmpstr = mx::getEnv(MAGAOX_env_config);
   if(tmpstr == "")
   {
      tmpstr = MAGAOX_configRelPath;
   }
   configDir = MagAOXPath + "/" + tmpstr;
   configPathGlobal = configDir + "/magaox.conf";

   //Setup default log path
   tmpstr = MagAOXPath + "/" + MAGAOX_logRelPath;

   m_log.logPath(tmpstr);

   //Setup default sys path
   tmpstr = MagAOXPath + "/" + MAGAOX_sysRelPath;
   sysPath = tmpstr;

   //Setup default secrets path
   tmpstr = MagAOXPath + "/" + MAGAOX_secretsRelPath;
   secretsPath = tmpstr;


   #ifdef MAGAOX_configBase
      //We use mx::application's configPathUser for this components base config file
      configPathUser = configDir + "/" + MAGAOX_configBase;
   #endif

   //Parse CL just to get the "name".
   config.add("name","n", "name",mx::argType::Required, "", "name", false, "string", "The name of the application, specifies config.");

   config.parseCommandLine(argc, argv, "name");
   config(m_configName, "name");

   if(m_configName == "")
   {
      boost::filesystem::path p(invokedName);
      m_configName = p.stem().string();
      log<text_log>("Application name (-n --name) not set.  Using argv[0].");
   }

   //We use mx::application's configPathLocal for this components config file
   configPathLocal = configDir + "/" + m_configName + ".conf";

   //Now we can setup common INDI properties
   REG_INDI_PROP_NOCB(indiP_state, "state", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   indiP_state.add (pcf::IndiElement("current"));


   return;

}

inline
void MagAOXApp::setupBasicConfig() //virtual
{
   //App stuff
   config.add("loopPause", "p", "loopPause", mx::argType::Required, "", "loopPause", false, "unsigned long", "The main loop pause time in ns");
   config.add("RTPriority", "P", "RTPriority", mx::argType::Required, "", "RTPriority", false, "unsigned", "The real-time priority (0-99)");

   //Logger Stuff
   m_log.setupConfig(config);

}

inline
void MagAOXApp::loadBasicConfig() //virtual
{
   //---------- Setup the logger ----------//
   m_log.logName(m_configName);
   m_log.loadConfig(config);

   //--------- Loop Pause Time --------//
   config(loopPause, "loopPause");

   //--------- RT Priority ------------//
   unsigned prio = m_RTPriority;
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
      state(stateCodes::FAILURE);
      log<text_log>("Failed to lock PID.", logLevels::FATAL);
      //Return immediately, not safe to go on.
      return -1;
   }

   //Begin the logger
   m_log.logThreadStart();

   setSigTermHandler();

   if( m_shutdown == 0 )
   {
      state(stateCodes::INITIALIZED);
      if(appStartup() < 0) m_shutdown = 1;
   }

   //====Begin INDI Communications
   if(startINDI() < 0)
   {
      state(stateCodes::FAILURE);
      m_shutdown = 1;
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

   //Stop INDI communications
   if(m_indiDriver != nullptr)
   {
      m_indiDriver->quitProcess();
      m_indiDriver->deactivate();
      log<indidriver_stop>();
   }

   unlockPID();

   return 0;
}

template<typename logT>
void MagAOXApp::log( const typename logT::messageT & msg,
                     logLevelT level
                   )
{
   m_log.log<logT>(msg, level);
}

template<typename logT>
void MagAOXApp::log( logLevelT level)
{
   m_log.log<logT>(level);
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

      log<software_error>({__FILE__, __LINE__, errno, logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGQUIT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGQUIT failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGINT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGINT failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, logss});

      return -1;
   }

   log<text_log>("Installed SIGTERM/SIGQUIT/SIGINT signal handler.");

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

   std::cerr << "\n" << logss << std::endl;
   log<text_log>(logss);
}

inline
int MagAOXApp::euidCalled()
{
   errno = 0;
   if(seteuid(m_euidCalled) < 0)
   {
      std::string logss = "Setting effective user id to euidCalled (";
      logss += mx::ioutils::convertToString<int>(m_euidCalled);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, logss});

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
      logss += mx::ioutils::convertToString<int>(m_euidReal);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, logss});

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
      log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to called failed."});
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
      log<software_error>({__FILE__, __LINE__, errno, logss.str()});
   }
   else
   {
      m_RTPriority = prio;

      std::stringstream logss;
      logss << "Scheduler priority (RT_priority) set to " << m_RTPriority << ".";
      log<text_log>(logss.str());
   }

   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to real failed."});
      return -1;
   }

   return rv;
}

inline
int MagAOXApp::lockPID()
{
   m_pid = getpid();

   std::string statusDir = sysPath;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to called failed."});
      return -1;
   }

   // Create statusDir root with read/write/search permissions for owner and group, and with read/search permissions for others.
   errno = 0;
   if( mkdir(statusDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
   {
      if( errno != EEXIST)
      {
         std::stringstream logss;
         logss << "Failed to create root of statusDir (" << statusDir << ").  Errno says: " << strerror(errno);
         log<software_critical>({__FILE__, __LINE__, errno, logss.str()});

         //Go back to regular privileges
         euidReal();

         return -1;
      }

   }

   statusDir += "/";
   statusDir += m_configName;

   pidFileName = statusDir + "/pid";

   // Create statusDir with read/write/search permissions for owner and group, and with read/search permissions for others.
   errno = 0;
   if( mkdir(statusDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
   {
      if( errno != EEXIST)
      {
         std::stringstream logss;
         logss << "Failed to create statusDir (" << statusDir << ").  Errno says: " << strerror(errno);
         log<software_critical>({__FILE__, __LINE__, errno, logss.str()});

         //Go back to regular privileges
         euidReal();

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

         try
         {
            procIn.open(procN.str());
            if(procIn.good()) procIn >> pidCmdLine;
            procIn.close();
         }
         catch( ... )
         {
            log<software_fatal>({__FILE__, __LINE__, 0, "exception caught testing /proc/pid"});
            euidReal();
            return -1;
         }

         //If pidCmdLine == "" at this point we just allow the rest of the
         //logic to run...

         //Search for invokedName in command line.
         size_t invokedPos = pidCmdLine.find( invokedName );

         //If invokedName found, then we check for configName.
         size_t configPos = std::string::npos;
         if(invokedPos != std::string::npos) configPos = pidCmdLine.find( m_configName );

         //Check if PID is already locked by this program+config combo:
         if(  invokedPos != std::string::npos && configPos != std::string::npos)
         {
            //This means that this app already exists for this config, and we need to die.
            std::stringstream logss;
            logss << "PID already locked (" << testPid  << ").  Time to die.";
            std::cerr << logss.str() << std::endl;

            log<text_log>(logss.str(), logLevels::CRITICAL);

            //Go back to regular privileges
            euidReal();

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

   if(!pidOut.good())
   {
      log<software_fatal>({__FILE__, __LINE__, errno, "could not open pid file for writing."});
      euidReal();
      return -1;
   }

   pidOut << m_pid;

   pidOut.close();

   std::stringstream logss;
   logss << "PID (" << m_pid << ") locked.";
   log<text_log>(logss.str());

   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, "Seeting euid to real failed."});
      return -1;
   }
}

inline
int MagAOXApp::unlockPID()
{
   /// \todo need error handling here.
   ::remove(pidFileName.c_str());

   std::stringstream logss;
   logss << "PID (" << m_pid << ") unlocked.";
   log<text_log>(logss.str());

   return 0;
}

inline
stateCodes::stateCodeT MagAOXApp::state()
{
   return m_state;
}

inline
void MagAOXApp::state(const stateCodes::stateCodeT & s)
{
   if(m_state == s) return;


   log<state_change>( {m_state, s} );

   m_state = s;
   m_stateLogged = 0;

   //And we keep INDI up to date
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.
   indiP_state["current"] = s;
   indiP_state.setState (pcf::IndiProperty::Ok);
   if(m_indiDriver) m_indiDriver->sendSetProperty (indiP_state);
}

inline
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

/*-------------------------------------------------------------------------------------*/
/*                                  INDI Support                                       */
/*-------------------------------------------------------------------------------------*/

inline
int MagAOXApp::registerIndiProperty( pcf::IndiProperty & prop,
                                     const std::string & propName,
                                     const pcf::IndiProperty::Type & propType,
                                     const pcf::IndiProperty::PropertyPermType & propPerm,
                                     const pcf::IndiProperty::PropertyStateType & propState,
                                     int (*newCallBack)( void *, const pcf::IndiProperty &ipRecv)
                                   )
{
   if(!m_useINDI) return 0;
   
   prop = pcf::IndiProperty (propType);
   prop.setDevice(m_configName);
   prop.setName(propName);
   prop.setPerm(propPerm);
   prop.setState( propState);


   callBackInsertResult result =  m_indiCallBacks.insert(callBackValueType( propName, {&prop, newCallBack}));

   if(!result.second)
   {
      return -1;
   }

   return 0;
}

inline
int MagAOXApp::createINDIFIFOS()
{
   if(!m_useINDI) return 0;
   
   ///\todo make driver FIFO path full configurable.
   std::string driverFIFOPath = MAGAOX_path;
   driverFIFOPath += "/";
   driverFIFOPath += MAGAOX_driverFIFORelPath;

   m_driverInName = driverFIFOPath + "/" + configName() + ".in";
   m_driverOutName = driverFIFOPath + "/" + configName() + ".out";

   //Get max permissions
   euidCalled();

   //Clear the file mode creation mask so mkfifo does what we want. Don't forget to restore it.
   mode_t prev = umask(0);

   errno = 0;
   if(mkfifo(m_driverInName.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP) !=0)
   {
      if(errno != EEXIST)
      {
         umask(prev);
         euidReal();
         log<software_fatal>({__FILE__, __LINE__, errno, "mkfifo failed"});
         log<text_log>("Failed to create input FIFO.", logLevels::FATAL);
         return -1;
      }
   }

   errno = 0;
   if(mkfifo(m_driverOutName.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP) !=0 )
   {
      if(errno != EEXIST)
      {
         umask(prev);
         euidReal();
         log<software_fatal>({__FILE__, __LINE__, errno, "mkfifo failed"});
         log<text_log>("Failed to create ouput FIFO.", logLevels::FATAL);
         return -1;
      }
   }

   umask(prev);
   euidReal();
   return 0;
}

inline
int MagAOXApp::startINDI()
{
   if(!m_useINDI) return 0;
   
   
   //===== Create the FIFOs for INDI communications ====
   if(createINDIFIFOS() < 0)
   {
      return -1;
   }

   //===== Create dummy conf file for libcommon to ignore
   //First create a unique filename with up to 12 chars of m_configName in it.
   char dummyConf[] = {"/tmp/XXXXXXXXXXXXXXXXXX"};
   for(int c=0; c< 12; ++c)
   {
      if(c > m_configName.size()-1) break;
      dummyConf[5+c] = m_configName[c];
   }
   int touch = mkstemp(dummyConf); //mkstemp replaces extra Xs.

   if( touch < 0) //Check if that failed.
   {
      log<software_fatal>({__FILE__, __LINE__, errno, "mkstemp failed"});
      log<text_log>("Failed to create dummy config file for pcf::Config.", logLevels::FATAL);
      return -1;
   }

   //Initialize the libcommon Config system with the empty dummy
   pcf::Config::init( "/" , dummyConf ); //This just gets ignored since it is empty.

   //======= Instantiate the indiDriver
   try
   {
      m_indiDriver = new indiDriver<MagAOXApp>(this, m_configName, "0", "0");
   }
   catch(...)
   {
      //Try to clean up the dummy config
      ::close(touch);
      remove(dummyConf);

      log<software_fatal>({__FILE__, __LINE__, 0, "INDI Driver construction exception."});
      return -1;
   }


   //clean up and delete the dummy config file.
   ::close(touch);
   remove(dummyConf);

   //Check for INDI failure
   if(m_indiDriver == nullptr)
   {
      log<software_fatal>({__FILE__, __LINE__, 0, "INDI Driver construction failed."});
      return -1;
   }

   //Check for INDI failure to open the FIFOs
   if(m_indiDriver->good() == false)
   {
      log<software_fatal>({__FILE__, __LINE__, 0, "INDI Driver failed to open FIFOs."});
      delete m_indiDriver;
      m_indiDriver = nullptr;
      return -1;
   }

   //======= Now we start talkin'
   m_indiDriver->activate();
   log<indidriver_start>();

   return 0;
}

inline
void MagAOXApp::handleGetProperties( const pcf::IndiProperty &ipRecv )
{
   if(m_indiDriver == nullptr) return;

   //Ignore if not our device
   if (ipRecv.hasValidDevice() && ipRecv.getDevice() != m_indiDriver->getName())
   {
      return;
   }

   //Send all properties if requested.
   if( !ipRecv.hasValidName() )
   {
      callBackIterator it = m_indiCallBacks.begin();

      while(it != m_indiCallBacks.end() )
      {
         m_indiDriver->sendDefProperty( *(it->second.property) );
         ++it;
      }

      return;
   }

   //Otherwise send just the requested property.
   m_indiDriver->sendDefProperty( *(m_indiCallBacks[ ipRecv.getName() ].property) );

   return;
}

inline
void MagAOXApp::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
   if(m_indiDriver == nullptr) return;

   int (*newCallBack)(void *, const pcf::IndiProperty &) = m_indiCallBacks[ ipRecv.getName() ].newCallBack;

   if(newCallBack) newCallBack( this, ipRecv);

   return;
}

inline
std::string MagAOXApp::configName()
{
   return m_configName;
}

inline
std::string MagAOXApp::driverInName()
{
   return m_driverInName;
}

inline
std::string MagAOXApp::driverOutName()
{
   return m_driverOutName;
}

} //namespace app
} //namespace MagAOX

#endif //app_MagAOXApp_hpp
