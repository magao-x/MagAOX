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

#include "../logger/logFileRaw.hpp"
#include "../logger/logManager.hpp"



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
template< bool _useINDI = true >
class MagAOXApp : public mx::application
{

public:

   ///The log manager type.
   typedef logger::logManager<logFileRaw> logManagerT;

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
              const bool git_modified       ///< [in] Whether or not the repo is modified.
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
   /** This is called after signal handling is installed.  FSM state is
     * stateCodes::INITIALIZED when this is called.
     *
     * Set m_shutdown = 1 on any fatal errors here.
     */
   virtual int appStartup() = 0;

   /// This is where derived applications implement their main FSM logic.
   /** This will be called every loopPause nanoseconds until the application terminates.
     *
     * FSM state will be whatever it is on exti from appStartup.
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
public:
   static logManagerT m_log;

   /// Make a log entry
   /** Wrapper for logManager::log
     */
   template<typename logT>
   static void log( const typename logT::messageT & msg, ///< [in] the message to log
             logPrioT level = logPrio::LOG_DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );

   /// Make a log entry
   /** Wrapper for logManager::log
     */
   template<typename logT>
   static void log( logPrioT level = logPrio::LOG_DEFAULT /**< [in] [optional] the log level.  The default is used if not specified.*/);

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
     * If prio < 0, it is changed to 0.  If prio is > 99, then it is changed to 99.
     *
     * \returns 0 on success.
     * \returns -1 on an error.  In this case priority will not have been changed.
     */
   int RTPriority( int prio /**< [in] the desired new RT priority */ );

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

   /// Updates and returns the value of m_stateLogged.  Will be 0 on first call after a state change, \>0 afterwards.
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
     * For reference: "Get" and "New" refer to properties we own. "Set" refers to properties owned by others.
     * So we respond to GetProperties by listing our own properties, and NewProperty is a request to change
     * a property we own.  Whereas SetProperty is a notification that someone else has changed a property.
     *
     * @{
     */
protected:

   ///Flag controlling whether INDI is used.  If false, then no INDI code ipRecv.getName()executes.
   constexpr static bool m_useINDI = _useINDI;

   ///The INDI driver wrapper.  Constructed and initialized by execute, which starts and stops communications.
   indiDriver<MagAOXApp> * m_indiDriver {nullptr};

   ///Mutex for locking INDI communications.
   std::mutex m_indiMutex;

   ///Structure to hold the call-back details for handling INDI communications.
   struct indiCallBack
   {
      pcf::IndiProperty * property {0}; ///< A pointer to an INDI property.
      int (*callBack)( void *, const pcf::IndiProperty &) {0}; ///< The function to call for a new or set property.
      bool m_defReceived {false}; ///< Flag indicating that a DefProperty has been received after a GetProperty.
   };

   ///Map to hold the NewProperty indiCallBacks for this App, with fast lookup by property name.
   /** The key for these is the property name.
     */
   std::unordered_map< std::string, indiCallBack> m_indiNewCallBacks;

   ///Map to hold the SetProperty indiCallBacks for this App, with fast lookup by property name.
   /** The key for these is device.name
     */
   std::unordered_map< std::string, indiCallBack> m_indiSetCallBacks;

   ///Flat indicating that all registered Set properties have been updated since last Get.
   bool m_allDefsReceived {false};

   ///Value type of the indiCallBack map.
   typedef std::pair<std::string, indiCallBack> callBackValueType;

   ///Iterator type of the indiCallBack map.
   typedef typename std::unordered_map<std::string, indiCallBack>::iterator callBackIterator;

   ///Return type of insert on the indiCallBack map.
   typedef std::pair<callBackIterator,bool> callBackInsertResult;

   ///Full path name of the INDI driver input FIFO.
   std::string m_driverInName;

   ///Full path name of the INDI driver output FIFO.
   std::string m_driverOutName;

   /// Register an INDI property which is exposed for others to request a New Property for.
   /**
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     * \todo needs error logging
     * \todo needs exception handling
     * \todo is a failure to register a FATAL error?
     */
   int registerIndiPropertyNew( pcf::IndiProperty & prop,                               ///< [out] the property to register
                                const std::string & propName,                           ///< [in] the name of the property
                                const pcf::IndiProperty::Type & propType,               ///< [in] the type of the property
                                const pcf::IndiProperty::PropertyPermType & propPerm,   ///< [in] the permissions of the property
                                const pcf::IndiProperty::PropertyStateType & propState, ///< [in] the state of the property
                                int (*)( void *, const pcf::IndiProperty &)             ///< [in] the callback for changing the property
                              );

   /// Register an INDI property which is monitored for updates from others.
   /**
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     * \todo needs error logging
     * \todo needs exception handling
     * \todo is a failure to register a FATAL error?
     */
   int registerIndiPropertySet( pcf::IndiProperty & prop,                   ///< [out] the property to register
                                const std::string & devName,                ///< [in] the device which owns this property
                                const std::string & propName,               ///< [in] the name of the property
                                int (*)( void *, const pcf::IndiProperty &) ///< [in] the callback for processing the property change
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

   void sendGetPropertySetList(bool all=false);

   /// Handler for the DEF INDI properties notification
   /** Uses the properties registered in m_indiSetCallBacks to process the notification.  This is called by
     * m_indiDriver's indiDriver::handleDefProperties.
     */
   void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] The property being sent. */ );

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

   /// Handler for the set INDI property request
   /**
     *
     * This is called by m_indiDriver's indiDriver::handleSetProperties.
     *
     * \todo handle errors, are they FATAL?
     */
   void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] The property being changed. */);

protected:

   /// Update an INDI property element value if it has changed.
   /** Will only peform a SetProperty if the new element value has changed
     * compared to the stored value.  This comparison is done in the true
     * type of the value.
     */
   template<typename T>
   void updateIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                         const std::string & el, ///< [in] The element name
                         const T & newVal ///< [in] the new value
                      );


   ///indi Property to report the application state.
   pcf::IndiProperty m_indiP_state;

   ///@} --INDI Interface

   /** \name Power Management 
     * For devices which have remote power management (e.g. from one of the PDUs) we implement
     * a standard power state monitoring and management component for the FSM.  This is only 
     * enabled if the power management configuration options are set.
     * 
     * If power management is enabled, then while power is off, appLogic will not be called.
     * Instead a parrallel set of virtual functions is called, onPowerOff (to allow apps to 
     * perform cleanup) and whilePowerOff (to allow apps to keep variables updated, etc).
     * Note that these could merely call appLogic if desired.
     * 
     */
protected:
   bool m_powerMgtEnabled {false};
   
   std::string m_powerDevice;
   std::string m_powerOutlet;
   std::string m_powerElement {"state"};
 
   int m_powerState {-1}; ///< Current power state, 1=On, 0=Off, -1=Unk.
   
   pcf::IndiProperty m_indiP_powerOutlet;
   
   /// This method is called when the change to poweroff is detected.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */ 
   virtual int onPowerOff() {return 0;}
   
   /// This method is called while the power is off, once per FSM loop.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */ 
   virtual int whilePowerOff() {return 0;}
   
public:
   
   INDI_SETCALLBACK_DECL(MagAOXApp, m_indiP_powerOutlet);
   
   ///@} Power Management
   
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
template<bool _useINDI> MagAOXApp<_useINDI> * MagAOXApp<_useINDI>::m_self = nullptr;

//Define the logger
template<bool _useINDI> typename MagAOXApp<_useINDI>::logManagerT MagAOXApp<_useINDI>::m_log;

template<bool _useINDI>
MagAOXApp<_useINDI>::MagAOXApp( const std::string & git_sha1,
                                const bool git_modified
                              )
{
   if( m_self != nullptr )
   {
      std::cerr << "Attempt to instantiate 2nd MagAOXApp.  Exiting immediately.\n";
      exit(-1);
   }

   m_self = this;

   //We log the current GIT status.
   logPrioT gl = logPrio::LOG_INFO;
   if(git_modified) gl = logPrio::LOG_WARNING;
   log<git_state>(git_state::messageT("MagAOX", git_sha1, git_modified), gl);

   gl = logPrio::LOG_INFO;
   if(MXLIB_UNCOMP_REPO_MODIFIED) gl = logPrio::LOG_WARNING;
   log<git_state>(git_state::messageT("mxlib", MXLIB_UNCOMP_CURRENT_SHA1, MXLIB_UNCOMP_REPO_MODIFIED), gl);

   //Get the uids of this process.
   getresuid(&m_euidReal, &m_euidCalled, &m_suid);
   euidReal(); //immediately step down to unpriveleged uid.

}

template<bool _useINDI>
MagAOXApp<_useINDI>::~MagAOXApp() noexcept(true)
{
   if(m_indiDriver) delete m_indiDriver;

   MagAOXApp<_useINDI>::m_self = nullptr;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::setDefaults( int argc,
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

   //We use mx::application's configPathLocal for this component's config file
   configPathLocal = configDir + "/" + m_configName + ".conf";

   //Now we can setup common INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_state, "state", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_state.add (pcf::IndiElement("current"));


   return;

}

template<bool _useINDI>
void MagAOXApp<_useINDI>::setupBasicConfig() //virtual
{
   //App stuff
   config.add("loopPause", "p", "loopPause", mx::argType::Required, "", "loopPause", false, "unsigned long", "The main loop pause time in ns");
   config.add("RTPriority", "P", "RTPriority", mx::argType::Required, "", "RTPriority", false, "unsigned", "The real-time priority (0-99)");

   //Logger Stuff
   m_log.setupConfig(config);

   //Power Management
   config.add("power.device", "", "power.device", mx::argType::Required, "power", "device", false, "string", "Device controlling power for this app's device (INDI name).");
   config.add("power.outlet", "", "power.outlet", mx::argType::Required, "power", "outlet", false, "string", "Outlet (or channel) on device for this app's device (INDI name).");
   config.add("power.element", "", "power.element", mx::argType::Required, "power", "element", false, "string", "INDI element name.  Default is \"state\", only need to specify if different.");
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::loadBasicConfig() //virtual
{
   //---------- Setup the logger ----------//
   m_log.logName(m_configName);
   m_log.loadConfig(config);

   //--------- Loop Pause Time --------//
   config(loopPause, "loopPause");

   //--------- RT Priority ------------//
   int prio = m_RTPriority;
   config(prio, "RTPriority");
   if(prio != m_RTPriority)
   {
      RTPriority(prio);
   }
   
   //--------Power Management --------//
   config(m_powerDevice, "power.device");
   config(m_powerOutlet, "power.outlet");
   config(m_powerElement, "power.element");
   
   if(m_powerDevice != "" && m_powerOutlet != "")
   {
      log<text_log>("enabling power management: " + m_powerDevice + "." + m_powerOutlet + "." + m_powerElement);
      
      m_powerMgtEnabled = true;
      REG_INDI_SETPROP(m_indiP_powerOutlet, m_powerDevice, m_powerOutlet);
   }
}



template<bool _useINDI>
int MagAOXApp<_useINDI>::execute() //virtual
{
   if( lockPID() < 0 )
   {
      state(stateCodes::FAILURE);
      log<text_log>({"Failed to lock PID."}, logPrio::LOG_CRITICAL);
      //Return immediately, not safe to go on.
      return -1;
   }

   //----------------------------------------//
   //        Begin the logger
   //----------------------------------------//
   m_log.logThreadStart();

   //Give up to 2 secs to make sure log thread has time to get started and try to open a file.
   for(int w=0;w<4;++w)
   {
      //Sleep for 500 msec
      std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(500000));

      //Verify that log thread is still running.
      if(m_log.logThreadRunning() == true) break;
   }

   if(m_log.logThreadRunning() == false)
   {
      //We don't log this, because it won't be logged anyway.
      std::cerr << "\nCRITICAL: log thread not running.  Exiting.\n\n";
        m_shutdown = 1;
   }

   //----------------------------------------//

   setSigTermHandler();

   if( m_shutdown == 0 )
   {
      state(stateCodes::INITIALIZED);
      if(appStartup() < 0) m_shutdown = 1;
   }

   //====Begin INDI Communications
   if(m_useINDI && m_shutdown == 0) //if we're using INDI and not already dead, that is
   {
      if(startINDI() < 0)
      {
         state(stateCodes::FAILURE);
         m_shutdown = 1;
      }
   }

   //This is the main event loop.
   while( m_shutdown == 0)
   {
      
      if(m_powerMgtEnabled)
      {
         if(state() == stateCodes::POWEROFF)
         {
            if(m_powerState == 1)
            {
               state(stateCodes::POWERON);
            }
         }
         else
         {
            if(m_powerState < 1)
            {
               state(stateCodes::POWEROFF);
               if(onPowerOff() < 0)
               {
                  m_shutdown = 1;
                  continue;
               }
               
            }
         }
      }
      else if(state() == stateCodes::POWEROFF)
      {
         //If power management is not enabled there's no way to get out of power off, so we just go
         state(stateCodes::POWERON);
      }
      
      if(state() != stateCodes::POWEROFF) 
      {
         if( appLogic() < 0) 
         {
            m_shutdown = 1;
            continue;
         }
      }
      
      /** \todo Need a heartbeat update here.
        */

      if(m_useINDI)
      {
         //Checkup on the INDI properties we're monitoring.
         //This will make sure we are up-to-date if indiserver restarts without us.
         //And handles cases where we miss a Def becuase the other driver wasn't started up
         //when we sent our Get.
         sendGetPropertySetList(false); //Only does anything if it needs to be done.
      }

      // This is purely to make sure INDI is up to date in case
      // mutex was locked on last attempt.
      state( state() );
      
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

template<bool _useINDI>
template<typename logT>
void MagAOXApp<_useINDI>::log( const typename logT::messageT & msg,
                               logPrioT level
                             )
{
   m_log.log<logT>(msg, level);
}

template<bool _useINDI>
template<typename logT>
void MagAOXApp<_useINDI>::log( logPrioT level)
{
   m_log.log<logT>(level);
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::setSigTermHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &MagAOXApp<_useINDI>::_handlerSigTerm;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGTERM, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGTERM failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGQUIT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGQUIT failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0,logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGINT, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGINT failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   log<text_log>("Installed SIGTERM/SIGQUIT/SIGINT signal handler.", logPrio::LOG_DEBUG);

   return 0;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::_handlerSigTerm( int signum,
                                           siginfo_t *siginf,
                                           void *ucont
                                         )
{
   m_self->handlerSigTerm(signum, siginf, ucont);
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::handlerSigTerm( int signum,
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

template<bool _useINDI>
int MagAOXApp<_useINDI>::euidCalled()
{
   errno = 0;
   if(seteuid(m_euidCalled) < 0)
   {
      std::string logss = "Setting effective user id to euidCalled (";
      logss += mx::ioutils::convertToString<int>(m_euidCalled);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::euidReal()
{
   errno = 0;
   if(seteuid(m_euidReal) < 0)
   {
      std::string logss = "Setting effective user id to euidReal (";
      logss += mx::ioutils::convertToString<int>(m_euidReal);
      logss += ") failed.  Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   return 0;

}

template<bool _useINDI>
int MagAOXApp<_useINDI>::RTPriority( int prio)
{
   struct sched_param schedpar;

   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;
   schedpar.sched_priority = prio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, 0,"Seeting euid to called failed."});
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
      log<software_error>({__FILE__, __LINE__, errno, 0, logss.str()});
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
      log<software_error>({__FILE__, __LINE__, 0, 0, "Setting euid to real failed."});
      return -1;
   }

   return rv;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::lockPID()
{
   m_pid = getpid();

   std::string statusDir = sysPath;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, 0, "Seeting euid to called failed."});
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
         log<software_critical>({__FILE__, __LINE__, errno, 0, logss.str()});

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
         log<software_critical>({__FILE__, __LINE__, errno, 0, logss.str()});

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
            log<software_critical>({__FILE__, __LINE__, 0, 0, "exception caught testing /proc/pid"});
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

            log<text_log>(logss.str(), logPrio::LOG_CRITICAL);

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
      log<software_critical>({__FILE__, __LINE__, errno, 0, "could not open pid file for writing."});
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
      log<software_error>({__FILE__, __LINE__, 0, 0, "Seeting euid to real failed."});
      return -1;
   }

   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::unlockPID()
{
   if( ::remove(pidFileName.c_str()) < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, 0, std::string("Failed to remove PID file: ") + strerror(errno)});
      return -1;
   }

   std::stringstream logss;
   logss << "PID (" << m_pid << ") unlocked.";
   log<text_log>(logss.str());

   return 0;
}

template<bool _useINDI>
stateCodes::stateCodeT MagAOXApp<_useINDI>::state()
{
   return m_state;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::state(const stateCodes::stateCodeT & s)
{
   //Only do anything if it's a change
   if(m_state != s)
   {
      logPrioT lvl = logPrio::LOG_INFO;
      if(s == stateCodes::ERROR) lvl = logPrio::LOG_ERROR;
      if(s == stateCodes::FAILURE) lvl = logPrio::LOG_CRITICAL;

      log<state_change>( {m_state, s}, lvl );

      m_state = s;
      m_stateLogged = 0;
   }
   
   //Check to make sure INDI is up to date
   std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);  //Lock the mutex before conducting INDI communications.
   
   if(lock.owns_lock())
   {
      updateIfChanged(m_indiP_state, "current", m_state);
   }
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::stateLogged()
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

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertyNew( pcf::IndiProperty & prop,
                                                  const std::string & propName,
                                                  const pcf::IndiProperty::Type & propType,
                                                  const pcf::IndiProperty::PropertyPermType & propPerm,
                                                  const pcf::IndiProperty::PropertyStateType & propState,
                                                  int (*callBack)( void *, const pcf::IndiProperty &ipRecv)
                                                )
{
   if(!m_useINDI) return 0;

   prop = pcf::IndiProperty (propType);
   prop.setDevice(m_configName);
   prop.setName(propName);
   prop.setPerm(propPerm);
   prop.setState( propState);


   callBackInsertResult result =  m_indiNewCallBacks.insert(callBackValueType( propName, {&prop, callBack}));

   if(!result.second)
   {
      return -1;
   }

   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertySet( pcf::IndiProperty & prop,
                                                  const std::string & devName,
                                                  const std::string & propName,
                                                  int (*callBack)( void *, const pcf::IndiProperty &ipRecv)
                                                )
{
   if(!m_useINDI) return 0;

   prop = pcf::IndiProperty();
   prop.setDevice(devName);
   prop.setName(propName);

   callBackInsertResult result =  m_indiSetCallBacks.insert(callBackValueType( devName + "." + propName, {&prop, callBack}));

   if(!result.second)
   {
      return -1;
   }

   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createINDIFIFOS()
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
         log<software_critical>({__FILE__, __LINE__, errno, 0, "mkfifo failed"});
         log<text_log>("Failed to create input FIFO.", logPrio::LOG_CRITICAL);
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
         log<software_critical>({__FILE__, __LINE__, errno, 0, "mkfifo failed"});
         log<text_log>("Failed to create ouput FIFO.", logPrio::LOG_CRITICAL);
         return -1;
      }
   }

   umask(prev);
   euidReal();
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::startINDI()
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
   for(size_t c=0; c< 12; ++c)
   {
      if(c > m_configName.size()-1) break;
      dummyConf[5+c] = m_configName[c];
   }
   int touch = mkstemp(dummyConf); //mkstemp replaces extra Xs.

   if( touch < 0) //Check if that failed.
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "mkstemp failed"});
      log<text_log>("Failed to create dummy config file for pcf::Config.", logPrio::LOG_CRITICAL);
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

      log<software_critical>({__FILE__, __LINE__, 0, 0, "INDI Driver construction exception."});
      return -1;
   }


   //clean up and delete the dummy config file.
   ::close(touch);
   remove(dummyConf);

   //Check for INDI failure
   if(m_indiDriver == nullptr)
   {
      log<software_critical>({__FILE__, __LINE__, 0, 0, "INDI Driver construction failed."});
      return -1;
   }

   //Check for INDI failure to open the FIFOs
   if(m_indiDriver->good() == false)
   {
      log<software_critical>({__FILE__, __LINE__, 0, 0, "INDI Driver failed to open FIFOs."});
      delete m_indiDriver;
      m_indiDriver = nullptr;
      return -1;
   }

   //======= Now we start talkin'
   m_indiDriver->activate();
   log<indidriver_start>();

   sendGetPropertySetList();

   return 0;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::sendGetPropertySetList(bool all)
{
   //Unless forced by all, we only do anything if allDefs are not received yet
   if(!all && m_allDefsReceived) return;

   callBackIterator it = m_indiSetCallBacks.begin();

   int nowFalse = 0;
   while(it != m_indiSetCallBacks.end() )
   {
      if(all || it->second.m_defReceived == false)
      {
         if( it->second.property )
         {
            m_indiDriver->sendGetProperties( *(it->second.property) );
         }

         it->second.m_defReceived = false;
         ++nowFalse;

      }
      ++it;
   }
   if(nowFalse != 0) m_allDefsReceived = false;
   if(nowFalse == 0) m_allDefsReceived = true;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
   handleSetProperty(ipRecv); //We have the same response to both Def and Set.
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::handleGetProperties( const pcf::IndiProperty &ipRecv )
{
   if(!m_useINDI) return;
   if(m_indiDriver == nullptr) return;

   //Ignore if not our device
   if (ipRecv.hasValidDevice() && ipRecv.getDevice() != m_indiDriver->getName())
   {
      return;
   }

   //Send all properties if requested.
   if( !ipRecv.hasValidName() )
   {
      callBackIterator it = m_indiNewCallBacks.begin();

      while(it != m_indiNewCallBacks.end() )
      {
         if( it->second.property )
         {
            m_indiDriver->sendDefProperty( *(it->second.property) );
         }
         ++it;
      }

      //This is a possible INDI server restart, so we re-register for all notifications.
      sendGetPropertySetList(true);

      return;
   }

   //Check if we actually have this.
   if( m_indiNewCallBacks.count(ipRecv.getName()) == 0)
   {
      std::stringstream s;
      s << "Received GetProperty for " << ipRecv.getName();
      log<text_log>(s.str(), logPrio::LOG_ERROR);
      return;
   }

   //Otherwise send just the requested property, if property is not null
   if(m_indiNewCallBacks[ ipRecv.getName() ].property)
   {
      m_indiDriver->sendDefProperty( *(m_indiNewCallBacks[ ipRecv.getName() ].property) );
   }
   return;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
   if(!m_useINDI) return;
   if(m_indiDriver == nullptr) return;

   //Check if this is a valid name for us.
   if( m_indiNewCallBacks.count(ipRecv.getName()) == 0 )
   {
      ///\todo log invalid NewProperty request, though it probably can't get this far.
      return;
   }

   int (*callBack)(void *, const pcf::IndiProperty &) = m_indiNewCallBacks[ ipRecv.getName() ].callBack;

   if(callBack) callBack( this, ipRecv);

   ///\todo log an error here because callBack should not be null

   return;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
   if(!m_useINDI) return;
   if(m_indiDriver == nullptr) return;

   std::string key = ipRecv.getDevice() + "." + ipRecv.getName();

   //Check if this is valid
   if( m_indiSetCallBacks.count(key) > 0 )
   {
      m_indiSetCallBacks[ key ].m_defReceived = true; //record that we got this Def/Set

      //And call the callback
      int (*callBack)(void *, const pcf::IndiProperty &) = m_indiSetCallBacks[ key ].callBack;
      if(callBack) callBack( this, ipRecv);

      ///\todo log an error here because callBack should not be null
   }
   else
   {
      ///\todo log invalid SetProperty request.
   }

   return;
}

template<bool _useINDI>
template<typename T>
void MagAOXApp<_useINDI>::updateIfChanged( pcf::IndiProperty & p,
                                           const std::string & el,
                                           const T & newVal
                                         )
{
   if(!_useINDI) return;
   
   if(!m_indiDriver) return;
   
   T oldVal = p[el].get<T>();

   if(oldVal != newVal)
   {
      p[el].set(newVal);
      p.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (p);
   }
}

template<bool _useINDI>
INDI_SETCALLBACK_DEFN( MagAOXApp<_useINDI>, m_indiP_powerOutlet)(const pcf::IndiProperty &ipRecv)
{
   //m_indiP_powerOutlet = ipRecv;

   std::string ps;
   
   try
   {
      ps = ipRecv[m_powerElement].get<std::string>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }
   
   if(ps == "On")
   {
      m_powerState = 1;
   }
   else if (ps == "Off")
   {
      m_powerState = 0;
   }
   else
   {
      m_powerState = -1;
   }
   
   return 0;
}

template<bool _useINDI>
std::string MagAOXApp<_useINDI>::configName()
{
   return m_configName;
}

template<bool _useINDI>
std::string MagAOXApp<_useINDI>::driverInName()
{
   return m_driverInName;
}

template<bool _useINDI>
std::string MagAOXApp<_useINDI>::driverOutName()
{
   return m_driverOutName;
}

} //namespace app
} //namespace MagAOX

#endif //app_MagAOXApp_hpp
