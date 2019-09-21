/** \file magAOXApp.hpp
  * \brief The basic MagAO-X Application
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-12-24 created by JRM
  * 
  * \ingroup app_files
  */

#ifndef app_MagAOXApp_hpp
#define app_MagAOXApp_hpp


#include <signal.h>
#include <sys/stat.h>

#include <cstdlib>
#include <fstream>

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

#include "../sys/thSetuid.hpp"

#include "stateCodes.hpp"
#include "indiDriver.hpp"
#include "indiMacros.hpp"
#include "indiUtils.hpp"

//#include "../../INDI/libcommon/System.hpp"
using namespace mx::app;

using namespace MagAOX::logger;

namespace MagAOX
{
namespace app
{

/// The base-class for MagAO-X applications.
/**
  * You can define a base configuration file for this class by defining
  * \code
    m_configBase = "base_name";
    \endcode
  * in the derived class constructor. This would be used, for instance to have a config common to
  * all filter wheels.
  *
  *
  * \todo implement libMagAOX error handling? (a stack?)
  * \todo make m_powerMgtEnabled a template parameter, and static_assert checki if _useINDI== false and power management is true
  *
  * \ingroup magaoxapp
  */
template< bool _useINDI = true >
class MagAOXApp : public application
{

public:

   ///The log manager type.
   typedef logger::logManager<logFileRaw> logManagerT;

protected:

   std::string MagAOXPath; ///< The base path of the MagAO-X system.

   std::string m_configName; ///< The name of the configuration file (minus .conf).

   std::string m_configDir; ///< The path to configuration files for MagAOX.
      
   std::string m_configBase; ///< The name of a base config class for this app (minus .conf).

   std::string m_calibDir; ///< The path to calibration files for MagAOX.
   
   std::string sysPath;  ///< The path to the system directory, for PID file, etc.

   std::string secretsPath; ///< Path to the secrets directory, where passwords, etc, are stored.

   unsigned long m_loopPause {MAGAOX_default_loopPause}; ///< The time in nanoseconds to pause the main loop.  The appLogic() function of the derived class is called every m_loopPause nanoseconds.  Default is 1,000,000,000 ns.  Config with loopPause=X.

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

   /// Get the value of the shutdown flag.
   /**
     * \returns the current value of m_shutdown 
     */ 
   int shutdown();
   
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
   /** Performs final startup steps.  That is:
     * - PID locking lockPID()
     * - log thread startup by logThreadStart()
     * - signal handling installation by setSigTermHandler()
     * - appStartup() is called
     * - INDI communications started by startINDI()
     * - power state is checked, pausing if unknown (if being managed)
     *
     * Errors in the above steps will cause a process exit.
     *
     * Then commences the main event loop.
     * Conditions on entry to the main loop:
     * - PID locked
     * - Log thread running
     * - Signal handling installed
     * - appStartup successful
     * - INDI communications started successfully (if being used)
     * - power state known (if being managed)
     *
     * In the event loop, the power state is checked (if being managed).  If power is off, then onPowerOff is called.
     * If power is on, or power is not managed, appLogic is called.  These methods are implemented in derived classes, and
     * are called every m_loopPause interval.
     *
     * If an error is returned by either onPowerOff or appLogic, or a signal is handled, then the shutdown is managed.
     * This includes shutting down INDI, calling appShutdown, and unlocking the PID.  The log thread will shutdown.
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
   /** This will be called every m_loopPause nanoseconds until the application terminates.
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
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
   template<typename logT, int retval=0>
   static int log( const typename logT::messageT & msg, ///< [in] the message to log
                   logPrioT level = logPrio::LOG_DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
                 );

   /// Make a log entry
   /** Wrapper for logManager::log
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
   template<typename logT, int retval=0>
   static int log( logPrioT level = logPrio::LOG_DEFAULT /**< [in] [optional] the log level.  The default is used if not specified.*/);

private:
   /// Callback for config system logging.
   /** Called by appConfigurator each time a value is set using the config() operator.
     * You never need to call this directly.
     */
   static void configLog( const std::string & name,  ///< [in] The name of the config value
                          const int & code,          ///< [in] numeric code specifying the type
                          const std::string & value, ///< [in] the value read by the config system
                          const std::string & source ///< [in] the source of the value.
                        );

   ///@} -- logging

   /** \name Signal Handling
     * @{libMagAOX/logger/types/software_log.hpp
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

   /// Internal class to manage setuid privilege escalation with RAII
   /** Upon construction this elevates to the called user id, root in a setuid process.
     * Restores privileges to real user id upon destruction (i.e. when it goes out of scope).
     */ 
   class elevatedPrivileges
   {
      private:
         MagAOXApp * m_app;
         bool m_elevated {false};
         
      public:
         elevatedPrivileges(MagAOXApp * app)
         {
            m_app = app;
            elevate();
         }
      
         void elevate()
         {
            if(m_elevated) return;
          
            m_app->euidCalled();
            m_elevated = true;
         }
         
         void restore()
         {
            if(!m_elevated) return;
            
            m_app->euidReal();
            m_elevated = false;
         }
         
         ~elevatedPrivileges()
         {
            restore();
         }
   };
   
   /// Set the effective user ID to the called value, i.e. the highest possible.
   /** If setuid is set on the file, this will be super-user privileges.
     *
     * Reference: http://pubs.opengroup.org/onlinepubs/009695399/functions/seteuid.html
     *
     * \todo make this private, and change name to enforce use of the above class.
     * 
     * \returns 0 on success
     * \returns -1 on error from setuid().
     */
   int euidCalled();

   /// Set the effective user ID to the real value, i.e. the file owner.
   /**
     * Reference: http://pubs.opengroup.org/onlinepubs/009695399/functions/seteuid.html
     *
     * \todo make this private, and change name to enforce use of the above class.
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

   /** \name Threads 
     *
     * @{
     */
public:
   
   /// Start a thread, using this class's privileges to set priority, etc.
   /** 
     * The thread initialization synchronizer `bool` is set to true at the beginning
     * of this function, then is set to false once all initialization is complete.  
     *
     * The interface of the thread start function is:
     \code
     static void impl::myThreadStart( impl * o )
     {
        o->myThreadExec(); //A member function which actually exectues the thread
     }
     \endcode
     * where `impl` is the derived class, and `mThreadStart` and `myThreadExec` are members
     * of `impl`.
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   template<class thisPtr, class Function>
   int threadStart( std::thread & thrd,           ///< [out] The thread object to start executing
                    bool & thrdInit,              ///< [in/out] The thread initilization synchronizer.  
                    int thrdPrio,                 ///< [in] The r/t priority to set for this thread
                    const std::string & thrdName, ///< [in] The name of the thread (just for logging)
                    thisPtr * thrdThis,           ///< [in] The `this` pointer to pass to the thread starter function
                    Function&& thrdStart          ///< [in] The thread starting function, a static function taking a `this` pointer as argument.
                  );
   
   ///@} -- Threads
   
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

///\todo instead of making this public, provide an accessor.
public:
   ///The INDI driver wrapper.  Constructed and initialized by execute, which starts and stops communications.
   indiDriver<MagAOXApp> * m_indiDriver {nullptr};

   ///Mutex for locking INDI communications.
   std::mutex m_indiMutex;
   
protected:
   ///Structure to hold the call-back details for handling INDI communications.
   struct indiCallBack
   {
      pcf::IndiProperty * property {0}; ///< A pointer to an INDI property.
      int (*callBack)( void *, const pcf::IndiProperty &) {0}; ///< The function to call for a new or set property.
      bool m_defReceived {false}; ///< Flag indicating that a DefProperty has been received after a GetProperty.
   };


public:

   ///Value type of the indiCallBack map.
   typedef std::pair<std::string, indiCallBack> callBackValueType;

   ///Iterator type of the indiCallBack map.
   typedef typename std::unordered_map<std::string, indiCallBack>::iterator callBackIterator;

   ///Return type of insert on the indiCallBack map.
   typedef std::pair<callBackIterator,bool> callBackInsertResult;

protected:
   ///Map to hold the NewProperty indiCallBacks for this App, with fast lookup by property name.
   /** The key for these is the property name.
     */
   std::unordered_map< std::string, indiCallBack> m_indiNewCallBacks;

   ///Map to hold the SetProperty indiCallBacks for this App, with fast lookup by property name.
   /** The key for these is device.name
     */
   std::unordered_map< std::string, indiCallBack> m_indiSetCallBacks;

protected:
   ///Flag indicating that all registered Set properties have been updated since last Get.
   bool m_allDefsReceived {false};

   ///Full path name of the INDI driver input FIFO.
   std::string m_driverInName;

   ///Full path name of the INDI driver output FIFO.
   std::string m_driverOutName;

   ///Full path name of the INDI driver control FIFO.
   /** This is currently only used to signal restarts.
     */
   std::string m_driverCtrlName;

public:

   /// Create a standard R/W INDI Text property with target and current elements.
   /**
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createStandardIndiText( pcf::IndiProperty & prop,       ///< [out] the property to create and setup
                               const std::string & propName,   ///< [in] the name of the property
                               const std::string & label = "", ///< [in] [optional] the GUI label suggestion for this property
                               const std::string & group = ""  ///< [in] [optional] the group for this property
                             );
   
   /// Create a standard ReadOnly INDI Text property, with at least one element.
   /**
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createROIndiText( pcf::IndiProperty & prop,           ///< [out] the property to create and setup
                         const std::string & propName,       ///< [in] the name of the property
                         const std::string & elName,         ///< [in] the name of the element
                         const std::string & propLabel = "", ///< [in] [optional] the GUI label suggestion for this property
                         const std::string & propGroup = "", ///< [in] [optional] the group for this property
                         const std::string & elLabel = ""    ///< [in] [optional] the GUI label suggestion for the element
                       );
   
   /// Create a standard R/W INDI Number property with target and current elements.
   /**
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   template<typename T>
   int createStandardIndiNumber( pcf::IndiProperty & prop,       ///< [out] the property to create and setup
                                 const std::string & name,       ///< [in] the name of the property
                                 const T & min,                  ///< [in] the minimum value for the elements, applied to both target and current
                                 const T & max,                  ///< [in] the minimum value for the elements, applied to both target and current
                                 const T & step,                 ///< [in] the step size for the elements, applied to both target and current
                                 const std::string & format,     ///< [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
                                 const std::string & label = "", ///< [in] [optional] the GUI label suggestion for this property
                                 const std::string & group = ""  ///< [in] [optional] the group for this property
                               );
   
   /// Create a ReadOnly INDI Number property
   /**
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createROIndiNumber( pcf::IndiProperty & prop,           ///< [out] the property to create and setup
                           const std::string & propName,       ///< [in] the name of the property
                           const std::string & propLabel = "", ///< [in] [optional] the GUI label suggestion for this property
                           const std::string & propGroup = ""  ///< [in] [optional] the group for this property
                         );
   
   /// Create a standard R/W INDI switch with a single toggle element.
   /** This switch is intended to function like an on/off toggle switch.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createStandardIndiToggleSw( pcf::IndiProperty & prop,       ///< [out] the property to create and setup
                                   const std::string & name,       ///< [in] the name of the property
                                   const std::string & label = "", ///< [in] [optional] the GUI label suggestion for this property
                                   const std::string & group = ""  ///< [in] [optional] the group for this property
                                 );   
   
   /// Create a standard R/W INDI switch with a single request element.
   /** This switch is intended to function like a momentary switch.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createStandardIndiRequestSw( pcf::IndiProperty & prop,       ///< [out] the property to create and setup
                                    const std::string & name,       ///< [in] the name of the property
                                    const std::string & label = "", ///< [in] [optional] the GUI label suggestion for this property
                                    const std::string & group = ""  ///< [in] [optional] the group for this property
                                  );
   
   /// Create a standard R/W INDI selection (one of many) switch with vector of elements
   /** This switch is intended to function like drop down menu.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int createStandardIndiSelectionSw( pcf::IndiProperty & prop,                  ///< [out] the property to create and setup
                                      const std::string & name,                  ///< [in] the name of the property,
                                      const std::vector<std::string> & elements, ///< [in] the element names to give to the switches
                                      const std::string & label = "",            ///< [in] [optional] the GUI label suggestion for this property
                                      const std::string & group = ""             ///< [in] [optional] the group for this property
                                    );

   /// Register an INDI property which is read only.
   /** This version requires the property be fully set up.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertyReadOnly( pcf::IndiProperty & prop /**< [in] the property to register, must be completely setup */ );

   /// Register an INDI property which is read only.
   /** This verison sets up the INDI property according to the arguments.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertyReadOnly( pcf::IndiProperty & prop,                              ///< [out] the property to register, will be configured
                                     const std::string & propName,                          ///< [in] the name of the property
                                     const pcf::IndiProperty::Type & propType,              ///< [in] the type of the property
                                     const pcf::IndiProperty::PropertyPermType & propPerm,  ///< [in] the permissions of the property
                                     const pcf::IndiProperty::PropertyStateType & propState ///< [in] the state of the property
                                   );
   
   /// Register an INDI property which is exposed for others to request a New Property for.
   /** In this version the supplied IndiProperty must be fully set up before passing in.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertyNew( pcf::IndiProperty & prop,                   ///< [in] the property to register, must be fully set up
                                int (*)( void *, const pcf::IndiProperty &) ///< [in] the callback for changing the property
                              );
   
   /// Register an INDI property which is exposed for others to request a New Property for.
   /** This verison sets up the INDI property according to the arguments.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertyNew( pcf::IndiProperty & prop,                               ///< [out] the property to register
                                const std::string & propName,                           ///< [in] the name of the property
                                const pcf::IndiProperty::Type & propType,               ///< [in] the type of the property
                                const pcf::IndiProperty::PropertyPermType & propPerm,   ///< [in] the permissions of the property
                                const pcf::IndiProperty::PropertyStateType & propState, ///< [in] the state of the property
                                int (*)( void *, const pcf::IndiProperty &)             ///< [in] the callback for changing the property
                              );
   
   /// Register an INDI property which is exposed for others to request a New Property for, with a switch rule
   /** This verison sets up the INDI property according to the arguments.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertyNew( pcf::IndiProperty & prop,                               ///< [out] the property to register
                                const std::string & propName,                           ///< [in] the name of the property
                                const pcf::IndiProperty::Type & propType,               ///< [in] the type of the property
                                const pcf::IndiProperty::PropertyPermType & propPerm,   ///< [in] the permissions of the property
                                const pcf::IndiProperty::PropertyStateType & propState, ///< [in] the state of the property
                                const pcf::IndiProperty::SwitchRuleType & propRule,     ///< [in] the switch rule type
                                int (*)( void *, const pcf::IndiProperty &)             ///< [in] the callback for changing the property
                              );
   
   /// Register an INDI property which is monitored for updates from others.
   /**
     *
     * \returns 0 on success.
     * \returns -1 on error.
     *
     */
   int registerIndiPropertySet( pcf::IndiProperty & prop,                   ///< [out] the property to register
                                const std::string & devName,                ///< [in] the device which owns this property
                                const std::string & propName,               ///< [in] the name of the property
                                int (*)( void *, const pcf::IndiProperty &) ///< [in] the callback for processing the property change
                              );

protected:
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
     * compared to the stored value, or if the property state has changed.  
     * 
     * This comparison is done in the true
     * type of the value.
     * 
     */
   template<typename T>
   void updateIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                         const std::string & el, ///< [in] The element name
                         const T & newVal, ///< [in] the new value
                         pcf::IndiProperty::PropertyStateType ipState = pcf::IndiProperty::Ok
                      );

   /// Update an INDI property element value if it has changed.
   /** Will only peform a SetProperty if the new element value has changed
     * compared to the stored value, or if the property state has changed.  
     * 
     * This comparison is done in the true
     * type of the value.
     * 
     * This is a specialization for `const char *` to `std::string`.
     * 
     * \overload
     */
   void updateIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                         const std::string & el, ///< [in] The element name
                         const char * newVal, ///< [in] the new value
                         pcf::IndiProperty::PropertyStateType ipState = pcf::IndiProperty::Ok
                      );
   
   /// Update an INDI switch element value if it has changed.
   /** Will only peform a SetProperty if the new element switch state has changed, or the propery state
     * has changed.
     * 
     */
   void updateSwitchIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                               const std::string & el, ///< [in] The element name
                               const pcf::IndiElement::SwitchStateType & newVal, ///< [in] the new value
                               pcf::IndiProperty::PropertyStateType ipState = pcf::IndiProperty::Ok
                             );
   
   /** 
     * \overload void updateIfChanged(pcf::IndiProperty & p, const std::string & el, const std::vector<T> & newVal)
     * Takes in a vector of values, such that each element in the vector will have an element name of el+(index of value)
     */
   template<typename T>
   void updateIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                         const std::string & el, ///< [in] Beginning of each element name
                         const std::vector<T> & newVal, ///< [in] the new values
                         pcf::IndiProperty::PropertyStateType ipState = pcf::IndiProperty::Ok
                      );

  /** 
     * \overload void updateIfChanged(pcf::IndiProperty & p, const std::string & el, const std::vector<T> & newVal)
     * Takes in a vector of values, such that each element in the vector will have an element name of el+(index of value)
     */
   template<typename T>
   void updateIfChanged( pcf::IndiProperty & p, ///< [in/out] The property containing the element to possibly update
                         const std::vector<std::string> & el, ///< [in] String vector of element names
                         const std::vector<T> & newVal ///< [in] the new values
                      );

   /// Get the target element value from an new property 
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   template<typename T>
   int indiTargetUpdate( pcf::IndiProperty & localProperty, ///< [out] The local property to update 
                         T & localTarget, ///< [out] The local value to update
                         const pcf::IndiProperty & remoteProperty, ///< [in] the new property received 
                         bool setBusy = true ///< [in] [optional] set property to busy if true
                       );

   /// Send a newProperty command to another device (using the INDI Client interface)
   /** Copies the input IndiProperty, then updates the element with the new value.
     *
     * \returns 0 on success.
     * \returns -1 on an errory.
     */
   template<typename T>
   int sendNewProperty( const pcf::IndiProperty & ipSend, ///< [in] The property to send a "new" INDI command for
                        const std::string & el, ///< [in] The element of the property to change
                        const T & newVal ///< [in] The value to request for the element.
                      );

   ///indi Property to report the application state.
   pcf::IndiProperty m_indiP_state;

   ///@} --INDI Interface

   /** \name Power Management
     * For devices which have remote power management (e.g. from one of the PDUs) we implement
     * a standard power state monitoring and management component for the FSM.  This needs to be enabled
     * in the derived app constructor.  To stay enabled, m_powerDevice and m_powerChannel must be
     * not empty strings after the configuration.  These could be set in the derived app defaults.
     *
     * If power management is enabled, then while power is off, appLogic will not be called.
     * Instead a parrallel set of virtual functions is called, onPowerOff (to allow apps to
     * perform cleanup) and whilePowerOff (to allow apps to keep variables updated, etc).
     * Note that these could merely call appLogic if desired.
     *
     */
protected:
   bool m_powerMgtEnabled {false}; ///< Flag controls whether power mgt is used.  Set this in the constructor of a derived app.  If true, then if after loadConfig the powerDevice and powerChannel are empty, then the app will exit with a critical error.

   /* Configurables . . . */
   std::string m_powerDevice; ///< The INDI device name of the power controller
   std::string m_powerChannel; ///< The INDI property name of the channel controlling this device's power.
   std::string m_powerElement {"state"}; ///< The INDI element name to monitor for this device's power state.
   std::string m_powerTargetElement {"target"}; ///< The INDI element name to monitor for this device's power state.

   unsigned long m_powerOnWait {0}; ///< Time in sec to wait for device to boot after power on.
   
   /* Power on waiting counter . . . */
   int m_powerOnCounter {-1}; ///< Counts numer of loops after power on, implements delay for device bootup.  If -1, then device was not powered off on startup.
   
   /* Power state . . . */
   int m_powerState {-1}; ///< Current power state, 1=On, 0=Off, -1=Unk.
   int m_powerTargetState {-1}; ///< Current target power state, 1=On, 0=Off, -1=Unk.
   
   pcf::IndiProperty m_indiP_powerChannel; ///< INDI property used to communicate power state.

   /// This method is called when the change to poweroff is detected.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
   virtual int onPowerOff();

   /// This method is called while the power is off, once per FSM loop.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
   virtual int whilePowerOff();
   
   /// This method tests whether the power on wait time has elapsed.
   /** You would call this once per appLogic loop while in state POWERON.  While false, you would return 0.  
     * Once it becomes true, take post-power-on actions and go on with life.
     * 
     * \returns true if the time since POWERON is greater than the power-on wait, or if power management is not enabled
     * \returns false otherwise
     */
   bool powerOnWaitElapsed();

public:

   /// Returns the current power state.
   /** If power management is not enabled, this always returns 1=On.
     *
     * \returns -1 if power state is unknown
     * \returns 0 if power is off 
     * \returns 1 if power is on or m_powerMgtEnabled==false
     */ 
   int powerState();
   
   /// Returns the target power state.
   /** If power management is not enabled, this always returns 1=On.
     *
     * \returns -1 if target power state is unknown
     * \returns 0 if target power state is off 
     * \returns 1 if target power is on or m_powerMgtEnabled==false
     */ 
   int powerStateTarget();
   
   INDI_SETCALLBACK_DECL(MagAOXApp, m_indiP_powerChannel);

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

   ///Get the config directory
   /**
     * \returns the current value of m_configDir
     */
   std::string configDir();
   
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

   ///Get the INDI control FIFO file name
   /**
     * \returns the current value of m_driverCtrlName
     */
   std::string driverCtrlName();

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

   //Set up config logging
   config.m_sources = true;
   config.configLog = configLog;

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
int MagAOXApp<_useINDI>::shutdown()
{
   return m_shutdown;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::setDefaults( int argc,
                                       char ** argv
                                     )   //virtual
{
   std::string tmpstr;

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
   m_configDir = MagAOXPath + "/" + tmpstr;
   configPathGlobal = m_configDir + "/magaox.conf";

   //Set the calib path relative to MagAOXPath
   tmpstr = mx::getEnv(MAGAOX_env_calib);
   if(tmpstr == "")
   {
      tmpstr = MAGAOX_calibRelPath;
   }
   m_calibDir = MagAOXPath + "/" + tmpstr;
   
   
   
   //Setup default log path
   tmpstr = MagAOXPath + "/" + MAGAOX_logRelPath;

   m_log.logPath(tmpstr);

   //Setup default sys path
   tmpstr = MagAOXPath + "/" + MAGAOX_sysRelPath;
   sysPath = tmpstr;

   //Setup default secrets path
   tmpstr = MagAOXPath + "/" + MAGAOX_secretsRelPath;
   secretsPath = tmpstr;


   if(m_configBase != "")
   {
      //We use mx::application's configPathUser for this components base config file
      configPathUser = m_configDir + "/" + m_configBase + ".conf";
   }

   //Parse CL just to get the "name".
   config.add("name","n", "name",argType::Required, "", "name", false, "string", "The name of the application, specifies config.");

   config.parseCommandLine(argc, argv, "name");
   config(m_configName, "name");

   if(m_configName == "")
   {
      boost::filesystem::path p(invokedName);
      m_configName = p.stem().string();
      log<text_log>("Application name (-n --name) not set.  Using argv[0].");
   }

   //We use mx::application's configPathLocal for this component's config file
   configPathLocal = m_configDir + "/" + m_configName + ".conf";

   //Now we can setup common INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_state, "fsm", pcf::IndiProperty::Text);
   m_indiP_state.add (pcf::IndiElement("state"));


   return;

}

template<bool _useINDI>
void MagAOXApp<_useINDI>::setupBasicConfig() //virtual
{
   //App stuff
   config.add("loopPause", "p", "loopPause", argType::Required, "", "loopPause", false, "unsigned long", "The main loop pause time in ns");
   config.add("RTPriority", "P", "RTPriority", argType::Required, "", "RTPriority", false, "unsigned", "The real-time priority (0-99)");

   //Logger Stuff
   m_log.setupConfig(config);

   if( m_powerMgtEnabled)
   {
      if(_useINDI == false)
      {
         //If this condition obtains, we should not go on because it means we'll never leave power off!!!
         log<software_critical>({__FILE__,__LINE__, "power management is enabled but we are not using INDI"});
         m_shutdown = true;
      }
      
      //Power Management
      config.add("power.device", "", "power.device", argType::Required, "power", "device", false, "string", "Device controlling power for this app's device (INDI name).");
      config.add("power.channel", "", "power.channel", argType::Required, "power", "channel", false, "string", "Channel on device for this app's device (INDI name).");
      config.add("power.element", "", "power.element", argType::Required, "power", "element", false, "string", "INDI power state element name.  Default is \"state\", only need to specify if different.");
      config.add("power.targetElement", "", "power.targetElement", argType::Required, "power", "targetElement", false, "string", "INDI power target element name.  Default is \"target\", only need to specify if different.");
      config.add("power.powerOnWait", "", "power.powerOnWait", argType::Required, "power", "powerOnWait", false, "int", "Time after power-on to wait before continuing [sec].  Default is 0 sec, max is 3600 sec.");
   }
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::loadBasicConfig() //virtual
{
   //---------- Setup the logger ----------//
   m_log.logName(m_configName);
   m_log.loadConfig(config);

   //--------- Loop Pause Time --------//
   config(m_loopPause, "loopPause");

   //--------- RT Priority ------------//
   int prio = m_RTPriority;
   config(prio, "RTPriority");
   if(prio != m_RTPriority)
   {
      RTPriority(prio);
   }

   //--------Power Management --------//
   if( m_powerMgtEnabled)
   {
      config(m_powerDevice, "power.device");
      config(m_powerChannel, "power.channel");
      config(m_powerElement, "power.element");
      config(m_powerTargetElement, "power.targetElement");
      
      if(m_powerDevice != "" && m_powerChannel != "")
      {
         log<text_log>("enabling power management: " + m_powerDevice + "." + m_powerChannel + "." + m_powerElement + "/" + m_powerTargetElement);
         REG_INDI_SETPROP(m_indiP_powerChannel, m_powerDevice, m_powerChannel);
      }
      else
      {
         log<text_log>("power management not configured!", logPrio::LOG_CRITICAL);
         m_shutdown = true;
      }
      
      config(m_powerOnWait, "power.powerOnWait");
      if(m_powerOnWait > 3600)
      {
         log<text_log>("powerOnWait longer than 1 hour.  Setting to 0.", logPrio::LOG_ERROR);
      }
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

   //We have to wait for power status to become available
   if(m_powerMgtEnabled)
   {
      while(m_powerState < 0 && !m_shutdown)
      {
         sleep(1);
         if(m_powerState < 0)
         {
            if(!stateLogged()) log<text_log>("waiting for power state");
         }
      }
      if(m_powerState > 0) 
      {
         state(stateCodes::POWERON);
      }
      else 
      {
         m_powerOnCounter = 0;
         state(stateCodes::POWEROFF);
         if(onPowerOff() < 0)
         {
            m_shutdown = 1;
         }
      }
   }

   //This is the main event loop.
   /* Conditions on entry:
    * -- PID locked
    * -- Log thread running
    * -- Signal handling installed
    * -- appStartup() successful
    * -- INDI communications started successfully (if being used)
    * -- power state known (if being managed)
    */
   while( m_shutdown == 0)
   {
      if(m_powerMgtEnabled)
      {
         if(state() == stateCodes::POWEROFF)
         {
            if(m_powerState == 1)
            {
               m_powerOnCounter = 0;
               state(stateCodes::POWERON);
            }
         }
         else //Any other state
         {
            if(m_powerState == 0)
            {
               state(stateCodes::POWEROFF);
               if(onPowerOff() < 0)
               {
                  m_shutdown = 1;
                  continue;
               }
            }
            //We don't do anything if m_powerState is -1, which is a startup condition.
         }
      }

      //Only run appLogic if power is on, or we are not managing power.
      if( !m_powerMgtEnabled || m_powerState > 0 )
      {
         if( appLogic() < 0)
         {
            m_shutdown = 1;
            continue;
         }
      }
      else if(m_powerState == 0)
      {
         if( whilePowerOff() < 0)
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
         std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(m_loopPause));
      }
   }

   appShutdown();

   state(stateCodes::SHUTDOWN);

   //Stop INDI communications
   if(m_indiDriver != nullptr)
   {
      pcf::IndiProperty ipSend;
      ipSend.setDevice(m_configName);
      m_indiDriver->sendDelProperty(ipSend);
      
      m_indiDriver->quitProcess();
      m_indiDriver->deactivate();
      log<indidriver_stop>();
   }

   unlockPID();

   sleep(1);
   return 0;
}

template<bool _useINDI>
template<typename logT, int retval>
int MagAOXApp<_useINDI>::log( const typename logT::messageT & msg,
                              logPrioT level
                            )
{
   m_log.log<logT>(msg, level);
   return retval;
}

template<bool _useINDI>
template<typename logT, int retval>
int MagAOXApp<_useINDI>::log( logPrioT level)
{
   m_log.log<logT>(level);
   return retval;
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::configLog( const std::string & name,
                                     const int & code,
                                     const std::string & value,
                                     const std::string & source
                                   )
{
   m_log.log<config_log>({name, code, value, source});
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

///Empty signal handler.  SIGUSR1 is used to interrupt sleep in various threads.   
void sigUsr1Handler( int signum,
                     siginfo_t * siginf,
                     void *ucont 
                   )
{
   static_cast<void>(signum);
   static_cast<void>(siginf);
   static_cast<void>(ucont);
   
   return;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::euidCalled()
{
   errno = 0;
   if(sys::th_seteuid(m_euidCalled) < 0)
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
   if(sys::th_seteuid(m_euidReal) < 0)
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
   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, 0, "Seeting euid to called failed."});
      return -1;
   }
   
   if( ::remove(pidFileName.c_str()) < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, 0, std::string("Failed to remove PID file: ") + strerror(errno)});
      return -1;
   }

   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, 0, 0, "Seeting euid to real failed."});
      return -1;
   }
   
   std::stringstream logss;
   logss << "PID (" << m_pid << ") unlocked.";
   log<text_log>(logss.str());

   return 0;
}

template<bool _useINDI>
template<class thisPtr, class Function>
int MagAOXApp<_useINDI>::threadStart( std::thread & thrd,
                                      bool & thrdInit,
                                      int thrdPrio,
                                      const std::string & thrdName,
                                      thisPtr * thrdThis,
                                      Function&& thrdStart
                                    )
{
   thrdInit = true;
   
   try
   {
      thrd  = std::thread( thrdStart, thrdThis);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on " + thrdName + " thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on " + thrdName + " thread start"});
      return -1;
   }

   if(!thrd.joinable())
   {
      log<software_error>({__FILE__, __LINE__, thrdName + " thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   if(thrdPrio < 0) thrdPrio = 0;
   if(thrdPrio > 99) thrdPrio = 99;

   sched_param sp;
   sp.sched_priority = thrdPrio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to called failed for " + thrdName});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(thrdPrio > 0) rv = pthread_setschedparam(thrd.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(thrd.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to real failed for " + thrdName});
   }
   
   if(rv < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, errno, "Setting " + thrdName + " thread scheduler priority to " + std::to_string(thrdPrio) + " failed."});
   }
   else
   {
      thrdInit = false;
      return log<text_log,0>(thrdName + " thread scheduler priority set to " + std::to_string(thrdPrio));
   }
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

   //Note this is called very execute loop to make sure we update eventually
   if(lock.owns_lock())
   {  
      ///\todo move this to a function in stateCodes
      pcf::IndiProperty::PropertyStateType stst = INDI_IDLE;
      if(m_state == stateCodes::READY) stst = INDI_OK;
      if(m_state == stateCodes::OPERATING || m_state == stateCodes::HOMING || m_state == stateCodes::CONFIGURING) stst = INDI_BUSY;
      if( m_state < stateCodes::NODEVICE ) stst = INDI_ALERT;
      else if (m_state <= stateCodes::LOGGEDIN ) stst = INDI_IDLE;
      else if (m_state == stateCodes::NOTHOMED || m_state == stateCodes::SHUTDOWN) stst = INDI_IDLE;
      
      updateIfChanged(m_indiP_state, "state", stateCodes::codeText(m_state), stst);
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
int MagAOXApp<_useINDI>::createStandardIndiText( pcf::IndiProperty & prop,       
                                                 const std::string & propName,   
                                                 const std::string & label, 
                                                 const std::string & group  
                                               )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Text);
   prop.setDevice(configName());
   prop.setName(propName);
   prop.setPerm(pcf::IndiProperty::ReadWrite); 
   prop.setState(pcf::IndiProperty::Idle);
   prop.add(pcf::IndiElement("current"));
   prop.add(pcf::IndiElement("target"));
   
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop.setLabel(label);
   }
   
   if(group != "")
   {
      prop.setGroup(group);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createROIndiText( pcf::IndiProperty & prop,           
                                           const std::string & propName,       
                                           const std::string & elName,         
                                           const std::string & propLabel, 
                                           const std::string & propGroup,  
                                           const std::string & elLabel  
                                         )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Text);
   prop.setDevice(configName());
   prop.setName(propName);
   prop.setPerm(pcf::IndiProperty::ReadOnly); 
   prop.setState(pcf::IndiProperty::Idle);
   
   //Don't set "" just in case libcommon does something with defaults
   if(propLabel != "")
   {
      prop.setLabel(propLabel);
   }
   
   if(propGroup != "")
   {
      prop.setGroup(propGroup);
   }
   
   prop.add(pcf::IndiElement(elName));
   
   if(elLabel != "")
   {
      prop[elName].setLabel(elLabel);
   }
   
   return 0;
}

template<bool _useINDI>
template<typename T>
int MagAOXApp<_useINDI>::createStandardIndiNumber( pcf::IndiProperty & prop,       
                                                   const std::string & name,       
                                                   const T & min,                  
                                                   const T & max,                  
                                                   const T & step,                 
                                                   const std::string & format,     
                                                   const std::string & label, 
                                                   const std::string & group  
                                                 )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Number);
   prop.setDevice(configName());
   prop.setName(name);
   prop.setPerm(pcf::IndiProperty::ReadWrite); 
   prop.setState(pcf::IndiProperty::Idle);
   prop.add(pcf::IndiElement("current"));
   prop["current"].setMin(min);
   prop["current"].setMax(max);
   prop["current"].setStep(step);
   prop["current"].setFormat(format);
   
   prop.add(pcf::IndiElement("target"));
   prop["target"].setMin(min);
   prop["target"].setMax(max);
   prop["target"].setStep(step);
   prop["target"].setFormat(format);
   
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop.setLabel(label);
   }
   
   if(group != "")
   {
      prop.setGroup(group);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createROIndiNumber( pcf::IndiProperty & prop,           
                                             const std::string & propName,       
                                             const std::string & propLabel, 
                                             const std::string & propGroup 
                                           )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Number);
   prop.setDevice(configName());
   prop.setName(propName);
   prop.setPerm(pcf::IndiProperty::ReadOnly); 
   prop.setState(pcf::IndiProperty::Idle);
   
   //Don't set "" just in case libcommon does something with defaults
   if(propLabel != "")
   {
      prop.setLabel(propLabel);
   }
   
   if(propGroup != "")
   {
      prop.setGroup(propGroup);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createStandardIndiToggleSw( pcf::IndiProperty & prop,       
                                                     const std::string & name,       
                                                     const std::string & label, 
                                                     const std::string & group  
                                                   )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Switch);
   prop.setDevice(configName());
   prop.setName(name);
   prop.setPerm(pcf::IndiProperty::ReadWrite); 
   prop.setState(pcf::IndiProperty::Idle);
   prop.setRule(pcf::IndiProperty::AtMostOne);
   
   //Add the toggle element initialized to Off
   prop.add(pcf::IndiElement("toggle", pcf::IndiElement::Off));
   
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop.setLabel(label);
   }
   
   if(group != "")
   {
      prop.setGroup(group);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createStandardIndiRequestSw( pcf::IndiProperty & prop,   
                                                      const std::string & name,   
                                                      const std::string & label,  
                                                      const std::string & group   
                                                    )
{
   prop = pcf::IndiProperty(pcf::IndiProperty::Switch);
   prop.setDevice(configName());
   prop.setName(name);
   prop.setPerm(pcf::IndiProperty::ReadWrite); 
   prop.setState(pcf::IndiProperty::Idle);
   prop.setRule(pcf::IndiProperty::AtMostOne);
   
   //Add the toggle element initialized to Off
   prop.add(pcf::IndiElement("request", pcf::IndiElement::Off));
   
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop.setLabel(label);
   }
   
   if(group != "")
   {
      prop.setGroup(group);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::createStandardIndiSelectionSw( pcf::IndiProperty & prop,                  
                                                        const std::string & name,                  
                                                        const std::vector<std::string> & elements, 
                                                        const std::string & label,            
                                                        const std::string & group             
                                                      )
{
   if(elements.size() == 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "elements vector has zero size"});
   }
   
   prop = pcf::IndiProperty(pcf::IndiProperty::Switch);
   prop.setDevice(configName());
   prop.setName(name);
   prop.setPerm(pcf::IndiProperty::ReadWrite); 
   prop.setState(pcf::IndiProperty::Idle);
   prop.setRule(pcf::IndiProperty::OneOfMany);
   
   //Add the toggle element initialized to Off
   for(size_t n=0; n < elements.size(); ++n)
   {
      prop.add(pcf::IndiElement(elements[n], pcf::IndiElement::Off));
   }
   
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop.setLabel(label);
   }
   
   if(group != "")
   {
      prop.setGroup(group);
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertyReadOnly( pcf::IndiProperty & prop )
{
   if(!m_useINDI) return 0;

   callBackInsertResult result =  m_indiNewCallBacks.insert(callBackValueType( prop.getName(), {&prop, nullptr}));

   try 
   {
      if(!result.second)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "failed to insert INDI property: " + prop.getName()});
      }
   }
   catch( std::exception & e)
   {
      return log<software_error, -1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});
   }
   catch(...)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "Unknown exception caught."});
   }
   
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertyReadOnly( pcf::IndiProperty & prop,
                                                       const std::string & propName,
                                                       const pcf::IndiProperty::Type & propType,
                                                       const pcf::IndiProperty::PropertyPermType & propPerm,
                                                       const pcf::IndiProperty::PropertyStateType & propState
                                                     )
{
   if(!m_useINDI) return 0;

   prop = pcf::IndiProperty (propType);
   prop.setDevice(m_configName);
   prop.setName(propName);
   prop.setPerm(propPerm);
   prop.setState( propState);


   callBackInsertResult result =  m_indiNewCallBacks.insert(callBackValueType( propName, {&prop, nullptr}));

   try 
   {
      if(!result.second)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "failed to insert INDI property: " + prop.getName()});
      }
   }
   catch( std::exception & e)
   {
      return log<software_error, -1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});
   }
   catch(...)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "Unknown exception caught."});
   }
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertyNew( pcf::IndiProperty & prop,
                                                  int (*callBack)( void *, const pcf::IndiProperty &ipRecv)
                                                )
{
   if(!m_useINDI) return 0;

   try
   {
      callBackInsertResult result =  m_indiNewCallBacks.insert(callBackValueType( prop.getName(), {&prop, callBack}));

      if(!result.second)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "failed to insert INDI property: " + prop.getName()});
      }
   }
   catch( std::exception & e)
   {
      return log<software_error, -1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});
   }
   catch(...)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "Unknown exception caught."});
   }
   
   return 0;
}

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

   return registerIndiPropertyNew(prop, callBack);
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::registerIndiPropertyNew( pcf::IndiProperty & prop,
                                                  const std::string & propName,
                                                  const pcf::IndiProperty::Type & propType,
                                                  const pcf::IndiProperty::PropertyPermType & propPerm,
                                                  const pcf::IndiProperty::PropertyStateType & propState,
                                                  const pcf::IndiProperty::SwitchRuleType & propRule,
                                                  int (*callBack)( void *, const pcf::IndiProperty &ipRecv)
                                                )
{
   if(!m_useINDI) return 0;

   prop = pcf::IndiProperty (propType);
   prop.setDevice(m_configName);
   prop.setName(propName);
   prop.setPerm(propPerm);
   prop.setState( propState);
   prop.setRule( propRule);
   return registerIndiPropertyNew(prop, callBack);
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

   callBackInsertResult result =  m_indiSetCallBacks.insert(callBackValueType( prop.createUniqueKey(), {&prop, callBack}));

   try 
   {
      if(!result.second)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "failed to insert INDI property: " + prop.createUniqueKey()});
      }
   }
   catch( std::exception & e)
   {
      return log<software_error, -1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});
   }
   catch(...)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "Unknown exception caught."});
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
   m_driverCtrlName = driverFIFOPath + "/" + configName() + ".ctrl";

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

   errno = 0;
   if(mkfifo(m_driverCtrlName.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP) !=0 )
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

   //======= Instantiate the indiDriver
   try
   {
      m_indiDriver = new indiDriver<MagAOXApp>(this, m_configName, "0", "0");
   }
   catch(...)
   {
      log<software_critical>({__FILE__, __LINE__, 0, 0, "INDI Driver construction exception."});
      return -1;
   }

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
                                           const T & newVal, 
                                           pcf::IndiProperty::PropertyStateType ipState
                                         )
{
   if(!_useINDI) return;

   if(!m_indiDriver) return;

   indi::updateIfChanged( p, el, newVal, m_indiDriver, ipState);
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::updateIfChanged( pcf::IndiProperty & p,
                                           const std::string & el,
                                           const char * newVal, 
                                           pcf::IndiProperty::PropertyStateType ipState
                                         )
{
   updateIfChanged<std::string>(p,el, std::string(newVal), ipState);
}

template<bool _useINDI>
void MagAOXApp<_useINDI>::updateSwitchIfChanged( pcf::IndiProperty & p,
                                                 const std::string & el,
                                                 const pcf::IndiElement::SwitchStateType & newVal, 
                                                 pcf::IndiProperty::PropertyStateType ipState
                                               )
{
   if(!_useINDI) return;

   if(!m_indiDriver) return;

   indi::updateSwitchIfChanged( p, el, newVal, m_indiDriver, ipState);
}

template<bool _useINDI>
template<typename T>
void MagAOXApp<_useINDI>::updateIfChanged( pcf::IndiProperty & p,
                                           const std::string & el,
                                           const std::vector<T> & newVals,
                                           pcf::IndiProperty::PropertyStateType ipState
                                         )
{
   if(!_useINDI) return;

   if(!m_indiDriver) return;

   for (size_t index = 0; index < newVals.size(); ++index) 
   {
      std::string descriptor = el+std::to_string(index);
      indi::updateIfChanged( p, descriptor, newVals.at(index), m_indiDriver, ipState);
   }
}

template<bool _useINDI>
template<typename T>
void MagAOXApp<_useINDI>::updateIfChanged( pcf::IndiProperty & p,
                                           const std::vector<std::string> & el,
                                           const std::vector<T> & newVals
                                         )
{
   if(!_useINDI) return;

   if(!m_indiDriver) return;

   for (size_t index = 0; index < newVals.size(); ++index) 
   {
      std::string descriptor = el.at(index);
      indi::updateIfChanged( p, descriptor, newVals.at(index), m_indiDriver);
   }
}


template<bool _useINDI>
template<typename T>
int MagAOXApp<_useINDI>::indiTargetUpdate( pcf::IndiProperty & localProperty,
                                           T & localTarget,
                                           const pcf::IndiProperty & remoteProperty,
                                           bool setBusy
                                         )
{
   if( remoteProperty.getName() != localProperty.getName())
   {
      return log<text_log,-1>("INDI property names do not match", logPrio::LOG_ERROR);
   }
   
   if( ! (remoteProperty.find("target") || remoteProperty.find("current") ) )
   {
      return log<text_log,-1>("not target or current element in INDI property", logPrio::LOG_ERROR);
   }

   bool set = false;
   
   if( remoteProperty.find("target") )
   {
      localTarget = remoteProperty["target"].get<T>();
      set = true;
   }
   
   if( !set )
   {
      if( remoteProperty.find("current") )
      {
         localTarget = remoteProperty["current"].get<T>();
         set = true;
      }
   }
   
   if( !set )
   {
      return log<text_log,-1>("no non-empty value found in INDI property", logPrio::LOG_ERROR);
   }
   
   if(setBusy)
   {
      updateIfChanged(localProperty, "target", localTarget, INDI_BUSY);
   }
   else
   {
      updateIfChanged(localProperty, "target", localTarget);
   }
   
   return 0;
}









/// \todo move propType to an INDI utils file, and document.

template<typename T>
pcf::IndiProperty::Type propType()
{
   return pcf::IndiProperty::Unknown;
}


template<>
inline
pcf::IndiProperty::Type propType<char *>()
{
   return pcf::IndiProperty::Text;
}

template<>
inline
pcf::IndiProperty::Type propType<std::string>()
{
   return pcf::IndiProperty::Text;
}

template<>
inline
pcf::IndiProperty::Type propType<int>()
{
   return pcf::IndiProperty::Number;
}

template<>
inline
pcf::IndiProperty::Type propType<double>()
{
   return pcf::IndiProperty::Number;
}

template<bool _useINDI>
template<typename T>
int MagAOXApp<_useINDI>::sendNewProperty( const pcf::IndiProperty & ipSend,
                                          const std::string & el,
                                          const T & newVal
                                        )
{
   if(!_useINDI) return 0;

   if(!m_indiDriver)
   {
      log<software_error>({__FILE__, __LINE__, "INDI communications not initialized."});
      return -1;
   }
   pcf::IndiProperty ipToSend = ipSend;

   try
   {
      ipToSend[el].setValue(newVal);
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught setting " + ipSend.getDevice() + "." + ipSend.getName() + "." + el});
      return -1;
   }

   int rv = m_indiDriver->sendNewProperty(ipToSend);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::onPowerOff() 
{
   return 0;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::whilePowerOff() 
{
   return 0;
}

template<bool _useINDI>
bool MagAOXApp<_useINDI>::powerOnWaitElapsed()
{
   if(!m_powerMgtEnabled || m_powerOnWait == 0 || m_powerOnCounter < 0) return true;
   
   if(m_powerOnCounter*m_loopPause > ((double) m_powerOnWait)*1e9)
   {
      return true;
   }
   else
   {
      ++m_powerOnCounter;
      return false;
   }
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::powerState()
{
   if(!m_powerMgtEnabled) return 1;
   
   return m_powerState;
}

template<bool _useINDI>
int MagAOXApp<_useINDI>::powerStateTarget()
{
   if(!m_powerMgtEnabled) return 1;
   
   return m_powerTargetState;
}

template<bool _useINDI>
INDI_SETCALLBACK_DEFN( MagAOXApp<_useINDI>, m_indiP_powerChannel)(const pcf::IndiProperty &ipRecv)
{
   std::string ps;

   if( ipRecv.find(m_powerElement))
   {
      ps = ipRecv[m_powerElement].get<std::string>();

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
   }
   
   if( ipRecv.find(m_powerTargetElement))
   {
      ps = ipRecv[m_powerTargetElement].get<std::string>();

      if(ps == "On")
      {
         m_powerTargetState = 1;
      }
      else if (ps == "Off")
      {
         m_powerTargetState = 0;
      }
      else
      {
         m_powerTargetState = -1;
      }
   }
   
   return 0;
}


   
template<bool _useINDI>
std::string MagAOXApp<_useINDI>::configName()
{
   return m_configName;
}

template<bool _useINDI>
std::string MagAOXApp<_useINDI>::configDir()
{
   return m_configDir;
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

template<bool _useINDI>
std::string MagAOXApp<_useINDI>::driverCtrlName()
{
   return m_driverCtrlName;
}

} //namespace app
} //namespace MagAOX

#endif //app_MagAOXApp_hpp
