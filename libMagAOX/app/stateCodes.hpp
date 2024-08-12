/** \file stateCodes.hpp 
  * \brief MagAO-X Application States
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-20 created by JRM
  * 
  * \ingroup app_files
  */ 

#ifndef app_stateCodes_hpp
#define app_stateCodes_hpp

#include <string>

namespace MagAOX 
{
namespace app
{

/// Scoping struct for application state codes.
/** We do not use the enum class feature since it does not have automatic integer conversion,
  * and it's nice to have static member functions associated.
  * 
  * \todo this ought to be renamed stateCode (no s)
  * 
  * \ingroup magaoxapp
  */
struct stateCodes 
{

   /// The type of the state code.
   /**
     */
   typedef int16_t stateCodeT;


   /// The numeric codes descrbing an application's state
   /** \ingroup magaoxapp
     *  
     */   
   enum : stateCodeT { FAILURE=-20,       ///< The application has failed, should be used when m_shutdown is set for an error.
                       ERROR=-10,         ///< The application has encountered an error, from which it is recovering (with or without intervention)
                       UNINITIALIZED = 0, ///< The application is unitialized, the default
                       INITIALIZED = 1,   ///< The application has been initialized, set just before calling appStartup().
                       NODEVICE = 2,      ///< No device exists for the application to control.
                       POWEROFF = 4,      ///< The device power is off.
                       POWERON = 6,       ///< The device power is on.
                       NOTCONNECTED = 8,  ///< The application is not connected to the device or service.
                       CONNECTED = 10,    ///< The application has connected to the device or service.
                       LOGGEDIN = 15,     ///< The application has logged into the device or service
                       CONFIGURING = 20,  ///< The application is configuring the device.
                       NOTHOMED = 24,     ///< The device has not been homed.
                       HOMING = 25,       ///< The device is homing.
                       OPERATING = 30,    ///< The device is operating, other than homing.
                       READY = 35,        ///< The device is ready for operation, but is not operating.
                       SHUTDOWN = 10000   ///< The application has shutdown, set just after calling appShutdown().
                    };
                  

   /// Get an ASCII string corresponding to an application stateCode.
   /**
     * \returns a string with the text name of the stateCode
     * 
     * \todo rename this to code2str
     */ 
   static std::string codeText( const stateCodeT & stateCode /**<[in] the stateCode for which the name is desired*/);

   /// Get the stateCode corresponding to an ASCII string.
   /**
     * \returns the stateCodeT which matches the string
     * 
     */
   static stateCodeT str2Code( const std::string & stateStr /**< [in] the string represenation of a state code */ );

   /// Get the stateCode corresponding to an ASCII string with minimal checks.
   /** This matches based on the first character and, if necessary, one more unique character.
     * Other than length checks when necessary, this does minimal validation, so you should only use it
     * if you trust the source of \p stateStr. For example, any string that starts with `'L'` will return
     * stateCodes::LOGGEDIN regardless of the value of `stateStr[1]`.
     *   
     * \returns the stateCodeT which matches the string
     * 
     */
   static stateCodeT str2CodeFast( const std::string & stateStr /**< [in] the string represenation of a state code */ );

}; //struct stateCodes           



} //namespace app 
} //namespace MagAOX

#endif //app_stateCodes_hpp
