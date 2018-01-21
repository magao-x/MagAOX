/** \file stateCodes.hpp 
  * \brief MagAO-X Application States
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-20 created by JRM
  */ 

#ifndef app_stateCodes_hpp
#define app_stateCodes_hpp


namespace MagAOX 
{
namespace app
{

typedef int stateCodeT;

namespace stateCodes 
{
   
/// The numeric codes descrbing an application's state
enum : stateCodeT { FAILURE=-20,       ///< The application has failed, should be used when m_shutdown is set for an error.
                    ERROR=-10,         ///< The application has encountered and error, from which it is recovering (with or without intervention)
                    UNINITIALIZED = 0, ///< The application is unitialized, the default
                    INITIALIZED = 1,   ///< The application has been initialized, set just before calling appStartup().
                    NODEVICE = 2,      ///<
                    NOTCONNECTED = 3,  ///<
                    CONNECTED = 4,     ///<
                    LOGGEDIN = 5,      ///<
                    CONFIGURING = 6,   ///<
                    HOMING = 10,       ///<
                    OPERATING = 20,    ///<
                    READY = 30,        ///<
                    SHUTDOWN = 10000   ///< The application has shutdown, set just after calling appShutdown().
                  };
                  
} //namespace stateCodes           

/// Get an ASCII string corresponding to an application stateCode.
/**
  * \returns a string with the text name of the stateCode
  */ 
std::string stateCodeText( stateCodeT stateCode /**<[in] the stateCode for which the name is desired*/)
{
   switch(stateCode)
   {
      case stateCodes::FAILURE:
         return "FAILURE";
      case stateCodes::ERROR:
         return "ERROR";
      case stateCodes::UNINITIALIZED:
         return "UNINITIALIZED";
      case stateCodes::INITIALIZED:
         return "INITIALIZED";
      case stateCodes::NODEVICE:
         return "NODEVICE";
      case stateCodes::NOTCONNECTED:
         return "NOTCONNECTED";
      case stateCodes::CONNECTED:
         return "CONNECTED";
      case stateCodes::LOGGEDIN:
         return "LOGGEDIN";
      case stateCodes::CONFIGURING:
         return "CONFIGURING";
      case stateCodes::HOMING:
         return "HOMING";
      case stateCodes::OPERATING:
         return "OPERATING";
      case stateCodes::READY:
         return "READY";
      case stateCodes::SHUTDOWN:
         return "SHUTDOWN";
      default:
         return "UNKNOWN";
   }

   return "UNKNOWN";
}

} //namespace app 
} //namespace MagAOX

#endif //app_stateCodes_hpp
