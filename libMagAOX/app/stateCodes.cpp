/** \file stateCodes.cpp 
  * \brief MagAO-X Application States
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * 
  * \ingroup app_files
  */ 

#include "stateCodes.hpp"

namespace MagAOX 
{
namespace app
{

std::string stateCodes::codeText( stateCodeT stateCode )
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
      case stateCodes::POWEROFF:
         return "POWEROFF";
      case stateCodes::POWERON:
         return "POWERON";
      case stateCodes::NOTCONNECTED:
         return "NOTCONNECTED";
      case stateCodes::CONNECTED:
         return "CONNECTED";
      case stateCodes::LOGGEDIN:
         return "LOGGEDIN";
      case stateCodes::CONFIGURING:
         return "CONFIGURING";
      case stateCodes::NOTHOMED:
         return "NOTHOMED";
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

