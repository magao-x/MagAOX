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

std::string stateCodes::codeText( const stateCodeT & stateCode )
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

stateCodes::stateCodeT stateCodes::str2Code( const std::string & stateStr )
{
   
   if(stateStr == "FAILURE")
   {
      return stateCodes::FAILURE;
   }
   else if(stateStr == "ERROR")
   {
      return stateCodes::ERROR;
   }
   else if(stateStr == "UNINITIALIZED")
   {
      return stateCodes::UNINITIALIZED;
   }
   else if(stateStr == "INITIALIZED" )
   {
      return stateCodes::INITIALIZED;
   }
   else if(stateStr == "NODEVICE" )
   {
      return stateCodes::NODEVICE;
   }
   else if(stateStr == "POWEROFF")
   {
      return stateCodes::POWEROFF;
   }
   else if(stateStr == "POWERON")
   {
      return stateCodes::POWERON;
   }
   else if(stateStr == "NOTCONNECTED")
   {
      return stateCodes::NOTCONNECTED;
   }
   else if(stateStr == "CONNECTED")
   {
      return stateCodes::CONNECTED;
   }
   else if(stateStr == "LOGGEDIN")
   {
      return stateCodes::LOGGEDIN;
   }
   else if(stateStr == "CONFIGURING")
   {
      return stateCodes::CONFIGURING;
   }
   else if(stateStr == "NOTHOMED")
   {
      return stateCodes::NOTHOMED;
   }
   else if(stateStr == "HOMING")
   {
      return stateCodes::HOMING;
   }
   else if(stateStr == "OPERATING")
   {
      return stateCodes::OPERATING;
   }
   else if(stateStr == "READY")
   {
      return stateCodes::READY;
   }
   else if(stateStr == "SHUTDOWN")
   {
      return stateCodes::SHUTDOWN;
   }
   else
   {
      return -999;
   }

}

stateCodes::stateCodeT stateCodes::str2CodeFast( const std::string & stateStr )
{
   switch(stateStr[0])
   {
      case 'C':   
         if(stateStr.size() < 4)
         {
            return -999;
         }
         
         if( stateStr[3] == 'F')
         {
            return stateCodes::CONFIGURING;
         }
         else if( stateStr[3] == 'N')
         {
            return stateCodes::CONNECTED;
         }
         else
         {
            return -999;
         }
      case 'E':
         return stateCodes::ERROR;
      case 'F':
         return stateCodes::FAILURE;
      case 'H':
         return stateCodes::HOMING;
      case 'I':
         return INITIALIZED;
      case 'L':
         return LOGGEDIN;
      case 'N':
         if(size(stateStr) < 4)
         {
            return -999;
         }
         if(stateStr[2] == 'D')
         {
            return stateCodes::NODEVICE;
         }
         else if(stateStr[3] == 'C')
         {
            return stateCodes::NOTCONNECTED;
         }
         else if(stateStr[3] == 'H')
         {
            return stateCodes::NOTHOMED;
         }
         else
         {
            return -999;
         }
      case 'O':
         return stateCodes::OPERATING;
      case 'P':
         if(stateStr.size() < 7)
         {
            return -999;
         }

         if(stateStr[6] == 'F')
         {
            return stateCodes::POWEROFF;
         }
         else if(stateStr[6] == 'N')
         {
            return stateCodes::POWERON;
         }
         else
         {
            return -999;
         }
      case 'R':
         return stateCodes::READY;
      case 'S':
         return stateCodes::SHUTDOWN;
      case 'U':
         return stateCodes::UNINITIALIZED;
      default:
         return -999;
   }
}

} //namespace app 
} //namespace MagAOX

