/** \file ttyErrors.cpp 
  * \brief Error numbers for the tty utilities.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup tty_files
  */

#include "ttyErrors.hpp"

namespace MagAOX 
{
namespace tty 
{

std::string ttyErrorString( int ec )
{
   switch(ec)
   {
      case TTY_E_NOERROR:
         return "TTY: success";
      case TTY_E_TCGETATTR:
         return "TTY: tcgetattr returned error";
      case TTY_E_SETISPEED:
         return "TTY: cfsetispeed returned error";
      case TTY_E_SETOSPEED:
         return "TTY: cfsetospeed returned error";
      case TTY_E_TCSETATTR:
         return "TTY: tcsetattr returned error";
      case TTY_E_TIMEOUTONWRITEPOLL:
         return "TTY: the write poll timed out";
      case TTY_E_ERRORONWRITEPOLL:
         return "TTY: an error was returned by the write poll";
      case TTY_E_ERRORONWRITE:
         return "TTY: an error occurred writing to the file";
      case TTY_E_TIMEOUTONWRITE:
         return "TTY: a timeout occurred during the write";
      case TTY_E_TIMEOUTONREADPOLL:
         return "TTY: the read poll timed out";
      case TTY_E_ERRORONREADPOLL:
         return "TTY: an error was returned by the read poll";
      case TTY_E_ERRORONREAD:
         return "TTY: an error occurred reading from the file";
      case TTY_E_TIMEOUTONREAD:
         return "TTY:  a timeout occurred during the read";
      case TTY_E_NODEVNAMES:
         return "TTY: no device names found in sys";
      case TTY_E_UDEVNEWFAILED:
         return "TTY: initializing libudev failed";
      case TTY_E_DEVNOTFOUND:
         return "TTY: no matching device found";
      case TTY_E_BADBAUDRATE:
         return "TTY: bad baud rate specified";
         
      case TELNET_E_GETADDR:
         return "TTY: getaddr failed";
      case TELNET_E_SOCKET:
         return "TTY: socket creation failed";
      case TELNET_E_BIND:
         return "TTY: socket bind failed";
      case TELNET_E_CONNECT:
         return "TTY: socket connect failed";
      case TELNET_E_TELNETINIT:
         return "TTY: failed to init telnet_t structure";
      case TELNET_E_EHERROR:
         return "TTY: error set in telnet event handler";
      case TELNET_E_LOGINTIMEOUT:
         return "TTY: login timed out";
      default:
         return "TTY: unknown error code";
   };
}

} //namespace tty 
} //namespace MagAOX 

 
