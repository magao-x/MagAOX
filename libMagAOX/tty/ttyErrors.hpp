/** \file ttyErrors.hpp 
  * \brief Error numbers for the tty utilities.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-17 created by JRM
  */

#ifndef tty_ttyErrors_hpp
#define tty_ttyErrors_hpp

#define TTY_E_NOERROR            (0)
#define TTY_E_TCGETATTR          (-42001)
#define TTY_E_SETISPEED          (-42002)
#define TTY_E_SETOSPEED          (-42003)
#define TTY_E_TCSETATTR          (-42004)
#define TTY_E_TIMEOUTONWRITEPOLL (-42011)
#define TTY_E_ERRORONWRITEPOLL   (-42012) 
#define TTY_E_ERRORONWRITE       (-42013)
#define TTY_E_TIMEOUTONWRITE     (-42014)
#define TTY_E_TIMEOUTONREADPOLL  (-42015)
#define TTY_E_ERRORONREADPOLL    (-42016)
#define TTY_E_ERRORONREAD        (-42017)
#define TTY_E_TIMEOUTONREAD      (-42018)
#define TTY_E_NODEVNAMES         (-42021)
#define TTY_E_UDEVNEWFAILED      (-42022)
#define TTY_E_DEVNOTFOUND        (-42023)
#define TTY_E_BADBAUDRATE        (-42030)

#define TELNET_E_GETADDR            (-42040)
#define TELNET_E_SOCKET             (-42041)
#define TELNET_E_BIND               (-42042)
#define TELNET_E_CONNECT            (-42043)
#define TELNET_E_TELNETINIT         (-42044)
#define TELNET_E_EHERROR            (-42045)

namespace MagAOX 
{
namespace tty 
{

  
/// Get a text explanation of a TTY_E_ error code.
inline
std::string ttyErrorString( int ec /**< [in] the error code */ )
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
         return "TTY; socket connect failed";
      case TELNET_E_TELNETINIT:
         return "TTY: failed to init telnet_t structure";
      case TELNET_E_EHERROR:
         return "TTY: error set in telnet event handler";
      default:
         return "TTY: unknown error code";
   };
}

} //namespace tty 
} //namespace MagAOX 

#endif //tty_ttyErrors_hpp
 
