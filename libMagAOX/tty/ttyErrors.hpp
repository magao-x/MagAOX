/** \file ttyErrors.hpp 
  * \brief Error numbers for the tty utilities.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup tty_files
  * History:
  * - 2018-01-17 created by JRM
  */

#ifndef tty_ttyErrors_hpp
#define tty_ttyErrors_hpp

#include <string>

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

#define TELNET_E_NOERROR            (0)
#define TELNET_E_GETADDR            (-42040)
#define TELNET_E_SOCKET             (-42041)
#define TELNET_E_BIND               (-42042)
#define TELNET_E_CONNECT            (-42043)
#define TELNET_E_TELNETINIT         (-42044)
#define TELNET_E_EHERROR            (-42045)
#define TELNET_E_LOGINTIMEOUT       (-42046)

namespace MagAOX 
{
namespace tty 
{
  
/// Get a text explanation of a TTY_E_ error code.
/** 
  * \ingroup tty 
  */ 
std::string ttyErrorString( int ec /**< [in] the error code */ );

} //namespace tty 
} //namespace MagAOX 

#endif //tty_ttyErrors_hpp
 
