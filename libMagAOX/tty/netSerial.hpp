/** \file netSerial.hpp
  * \brief Managing a connection to a serial device over a network socket.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * This code is taken from the LBTAO/MagAO supervisor source code, modifications
  * mainly for c++.
  * 
  * \ingroup tty_files
  */

#ifndef tty_netSerial_hpp
#define tty_netSerial_hpp

#define NETSERIAL_E_NOERROR   (0)
#define NETSERIAL_E_NETWORK   (-50000)
#define NETSERIAL_E_CONNECT   (-50010)
#define NETSERIAL_E_COMM      (-50020)

namespace MagAOX
{
namespace tty
{

/// Manage a connectio to a serial device over a network 
/**
  *
  * \todo document this, including methods
  * \todo add errors to ttyErrors
  */ 
struct netSerial
{
   
protected:
   int m_sockfd {-1};
   
public:

   int serialInit( const char *address, 
                   int port
                 );
   
   int serialClose(void);
   
   int serialOut( const char *buf, 
                  int len
                );
   
   int serialIn( char *buf, 
                 int len, 
                 int timeout
               );
   
   int serialInString( char *buf, 
                       int len, 
                       int timeout, 
                       char terminator
                     );
   
   int serialInString2( char *buf, 
                        int len, 
                        int timeout, 
                        char *terminator
                      );

   int getSocketFD(void);

};



} // namespace tty
} // namespace MagAOX

#endif // netSerial_hpp
