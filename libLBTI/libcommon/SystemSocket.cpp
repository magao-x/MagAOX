// $Id: SystemSocket.cpp 7418 2009-12-16 22:08:43Z pgrenz $
////////////////////////////////////////////////////////////////////////////////

#include "SystemSocket.hpp"
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <sstream>
#include <sstream>
#include <sys/types.h>
#include <string.h> // For 'memset'.
#include <stdlib.h> // For 'malloc', 'free', etc.
// For the "TCP_NODELAY" constant.
#include <netinet/tcp.h>
// For the "geLocalIP" function:
#include <sys/socket.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <net/if_arp.h>
#include <arpa/inet.h>               // for inet_addr
// End of includes for the "getLocalIP" function.

using ::std::string;
using ::std::vector;
using pcf::SystemSocket;

// On the MAC, there is no "MSG_NOSIGNAL" flag. The equivalent is
// "SO_NOSIGPIPE" (on version 10.2 and later)
#if defined(__APPLE__)
//#ifdef __FreeBSD__
//# if __FreeBSD_version >= 400014
//#  define s6_addr16 __u6_addr.__u6_addr16
# if !defined(MSG_NOSIGNAL)
# define MSG_NOSIGNAL SO_NOSIGPIPE
# endif
//# endif
#endif

////////////////////////////////////////////////////////////////////////////////

SystemSocket::SystemSocket()
{
  m_nSocket = -1;
  // default the connection timeout to 1 second.
  m_nConnectTimeout = 1000;
  // The Nagle algorithm is enabled by default.
  m_oIsNagleDisabled = false;
}

////////////////////////////////////////////////////////////////////////////////

SystemSocket::~SystemSocket()
{
  close();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Initialize this SystemSocket from another one.

SystemSocket::SystemSocket( const SystemSocket &rhs )
{
  // The socket file descriptor should not be copied,
  // since each must be unique.
  m_nSocket = -1;
  m_nConnectTimeout = rhs.m_nConnectTimeout;
  m_szHost = rhs.m_szHost;
  m_nPort = rhs.m_nPort;
  // The Nagle algorithm is enabled by default.
  m_oIsNagleDisabled = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator. Assigns this object from another SystemSocket object.

const SystemSocket &SystemSocket::operator= ( const SystemSocket &rhs )
{
  if ( &rhs != this )
  {
    // The socket file descriptor should not be copied,
    // since each must be unique.
    m_nSocket = -1;
    m_nConnectTimeout = rhs.m_nConnectTimeout;
    m_szHost = rhs.m_szHost;
    m_nPort = rhs.m_nPort;
    // The Nagle algorithm is enabled by default.
    m_oIsNagleDisabled = false;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
//  Returns the socket descriptor.

int SystemSocket::getFd() const
{
  return m_nSocket;
}

////////////////////////////////////////////////////////////////////////////////
// Returns true if the listen or connect was successful, false otherwise.

bool SystemSocket::isValid() const
{
  return ( bool )( m_nSocket >= 0 );
}

////////////////////////////////////////////////////////////////////////////////
// Set the connect timeout in milliseconds.

void SystemSocket::setConnectTimeout( const int &nTimeout )
{
  m_nConnectTimeout = nTimeout;
}

////////////////////////////////////////////////////////////////////////////////
//  returns the socket host address.

string SystemSocket::getHost() const
{
  return string( "127.0.0.1" );
  /*
  string szHost;

  sockaddr_in saAddr;
  memset( &saAddr, 0, sizeof ( saAddr ) );
  socklen_t tSize = sizeof( saAddr );
  if ( getsockname( m_nSocket, ( sockaddr * )( &saAddr ), &tSize ) == 0 )
  {
    szHost = string( inet_ntoa( saAddr.sin_addr ) );
  }
  return szHost;
  */
}

////////////////////////////////////////////////////////////////////////////////
//  returns the socket port.

int SystemSocket::getPort() const
{
  int nPort = -1;

  sockaddr_in saAddr;
  memset( &saAddr, 0, sizeof ( saAddr ) );
  socklen_t tSize = sizeof( saAddr );
  if ( getsockname( m_nSocket, ( sockaddr * )( &saAddr ), &tSize ) == 0 )
    nPort = ntohs( saAddr.sin_port );
  return nPort;
}

////////////////////////////////////////////////////////////////////////////////
//  returns 0 if the listen was successful, a negative error code otherwise.

int SystemSocket::create( const Type &tType )
{
  //  The  domain parameter specifies a communication domain; this selects
  //  the protocol family which will be used for communication.
  //  These families are defined in  <sys/socket.h>
  //  PF_INET:
  //  IPv4 Internet protocols.
  //  SOCK_STREAM
  //  Provides sequenced, reliable, two-way, connection-based byte streams.
  //  An out-of-band data transmission mechanism may be supported.

  int nTrue = 1;

  switch ( tType )
  {
    case enumStream:
      if ( ( m_nSocket = ::socket( AF_INET, SOCK_STREAM, 0 ) ) < 0 )
        return -errno;
      break;
    case enumDatagram:
      if ( ( m_nSocket = ::socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP ) ) < 0 )
        return -errno;
      if ( ( ::setsockopt( m_nSocket, SOL_SOCKET, SO_BROADCAST,
                           ( const char * ) &nTrue, sizeof ( nTrue ) ) ) < 0 )
        return -errno;
      break;
    case enumMulticast:
      if ( ( m_nSocket = ::socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP ) ) < 0 )
        return -errno;
      break;
  }

  // If we got here, there were no errors.
  return enumNoError;
}


////////////////////////////////////////////////////////////////////////////////
/// This function creates a sockaddr_in struct based on the arguments
/// passed in. "sockaddr_in.sin_addr.s_addr" == INADDR_NONE if there was an error.

sockaddr_in SystemSocket::createSockAddr( const int &nPort,
                                          const string &szHost )
{
  sockaddr_in saAddr;
  memset( &saAddr, 0, sizeof ( saAddr ) );

  saAddr.sin_family = AF_INET;
  saAddr.sin_port = htons( nPort );
  saAddr.sin_addr.s_addr = ( szHost.size() == 0 ) ?
      htonl( INADDR_ANY ) : ::inet_addr( szHost.c_str() );

  return saAddr;
}

////////////////////////////////////////////////////////////////////////////////
//  This function converts the character string szHost into a network address
//  structure in the af address family, then copies  the  network  address
//  structure to psaAddr.

int SystemSocket::convertStrToAddr( const string &szHost, sockaddr_in *psaAddr )
{
  int nRet = enumNoError;

  if ( szHost.size() == 0 )
  {
    //  no host specified - use all available interface.
    psaAddr->sin_addr.s_addr = htonl( INADDR_ANY );
  }
  else
  {
#ifdef WIN32
    char[256] pcAddr;
    memset( pcAddr, 0, 256 );
    nRet = WSAStringToAddress( szHost.c_str(), AF_INET, NULL,
                               pcAddr, 256 );
    memcpy( psaAddr->sin_addr, pcAddr, 256 )
    switch ( nRet )
    {
      //  no error
      case 0:
        nRet = enumNoError;
        break;
      //  The specified Address buffer is too small.
      case WSAEFAULT:
        nRet = EDQUOT;
        break;
      //  Unable to translate the string into a sockaddr.
      case WSAEINVAL:
        nRet = EINVAL;
        break;
      //  The WS2_32.DLL has not been initialized. Call "WSAStartup".
      case WSANOTINITIALISED:
        nRet = EBADF;
        break;
      //  There was insufficient memory to perform the operation.
      case WSA_NOT_ENOUGH_MEMORY:
nRet = ENOMEM:
               break;
    }
#else  //  assume linux
    nRet = inet_pton( AF_INET, szHost.c_str(), ( void * )&psaAddr->sin_addr );

    if ( nRet > 0 )
    {
      //  this is "no error" for the "inet_pton" call.
      nRet = enumNoError;
    }
    else if ( nRet == 0 )
    {
      //  input not IPv4 dotted decimal string.
      nRet = EINVAL;
    }
    else
    {
      //  the only error is: errno == EAFNOSUPPORT
      nRet = EAFNOSUPPORT;
    }
#endif
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns enumNoError if the bind was successful. In the case of broadcast
/// or multicast, szHost should be "".
/// On error, may return:
/// enumInvalidSocketDescriptor
/// enumInvalidParameter
/// enumAddressNotSupported
/// enumAlreadyBound
/// enumNoAccess
/// enumAddressInUse
/// enumDescriptorNotSocket
/// enumAddressNotAvailable
/// enumSymbolicLoopExists
/// enumPathNameTooLong
/// enumPathDoesNotExist
/// enumNotEnoughMemory
/// enumPathIsNotDirectory
/// enumReadOnlyFileSystem

int SystemSocket::bind( const int &nPort, const string &szHost )
{
  if ( isValid() == false )
    return -EBADF;

  //  SO_REUSEADDR
  //  Indicates that the rules used in validating addresses supplied
  //  in a bind(2) call should allow reuse of local addresses.
  //  For PF_INET sockets this means that a socket may bind, except when
  //  there is an active listening socket bound to the address. When the
  //  listening socket is bound to INADDR_ANY with a specific port then it
  //  is not possible to bind to this port for any local address.

  //  try to set a reusable address.
  int nTrue = 1;
  if ( ::setsockopt( m_nSocket, SOL_SOCKET, SO_REUSEADDR,
                     ( const char * ) &nTrue, sizeof ( nTrue ) ) == -1 )
    return -EOPNOTSUPP;

  //  bind gives the socket 'm_nSocket' the local address 'm_saAddr' .
  //  my_addr is addrlen bytes long.  Traditionally, this is called
  //  "assigning a name to a socket." When a socket is created with
  //  socket(2) , it exists in a name space (address family) but has no
  //  name assigned.
  //  It is normally necessary to assign a local address using bind before
  //  a SOCK_STREAM socket may receive connections (see accept(2) ).
  //  The rules used in name binding vary between address families.
  //  Consult the manual entries in Section 7 for detailed information.
  //  For AF_INET see ip(7) , for AF_UNIX see unix(7) ,
  //  for AF_APPLETALK see ddp(7) , for AF_PACKET see packet(7) ,
  //  for AF_X25 see x25(7) and for AF_NETLINK see netlink(7) .

  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  if ( ::bind( m_nSocket, ( sockaddr * ) &saAddr, sizeof( saAddr ) ) != 0 )
  {
    int nRet = errno;
    //  convert to our error code as necessary.
    nRet = ( nRet == EINVAL ) ? ( EEXIST ) : ( nRet );
    nRet = ( nRet == EDESTADDRREQ ) ? ( EBADF ) : ( nRet );
    return -nRet;
  }

  // If we got here, there was no error.
  return enumNoError;
}

////////////////////////////////////////////////////////////////////////////////
//  returns 0 if the listen was successful, a negative error code otherwise.

int SystemSocket::listen()
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  To accept connections, a socket is first created with  socket(2) ,
  //  a willingness to accept incoming connections and a queue limit for
  //  incoming connections are specified with  listen , and then the
  //  connections are accepted with  accept(2) . The  listen call applies
  //  only to sockets of type  SOCK_STREAM or  SOCK_SEQPACKET .

  if ( ::listen( m_nSocket, enumMaxConnections ) != 0 )
  {
    nRet = errno;
    //  convert to our error code as necessary.
    nRet = ( nRet == EINVAL ) ? ( EISCONN ) : ( nRet );
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
//  sets the socket descriptor if it succeeds and returns 0, on failure returns
//  a negative error.

int SystemSocket::accept( SystemSocket &socNew )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  The accept function is used with connection-based socket types
  //  SOCK_SEQPACKET and  SOCK_RDM ). It extracts the first connection request
  //  on the queue of pending connections, creates a new connected socket with
  //  mostly the same properties as, and allocates a new file descriptor
  //  for the socket, which is returned. The newly created socket is no longer
  //  in the listening state. The original socket is unaffected by this call.
  //  Note that any per file descriptor flags (everything that can be set with
  //  the  F_SETFL fcntl, like non blocking or async state) are not inherited
  //  across an  accept .

  sockaddr_in saAddr;
  memset( &saAddr, 0, sizeof ( saAddr ) );
  int nAddrLen = sizeof( saAddr );

  if ( ( socNew.m_nSocket = ::accept( m_nSocket, ( sockaddr * ) &saAddr,
                                      ( socklen_t * ) &nAddrLen ) ) < 0 )
  {
    nRet = errno;
    //  convert to our error code as necessary.
    nRet = ( nRet == EINVAL ) ? ( EBUSY ) : ( nRet );
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
///  Sends a fixed-size chunk. Returns any error encountered.
///  nNumBytes will be modified to reflect the number of bytes actually sent.

int SystemSocket::sendChunk( char *pcData, int &nNumBytes )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  The  send call may be used only when the socket is in a  connected state
  //  (so that the intended recipient is known). The only difference between
  //  send and  write is the presence of  flags . With zero  flags parameter,
  //  send is equivalent to  write . Also,  send( s , buf , len ) is equivalent
  //  to  sendto( s, buf, len, NULL, 0 ).
  //  MSG_NOSIGNAL
  //  Requests not to send SIGPIPE on errors on stream oriented sockets when
  //  the other end breaks the connection. The EPIPE error is still returned.

  int nSent = 0;
  int nTotalSent = 0;
  for ( nTotalSent = 0; nTotalSent < nNumBytes; nTotalSent += nSent )
  {
    nSent = ::send( m_nSocket, pcData + nTotalSent,
                    nNumBytes - nTotalSent, MSG_NOSIGNAL );

    // Repeat if we got EAGAIN, otherwise drop out of the loop.
    if ( nSent < 0 )
    {
      nRet = errno;
      if ( nRet == EAGAIN )
        nSent = 0;
      else
        break;
    }
  }

  // report total sent
  nNumBytes = nTotalSent;

  // return any error
  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
///  Sends a fixed-size chunk. Returns any error encountered.
///  nNumBytes will be modified to reflect the number of bytes actually sent.

int SystemSocket::sendChunkTo( char *pcData, int &nNumBytes, const int &nPort,
                               const string &szHost )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  we need the sockaddr struct to pass the port and host.
  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  //  The sendto() function shall send a message through a connection-mode
  //  or connectionless-mode socket. If the socket is connectionless-mode,
  //  the message shall be sent to the address specified by dest_addr. If
  //  the socket is connection-mode, dest_addr shall be ignored.

  nRet = ::sendto( m_nSocket, pcData, nNumBytes, 0,
                   ( const sockaddr * )( &saAddr ), sizeof( saAddr ) );

  // Did we get an error?
  if ( nRet < 0 )
  {
    //  for an error, assume no bytes were sent.
    nNumBytes = 0;
    //  convert to our error code as necessary.
    nRet = errno;
  }
  // Otherwise, we sent some data, but did we send it all?
  else
  {
    if ( nRet == nNumBytes )
      nRet = enumNoError;
    else
    {
      nNumBytes = nRet;
      nRet = EMSGSIZE;
    }
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
///  Receives a fixed-size chunk. Returns any error encountered. nNumBytes will
///  be modified to reflect the number of bytes received.

int SystemSocket::recvChunk( char *pcData, int &nNumBytes )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  The  recvfrom and  recvmsg calls are used to receive messages from a
  //  socket, and may be used to receive data on a socket whether or not it is
  //  connection-oriented.
  //  If from is not NULL, and the underlying protocol provides the source
  //  address, this source address is filled in. The argument fromlen is a
  //  value-result parameter, initialized to the size of the buffer associated
  //  with from , and modified on return to indicate the actual size of the
  //  address stored there.
  //  The recv call is normally used only on a connected socket (see connect(2))
  //  and is identical to recvfrom with a NULL from parameter.
  //  a flag of EAGAIN can be used to indicate a timeout.

  memset( pcData, 0, nNumBytes );
  int nBytesRead = 0;
  int nBytesLeft = nNumBytes;

  //  read all the data.
  while ( nBytesLeft > 0 &&
          ( nRet = ::recv( m_nSocket, pcData + nBytesRead, nBytesLeft, 0 ) ) > 0 )
  {
    nBytesRead += nRet;
    nBytesLeft -= nRet;
  }

  //  did the peer close the socket?
  if ( nRet == 0 )
  {
    //  peer shut down - this is okay - just return empty data.
    memset( pcData, 0, nNumBytes );
    nBytesRead = 0;
    nBytesLeft = nNumBytes;
    //m_nSocket = -1;
    nRet = ECONNRESET;
  }
  //  did we complete okay?
  else if ( nRet > 0 )
  {
    if ( nBytesRead < nNumBytes )
      nRet = EFBIG;
    else
      nRet = enumNoError;
  }
  //  we had an error....
  else
  {
    //  convert to our error code as necessary.
    nRet = errno;
    nRet = ( nRet == EINVAL ) ? ( ENODATA ) : ( nRet );
  }

  //  set nNumBytes to nBytesRead to show how many were actually read.
  nNumBytes = nBytesRead;

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
/// Receives a fixed-size chunk. Returns any error encountered. nNumBytes will
/// be modified to reflect the number of bytes received. In the case of a
/// multicast, szHost should be "".

int SystemSocket::recvChunkFrom( char *pcData, int &nNumBytes,
                                 const int &nPort, const string &szHost )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  we need the sockaddr struct to pass the port and host.
  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  memset( pcData, 0, nNumBytes );
  int nBytesLeft = nNumBytes;
  int nBytesRead = 0;

  //  The recvfrom() function shall receive a message from a connection-
  //  mode or connectionless-mode socket. It is normally used with
  //  connectionless-mode sockets because it permits the application to
  //  retrieve the source address of received data.

  socklen_t tSockLen = sizeof( saAddr );
  nRet = ::recvfrom( m_nSocket, pcData, nNumBytes, MSG_WAITALL,
                     ( sockaddr * )( &saAddr ), &tSockLen );

  //  how many bytes are left and how many were read?
  nBytesLeft = nNumBytes - nRet;
  nBytesRead = nRet;

  //  did the peer close the socket?
  if ( nRet == 0 )
  {
    //  peer shut down - this is okay - just return empty data.
    memset( pcData, 0, nNumBytes );
    //m_nSocket = -1;
    nRet = ECONNRESET;
  }
  //  did we complete okay?
  else if ( nBytesLeft > 0 )
  {
    nRet = EFBIG;
  }
  //  we had an error....
  else
  {
    //  convert to our error code as necessary.
    nRet = errno;
    nRet = ( nRet == EINVAL ) ? ( ENODATA ) : ( nRet );
  }

  //  set nNumBytes to nBytesRead to show how many were actually read.
  nNumBytes = nBytesRead;

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
///  attempts to send data. Returns an error code.

int SystemSocket::send( const string &szData )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  The  send call may be used only when the socket is in a  connected state
  //  (so that the intended recipient is known). The only difference between
  //  send and  write is the presence of  flags . With zero  flags parameter,
  //  send is equivalent to  write . Also,  send( s , buf , len ) is equivalent
  //  to  sendto( s, buf, len, NULL, 0 ).
  //  MSG_NOSIGNAL
  //  Requests not to send SIGPIPE on errors on stream oriented sockets when
  //  the other end breaks the connection. The EPIPE error is still returned.

  //  nRet will now hold the number of bytes sent or an error code.
  nRet = ::send( m_nSocket, szData.c_str(), szData.size(), MSG_NOSIGNAL );

  // Was there an error?
  if ( nRet < 0 )
  {
    nRet = errno;
  }
  // Otherwise, we sent some data, but did we send it all?
  else
  {
    if ( static_cast<unsigned int>( nRet ) == szData.size() )
      nRet = enumNoError;
    else
      nRet = EMSGSIZE;
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
///  attempts to send data. Returns an error code.

int SystemSocket::sendTo( const string &szData,
                          const int &nPort, const string &szHost )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  we need the sockaddr struct to pass the port and host.
  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  //  The sendto() function shall send a message through a connection-
  //  mode or connectionless-mode socket. If the socket is connectionless-
  //  mode, the message shall be sent to the address specified by dest_addr.
  //  If the socket is connection-mode, dest_addr shall be ignored.

  //  nRet will now hold the number of bytes sent or an error code.
  nRet = ::sendto( m_nSocket, szData.c_str(), szData.size(), 0,
                   ( sockaddr * )( &saAddr ), sizeof( saAddr ) );

  // Was there an error?
  if ( nRet < 0 )
  {
    nRet = errno;
  }
  //  Otherwise, we sent some data, but did we send it all?
  else
  {
    if ( static_cast<unsigned int>( nRet ) == szData.size() )
      nRet = enumNoError;
    else
      nRet = EMSGSIZE;
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::recv( string &szData )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  The  recvfrom and  recvmsg calls are used to receive messages from a
  //  socket, and may be used to receive data on a socket whether or not it is
  //  connection-oriented.
  //  If from is not NULL, and the underlying protocol provides the source
  //  address, this source address is filled in. The argument fromlen is a
  //  value-result parameter, initialized to the size of the buffer associated
  //  with from , and modified on return to indicate the actual size of the
  //  address stored there.
  //  The recv call is normally used only on a connected socket (see connect(2))
  //  and is identical to recvfrom with a NULL from parameter.
  //  a flag of EAGAIN can be used to indicate a timeout.

  // Get some memory to hold the data.
  char pcBuf[ enumMaxRecv + 1 ];
  memset( pcBuf, 0, enumMaxRecv + 1 );

  // Reset the data, just in case nothing comes...
  szData = string( "" );
  nRet = ::recv( m_nSocket, pcBuf, enumMaxRecv, 0 );

  // Did the peer shut down? - this is okay - just return empty data.
  if ( nRet == 0 )
  {
    //m_nSocket = -1;
    nRet = ECONNRESET;
  }
  // Otherwise, this is the number of bytes received.
  else if ( nRet > 0 )
  {
    szData = string( pcBuf, nRet );
    nRet = enumNoError;
  }
  // Convert to our error code as necessary.
  else
  {
    nRet = errno;
    nRet = ( nRet == EINVAL ) ? ( ENODATA ) : ( nRet );
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::recvFrom( string &szData,
                            const int &nPort, const string &szHost )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  we need the sockaddr struct to pass the port and host.
  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  //  The  recvfrom()  function  shall receive a message from a connection-
  //  mode or connectionless-mode socket. It is  normally  used  with
  //  connectionless-mode sockets because it permits the application to
  //  retrieve the source address of received data.

  //  get some memory to hold the data.
  char pcBuf[ enumMaxRecv + 1 ];
  memset( pcBuf, 0, enumMaxRecv + 1 );

  //  reset the data, just in case nothing comes...
  szData = string( "" );
  socklen_t tSockLen = sizeof( saAddr );
  nRet = ::recvfrom( m_nSocket, pcBuf, enumMaxRecv, MSG_WAITALL,
                     ( sockaddr * )( &saAddr ), &tSockLen );

  // Did the peer shut down? - this is okay - just return empty data.
  if ( nRet == 0 )
  {
    memset( pcBuf, 0, enumMaxRecv + 1 );
    nRet = ECONNRESET;
  }
  // Otherwise, this is the number of bytes received.
  else if ( nRet > 0 )
  {
    szData = string( pcBuf );
    nRet = enumNoError;
  }
  else
  {
    //  convert to our error code as necessary.
    nRet = -errno;
    nRet = ( nRet == EINVAL ) ? ( ENODATA ) : ( nRet );
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::connect()
{
  return connect( m_nPort, m_szHost );
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::connect( const int &nPort, const string &szHost )
{
  int nRet = enumNoError;

  // Store the information for later.
  m_szHost = szHost;
  m_nPort = nPort;

  if ( isValid() == false )
    return -EBADF;

  //  we need the sockaddr struct to pass the port and host.
  sockaddr_in saAddr = createSockAddr( nPort, szHost );
  if ( saAddr.sin_addr.s_addr == INADDR_NONE )
    return -EINVAL;

  // we need a set to use with 'select' so we can time-out.
  fd_set setRead;
  FD_ZERO( &setRead );
  FD_SET( m_nSocket, &setRead );

  fd_set setWrite;
  FD_ZERO ( &setWrite );
  FD_SET ( m_nSocket, &setWrite );

  // we need a timeval struct to hold our timeout value.
  timeval tvTimeout;
  tvTimeout.tv_sec = m_nConnectTimeout / 1000;
  tvTimeout.tv_usec = m_nConnectTimeout % 1000;

  // set the socket descriptor to non-blocking BEFORE we call 'connect'.
  setNonBlocking( true );

  //  The  file descriptor sockfd must refer to a socket.  If the socket
  //  is of type SOCK_DGRAM then the serv_addr address is the address to
  //  which datagrams are sent by default, and the only address from which
  //  datagrams are received.  If the socket is of type SOCK_STREAM or
  //  SOCK_SEQPACKET,  this call attempts to make a connection to another
  //  socket.  The other socket is specified by serv_addr, which is an
  //  address (of length addrlen) in the communications space of the
  //  socket.  Each communications space interprets the serv_addr
  //  parameter in its own way.
  //  Generally, connection-based protocol sockets may successfully
  //  connect only once; connectionless protocol sockets may use connect
  //  multiple  times to  change their association.  Connectionless
  //  sockets may dissolve the association by connecting to an address
  //  with the sa_family member of sockaddr set to AF_UNSPEC.

  // will we connect immediately?
  if ( ::connect( m_nSocket, ( sockaddr * ) &saAddr, sizeof( saAddr ) ) < 0 )
  {
    // save errno for later.
    nRet = errno;

    // the only acceptable error is 'in progress'.
    if ( nRet == EINPROGRESS )
    {
      // reset the return value.
      nRet = enumNoError;
      int nNum = ::select( m_nSocket + 1, &setRead, &setWrite, NULL, &tvTimeout );

      // did we timeout (no descriptors are ready)?
      if ( nNum == 0 )
      {
        nRet = ETIMEDOUT;
        close();
      }
      // did select have an error?
      else if ( nNum < 0 )
      {
        nRet = errno;
        close();
      }
      // n > 0 - are we possibly connected?
      else if ( FD_ISSET( m_nSocket, &setWrite )
                || FD_ISSET( m_nSocket, &setRead ) )
      {
        int nErr = 0;
        socklen_t nSizeErr = sizeof( nErr );
        if ( ::getsockopt( m_nSocket, SOL_SOCKET, SO_ERROR, &nErr,
                           &nSizeErr ) < 0 )
        {
          nRet = errno;
          close();
        }
        else if ( nErr != 0 )
        {
          nRet = nErr;
          close();
        }
        /*
        sockaddr_in addrPeer;
        socklen_t nSizePeer = sizeof( addrPeer );

        // does our peer have a name?
        if ( getpeername( m_nSocket, (sockaddr *)( &addrPeer ),
            &nSizePeer) < 0 )
        {
          nRet = ETIMEDOUT;
          close();
        }
        */
      }
    }
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
/// Join a multicast group in order to receive data.

int SystemSocket::join( const int &nPort,
                        const string &szMulticast )
{
  // Store the information for later.
  m_szHost = szMulticast;
  m_nPort = nPort;

  if ( isValid() == false )
    return -EBADF;

  // Bind such that any address is ok.
  string szAnyAddress = "";
  int nRet = enumNoError;
  if ( ( nRet = bind( nPort, szAnyAddress ) ) < 0 )
    return nRet;

  // use setsockopt() to request that the kernel join a multicast group.
  struct ip_mreq mreq;
  mreq.imr_multiaddr.s_addr = ::inet_addr( szMulticast.c_str() );
  mreq.imr_interface.s_addr = htonl( INADDR_ANY );
  if ( ::setsockopt( m_nSocket, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                     &mreq, sizeof(mreq) ) < 0 )
  {
    return -errno;
  }

  // If we got here, there was no error.
  return enumNoError;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::close()
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  close closes a file descriptor, so that it no longer refers to any file
  //  and may be reused. Any locks held on the file it was associated with,
  //  and owned by the process, are removed (regardless of the file
  //  descriptor that was used to obtain the lock).
#ifdef WIN32
  if ( ( nRet = closesocket( m_nSocket ) ) != 0 )
#else
  if ( ( nRet = ::close( m_nSocket ) ) != 0 )
#endif
  {
    //  convert to our error code as necessary.
    nRet = errno;
  }
  m_nSocket = -1;

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::setRecvTimeout( const int &nMillis )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

  //  create the timeval structure.
  timeval tvTimeout;
  tvTimeout.tv_sec = nMillis / 1000;
  tvTimeout.tv_usec = nMillis * 1000 - tvTimeout.tv_sec * 1000000;

  //  try to set a time on the recv.
  if ( ::setsockopt( m_nSocket, SOL_SOCKET, SO_RCVTIMEO,
                     ( const char * ) &tvTimeout, sizeof ( tvTimeout ) ) == -1 )
  {
    //  convert to our error code as necessary.
    nRet = errno;
    nRet = ( nRet == ENOPROTOOPT ) ? ( EOPNOTSUPP ) : ( nRet );
  }

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////

int SystemSocket::setNonBlocking( const bool &oIsNonBlocking )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

#ifdef WIN32
  // Set the socket I/O mode: In this case FIONBIO
  // enables or disables the blocking mode for the
  // socket based on the numerical value of iMode.
  // If uiMode = 0, blocking is enabled;
  // If uiMode != 0, non-blocking mode is enabled.
  unsigned int uiMode = ( oIsNonBlocking == true ) ? ( 1 ) : ( 0 );
  ioctlsocket( m_nSocket, FIONBIO, &uiMode );
#else
  //  fcntl performs one of various miscellaneous operations on  fd .
  //  The operation in question is determined by  cmd .
  //  F_GETFL
  //  Read the file descriptor's flags.
  int nOpts = ::fcntl( m_nSocket, F_GETFL );
  if ( nOpts >= 0 )
  {
    nOpts = ( oIsNonBlocking == true ) ?
            ( nOpts | O_NONBLOCK ) : ( nOpts & ~O_NONBLOCK );
    //  add in this new setting.
    ::fcntl ( m_nSocket, F_SETFL, nOpts );
  }
#endif

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if the nagle algorithm is disabled, false otherwise.

bool SystemSocket::isNagleDisabled() const
{
  return m_oIsNagleDisabled;
}

////////////////////////////////////////////////////////////////////////////////
/// Disables the algorithm that conglomerates lots of
/// small sends into a large one.

int SystemSocket::disableNagle( const bool &oDisable )
{
  int nRet = enumNoError;

  if ( isValid() == false )
    return -EBADF;

#ifdef WIN32
  // todo: Look up the way to do this in windows.
#else
  int nFlag = ( oDisable == true ) ? ( 1 ) : ( 0 );
  nRet = ::setsockopt( m_nSocket, IPPROTO_TCP, TCP_NODELAY,
                       ( char * ) &nFlag, sizeof( int ) );

  // If we got an error, the state of the nagle algorithm usage
  // has not changed. Return the error.
  if ( nRet == -1 )
  {
    nRet = errno;
  }
  // Otherwise, store the new state of the algorithm usage.
  else
  {
    m_oIsNagleDisabled = oDisable;
  }
#endif

  return -nRet;
}

////////////////////////////////////////////////////////////////////////////////
//  this function will block until data is ready to be read on the socket.
/*
int SystemSocket::waitForData()
{
  if ( isValid() == false )
    return -EBADF;

  fd_set ReadSet;
  FD_ZERO( &ReadSet );
  FD_SET( m_nSocket, &ReadSet );

  //  The functions select and pselect wait for a number of file
  //  descriptors to change status.
  int nRet = ::select( m_nSocket+1, &ReadSet, NULL, NULL, NULL );

  if ( nRet == -1 && errno == EINTR )
  {
  }
  //  there is possibly data available -
  //  or the connection may have been closed.
  else if ( FD_ISSET( m_nSocket, &ReadSet ) )
  {
  }
  else if ( nRet < 0 )
  {
  }
  else
  {
  }
    return- nRet;
}
*/
////////////////////////////////////////////////////////////////////////////////
///  return the message concerning the error.

string SystemSocket::getErrorMsg( const int &nError )
{
  switch ( nError )
  {
    case enumNoError:
      return string( "No Error" );
      break;
    case -EBUSY:
      return string( "Unwilling to accept connections" );
      break;
    case -EEXIST:
      return string( "Shut down or already bound to an address" );
      break;
    case -EADDRINUSE:
      return string( "Address already in use" );
      break;
    case -EADDRNOTAVAIL:
      return string( "Address not available" );
      break;
    case -EAFNOSUPPORT:
      return string( "Address not supported" );
      break;
    case -ECONNABORTED:
      return string( "Connection aborted" );
      break;
    case -ECONNRESET:
      return string( "Connection reset by peer" );
      break;
    case -EALREADY:
      return string( "A previous connection attempt has not been completed" );
      break;
    case -EINPROGRESS:
      return string( "Connection cannot be completed immediately" );
      break;
    case -ECONNREFUSED:
      return string( "Connection refused" );
      break;
    case -ENOTSOCK:
      return string( "Descriptor not a socket" );
      break;
    case -EISDIR:
      return string( "An empty pathname was specified" );
      break;
    case -EDESTADDRREQ:
      return string( "Destination address required or socket not bound" );
      break;
    case -EHOSTUNREACH:
      return string( "Host unreachable" );
      break;
    case EINVAL:
      return string( "Invalid parameter" );
      break;
    case -EBADF:
      return string( "Invalid socket descriptor" );
      break;
    case -EIO:
      return string( "IO error" );
      break;
    case -EPIPE:
      return string( "Local shutdown" );
      break;
    case -EMFILE:
      return string( "Max descriptors already open" );
      break;
    case -ENETDOWN:
      return string( "Network down" );
      break;
    case -ENETUNREACH:
      return string( "Network unreachable" );
      break;
    case -EACCES:
      return string( "No Access" );
      break;
    case -ENOBUFS:
      return string( "No buffers available" );
      break;
    case -ENODATA:
      return string( "No out-of-band data available" );
      break;
    case -ENOTCONN:
      return string( "Not connected" );
      break;
    case -ENOMEM:
      return string( "Not enough memory" );
      break;
    case -EOPNOTSUPP:
      return string( "Type and/or protocol is not supported for this operation" );
      break;
    case -EDQUOT:
      return string( "Message too big" );
      break;
    case -EFBIG:
      return string( "Only partial data was received" );
      break;
    case -EMSGSIZE:
      return string( "Only partial data was sent" );
      break;
    case -ENAMETOOLONG:
      return string( "Path name too long" );
      break;
    case -ENOENT:
      return string( "Path does not exist" );
      break;
    case -ENOTDIR:
      return string( "Path is not a directory" );
      break;
    case -EPROTO:
      return string( "Protocol error" );
      break;
    case -EROFS:
      return string( "Read-only file system" );
      break;
    case -EINTR:
      return string( "Interrupt signal received" );
      break;
    case -EISCONN:
      return string( "Socket is already connected" );
      break;
    case -ELOOP:
      return string( "Symbolic loop exists" );
      break;
    case -EDOM:
      return string( "Timeout field too big" );
      break;
    case -ETIMEDOUT:
      return string( "Timed out" );
      break;
    case -EAGAIN:
      return string( "Would block (or timed out)" );
      break;
    case -EPROTOTYPE:
      return string( "Wrong address type" );
      break;
    case enumUnknownError:
      return string( "Unknown error" );
      break;
  }
  return string( "" );
}

////////////////////////////////////////////////////////////////////////////////
/// Return the standard host name for the current machine.

int SystemSocket::getHostName( string &szHostname )
{
  char pcHostName[256];
  memset( pcHostName, 0, 256 );
  int nRet = ::gethostname( pcHostName, 255 );
  szHostname = string( pcHostName );
  return nRet;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the interface named 'szName'.
/// @param szName The name of the interface.
/// @return The interface which corresponds to the name 'szName', or the
/// loopback if it is not found.

SystemSocket::Interface SystemSocket::getInterface( const string &szName )
{
  // As a default, our interface is undefined.
  Interface interface( "", "127.0.0.1", "255.0.0.0" );

  // First, get a vector of all the interfaces.
  vector<Interface> vecInterfaces;
  SystemSocket::getInterfaces( vecInterfaces );

  // find the one with the name we want.
  for ( unsigned int ii = 0; ii < vecInterfaces.size(); ii++ )
  {
    if ( vecInterfaces[ii].getName() == szName )
    {
      interface = vecInterfaces[ii];
      break;
    }
  }

  return interface;
}

////////////////////////////////////////////////////////////////////////////////
/// This returns the local interfaces as a vector.
/// The return value is the number of interfaces found.

int SystemSocket::getInterfaces( vector<SystemSocket::Interface> &vecInterfaces )
{
  /**
    SIOCGIFCONF
    Return a list of interface (transport layer) addresses. This currently means
    only addresses of the AF_INET (IPv4) family for compatibility. The user
    passes a ifconf structure as argument to the ioctl. It contains a pointer to
    an array of ifreq structures in ifc_req and its length in bytes in ifc_len.
    The kernel fills the ifreqs with all current L3 interface addresses that are
    running: ifr_name contains the interface name (eth0:1 etc.), ifr_addr the
    address. The kernel returns with the actual length in ifc_len. If ifc_len is
    equal to the original length the buffer probably has overflowed and you
    should retry with a bigger buffer to get all addresses. When no error occurs
    the ioctl returns 0; otherwise -1. Overflow is not an error.

    See: http://www.die.net/doc/linux/man/man7/netdevice.7.html
  **/

  // This will be an error (<0) or the number of interfaces (>0).
  int nError = 0;
  ifconf ifc;
  sockaddr_in addrSock;

  // Make sure the list is empty.
  vecInterfaces.clear();

  // Create a fd with the correct protocol.
  int fdSock = ::socket( AF_INET, SOCK_DGRAM, 0 );

  if ( fdSock < 0 )
  {
    nError = -1;
  }
  else
  {
    int nRetVal = 0;
    unsigned int uiAmt = 0;
    memset( &ifc, 0, sizeof( ifc ) );

    // Repeat the ioctl call and resize the data structure until
    // it is more than big enough to hold information for all the network
    // interfaces. Of course, we will stop if we have an error.
    while ( nRetVal == 0 &&
            sizeof( struct ifreq ) * uiAmt <= ( unsigned int )( ifc.ifc_len ) )
    {
      // Try an increment of 10....
      uiAmt += 10;
      ifc.ifc_len = sizeof( struct ifreq ) * uiAmt;
      ifc.ifc_buf = ( char * )( realloc( ifc.ifc_buf, ifc.ifc_len ) );
      // Make the call.
      nRetVal = ::ioctl( fdSock, SIOCGIFCONF, &ifc );
    }

    if ( nRetVal < 0 )
    {
      nError = -1;
    }
    else
    {
      //std::cout << "Final size is: " << uiAmt << std::endl;
      ifreq *ifr = ifc.ifc_req;
      for ( unsigned int ii = 0; ii < uiAmt; ii++ )
      {
        // The name is set in the struct.
        string szName( ifr->ifr_name );

        // The IP address is set in the struct.
        struct in_addr *psia1 = ( struct in_addr * )
                                ( &( ifr->ifr_addr.sa_data[sizeof addrSock.sin_port] ) );
        string szIP( string( ::inet_ntoa( *psia1 ) ) );

        // Overwrite the same piece of memory with the broadcast address.
        nRetVal = ::ioctl( fdSock, SIOCGIFBRDADDR, ifr );

        // The broadcast address has replaced the IP address in the struct.
        struct in_addr *psia2 = ( struct in_addr * )
                                ( &( ifr->ifr_broadaddr.sa_data[sizeof addrSock.sin_port] ) );
        string szBCast( string( ::inet_ntoa( *psia2 ) ) );

        // Add this info to our vector.
        vecInterfaces.push_back( Interface( szName, szIP, szBCast ) );

        //std::cout << "adding: '" << szName << "' '" << szIP
        //    << "' '" << szBCast <<"'." << std::endl;
        ifr++;
      }
      // Return a positive size as the error value (no error).
      nError = vecInterfaces.size();
    }
    // Free the memory we allocated.
    free( ifc.ifc_buf );
  }
  return nError;
}

////////////////////////////////////////////////////////////////////////////////
