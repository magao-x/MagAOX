/// Socket.hpp
///
/// @author Paul Grenz
///
/// The "SystemSocket" class wraps the basic socket functionality as
/// exposed by the system.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_SYSTEM_SOCKET_HPP
#define PCF_SYSTEM_SOCKET_HPP

#include <errno.h>
#include <string>
#include <vector>
#ifdef WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <fcntl.h>
#endif

namespace pcf
{
class SystemSocket
{
  // This class holds some information about an interface.
  public:
    class Interface
    {
      public:
        Interface( const std::string &szName, const std::string &szIp,
                   const std::string &szBroadcast )
          : m_szName( szName ), m_szIp( szIp ), m_szBroadcast( szBroadcast ) {}
        ~Interface() {}
        std::string getName() const
        {
          return m_szName;
        }
        std::string getIp() const
        {
          return m_szIp;
        }
        std::string getBroadcast() const
        {
          return m_szBroadcast;
        }
      private:
        std::string m_szName;
        std::string m_szIp;
        std::string m_szBroadcast;
    };

  // Constants.
  public:
    enum Type
    {
      enumStream =      0,
      enumDatagram =    1,
      enumMulticast =   2
    };

    enum
    {
      enumMaxHostName =     1024,
      enumMaxConnections =  1024,
      enumMaxRecv =         2048
    };

    enum Error
    {
      // The 'no error' value must stay 0
      // to be compatible with the system errors.
      enumNoError  =                 0,
      enumAlreadyBound  =           -EEXIST,
      enumAddressInUse  =           -EADDRINUSE,
      enumAddressNotAvailable  =    -EADDRNOTAVAIL,
      enumAddressNotSupported  =    -EAFNOSUPPORT,
      enumAlreadyConnected  =       -EISCONN,
      enumBusy =                    -EBUSY,
      enumConnectionAborted  =      -ECONNABORTED,
      enumConnectionBusy  =         -EINPROGRESS,
      enumConnectionInProgress  =   -EALREADY,
      enumConnectionRefused  =      -ECONNREFUSED,
      enumDescriptorNotSocket  =    -ENOTSOCK,
      enumEmptyPathname  =          -EISDIR,
      enumHostUnreachable  =        -EHOSTUNREACH,
      enumInterruptRecv  =          -EINTR,
      enumInvalidParameter  =       -EINVAL,
      enumInvalidSocketDescriptor = -EBADF,
      enumIoError  =                -EIO,
      enumLocalShutdown  =          -EPIPE,
      enumMaxDescriptorsUsed  =     -EMFILE,
      enumMessageTooBig  =          -EDQUOT,
      enumNetworkDown  =            -ENETDOWN,
      enumNetworkUnreachable  =     -ENETUNREACH,
      enumNoAccess  =               -EACCES,
      enumNoBuffers  =              -ENOBUFS,
      enumNoOobData  =              -ENODATA,
      enumNotConnected  =           -ENOTCONN,
      enumNotEnoughMemory  =        -ENOMEM,
      enumOperationNotSupported  =  -EOPNOTSUPP,
      enumPartialDataRecv  =        -EFBIG,
      enumPartialDataSent  =        -EMSGSIZE,
      enumPathDoesNotExist  =       -ENOENT,
      enumPathIsNotDirectory  =     -ENOTDIR,
      enumPathNameTooLong  =        -ENAMETOOLONG,
      enumPeerShutdown  =           -ECONNRESET,
      enumProtocolError  =          -EPROTO,
      enumReadOnlyFileSystem  =     -EROFS,
      enumSocketAlreadyConnected  = -EISCONN,
      enumSocketNotBound  =         -EDESTADDRREQ,
      enumSymbolicLoopExists  =     -ELOOP,
      enumTimeoutFieldTooBig  =     -EDOM,
      enumTimedOut  =               -ETIMEDOUT,
      enumWouldBlock =              -EAGAIN,
      enumWrongAddressType  =       -EPROTOTYPE,
      enumUnknownError =            -999999
    };

  // Constructor/destructor
  public:
    SystemSocket();
    virtual ~SystemSocket();
    SystemSocket( const SystemSocket &rhs );
    const SystemSocket &operator= ( const SystemSocket &rhs );

    // Server-type operations
  public:
    virtual int accept( SystemSocket &socNew );
    /// Returns enumNoError if the bind was successful. In the case of broadcast
    /// or multicast, szHost should be "".
    virtual int bind( const int &nPort = 0,
                      const std::string &szHost = std::string( "" ) );
    virtual int create( const Type &tType = enumStream );
    virtual int listen();

    // Client-type operations
  public:
    virtual int connect();
    virtual int connect( const int &nPort,
                         const std::string &szHost );

  // For both
  public:
    virtual int close();
    static int convertStrToAddr( const std::string &szHost,
                                 sockaddr_in *psaAddr );
    /// This function creates a sockaddr_in struct based on the arguments
    /// passed in. "sockaddr_in.sin_addr.s_addr" == INADDR_NONE if there was an error.
    sockaddr_in createSockAddr( const int &nPort,
                                const std::string &szHost = std::string( "" ) );
    /// Disables the algorithm that conglomerates lots of
    /// small sends into a large one.
    int disableNagle( const bool &oDisable );
    /// Return the message concerning the error.
    static std::string getErrorMsg( const int &nError );
    std::string getHost() const;
    /// Returns the socket descriptor.
    int getFd() const;
    // Gets the standard host name for the current machine.
    static int getHostName( std::string &szHostname );
    /// Get the interface named 'szName'.
    static Interface getInterface( const std::string &szName );
    // This returns the local interfaces as a vector.
    // The return value is the number of interfaces found.
    static int getInterfaces( std::vector<Interface> &vecInterfaces );
    int getPort() const;
    /// Returns true if the nagle algorithm is disabled, false otherwise.
    bool isNagleDisabled() const;
    /// Returns true if the listen or connect was successful, false otherwise.
    bool isValid() const;
    /// Join a multicast group in order to receive data.
    int join( const int &nPort,
              const std::string &szMulticast );
    virtual int recv( std::string &szData );
    virtual int recvFrom( std::string &szData,
                          const int &nPort,
                          const std::string &szHost );
    virtual int recvChunk( char *pcData,
                           int &nNumBytes );
    /// Receives a fixed-size chunk. Returns any error encountered. nNumBytes will
    /// be modified to reflect the number of bytes received. In the case of a
    /// multicast, szHost should be "".
    virtual int recvChunkFrom( char *pcData,
                               int &nNumBytes,
                               const int &nPort,
                               const std::string &szHost = std::string( "" ) );
    virtual int send( const std::string &szData );
    virtual int sendTo( const std::string &szData,
                        const int &nPort,
                        const std::string &szHost );
    virtual int sendChunk( char *pcData,
                           int &nNumBytes );
    virtual int sendChunkTo( char *pcData,
                             int &nNumBytes,
                             const int &nPort,
                             const std::string &szHost );
    /// Set the connect timeout in milliseconds.
    inline void setConnectTimeout( const int &nTimeout );
    int setNonBlocking( const bool &oIsNonBlocking );
    int setRecvTimeout( const int &nMillis );

  // Member variables
  private:
    // The socket file descriptor.
    int m_nSocket;
    int m_nConnectTimeout;
    // Stored host from the first connect/join call.
    std::string m_szHost;
    // Stored port from the first connect/join call.
    int m_nPort;
    // Are we using the nagle algorithm?
    bool m_oIsNagleDisabled;

}; // class SystemSocket
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_SYSTEM_SOCKET_HPP
