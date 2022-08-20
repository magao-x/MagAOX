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

#include <stdexcept>
#include <exception>
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
  // We have a special exception defined here.
  public:
    class Error : public std::runtime_error
    {
      public:
        Error( const std::string &szMsg ) : std::runtime_error( szMsg ) {}
    };

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
      UnknownType =     -1,
      Stream =          0,
      Datagram =        1,
      enumMulticast =   2
    };

    enum
    {
      enumMaxHostName =     1024,
      enumMaxConnections =  1024,
      enumMaxRecv =         2048
    };
/*
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
*/
  // Constructor/destructor
  public:
    SystemSocket();
    SystemSocket( const Type &tType,
                  const int &nPort,
                  const std::string &szHost );
    virtual ~SystemSocket();
    SystemSocket( const SystemSocket &rhs );
    const SystemSocket &operator= ( const SystemSocket &rhs );

  // Methods
  public:
    virtual void accept( SystemSocket &socNew );
    virtual void bind();
    virtual void close();
    virtual void connect();
    static sockaddr_in convertStrToAddr( const std::string &szHost );
    virtual void create();
    /// This function creates a sockaddr_in struct based on the arguments.
    static sockaddr_in createSockAddr( const int &nPort,
                                       const std::string &szHost = std::string( "" ) );
    /// Disables the algorithm that conglomerates lots of
    /// small sends into a large one.
    void disableNagle( const bool &oDisable );
    /// Returns the socket descriptor.
    int getFd() const;
    /// Returns the host set in the constructor.
    std::string getHost() const;
    /// Get the interface named 'szName'.
    static Interface getInterface( const std::string &szName );
    // This returns the local interfaces as a vector.
    /// The return value is the number of interfaces found.
    static int getInterfaces( std::vector<Interface> &vecInterfaces );
    /// This is the last error that occurred when manipluating the socket.
    int getLastError() const;
    virtual void listen();
    /// Gets the standard host name for the current machine.
    static std::string getLocalHostName();
    /// Returns the option set for the socket.
    void getOption( const int &nLevel,
                    const int &nOption,
                    void *pvOptionValue,
                    socklen_t &nOptionLength );
    /// Returns the port set in the constructor.
    int getPort() const;
    /// Returns the type set in the constructor.
    Type getType() const;
    /// Returns true if the socket is bound to a port.
    bool isBound() const;
    /// Returns true if the nagle algorithm is disabled, false otherwise.
    bool isNagleDisabled() const;
    /// Returns true if the listen or connect was successful, false otherwise.
    bool isValid() const;
    virtual void join();
    virtual std::string recv();
    virtual std::string recvFrom();
    virtual void recvChunk( char *pcData,
                            int &nNumBytes );
    /// Receives a fixed-size chunk. Returns any error encountered. nNumBytes will
    /// be modified to reflect the number of bytes received.
    virtual void recvChunkFrom( char *pcData,
                                int &nNumBytes );
    virtual void send( const std::string &szData );
    virtual void sendTo( const std::string &szData );
    virtual void sendChunk( char *pcData,
                            int &nNumBytes );
    virtual void sendChunkTo( char *pcData,
                              int &nNumBytes );
    /// Set the connect timeout in milliseconds.
    void setConnectTimeout( const int &nTimeout );
    void setNonBlocking( const bool &oIsNonBlocking );
    void setOption( const int &nLevel,
                    const int &nOption,
                    void *pvOptionValue,
                    const socklen_t &nOptionLength );
    void setRecvTimeout( const int &nMillis );

  // Member variables
  private:
    /// The type of socket this is.
    SystemSocket::Type m_tType;
    /// The last error that occurred when manipulating the socket.
    int m_nLastError;
    /// The socket file descriptor.
    int m_nSocket;
    /// The timeout set for the 'connect' call.
    int m_nConnectTimeout;
    /// Stored host used.
    std::string m_szHost;
    /// Stored port used.
    int m_nPort;
    /// Are we using the nagle algorithm?
    bool m_oIsNagleDisabled;
    /// have we bound this socket to a port?
    bool m_oIsBound;

}; // class SystemSocket
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_SYSTEM_SOCKET_HPP
