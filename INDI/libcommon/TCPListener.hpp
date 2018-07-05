/// TCPListener.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_TCP_LISTENER_HPP
#define PCF_TCP_LISTENER_HPP

#include <string>
#include "errno.h"
#include "Thread.hpp"
#include "SystemSocket.hpp"
#include "TCPConnection.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class TCPListener : public pcf::Thread
{
  // We have a special exception defined here.
  public:
    class Error : public std::runtime_error
    {
      public:
        Error( const std::string &szMsg ) : std::runtime_error( szMsg ) {}
    };
/*
  /// constants
  public:
    enum Error
    {
      enumNoError =                0, // This must stay zero.
      enumBindFailed =            -EFAULT,
      enumStartFailed =           -EAGAIN,
      enumAlreadyStarted =        -EEXIST,
      enumUnknownError =          -9999
    };
*/
    // Constructor/destructor
  public:
    TCPListener();
    TCPListener( const char *pcName );
    TCPListener( const std::string &szName );
    virtual ~TCPListener();

    // Methods.
  public:
    /// Close one connection to this server.
    void close( const int &nConnectionId );
    /// Close all connections to this server.
    void closeAll();
    /// Return the message concerning the error.
    //std::string getErrorMsg( const int &nErr ) const;
    /// There is one object for each connection.
    TCPConnection::Map *getConnections();
    /// This returns the name of this server.
    std::string getName() const;
    /// Return the port we are using currently.
    int getPort() const;
    /// Are we currently listening for new connections?
    bool isListening() const;
    /// Listen for new connections.
    void listen( const int &nPort, TCPConnection *pConnectionPrototype,
                 const std::string &szHost = std::string( "" ) );
    /// Set the name this server.
    void setName( const char *pcName );
    void setName( const std::string &szName );
    /// How many connections do we currently have?
    int size() const;
    /// Stop listening for new connections (keeps old connections).
    void stopListening();

    // Helper functions.
  protected:
    /// This function will be called after the thread exits the main
    /// update loop. (After calling "execute" for the last time).
    virtual void afterExecute();
    /// This function will be called before the thread enters the main
    /// update loop. (Before calling "execute" the first time).
    virtual void beforeExecute();
    /// This function will be called every "Interval" milliseconds.
    virtual void execute();

    // Variables
  private:
    /// Are we waiting for connections?
    bool m_oIsListening;
    /// What is the name of our server?
    std::string m_szName;
    /// The port we will use to listen for incoming connections.
    //int m_nPort;
    /// The interface we will use to listen for incoming connections.
    //std::string m_szHost;
    /// The socket we will use to listen on.
    SystemSocket m_socServer;
    /// Should we stop listening for new connections?
    //bool m_oStopListening;
    /// This is the prototype of Connections for this server
    /// it is not destroyed when this class is deleted.
    TCPConnection *m_pConnectionPrototype;
    /// A map of all connected Connections.
    TCPConnection::Map m_mapConnections;
    /// The currently allocated prototype.
    TCPConnection *m_pConnection;
    /// The running count of connections (and the identifier of it)
    unsigned long long m_nnCount;

    /// A map of the TCP/IP connections and their privileges.
    //map<int, int> m_mapPrivilege;
    /// The pointer to the last connection made - used by "WaitForConnections".
    //Socket *m_psocLastConnected;
    /// A mutex for the call to the process function.
    //Mutex m_mutProcessData;

}; // Class TCPListener
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_TCP_LISTENER_HPP
