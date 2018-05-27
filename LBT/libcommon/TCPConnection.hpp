// $Id: TCPConnection.hpp,v 1.4 2007/09/11 03:46:50 pgrenz Exp $
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_TCP_CONNECTION_HPP
#define PCF_TCP_CONNECTION_HPP

////////////////////////////////////////////////////////////////////////////////
///
/// @author Paul Grenz
///
/// The "TCPConnection" class is a wrapper around an execute function
/// registered for some task and a socket to receive the data.
/// The task can be the arrival of a command, through a socket, the receipt of
/// an event or a spawning of a thread. To use this class, derive a new class
/// from it and fill in the "execute" function.
///
////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <string>
#include "Thread.hpp"
#include "SystemSocket.hpp"
#include "TimeStamp.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class TCPConnection : public pcf::Thread
{
    //  definitions.
  public:
    typedef std::map<unsigned long long, TCPConnection *> Map;

    //  construction/destruction/copy/assignment
  public:
    TCPConnection();
    TCPConnection( TCPConnection::Map *pmapSiblings );
    TCPConnection( const char *pcName );
    TCPConnection( const std::string &szName );
    TCPConnection( const char *pcName, TCPConnection::Map *pmapSiblings );
    TCPConnection( const std::string &szName, TCPConnection::Map *pmapSiblings );
    TCPConnection( const TCPConnection &copy );
    const TCPConnection &operator=( const TCPConnection &copy );
    virtual ~TCPConnection();

    // get/set methods
  public:
    ///  this returns the connection name.
    inline std::string getName() const
    {
      return m_szName;
    }
    ///  returns the privilege associated with this connection.
    inline int getPrivilege() const
    {
      return m_nPrivilege;
    }
    /// returns a pointer to the parent's connection map.
    inline TCPConnection::Map *getSiblingMap()
    {
      return m_pmapSiblings;
    }
    /// This is the time that the data arrived.
    inline pcf::TimeStamp getDataArrivalTime() const
    {
      return m_tsDataArrival;
    }
    /// returns the size of the data we will receive.
    inline int getDataSize() const
    {
      return m_nSize;
    }
    /// returns a reference to the internal socket used for this connection.
    inline SystemSocket &getSocket()
    {
      return m_socConnection;
    }
    /// sets the size of the data we will be receiving. if the size = 0,
    /// the data does not have a fixed size and is considered to be a string.
    /// If the size is > 0, we have 'chunk' data (binary). If it is < 0, we
    /// will not do any processing at all.
    inline void setDataSize( const int &nSize )
    {
      m_nSize = nSize;
    }
    /// sets the name of this connection.
    inline void setName( const char *pcName )
    {
      m_szName = std::string( pcName );
    }
    inline void setName( const std::string &szName )
    {
      m_szName = szName;
    }
    /// sets the privilege associated with this connection.
    inline void setPrivilege( const int &nPrivilege )
    {
      m_nPrivilege = nPrivilege;
    }
    /// sets the thread id for the listener which created this connection.
    /// This is where the signal will be sent to cause this object to be deleted.
    inline void setListenerThreadId( const int &nThreadId )
    {
      m_nThreadId = nThreadId;
    }
    /// sets the map associated with the other connections.
    inline void setSiblingMap( TCPConnection::Map *pmapSiblings )
    {
      m_pmapSiblings = pmapSiblings;
    }

  public:
    void close();
    /// Ensures that we will make a copy of this class, not a base class.
    virtual TCPConnection *copyInstance();
    /// The method called when the connection is set up.
    virtual void setupConnection();
    ///  This function should be used if the user wants control over the
    /// socket at a low level. Call 'setDataSize' with -1 to make this the call.
    virtual std::string executeSocket( SystemSocket &socData );
    /// The method called when string data arrives. Call 'setDataSize' with
    /// an arg of 0 to indicate CRLF terminated text is expected.
    virtual std::string executeString( const std::string &szData );
    /// The method called when chunk data arrives. Call 'setDataSize' with the
    /// size, in bytes, of the data we want.
    virtual char *executeChunk( const char *pcData, int &nSize );
    /// returns whether or not we are connected.
    inline bool isConnected() const
    {
      return m_oIsConnected;
    }
    /// Send an unsolicited chunk of data to all connected clients.
    int sendChunk( const char *pcData, int &nSize  );

    // helper methods.
  protected:
    ///  this function will be called every "Interval" milliseconds.
    virtual void execute();
    ///  this function will be called before the thread enters the main
    ///  update loop. (Before calling "execute" the first time).
    virtual void beforeExecute();
    ///  this function will be called after the thread exits the main
    ///  update loop. (After calling "execute" for the last time).
    virtual void afterExecute();

  protected:
    ///  what is the name of this connection?
    std::string m_szName;
    /// The socket used to make the connection.
    SystemSocket m_socConnection;
    /// What is the size of the data we will receive?
    /// < 0 = no processing.
    /// = 0 = string data of no fixed size.
    /// > 0 = chunk data (binary) of a fixed size.
    /// default is '0' - string data.
    int m_nSize;
    /// Are we connecetd?
    bool m_oIsConnected;
    /// What is the privilege level for this connection?
    int m_nPrivilege;
    /// This is the thread id of the listener which created this connection.
    int m_nThreadId;
    /// A map of all other connections made by the listener (server).
    TCPConnection::Map *m_pmapSiblings;
    /// At what time did the current data packet arrive?
    pcf::TimeStamp m_tsDataArrival;

}; // class TCPConnection
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_TCP_CONNECTION_HPP
