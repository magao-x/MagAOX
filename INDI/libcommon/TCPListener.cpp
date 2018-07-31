/// $Id: TCPListener.cpp,v 1.10 2007/09/11 03:46:50 pgrenz Exp $
///
/// @author Paul grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <errno.h>
#include "SystemSocket.hpp"
#include "TCPListener.hpp"
#include "Logger.hpp"
#include <iostream>

using pcf::Logger;
using pcf::Thread;
using pcf::TCPListener;
using pcf::TCPConnection;
using std::string;
using std::endl;

////////////////////////////////////////////////////////////////////////////////

TCPListener::TCPListener() : Thread()
{
  m_oIsListening = false;
  m_pConnectionPrototype = NULL;
  m_pConnection = NULL;
  m_nnCount = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with arguments.

TCPListener::TCPListener( const char *pcName ) : Thread()
{
  m_szName = string( pcName );
  m_oIsListening = false;
  m_pConnectionPrototype = NULL;
  m_pConnection = NULL;
  m_nnCount = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with arguments.

TCPListener::TCPListener( const string &szName ) : Thread()
{
  m_szName = szName;
  m_oIsListening = false;
  m_pConnectionPrototype = NULL;
  m_pConnection = NULL;
  m_nnCount = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TCPListener::~TCPListener()
{
  stopListening();
  closeAll();

  if ( m_pConnection != NULL )
    delete m_pConnection;
}

////////////////////////////////////////////////////////////////////////////////
/// This returns the name of this server.

string TCPListener::getName() const
{
  return m_szName;
}

////////////////////////////////////////////////////////////////////////////////
/// Assign a server name.

void TCPListener::setName( const char *pcName )
{
  m_szName = string( pcName );
}

////////////////////////////////////////////////////////////////////////////////
/// Assign a server name.

void TCPListener::setName( const string &szName )
{
  m_szName = szName;
}

////////////////////////////////////////////////////////////////////////////////
/// Close one connection to this server.

void TCPListener::close( const int &nConnectionId )
{
  TCPConnection::Map::iterator itr = m_mapConnections.find( nConnectionId );

  if ( itr != m_mapConnections.end() )
  {
    itr->second->close();
    delete itr->second;
    m_mapConnections.erase( itr );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Close all connections to this server.

void TCPListener::closeAll()
{
  TCPConnection::Map::iterator itr = m_mapConnections.begin();

  // Each time we erase a Connection, this will remove the entry from the map.
  // therefore, we cannot use a 'for' loop or other such construct.
  while ( itr != m_mapConnections.end() )
  {
    itr->second->close();
    delete itr->second;
    m_mapConnections.erase( itr );
    // The iterator needs to be reset back to a valid entry.
    // stl documentation indicates that the iterator should be placed on
    // the next valid entry, but in practice this seems to not be
    // the case, as a seg fault is the result of not doing this step.
    itr = m_mapConnections.begin();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Stop listening for new connections (keeps old connections).

void TCPListener::stopListening()
{
  stop();
  join();
}

////////////////////////////////////////////////////////////////////////////////
/// There is one object for each connection.

TCPConnection::Map *TCPListener::getConnections()
{
  return &m_mapConnections;
}

////////////////////////////////////////////////////////////////////////////////
/// How many connections do we currently have?

int TCPListener::size() const
{
  return m_mapConnections.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the port we are using currently.

int TCPListener::getPort() const
{
  return m_socServer.getPort();
}

////////////////////////////////////////////////////////////////////////////////
/// Are we currently listening for new connections?

bool TCPListener::isListening() const
{
  return m_oIsListening;
}

////////////////////////////////////////////////////////////////////////////////
///  return the message concerning the error.
/*
string TCPListener::getErrorMsg( const int &nErr ) const
{
  string szErrorMsg;
  switch ( nErr )
  {
    case enumNoError:
      szErrorMsg = "No Error.";
      break;
    case  enumBindFailed:
      szErrorMsg = "The 'bind' call failed.";
      break;
    case enumStartFailed:
      szErrorMsg = "Could not start thread to listen for connections.";
      break;
    default:
      szErrorMsg = "Unknown error.";
      break;
  }
  return szErrorMsg;
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Tries to listen on a port using the given host address. If the listen
/// is successful, every time a connection is made, the connection prototype
/// will be copied to handle it. The new connection object will run in a
/// new thread and be added to the connection sibling map.

void TCPListener::listen( const int &nPort,
                          TCPConnection *pConnectionPrototype,
                          const string &szHost )
{
  try
  {
    if ( m_oIsListening != true )
    {
      //  we need to start fresh.
      closeAll();
      //  assign the sibling map inside the prototype.
      //  this way, all subsequent connections will have access to it.
      pConnectionPrototype->setSiblingMap( &m_mapConnections );
      //  save the prototype to use.
      m_pConnectionPrototype = pConnectionPrototype;

      //  first setup the socket....
      m_socServer = SystemSocket( SystemSocket::Stream, nPort, szHost );

      //  bind it to the port supplied....
      m_socServer.bind ();
    }
  }
  catch ( const SystemSocket::Error &err )
  {
    Logger logMsg;
    logMsg << Logger::enumError << "ERROR   [TCPListener::listen] "
           << "(fd=" << m_socServer.getFd() << ") " << err.what()
           << endl;
    throw Error( string( "::listen" ) + err.what() );
  }
}

////////////////////////////////////////////////////////////////////////////////
///  this function will be called before the thread enters the main
///  update loop. (Before calling "execute" the first time).

void TCPListener::beforeExecute()
{
  try
  {
    // We want to loop as fast as possible....
    setInterval( 0 );

    if ( m_pConnection != NULL )
    {
      delete m_pConnection;
      m_pConnection = NULL;
    }

    // Listen for incoming connections....
    m_socServer.listen();

    m_oIsListening = true;
    Logger logMsg;
    logMsg << Logger::enumDebug << "OK      [TCPListener::beforeExecute] "
           << "(fd=" << m_socServer.getFd() << ") "
           << "started listening on '"
           << m_socServer.getHost() << ":"
           << m_socServer.getPort() << "'."
           << endl;
  }
  catch ( const SystemSocket::Error &err )
  {
    Logger logMsg;
    logMsg << Logger::enumError << "ERROR   [TCPListener::beforeExecute] "
           << "(fd=" << m_socServer.getFd() << ") " << err.what()
           << endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
///  this function will be called after the thread exits the main
///  update loop. (After calling "execute" for the last time).

void TCPListener::afterExecute()
{
  Logger logMsg;
  logMsg << Logger::enumDebug << "OK      [TCPListener::afterExecute] "
         << "(fd=" << m_socServer.getFd() << ") "
         << "stopped listening on '"
         << m_socServer.getHost() << ":"
         << m_socServer.getPort() << "'."
         << endl;

  if ( m_oIsListening == true )
  {
    m_oIsListening = false;
    m_socServer.close();
    if ( m_pConnection != NULL )
    {
      delete m_pConnection;
      m_pConnection = NULL;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  this function will be called every "Interval" milliseconds.

void TCPListener::execute()
{
  try
  {
    if ( m_oIsListening == true )
    {
      //  make a copy of the current Connection prototype, if we have not
      //  already done so the last time through this loop.
      //  if it is already allocated, that would indicate that no
      //  connection was made the last time around.
      if ( m_pConnection == NULL )
      {
        //  get a new copy of the Connection. We want a copy of the _derived_
        // class, not the base class, so we need to call 'copyInstance'.
        m_pConnection = m_pConnectionPrototype->copyInstance();
      }

      //  connect the Connection's socket to the current connection.
      m_socServer.accept( m_pConnection->getSocket() );

      Logger logMsg;
      logMsg << Logger::enumDebug << "OK      [TCPListener::execute] "
             << "(fd=" << m_socServer.getFd() << ") "
             << "accepting a new connection."
             << endl;

      //  set the name of the Connection - this is the same as this server.
      m_pConnection->setName( getName() );
      //  set the sibling map within it.
      m_pConnection->setSiblingMap( &m_mapConnections );
      //  set the data size.
      m_pConnection->setDataSize( m_pConnectionPrototype->getDataSize() );
      //  set the privilege - this is only 0 for now.
      m_pConnection->setPrivilege( 0 );
      //  add to the map of connections.
      //m_mapConnections[ m_pConnection->getSocket().getId() ] = m_pConnection;
      m_mapConnections[ m_nnCount++ ] = m_pConnection;

      //  this thread is a child of the TCPListener as well.
      m_pConnection->start();
      //  detach the pointer from the object - the map will manage it now.
      m_pConnection = NULL;
    }
  }
  catch ( const SystemSocket::Error &err )
  {
    Thread::msleep( 1 );

    //  remove closed connections.
    TCPConnection::Map::iterator itr = m_mapConnections.begin();

    //  each time we erase a Connection, this will remove the entry from
    //  the map. therefore, we cannot use a 'for' loop or other such
    //  construct.
    while ( itr != m_mapConnections.end() )
    {
      ///  are we an active connection?
      if ( itr->second->isConnected() == true )
      {
        ++itr;
      }
      else
      {
        itr->second->join();
        delete itr->second;
        m_mapConnections.erase( itr );
        //  the iterator needs to be reset back to a valid entry.
        //  stl documentation indicates that the iterator should be placed on
        //  the next valid entry, but in practice this seems to not be
        //  the case, as a seg fault is the result of not doing this step.
        itr = m_mapConnections.begin();
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
