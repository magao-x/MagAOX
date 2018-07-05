/// $Id: TCPConnection.cpp,v 1.7 2007/09/11 03:46:50 pgrenz Exp $
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <string.h> // For 'strcpy'.
#include <iostream>
#include "Logger.hpp"
#include "TCPConnection.hpp"

using pcf::TCPConnection;
using pcf::SystemSocket;
using pcf::TimeStamp;
using pcf::Thread;
using pcf::Logger;
using std::string;
using std::endl;

////////////////////////////////////////////////////////////////////////////////
///  default constructor.

TCPConnection::TCPConnection()
  : Thread()
{
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_nSize = 0;
  m_oIsConnected = false;
  m_pmapSiblings = NULL;
}

////////////////////////////////////////////////////////////////////////////////
///  default constructor.

TCPConnection::TCPConnection( TCPConnection::Map *pmapSiblings )
  : Thread()
{
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_nSize = 0;
  m_oIsConnected = false;
  m_pmapSiblings = pmapSiblings;
}

////////////////////////////////////////////////////////////////////////////////
///  constructor with arguments.

TCPConnection::TCPConnection( const char *pcName )
  : Thread()
{
  m_szName = string( pcName );
  m_nSize = 0;
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_oIsConnected = false;
  m_pmapSiblings = NULL;
}

////////////////////////////////////////////////////////////////////////////////
///  constructor with arguments.

TCPConnection::TCPConnection( const char *pcName,
                              TCPConnection::Map *pmapSiblings )
  : Thread()
{
  m_szName = string( pcName );
  m_nSize = 0;
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_oIsConnected = false;
  m_pmapSiblings = pmapSiblings;
}

////////////////////////////////////////////////////////////////////////////////
///  constructor with arguments.

TCPConnection::TCPConnection( const string &szName ) : Thread()
{
  m_szName = szName;
  m_nSize = 0;
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_oIsConnected = false;
  m_pmapSiblings = NULL;
}

////////////////////////////////////////////////////////////////////////////////
///  constructor with arguments.

TCPConnection::TCPConnection( const string &szName,
                              TCPConnection::Map *pmapSiblings ) : Thread()
{
  m_szName = szName;
  m_nSize = 0;
  m_nPrivilege = 0;
  m_nThreadId = 0;
  m_oIsConnected = false;
  m_pmapSiblings = pmapSiblings;
}

////////////////////////////////////////////////////////////////////////////////
///  default destructor.

TCPConnection::~TCPConnection()
{
  //  if the connection is active, make sure it closes properly.
  close();
}

////////////////////////////////////////////////////////////////////////////////
///  copy constructor.

TCPConnection::TCPConnection( const TCPConnection &copy )
  : Thread( copy )
{
  m_szName = copy.m_szName;
  m_nSize = copy.m_nSize;
  m_nPrivilege = copy.m_nPrivilege;
  m_nThreadId = copy.m_nThreadId;
  m_oIsConnected = false;
  m_pmapSiblings = copy.m_pmapSiblings;
}

////////////////////////////////////////////////////////////////////////////////
///  assignment operator.

const TCPConnection &TCPConnection::operator=( const TCPConnection &copy )
{
  if ( &copy != this )
  {
    Thread::operator=( copy );
    m_szName = copy.m_szName;
    m_nSize = copy.m_nSize;
    m_nPrivilege = copy.m_nPrivilege;
    m_nThreadId = copy.m_nThreadId;
    m_oIsConnected = false;
    m_pmapSiblings = copy.m_pmapSiblings;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
///  this object can be used to manage several connections and several
///  commands, so the server needs a way of creating many copies of it.

TCPConnection *TCPConnection::copyInstance()
{
  return new TCPConnection( getName() );
}

////////////////////////////////////////////////////////////////////////////////
//  close the connection if we are connected.

void TCPConnection::close()
{
  //  stop the thread running.
  stop();
  join();

  if ( m_socConnection.isValid() == true )
  {
    m_socConnection.close();
    Thread::msleep( 1 );
    m_oIsConnected = false;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// The method called when the connection is set up.

void TCPConnection::setupConnection()
{
}

////////////////////////////////////////////////////////////////////////////////
///  this function will be called before the thread enters the main
///  update loop. (Before calling "execute" the first time).

void TCPConnection::beforeExecute()
{
  //  we want to loop as fast as possible....
  setInterval( 0 );

  //  when this connection is broken, delete all vestiges of this object.
  //setDeleteThis( true );

  //  set the socket to non-blocking.
  m_socConnection.setNonBlocking( true );

  // The default is to trigger the running of 'execute' based on whether
  // there is data available to be read on the socket.
  setTrigger( &m_socConnection );

  //  create the socket number string.
  //stringstream ssSocketId;
  //ssSocketId << m_socConnection.getId();
  //  send a "hello"
  //m_socConnection.send( "Hello - You are connection id #" + ssSocketId.str() + ".\n" );

  //  we are now waiting for data.....
  m_oIsConnected = true;

  Logger logMsg;
  logMsg << Logger::enumDebug << "OK      [TCPConnection::beforeExecute] "
         << "(fd=" << m_socConnection.getFd() << ") "
         << "starting connection."
         << endl;

  setupConnection();
}

////////////////////////////////////////////////////////////////////////////////
///  this function will be called after the thread exits the main
///  update loop. (After calling "execute" for the last time).

void TCPConnection::afterExecute()
{
  //  detach this Connection from the sibling map.
  //if ( m_pmapSiblings != NULL )
  //{
  //  TCPConnection::Map::iterator itr =
  //      m_pmapSiblings->find( m_socConnection.getId() );
  //  if ( itr != m_pmapSiblings->end() )
  //  {
  //    itr->second = NULL;
  //    m_pmapSiblings->erase( itr );
  //  }
  //}

  //  send a goodbye!
  //m_socConnection.send( "Goodbye - connection id #" + ssSocketId.str() +
  //    " is now closed.\n" );

  Logger logMsg;
  logMsg << Logger::enumDebug << "OK      [TCPConnection::afterExecute] "
         << "(fd=" << m_socConnection.getFd() << ") "
         << "stopping connection."
         << endl;

  //  close from this end.
  // changed to work on windows....
  //close();
  if ( m_socConnection.isValid() == true )
  {
    m_socConnection.close();
    Thread::msleep( 1 );
    m_oIsConnected = false;
  }
}

////////////////////////////////////////////////////////////////////////////////
///  when using the thread capabilities of this class,
///  this function gets the thread going.

void TCPConnection::execute()
{
  try
  {
    int nSize = getDataSize();

    // Is the socket ok?
    if ( m_socConnection.isValid() == false || m_oIsConnected == false )
    {
      Logger logMsg;
      logMsg << Logger::enumError << "ERROR   [TCPConnection::execute] "
             << "(fd=" << m_socConnection.getFd() << ") "
             << "not valid or not connected."
             << endl;
      stop();
    }
    // If the size is less than 0, the user wants to do all the processing.
    else if ( nSize < 0 )
    {
      executeSocket( m_socConnection );
    }
    // We have string data of indeterminate size.
    else if ( nSize == 0 )
    {
      string szData = m_socConnection.recv();

      // We capture the arrival time for use later.
      m_tsDataArrival = TimeStamp::now();

      m_socConnection.send( executeString( szData ) );
    }
    // We are processing a fixed-size chunk.
    else
    {
      char *pcData = new char[nSize];
      m_socConnection.recvChunk( pcData, nSize );

      // We capture the arrival time for use later.
      m_tsDataArrival = TimeStamp::now();

      // pcSend must point to valid data.
      char *pcSend = executeChunk( pcData, nSize );
      m_socConnection.sendChunk( pcSend, nSize );

      // We own this data now, so release it.
      if ( pcSend != NULL )
        delete pcSend;
      if ( pcData != NULL )
        delete[]( pcData );
    }
  }
  catch ( const SystemSocket::Error &err )
  {
    //  we received no data, but the connection is still good
    //  just go back around again.
    if ( m_socConnection.getLastError() == EAGAIN )
    {
      Thread::msleep( 1 );
    }
    //  did the socket close?
    else if ( m_socConnection.getLastError() == ECONNRESET )
    {
      Logger logMsg;
      logMsg << Logger::enumInfo << "OK      [TCPConnection::execute] "
             << "(fd=" << m_socConnection.getFd() << ") " << err.what()
             << endl;
      stop();
    }
    //  was there some kind of other error?
    else
    {
      Logger logMsg;
      logMsg << Logger::enumError << "ERROR   [TCPConnection::execute] "
             << "(fd=" << m_socConnection.getFd() << ") " << err.what()
             << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///  This function should be filled in a derived class to actually do something.

string TCPConnection::executeString( const string &szData )
{
  //  print a message containing the received data as a string.
  Logger logMsg;
  logMsg << Logger::enumInfo << "OK      [TCPConnection::executeString] "
         << "received: '" << szData << "'."
         << endl;

  return "OK";
}

////////////////////////////////////////////////////////////////////////////////
///  This function should be filled in a derived class to actually do something.

char *TCPConnection::executeChunk( const char *pcData, int &nSize  )
{
  //  print a message containing the received data as a string.
  Logger logMsg;
  logMsg << Logger::enumInfo << "OK      [TCPConnection::executeChunk] "
         << "received chunk: '" << pcData << "'."
         << endl;

  char *pcReturn = new char[4];
  strcpy( pcReturn, "OK" );
  return pcReturn;
}

////////////////////////////////////////////////////////////////////////////////
///  This function should be filled in a derived class to actually do something.

string TCPConnection::executeSocket( SystemSocket &socData )
{
  //  print a message containing the socket's file descriptor.
  Logger logMsg;
  logMsg << Logger::enumInfo << "OK      [TCPConnection::executeSocket] "
         << "(fd=" << socData.getFd() << ") "
         << endl;

  return string( "OK" );
}

////////////////////////////////////////////////////////////////////////////////
///  This function will send an unsolicited chunk of data to all clients.

int TCPConnection::sendChunk( const char *pcData, int &nSize  )
{
  try
  {
    //std::cerr << "[TCPConnection] sending chunk of " << nSize
    //          << " bytes." << std::endl;
    // This interface does not accept "const char *".
    m_socConnection.sendChunk( const_cast<char *>( pcData ), nSize );

    return 0;
  }
  catch ( const SystemSocket::Error &err )
  {
    return m_socConnection.getLastError();
  }
}

////////////////////////////////////////////////////////////////////////////////
