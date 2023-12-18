/// $Id: IndiConnection.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <iostream>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <stdexcept>
#include <sys/types.h>  // provides 'umask'
#include <sys/stat.h>  // provides 'umask'
#include <sys/time.h>  // provides 'setrlimit'
#include <sys/resource.h>  // provides 'setrlimit'
#include "IndiConnection.hpp"
#include "TimeStamp.hpp"

using std::exception;
using std::runtime_error;
using std::string;
using std::vector;
using std::stringstream;
using std::endl;
using pcf::TimeStamp;
using pcf::IndiConnection;
using pcf::IndiXmlParser;
using pcf::IndiMessage;
using pcf::IndiProperty;


////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::IndiConnection
/// Standard constructor.

IndiConnection::IndiConnection()
{
  construct( "generic_indi_process", "1", "1" );
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::IndiConnection
/// Constructor which sets the name, version, and INDI protocol version.
/// \param szName Name of this object.
/// \param szVersion Version of this object.
/// \param szProtocolVersion INDI protocol version.

IndiConnection::IndiConnection( const string &szName,
                                const string &szVersion,
                                const string &szProtocolVersion )
{
  construct( szName, szVersion, szProtocolVersion );
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::IndiConnection
/// Copy constructor.
/// \param idRhs Another version of the driver.

IndiConnection::IndiConnection( const IndiConnection &idRhs ) : Thread()
{
  static_cast<void>(idRhs);
  // Empty because this is private.
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::operator =
/// Assignment operator.
/// \param idRhs The right-hand side of the operation.
/// \return This object.

const IndiConnection &IndiConnection::operator= ( const IndiConnection &idRhs )
{
  static_cast<void>(idRhs);
  // Empty because this is private.
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::~IndiConnection
/// Standard destructor.

IndiConnection::~IndiConnection()
{
    try
    {
        deactivate();
    }
    catch(...)
    {
        //do nothing
    }
    
    if(m_fstreamOutput && m_fstreamOutput != m_fstreamSTDOUT)
    {
        fclose(m_fstreamOutput);
    }

}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::construct
/// Called from the constructor to initialize member variables.
/// \param szName Name for this INDI process to use.
/// \param szVersion Version of the process.
/// \param szProtocolVersion Version of the INDI protocol.

void IndiConnection::construct( const string &szName,
                                const string &szVersion,
                                const string &szProtocolVersion )
{
  // Make sure we are in a known state.
  m_oQuitProcess = false;

  // The process thread has not been set up yet.
  m_idProcessThread = 0;

  // These are the two descriptors we will use to talk to the outside world.
  m_fdInput = STDIN_FILENO;

  //We start with STDOUT.  
  m_fstreamSTDOUT = fdopen(STDOUT_FILENO, "w+");  
  setOutputFd(STDOUT_FILENO);

  // setup the signal handler.
  //::signal( SIGHUP, IndiConnection::handleSignal );
  //::signal( SIGINT, IndiConnection::handleSignal );
  //::signal( SIGTERM, IndiConnection::handleSignal );

  // Set our information that sets us up as a unique INDI component.
  setName( szName );
  setVersion( szVersion );
  setProtocolVersion( szProtocolVersion );

  m_oIsVerboseModeEnabled = false;

  // What is the interval at which our 'execute' function is called?
  // This is the same if we are in simulation mode or not.
  // The default is one second.
  setInterval(1000);
  
  // What is our CPU affinity? This is the CPU we will run on.
  // A -1 indicates we don't care where it runs.
  m_iCpuAffinity = -1;

  // allocate a big buffer to hold the input data.
  m_vecInputBuf = vector<unsigned char>( InputBufSize );

}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::beforeExecute
/// Override this function to do something before this device has been told to
/// start running the thread, like allocate memory.

void IndiConnection::beforeExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::afterExecute
/// Override this function to do something after this device has been told to
/// stop running the thread, like clean up allocated memory.

void IndiConnection::afterExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::execute
/// Function which executes in a loop in a separate thread.
/// Override in derived class to perform some action.

void IndiConnection::execute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::activate
/// Try to start the driver 'execute' thread. If it is already running, this
/// will throw. When the thread starts, it calls 'beforeExecute' before
/// calling 'execute' in a loop.

void IndiConnection::activate()
{
  // is the thread already running?
  if ( isRunning() == true )
    throw runtime_error( string( "Tried to activate when already active." ) );

  // Start the 'execute' thread running to perform the component.
  start( m_iCpuAffinity );
}

/////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::deactivate
/// Try to stop the driver 'execute' thread and the 'process' thread. If
/// neither is running, this will have no effect. This will stop the 'execute'
/// being called in a loop and call 'afterExecute' before stopping the thread.

void IndiConnection::deactivate()
{
  if ( isRunning() == true )
  {
    stop();
    join();
    if ( m_idProcessThread != 0 )
    {
      ::pthread_join( m_idProcessThread, NULL );
      m_idProcessThread = 0;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::isActive
/// Is the driver currently active ('execute' thread running)?
/// \return true or false.

bool IndiConnection::isActive() const
{
  return isRunning();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::processIndiRequests
/// Called to ensure that incoming INDI messages are received and handled.
/// It will not exit until we receive a signal. May create a new thread.
/// \param oUseThread Run this in a separate thread or not.

void IndiConnection::processIndiRequests( const bool &oUseThread )
{
  if ( oUseThread == false )
  {
    process();
  }
  else
  {
    m_idProcessThread = 0;
    ::pthread_create( &m_idProcessThread, NULL,
                      IndiConnection::pthreadProcess, this );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::pthreadProcess
/// 'pthread_create' needs a static function to get the thread going.
/// Passing a pointer back to this class allows us to call the 'runLoop'
/// function from within the new thread.
/// \param pUnknown A pointer to this class that the thread function can use.
/// \return NULL - no return is needed.

void *IndiConnection::pthreadProcess( void *pUnknown )
{
  //  do a static cast to get the pointer back to a "Thread" object.
  IndiConnection *pThis = static_cast<IndiConnection *>( pUnknown );

  //  we are now within the new thread and up and running.
  try
  {
    pThis->process();
  }
  catch ( const std::exception &excep )
  {
    std::cerr << "Process thread exited: " << excep.what() << std::endl;
  }
  catch ( ... )
  {
    std::cerr << "An exception was thrown, process thread exited." << std::endl;
  }

  ///  no return is necessary, since it is not examined.
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::process
/// Listens on the file descriptor for incoming INDI messages. Exits when the
/// 'Quit Process' flag becomes true.

void IndiConnection::process()
{
  // Loop here until we are told to quit or we hit an error.
  while ( m_oQuitProcess == false )
  {
    try
    {
      // Call the 'update' function to do something each time we pass through
      // this loop reading input.
      update();

      // The length of the command received.
      int nInputBufLen = 0;
      ::memset( &m_vecInputBuf[0], 0, InputBufSize );

      // Create and clear out the FD set.
      fd_set fdsRead;
      FD_ZERO( &fdsRead );
      // Watch input to see when we get some input.
      FD_SET( m_fdInput, &fdsRead );

      // The argument to 'select' must be +1 greater than the largest fd.
      int nHighestNumberedFd = m_fdInput;

      // Set the timeout on the select call.
      timeval tv;
      tv.tv_sec = 1; //0;
      tv.tv_usec = 0; //10000;

      // We need a timeout on the select to ensure that we loop around and
      // call the 'update' function regularly.
      int nRetval = ::select( nHighestNumberedFd + 1, &fdsRead, NULL, NULL, &tv );
      //int nRetval = ::select( nHighestNumberedFd+1, &fdsRead, NULL, NULL, NULL );

      if ( nRetval == -1 )
      {
        if ( m_oQuitProcess == false )
        {
          Thread::sleep( 1 );
        }
      }
      else if ( nRetval == 0 )
      {
        // Timed out - just loop back around.
      }
      // We must check the input file descriptor.
      else if ( FD_ISSET( m_fdInput, &fdsRead ) != 0 )
      {
        // Receive a command
        nInputBufLen = ::read( m_fdInput, &m_vecInputBuf[0], InputBufSize );
        if ( nInputBufLen < 0 )
        {
          m_oQuitProcess = true;
        }
        else if ( nInputBufLen == 0 )
        {
          // If we read an EOF, this is a signal that we should die.
          m_oQuitProcess = true;
        }
        else
        {
          // A message for the error.
          std::string szErrorMsg;
          // Now, is this a command which fits our requirements?
          m_ixpIndi.parseXml( ( char * )( &m_vecInputBuf[0] ), nInputBufLen, szErrorMsg );

          while( m_ixpIndi.getState() == IndiXmlParser::CompleteState )
          {
            // Create the message from the XML.
            IndiMessage imRecv = m_ixpIndi.createIndiMessage();
            const IndiProperty &ipRecv = imRecv.getProperty();

            // Dispatch!
            dispatch( imRecv.getType(), ipRecv );

            // Get ready for some new XML.
            //m_ixpIndi.clear();

            //Test whether there is more unparsed data.
            m_ixpIndi.parseXml( "", szErrorMsg);
          }
        }
      }
    }
    catch ( const runtime_error &excep )
    {
    }
    catch ( const exception &excep )
    {
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::sendXml
/// Sends an XML string out. 
/// \param szXml The XML to send.
void IndiConnection::sendXml( const string &szXml ) const
{
  MutexLock::AutoLock autoOut( &m_mutOutput );
  
  if(!m_fstreamOutput)
  {
    return;
  }

  ::fprintf( m_fstreamOutput, "%s", szXml.c_str() );
  fflush(m_fstreamOutput);

}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::isVerboseModeEnabled
/// Are we logging additional messages? True if yes, false otherwise.
/// \return true or false.

bool IndiConnection::isVerboseModeEnabled() const
{
  return m_oIsVerboseModeEnabled;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::enableVerboseMode
/// Turns the additional logging on or off.
/// \param oEnable true or false to turn it on or off.

void IndiConnection::enableVerboseMode( const bool &oEnable )
{
  m_oIsVerboseModeEnabled = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief setInputFd
/// Which FD will be used for input?
/// \param iFd The file descriptor to use.

void IndiConnection::setInputFd( const int &iFd )
{
  m_fdInput = iFd;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief setOutputFd
/// Which FD will be used for output?
/// \param iFd The file descriptor to use.

void IndiConnection::setOutputFd( const int &iFd )
{
    m_fdOutput = iFd;

    //Close if it's open as long as it isn't STDOUT
    if(m_fstreamOutput && m_fstreamOutput != m_fstreamSTDOUT)
    {
        fclose(m_fstreamOutput);
    }

    if(iFd == STDOUT_FILENO)
    {
        m_fstreamOutput = m_fstreamSTDOUT;
    }
    else
    {
        m_fstreamOutput = fdopen(m_fdOutput, "w+");
    }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::setName
/// Sets the name of this component.
/// \param szName The name to use.

void IndiConnection::setName( const string &szName )
{
  m_szName = szName;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::getName
/// Returns the name of this component.
/// \return The name

string IndiConnection::getName() const
{
  return m_szName;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::setVersion
/// Sets the version of this component.
/// \param szVersion The version as a string.

void IndiConnection::setVersion( const string &szVersion )
{
  m_szVersion = szVersion;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::getVersion
/// Returns the version of this component.
/// \return The version as a string.

string IndiConnection::getVersion() const
{
  return m_szVersion;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::setProtocolVersion
/// Sets the INDI protocol version.
/// \param ProtocolVersion The protocol version as a string.

void IndiConnection::setProtocolVersion( const string &ProtocolVersion )
{
  m_ixpIndi.setProtocolVersion( ProtocolVersion );
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::getProtocolVersion
/// Returns the INDI protocol version.
/// \param szVersion The protocol version as a string.

string IndiConnection::getProtocolVersion() const
{
  return m_ixpIndi.getProtocolVersion();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiConnection::quitProcess
/// Causes the process to quit, the same as if a ctrl-c was sent.

void IndiConnection::quitProcess()
{
  m_oQuitProcess = true;
}

////////////////////////////////////////////////////////////////////////////////
