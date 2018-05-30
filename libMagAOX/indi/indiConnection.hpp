/** \file indiConnection.hpp 
  * \brief A Class to mannage the basic INDI protocol connection.
  * \author Paul Grenz (LBTI)
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - Original code by Paul Grenz for the LBTI project.
  * - 2018-05-28 Ported to libMagAOX and refactoring begun by Jared Males.
  *   Changes made:
  *     - Converted to header only (inlined all function def'n after dec'l).
  *     - Deleted the private constructor and assignmetn op.
  *     - Reordered member dec'l, moved def'n to same order.
  *     - Removed libcommon config system.
  *     - Removed signal handling from this class.  Signal handling somewhere else must call quitProcess() now.
  *     - Changed activate(), deactivate(), to return int for error signaling.
  *     - Using initialization in declaration.  Default constructor no long sets name and version.
  */


#ifndef indi_indiConnection_hpp
#define indi_indiConnection_hpp

#include <string>
#include <vector>
#include "../../libLBTI/libcommon/Thread.hpp"
#include "../../libLBTI/libcommon/MutexLock.hpp"
#include "../../libLBTI/libcommon/TimeStamp.hpp"
#include "../../libLBTI/libcommon/IndiXmlParser.hpp"
#include "../../libLBTI/libcommon/IndiMessage.hpp"
#include "../../libLBTI/libcommon/IndiProperty.hpp"

#include "../../libLBTI/libcommon/Logger.hpp"

namespace MagAOX
{
namespace indi
{
   
/// A Class to mannage the basic INDI protocol connection.
/** This is a virtual class which must be derived from to be useful.
  * See 'IndiClient' and 'IndiDriver' for classes that can be insantiated.
  */
class indiConnection : public pcf::Thread
{

private:
   enum Constants
   {
      /// Size of the input buffer to hold the incoming commands.
      InputBufSize = 65536,
   };

public:
    
   /// Standard constructor.
   indiConnection();
    
   /// Constructor which sets the name, version, and INDI protocol version.
   indiConnection( const std::string &szName,           ///< [in] Name of this object.
                   const std::string &szVersion,        ///< [in] Version of this object.
                   const std::string &szProtocolVersion ///< [in] INDI protocol version.
                 );
    
   /// Standard destructor.
   virtual ~indiConnection();

private:
   /// Copy constructor, deleted.
   indiConnection( const indiConnection &idRhs ) = delete;
   
   /// Assignment operator, deleted.
   const indiConnection &operator= ( const indiConnection &idRhs /**< [in] The right-hand side of the operation. */) = delete;
   
   /// Called from the constructor to initialize member variables.
   void construct();
   

public:
   
   /// Try to start the driver 'execute' thread. 
   /** If it is already running, this will throw. When the thread starts, it calls 'beforeExecute' before
     * calling 'execute' in a loop.
     * 
     * \returns 0 on success
     * \returns an indi error code on an error.
     */
   int activate();

protected:
   /// 'pthread_create' needs a static function to get the thread going.
   /// Passing a pointer back to this class allows us to call the 'runLoop'
   /// function from within the new thread.
   static void *pthreadProcess( void *pUnknown /**< [in] A pointer to this class that the thread function can use. */);
   
private:
   /// Listens on the file descriptor in a loop for incoming INDI messages.
   /** Exits when the 'Quit Process' flag becomes true.
     */
   void process();
   
public:
   
   /// Called to ensure that incoming INDI messages are received and handled.
   /** It will not exit until we receive a signal. May create a new thread.
     */
   void processIndiRequests( const bool &oUseThread = false /**< [in] Run this in a separate thread or not. */);

   /// Is the driver currently active ('execute' thread running)?
   /**
     * \returns true or false
     */ 
   bool isActive() const;

   /// This will cause the process to quit, the same as if a ctrl-c was sent.
   void quitProcess();
   
   /// Try to stop the driver 'execute' thread and the 'process' thread. 
   /** If neither is running, this will have no effect. This will stop the 'execute'
     * being called in a loop and call 'afterExecute' before stopping the thread.
     * 
     * \returns 0 on success
     * \returns an indi error code on an error.
     * 
     */
   int deactivate();
   
   
   /// Override this function to do something before this device has been told to
   /// start running the thread, like allocate memory.
   virtual void beforeExecute();

   /// Function which executes in a loop in a separate thread.
   /** Override in derived class to perform some action.
     */
   virtual void execute();

   /// Called in the process loop to perform an action each time through.
   virtual void update() = 0;
   
   /// Override this function to do something after this device has been told to
   /// stop running the thread, like clean up allocated memory.
   virtual void afterExecute();
    
   /// Chooses what to do with the received property.
   virtual void dispatch( const pcf::IndiMessage::Type &tType,
                          const pcf::IndiProperty &ipDispatch 
                        ) = 0;
                     
   /// Sends an XML string out to a file descriptor. If there is an error,
   /// it will be logged.
   virtual void sendXml( const std::string &szXml /**< [in] The XML to send. */) const;
              
   
   /// Return the name of this client.
   /**
     * \returns the current value of m_szName
     */ 
   std::string getName() const;

   /// Set the name of this component.
   void setName( const std::string &szName  /**< [in] The new name*/);

   /// Returns the version of this component.
   std::string getVersion() const;

   /// Sets the version of this component.
   void setVersion( const std::string &szVersion /**< [in] The new driver version*/);
   
   /// Return the INDI protocol version.
   std::string getProtocolVersion() const;
   
   /// Set the version of the INDI protocol.
   void setProtocolVersion( const std::string &szProtocolVersion /**< [in] The new protocol version*/);
   
   /// Which FD will be used for input?
   void setInputFd( const int &iFd /**< [in] The file descriptor to use for input. */);
   
   /// Which FD will be used for output?
   void setOutputFd( const int &iFd /**< [out] The file descriptor to use for output. */);
      
  // Variables
protected:
   /// The name of this client.
   std::string m_szName;
   
   /// the version of this software. may be "none".
   std::string m_szVersion;
   
   /// Which CPU do we want to run the worker thread on?
   int m_iCpuAffinity {-1};
   
   /// allocate a big buffer to hold the input data.
   std::vector<unsigned char> m_vecInputBuf;
   
   /// The flag to tell this to quit.
   bool sm_oQuitProcess {false};
   
   /// This is the object that conglomerates all the INDI XML
   pcf::IndiXmlParser m_ixpIndi;
   
   /// A mutex to protect output.
   mutable pcf::MutexLock m_mutOutput;
   
   /// The file descriptor to read from.
   int m_fdInput {STDIN_FILENO};
   
   /// The file descriptor to write to.
   int m_fdOutput {STDOUT_FILENO};
   
   /// If the processing of INDI messages is put in a separate thread,
   /// this is the thread id of it.
   pthread_t m_idProcessThread {0};

}; // class indiConnection

   
//bool pcf::indiConnection::sm_oQuitProcess = false;

inline
indiConnection::indiConnection()
{
  construct();
}

inline
indiConnection::indiConnection( const std::string &szName,
                                const std::string &szVersion,
                                const std::string &szProtocolVersion )
{
  construct();
  
  // Set our information that sets us up as a unique INDI component.
  setName( szName );
  setVersion( szVersion );
  setProtocolVersion( szProtocolVersion );
  
}

inline
indiConnection::~indiConnection()
{
}

inline
void indiConnection::construct()
{

  // What is the interval at which our 'execute' function is called?
  // This is the same if we are in simulation mode or not.
  // The default is 0 seconds -- called without sleeping.
  setInterval( 0 );

  // allocate a big buffer to hold the input data.
  m_vecInputBuf = std::vector<unsigned char>( InputBufSize );

}

inline
int indiConnection::activate()
{
  // is the thread already running?
  if ( isRunning() == true )
  {
     return -1;
  }

  // Start the 'execute' thread running to perform the component.
  start( m_iCpuAffinity );
  
  return 0;
}



inline //It's static anyway
void *indiConnection::pthreadProcess( void *pUnknown )
{
  //  do a static cast to get the pointer back to a "Thread" object.
  indiConnection *pThis = static_cast<indiConnection *>( pUnknown );

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

inline
void indiConnection::process()
{
  pcf::Logger logMsg;
  logMsg.enableClearAfterLog( true );

  logMsg << pcf::Logger::Info << m_szName << "::process: "
         << "Starting processing incoming INDI messages." << std::endl;

  // Loop here until we are told to quit or we hit an error.
  while ( sm_oQuitProcess == false )
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
        if ( sm_oQuitProcess == false )
        {
          logMsg << pcf::Logger::Error << "select ERROR: " << strerror( errno ) << std::endl;
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
          logMsg << pcf::Logger::Error
                 << "Failed to get data from input." << std::endl;
          sm_oQuitProcess = true;
        }
        else if ( nInputBufLen == 0 )
        {
          // If we read an EOF, this is a signal that we should die.
          sm_oQuitProcess = true;
        }
        else
        {
          // A message for the error.
          std::string szErrorMsg;
          // Now, is this a command which fits our requirements?
          m_ixpIndi.parseXml( ( char * )( &m_vecInputBuf[0] ), nInputBufLen, szErrorMsg );

          if ( m_ixpIndi.getState() == pcf::IndiXmlParser::CompleteState )
          {
            // Create the message from the XML.
            pcf::IndiMessage imRecv = m_ixpIndi.createIndiMessage();
            const pcf::IndiProperty &ipRecv = imRecv.getProperty();

            // Dispatch!
            dispatch( imRecv.getType(), ipRecv );

            // Get ready for some new XML.
            m_ixpIndi.clear();
          }
        }
      }
    }
    catch ( const std::runtime_error &excep )
    {
      pcf::Logger logMsg;
      logMsg << pcf::Logger::enumError << excep.what() << std::endl;
      logMsg << pcf::Logger::enumError << "Received XML: "
             << m_ixpIndi.createXmlString() << std::endl;
    }
    catch ( const std::exception &excep )
    {
      pcf::Logger logMsg;
      logMsg << pcf::Logger::enumError << excep.what() << std::endl;
      logMsg << pcf::Logger::enumError << "Received XML: "
             << m_ixpIndi.createXmlString() << std::endl;
    }
  }

  logMsg << pcf::Logger::Info << m_szName << "::process: "
         << "Stopped processing incoming INDI messages." << std::endl;
}

inline
void indiConnection::processIndiRequests( const bool &oUseThread )
{
  if ( oUseThread == false )
  {
    process();
  }
  else
  {
    m_idProcessThread = 0;
    ::pthread_create( &m_idProcessThread, NULL,
                      indiConnection::pthreadProcess, this );
  }
}

inline
bool indiConnection::isActive() const
{
  return isRunning();
}

inline
void indiConnection::quitProcess()
{
  sm_oQuitProcess = true;
}

inline
int indiConnection::deactivate()
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

  return 0;
}


inline
void indiConnection::beforeExecute()
{
}

inline
void indiConnection::execute()
{
}

inline
void indiConnection::afterExecute()
{
}

inline
void indiConnection::sendXml( const std::string &szXml ) const
{
  pcf::MutexLock::AutoLock autoOut( &m_mutOutput );
  int nNumPrinted = ::dprintf( m_fdOutput, "%s", szXml.c_str() );
  //if ( fflush( m_fdOutput ) != 0 || nNumPrinted < (int)( szXml.length() ) )
  if ( nNumPrinted < ( int )( szXml.length() ) )
  {
    pcf::Logger logMsg;
    logMsg << pcf::Logger::Error << "Failed to write XML to output." << std::endl;
  }
}

inline
std::string indiConnection::getName() const
{
  return m_szName;
}

inline
void indiConnection::setName( const std::string &szName )
{
  m_szName = szName;
}

inline
std::string indiConnection::getVersion() const
{
  return m_szVersion;
}

inline
void indiConnection::setVersion( const std::string &szVersion )
{
  m_szVersion = szVersion;
}

std::string indiConnection::getProtocolVersion() const
{
  return m_ixpIndi.getProtocolVersion();
}

inline
void indiConnection::setProtocolVersion( const std::string &ProtocolVersion )
{
  m_ixpIndi.setProtocolVersion( ProtocolVersion );
}

inline
void indiConnection::setInputFd( const int &iFd )
{
  m_fdInput = iFd;
}

inline
void indiConnection::setOutputFd( const int &iFd )
{
  m_fdOutput = iFd;
}

} // namespace indi
} // namespace MagAOX

#endif // indi_indiConnection_hpp
