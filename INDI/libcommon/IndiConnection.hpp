/// IndiConnection.hpp
///
/// @author Paul Grenz
///
/// This is a virtual class which must be derived from to be useful.
/// See 'IndiClient' and 'IndiDriver' for a class that can be insantiated.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_INDI_CONNECTION_HPP
#define PCF_INDI_CONNECTION_HPP

#include <string>
#include <vector>
#include "Thread.hpp"
#include "MutexLock.hpp"
#include "TimeStamp.hpp"
//#include "ConfigFile.hpp"
#include "IndiXmlParser.hpp"
#include "IndiMessage.hpp"
#include "IndiProperty.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class IndiConnection : public pcf::Thread
{
  private:
    enum Constants
    {
      // This is the size of the input buffer to hold the incoming commands.
      InputBufSize = 65536,
    };

  // construction/destruction/assign/copy
  public:
    /// Standard constructor.
    IndiConnection();
    /// Constructor which sets the name, version, and INDI protocol version.
    IndiConnection( const std::string &szName,
                    const std::string &szVersion,
                    const std::string &szProtocolVersion );
    /// Standard destructor.
    virtual ~IndiConnection();

  // Prevent these from being invoked.
  private:
    /// Copy constructor.
    IndiConnection( const IndiConnection &idRhs );
    /// Assignment operator.
    const IndiConnection &operator= ( const IndiConnection &idRhs );
    /// Called from the constructor to initialize member variables.
    void construct( const std::string &szName,
                    const std::string &szVersion,
                    const std::string &szProtocolVersion );
    /// Listens on the file descriptor in a loop for incoming INDI messages.
    /// Exits when the 'Quit Process' flag becomes true.
    void process();

  // Standard client interface methods.
  public:
    /// Try to start the driver 'execute' thread. If it is already running, this
    /// will throw. When the thread starts, it calls 'beforeExecute' before
    /// calling 'execute' in a loop.
    void activate();
    /// Override this function to do something after this device has been told to
    /// stop running the thread, like clean up allocated memory.
    virtual void afterExecute();
    /// Override this function to do something before this device has been told to
    /// start running the thread, like allocate memory.
    virtual void beforeExecute();
    /// Try to stop the driver 'execute' thread and the 'process' thread. If
    /// neither is running, this will have no effect. This will stop the 'execute'
    /// being called in a loop and call 'afterExecute' before stopping the thread.
    void deactivate();
    /// Chooses what to do with the received property.
    virtual void dispatch( const IndiMessage::Type &tType,
                           const IndiProperty &ipDispatch ) = 0;
    /// Turns the additional logging on or off.
    void enableVerboseMode( const bool &oEnable );
    /// Function which executes in a loop in a separate thread.
    /// Override in derived class to perform some action.
    virtual void execute();
    /// Return the config file path.
    std::string getConfigPath() const;
    /// Return the log file path.
    std::string getLogPath() const;
    /// Return the name of this client.
    std::string getName() const;
    /// Return the INDI protocol version.
    std::string getProtocolVersion() const;
    /// Returns the version of this component.
    std::string getVersion() const;

    /// Is the driver currently active ('execute' thread running)?
    bool isActive() const;
    /// Are we logging additional messages?
    bool isVerboseModeEnabled() const;
    /// Called to ensure that incoming INDI messages are received and handled.
    /// It will not exit until we receive a signal. May create a new thread.
    void processIndiRequests( const bool &oUseThread = false );
    /// Sends an XML string out to a file descriptor. If there is an error,
    /// it will be logged.
    virtual void sendXml( const std::string &szXml ) const;

    /// Which FD will be used for input?
    void setInputFd( const int &iFd );
    /// Set the name of this component.
    void setName( const std::string &szName );
    /// Which FD will be used for output?
    void setOutputFd( const int &iFd );
    /// Set the version of the INDI protocol.
    void setProtocolVersion( const std::string &szProtocolVersion );
    /// Sets the INDI protocol version.
    void setVersion( const std::string &szVersion );

    /// Called in the process loop to perform an action each time through.
    virtual void update() = 0;
    /// This will cause the process to quit, the same as if a ctrl-c was sent.
    void quitProcess();

    bool getQuitProcess() 
    {
       return m_oQuitProcess;
    }
    
  // Helper functions.
  protected:
    /// 'pthread_create' needs a static function to get the thread going.
    /// Passing a pointer back to this class allows us to call the 'runLoop'
    /// function from within the new thread.
    static void *pthreadProcess( void *pUnknown );

  // Variables
  private:
    /// The name of this client.
    std::string m_szName;
    /// the version of this software. may be "none".
    std::string m_szVersion;
    /// Is this client generating additional messages?
    bool m_oIsVerboseModeEnabled;
    /// Which CPU do we want to run the worker thread on?
    int m_iCpuAffinity;
    /// allocate a big buffer to hold the input data.
    std::vector<unsigned char> m_vecInputBuf;
    
    /// The flag to tell this to quit.
    //Changed from static to prevent app-wide INDI shutdown.
    bool m_oQuitProcess {false};
    
    /// This is the object that conglomerates all the INDI XML
    pcf::IndiXmlParser m_ixpIndi;
    /// A mutex to protect output.
    mutable pcf::MutexLock m_mutOutput;
    /// The file descriptor to read from.
    int m_fdInput;
    
    /// The file descriptor to write to.
    int m_fdOutput;

    /// Stream for safer output
    FILE * m_fstreamOutput {NULL};

    FILE * m_fstreamSTDOUT {NULL};

    /// If the processing of INDI messages is put in a separate thread,
    /// this is the thread id of it.
    pthread_t m_idProcessThread;

}; // class IndiConnection
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_INDI_CONNECTION_HPP
