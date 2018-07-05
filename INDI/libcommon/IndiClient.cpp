/// $Id: IndiClient.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include "IndiClient.hpp"
#include "SystemSocket.hpp"
#include "Config.hpp"

using std::runtime_error;
using std::string;
using std::endl;
using std::vector;
using pcf::Thread;
using pcf::Logger;
using pcf::Config;
using pcf::IndiClient;
using pcf::IndiMessage;
using pcf::IndiProperty;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

IndiClient::IndiClient()
  : IndiConnection()
{
  setup();
}

////////////////////////////////////////////////////////////////////////////////

IndiClient::IndiClient( const string &szName,
                        const string &szVersion,
                        const string &szProtocolVersion )
    : IndiConnection( szName, szVersion, szProtocolVersion )
{
  setup();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::IndiClient
/// Copy constructor.
/// \param icRhs Another version of the driver.

IndiClient::IndiClient(const IndiClient &icRhs )
//  : IndiConnection( icRhs )  // can't invoke - private
{
  // Empty because this is private.
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::operator =
/// Assignment operator.
/// \param icRhs The right-hand side of the operation.
/// \return This object.

const IndiClient &IndiClient::operator= ( const IndiClient &icRhs )
//  : IndiConnection::operator= ( icRhs )  // can't invoke - private
{
  // Empty because this is private.
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::~IndiClient
/// Standard destructor.

IndiClient::~IndiClient()
{
  if ( m_socClient.isValid() == true )
  {
    m_socClient.close();
    Thread::msleep( 10 );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::setup Sets up file descriptors and other things that
/// need to be initialized at construction time.

void IndiClient::setup()
{
  try
  {
    // These are the two descriptors we will use to talk to the outside world.
    // Set them by default to an invalid value.
    setInputFd( -1 ); // STDIN_FILENO;
    setOutputFd( -1 ); // STDOUT_FILENO;

    Logger logMsg;
    logMsg.enableClearAfterLog( true );

    logMsg << Logger::enumInfo << getName() << "::setup: "
           << " (v " << getVersion() << ")." << endl;

    if ( m_socClient.isValid() == true )
    {
      m_socClient.close();
      Thread::msleep( 10 );
    }

    Config cfReader;
    m_socClient = SystemSocket( SystemSocket::Stream,
                                cfReader.get<int>( "indi_server_port", 9752 ),
                                cfReader.get<string>( "indi_server_ip", "127.0.0.1" ) );

    logMsg << Logger::Info << "    Read & set the client configuration." << endl;

    m_socClient.connect();

    logMsg << Logger::enumInfo
           << "    Connected to server at " << m_socClient.getHost() << ":"
           << m_socClient.getPort() << endl;

    // Make sure we have a limit on how long we wait for a response.
    //m_socClient.setRecvTimeout( 1000 );
    // Make sure the socket will wait for data.
    //m_socClient.setNonBlocking( false );
    // Send all data, no matter how small.
    m_socClient.disableNagle( true );

    // Give the server a moment to set up.
    Thread::msleep( 10 );

    // Assign the file descriptor to the member variables.
    setInputFd( m_socClient.getFd() );
    setOutputFd( m_socClient.getFd() );
  }
  catch ( const SystemSocket::Error &err )
  {
    Logger logMsg;
    logMsg << Logger::enumError
           << "Cannot connect to " << m_socClient.getHost() << ":"
           << m_socClient.getPort() << " '" << err.what() << "'." << endl;
    m_socClient.close();
    Thread::msleep( 10 );
  }
  catch ( const runtime_error &excepRuntime )
  {
    Logger logMsg;
    logMsg << Logger::enumError << excepRuntime.what() << endl;
    m_socClient.close();
    Thread::msleep( 10 );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::update
/// Called in the process loop to perform an action each time through.

void IndiClient::update()
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::dispatch
/// Chooses what to do with the received property.
/// \param tType Type of the message we received.
/// \param ipDispatch The property contained in the message.

void IndiClient::dispatch( const IndiMessage::Type &tType,
                           const IndiProperty &ipDispatch )
{
  Logger logMsg;
  logMsg.enableClearAfterLog( true );

  // Decide what we should do based on the type of the message.
  switch ( tType )
  {
    case IndiMessage::Define:
      handleDefProperty( ipDispatch ); break;
    case IndiMessage::Delete:
      handleDelProperty( ipDispatch ); break;
    case IndiMessage::Message:
      handleMessage( ipDispatch ); break;
    case IndiMessage::NewProperty:
      handleNewProperty( ipDispatch ); break;
    case IndiMessage::SetProperty:
      handleSetProperty( ipDispatch ); break;
    default:
      logMsg << Logger::Error << "Client unable to dispatch INDI message." << endl;
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something before this device has been told to
/// start, like allocate memory.

void IndiClient::beforeExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something after this device has been told to
/// stop, like clean up allocated memory.

void IndiClient::afterExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override in derived class, place the code to do something here.

void IndiClient::execute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a DEF PROPERTY. A remote device sends a 'DEF' to tell other
/// INDI devices that are interested what properties it has available.
/// (see 'sendDefProperty')

void IndiClient::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a DEL PROPERTY. A remote device is telling us that one of its
/// properties is no longer available, or a device is no longer available.
/// (see 'sendDelProperty')

void IndiClient::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a MESSAGE. a remote device sent a generic message,
/// associated with a device or entire system.

void IndiClient::handleMessage( const pcf::IndiProperty &ipRecv )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a NEW PROPERTY. This is a request to update one of the
/// INDI properties we own.

void IndiClient::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a SET PROPERTY. This is a notification telling us that a
/// remote device changed one of its INDI properties.

void IndiClient::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Send an ENABLE BLOB. This behavior is only to be implemented in
/// intermediate INDI server processes; individual devices shall
/// disregard enableBLOB and send all elements at will.

void IndiClient::sendEnableBLOB( const IndiProperty &ipSend )
{
  IndiXmlParser ixp( IndiMessage( IndiMessage::EnableBLOB, ipSend ),
                     getProtocolVersion() );
  sendXml( ixp.createXmlString() );
}

////////////////////////////////////////////////////////////////////////////////
/// Send a GET PROPERTIES. When a Client first starts up, it begins
/// by sending the getProperties command. This includes the protocol
/// version and may include the name of a specific Device and Property
/// if it is known by some other means. If no device is specified,
/// then all devices are reported; if no name is specified,
/// then all properties for the given device are reported. The Device
/// then calls 'sendDefProperty' for each matching Property it offers
/// for control, limited to the Properties of the specified Device if
/// included.

void IndiClient::sendGetProperties( const IndiProperty &ipSend )
{
  IndiXmlParser ixp( IndiMessage( IndiMessage::GetProperties, ipSend ),
                     getProtocolVersion() );
  sendXml( ixp.createXmlString() );
}

////////////////////////////////////////////////////////////////////////////////
// Send a MESSAGE.

void IndiClient::sendMessage( const IndiProperty &ipSend )
{
  IndiXmlParser ixp( IndiMessage( IndiMessage::Message, ipSend ),
                     getProtocolVersion() );;
  sendXml( ixp.createXmlString() );
}

////////////////////////////////////////////////////////////////////////////////
// Send a NEW PROPERTY. This is a request to a remote device to
// update a property that the remote device owns.

void IndiClient::sendNewProperty( const IndiProperty &ipSend )
{
  IndiXmlParser ixp( IndiMessage( IndiMessage::NewProperty, ipSend ),
                     getProtocolVersion() );
  sendXml( ixp.createXmlString() );
}

////////////////////////////////////////////////////////////////////////////////
// Send an NEW PROPERTY vector. This is a request to a remote device to
// update a property that the remote device owns.

void IndiClient::sendNewProperties( const vector<IndiProperty> &vecIpSend )
{
  for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::NewProperty, vecIpSend[ii] ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
