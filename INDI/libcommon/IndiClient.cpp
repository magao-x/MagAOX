/// $Id: IndiClient.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include "IndiClient.hpp"
#include "SystemSocket.hpp"
//#include "Config.hpp"

using std::runtime_error;
using std::string;
using std::endl;
using std::vector;
using pcf::Thread;
using pcf::IndiClient;
using pcf::IndiMessage;
using pcf::IndiProperty;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

IndiClient::IndiClient( const string & szIPAddr,
                        const int & port
                      )
  : IndiConnection()
{
  setup(szIPAddr, port);
}

////////////////////////////////////////////////////////////////////////////////

IndiClient::IndiClient( const string &szName,
                        const string &szVersion,
                        const string &szProtocolVersion,
                        const string & szIPAddr,
                        const int & port
                      )
    : IndiConnection( szName, szVersion, szProtocolVersion )
{
  setup(szIPAddr, port);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiClient::IndiClient
/// Copy constructor.
/// \param icRhs Another version of the driver.

IndiClient::IndiClient(const IndiClient &icRhs ) : IndiConnection()
//  : IndiConnection( icRhs )  // can't invoke - private
{
  static_cast<void>(icRhs);
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
  static_cast<void>(icRhs);
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

void IndiClient::setup( const string & szIPAddr,
                        const int & port
                      )
{
  try
  {
    // These are the two descriptors we will use to talk to the outside world.
    // Set them by default to an invalid value.
    setInputFd( -1 ); // STDIN_FILENO;
    setOutputFd( -1 ); // STDOUT_FILENO;

    if ( m_socClient.isValid() == true )
    {
      m_socClient.close();
      Thread::msleep( 10 );
    }

    //Config cfReader;
    m_socClient = SystemSocket( SystemSocket::Stream, port, szIPAddr.c_str());

    m_socClient.connect();

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
    m_socClient.close();
    Thread::msleep( 10 );
    return;
  }
  catch ( const runtime_error &excepRuntime )
  {
    m_socClient.close();
    Thread::msleep( 10 );
    return;
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
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a DEL PROPERTY. A remote device is telling us that one of its
/// properties is no longer available, or a device is no longer available.
/// (see 'sendDelProperty')

void IndiClient::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a MESSAGE. a remote device sent a generic message,
/// associated with a device or entire system.

void IndiClient::handleMessage( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a NEW PROPERTY. This is a request to update one of the
/// INDI properties we own.

void IndiClient::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a SET PROPERTY. This is a notification telling us that a
/// remote device changed one of its INDI properties.

void IndiClient::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
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
