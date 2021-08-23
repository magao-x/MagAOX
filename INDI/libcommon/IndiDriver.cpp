/// $Id: IndiDriver.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include "IndiDriver.hpp"
#include "System.hpp"

using std::runtime_error;
using std::string;
using std::stringstream;
using std::endl;
using std::vector;
using pcf::System;
using pcf::TimeStamp;
using pcf::IndiConnection;
using pcf::IndiDriver;
using pcf::IndiMessage;
using pcf::IndiProperty;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

IndiDriver::IndiDriver()
  : IndiConnection()
{
  setup();
}

////////////////////////////////////////////////////////////////////////////////

IndiDriver::IndiDriver( const string &szName,
                        const string &szVersion,
                        const string &szProtocolVersion )
  : IndiConnection( szName, szVersion, szProtocolVersion )
{
  setup();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::IndiDriver
/// Copy constructor.
/// \param idRhs Another version of the driver.

IndiDriver::IndiDriver(const IndiDriver &idRhs ) : IndiConnection()
//  : IndiConnection( idRhs )  // can't invoke - private
{
  static_cast<void>(idRhs);
  // Empty because this is private.
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::operator =
/// Assignment operator.
/// \param idRhs The right-hand side of the operation.
/// \return This object.

const IndiDriver &IndiDriver::operator= ( const IndiDriver &idRhs )
//  : IndiConnection::operator= ( idRhs )  // can't invoke - private
{
  static_cast<void>(idRhs);
  // Empty because this is private.
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::~IndiDriver
/// Standard destructor.

IndiDriver::~IndiDriver() noexcept(true)
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::setup Sets up file descriptors and other things that
/// need to be initialized at construction time.

void IndiDriver::setup()
{
  // Grab the current time to use for the 'uptime' message.
  TimeStamp m_tsStartTime = TimeStamp::now();
  // Initialize the the last time sent with the current time.
  TimeStamp m_tsLastSent = TimeStamp::now();


  // If this driver saves any files or data to disk, this is where to do it.
  m_szDataDirectory = "";


  // This is a flag which can be used to fake data or pretend to be
  // connected to hardware. By itself, it does nothing.
  m_oIsSimulationModeEnabled = false;

  // This is also a flag which does nothing on its own, but can be used to
  // decide whether or not to send an alarm email.
  m_oIsAlarmModeEnabled = true;

  // This is a comma-separated list of email recipients which will receive
  // an email when a alarm is logged.
  m_szEmailList = "";

  // What is our alarm interval? This is the same if we are in
  // simulation mode or not. The default is one day (1440 minutes).
  // This is the maximun speed that alarms will go out.
  m_uiAlarmInterval = 1440;

  m_oIsAlarmActive = false;
  m_uiAlarmInterval = 1440; // Once a day (every 1440 minutes)


  // This will not be set to true until at least one 'getProperties' has
  // been received, or the user of this class sets it manually.
  m_oIsResponseModeEnabled = false;

  // Create and initialize the uptime message.
  m_ipUpTime = IndiProperty( IndiProperty::Number, getName(), "Version" );
  m_ipUpTime.setPerm( IndiProperty::ReadOnly );
  m_ipUpTime.setState( IndiProperty::Ok );
  m_ipUpTime.setTimeStamp( TimeStamp::now() );

  // Device version number.
  m_ipUpTime.add( IndiElement( "Driver", getVersion() ) );
  // Protocol version number.
  m_ipUpTime.add( IndiElement( "Properties", getProtocolVersion() ) );
  // Seconds since this device started.
  m_ipUpTime.add( IndiElement( "Uptime", m_tsStartTime.elapsedMillis() / 1000 ) );
  // Whether or not the driver thread is running.
  m_ipUpTime.add( IndiElement( "active", ( isActive() ? (1) : (0) ) ) );
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::update
/// Called in the process loop to perform an action each time through.

void IndiDriver::update()
{
  // Should we send an 'uptime' message?
  if ( m_tsLastSent.elapsedMillis() > 5000 &&
       isResponseModeEnabled() == true)
  {
    // Update the "uptime" message.
    m_ipUpTime[ "Uptime" ] = m_tsStartTime.elapsedMillis() / 1000;
    m_ipUpTime[ "active" ] = ( isActive() ? (1) : (0) );
    // Reset the last sent to wait another 5 seconds.
    m_tsLastSent = TimeStamp::now();
    sendSetProperty( m_ipUpTime );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief IndiDriver::dispatch
/// Chooses what to do with the received property.
/// \param tType Type of the message we received.
/// \param ipDispatch The property contained in the message.

void IndiDriver::dispatch( const IndiMessage::Type &tType,
                           const IndiProperty &ipDispatch )
{

  // Make sure the client knows about the basic properties the driver supports.
  if ( tType == IndiMessage::GetProperties )
  {
    handleDriverGetProperties( ipDispatch );
  }

  // Decide what we should do based on the type of the message.
  switch ( tType )
  {
    case IndiMessage::Define:
      handleDefProperty( ipDispatch ); break;
    case IndiMessage::Delete:
      handleDelProperty( ipDispatch ); break;
    case IndiMessage::GetProperties:
      handleGetProperties( ipDispatch ); break;
    case IndiMessage::Message:
      handleMessage( ipDispatch ); break;
    case IndiMessage::NewProperty:
      if ( handleDriverNewProperty( ipDispatch ) == false )
      {
        handleNewProperty( ipDispatch );
      }
      break;
    case IndiMessage::SetProperty:
      handleSetProperty( ipDispatch ); break;
    default:
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something before this device has been told to
/// start, like allocate memory.

void IndiDriver::beforeExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something after this device has been told to
/// stop, like clean up allocated memory.

void IndiDriver::afterExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override in derived class, place the code to do something here.

void IndiDriver::execute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Received a GET property, Respond with the properties the driver handles
/// at a basic level. These properties are common to all drivers.
/// Some of these are:
///
///   enable_active -> rmeoved for MagAO-X
///
/// Returns true (for now).

bool IndiDriver::handleDriverGetProperties( const IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);

  // Log the fact that we can now respond to INDI requests.
  if ( isResponseModeEnabled() == false )
  {
    enableResponseMode( true );
  }


  // For now, this always succeeds.
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Received a NEW property, decide if it should be handled here. A remote
/// device can send a property which can change the state of the driver at
/// a basic level common to all drivers. Some of these are:
///
///   enable_active.value=On/Off -  start or stop the internal worker thread.
///
/// Returns 'true' if the property was handled, 'false' otherwise.

bool IndiDriver::handleDriverNewProperty( const IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);

  // Assume we didn't handle the NEW and it will be handled by
  // the derived class.
  bool oHandledProperty = false;
  
  return oHandledProperty;
}

////////////////////////////////////////////////////////////////////////////////
/// Received a DEF PROPERTY. A remote device sends a 'DEF' to tell other
/// INDI devices that are interested what properties it has available.
/// (see 'sendDefProperty')

void IndiDriver::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a DEL PROPERTY. A remote device is telling us that one of its
/// properties is no longer available, or a device is no longer available.
/// (see 'sendDelProperty')

void IndiDriver::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a GET PROPERTIES. An remote device will send this command to
/// learn what INDI properties this device has. A DEF PROPERTY can be
/// sent as a reply. (see 'sendDefProperty')

void IndiDriver::handleGetProperties( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a MESSAGE. a remote device sent a generic message,
/// associated with a device or entire system.

void IndiDriver::handleMessage( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a NEW PROPERTY. This is a request to update one of the
/// INDI properties we own.

void IndiDriver::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Received a SET PROPERTY. This is a notification telling us that a
/// remote device changed one of its INDI properties.

void IndiDriver::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
  static_cast<void>(ipRecv);
}

////////////////////////////////////////////////////////////////////////////////
/// Send an DEF property to a client. This is usually done in response
/// to receiving a 'GetProperties' message. (see 'sendGetProperties')

void IndiDriver::sendDefProperty( const IndiProperty &ipSend ) const
{
  if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::Define, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Send an DEF property vector to a client. This is usually done in response
/// to receiving a 'GetProperties' message. (see 'sendGetProperties')

void IndiDriver::sendDefProperties( const vector<IndiProperty> &vecIpSend ) const
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      IndiXmlParser ixp( IndiMessage( IndiMessage::Define, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Send a DEL PROPERTY to a client. This tells a Client a given Property
/// is no longer available. If the property only specifies
/// a Device without a Name, the Client must assume all the Properties
/// for that Device, and indeed the Device itself, are no longer available.

void IndiDriver::sendDelProperty( const pcf::IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::Delete, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Send a DEL PROPERTY vector to a client. This tells a Client a given Property
/// is no longer available. If the property only specifies
/// a Device without a Name, the Client must assume all the Properties
/// for that Device, and indeed the Device itself, are no longer available.

void IndiDriver::sendDelProperties( const vector<IndiProperty> &vecIpSend )
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      IndiXmlParser ixp( IndiMessage( IndiMessage::Delete, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Send an ENABLE BLOB. This behavior is only to be implemented in
/// intermediate INDI server processes; individual devices shall
/// disregard enableBLOB and send all elements at will.

void IndiDriver::sendEnableBLOB( const IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::EnableBLOB, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
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

void IndiDriver::sendGetProperties( const IndiProperty &ipSend )
{
  // todo: Should this be disabled, or is this the one exception?
  //if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::GetProperties, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Send a MESSAGE.

void IndiDriver::sendMessage( const IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::Message, ipSend ),
                       getProtocolVersion() );;
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Send an SET PROPERTY. This is a notification that a property owned
// by this device has changed.

void IndiDriver::sendSetProperty( const IndiProperty &ipSend ) const
{
   IndiProperty _ipSend = ipSend;
   
   _ipSend.setTimeStamp(TimeStamp::now());
   
  if ( isResponseModeEnabled() == true )
  {
    IndiXmlParser ixp( IndiMessage( IndiMessage::SetProperty, _ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Send an SET PROPERTY vector. This is a notification that a property owned
// by this device has changed.

void IndiDriver::sendSetProperties( const vector<IndiProperty> &vecIpSend )
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      IndiXmlParser ixp( IndiMessage( IndiMessage::SetProperty, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Are the alarms enabled? True if yes, false otherwise.

bool IndiDriver::isAlarmModeEnabled() const
{
  return m_oIsAlarmModeEnabled;
}

////////////////////////////////////////////////////////////////////////////////
/// Turn on or off the alarms.

void IndiDriver::enableAlarmMode( const bool &oEnable )
{
  m_oIsAlarmModeEnabled = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Are we sending outgoing messages? True if yes, false otherwise.

bool IndiDriver::isResponseModeEnabled() const
{
  return m_oIsResponseModeEnabled;
}

////////////////////////////////////////////////////////////////////////////////
/// Turns the outgoing messages like 'SET', 'NEW', and 'DEF' on or off.

void IndiDriver::enableResponseMode( const bool &oEnable )
{
  m_oIsResponseModeEnabled = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Is this device's data being simulated? True if yes, false otherwise.

bool IndiDriver::isSimulationModeEnabled() const
{
  return m_oIsSimulationModeEnabled;
}

////////////////////////////////////////////////////////////////////////////////
// Turn on or off simulation mode. This can only be changed when this
// device is not active.

void IndiDriver::enableSimulationMode( const bool &oEnable )
{
  if ( isRunning() == true )
    throw runtime_error( string( "Tried to enabled simulation mode while active." ) );

  m_oIsSimulationModeEnabled = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
// Return the list of email recipients in case an alarm needs to be sent.
// This list is comma-separated.

string IndiDriver::getEmailList() const
{
  return m_szEmailList;
}

////////////////////////////////////////////////////////////////////////////////
// Return the path to where data files that are generated by this driver
// should be saved.

string IndiDriver::getDataDirectory() const
{
  return m_szDataDirectory;
}

////////////////////////////////////////////////////////////////////////////////
// The interval between giving alarms for the same alarm condition (sec).

unsigned int IndiDriver::getAlarmInterval() const
{
  return m_uiAlarmInterval;
}

////////////////////////////////////////////////////////////////////////////////
/// At what time was this object instantiated (constructor called)?

TimeStamp IndiDriver::getStartTime() const
{
  return m_tsStartTime;
}

////////////////////////////////////////////////////////////////////////////////
