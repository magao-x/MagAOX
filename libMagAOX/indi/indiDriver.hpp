/** \file indiDriver.hpp 
  * \brief A Class to implement a basic INDI Driver.
  * \author Paul Grenz (LBTI, original author)
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - Original code by Paul Grenz for the LBTI project.
  * - 2018-05-28 Ported to libMagAOX and refactoring begun by Jared Males.
  *   Changes made:
  *     - Converted to header only (inlined all function def'n after dec'l).
  *     - Deleted the private constructor and assignment op.
  *     - Reordered member dec'l, moved def'n to same order.
  *     - Removed libcommon config system.
  *     - Using initialization in declaration.  Default constructor no longer sets name and version.
  *     - Edited the handleDriver* methods to make them no-ops.  We are not using the enable_active property at this level.
  * 
  * \todo manage updating the upTime property for later updates to Versions.
  * \todo add uptime interval configuration
  * \todo move upTime to higher level -- keep this a low-level comms only class.
  */

#ifndef indi_indiDriver_hpp
#define indi_indiDriver_hpp

#include "indiConnection.hpp"

namespace MagAOX
{
namespace indi
{

class indiDriver : public indiConnection
{
   
public:
   /// Standard constructor.
   indiDriver();
    
   /// Constructor which sets the name, version, and INDI protocol version.
   indiDriver( const std::string &szName,
               const std::string &szVersion,
               const std::string &szProtocolVersion
             );
   
   /// Standard destructor.
   virtual ~indiDriver();

private:
   
   /// Copy constructor, deleted.
   indiDriver( const indiDriver &idRhs ) = delete;
    
   /// Assignment operator, deleted.
   const indiDriver &operator= ( const indiDriver &idRhs ) = delete;
    
   /// Sets up file descriptors and other things that need to be
   /// initialized at construction time.
   void setup();

public:
   
   /// Before the main execution loop, before the thread update loop starts.
   virtual void beforeExecute();

   /// Function which executes in the separate thread.
   virtual void execute();

   /// Called in the process loop to perform an action each time through.
   virtual void update();
   
   /// After the main execution loop, before the thread exits.
   virtual void afterExecute();
   
   /// Chooses what to do with the received property.
   virtual void dispatch( const pcf::IndiMessage::Type &tType, ///< [in] Type of the message we received.
                          const pcf::IndiProperty &ipDispatch  ///< [in] The property contained in the message.
                        );

   /// Handle a received DEF PROPERTY. 
   /** A remote device sends a 'DEF' to tell other
     * INDI devices that are interested what properties it has available.
     * (see 'sendDefProperty')
     */
   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
    
   /// Handle a received DEL PROPERTY. 
   /** A remote device is telling us that one of its
     * properties is no longer available, or a device is no longer available.
     * (see 'sendDelProperty')
     */
   virtual void handleDelProperty( const pcf::IndiProperty &ipRecv );
    
   /// Handle a received GET PROPERTIES. 
   /** A remote device will send this command to
     * learn what INDI properties this device has. A DEF PROPERTY can be
     * sent as a reply. (see 'sendDefProperty')
     */
   virtual void handleGetProperties( const pcf::IndiProperty &ipRecv );
    
    
   /// Handle a received a MESSAGE. 
   /** A remote device sent a generic message,
     * associated with a device or entire system.
     */
   virtual void handleMessage( const pcf::IndiProperty &ipRecv );
    
   /// Handle a received NEW PROPERTY. 
   /** This is a request to update one of the
     * INDI properties we own.
     */
   virtual void handleNewProperty( const pcf::IndiProperty &ipRecv );
    
   /// Handle a received SET PROPERTY. 
   /** This is a notification telling us that a
     * remote device changed one of its INDI properties.
     */
   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

private:
   /** Hanel a received GET property
     * Respond with the properties the driver handles
     * at a basic level. These properties are common to all drivers.
     * Some of these are:
     *   - enable_active.value=On/Off : start or stop the internal worker thread.
     *
     * \returns true (for now).
     */
   bool handleDriverGetProperties( const pcf::IndiProperty &ipRecv );
    
   /// Handle a received NEW property
   /** Decide if it should be handled here. A remote
     * device can send a property which can change the state of the driver at
     * a basic level common to all drivers.
     * Some of these are:
     *   - enable_active.value=On/Off -  start or stop the internal worker thread.
     *
     * \retval true if the property was handled
     * \retval false if the property was not handled.
     */
   bool handleDriverNewProperty( const pcf::IndiProperty &ipRecv );
   
public:   
   
   /// Send a DEF property to a client. 
   /** This is usually done in response
     * to receiving a 'GetProperties' message. (see 'sendGetProperties')
     */
   void sendDefProperty( const pcf::IndiProperty &ipSend ) const;

   /// Send a DEF property vector to a client. 
   /** This is usually done in response
     * to receiving a 'GetProperties' message. (see 'sendGetProperties')
     */
   void sendDefProperties( const std::vector<pcf::IndiProperty> &vecIpSend ) const;

   /// Send a DEL PROPERTY to a client. 
   /** This tells a Client a given Property
     * is no longer available. If the property only specifies
     * a Device without a Name, the Client must assume all the Properties
     * for that Device, and indeed the Device itself, are no longer available.
     */
   void sendDelProperty( const pcf::IndiProperty &ipSend );

   /// Send a DEL PROPERTY vector to a client. 
   /** This tells a Client a given Property
     * is no longer available. If the property only specifies
     * a Device without a Name, the Client must assume all the Properties
     * for that Device, and indeed the Device itself, are no longer available.
     */
   void sendDelProperties( const std::vector<pcf::IndiProperty> &vecIpSend );

   /// Send an ENABLE BLOB. 
   /** This behavior is only to be implemented in
     * intermediate INDI server processes; individual devices shall
     * disregard enableBLOB and send all elements at will.
     */
   void sendEnableBLOB( const pcf::IndiProperty &ipSend );

   /// Send a GET PROPERTIES. 
   /** When a Client first starts up, it begins
     * by sending the getProperties command. This includes the protocol
     * version and may include the name of a specific Device and Property
     * if it is known by some other means. If no device is specified,
     * then all devices are reported; if no name is specified,
     * then all properties for the given device are reported. The Device
     * then calls 'sendDefProperty' for each matching Property it offers
     * for control, limited to the Properties of the specified Device if
     * included.
     */
   void sendGetProperties( const pcf::IndiProperty &ipSend );
    
   /// Send a MESSAGE.
   void sendMessage( const pcf::IndiProperty &ipSend );
    
   /// Send a SET PROPERTY. 
   /** This is a notification that a property owned
     * by this device has changed.
     */
   void sendSetProperty( const pcf::IndiProperty &ipSend ) const;
    
   /// Send a SET PROPERTY vector. 
   /** This is a notification that a property owned
     * by this device has changed.
     */
   void sendSetProperties( const std::vector<pcf::IndiProperty> &vecIpSend );

   
   // Once we have seen at least 1 'GET PROPERTIES' message, we can send
   // 'SET', 'NEW', and 'DEF', but not before.
   bool isResponseModeEnabled() const;
   
   // Enables the sending of outgoing messages, like 'SET', 'NEW', and 'DEF'.
   // If this is false no messages will be sent.
   void enableResponseMode( const bool & oEnable );
    
   /// At what time was this object instantiated (constructor called)?
   pcf::TimeStamp getStartTime() const;

   // Variables
private:
     
   // Can we send outgoing messages? This can only happen after we
   // have received at least 1 'getProperties'.
   bool m_oIsResponseModeEnabled {false};
    
   /// The time this program was started.
   pcf::TimeStamp m_tsStartTime;
   /// The last instant when the time was sent.
   pcf::TimeStamp m_tsLastSent;
   /// A property to handle the "uptime" message.
   pcf::IndiProperty m_ipUpTime;
   
}; // class indiDriver

inline
indiDriver::indiDriver() : indiConnection()
{
   setup();
}

inline
indiDriver::indiDriver( const std::string &szName,
                        const std::string &szVersion,
                        const std::string &szProtocolVersion
                      ) : indiConnection( szName, szVersion, szProtocolVersion )
{
   setup();
}

inline
indiDriver::~indiDriver()
{
}

inline
void indiDriver::setup()
{   
   // Grab the current time to use for the 'uptime' message.
   pcf::TimeStamp m_tsStartTime = pcf::TimeStamp::now();
  
   // Initialize the the last time sent with the current time.
   pcf::TimeStamp m_tsLastSent = pcf::TimeStamp::now();

   // Create and initialize the uptime message.
   m_ipUpTime = pcf::IndiProperty( pcf::IndiProperty::Number, getName(), "Version" );
   m_ipUpTime.setPerm( pcf::IndiProperty::ReadOnly );
   m_ipUpTime.setState( pcf::IndiProperty::Ok );
   m_ipUpTime.setTimeStamp( pcf::TimeStamp::now() );

   // Device version number.
   m_ipUpTime.add( pcf::IndiElement( "Driver", getVersion() ) );
   // Protocol version number.
   m_ipUpTime.add( pcf::IndiElement( "Properties", getProtocolVersion() ) );
   // Seconds since this device started.
   m_ipUpTime.add( pcf::IndiElement( "Uptime", m_tsStartTime.elapsedMillis() / 1000 ) );
   // Whether or not the driver thread is running.
   m_ipUpTime.add( pcf::IndiElement( "active", ( isActive() ? (1) : (0) ) ) );
   
}


inline
void indiDriver::beforeExecute()
{
}

inline
void indiDriver::execute()
{
   if(sm_oQuitProcess == false)
   {  
      processIndiRequests(false);
   }
}

inline
void indiDriver::afterExecute()
{
}

inline
void indiDriver::update()
{
  // Should we send an 'uptime' message?
  if ( m_tsLastSent.elapsedMillis() > 5000 &&
       isResponseModeEnabled() == true)
  {
    // Update the "uptime" message.
    m_ipUpTime[ "Uptime" ] = m_tsStartTime.elapsedMillis() / 1000;
    m_ipUpTime[ "active" ] = ( isActive() ? (1) : (0) );
    // Reset the last sent to wait another 5 seconds.
    m_tsLastSent = pcf::TimeStamp::now();
    sendSetProperty( m_ipUpTime );
  }
}

inline
void indiDriver::dispatch( const pcf::IndiMessage::Type &tType,
                           const pcf::IndiProperty &ipDispatch )
{
  pcf::Logger logMsg;
  logMsg.enableClearAfterLog( true );

  // Make sure the client knows about the basic properties the driver supports.
  if ( tType == pcf::IndiMessage::GetProperties )
  {
    handleDriverGetProperties( ipDispatch );
  }

  // Decide what we should do based on the type of the message.
  switch ( tType )
  {
     case pcf::IndiMessage::Define:
      handleDefProperty( ipDispatch ); break;
    case pcf::IndiMessage::Delete:
      handleDelProperty( ipDispatch ); break;
    case pcf::IndiMessage::GetProperties:
      handleGetProperties( ipDispatch ); break;
    case pcf::IndiMessage::Message:
      handleMessage( ipDispatch ); break;
    case pcf::IndiMessage::NewProperty:
      if ( handleDriverNewProperty( ipDispatch ) == false )
      {
        handleNewProperty( ipDispatch ); break;
      }
    case pcf::IndiMessage::SetProperty:
      handleSetProperty( ipDispatch ); break;
    default:
      logMsg << pcf::Logger::Error << "Driver unable to dispatch INDI message." << std::endl;
      break;
  }
}

inline
void indiDriver::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
}

inline
void indiDriver::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
}

inline
void indiDriver::handleGetProperties( const pcf::IndiProperty &ipRecv )
{
}

inline
void indiDriver::handleMessage( const pcf::IndiProperty &ipRecv )
{
}

inline
void indiDriver::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
}

inline
void indiDriver::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
}

inline
bool indiDriver::handleDriverGetProperties( const pcf::IndiProperty &ipRecv )
{
  enableResponseMode( true );
  
  // For now, this always succeeds.
  return true;
}

bool indiDriver::handleDriverNewProperty( const pcf::IndiProperty &ipRecv )
{
   return false;
}

inline
void indiDriver::sendDefProperty( const pcf::IndiProperty &ipSend ) const
{
  if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::Define, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendDefProperties( const std::vector<pcf::IndiProperty> &vecIpSend ) const
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::Define, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

inline
void indiDriver::sendDelProperty( const pcf::IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::Delete, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendDelProperties( const std::vector<pcf::IndiProperty> &vecIpSend )
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::Delete, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

inline
void indiDriver::sendEnableBLOB( const pcf::IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::EnableBLOB, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendGetProperties( const pcf::IndiProperty &ipSend )
{
  /// \todo Should this be disabled, or is this the one exception?
  //if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::GetProperties, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendMessage( const pcf::IndiProperty &ipSend )
{
  if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::Message, ipSend ),
                       getProtocolVersion() );;
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendSetProperty( const pcf::IndiProperty &ipSend ) const
{
  if ( isResponseModeEnabled() == true )
  {
    pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::SetProperty, ipSend ),
                       getProtocolVersion() );
    sendXml( ixp.createXmlString() );
  }
}

inline
void indiDriver::sendSetProperties( const std::vector<pcf::IndiProperty> &vecIpSend )
{
  if ( isResponseModeEnabled() == true )
  {
    for ( unsigned int ii = 0; ii < vecIpSend.size(); ii++ )
    {
      pcf::IndiXmlParser ixp( pcf::IndiMessage( pcf::IndiMessage::SetProperty, vecIpSend[ii] ),
                         getProtocolVersion() );
      sendXml( ixp.createXmlString() );
    }
  }
}

inline
void indiDriver::enableResponseMode( const bool &oEnable )
{
  m_oIsResponseModeEnabled = oEnable;
}

inline
bool indiDriver::isResponseModeEnabled() const
{
  return m_oIsResponseModeEnabled;
}

   
} // namespace indi
} // namespace MagAOX


#endif // indi_indiDriver_hpp
