/// IndiClient.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_INDI_CLIENT_HPP
#define PCF_INDI_CLIENT_HPP

#include <string>
#include "IndiConnection.hpp"
#include "SystemSocket.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class IndiClient : public pcf::IndiConnection
{
  // construction/destruction/assign/copy
  public:
    /// Standard constructor.
    IndiClient( const std::string & szIPAddr,
                const int & port
              );
    /// Constructor which sets the name, version, and INDI protocol version.
    IndiClient( const std::string &szName,
                const std::string &szVersion,
                const std::string &szProtocolVersion,
                const std::string & szIPAddr,
                const int & port
              );

    /// Standard destructor.
    virtual ~IndiClient();

  // Prevent these from being invoked.
  private:
    /// Copy constructor.
    IndiClient( const IndiClient &icRhs );

    /// Assignment operator.
    const IndiClient &operator= ( const IndiClient &icRhs );

    /// Sets up file descriptors and other things that need to be
    /// initialized at construction time.
    void setup( const std::string & szIPAddr,
                const int & port
              );

  // Standard client interface methods.
  public:
    /// After the main execution loop, before the thread exits.
    virtual void afterExecute();
    /// Before the main execution loop, before the thread update loop starts.
    virtual void beforeExecute();
    /// Function which executes in the separate thread.
    virtual void execute();
    /// Chooses what to do with the received property.
    virtual void dispatch( const IndiMessage::Type &tType,
                           const IndiProperty &ipDispatch );
    /// Received a DEF PROPERTY. A remote device sends a 'DEF' to tell other
    /// INDI clients that are interested what properties it has available.
    /// (see 'sendDefProperty')
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
    /// Received a DEL PROPERTY. A remote device is telling us that one of its
    /// properties is no longer available, or a device is no longer available.
    /// (see 'sendDelProperty')
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv );
    /// Received a MESSAGE. a remote device sent a generic message,
    /// associated with a device or entire system.
    virtual void handleMessage( const pcf::IndiProperty &ipRecv );
    /// Received a NEW PROPERTY. This is a request to update one of the
    /// INDI properties we own.
    virtual void handleNewProperty( const pcf::IndiProperty &ipRecv );
    /// Received a SET PROPERTY. This is a notification telling us that a
    /// remote device changed one of its INDI properties.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );
    /// Called in the process loop to perform an action each time through.
    virtual void update();

    /// Send an ENABLE BLOB. This behavior is only to be implemented in
    /// intermediate INDI server processes; individual devices shall
    /// disregard enableBLOB and send all elements at will.
    void sendEnableBLOB( const pcf::IndiProperty &ipSend );
    /// Send a GET PROPERTIES. When a client first starts up, it begins
    /// by sending the getProperties command. This includes the protocol
    /// version and may include the name of a specific Device and Property
    /// if it is known by some other means. If no device is specified,
    /// then all devices are reported; if no name is specified,
    /// then all properties for the given device are reported. The Device
    /// then calls 'sendDefProperty' for each matching Property it offers
    /// for control, limited to the Properties of the specified Device if
    /// included.
    void sendGetProperties( const pcf::IndiProperty &ipSend );
    /// Send a MESSAGE.
    void sendMessage( const pcf::IndiProperty &ipSend );
    /// Send an NEW PROPERTY. This is a request to a remote device to
    /// update a property that the remote device owns.
    void sendNewProperty( const pcf::IndiProperty &ipSend );
    void sendNewProperties( const std::vector<pcf::IndiProperty> &vecIpSend );

  // Variables
  private:
    /// This is used to make a client connection.
    pcf::SystemSocket m_socClient;

}; // class IndiClient
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_INDI_CLIENT_HPP
