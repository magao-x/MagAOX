#ifndef multiIndiPublisher_hpp
#define multiIndiPublisher_hpp

#include <unordered_map>
#include <set>

#include "../../INDI/libcommon/IndiClient.hpp"

#include "multiIndiSubscriber.hpp"

/// Define this before including this file if you wish to specify a version of the client.
#ifndef MULTI_INDI_CLIENT_VERSION
#define MULTI_INDI_CLIENT_VERSION "none"
#endif

#define MULTI_INDI_PROTO_VERSION  "1.7"

/// An INDI client which serves as a publisher for many subscribers.
/** This allows many widgets to use a single INDI client connection from within
  * one overall application (e.g. a GUI window).
  * 
  * Subscribers, instances of classes derived from multiIndiSubscriber, register
  * callbacks to be notified when a specific property is changed.
  * 
  */ 
class multiIndiPublisher : public pcf::IndiClient, public multiIndiSubscriber
{

protected:
   
   std::string m_hostAddress;
   std::string m_hostPort;
   
public:

   /// Constructor, which establishes the INDI client connection.
   multiIndiPublisher( const std::string & clientName,
                       const std::string & hostAddress,
                       const int hostPort
                     );
   
   ~multiIndiPublisher() noexcept {};
   

   /// Subscribes the given instance of multiIndiSubscriber for notifications on the given property.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */ 
   virtual int addSubscriberProperty( multiIndiSubscriber * sub, ///< [in] pointer to the subscriber 
                                      pcf::IndiProperty & ipSub  ///< [in] the property being subscribed to.
                                    );
   
   virtual int addSubscriberProperty( multiIndiSubscriber * sub, ///< [in] pointer to the subscriber 
                                      const std::string & device,  ///< [in] the property being subscribed to.
                                      const std::string & propName
                                    )
   {            
      pcf::IndiProperty ipSub;
      ipSub.setDevice(device);
      ipSub.setName(propName);

      return addSubscriberProperty(sub, ipSub);
   }

   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   virtual void handleDelProperty( const pcf::IndiProperty &ipRecv );

   /// Responds to SET PROPERTY
   /** This is the implementation of the pcf::IndiClient interface function.
     * Calls the handleSetProperty callback for any subscribers registered on the property.
     */ 
   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the INDI property which has changed */);
   
   /// Implementation of the pcf::IndiClient interface, called by activate to actually beging the INDI event loop.
   /** 
     */
   void execute();
   
   /// Called once the parent is connected.
   virtual void onConnect()
   {
      //pcf::IndiProperty ipSend;
      //sendGetProperties( ipSend );  
      multiIndiSubscriber::onConnect(); 
   }

   virtual void sendNewProperty( const pcf::IndiProperty &ipSend)
   {
      pcf::IndiClient::sendNewProperty(ipSend);
   }

   virtual void sendGetProperties(const pcf::IndiProperty &ipSend)
   {
       pcf::IndiClient::sendGetProperties(ipSend);
   }
};

inline
multiIndiPublisher::multiIndiPublisher( const std::string & clientName,
                                        const std::string & hostAddress,
                                        const int hostPort
                                      ) : pcf::IndiClient( clientName, MULTI_INDI_CLIENT_VERSION, MULTI_INDI_PROTO_VERSION, hostAddress, hostPort)
{
   //pcf::IndiProperty ipSend;
   //sendGetProperties( ipSend );
}


inline
int multiIndiPublisher::addSubscriberProperty( multiIndiSubscriber * sub,
                                               pcf::IndiProperty & ipSub
                                             )
{
   if(multiIndiSubscriber::addSubscriberProperty(sub, ipSub) != 0)
   {
      return -1;
   }

   //note: we have to send this every time b/c otherwise late subscribers won't get an update on subscribe
   sendGetProperties(ipSub);


   return 0;
}

inline
void multiIndiPublisher::handleDefProperty( const pcf::IndiProperty &ipRecv )
{   
   for(subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it)
   {
      (*it)->handleDefProperty(ipRecv);
   }      
}

inline
void multiIndiPublisher::handleDelProperty( const pcf::IndiProperty &ipRecv )
{   
   for(subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it)
   {
      (*it)->handleDelProperty(ipRecv);
   }   
}

inline
void multiIndiPublisher::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
   std::pair< propMapIteratorT, propMapIteratorT> range = subscribedProperties.equal_range(ipRecv.createUniqueKey());

   if(range.first == subscribedProperties.end()) return;

   for(propMapIteratorT it = range.first; it != range.second; ++it)
   {
      it->second->handleSetProperty(ipRecv);
   }
}

inline
void multiIndiPublisher::execute()
{
   processIndiRequests(false);
}

#endif //multiIndiPublisher_hpp
