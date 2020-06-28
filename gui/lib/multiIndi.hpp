
#ifndef multiIndiPublisher_hpp
#define multiIndiPublisher_hpp

#include <unordered_map>
#include <set>

#include "../../INDI/libcommon/IndiClient.hpp"

/// Define this before including this file if you wish to specify a version of the client.
#ifndef MULTI_INDI_CLIENT_VERSION
#define MULTI_INDI_CLIENT_VERSION "none"
#endif

#define MULTI_INDI_PROTO_VERSION  "1.7"

//Forward:
class multiIndiPublisher;

/// Class implementing the functions of a subscriber to a multiIndiPublisher.
/** Derived classes will implement handleSetProperty, which is the callback 
  * for the publisher to use when a property changes.
  */ 
class multiIndiSubscriber
{
   /// We allos multiIndiPublisher to access the publisher pointer
   friend class multiIndiPublisher;
   
protected:
   
   /// The parent publisher to which the instance is subscribed.
   multiIndiPublisher * publisher {nullptr};

   /// Set the publisher to which this instance is subscribed.
   void setPublisher(multiIndiPublisher * p /**< [in] the publisher to which this instance is subscribed */);
   
public:
   
   multiIndiSubscriber();
   
   /// Destructor
   /** Unsubscribes from the publisher.
     */
   virtual ~multiIndiSubscriber();
   
   virtual int subscribe( multiIndiPublisher * publisher ) = 0;
   
   /// Called when the publisher disconnects.
   virtual void onDisconnect()
   {
      std::cerr << "disconnected\n";
   }
   
   virtual int handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   /// Callback for a SET PROPERTY message notifying us that the propery has changed.
   /** This is called by the publisher which is subscribed to.
     * 
     * Derived classes shall implement this.
     */
   virtual int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   /// Send an NEW PROPERTY request to a remote device.
   /** This is a request to update a property that the remote device owns.
     */
   void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] the property to send a change request for*/);
   
};

/// An INDI client which serves as a publisher for many subscribers.
/** This allows many widgets to use a single INDI client connection from within
  * one overall application (e.g. a GUI window).
  * 
  * Subscribers, instances of classes derived from multiIndiSubscriber, register
  * callbacks to be notified when a specific property is changed.
  * 
  */ 
class multiIndiPublisher : public pcf::IndiClient
{
public:

   /// Subscriber pointers are stored in a map, keyed by the property `device.name` unique key.
   typedef std::unordered_multimap< std::string, multiIndiSubscriber *> propMapT;

   /// The forward iterator for the unordered_multimap of subscribers
   typedef propMapT::iterator propMapIteratorT;
   
   /// Subscriber pointers are also stored in a set, to allow iteration over them
   typedef std::set<multiIndiSubscriber*> subSetT;
   
   /// The iterator for the set of subscriber pointers
   typedef subSetT::iterator subSetIteratorT;

protected:
   
   std::string m_hostAddress;
   std::string m_hostPort;
   
   /// Contains the subscriber pointers
   propMapT subscribedProperties;

   subSetT subscribers;
public:

   /// Constructor, which establishes the INDI client connection.
   multiIndiPublisher( const std::string & clientName,
                       const std::string & hostAddress,
                       const int hostPort
                     );
   
   /// Subscribes the given instance of multiIndiSubscriber for notifications on the given property.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */ 
   int subscribe( multiIndiSubscriber * sub /**< [in] pointer to the subscriber */ );
   
   /// Subscribes the given instance of multiIndiSubscriber for notifications on the given property.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */ 
   int subscribeProperty( multiIndiSubscriber * sub, ///< [in] pointer to the subscriber 
                          pcf::IndiProperty & ipSub  ///< [in] the property being subscribed to.
                        );

   int subscribeProperty( multiIndiSubscriber * sub, ///< [in] pointer to the subscriber 
                          const std::string & device,  ///< [in] the property being subscribed to.
                          const std::string & propName
                        );
   
   /// Remove all subscriptions for this subscriber.
   /** This is mainly called by the multiIndiSubscriber destructor.
     */
   void unsubscribe( multiIndiSubscriber * sub /**< [in] the subscriber being un-subscribed*/);
   
   void handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   /// Responds to SET PROPERTY
   /** This is the implementation of the pcf::IndiClient interface function.
     * Calls the handleSetProperty callback for any subscribers registered on the property.
     */ 
   void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the INDI property which has changed */);
   
   /// Implementation of the pcf::IndiClient interface, called by activate to actually beging the INDI event loop.
   /** 
     */
   void execute();
   
};

//--**************************************************************--//
//--               multiIndiSubscriber Definitions                --//
//--**************************************************************--//

inline
multiIndiSubscriber::multiIndiSubscriber()
{

}

inline
multiIndiSubscriber::~multiIndiSubscriber()
{
   if(publisher) 
   {
      publisher->unsubscribe(this);
   }
}

inline
void multiIndiSubscriber::setPublisher(multiIndiPublisher * p)
{
   publisher = p;
}

inline
int multiIndiSubscriber::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   std::cerr << "Received Def: " << ipRecv.createUniqueKey() << "\n";

   return 0;
}

inline
int multiIndiSubscriber::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::cerr << "Received: " << ipRecv.createUniqueKey() << "\n";

   return 0;
}
   
inline
void multiIndiSubscriber::sendNewProperty( const pcf::IndiProperty &ipSend )
{
   if(publisher == nullptr) return;
   
   publisher->sendNewProperty(ipSend);
}

//--**************************************************************--//
//--                multiIndiPublisher Definitions                --//
//--**************************************************************--//

inline
multiIndiPublisher::multiIndiPublisher( const std::string & clientName,
                                        const std::string & hostAddress,
                                        const int hostPort
                                      ) : pcf::IndiClient( clientName, MULTI_INDI_CLIENT_VERSION, MULTI_INDI_PROTO_VERSION, hostAddress, hostPort)
{
   pcf::IndiProperty ipSend;
   sendGetProperties( ipSend );
}

inline
int multiIndiPublisher::subscribe( multiIndiSubscriber * sub )
{
   subscribers.insert(sub);
   
   sub->setPublisher(this);
   
   return 0;
}

inline
int multiIndiPublisher::subscribeProperty( multiIndiSubscriber * sub,
                                           pcf::IndiProperty & ipSub
                                         )
{
   size_t already = subscribedProperties.count(ipSub.createUniqueKey());

   subscribedProperties.insert(std::pair<std::string,multiIndiSubscriber*>(ipSub.createUniqueKey(), sub));
   subscribers.insert(sub);
   
   sub->setPublisher(this);
   
   if(already == 0)
   {
      sendGetProperties(ipSub);
   }
   return 0;
}

inline
int multiIndiPublisher::subscribeProperty( multiIndiSubscriber * sub,
                                           const std::string & device,
                                           const std::string & propName
                                         )
{
   pcf::IndiProperty ipSub;
   ipSub.setDevice(device);
   ipSub.setName(propName);
   
   return subscribeProperty(sub, ipSub);
   
}

inline
void multiIndiPublisher::unsubscribe( multiIndiSubscriber * sub )
{
   //Since this is a forward iterator, we can't just for-loop through this because the erase invalidates, and we can't decrement!
   auto it = subscribedProperties.begin();
   while(it != subscribedProperties.end() )
   {
      if(it->second == sub) 
      {
         subscribedProperties.erase(it);
         it=subscribedProperties.begin();
      }
      else  ++it;
   }
   
   subscribers.erase(sub);
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
