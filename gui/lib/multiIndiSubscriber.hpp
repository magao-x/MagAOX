#ifndef multiIndiSubscriber_hpp
#define multiIndiSubscriber_hpp

#include <iostream>

#include <unordered_map>
#include <set>

#include "../../INDI/libcommon/IndiProperty.hpp"

/// Class implementing the functions of a subscriber to a multiIndiPublisher.
/** Derived classes will implement handleSetProperty, which is the callback 
  * for the publisher to use when a property changes.
  */ 
class multiIndiSubscriber
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
   
   /// The parent subscriber to which the instance is subscribed.
   multiIndiSubscriber * m_parent {nullptr};

   /// Child subscribers for which this instance is the parent
   propMapT subscribedProperties;

   subSetT subscribers;

   bool m_disconnect {false};

public:
   
   multiIndiSubscriber();
   
   /// Destructor
   /** Unsubscribes from the publisher.
     */
   virtual ~multiIndiSubscriber() noexcept;
   
   virtual void subscribe() {}

   /// Subscribes the given instance of multiIndiSubscriber to this instance
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */ 
   virtual int addSubscriber( multiIndiSubscriber * sub /**< [in] pointer to the subscriber */ );
   
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
                                    );
   
   /// Remove all subscriptions for this subscriber.
   /** This is mainly called by the multiIndiSubscriber destructor.
     */
   virtual void unsubscribe( multiIndiSubscriber * sub /**< [in] the subscriber being un-subscribed*/);

   /// Called by the parent once the parent is connected.
   /** If this is reimplemented, you should call multiIndiSubscriber::onConnect() to ensure children are notified.
     *
     */ 
   virtual void onConnect();

   /// Called by the parent once the parent is disconnected.
   /** If this is reimplemented, you should call multiIndiSubscriber::onDisconnect() to ensure children are notified.
     *
     */ 
   virtual void onDisconnect();

   void setDisconnect()
   {
      m_disconnect = true;
   }

   bool disconnect()
   {
      bool disc = m_disconnect;
      m_disconnect = false;
      return disc;
   }

   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   /// Callback for a SET PROPERTY message notifying us that the propery has changed.
   /** This is called by the publisher which is subscribed to.
     * 
     * Derived classes shall implement this.
     */
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   /// Send an NEW PROPERTY request to a remote device.
   /** This is a request to update a property that the remote device owns.
     */
   virtual void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] the property to send a change request for*/);
   
};

inline
multiIndiSubscriber::multiIndiSubscriber()
{
}

inline
multiIndiSubscriber::~multiIndiSubscriber() noexcept
{
   if(m_parent) 
   {
      m_parent->unsubscribe(this);
   }
}

inline
int multiIndiSubscriber::addSubscriber( multiIndiSubscriber * sub )
{
   subscribers.insert(sub);
   sub->m_parent = this;
   sub->subscribe();   
   return 0;
}

inline
int multiIndiSubscriber::addSubscriberProperty( multiIndiSubscriber * sub,
                                                pcf::IndiProperty & ipSub
                                              )
{
   subscribedProperties.insert(std::pair<std::string,multiIndiSubscriber*>(ipSub.createUniqueKey(), sub));

   subscribers.insert(sub);
   sub->m_parent = this;
   
   return 0;
}

inline
int multiIndiSubscriber::addSubscriberProperty( multiIndiSubscriber * sub,
                                                const std::string & device,
                                                const std::string & propName
                                              )
{
   pcf::IndiProperty ipSub;
   ipSub.setDevice(device);
   ipSub.setName(propName);
   
   return addSubscriberProperty(sub, ipSub);
}

inline
void multiIndiSubscriber::unsubscribe( multiIndiSubscriber * sub )
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
   sub->m_parent = nullptr;
}

inline
void multiIndiSubscriber::onConnect()
{
   for(subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it)
   {
      (*it)->onConnect();
   }
}

inline
void multiIndiSubscriber::onDisconnect()
{
   for(subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it)
   {
      (*it)->onDisconnect();
   }
}

inline
void multiIndiSubscriber::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   std::cerr << "Received Def: " << ipRecv.createUniqueKey() << "\n";
}

inline
void multiIndiSubscriber::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::cerr << "Received: " << ipRecv.createUniqueKey() << "\n";
}
   
inline
void multiIndiSubscriber::sendNewProperty( const pcf::IndiProperty &ipSend )
{
   if(m_parent == nullptr) return;
   m_parent->sendNewProperty(ipSend);
}

#endif //multiIndiSubscriber_hpp
