/** \file multiIndiSubscriber.hpp
 * \brief Subscriber interface and routing for shared INDI publishers.
 */
#ifndef multiIndiSubscriber_hpp
#define multiIndiSubscriber_hpp

#include <atomic>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "../../INDI/libcommon/IndiProperty.hpp"

/// Class implementing the functions of a subscriber to a multiIndiPublisher.
/** Derived classes implement property callbacks used by a publisher.
 */
class multiIndiSubscriber
{

  public:
    /// Subscriber pointers are stored in a map keyed by `device.name` unique key.
    typedef std::unordered_multimap<std::string, multiIndiSubscriber *> propMapT;

    /// Forward iterator for the unordered_multimap of subscribers.
    typedef propMapT::iterator propMapIteratorT;

    /// Subscriber pointers are also stored in a set, to allow iteration.
    typedef std::set<multiIndiSubscriber *> subSetT;

    /// Iterator for the set of subscriber pointers.
    typedef subSetT::iterator subSetIteratorT;

  protected:
    multiIndiSubscriber *m_parent{ nullptr }; ///< Parent subscriber/publisher this instance is attached to.

    propMapT subscribedProperties; ///< Property-specific child subscriptions.

    subSetT subscribers; ///< Child subscribers registered under this instance.

    std::atomic_bool m_disconnect{ false }; ///< One-shot disconnect flag consumed by disconnect().

    /// Registers a child subscriber under this parent without invoking `subscribe()`.
    void registerSubscriber( multiIndiSubscriber *sub /**< [in] Subscriber being attached to this parent. */ );

    /// Clears parent pointers and subscription maps throughout this descendant tree.
    /** Used during publisher teardown to invalidate descendant back-pointers before the publisher is deleted.
     */
    void detachSubscribersRecursive();

  public:
    /// Constructs an unbound subscriber.
    multiIndiSubscriber();

    /// Destructor.
    /** Unsubscribes from parent publisher if currently attached.
     */
    virtual ~multiIndiSubscriber() noexcept;

    /// Hook invoked when the subscriber is attached to a parent.
    virtual void subscribe();

    /// Subscribes the given instance of multiIndiSubscriber to this instance.
    /** \returns 0 on success, -1 on error.
     */
    virtual int addSubscriber( multiIndiSubscriber *sub /**< [in] Pointer to the subscriber. */ );

    /// Subscribes a child to a specific property.
    /** \returns 0 on success, -1 on error.
     */
    virtual int addSubscriberProperty( multiIndiSubscriber *sub,  /**< [in] Pointer to the subscriber. */
                                       pcf::IndiProperty   &ipSub /**< [in] Property being subscribed to. */
    );

    /// Subscribes a child to a property by device and property name.
    /** \returns 0 on success, -1 on error.
     */
    virtual int addSubscriberProperty( multiIndiSubscriber *sub,     /**< [in] Pointer to the subscriber. */
                                       const std::string   &device,  /**< [in] Device name being subscribed to. */
                                       const std::string   &propName /**< [in] Property name being subscribed to. */
    );

    /// Removes all subscriptions for a child subscriber.
    /** Mainly called by the multiIndiSubscriber destructor.
     */
    virtual void unsubscribe( multiIndiSubscriber *sub /**< [in] Subscriber being unsubscribed. */ );

    /// Removes a single property subscription for a child subscriber.
    virtual void unsubscribe( multiIndiSubscriber *sub,     /**< [in] Subscriber being unsubscribed. */
                              const std::string   &device,  /**< [in] Device name being unsubscribed from. */
                              const std::string   &propName /**< [in] Property name being unsubscribed from. */
    );

    /// Called by the parent once connected.
    /** If reimplemented, call `multiIndiSubscriber::onConnect()` to notify children.
     */
    virtual void onConnect();

    /// Called by the parent once disconnected.
    /** If reimplemented, call `multiIndiSubscriber::onDisconnect()` to notify children.
     */
    virtual void onDisconnect();

    /// Sets the disconnect flag consumed by `disconnect()`.
    void setDisconnect();

    /// Consumes and returns the current disconnect flag.
    /** \returns True if a disconnect was requested since last call.
     */
    bool disconnect();

    /// Callback for a `defProperty` message.
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has been defined. */ );

    /// Callback for a `delProperty` message.
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has been deleted. */ );

    /// Callback for a `setProperty` message.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has changed. */ );

    /// Sends a `newProperty` request to a remote device.
    virtual void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] Property change request. */ );

    /// Sends a `getProperties` request to a remote device.
    virtual void sendGetProperties( const pcf::IndiProperty &ipSend /**< [in] Property query request. */ );
};

inline multiIndiSubscriber::multiIndiSubscriber()
{
}

inline multiIndiSubscriber::~multiIndiSubscriber() noexcept
{
    if( m_parent )
    {
        m_parent->unsubscribe( this );
    }
}

inline void multiIndiSubscriber::subscribe()
{
}

inline void multiIndiSubscriber::registerSubscriber( multiIndiSubscriber *sub )
{
    subscribers.insert( sub );
    sub->m_parent = this;
}

inline int multiIndiSubscriber::addSubscriber( multiIndiSubscriber *sub )
{
    registerSubscriber( sub );
    sub->subscribe();
    return 0;
}

inline int multiIndiSubscriber::addSubscriberProperty( multiIndiSubscriber *sub, pcf::IndiProperty &ipSub )
{
    subscribedProperties.insert( std::pair<std::string, multiIndiSubscriber *>( ipSub.createUniqueKey(), sub ) );

    registerSubscriber( sub );

    return 0;
}

inline int multiIndiSubscriber::addSubscriberProperty( multiIndiSubscriber *sub,
                                                       const std::string   &device,
                                                       const std::string   &propName )
{
    pcf::IndiProperty ipSub;
    ipSub.setDevice( device );
    ipSub.setName( propName );

    return addSubscriberProperty( sub, ipSub );
}

inline void multiIndiSubscriber::unsubscribe( multiIndiSubscriber *sub )
{
    // Since this is a forward iterator, erase invalidates and we cannot decrement.
    auto it = subscribedProperties.begin();
    while( it != subscribedProperties.end() )
    {
        if( it->second == sub )
        {
            subscribedProperties.erase( it );
            it = subscribedProperties.begin();
        }
        else
        {
            ++it;
        }
    }

    subscribers.erase( sub );
    sub->m_parent = nullptr;
}

inline void
multiIndiSubscriber::unsubscribe( multiIndiSubscriber *sub, const std::string &device, const std::string &propName )
{
    pcf::IndiProperty ipSub;
    ipSub.setDevice( device );
    ipSub.setName( propName );

    std::string key = ipSub.createUniqueKey();
    // Since this is a forward iterator, erase invalidates and we cannot decrement.
    auto it = subscribedProperties.begin();
    while( it != subscribedProperties.end() )
    {
        if( it->first == key && it->second == sub )
        {
            subscribedProperties.erase( it );
            it = subscribedProperties.begin();
        }
        else
        {
            ++it;
        }
    }

    bool have = false;
    for( auto it2 = subscribedProperties.begin(); it2 != subscribedProperties.end(); ++it2 )
    {
        if( it2->second == sub )
        {
            have = true;
            break;
        }
    }

    if( !have )
    {
        subscribers.erase( sub );
        sub->m_parent = nullptr;
    }
}

inline void multiIndiSubscriber::onConnect()
{
    for( subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it )
    {
        ( *it )->onConnect();
    }
}

inline void multiIndiSubscriber::onDisconnect()
{
    for( subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it )
    {
        ( *it )->onDisconnect();
    }
}

inline void multiIndiSubscriber::setDisconnect()
{
    m_disconnect.store( true, std::memory_order_relaxed );
}

inline void multiIndiSubscriber::detachSubscribersRecursive()
{
    std::vector<multiIndiSubscriber *> childSubs;
    childSubs.reserve( subscribers.size() );

    for( auto it = subscribers.begin(); it != subscribers.end(); ++it )
    {
        childSubs.push_back( *it );
    }

    for( auto *sub : childSubs )
    {
        sub->detachSubscribersRecursive();
        sub->m_parent = nullptr;
    }

    subscribedProperties.clear();
    subscribers.clear();
}

inline bool multiIndiSubscriber::disconnect()
{
    return m_disconnect.exchange( false, std::memory_order_relaxed );
}

inline void multiIndiSubscriber::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );
}

inline void multiIndiSubscriber::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );
}

inline void multiIndiSubscriber::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );
}

inline void multiIndiSubscriber::sendNewProperty( const pcf::IndiProperty &ipSend )
{
    if( m_parent == nullptr )
    {
        return;
    }

    m_parent->sendNewProperty( ipSend );
}

inline void multiIndiSubscriber::sendGetProperties( const pcf::IndiProperty &ipSend )
{
    if( m_parent == nullptr )
    {
        return;
    }

    m_parent->sendGetProperties( ipSend );
}

#endif // multiIndiSubscriber_hpp
