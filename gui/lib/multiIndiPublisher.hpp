/** \file multiIndiPublisher.hpp
 * \brief Shared INDI client publisher for multiple subscribers.
 */
#ifndef multiIndiPublisher_hpp
#define multiIndiPublisher_hpp

#include <algorithm>
#include <chrono>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <QObject>
#include <QTimer>

#include "../../INDI/libcommon/IndiClient.hpp"

#include "multiIndiSubscriber.hpp"

/// Define this before including this file if you wish to specify a version of the client.
#ifndef MULTI_INDI_CLIENT_VERSION
    #define MULTI_INDI_CLIENT_VERSION "none"
#endif

#define MULTI_INDI_PROTO_VERSION "1.7"

/// Dispatches a defProperty callback, queued through Qt for QObject subscribers.
/** Copies the received property so queued delivery does not reference connection-thread storage.
 */
inline void _dispatchDefProperty( multiIndiSubscriber     *sub, /**< [in] Subscriber receiving the callback. */
                                  const pcf::IndiProperty &ipRecv /**< [in] Property copied for queued delivery. */ )
{
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        pcf::IndiProperty ipCopy = ipRecv;
        QTimer::singleShot( 0, obj, [sub, ipCopy]() { sub->handleDefProperty( ipCopy ); } );
        return;
    }

    sub->handleDefProperty( ipRecv );
}

/// Dispatches a delProperty callback, queued through Qt for QObject subscribers.
/** Copies the received property so queued delivery does not reference connection-thread storage.
 */
inline void _dispatchDelProperty( multiIndiSubscriber     *sub, /**< [in] Subscriber receiving the callback. */
                                  const pcf::IndiProperty &ipRecv /**< [in] Property copied for queued delivery. */ )
{
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        pcf::IndiProperty ipCopy = ipRecv;
        QTimer::singleShot( 0, obj, [sub, ipCopy]() { sub->handleDelProperty( ipCopy ); } );
        return;
    }

    sub->handleDelProperty( ipRecv );
}

/// Dispatches a setProperty callback, queued through Qt for QObject subscribers.
/** Copies the received property so queued delivery does not reference connection-thread storage.
 */
inline void _dispatchSetProperty( multiIndiSubscriber     *sub, /**< [in] Subscriber receiving the callback. */
                                  const pcf::IndiProperty &ipRecv /**< [in] Property copied for queued delivery. */ )
{
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        pcf::IndiProperty ipCopy = ipRecv;
        QTimer::singleShot( 0, obj, [sub, ipCopy]() { sub->handleSetProperty( ipCopy ); } );
        return;
    }

    sub->handleSetProperty( ipRecv );
}

/// Dispatches an onConnect callback, queued through Qt for QObject subscribers.
inline void _dispatchOnConnect( multiIndiSubscriber *sub /**< [in] Subscriber receiving connection notification. */ )
{
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        QTimer::singleShot( 0, obj, [sub]() { sub->onConnect(); } );
        return;
    }

    sub->onConnect();
}

/// An INDI client which serves as a publisher for many subscribers.
/** This allows many widgets to use a single INDI client connection from within
 * one overall application (e.g. a GUI window). Named subscriptions share discovery
 * state across listeners. Unresolved queries back off from one second to sixty
 * seconds until a Def/Set arrives; device/FSM deletion rearms the device, and
 * named deletion rearms only that property. The receive loop services retries
 * during idle periods as well as incoming traffic. No remote commands are replayed.
 * Subscriber routing and discovery bookkeeping share m_subMutex; transport sends
 * and queued Qt callbacks occur outside the lock. A query already selected for
 * sending can race a receipt or unsubscription, producing at most that extra read.
 */
class multiIndiPublisher : public pcf::IndiClient, public multiIndiSubscriber
{

  protected:
    std::string m_hostAddress; ///< Legacy address storage; connection setup is owned by IndiClient.

    std::string m_hostPort; ///< Legacy port storage; connection setup is owned by IndiClient.

    std::recursive_mutex m_subMutex; ///< Guards subscriber routing and discovery state across Qt and receive threads.

    /// Discovery state shared by all listeners of one explicitly named property.
    struct discoveryState
    {
        pcf::IndiProperty m_query; ///< Targeted getProperties query with no element payload.

        bool m_received{ false }; ///< A Def/Set has arrived since registration or invalidation.

        std::chrono::seconds m_delay{ 0 }; ///< Delay after the last query, capped at sixty seconds.

        std::chrono::steady_clock::time_point m_next{}; ///< Next eligible discovery attempt.
    };

    /// Unique subscribed properties; guarded by m_subMutex with subscriber routing.
    std::unordered_map<std::string, discoveryState> m_discovery;

    /// Read the monotonic clock; tests override this without sleeping or using a transport.
    virtual std::chrono::steady_clock::time_point discoveryNow() const;

    /// Advance one unresolved property's capped exponential schedule after a query.
    void noteDiscoveryRequest( discoveryState                       &state, /**< [in,out] Property discovery state. */
                               std::chrono::steady_clock::time_point now /**< [in] Request time. */ );

    /// Remove discovery records whose last listener has unsubscribed; caller holds m_subMutex.
    void pruneDiscovery();

  public:
    /// Constructor, which establishes the INDI client connection.
    multiIndiPublisher( const std::string &clientName,  /**< [in] INDI client name. */
                        const std::string &hostAddress, /**< [in] INDI server host address. */
                        const int          hostPort     /**< [in] INDI server host port. */
    );

    /// Destructor.
    ~multiIndiPublisher() noexcept;

    /// Adds a subscriber to receive publisher events.
    /** QObject subscribers queue `subscribe()` onto their own Qt thread.
     */
    virtual int addSubscriber( multiIndiSubscriber *sub /**< [in] Subscriber to add. */ );

    /// Unsubscribes a subscriber from all events.
    virtual void unsubscribe( multiIndiSubscriber *sub /**< [in] Subscriber to remove. */ );

    /// Unsubscribes a subscriber from one specific property.
    virtual void unsubscribe( multiIndiSubscriber *sub,     /**< [in] Subscriber to remove. */
                              const std::string   &device,  /**< [in] Device name. */
                              const std::string   &propName /**< [in] Property name. */
    );

    /// Subscribe once per subscriber/property and discover its current definition.
    /** An unresolved property shares its outstanding discovery schedule. A new
     * listener for an already resolved property triggers a fresh targeted query.
     */
    /** \returns 0 on success, -1 on error.
     */
    virtual int addSubscriberProperty( multiIndiSubscriber *sub,  /**< [in] Subscriber pointer. */
                                       pcf::IndiProperty   &ipSub /**< [in] Property being subscribed to. */
    );

    /// Subscribes a subscriber for notifications using device/property names.
    /** \returns 0 on success, -1 on error.
     */
    virtual int addSubscriberProperty( multiIndiSubscriber *sub,     /**< [in] Subscriber pointer. */
                                       const std::string   &device,  /**< [in] Device name. */
                                       const std::string   &propName /**< [in] Property name. */
    );

    /// Handles defProperty messages from the INDI client.
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Received INDI property. */ );

    /// Handles delProperty messages from the INDI client.
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Received INDI property. */ );

    /// Handles setProperty messages from the INDI client.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Received INDI property. */ );

    /// Implementation of the IndiClient execution loop.
    void execute();

    /// Retry unresolved properties from the INDI receive loop, including idle iterations.
    void update() override;

    /// Called once the parent is connected.
    virtual void onConnect();

    /// Sends a newProperty request.
    virtual void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] Property update request. */ );

    /// Sends a getProperties request.
    virtual void sendGetProperties( const pcf::IndiProperty &ipSend /**< [in] Property query request. */ );

    /// Detaches the full subscriber tree from this publisher during teardown.
    /** Clears descendant back-pointers before the publisher object is destroyed.
     */
    void detachAllSubscribers();
};

inline multiIndiPublisher::multiIndiPublisher( const std::string &clientName,
                                               const std::string &hostAddress,
                                               const int          hostPort )
    : pcf::IndiClient( clientName, MULTI_INDI_CLIENT_VERSION, MULTI_INDI_PROTO_VERSION, hostAddress, hostPort )
{
    // pcf::IndiProperty ipSend;
    // sendGetProperties( ipSend );
}

inline multiIndiPublisher::~multiIndiPublisher() noexcept
{
}

inline int multiIndiPublisher::addSubscriber( multiIndiSubscriber *sub )
{
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        registerSubscriber( sub );
    }

    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        QTimer::singleShot( 0, obj, [sub]() { sub->subscribe(); } );
        return 0;
    }

    sub->subscribe();
    return 0;
}

inline void multiIndiPublisher::unsubscribe( multiIndiSubscriber *sub )
{
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    multiIndiSubscriber::unsubscribe( sub );
    pruneDiscovery();
}

inline void
multiIndiPublisher::unsubscribe( multiIndiSubscriber *sub, const std::string &device, const std::string &propName )
{
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    multiIndiSubscriber::unsubscribe( sub, device, propName );
    pruneDiscovery();
}

inline int multiIndiPublisher::addSubscriberProperty( multiIndiSubscriber *sub, pcf::IndiProperty &ipSub )
{
    pcf::IndiProperty query;
    query.setDevice( ipSub.getDevice() );
    query.setName( ipSub.getName() );
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        const auto                            key   = ipSub.createUniqueKey();
        const auto                            range = subscribedProperties.equal_range( key );
        for( auto it = range.first; it != range.second; ++it )
            if( it->second == sub )
                return 0;
        if( multiIndiSubscriber::addSubscriberProperty( sub, ipSub ) != 0 )
            return -1;

        if( !query.getDevice().empty() && !query.getName().empty() )
        {
            auto  result = m_discovery.emplace( key, discoveryState{} );
            auto &state  = result.first->second;
            if( !result.second && !state.m_received )
                return 0; // An existing unresolved query also serves this new listener.
            state         = discoveryState{};
            state.m_query = query;
            noteDiscoveryRequest( state, discoveryNow() );
        }
    }

    // QObject subscribe() already runs on its Qt thread. Send outside the routing lock;
    // do not leave a queued query that could follow a subscriber onto a new publisher.
    sendGetProperties( query );

    return 0;
}

inline int multiIndiPublisher::addSubscriberProperty( multiIndiSubscriber *sub,
                                                      const std::string   &device,
                                                      const std::string   &propName )
{
    pcf::IndiProperty ipSub;
    ipSub.setDevice( device );
    ipSub.setName( propName );

    return addSubscriberProperty( sub, ipSub );
}

inline void multiIndiPublisher::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    std::vector<multiIndiSubscriber *> subs;
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        auto                                  found = m_discovery.find( ipRecv.createUniqueKey() );
        if( found != m_discovery.end() )
            found->second.m_received = true;
        subs.reserve( subscribers.size() );
        for( subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it )
            subs.push_back( *it );
    }

    for( auto *sub : subs )
    {
        _dispatchDefProperty( sub, ipRecv );
    }
}

inline void multiIndiPublisher::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    std::vector<multiIndiSubscriber *> subs;
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        for( auto &entry : m_discovery )
        {
            auto &state = entry.second;
            if( state.m_query.getDevice() == ipRecv.getDevice() &&
                ( ipRecv.getName().empty() || ipRecv.getName() == "fsm" ||
                  state.m_query.getName() == ipRecv.getName() ) )
            {
                state.m_received = false;
                state.m_delay    = std::chrono::seconds( 0 );
                state.m_next     = discoveryNow();
            }
        }
        subs.reserve( subscribers.size() );
        for( subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it )
            subs.push_back( *it );
    }

    for( auto *sub : subs )
    {
        _dispatchDelProperty( sub, ipRecv );
    }
}

inline void multiIndiPublisher::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    std::vector<multiIndiSubscriber *> subs;
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        auto                                  found = m_discovery.find( ipRecv.createUniqueKey() );
        if( found != m_discovery.end() )
            found->second.m_received = true;
        std::pair<propMapIteratorT, propMapIteratorT> range =
            subscribedProperties.equal_range( ipRecv.createUniqueKey() );

        if( range.first == subscribedProperties.end() )
            return;

        for( propMapIteratorT it = range.first; it != range.second; ++it )
            subs.push_back( it->second );
    }

    for( auto *sub : subs )
    {
        _dispatchSetProperty( sub, ipRecv );
    }
}

inline void multiIndiPublisher::execute()
{
    processIndiRequests( false );
}

inline std::chrono::steady_clock::time_point multiIndiPublisher::discoveryNow() const
{
    return std::chrono::steady_clock::now();
}

inline void multiIndiPublisher::noteDiscoveryRequest( discoveryState &state, std::chrono::steady_clock::time_point now )
{
    state.m_delay = state.m_delay.count() == 0 ? std::chrono::seconds( 1 )
                                               : std::min( state.m_delay * 2, std::chrono::seconds( 60 ) );
    state.m_next  = now + state.m_delay;
}

inline void multiIndiPublisher::pruneDiscovery()
{
    for( auto it = m_discovery.begin(); it != m_discovery.end(); )
    {
        if( subscribedProperties.count( it->first ) == 0 )
            it = m_discovery.erase( it );
        else
            ++it;
    }
}

inline void multiIndiPublisher::update()
{
    std::vector<pcf::IndiProperty> queries;
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        const auto                            now = discoveryNow();
        for( auto &entry : m_discovery )
        {
            auto &state = entry.second;
            if( !state.m_received && now >= state.m_next )
            {
                queries.push_back( state.m_query );
                noteDiscoveryRequest( state, now );
            }
        }
    }
    for( const auto &query : queries )
        sendGetProperties( query );
}

inline void multiIndiPublisher::onConnect()
{
    std::vector<multiIndiSubscriber *> subs;
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        subs.reserve( subscribers.size() );
        for( subSetIteratorT it = subscribers.begin(); it != subscribers.end(); ++it )
            subs.push_back( *it );
    }

    for( auto *sub : subs )
    {
        _dispatchOnConnect( sub );
    }
}

inline void multiIndiPublisher::sendNewProperty( const pcf::IndiProperty &ipSend )
{
    pcf::IndiClient::sendNewProperty( ipSend );
}

inline void multiIndiPublisher::sendGetProperties( const pcf::IndiProperty &ipSend )
{
    pcf::IndiClient::sendGetProperties( ipSend );
}

inline void multiIndiPublisher::detachAllSubscribers()
{
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    detachSubscribersRecursive();
    m_discovery.clear();
}

#endif // multiIndiPublisher_hpp
