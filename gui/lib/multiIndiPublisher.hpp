/** \file multiIndiPublisher.hpp
 * \brief Shared INDI client publisher for multiple subscribers.
 */
#ifndef multiIndiPublisher_hpp
#define multiIndiPublisher_hpp

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
inline void _dispatchDefProperty( multiIndiSubscriber *sub, const pcf::IndiProperty &ipRecv )
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
inline void _dispatchDelProperty( multiIndiSubscriber *sub, const pcf::IndiProperty &ipRecv )
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
inline void _dispatchSetProperty( multiIndiSubscriber *sub, const pcf::IndiProperty &ipRecv )
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
inline void _dispatchOnConnect( multiIndiSubscriber *sub )
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
 * one overall application (e.g. a GUI window).
 */
class multiIndiPublisher : public pcf::IndiClient, public multiIndiSubscriber
{

  protected:
    std::string          m_hostAddress; ///< Host address configured for the publisher connection.
    std::string          m_hostPort;    ///< Host port configured for the publisher connection.
    std::recursive_mutex m_subMutex;    ///< Guards subscriber structures during publish/subscribe updates.

  public:
    /// Constructor, which establishes the INDI client connection.
    multiIndiPublisher( const std::string &clientName,  ///< [in] INDI client name.
                        const std::string &hostAddress, ///< [in] INDI server host address.
                        const int          hostPort     ///< [in] INDI server host port.
    );

    /// Destructor.
    ~multiIndiPublisher() noexcept;

    /// Adds a subscriber to receive publisher events.
    virtual int addSubscriber( multiIndiSubscriber *sub /**< [in] Subscriber to add. */ );

    /// Unsubscribes a subscriber from all events.
    virtual void unsubscribe( multiIndiSubscriber *sub /**< [in] Subscriber to remove. */ );

    /// Unsubscribes a subscriber from one specific property.
    virtual void unsubscribe( multiIndiSubscriber *sub,     /**< [in] Subscriber to remove. */
                              const std::string   &device,  /**< [in] Device name. */
                              const std::string   &propName /**< [in] Property name. */
    );

    /// Subscribes a subscriber for notifications on the given property.
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

    /// Called once the parent is connected.
    virtual void onConnect();

    /// Sends a newProperty request.
    virtual void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] Property update request. */ );

    /// Sends a getProperties request.
    virtual void sendGetProperties( const pcf::IndiProperty &ipSend /**< [in] Property query request. */ );
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
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    return multiIndiSubscriber::addSubscriber( sub );
}

inline void multiIndiPublisher::unsubscribe( multiIndiSubscriber *sub )
{
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    multiIndiSubscriber::unsubscribe( sub );
}

inline void
multiIndiPublisher::unsubscribe( multiIndiSubscriber *sub, const std::string &device, const std::string &propName )
{
    std::lock_guard<std::recursive_mutex> lock( m_subMutex );
    multiIndiSubscriber::unsubscribe( sub, device, propName );
}

inline int multiIndiPublisher::addSubscriberProperty( multiIndiSubscriber *sub, pcf::IndiProperty &ipSub )
{
    { // mutex scope
        std::lock_guard<std::recursive_mutex> lock( m_subMutex );
        if( multiIndiSubscriber::addSubscriberProperty( sub, ipSub ) != 0 )
        {
            return -1;
        }
    }

    // Request current value for late subscribers. For QObject-based subscribers,
    // queue via their event loop instead of sending inline from connection thread.
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        pcf::IndiProperty ipReq = ipSub;
        QTimer::singleShot( 0, obj, [sub, ipReq]() { sub->sendGetProperties( ipReq ); } );
    }
    else
    {
        sendGetProperties( ipSub );
    }

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
        std::lock_guard<std::recursive_mutex>         lock( m_subMutex );
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

#endif // multiIndiPublisher_hpp
