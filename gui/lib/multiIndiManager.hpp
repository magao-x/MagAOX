/** \file multiIndiManager.hpp
 * \brief Manager for a shared INDI publisher with reconnect handling.
 */
#ifndef multiIndiManager_hpp
#define multiIndiManager_hpp

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

#include <QObject>
#include <QTimer>

#include <unistd.h>

#include "multiIndiPublisher.hpp"

/// Dispatches disconnect callbacks on the Qt event loop when the subscriber is a QObject.
/** This avoids direct cross-thread QObject callback execution during disconnect cleanup.
 */
inline void _dispatchOnDisconnect( multiIndiSubscriber *sub )
{
    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        QTimer::singleShot( 0, obj, [sub]() { sub->onDisconnect(); } );
        return;
    }

    sub->onDisconnect();
}

/// Class to manage an INDI publisher and multiple INDI subscribers
/** Primary purpose of this class is to detect lack/loss of connection and
 * reconnect when able, then re-initialize the subscriptions.
 *
 *
 */
class multiIndiManager : public multiIndiSubscriber
{

  protected:
    std::string m_clientName;    ///< Name used for the INDI client.
    std::string m_hostAddress;   ///< Address of the INDI server host to connect to.
    int         m_hostPort{ 0 }; ///< Port of the INDI server host to connect to.

    multiIndiPublisher *m_publisher{ nullptr }; ///< Managed publisher/client instance.
    std::mutex          m_mutex;                ///< Guards publisher pointer and subscriber snapshots.

    std::thread m_monThread; ///< Connection monitor/reconnect thread.

    std::atomic_bool m_shutdown{ false }; ///< Signals monitor thread shutdown.

  public:
    /// Constructs an unconfigured manager.
    multiIndiManager();

    /// Constructs a manager with connection settings.
    multiIndiManager( const std::string &clientName,  ///< [in] INDI client name.
                      const std::string &hostAddress, ///< [in] INDI server host address.
                      const int          hostPort     ///< [in] INDI server host port.
    );

    /// Destroys the manager and stops the monitor thread.
    ~multiIndiManager();

    /// Gets the INDI client name.
    /** \returns The current INDI client name. */
    std::string clientName();

    /// Sets the INDI client name.
    /** After setting this, call `activate(true)` to reconnect with the new name.
     */
    void clientName( const std::string &cn /**< [in] New INDI client name. */ );

    /// Gets the INDI server host address.
    /** \returns The current host address. */
    std::string hostAddress();

    /// Sets the INDI server host address.
    /** After setting this, call `activate(true)` to reconnect with the new host.
     */
    void hostName( const std::string &hn /**< [in] New host address. */ );

    /// Gets the INDI server port.
    /** \returns The current host port. */
    int hostPort();

    /// Sets the INDI server port.
    /** After setting this, call `activate(true)` to reconnect with the new port.
     */
    void hostPort( int hp /**< [in] New host port. */ );

    /// Add a subscriber.
    /** If connected, this forwards registration to the active publisher.
     * QObject subscribers then queue `subscribe()` onto their own Qt thread.
     */
    virtual int addSubscriber( multiIndiSubscriber *sub /**< [in] Subscriber to add. */ );

    /// Removes a subscriber.
    virtual void unsubscribe( multiIndiSubscriber *sub /**< [in] Subscriber to remove. */ );

    /// Sends an INDI newProperty message through the managed publisher.
    virtual void sendNewProperty( const pcf::IndiProperty &ipSend /**< [in] Property update request. */ );

    /// Sends an INDI getProperties message through the managed publisher.
    virtual void sendGetProperties( const pcf::IndiProperty &ipSend /**< [in] Property query request. */ );

    /// Starts or restarts the connection monitor.
    void activate( bool force = false /**< [in] True forces reconnect by restarting the monitor thread. */ );

  public: // todo: make a protected static member
    /// Monitor thread entry point that manages connect/disconnect/reconnect.
    void connectClient();
};

inline multiIndiManager::multiIndiManager()
{
}

inline multiIndiManager::multiIndiManager( const std::string &clientName,
                                           const std::string &hostAddress,
                                           const int          hostPort )
    : m_clientName{ clientName }, m_hostAddress{ hostAddress }, m_hostPort{ hostPort }
{
}

inline multiIndiManager::~multiIndiManager()
{
    m_shutdown.store( true, std::memory_order_relaxed );

    if( m_monThread.joinable() )
    {
        try
        {
            m_monThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }
}

/// Starts the reconnect monitor thread body for a manager instance.
inline void _connectStart( multiIndiManager *mim )
{
    mim->connectClient();
}

inline void multiIndiManager::sendNewProperty( const pcf::IndiProperty &ipSend )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_publisher )
        m_publisher->sendNewProperty( ipSend );
}

inline void multiIndiManager::sendGetProperties( const pcf::IndiProperty &ipSend )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_publisher )
        m_publisher->sendGetProperties( ipSend );
}

inline void multiIndiManager::activate( bool force )
{
    if( force )
    {
        m_shutdown.store( true, std::memory_order_relaxed );

        if( m_monThread.joinable() )
        {
            try
            {
                m_monThread.join(); // this will throw if it was already joined
            }
            catch( ... )
            {
            }
        }
        m_shutdown.store( false, std::memory_order_relaxed );
    }

    if( m_monThread.joinable() )
        return; // Already running

    try
    {
        m_monThread = std::thread( _connectStart, this );
    }
    catch( const std::exception &e )
    {
        std::cerr << "Exception while activating INDI connection thread: " << e.what() << "\n";
    }
    catch( ... )
    {
        std::cerr << "Unknown exception while activating INDI connection thread.\n";
    }
}

inline int multiIndiManager::addSubscriber( multiIndiSubscriber *sub )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    subscribers.insert( sub );

    if( auto *obj = dynamic_cast<QObject *>( sub ) )
    {
        QObject::connect( obj, &QObject::destroyed, [this, sub]() { this->unsubscribe( sub ); } );
    }

    if( m_publisher != nullptr )
    {
        m_publisher->addSubscriber( sub );
    }

    return 0;
}

inline void multiIndiManager::unsubscribe( multiIndiSubscriber *sub )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    subscribers.erase( sub );
    if( m_publisher != nullptr )
    {
        m_publisher->unsubscribe( sub );
    }
}

inline void multiIndiManager::connectClient()
{
    while( !m_shutdown.load( std::memory_order_relaxed ) )
    {
        multiIndiPublisher                *pub{ nullptr };
        std::vector<multiIndiSubscriber *> subs;
        bool                               doDisconnect{ false };
        bool                               doConnect{ false };

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            if( m_publisher != nullptr ) // Check to see if we're still connected
            {
                if( m_publisher->getQuitProcess() || m_publisher->disconnect() ||
                    m_shutdown.load( std::memory_order_relaxed ) )
                {
                    pub         = m_publisher;
                    m_publisher = nullptr;
                    subs.reserve( subscribers.size() );
                    for( auto it = subscribers.begin(); it != subscribers.end(); ++it )
                        subs.push_back( *it );
                    doDisconnect = true;
                }
            }
            else
            {
                doConnect = true;
            }
        }

        if( doDisconnect )
        {
            pub->quitProcess();
            pub->deactivate();

            for( auto *sub : subs )
            {
                _dispatchOnDisconnect( sub );
            }

            pub->detachAllSubscribers();
            delete pub;
        }

        if( doConnect ) // try to connect
        {
            multiIndiPublisher *candidate{ nullptr };
            try
            {
                candidate = new multiIndiPublisher( m_clientName, m_hostAddress, m_hostPort );
            }
            catch( ... )
            {
                sleep( 1 );
                continue;
            }

            candidate->activate();

            sleep( 5 );

            // Check connection
            if( candidate->getQuitProcess() || m_shutdown.load( std::memory_order_relaxed ) ) // not connected
            {
                candidate->deactivate();
                delete candidate;
            }
            else // connected
            {
                { // mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    m_publisher = candidate;
                    subs.reserve( subscribers.size() );
                    for( auto it = subscribers.begin(); it != subscribers.end(); ++it )
                        subs.push_back( *it );
                }

                for( auto *sub : subs )
                {
                    candidate->addSubscriber( sub );
                }

                for( auto *sub : subs )
                {
                    _dispatchOnConnect( sub );
                }
            }
        }

        sleep( 1 );
    }

    multiIndiPublisher                *pub{ nullptr };
    std::vector<multiIndiSubscriber *> subs;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( m_publisher != nullptr ) // Before exiting, disconnect.
        {
            pub         = m_publisher;
            m_publisher = nullptr;
            subs.reserve( subscribers.size() );
            for( auto it = subscribers.begin(); it != subscribers.end(); ++it )
                subs.push_back( *it );
        }
    }

    if( pub != nullptr )
    {
        pub->quitProcess();
        pub->deactivate();

        for( auto *sub : subs )
        {
            _dispatchOnDisconnect( sub );
        }

        pub->detachAllSubscribers();
        delete pub;
    }
}

#endif // multiIndiManager_hpp
