#ifndef multiIndiManager_hpp
#define multiIndiManager_hpp

#include <unistd.h>

#include <QObject>
#include <QTimer>

#include "multiIndi.hpp"

/// Class to manage an INDI publisher and multiple INDI subscribers 
/** Primary purpose of this class is to detect lack/loss of connection and
  * reconnect when able, then re-initialize the subscriptions.
  * 
  * 
  */
class multiIndiManager : public QObject
{
   Q_OBJECT 
   
protected:
   std::string m_clientName;  ///< Name used for the INDI client
   std::string m_hostAddress; ///< Address of the indiserver host to connect to
   int m_hostPort {0};        ///< Port on the host for indiserver
   
   std::vector<multiIndiSubscriber *> m_subscribers; ///< Pointers to the subscribers themselves
   
   multiIndiPublisher * m_publisher {nullptr}; ///< The publisher, which is the INDI client which manages the distrubtion of properties to subscribers.
   
   QTimer m_timer; ///< Timer object for checking connection status.

public:
   /// Default c'tor
   /*
    */ 
   multiIndiManager();
   
   /// Constructor which sets up and initiates the connection
   /*
    */
   multiIndiManager( const std::string & clientName,  ///< [in]
                     const std::string & hostAddress, ///< [in]
                     const int hostPort               ///< [in]
                   );
   
   /// Destructor
   /* Disconnects and cleans up the client.
    */
   ~multiIndiManager();
   
   /// Get the 
   /**
     * \returns the current value of
     */ 
   std::string clientName();
   
   /// Set the 
   /** After setting this, you will need to call activate(true) to reset the client.
     */
   void clientName( const std::string & cn /**< [in] the new*/);
   
   /// Get the 
   /**
     * \returns the current value of
     */
   std::string hostAddress();
   
   /// Set the 
   /** After setting this, you will need to call activate(true) to reset the client.
     */
   void hostName( const std::string & hn /**< [in] the new*/);
   
   /// Get the 
   /**
     * \returns the current value of
     */
   int hostPort();
   
   /// Set the 
   /** After setting this, you will need to call activate(true) to reset the client.
     */
   void hostPort( int hp  /**< [in] the new*/);
   
   /// Add a subscriber.
   /** If connected, this immediately calls the subscribers subscribe member function.
     */
   void addSubscriber( multiIndiSubscriber * sub /**< [in] the subscriber to add*/);
   
   ///
   /*
    */
   void activate(bool force = false /**< [in] if true, then this will force a reconnection */);
   
   
public slots:
   
   ///
   /*
    */
   void timerout();

protected:
   
   void connectClient(bool force = false /**< [in] if true, then this will force a reconnection */);
};

multiIndiManager::multiIndiManager()
{
   connect(&m_timer, SIGNAL(timeout()), this, SLOT(timerout()));
}

multiIndiManager::multiIndiManager( const std::string & clientName,
                                    const std::string & hostAddress,
                                    const int hostPort 
                                  ) : m_clientName {clientName}, m_hostAddress{hostAddress}, m_hostPort{hostPort}
{
   connect(&m_timer, SIGNAL(timeout()), this, SLOT(timerout()));
   activate();
}

void multiIndiManager::activate(bool force)
{
   connectClient(force);
}

multiIndiManager::~multiIndiManager()
{
   if(m_publisher)
   {
      m_publisher->quitProcess();
      m_publisher->deactivate();
   }
}

void multiIndiManager::addSubscriber( multiIndiSubscriber * sub )
{
   m_subscribers.push_back(sub);
   if(m_publisher) sub->subscribe(m_publisher);
}

void multiIndiManager::timerout()
{
   connectClient();
}

void multiIndiManager::connectClient(bool force)
{
   m_timer.stop();
   
   if(m_publisher != nullptr)
   {
      if(m_publisher->getQuitProcess() || force)
      {
         m_publisher->quitProcess();
         m_publisher->deactivate();
         delete m_publisher;
         m_publisher = nullptr;
         
         for(size_t n=0;n<m_subscribers.size();++n)
         {
            m_subscribers[n]->onDisconnect();
         }
   
      }
      else 
      {
         m_timer.start(1000);
         return;
      }
   }
   
   //std::cerr << "connecting\n";
   
   m_publisher = new multiIndiPublisher(m_clientName, m_hostAddress, m_hostPort);
   m_publisher->activate();
   
   sleep(2);
   
   if(m_publisher->getQuitProcess())
   {
      //std::cerr << "connection failed\n";
      m_publisher->quitProcess();
      m_publisher->deactivate();
      delete m_publisher;
      m_publisher = nullptr;
      m_timer.start(1000);
      return;
   }
      
   for(size_t n=0;n<m_subscribers.size();++n)
   {
      m_subscribers[n]->subscribe(m_publisher);
   }
   
   m_timer.start(1000);
}

//Include moc so this can be a "header only" file
#include "moc_multiIndiManager.cpp"

#endif  //multiIndiManager_hpp
