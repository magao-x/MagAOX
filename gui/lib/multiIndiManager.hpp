#ifndef multiIndiManager_hpp
#define multiIndiManager_hpp

#include <thread>
#include <mutex>

#include <unistd.h>

#include "multiIndiPublisher.hpp"

/// Class to manage an INDI publisher and multiple INDI subscribers 
/** Primary purpose of this class is to detect lack/loss of connection and
  * reconnect when able, then re-initialize the subscriptions.
  * 
  * 
  */
class multiIndiManager : public multiIndiSubscriber
{

protected:
   std::string m_clientName;  ///< Name used for the INDI client
   std::string m_hostAddress; ///< Address of the indiserver host to connect to
   int m_hostPort {0};        ///< Port on the host for indiserver
   
   //std::vector<multiIndiSubscriber *> m_subscribers; ///< Pointers to the subscribers themselves
   
   multiIndiPublisher * m_publisher {nullptr}; ///< The publisher, which is the INDI client which manages the distrubtion of properties to subscribers.
   
   std::thread m_monThread;

   bool m_shutdown {false};

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
   virtual int addSubscriber( multiIndiSubscriber * sub /**< [in] the subscriber to add*/);
   
   ///
   /*
    */
   void activate(bool force = false /**< [in] if true, then this will force a reconnection */);
   
public: //todo: make a protected static member
   
   void connectClient();
};

inline
multiIndiManager::multiIndiManager()
{
}

inline
multiIndiManager::multiIndiManager( const std::string & clientName,
                                    const std::string & hostAddress,
                                    const int hostPort 
                                  ) : m_clientName {clientName}, m_hostAddress{hostAddress}, m_hostPort{hostPort}
{
}

inline
multiIndiManager::~multiIndiManager()
{
   m_shutdown = true;

   if(m_monThread.joinable())
   {
      try
      {
         m_monThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }
}

inline
void _connectStart( multiIndiManager * mim )
{
   mim->connectClient();
}

inline
void multiIndiManager::activate(bool force)
{
   if(force)
   {
      m_shutdown = true;

      if(m_monThread.joinable())
      {
         try
         {
            m_monThread.join(); //this will throw if it was already joined
         }
         catch(...)
         {
          }
      }
      m_shutdown = false;
   }
   
   if(m_monThread.joinable()) return; //Already running

   try
   {
      m_monThread  = std::thread( _connectStart, this);
   }
   catch( const std::exception & e )
   {
      std::cerr << "Exception while activating INDI connection thread: " << e.what() << "\n";
   }
   catch( ... )
   {
      std::cerr << "Unknown exception while activating INDI connection thread.\n";
   }
}



inline
int multiIndiManager::addSubscriber( multiIndiSubscriber * sub )
{
   subscribers.insert(sub);
   if(m_publisher) 
   {
      m_publisher->addSubscriber(sub);
   }

   return 0;
}

inline
void multiIndiManager::connectClient()
{
   while( !m_shutdown )
   {
      if(m_publisher != nullptr) //Check to see if we're still connected
      {
         if(m_publisher->getQuitProcess() || m_publisher->disconnect() || m_shutdown)
         {
            m_publisher->quitProcess();
            m_publisher->deactivate();

            for(auto it = subscribers.begin(); it != subscribers.end(); ++it)
            {
               (*it)->onDisconnect();
               m_publisher->unsubscribe(*it);
            }
   
            delete m_publisher;
            m_publisher = nullptr;   
        }
      }
      else //try to connect
      {
         try
         {
            m_publisher = new multiIndiPublisher(m_clientName, m_hostAddress, m_hostPort);
         }
         catch(...) 
         {
            sleep(1);
            continue;
         }

         m_publisher->activate();
   
         sleep(5);

         //Check connection
         if(m_publisher->getQuitProcess()) //not connected
         {
            m_publisher->deactivate();
            delete m_publisher;
            m_publisher = nullptr;
      
         }
         else //connected
         {
            for(auto it = subscribers.begin(); it != subscribers.end(); ++it)
            {
               m_publisher->addSubscriber(*it);
            }
            m_publisher->onConnect();
         }
      }

      sleep(1);
   }

   if(m_publisher != nullptr) //Before exiting, disconnect.
   {
      m_publisher->quitProcess();
      m_publisher->deactivate();

      for(auto it = subscribers.begin(); it != subscribers.end(); ++it)
      {
         (*it)->onDisconnect();
         m_publisher->unsubscribe(*it);
      }

      delete m_publisher;
      m_publisher = nullptr;   
   }
}


#endif  //multiIndiManager_hpp
