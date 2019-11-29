/** \file mzmqClient.hpp
  * \brief The MagAO-X milkzmqClient wrapper
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup mzmqClient_files
  */

#ifndef mzmqClient_hpp
#define mzmqClient_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <milkzmqClient.hpp>

#include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"



namespace MagAOX
{
namespace app
{

/** \defgroup mzmqClient ImageStreamIO 0mq Stream Client
  * \brief Reads the contents of an ImageStreamIO image stream over a zeroMQ channel
  *
  * <a href="../handbook/operating/software/apps/mzmqClient.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup mzmqClient_files ImageStreamIO Stream Synchronization
  * \ingroup mzmqClient
  */

/// MagAO-X application to control reading ImageStreamIO streams from a zeroMQ channel
/** Contents are published to a local ImageStreamIO shmem buffer.
  * 
  * \todo handle the alternate local name option as in the base milkzmqClient
  * \todo md docs for this.
  * 
  * \ingroup mzmqClient
  * 
  */
class mzmqClient : public MagAOXApp<>, public milkzmq::milkzmqClient
{

         
public:

   ///Default c'tor
   mzmqClient();

   ///Destructor
   ~mzmqClient() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();


   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   
protected:
   
   std::vector<std::string> m_shMemImNames;
   
   /** \name SIGSEGV & SIGBUS signal handling
     * These signals occur as a result of a ImageStreamIO source server resetting (e.g. changing frame sizes).
     * When they occur a restart of the framegrabber and framewriter thread main loops is triggered.
     * 
     * @{
     */ 
   bool m_restart {false};
   
   static mzmqClient * m_selfClient; ///< Static pointer to this (set in constructor).  Used for getting out of the static SIGSEGV handler.

   ///Sets the handler for SIGSEGV and SIGBUS
   /** These are caused by ImageStreamIO server resets.
     */
   int setSigSegvHandler();
   
   ///The handler called when SIGSEGV or SIGBUS is received, which will be due to ImageStreamIO server resets.  Just a wrapper for handlerSigSegv.
   static void _handlerSigSegv( int signum,
                                siginfo_t *siginf,
                                void *ucont
                              );

   ///Handles SIGSEGV and SIGBUS.  Sets m_restart to true.
   void handlerSigSegv( int signum,
                        siginfo_t *siginf,
                        void *ucont
                      );
   ///@}


};

//Set self pointer to null so app starts up uninitialized.
mzmqClient * mzmqClient::m_selfClient = nullptr;

inline
mzmqClient::mzmqClient() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
 
   return;
}

inline
mzmqClient::~mzmqClient() noexcept
{
   return;
}

inline
void mzmqClient::setupConfig()
{
   config.add("server.address", "", "server.address", argType::Required, "server", "address", false, "string", "The server's remote address. Usually localhost if using a tunnel.");
   config.add("server.imagePort", "", "server.imagePort", argType::Required, "server", "imagePort", false, "int", "The server's port.  Usually the port on localhost forwarded to the host.");
   
   config.add("server.shmimNames", "", "server.shmimNames", argType::Required, "server", "shmimNames", false, "string", "List of names of the remote shmim streams to get.");
   
   
 
}



inline
void mzmqClient::loadConfig()
{
   m_argv0 = m_configName;
   
   config(m_address, "server.address");
   config(m_imagePort, "server.imagePort");
   
   config(m_shMemImNames, "server.shmimNames");
   
   std::cerr << "m_imagePort = " << m_imagePort << "\n";
   
}


#include <sys/syscall.h>

inline
int mzmqClient::appStartup()
{
   if(setSigSegvHandler() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   for(size_t n=0; n < m_shMemImNames.size(); ++n)
   {
      shMemImName(m_shMemImNames[n]);
   }
   
   for(size_t n=0; n < m_imageThreads.size(); ++n)
   {
      if( imageThreadStart(n) > 0)
      {
         log<software_critical>({__FILE__, __LINE__, "Starting image thread " + m_imageThreads[n].m_imageName});
         return -1;
      }
   }
   
   return 0;

}



inline
int mzmqClient::appLogic()
{
   //first do a join check to see if other threads have exited.
   
   for(size_t n=0; n < m_imageThreads.size(); ++n)
   {
      if(pthread_tryjoin_np(m_imageThreads[n].m_thread->native_handle(),0) == 0)
      {
         log<software_error>({__FILE__, __LINE__, "image thread " + m_imageThreads[n].m_imageName + " has exited"});
      
         return -1;
      }
   }
   
   
   return 0;

}

inline
int mzmqClient::appShutdown()
{
   m_timeToDie = true;
   
   for(size_t n=0; n < m_imageThreads.size(); ++n)
   {
      imageThreadKill(n);
   }
   
   for(size_t n=0; n < m_imageThreads.size(); ++n)
   {
      if( m_imageThreads[n].m_thread->joinable())
      {
         m_imageThreads[n].m_thread->join();
      }
   }
   
   return 0;
}

inline
int mzmqClient::setSigSegvHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &mzmqClient::_handlerSigSegv;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGSEGV, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGSEGV failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGBUS, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGBUS failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0,logss});

      return -1;
   }

   log<text_log>("Installed SIGSEGV/SIGBUS signal handler.", logPrio::LOG_DEBUG);

   return 0;
}

inline
void mzmqClient::_handlerSigSegv( int signum,
                                    siginfo_t *siginf,
                                    void *ucont
                                  )
{
   m_selfClient->handlerSigSegv(signum, siginf, ucont);
}

inline
void mzmqClient::handlerSigSegv( int signum,
                                   siginfo_t *siginf,
                                   void *ucont
                                 )
{
   static_cast<void>(signum);
   static_cast<void>(siginf);
   static_cast<void>(ucont);
   
   m_restart = true;

   return;
}




}//namespace app
} //namespace MagAOX
#endif
