/** \file mzmqServer.hpp
  * \brief The MagAO-X milkzmqServer wrapper
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup mzmqServer_files
  */

#ifndef mzmqServer_hpp
#define mzmqServer_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <milkzmqServer.hpp>

#include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"



namespace MagAOX
{
namespace app
{

/** \defgroup mzmqServer ImageStreamIO Stream Server
  * \brief Writes the contents of an ImageStreamIO image stream over a zeroMQ channel
  *
  *  <a href="../apps_html/page_module_mzmqServer.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup mzmqServer_files ImageStreamIO Stream Synchronization
  * \ingroup mzmqServer
  */

/** MagAO-X application to control writing ImageStreamIO streams to a zeroMQ channel
  *
  * \ingroup mzmqServer
  * 
  */
class mzmqServer : public MagAOXApp<>, public milkzmq::milkzmqServer
{

         
public:

   ///Default c'tor
   mzmqServer();

   ///Destructor
   ~mzmqServer() noexcept;

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
   
   static mzmqServer * m_selfWriter; ///< Static pointer to this (set in constructor).  Used for getting out of the static SIGSEGV handler.

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
mzmqServer * mzmqServer::m_selfWriter = nullptr;

inline
mzmqServer::mzmqServer() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
 
   return;
}

inline
mzmqServer::~mzmqServer() noexcept
{
   return;
}

inline
void mzmqServer::setupConfig()
{
   config.add("server.imagePort", "", "imagePort", argType::Required, "server", "imagePort", false, "int", "");
   
   config.add("server.shmimNames", "", "server.shmimNames", argType::Required, "server", "shmimNames", false, "string", "");
   
   config.add("server.usecSleep", "", "server.usecSleep", argType::Required, "server", "usecSleep", false, "int", "");
   
   config.add("server.fpsTgt", "", "writer.fpsTgt", argType::Required, "server", "fpsTgt", false, "float", "");
   
   config.add("server.fpsGain", "", "server.fpsGain", argType::Required, "server", "fpsGain", false, "float", "");
 
}



inline
void mzmqServer::loadConfig()
{
   m_argv0 = m_configName;
   
   config(m_imagePort, "server.imagePort");
   
   config(m_shMemImNames, "server.shmimNames");
   config(m_usecSleep, "server.usecSleep");
   config(m_fpsTgt, "server.fpsTgt");
   
   config(m_fpsGain, "server.fpsGain");
   
   
   
}


#include <sys/syscall.h>

inline
int mzmqServer::appStartup()
{

   //Now set up the framegrabber and writer threads.
   // - need SIGSEGV and SIGBUS handling for ImageStreamIO restarts
   // - initialize the semaphore 
   // - start the threads
   
   if(setSigSegvHandler() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   
   for(size_t n=0; n < m_shMemImNames.size(); ++n)
   {
      shMemImName(m_shMemImNames[n]);
   }
   
   if(serverThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }

   for(size_t n=0; n < m_imageThreads.size(); ++n)
   {
      if( imageThreadStart(n) > 0)
      {
         log<software_critical>({__FILE__, __LINE__, "Starting image thread " + m_imageThreads[n].m_imageName});
         return -1;
      }
   }
   

   std::cerr << "Main Thread: " << syscall(SYS_gettid) << "\n";   
   return 0;

}



inline
int mzmqServer::appLogic()
{
   //first do a join check to see if other threads have exited.
   
   if(pthread_tryjoin_np(m_serverThread.native_handle(),0) == 0)
   {
      log<software_error>({__FILE__, __LINE__, "server thread has exited"});
      
      return -1;
   }
   
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
int mzmqServer::appShutdown()
{
   m_timeToDie = true;
   
   if(m_serverThread.joinable())
   {
      m_serverThread.join();
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
int mzmqServer::setSigSegvHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &mzmqServer::_handlerSigSegv;
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
void mzmqServer::_handlerSigSegv( int signum,
                                    siginfo_t *siginf,
                                    void *ucont
                                  )
{
   m_selfWriter->handlerSigSegv(signum, siginf, ucont);
}

inline
void mzmqServer::handlerSigSegv( int signum,
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
