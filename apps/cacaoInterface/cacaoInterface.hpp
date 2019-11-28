/** \file cacaoInterface.hpp
  * \brief The MagAO-X CACAO Interface header file
  *
  * \ingroup cacaoInterface_files
  */

#ifndef cacaoInterface_hpp
#define cacaoInterface_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup cacaoInterface
  * \brief The CACAO Interface to provide loop status
  *
  * <a href="../handbook/apps/cacaoInterface.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup cacaoInterface_files
  * \ingroup cacaoInterface
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X CACAO Interface
/** 
  * \ingroup cacaoInterface
  */
class cacaoInterface : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class cacaoInterface_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}


   int m_loopState {0};
   float m_gain {0.0};
   
public:
   /// Default c'tor.
   cacaoInterface();

   /// D'tor, declared and defined for noexcept.
   ~cacaoInterface() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for cacaoInterface.
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /** 
     *
     */
   virtual int appShutdown();

   
   pcf::IndiProperty m_indiP_loopState;
   pcf::IndiProperty m_indiP_loopGain;
   
   /** \name File Monitoring Thread
     * Handling of offloads from the average woofer shape
     * @{
     */
   
   bool m_fmThreadInit {true}; ///< Initialization flag for the file monitoring thread.
   
   std::thread m_fmThread; ///< The file monitoring thread.

   /// File monitoring thread starter function
   static void fmThreadStart( cacaoInterface * c /**< [in] pointer to this */);
   
   /// File monitoring thread function
   /** Runs until m_shutdown is true.
     */
   void fmThreadExec();
   
   
   ///@}

};

cacaoInterface::cacaoInterface() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void cacaoInterface::setupConfig()
{
}

int cacaoInterface::loadConfigImpl( mx::app::appConfigurator & _config )
{
   static_cast<void>(_config); //be unused
   
   return 0;
}

void cacaoInterface::loadConfig()
{
   loadConfigImpl(config);
}

int cacaoInterface::appStartup()
{
   
   createROIndiText( m_indiP_loopState, "loopState", "state", "Loop State");
   registerIndiPropertyReadOnly(m_indiP_loopState);  
   
   createROIndiNumber( m_indiP_loopGain, "loopGain", "Loop Gain");
   indi::addNumberElement<float>(m_indiP_loopGain, "gain", 0, 1.0, 0.0, "%0.2f", "Loop Gain");
   registerIndiPropertyReadOnly(m_indiP_loopGain);  
   
   if(threadStart( m_fmThread, m_fmThreadInit, 0, "offload", this, fmThreadStart) < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;
}

int cacaoInterface::appLogic()
{
   
   //do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fmThread.native_handle(),0) == 0)
   {
      log<software_critical>({__FILE__, __LINE__, "cacao file monitoring thread has exited"});
      
      return -1;
   }
   
   if(m_loopState == 0) state(stateCodes::READY);
   else state(stateCodes::OPERATING);

   if(m_loopState == 0)  updateIfChanged(m_indiP_loopState, "state", "open", INDI_IDLE);
   if(m_loopState == 1)  updateIfChanged(m_indiP_loopState, "state", "paused", INDI_OK);
   if(m_loopState == 2)  updateIfChanged(m_indiP_loopState, "state", "closed", INDI_BUSY);
   
   updateIfChanged(m_indiP_loopGain, "gain", m_gain);
   
   return 0;
}

int cacaoInterface::appShutdown()
{
   
   if(m_fmThread.joinable())
   {
      try
      {
         m_fmThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }
   
   return 0;
}

void cacaoInterface::fmThreadStart( cacaoInterface * c )
{
   c->fmThreadExec();
}

void cacaoInterface::fmThreadExec( )
{
   while( m_fmThreadInit == true && shutdown() == 0)
   {
      sleep(1);
   }
   
   std::ifstream f_loop;
   std::string loop;
   
   std::ifstream f_gain;
   
  
   
   while(shutdown() == 0)
   {
      f_loop.open("/opt/MagAOX/cacao/tweeter/status/stat_loopON.txt");
      
      if(!f_loop.is_open()) 
      {
         sleep(1);
         continue;
      }
      
      
      f_gain.open("/opt/MagAOX/cacao/tweeter/conf/param_loopgain.txt");
      
      if(!f_gain.is_open()) 
      {
         sleep(1);
         continue;
      }
      
      
      f_loop >> loop;
      f_gain >> m_gain;
       
      if(loop[1] == 'F')
      {
         m_loopState = 0; //open
      }
      else if(m_gain == 0)
      {
         m_loopState = 1; //paused
      }
      else m_loopState = 2; //closed
      
      
      f_loop.close();
      f_gain.close();
      
      mx::milliSleep(100);

   }
   
   return;
}

} //namespace app
} //namespace MagAOX

#endif //cacaoInterface_hpp
