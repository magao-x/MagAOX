/** \file shmimMonitor.hpp
  * \brief The MagAO-X generic shared memory monitor.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef shmimMonitor_hpp
#define shmimMonitor_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include "../../common/paths.hpp"


namespace MagAOX
{
namespace app
{
namespace dev 
{
   



/** MagAO-X generic frame grabber
  *
  * 
  * The derived class `derivedT` must expose the following interface
  * \code 
    
  * \endcode  
  * Each of the above functions should return 0 on success, and -1 on an error. 
  * 
  *
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic` and `appShutdown`
  * functions must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT>
class shmimMonitor 
{
protected:

   /** \name Configurable Parameters
    * @{
    */
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_smThreadPrio {2}; ///< Priority of the shmimMonitor thread, should normally be > 00.
    
    
   ///@}
   
   uint32_t m_width {0}; ///< The width of the images in the stream
   uint32_t m_height {0}; ///< The height of the images in the stream
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   
   IMAGE m_imageStream; ///< The ImageStreamIO shared memory buffer.
   
   
   
public:

   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       shmimMonitor<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       shmimMonitor<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** Starts the shmimMonitor thread
     * This should be called in `derivedT::appStartup` as
     * \code
       shmimMonitor<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Checks the shmimMonitor thread
   /** This should be called in `derivedT::appLogic` as
     * \code
       shmimMonitor<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// Shuts down the shmimMonitor thread
   /** This should be called in `derivedT::appShutdown` as
     * \code
       shmimMonitor<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
protected:
   

   /** \name shmimMonitor Thread
     * This thread waits on the stream semaphore
     * @{
     */
   
   bool m_smThreadInit {true}; ///< Synchronizer for thread startup, to allow priority setting to finish.
   
   std::thread m_smThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls smThreadExec.
   static void smThreadStart( shmimMonitor * s /**< [in] a pointer to a shmimMonitor instance (normally this) */);

   /// Execute the thread monitoring
   void smThreadExec();
   
   ///@}
  
   
    
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_shmimName; ///< Property used to report the shmim buffer name
   
   pcf::IndiProperty m_indiP_frameSize; ///< Property used to report the current frame size

public:

   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
void shmimMonitor<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("shmimMonitor.threadPrio", "", "shmimMonitor.threadPrio", argType::Required, "shmimMonitor", "threadPrio", false, "int", "The real-time priority of the shmimMonitor thread.");
   
   config.add("shmimMonitor.shmimName", "", "shmimMonitor.shmimName", argType::Required, "shmimMonitor", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /tmp/<shmimName>.im.shm.");
         
}

template<class derivedT>
void shmimMonitor<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_smThreadPrio, "shmimMonitor.threadPrio");
   m_shmimName = derived().configName();
   config(m_shmimName, "shmimMonitor.shmimName");
  
}
   

template<class derivedT>
int shmimMonitor<derivedT>::appStartup()
{
   //Register the shmimName INDI property
   m_indiP_shmimName = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_shmimName.setDevice(derived().configName());
   m_indiP_shmimName.setName("shmimName");
   m_indiP_shmimName.setPerm(pcf::IndiProperty::ReadWrite); ///\todo why is this ReadWrite?
   m_indiP_shmimName.setState(pcf::IndiProperty::Idle);
   m_indiP_shmimName.add(pcf::IndiElement("name"));
   m_indiP_shmimName["name"] = m_shmimName;
   
   if( derived().registerIndiPropertyNew( m_indiP_shmimName, nullptr) < 0)
   {
      #ifndef SHMIMMONITOR_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the frameSize INDI property
   m_indiP_frameSize = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_frameSize.setDevice(derived().configName());
   m_indiP_frameSize.setName("frameSize");
   m_indiP_frameSize.setPerm(pcf::IndiProperty::ReadWrite); ///\todo why is this ReadWrite?
   m_indiP_frameSize.setState(pcf::IndiProperty::Idle);
   m_indiP_frameSize.add(pcf::IndiElement("width"));
   m_indiP_frameSize["width"] = 0;
   m_indiP_frameSize.add(pcf::IndiElement("height"));
   m_indiP_frameSize["height"] = 0;
   
   if( derived().registerIndiPropertyNew( m_indiP_frameSize, nullptr) < 0)
   {
      #ifndef SHMIMMONITOR_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   if(derived().threadStart( m_smThread, m_smThreadInit, m_smThreadPrio, "shmimMonitor", this, smThreadStart) < 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}

template<class derivedT>
int shmimMonitor<derivedT>::appLogic()
{
   //do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_smThread.native_handle(),0) == 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "shmimMonitor thread has exited"});
      
      return -1;
   }
   
   return 0;

}


template<class derivedT>
int shmimMonitor<derivedT>::appShutdown()
{
   if(m_smThread.joinable())
   {
      try
      {
         m_smThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }
   
   
   
   return 0;
}



template<class derivedT>
void shmimMonitor<derivedT>::smThreadStart( shmimMonitor * s)
{
   s->smThreadExec();
}


template<class derivedT>
void shmimMonitor<derivedT>::smThreadExec()
{
   timespec writestart;
   
   //Wait fpr the thread starter to finish initializing this thread.
   while(m_smThreadInit == true && derived().shutdown() == 0)
   {
      sleep(1);
   }
   
   while(1);
   
}





template<class derivedT>
int shmimMonitor<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   indi::updateIfChanged(m_indiP_shmimName, "name", m_shmimName, derived().m_indiDriver);                     
   indi::updateIfChanged(m_indiP_frameSize, "width", m_width, derived().m_indiDriver);
   indi::updateIfChanged(m_indiP_frameSize, "height", m_height, derived().m_indiDriver);
   
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
