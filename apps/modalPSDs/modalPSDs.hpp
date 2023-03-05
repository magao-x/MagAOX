/** \file modalPSDs.hpp
  * \brief The MagAO-X modalPSDs app header file
  *
  * \ingroup modalPSDs_files
  */

#ifndef modalPSDs_hpp
#define modalPSDs_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/sigproc/circularBuffer.hpp>

/** \defgroup modalPSDs
  * \brief An application to calculate rolling PSDs of modal amplitudes
  *
  * <a href="../handbook/operating/software/apps/modalPSDs.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup modalPSDs_files
  * \ingroup modalPSDs
  */

namespace MagAOX
{
namespace app
{

/// Class for application to calculate rolling PSDs of modal amplitudes.
/** 
  * \ingroup modalPSDs
  */
class modalPSDs : public MagAOXApp<true>, public dev::shmimMonitor<modalPSDs>
{

   //Give the test harness access.
   friend class modalPSDs_test;

   friend class dev::shmimMonitor<modalPSDs>;

public:

   /// The base shmimMonitor type
   typedef dev::shmimMonitor<modalPSDs> shmimMonitorT;

   /// The amplitude circular buffer type
   typedef mx::sigproc::circularBufferIndex<float, unsigned> ampCircBuffT;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_fpsSource; ///< Device name for getting fps to set circular buffer length.  This device should have *.fps.current.

   float m_psdTime {1}; ///< The length of time over which to calculate PSDs.  The default is 1 sec.
   float m_overSize {2}; ///< Multiplicative factor by which to oversize the circular buffer, to account for time-to-calculate.

   float m_psdOverlapFraction {0.5}; ///< The fraction of the sample time to overlap by.

   ///@}

   std::vector<ampCircBuffT> m_ampCircBuffs;

   float m_fps {0};

   unsigned m_tsCircBuffLength {4000}; ///< Length of the time-series circular buffers.  This is updated by m_fpsSource and m_psdTime. 

   unsigned m_psdSize {2000}; ///< The length of the time series sample over which the PSD is calculated
   unsigned m_psdOverlapSize {1000}; ///< The number of samples in the overlap

   /** \name PSD Calculation Thread
     * Handling of offloads from the average woofer shape
     * @{
     */
   int m_psdThreadPrio {0}; ///< Priority of the PSD Calculation thread.
   std::string m_psdThreadCpuset; ///< The cpuset to use for the PSD Calculation thread.

   std::thread m_psdThread; ///< The PSD Calculation thread.
   
   bool m_psdThreadInit {true}; ///< Initialization flag for the PSD Calculation thread.
   
   bool m_psdRestarting {true}; ///< Synchronization flag.  This will only become false after a successful call to allocate.
   bool m_psdWaiting {false}; ///< Synchronization flag.  This is set to true when the PSD thread is safely waiting for allocation to complete.

   pid_t m_psdThreadID {0}; ///< PSD Calculation thread PID.

   pcf::IndiProperty m_psdThreadProp; ///< The property to hold the PSD Calculation thread details.
   
   /// PS Calculation thread starter function
   static void psdThreadStart( modalPSDs * p /**< [in] pointer to this */);
   
   /// PSD Calculation thread function
   /** Runs until m_shutdown is true.
     */
   void psdThreadExec();

public:
   /// Default c'tor.
   modalPSDs();

   /// D'tor, declared and defined for noexcept.
   ~modalPSDs() noexcept
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

   /// Implementation of the FSM for modalPSDs.
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

   //shmimMonitor Interface
protected:

   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );


   //INDI Interface
protected:

   pcf::IndiProperty m_indiP_psdTime;
   INDI_NEWCALLBACK_DECL(modalPSDs, m_indiP_psdTime);

   pcf::IndiProperty m_indiP_overSize;
   INDI_NEWCALLBACK_DECL(modalPSDs, m_indiP_overSize);

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(modalPSDs, m_indiP_fpsSource);

   pcf::IndiProperty m_indiP_fps;

};

modalPSDs::modalPSDs() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void modalPSDs::setupConfig()
{
   shmimMonitorT::setupConfig(config);

   config.add("circBuff.fpsSource", "", "circBuff.fpsSource", argType::Required, "circBuff", "fpsSource", false, "string", "Device name for getting fps to set circular buffer length.  This device should have *.fps.current.");
   config.add("circBuff.defaultFPS", "", "circBuff.defaultFPS", argType::Required, "circBuff", "defaultFPS", false, "float", "Default FPS at startup, will enable changing average length with psdTime before INDI available.");
   config.add("circBuff.psdTime", "", "circBuff.psdTime", argType::Required, "circBuff", "psdTime", false, "float", "The length of time over which to calculate PSDs.  The default is 1 sec.");
   config.add("circBuff.overSize", "", "circBuff.overSize", argType::Required, "circBuff", "overSize", false, "float", "Multiplicative factor by which to oversize the circular buffer, to account for time-to-calculate..");

}

int modalPSDs::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);
   
   _config(m_fpsSource, "circBuff.fpsSource");
   _config(m_fps, "circBuff.defaultFPS");
   _config(m_psdTime, "circBuff.psdTime");
   _config(m_overSize, "circBuff.overSize");

   return 0;
}

void modalPSDs::loadConfig()
{
   loadConfigImpl(config);
}

int modalPSDs::appStartup()
{
   createStandardIndiNumber<unsigned>( m_indiP_psdTime, "psdTime", 0, 60, 0.1, "%0.1f");
   m_indiP_psdTime["current"].set(m_psdTime);
   m_indiP_psdTime["target"].set(m_psdTime);
   
   if( registerIndiPropertyNew( m_indiP_psdTime, INDI_NEWCALLBACK(m_indiP_psdTime)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiNumber<float>( m_indiP_overSize, "overSize", 0, 10, 0.1, "%0.1f");
   m_indiP_overSize["current"].set(m_overSize);
   m_indiP_overSize["target"].set(m_overSize);
   
   if( registerIndiPropertyNew( m_indiP_overSize, INDI_NEWCALLBACK(m_indiP_overSize)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createROIndiNumber( m_indiP_fps, "fps", "current", "Circular Buffer");
   m_indiP_fps.add(pcf::IndiElement("current"));
   m_indiP_fps["current"] = m_fps;

   if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(threadStart( m_psdThread, m_psdThreadInit, m_psdThreadID, m_psdThreadProp, m_psdThreadPrio, m_psdThreadCpuset, "psdcalc", this, psdThreadStart) < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   state(stateCodes::OPERATING);

   return 0;
}

int modalPSDs::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   return 0;
}

int modalPSDs::appShutdown()
{
   shmimMonitorT::appShutdown();

   if(m_psdThread.joinable())
   {
      pthread_kill(m_psdThread.native_handle(), SIGUSR1);
      try
      {
         m_psdThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }

   return 0;
}

int modalPSDs::allocate( const dev::shmimT & dummy)
{
   static_cast<void>(dummy);

   m_psdRestarting = true;

   //Prevent reallocation while the psd thread might be calculating
   while(m_psdWaiting == false && !shutdown()) mx::sys::microSleep(100); 

   if(shutdown()) return 0; //If shutdown() is true then shmimMonitor will cleanup

   if( m_fps > 0)
   {
      m_tsCircBuffLength = m_fps * m_psdTime * m_overSize;
      m_psdSize = m_fps*m_psdTime;
      m_psdOverlapSize = m_psdSize * m_psdOverlapFraction;
   }

   if(m_tsCircBuffLength <= 0 || !std::isnormal(m_tsCircBuffLength))
   {
      log<software_error>({__FILE__,__LINE__, "bad m_tsCircBuffLength value: " + std::to_string(m_tsCircBuffLength)});
      return -1;
   }

   if(m_psdSize <= 0 || !std::isnormal(m_psdSize))
   {
      log<software_error>({__FILE__,__LINE__, "bad m_psdSize value: " + std::to_string(m_psdSize)});
      return -1;
   }

   if(m_psdOverlapSize <= 0 || !std::isnormal(m_psdOverlapSize))
   {
      log<software_error>({__FILE__,__LINE__, "bad m_psdOverlapSize value: " + std::to_string(m_psdOverlapSize)});
      return -1;
   }

   //Check for unsupported type (must be float)
   if(shmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT)
   {
      //must be a vector of size 1 on one axis
      log<software_error>({__FILE__,__LINE__, "unsupported data type: must be float"});
      return -1;
   }

   //Check for unexpected format
   if(shmimMonitorT::m_width != 1 && shmimMonitorT::m_height != 1)
   {
      //must be a vector of size 1 on one axis
      log<software_error>({__FILE__,__LINE__, "unexpected shmim format"});
      return -1;
   }
   
   
   m_ampCircBuffs.resize(shmimMonitorT::m_width*shmimMonitorT::m_height);

   for(size_t n=0; n < m_ampCircBuffs.size(); ++n)
   {
      m_ampCircBuffs[n].maxEntries(m_tsCircBuffLength);
   }

   m_psdRestarting = false;

   return 0;
}
   
int modalPSDs::processImage( void * curr_src,
                             const dev::shmimT & dummy
                           )
{
   static_cast<void>(dummy);

   float * f_src = static_cast<float *>(curr_src);

   for(size_t n=0; n < m_ampCircBuffs.size(); ++n)
   {
      m_ampCircBuffs[n].nextEntry(f_src[n]);
   }

   return 0;
}

void modalPSDs::psdThreadStart( modalPSDs * p )
{
   p->psdThreadExec();
}


void modalPSDs::psdThreadExec( )
{
   m_psdThreadID = syscall(SYS_gettid);
   
   while( m_psdThreadInit == true && shutdown() == 0)
   {
      sleep(1);
   }
      
   while(shutdown() == 0)
   {
      if(m_psdRestarting == true || m_ampCircBuffs.size() == 0) m_psdWaiting = true;

      while((m_psdRestarting == true || m_ampCircBuffs.size() == 0) && !shutdown()) mx::sys::microSleep(100);

      if(shutdown()) break;

      m_psdWaiting = false;

      if(m_ampCircBuffs.size() == 0)
      {
         log<software_error>({__FILE__, __LINE__, "amp circ buff has zero size"});
         return;
      }

      if(m_ampCircBuffs.back().maxEntries() == 0)
      {
         log<software_error>({__FILE__, __LINE__, "amp circ buff entries have zero size"});
         return;
      }

      std::cerr << "waiting to grow\n";
      while( m_ampCircBuffs.back().size() < m_ampCircBuffs.back().maxEntries()  && m_psdRestarting == false && !shutdown())
      {
         //shrinking sleep
         double stime = (1.0*m_ampCircBuffs.back().maxEntries() - 1.0*m_ampCircBuffs.back().size())/m_fps * 0.5*1e9;
         mx::sys::nanoSleep(stime);
      }

      std::cerr << "all grown.  starting to calculate\n";

      ampCircBuffT::indexT ne0;
      ampCircBuffT::indexT ne1 = m_ampCircBuffs.back().latest(); 
      if(ne1 > m_psdOverlapSize) ne1 -= m_psdOverlapSize;
      else ne1 = m_ampCircBuffs.back().size() + ne1 - m_psdOverlapSize;

      while(m_psdRestarting == false && !shutdown())
      {
         //Used to check if we are getting too behind
         uint64_t mono0 = m_ampCircBuffs.back().mono();

         //Calc PSDs here
         ne0 = ne1;

         std::cerr << "calculating: " << ne0 << "\n";

         // *****
         //Loop over m_ampCircBuffs[n] with ne0 as refEntry
         // *****

         //Have to be cycling within the overlap
         if(m_ampCircBuffs.back().mono() - mono0 >= m_psdOverlapSize)
         {
            log<text_log>("PSD calculations getting behind, skipping ahead.", logPrio::LOG_WARNING);
            ne0 = m_ampCircBuffs.back().latest();
            if(ne0 > m_psdOverlapSize) ne0 -= m_psdOverlapSize;
            else ne0 = m_ampCircBuffs.back().size() + ne0 - m_psdOverlapSize;
         }

         //Now wait until we get to next one
         ne1 = ne0 + m_psdOverlapSize;
         if( ne1 >= m_ampCircBuffs.back().size()) 
         {
            ne1 -= m_ampCircBuffs.back().size();
         }

         ampCircBuffT::indexT ce = m_ampCircBuffs.back().latest(); 
         //wrapped difference
         long dn;
         if( ce >= ne1 ) dn = ce - ne1;
         else dn = ce + (m_ampCircBuffs[0].size() - ne1);

         while(dn < m_psdOverlapSize && !shutdown())
         {
            double stime = (1.0*dn)/m_fps * 0.5 * 1e9;
            mx::sys::nanoSleep(stime);

            ce = m_ampCircBuffs.back().latest();

            if( ce >= ne1 ) dn = ce - ne1;
            else dn = ce + (m_ampCircBuffs.back().size() - ne1);
         }


         
      }



   }
}

INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdTime)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_psdTime.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_psdTime, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(m_psdTime != target)
   {   
      std::lock_guard<std::mutex> guard(m_indiMutex);

      m_psdTime = target;
   
      updateIfChanged(m_indiP_psdTime, "current", m_psdTime, INDI_IDLE);
      updateIfChanged(m_indiP_psdTime, "target", m_psdTime, INDI_IDLE);

      shmimMonitorT::m_restart = true;
   
      log<text_log>("set psdTime to " + std::to_string(m_psdTime), logPrio::LOG_NOTICE);
   }

   return 0;
} //INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdTime)

INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_overSize)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_overSize.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_overSize, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(m_overSize != target)
   {   
      std::lock_guard<std::mutex> guard(m_indiMutex);

      m_overSize = target;
   
      updateIfChanged(m_indiP_overSize, "current", m_overSize, INDI_IDLE);
      updateIfChanged(m_indiP_overSize, "target", m_overSize, INDI_IDLE);

      shmimMonitorT::m_restart = true;
   
      log<text_log>("set overSize to " + std::to_string(m_overSize), logPrio::LOG_NOTICE);
   }

   return 0;
} //INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_overSize)

INDI_SETCALLBACK_DEFN(modalPSDs, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_fpsSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("current") != true ) //this isn't valid
   {
      log<software_error>({__FILE__, __LINE__, "No current property in fps source."});
      return 0;
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   float fps = ipRecv["current"].get<float>();
   
   if(fps != m_fps)
   {
      m_fps = fps;
      log<text_log>("set fps to " + std::to_string(m_fps), logPrio::LOG_NOTICE);
      updateIfChanged(m_indiP_fps, "current", m_fps, INDI_IDLE);
      shmimMonitorT::m_restart = true;
   }

   return 0;

} //INDI_SETCALLBACK_DEFN(modalPSDs, m_indiP_fpsSource)

} //namespace app
} //namespace MagAOX

#endif //modalPSDs_hpp
