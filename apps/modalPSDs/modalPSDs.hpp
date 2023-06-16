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
#include <mx/sigproc/signalWindows.hpp>

#include <mx/math/fft/fftwEnvironment.hpp>
#include <mx/math/fft/fft.hpp>

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
   friend class dev::shmimMonitor<modalPSDs>;

public:

   typedef float realT;
   typedef std::complex<realT> complexT;

   /// The base shmimMonitor type
   typedef dev::shmimMonitor<modalPSDs> shmimMonitorT;

   /// The amplitude circular buffer type
   typedef mx::sigproc::circularBufferIndex<float *, unsigned> ampCircBuffT;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_fpsSource; ///< Device name for getting fps to set circular buffer length.  This device should have *.fps.current.

   realT m_psdTime {1}; ///< The length of time over which to calculate PSDs.  The default is 1 sec.
   realT m_psdAvgTime {10}; ///< The time over which to average PSDs.  The default is 10 sec.

   //realT m_overSize {10}; ///< Multiplicative factor by which to oversize the circular buffer, to give good mean estimates and account for time-to-calculate.

   realT m_psdOverlapFraction {0.5}; ///< The fraction of the sample time to overlap by.

   int m_nPSDHistory {100}; //

   
   ///@}

   int m_nModes; ///< the number of modes to calculate PSDs for.

   ampCircBuffT m_ampCircBuff;

   //std::vector<ampCircBuffT> m_ampCircBuffs;

   realT m_fps {0};
   realT m_df {1};

   //unsigned m_tsCircBuffLength {4000}; ///< Length of the time-series circular buffers.  This is updated by m_fpsSource and m_psdTime. 

   unsigned m_tsSize {2000}; ///< The length of the time series sample over which the PSD is calculated
   unsigned m_tsOverlapSize {1000}; ///< The number of samples in the overlap

   std::vector<realT> m_win; ///< The window function.  By default this is Hann.
   
   realT * m_tsWork {nullptr};
   size_t m_tsWorkSize {0};
   
   std::complex<realT> * m_fftWork {nullptr};
   size_t m_fftWorkSize {0};
   
   std::vector<realT> m_psd;

   mx::math::fft::fftT< realT, std::complex<realT>, 1, 0> m_fft;
   mx::math::fft::fftwEnvironment<realT> m_fftEnv;

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

   IMAGE * m_freqStream {nullptr}; ///< The ImageStreamIO shared memory buffer to hold the frequency scale

   mx::improc::eigenImage<realT> m_psdBuffer;

   IMAGE * m_rawpsdStream {nullptr};

   IMAGE * m_avgpsdStream {nullptr}; ///< The ImageStreamIO shared memory buffer to hold the average psds

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
   
   int allocatePSDStreams();

   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );


   //INDI Interface
protected:

   pcf::IndiProperty m_indiP_psdTime;
   INDI_NEWCALLBACK_DECL(modalPSDs, m_indiP_psdTime);

   pcf::IndiProperty m_indiP_psdAvgTime;
   INDI_NEWCALLBACK_DECL(modalPSDs, m_indiP_psdAvgTime);

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
   config.add("circBuff.defaultFPS", "", "circBuff.defaultFPS", argType::Required, "circBuff", "defaultFPS", false, "realT", "Default FPS at startup, will enable changing average length with psdTime before INDI available.");
   config.add("circBuff.psdTime", "", "circBuff.psdTime", argType::Required, "circBuff", "psdTime", false, "realT", "The length of time over which to calculate PSDs.  The default is 1 sec.");
}

int modalPSDs::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);
   
   _config(m_fpsSource, "circBuff.fpsSource");
   _config(m_fps, "circBuff.defaultFPS");
   _config(m_psdTime, "circBuff.psdTime");

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

   createStandardIndiNumber<unsigned>( m_indiP_psdTime, "psdAvgTime", 0, 60, 0.1, "%0.1f");
   m_indiP_psdTime["current"].set(m_psdAvgTime);
   m_indiP_psdTime["target"].set(m_psdAvgTime);
   
   if( registerIndiPropertyNew( m_indiP_psdAvgTime, INDI_NEWCALLBACK(m_indiP_psdAvgTime)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(m_fpsSource != "")
   {
      REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
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
      //m_tsCircBuffLength = m_fps * m_psdTime * m_overSize;
      m_tsSize = m_fps*m_psdTime;
      m_tsOverlapSize = m_tsSize * m_psdOverlapFraction;
   }

   if(m_tsOverlapSize <= 0 || !std::isnormal(m_tsOverlapSize))
   {
      log<software_error>({__FILE__,__LINE__, "bad m_tsOverlapSize value: " + std::to_string(m_tsOverlapSize)});
      return -1;
   }

   //Check for unsupported type (must be realT)
   if(shmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT)
   {
      //must be a vector of size 1 on one axis
      log<software_error>({__FILE__,__LINE__, "unsupported data type: must be realT"});
      return -1;
   }

   //Check for unexpected format
   if(shmimMonitorT::m_width != 1 && shmimMonitorT::m_height != 1)
   {
      //must be a vector of size 1 on one axis
      log<software_error>({__FILE__,__LINE__, "unexpected shmim format"});
      return -1;
   }

   std::cerr << "connected to " << shmimMonitorT::m_shmimName << " " << shmimMonitorT::m_width << " " << shmimMonitorT::m_height << " " << shmimMonitorT::m_depth << "\n";


   m_nModes = shmimMonitorT::m_width*shmimMonitorT::m_height;

   //Size the circ buff
   m_ampCircBuff.maxEntries(shmimMonitorT::m_depth);
   
   //Create the window
   m_win.resize(m_tsSize);
   mx::sigproc::window::hann(m_win);

   //Set up the FFT and working memory
   m_fft.plan(m_tsSize, MXFFT_FORWARD, false);
   
   if(m_tsWork) fftw_free(m_tsWork);
   m_tsWork = mx::math::fft::fftw_malloc<realT>( m_tsSize );
   
   if(m_fftWork) fftw_free(m_fftWork);
   m_fftWork = mx::math::fft::fftw_malloc<std::complex<realT>>( (m_tsSize/2 + 1) );

   m_psd.resize(m_tsSize/2 + 1);

   if(m_fps > 0)
   {
      m_df = 1.0/(m_tsSize/m_fps);
   }
   else
   {
      m_df = 1.0/(m_tsSize);
   }


   //Create the shared memory images
   uint32_t imsize[3];

   //First the frequency
   imsize[0] = 1;
   imsize[1] = m_psd.size();
   imsize[2] = 1;

   if(m_freqStream)
   {
      ImageStreamIO_destroyIm(m_freqStream);
      free(m_freqStream);
   }
   m_freqStream = (IMAGE *) malloc(sizeof(IMAGE));

   ImageStreamIO_createIm_gpu(m_freqStream, (m_configName + "_freq").c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
   m_freqStream->md->write=1;
   for(size_t n = 0; n < m_psd.size(); ++n)
   {
      m_freqStream->array.F[n] = n * m_df;
   }

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_freqStream->md->writetime);
   m_freqStream->md->atime = m_freqStream->md->writetime;

   //Update cnt1
   m_freqStream->md->cnt1 = 0;

   //Update cnt0
   m_freqStream->md->cnt0 = 0;

   m_freqStream->md->write=0;
   ImageStreamIO_sempost(m_freqStream,-1);


   allocatePSDStreams();

   m_psdRestarting = false;



   return 0;
}
   
int modalPSDs::allocatePSDStreams()
{
   if(m_rawpsdStream)
   {
      ImageStreamIO_destroyIm( m_rawpsdStream );
      free(m_rawpsdStream);
   }

   uint32_t imsize[3];
   imsize[0] = m_psd.size();
   imsize[1] = m_nModes;
   imsize[2] = m_nPSDHistory;

   m_rawpsdStream = (IMAGE *) malloc(sizeof(IMAGE));
   ImageStreamIO_createIm_gpu(m_rawpsdStream, (m_configName + "_rawpsds").c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
   
   if(m_avgpsdStream)
   {
      ImageStreamIO_destroyIm( m_avgpsdStream );
      free(m_avgpsdStream);
   }

   imsize[0] = m_psd.size();
   imsize[1] = m_nModes;
   imsize[2] = 1;

   m_avgpsdStream = (IMAGE *) malloc(sizeof(IMAGE));
   ImageStreamIO_createIm_gpu(m_avgpsdStream, (m_configName + "_psds").c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
   
   m_psdBuffer.resize(m_psd.size(), m_nModes);

   return 0;
}

int modalPSDs::processImage( void * curr_src,
                             const dev::shmimT & dummy
                           )
{
   static_cast<void>(dummy);

   float * f_src = static_cast<float *>(curr_src);

   m_ampCircBuff.nextEntry(f_src);

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
      if(m_psdRestarting == true || m_ampCircBuff.maxEntries() == 0) m_psdWaiting = true;

      while((m_psdRestarting == true || m_ampCircBuff.maxEntries() == 0) && !shutdown()) mx::sys::microSleep(100);

      if(shutdown()) break;

      m_psdWaiting = false;

      if(m_ampCircBuff.maxEntries() == 0)
      {
         log<software_error>({__FILE__, __LINE__, "amp circ buff has zero size"});
         return;
      }

      std::cerr << "waiting to grow\n";
      while( m_ampCircBuff.size() < m_ampCircBuff.maxEntries()  && m_psdRestarting == false && !shutdown())
      {
         //shrinking sleep
         double stime = (1.0*m_ampCircBuff.maxEntries() - 1.0*m_ampCircBuff.size())/m_fps * 0.5*1e9;
         mx::sys::nanoSleep(stime);
      }

      std::cerr << "all grown.  starting to calculate\n";

      ampCircBuffT::indexT ne0;
      ampCircBuffT::indexT ne1 = m_ampCircBuff.latest(); 
      if(ne1 > m_tsOverlapSize) ne1 -= m_tsSize;
      else ne1 = m_ampCircBuff.size() + ne1 - m_tsSize;

      while(m_psdRestarting == false && !shutdown())
      {
         //Used to check if we are getting too behind
         uint64_t mono0 = m_ampCircBuff.mono();

         //Calc PSDs here
         ne0 = ne1;

         std::cerr << "calculating: " << ne0 << " " << m_ampCircBuff.size() << " " << m_tsSize << "\n";
         double t0 = mx::sys::get_curr_time();

         for(size_t m = 0; m < shmimMonitorT::m_width*shmimMonitorT::m_height; ++m) //Loop over each mode
         {
            //get mean going over entire TS
            float mn = 0;
            for(size_t n =0; n < m_ampCircBuff.size(); ++n)
            {
               realT mn = 0;
               mn += m_ampCircBuff[n][m];
            }
            mn /= m_ampCircBuff.size();
            
            double var = 0;
            for(size_t n = 0; n < m_tsSize; ++n)
            {
               m_tsWork[n] = (m_ampCircBuff.at(ne0,n)[m]-mn);
               var += pow(m_tsWork[n],2);
               
               m_tsWork[n] *= m_win[n];
            }
            var /= m_tsSize;

            m_fft( m_fftWork, m_tsWork);

            double nm = 0;
            for(size_t n=0; n < m_psd.size(); ++n) 
            {
               m_psd[n] = norm(m_fftWork[n]);
               nm += m_psd[n] * m_df;
            }

            for(size_t n=0; n < m_psd.size(); ++n) 
            {
               m_psd[n] *= (var/nm);
            }

            //Put it in the buffer for uploading to shmim
            for(size_t n=0; n < m_psd.size(); ++n) m_psdBuffer(n,m) = m_psd[n];

         }

         //------------------------- the raw psds ---------------------------
         m_rawpsdStream->md->write=1;
      
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &m_rawpsdStream->md->writetime);
         m_rawpsdStream->md->atime = m_rawpsdStream->md->writetime;

         uint64_t cnt1 = m_rawpsdStream->md->cnt1 + 1;
         if(cnt1 >= m_rawpsdStream->md->size[2]) cnt1 = 0;

         //Move to next pointer
         float * F = m_rawpsdStream->array.F + m_psdBuffer.rows()*m_psdBuffer.cols()*cnt1;

         memcpy(F, m_psdBuffer.data(), m_psdBuffer.rows()*m_psdBuffer.cols()*sizeof(float));

         //Update cnt1
         m_rawpsdStream->md->cnt1 = cnt1;

         //Update cnt0
         ++m_rawpsdStream->md->cnt0;

         m_rawpsdStream->md->write=0;
         ImageStreamIO_sempost(m_rawpsdStream,-1);

         //-------------------------- now average the psds ----------------------------

         int nPSDAverage = (m_psdAvgTime/m_psdTime) / m_psdOverlapFraction;

         if(nPSDAverage <= 0) nPSDAverage = 1;
         else if((uint64_t) nPSDAverage > m_rawpsdStream->md->size[2]) nPSDAverage = m_rawpsdStream->md->size[2];

         //Move to next pointer
         F = m_rawpsdStream->array.F + m_psdBuffer.rows()*m_psdBuffer.cols()*cnt1;

         memcpy(m_psdBuffer.data(), F, m_psdBuffer.rows()*m_psdBuffer.cols()*sizeof(float));

         for(int n =1; n < nPSDAverage; ++n)
         {
            if(cnt1 == 0) cnt1 = m_rawpsdStream->md->size[2] - 1;
            else --cnt1;

            F = m_rawpsdStream->array.F + m_psdBuffer.rows()*m_psdBuffer.cols()*cnt1;

            m_psdBuffer += Eigen::Map<Eigen::Array<float,-1,-1>>(F, m_psdBuffer.rows(), m_psdBuffer.cols());
         }

         m_psdBuffer /= nPSDAverage;

         m_avgpsdStream->md->write=1;
      
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &m_avgpsdStream->md->writetime);
         m_avgpsdStream->md->atime = m_avgpsdStream->md->writetime;

         //Move to next pointer
         F = m_avgpsdStream->array.F;

         memcpy(F, m_psdBuffer.data(), m_psdBuffer.rows()*m_psdBuffer.cols()*sizeof(float));

         //Update cnt1
         m_avgpsdStream->md->cnt1 = 0;

         //Update cnt0
         ++m_avgpsdStream->md->cnt0;

         m_avgpsdStream->md->write=0;
         ImageStreamIO_sempost(m_avgpsdStream,-1);

         double t1 = mx::sys::get_curr_time();
         std::cerr << "done " << t1-t0 << "\n";

         //Have to be cycling within the overlap
         if(m_ampCircBuff.mono() - mono0 >= m_tsOverlapSize)
         {
            log<text_log>("PSD calculations getting behind, skipping ahead.", logPrio::LOG_WARNING);
            ne0 = m_ampCircBuff.latest();
            if(ne0 > m_tsOverlapSize) ne0 -= m_tsOverlapSize;
            else ne0 = m_ampCircBuff.size() + ne0 - m_tsOverlapSize;
         }

         //Now wait until we get to next one
         ne1 = ne0 + m_tsOverlapSize;
         if( ne1 >= m_ampCircBuff.size()) 
         {
            ne1 -= m_ampCircBuff.size();
         }

         ampCircBuffT::indexT ce = m_ampCircBuff.latest(); 
         //wrapped difference
         long dn;
         if( ce >= ne1 ) dn = ce - ne1;
         else dn = ce + (m_ampCircBuff.size() - ne1);

         while(dn < m_tsOverlapSize && !shutdown() && m_psdRestarting == false)
         {
            double stime = (1.0*dn)/m_fps * 0.5 * 1e9;
            mx::sys::nanoSleep(stime);

            ce = m_ampCircBuff.latest();

            if( ce >= ne1 ) dn = ce - ne1;
            else dn = ce + (m_ampCircBuff.size() - ne1);
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
   
   realT target;
   
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


INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdAvgTime)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_psdAvgTime.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   realT target;
   
   if( indiTargetUpdate( m_indiP_psdAvgTime, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(m_psdAvgTime != target)
   {   
      std::lock_guard<std::mutex> guard(m_indiMutex);

      m_psdAvgTime = target;
   
      updateIfChanged(m_indiP_psdTime, "current", m_psdAvgTime, INDI_IDLE);
      updateIfChanged(m_indiP_psdTime, "target", m_psdAvgTime, INDI_IDLE);

      log<text_log>("set psdAvgTime to " + std::to_string(m_psdAvgTime), logPrio::LOG_NOTICE);
   }

   return 0;
} //INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdTime)

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

   realT fps = ipRecv["current"].get<realT>();
   
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
