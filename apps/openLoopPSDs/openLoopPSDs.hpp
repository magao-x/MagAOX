/** \file openLoopPSDs.hpp
  * \brief The MagAO-X openLoopPSDs app header file
  *
  * \ingroup openLoopPSDs_files
  */

#ifndef openLoopPSDs_hpp
#define openLoopPSDs_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/sigproc/circularBuffer.hpp>
#include <mx/sigproc/signalWindows.hpp>

#include <mx/math/fft/fftwEnvironment.hpp>
#include <mx/math/fft/fft.hpp>

/** \defgroup openLoopPSDs
  * \brief An application to calculate rolling PSDs of modal amplitudes
  *
  * <a href="../handbook/operating/software/apps/openLoopPSDs.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup openLoopPSDs_files
  * \ingroup openLoopPSDs
  */

namespace MagAOX
{
namespace app
{

/// Class for application to calculate rolling PSDs of modal amplitudes.
/** 
  * \ingroup openLoopPSDs
  */
class openLoopPSDs : public MagAOXApp<true>, public dev::shmimMonitor<openLoopPSDs>
{
   friend class dev::shmimMonitor<openLoopPSDs>;

public:

   typedef float realT;
   typedef std::complex<realT> complexT;

   /// The base shmimMonitor type
   typedef dev::shmimMonitor<openLoopPSDs> shmimMonitorT;

   /// The amplitude circular buffer type
   typedef mx::sigproc::circularBufferIndex<float *, unsigned> ampCircBuffT;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_clPSDSource; ///< Device name for getting the C.L. PSDs.  This is used as the shmim name (unless overriden) and this INDI device should have *.fps.current.

   int m_nPSDHistory {100}; //
   
   ///@}

   int m_nModes; ///< the number of modes to calculate PSDs for.

   realT m_fps {0};
   realT m_df {1};
   
   std::vector<realT> m_freq;

   IMAGE * m_olpsdStream {nullptr};

public:
   /// Default c'tor.
   openLoopPSDs();

   /// D'tor, declared and defined for noexcept.
   ~openLoopPSDs() noexcept
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

   /// Implementation of the FSM for openLoopPSDs.
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

   pcf::IndiProperty m_indiP_clPSDSource;
   INDI_SETCALLBACK_DECL(openLoopPSDs, m_indiP_clPSDSource);

   pcf::IndiProperty m_indiP_fps;

};

openLoopPSDs::openLoopPSDs() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void openLoopPSDs::setupConfig()
{
   shmimMonitorT::setupConfig(config);

   config.add("psds.clPSDSource", "", "psds.clPSDSource", argType::Required, "psds", "clPSDSource", false, "string", "Device name for getting the C.L. PSDs.  This is used as the shmim name (unless overriden) and this INDI device should have *.fps.current.");
}

int openLoopPSDs::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_clPSDSource, "psds.clPSDSource");

   shmimMonitorT::m_shmimName = m_clPSDSource + "_psds";
   shmimMonitorT::loadConfig(_config);
   
   
   return 0;
}

void openLoopPSDs::loadConfig()
{
   loadConfigImpl(config);
}

int openLoopPSDs::appStartup()
{
   if(m_clPSDSource != "")
   {
      REG_INDI_SETPROP(m_indiP_clPSDSource, m_clPSDSource, std::string("fps"));
   }
   else
   {
      log<text_log>("must specify psds.clPSDSource\n", logPrio::LOG_ERROR);
      m_shutdown = 1;
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

   state(stateCodes::OPERATING);

   return 0;
}

int openLoopPSDs::appLogic()
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

int openLoopPSDs::appShutdown()
{
   shmimMonitorT::appShutdown();

   return 0;
}

int openLoopPSDs::allocate( const dev::shmimT & dummy)
{
   static_cast<void>(dummy);

   //Create the shared memory images
   uint32_t imsize[3];

   //First the frequency
   imsize[0] = shmimMonitorT::m_width;
   imsize[1] = shmimMonitorT::m_height;
   imsize[2] = 1;

   if(m_olpsdStream)
   {
      ImageStreamIO_destroyIm(m_olpsdStream);
      free(m_olpsdStream);
   }
   m_olpsdStream = (IMAGE *) malloc(sizeof(IMAGE));

   ImageStreamIO_createIm_gpu(m_olpsdStream, (m_configName + "_olpsds").c_str(), 3, imsize, IMAGESTRUCT_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
   
   return 0;
}
   
int openLoopPSDs::processImage( void * curr_src,
                             const dev::shmimT & dummy
                           )
{
   static_cast<void>(dummy);

   float * src = static_cast<float *>(curr_src);

   m_olpsdStream->md->write=1;

   for(uint32_t n = 0; n < shmimMonitorT::m_height; ++n)
   {
      for(uint32_t f = 0; f < shmimMonitorT::m_width; ++f)
      {
         m_olpsdStream->array.F[n*shmimMonitorT::m_width + f] = src[n*shmimMonitorT::m_width + f]; // <- divide by t.f. here.
      }
   }

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &m_olpsdStream->md->writetime);
   m_olpsdStream->md->atime = m_olpsdStream->md->writetime;

   //Update cnt1
   m_olpsdStream->md->cnt1 = 0;

   //Update cnt0
   m_olpsdStream->md->cnt0 = 0;

   m_olpsdStream->md->write=0;
   ImageStreamIO_sempost(m_olpsdStream,-1);

   return 0;
}

INDI_SETCALLBACK_DEFN(openLoopPSDs, m_indiP_clPSDSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_clPSDSource.getName())
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

} //INDI_SETCALLBACK_DEFN(openLoopPSDs, m_indiP_clPSDSource)

} //namespace app
} //namespace MagAOX

#endif //openLoopPSDs_hpp
