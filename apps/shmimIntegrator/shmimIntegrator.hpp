/** \file shmimIntegrator.hpp
  * \brief The MagAO-X generic ImageStreamIO stream integrator
  *
  * \ingroup app_files
  */

#ifndef shmimIntegrator_hpp
#define shmimIntegrator_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

struct darkShmimT 
{
   static std::string configSection()
   {
      return "darkShmim";
   };
   
   static std::string indiPrefix()
   {
      return "dark";
   };
};

struct dark2ShmimT 
{
   static std::string configSection()
   {
      return "dark2Shmim";
   };
   
   static std::string indiPrefix()
   {
      return "dark2";
   };
};

/** \defgroup shmimIntegrator ImageStreamIO Stream Integrator
  * \brief Integrates (i.e. averages) an ImageStreamIO image stream.
  *
  * <a href="../handbook/operating/software/apps/shmimIntegrator.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup shmimIntegrator_files ImageStreamIO Stream Integrator
  * \ingroup shmimIntegrator
  */

/** MagAO-X application to control integrating (averaging) an ImageStreamIO stream
  *
  * \ingroup shmimIntegrator
  * 
  */
class shmimIntegrator : public MagAOXApp<true>, public dev::shmimMonitor<shmimIntegrator>, public dev::shmimMonitor<shmimIntegrator,darkShmimT>, public dev::shmimMonitor<shmimIntegrator,dark2ShmimT>, public dev::frameGrabber<shmimIntegrator>
{

   //Give the test harness access.
   friend class shmimIntegrator_test;

   friend class dev::shmimMonitor<shmimIntegrator>;
   friend class dev::shmimMonitor<shmimIntegrator,darkShmimT>;
   friend class dev::shmimMonitor<shmimIntegrator,dark2ShmimT>;
   friend class dev::frameGrabber<shmimIntegrator>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<shmimIntegrator> shmimMonitorT;
   
   //The dark shmimMonitor type
   typedef dev::shmimMonitor<shmimIntegrator, darkShmimT> darkMonitorT;
   
   //The dark shmimMonitor type for a 2nd dark
   typedef dev::shmimMonitor<shmimIntegrator, dark2ShmimT> dark2MonitorT;
   
   //The base frameGrabber type
   typedef dev::frameGrabber<shmimIntegrator> frameGrabberT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
public:
   /** \name app::dev Configurations
     *@{
     */
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber these images can not be flipped
   
   ///@}
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   unsigned m_nAverageDefault {10}; ///< The number of frames to average.  Default 10.

   std::string m_fpsSource; ///< Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.

   float m_avgTime {0}; ///< If non zero, then m_nAverage adjusts automatically to keep a constant averaging time [sec].  Default 0.

   unsigned m_nUpdate {0}; ///< The rate at which to update the average.  If 0 < m_nUpdate < m_nAverage then this is a moving averager. Default 0.
   
   bool m_continuous {true}; ///< Set to false in configuration to have this run once then stop until triggered.

   bool m_running {true}; ///< Set to false in configuration to have it not start averaging until triggered.   
   
   std::string m_stateSource; ///< The source of the state string used for file management

   bool m_fileSaver {false}; ///< Set to true in configuration to have this save and reload files automatically.

   ///@}

   mx::improc::eigenCube<realT> m_accumImages; ///< Cube used to accumulate images
   
   mx::improc::eigenImage<realT> m_avgImage; ///< The average image.

   unsigned m_nAverage {10};

   float m_fps {0}; ///< Current FPS from the FPS source.

   unsigned m_nprocessed {0};
   size_t m_currImage {0};
   size_t m_sinceUpdate {0};
   bool m_updated {false};
   
   bool m_imageValid {false};
   std::string m_stateString;
   bool m_stateStringValid {false};
   bool m_stateStringValidOnStart {false};
   bool m_stateStringChanged {false};
   std::string m_fileSaveDir;


   sem_t m_smSemaphore {0}; ///< Semaphore used to synchronize the fg thread and the sm thread.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   mx::improc::eigenImage<realT> m_darkImage;
   bool m_darkSet {false};
   realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   mx::improc::eigenImage<realT> m_dark2Image;
   bool m_dark2Set {false};
   realT (*dark2_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   
public:
   /// Default c'tor.
   shmimIntegrator();

   /// D'tor, declared and defined for noexcept.
   ~shmimIntegrator() noexcept
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

   /// Implementation of the FSM for shmimIntegrator.
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

protected:

   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int allocate( const darkShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const darkShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int allocate( const dark2ShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dark2ShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );

   int findMatchingDark();


   /** \name dev::frameGrabber interface
     *
     * @{
     */
   
   /// Implementation of the framegrabber configureAcquisition interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int configureAcquisition();
   
   /// Implementation of the framegrabber fps interface
   /**
     * \todo this needs to infer the stream fps and return it
     */  
   float fps()
   {
      return 1.0;
   }
   
   /// Implementation of the framegrabber startAcquisition interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int startAcquisition();
   
   /// Implementation of the framegrabber acquireAndCheckValid interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int acquireAndCheckValid();
   
   /// Implementation of the framegrabber loadImageIntoStream interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int loadImageIntoStream( void * dest  /**< [in] */);
   
   /// Implementation of the framegrabber reconfig interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int reconfig();
   
   ///@}
   
   pcf::IndiProperty m_indiP_nAverage;
   
   pcf::IndiProperty m_indiP_avgTime;

   pcf::IndiProperty m_indiP_nUpdate;
   
   pcf::IndiProperty m_indiP_startAveraging;
   
   INDI_NEWCALLBACK_DECL(shmimIntegrator, m_indiP_nAverage);
   INDI_NEWCALLBACK_DECL(shmimIntegrator, m_indiP_avgTime);
   INDI_NEWCALLBACK_DECL(shmimIntegrator, m_indiP_nUpdate);
   INDI_NEWCALLBACK_DECL(shmimIntegrator, m_indiP_startAveraging);

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(shmimIntegrator, m_indiP_fpsSource);

   pcf::IndiProperty m_indiP_stateSource;
   INDI_SETCALLBACK_DECL(shmimIntegrator, m_indiP_stateSource);

   pcf::IndiProperty m_indiP_imageValid;
   
};

inline
shmimIntegrator::shmimIntegrator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkMonitorT::m_getExistingFirst = true;
   return;
}

inline
void shmimIntegrator::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkMonitorT::setupConfig(config);
   dark2MonitorT::setupConfig(config);
   
   frameGrabberT::setupConfig(config);
   
   config.add("integrator.nAverage", "", "integrator.nAverage", argType::Required, "integrator", "nAverage", false, "unsigned", "The default number of frames to average.  Default 10. Can be changed via INDI.");
   config.add("integrator.fpsSource", "", "integrator.fpsSource", argType::Required, "integrator", "fpsSource", false, "string", "///< Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.");

   config.add("integrator.avgTime", "", "integrator.avgTime", argType::Required, "integrator", "avgTime", false, "float", "///< If non zero, then m_nAverage adjusts automatically to keep a constant averaging time [sec].  Default 0. Can be changed via INDI.");

   config.add("integrator.nUpdate", "", "integrator.nUpdate", argType::Required, "integrator", "nUpdate", false, "unsigned", "The rate at which to update the average.  If 0 < m_nUpdate < m_nAverage then this is a moving averager. Default 0.  If 0, then it is a simple average.");
   
   config.add("integrator.continuous", "", "integrator.continuous", argType::Required, "integrator", "continuous", false, "bool", "Flag controlling whether averaging is continuous or only when triggered.  Default true.");
   config.add("integrator.running", "", "integrator.running", argType::Required, "integrator", "running", false, "bool", "Flag controlling whether averaging is running at startup.  Default true.");

   config.add("integrator.stateSource", "", "integrator.stateSource", argType::Required, "integrator", "stateSource", false, "string", "///< Device name for getting the state string for file management.  This device should have *.state_string.current.");
   config.add("integrator.fileSaver", "", "integrator.fileSaver", argType::Required, "integrator", "fileSaver", false, "bool", "Flag controlling whether this saves and reloads files automatically.  Default false.");
}

inline
int shmimIntegrator::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(config);
   darkMonitorT::loadConfig(config);
   dark2MonitorT::loadConfig(config);
   
   frameGrabberT::loadConfig(config);
   
   _config(m_nAverageDefault, "integrator.nAverage");
   m_nAverage=m_nAverageDefault;
   _config(m_fpsSource, "integrator.fpsSource");
   _config(m_avgTime, "integrator.avgTime");
   _config(m_nUpdate, "integrator.nUpdate");
   
   _config(m_continuous, "integrator.continuous");
   
   _config(m_running, "integrator.running");
   
   _config(m_stateSource, "integrator.stateSource");
   _config(m_fileSaver, "integrator.fileSaver");

   return 0;
}

inline
void shmimIntegrator::loadConfig()
{
   loadConfigImpl(config);
}

inline
int shmimIntegrator::appStartup()
{
   
   createStandardIndiNumber<unsigned>( m_indiP_nAverage, "nAverage", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_nAverage["current"].set(m_nAverage);
   m_indiP_nAverage["target"].set(m_nAverage);
   
   if( registerIndiPropertyNew( m_indiP_nAverage, INDI_NEWCALLBACK(m_indiP_nAverage)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<float>( m_indiP_avgTime, "avgTime", 0, std::numeric_limits<float>::max(),0 , "%0.1f");
   m_indiP_avgTime["current"].set(m_avgTime);
   m_indiP_avgTime["target"].set(m_avgTime);
   
   if( registerIndiPropertyNew( m_indiP_avgTime, INDI_NEWCALLBACK(m_indiP_avgTime)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiNumber<unsigned>( m_indiP_nUpdate, "nUpdate", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_nUpdate["current"].set(m_nUpdate);
   m_indiP_nUpdate["target"].set(m_nUpdate);
   
   if( registerIndiPropertyNew( m_indiP_nUpdate, INDI_NEWCALLBACK(m_indiP_nUpdate)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiToggleSw( m_indiP_startAveraging, "start");
   if( registerIndiPropertyNew( m_indiP_startAveraging, INDI_NEWCALLBACK(m_indiP_startAveraging)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   if(m_fpsSource != "")
   {
      REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
   }

   if(m_fileSaver == true && m_stateSource != "")
   {
      REG_INDI_SETPROP(m_indiP_stateSource, m_stateSource, std::string("state_string"));
      
      createROIndiText( m_indiP_imageValid, "image_valid", "flag", "Image Valid", "Image", "Valid");
      if(!m_imageValid) //making sure we stay up with default
      {
         m_indiP_imageValid["flag"] = "no";
      }
      else
      {
         m_indiP_imageValid["flag"] = "yes";
      }

      if( registerIndiPropertyReadOnly( m_indiP_imageValid ) < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         return -1;
      }


      m_fileSaveDir = m_calibDir + "/" + m_configName;

      //Create save directory.
      errno = 0;
      if( mkdir(m_fileSaveDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
      {
         if( errno != EEXIST)
         {
            std::stringstream logss;
            logss << "Failed to create image directory (" << m_fileSaveDir << ").  Errno says: " << strerror(errno);
            log<software_critical>({__FILE__, __LINE__, errno, 0, logss.str()});

            return -1;
         }
      }
   }


   if(sem_init(&m_smSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.M. semaphore"});
      return -1;
   }

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
  
   if(darkMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(dark2MonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   state(stateCodes::READY);
    
   return 0;
}

inline
int shmimIntegrator::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( darkMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( dark2MonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(darkMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(dark2MonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
      
   if(m_running == false)
   {
      state(stateCodes::READY);
      updateSwitchIfChanged(m_indiP_startAveraging, "toggle", pcf::IndiElement::Off, INDI_IDLE);

      if(m_fileSaver)
      {
         if(m_stateStringChanged) //So if not running and the state has changed, we check
         {
            if(findMatchingDark() < 0)
            {
               log<software_error>({__FILE__, __LINE__});
            }
         }

         if(!m_imageValid)
         {
            updateIfChanged(m_indiP_imageValid, "flag", "no");
         }
         else
         {
            updateIfChanged(m_indiP_imageValid, "flag", "yes");
         }
      }
   }
   else
   {
      state(stateCodes::OPERATING);
      updateSwitchIfChanged(m_indiP_startAveraging, "toggle", pcf::IndiElement::On, INDI_BUSY);
   }

   updateIfChanged(m_indiP_nAverage, "current", m_nAverage, INDI_IDLE);
   updateIfChanged(m_indiP_nAverage, "target", m_nAverage, INDI_IDLE);
   
   updateIfChanged(m_indiP_avgTime, "current", m_avgTime, INDI_IDLE);
   updateIfChanged(m_indiP_avgTime, "target", m_avgTime, INDI_IDLE);

   updateIfChanged(m_indiP_nUpdate, "current", m_nUpdate, INDI_IDLE);
   updateIfChanged(m_indiP_nUpdate, "target", m_nUpdate, INDI_IDLE);
   
   return 0;
}

inline
int shmimIntegrator::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   darkMonitorT::appShutdown();
   
   dark2MonitorT::appShutdown();
   
   frameGrabberT::appShutdown();
   
   return 0;
}

inline
int shmimIntegrator::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
  
   if(m_avgTime > 0 && m_fps > 0)
   {
      m_nAverage = m_avgTime * m_fps;
      log<text_log>("set nAverage to " + std::to_string(m_nAverage) + " based on FPS", logPrio::LOG_NOTICE);
   }
   else if(m_fps == 0) //Haven't gotten the update yet so we keep going for now
   {
      if(m_nAverage != m_nAverageDefault)
      {
         m_nAverage = m_nAverageDefault;
         log<text_log>("set nAverage to default " + std::to_string(m_nAverage), logPrio::LOG_NOTICE);
      }
   }

   if(m_nUpdate > 0)
   {
      m_accumImages.resize(shmimMonitorT::m_width, shmimMonitorT::m_height, m_nAverage);
      m_accumImages.setZero();
   }
   else
   {
      m_accumImages.resize(1,1,1);
   }
   
   m_nprocessed = 0;
   m_currImage = 0;
   m_sinceUpdate = 0;
   
   m_avgImage.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   //m_avgImage.setZero();
   
   pixget = getPixPointer<realT>(shmimMonitorT::m_dataType);
   
   if(pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   updateIfChanged(m_indiP_nAverage, "current", m_nAverage, INDI_IDLE);
   updateIfChanged(m_indiP_nAverage, "target", m_nAverage, INDI_IDLE);
   
   updateIfChanged(m_indiP_avgTime, "current", m_avgTime, INDI_IDLE);
   updateIfChanged(m_indiP_avgTime, "target", m_avgTime, INDI_IDLE);

   updateIfChanged(m_indiP_nUpdate, "current", m_nUpdate, INDI_IDLE);
   updateIfChanged(m_indiP_nUpdate, "target", m_nUpdate, INDI_IDLE);
   
   m_reconfig = true;
   
   return 0;
}

inline
int shmimIntegrator::processImage( void * curr_src, 
                                   const dev::shmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused
   
   if(!m_running) return 0;
   
   if(m_nUpdate == 0)
   {
      if(m_updated) return 0;
      if(m_sinceUpdate == 0) m_avgImage.setZero();
      
      realT * data = m_avgImage.data();
      
      for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
      {
         data[nn] += pixget(curr_src, nn);
      }
      ++m_sinceUpdate;
      if(m_sinceUpdate >= m_nAverage)
      {
         m_avgImage /= m_nAverage;
         
         if(m_darkSet && !m_dark2Set) m_avgImage -= m_darkImage;
         else if(!m_darkSet && m_dark2Set) m_avgImage -= m_dark2Image;
         else if(m_darkSet && m_dark2Set) m_avgImage -= m_darkImage + m_dark2Image;
         
         m_updated = true;
         
         //Now tell the f.g. to get going
         if(sem_post(&m_smSemaphore) < 0)
         {
            log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            return -1;
         }
         
         m_sinceUpdate = 0;
         if(!m_continuous) 
         {
            m_running = false;
            if(m_fileSaver) 
            {
               if(m_stateStringChanged || !m_stateStringValid || !m_stateStringValidOnStart)
               {
                  m_imageValid = false;
                  log<text_log>("state changed during acquisition, not saving", logPrio::LOG_NOTICE);
               }
               else
               {
                  m_imageValid = true;
                  m_stateStringChanged=false;

                  ///\todo this should happen in a different less-real-time thread.
                  //Otherwise we save:
                  timespec fts;
                  clock_gettime(CLOCK_REALTIME, &fts);
         
                  tm uttime;//The broken down time.   
        
                  if(gmtime_r(&fts.tv_sec, &uttime) == 0)
                  {
                     //Yell at operator but keep going
                     log<software_alert>({__FILE__,__LINE__,errno,0,"gmtime_r error.  possible loss of timing information."}); 
                  }
   
                  char cts[] = "YYYYMMDDHHMMSSNNNNNNNNN";
                  int rv = snprintf(cts, sizeof(cts), "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year+1900, 
                               uttime.tm_mon+1, uttime.tm_mday, uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(fts.tv_nsec));
      
                  if(rv != sizeof(cts)-1) 
                  {
                     //Something is very wrong.  Keep going to try to get it on disk.
                     log<software_alert>({__FILE__,__LINE__, errno, rv, "did not write enough chars to timestamp"}); 
                  }
   
                  std::string fname = m_fileSaveDir + "/" + m_configName + "_" + m_stateString + "__T" + cts + ".fits";  
                  
                  mx::fits::fitsFile<float> ff;
                  ff.write(fname, m_avgImage);
                  log<text_log>("Wrote " + fname);

               }   
            }
         }
      }
   }
   else
   {
      realT * data = m_accumImages.image(m_currImage).data();
      
      for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
      {
         data[nn] = pixget(curr_src, nn);
      }
      ++m_nprocessed;
      ++m_currImage;
      if(m_currImage >= m_nAverage) m_currImage = 0;
         
      if(m_nprocessed < m_nAverage) //Check that we are burned in on first pass through cube
      {
         return 0;
      }
      
      ++m_sinceUpdate;
      
      if(m_sinceUpdate >= m_nUpdate)
      {
         if(m_updated)
         {
            return 0; //In case f.g. thread is behind, we skip and come back.
         }
         //Don't use eigenCube functions to avoid any omp 
         m_avgImage.setZero();
         for(size_t n =0; n < m_nAverage; ++n)
         {
            for(size_t ii=0; ii< shmimMonitorT::m_width; ++ii)
            {
               for(size_t jj=0; jj< shmimMonitorT::m_height; ++jj)
               {
                  m_avgImage(ii,jj) += m_accumImages.image(n)(ii,jj);
               }
            }
         }
         m_avgImage /= m_nAverage;
         
         if(m_darkSet) m_avgImage -= m_darkImage;
         
         m_updated = true;
         
         //Now tell the f.g. to get going
         if(sem_post(&m_smSemaphore) < 0)
         {
            log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            return -1;
         }
            
         m_sinceUpdate = 0;
      }
   }
   return 0;
}

inline
int shmimIntegrator::allocate(const darkShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height)
   {
      m_darkSet = false;
      darkMonitorT::m_restart = true;
   }
   
   m_darkImage.resize(darkMonitorT::m_width, darkMonitorT::m_height);
   
   dark_pixget = getPixPointer<realT>(darkMonitorT::m_dataType);
   
   if(dark_pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   return 0;
}

inline
int shmimIntegrator::processImage( void * curr_src, 
                                   const darkShmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused
   
   realT * data = m_darkImage.data();
   
   for(unsigned nn=0; nn < darkMonitorT::m_width*darkMonitorT::m_height; ++nn)
   {
      //data[nn] = *( (int16_t * ) (curr_src + nn*shmimMonitorT::m_typeSize));
      data[nn] = dark_pixget(curr_src, nn);
   }
   
   m_darkSet = true;
   
   return 0;
}

inline
int shmimIntegrator::allocate(const dark2ShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(dark2MonitorT::m_width != shmimMonitorT::m_width || dark2MonitorT::m_height != shmimMonitorT::m_height)
   {
      m_dark2Set = false;
      dark2MonitorT::m_restart = true;
   }
   
   m_dark2Image.resize(dark2MonitorT::m_width, dark2MonitorT::m_height);
   
   dark2_pixget = getPixPointer<realT>(dark2MonitorT::m_dataType);
   
   if(dark2_pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   return 0;
}

inline
int shmimIntegrator::processImage( void * curr_src, 
                                   const dark2ShmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused
   
   realT * data = m_dark2Image.data();
   
   for(unsigned nn=0; nn < dark2MonitorT::m_width*dark2MonitorT::m_height; ++nn)
   {
      //data[nn] = *( (int16_t * ) (curr_src + nn*shmimMonitorT::m_typeSize));
      data[nn] = dark2_pixget(curr_src, nn);
   }
   
   m_dark2Set = true;
   
   return 0;
}

inline
int shmimIntegrator::findMatchingDark()
{
   std::vector<std::string> fnames = mx::ioutils::getFileNames(m_fileSaveDir, m_configName, "", ".fits");

   //getFileNames sorts, so these will be in oldest to newest order by lexical timestamp sort
   //So we search in reverse to always pick newest
   long N = fnames.size();
   for(long n = N-1; n >= 0; --n)
   {
      std::string fn = mx::ioutils::pathStem(fnames[n]);

      if(fn.size() < m_configName.size()+1) continue;

      size_t st = m_configName.size()+1;
      size_t ed = fn.find("__T");
      if(ed == std::string::npos || ed - st < 2) continue;
      std::string stateStr = fn.substr(st,ed-st);

      if(stateStr == m_stateString)
      {
         mx::fits::fitsFile<float> ff;
         ff.read(m_avgImage, fnames[n]);

         if(m_avgImage.rows() != shmimMonitorT::m_width || m_avgImage.cols() != shmimMonitorT::m_height)
         {
            //Means the camera has changed but stream hasn't caught up
            //(This happens on startup before stream connection completes.)  

            // And possibly that we haven't turned the shmimMonitor on yet by switching to OPERATING
            if( shmimMonitorT::m_width == 0 && shmimMonitorT::m_height == 0)
            {
               m_updated = true;
               shmimMonitorT::m_width = m_avgImage.rows();
               shmimMonitorT::m_height = m_avgImage.cols();
               m_reconfig = true;
               
            }
            else
            {
               m_imageValid = false;
               m_stateStringChanged = true; //So we let appLogic try again next time around.
               continue;
            }
         }

         //Now tell the f.g. to get going
         if(sem_post(&m_smSemaphore) < 0)
         {
            log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            return -1;
         }
         m_imageValid = true;
         m_stateStringChanged = false;
         log<text_log>("loaded last matching dark from disk", logPrio::LOG_NOTICE);
         return 0;
      }
   }

   m_imageValid = false;
   m_stateStringChanged = false; //stop trying b/c who else is going to add a dark?
   log<text_log>("dark is not valid", logPrio::LOG_WARNING);

   return 0;
}

inline
int shmimIntegrator::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(shmimMonitorT::m_width==0 || shmimMonitorT::m_height==0)
   {
      //This means we haven't connected to the stream to average. so wait.
      lock.unlock(); //don't hold the lock for a whole second.
      sleep(1);
      return -1;
   }
   
   frameGrabberT::m_width = shmimMonitorT::m_width;
   frameGrabberT::m_height = shmimMonitorT::m_height;
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;
   
   return 0;
}

inline
int shmimIntegrator::startAcquisition()
{
   return 0;
}

inline
int shmimIntegrator::acquireAndCheckValid()
{
   timespec ts;
         
   if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
      return -1;
   }
         
   ts.tv_sec += 1;
        
   if(sem_timedwait(&m_smSemaphore, &ts) == 0)
   {
      if( m_updated )
      {
         clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
         return 0;
      }
      else
      {
         return 1;
      }
   }
   else
   {
      return 1;
   }
}

inline
int shmimIntegrator::loadImageIntoStream(void * dest)
{
   memcpy(dest, m_avgImage.data(), shmimMonitorT::m_width*shmimMonitorT::m_height*frameGrabberT::m_typeSize  ); 
   m_updated = false;
   return 0;
}

inline
int shmimIntegrator::reconfig()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(shmimIntegrator, m_indiP_nAverage)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_nAverage.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_nAverage, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nAverage = target;

   if(m_avgTime > 0 && m_fps > 0)
   {
      m_avgTime = m_nAverage/m_fps;
   }
   
   updateIfChanged(m_indiP_nAverage, "current", m_nAverage, INDI_IDLE);
   updateIfChanged(m_indiP_nAverage, "target", m_nAverage, INDI_IDLE);

   updateIfChanged(m_indiP_avgTime, "current", m_avgTime, INDI_IDLE);
   updateIfChanged(m_indiP_avgTime, "target", m_avgTime, INDI_IDLE);
   
   shmimMonitorT::m_restart = true;
   
   log<text_log>("set nAverage to " + std::to_string(m_nAverage), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(shmimIntegrator, m_indiP_avgTime)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_avgTime.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_avgTime, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_avgTime = target;
   
   updateIfChanged(m_indiP_avgTime, "current", m_avgTime, INDI_IDLE);
   updateIfChanged(m_indiP_avgTime, "target", m_avgTime, INDI_IDLE);
   
   shmimMonitorT::m_restart = true;
   
   log<text_log>("set avgTime to " + std::to_string(m_avgTime), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(shmimIntegrator, m_indiP_nUpdate)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_nUpdate.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_nUpdate, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nUpdate = target;
   
   updateIfChanged(m_indiP_nUpdate, "current", m_nUpdate, INDI_IDLE);
   updateIfChanged(m_indiP_nUpdate, "target", m_nUpdate, INDI_IDLE);
   
   
   shmimMonitorT::m_restart = true;
   
   log<text_log>("set nUpdate to " + std::to_string(m_nUpdate), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(shmimIntegrator, m_indiP_startAveraging)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_startAveraging.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);

      m_running = false;
      
      state(stateCodes::READY);
      
      updateSwitchIfChanged(m_indiP_startAveraging, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);

      if(m_fileSaver && !m_continuous) m_stateStringChanged = false; //We reset this here so we can detect a change at the end of the integration
      
      m_stateStringValidOnStart = m_stateStringValid;
      m_running = true;      
      
      state(stateCodes::OPERATING);
      
      updateSwitchIfChanged(m_indiP_startAveraging, "toggle", pcf::IndiElement::On, INDI_BUSY);
   }
   return 0;
}

INDI_SETCALLBACK_DEFN( shmimIntegrator, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_fpsSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("current") != true ) //this isn't valie
   {
      return 0;
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   realT fps = ipRecv["current"].get<float>();
   
   if(fps != m_fps)
   {
      m_fps = fps;
      std::cout << "Got fps: " << m_fps << "\n";   
      shmimMonitorT::m_restart = true;
   }

   return 0;
}

INDI_SETCALLBACK_DEFN( shmimIntegrator, m_indiP_stateSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stateSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("valid") == true ) 
   {
      bool stateStringValid;
      if(ipRecv["valid"].get<std::string>() == "yes") stateStringValid = true;
      else stateStringValid = false;

      if(stateStringValid != m_stateStringValid) m_stateStringChanged = true;

      m_stateStringValid = stateStringValid;      
   }

   if( ipRecv.find("current") != true )
   {
      return 0;
   }


   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   std::string ss = ipRecv["current"].get<std::string>();
   
   if(ss != m_stateString)
   {
      m_stateString = ss;
      m_imageValid = false; //This will mark the current dark invalid
      updateIfChanged(m_indiP_imageValid, "flag", "no");   
      m_stateStringChanged = true; //We declare it changed.  This can have two effects:
                                   // 1) if we are not currently integrating, it will start a lookup in appLogic
                                   // 2) if we are integrating, after it finishes it will not be declared valid and then we'll lookup in appLogic
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //shmimIntegrator_hpp
