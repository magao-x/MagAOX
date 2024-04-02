/** \file psfFit.hpp
  * \brief The MagAO-X PSF Fitter application header
  *
  * \ingroup psfFit_files
  */

#ifndef psfFit_hpp
#define psfFit_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/fit/fitGaussian.hpp>
#include <mx/improc/imageFilters.hpp>

/** \defgroup psfFit 
  * \brief The MagAO-X PSF fitter.
  *
  * <a href="../handbook/operating/software/apps/psfFit.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup psfFit_files
  * \ingroup psfFit
  */

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


/// The MagAO-X PSF Fitter
/** 
  * \ingroup psfFit
  */
class psfFit : public MagAOXApp<true>, public dev::shmimMonitor<psfFit>, public dev::shmimMonitor<psfFit,darkShmimT>, 
                                      public dev::frameGrabber<psfFit>, public dev::telemeter<psfFit>
{
   //Give the test harness access.
   friend class psfFit_test;

   friend class dev::shmimMonitor<psfFit>;
   friend class dev::shmimMonitor<psfFit,darkShmimT>;
   friend class dev::frameGrabber<psfFit>;

   friend class dev::telemeter<psfFit>;

public:
   //The base shmimMonitor type
   typedef dev::shmimMonitor<psfFit> shmimMonitorT;
   
   typedef dev::shmimMonitor<psfFit,darkShmimT> darkShmimMonitorT;

   //The base frameGrabber type
   typedef dev::frameGrabber<psfFit> frameGrabberT;

   //The base telemeter type
   typedef dev::telemeter<psfFit> telemeterT;

   ///Floating point type in which to do all calculations.
   typedef float realT;
   
   /** \name app::dev Configurations
     *@{
     */
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber these images can not be flipped
   
   ///@}

protected:

    /** \name Configurable Parameters
      *@{
      */
   
    std::string m_fpsSource; ///< Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.

    uint16_t m_fitCircBuffMaxLength {3600}; ///< Maximum length of the latency measurement circular buffers
    float m_fitCircBuffMaxTime {5}; ///< Maximum time of the latency meaurement circular buffers

    float m_fwhmGuess {2};
   ///@}

    mx::improc::eigenImage<float> m_image;
    mx::improc::eigenImage<float> m_sm;

    mx::improc::eigenImage<float> m_dark;
   
    bool m_updated {false};
    float m_x {0};
    float m_y {0};

    float m_dx {0};
    float m_dy {0};

    float m_fps {0};

    mx::sigproc::circularBufferIndex<float, cbIndexT> m_xcb;
    mx::sigproc::circularBufferIndex<float, cbIndexT> m_ycb;

    std::vector<float> m_xcbD;
    std::vector<float> m_ycbD;

    float m_mnx {0};
    float m_rmsx {0};
    float m_mny  {0};
    float m_rmsy {0};

    //Working memory for poke fitting
    mx::math::fit::fitGaussian2Dsym<float> m_gfit;

public:
   /// Default c'tor.
   psfFit();

   /// D'tor, declared and defined for noexcept.
   ~psfFit() noexcept;
   
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

   /// Implementation of the FSM for psfFit.
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

   // shmimMonitor interface:
   int allocate( const dev::shmimT &);
   
   int processImage( void* curr_src,
                     const dev::shmimT &
                    );
   
   // shmimMonitor interface for referenc:
   int allocate( const darkShmimT &);
   
   int processImage( void* curr_src,
                     const darkShmimT &
                   );

protected:

   std::mutex m_imageMutex;

   sem_t m_smSemaphore {0}; ///< Semaphore used to synchronize the fg thread and the sm thread.

public:

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
      return m_fps;
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

protected:

   /** \name INDI
     * @{
     */ 
   
   pcf::IndiProperty m_indiP_dx;
   
   INDI_NEWCALLBACK_DECL(psfFit, m_indiP_dx);
   
   pcf::IndiProperty m_indiP_dy;
   
   INDI_NEWCALLBACK_DECL(psfFit, m_indiP_dy);

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(psfFit, m_indiP_fpsSource);

   ///@}

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_fgtimings * );

   ///@}
};

inline
psfFit::psfFit() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkShmimMonitorT::m_getExistingFirst = true;
   return;
}

inline
psfFit::~psfFit() noexcept
{

}

inline
void psfFit::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkShmimMonitorT::setupConfig(config);
   frameGrabberT::setupConfig(config);
   telemeterT::setupConfig(config);
   
   config.add("fitter.fpsSource", "", "fitter.fpsSource", argType::Required, "fitter", "fpsSource", false, "string", "Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.");

}


inline
int psfFit::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);
   darkShmimMonitorT::loadConfig(_config);

   frameGrabberT::loadConfig(_config);
   telemeterT::loadConfig(_config);

    _config(m_fpsSource, "fitter.fpsSource");


   
   return 0;
}

inline
void psfFit::loadConfig()
{
   loadConfigImpl(config);
}

inline
int psfFit::appStartup()
{
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(darkShmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(sem_init(&m_smSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.M. semaphore"});
      return -1;
   }

   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(m_fpsSource != "")
   {
      REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
   }

   CREATE_REG_INDI_NEW_NUMBERF(m_indiP_dx, "dx", -100, 100, 1e-2, "%0.02f", "", "");
   m_indiP_dx["current"].setValue(m_dx);
   m_indiP_dx["target"].setValue(m_dx);

   CREATE_REG_INDI_NEW_NUMBERF(m_indiP_dy, "dy", -100, 100, 1e-2, "%0.02f", "", "");
   m_indiP_dy["current"].setValue(m_dy);
   m_indiP_dy["target"].setValue(m_dy);


   state(stateCodes::OPERATING);
   
   return 0;
}

inline
int psfFit::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( darkShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( telemeterT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }


    if( state() == stateCodes::OPERATING && m_xcb.size() > 0 )
    {
        if(m_xcb.size() >= m_xcb.maxEntries())
        {
            cbIndexT refEntry = m_xcb.earliest();
         
            m_xcbD.resize(m_xcb.maxEntries()-1);
            m_ycbD.resize(m_xcb.maxEntries()-1);

            for(size_t n=0; n <= m_atimesD.size(); ++n)
            {
                m_xcbD[n] = m_xcb.at(refEntry,n);
                m_ycbD[n] = m_ycb.at(refEntry,n);
            }
         
            m_mnx = mx::math::vectorMean(m_xcbD);
            m_rmsx = sqrt(mx::math::vectorVariance(m_xcbD, m_mnx));
         
            m_mny = mx::math::vectorMean(m_ycbD);
            m_rmsy = sqrt(mx::math::vectorVariance(m_ycbD, m_mny));

        }
        else
        {
            m_mnx = 0;
            m_rmsx = 0;
            m_mny = 0;
            m_rmsy = 0;
        }
    }
    else
    {
        m_mnx = 0;
        m_rmsx = 0;
        m_mny = 0;
        m_rmsy = 0;
    }


   shmimMonitorT::updateINDI();
   darkShmimMonitorT::updateINDI();
   
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   updateIfChanged(m_indiP_dx, "current", m_dx);
   updateIfChanged(m_indiP_dy, "current", m_dy);

   return 0;
}

inline
int psfFit::appShutdown()
{
   shmimMonitorT::appShutdown();
   darkShmimMonitorT::appShutdown();
   frameGrabberT::appShutdown();
   telemeterT::appShutdown();

   return 0;
}

inline
int psfFit::allocate(const dev::shmimT & dummy)
{
    static_cast<void>(dummy);
   
    std::lock_guard<std::mutex> guard(m_imageMutex);

    m_image.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
    m_image.setZero();

    m_sm.resize(m_image.rows(), m_image.cols());

    if(m_fitCircBuffMaxLength == 0 || m_fitCircBuffMaxTime == 0 || m_fps <= 0)
    {
        m_xcb.maxEntries(0);
        m_ycb.maxEntries(0);
    }
    else 
    {
        //Set up the fit circ. buffs
        cbIndexT cbSz = m_fitCircBuffMaxTime * m_fps;
        if(cbSz > m_fitCircBuffMaxLength) cbSz = m_fitCircBuffMaxLength;
        if(cbSz < 3) cbSz = 3; //Make variance meaningful
        m_xcb.maxEntries(cbSz);
        m_ycb.maxEntries(cbSz);
    }

   m_updated = false;
   return 0;
}
   
inline
int psfFit::processImage( void* curr_src,
                          const dev::shmimT & dummy
                        )
{
    static_cast<void>(dummy);
   
    std::unique_lock<std::mutex> lock(m_imageMutex);
   
    if(m_dark.rows() == m_image.rows() && m_dark.cols() == m_image.cols())
    {
        for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
        {
            m_image.data()[nn] = ((uint16_t*)curr_src) [nn] - m_dark.data()[nn];
        }
    }
    else
    {
        for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
        {
            m_image.data()[nn] = ((uint16_t*)curr_src) [nn];
        }
    }

    lock.unlock();

    // fit
    //std::cerr << m_image.sum() << "\n";
    
    //float max;
    //int x=0;
    //int y=0;

    //max = m_image.maxCoeff(&x, &y);

    //mx::improc::medianSmooth(m_sm, x, y, max, m_image, 3);

    mx::improc::imageCenterOfLight(m_x, m_y, m_image);


    /*if(fabs(m_x-x) > 2 || fabs(m_y-y) > 2)
    {
        std::cerr << "skip frame\n";
        return 0;
    }*/
/*
    m_gfit.set_itmax(1000);
    m_gfit.setArray(m_image.data(), m_image.rows(), m_image.cols());
    m_gfit.setGuess(0, max, x, y, mx::math::func::sigma2fwhm(m_fwhmGuess));
    m_gfit.fit();

    m_x = m_gfit.x0();
    m_y = m_gfit.y0();
*/


    m_updated = true;

    //signal framegrabber
    //Now tell the f.g. to get going
    if(sem_post(&m_smSemaphore) < 0)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
        return -1;
    }
         
    //Update the latency circ. buffs
    if(m_xcb.maxEntries()  >  0)
    {
        m_xcb.nextEntry(m_x);
        m_ycb.nextEntry(m_y);
    }

    return 0;
}
   
inline
int psfFit::allocate(const darkShmimT & dummy)
{
    static_cast<void>(dummy);
   
    std::lock_guard<std::mutex> guard(m_imageMutex);
   
    if(darkShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT)
    {
        return log<software_error,-1>({__FILE__, __LINE__, "dark is not float"});
    }

    m_dark.resize( darkShmimMonitorT::m_width, darkShmimMonitorT::m_height );
    m_dark.setZero();

    return 0;
}   
   
inline
int psfFit::processImage( void* curr_src,
                          const darkShmimT & dummy
                        )
{
    static_cast<void>(dummy);
   
    std::unique_lock<std::mutex> lock(m_imageMutex);

    for(unsigned nn=0; nn < darkShmimMonitorT::m_width*darkShmimMonitorT::m_height; ++nn)
    {
        m_dark.data()[nn] += ((float*)curr_src) [nn];
    }

    lock.unlock();

    log<text_log>("dark updated", logPrio::LOG_INFO);

    return 0;
}

inline
int psfFit::configureAcquisition()
{

   frameGrabberT::m_width = 2;
   frameGrabberT::m_height = 1;
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;
   
   return 0;
}

inline
int psfFit::startAcquisition()
{
   return 0;
}

inline
int psfFit::acquireAndCheckValid()
{
   timespec ts;
         
   if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__,__LINE__,errno, 0, "clock_gettime"}); 
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
int psfFit::loadImageIntoStream(void * dest)
{
   ((float *) dest)[0] = m_x - m_dx;
   ((float *) dest)[1] = m_y - m_dy;

   m_updated = false;

   return 0;
}

inline
int psfFit::reconfig()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(psfFit, m_indiP_dx)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_dx.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_dx, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_dx = target;
   
   log<text_log>("set dx = " + std::to_string(m_dx), logPrio::LOG_NOTICE);
   return 0;
}

INDI_NEWCALLBACK_DEFN(psfFit, m_indiP_dy)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_dy.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_dy, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_dy = target;
   
   log<text_log>("set dy = " + std::to_string(m_dy), logPrio::LOG_NOTICE);
   return 0;
}

INDI_SETCALLBACK_DEFN( psfFit, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
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
      shmimMonitorT::m_restart = true;
   }

   return 0;
}

inline
int psfFit::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_fgtimings());
}
   
inline
int psfFit::recordTelem( const telem_fgtimings * )
{
   return recordFGTimings(true);
}



} //namespace app
} //namespace MagAOX

#endif //psfFit_hpp
