/** \file refRMS.hpp
  * \brief The MagAO-X user gain control app
  *
  * \ingroup app_files
  */

#ifndef refRMS_hpp
#define refRMS_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

struct refShmimT 
{
   static std::string configSection()
   {
      return "refShmim";
   };
   
   static std::string indiPrefix()
   {
      return "refShmim";
   };
};


struct maskShmimT 
{
   static std::string configSection()
   {
      return "maskShmim";
   };
   
   static std::string indiPrefix()
   {
      return "maskShmim";
   };
};

/** \defgroup refRMS Calculate the RMS of the ref subtracted image
  * \brief Calculates the r.m.s. of the reference subtracted WFS image.
  *
  * <a href="../handbook/operating/software/apps/refRMS.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup refRMS_files User Gain Control
  * \ingroup refRMS
  */

/** MagAO-X application to calculate the RMS of the reference subtracted WFS image.
  *
  * \ingroup refRMS
  * 
  */
class refRMS : public MagAOXApp<true>, public dev::shmimMonitor<refRMS, refShmimT>, 
                     public dev::shmimMonitor<refRMS,maskShmimT>
{

   //Give the test harness access.
   friend class refRMS_test;

   friend class dev::shmimMonitor<refRMS,refShmimT>;
   friend class dev::shmimMonitor<refRMS,maskShmimT>;
  
public:

   //The base shmimMonitor type
   typedef dev::shmimMonitor<refRMS,refShmimT> refShmimMonitorT;
   typedef dev::shmimMonitor<refRMS,maskShmimT> maskShmimMonitorT;

   ///Floating point type in which to do all calculations.
   typedef float realT;
   
   typedef uint16_t cbIndexT;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_fpsSource; ///< Device name for getting fps. This device should have *.fps.current.

   ///@}
 
   mx::improc::eigenImage<realT> m_currRef;
   mx::improc::eigenImage<realT> m_mask;
   bool m_maskValid {false};
   realT m_maskSum {0};

   mx::sigproc::circularBufferIndex<float, cbIndexT> m_rms;
   
   mx::sigproc::circularBufferIndex<float, cbIndexT> m_mean;

   float m_fps {0}; ///< Current FPS from the FPS source.

public:
   /// Default c'tor.
   refRMS();

   /// D'tor, declared and defined for noexcept.
   ~refRMS() noexcept
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

   /// Implementation of the FSM for refRMS.
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


   int allocate( const refShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const refShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   

   int allocate( const maskShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const maskShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );


   pcf::IndiProperty m_indiP_refrms;

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(refRMS, m_indiP_fpsSource);


};

inline
refRMS::refRMS() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

   refShmimMonitorT::m_getExistingFirst = true;
   maskShmimMonitorT::m_getExistingFirst = true;
   
   return;
}

inline
void refRMS::setupConfig()
{
   refShmimMonitorT::setupConfig(config);
   maskShmimMonitorT::setupConfig(config); 

   config.add("rms.fpsSource", "", "rms.fpsSource", argType::Required, "rms", "fpsSource", false, "string", "Device name for getting fps.  This device should have *.fps.current."); 
}

inline
int refRMS::loadConfigImpl( mx::app::appConfigurator & _config )
{
   refShmimMonitorT::loadConfig(config);

   maskShmimMonitorT::loadConfig(config);

   _config(m_fpsSource, "rms.fpsSource");

   return 0;
}

inline
void refRMS::loadConfig()
{
   loadConfigImpl(config);
}

inline
int refRMS::appStartup()
{
   createROIndiNumber( m_indiP_refrms, "refrms", "Reference RMS", "");
   indi::addNumberElement(m_indiP_refrms, "one_sec", 0, 1, 1000, "One Second rms");
   indi::addNumberElement(m_indiP_refrms, "two_sec", 0, 1, 1000, "Two Second rms");
   indi::addNumberElement(m_indiP_refrms, "five_sec", 0, 1, 1000, "Five Second rms");
   indi::addNumberElement(m_indiP_refrms, "ten_sec", 0, 1, 1000, "Ten Second rms");
   registerIndiPropertyReadOnly(m_indiP_refrms);

   if(m_fpsSource != "")
   {
      REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
   }

   state(stateCodes::READY);
    
   return 0;
}

inline
int refRMS::appLogic()
{
   if( refShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( maskShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( m_rms.size() > 0 )
   {
      if(m_rms.size() >= m_rms.maxEntries())
      {
         cbIndexT refEntry = m_rms.nextEntry();

         //Calculate shit
         //need some way to detect that we shouldn't bother when not updating
      }
   }         


   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(refShmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(maskShmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }


   return 0;
}

inline
int refRMS::appShutdown()
{
   refShmimMonitorT::appShutdown();
   maskShmimMonitorT::appShutdown();
   
   return 0;
}


inline
int refRMS::allocate(const refShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
     
   m_currRef.resize(refShmimMonitorT::m_width, refShmimMonitorT::m_height);

   if(m_mask.rows() != m_currRef.rows() || m_mask.cols() != m_currRef.cols())
   {
      m_maskValid = false;
      m_mask.resize(m_currRef.rows(), m_currRef.cols());
      m_mask.setConstant(1);
   }

   if(m_fps == 0)
   {
      sleep(5);
      if(m_fps == 0)
      {
         return log<text_log,-1>("fps not updated", logPrio::LOG_ERROR);
      }
   }

   cbIndexT cbSz = 10 * m_fps;
   m_mean.maxEntries(cbSz);
   m_rms.maxEntries(cbSz);
   return 0;
}

inline
int refRMS::processImage( void * curr_src, 
                          const refShmimT & dummy 
                        )
{
   static_cast<void>(dummy); //be unused
  
   //Copy it out first so we can afford to be slow and skipping frames 
   m_currRef = Eigen::Map<Eigen::Matrix<float,-1,-1>>((float *)curr_src,  refShmimMonitorT::m_width*refShmimMonitorT::m_height,1);

   //If mask has changed we skip
   if(m_mask.rows() != m_currRef.rows() || m_mask.cols() != m_currRef.cols()) return 0;

   //mult by mask, etc.      
   float mean = (m_currRef * m_mask).sum()/m_maskSum;
   float rms = sqrt(((m_currRef - mean)*m_mask).square().sum()/m_maskSum);

   m_mean.nextEntry(mean);
   m_rms.nextEntry(rms);
   
   return 0;
}


inline
int refRMS::allocate(const maskShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
     
   m_mask.resize(maskShmimMonitorT::m_width, maskShmimMonitorT::m_height);

   return 0;
}

inline
int refRMS::processImage( void * curr_src, 
                          const maskShmimT & dummy 
                        )
{
   static_cast<void>(dummy); //be unused
  
   //copy curr_src to mask
   m_mask = Eigen::Map<Eigen::Matrix<float,-1,-1>>((float *)curr_src,  maskShmimMonitorT::m_width*maskShmimMonitorT::m_height,1);

   m_maskSum = m_mask.sum();

   if(m_mask.rows() == m_currRef.rows() || m_mask.cols() == m_currRef.cols())
   {
      m_maskValid = true;
   }

   return 0;
}

INDI_SETCALLBACK_DEFN( refRMS, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
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
      refShmimMonitorT::m_restart = true;
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //refRMS_hpp
