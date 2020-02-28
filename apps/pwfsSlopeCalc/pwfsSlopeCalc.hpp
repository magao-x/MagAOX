/** \file pwfsSlopeCalc.hpp
  * \brief The MagAO-X PWFS Slope Calculator
  *
  * \ingroup app_files
  */

#ifndef pwfsSlopeCalc_hpp
#define pwfsSlopeCalc_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

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


/** \defgroup pwfsSlopeCalc PWFS Slope Calculator
  * \brief Calculates slopes from a PWFS image.
  *
  * <a href="../handbook/operating/software/apps/pwfsSlopeCalc.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup pwfsSlopeCalc_files PWFS Slope Calculator Files
  * \ingroup pwfsSlopeCalc
  */

/** MagAO-X application to calculate slopes from PWFS images.
  *
  * \ingroup pwfsSlopeCalc
  * 
  */
class pwfsSlopeCalc : public MagAOXApp<true>, public dev::shmimMonitor<pwfsSlopeCalc>, public dev::shmimMonitor<pwfsSlopeCalc,darkShmimT>, public dev::frameGrabber<pwfsSlopeCalc>
{

   //Give the test harness access.
   friend class pwfsSlopeCalc_test;

   friend class dev::shmimMonitor<pwfsSlopeCalc>;
   friend class dev::shmimMonitor<pwfsSlopeCalc,darkShmimT>;
   friend class dev::frameGrabber<pwfsSlopeCalc>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<pwfsSlopeCalc> shmimMonitorT;
   
   //The dark shmimMonitor type
   typedef dev::shmimMonitor<pwfsSlopeCalc, darkShmimT> darkMonitorT;
   
   //The base frameGrabber type
   typedef dev::frameGrabber<pwfsSlopeCalc> frameGrabberT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   ///@}

   sem_t m_smSemaphore; ///< Semaphore used to synchronize the fg thread and the sm thread.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   void * m_curr_src {nullptr};
   
   int m_quadSize {60};
   
   mx::improc::eigenImage<realT> m_darkImage;
   realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   bool m_darkSet {false};
   
public:
   /// Default c'tor.
   pwfsSlopeCalc();

   /// D'tor, declared and defined for noexcept.
   ~pwfsSlopeCalc() noexcept
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

   /// Implementation of the FSM for pwfsSlopeCalc.
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

   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int allocate( const darkShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const darkShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
protected:

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
   
};

inline
pwfsSlopeCalc::pwfsSlopeCalc() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkMonitorT::m_getExistingFirst = true;
   return;
}

inline
void pwfsSlopeCalc::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkMonitorT::setupConfig(config);
   
   frameGrabberT::setupConfig(config);
}

inline
int pwfsSlopeCalc::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   darkMonitorT::loadConfig(_config);
   frameGrabberT::loadConfig(_config);
   
   return 0;
}

inline
void pwfsSlopeCalc::loadConfig()
{
   loadConfigImpl(config);
}

inline
int pwfsSlopeCalc::appStartup()
{
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
   
   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int pwfsSlopeCalc::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( darkMonitorT::appLogic() < 0)
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
      
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
      
  
   
   
   return 0;
}

inline
int pwfsSlopeCalc::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   darkMonitorT::appShutdown();
   
   frameGrabberT::appShutdown();
   
   return 0;
}

inline
int pwfsSlopeCalc::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused

   //Initialize dark image if not correct size.
   if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height)
   {
      m_darkImage.resize(shmimMonitorT::m_width,shmimMonitorT::m_height);
      m_darkImage.setZero();
      m_darkSet = false;
   }
   
   m_reconfig = true;
   
   return 0;
}

inline
int pwfsSlopeCalc::processImage( void * curr_src, 
                                       const dev::shmimT & dummy 
                                     )
{
   static_cast<void>(dummy); //be unused

   m_curr_src = curr_src;

   //Now tell the f.g. to get going
   if(sem_post(&m_smSemaphore) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
       return -1;
   }

   return 0;
}

inline
int pwfsSlopeCalc::allocate(const darkShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   m_darkSet = false;
   
//    if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height)
//    {
//       darkMonitorT::m_restart = true;
//    }
   
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
int pwfsSlopeCalc::processImage( void * curr_src, 
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
int pwfsSlopeCalc::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::m_width==0 || shmimMonitorT::m_height==0 || shmimMonitorT::m_dataType == 0)
   {
      //This means we haven't connected to the stream to average. so wait.
      sleep(1);
      return -1;
   }
   
   m_quadSize = shmimMonitorT::m_width/2;
   frameGrabberT::m_width = shmimMonitorT::m_width/2;
   frameGrabberT::m_height = shmimMonitorT::m_height;
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;
   
   return 0;
}

inline
int pwfsSlopeCalc::startAcquisition()
{
   return 0;
}

inline
int pwfsSlopeCalc::acquireAndCheckValid()
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
      clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
      return 0;
   }
   else
   {
      return 1;
   }
}

inline
int pwfsSlopeCalc::loadImageIntoStream(void * dest)
{
   //Here is where we do it.
   Eigen::Map<eigenImage<unsigned short>> pwfsIm( static_cast<unsigned short *>(m_curr_src), shmimMonitorT::m_width, shmimMonitorT::m_height );
   Eigen::Map<eigenImage<float>> slopesIm(static_cast<float*>(dest), frameGrabberT::m_width, frameGrabberT::m_height );
   
   for(int ii=0; ii< m_quadSize; ++ii)
   {
      for(int jj=0; jj< m_quadSize; ++jj)
      {
         float I1 = pwfsIm(jj,ii) - m_darkImage(jj,jj);
         float I2 = pwfsIm(jj + m_quadSize,ii) - m_darkImage(jj+m_quadSize,jj);
         float I3 = pwfsIm(jj, ii + m_quadSize) - m_darkImage(jj, ii+m_quadSize);
         float I4 = pwfsIm(jj+m_quadSize, ii + m_quadSize) - m_darkImage(jj+m_quadSize, ii+m_quadSize);
         
         float sum = I1+I2+I3+I4;
         
         slopesIm(jj,ii) = ((I1+I3) - (I2+I4))/sum;
         slopesIm(jj,ii+m_quadSize) = ((I1+I2)-(I3+I4))/sum;
      }
   }
      
   return 0;
}

inline
int pwfsSlopeCalc::reconfig()
{
   return 0;
}



} //namespace app
} //namespace MagAOX

#endif //pwfsSlopeCalc_hpp
