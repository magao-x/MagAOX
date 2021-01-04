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
   
   std::string m_fitter; ///< Device name of the pupil fitter process. If set, the number of pupils
   int m_numPupils {4};
   
   float m_pupil_cx_1; ///< the center x coordinate of pupil 1
   float m_pupil_cy_1; ///< the center y coordinate of pupil 1
   float m_pupil_D_1 {0}; ///< the diameter of pupil 1, used only for averaging the fitter output
   
   float m_pupil_cx_2; ///< the center x coordinate of pupil 2
   float m_pupil_cy_2; ///< the center y coordinate of pupil 2
   float m_pupil_D_2 {0}; ///< the diameter of pupil 2, used only for averaging the fitter output
   
   float m_pupil_cx_3; ///< the center x coordinate of pupil 3
   float m_pupil_cy_3; ///< the center y coordinate of pupil 3
   float m_pupil_D_3 {0}; ///< the diameter of pupil 3, used only for averaging the fitter output
   
   float m_pupil_cx_4; ///< the center x coordinate of pupil 4
   float m_pupil_cy_4; ///< the center y coordinate of pupil 4
   float m_pupil_D_4 {0}; ///< the diameter of pupil 4, used only for averaging the fitter output

   int m_pupil_D {56}; ///< the pupil diameter, just one applied to all pupils.
   
   int m_pupil_buffer {1}; ///< the edge buffer for the pupils, just one applied to all pupils.  Default is 1.
   
   ///@}

   sem_t m_smSemaphore; ///< Semaphore used to synchronize the fg thread and the sm thread.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   void * m_curr_src {nullptr};
   
   int m_quadSize {60};
   
   mx::improc::eigenImage<realT> m_darkImage;
   realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   bool m_darkSet {false};
   
   int m_pupil_sx_1; ///< the starting x-coordinate of pupil 1 quadrant, calculated from the pupil center, diameter, and buffer.
   int m_pupil_sy_1; ///< the starting y-coordinate of pupil 1 quadrant, calculated from the pupil center, diameter, and buffer.
   
   int m_pupil_sx_2; ///< the starting x-coordinate of pupil 2 quadrant, calculated from the pupil center, diameter, and buffer.
   int m_pupil_sy_2; ///< the starting y-coordinate of pupil 2 quadrant, calculated from the pupil center, diameter, and buffer.
   
   int m_pupil_sx_3; ///< the starting x-coordinate of pupil 3 quadrant, calculated from the pupil center, diameter, and buffer.
   int m_pupil_sy_3; ///< the starting y-coordinate of pupil 3 quadrant, calculated from the pupil center, diameter, and buffer.
   
   int m_pupil_sx_4; ///< the starting x-coordinate of pupil 4 quadrant, calculated from the pupil center, diameter, and buffer.
   int m_pupil_sy_4; ///< the starting y-coordinate of pupil 4 quadrant, calculated from the pupil center, diameter, and buffer.
   
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
   
   pcf::IndiProperty m_indiP_quad1;
   pcf::IndiProperty m_indiP_quad2;
   pcf::IndiProperty m_indiP_quad3;
   pcf::IndiProperty m_indiP_quad4;
   
public:
   INDI_SETCALLBACK_DECL(pwfsSlopeCalc, m_indiP_quad1);
   INDI_SETCALLBACK_DECL(pwfsSlopeCalc, m_indiP_quad2);
   INDI_SETCALLBACK_DECL(pwfsSlopeCalc, m_indiP_quad3);
   INDI_SETCALLBACK_DECL(pwfsSlopeCalc, m_indiP_quad4);
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
   
   config.add("pupil.fitter", "", "pupil.fitter", argType::Required, "pupil", "fitter", false, "int", "The device name of the pupil fitter.  If set, then pupil position is set by the fitter reference.");

   config.add("pupil.D", "", "pupil.D", argType::Required, "pupil", "D", false, "int", "The diameter of the pupils, fixed. Default is 56.");

   config.add("pupil.buffer", "", "pupil.buffer", argType::Required, "pupil", "buffer", false, "int", "The edge buffer for the pupils.  Default is 1.");
   
   config.add("pupil.numPupils", "", "pupil.numPupils", argType::Required, "pupil", "numPupils", false, "int", "The number of pupils.  Default is 4.  3 is also supported.");
   
   
   
   config.add("pupil.cx_1", "", "pupil.cx_1", argType::Required, "pupil", "cx_1", false, "int", "The default x-coordinate of pupil 1 (LL).  Can be updated from real-time fitter.");
   config.add("pupil.cy_1", "", "pupil.cy_1", argType::Required, "pupil", "cy_1", false, "int", "The default y-coordinate of pupil 1 (LL).  Can be updated from real-time fitter.");
   
   config.add("pupil.cx_2", "", "pupil.cx_2", argType::Required, "pupil", "cx_2", false, "int", "The default x-coordinate of pupil 2 (LL).  Can be updated from real-time fitter.");
   config.add("pupil.cy_2", "", "pupil.cy_2", argType::Required, "pupil", "cy_2", false, "int", "The default y-coordinate of pupil 2 (LL).  Can be updated from real-time fitter.");
   
   config.add("pupil.cx_3", "", "pupil.cx_3", argType::Required, "pupil", "cx_3", false, "int", "The default x-coordinate of pupil 3 (LL).  Can be updated from real-time fitter.");
   config.add("pupil.cy_3", "", "pupil.cy_3", argType::Required, "pupil", "cy_3", false, "int", "The default y-coordinate of pupil 3 (LL).  Can be updated from real-time fitter.");
   
   config.add("pupil.cx_4", "", "pupil.cx_4", argType::Required, "pupil", "cx_4", false, "int", "The default x-coordinate of pupil 4 (LL).  Can be updated from real-time fitter.");
   config.add("pupil.cy_4", "", "pupil.cy_4", argType::Required, "pupil", "cy_4", false, "int", "The default y-coordinate of pupil 4 (LL).  Can be updated from real-time fitter.");
}

inline
int pwfsSlopeCalc::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   darkMonitorT::loadConfig(_config);
   frameGrabberT::loadConfig(_config);
   
   config(m_fitter, "pupil.fitter");
   config(m_numPupils, "pupil.numPupils");
   config(m_pupil_D, "pupil.D");
   config(m_pupil_buffer, "pupil.buffer");
   config(m_pupil_cx_1, "pupil.cx_1");
   config(m_pupil_cy_1, "pupil.cy_1");
   config(m_pupil_cx_2, "pupil.cx_2");
   config(m_pupil_cy_2, "pupil.cy_2");
   config(m_pupil_cx_3, "pupil.cx_3");
   config(m_pupil_cy_3, "pupil.cy_3");
   config(m_pupil_cx_4, "pupil.cx_4");
   config(m_pupil_cy_4, "pupil.cy_4");
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
   
   if(m_fitter != "")
   {
      REG_INDI_SETPROP(m_indiP_quad1, m_fitter, "quadrant1");
      REG_INDI_SETPROP(m_indiP_quad2, m_fitter, "quadrant2");
      REG_INDI_SETPROP(m_indiP_quad3, m_fitter, "quadrant3");
      if(m_numPupils == 4)
      {
         REG_INDI_SETPROP(m_indiP_quad4, m_fitter, "quadrant4");
      }
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
   
   /*if(m_fitter != "")
   {
      if(m_numPupils == 3)
      {
         m_pupil_D = (1./3.)*(m_pupil_D_1 + m_pupil_D_2 + m_pupil_D_3);
      }
      else
      {
         m_pupil_D = (1./4.)*(m_pupil_D_1 + m_pupil_D_2 + m_pupil_D_3 + m_pupil_D_4);
      }
   }*/
   
   m_quadSize = m_pupil_D + 2*m_pupil_buffer;
   
   m_pupil_sx_1 = m_pupil_cx_1 - 0.5*m_quadSize;
   m_pupil_sy_1 = m_pupil_cy_1 - 0.5*m_quadSize;
   
   m_pupil_sx_2 = m_pupil_cx_2 - 0.5*m_quadSize;
   m_pupil_sy_2 = m_pupil_cy_2 - 0.5*m_quadSize;

   m_pupil_sx_3 = m_pupil_cx_3 - 0.5*m_quadSize;
   m_pupil_sy_3 = m_pupil_cy_3 - 0.5*m_quadSize;
   
   m_pupil_sx_4 = m_pupil_cx_4 - 0.5*m_quadSize;
   m_pupil_sy_4 = m_pupil_cy_4 - 0.5*m_quadSize;
   
   //m_quadSize = shmimMonitorT::m_width/2;
   frameGrabberT::m_width = m_quadSize;
   frameGrabberT::m_height = 2*m_quadSize;
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
   
   static float sqrt32 = sqrt(3.0)/2;
   
   float norm;
   
   if(m_numPupils == 3)
   {
      for(int rr=0; rr< m_quadSize; ++rr)
      {
         for(int cc=0; cc< m_quadSize; ++cc)
         {
            float I2 = pwfsIm(rr+m_pupil_sx_1,cc+m_pupil_sy_1) - m_darkImage(rr+m_pupil_sx_1,cc+m_pupil_sy_1);
            float I3 = pwfsIm(rr+m_pupil_sx_2,cc+m_pupil_sy_2) - m_darkImage(rr+m_pupil_sx_2,cc+m_pupil_sy_2);
            float I1 = pwfsIm(rr+m_pupil_sx_3,cc+m_pupil_sy_3) - m_darkImage(rr+m_pupil_sx_3,cc+m_pupil_sy_3);
           
            norm = 1;//I1+I2+I3+I4;
         
            slopesIm(rr,cc) = sqrt32*(I2-I3)/norm;
            slopesIm(rr,cc+m_quadSize) = (I1-0.5*(I2+I3))/norm;
         }
      }
   }
   else
   {
   
      for(int rr=0; rr< m_quadSize; ++rr)
      {
         for(int cc=0; cc< m_quadSize; ++cc)
         {
            float I1 = pwfsIm(rr+m_pupil_sx_1,cc+m_pupil_sy_1) - m_darkImage(rr+m_pupil_sx_1,cc+m_pupil_sy_1);
            float I2 = pwfsIm(rr+m_pupil_sx_2,cc+m_pupil_sy_2) - m_darkImage(rr+m_pupil_sx_2,cc+m_pupil_sy_2);
            float I3 = pwfsIm(rr+m_pupil_sx_3,cc+m_pupil_sy_3) - m_darkImage(rr+m_pupil_sx_3,cc+m_pupil_sy_3);
            float I4 = pwfsIm(rr+m_pupil_sx_4,cc+m_pupil_sy_4) - m_darkImage(rr+m_pupil_sx_4,cc+m_pupil_sy_4);
         
            norm = 1;//I1+I2+I3+I4;
         
            slopesIm(rr,cc) = ((I1+I3) - (I2+I4))/norm;
            slopesIm(rr,cc+m_quadSize) = ((I1+I2)-(I3+I4))/norm;
         }
      }
   }
    /*
   norm/=(120.*120.) * 10;  
   for(size_t ii=0; ii< frameGrabberT::m_height; ++ii)
   {
      for(size_t jj=0; jj < frameGrabberT::m_width; ++jj)
      {
         slopesIm(jj,ii)/=norm;
      }
   }*/
   
   return 0;
}

inline
int pwfsSlopeCalc::reconfig()
{
   return 0;
}

INDI_SETCALLBACK_DEFN(pwfsSlopeCalc, m_indiP_quad1)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_quad1.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});
      
      return -1;
   }
   
   if(ipRecv.find("set-x"))
   {
      float newval = ipRecv["set-x"].get<float>();
      if(newval != m_pupil_cx_1)
      {
         m_pupil_cx_1 = newval;
         m_reconfig = true;
      }
   }
   
   if(ipRecv.find("set-y"))
   {
      float newval = ipRecv["set-y"].get<float>();
      if(newval != m_pupil_cy_1)
      {
         m_pupil_cy_1 = newval;
         m_reconfig = true;
      }
   }
   
   if(ipRecv.find("set-D"))
   {
      float newval = ipRecv["set-D"].get<float>();
      if(newval != m_pupil_D_1)
      {
         m_pupil_D_1 = newval;
         m_reconfig = true;
      }
   }
   
   
   return 0;
}

INDI_SETCALLBACK_DEFN(pwfsSlopeCalc, m_indiP_quad2)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_quad2.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});
      
      return -2;
   }
   
   if(ipRecv.find("set-x"))
   {
      m_pupil_cx_2 = ipRecv["set-x"].get<float>();
   }
   
   if(ipRecv.find("set-y"))
   {
      m_pupil_cy_2 = ipRecv["set-y"].get<float>();
   }
   if(ipRecv.find("set-D"))
   {
      m_pupil_D_2 = ipRecv["set-D"].get<float>();
   }
   
   
   return 0;
}

INDI_SETCALLBACK_DEFN(pwfsSlopeCalc, m_indiP_quad3)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_quad3.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});
      
      return -3;
   }
   
   if(ipRecv.find("set-x"))
   {
      m_pupil_cx_3 = ipRecv["set-x"].get<float>();
   }
   
   if(ipRecv.find("set-y"))
   {
      m_pupil_cy_3 = ipRecv["set-y"].get<float>();
   }
   if(ipRecv.find("set-D"))
   {
      m_pupil_D_3 = ipRecv["set-D"].get<float>();
   }
   
   
   return 0;
}

INDI_SETCALLBACK_DEFN(pwfsSlopeCalc, m_indiP_quad4)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_quad4.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});
      
      return -4;
   }
   
   if(ipRecv.find("set-x"))
   {
      m_pupil_cx_4 = ipRecv["set-x"].get<float>();
   }
   
   if(ipRecv.find("set-y"))
   {
      m_pupil_cy_4 = ipRecv["set-y"].get<float>();
   }
   if(ipRecv.find("set-D"))
   {
      m_pupil_D_4 = ipRecv["set-D"].get<float>();
   }
   
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //pwfsSlopeCalc_hpp
