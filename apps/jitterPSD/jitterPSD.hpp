/** \file jitterPSD.hpp
  * \brief Calculates the PSDs of the movement and Strehl of the camtip
  * images
  *
  * \ingroup app_files
  */

#pragma once

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <fftw3.h>
#include "buffer.hpp"
#include "welchpsd.hpp"


/**************************************************************************
 *  Function: window()                                                    *
 *  Description - Applying the Hann window to the time domain signal      *
 *  --------------------------------------------------------------------  *
 *  List of Arguments:                                                    *
 *  size_t n - index of the nth data point                                *
 *  size_t N - number of points used to calculate one PSD                 *        
 *************************************************************************/
static double window(size_t n, size_t N) {
         return sin(n*M_PI/N)*sin(n*M_PI/N);  
}


namespace MagAOX::app
{
  
/** \defgroup jitterPSD Estimates PSD of camtip image movements and Strehl ratio
  * \brief Calculates the PSD of camtip image movements and Strehl ratio
  *
  * \ingroup apps
  *
  */

/** \defgroup jitterPSD_files PSD estimation
  * \ingroup jitterPSD
  */

/** MagAO-X application to get PSDs of camtip image motion and Strehl ratios.
  *
  * \ingroup jitterPSD
  * 
  */
class jitterPSD : public MagAOXApp<true>, 
                  public dev::shmimMonitor<jitterPSD>, 
                  public welchmethod,
                  public dev::frameGrabber<jitterPSD>
{

   friend class dev::shmimMonitor<jitterPSD>;
 
   friend class dev::frameGrabber<jitterPSD>;
  
   //The base shmimMonitor type
   typedef dev::shmimMonitor<jitterPSD> shmimMonitorT;

   //The base frameGrabber type
   typedef dev::frameGrabber<jitterPSD> frameGrabberT;
      
   ///Floating point type in which to do all calculations.
   typedef double realT;

   typedef fftw_complex complexT;
   
protected:
  
   bool m_alloc0 {true};

   float m_fps {0.0};
   pcf::IndiProperty m_indiP_fps;
   INDI_SETCALLBACK_DECL(jitterPSD, m_indiP_fps);


public:
   /// Default c'tor.
   jitterPSD();

   /// D'tor, declared and defined for noexcept.   ~jitterPSD() noexcept
   ~jitterPSD() noexcept
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

   /// Implementation of the FSM for jitterPSD.
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


   protected: //frameGrabber functionality
      static constexpr bool c_frameGrabber_flippable = false;

      sem_t m_smSemaphore; ///< synchronizes the fg and sm threads

      bool m_update {false};

      float fps();

      int configureAcquisition(); 
      int startAcquisition();
      int acquireAndCheckValid();
      int loadImageIntoStream(void * dest);
      int reconfig();

};


//===============================/
//          FUNCTIONS            /
//===============================/
inline
jitterPSD::jitterPSD()
: MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{}


inline
void jitterPSD::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   frameGrabberT::setupConfig(config);
}


inline
int jitterPSD::loadConfigImpl( mx::app::appConfigurator & _config )
{ 
   shmimMonitorT::loadConfig(_config);
   
   return 0;
}


inline
void jitterPSD::loadConfig()
{
   loadConfigImpl(config);
   frameGrabberT::loadConfig(config);
}


inline
int jitterPSD::appStartup()
{
  
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }

   if (frameGrabberT::appStartup() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }
   
   if (sem_init(&m_smSemaphore, 0, 0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "Initializing S.M. semaphore."});
      return -1;
   }

   REG_INDI_SETPROP( m_indiP_fps, "camtip", "fps");

   state(stateCodes::OPERATING);  
   return 0;
}


inline
int jitterPSD::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
 
   if (frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
     
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if (frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);
   }

   return 0;
}


inline
int jitterPSD::appShutdown()
{
   shmimMonitorT::appShutdown(); 
   frameGrabberT::appShutdown();
 
   return 0;
}


inline
int jitterPSD::allocate(const dev::shmimT & dummy)
{
      static_cast<void>(dummy);

      double sampleTime = 1 / fps();
      size_t num_modes = shmimMonitor::m_width;
      size_t pts_10sec = (size_t) 10 * fps();  // we are using a 10 sec window

      while (m_fps == 0) {}
      size_t pts_1sec = (size_t) fps(); 

      welch_init(num_modes, pts_1sec, pts_10sec, sampleTime, window, &m_smSemaphore);
        
      m_psd0 = true;
      m_welchThreadRestart = true;
      
      if (m_alloc0) 
      {
         if (threadStart( m_welchThread, m_welchThreadInit, m_welchThreadID, 
                          m_welchThreadProp, m_welchThreadPrio, "camtipWelchMethod", 
                          this, &welchmethod::welchCalculate) < 0)
         {
            jitterPSD::template log<software_error>({__FILE__, __LINE__});
            return -1;
         }

         m_alloc0 = false;
      }
   
      return 0;
}




inline
int jitterPSD::processImage( void * curr_src, const dev::shmimT & dummy)
{
   static_cast<void>(dummy);
   welchFetch( (double *) curr_src); 
   m_update = true;
   return 0;
}



// ============================ //
//    frameGrabber functions    //
// ============================ //
inline
float jitterPSD::fps()
{
   return m_fps;
}


inline
int jitterPSD::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if (shmimMonitorT::m_width == 0 || shmimMonitorT::m_height == 0 || shmimMonitorT::m_dataType == 0)
   {
      sleep(1);
      return -1;
   }
   
   frameGrabberT::m_width = shmimMonitorT::m_width; //columns
   frameGrabberT::m_height = (fps() / 2) + 1; //rows
   frameGrabberT::m_dataType = _DATATYPE_DOUBLE;
   
   std::cerr << "shmimMonitorT::m_dataType: " << (int) shmimMonitorT::m_dataType << "\n";
   std::cerr << "frameGrabberT::m_dataType: " << (int) frameGrabberT::m_dataType << "\n";

   return 0;
}


inline
int jitterPSD::startAcquisition()
{
   state(stateCodes::OPERATING); 
   return 0;
}


inline
int jitterPSD::acquireAndCheckValid()
{
   timespec ts;
         
   if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"}); 
      return -1;
   }
         
   ts.tv_sec += 1;
        
   if (sem_timedwait(&m_smSemaphore, &ts) == 0)
   {
      if (m_update)
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

   return -1;
}


inline
int jitterPSD::loadImageIntoStream(void * dest)
{
   memcpy(dest, m_resultArray, frameGrabberT::m_width*frameGrabberT::m_height*frameGrabberT::m_typeSize); 
   m_update = false;
   return 0;
}


inline
int jitterPSD::reconfig()
{
   return 0;
}



INDI_SETCALLBACK_DEFN( jitterPSD, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() != m_indiP_fps.getDevice() || ipRecv.getName() != m_indiP_fps.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if (ipRecv.find("current") != true )
   {
      return 0;
   }

   m_indiP_fps = ipRecv;
   
   m_fps = ipRecv["current"].get<float>();
   
   return 0;
}


} // namespace magAOX::app 
