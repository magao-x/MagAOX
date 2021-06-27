/** \file camtipPSD.hpp
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
  
/** \defgroup camtipPSD Estimates PSD of camtip image movements and Strehl ratio
  * \brief Calculates the PSD of camtip image movements and Strehl ratio
  *
  * \ingroup apps
  *
  */

/** \defgroup camtipPSD_files PSD estimation
  * \ingroup camtipPSD
  */

/** MagAO-X application to get PSDs of camtip image motion and Strehl ratios.
  *
  * \ingroup camtipPSD
  * 
  */
class camtipPSD : public MagAOXApp<false>, 
                  public dev::shmimMonitor<camtipPSD>, 
                  public welchmethod,
                  public dev::frameGrabber<camtipPSD>
{

   friend class dev::shmimMonitor<camtipPSD>;
 
   friend class dev::frameGrabber<camtipPSD>;
  
   //The base shmimMonitor type
   typedef dev::shmimMonitor<camtipPSD> shmimMonitorT;

   //The base frameGrabber type
   typedef dev::frameGrabber<camtipPSD>;
      
   ///Floating point type in which to do all calculations.
   typedef double realT;

   typedef fftw_complex complexT;
   
protected:
  
   bool m_imOpened  {false};
   bool m_imRestart {false};
 
   IMAGE m_shifts;
   std::string m_shiftsKey {"camtip-shifts"};

   bool m_alloc0 {true};

public:
   /// Default c'tor.
   camtipPSD();

   /// D'tor, declared and defined for noexcept.   ~camtipPSD() noexcept
   ~camtipPSD() noexcept
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

   /// Implementation of the FSM for camtipPSD.
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
camtipPSD::camtipPSD() 
: MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{}

inline
void camtipPSD::setupConfig()
{
   shmimMonitorT::setupConfig(config);
}


inline
int camtipPSD::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   
   return 0;
}

inline
void camtipPSD::loadConfig()
{
   loadConfigImpl(config);
}

inline
int camtipPSD::appStartup()
{
  
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }


   state(stateCodes::OPERATING);  
   return 0;
}

inline
int camtipPSD::appLogic()
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

inline
int camtipPSD::appShutdown()
{
   shmimMonitorT::appShutdown(); 
   
   return 0;
}

inline
int camtipPSD::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused

   if(m_imOpened)
   {
      ImageStreamIO_closeIm(&m_shifts);
   }
   
   m_imOpened  = false;
   m_imRestart = false;

   if (ImageStreamIO_openIm(&m_shifts, m_shiftsKey.c_str()) == 0)
   {
      if (m_shifts.md[0].sem < 10) 
      {
            ImageStreamIO_closeIm(&m_shifts);
      }
      else
      {
         m_imOpened = true;
      }
   }
      
   if (!m_imOpened) {

      log<software_error>({__FILE__, __LINE__, m_outputKey + " not opened."});
      return -1;

   } else {
     
      double sampleTime = 0.02; // 500 Hz, this value should be hard-coded 
      size_t num_modes = m_shifts.md[0].size[0];
      size_t pts_10sec = (size_t) 10/sampleTime;  // we are using a 10 sec window
      size_t pts_1sec = 500;  //\todo this should not be hard-coded 

      welch_init(num_modes, pts_1sec, pts_10sec, sampleTime, window, &m_shifts);
        
      m_psd0 = true;
      m_welchThreadRestart = true;
      
      if (m_alloc0) 
      {
         if( threadStart( m_welchThread, 
                          m_welchThreadInit, 
                          m_welchThreadID, 
                          m_welchThreadProp, 
                          m_welchThreadPrio, 
                          "camtipWelchMethod", 
                          this, 
                          &welchmethod::welchCalculate
                        ) < 0
           )
         {
            camtipPSD::template log<software_error>({__FILE__, __LINE__});
            return -1;
         }
         m_alloc0 = false;
      }
   
   }
 
   //state(stateCodes::OPERATING);
   return 0;
}




inline
int camtipPSD::processImage( void * curr_src __attribute__((unused)), 
                             const dev::shmimT & dummy 
                           )
{
   static_cast<void>(dummy); //be unused

   welchFetch(); 

   return 0;
}



// ============================ //
//    frameGrabber functions    //
// ============================ //
inline
float camtipPSD::fps()
{
   return m_fps;
}


inline
int camtipPSD::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if (shmimMonitorT::m_width == 0 || shmimMonitorT::m_height == 0 || shmimMonitorT::m_dataType == 0)
   {
      sleep(1);
      return -1;
   }
   
   frameGrabberT::m_width = shmimMonitorT::m_width; //columns
   frameGrabberT::m_height = (shmimMonitorT::m_height / 2) + 1; //rows
   frameGrabberT::m_dataType = _DATATYPE_DOUBLE;
   
   std::cerr << "shmimMonitorT::m_dataType: " << (int) shmimMonitorT::m_dataType << "\n";
   std::cerr << "frameGrabberT::m_dataType: " << (int) frameGrabberT::m_dataType << "\n";

   return 0;
}


inline
int camtipPSD::startAcquisition()
{
   state(stateCodes::OPERATING); 
   return 0;
}


inline
int camtipPSD::acquireAndCheckValid()
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
int camtipPSD::loadImageIntoStream(void * dest)
{
   memcpy(dest, m_data, frameGrabberT::m_width*frameGrabberT::m_height*frameGrabberT::m_typeSize); 
   m_update = false;
   return 0;
}


inline
int camtipPSD::reconfig()
{
   return 0;
}


} // namespace magAOX::app 
