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
class camtipPSD : public MagAOXApp<false>, public dev::shmimMonitor<camtipPSD>
{

   friend class dev::shmimMonitor<camtipPSD>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<camtipPSD> shmimMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

  

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
   
   // function implementation
 
   ///\todo size checks here.
   
   //state(stateCodes::OPERATING);
   return 0;
}

inline
int camtipPSD::processImage( void * curr_src, 
                              const dev::shmimT & dummy 
                            )
{
   static_cast<void>(dummy); //be unused
   
   // function implementation 
   
   return 0;
}


} // namespace magAOX::app 
