/** \file lowfsPredCtrl.hpp
  * \brief The MagAO-X predictive controller for the LOWFS.
  *
  * \ingroup app_files
  */

#ifndef lowfsPredCtrl_hpp
#define lowfsPredCtrl_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{


   
/** \defgroup lowfsPredCtrl Controller
  * \brief TODO
  *
  * <a href="../handbook/operating/software/apps/lowfsPredCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup lowfsPredCtrl_files Controller
  * \ingroup lowfsPredCtrl
  */

/** MagAO-X application to control offloading the tweeter to the woofer.
  *
  * \ingroup lowfsPredCtrl
  * 
  */
class lowfsPredCtrl : public MagAOXApp<true>, public dev::shmimMonitor<lowfsPredCtrl>
{

   //Give the test harness access.
   friend class lowfsPredCtrl_test;

   friend class dev::shmimMonitor<lowfsPredCtrl>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<lowfsPredCtrl> shmimMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   // std::string m_dmChannel;
   
   // float m_gain {0.1};
   // float m_leak {0.0};   
   // float m_actLim {7.0}; ///< the upper limit on woofer actuator commands.  default is 7.0.
   //std::string m_twRespMPath;
   ///@}

   mx::improc::eigenImage<realT> new_image;
   mx::improc::eigenImage<realT> transfer_matrix;
   // mx::improc::eigenImage<realT> centroid;


   // IMAGE m_dmStream; 
   // uint32_t m_dmWidth {0}; ///< The width of the image
   // uint32_t m_dmHeight {0}; ///< The height of the image.
   
   // uint8_t m_dmDataType{0}; ///< The ImageStreamIO type code.
   // size_t m_dmTypeSize {0}; ///< The size of the type, in bytes.  
   
   // bool m_dmOpened {false};
   // bool m_dmRestart {false};
   // bool m_offloading {false};
   
public:
   /// Default c'tor.
   lowfsPredCtrl();

   /// D'tor, declared and defined for noexcept.
   ~lowfsPredCtrl() noexcept
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

   /// Implementation of the FSM for lowfsPredCtrl.
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
   
   // int zero();
   
};

inline
lowfsPredCtrl::lowfsPredCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void lowfsPredCtrl::setupConfig()
{
   // Here the shimMonitor is initialized. The parameters are defined in the config file.
   shmimMonitorT::setupConfig(config);
   
}

inline
int lowfsPredCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);   
   return 0;
}

inline
void lowfsPredCtrl::loadConfig()
{
   loadConfigImpl(config);
}

inline
int lowfsPredCtrl::appStartup()
{

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int lowfsPredCtrl::appLogic()
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
int lowfsPredCtrl::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   
   return 0;
}

inline
int lowfsPredCtrl::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   new_image.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   transfer_matrix.resize(2, shmimMonitorT::m_width * shmimMonitorT::m_height);
   
   size_t width = shmimMonitorT::m_width;
   size_t height = shmimMonitorT::m_width;
   for(size_t j = 0; j<height; j++){
	   for(size_t i = 0; i<width; i++){
		   // Column major or raw major ????
		   transfer_matrix(0, j * width + i) = -m_width / 2.0 + i;
		   transfer_matrix(1, j * width + i) = -m_height / 2.0 + j;
	   }
   }
   

   //m_woofer = m_gain* Eigen::Map<Array<float,-1,-1>>( m_wooferDelta.data(), m_dmWidth, m_dmHeight) + (1.0-m_leak)*m_woofer;

   return 0;
}

inline
int lowfsPredCtrl::processImage( void * curr_src, 
                                const dev::shmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused
   
   // Cast data to a 1D vector
   new_image =  Eigen::Map<Matrix<float,-1,-1>>((float *)curr_src,  m_width*m_height, 1)

   // Calculate the centroid
   centroid = transfer_matrix * new_image;
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //lowfsPredCtrl_hpp
