/** \file hoPredCtrl.hpp
  * \brief The MagAO-X tweeter to woofer offloading manager
  *
  * \ingroup app_files
  */

#ifndef hoPredCtrl_hpp
#define hoPredCtrl_hpp

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
   
/** \defgroup hoPredCtrl Tweeter to Woofer Offloading
  * \brief Monitors the averaged tweeter shape, and sends it to the woofer.
  *
  * <a href="../handbook/operating/software/apps/hoPredCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup hoPredCtrl_files Tweeter to Woofer Offloading
  * \ingroup hoPredCtrl
  */

/** MagAO-X application to control offloading the tweeter to the woofer.
  *
  * \ingroup hoPredCtrl
  * 
  */
class hoPredCtrl : public MagAOXApp<true>, public dev::shmimMonitor<hoPredCtrl>, public dev::shmimMonitor<hoPredCtrl,darkShmimT>
{

   //Give the test harness access.
   friend class hoPredCtrl_test;

   friend class dev::shmimMonitor<hoPredCtrl>;
   friend class dev::shmimMonitor<hoPredCtrl,darkShmimT>;

   //The base shmimMonitor type
   typedef dev::shmimMonitor<hoPredCtrl> shmimMonitorT;

    //The dark shmimMonitor type
   typedef dev::shmimMonitor<hoPredCtrl, darkShmimT> darkMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

	/** \name Configurable Parameters
	 *@{
	 */

	// std::string m_twRespMPath;
	// std::string m_dmChannel;
	// float m_gain {0.1};
	// float m_leak {0.0};
	// float m_actLim {7.0}; ///< the upper limit on woofer actuator commands.  default is 7.0.

	///@}
   
	std::string m_pupilMaskFilename;

	// IMAGE m_dmStream; 
	size_t m_pwfsWidth {0}; ///< The width of the image
	size_t m_pwfsHeight {0}; ///< The height of the image.

	size_t m_quadWidth {0}; ///< The width of the image
	size_t m_quadHeight {0}; ///< The height of the image.

	uint8_t m_pwfsDataType{0}; ///< The ImageStreamIO type code.
	size_t m_pwfsTypeSize {0}; ///< The size of the type, in bytes.  

	size_t m_illuminatedPixels;
	eigenImage<realT> m_pupilMask;
	eigenImage<realT> m_measurementVector;

	realT (*pwfs_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.

	eigenImage<realT> m_darkImage;
	realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
	bool m_darkSet {false};

	// bool m_dmOpened {false};
	// bool m_dmRestart {false};

	// bool m_offloading {false};

public:
   /// Default c'tor.
   hoPredCtrl();

   /// D'tor, declared and defined for noexcept.
   ~hoPredCtrl() noexcept
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

   /// Implementation of the FSM for hoPredCtrl.
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
   
   int zero();
   
   int set_pupil_mask(std::string pupil_mask_filename);

protected:


};

inline
hoPredCtrl::hoPredCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    darkMonitorT::m_getExistingFirst = true;
   return;
}

inline
void hoPredCtrl::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkMonitorT::setupConfig(config);

  config.add("parameters.pupil_mask", "", "parameters.pupil_mask", argType::Required, "parameters", "pupil_mask", false, "string", "The path to the response matrix.");
}

inline
int hoPredCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   darkMonitorT::loadConfig(_config);
   
   _config(m_pupilMaskFilename, "parameters.pupil_mask");
  std::cout << m_pupilMaskFilename << std::endl;

   return 0;
}

inline
void hoPredCtrl::loadConfig()
{
   loadConfigImpl(config);
}

inline
int hoPredCtrl::appStartup()
{

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(darkMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   


   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int hoPredCtrl::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
      
   if( darkMonitorT::appLogic() < 0)
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
   
   
   return 0;
}

inline
int hoPredCtrl::appShutdown()
{
   shmimMonitorT::appShutdown();

	darkMonitorT::appShutdown();
   
   return 0;
}

inline
int hoPredCtrl::allocate(const dev::shmimT & dummy)
{

	static_cast<void>(dummy); //be unused

	m_pwfsWidth = shmimMonitorT::m_width;
	m_quadWidth = m_pwfsWidth / 2;
	std::cout << "Width " << m_pwfsWidth << std::endl;

	m_pwfsHeight = shmimMonitorT::m_height;
	m_quadHeight = m_pwfsHeight / 2;
	std::cout << "Height " << m_pwfsHeight << std::endl;

	set_pupil_mask(m_pupilMaskFilename);
   
	//Initialize dark image if not correct size.
	if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height)
	{
		m_darkImage.resize(shmimMonitorT::m_width,shmimMonitorT::m_height);
		m_darkImage.setZero();
		m_darkSet = false;
	}

   return 0;
}

inline
int hoPredCtrl::processImage( void * curr_src, 
                                const dev::shmimT & dummy 
                              )
{
	static_cast<void>(dummy); //be unused

	Eigen::Map<eigenImage<unsigned short>> pwfsIm( static_cast<unsigned short *>(curr_src), m_pwfsHeight, m_pwfsWidth);
	realT mean_value = 0;
	realT Ia = 0, Ib = 0, Ic = 0, Id = 0;
	
	size_t ki = 0;
	for(uint32_t col_i=0; col_i < m_quadWidth; ++col_i){
		for(uint32_t row_i=0; row_i < m_quadHeight; ++row_i){
			// Select the pixel from the correct quadrant and subtract dark			
			Ia = pwfsIm(row_i, col_i) - m_darkImage(row_i, col_i);
			Ib = pwfsIm(row_i + m_quadWidth, col_i) - m_darkImage(row_i + m_quadWidth, col_i);
			Ic = pwfsIm(row_i, col_i + m_quadHeight) - m_darkImage(row_i, col_i + m_quadHeight);
			Id = pwfsIm(row_i + m_quadWidth, col_i + m_quadHeight) - m_darkImage(row_i + m_quadWidth, col_i + m_quadHeight);

			// m_pwfsSlopeX(row_i, col_i) = Ia + Ic - Ib - Id;
			// m_pwfsSlopeY(row_i, col_i) = Ia + Ib - Ic - Id;
			// m_pwfsSlopeF(row_i, col_i) = Ia + Id - Ib - Ic;
			// m_pwfsNorm(row_i, col_i) = Ia + Id + Ib + Ic;
			
			// Take all linear combinations of the measurements and concatenate in vector
			realT pwfsNorm = Ia + Id + Ib + Ic;
			if(m_pupilMask(row_i, col_i) > 0.5){
				m_measurementVector(ki, 0) = (Ia + Ic - Ib - Id) / pwfsNorm;
				m_measurementVector(ki + m_illuminatedPixels, 0) = (Ia + Ib - Ic - Id) / pwfsNorm;
				m_measurementVector(ki + 2 * m_illuminatedPixels, 0) = (Ia + Id - Ib - Ic) / pwfsNorm;
				++ki;
			}
		}
	}
	

	std::cout << "Average counts: " << mean_value/(shmimMonitorT::m_width * shmimMonitorT::m_height) << std::endl;

	// Send the new command to the DM.
	
	return 0;
}

inline 
int hoPredCtrl::set_pupil_mask(std::string pupil_mask_filename){
/*
	This function reads in the filename to create a pupil mask and to initialize the measurement vector.
*/
	// Read in the pupil mask
	mx::fits::fitsFile<realT> ff;
	ff.read(m_pupilMask, pupil_mask_filename);
	std::cerr << "Read a " << m_pupilMask.rows() << " x " << m_pupilMask.cols() << " matrix.\n";

	// Count the number of pixels that are used for the wavefront sensing
	m_illuminatedPixels = 0;
	realT * data = m_pupilMask.data();
   
	for(size_t nn=0; nn < m_pupilMask.rows() * m_pupilMask.cols(); ++nn){
		if(data[nn] > 0.5)
			++m_illuminatedPixels;
	}
	
	// Create the measurement vector
	m_measurementVector.resize(3 * m_illuminatedPixels, 1);
    m_measurementVector.setZero();

	return 0;
}

inline
int hoPredCtrl::allocate(const darkShmimT & dummy)
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
int hoPredCtrl::processImage( void * curr_src, 
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
int hoPredCtrl::zero()
{
   
   return 0;
      
}

} //namespace app
} //namespace MagAOX

#endif //hoPredCtrl_hpp
