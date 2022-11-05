/** \file photonCounter.hpp
  * \brief The MagAO-X PWFS Slope Calculator
  *
  * \ingroup app_files
  */

#ifndef photonCounter_hpp
#define photonCounter_hpp

#include <limits>
#include <algorithm>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup photonCounter PWFS Slope Calculator
  * \brief Calculates slopes from a PWFS image.
  *
  * <a href="../handbook/operating/software/apps/photonCounter.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup photonCounter_files PWFS Slope Calculator Files
  * \ingroup photonCounter
  */

/** MagAO-X application to calculate slopes from PWFS images.
  *
  * \ingroup photonCounter
  * 
  */
class photonCounter : public MagAOXApp<true>, public dev::shmimMonitor<photonCounter>, public dev::frameGrabber<photonCounter>
{

   //Give the test harness access.
   friend class photonCounter_test;

   friend class dev::shmimMonitor<photonCounter>;
   friend class dev::frameGrabber<photonCounter>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<photonCounter> shmimMonitorT;
   
   //The base frameGrabber type
   typedef dev::frameGrabber<photonCounter> frameGrabberT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber these images can not be flipped
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   ///@}

	sem_t m_smSemaphore; ///< Semaphore used to synchronize the fg thread and the sm thread.
	realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
	void * m_curr_src {nullptr};

	int m_image_width;
	int m_image_height;

	mx::improc::eigenCube<realT> m_calibrationCube;
	mx::improc::eigenImage<realT> m_dark_image;   
	mx::improc::eigenImage<realT> m_thresholdImage;
	mx::improc::eigenImage<realT> m_photonCountedImage;
	realT m_quantile_cut;

	//
	bool m_calibrate;
	bool m_calibrationSet;

	int calibration_steps;
	uint32_t current_calibration_iteration;
	int m_stack_frames;
	int m_stack_frames_index;

   
public:
   /// Default c'tor.
   photonCounter();

   /// D'tor, declared and defined for noexcept.
   ~photonCounter() noexcept
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

   /// Implementation of the FSM for photonCounter.
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
   
   float fps()
   {
      return 250;
   }
   
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
	pcf::IndiProperty m_indiP_calibrateToggle;
	pcf::IndiProperty m_indiP_calibrateSteps;
	pcf::IndiProperty m_indiP_stackFrames;
	pcf::IndiProperty m_indiP_quantileCut;
   
public:
	
	// Control states
	INDI_NEWCALLBACK_DECL(photonCounter, m_indiP_calibrateToggle);
	INDI_NEWCALLBACK_DECL(photonCounter, m_indiP_calibrateSteps);
	INDI_NEWCALLBACK_DECL(photonCounter, m_indiP_stackFrames);
    INDI_NEWCALLBACK_DECL(photonCounter, m_indiP_quantileCut);

};


inline
photonCounter::photonCounter() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void photonCounter::setupConfig()
{
	shmimMonitorT::setupConfig(config);
	frameGrabberT::setupConfig(config);

	config.add("parameters.quantile", "", "parameters.quantile", argType::Required, "parameters", "quantile", false, "float", "The quantile of the threshold.");
	config.add("parameters.Nstack", "", "parameters.Nstack", argType::Required, "parameters", "Nstack", false, "string", "The number of frames to stack.");
	config.add("parameters.Ncalibrate", "", "parameters.Ncalibrate", argType::Required, "parameters", "Ncalibrate", false, "string", "The number of frames for calibration.");
}

inline
int photonCounter::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
	shmimMonitorT::loadConfig(_config);
	frameGrabberT::loadConfig(_config);

	_config(m_stack_frames, "parameters.Nstack");
	_config(m_quantile_cut, "parameters.quantile");
	_config(calibration_steps, "parameters.Ncalibrate");
   
   return 0;
}

inline
void photonCounter::loadConfig()
{
   loadConfigImpl(config);
}

inline
int photonCounter::appStartup()
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
   
   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

	// createStandardIndiToggleSw( m_indiP_calibrateToggle, "calibrate", "Control calibration state", "Calibration control");
	// registerIndiPropertyNew( m_indiP_calibrateToggle, INDI_NEWCALLBACK(m_indiP_calibrateToggle) ); 

	createStandardIndiRequestSw( m_indiP_calibrateToggle, "calibrate", "Start calibration", "Calibration control");
	registerIndiPropertyNew( m_indiP_calibrateToggle, INDI_NEWCALLBACK(m_indiP_calibrateToggle) ); 
   
	createStandardIndiNumber<int>( m_indiP_calibrateSteps, "nFrames", 0, 1000000, 1, "%d", "The calibration ", "Calibration control");
	registerIndiPropertyNew( m_indiP_calibrateSteps, INDI_NEWCALLBACK(m_indiP_calibrateSteps) );  

	createStandardIndiNumber<int>( m_indiP_stackFrames, "stackNframes", 0, 1000000, 1, "%d", "The calibration ", "Calibration control");
	registerIndiPropertyNew( m_indiP_stackFrames, INDI_NEWCALLBACK(m_indiP_stackFrames) );  

	createStandardIndiNumber<float>( m_indiP_quantileCut, "quantile", 0, 1.0, 0.0001, "%0.3f", "The quantile cut", "Calibration control");
	registerIndiPropertyNew( m_indiP_quantileCut, INDI_NEWCALLBACK(m_indiP_quantileCut));  

   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int photonCounter::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
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
      
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

	updateIfChanged(m_indiP_calibrateSteps, "current", calibration_steps);
	updateIfChanged(m_indiP_stackFrames, "current", m_stack_frames);
	updateIfChanged(m_indiP_quantileCut, "current", m_quantile_cut);
   
   return 0;
}

inline
int photonCounter::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   frameGrabberT::appShutdown();
   
   return 0;
}

inline
int photonCounter::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused

	m_image_height = shmimMonitorT::m_height;
	m_image_width = shmimMonitorT::m_width;

	m_calibrationCube.resize(shmimMonitorT::m_width, shmimMonitorT::m_height, calibration_steps);
	m_calibrationCube.setZero();

	m_thresholdImage.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
	m_thresholdImage.setZero();

	m_dark_image.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
	m_dark_image.setZero();

	m_photonCountedImage.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
	m_photonCountedImage.setZero();
	
	// m_stack_frames = 1;
	// calibration_steps = 1000;

	m_calibrate = false;
	m_calibrationSet = false;
	current_calibration_iteration = 0;
	m_stack_frames_index = 0;

   	return 0;
}

inline
int photonCounter::processImage( void * curr_src, 
                                       const dev::shmimT & dummy 
                                     )
{
   static_cast<void>(dummy); //be unused

   // Set the internal pointer to the new data stream
   	Eigen::Map<eigenImage<unsigned short>> camera_image( static_cast<unsigned short *>(curr_src), m_image_height, m_image_width);

   if(m_calibrate){
	
	   if(current_calibration_iteration == 0)
	   		std::cout << "Start data collection" << std::endl;
		
		m_calibrationSet = false;
		
		// Copy the data into the stream
		for(uint32_t col_i=0; col_i < m_image_width; ++col_i){
			for(uint32_t row_i=0; row_i < m_image_height; ++row_i){

				m_calibrationCube.image(current_calibration_iteration)(row_i, col_i) = (float)camera_image(row_i, col_i);

			}
		}

		current_calibration_iteration += 1;

		if(current_calibration_iteration >= calibration_steps){
			// We have collected enough data!
			current_calibration_iteration = 0;
			m_calibrate = false;
			m_calibrationSet = true;

			// Measure the average dark frame
			m_calibrationCube.mean(m_dark_image);

			// Copy the data into the stream
			for(uint32_t col_i=0; col_i < m_image_width; ++col_i){
				for(uint32_t row_i=0; row_i < m_image_height; ++row_i){
					
					if( col_i == 0){
					 	std::cout << "The dark : " << m_dark_image(row_i, col_i) << " ";
					}

					// Get the time-series of the pixel of interest
					// auto pixelVec = m_calibrationCube.pixel(row_i, col_i);
					
					// Dark subtract the timeseries
					// for(size_t p=0; p<pixelVec.size(); p++){
					// 	pixelVec(p, 0) = pixelVec(p, 0) - m_dark_image(row_i, col_i);
					// }
					std::vector<float> tempVec;
					for(int ti=0; ti < calibration_steps; ti++){
						float val = m_calibrationCube.image(ti)(row_i, col_i);
						tempVec.push_back(val);
					}
					
					// Make it a standard library vector
					// float* start = &pixelVec(0,0);
					// float* end = &pixelVec(0,0) + pixelVec.size() * sizeof(float);
					//
					// Sort the array
					std::sort(tempVec.begin(), tempVec.end());
					
					// Find the qth quantile
					int q_index = (int)(m_quantile_cut * tempVec.size());
					if( col_i == 0 )
						std::cout <<"q0: [" << tempVec[0] << ", " << tempVec[q_index] << ", " << tempVec[tempVec.size()-1] << "]" << std::endl;
					
					m_thresholdImage(row_i, col_i) = tempVec[q_index];
					
				}
			}

			std::cout << "Calibration done!" << std::endl;
		}


   }else{
	   
		if(m_calibrationSet){
			// if( m_stack_frames_index == 0 )
			// 	std::cout << "Start collecting data for stacking..." << std::endl;

			// Apply photon counting
			for(uint32_t col_i=0; col_i < m_image_width; ++col_i){
				for(uint32_t row_i=0; row_i < m_image_height; ++row_i){
					float dI = (float)camera_image(row_i, col_i); // - m_dark_image(row_i, col_i);
					
					if( dI > m_thresholdImage(row_i, col_i)){
						m_photonCountedImage(row_i, col_i) += 1.0;
					}

				}
			}
			
			m_stack_frames_index += 1;

			if( (m_stack_frames_index >= m_stack_frames) ){

				// std::cout << "Send stacked data..." << std::endl;
				m_stack_frames_index = 0;

				//Now tell the f.g. to get going
				if(sem_post(&m_smSemaphore) < 0)
				{
					log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
					return -1;
				}

			}

		}

   }

   return 0;
}

inline
int photonCounter::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::m_width==0 || shmimMonitorT::m_height==0 || shmimMonitorT::m_dataType == 0)
   {
      //This means we haven't connected to the stream to average. so wait.
      sleep(1);
      return -1;
   }
   
   // The frame grabber has the exact same size as the imagestream
   frameGrabberT::m_width = shmimMonitorT::m_width;
   frameGrabberT::m_height = shmimMonitorT::m_height;
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;
   
   return 0;
}

inline
int photonCounter::startAcquisition()
{
   return 0;
}

inline
int photonCounter::acquireAndCheckValid()
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
int photonCounter::loadImageIntoStream(void * dest)
{
   //Here is where we do it.   
   Eigen::Map<eigenImage<float>> photon_counted_out(static_cast<uint16_t*>(dest), frameGrabberT::m_width, frameGrabberT::m_height );
   
	// Copy the data into the stream
	for(uint32_t col_i=0; col_i < m_image_width; ++col_i){
		for(uint32_t row_i=0; row_i < m_image_height; ++row_i){
			
			photon_counted_out(row_i, col_i) = m_photonCountedImage(row_i, col_i);

			m_photonCountedImage(row_i, col_i) = 0.0;

		}
	}

   return 0;
}

inline
int photonCounter::reconfig()
{
   return 0;
}


INDI_NEWCALLBACK_DEFN(photonCounter, m_indiP_calibrateToggle )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_calibrateToggle.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		
		if(!m_calibrate){
			std::cout << "Request calibration" << std::endl;
			m_calibrationSet = false;
			m_calibrate = true;
		}

		updateSwitchIfChanged(m_indiP_calibrateToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(photonCounter, m_indiP_quantileCut )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_quantileCut.getName()){
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	float current = -1;
	float target = -1;

	if(ipRecv.find("current"))
		current = ipRecv["current"].get<float>();

	if(ipRecv.find("target"))
		target = ipRecv["target"].get<float>();

	if(target == -1) target = current;

	if(target == -1)
		return 0;

	std::lock_guard<std::mutex> guard(m_indiMutex);

	m_quantile_cut = target;
	updateIfChanged(m_indiP_quantileCut, "target", m_quantile_cut);

	return 0;
}

INDI_NEWCALLBACK_DEFN(photonCounter, m_indiP_calibrateSteps )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_calibrateSteps.getName())
	{
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	int current = -1;
	int target = -1;

	if(ipRecv.find("current"))
	{
		current = ipRecv["current"].get<int>();
	}

	if(ipRecv.find("target"))
	{
		target = ipRecv["target"].get<int>();
	}

	if(target == -1) target = current;

	if(target == -1)
	{
		return 0;
	}

	std::lock_guard<std::mutex> guard(m_indiMutex);

	calibration_steps = target;
	m_calibrationCube.resize(m_image_width, m_image_height, calibration_steps);
	m_calibrationCube.setZero();

	updateIfChanged(m_indiP_calibrateSteps, "target", calibration_steps);

	return 0;
}

INDI_NEWCALLBACK_DEFN(photonCounter, m_indiP_stackFrames )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_stackFrames.getName())
	{
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	int current = -1;
	int target = -1;

	if(ipRecv.find("current"))
	{
		current = ipRecv["current"].get<int>();
	}

	if(ipRecv.find("target"))
	{
		target = ipRecv["target"].get<int>();
	}

	if(target == -1) target = current;

	if(target == -1)
	{
		return 0;
	}

	std::lock_guard<std::mutex> guard(m_indiMutex);

	m_stack_frames = target;
	updateIfChanged(m_indiP_stackFrames, "target", m_stack_frames);

	return 0;
}

} //namespace app
} //namespace MagAOX

#endif //photonCounter_hpp
