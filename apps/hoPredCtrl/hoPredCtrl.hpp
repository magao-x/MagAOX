/** \file hoPredCtrl.hpp
  * \brief The MagAO-X tweeter to woofer offloading manager
  *
  * \ingroup app_files
  */

#ifndef hoPredCtrl_hpp
#define hoPredCtrl_hpp

#include <iostream>
#include <limits>
#include <chrono>
#include <thread>


#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "predictive_controller.cuh"

using namespace DDSPC;

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
	std::string m_interaction_matrix_filename;

	// IMAGE m_dmStream; 
	size_t m_pwfsWidth {0}; ///< The width of the image
	size_t m_pwfsHeight {0}; ///< The height of the image.

	size_t m_quadWidth {0}; ///< The width of the image
	size_t m_quadHeight {0}; ///< The height of the image.

	uint8_t m_pwfsDataType{0}; ///< The ImageStreamIO type code.
	size_t m_pwfsTypeSize {0}; ///< The size of the type, in bytes.  
	
	unsigned long long duration;
	unsigned long long iterations;

	// The wavefront sensor variables
	size_t m_illuminatedPixels;
	size_t m_measurement_size;
	eigenImage<realT> m_pupilMask;
	eigenImage<realT> m_interaction_matrix;
	eigenImage<realT> m_measurementVector;
	realT (*pwfs_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.

	// The dark image parameters
	eigenImage<realT> m_darkImage;
	realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
	bool m_darkSet {false};

	// Predictive control parameters
	float m_clip_val;
	int m_numModes;
	int m_numHist;			///< The number of past states to use for the prediction
	int m_numFut;			///< The number of future states to predict
	realT m_gamma;			///< The forgetting factore (0, 1]
	realT m_inv_covariance; ///< The starting point of the inverse covariance matrix
	realT m_lambda;		///< The regularization parameter
	
	DDSPC::PredictiveController* controller;

	// Learning time
	int m_exploration_steps;
	int m_learning_counter;
	float m_exploration_rms;

	bool m_is_closed_loop;

	// Interaction for the DM
	float* m_command;
	// If true we can just use a mask to map the controlled subset of actuators to the 50x50 shmim.
	// Otherwise we should use another matrix that maps mode coefficients to DM actuator patterns.
	bool use_actuators;
	eigenImage<realT> m_illuminated_actuators_mask;
	eigenImage<realT> m_shaped_command;	// 50x50

	std::string m_dmChannel;
	IMAGE m_dmStream; 
	uint32_t m_dmWidth {0}; ///< The width of the image
	uint32_t m_dmHeight {0}; ///< The height of the image.
	
	uint8_t m_dmDataType{0}; ///< The ImageStreamIO type code.
	size_t m_dmTypeSize {0}; ///< The size of the type, in bytes.  
	
	bool m_dmOpened {false};
	bool m_dmRestart {false};


	pcf::IndiProperty m_indiP_controlToggle;
	pcf::IndiProperty m_indiP_reset_bufferRequest;
	pcf::IndiProperty m_indiP_reset_modelRequest;

	pcf::IndiProperty m_indiP_learningSteps;
	
	// The control parameters
	pcf::IndiProperty m_indiP_lambda;
	// pcf::IndiProperty m_indiP_gamma;
	// pcf::IndiProperty m_indiP_inv_cov;

	/*
		TODO:
			2) Add a check for closed-loop operation in model/buffer reset.
			3) Make clip value accesible through INDI.
			4) Set gamma and inverse covariance.
			5) Create a Double variant? Use a typedef for the variables.
			6) Create a GUI.
			7) Update reconstruction matrix.
			8) Update pupil mask.
			9) Add a toggle to create an empty loop?
	*/

	// Control states
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_controlToggle);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_bufferRequest);
    INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_modelRequest);

	// Control parameters
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_learningSteps);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_lambda);

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
   int send_dm_command();
   int map_command_vector_to_dmshmim();
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

	config.add("parameters.pupil_mask", "", "parameters.pupil_mask", argType::Required, "parameters", "pupil_mask", false, "string", "The path to the PyWFS pupil mask.");
	config.add("parameters.interaction_matrix", "", "parameters.interaction_matrix", argType::Required, "parameters", "interaction_matrix", false, "string", "The path to the PyWFS interaction matrix.");
	config.add("parameters.Nhist", "", "parameters.Nhist", argType::Required, "parameters", "Nhist", false, "int", "The history length.");
	config.add("parameters.Nfut", "", "parameters.Nfut", argType::Required, "parameters", "Nfut", false, "int", "The prediction horizon.");
	config.add("parameters.Nmodes", "", "parameters.Nmodes", argType::Required, "parameters", "Nmodes", false, "int", "The number of modes to control.");
	config.add("parameters.gamma", "", "parameters.gamma", argType::Required, "parameters", "gamma", false, "float", "The prediction horizon.");
	config.add("parameters.inv_covariance", "", "parameters.inv_covariance", argType::Required, "parameters", "inv_covariance", false, "float", "The prediction horizon.");
	config.add("parameters.lambda", "", "parameters.lambda", argType::Required, "parameters", "lambda", false, "float", "The prediction horizon.");
	config.add("parameters.clip_val", "", "parameters.clip_val", argType::Required, "parameters", "clip_val", false, "float", "The update clip value.");
	
	//
	config.add("parameters.exploration_steps", "", "parameters.exploration_steps", argType::Required, "parameters", "exploration_steps", false, "int", "The update clip value.");
	config.add("parameters.exploration_rms", "", "parameters.exploration_rms", argType::Required, "parameters", "exploration_rms", false, "float", "The update clip value.");
  
	config.add("parameters.channel", "", "parameters.channel", argType::Required, "parameters", "channel", false, "string", "The DM channel to control.");
}

inline
int hoPredCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
	shmimMonitorT::loadConfig(_config);
	darkMonitorT::loadConfig(_config);

	_config(m_pupilMaskFilename, "parameters.pupil_mask");
	std::cout << m_pupilMaskFilename << std::endl;
	_config(m_interaction_matrix_filename, "parameters.interaction_matrix");
	std::cout << m_interaction_matrix_filename << std::endl;

	_config(m_numHist, "parameters.Nhist");
	std::cout << "Nhist:: "<< m_numHist << std::endl;
	_config(m_numFut, "parameters.Nfut");
	std::cout << "Nfut:: "<< m_numFut << std::endl;
	_config(m_numModes, "parameters.Nmodes");
	std::cout << "Nmodes:: "<< m_numModes << std::endl;

	_config(m_gamma, "parameters.gamma");
	_config(m_inv_covariance, "parameters.inv_covariance");
	_config(m_lambda, "parameters.lambda");
	_config(m_clip_val, "parameters.clip_val");

	_config(m_exploration_steps, "parameters.exploration_steps");
	_config(m_exploration_rms, "parameters.exploration_rms");
	std::cout << "Nexplore:: "<< m_exploration_steps << " with " << m_exploration_rms << " rms." << std::endl;

	_config(m_dmChannel, "parameters.channel");

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

	createStandardIndiToggleSw( m_indiP_controlToggle, "control", "Control State", "Loop Controls");
	registerIndiPropertyNew( m_indiP_controlToggle, INDI_NEWCALLBACK(m_indiP_controlToggle) ); 

	createStandardIndiRequestSw( m_indiP_reset_bufferRequest, "reset_buffer", "Reset the data buffers", "Loop Controls");
	registerIndiPropertyNew( m_indiP_reset_bufferRequest, INDI_NEWCALLBACK(m_indiP_reset_bufferRequest) ); 

	createStandardIndiRequestSw( m_indiP_reset_modelRequest, "reset_model", "Reset the RLS model", "Loop Controls");
	registerIndiPropertyNew( m_indiP_reset_modelRequest, INDI_NEWCALLBACK(m_indiP_reset_modelRequest) ); 

	createStandardIndiNumber<int>( m_indiP_learningSteps, "learning_steps", -1, 20000, 1, "%d", "Learning Steps", "Learning control");
	registerIndiPropertyNew( m_indiP_learningSteps, INDI_NEWCALLBACK(m_indiP_learningSteps) );  

	// createStandardIndiNumber<int>( m_indiP_gamma, "gamma", 0, 1.0, 0.0001, "%0.3f", "Forgetting parameter", "Learning control");
	// registerIndiPropertyNew( m_indiP_gamma, INDI_NEWCALLBACK(m_indiP_gamma) );  

	createStandardIndiNumber<float>( m_indiP_lambda, "lambda", 0, 100.0, 0.0001, "%0.3f", "Regularization", "Learning control");
	registerIndiPropertyNew( m_indiP_lambda, INDI_NEWCALLBACK(m_indiP_lambda) );  

	// createStandardIndiNumber<int>( m_indiP_inv_cov, "lambda", 0, 1e8, 0.1, "%0.3f", "Inverse Covariance", "Learning control");
	// registerIndiPropertyNew( m_indiP_inv_cov, INDI_NEWCALLBACK(m_indiP_inv_cov) );  



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

	// Send updates to indi if the values changed due to changes caused by software

	updateIfChanged(m_indiP_learningSteps, "current", m_learning_counter);
	// updateIfChanged(m_indiP_gamma, "current", m_gamma);
	updateIfChanged(m_indiP_lambda, "current", m_lambda);
	// updateIfChanged(m_indiP_inv_cov, "current", m_inv_covariance);

	if(m_is_closed_loop){
		updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	}else{
		updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}


	return 0;
}

inline
int hoPredCtrl::appShutdown()
{
	shmimMonitorT::appShutdown();

	darkMonitorT::appShutdown();

	// delete controller;

	delete 	[] m_command;
	return 0;
}

inline
int hoPredCtrl::allocate(const dev::shmimT & dummy)
{

	static_cast<void>(dummy); //be unused

	// Wavefront sensor setup
	m_pwfsWidth = shmimMonitorT::m_width;
	m_quadWidth = m_pwfsWidth / 2;
	std::cout << "Width " << m_pwfsWidth << std::endl;

	m_pwfsHeight = shmimMonitorT::m_height;
	m_quadHeight = m_pwfsHeight / 2;
	std::cout << "Height " << m_pwfsHeight << std::endl;

	set_pupil_mask(m_pupilMaskFilename);
	
	// Controller setup
	controller = new DDSPC::PredictiveController(m_numHist, m_numFut, m_numModes, m_measurement_size, m_gamma, m_lambda, m_inv_covariance);

	// Read in the pupil mask
	mx::fits::fitsFile<realT> ff;
	ff.read(m_interaction_matrix, m_interaction_matrix_filename);
	std::cerr << "Read a " << m_interaction_matrix.rows() << " x " << m_interaction_matrix.cols() << " interaction matrix.\n";
	controller->set_interaction_matrix(m_interaction_matrix.data());

	m_command = new float[m_numModes];

	// controller->create_exploration_buffer(m_exploration_rms, m_exploration_steps);
	m_learning_counter = -1;

	//Initialize dark image if not correct size.
	if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height){
		m_darkImage.resize(shmimMonitorT::m_width,shmimMonitorT::m_height);
		m_darkImage.setZero();
		m_darkSet = false;
	}

	// Allocate the DM
	if(m_dmOpened){
		ImageStreamIO_closeIm(&m_dmStream);
	}
	
	m_dmOpened = false;
	m_dmRestart = false; //Set this up front, since we're about to restart.
		
	if( ImageStreamIO_openIm(&m_dmStream, m_dmChannel.c_str()) == 0){
		if(m_dmStream.md[0].sem < 10){
			ImageStreamIO_closeIm(&m_dmStream);
		}else{
			m_dmOpened = true;
		}
	}
		
	if(!m_dmOpened){
		log<software_error>({__FILE__, __LINE__, m_dmChannel + " not opened."});
		return -1;
	}else{
		m_dmWidth = m_dmStream.md->size[0]; 
		m_dmHeight = m_dmStream.md->size[1]; 

		m_dmDataType = m_dmStream.md->datatype;
		m_dmTypeSize = ImageStreamIO_typesize(m_dmDataType);
		
		log<text_log>( "Opened " + m_dmChannel + " " + std::to_string(m_dmWidth) + " x " + std::to_string(m_dmHeight) + " with data type: " + std::to_string(m_dmDataType)); 
		
		m_shaped_command.resize(m_dmWidth, m_dmHeight);
		m_shaped_command.setZero();
	}

	duration = 0;
	iterations = 0;

   return 0;
}

inline
int hoPredCtrl::processImage( void * curr_src, 
                                const dev::shmimT & dummy 
                              )
{
	static_cast<void>(dummy); //be unused
	auto start = std::chrono::steady_clock::now();

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
	
	// If true the DDSPC algorithm is running
	// Should I extend to all operations in this loop ?
	if( m_is_closed_loop ){

		controller->add_measurement(m_measurementVector.data());
		
		// If -1 always learn, otherwise learn for N steps
		if(m_learning_counter == -1){

			controller->update_predictor();
			controller->update_controller();

		}else if(m_learning_counter > 0){

			// std::cout << m_learning_counter << std::endl;
			controller->update_predictor();
			controller->update_controller();
			m_learning_counter -= 1;

		}

		m_command = controller->get_command(m_clip_val);

		if(use_actuators){
			map_command_vector_to_dmshmim();
		}
		send_dm_command();

	}
	auto end = std::chrono::steady_clock::now();
	
	duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	iterations += 1;
	
	if(iterations % 2000 == 0)
		std::cout << "elapsed " << (double)duration / (double)iterations << " us." << std::endl;

	// std::chrono::milliseconds timespan(10); // or whatever
	// std::this_thread::sleep_for(timespan);

	// std::cout << "Average counts: " << mean_value/(shmimMonitorT::m_width * shmimMonitorT::m_height) << std::endl;

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
	
	m_measurement_size = 3 * m_illuminatedPixels;

	std::cout << "Number of illuminated pixels :: " << m_illuminatedPixels << std::endl;
	std::cout << "Measurement vector size :: " << m_measurement_size << std::endl;

	// Create the measurement vector
	m_measurementVector.resize(m_measurement_size, 1);
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
      //data[nn] = *( (int16_t * ) (curr_src + nn*shmimMonitorT::m_dmDataType));
      data[nn] = dark_pixget(curr_src, nn);
   }
   
   m_darkSet = true;
   
   return 0;
}


inline
int hoPredCtrl::zero()
{
	m_shaped_command.setZero();
	// Reset buffer and controller too?

	return 0;
      
}

inline
int hoPredCtrl::map_command_vector_to_dmshmim(){
	// Convert the 2000 modes into a 50x50 image.
	/*
		This function maps the command vector to a masked 2D image. A mapping is implicitely assumed due to the way the array is accessed.
	*/
	int ki = 0;
	for(uint32_t col_i=0; col_i < m_dmHeight; ++col_i){
		for(uint32_t row_i=0; row_i < m_dmHeight; ++row_i){
			if(m_illuminated_actuators_mask(row_i, col_i) > 0.5){
				m_shaped_command(row_i, col_i) = m_command[ki];
				ki += 1;
			}
		}
	}
}

inline
int hoPredCtrl::send_dm_command(){

	//Check if processImage is running
	while(m_dmStream.md[0].write == 1);

	m_dmStream.md[0].write = 1;
	memcpy(m_dmStream.array.raw, m_shaped_command.data(),  m_shaped_command.rows() * m_shaped_command.cols() * m_dmDataType);

	m_dmStream.md[0].cnt0++;

	m_dmStream.md->write=0;
	ImageStreamIO_sempost(&m_dmStream,-1);

	// log<text_log>("zeroed", logPrio::LOG_NOTICE);

	return 0;

}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_learningSteps )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_learningSteps.getName())
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

	m_learning_counter = target;

	updateIfChanged(m_indiP_learningSteps, "target", m_learning_counter);

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_lambda )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_lambda.getName()){
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
	if(!m_is_closed_loop){
		m_lambda = target;
		controller->set_new_regularization(m_lambda);
		updateIfChanged(m_indiP_lambda, "target", m_lambda);
	}else{
		 log<text_log>("Lambda not changed. Loop is still running.", logPrio::LOG_NOTICE);
	}

	return 0;
}
/*
INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_inv_cov )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_inv_cov.getName()){
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

	m_inv_covariance = target;

	updateIfChanged(m_indiP_inv_cov, "target", m_inv_covariance);

	return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_gamma )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_gamma.getName()){
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

	m_gamma = target;
	
	updateIfChanged(m_indiP_gamma, "target", m_gamma);

	return 0;
}
*/

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_controlToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_controlToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!m_is_closed_loop) //not offloading so change
      {
		// m_woofer.setZero(); //always zero when offloading starts
		// log<text_log>("zeroed", logPrio::LOG_NOTICE);
		// m_offloading = true;
		m_is_closed_loop = true;
		log<text_log>("started closed-loop operation", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);

      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(m_is_closed_loop) //offloading so change it
      {
         m_is_closed_loop = false;
         log<text_log>("stopped closed-loop operation", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_reset_bufferRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_reset_bufferRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);
		controller->reset_data_buffer();
		updateSwitchIfChanged(m_indiP_reset_bufferRequest, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_reset_modelRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_reset_modelRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);
		controller->reset_controller();
		updateSwitchIfChanged(m_indiP_reset_bufferRequest, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //hoPredCtrl_hpp
