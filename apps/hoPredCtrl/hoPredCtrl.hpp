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
   
	std::string m_pupilMaskFilename;			// aol1_wfsmask.fits
	std::string m_interaction_matrix_filename; 	// aol1_modesWFS.fits in cacao
	std::string m_mapping_matrix_filename; 		// aol1_DMmodes.fits
	std::string m_refWavefront_filename;		// aol1_wfsref.fits 
	uint64_t loading_timestamp;

	// std::string m_actuator_mask_filename;		// 
	
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
	
	eigenImage<realT> m_interaction_matrix;
	eigenImage<realT> m_mapping_matrix;

	eigenImage<realT> m_pupilMask;
	eigenImage<realT> m_measurementVector;
	eigenImage<realT> m_refWavefront;
	
	realT average_pupil_intensity;

	bool use_full_image_reconstructor;

	realT (*pwfs_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.

	// The dark image parameters
	eigenImage<realT> m_darkImage;
	realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
	bool m_darkSet {false};

	// Predictive control parameters
	float m_clip_val;
	int m_numModes;
	int m_numVoltages;
	int m_numHist;			///< The number of past states to use for the prediction
	int m_numFut;			///< The number of future states to predict
	realT m_gamma;			///< The forgetting factore (0, 1)
	
	realT m_inv_covariance; ///< The starting point of the inverse covariance matrix
	realT m_lambda;		///< The regularization parameter
	
	// Integrator commands
	bool m_use_predictive_control;
	realT m_intgain;
	realT m_intleak;

	DDSPC::PredictiveController* controller;

	// Learning time
	int m_exploration_steps;
	int m_learning_counter;
	int m_learning_steps;
	float m_exploration_rms;
	int m_learning_iterations;

	bool m_is_closed_loop;

	// Interaction for the DM
	float* m_command;
	float* m_temp_command;

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

	std::string savepath;

	pcf::IndiProperty m_indiP_controlToggle;
	pcf::IndiProperty m_indiP_predictorToggle;
	pcf::IndiProperty m_indiP_reset_bufferRequest;
	pcf::IndiProperty m_indiP_reset_modelRequest;
	pcf::IndiProperty m_indiP_reset_cleanRequest;

	pcf::IndiProperty m_indiP_updateControllerRequest;

	pcf::IndiProperty m_indiP_learningSteps;
	pcf::IndiProperty m_indiP_learningIterations;
	pcf::IndiProperty m_indiP_explorationRms;
	pcf::IndiProperty m_indiP_explorationSteps;
	pcf::IndiProperty m_indiP_reset_exploreRequest;
	pcf::IndiProperty m_indiP_zeroRequest;
	
	pcf::IndiProperty m_indiP_saveRequest;
	pcf::IndiProperty m_indiP_loadRequest;
	pcf::IndiProperty m_indiP_timestamp;

	// The control parameters
	pcf::IndiProperty m_indiP_lambda;
	pcf::IndiProperty m_indiP_clipval;
	pcf::IndiProperty m_indiP_gamma;

	// Integrator parameters
	pcf::IndiProperty m_indiP_intgain;
	pcf::IndiProperty m_indiP_intleak;

	// pcf::IndiProperty m_indiP_inv_cov;

	/*
		TODO:
			5) Create a Double variant? Use a typedef for the variables.
			6) Create a GUI.
			7) Update reconstruction matrix.
			8) Update pupil mask.
			9) Add a toggle to create an empty loop?
			10) Save and load model.
			11) Reset to loaded model.
	*/

	// Control states
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_controlToggle);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_bufferRequest);
    INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_modelRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_cleanRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_updateControllerRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_reset_exploreRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_zeroRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_saveRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_loadRequest);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_timestamp);
	

	// Control parameters
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_learningSteps);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_learningIterations);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_explorationRms);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_explorationSteps);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_lambda);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_clipval);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_gamma);

	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_predictorToggle);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_intgain);
	INDI_NEWCALLBACK_DECL(hoPredCtrl, m_indiP_intleak);

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

	void save_state(){};

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

	config.add("parameters.calib_directory", "", "parameters.calib_directory", argType::Required, "parameters", "calib_directory", false, "string", "The path to the PyWFS pupil mask.");
	config.add("parameters.pupil_mask", "", "parameters.pupil_mask", argType::Required, "parameters", "pupil_mask", false, "string", "The path to the PyWFS pupil mask.");
	config.add("parameters.interaction_matrix", "", "parameters.interaction_matrix", argType::Required, "parameters", "interaction_matrix", false, "string", "The path to the PyWFS interaction matrix.");
	config.add("parameters.mapping_matrix", "", "parameters.mapping_matrix", argType::Required, "parameters", "mapping_matrix", false, "string", "The path to the DM mapping matrix.");
	config.add("parameters.act_mask", "", "parameters.act_mask", argType::Required, "parameters", "act_mask", false, "string", "The path to the PyWFS interaction matrix.");
	config.add("parameters.reference_image", "", "parameters.reference_image", argType::Required, "parameters", "reference_image", false, "string", "The path to the PyWFS interaction matrix.");


	config.add("parameters.Nhist", "", "parameters.Nhist", argType::Required, "parameters", "Nhist", false, "int", "The history length.");
	config.add("parameters.Nfut", "", "parameters.Nfut", argType::Required, "parameters", "Nfut", false, "int", "The prediction horizon.");
	config.add("parameters.gamma", "", "parameters.gamma", argType::Required, "parameters", "gamma", false, "float", "The prediction horizon.");
	config.add("parameters.inv_covariance", "", "parameters.inv_covariance", argType::Required, "parameters", "inv_covariance", false, "float", "The prediction horizon.");
	config.add("parameters.lambda", "", "parameters.lambda", argType::Required, "parameters", "lambda", false, "float", "The prediction horizon.");
	config.add("parameters.clip_val", "", "parameters.clip_val", argType::Required, "parameters", "clip_val", false, "float", "The update clip value.");
	
	//
	config.add("parameters.learning_steps", "", "parameters.learning_steps", argType::Required, "parameters", "learning_steps", false, "int", "The update clip value.");
	config.add("parameters.learning_iterations", "", "parameters.learning_iterations", argType::Required, "parameters", "learning_iterations", false, "int", "The amount of learning cycles.");
	config.add("parameters.exploration_steps", "", "parameters.exploration_steps", argType::Required, "parameters", "exploration_steps", false, "int", "The update clip value.");
	config.add("parameters.exploration_rms", "", "parameters.exploration_rms", argType::Required, "parameters", "exploration_rms", false, "float", "The update clip value.");

	// Read in the learning parameters as a vector.
	// config.add("parameters.exploration_steps", "", "parameters.exploration_steps",  argType::Required, "parameters", "exploration_steps", false, "vector<int>", "The number of steps for each training iteration.");
	// config.add("parameters.exploration_rms", "", "parameters.exploration_rms",  argType::Required, "parameters", "exploration_rms", false, "vector<double>", "The rms for each training iteration.");
	// config.add("parameters.exploration_lambda", "", "parameters.exploration_lambda",  argType::Required, "parameters", "exploration_lambda", false, "vector<double>", "The regularization for each training iteration.");

	//
	config.add("parameters.channel", "", "parameters.channel", argType::Required, "parameters", "channel", false, "string", "The DM channel to control.");

	// The integrator parameters
	config.add("integrator.gain", "", "integrator.gain", argType::Required, "integrator", "gain", false, "float", "The integrator gain value.");
	config.add("integrator.leakage", "", "integrator.leakage", argType::Required, "integrator", "leakage", false, "float", "The integrator leakage.");
}

inline
int hoPredCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
	shmimMonitorT::loadConfig(_config);
	darkMonitorT::loadConfig(_config);

	// The integrator control parameters
	_config(m_intgain, "integrator.gain");
	_config(m_intleak, "integrator.leakage");

	// Calibration files
	std::string calibration_directory;
	_config(calibration_directory, "parameters.calib_directory");
	
	_config(m_pupilMaskFilename, "parameters.pupil_mask");
	m_pupilMaskFilename = calibration_directory + m_pupilMaskFilename;
	std::cout << m_pupilMaskFilename << std::endl;

	_config(m_interaction_matrix_filename, "parameters.interaction_matrix");
	m_interaction_matrix_filename = calibration_directory + m_interaction_matrix_filename;
	std::cout << m_interaction_matrix_filename << std::endl;

	_config(m_mapping_matrix_filename, "parameters.mapping_matrix");
	m_mapping_matrix_filename = calibration_directory + m_mapping_matrix_filename;
	std::cout << m_mapping_matrix_filename << std::endl;

	_config(m_refWavefront_filename, "parameters.reference_image");
	m_refWavefront_filename = calibration_directory + m_refWavefront_filename;
	std::cout << m_refWavefront_filename << std::endl;
	
	// Controller parameters
	_config(m_numHist, "parameters.Nhist");
	std::cout << "Nhist="<< m_numHist << std::endl;
	_config(m_numFut, "parameters.Nfut");
	std::cout << "Nfut="<< m_numFut << std::endl;

	_config(m_gamma, "parameters.gamma");
	std::cout << "gamma="<< m_gamma << std::endl;
	_config(m_inv_covariance, "parameters.inv_covariance");
	std::cout << "inv_covariance="<< m_inv_covariance << std::endl;
	_config(m_lambda, "parameters.lambda");
	std::cout << "lambda="<< m_lambda << std::endl;
	_config(m_clip_val, "parameters.clip_val");
	std::cout << "clip_val="<< m_clip_val << std::endl;

	// The learning parameters
	_config(m_exploration_steps, "parameters.exploration_steps");
	_config(m_learning_steps, "parameters.learning_steps");
	m_learning_counter = m_learning_steps;
	_config(m_learning_iterations, "parameters.learning_iterations");

	_config(m_exploration_rms, "parameters.exploration_rms");
	std::cout << "Nexplore:: "<< m_exploration_steps << " with " << m_exploration_rms << " rms." << std::endl;

	_config(m_dmChannel, "parameters.channel");
	std::cout << "Open DM tweeter channel at " << m_dmChannel << std::endl;
	std::cout << "Done reading config Impl." << std::endl;

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
	
	createStandardIndiRequestSw( m_indiP_reset_cleanRequest, "clean", "Clean the complete model.", "Loop Controls");
	registerIndiPropertyNew( m_indiP_reset_cleanRequest, INDI_NEWCALLBACK(m_indiP_reset_cleanRequest) ); 

	createStandardIndiRequestSw( m_indiP_updateControllerRequest, "calc_controller", "Update the controller.", "Loop Controls");
	registerIndiPropertyNew( m_indiP_updateControllerRequest, INDI_NEWCALLBACK(m_indiP_updateControllerRequest) ); 

	createStandardIndiRequestSw( m_indiP_reset_exploreRequest, "reset_exploration", "Reset the exploration model", "Loop Controls");
	registerIndiPropertyNew( m_indiP_reset_exploreRequest, INDI_NEWCALLBACK(m_indiP_reset_exploreRequest) ); 

	createStandardIndiRequestSw( m_indiP_zeroRequest, "zero", "Zero the dm", "Loop Controls");
	registerIndiPropertyNew( m_indiP_zeroRequest, INDI_NEWCALLBACK(m_indiP_zeroRequest) ); 

	createStandardIndiRequestSw( m_indiP_saveRequest, "save", "Save the controller", "Loop Controls");
	registerIndiPropertyNew( m_indiP_saveRequest, INDI_NEWCALLBACK(m_indiP_saveRequest) ); 

	createStandardIndiRequestSw( m_indiP_loadRequest, "load", "Load the controller", "Loop Controls");
	registerIndiPropertyNew( m_indiP_loadRequest, INDI_NEWCALLBACK(m_indiP_loadRequest) ); 
	
	createStandardIndiNumber<uint64_t>( m_indiP_timestamp, "timestamp", -1, 20000000000000000, 1, "%d", "Timestamp", "Loading the controller");
	registerIndiPropertyNew( m_indiP_timestamp, INDI_NEWCALLBACK(m_indiP_timestamp) );  

	createStandardIndiNumber<int>( m_indiP_learningSteps, "learning_steps", -1, 200000, 1, "%d", "Learning Steps", "Learning control");
	registerIndiPropertyNew( m_indiP_learningSteps, INDI_NEWCALLBACK(m_indiP_learningSteps) );  

	createStandardIndiNumber<int>( m_indiP_learningSteps, "learning_steps", -1, 200000, 1, "%d", "Learning Steps", "Learning control");
	registerIndiPropertyNew( m_indiP_learningSteps, INDI_NEWCALLBACK(m_indiP_learningSteps) );  

	createStandardIndiNumber<int>( m_indiP_learningIterations, "learning_iterations", -1, 200000, 1, "%d", "Learning iterations", "Learning control");
	registerIndiPropertyNew( m_indiP_learningIterations, INDI_NEWCALLBACK(m_indiP_learningIterations) );  

	createStandardIndiNumber<float>( m_indiP_explorationRms, "exploration_rms", 0.0, 1.0, 0.00001, "%0.4f", "Learning Steps", "Learning control");
	registerIndiPropertyNew( m_indiP_explorationRms, INDI_NEWCALLBACK(m_indiP_explorationRms) );  

	createStandardIndiNumber<float>( m_indiP_explorationSteps, "exploration_steps", 0, 200000, 1, "%d", "Exploration Steps", "Learning control");
	registerIndiPropertyNew( m_indiP_explorationSteps, INDI_NEWCALLBACK(m_indiP_explorationSteps) );  

	createStandardIndiNumber<float>( m_indiP_gamma, "gamma", 0, 1.0, 0.0001, "%0.3f", "Forgetting parameter", "Learning control");
	registerIndiPropertyNew( m_indiP_gamma, INDI_NEWCALLBACK(m_indiP_gamma) );  

	createStandardIndiNumber<float>( m_indiP_lambda, "lambda", 0, 1000.0, 0.0001, "%0.3f", "Regularization", "Learning control");
	registerIndiPropertyNew( m_indiP_lambda, INDI_NEWCALLBACK(m_indiP_lambda) );

	createStandardIndiNumber<float>( m_indiP_clipval, "clipval", 0, 1000.0, 0.0001, "%0.3f", "Regularization", "Learning control");
	registerIndiPropertyNew( m_indiP_clipval, INDI_NEWCALLBACK(m_indiP_clipval) );  

	// createStandardIndiNumber<int>( m_indiP_inv_cov, "lambda", 0, 1e8, 0.1, "%0.3f", "Inverse Covariance", "Learning control");
	// registerIndiPropertyNew( m_indiP_inv_cov, INDI_NEWCALLBACK(m_indiP_inv_cov) );  

	createStandardIndiToggleSw( m_indiP_predictorToggle, "use_predictor", "Choose controller", "Loop Controls");
	registerIndiPropertyNew( m_indiP_predictorToggle, INDI_NEWCALLBACK(m_indiP_predictorToggle) ); 
	
	createStandardIndiNumber<float>( m_indiP_intgain, "intgain", 0, 1.0, 0.0001, "%0.3f", "Integrator gain", "Learning control");
	registerIndiPropertyNew( m_indiP_intgain, INDI_NEWCALLBACK(m_indiP_intgain) );  
	
	createStandardIndiNumber<float>( m_indiP_intleak, "intleak", 0, 1.0, 0.0001, "%0.3f", "Integrator gain", "Learning control");
	registerIndiPropertyNew( m_indiP_intleak, INDI_NEWCALLBACK(m_indiP_intleak) );  

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
	updateIfChanged(m_indiP_learningIterations, "current", m_learning_iterations);
	updateIfChanged(m_indiP_explorationRms, "current", m_exploration_rms);
	updateIfChanged(m_indiP_explorationSteps, "current", m_exploration_steps);
	updateIfChanged(m_indiP_gamma, "current", m_gamma);
	updateIfChanged(m_indiP_lambda, "current", m_lambda);
	updateIfChanged(m_indiP_clipval, "current", m_clip_val);

	updateIfChanged(m_indiP_intgain, "current", m_intgain);
	updateIfChanged(m_indiP_intleak, "current", m_intleak);
	// updateIfChanged(m_indiP_inv_cov, "current", m_inv_covariance);

	if(m_is_closed_loop){
		updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	}else{
		updateSwitchIfChanged(m_indiP_controlToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}

	if(m_use_predictive_control){
		updateSwitchIfChanged(m_indiP_predictorToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	}else{
		updateSwitchIfChanged(m_indiP_predictorToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	}


	return 0;
}

inline
int hoPredCtrl::appShutdown()
{
	shmimMonitorT::appShutdown();

	darkMonitorT::appShutdown();

	m_shaped_command.setZero();
	send_dm_command();
	delete controller;

	if(m_temp_command){
		delete m_temp_command;
	}

	return 0;
}

inline
int hoPredCtrl::allocate(const dev::shmimT & dummy)
{
	static_cast<void>(dummy); //be unused
	
	// Or get from config
	use_full_image_reconstructor = true;

	// Wavefront sensor setup
	m_pwfsWidth = shmimMonitorT::m_width;
	m_quadWidth = m_pwfsWidth / 2;
	std::cout << "Width " << m_pwfsWidth << std::endl;

	m_pwfsHeight = shmimMonitorT::m_height;
	m_quadHeight = m_pwfsHeight / 2;
	std::cout << "Height " << m_pwfsHeight << std::endl;

	set_pupil_mask(m_pupilMaskFilename);
	std::cout << "Start reading in calibration files\n";

	// Read in the pupil mask
	mx::fits::fitsFile<realT> ff;
	eigenCube<realT>  temp_matrix;
	
	//
	ff.read(temp_matrix, m_interaction_matrix_filename);
	// std::cout << temp_matrix.shape() << "\n";
	std::cerr << "Read a " << temp_matrix.rows() << " x " << temp_matrix.cols() << " x " << temp_matrix.planes() << " interaction matrix.\n";
	m_interaction_matrix = temp_matrix.asVectors().matrix().transpose().array();

	//
	std::cerr << "Read a " << m_interaction_matrix.rows() << " x " << m_interaction_matrix.cols() << " interaction matrix.\n";
	m_numModes = m_interaction_matrix.rows();

	bool use_cacao_calib = true;
	if(use_cacao_calib){
		
		for(int row_i = 0; row_i < m_interaction_matrix.rows(); row_i++ ){
			
			realT norm = 0;
			for(int col_i = 0; col_i < m_interaction_matrix.cols(); col_i++ ){
				if(col_i==(25*30))
					std::cout << m_interaction_matrix(row_i, col_i) << std::endl;
				
				norm += m_interaction_matrix(row_i, col_i) * m_interaction_matrix(row_i, col_i);
			}

			std::cout << "Norm for mode " << row_i << " " << norm << std::endl;


			for(int col_i = 0; col_i < m_interaction_matrix.cols(); col_i++ ){
				m_interaction_matrix(row_i, col_i) = m_interaction_matrix(row_i, col_i) / norm;

				if(col_i==(25*30))
					std::cout << m_interaction_matrix(row_i, col_i) << std::endl;

				// 
			}

			realT test_norm = 0.0;
			for(int col_i = 0; col_i < m_interaction_matrix.cols(); col_i++ ){
				test_norm += m_interaction_matrix(row_i, col_i) * m_interaction_matrix(row_i, col_i);
			}
			std::cout << "Test Norm for mode " << row_i << " " << test_norm << std::endl;
		}	
		
	}

	// Read in the reference image
	ff.read(m_refWavefront, m_refWavefront_filename);
	std::cerr << "Read a " << m_refWavefront.rows() << " x " << m_refWavefront.cols() << " reference image.\n";

	// Read in the pupil mask
	ff.read(temp_matrix, m_mapping_matrix_filename);
	m_mapping_matrix = temp_matrix.asVectors();

	std::cerr << "Read a " << m_mapping_matrix.rows() << " x " << m_mapping_matrix.cols() << " mapping matrix.\n";
	m_numVoltages = m_mapping_matrix.rows();

	m_temp_command = new float[m_numVoltages];
	for(int i = 0; i < m_numVoltages; ++i)
		m_temp_command[i] = 0.001;
	m_command = m_temp_command;
	std::cerr << "Initialized temp command.\n";

	// Allocate the DM
	if(m_dmOpened){
		ImageStreamIO_closeIm(&m_dmStream);
	}

	m_dmOpened = false;
	m_dmRestart = false; //Set this up front, since we're about to restart.
	use_actuators = true;

	if( ImageStreamIO_openIm(&m_dmStream, m_dmChannel.c_str()) == 0){
		if(m_dmStream.md[0].sem < 10){
			ImageStreamIO_closeIm(&m_dmStream);
		}else{
			m_dmOpened = true;
		}
	}
		
	if(!m_dmOpened){
		// log<software_error>({__FILE__, __LINE__, m_dmChannel + " not opened."});

		log<text_log>( m_dmChannel + " not opened.", logPrio::LOG_NOTICE); 
		return -1;
	}else{
		m_dmWidth = m_dmStream.md->size[0]; 
		m_dmHeight = m_dmStream.md->size[1]; 

		m_dmDataType = m_dmStream.md->datatype;
		m_dmTypeSize = sizeof(float);
		
		log<text_log>( "Opened " + m_dmChannel + " " + std::to_string(m_dmWidth) + " x " + std::to_string(m_dmHeight) + " with data type: " + std::to_string(m_dmDataType), logPrio::LOG_NOTICE); 
		m_shaped_command.resize(m_dmWidth, m_dmHeight);
		std::cout << m_shaped_command.rows() << " x " << m_shaped_command.cols() << '\n';
		m_shaped_command.setZero();
		send_dm_command();
	}

	// Controller setup
	controller = new DDSPC::PredictiveController(m_numHist, m_numFut, m_numModes, m_measurement_size, m_gamma, m_lambda, m_inv_covariance, m_dmWidth * m_dmHeight);
	controller->set_interaction_matrix(m_interaction_matrix.data());
	controller->set_mapping_matrix(m_mapping_matrix.data());
	std::cerr << "Finished intializing the controller.\n";

	// mx::fits::fitsFile<realT> ff2;
	// ff2.read(m_illuminated_actuators_mask, m_actuator_mask_filename);
	// std::cerr << "Read a " << m_illuminated_actuators_mask.rows() << " x " << m_illuminated_actuators_mask.cols() << " actuator mask.\n";

	controller->create_exploration_buffer(m_exploration_rms, m_exploration_steps);
	std::cerr << "Initialized exploration buffer.\n";

	//Initialize dark image if not correct size.
	if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height){
		m_darkImage.resize(shmimMonitorT::m_width,shmimMonitorT::m_height);
		m_darkImage.setZero();
		m_darkSet = false;
	}

	duration = 0;
	iterations = 0;
	m_is_closed_loop = false;

	m_use_predictive_control = false;
	controller->controller->set_integrator(m_use_predictive_control, m_intgain, m_intleak);
	std::cerr << "Finished setup.\n";

	average_pupil_intensity = -100000.0;

	savepath = "/data/users/xsup/PredCtrlData/"; 
	return 0;
}

inline
int hoPredCtrl::processImage( void * curr_src, const dev::shmimT & dummy )
{
		
	static_cast<void>(dummy); //be unused
	auto start = std::chrono::steady_clock::now();
	
	Eigen::Map<eigenImage<unsigned short>> pwfsIm( static_cast<unsigned short *>(curr_src), m_pwfsHeight, m_pwfsWidth);
	// Calculate the norm
	realT pwfs_norm = 0;

	if(!use_full_image_reconstructor){
		// realT mean_value = 0;
		realT Ia = 0, Ib = 0, Ic = 0, Id = 0;
		realT total_norm = 0;
		size_t number_of_pixels = 0;
		
		size_t ki = 0;
		for(uint32_t col_i=0; col_i < m_quadWidth; ++col_i){
			for(uint32_t row_i=0; row_i < m_quadHeight; ++row_i){
				// Select the pixel from the correct quadrant and subtract dark			
				Ic = (realT)pwfsIm(row_i, col_i) - m_darkImage(row_i, col_i);															
				Id = (realT)pwfsIm(row_i + m_quadWidth, col_i) - m_darkImage(row_i + m_quadWidth, col_i);								
				Ia = (realT)pwfsIm(row_i, col_i + m_quadHeight) - m_darkImage(row_i, col_i + m_quadHeight);								
				Ib = (realT)pwfsIm(row_i + m_quadWidth, col_i + m_quadHeight) - m_darkImage(row_i + m_quadWidth, col_i + m_quadHeight);

				// Calculate the norm
				// TODO: Add an exponential learning to the PWFS norm?
				pwfs_norm = Ia + Ib + Ic + Id;
				
				// Take all linear combinations of the measurements and concatenate in vector
				if(m_pupilMask(row_i, col_i) > 0.5){
					m_measurementVector(ki, 0) = (Ia - Ib + Ic - Id) / pwfs_norm;
					m_measurementVector(ki + m_illuminatedPixels, 0) = (Ia + Ib - Ic - Id) / pwfs_norm;
					m_measurementVector(ki + 2 * m_illuminatedPixels, 0) = (Ia - Ib - Ic + Id) / pwfs_norm;
					++ki;
					total_norm += pwfs_norm;
				}
			}
		}
		
		total_norm /= ki;

	}else{

		for(uint32_t col_i=0; col_i < m_pwfsWidth; ++col_i){
			for(uint32_t row_i=0; row_i < m_pwfsHeight; ++row_i){
				pwfs_norm += m_pupilMask(row_i,col_i) * ((realT)pwfsIm(row_i, col_i) - m_darkImage(row_i, col_i));
			}
		}
		
		// Extract the illuminated pixels.
		size_t ki = 0;
		for(uint32_t col_i=0; col_i < m_pwfsWidth; ++col_i){
			for(uint32_t row_i=0; row_i < m_pwfsHeight; ++row_i){
				// Subtract a dark and the reference.
				m_measurementVector(ki, 0) = m_pupilMask(row_i,col_i) * (((realT)pwfsIm(row_i, col_i) - m_darkImage(row_i, col_i)) / pwfs_norm - m_refWavefront(row_i, col_i));
				++ki;
			}
		}

	}
	
	controller->add_measurement(m_measurementVector.data());

	// Okay reconstruction matrix is used correctly!
	// The error is now in the slope measurement.
	if( m_is_closed_loop ){
		m_command = controller->get_command(m_clip_val);

		// This works!
		if(use_actuators){
			map_command_vector_to_dmshmim();
		}
		send_dm_command();
				 
		// Only learn if we have a predictor
		if(m_use_predictive_control){

			// If -1 always learn, otherwise learn for N steps
			if(m_learning_counter == -1){
				
				controller->update_predictor();
				controller->update_controller();

			}else if(m_learning_counter > 0){
				
				controller->update_predictor();
				controller->update_controller();
				m_learning_counter -= 1;

			}

			if(m_learning_counter == 0 && m_learning_iterations > 0){
				m_learning_iterations--;
				m_learning_counter = m_learning_steps;

				m_lambda /= 10.0;
		
				controller->set_zero(); 														// set current control to zero
				zero();																			// set the DM shape to zero
				controller->create_exploration_buffer(m_exploration_rms, m_exploration_steps);  // Make a new buffer
				controller->set_new_regularization(m_lambda); 									// set a new regularization
				controller->reset_data_buffer();												// remove the history
			}
			

		}

	}

	auto end = std::chrono::steady_clock::now();
	

	if(iterations == 0){
		duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}else{
		duration = 0.95 * duration + 0.05 * std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	}
	iterations += 1;
	
	if(iterations % 10000 == 0){
		// std::cout << "PWFS NORM: " << pwfs_norm << std::endl;
		// std::cout << m_measurementVector(25*30,0)/1e-5 <<  " " << m_measurementVector(25*30+1,0)/1e-5 <<  " " << m_measurementVector(25*30+2,0)/1e-5 << std::endl;
		// controller->m_measurement->print(true);
		std::cout << "elapsed " << (double)duration << " us." << std::endl;
		std::cout << '\n';
	}

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
	realT * data = m_pupilMask.data();
   	m_illuminatedPixels = 0;

	if(use_full_image_reconstructor){
		m_measurement_size = m_pupilMask.rows() * m_pupilMask.cols();
	}else{
		for(size_t nn=0; nn < m_pupilMask.rows() * m_pupilMask.cols(); ++nn){
			if(data[nn] > 0.5)
				++m_illuminatedPixels;
		}
		m_measurement_size = 3 * m_illuminatedPixels;
	}

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
   
   m_darkImage.resize(darkMonitorT::m_width, darkMonitorT::m_height);
   
   dark_pixget = getPixPointer<realT>(darkMonitorT::m_dataType);
   
   if(dark_pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   std::cout << "Allocated dark frames stuff. \n";
	

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
	send_dm_command();
	return 0;
	
}

inline
int hoPredCtrl::map_command_vector_to_dmshmim(){
	
	// Convert the actuators modes into a 50x50 image.
	/*
		This function maps the command vector to a masked 2D image. A mapping is implicitely assumed due to the way the array is accessed.
	*/

	
	// The new output of the controller is a nact length vector. So this can be replaced by a single copy statement.
	// For now let's keep the dumb copy.
	int ki = 0;
	for(uint32_t col_i=0; col_i < m_dmHeight; ++col_i){
		for(uint32_t row_i=0; row_i < m_dmHeight; ++row_i){
			m_shaped_command(row_i, col_i) = m_command[ki];
			ki += 1;
		}
	}
	

	return 0;
}

inline
int hoPredCtrl::send_dm_command(){
	// Check if processImage is running
	// while(m_dmStream.md[0].write == 1);

	m_dmStream.md[0].write = 1;
	memcpy(m_dmStream.array.raw, m_shaped_command.data(),  2500 * sizeof(float));
	m_dmStream.md[0].cnt0++;
	m_dmStream.md[0].write = 0;

	ImageStreamIO_sempost(&m_dmStream,-1);

	// log<text_log>("zeroed", logPrio::LOG_NOTICE);W
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
	m_learning_steps = target;

	updateIfChanged(m_indiP_learningSteps, "target", m_learning_counter);

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_learningIterations )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_learningIterations.getName())
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

	m_learning_iterations = target;

	updateIfChanged(m_indiP_learningIterations, "target", m_learning_iterations);

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_explorationRms )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_explorationRms.getName())
	{
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	float current = -1;
	float target = -1;

	if(ipRecv.find("current"))
	{
		current = ipRecv["current"].get<float>();
	}

	if(ipRecv.find("target"))
	{
		target = ipRecv["target"].get<float>();
	}

	if(target == -1) target = current;

	if(target == -1)
	{
		return 0;
	}

	std::lock_guard<std::mutex> guard(m_indiMutex);

	m_exploration_rms = target;
	std::cout << "New expl. rms: " << m_exploration_rms << "\n";

	updateIfChanged(m_indiP_explorationRms, "target", m_exploration_rms);

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_explorationSteps )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_explorationSteps.getName())
	{
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	float current = -1;
	float target = -1;

	if(ipRecv.find("current"))
	{
		current = ipRecv["current"].get<float>();
	}

	if(ipRecv.find("target"))
	{
		target = ipRecv["target"].get<float>();
	}

	if(target == -1) target = current;

	if(target == -1)
	{
		return 0;
	}

	std::lock_guard<std::mutex> guard(m_indiMutex);

	m_exploration_steps = target;

	updateIfChanged(m_indiP_explorationSteps, "target", m_exploration_steps);

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

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_clipval )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_clipval.getName()){
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
		m_clip_val = target;
		updateIfChanged(m_indiP_clipval, "target", m_clip_val);
	}else{
		 log<text_log>("Clip value not changed. Loop is still running.", logPrio::LOG_NOTICE);
	}

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

	if(!m_is_closed_loop){
		m_gamma = target;
		
		controller->set_new_gamma(m_gamma);
		updateIfChanged(m_indiP_gamma, "target", m_gamma);
	}else{
		 log<text_log>("Gamma value not changed. Loop is still running.", logPrio::LOG_NOTICE);
	}

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_intgain )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_intgain.getName()){
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


	m_intgain = target;
	controller->controller->set_integrator(m_use_predictive_control, m_intgain, m_intleak);
	updateIfChanged(m_indiP_intgain, "target", m_intgain);
	
	// if(!m_is_closed_loop){
	// }else{
	// 	 log<text_log>("Integrator gain value not changed. Loop is still running.", logPrio::LOG_NOTICE);
	// }

	return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_intleak )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_intleak.getName()){
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

	
	m_intleak = target;
	controller->controller->set_integrator(m_use_predictive_control, m_intgain, m_intleak);
	updateIfChanged(m_indiP_intleak, "target", m_intleak);
	
	// if(!m_is_closed_loop){
	// }else{
	// 	 log<text_log>("Integrator leakage value not changed. Loop is still running.", logPrio::LOG_NOTICE);
	// }

	return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_timestamp )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_timestamp.getName()){
		log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
		return -1;
	}

	uint64_t current = -1;
	uint64_t target = -1;

	if(ipRecv.find("current"))
		current = ipRecv["current"].get<uint64_t>();

	if(ipRecv.find("target"))
		target = ipRecv["target"].get<uint64_t>();

	if(target == -1) target = current;

	if(target == -1)
		return 0;

	std::lock_guard<std::mutex> guard(m_indiMutex);

	loading_timestamp = target;
	updateIfChanged(m_indiP_timestamp, "target", loading_timestamp);

	return 0;
}


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

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_predictorToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_predictorToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
		m_use_predictive_control = true;
		controller->controller->set_integrator(m_use_predictive_control, m_intgain, m_intleak);
		log<text_log>("Switched to predictive control.", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_predictorToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);
		return 0;
   }

   //switch is toggled to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(!m_is_closed_loop)
      {
         m_use_predictive_control = false;
		 controller->controller->set_integrator(m_use_predictive_control, m_intgain, m_intleak);
         log<text_log>("Switched to integrator.", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_predictorToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
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
		updateSwitchIfChanged(m_indiP_reset_bufferRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_reset_exploreRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_reset_exploreRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);
		controller->create_exploration_buffer(m_exploration_rms, m_exploration_steps);
		updateSwitchIfChanged(m_indiP_reset_exploreRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
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
		updateSwitchIfChanged(m_indiP_reset_modelRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}



INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_reset_cleanRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_reset_cleanRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);

		// Regenerate the exploration buffer
		controller->create_exploration_buffer(m_exploration_rms, m_exploration_steps);

		// Reset the controller
		controller->reset_controller();
		controller->reset_data_buffer();	// Clear the data buffer

		// Set the current DM shape and command to zero
		zero();
		controller->set_zero();

		updateSwitchIfChanged(m_indiP_reset_cleanRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_updateControllerRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_updateControllerRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);

		// Regenerate the exploration buffer
		controller->update_controller();
		updateSwitchIfChanged(m_indiP_updateControllerRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_zeroRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_zeroRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		
		if(!m_is_closed_loop){
			zero();
			controller->set_zero();
		}else{
			m_shaped_command.setZero();
		}

		updateSwitchIfChanged(m_indiP_zeroRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_saveRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_saveRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		controller->save_state(savepath);
		updateSwitchIfChanged(m_indiP_saveRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(hoPredCtrl, m_indiP_loadRequest )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_loadRequest.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		controller->load_state(savepath, std::to_string(loading_timestamp));
		updateSwitchIfChanged(m_indiP_loadRequest, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //hoPredCtrl_hpp
