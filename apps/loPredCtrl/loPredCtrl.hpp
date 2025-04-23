/** \file loPredCtrl.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef loPredCtrl_hpp
#define loPredCtrl_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <thread>
#include <random>

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "ar_controller.hpp"

 // #define MAGAOX_CURRENT_SHA1 0
 // #define MAGAOX_REPO_MODIFIED 0
 namespace MagAOX
 {
 namespace app
 {
 
 class loPredCtrl : public MagAOXApp<true>, public dev::shmimMonitor<loPredCtrl>
 {
     // Give the test harness access.
     friend class loPredCtrl_test;
 
     friend class dev::shmimMonitor<loPredCtrl>;
 
     // The base shmimMonitor type
     typedef dev::shmimMonitor<loPredCtrl> shmimMonitorT;
 
     /// Floating point type in which to do all calculations.
     typedef float realT;
 
   public:
     /** \name app::dev Configurations
      *@{
      */
 
     ///@}
 
   protected:
     /** \name Configurable Parameters
      *@{
      */
 
    // variables for sending the output to an output shmim.
    std::string m_outputName;
	IMAGE m_outputStream; 
	uint32_t m_outputWidth {0}; ///< The width of the image
	uint32_t m_outputHeight {0}; ///< The height of the image.
	
	uint8_t m_outputDataType{0}; ///< The ImageStreamIO type code.
	size_t m_outputTypeSize {0}; ///< The size of the type, in bytes.  
	
	bool m_outputOpened {false};
	bool m_outputRestart {false};

    // The incoming stream name
    uint32_t m_modevalWidth {0}; ///< The width of the shmim
    uint32_t m_modevalHeight {0}; ///< The height of the shmim
    uint32_t m_modevalTypeSize{0};

    long long frame_counter {0};

    // The predictive control parameters
    float m_gainCtrl {0.0};
    float m_regularizationCtrl {1.0};
    float m_gammaCtrl {1.00};
    float m_covarianceCtrl {100000.0};

    int m_num_modes {1};
    int m_history {5};
    int m_future {3};

    DDSPC::Matrix new_command;
    DDSPC::Matrix new_measurement;
    DDSPC::Matrix full_command;

    DDSPC::PredictiveController* controller {nullptr};

    // Process control parameters
    bool is_learning {false};
    bool is_predictive_control {false};
    
    //  Learning variables
    std::vector<float> m_exploration_noise_strength_01;
    std::vector<int> m_exploration_steps_01;
    std::vector<float> m_regularization_steps_01;

    std::vector<float> m_exploration_noise_strength_02;
    std::vector<int> m_exploration_steps_02;
    std::vector<float> m_regularization_steps_02;

    bool switch_exploration {false};
    bool use_set_01 {true};

    //
    std::default_random_engine generator;
    std::normal_distribution<DDSPC::realT> distribution;

    std::string m_exploration_sequence {""};

    pcf::IndiProperty m_indiP_exploration;
    pcf::IndiProperty m_indiP_learningToggle;
    pcf::IndiProperty m_indiP_predictingToggle;
    pcf::IndiProperty m_indiP_resetToggle;

   public:

    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_exploration );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_learningToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_predictingToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_resetToggle );

     /// Default c'tor.
     loPredCtrl();
 
     /// D'tor, declared and defined for noexcept.
     ~loPredCtrl() noexcept
     {
     }
 
     virtual void setupConfig();
 
     /// Implementation of loadConfig logic, separated for testing.
     /** This is called by loadConfig().
      */
     int loadConfigImpl(
         mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );
 
     virtual void loadConfig();
 
     /// Startup function
     /**
      *
      */
     virtual int appStartup();
 
     /// Implementation of the FSM for loPredCtrl.
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
 
     // Custom functions
     int send_to_shmim();
    
   protected:
     int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );
 
     int processImage( void *curr_src,          ///< [in] pointer to start of current frame.
                       const dev::shmimT &dummy ///< [in] tag to differentiate shmimMonitor parents.
     );

     // TODO ::: ADD SAVE AND LOAD FUNCTIONALITY
     void save(std::string directory);
     void load(std::string directory);
 }; 
 
 inline int loPredCtrl::send_to_shmim()
 {
    // Check if processImage is running
    // while(m_outputStream.md[0].write == 1);
    
    m_outputStream.md[0].write = 1;
    memcpy( m_outputStream.array.raw, full_command.data(), m_modevalWidth * m_modevalTypeSize );
    m_outputStream.md[0].cnt0++;
    m_outputStream.md[0].write = 0;

    ImageStreamIO_sempost( &m_outputStream, -1 );

    return 0;
 }
 
 inline loPredCtrl::loPredCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
 {
     return;
 }
 
 inline void loPredCtrl::setupConfig()
 {
     shmimMonitorT::setupConfig( config );
     config.add("outputShmim.shmimName", "", "outputShmim.shmimName", argType::Required, "outputShmim", "shmimName", false, "string", "The output shmim to write to.");

     config.add("parameters.gain", "", "parameters.gain", argType::Required, "parameters", "gain", false, "float", "The initial feedback gain.");
     config.add("parameters.regularization", "", "parameters.regularization", argType::Required, "parameters", "regularization", false, "float", "The regularization parameter.");
     config.add("parameters.gamma", "", "parameters.gamma", argType::Required, "parameters", "gamma", false, "float", "The forgetting factor.");
     config.add("parameters.covariance", "", "parameters.covariance", argType::Required, "parameters", "covariance", false, "float", "The initial covariance.");

     config.add("parameters.num_modes", "", "parameters.num_modes", argType::Required, "parameters", "num_modes", false, "int", "The number of modes that will be controlled through predictive control.");
     config.add("parameters.history", "", "parameters.history", argType::Required, "parameters", "history", false, "int", "The number of past measurements for the prediction.");
     config.add("parameters.future", "", "parameters.future", argType::Required, "parameters", "future", false, "int", "The number of future steps that are predicted.");
 }
 
 inline int loPredCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
 {
     shmimMonitorT::loadConfig( config );

    _config(m_gainCtrl, "parameters.gain");
    _config(m_regularizationCtrl, "parameters.regularization");
    _config(m_gammaCtrl, "parameters.gamma");
    _config(m_covarianceCtrl, "parameters.covariance");

    _config(m_num_modes, "parameters.num_modes");
    _config(m_history, "parameters.history");
    _config(m_future, "parameters.future");

    _config(m_outputName, "outputShmim.shmimName");

	std::cout << "Open output channel at " << m_outputName << std::endl;
    std::cout << "Gain " << m_gainCtrl << std::endl;
    std::cout << "Regularization " << m_regularizationCtrl << std::endl;
    std::cout << "Gamma " << m_gammaCtrl << std::endl;

    std::cout << "History " << m_history << std::endl;
    std::cout << "Future " << m_future << std::endl;

    std::cout << "Done reading config Impl." << std::endl;

     return 0;
 }
 
 inline void loPredCtrl::loadConfig()
 {
     loadConfigImpl( config );
 }
 
 inline int loPredCtrl::appStartup()
 {
     if( shmimMonitorT::appStartup() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }
     
     CREATE_REG_INDI_NEW_TEXT( m_indiP_exploration, "exploration_sequence", "", "");

     createStandardIndiToggleSw( m_indiP_learningToggle, "learn", "Learning State", "Learn Controls");
	 registerIndiPropertyNew( m_indiP_learningToggle, INDI_NEWCALLBACK(m_indiP_learningToggle) );
     
     createStandardIndiToggleSw( m_indiP_predictingToggle, "predict", "Predict State", "Predictive Controls");
	 registerIndiPropertyNew( m_indiP_predictingToggle, INDI_NEWCALLBACK(m_indiP_predictingToggle) );

     createStandardIndiRequestSw( m_indiP_resetToggle, "reset_model", "Reset the RLS model", "Reset Model");
	 registerIndiPropertyNew( m_indiP_resetToggle, INDI_NEWCALLBACK(m_indiP_resetToggle) ); 

     // state(stateCodes::READY);
     state( stateCodes::OPERATING );
     return 0;
 }
 
 inline int loPredCtrl::appLogic()
 {
     if( shmimMonitorT::appLogic() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }
 
     std::unique_lock<std::mutex> lock( m_indiMutex );
 
     if( shmimMonitorT::updateINDI() < 0 )
     {
         log<software_error>( { __FILE__, __LINE__ } );
     }

     updatesIfChanged<std::string>( m_indiP_exploration, { "current", "target" }, { m_exploration_sequence, m_exploration_sequence } );     

     if(is_learning){
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	 }else{
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	 }

     if(is_predictive_control){
        updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::On, INDI_OK);
    }else{
        updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
    }
 
     return 0;
 }
 
 inline int loPredCtrl::appShutdown()
 {
     shmimMonitorT::appShutdown();

     if(controller)
        delete controller;

     return 0;
 }
 
 inline int loPredCtrl::allocate( const dev::shmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused

    m_modevalWidth = shmimMonitorT::m_width;
    m_modevalHeight = shmimMonitorT::m_height;
    m_modevalTypeSize = sizeof(realT);
    std::cout << "m_modevalWidth: " << m_modevalWidth << std::endl;
    std::cout << "m_modevalHeight: " << m_modevalHeight << std::endl;

    full_command.resize(m_modevalWidth, m_modevalHeight);
    new_command.resize(m_num_modes, 1);
    new_measurement.resize(m_num_modes, 1);

    generator = std::default_random_engine();
    distribution = std::normal_distribution<DDSPC::realT>(0.0, 1.0);

    // Allocate the DM
	if(m_outputOpened){
		ImageStreamIO_closeIm(&m_outputStream);
	}

	m_outputOpened = false;
	m_outputRestart = false; //Set this up front, since we're about to restart.

	if( ImageStreamIO_openIm(&m_outputStream, m_outputName.c_str()) == 0){
		if(m_outputStream.md[0].sem < 10){
			ImageStreamIO_closeIm(&m_outputStream);
		}else{
			m_outputOpened = true;
		}
	}
		
	if(!m_outputOpened){
		log<text_log>( m_outputName + " not opened.", logPrio::LOG_NOTICE); 
		return -1;
	}else{
		m_outputWidth = m_outputStream.md->size[0]; 
		m_outputHeight = m_outputStream.md->size[1]; 

		m_outputDataType = m_outputStream.md->datatype;
		m_outputTypeSize = sizeof(float);
		
		log<text_log>( "Opened " + m_outputName + " " + std::to_string(m_outputWidth) + " x " + std::to_string(m_outputHeight) + " with data type: " + std::to_string(m_outputDataType), logPrio::LOG_NOTICE); 
	}


    controller = new DDSPC::PredictiveController(m_num_modes, m_history, m_future, m_gainCtrl, m_gammaCtrl, m_regularizationCtrl, m_covarianceCtrl);

    return 0;
 }
 
 inline int loPredCtrl::processImage( void *curr_src, const dev::shmimT &dummy )
 {
    // static_cast<void>( dummy ); // be unused   
    // This could be made more efficient by doing only a single copy statement.
    Eigen::Map<eigenImage<realT>> m_modeval( static_cast<realT *>(curr_src), m_modevalWidth, m_modevalHeight);

    DDSPC::Matrix exp_noise;
    exp_noise.resize(m_num_modes, 1);
    exp_noise.setZero();

    if(switch_exploration){
        use_set_01 = !use_set_01;
        switch_exploration = false;

        if(use_set_01){
            controller->set_regularization(m_regularization_steps_01[0]);
        }else{
            controller->set_regularization(m_regularization_steps_02[0]);
        }
    }
    
    if(use_set_01){
        if(!m_exploration_steps_01.empty() and !m_exploration_noise_strength_01.empty()){
            for(int i=0; i < m_num_modes; i++){
                exp_noise(i,0) = m_exploration_noise_strength_01[0] * distribution(generator);
            }
            
            // If no more steps are left pop it!
            m_exploration_steps_01[0]--;
            if(m_exploration_steps_01[0] == 0){
                m_exploration_steps_01.erase(m_exploration_steps_01.begin());
                m_exploration_noise_strength_01.erase(m_exploration_noise_strength_01.begin());
                
                // Erase and apply the next regularization step?
                m_regularization_steps_01.erase(m_regularization_steps_01.begin());
                if(!m_regularization_steps_01.empty())
                    controller->set_regularization(m_regularization_steps_01[0]);
            }
        }
    }else{
        if(!m_exploration_steps_02.empty() and !m_exploration_noise_strength_02.empty()){
            for(int i=0; i < m_num_modes; i++){
                exp_noise(i,0) = m_exploration_noise_strength_02[0] * distribution(generator);
            }
            
            // If no more steps are left pop it!
            m_exploration_steps_02[0]--;
            if(m_exploration_steps_02[0] == 0){
                m_exploration_steps_02.erase(m_exploration_steps_02.begin());
                m_exploration_noise_strength_02.erase(m_exploration_noise_strength_02.begin());
                
                // Erase and apply the next regularization step?
                m_regularization_steps_02.erase(m_regularization_steps_02.begin());
                if(!m_regularization_steps_02.empty())
                    controller->set_regularization(m_regularization_steps_02[0]);
            }
        }
    }

    for(int i=0; i < m_num_modes; i++){
        new_measurement(i, 0) = m_modeval(i,0);
    }

    if(is_predictive_control){
        new_command = controller->calculate_command(new_measurement, exp_noise);
    }

    for(int i=0; i < m_modevalWidth; i++){
        if(i < m_num_modes){
            full_command(i, 0) = new_command(i, 0);
        }else{
            full_command(i, 0) = m_modeval(i,0);
        }   
    }

    send_to_shmim();

    if(is_learning){
        controller->update_system();
        controller->update_controller();
    }

    if(frame_counter % 2000 == 0){
        std::cout << "HOWDY" << std::endl;
    }
 
     frame_counter++;
     return 0;
 }

 INDI_NEWCALLBACK_DEFN( loPredCtrl, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_exploration, ipRecv );

    std::string target;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( indiTargetUpdate( m_indiP_exploration, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    // Now we need to parse the string!
    m_exploration_sequence = target;
    std::cout << target << std::endl;

    std::stringstream csvStringStream(m_exploration_sequence);
    std::string entry;
    
    int k = 0;
    while (getline(csvStringStream, entry, ',')){
        if(k % 3 == 0){
            // std::cout << std::stoi(entry) << std::endl;
            if(use_set_01){
                m_exploration_steps_02.push_back(std::stoi(entry));
            }else{
                m_exploration_steps_01.push_back(std::stoi(entry));
            }            
        }else if(k % 3 == 1){
            // std::cout << static_cast<DDSPC::realT>() << std::endl;
            if(use_set_01){
                m_exploration_noise_strength_02.push_back(std::stod(entry));
            }else{
                m_exploration_noise_strength_01.push_back(std::stod(entry));
            }
        }else{
            if(use_set_01){
                m_regularization_steps_02.push_back(std::stof(entry));
            }else{
                m_regularization_steps_01.push_back(std::stof(entry));
            }
        }
        k++;
    }
    switch_exploration = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_learningToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_learningToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!is_learning) //is actively learning so change it
      {
		is_learning = true;
		log<text_log>("started learning", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);

      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(is_learning) //is actively learning so change it
      {
        is_learning = false;
         log<text_log>("stopped learning", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_predictingToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_predictingToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!is_predictive_control) //is actively learning so change it
      {
		is_predictive_control = true;
		log<text_log>("started predicting", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);

      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(is_predictive_control) //is actively learning so change it
      {
        is_predictive_control = false;
         log<text_log>("stopped predicting", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_resetToggle )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_resetToggle.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{   
		std::lock_guard<std::mutex> guard(m_indiMutex);
		controller->reset();
		updateSwitchIfChanged(m_indiP_resetToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}
   
   return 0;
}
 
 } // namespace app
 } // namespace MagAOX
 
 #endif // loPredCtrl_hpp
 