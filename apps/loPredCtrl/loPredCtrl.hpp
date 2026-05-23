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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>

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

struct accelShmimT
{
    static std::string configSection()
    {
        return "accelShmim";
    };

    static std::string indiPrefix()
    {
        return "accel";
    };
};

 class loPredCtrl : public MagAOXApp<true>, public dev::shmimMonitor<loPredCtrl>,
                    public dev::shmimMonitor<loPredCtrl, accelShmimT>
 {
     // Give the test harness access.
     friend class loPredCtrl_test;

     friend class dev::shmimMonitor<loPredCtrl>;
     friend class dev::shmimMonitor<loPredCtrl, accelShmimT>;

     // The base shmimMonitor type
     typedef dev::shmimMonitor<loPredCtrl> shmimMonitorT;
     typedef dev::shmimMonitor<loPredCtrl, accelShmimT> accelShmimMonitorT;

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
    int m_accelChannels {0}; ///< Number of accelerometer telemetry channels consumed per frame.
    int m_accelHistory {0}; ///< Number of accel lag steps included in the DDSPC regressor.
    bool m_accelNormalize {true}; ///< Enables per-channel running z-score normalization for accel samples.
    realT m_accelStdFloor {1.0e-4f}; ///< Lower bound applied to per-channel accel standard deviation.
    realT m_accelClipSigma {0.0f}; ///< Optional post-normalization sigma clipping threshold (0 disables clipping).

    uint32_t m_accelWidth {0}; ///< The width of the accelerometer shmim.
    uint32_t m_accelHeight {0}; ///< The height of the accelerometer shmim.
    DDSPC::Matrix m_latestAccelSample; ///< Most recent normalized accel sample under synchronized-stream assumption.
    bool m_haveAccelSample {false}; ///< True after at least one accel frame has been ingested.

    std::mutex m_accelMutex; ///< Protects shared latest accel sample access across shmim monitor threads.

    DDSPC::Matrix m_accelMean; ///< Running per-channel accel mean for online normalization.
    DDSPC::Matrix m_accelM2; ///< Running per-channel second moment accumulator for online normalization.
    uint64_t m_accelNormCount {0}; ///< Number of accel samples incorporated into normalization statistics.

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
    bool do_reset_model {false};

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

     int allocate( const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/ );

     int processImage( void *curr_src,             ///< [in] pointer to start of current accelerometer frame.
                       const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/
     );

     /// Reset accelerometer sample and normalization state.
     void resetAccelTelemetryState();

     /// Apply per-channel online z-score normalization to one accelerometer sample.
     void normalizeAccelSample( DDSPC::Matrix &sample /**< [in.out] raw sample replaced by normalized sample */ );

     // TODO ::: ADD SAVE AND LOAD FUNCTIONALITY
     void save(std::string directory);
     void load(std::string directory);
 };

 inline int loPredCtrl::send_to_shmim()
 {
    // Check if processImage is running
    // while(m_outputStream.md[0].write == 1);

    // m_outputStream.md[0].write = 1;
    // memcpy( m_outputStream.array.raw, full_command.data(), m_modevalWidth * m_modevalTypeSize );
    // m_outputStream.md[0].cnt0++;
    // m_outputStream.md[0].write = 0;

    //ImageStreamIO_sempost( &m_outputStream, -1 );

    return 0;
 }

 inline loPredCtrl::loPredCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
 {
     accelShmimMonitorT::m_getExistingFirst = true;
     return;
 }

 inline void loPredCtrl::setupConfig()
 {
     shmimMonitorT::setupConfig( config );
     accelShmimMonitorT::setupConfig( config );
     config.add("outputShmim.shmimName", "", "outputShmim.shmimName", argType::Required, "outputShmim", "shmimName", false, "string", "The output shmim to write to.");

     config.add("parameters.gain", "", "parameters.gain", argType::Required, "parameters", "gain", false, "float", "The initial feedback gain.");
     config.add("parameters.regularization", "", "parameters.regularization", argType::Required, "parameters", "regularization", false, "float", "The regularization parameter.");
     config.add("parameters.gamma", "", "parameters.gamma", argType::Required, "parameters", "gamma", false, "float", "The forgetting factor.");
     config.add("parameters.covariance", "", "parameters.covariance", argType::Required, "parameters", "covariance", false, "float", "The initial covariance.");

     config.add("parameters.num_modes", "", "parameters.num_modes", argType::Required, "parameters", "num_modes", false, "int", "The number of modes that will be controlled through predictive control.");
     config.add("parameters.history", "", "parameters.history", argType::Required, "parameters", "history", false, "int", "The number of past measurements for the prediction.");
     config.add("parameters.future", "", "parameters.future", argType::Required, "parameters", "future", false, "int", "The number of future steps that are predicted.");
     config.add("parameters.accel_channels", "", "parameters.accel_channels", argType::Required, "parameters", "accel_channels", false, "int", "Number of accelerometer telemetry channels to ingest.");
     config.add("parameters.accel_history", "", "parameters.accel_history", argType::Required, "parameters", "accel_history", false, "int", "Number of lagged accelerometer samples to include in the predictor.");
     config.add("parameters.accel_normalize", "", "parameters.accel_normalize", argType::Required, "parameters", "accel_normalize", false, "bool", "Enable online per-channel z-score normalization for accelerometer telemetry.");
     config.add("parameters.accel_std_floor", "", "parameters.accel_std_floor", argType::Required, "parameters", "accel_std_floor", false, "float", "Standard-deviation floor used for accelerometer normalization.");
     config.add("parameters.accel_clip_sigma", "", "parameters.accel_clip_sigma", argType::Required, "parameters", "accel_clip_sigma", false, "float", "Optional post-normalization clip limit in sigma units (0 disables clipping).");
 }

 inline int loPredCtrl::loadConfigImpl( mx::app::appConfigurator &_config )
 {
     shmimMonitorT::loadConfig( _config );
     accelShmimMonitorT::loadConfig( _config );

    _config(m_gainCtrl, "parameters.gain");
    _config(m_regularizationCtrl, "parameters.regularization");
    _config(m_gammaCtrl, "parameters.gamma");
    _config(m_covarianceCtrl, "parameters.covariance");

    _config(m_num_modes, "parameters.num_modes");
    _config(m_history, "parameters.history");
    _config(m_future, "parameters.future");
    _config(m_accelChannels, "parameters.accel_channels");
    _config(m_accelHistory, "parameters.accel_history");
    _config(m_accelNormalize, "parameters.accel_normalize");
    _config(m_accelStdFloor, "parameters.accel_std_floor");
    _config(m_accelClipSigma, "parameters.accel_clip_sigma");

    m_accelChannels = std::max(0, m_accelChannels);
    m_accelHistory = std::max(0, m_accelHistory);

    if(m_accelChannels == 0 || m_accelHistory == 0){
        accelShmimMonitorT::m_shmimName = "";
    }

    _config(m_outputName, "outputShmim.shmimName");

	std::cout << "Open output channel at " << m_outputName << std::endl;
    std::cout << "Gain " << m_gainCtrl << std::endl;
    std::cout << "Regularization " << m_regularizationCtrl << std::endl;
    std::cout << "Gamma " << m_gammaCtrl << std::endl;

    std::cout << "History " << m_history << std::endl;
    std::cout << "Future " << m_future << std::endl;
    std::cout << "Accel channels " << m_accelChannels << std::endl;
    std::cout << "Accel history " << m_accelHistory << std::endl;

    std::cout << "Done reading config Impl." << std::endl;

    resetAccelTelemetryState();

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

     if( accelShmimMonitorT::appStartup() < 0 )
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

     if( accelShmimMonitorT::appLogic() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }

     std::unique_lock<std::mutex> lock( m_indiMutex );

     if( shmimMonitorT::updateINDI() < 0 )
     {
         log<software_error>( { __FILE__, __LINE__ } );
     }

     if( accelShmimMonitorT::updateINDI() < 0 )
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
     accelShmimMonitorT::appShutdown();

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

    /*
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
    */

    if(controller){
        delete controller;
        controller = nullptr;
    }

    controller = new DDSPC::PredictiveController(m_num_modes,
                                                  m_history,
                                                  m_future,
                                                  m_gainCtrl,
                                                  m_gammaCtrl,
                                                  m_regularizationCtrl,
                                                  m_covarianceCtrl,
                                                  m_accelChannels,
                                                  m_accelHistory);

    return 0;
 }

 inline int loPredCtrl::allocate( const accelShmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused

    m_accelWidth = accelShmimMonitorT::m_width;
    m_accelHeight = accelShmimMonitorT::m_height;

    if(m_accelChannels < 0){
        m_accelChannels = 0;
    }

    if(m_accelChannels > 0){
        size_t totalAccelValues = static_cast<size_t>(m_accelWidth) * static_cast<size_t>(m_accelHeight);
        if(static_cast<size_t>(m_accelChannels) > totalAccelValues){
            return log<software_error, -1>({__FILE__, __LINE__, "Accelerometer channel count exceeds accel shmim size."});
        }
    }

    resetAccelTelemetryState();
    return 0;
 }

 inline void loPredCtrl::resetAccelTelemetryState()
 {
    std::lock_guard<std::mutex> guard(m_accelMutex);

    m_accelNormCount = 0;
    m_haveAccelSample = false;

    if(m_accelChannels <= 0){
        m_accelMean.resize(0, 1);
        m_accelM2.resize(0, 1);
        m_latestAccelSample.resize(0, 1);
        return;
    }

    m_accelMean.resize(m_accelChannels, 1);
    m_accelMean.setZero();
    m_accelM2.resize(m_accelChannels, 1);
    m_accelM2.setZero();

    m_latestAccelSample.resize(m_accelChannels, 1);
    m_latestAccelSample.setZero();
 }

 inline void loPredCtrl::normalizeAccelSample( DDSPC::Matrix &sample )
 {
    if(!m_accelNormalize){
        return;
    }

    if(m_accelChannels <= 0 || sample.rows() != m_accelChannels || sample.cols() != 1){
        return;
    }

    m_accelNormCount++;
    realT stdFloor = std::max(m_accelStdFloor, static_cast<realT>(1.0e-8));

    for(int i = 0; i < m_accelChannels; ++i){
        realT value = sample(i, 0);
        realT delta = value - m_accelMean(i, 0);
        m_accelMean(i, 0) += delta / static_cast<realT>(m_accelNormCount);
        realT delta2 = value - m_accelMean(i, 0);
        m_accelM2(i, 0) += delta * delta2;

        if(m_accelNormCount < 2){
            sample(i, 0) = 0.0;
            continue;
        }

        realT variance = m_accelM2(i, 0) / static_cast<realT>(m_accelNormCount - 1);
        realT sigma = std::sqrt(std::max(variance, stdFloor * stdFloor));
        sample(i, 0) = (value - m_accelMean(i, 0)) / sigma;

        if(m_accelClipSigma > 0.0f){
            sample(i, 0) = std::max(-m_accelClipSigma, std::min(m_accelClipSigma, sample(i, 0)));
        }
    }
 }

 inline int loPredCtrl::processImage( void *curr_src, const accelShmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused

    if(m_accelChannels <= 0){
        return 0;
    }

    Eigen::Map<eigenImage<realT>> accelFrame(static_cast<realT *>(curr_src), m_accelWidth, m_accelHeight);
    DDSPC::Matrix accelSample;
    accelSample.resize(m_accelChannels, 1);

    size_t availableValues = static_cast<size_t>(m_accelWidth) * static_cast<size_t>(m_accelHeight);
    for(int i = 0; i < m_accelChannels; ++i){
        if(static_cast<size_t>(i) >= availableValues){
            accelSample(i, 0) = 0.0;
            continue;
        }

        uint32_t row = static_cast<uint32_t>(i) % m_accelWidth;
        uint32_t col = static_cast<uint32_t>(i) / m_accelWidth;
        accelSample(i, 0) = accelFrame(row, col);
    }

    // Normalization is applied on ingestion so both learning and runtime use the same scale.
    normalizeAccelSample(accelSample);

    std::lock_guard<std::mutex> guard(m_accelMutex); //mutex scope
    m_latestAccelSample = accelSample;
    m_haveAccelSample = true;

    return 0;
 }

 inline int loPredCtrl::processImage( void *curr_src, const dev::shmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused
    // This could be made more efficient by doing only a single copy statement.
    Eigen::Map<eigenImage<realT>> m_modeval( static_cast<realT *>(curr_src), m_modevalWidth, m_modevalHeight);

    DDSPC::Matrix exp_noise;
    exp_noise.resize(m_num_modes, 1);
    exp_noise.setZero();

    if(do_reset_model){
        controller->reset();
        do_reset_model = false;
    }


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

    if(controller){
        DDSPC::Matrix accelSample;
        accelSample.resize(m_accelChannels, 1);
        accelSample.setZero();

        // Stream assumption: accelerometer telemetry is already synchronized to WFS frames.
        {
            std::lock_guard<std::mutex> guard(m_accelMutex); //mutex scope
            if(m_haveAccelSample && m_latestAccelSample.rows() == m_accelChannels && m_latestAccelSample.cols() == 1){
                accelSample = m_latestAccelSample;
            }
        }

        controller->push_accelerometer_sample(accelSample);
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

    // send_to_shmim();

    if(is_learning){
        controller->update_system();
        controller->update_controller();
    }

    if(frame_counter % 20 == 0){
        std::cout << "HOWDY" << std::endl;
    }

     frame_counter++;
     return 0;
 }

 INDI_NEWCALLBACK_DEFN( loPredCtrl, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_exploration, ipRecv );
    // Called in indi like: num_explore, std, regularization, num_explore, std, regularization, ....

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

        //controller->reset();
        do_reset_model = true;
        log<text_log>("request reset.", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_resetToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}

   return 0;
}

 } // namespace app
 } // namespace MagAOX

 #endif // loPredCtrl_hpp
