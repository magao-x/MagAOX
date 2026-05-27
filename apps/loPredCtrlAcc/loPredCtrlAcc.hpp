/** \file loPredCtrlAcc.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef loPredCtrlAcc_hpp
#define loPredCtrlAcc_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <thread>
#include <random>
#include <semaphore.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <mutex>

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

 enum class controllerModeT
 {
     legacy,
     accel
 };

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

 class loPredCtrlAcc : public MagAOXApp<true>, public dev::shmimMonitor<loPredCtrlAcc>, public dev::shmimMonitor<loPredCtrlAcc, accelShmimT>, public dev::frameGrabber<loPredCtrlAcc>, public dev::telemeter<loPredCtrlAcc>
 {
     // Give the test harness access.
     friend class loPredCtrlAcc_test;

     friend class dev::shmimMonitor<loPredCtrlAcc>;
     friend class dev::shmimMonitor<loPredCtrlAcc, accelShmimT>;

     // The base shmimMonitor type
     typedef dev::shmimMonitor<loPredCtrlAcc> shmimMonitorT;
     typedef dev::shmimMonitor<loPredCtrlAcc, accelShmimT> accelShmimMonitorT;

     friend class dev::frameGrabber<loPredCtrlAcc>;

     typedef dev::frameGrabber<loPredCtrlAcc> frameGrabberT;

     friend class dev::telemeter<loPredCtrlAcc>;

     typedef dev::telemeter<loPredCtrlAcc> telemeterT;

     /// Floating point type in which to do all calculations.
     typedef float realT;

   public:
     /** \name app::dev Configurations
      *@{
      */

     /// This framegrabber can't be flipped
     static constexpr bool c_frameGrabber_flippable = false;

     ///@}

   protected:
     /** \name Configurable Parameters
      *@{
      */


    // The incoming stream name
    uint32_t m_modevalWidth {0}; ///< The width of the shmim
    uint32_t m_modevalHeight {0}; ///< The height of the shmim
    uint32_t m_modevalTypeSize{0};

    long long frame_counter {0};
    std::chrono::high_resolution_clock::time_point m_lastPrintTime { std::chrono::high_resolution_clock::now() };

    // The predictive control parameters
    realT m_gainCtrl {0.0};
    realT m_copygainCtrl {1.0};
    realT m_regularizationCtrl {100.0};
    realT m_gammaCtrl {1.00};
    realT m_covarianceCtrl {1.0};

    int m_num_modes {1};
    int m_history {5};
    int m_future {3};
    std::string m_controllerModeConfig{"legacy"};
    controllerModeT m_controllerMode {controllerModeT::legacy};
    bool m_accelConfigured {false};
    bool m_accelEnabled {false};
    int m_accelChannels {2};
    int m_accelHistory {20};
    bool m_accelNormalize {true};
    realT m_accelStdFloor {1.0e-4f};
    realT m_accelClipSigma {0.0f};
    int m_accelMissingFrameLimit {20};

    uint32_t m_accelWidth {0};
    uint32_t m_accelHeight {0};
    DDSPC::Matrix m_latestAccelSample;
    bool m_haveAccelSample {false};
    int m_accelMissingFrameCount {0};
    std::mutex m_accelMutex;
    DDSPC::Matrix m_accelMean;
    DDSPC::Matrix m_accelM2;
    uint64_t m_accelNormCount {0};
    bool m_accelMonitorStarted {false};

    DDSPC::Matrix new_command;
    DDSPC::Matrix new_measurement;
    DDSPC::Matrix full_command;
    DDSPC::Matrix zero_exp_noise;
    

    DDSPC::PredictiveController* controller {nullptr};

    // Process control parameters
    bool is_learning {false};
    bool is_std_learning {false};
    bool is_predictive_control {false};
    bool is_integrating {false};
    bool own_shmim {true};

    double loop_time_elapsed {0.0};

    //  Learning variables
    std::vector<realT> m_exploration_noise_strength_01;
    std::vector<int> m_exploration_steps_01;
    std::vector<realT> m_regularization_steps_01;

    std::vector<realT> m_exploration_noise_strength_02;
    std::vector<int> m_exploration_steps_02;
    std::vector<realT> m_regularization_steps_02;

    bool switch_exploration {false};
    bool use_set_01 {true};
    bool do_reset_model {false};
    bool do_trigger_load {false};
    bool do_trigger_save {false};
    bool use_qrd{false};

    //
    std::default_random_engine generator;
    std::normal_distribution<DDSPC::realT> distribution;

    std::string m_exploration_sequence {""};

    std::string m_filename {""};

    std::string m_fpsSource{ "camwfs" }; /**< Device name for getting fps of the loop.
                                              This device must have *.fps.current.  Default is camwfs*/

    realT m_fps {0}; ///< The current fps

    /** \name frameGrabber Interface
     * @{
     */

    bool m_updated{ false }; ///< Flag indicating that the commands have been updated
    sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the fg thread and the process thread.

    ///@}

    pcf::IndiProperty m_indiP_exploration;
    pcf::IndiProperty m_indiP_filename;
    pcf::IndiProperty m_indiP_learningToggle;
    pcf::IndiProperty m_indiP_learningStdToggle;
    pcf::IndiProperty m_indiP_integratingToggle;
    pcf::IndiProperty m_indiP_predictingToggle;
    pcf::IndiProperty m_indiP_resetToggle;

    pcf::IndiProperty m_indiP_saveToggle;
    pcf::IndiProperty m_indiP_loadToggle;

    pcf::IndiProperty m_indiP_fpsSource;
    pcf::IndiProperty m_indiP_fps;
    pcf::IndiProperty m_indiP_controllerMode;

   public:

    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_exploration );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_filename );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_learningToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_learningStdToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_integratingToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_predictingToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_resetToggle );

    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_saveToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_loadToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc, m_indiP_controllerMode );

    INDI_SETCALLBACK_DECL( loPredCtrlAcc, m_indiP_fpsSource );

     /// Default c'tor.
     loPredCtrlAcc();

     /// D'tor, declared and defined for noexcept.
     ~loPredCtrlAcc() noexcept
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

     /// Implementation of the FSM for loPredCtrlAcc.
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

     /** \name frameGrabber Interface
      * @{
      */

     /// Configure the output stream for acquistion.
     /** Tests if stream exists and is expected size.  Creates it if needed.
      *  will set m_width, m_height, and m_dataType.
      */
     int configureAcquisition();

     /// Gets the frames-per-second readout rate
     /** Used for the latency statistics
       */
     float fps();

     /// Start acquisition.
     /** A no-op in this class.
      */
     int startAcquisition();

     /// Acquire data.
     /** Here just waits on the semaphore.
      */
     int acquireAndCheckValid();

     /// Loads the commands into the stream
     int loadImageIntoStream(void * dest);

     ///Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
     int reconfig();

     ///@}

     /** \name telemeter Interface
      * @{
      */

     int checkRecordTimes();

     int recordTelem( const telem_fgtimings * );

     ///@}

   protected:
     int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

     int processImage( void *curr_src,          ///< [in] pointer to start of current frame.
                       const dev::shmimT &dummy ///< [in] tag to differentiate shmimMonitor parents.
     );

     int allocate( const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/ );

     int processImage( void *curr_src,             ///< [in] pointer to start of current accelerometer frame.
                       const accelShmimT &dummy /**< [in] tag to differentiate accelerometer shmim monitor parent.*/
     );

     controllerModeT parseControllerMode( const std::string &modeName );
     const char *controllerModeElement( controllerModeT mode );
     int rebuildController( bool enableAccelFeatures );
     int setControllerMode( controllerModeT mode, const std::string &reason );
     void resetAccelTelemetryState();
     void normalizeAccelSample( DDSPC::Matrix &sample );
     void disableAccelIntegration( const std::string &reason, bool rebuild = false );

     inline void save(std::string directory)
     {
         if(controller)
             controller->save_state(directory);
     }

     inline void load(std::string directory)
     {
         if(controller)
             controller->load_state(directory);
     }
 };


inline loPredCtrlAcc::loPredCtrlAcc() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    accelShmimMonitorT::m_getExistingFirst = true;
     return;
}

inline controllerModeT loPredCtrlAcc::parseControllerMode( const std::string &modeName )
{
    std::string normalized = modeName;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        []( unsigned char c )
        {
            return static_cast<char>( std::tolower( c ) );
        } );

    if( normalized == "accel" || normalized == "accelerometer" )
    {
        return controllerModeT::accel;
    }

    return controllerModeT::legacy;
}

inline const char *loPredCtrlAcc::controllerModeElement( controllerModeT mode )
{
    if( mode == controllerModeT::accel )
    {
        return "accel";
    }

    return "legacy";
}

inline int loPredCtrlAcc::rebuildController( bool enableAccelFeatures )
{
    if( controller )
    {
        delete controller;
        controller = nullptr;
    }

    int accelChannels = enableAccelFeatures ? m_accelChannels : 0;
    int accelHistory = enableAccelFeatures ? m_accelHistory : 0;

    controller = new DDSPC::PredictiveController(
        m_num_modes, m_history, m_future, m_gainCtrl, m_gammaCtrl, m_regularizationCtrl, m_covarianceCtrl, accelChannels, accelHistory );
    controller->use_qrd = use_qrd;
    return 0;
}

inline void loPredCtrlAcc::resetAccelTelemetryState()
{
    std::lock_guard<std::mutex> guard( m_accelMutex ); //mutex scope
    m_accelNormCount = 0;
    m_haveAccelSample = false;
    m_accelMissingFrameCount = 0;

    if( m_accelChannels <= 0 )
    {
        m_accelMean.resize( 0, 1 );
        m_accelM2.resize( 0, 1 );
        m_latestAccelSample.resize( 0, 1 );
        return;
    }

    m_accelMean.resize( m_accelChannels, 1 );
    m_accelMean.setZero();
    m_accelM2.resize( m_accelChannels, 1 );
    m_accelM2.setZero();
    m_latestAccelSample.resize( m_accelChannels, 1 );
    m_latestAccelSample.setZero();
}

inline void loPredCtrlAcc::normalizeAccelSample( DDSPC::Matrix &sample )
{
    if( !m_accelNormalize )
    {
        return;
    }

    if( m_accelChannels <= 0 || sample.rows() != m_accelChannels || sample.cols() != 1 )
    {
        return;
    }

    m_accelNormCount++;
    realT stdFloor = std::max( m_accelStdFloor, static_cast<realT>( 1.0e-8 ) );

    for( int i = 0; i < m_accelChannels; ++i )
    {
        realT value = sample( i, 0 );
        realT delta = value - m_accelMean( i, 0 );
        m_accelMean( i, 0 ) += delta / static_cast<realT>( m_accelNormCount );
        realT delta2 = value - m_accelMean( i, 0 );
        m_accelM2( i, 0 ) += delta * delta2;

        if( m_accelNormCount < 2 )
        {
            sample( i, 0 ) = 0.0;
            continue;
        }

        realT variance = m_accelM2( i, 0 ) / static_cast<realT>( m_accelNormCount - 1 );
        realT sigma = std::sqrt( std::max( variance, stdFloor * stdFloor ) );
        sample( i, 0 ) = ( value - m_accelMean( i, 0 ) ) / sigma;

        if( m_accelClipSigma > 0.0f )
        {
            sample( i, 0 ) = std::max( -m_accelClipSigma, std::min( m_accelClipSigma, sample( i, 0 ) ) );
        }
    }
}

inline void loPredCtrlAcc::disableAccelIntegration( const std::string &reason, bool rebuild )
{
    if( !m_accelEnabled )
    {
        return;
    }

    log<text_log>( reason + " Falling back to legacy mode.", logPrio::LOG_WARNING );
    m_controllerMode = controllerModeT::legacy;
    m_accelEnabled = false;

    if( m_accelMonitorStarted )
    {
        accelShmimMonitorT::appShutdown();
        m_accelMonitorStarted = false;
    }

    if( rebuild )
    {
        rebuildController( false );
    }

    resetAccelTelemetryState();
}

inline int loPredCtrlAcc::setControllerMode( controllerModeT mode, const std::string &reason )
{
    if( mode == m_controllerMode )
    {
        return 0;
    }

    if( mode == controllerModeT::accel )
    {
        if( !m_accelConfigured )
        {
            log<text_log>( reason + " Accel mode requested but accel is not configured.", logPrio::LOG_WARNING );
            return -1;
        }

        if( !m_accelMonitorStarted )
        {
            if( accelShmimMonitorT::appStartup() < 0 )
            {
                log<text_log>( reason + " Accel monitor startup failed; staying in legacy mode.", logPrio::LOG_WARNING );
                m_controllerMode = controllerModeT::legacy;
                m_accelEnabled = false;
                rebuildController( false );
                return -1;
            }

            m_accelMonitorStarted = true;
        }

        m_controllerMode = controllerModeT::accel;
        m_accelEnabled = true;
        resetAccelTelemetryState();
        rebuildController( true );
        return 0;
    }

    if( m_accelMonitorStarted )
    {
        accelShmimMonitorT::appShutdown();
        m_accelMonitorStarted = false;
    }

    m_controllerMode = controllerModeT::legacy;
    m_accelEnabled = false;
    resetAccelTelemetryState();
    rebuildController( false );
    return 0;
}

inline void loPredCtrlAcc::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    accelShmimMonitorT::setupConfig( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG(config);

     config.add("parameters.fpsSource", "", "parameters.fpsSource", argType::Required, "parameters", "fpsSource", false, "string", "The device name for getting fps of the loop.");

     config.add("parameters.gain", "", "parameters.gain", argType::Required, "parameters", "gain", false, "float", "The initial feedback gain.");
     config.add("parameters.copy_gain", "", "parameters.copy_gain", argType::Required, "parameters", "copy_gain", false, "float", "The initial feedback gain.");
     config.add("parameters.regularization", "", "parameters.regularization", argType::Required, "parameters", "regularization", false, "float", "The regularization parameter.");
     config.add("parameters.gamma", "", "parameters.gamma", argType::Required, "parameters", "gamma", false, "float", "The forgetting factor.");
     config.add("parameters.covariance", "", "parameters.covariance", argType::Required, "parameters", "covariance", false, "float", "The initial covariance.");

     config.add("parameters.num_modes", "", "parameters.num_modes", argType::Required, "parameters", "num_modes", false, "int", "The number of modes that will be controlled through predictive control.");
     config.add("parameters.history", "", "parameters.history", argType::Required, "parameters", "history", false, "int", "The number of past measurements for the prediction.");
     config.add("parameters.future", "", "parameters.future", argType::Required, "parameters", "future", false, "int", "The number of future steps that are predicted.");

     config.add("parameters.qrd", "", "parameters.qrd", argType::Required, "parameters", "qrd", false, "bool", "The use QRD-RLS or Classic RLS.");
     config.add("parameters.own_shmim", "", "parameters.own_shmim", argType::Required, "parameters", "own_shmim", false, "bool", "Does the predictive control own the output shmim or not.");
     config.add("parameters.is_integrating", "", "parameters.is_integrating", argType::Required, "parameters", "is_integrating", false, "bool", "Whether the control signal is integrated or not.");
     config.add("parameters.controller_mode", "", "parameters.controller_mode", argType::Optional, "parameters", "controller_mode", false, "string", "Controller mode at startup: legacy or accel.");
     config.add("parameters.accel_enabled", "", "parameters.accel_enabled", argType::Optional, "parameters", "accel_enabled", false, "bool", "Enable accelerometer integration.");
     config.add("parameters.accel_channels", "", "parameters.accel_channels", argType::Optional, "parameters", "accel_channels", false, "int", "Number of accelerometer channels.");
     config.add("parameters.accel_history", "", "parameters.accel_history", argType::Optional, "parameters", "accel_history", false, "int", "Number of accelerometer history samples.");
     config.add("parameters.accel_normalize", "", "parameters.accel_normalize", argType::Optional, "parameters", "accel_normalize", false, "bool", "Enable online accelerometer normalization.");
     config.add("parameters.accel_std_floor", "", "parameters.accel_std_floor", argType::Optional, "parameters", "accel_std_floor", false, "float", "Standard deviation floor for accelerometer normalization.");
     config.add("parameters.accel_clip_sigma", "", "parameters.accel_clip_sigma", argType::Optional, "parameters", "accel_clip_sigma", false, "float", "Optional sigma clipping applied after normalization.");
     config.add("parameters.accel_missing_frame_limit", "", "parameters.accel_missing_frame_limit", argType::Optional, "parameters", "accel_missing_frame_limit", false, "int", "Number of WFS frames to wait before accel mode falls back to legacy.");
}

inline int loPredCtrlAcc::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::loadConfig( _config );

    _config(m_fpsSource, "parameters.fpsSource");
    _config(m_controllerModeConfig, "parameters.controller_mode");
    _config(m_accelConfigured, "parameters.accel_enabled");

    _config(m_gainCtrl, "parameters.gain");
    _config(m_copygainCtrl, "parameters.gain");
    _config(m_regularizationCtrl, "parameters.regularization");
    _config(m_gammaCtrl, "parameters.gamma");
    _config(m_covarianceCtrl, "parameters.covariance");

    _config(m_num_modes, "parameters.num_modes");
    _config(m_history, "parameters.history");
    _config(m_future, "parameters.future");
    _config(use_qrd, "parameters.qrd");
    _config(own_shmim, "parameters.own_shmim");
    _config(is_integrating, "parameters.is_integrating");
    _config(m_accelChannels, "parameters.accel_channels");
    _config(m_accelHistory, "parameters.accel_history");
    _config(m_accelNormalize, "parameters.accel_normalize");
    _config(m_accelStdFloor, "parameters.accel_std_floor");
    _config(m_accelClipSigma, "parameters.accel_clip_sigma");
    _config(m_accelMissingFrameLimit, "parameters.accel_missing_frame_limit");

    if(m_accelConfigured)
    {
        accelShmimMonitorT::loadConfig( _config );
    }
    else
    {
        accelShmimMonitorT::m_shmimName = "";
    }

    m_accelChannels = std::max(0, m_accelChannels);
    m_accelHistory = std::max(0, m_accelHistory);
    if(m_accelMissingFrameLimit < 1)
    {
        m_accelMissingFrameLimit = 1;
    }

    if(m_accelChannels == 0 || m_accelHistory == 0)
    {
        m_accelConfigured = false;
    }

    if(!m_accelConfigured)
    {
        accelShmimMonitorT::m_shmimName = "";
    }

    m_controllerMode = parseControllerMode(m_controllerModeConfig);
    if(m_controllerMode == controllerModeT::accel && !m_accelConfigured)
    {
        m_controllerMode = controllerModeT::legacy;
    }
    m_accelEnabled = m_controllerMode == controllerModeT::accel;

    frameGrabberT::m_ownShmim = own_shmim;
    FRAMEGRABBER_LOAD_CONFIG(_config);
    TELEMETER_LOAD_CONFIG(_config);

	std::cout << "Gain " << m_gainCtrl << std::endl;
    std::cout << "Copy Gain " << m_copygainCtrl << std::endl;
    std::cout << "Regularization " << m_regularizationCtrl << std::endl;
    std::cout << "Covariance " << m_covarianceCtrl << std::endl;
    std::cout << "Gamma " << m_gammaCtrl << std::endl;

    std::cout << "num modes " << m_num_modes << std::endl;
    std::cout << "History " << m_history << std::endl;
    std::cout << "Future " << m_future << std::endl;
    std::cout << "Use QRD " << use_qrd << std::endl;
    std::cout << "Own shmim " << own_shmim << std::endl;
    std::cout << "Controller mode " << controllerModeElement(m_controllerMode) << std::endl;
    std::cout << "Accel configured " << m_accelConfigured << std::endl;
    std::cout << "Accel enabled " << m_accelEnabled << std::endl;
    std::cout << "Accel channels " << m_accelChannels << std::endl;
    std::cout << "Accel history " << m_accelHistory << std::endl;

    std::cout << "Done reading config Impl." << std::endl;
    resetAccelTelemetryState();

     return 0;
 }

 inline void loPredCtrlAcc::loadConfig()
 {
     loadConfigImpl( config );
 }

inline int loPredCtrlAcc::appStartup()
{
     if( shmimMonitorT::appStartup() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }

     if(m_accelEnabled)
     {
         if(accelShmimMonitorT::appStartup() < 0)
         {
             disableAccelIntegration("Accelerometer monitor startup failed.", true);
         }
         else
         {
             m_accelMonitorStarted = true;
         }
     }

     CREATE_REG_INDI_NEW_TEXT( m_indiP_exploration, "exploration_sequence", "", "");

     CREATE_REG_INDI_NEW_TEXT( m_indiP_filename, "filename", "", "");

     std::vector<std::string> controllerModeElements{"legacy", "accel"};
     if(createStandardIndiSelectionSw(m_indiP_controllerMode, "controller_mode", controllerModeElements, "Controller Mode", "Predictive Controls") < 0)
     {
         return log<software_error, -1>( { __FILE__, __LINE__, "error creating controller mode property" } );
     }
     registerIndiPropertyNew(m_indiP_controllerMode, INDI_NEWCALLBACK(m_indiP_controllerMode));

     createStandardIndiToggleSw( m_indiP_learningToggle, "learn", "Learning State", "Learn Controls");
	 registerIndiPropertyNew( m_indiP_learningToggle, INDI_NEWCALLBACK(m_indiP_learningToggle) );

     createStandardIndiToggleSw( m_indiP_learningStdToggle, "learn_std", "Learning State", "Learn Controls");
	 registerIndiPropertyNew( m_indiP_learningStdToggle, INDI_NEWCALLBACK(m_indiP_learningStdToggle) );

     createStandardIndiToggleSw( m_indiP_predictingToggle, "predict", "Predict State", "Predictive Controls");
	 registerIndiPropertyNew( m_indiP_predictingToggle, INDI_NEWCALLBACK(m_indiP_predictingToggle) );

     createStandardIndiToggleSw( m_indiP_integratingToggle, "integrate", "Integration State", "Integration Controls");
	 registerIndiPropertyNew( m_indiP_integratingToggle, INDI_NEWCALLBACK(m_indiP_integratingToggle) );

     createStandardIndiRequestSw( m_indiP_resetToggle, "reset_model", "Reset the RLS model", "Reset Model");
	 registerIndiPropertyNew( m_indiP_resetToggle, INDI_NEWCALLBACK(m_indiP_resetToggle) );

     createStandardIndiRequestSw( m_indiP_saveToggle, "save_state", "Save the controller state", "Save State");
	 registerIndiPropertyNew( m_indiP_saveToggle, INDI_NEWCALLBACK(m_indiP_saveToggle) );

     createStandardIndiRequestSw( m_indiP_loadToggle, "load_state", "Load the controller state", "Load State");
	 registerIndiPropertyNew( m_indiP_loadToggle, INDI_NEWCALLBACK(m_indiP_loadToggle) );

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { errno, "Initializing S.M. semaphore" } );
        return -1;
    }

    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

     state( stateCodes::OPERATING );
     return 0;
 }

inline int loPredCtrlAcc::appLogic()
{
     if( shmimMonitorT::appLogic() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }

     if(m_accelEnabled)
     {
         if(accelShmimMonitorT::appLogic() < 0)
         {
             disableAccelIntegration("Accelerometer monitor thread exited.", true);
         }
     }

    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

     std::unique_lock<std::mutex> lock( m_indiMutex );

     if( shmimMonitorT::updateINDI() < 0 )
     {
         log<software_error>( { __FILE__, __LINE__ } );
     }

     if(m_accelEnabled)
     {
         if(accelShmimMonitorT::updateINDI() < 0)
         {
             disableAccelIntegration("Accelerometer INDI update failed.", true);
         }
     }

    FRAMEGRABBER_UPDATE_INDI;

     updatesIfChanged<std::string>( m_indiP_exploration, { "current", "target" }, { m_exploration_sequence, m_exploration_sequence } );

     updatesIfChanged<std::string>( m_indiP_filename, { "current", "target" }, { m_filename, m_filename } );
     indi::updateSelectionSwitchIfChanged(m_indiP_controllerMode, controllerModeElement(m_controllerMode), m_indiDriver, INDI_OK);

     if(is_learning){
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	 }else{
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	 }

     if(is_std_learning){
		 updateSwitchIfChanged(m_indiP_learningStdToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	 }else{
		 updateSwitchIfChanged(m_indiP_learningStdToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	 }

     if(is_integrating){
		 updateSwitchIfChanged(m_indiP_integratingToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	 }else{
		 updateSwitchIfChanged(m_indiP_integratingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
	 }

     if(is_predictive_control){
        updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::On, INDI_OK);
    }else{
        updateSwitchIfChanged(m_indiP_predictingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
    }

     return 0;
 }

inline int loPredCtrlAcc::appShutdown()
{
     shmimMonitorT::appShutdown();
     if(m_accelMonitorStarted)
     {
         accelShmimMonitorT::appShutdown();
         m_accelMonitorStarted = false;
     }

    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

     if(controller)
        delete controller;

     return 0;
 }

 inline int loPredCtrlAcc::allocate( const dev::shmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused

    m_modevalWidth = shmimMonitorT::m_width;
    m_modevalHeight = shmimMonitorT::m_height;
    m_modevalTypeSize = sizeof(realT);
    std::cout << "m_modevalWidth: " << m_modevalWidth << std::endl;
    std::cout << "m_modevalHeight: " << m_modevalHeight << std::endl;

    // Only resize if dimensions change to avoid Eigen block resize issues
    if(full_command.rows() != (int)m_modevalWidth || full_command.cols() != (int)m_modevalHeight) {
        full_command = DDSPC::Matrix(m_modevalWidth, m_modevalHeight);
    }
    if(new_command.rows() != m_num_modes || new_command.cols() != 1) {
        new_command = DDSPC::Matrix(m_num_modes, 1);
    }
    if(new_measurement.rows() != m_num_modes || new_measurement.cols() != 1) {
        new_measurement = DDSPC::Matrix(m_num_modes, 1);
    }
    
    // allocate the exploration noise matrix
    zero_exp_noise.resize(m_num_modes, 1);
    zero_exp_noise.setZero();

    generator = std::default_random_engine();
    distribution = std::normal_distribution<DDSPC::realT>(0.0, 1.0);
    rebuildController(m_accelEnabled);

    return 0;
 }

 inline int loPredCtrlAcc::allocate( const accelShmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused

    m_accelWidth = accelShmimMonitorT::m_width;
    m_accelHeight = accelShmimMonitorT::m_height;

    if(m_accelChannels > 0)
    {
        size_t totalAccelValues = static_cast<size_t>(m_accelWidth) * static_cast<size_t>(m_accelHeight);
        if(static_cast<size_t>(m_accelChannels) > totalAccelValues)
        {
            return log<software_error, -1>({__FILE__, __LINE__, "Accelerometer channel count exceeds accel shmim size."});
        }
    }

    resetAccelTelemetryState();
    return 0;
 }

 inline int loPredCtrlAcc::processImage( void *curr_src, const accelShmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused
    if(!m_accelEnabled || m_accelChannels <= 0)
    {
        return 0;
    }

    Eigen::Map<eigenImage<realT>> accelFrame( static_cast<realT *>(curr_src), m_accelWidth, m_accelHeight);
    DDSPC::Matrix accelSample;
    accelSample.resize(m_accelChannels, 1);

    size_t availableValues = static_cast<size_t>(m_accelWidth) * static_cast<size_t>(m_accelHeight);
    for(int i = 0; i < m_accelChannels; ++i)
    {
        if(static_cast<size_t>(i) >= availableValues)
        {
            accelSample(i, 0) = 0.0;
            continue;
        }

        uint32_t row = static_cast<uint32_t>(i) % m_accelWidth;
        uint32_t col = static_cast<uint32_t>(i) / m_accelWidth;
        accelSample(i, 0) = accelFrame(row, col);
    }

    normalizeAccelSample(accelSample);

    std::lock_guard<std::mutex> guard(m_accelMutex); //mutex scope
    m_latestAccelSample = accelSample;
    m_haveAccelSample = true;
    m_accelMissingFrameCount = 0;

    return 0;
 }

 inline int loPredCtrlAcc::processImage( void *curr_src, const dev::shmimT &dummy )
 {
    static_cast<void>( dummy ); // be unused
    //record arrival time as the atime
    if( clock_gettime( CLOCK_REALTIME, &(frameGrabberT::m_currImageTimestamp) ) < 0 )
    {
        m_shutdown = true;
        return log<software_critical,-1>( { errno, "clock_gettime" } );
    }

    auto start = std::chrono::high_resolution_clock::now();
    // static_cast<void>( dummy ); // be unused
    // This could be made more efficient by doing only a single copy statement.
    Eigen::Map<eigenImage<realT>> m_modeval( static_cast<realT *>(curr_src), m_modevalWidth, m_modevalHeight);

    DDSPC::Matrix exp_noise;
    exp_noise.resize(m_num_modes, 1);
    exp_noise.setZero();

    if(do_reset_model){
        controller->reset();
        do_reset_model = false;
    }

    if(do_trigger_load){
        load("/opt/MagAOX/calib/loPredCtrlAcc/" + m_filename);
        do_trigger_load = false;
    }

    if(do_trigger_save){
        save("/opt/MagAOX/calib/loPredCtrlAcc/" + m_filename);
        do_trigger_save = false;
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

    if(is_learning){
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
    }

    for(int i=0; i < m_num_modes; i++){
        new_measurement(i, 0) = m_modeval(i,0);
    }

    if(controller && m_accelEnabled)
    {
        DDSPC::Matrix accelSample;
        accelSample.resize(m_accelChannels, 1);
        accelSample.setZero();

        bool haveAccel = false;
        { //mutex scope
            std::lock_guard<std::mutex> guard(m_accelMutex);
            if(m_haveAccelSample && m_latestAccelSample.rows() == m_accelChannels && m_latestAccelSample.cols() == 1)
            {
                accelSample = m_latestAccelSample;
                haveAccel = true;
                m_accelMissingFrameCount = 0;
            }
            else
            {
                m_accelMissingFrameCount++;
            }
        }

        if(!haveAccel && m_accelMissingFrameCount >= m_accelMissingFrameLimit)
        {
            disableAccelIntegration("No accelerometer frames received in accel mode.", true);
        }
        else if(m_accelEnabled)
        {
            controller->push_accelerometer_sample(accelSample);
        }
    }

    if(is_predictive_control){
        if(is_learning){
            new_command = controller->calculate_command(new_measurement, exp_noise);
        }else{
            new_command = controller->calculate_command(new_measurement, zero_exp_noise);
        }
        
    }else{
        for(int i=0; i < m_num_modes; i++){
            new_command(i,0) = m_copygainCtrl * new_measurement(i, 0);
        }
    }
    
    //
    for(size_t i=0; i < m_modevalWidth; i++){
        if(i < static_cast<size_t>(m_num_modes)){
            if(is_integrating){
                full_command(i, 0) = full_command(i, 0) + new_command(i, 0);
            }else{
                full_command(i, 0) = new_command(i, 0);
            }
        }else{
            full_command(i, 0) = m_modeval(i,0);
        }
    }

    // Send modal coefficients to the correct stream
    m_updated = true;

    // trigger framegrabber
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { errno, 0, "Error posting to semaphore" } );
        return -1;
    }

    if(is_learning){
        controller->update_system();
        controller->update_controller();
    }

    auto end = std::chrono::high_resolution_clock::now();
    loop_time_elapsed += std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count();
    
    if(frame_counter % 4000 == 0){
        std::cout << "HOWDY elapsed us: " << loop_time_elapsed / 4000.0 << " us" << std::endl;
        loop_time_elapsed = 0.0;
    }
    
    frame_counter++;
    return 0;
 }

int loPredCtrlAcc::configureAcquisition()
{
    frameGrabberT::m_width = m_modevalWidth;
    frameGrabberT::m_height = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    return 0;
}

float loPredCtrlAcc::fps()
{
    return m_fps;
}

int loPredCtrlAcc::startAcquisition()
{
    return 0;
}

int loPredCtrlAcc::acquireAndCheckValid()
{
    timespec ts;

    errno = 0;
    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        log<software_critical>( { errno, "clock_gettime" } );
        return -1;
    }

    ts.tv_sec += 1;

    if( sem_timedwait( &m_smSemaphore, &ts ) == 0 )
    {
        if( m_updated )
        {
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
}

int loPredCtrlAcc::loadImageIntoStream(void * dest)
{
    memcpy( dest, full_command.data(), full_command.rows() * frameGrabberT::m_typeSize );

    m_updated = false;

    return 0;
}

int loPredCtrlAcc::reconfig()
{
    return 0;
}

int loPredCtrlAcc::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int loPredCtrlAcc::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

 INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
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
            std::cout << std::stoi(entry) << std::endl;
            if(use_set_01){
                m_exploration_steps_02.push_back(std::stoi(entry));
            }else{
                m_exploration_steps_01.push_back(std::stoi(entry));
            }
        }else if(k % 3 == 1){
            std::cout << static_cast<DDSPC::realT>(std::stod(entry)) << std::endl;
            if(use_set_01){
                m_exploration_noise_strength_02.push_back(std::stod(entry));
            }else{
                m_exploration_noise_strength_01.push_back(std::stod(entry));
            }
        }else{
            std::cout << static_cast<DDSPC::realT>(std::stof(entry)) << std::endl;
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

INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_filename )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_filename, ipRecv );

    std::string target;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( indiTargetUpdate( m_indiP_filename, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_filename = target;
    log<text_log>( "Filename set to: " + m_filename, logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( loPredCtrlAcc, m_indiP_controllerMode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_controllerMode, ipRecv );

    controllerModeT requestedMode = m_controllerMode;
    bool found = false;
    for(auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit)
    {
        if(elit->second.getSwitchState() != pcf::IndiElement::On)
        {
            continue;
        }

        if(found)
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "multiple controller modes selected in one update" } );
        }

        if(elit->first == "legacy")
        {
            requestedMode = controllerModeT::legacy;
        }
        else if(elit->first == "accel")
        {
            requestedMode = controllerModeT::accel;
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "invalid controller mode: " + elit->first } );
        }

        found = true;
    }

    if(!found)
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard(m_indiMutex); //mutex scope
    return setControllerMode(requestedMode, "INDI controller mode request.");
}

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_learningToggle )(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_learningStdToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_learningStdToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }

   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!is_learning) //is actively learning so change it
      {

        if(use_set_01){
            m_exploration_steps_02.push_back(1000);
            m_exploration_noise_strength_02.push_back(0.1);
            m_regularization_steps_02.push_back(100.0);

            m_exploration_steps_02.push_back(1000);
            m_exploration_noise_strength_02.push_back(0.1);
            m_regularization_steps_02.push_back(3.0);

            m_exploration_steps_02.push_back(1000);
            m_exploration_noise_strength_02.push_back(0.1);
            m_regularization_steps_02.push_back(0.1);
        }else{
            m_exploration_steps_01.push_back(1000);
            m_exploration_noise_strength_01.push_back(0.1);
            m_regularization_steps_01.push_back(100.0);

            m_exploration_steps_01.push_back(1000);
            m_exploration_noise_strength_01.push_back(0.1);
            m_regularization_steps_01.push_back(3.0);

            m_exploration_steps_01.push_back(1000);
            m_exploration_noise_strength_01.push_back(0.1);
            m_regularization_steps_01.push_back(0.1);
        }

        // Setup all the correct triggers
        switch_exploration = true;
        is_predictive_control = true;
        is_learning = true;

		log<text_log>("started standard learning", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_learningStdToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);

      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(is_learning) //is actively learning so change it
      {
        is_learning = false;
        log<text_log>("stopped STD learning", logPrio::LOG_NOTICE);
        updateSwitchIfChanged(m_indiP_learningStdToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_integratingToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_integratingToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }

   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!is_integrating) //is actively integrating so change it
      {
		is_integrating = true;
		log<text_log>("started integrating", logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_integratingToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);
      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(is_integrating) //is actively integrating so change it
      {
        is_integrating = false;
        log<text_log>("stopped integrating", logPrio::LOG_NOTICE);
        updateSwitchIfChanged(m_indiP_integratingToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_predictingToggle )(const pcf::IndiProperty &ipRecv)
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


INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_resetToggle )(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_saveToggle )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_saveToggle.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);

        // save(m_filename);
        do_trigger_save = true;
        log<text_log>("saved state to " + m_filename, logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_saveToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}

   return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrlAcc, m_indiP_loadToggle )(const pcf::IndiProperty &ipRecv)
{
	if(ipRecv.getName() != m_indiP_loadToggle.getName())
	{
		log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
		return -1;
	}

	if(!ipRecv.find("request")) return 0;

	if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
	{
		std::lock_guard<std::mutex> guard(m_indiMutex);

        // load(m_filename);
        do_trigger_load = true;
        log<text_log>("loaded state from " + m_filename, logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_loadToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}

   return 0;
}

INDI_SETCALLBACK_DEFN( loPredCtrlAcc, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( "current" ) != true ) // this isn't valid
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        updateIfChanged( m_indiP_fps, "current", m_fps );
    }

    return 0;
}

 } // namespace app
 } // namespace MagAOX

 #endif // loPredCtrlAcc_hpp
