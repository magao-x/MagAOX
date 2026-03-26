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
#include <semaphore.h>

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

 class loPredCtrl : public MagAOXApp<true>, public dev::shmimMonitor<loPredCtrl>, public dev::frameGrabber<loPredCtrl>, public dev::telemeter<loPredCtrl>
 {
     // Give the test harness access.
     friend class loPredCtrl_test;

     friend class dev::shmimMonitor<loPredCtrl>;

     // The base shmimMonitor type
     typedef dev::shmimMonitor<loPredCtrl> shmimMonitorT;

     friend class dev::frameGrabber<loPredCtrl>;

     typedef dev::frameGrabber<loPredCtrl> frameGrabberT;

     friend class dev::telemeter<loPredCtrl>;

     typedef dev::telemeter<loPredCtrl> telemeterT;

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
    realT m_regularizationCtrl {1.0};
    realT m_gammaCtrl {1.00};
    realT m_covarianceCtrl {100000.0};

    int m_num_modes {1};
    int m_history {5};
    int m_future {3};

    DDSPC::Matrix new_command;
    DDSPC::Matrix new_measurement;
    DDSPC::Matrix full_command;
    DDSPC::Matrix zero_exp_noise;
    

    DDSPC::PredictiveController* controller {nullptr};

    // Process control parameters
    bool is_learning {false};
    bool is_predictive_control {false};
    bool is_integrating {true};

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
    pcf::IndiProperty m_indiP_integratingToggle;
    pcf::IndiProperty m_indiP_predictingToggle;
    pcf::IndiProperty m_indiP_resetToggle;

    pcf::IndiProperty m_indiP_saveToggle;
    pcf::IndiProperty m_indiP_loadToggle;

    pcf::IndiProperty m_indiP_fpsSource;
    pcf::IndiProperty m_indiP_fps;

   public:

    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_exploration );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_filename );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_learningToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_integratingToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_predictingToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_resetToggle );

    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_saveToggle );
    INDI_NEWCALLBACK_DECL( loPredCtrl, m_indiP_loadToggle );

    INDI_SETCALLBACK_DECL( loPredCtrl, m_indiP_fpsSource );

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


 inline loPredCtrl::loPredCtrl() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
 {
     return;
 }

 inline void loPredCtrl::setupConfig()
 {
     shmimMonitorT::setupConfig( config );
     FRAMEGRABBER_SETUP_CONFIG( config );
     TELEMETER_SETUP_CONFIG(config);

     config.add("parameters.fpsSource", "", "parameters.fpsSource", argType::Required, "parameters", "fpsSource", false, "string", "The device name for getting fps of the loop.");

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

    frameGrabberT::m_ownShmim = true;
    FRAMEGRABBER_LOAD_CONFIG(_config);
    TELEMETER_LOAD_CONFIG(_config);

    _config(m_fpsSource, "parameters.fpsSource");

    _config(m_gainCtrl, "parameters.gain");
    _config(m_regularizationCtrl, "parameters.regularization");
    _config(m_gammaCtrl, "parameters.gamma");
    _config(m_covarianceCtrl, "parameters.covariance");

    _config(m_num_modes, "parameters.num_modes");
    _config(m_history, "parameters.history");
    _config(m_future, "parameters.future");

	std::cout << "Gain " << m_gainCtrl << std::endl;
    std::cout << "Regularization " << m_regularizationCtrl << std::endl;
    std::cout << "Gamma " << m_gammaCtrl << std::endl;

    std::cout << "num modes " << m_num_modes << std::endl;
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

     CREATE_REG_INDI_NEW_TEXT( m_indiP_filename, "filename", "", "");

     createStandardIndiToggleSw( m_indiP_learningToggle, "learn", "Learning State", "Learn Controls");
	 registerIndiPropertyNew( m_indiP_learningToggle, INDI_NEWCALLBACK(m_indiP_learningToggle) );

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

 inline int loPredCtrl::appLogic()
 {
     if( shmimMonitorT::appLogic() < 0 )
     {
         return log<software_error, -1>( { __FILE__, __LINE__ } );
     }

    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

     std::unique_lock<std::mutex> lock( m_indiMutex );

     if( shmimMonitorT::updateINDI() < 0 )
     {
         log<software_error>( { __FILE__, __LINE__ } );
     }

    FRAMEGRABBER_UPDATE_INDI;

     updatesIfChanged<std::string>( m_indiP_exploration, { "current", "target" }, { m_exploration_sequence, m_exploration_sequence } );

     updatesIfChanged<std::string>( m_indiP_filename, { "current", "target" }, { m_filename, m_filename } );

     if(is_learning){
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::On, INDI_OK);
	 }else{
		 updateSwitchIfChanged(m_indiP_learningToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
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

 inline int loPredCtrl::appShutdown()
 {
     shmimMonitorT::appShutdown();

    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

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

    controller = new DDSPC::PredictiveController(m_num_modes, m_history, m_future, m_gainCtrl, m_gammaCtrl, m_regularizationCtrl, m_covarianceCtrl);

    return 0;
 }

 inline int loPredCtrl::processImage( void *curr_src, const dev::shmimT &dummy )
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
    // DDSPC::print_matrix(new_measurement, "new measurement");

    if(is_predictive_control){
        if(is_learning){
            new_command = controller->calculate_command(new_measurement, exp_noise);
        }else{
            new_command = controller->calculate_command(new_measurement, zero_exp_noise);
        }
        
    }else{
        for(int i=0; i < m_num_modes; i++){
            new_command(i,0) = -m_gainCtrl * new_measurement(i, 0);
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

int loPredCtrl::configureAcquisition()
{
    frameGrabberT::m_width = m_modevalWidth;
    frameGrabberT::m_height = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    return 0;
}

float loPredCtrl::fps()
{
    return m_fps;
}

int loPredCtrl::startAcquisition()
{
    return 0;
}

int loPredCtrl::acquireAndCheckValid()
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

int loPredCtrl::loadImageIntoStream(void * dest)
{
    memcpy( dest, full_command.data(), full_command.rows() * frameGrabberT::m_typeSize );

    m_updated = false;

    return 0;
}

int loPredCtrl::reconfig()
{
    return 0;
}

int loPredCtrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int loPredCtrl::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
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

INDI_NEWCALLBACK_DEFN( loPredCtrl, m_indiP_filename )( const pcf::IndiProperty &ipRecv )
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

INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_integratingToggle )(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_saveToggle )(const pcf::IndiProperty &ipRecv)
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

        save(m_filename);
        log<text_log>("saved state to " + m_filename, logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_saveToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}

   return 0;
}

INDI_NEWCALLBACK_DEFN(loPredCtrl, m_indiP_loadToggle )(const pcf::IndiProperty &ipRecv)
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

        load(m_filename);
        log<text_log>("loaded state from " + m_filename, logPrio::LOG_NOTICE);
		updateSwitchIfChanged(m_indiP_loadToggle, "request", pcf::IndiElement::Off, INDI_IDLE);
	}

   return 0;
}

INDI_SETCALLBACK_DEFN( loPredCtrl, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
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

 #endif // loPredCtrl_hpp
