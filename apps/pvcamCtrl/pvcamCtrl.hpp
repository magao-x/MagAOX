/** \file pvcamCtrl.hpp
  * \brief The MagAO-X pvcam controller header file
  *
  * \ingroup pvcamCtrl_files
  */

#ifndef pvcamCtrl_hpp
#define pvcamCtrl_hpp

#include <string.h>

// PVCAM
#include <master.h>
#include <pvcam.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup pvcamCtrl
  * \brief The pvcam controller application for teledyne cameras
  *
  * <a href="../handbook/operating/software/apps/pvcamCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

#ifndef PL_ERR_LIBRARY_NOT_INITIALIZED
#define PL_ERR_LIBRARY_NOT_INITIALIZED (157)
#endif

/// Format an error message using pvcam facilities
/**
  * \returns a string of format "<func> failed: <pl_error_message>. <more>"
  */ 
std::string pvcamErrMessage( const std::string & func,  ///< [in] the pvcam function which failed
                             int pec,                   ///< [in] the code from pl_error_code
                             const std::string & more   ///< [in] extra information to include
                           )
{
    char pvmsg[ERROR_MSG_LEN];
    pl_error_message(pec, pvmsg);
    
    std::string msg = func + " failed: " + pvmsg;
    if(more != "") msg += " " + more;

    return msg;
}

/// Helper for logging an error from pvcam
#define log_pvcam_software_error( func, /**< [in] the pvcam function which failed*/            \
                                  more  /**< [in] extra information to include*/               \
                                )                                                              \
{                                                                                              \
    int pec = pl_error_code();                                                                 \
    log<software_error>({__FILE__, __LINE__, 0, pec , pvcamErrMessage(func, pec, more)});      \
}


/** \defgroup pvcamCtrl_files
  * \ingroup pvcamCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X pvcam controller
/** 
  * \ingroup pvcamCtrl
  */
class pvcamCtrl : public MagAOXApp<true>, public dev::stdCamera<pvcamCtrl>, public dev::frameGrabber<pvcamCtrl>, 
                                                 public dev::dssShutter<pvcamCtrl>, 
                                                        public dev::telemeter<pvcamCtrl>
{

    typedef dev::stdCamera<pvcamCtrl> stdCameraT;
    typedef dev::frameGrabber<pvcamCtrl> frameGrabberT;
    typedef dev::dssShutter<pvcamCtrl> shutterT;
    typedef dev::telemeter<pvcamCtrl> telemeterT;

    //Give the test harness access.
    friend class pvcamCtrl_test;

    friend class dev::stdCamera<pvcamCtrl>;
    friend class dev::frameGrabber<pvcamCtrl>;
    friend class dev::dssShutter<pvcamCtrl>;
    friend class dev::telemeter<pvcamCtrl>;

    typedef MagAOXApp<> MagAOXAppT;

public:
    /** \name app::dev Configurations
      *@{
      */
    static constexpr bool c_stdCamera_tempControl = false; ///< app::dev config to tell stdCamera to expose temperature controls
   
    static constexpr bool c_stdCamera_temp = false; ///< app::dev config to tell stdCamera to expose temperature
   
    static constexpr bool c_stdCamera_readoutSpeed = true; ///< app::dev config to tell stdCamera to expose readout speed controls
   
    static constexpr bool c_stdCamera_vShiftSpeed = false; ///< app:dev config to tell stdCamera to expose vertical shift speed control

    static constexpr bool c_stdCamera_emGain = false; ///< app::dev config to tell stdCamera to expose EM gain controls 

    static constexpr bool c_stdCamera_exptimeCtrl = true; ///< app::dev config to tell stdCamera to expose exposure time controls
   
    static constexpr bool c_stdCamera_fpsCtrl = true; ///< app::dev config to tell stdCamera not to expose FPS controls

    static constexpr bool c_stdCamera_fps = true; ///< app::dev config to tell stdCamera not to expose FPS status
   
    static constexpr bool c_stdCamera_synchro = false; ///< app::dev config to tell stdCamera to not expose synchro mode controls
   
    static constexpr bool c_stdCamera_usesModes = false; ///< app:dev config to tell stdCamera not to expose mode controls
   
    static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

    static constexpr bool c_stdCamera_cropMode = false; ///< app:dev config to tell stdCamera to expose Crop Mode controls
   
    static constexpr bool c_stdCamera_hasShutter = true; ///< app:dev config to tell stdCamera to expose shutter controls
      
    static constexpr bool c_stdCamera_usesStateString = false; ///< app::dev confg to tell stdCamera to expose the state string property
   
    static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber this camera can be flipped
   
    ///@}

    
protected:

    /** \name Configurable Parameters
      *@{
      */
   
    std::string m_serialNumber; ///< The camera serial number.  

    ///\todo implement config-ing of this
    float m_tempTol {0.1}; ///< Tolerance in degrees C to declare the temperature control locked.

    uint32_t m_circBuffMaxBytes {536870912}; ///< Max size in bytes of the circular buffer to allocate.  Default is 0.5 GB.

    ///@}

    int16 m_handle{ -1 }; ///< Camera handle, set when camera is opened

    std::string m_camName; ///< Camera name, filled in as part of opening the camera.

    struct gain
    {
        int index;
        std::string name;

        int bitDepth;
    };

    struct speed
    {
        int index;
        int pixTime; 
        int minG;
        int maxG;
        std::vector<gain> gains;
    };

    struct port
    {
        int index;
        int value;
        std::string name;

        std::vector<speed> speeds;
    };

    std::vector<port> m_ports;

    bool m_8bit {false};

    uns32  m_circBuffBytes {0};
    uns8 * m_circBuff {nullptr};

    FRAME_INFO m_frameInfo;

    sem_t m_frSemaphore; ///< Semaphore used to signal that a frame is ready.


public:
    /// Default c'tor.
    pvcamCtrl();
 
    /// D'tor, declared and defined for noexcept.
    ~pvcamCtrl() noexcept
    {}
 
 
     // MagAOXApp:
 
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
 
    /// Implementation of the FSM for pvcamCtrl.
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


    // stdCamera interface:
    
    //This must set the power-on default values of
    /* -- m_ccdTempSetpt
     * -- m_currentROI 
     */
    int powerOnDefaults();
    
    int setTempControl();
    int setTempSetPt();
    int setReadoutSpeed();
    int setVShiftSpeed();
    int setEMGain();
    int setExpTime();
    int setFPS();
 
    /// Check the next ROI
    /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
      *
      * \returns 0 if successful
      * \returns -1 otherwise
      */
    int checkNextROI();
 
    int setNextROI();

    /// Sets the shutter state, via call to dssShutter::setShutterState(int) [stdCamera interface]
    /**
      * \returns 0 always
      */
    int setShutter(int sh);
    
    //Framegrabber interface:
    int configureAcquisition();
    float fps();
    int startAcquisition();
    int acquireAndCheckValid();
    int loadImageIntoStream(void * dest);
    int reconfig();

    // pvcam specific:
    int connect();

    int fillSpeedTable();

    void dumpEnum( uns32 paramID,
                   const std::string & paramMnem
                 );

    int getTemp();

    static void st_endOfFrameCallback( FRAME_INFO * finfo,
                                       void * pvcamCtrlInst 
                                     );

    void endOfFrameCallback( FRAME_INFO * finfo );

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_stdcam * );
   
   int recordTelem( const telem_fgtimings * );
   
   ///@}

};

pvcamCtrl::pvcamCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    m_powerMgtEnabled = true;
    m_powerOnWait = 15;

    m_expTime = 0.01;
    m_expTimeSet = 0.01;

    m_default_x = 1599.5;
    m_default_y = 1599.5;
    m_default_w = 3200;
    m_default_h = 3200;
    m_default_bin_x = 1;
    m_default_bin_y = 1;

    m_full_x = 1599.5;
    m_full_y = 1599.5;
    m_full_w = 3200;
    m_full_h = 3200;

    m_currentROI.x = m_default_x;
    m_currentROI.y = m_default_y;
    m_currentROI.w = m_default_w;
    m_currentROI.h = m_default_h;
    m_currentROI.bin_x = m_default_bin_x;
    m_currentROI.bin_y = m_default_bin_y;

    m_nextROI.x = m_currentROI.x;
    m_nextROI.y = m_currentROI.y;
    m_nextROI.w = m_currentROI.w;
    m_nextROI.h = m_currentROI.h;
    m_nextROI.bin_x = m_currentROI.bin_x;
    m_nextROI.bin_y = m_currentROI.bin_y;

    m_defaultReadoutSpeed  = "dynamic_range";
    m_readoutSpeedNames = {"sensitivity", "speed", "dynamic_range", "sub_electron"};
    m_readoutSpeedNameLabels = {"Sensitivity", "Speed", "Dynamic Range", "Sub-Electron"};

    m_readoutSpeedName  = m_defaultReadoutSpeed;
    m_readoutSpeedNameSet = m_defaultReadoutSpeed;


    return;
}

void pvcamCtrl::setupConfig()
{
    config.add("camera.serialNumber", "", "camera.serialNumber", argType::Required, "camera", "serialNumber", false, "int", "The identifying serial number of the camera.");

    //put this in camera because it is a camera interface config, not a framegrabber thing per se:
    config.add("camera.circBuffMaxBytes", "", "camera.circBuffMaxBytes", argType::Required, "camera", "circBuffMaxBytes", false, "int", "Maximum size in bytes of the circular buffer to allocate.  Default is 0.5 GB.");

    stdCameraT::setupConfig(config);
    dev::frameGrabber<pvcamCtrl>::setupConfig(config);
    shutterT::setupConfig(config);
    telemeterT::setupConfig(config);
}

int pvcamCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
    _config(m_serialNumber, "camera.serialNumber");
    _config(m_circBuffMaxBytes, "camera.circBuffMaxBytes");

    if(m_serialNumber == "")
    {
        log<text_log>("camera serial number not provided", logPrio::LOG_CRITICAL);
        return -1;
    }

    stdCameraT::loadConfig(_config);

    m_currentROI.x = m_default_x;
    m_currentROI.y = m_default_y;
    m_currentROI.w = m_default_w;
    m_currentROI.h = m_default_h;
    m_currentROI.bin_x = m_default_bin_x;
    m_currentROI.bin_y = m_default_bin_y;

    m_nextROI.x = m_currentROI.x;
    m_nextROI.y = m_currentROI.y;
    m_nextROI.w = m_currentROI.w;
    m_nextROI.h = m_currentROI.h;
    m_nextROI.bin_x = m_currentROI.bin_x;
    m_nextROI.bin_y = m_currentROI.bin_y;
    
    dev::frameGrabber<pvcamCtrl>::loadConfig(_config);
    shutterT::loadConfig(_config);
    telemeterT::loadConfig(_config);
   

   return 0;
}

void pvcamCtrl::loadConfig()
{
    if(loadConfigImpl(config) != 0)
    {
        log<text_log>("error loading config", logPrio::LOG_CRITICAL);
        m_shutdown = true;
    }
}

int pvcamCtrl::appStartup()
{
    if(stdCameraT::appStartup() < 0)
    {
        return log<software_critical,-1>({__FILE__,__LINE__});
    }

    if(sem_init(&m_frSemaphore, 0, 0) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Initializing frame ready semaphore"});
    }

    if(frameGrabberT::appStartup() < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__});
    }

    if(shutterT::appStartup() < 0)
    {
       return log<software_critical,-1>({__FILE__,__LINE__});
    }

    if(telemeterT::appStartup() < 0)
    {
       return log<software_critical,-1>({__FILE__,__LINE__});
    }

    return 0;
}

int pvcamCtrl::appLogic()
{
    ///\todo why do we run dev appLogics first?

    if( state() == stateCodes::POWERON)
    {
        if(!powerOnWaitElapsed())
        {
            return 0;
        }

        state(stateCodes::NOTCONNECTED);
    }

    //run stdCamera's appLogic
    if(stdCameraT::appLogic() < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__});
    }

    // run frammerGrabbers's appLogic
    if(frameGrabberT::appLogic() < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__});
    }

    //and run dssShutter's appLogic
    if(shutterT::appLogic() < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__});
    }

    if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
    {
        m_reconfig = true; //Trigger a f.g. thread reconfig.
  
        //Might have gotten here because of a power off.
        if(powerState() != 1) 
        {
            return 0;
        }

        std::unique_lock<std::mutex> lock(m_indiMutex);
        if(connect() < 0)
        {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__});
        }
  
        if(state() != stateCodes::CONNECTED) return 0;
    }

    if( state() == stateCodes::CONNECTED)
    {
        state(stateCodes::READY);
        state(stateCodes::OPERATING);
    }

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING)
    {
        if(getTemp() != 0)
        {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__});
            return 0;
        }
        
        if(stdCameraT::updateINDI() < 0)
        {
            return log<software_error,0>({__FILE__,__LINE__});
        }
      
        if(frameGrabberT::updateINDI() < 0)
        {
            return log<software_error,0>({__FILE__,__LINE__});
        }

        if(telemeterT::appLogic() < 0)
        {
            log<software_error>({__FILE__, __LINE__});
        }

        recordCamera();
    }


    return 0;
}

int pvcamCtrl::appShutdown()
{
    if(m_handle != -1)
    {
        if(!pl_cam_close(m_handle))
        {
            log_pvcam_software_error("pl_cam_close", "continuing");
        }
        m_handle = -1;
    }

    if(!pl_pvcam_uninit())
    {
        if(pl_error_code() != PL_ERR_LIBRARY_NOT_INITIALIZED) /// \todo this error code is manually defined
        {
            log_pvcam_software_error("pl_pvcam_uninit", "continuing");
        }
    }

    ///\todo error check these base class fxns.
    if(frameGrabberT::appShutdown() < 0)
    {
        log<software_error>({__FILE__, __LINE__, "error from frameGrabberT::appShutdown()"});
    }

    if(shutterT::appShutdown() < 0)
    {
        log<software_error>({__FILE__, __LINE__, "error from shutterT::appShutdown()"});
    }

    if(telemeterT::appShutdown() < 0)
    {
        log<software_error>({__FILE__, __LINE__, "error from telemeterT::appShutdown()"});
    }

    return 0;
}

int pvcamCtrl::powerOnDefaults()
{
    m_expTime = 0.01;
    m_expTimeSet = 0.01;

    m_currentROI.x = m_default_x;
    m_currentROI.y = m_default_y;
    m_currentROI.w = m_default_w;
    m_currentROI.h = m_default_h;
    m_currentROI.bin_x = m_default_bin_x;
    m_currentROI.bin_y = m_default_bin_y;
   
    m_nextROI.x = m_default_x;
    m_nextROI.y = m_default_y;
    m_nextROI.w = m_default_w;
    m_nextROI.h = m_default_h;
    m_nextROI.bin_x = m_default_bin_x;
    m_nextROI.bin_y = m_default_bin_y;

    m_readoutSpeedName  = m_defaultReadoutSpeed;
    m_readoutSpeedNameSet = m_defaultReadoutSpeed;

    return 0;
}

int pvcamCtrl::setTempControl()
{
    return 0;
}

int pvcamCtrl::setTempSetPt()
{
    return 0;
}

int pvcamCtrl::setReadoutSpeed()
{
    frameGrabberT::m_reconfig = true;

    return 0;
}

int pvcamCtrl::setVShiftSpeed()
{
    return 0;
}

int pvcamCtrl::setEMGain()
{
    return 0;
}

int pvcamCtrl::setExpTime()
{
    ulong64 minExpTime, maxExpTime;

    if(!pl_get_param(m_handle, PARAM_EXPOSURE_TIME, ATTR_MIN, static_cast<void *>(&minExpTime)))
    {
        log_pvcam_software_error("pl_get_param", "PARAM_EXPOSURE_TIME ATTR_MIN");
        log<software_error>({__FILE__, __LINE__, "could not set exposure time"});
        return -1;
    }

    if(!pl_get_param(m_handle, PARAM_EXPOSURE_TIME, ATTR_MAX, static_cast<void *>(&maxExpTime)))
    {
        log_pvcam_software_error("pl_get_param", "PARAM_EXPOSURE_TIME ATTR_MAX");
        log<software_error>({__FILE__, __LINE__, "could not set exposure time"});
        return -1;
    }

    std::cerr << "Exposure time min: " << minExpTime << " max: " << maxExpTime << "\n";

    if(m_expTimeSet*1e6 < minExpTime)
    {
        m_expTimeSet = (int) (minExpTime/1e6+0.5);
        log<text_log>("increased m_expTimeSet to: " + std::to_string(m_expTimeSet), logPrio::LOG_INFO);
    }

    if(m_expTimeSet * 1e6 > maxExpTime)
    {
        m_expTimeSet = (int) (maxExpTime / 1e6 - 0.5);
        log<text_log>("decreased m_expTimeSet to: " + std::to_string(m_expTimeSet), logPrio::LOG_INFO); 
    }

    frameGrabberT::m_reconfig = true;
    
    return 0;
}

int pvcamCtrl::setFPS()
{
    m_expTimeSet = 1.0/m_fpsSet;
    return setExpTime();
}

int pvcamCtrl::checkNextROI()
{
    if(m_nextROI.w > 3200)
    {
        m_nextROI.w = 3200;
    }

    int x0 = m_nextROI.x - 0.5*m_nextROI.w;

    if(x0 < 0) 
    {
        m_nextROI.w += x0;
        m_nextROI.x = 0;
    }

    int x1 = m_nextROI.x + 0.5*m_nextROI.w;

    if(x1 > 3199)
    {
        m_nextROI.w = 3199 - x0;
        m_nextROI.x = x0 + 0.5*m_nextROI.w;
    }

    if(m_nextROI.h > 3200)
    {
        m_nextROI.h = 3200;
    }

    int y0 = m_nextROI.y - 0.5 * m_nextROI.h;

    if(y0 < 0)
    {
        m_nextROI.h += y0;
        m_nextROI.y = 0;
    }

    int y1 = m_nextROI.y + 0.5 * m_nextROI.h;

    if(y1 > 3199)
    {
        m_nextROI.h = 3199 - y0;
        m_nextROI.y = y0 + 0.5 * m_nextROI.h;
    }

    return 0;
}

int pvcamCtrl::setNextROI()
{
    //This is done in setup acq
    frameGrabberT::m_reconfig = true;
    return 0;
}

int pvcamCtrl::setShutter(int sh)
{
    recordCamera(true);
    return shutterT::setShutterState(sh);
}

int pvcamCtrl::configureAcquisition()
{

    recordCamera(true);
    //-- 0: register the callback
    if(pl_cam_deregister_callback(m_handle, PL_CALLBACK_EOF) == false) // Because we registered it last time we configured acq:
    {
        log_pvcam_software_error("pl_cam_deregister_callback", "PL_CALLBACK_EOF");
    }

    if(pl_cam_register_callback_ex3(m_handle, PL_CALLBACK_EOF, reinterpret_cast<void *>(&st_endOfFrameCallback), static_cast<void *>(this)) != true)
    {
        log_pvcam_software_error("pl_cam_register_callback_ex3", "PL_CALLBACK_EOF");
        return -1;
    }

    //-- 1: Set speed:
    int32 value;

    if(m_readoutSpeedNameSet == "sensitivity")
    {
        value=0;
        m_8bit = false;
    }
    else if(m_readoutSpeedNameSet == "speed")
    {
        value=1;
        m_8bit = true;
    }
    else if(m_readoutSpeedNameSet == "dynamic_range")
    {
        value=2;
        m_8bit = false;
    }
    else if(m_readoutSpeedNameSet == "sub_electron")
    {
        value=3;
        m_8bit = false;
    }
    else
    {
        value = 2;
        m_8bit = false;
        m_readoutSpeedNameSet = "dynamic_range";
    }
    m_readoutSpeedName = m_readoutSpeedNameSet;

    if(pl_set_param(m_handle, PARAM_READOUT_PORT, static_cast<void *>(&value)) == false)
    {
        log_pvcam_software_error("pl_set_param", "PARAM_READOUT_PORT");
        return -1;
    }

    //-- 2: Set ROI:
    rgn_type pvROI;
    pvROI.s1 = m_nextROI.x - 0.5*m_nextROI.w;
    pvROI.s2 = pvROI.s1 + m_nextROI.w-1;
    pvROI.sbin = m_nextROI.bin_x;
    pvROI.p1 = m_nextROI.y - 0.5*m_nextROI.h;
    pvROI.p2 = pvROI.p1 + m_nextROI.h-1;
    pvROI.pbin = m_nextROI.bin_y;

    m_currentROI = m_nextROI;

    m_width = (pvROI.s2 - pvROI.s1 + 1) / pvROI.sbin;
    m_height = (pvROI.p2 - pvROI.p1 + 1) / pvROI.pbin;
    m_dataType = _DATATYPE_UINT16;


    //-- 3: Setup continuous acquisition
    //std::cerr << pvROI.s1 << " " << pvROI.s2 << " " << pvROI.sbin << " " << pvROI.p1 << " " << pvROI.p2 << " " << pvROI.pbin << "\n";
    uns32 fsize;

    uns32 exptime = m_expTimeSet * 1e6;
    if(pl_exp_setup_cont(m_handle, 1, &pvROI, TIMED_MODE, exptime, &fsize, CIRC_OVERWRITE) == false)
    {
        log_pvcam_software_error("pl_exp_setup_cont", "");
        m_shutdown = true;
        return -1;
    }

    if (pl_get_param(m_handle, PARAM_EXPOSURE_TIME, ATTR_CURRENT, &exptime) == false)
    {
        log_pvcam_software_error("pl_get_param", "PARAM_EXPOSURE_TIME");
    }
    m_expTime = (1.0*exptime) / 1e6;
    m_expTimeSet = m_expTime;

    uns32 readouttime=0;
    if(pl_get_param(m_handle, PARAM_READOUT_TIME, ATTR_CURRENT, &readouttime) == false)
    {
        log_pvcam_software_error("pl_get_param", "PARAM_READOUT_TIME");
    }

    long64 predelay = 0;
    if(pl_get_param(m_handle, PARAM_PRE_TRIGGER_DELAY, ATTR_CURRENT, &predelay) == false)
    {
        log_pvcam_software_error("pl_get_param", "PARAM_PRE_TRIGGER_DELAY");
    }

    long64 postdelay = 0;
    if(pl_get_param(m_handle, PARAM_POST_TRIGGER_DELAY, ATTR_CURRENT, &postdelay) == false)
    {
        log_pvcam_software_error("pl_get_param", "PARAM_POST_TRIGGER_DELAY");
    }

    m_fps = 1.0/(m_expTime + predelay/1e9 + postdelay/1e9);
    m_fpsSet = m_fps;

    //-- 4: Allocate the acq circular buffer
    if(m_circBuff != nullptr)
    {
        delete[] m_circBuff;
        m_circBuffBytes = 0;
        m_circBuff = nullptr;
    }

    m_circBuffBytes = ( (uns32) (m_circBuffMaxBytes / fsize)) * fsize;

    m_circBuff = new (std::nothrow) uns8[m_circBuffBytes];

    if(m_circBuff == nullptr)
    {
        log<software_critical>({__FILE__, __LINE__, "failed to allocate acquisition circular buffer."});
        state(stateCodes::FAILURE);
        return -1;
    }

    recordCamera(true);

    return 0;
}

float pvcamCtrl::fps()
{
    return m_fps;
}

int pvcamCtrl::startAcquisition()
{
    if(pl_exp_start_cont(m_handle, m_circBuff, m_circBuffBytes) == false)
    {
        log_pvcam_software_error("pv_exp_start_cont", "");
        return -1;
    }

    log<text_log>("continuous acquisition started", logPrio::LOG_INFO);
    return 0;
}

int pvcamCtrl::acquireAndCheckValid()
{
    int rv = sem_trywait(&m_frSemaphore);
    if( rv == 0)
    {
        return 0;
    }
    else if(errno != EAGAIN)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "sem_trywait"});
        return -1;
    }

    timespec ts;

    if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"});
        return -1;
    }

    mx::sys::timespecAddNsec(ts, 1e9);

    rv = sem_timedwait(&m_frSemaphore, &ts);
    if(rv == 0)
    {
        return 0;
    }
    else if(errno != EAGAIN && errno != ETIMEDOUT)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "sem_timedwait"});
        return -1;
    }

    return 1;
}

int pvcamCtrl::loadImageIntoStream(void *dest)
{
    clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
    
    // Obtain a pointer to the last acquired frame
    uns8 *frame;
    if(pl_exp_get_latest_frame(m_handle, reinterpret_cast<void **>(&frame)) == false)
    {
        log_pvcam_software_error("pl_exp_get_latest_frame", "");
    }

    if(m_8bit)
    {
        uint16_t * dest16 = static_cast<uint16_t*>(dest);
        uint8_t * src8 = static_cast<uint8_t*>(frame);
        for(uint32_t i=0; i < m_width*m_height; ++i)
        {
            dest16[i] = src8[i];
        }
    }
    else
    {
        memcpy(dest, frame, m_width*m_height*2);
    }

    return 0;
}

int pvcamCtrl::reconfig()
{
    if(pl_exp_stop_cont(m_handle,CCS_HALT) == false)
    {
        log_pvcam_software_error("pl_exp_stop_cont", "");
    }
    return 0;
}

int pvcamCtrl::connect()
{

    //In picam, we had to initialize every time.  We'll do that here too.

    //So close handle if it's open
    if(m_handle != -1)
    {
        if(!pl_cam_close(m_handle))
        {
            log_pvcam_software_error("pl_cam_close", "");
            return -1;
        }
        m_handle = -1;
    }

    //Uninit
    if(!pl_pvcam_uninit())
    {
        if(pl_error_code() != PL_ERR_LIBRARY_NOT_INITIALIZED) /// \todo this error code is manually defined
        {
            log_pvcam_software_error("pl_pvcam_uninit", "continuing");
        }
    }

    if(!pl_pvcam_init())
    {
        log_pvcam_software_error("pl_pvcam_init", "");
        return -1;
    }

    int16 nrOfCameras;

    // Read the number of cameras in the system.
    // This will return total number of PVCAM cameras regardless of interface.
    if(pl_cam_get_total(&nrOfCameras) != PV_OK)
    {
        log_pvcam_software_error("pl_cam_get_total", "");
        return -1;
    }

    if(nrOfCameras == 0)
    {
        if(!stateLogged())
        {
            log<text_log>("Found 0 pvcam cameras.", logPrio::LOG_INFO);
        }
        state(stateCodes::NODEVICE);
        return 0;
    }

    log<text_log>("Found " + std::to_string(nrOfCameras) + " pvcam cameras.", logPrio::LOG_INFO);

    
    for(int n=0; n < nrOfCameras; ++n)
    {
        char camName[CAM_NAME_LEN] {'\0'};

        // Obtain PVCAM-name for this particular camera
        if(pl_cam_get_name(n, camName) != PV_OK)
        {
            log_pvcam_software_error("pl_cam_get_name", "");
            return -1;
        }
    
        int16_t handle = -1;

        //Open to check its serial number
        if(!pl_cam_open(camName, &handle, OPEN_EXCLUSIVE))
        {
            log_pvcam_software_error("pl_cam_open", ""); //We log this for now, but with 2 apps running we prob want to ignore 
            continue;
        }
        
        // Read the version of the Device Driver

        rs_bool isAvailable;
        if(!pl_get_param(handle, PARAM_HEAD_SER_NUM_ALPHA, ATTR_AVAIL, static_cast<void*>(&isAvailable)))
        {
            log_pvcam_software_error("pl_get_param", "PARAM_HEAD_SER_NUM_ALPHA ATTR_AVAIL");

            if(!pl_cam_close(handle))
            {
                log_pvcam_software_error("pl_cam_close", "");
            }

            return -1;
        }

        if(isAvailable)
        {
            char camSerial[MAX_ALPHA_SER_NUM_LEN]{ '\0' };

            if(!pl_get_param(handle, PARAM_HEAD_SER_NUM_ALPHA, ATTR_CURRENT, static_cast<void*>(camSerial)))
            {
                log_pvcam_software_error("pl_get_param", "PARAM_HEAD_SER_NUM_ALPHA ATTR_CURRENT");

                if(!pl_cam_close(handle))
                {
                    log_pvcam_software_error("pl_cam_close", "");
                }

                return -1;
            }       

            if(camSerial == m_serialNumber) 
            {
                state(stateCodes::NOTCONNECTED); //not strictly true, but logically true until m_handle is set
                m_camName = camName; 
                m_handle = handle;
                break;
            }    
        }  

        //If we're here then either it didn't have a serial number, or it didn't have the right serial number
        if(!pl_cam_close(handle))
        {
            log_pvcam_software_error("pl_cam_close", "");
            return -1;
        }

    }

    if(m_handle > -1)
    {
        log<text_log>("Opened camera " + m_serialNumber + " at " + m_camName, logPrio::LOG_INFO);
        state(stateCodes::CONNECTED);


        int32 res;
        uns16 idx;

        //Set exposure resolution to usec.

        idx = 1;
        if(pl_set_param(m_handle, PARAM_EXP_RES_INDEX, &idx) == false)
        {
            log_pvcam_software_error("pl_set_param", "PARAM_EXP_RES_INDEX");
        }

        if (pl_get_param(m_handle, PARAM_EXP_RES, ATTR_CURRENT, &res) == false)
        {
            log_pvcam_software_error("pl_get_param", "PARAM_EXP_RES");
        }

        if (pl_get_param(m_handle, PARAM_EXP_RES_INDEX, ATTR_CURRENT, &idx) == false)
        {
            log_pvcam_software_error("pl_get_param", "PARAM_EXP_RES_INDEX");
        }

        fillSpeedTable();
    }
    else
    {
        if(!stateLogged())
        {
            log<text_log>("camera not found", logPrio::LOG_INFO);
        }
        state(stateCodes::NODEVICE);
    }

    return 0;
}

int pvcamCtrl::fillSpeedTable()
{
    if(state() != stateCodes::CONNECTED && state() != stateCodes::READY )
    {
        return -1;
    }
        
    uns32 nports;

    if(pl_get_param(m_handle, PARAM_READOUT_PORT, ATTR_COUNT, static_cast<void *>(&nports)) == false)
    {
        log_pvcam_software_error("pl_get_param", "PARAM_READOUT_PORT");
        return -1;
    }

    std::cerr << "Found " << nports << " ports\n";

    m_ports.resize(nports);

    for(uns32 p=0; p < nports; ++p)
    {
        uns32 strLength;
        if(pl_enum_str_length(m_handle, PARAM_READOUT_PORT, p, &strLength) == false)
        {
            log_pvcam_software_error("pl_enum_str_length", "PARAM_READOUT_PORT");
            return -1;
        }

        char* text = new (std::nothrow) char[strLength];
        if (!text)
        {
            ///\todo log this properly
            std::cerr << "failed to allocate string\n";
            return -1;
        }

        int32 value;
        if(pl_get_enum_param(m_handle, PARAM_READOUT_PORT, p, &value, text, strLength) == false)
        {
            log_pvcam_software_error("pl_get_enum_param", "PARAM_READOUT_PORT");
            delete [] text;
            return false;
        }

        m_ports[p].index = p;
        m_ports[p].value = value;
        m_ports[p].name = text;

        std::cerr << "Port: " << p << " name: " << text << " value: " << value << "\n";

        delete [] text;

        if(pl_set_param(m_handle, PARAM_READOUT_PORT, static_cast<void *>(&value)) == false)
        {
            log_pvcam_software_error("pl_set_param", "PARAM_READOUT_PORT");
            return -1;
        }

        uns32 nspeeds;
        if(pl_get_param(m_handle, PARAM_SPDTAB_INDEX, ATTR_COUNT, static_cast<void *>(&nspeeds)) == false)
        {
            log_pvcam_software_error("pl_get_param", "PARAM_SPDTAB_INDEX");
            return -1;
        }
        std::cerr << "    Speeds: " << nspeeds << "\n";

        m_ports[p].speeds.resize(nspeeds);

        for(uns32 s = 0; s < nspeeds; ++s)
        {
            if(pl_set_param(m_handle, PARAM_SPDTAB_INDEX, static_cast<void *>(&s)) == false)
            {
                log_pvcam_software_error("pl_set_param", "PARAM_SPDTAB_INDEX");
                return -1;
            }

            uns16 pixtime;
            if(pl_get_param(m_handle, PARAM_PIX_TIME, ATTR_CURRENT, static_cast<void *>(&pixtime)) == false)
            {
                log_pvcam_software_error("pl_get_param", "PARAM_PIX_TIME");
                return -1;
            }

            m_ports[p].speeds[s].pixTime = pixtime;

            uns32 ngains;
            if(pl_get_param(m_handle, PARAM_GAIN_INDEX, ATTR_COUNT, static_cast<void *>(&ngains)) == false)
            {
                log_pvcam_software_error("pl_get_param", "PARAM_GAIN_INDEX ATTR_COUNT");
                return -1;
            }

            int16 ming;
            if(pl_get_param(m_handle, PARAM_GAIN_INDEX, ATTR_MIN, static_cast<void *>(&ming)) == false)
            {
                log_pvcam_software_error("pl_get_param", "PARAM_GAIN_INDEX ATTR_MIN");
                return -1;
            }

            int16 maxg;
            if(pl_get_param(m_handle, PARAM_GAIN_INDEX, ATTR_MIN, static_cast<void *>(&maxg)) == false)
            {
                log_pvcam_software_error("pl_get_param", "PARAM_GAIN_INDEX ATTR_MAX");
                return -1;
            }

            std::cerr << "      Speed: " << s << " " << " pixtime: " << pixtime << " gains: " << ngains << " [" << ming << "-" << maxg << "]\n";

            m_ports[p].speeds[s].minG = ming;
            m_ports[p].speeds[s].maxG = maxg;

            m_ports[p].speeds[s].gains.resize(ngains);

            for(uns32 g=0; g < ngains; ++g)
            {
                
                int16 gg = ming + g;
                if(pl_set_param(m_handle, PARAM_GAIN_INDEX, static_cast<void *>(&gg)) == false)
                {
                    log_pvcam_software_error("pl_set_param", "PARAM_GAIN_INDEX");
                    return -1;
                }

                int16 bitdepth;
                if(pl_get_param(m_handle, PARAM_BIT_DEPTH, ATTR_CURRENT, static_cast<void *>(&bitdepth)) == false)
                {
                    log_pvcam_software_error("pl_get_param", "PARAM_BIT_DEPTH ATTR_CURRENT");
                    return -1;
                }
                std::cerr << "         Gain: " << g << " bitdepth: " << bitdepth << "\n";
            }
        }
    }

    return 0;
}

void pvcamCtrl::dumpEnum(uns32 paramID, const std::string & paramMnem)
{
    
    if(state() == stateCodes::CONNECTED)
    {
        uns32 count;

        if(PV_OK != pl_get_param(m_handle, paramID, ATTR_COUNT, static_cast<void *>(&count)))
        {
            log_pvcam_software_error("pl_get_param", paramMnem);
            // TODO: Handle error
            return;
        }
        if(count == 0)
        {
            std::cerr << paramMnem << ": count 0\n";
            return;
        }
        for (uns32 n = 0; n < count; ++n)
        {
            uns32 strLength;
            if(PV_OK != pl_enum_str_length(m_handle, paramID, n, &strLength))
            {
                log_pvcam_software_error("pl_enum_str_length", paramMnem);
                // TODO: Handle error
                break;
            }
            char *text = new (std::nothrow) char[strLength];
            if(!text)
            {
                // TODO: Handle error
                break;
            }
            int32 value;
            if(PV_OK != pl_get_enum_param(m_handle, paramID, n, &value, text, strLength))
            {
                log_pvcam_software_error("pl_get_enum_param", paramMnem);
                // TODO: Handle error
                delete[] text;
                break;
            }
            std::cerr << paramMnem;
            fprintf(stderr, " item at index %u, value: %d, text: '%s'\n", n, value, text);
            delete[] text;
        }        
    }
    else
    {
        std::cerr << "dumpEnum: not CONNECTED\n";
    }
}


int pvcamCtrl::getTemp()
{
    if(state() == stateCodes::OPERATING)
    {
        return 0;
    }

    rs_bool isAvailable;
    if(!pl_get_param(m_handle, PARAM_TEMP_SETPOINT, ATTR_AVAIL, static_cast<void*>(&isAvailable)))
    {
        if(powerState() != 1 || powerStateTarget() != 1) return 0;
        log_pvcam_software_error("pl_get_param", "PARAM_TEMP ATTR_AVAIL");
        state(stateCodes::ERROR);
    }

    int16 stemp;
    if(isAvailable) //Maybe this is a separate check.  Don't yet know what happens when acquiring
    {
        if(!pl_get_param(m_handle, PARAM_TEMP_SETPOINT, ATTR_CURRENT, static_cast<void*>(&stemp)))
        {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log_pvcam_software_error("pl_get_param", "PARAM_TEMP ATTR_AVAIL");
            state(stateCodes::ERROR);
        }

        m_ccdTempSetpt = stemp/100.0;
    }

    if(!pl_get_param(m_handle, PARAM_TEMP, ATTR_AVAIL, static_cast<void*>(&isAvailable)))
    {
        if(powerState() != 1 || powerStateTarget() != 1) return 0;
        log_pvcam_software_error("pl_get_param", "PARAM_TEMP ATTR_AVAIL");
        state(stateCodes::ERROR);
    }

    int16 ctemp;
    if(isAvailable) //Maybe this is a separate check.  Don't yet know what happens when acquiring
    {
        if(!pl_get_param(m_handle, PARAM_TEMP, ATTR_CURRENT, static_cast<void*>(&ctemp)))
        {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log_pvcam_software_error("pl_get_param", "PARAM_TEMP ATTR_AVAIL");
            state(stateCodes::ERROR);
        }

        m_ccdTemp = ctemp/100.0;
    }

    if(fabs(m_ccdTemp - m_ccdTempSetpt) > m_tempTol) 
    {
        m_tempControlStatus = true;
        m_tempControlOnTarget = false;
        m_tempControlStatusStr = "UNLOCKED";
    }
    else 
    {
        m_tempControlStatus = true;
        m_tempControlOnTarget = true;
        m_tempControlStatusStr = "LOCKED";
    }

    return 0;
}

void pvcamCtrl::st_endOfFrameCallback( FRAME_INFO *finfo,
                                       void *pvcamCtrlInst
                                     )
{
    static_cast<pvcamCtrl *>(pvcamCtrlInst)->endOfFrameCallback(finfo);
}

void pvcamCtrl::endOfFrameCallback(FRAME_INFO *finfo)
{
    m_frameInfo = *finfo;

    // Now tell the writer to get going
    if(sem_post(&m_frSemaphore) < 0)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to frame ready semaphore"});
        return;
    }
}

int pvcamCtrl::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_stdcam());
}
   
int pvcamCtrl::recordTelem(const telem_stdcam *)
{
   return recordCamera(true);
}

inline
int pvcamCtrl::recordTelem( const telem_fgtimings * )
{
   return recordFGTimings(true);
}

} // namespace app
} //namespace MagAOX

#endif //pvcamCtrl_hpp
