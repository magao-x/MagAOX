/** \file dmPokeCenter.hpp
  * \brief The MagAO-X DM Poke Centering header file
  *
  * \ingroup dmPokeCenter_files
  */



#ifndef dmPokeCenter_hpp
#define dmPokeCenter_hpp

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/improc/circleOuterpix.hpp>
using namespace mx::improc;

#include <mx/math/fit/fitGaussian.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup dmPokeCenter
  * \brief The MagAO-X application to center a DM pupil by poking actuators
  *
  * <a href="../handbook/operating/software/apps/dmPokeCenter.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

struct wfsShmimT
{
   static std::string configSection()
   {
      return "wfscam";
   };
   
   static std::string indiPrefix()
   {
      return "wfscam";
   };
};

/** \defgroup dmPokeCenter_files
  * \ingroup dmPokeCenter
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X DM Pupil Centering Application
/** 
  * \ingroup dmPokeCenter
  */
class dmPokeCenter : public MagAOXApp<true>, public dev::shmimMonitor<dmPokeCenter, wfsShmimT>, public dev::telemeter<dmPokeCenter>
{
    //Give the test harness access.
    friend class dmPokeCenter_test;

    friend class dev::shmimMonitor<dmPokeCenter, wfsShmimT>;

    friend class dev::telemeter<dmPokeCenter>;

    typedef dev::shmimMonitor<dmPokeCenter, wfsShmimT> shmimMonitorT;

    typedef dev::telemeter<dmPokeCenter> telemeterT;

protected:

    /** \name Configurable Parameters
      *@{
      */
   
    std::string m_wfsCamDevName; ///<INDI device name of the WFS camera.  Default is wfscam.shmimName.

    double m_wfsSemWait {1.5}; ///< The time in sec to wait on the WFS semaphore.  Default 0.5 sec.

    double m_imageSemWait {0.5}; ///< The time in sec to wait on the image semaphore.  Default 0.5 sec.

    unsigned m_nDarks {5}; ///< The number of images to average for the dark.  Default is 5.

    unsigned m_nPupilImages {20}; ///< The number of images to average for the pupil image.  Default is 20.

    unsigned m_nPokeImages {5}; ///< The number of images to average for the poke images.  Default is 5.

    std::string m_dmChan;

    std::vector<int> m_poke_x;
    std::vector<int> m_poke_y;

    float m_poke_amp {0.0};

    float m_dmSleep {10000}; ///<The time to sleep for the DM command to be applied, in microseconds. Default is 10000.

    // Pupil fitting:
    int m_pupilPixels = 68600; ///< The number of pixels in the pupil. Default is 68600.

    int m_pupilCutBuff = 20; ///< The buffer around the initial found-pupil to include in the cut image.  />= 0, default 20.

    float m_pupilMag = 10; ///< The magnification to apply to the pupil image. />= 1, default 10.

    float m_pupilMedThresh = 0.9; ///< Threshold in the magnified image as a fraction of the median.  />0, /<=1, default 0.9.

    int m_pokeBlockW = 64; ///< The size of the sub-image for the poke analysis
    
    int m_pokeFWHMGuess = 2; ///< The initial guess for the FWHM of the Gaussian fit to the poke.

    ///@}

    std::mutex m_wfsImageMutex;

    milkImage<float> m_rawImage;
    
    milkImage<float> m_wfsDark;

    milkImage<float> m_pupilImage;

    std::vector<milkImage<float>> m_pokeImages;

    float (*wfsPixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as float

    float m_wfsFps {-1}; ///< The WFS camera FPS

    int m_shutter {-1}; ///< Shutter status.  -1 is unknown, 0 open, 1 shut.

    milkImage<float> m_dmStream;

    eigenImage<float> m_dmImage;
    
    //Working memory for pupil fitting
    eigenImage<float> m_pupilCopy;
    eigenImage<float> m_fullEdge;
    eigenImage<float> m_fullMask;
    eigenImage<float> m_cutEdge;
    eigenImage<float> m_cutMask;
    eigenImage<float> m_pupilCut;
    eigenImage<float> m_pupilMagnified;
    eigenImage<float> m_magMask;
    eigenImage<float> m_magEdge;

    float m_pupilX {0};
    float m_pupilY {0};
    
    //Working memory for poke fitting
    mx::math::fit::fitGaussian2Dsym<float> m_gfit;
    eigenImage<float> m_pokeBlock;

    float m_pokeX {0};
    float m_pokeY {0};

public:
    /// Default c'tor.
    dmPokeCenter();

    /// D'tor, declared and defined for noexcept.
    ~dmPokeCenter() noexcept
    {}

    /**\name MagAOX Interface
      *
      * @{ 
      */
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

    /// Implementation of the FSM for dmPokeCenter.
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

    ///@}

    /** \name shmimMonitor Interface
      * @{
      */

    int allocate( const wfsShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/);
    
    int processImage( void * curr_src,   ///< [in] pointer to the start of the current frame 
                      const wfsShmimT &  ///< [in] tag to differentiate shmimMonitor parents. 
                    );
    ///@}

    /** \name WFS Thread 
      * This thread coordinates the WFS process
      *
      * @{
      */ 
protected:

    int m_wfsThreadPrio {1}; ///< Priority of the WFS thread, should normally be > 00.

    std::string m_wfsCpuset; ///< The cpuset for the framegrabber thread.  Ignored if empty (the default).

    std::thread m_wfsThread; ///< A separate thread for the actual WFSing

    bool m_wfsThreadInit {true}; ///< Synchronizer to ensure wfs thread initializes before doing dangerous things.
  
    pid_t m_wfsThreadID {0}; ///< WFS thread PID.

    pcf::IndiProperty m_wfsThreadProp; ///< The property to hold the WFS thread details.
 
    ///Thread starter, called by wfsThreadStart on thread construction.  Calls wfsThreadExec.
    static void wfsThreadStart( dmPokeCenter * s /**< [in] a pointer to an streamWriter instance (normally this) */);

    /// Execute the frame grabber main loop.
    void wfsThreadExec();

    sem_t m_wfsSemaphore; ///< Semaphore used to signal the WFS thread to start WFSing.

    unsigned m_wfsSemWait_sec {1}; ///< The timeout for the WFS semaphore, seconds component.

    unsigned m_wfsSemWait_nsec {0}; ///< The timeoutfor the WFS semaphore, nanoseconds component.

    int m_measuring {0}; ///< Status of measuring: 0 no, 1 single in progress, 2 continuous in progress.

    bool m_single {false}; ///< True a single measurement is in progress.

    bool m_continuous {false}; ///< True if continuous measurements are in progress.

    bool m_stopMeasurement {false}; ///< Used to request that the measurement in progress stop.

    ///@}

    sem_t m_imageSemaphore; ///< Semaphore used to signal that an image is ready
    
    unsigned m_imageSemWait_sec {1}; ///< The timeout for the image semaphore, seconds component.

    unsigned m_imageSemWait_nsec {0}; ///< The timeout for the image semaphore, nanoseconds component.

    /// Run the sensor steps 
    /** Coordinates the actions of poking and collecting images.
      * Upon completion this calls runSensor.  If \p firstRun == true, then this takes a dark.
      *
      * \returns 0 on success
      * \returns \< 0 on an error
      */
    int runSensor(bool firstRun /**< [in] flag indicating this is the first call.  triggers taking a dark if true.*/);

    /// Analyze the images
    /** Calls fitPupil and fitPokes and updates INDI.
      *
      * \returns 0 on success
      * \returns \< 0 on an error
      */
    int analyzeSensor();

    /// Fit the pupil parameters
    /** 
      * \returns 0 on success
      * \returns \< 0 on an error
      */
    int fitPupil();

    /// Fit the poke parameters
    /** 
      * \returns 0 on success
      * \returns \< 0 on an error
      */
    int fitPokes();

    /** \name INDI Interface 
      * @{ 
      */
protected:

    pcf::IndiProperty m_indiP_poke_amp;
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_poke_amp);

    pcf::IndiProperty m_indiP_nPupilImages;
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_nPupilImages);

    pcf::IndiProperty m_indiP_nPokeImages;
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_nPokeImages);

    pcf::IndiProperty m_indiP_wfsFps; ///< Property to get the FPS from the WFS camera
    INDI_SETCALLBACK_DECL(dmPokeCenter, m_indiP_wfsFps);

    pcf::IndiProperty m_indiP_shutter; ///< Property to get the status from the WFS camera
    INDI_SETCALLBACK_DECL(dmPokeCenter, m_indiP_shutter);

    pcf::IndiProperty m_indiP_single; ///< Property to start a single measurement
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_single);

    pcf::IndiProperty m_indiP_continuous; ///< Property to start continuous measurement
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_continuous);

    pcf::IndiProperty m_indiP_stop; ///< Property to request that measurement stop
    INDI_NEWCALLBACK_DECL(dmPokeCenter, m_indiP_stop);


    pcf::IndiProperty m_indiP_pupilPos; ///< Property to report the pupil position

    pcf::IndiProperty m_indiP_pokePos; ///< Property to report the poke positions
    std::vector<std::string> m_pokePosEls; ///< Vector of element names for easy calls to UpdateIfChanged.  One per poke, plus last two are for the average.
    std::vector<float> m_pokePositions; ///< Vector of positions for easy calls to UpdateIfChanged.  One per poke, plus last two are for the average.

    pcf::IndiProperty m_indiP_deltaPos; ///< Property to report the difference in pupil and average poke position

    ///@}

    /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_pokecenter * );

   int recordPokeCenter( bool force = false );
   ///@}
};

dmPokeCenter::dmPokeCenter() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    return;
}

void dmPokeCenter::setupConfig()
{
    shmimMonitorT::setupConfig(config);
    telemeterT::setupConfig(config);

    config.add("wfscam.camDevName", "", "wfscam.camDevName", argType::Required, "wfs", "camDevName", false, "string", "INDI device name of the WFS camera.  Default is wfscam.shmimName.");
    config.add("wfscam.loopSemWait", "", "wfscam.loopSemWait", argType::Required, "wfs", "loopSemWait", false, "float", "The semaphore wait time for the wfs loop start signal");
    config.add("wfscam.imageSemWait", "", "wfscam.imageSemWait", argType::Required, "wfs", "imageSemWait", false, "float", "The semaphore wait time for the image availability signal");

    config.add("pokecen.dmChannel", "", "pokecen.dmChannel", argType::Required, "pokecen", "dmChannel", false, "string", "The dm channel to use for pokes, e.g. dm01disp06.");
    config.add("pokecen.pokeX", "", "pokecen.pokeX", argType::Required, "pokecen", "pokeX", false, "vector<int>", "The x-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeY", "", "pokecen.pokeY", argType::Required, "pokecen", "pokeY", false, "vector<int>", "The y-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeAmp", "", "pokecen.pokeAmp", argType::Required, "pokecen", "pokeAmp", false, "float", "The poke amplitude, in DM command units. Default is 0.");
    config.add("pokecen.dmSleep", "", "pokecen.dmSleep", argType::Required, "pokecen", "dmSleep", false, "float", "The time to sleep for the DM command to be applied, in microseconds. Default is 10000.");
    config.add("pokecen.nPokeImages", "", "pokecen.nPokeImages", argType::Required, "pokecen", "nPokeImages", false, "int", "The number of poke images to average.  Default 5.");
    config.add("pokecen.nPupilImages", "", "pokecen.nPupilImages", argType::Required, "pokecen", "nPupilImages", false, "int", "The number of pupil images to average. Default 20.");
    config.add("pokecen.pupilPixels", "", "pokecen.pupilPixels", argType::Required, "pokecen", "pupilPixels", false, "int", "The number of pixels in the pupil. Default is 68600.");
    config.add("pokecen.pupilCutBuff", "", "pokecen.pupilCutBuff", argType::Required, "pokecen", "pupilCutBuff", false, "int", "The buffer around the initial found-pupil to include in the cut image.  />= 0, default 20.");
    config.add("pokecen.pupilMag", "", "pokecen.pupilMag", argType::Required, "pokecen", "pupilMag", false, "float", "The magnification to apply to the pupil image. >= 1, default 10.");
    config.add("pokecen.pupilMedThresh", "", "pokecen.pupilMedThresh", argType::Required, "pokecen", "pupilMedThresh", false, "float", "Threshold in the magnified image as a fraction of the median.  >0, <=1, default 0.9.");
    config.add("pokecen.pokeBlockW", "", "pokecen.pokeBlockW", argType::Required, "pokecen", "pokeBlockW", false, "int", "The size of the sub-image for the poke analysis");
    config.add("pokecen.pokeFWHMGuess", "", "pokecen.pokeFWHMGuess", argType::Required, "pokecen", "pokeFWHMGuess", false, "int", "The initial guess for the FWHM of the Gaussian fit to the poke.");
}

int dmPokeCenter::loadConfigImpl( mx::app::appConfigurator & _config )
{
    shmimMonitorT::loadConfig(_config);
    telemeterT::loadConfig(_config);

    m_wfsCamDevName = shmimMonitorT::m_shmimName;
    _config(m_wfsCamDevName, "wfscam.camDevName");

    //configure the semaphore waits
    _config(m_wfsSemWait, "wfscam.loopSemWait");

    m_wfsSemWait_sec = floor(m_wfsSemWait);
    m_wfsSemWait_nsec = (m_wfsSemWait - m_wfsSemWait_sec) * 1e9;

    _config(m_imageSemWait, "wfscam.imageSemWait");

    m_imageSemWait_sec = floor(m_imageSemWait);
    m_imageSemWait_nsec = (m_imageSemWait - m_imageSemWait_sec) * 1e9;

    _config(m_dmChan, "pokecen.dmChannel");

    _config(m_poke_x, "pokecen.pokeX");

    _config(m_poke_y, "pokecen.pokeY");

    if(m_poke_x.size() == 0 || (m_poke_x.size() != m_poke_y.size()))
    {
        return log<software_error,-1>({__FILE__, __LINE__, "invalid poke specification"});
    }

    _config(m_poke_amp, "pokecen.pokeAmp");

    _config(m_dmSleep, "pokecen.dmSleep");

    _config(m_nPokeImages, "pokecen.nPokeImages");
    _config(m_nPupilImages, "pokecen.nPupilImages");
    _config(m_pupilPixels, "pokecen.pupilPixels");
    _config(m_pupilCutBuff, "pokecen.pupilCutBuff");
    _config(m_pupilMag, "pokecen.pupilMag");
    _config(m_pupilMedThresh, "pokecen.pupilMedThresh");
    _config(m_pokeBlockW, "pokecen.pokeBlockW");
    _config(m_pokeFWHMGuess, "pokecen.pokeFWHMGuess");

    return 0;
}

void dmPokeCenter::loadConfig()
{
    if( loadConfigImpl(config)< 0)
    {
        m_shutdown = true;
    }
}

int dmPokeCenter::appStartup()
{
    if( shmimMonitorT::appStartup() < 0)
    {
        return log<software_error, -1>({__FILE__,__LINE__});
    }
    
    if(telemeterT::appStartup() < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__});
    }

    CREATE_REG_INDI_NEW_NUMBERF(m_indiP_poke_amp, "poke_amp", -1, 1, 1e-1, "%0.01f", "", "");
    m_indiP_poke_amp["current"].setValue(m_poke_amp);
    m_indiP_poke_amp["target"].setValue(m_poke_amp);

    CREATE_REG_INDI_NEW_NUMBERI(m_indiP_nPupilImages, "nPupilImages", 1, 1000, 1, "%d", "", "");
    m_indiP_nPupilImages["current"].setValue(m_nPupilImages);
    m_indiP_nPupilImages["target"].setValue(m_nPupilImages);

    CREATE_REG_INDI_NEW_NUMBERI(m_indiP_nPokeImages, "nPokeImages", 1, 1000, 1, "%d", "", "");
    m_indiP_nPokeImages["current"].setValue(m_nPokeImages);
    m_indiP_nPokeImages["target"].setValue(m_nPokeImages);

    REG_INDI_SETPROP(m_indiP_wfsFps, m_wfsCamDevName, std::string("fps"));
    
    REG_INDI_SETPROP(m_indiP_shutter, m_wfsCamDevName, std::string("shutter"));

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_single, "single");

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_continuous, "continuous");

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_stop, "stop");

    registerIndiPropertyReadOnly( m_indiP_pupilPos, "pupil_position", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
    m_indiP_pupilPos.add({"x", 0.0});
    m_indiP_pupilPos.add({"y", 0.0});
        
    registerIndiPropertyReadOnly( m_indiP_pokePos, "poke_position", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
    m_indiP_pokePos.add({"avg_x", 0.0});
    m_indiP_pokePos.add({"avg_y", 0.0});
    for(size_t n = 0; n < m_poke_x.size(); ++n)
    {
        std::string pstr = "poke" + std::to_string(n) + "_";
        m_indiP_pokePos.add({pstr + "x", 0.0});
        m_indiP_pokePos.add({pstr + "y", 0.0});
        m_pokePosEls.push_back(pstr + "x");
        m_pokePosEls.push_back(pstr + "y");
    }

    m_pokePosEls.push_back("avg_x"); //keep vector of element names for UpdateIfChanged
    m_pokePosEls.push_back("avg_y");

    m_pokePositions.resize(m_pokePosEls.size());

    registerIndiPropertyReadOnly( m_indiP_deltaPos, "delta_position", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
    m_indiP_deltaPos.add({"x", 0.0});
    m_indiP_deltaPos.add({"y", 0.0});

    if(sem_init(&m_wfsSemaphore, 0,0) < 0) 
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno,0, "Initializing wfs semaphore"});
    }

    if(sem_init(&m_imageSemaphore, 0,0) < 0) 
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno,0, "Initializing image semaphore"});
    }

    if(threadStart( m_wfsThread, m_wfsThreadInit, m_wfsThreadID, m_wfsThreadProp, m_wfsThreadPrio, m_wfsCpuset, "wfs", this, wfsThreadStart)  < 0)
    {
        return log<software_critical,-1>({__FILE__, __LINE__});
    }

    state(stateCodes::READY);

    return 0;
}

int dmPokeCenter::appLogic()
{
    if( shmimMonitorT::appLogic() < 0)
    {
        return log<software_error, -1>({__FILE__,__LINE__});
    }

    if( telemeterT::appLogic() < 0)
    {
        return log<software_error,-1>({__FILE__,__LINE__});
    }

    //first do a join check to see if other threads have exited.
    //these will throw if the threads are really gone
    try
    {
        if(pthread_tryjoin_np(m_wfsThread.native_handle(),0) == 0)
        {
            log<software_error>({__FILE__, __LINE__, "WFS thread has exited"});
            return -1;
        }
    }
    catch(...)
    {
        log<software_error>({__FILE__, __LINE__, "WFS thread has exited"});
        return -1;
    }

    if(m_measuring > 0)
    {
        if(m_continuous)
        {
            updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::On, INDI_OK);
        }
        else
        {
            updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        }

        if(m_single)
        {
            updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::On, INDI_OK);
        }
        else
        {
            updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        }
    }
    else
    {
        updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
    }

    updateIfChanged( m_indiP_nPupilImages, "current", m_nPupilImages);
    updateIfChanged( m_indiP_nPokeImages, "current", m_nPokeImages);
    updateIfChanged( m_indiP_poke_amp, "current", m_poke_amp);

    return 0;
}

int dmPokeCenter::appShutdown()
{
    shmimMonitorT::appShutdown();
    telemeterT::appShutdown();

    try
    {
        if(m_wfsThread.joinable())
        {
            m_wfsThread.join();
        }
    }
    catch(...){}

    return 0;
}

int dmPokeCenter::allocate( const wfsShmimT & dummy)
{
    static_cast<void>(dummy); //be unused
  
    //This is a call to the pokeSensor::allocate, unless we can have dev::pokeSensor : public shmimMonitor<pokeSensor>
    std::unique_lock<std::mutex> lock(m_wfsImageMutex);

    m_rawImage.create( m_configName + "_raw", shmimMonitorT::m_width, shmimMonitorT::m_height);

    m_wfsDark.create( m_configName + "_dark", shmimMonitorT::m_width, shmimMonitorT::m_height);

    m_pupilImage.create(m_configName + "_pupil", shmimMonitorT::m_width, shmimMonitorT::m_height);

    wfsPixget = getPixPointer<float>(shmimMonitorT::m_dataType);

    try
    {
        m_dmStream.open(m_dmChan);    
    }
    catch(const std::exception& e) //this can check for invalid_argument and distinguish not existing
    {
        return log<software_error,-1>({__FILE__, __LINE__, std::string("exception opening DM: ") + e.what()});
    }
    
    m_dmImage.resize(m_dmStream.rows(), m_dmStream.cols());

    //end of call to pokeSensor::allocate

    m_pokeImages.resize(m_poke_x.size());
    for(size_t n = 0; n < m_pokeImages.size(); ++n)
    {
        m_pokeImages[n].create(m_configName + "_poke_" + std::to_string(n),shmimMonitorT::m_width, shmimMonitorT::m_height);
    }
    
    return 0;
}
    
int dmPokeCenter::processImage( void * curr_src,    
                                const wfsShmimT &  dummy
                              )
{
    static_cast<void>(dummy); //be unused

    float * data = m_rawImage().data();

    //Copy the data out as float no matter what type it is
    for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
    {
        data[nn] = wfsPixget(curr_src, nn);
    }

    if(sem_post(&m_imageSemaphore) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
    }

    return 0;
}

inline
void dmPokeCenter::wfsThreadStart( dmPokeCenter * d)
{
   d->wfsThreadExec();
}


inline
void dmPokeCenter::wfsThreadExec()
{
    m_wfsThreadID = syscall(SYS_gettid);

    //Wait fpr the thread starter to finish initializing this thread.
    while(m_wfsThreadInit == true && m_shutdown == 0)
    {
        sleep(1);
    }

    while(m_shutdown == 0)
    {
        timespec ts;
        XWC_SEM_WAIT_TS_RETVOID(ts, m_wfsSemWait_sec, m_wfsSemWait_nsec);
      
        XWC_SEM_TIMEDWAIT_LOOP( m_wfsSemaphore, ts )

        //Lock a mutex here
        if(m_single)
        {
            m_measuring = 1;    
        }
        else if(m_continuous)
        {
            m_measuring = 2;
        }
        else
        {
            m_measuring = 0;
            return;
        }

        state(stateCodes::OPERATING);

        m_stopMeasurement = false;

        bool firstRun = true;

        while(!m_stopMeasurement && !m_shutdown)
        {
            if( runSensor(firstRun) < 0)
            {
                log<software_error>({__FILE__, __LINE__, "runSensor returned error"});
                break;
            }
            
            firstRun = false;

            if(m_measuring == 1)
            {
                break;
            }
        }

        m_measuring = 0;
        m_single = 0;
        m_continuous = 0;

        state(stateCodes::READY);

        
    } //outer loop, will exit if m_shutdown==true

    return;

}

inline
int dmPokeCenter::runSensor(bool firstRun)
{

    mx::fits::fitsFile<float> tmpFF;

    timespec ts;
    
    //Wait two seconds for it to shut 
    ///\todo this should be configurable and based on fps
    unsigned n = 0;
    while(!m_wfsDark.valid() && n < 200)
    {
        mx::sys::milliSleep(10);
        ++n;
    }

    if(!m_wfsDark.valid())
    {
        return log<software_error, -1>({__FILE__,__LINE__, "not allocated"});
    }

    if(firstRun)
    {
        //Shut the shutter
        if( sendNewStandardIndiToggle(m_wfsCamDevName, "shutter", true) < 0)
        {
            return log<software_error,-1>({__FILE__,__LINE__});
        }

        //Wait two seconds for it to shut 
        ///\todo this should be configurable
        n = 0;
        while(m_shutter != 1 && n < 200)
        {
            mx::sys::milliSleep(10);
            ++n;
        }

        if(m_shutter != 1)
        {
            return log<software_error,-1>({__FILE__,__LINE__, "shutter did not shut"});
        }

        m_wfsDark().setZero();
        n = 0;

        //flush semaphore so we take the _next_ good image
        XWC_SEM_FLUSH(m_imageSemaphore);

        while(n < m_nDarks && !m_stopMeasurement && !m_shutdown)
        {
            XWC_SEM_WAIT_TS(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
            XWC_SEM_TIMEDWAIT_LOOP( m_imageSemaphore, ts )

            //If here we got an image
            m_wfsDark() += m_rawImage();
            ++n;
        }

        m_wfsDark() /= m_nDarks;

    }


    //Open the shutter
    if( sendNewStandardIndiToggle(m_wfsCamDevName, "shutter", false) < 0)
    {
        return log<software_error>({__FILE__,__LINE__});
    }

    //Wait two seconds for it to open
    ///\todo this should be configurable
    n = 0;
    while(m_shutter != 0 && n < 200)
    {
        mx::sys::milliSleep(10);
        ++n;
    }

    if(m_shutter != 0)
    {
        return log<software_error,-1>({__FILE__,__LINE__, "shutter did not open"});
    }

    //** Now we record the pupil image **//
    m_pupilImage.setWrite();
    m_pupilImage().setZero();
    n = 0;
        
    //flush semaphore so we take the _next_ good image
    XWC_SEM_FLUSH(m_imageSemaphore);

    while(n < m_nPupilImages && !m_stopMeasurement && !m_shutdown)
    {
        XWC_SEM_WAIT_TS(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
        XWC_SEM_TIMEDWAIT_LOOP( m_imageSemaphore, ts )

        //If here we got an image.  m_rawImage will have been updated
        m_pupilImage() += m_rawImage();
        ++n;
    }

    m_pupilImage() = m_pupilImage()/m_nPupilImages - m_wfsDark();
    m_pupilImage.post();


    tmpFF.write("/tmp/pupilImage.fits", m_pupilImage());

    for(size_t nn = 0; nn < m_pokeImages.size(); ++nn)
    {
        m_dmImage.setZero();

        m_pokeImages[nn].setWrite();
        m_pokeImages[nn]().setZero();
        n = 0;
        
        m_dmImage( m_poke_x[nn], m_poke_y[nn]) = m_poke_amp;
        
        m_dmStream = m_dmImage;

        mx::sys::microSleep(m_dmSleep);

        //flush semaphore so we take the _next_ good image
        XWC_SEM_FLUSH(m_imageSemaphore);

        while(n < m_nPokeImages && !m_stopMeasurement && !m_shutdown)
        {    
            /* POSITIVE POKE */

            //** Now we record the poke image **//
            XWC_SEM_WAIT_TS(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
            XWC_SEM_TIMEDWAIT_LOOP( m_imageSemaphore, ts )

            //If here, we got an image.  m_rawImage will have been updated
            m_pokeImages[nn]() += m_rawImage();

            ++n;
        }

        m_pokeImages[nn]() = m_pokeImages[nn]()/m_nPokeImages - m_wfsDark() - m_pupilImage();
        m_pokeImages[nn].post();

        if(nn == 0)
        {
            tmpFF.write("/tmp/poke0.fits", m_pokeImages[nn]());
        }
    }

    m_dmImage.setZero();
    m_dmStream = m_dmImage;

    return analyzeSensor();
}

inline
int dmPokeCenter::analyzeSensor()
{
    if(fitPupil() < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__, "error from fitPupil"});
    }

    if(fitPokes() < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__, "error from fitPupil"});
    }

    updateIfChanged(m_indiP_deltaPos, std::vector<std::string>({"x", "y"}), std::vector<float>({m_pupilX - m_pokeX, m_pupilY - m_pokeY}));

    recordPokeCenter();

    return 0;
}

inline
int dmPokeCenter::fitPupil()
{
    float threshPerc = (1.0*m_pupilPixels)/(m_pupilImage().rows() * m_pupilImage().cols());

    //Threshold to find the initial pupil mask geometrically
    size_t pos = (1-threshPerc)*m_pupilImage().rows()*m_pupilImage().cols();
    m_pupilCopy = m_pupilImage();
    std::nth_element(m_pupilCopy.data(), m_pupilCopy.data()+pos, m_pupilCopy.data()+m_pupilCopy.rows()*m_pupilCopy.cols());

    float pupilThresh = m_pupilCopy.data()[pos];

    float x0, y0, avgr0, avgx, avgy, avgr;

    
    m_fullMask.resize(m_pupilImage().rows(), m_pupilImage().cols());

    for(int cc=0; cc < m_fullMask.cols(); ++cc)
    {
        for(int rr=0; rr < m_fullMask.rows(); ++rr)
        {
            if(m_pupilImage()(rr,cc) < pupilThresh)
            {
                m_fullMask(rr,cc) = 0;
            }
            else
            {
                m_fullMask(rr,cc) = 1;
            }
        }
    }

    //Now find the outer edge
    if(circleOuterpix( x0, y0, avgr0, avgx, avgy, avgr, m_fullEdge, m_fullMask) < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__, "circle fit failed"});
    }

    //And cut out the pupil plust buffer
    float cutx = avgx - avgr-m_pupilCutBuff;
    float cuty = avgy - avgr-m_pupilCutBuff;
    float cutw =  2*avgr+2*m_pupilCutBuff;

    if(cutx < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__, "pupilCutBuff is too big for pupil position"});
    }

    if(cuty < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__, "pupilCutBuff is too big for pupil position"});
    }

    if(cutx + cutw > m_pupilImage.rows())
    {
        return log<software_error, -1>({__FILE__, __LINE__, "pupilCutBuff is too big for pupil position"});
    }

    if(cuty + cutw > m_pupilImage.rows())
    {
        return log<software_error, -1>({__FILE__, __LINE__, "pupilCutBuff is too big for pupil position"});
    }

    m_pupilCut = m_pupilImage().block( cutx, cuty, cutw, cutw);
    
    m_pupilMagnified.resize(m_pupilCut.rows()*m_pupilMag, m_pupilCut.cols()*m_pupilMag);

    imageMagnify(m_pupilMagnified, m_pupilCut, mx::improc::bilinearTransform<float>());

    float med = imageMedian(m_pupilMagnified); /// \todo use work version
    
    float dthresh = med * m_pupilMedThresh;

    m_magMask.resize(m_pupilMagnified.rows(), m_pupilMagnified.cols()); //This is a different mask-- maskMag

    for(int cc=0; cc < m_pupilMagnified.cols(); ++cc)
    {
        for(int rr=0; rr < m_pupilMagnified.rows(); ++rr)
        {
            if(m_pupilMagnified(rr,cc) < dthresh)
            {
                m_magMask(rr,cc) = 0;
            }
            else
            {
                m_magMask(rr,cc) = 1;
            }
        }
    }

    if(circleOuterpix( x0, y0, avgr0, avgx, avgy, avgr, m_magEdge, m_magMask) < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__, "circle fit failed"});
    }

    x0 = cutx + x0/m_pupilMag;
    y0 = cuty + y0/m_pupilMag;
    avgr0 /= m_pupilMag;

    avgx = cutx + avgx/m_pupilMag;
    avgy = cuty + avgy/m_pupilMag;
    avgr /= m_pupilMag;

    m_pupilX = avgx;
    m_pupilY = avgy;

    updateIfChanged(m_indiP_pupilPos, std::vector<std::string>({"x", "y"}), std::vector<float>({m_pupilX, m_pupilY}));
    
    return 0;

}

inline
int dmPokeCenter::fitPokes()
{

    m_pokeX = 0;
    m_pokeY = 0;
    for(size_t nn = 0; nn < m_pokeImages.size(); ++nn)
    {
        int xmx, ymx;

        float mx = m_pokeImages[nn]().maxCoeff(&xmx, &ymx);

        int x0 = xmx - m_pokeBlockW/2;
        int y0 = ymx - m_pokeBlockW/2;

        m_pokeBlock = m_pokeImages[nn]().block(x0, y0, m_pokeBlockW, m_pokeBlockW);

        m_gfit.setArray(m_pokeBlock .data(), m_pokeBlock.rows(), m_pokeBlock.cols());
        m_gfit.setGuess(0, mx, 0.5*(m_pokeBlock.rows()-1.0), 0.5*(m_pokeBlock.cols()-1.0), mx::math::func::sigma2fwhm(m_pokeFWHMGuess));
        m_gfit.fit();

        int rc = m_gfit.get_reason_code();
        if(rc != 1 && rc != 2)
        {
            return log<software_error, -1>({__FILE__,__LINE__, "fit to poke " + std::to_string(nn) + " failed: " + m_gfit.get_reason_string()});
        }

        m_pokePositions[nn*2 + 0] = x0 + m_gfit.x0();
        m_pokePositions[nn*2 + 1] = y0 + m_gfit.y0();

        m_pokeX += x0 + m_gfit.x0();
        m_pokeY += y0 + m_gfit.y0();
    }

    m_pokeX /= m_pokeImages.size();
    m_pokeY /= m_pokeImages.size();

    m_pokePositions[m_pokeImages.size()*2 + 0] = m_pokeX;
    m_pokePositions[m_pokeImages.size()*2 + 1] = m_pokeY;

    updateIfChanged(m_indiP_pokePos, m_pokePosEls, m_pokePositions);


    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_nPupilImages )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_nPupilImages, ipRecv)
   
    float target;

    if( indiTargetUpdate(m_indiP_nPupilImages, target, ipRecv, false) < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__});
    }

    m_nPupilImages = target;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_nPokeImages )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_nPokeImages, ipRecv)
   
    float target;

    if( indiTargetUpdate(m_indiP_nPokeImages, target, ipRecv, false) < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__});
    }

    m_nPokeImages = target;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_poke_amp )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_poke_amp, ipRecv)
   
    float target;

    if( indiTargetUpdate(m_indiP_poke_amp, target, ipRecv, false) < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__});
    }

    m_poke_amp = target;

    return 0;
}

INDI_SETCALLBACK_DEFN( dmPokeCenter, m_indiP_wfsFps )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_wfsFps, ipRecv)
   
    if( ipRecv.find("current") != true ) //this isn't valid
    {
        return 0;
    }
   
    m_wfsFps = ipRecv["current"].get<float>();

    return 0;
}

INDI_SETCALLBACK_DEFN( dmPokeCenter, m_indiP_shutter )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_shutter, ipRecv)
   
    if( ipRecv.find("toggle") != true ) //this isn't valid
    {
        return -1;
    }
   
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_shutter = false; //open
    }
    else
    {
        m_shutter = true; //shut
    }
   
    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_single )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_single, ipRecv)
   
    if( ipRecv.find("toggle") != true ) //this isn't valid
    {
        return -1;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if(m_measuring == 0)
        {
            m_continuous = 0;
            m_single = 1;
            if(sem_post(&m_wfsSemaphore) < 0)
            {
                return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            }
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_continuous )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_continuous, ipRecv)
   
    if( ipRecv.find("toggle") != true ) //this isn't valid
    {
        return -1;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        if(m_measuring == 0)
        {
            m_continuous = 1;
            m_single = 0;
            if(sem_post(&m_wfsSemaphore) < 0)
            {
                return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            }
        }
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        if(m_measuring != 0)
        {
            m_stopMeasurement = true;
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmPokeCenter, m_indiP_stop )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_stop, ipRecv)
   
    if( ipRecv.find("request") != true ) //this isn't valid
    {
        return -1;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        if(m_measuring != 0)
        {
            m_stopMeasurement = true;
        }
    }

    return 0;
}

inline
int dmPokeCenter::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_pokecenter());
}
   
inline
int dmPokeCenter::recordTelem( const telem_pokecenter * )
{
   return recordPokeCenter(true);
}

inline
int dmPokeCenter::recordPokeCenter( bool force )
{
    static int measuring = -1;
    static float pupilX = 0;
    static float pupilY = 0;
    static std::vector<float> pokePositions;

    if(pokePositions.size() != m_pokePositions.size())
    {
        pokePositions.resize(m_pokePositions.size(), 0);
    }

    bool changed = false;
    if(!force)
    {
        if(m_measuring != measuring) changed = true;
        else if(m_pupilX != pupilX) changed = true;
        else if(m_pupilY != pupilY) changed = true;
        else
        {
            for(size_t n = 0; n < m_pokePositions.size(); ++n)
            {
                if(m_pokePositions[n] != pokePositions[n])
                {
                    changed = true;
                    break;
                }
            }
        }
    }   

    if(changed || force)
    {
        uint8_t meas = m_measuring;
        telem<telem_pokecenter>({meas, m_pupilX, m_pupilY, m_pokePositions});

        measuring = m_measuring;
        pupilX = m_pupilX;
        pupilY = m_pupilY;
        pokePositions.assign(m_pokePositions.begin(), m_pokePositions.end());
    }

    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //dmPokeCenter_hpp
