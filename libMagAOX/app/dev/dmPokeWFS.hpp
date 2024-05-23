/** \file dmPokeWFS.hpp
  * \brief The MagAO-X DM Poke Centering header file
  *
  * \ingroup dmPokeWFS_files
  */

#ifndef dmPokeWFS_hpp
#define dmPokeWFS_hpp

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/improc/eigenCube.hpp>
using namespace mx::improc;

#include "../../ImageStreamIO/pixaccess.hpp"

/** \defgroup dmPokeWFS
  * \brief The MagAO-X device to coordinate poking a deformable mirror's actuators and synchronize reads of a camera image.
  *
  * <a href="../handbook/operating/software/apps/dmPokeWFS.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */


/** \defgroup dmPokeWFS_files
  * \ingroup dmPokeWFS
  */

namespace MagAOX
{
namespace app
{
namespace dev
{



/// A base class to coordinate poking a deformable mirror's actuators and synchronizedreads of a camera image.
/** CRTP class `derivedT` has the following requirements:
  * 
  * - Must be derived from MagAOXApp<true>
  * 
  * - Must be derived from `dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::wfsShmimT>` (replace DERIVEDNAME with derivedT class name)
  * 
  * - Must be derived from  `dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::darkShmimT>` (replace DERIVEDNAME with derivedT class name)
  * 
  * - Must contain the following friend declalibMagAOX/app/indiMacros.hpprations (replace DERIVEDNAME with derivedT class name):
  *   \code
  *      friend class dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::wfsShmimT>;
  *      friend class dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::darkShmimT>;
  *      friend class dev::dmPokeWFS<DERIVEDNAME>
  *   \endcode
  * 
  * - Must contain the following typedefs (replace DERIVEDNAME with derivedT class name):
  *   \code
  *       typedef dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::wfsShmimT> shmimMonitorT;
  *       typedef dev::shmimMonitor<DERIVEDNAME, dev::dmPokeWFS<DERIVEDNAME>::darkShmimT> darkShmimMonitorT;
  *       typedef dev::dmPokeWFS<DERIVEDNAME> dmPokeWFST;
  * 
  *   \endcode
  * - Must provide the following interfaces:
  *   \code 
  *       shmimMonitorT & shmimMonitor() 
  *       {
  *           return *static_cast<shmimMonitorT *>(this);
  *       }
  * 
  *       darkShmimMonitorT & darkShmimMonitor() 
  *       {
  *           return *static_cast<darkShmimMonitorT *>(this);
  *       }
  *   \endcode
  * 
  * - If derivedT has additional shmimMonitor parents, you will need to include these lines in the class 
  *   declaration:
  *   \code
  *       using dmPokeWFST::allocate;
  *       using dmPokeWFST::processImage;
  *   \endcode
  *
  * - Must provide the following interface:
  *   \code
  *       // Run the sensor steps 
  *       // Coordinates the actions of poking and collecting images.
  *       // Upon completion this calls runSensor.  If \p firstRun == true, one time
  *       // actions such as taking a dark can be executed.
  *       // 
  *       // returns 0 on success
  *       // returns \< 0 on an error
  *       int runSensor(bool firstRun ///< [in] flag indicating this is the first call.  triggers taking a dark if true.
  *                     );
  *   \endcode
  * 
  * - Must provide the following interface:
  *   \code 
  *       // Analyze the poke image
  *       // This analyzes the resulting poke images. 
  *       //
  *       // returns 0 on success
  *       // returns \< 0 on an error
  *       int analyzeSensor();
  *   \endcode
  *   At the conclusion of analyzeSensor the measured signal (e.g. deltaX and deltaY) should be updated and set in m_indiP_measurement.
  *   The function \ref updateMeasurement() can be used for this.  However, the updating of the loop counter and the subsequent INDI 
  *   property update is handled automatically after that.
  * 
  * - Must be a telemeter with the following interface:
  * 
  *     - Must be derived from `dev::telemeter<DERIVEDNAME>` (replace DERIVEDNAME with derivedT class name) and meet the requirements
  *       of `dev::telemeter`
  * 
  *     - In the function `derivedT::checkRecordTimes()` required by `dev::telemeter`,  the `telem_pokeloop` type must be checked.
  *       The minimum `derivedT::checkRecordTimes()` is:
  *       \code
  *         int checkRecordTimes()
  *         {
  *             return telemeterT::checkRecordTimes(telem_pokeloop());
  *         }
  *       \endcode
  *
  * - Must call this base class's setupConfig(), loadConfig(), appStartup(), appStartup(), and appShutdown() in the 
  *    appropriate functions.  For convenience the following macros are defined to provide error checking:
  *    \code  
  *       DMPOKEWFS_SETUP_CONFIG( cfig )
  *       DMPOKEWFS_LOAD_CONFIG( cfig )
  *       DMPOKEWFS_APP_STARTUP
  *       DMPOKEWFS_APP_LOGIC
  *       DMPOKEWFS_APP_SHUTDOWN
  *    \endcode
  * 
  * \ingroup appdev
  */
template<class derivedT>
class dmPokeWFS 
{

public:

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

    struct darkShmimT
    {
        static std::string configSection()
        {
            return "wfsdark";
        };

        static std::string indiPrefix()
        {
            return "wfsdark";
        };
    };

protected:

    /** \name Configurable Parameters
      *@{
      */
   
    std::string m_wfsCamDevName; ///<INDI device name of the WFS camera.  Default is wfscam.shmimName.

    double m_wfsSemWait {1.5}; ///< The time in sec to wait on the WFS semaphore.  Default 0.5 sec.

    double m_imageSemWait {0.5}; ///< The time in sec to wait on the image semaphore.  Default 0.5 sec.

    unsigned m_nPokeImages {5}; ///< The number of images to average for the poke images.  Default is 5.

    unsigned m_nPokeAverage {10}; ///< The number of poke sequences to average.  Default is 10.

    std::string m_dmChan;

    std::vector<int> m_poke_x;
    std::vector<int> m_poke_y;

    float m_poke_amp {0.0};

    float m_dmSleep {10000}; ///<The time to sleep for the DM command to be applied, in microseconds. Default is 10000.

    ///@}

    std::mutex m_wfsImageMutex;

    mx::improc::milkImage<float> m_rawImage;
    
    mx::improc::milkImage<float> m_pokeImage;
    mx::improc::eigenImage<float> m_pokeLocal;

    float (*wfsPixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as float

    float m_wfsFps {-1}; ///< The WFS camera FPS

    mx::improc::eigenImage<float> m_darkImage; ///< The dark image

    bool m_darkValid {false}; ///< Flag indicating if dark is valid based on its size.

    float (*darkPixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the dark image data as float

    mx::improc::milkImage<float> m_dmStream;

    mx::improc::eigenImage<float> m_dmImage;

    float m_deltaX {0};
    float m_deltaY {0};
    uint64_t m_counter {0};

public:

    /**\name MagAOXApp Interface
      *
      * @{ 
      */

    /// Setup the configuration system
    /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       dmPokeWFS<derivedT,realT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
    int setupConfig( mx::app::appConfigurator & config /**< [in] an application configuration to load values to*/);

    /// load the configuration system results
    /**
      * This should be called in `derivedT::loadConfig` as
      * \code
        dmPokeWFS<derivedT,realT>::loadConfig(config);
        \endcode
      * with appropriate error checking.
      */
    int loadConfig( mx::app::appConfigurator & config /**< [in] an application configuration from which to load values */);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       dmPokeWFS<derivedT,realT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
    int appStartup();

   /// dmPokeWFS application logic
   /** This should be called in `derivedT::appLogic` as
     * \code
       dmPokeWFS<derivedT,realT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
    int appLogic();

   /// dmPokeWFS shutdown
   /** This should be called in `derivedT::appShutdown` as
     * \code
       dmPokeWFS<derivedT,realT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
    int appShutdown();

    ///@}

    /** \name shmimMonitor Interface
      * @{
      */

    int allocate( const wfsShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/);
    
    int processImage( void * curr_src,   ///< [in] pointer to the start of the current frame 
                      const wfsShmimT &  ///< [in] tag to differentiate shmimMonitor parents. 
                    );
    ///@}


    /** \name darkShmimMonitor Interface
      * @{
      */

    int allocate( const darkShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/);
    
    int processImage( void * curr_src,   ///< [in] pointer to the start of the current frame 
                      const darkShmimT &  ///< [in] tag to differentiate shmimMonitor parents. 
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
    static void wfsThreadStart( dmPokeWFS * s /**< [in] a pointer to an streamWriter instance (normally this) */);

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


    /// Apply a single DM poke pattern and record the results
    /** This accumulates m_nPokeImages*m_nPokeAverage images in m_pokeLocal, so m_pokeLocal
      * should be zeroed before the first call to this (e.g. for a +1 poke), 
      * but not zeroed before the second call (e.g. for the -1 poke). You also need
      * to 0 the DM after finishing a poke pair.
      * See basicRunSensor() for how to use.
      * 
      * \returns +1 if exit is due to shutdown or stop request
      * \returns 0 if no error
      * \returns -1 if an error occurs
      */
    int basicTimedPoke(float pokeSign /**< [in] the sign, and possibly a scaling, to apply to m_pokeAmplitude*/);

    /// Run the basic +/- poke sensor steps 
    /** Coordinates the actions of poking and collecting images.
      * 
      * This can be called from the derived class runSensor.
      * 
      * \returns +1 if exit is due to shutdown or stop request
      * \returns 0 if no error
      * \returns -1 if an error occurs
      */
    int basicRunSensor();

    int updateMeasurement( float deltaX,
                           float deltaY
                         );

    /** \name INDI Interface 
      * @{ 
      */
protected:

    pcf::IndiProperty m_indiP_poke_amp;
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_poke_amp);

    pcf::IndiProperty m_indiP_nPokeImages;
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_nPokeImages);

    pcf::IndiProperty m_indiP_nPokeAverage;
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_nPokeAverage);

    pcf::IndiProperty m_indiP_wfsFps; ///< Property to get the FPS from the WFS camera
    INDI_SETCALLBACK_DECL(derivedT, m_indiP_wfsFps);

    pcf::IndiProperty m_indiP_single; ///< Switch to start a single measurement
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_single);

    pcf::IndiProperty m_indiP_continuous; ///< Switch to start continuous measurement
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_continuous);

    pcf::IndiProperty m_indiP_stop; ///< Switch to request that measurement stop
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_stop);

    pcf::IndiProperty m_indiP_measurement; ///< Property to report the delta measurement, including the loop counter.


    ///@}

    /** \name Telemeter Interface 
      * @{ 
      */

    int recordTelem(const telem_pokeloop *);

    int recordPokeLoop(bool force = false);

    ///@}

private:
    derivedT & derived()
    {
        return *static_cast<derivedT *>(this);
    }

};

template<class derivedT>
int dmPokeWFS<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
    if(derived().shmimMonitor().setupConfig(config) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "shmimMonitorT::setupConfig"});
        return -1;
    }

    config.add("wfscam.camDevName", "", "wfscam.camDevName", argType::Required, "wfscam", "camDevName", false, "string", "INDI device name of the WFS camera.  Default is wfscam.shmimName.");
    config.add("wfscam.loopSemWait", "", "wfscam.loopSemWait", argType::Required, "wfscam", "loopSemWait", false, "float", "The semaphore wait time for the wfs loop start signal");
    config.add("wfscam.imageSemWait", "", "wfscam.imageSemWait", argType::Required, "wfscam", "imageSemWait", false, "float", "The semaphore wait time for the image availability signal");

    if(derived().darkShmimMonitor().setupConfig(config) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "darkShmimMonitorT::setupConfig"});
        return -1;
    }

    config.add("pokecen.dmChannel", "", "pokecen.dmChannel", argType::Required, "pokecen", "dmChannel", false, "string", "The dm channel to use for pokes, e.g. dm01disp06.");
    config.add("pokecen.pokeX", "", "pokecen.pokeX", argType::Required, "pokecen", "pokeX", false, "vector<int>", "The x-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeY", "", "pokecen.pokeY", argType::Required, "pokecen", "pokeY", false, "vector<int>", "The y-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeAmp", "", "pokecen.pokeAmp", argType::Required, "pokecen", "pokeAmp", false, "float", "The poke amplitude, in DM command units. Default is 0.");
    config.add("pokecen.dmSleep", "", "pokecen.dmSleep", argType::Required, "pokecen", "dmSleep", false, "float", "The time to sleep for the DM command to be applied, in microseconds. Default is 10000.");
    config.add("pokecen.nPokeImages", "", "pokecen.nPokeImages", argType::Required, "pokecen", "nPokeImages", false, "int", "The number of poke images to average.  Default 5.");
    config.add("pokecen.nPokeAverage", "", "pokecen.nPokeAverage", argType::Required, "pokecen", "nPokeAverage", false, "int", "The number of poke sequences to average.  Default 10.");


    return 0;    
}


template<class derivedT>
int dmPokeWFS<derivedT>::loadConfig( mx::app::appConfigurator & config)
{
    if(derived().shmimMonitor().loadConfig(config) < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "shmimMonitorT::loadConfig"});
    }

    m_wfsCamDevName = derived().shmimMonitor().shmimName();
    config(m_wfsCamDevName, "wfscam.camDevName");

    //configure the semaphore waits
    config(m_wfsSemWait, "wfscam.loopSemWait");

    m_wfsSemWait_sec = floor(m_wfsSemWait);
    m_wfsSemWait_nsec = (m_wfsSemWait - m_wfsSemWait_sec) * 1e9;

    config(m_imageSemWait, "wfscam.imageSemWait");

    m_imageSemWait_sec = floor(m_imageSemWait);
    m_imageSemWait_nsec = (m_imageSemWait - m_imageSemWait_sec) * 1e9;

    if(derived().darkShmimMonitor().loadConfig(config) < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "darkShmimMonitorT::loadConfig"});
    }

    config(m_dmChan, "pokecen.dmChannel");

    config(m_poke_x, "pokecen.pokeX");

    config(m_poke_y, "pokecen.pokeY");

    if(m_poke_x.size() == 0 || (m_poke_x.size() != m_poke_y.size()))
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "invalid poke specification"});
    }

    config(m_poke_amp, "pokecen.pokeAmp");

    config(m_dmSleep, "pokecen.dmSleep");

    config(m_nPokeImages, "pokecen.nPokeImages");

    config(m_nPokeAverage, "pokecen.nPokeAverage");

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::appStartup()
{
    if( derived().shmimMonitor().appStartup() < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__,__LINE__});
    }

    if( derived().darkShmimMonitor().appStartup() < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__,__LINE__});
    }

    CREATE_REG_INDI_NEW_NUMBERF_DERIVED(m_indiP_poke_amp, "poke_amp", -1, 1, 1e-1, "%0.01f", "", "");
    m_indiP_poke_amp["current"].setValue(m_poke_amp);
    m_indiP_poke_amp["target"].setValue(m_poke_amp);

    CREATE_REG_INDI_NEW_NUMBERI_DERIVED(m_indiP_nPokeImages, "nPokeImages", 1, 1000, 1, "%d", "", "");
    m_indiP_nPokeImages["current"].setValue(m_nPokeImages);
    m_indiP_nPokeImages["target"].setValue(m_nPokeImages);

    CREATE_REG_INDI_NEW_NUMBERI_DERIVED(m_indiP_nPokeAverage, "nPokeAverage", 1, 1000, 1, "%d", "", "");
    m_indiP_nPokeAverage["current"].setValue(m_nPokeAverage);
    m_indiP_nPokeAverage["target"].setValue(m_nPokeAverage);

    REG_INDI_SETPROP_DERIVED(m_indiP_wfsFps, m_wfsCamDevName, std::string("fps"));
    
    CREATE_REG_INDI_NEW_TOGGLESWITCH_DERIVED( m_indiP_single, "single");

    CREATE_REG_INDI_NEW_TOGGLESWITCH_DERIVED( m_indiP_continuous, "continuous");

    CREATE_REG_INDI_NEW_REQUESTSWITCH_DERIVED( m_indiP_stop, "stop");
        
    derived().template registerIndiPropertyReadOnly( m_indiP_measurement, "measurement", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
    m_indiP_measurement.add({"delta_x", 0.0}); 
    m_indiP_measurement.add({"delta_y", 0.0});
    m_indiP_measurement.add({"counter", 0});

    if(sem_init(&m_wfsSemaphore, 0,0) < 0) 
    {
        return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno,0, "Initializing wfs semaphore"});
    }

    if(sem_init(&m_imageSemaphore, 0,0) < 0) 
    {
        return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno,0, "Initializing image semaphore"});
    }

    if(derived().template threadStart( m_wfsThread, m_wfsThreadInit, m_wfsThreadID, m_wfsThreadProp, m_wfsThreadPrio, m_wfsCpuset, "wfs", this, wfsThreadStart)  < 0)
    {
        return derivedT::template log<software_critical,-1>({__FILE__, __LINE__});
    }

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::appLogic()
{

    if( derived().shmimMonitor().appLogic() < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__,__LINE__});
    }

    if( derived().darkShmimMonitor().appLogic() < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__,__LINE__});
    }

    //first do a join check to see if other threads have exited.
    //these will throw if the threads are really gone
    try
    {
        if(pthread_tryjoin_np(m_wfsThread.native_handle(),0) == 0)
        {
            derivedT::template log<software_error>({__FILE__, __LINE__, "WFS thread has exited"});
            return -1;
        }
    }
    catch(...)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "WFS thread has exited"});
        return -1;
    }

    if(m_measuring > 0)
    {
        if(m_continuous)
        {
            derived().template updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::On, INDI_OK);
        }
        else
        {
            derived().template updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        }

        if(m_single)
        {
            derived().template updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::On, INDI_OK);
        }
        else
        {
            derived().template updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        }
    }
    else
    {
        derived().template updateSwitchIfChanged(m_indiP_continuous, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
        derived().template updateSwitchIfChanged(m_indiP_single, "toggle", pcf::IndiElement::SwitchStateType::Off, INDI_IDLE);
    }

    derived().template updateIfChanged( m_indiP_nPokeImages, "current", m_nPokeImages);
    derived().template updateIfChanged( m_indiP_nPokeAverage, "current", m_nPokeAverage);
    derived().template updateIfChanged( m_indiP_poke_amp, "current", m_poke_amp);

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::appShutdown()
{
    if(derived().shmimMonitor().appShutdown() < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "error from shmimMonitorT::appShutdown"});
    }
    
    if(derived().darkShmimMonitor().appShutdown() < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "error from darkShmimMonitorT::appShutdown"});
    }

    if (m_wfsThread.joinable())
    {
        pthread_kill(m_wfsThread.native_handle(), SIGUSR1);
        try
        {
            m_wfsThread.join(); // this will throw if it was already joined
        }
        catch (...)
        {
        }
    }

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::allocate( const wfsShmimT & dummy)
{
    static_cast<void>(dummy); //be unused
  
    std::unique_lock<std::mutex> lock(m_wfsImageMutex);

    m_rawImage.create( derived().m_configName + "_raw", derived().shmimMonitor().width(), derived().shmimMonitor().height());

    wfsPixget = getPixPointer<float>(derived().shmimMonitor().dataType());

    try
    {
        m_dmStream.open(m_dmChan);    
    }
    catch(const std::exception& e) 
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, std::string("exception opening DM: ") + e.what()});
    }
    
    m_dmImage.resize(m_dmStream.rows(), m_dmStream.cols());

    if(derived().darkShmimMonitor().width() == derived().shmimMonitor().width() && 
         derived().darkShmimMonitor().height() == derived().shmimMonitor().height() )
    {
        m_darkValid = true;
    }
    else
    {
        m_darkValid = false;
    }

    if(m_pokeImage.rows() != derived().shmimMonitor().width() || m_pokeImage.cols() != derived().shmimMonitor().height())
    {
        m_pokeImage.create(derived().m_configName + "_poke", derived().shmimMonitor().width(), derived().shmimMonitor().height());
    }

    m_pokeLocal.resize(derived().shmimMonitor().width(), derived().shmimMonitor().height());

    return 0;
}

template<class derivedT>   
int dmPokeWFS<derivedT>::processImage( void * curr_src,    
                                       const wfsShmimT &  dummy
                                     )
{
    static_cast<void>(dummy); //be unused

    std::unique_lock<std::mutex> lock(m_wfsImageMutex);

    float * data = m_rawImage().data();
    float * darkData = m_darkImage.data();

    //Copy the data out as float no matter what type it is
    uint64_t Npix = derived().shmimMonitor().width()*derived().shmimMonitor().height();

    if(m_darkValid)
    {
        for(unsigned nn=0; nn < Npix; ++nn)
        {
            data[nn] = wfsPixget(curr_src, nn) - darkData[nn];
        }
    }
    else
    {
        for(unsigned nn=0; nn < Npix; ++nn)
        {
            data[nn] = wfsPixget(curr_src, nn);
        }
    }

    if(sem_post(&m_imageSemaphore) < 0)
    {
        return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
    }

    return 0;
}

//--dark shmim

template<class derivedT>
int dmPokeWFS<derivedT>::allocate( const darkShmimT & dummy)
{
    static_cast<void>(dummy); //be unused
  
    std::unique_lock<std::mutex> lock(m_wfsImageMutex);

    m_darkImage.resize(derived().darkShmimMonitor().width(), derived().darkShmimMonitor().height());

    darkPixget = getPixPointer<float>(derived().darkShmimMonitor().dataType());

    if(derived().darkShmimMonitor().width() == derived().shmimMonitor().width() && 
         derived().darkShmimMonitor().height() == derived().shmimMonitor().height() )
    {
        std::cerr << "dark is valid " << derived().darkShmimMonitor().width() << " " << derived().shmimMonitor().width() << " ";
        std::cerr << derived().darkShmimMonitor().height() << " " << derived().shmimMonitor().height() << "\n";
        m_darkValid = true;
    }
    else
    {
        m_darkValid = false;
    }
    
    return 0;
}

template<class derivedT> 
int dmPokeWFS<derivedT>::processImage( void * curr_src,    
                                       const darkShmimT &  dummy
                                     )
{
    static_cast<void>(dummy); //be unused

    std::unique_lock<std::mutex> lock(m_wfsImageMutex);

    float * darkData = m_darkImage.data();

    //Copy the data out as float no matter what type it is
    uint64_t nPix = derived().darkShmimMonitor().width()*derived().darkShmimMonitor().height();
    for(unsigned nn=0; nn < nPix; ++nn)
    {
        darkData[nn] = darkPixget(curr_src, nn);
    }

    return 0;
}

template<class derivedT>
void dmPokeWFS<derivedT>::wfsThreadStart( dmPokeWFS * d)
{
   d->wfsThreadExec();
}

template<class derivedT>
void dmPokeWFS<derivedT>::wfsThreadExec()
{
    m_wfsThreadID = syscall(SYS_gettid);

    //Wait fpr the thread starter to finish initializing this thread.
    while(m_wfsThreadInit == true && derived().m_shutdown == 0)
    {
        sleep(1);
    }

    while(derived().m_shutdown == 0)
    {
        timespec ts;
        XWC_SEM_WAIT_TS_RETVOID_DERIVED(ts, m_wfsSemWait_sec, m_wfsSemWait_nsec);
      
        XWC_SEM_TIMEDWAIT_LOOP_DERIVED( m_wfsSemaphore, ts )

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

        derived().template state(stateCodes::OPERATING);

        while(!m_pokeImage.valid())
        {
            mx::sys::milliSleep(10);
        }

        m_stopMeasurement = false;

        bool firstRun = true;

        while(!m_stopMeasurement && !derived().m_shutdown)
        {   
            if( derived().runSensor(firstRun) < 0)
            {
                derivedT::template log<software_error>({__FILE__, __LINE__, "runSensor returned error"});
                break;
            }
            
            if(m_stopMeasurement || derived().m_shutdown)
            {
                break;
            }

            if( derived().analyzeSensor() < 0)
            {
                derivedT::template log<software_error>({__FILE__, __LINE__, "runSensor returned error"});
                break;
            }

            ++m_counter;
            derived().updateIfChanged(m_indiP_measurement, "counter", m_counter);   
            derived().recordPokeLoop();

            firstRun = false;

            if(m_measuring == 1)
            {
                break;
            }
        }

        m_measuring = 0;
        m_single = 0;
        m_continuous = 0;

        derived().template state(stateCodes::READY);

        
    } //outer loop, will exit if derived().m_shutdown==true

    return;

}

template<class derivedT>
int dmPokeWFS<derivedT>::basicTimedPoke(float pokeSign)
{
    timespec ts;

    int sign = 1;
    if(pokeSign < 0) sign = -1;

    //Prepare the DM image with the pokes
    m_dmImage.setZero();

    for(size_t nn = 0; nn < m_poke_x.size(); ++nn)
    {
        m_dmImage( m_poke_x[nn], m_poke_y[nn]) = pokeSign*m_poke_amp;
    }

    //This is where the pokes are applied to the DM
    m_dmStream = m_dmImage;

    mx::sys::microSleep(m_dmSleep);

    //flush semaphore so we take the _next_ good image
    XWC_SEM_FLUSH_DERIVED(m_imageSemaphore);

    //** And wait one image to be sure we are on a whole poke**//
    XWC_SEM_WAIT_TS_DERIVED(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
    bool ready = false;
    while(!ready && !(m_stopMeasurement || derived().m_shutdown))
    {
        XWC_SEM_TIMEDWAIT_LOOP_DERIVED( m_imageSemaphore, ts )
        else
        {
            ready = true;
        }        
    }

    uint32_t n = 0;
    while(n < m_nPokeImages && !(m_stopMeasurement || derived().m_shutdown))
    {    
        //** Now we record the poke image **//
        XWC_SEM_WAIT_TS_DERIVED(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
        XWC_SEM_TIMEDWAIT_LOOP_DERIVED( m_imageSemaphore, ts )

        //If here, we got an image.  m_rawImage will have been updated
        m_pokeLocal +=  sign*m_rawImage();

        ++n;
    }
        
    if(m_stopMeasurement || derived().m_shutdown) 
    {
        m_dmImage.setZero();
        m_dmStream = m_dmImage;
        return 1;
    }

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::basicRunSensor()
{
    int rv;

    if(!m_pokeImage.valid())
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "poke image is not allocated"});
    }

    m_pokeLocal.setZero();

    for(unsigned nseq = 0; nseq < m_nPokeAverage; ++nseq)
    {

        //************** positive POKE **********************/

        rv = basicTimedPoke(+1);

        if(rv < 0)
        {
            derivedT::template log<software_error>({__FILE__, __LINE__});
            return rv;
        }
        else if (rv > 0) // shutdown
        {
            return rv;
        }

        if(m_stopMeasurement || derived().m_shutdown) 
        {
            break;
        }

        //************** NEGATIVE POKE **********************/

        rv = basicTimedPoke(-1);

        if(rv < 0)
        {
            derivedT::template log<software_error>({__FILE__, __LINE__});
            return rv;
        }
        else if (rv > 0) // shutdown
        {
            return rv;
        }
    
        if(m_stopMeasurement || derived().m_shutdown) 
        {
            break;
        }
    }

    try
    {
        m_pokeImage = m_pokeLocal/(2.0*m_nPokeImages*m_nPokeAverage);
    }
    catch(const std::exception& e)
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, e.what()});
    }

   

    m_dmImage.setZero();
    m_dmStream = m_dmImage;

    return 0;
}

template<class derivedT>
int dmPokeWFS<derivedT>::updateMeasurement( float deltaX,
                                            float deltaY
                                          )
{
    m_deltaX = deltaX;
    m_deltaY = deltaY;
    m_indiP_measurement["delta_x"] = deltaX;
    m_indiP_measurement["delta_y"] = deltaY;

    return 0;
}

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_nPokeImages )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_nPokeImages, ipRecv)
   
    float target;

    if( derived().template indiTargetUpdate(m_indiP_nPokeImages, target, ipRecv, false) < 0)
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__});
    }

    m_nPokeImages = target;

    return 0;
}

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_nPokeAverage )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_nPokeAverage, ipRecv)
   
    float target;

    if( derived().template indiTargetUpdate(m_indiP_nPokeAverage, target, ipRecv, false) < 0)
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__});
    }

    m_nPokeAverage = target;

    return 0;
}

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_poke_amp )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_poke_amp, ipRecv)
   
    float target;

    if( derived().template indiTargetUpdate(m_indiP_poke_amp, target, ipRecv, false) < 0)
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__});
    }

    m_poke_amp = target;

    return 0;
}

template<class derivedT>
INDI_SETCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_wfsFps )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_wfsFps, ipRecv)
   
    if( ipRecv.find("current") != true ) //this isn't valid
    {
        return 0;
    }
   
    m_wfsFps = ipRecv["current"].get<float>();

    return 0;
}

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_single )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_single, ipRecv)
   
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
                return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
            }
        }
    }

    return 0;
}

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_continuous )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_continuous, ipRecv)
   
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
                return derivedT::template log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
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

template<class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_stop )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_stop, ipRecv)
   
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

template<class derivedT>
int dmPokeWFS<derivedT>::recordTelem(const telem_pokeloop *)
{
    return recordPokeLoop(true);
}

template<class derivedT>
int dmPokeWFS<derivedT>::recordPokeLoop(bool force)
{
    static int measuring = -1;
    static float deltaX = std::numeric_limits<float>::max();
    static float deltaY = std::numeric_limits<float>::max();
    static uint64_t counter = std::numeric_limits<uint64_t>::max();

    if(force || (m_counter != counter) || (m_deltaX != deltaX) || (m_deltaY != deltaY) || (m_measuring != measuring))
    {
        uint8_t meas = m_measuring;
        derived().template telem<telem_pokeloop>({meas, m_deltaX, m_deltaY, m_counter});

        measuring = m_measuring;
        deltaX = m_deltaX;
        deltaY = m_deltaY;
        counter = m_counter;
    }

    return 0;
}

/// Call dmPokeWFS::setupConfig with error checking
/**
  * \param cfig the application configurator 
  */
#define DMPOKEWFS_SETUP_CONFIG( cfig )                                                   \
    if(dmPokeWFST::setupConfig(cfig) < 0)                                                \
    {                                                                                    \
        log<software_error>({__FILE__, __LINE__, "Error from dmPokeWFST::setupConfig"}); \
        m_shutdown = true;                                                               \
        return;                                                                          \
    }

/// Call dmPokeWFS::loadConfig with error checking
/** This must be inside a function that returns int, e.g. the standard loadConfigImpl.
  * \param cfig the application configurator 
  */
#define DMPOKEWFS_LOAD_CONFIG( cfig )                                                             \
    if(dmPokeWFST::loadConfig(cfig) < 0)                                                          \
    {                                                                                             \
        return log<software_error,-1>({__FILE__, __LINE__, "Error from dmPokeWFST::loadConfig"}); \
    } 

/// Call dmPokeWFS::appStartup with error checking
#define DMPOKEWFS_APP_STARTUP                                \
    if( dmPokeWFST::appStartup() < 0)                        \
    {                                                        \
        return log<software_error, -1>({__FILE__,__LINE__}); \
    }

/// Call dmPokeWFS::appLogic with error checking
#define DMPOKEWFS_APP_LOGIC                                  \
    if( dmPokeWFST::appLogic() < 0)                          \
    {                                                        \
        return log<software_error, -1>({__FILE__,__LINE__}); \
    }

/// Call dmPokeWFS::appShutdown with error checking
#define DMPOKEWFS_APP_SHUTDOWN                                                           \
    if(dmPokeWFST::appShutdown() < 0)                                                    \
    {                                                                                    \
        log<software_error>({__FILE__, __LINE__, "error from dmPokeWFST::appShutdown"}); \
    }

} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //dmPokeWFS_hpp
