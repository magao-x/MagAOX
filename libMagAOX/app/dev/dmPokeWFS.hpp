/** \file dmPokeWFS.hpp
  * \brief The MagAO-X DM Poke Centering header file
  *
  * \ingroup dmPokeWFS_files
  */

#ifndef dmPokeWFS_hpp
#define dmPokeWFS_hpp

#include <atomic>

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/improc/eigenCube.hpp>
using namespace mx::improc;

#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "semUtilsDerived.hpp"

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
  * - Must contain the following friend declarations (replace DERIVEDNAME with derivedT class name):
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
  *   property update is handled automatically after that.  The measurement should first be checked with \ref measurementInRange(),
  *   and if \ref m_saveDiagnostics is true analyzeSensor should clear it and save diagnostics with \ref writeDiagnostics().
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

    unsigned m_nSettleImages{ 1 }; ///< The number of WFS images to discard after the DM trigger.  Default is 1.

    std::string m_dmChan; ///< The DM channel stream to write pokes to, e.g. dm01disp06.

    std::vector<int> m_poke_x; ///< The x-coordinates of the actuators to poke.

    std::vector<int> m_poke_y; ///< The y-coordinates of the actuators to poke.

    float m_poke_amp{ 0.0 }; ///< The poke amplitude, in DM command units.  Default is 0.

    float m_dmSleep {10000}; ///<The time to sleep for the DM command to be applied, in microseconds. Default is 10000.

    std::string m_dmTriggerChan; ///< ImageStreamIO channel whose semaphore indicates the DM command has updated.

    float m_maxDelta{ 10 }; /**< Maximum magnitude of either measured delta component, in pixels.  Larger measurements
                                 are rejected and diagnostics are written.  <= 0 disables the check.  Default is 10. */

    std::string m_diagnosticsDir; /**< Directory for diagnostics written when a measurement is rejected.
                                       Default is `<basePath>/rawimages/<configName>`. */

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

    IMAGE m_dmTriggerStream {};

    int m_dmTriggerSemaphoreNumber {0}; ///< ImageStreamIO semaphore number claimed on m_dmTriggerChan.

    sem_t *m_dmTriggerSemaphore {nullptr}; ///< Semaphore posted when m_dmTriggerChan updates.

    bool m_dmTriggerStreamOpen {false}; ///< True when m_dmTriggerStream is connected.

    float m_deltaX{ 0 }; ///< The most recently published x delta measurement.

    float m_deltaY{ 0 }; ///< The most recently published y delta measurement.

    uint64_t m_counter{ 0 }; ///< The number of completed measurements.

    uint64_t m_wfsFrames{ 0 }; ///< The number of WFS frames received by processImage, protected by m_wfsImageMutex.

    timespec m_measStart{}; ///< The time the most recent measurement started.

    timespec m_measEnd{}; ///< The time the most recent measurement finished.

    uint64_t m_wfsFramesStart{ 0 }; ///< The value of m_wfsFrames when the most recent measurement started.

    uint64_t m_wfsFramesEnd{ 0 }; ///< The value of m_wfsFrames when the most recent measurement finished.

    uint64_t m_dmTriggerCnt0Start{ 0 }; ///< The DM trigger stream cnt0 when the most recent measurement started.

    uint64_t m_dmTriggerCnt0End{ 0 }; ///< The DM trigger stream cnt0 when the most recent measurement finished.

    double m_dmPlusSum{ 0 }; /**< Sum over + pokes of the mean DM trigger stream value at the poke actuators when
                                  accumulation starts, for the most recent measurement. */

    unsigned m_dmPlusN{ 0 }; ///< Number of + pokes summed in m_dmPlusSum.

    double m_dmMinusSum{ 0 }; /**< Sum over - pokes of the mean DM trigger stream value at the poke actuators when
                                   accumulation starts, for the most recent measurement. */

    unsigned m_dmMinusN{ 0 }; ///< Number of - pokes summed in m_dmMinusSum.

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

    std::atomic<bool> m_saveDiagnostics{ false }; /**< Set by the save_diagnostics INDI request, cleared when
                                                       diagnostics are saved after the next completed measurement. */

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

    /// Wait for the combined DM stream to update.
    /**
     * \returns +1 if exit is due to shutdown or stop request
     * \returns 0 if the DM trigger stream updated
     * \returns -1 if an error occurs
     */
    int waitForDmTrigger();

    /// Publish a delta measurement.
    /** Sets m_deltaX and m_deltaY and the corresponding elements of m_indiP_measurement.
     *
     * \returns 0 on success
     */
    int updateMeasurement( float deltaX, /**< [in] the measured x delta */
                           float deltaY  /**< [in] the measured y delta */
    );

    /// Check whether a delta measurement is within the allowed range.
    /**
     * \returns true if both components have magnitude no larger than m_maxDelta, or if m_maxDelta <= 0
     * \returns false otherwise
     */
    bool measurementInRange( float deltaX, /**< [in] the measured x delta, in pixels */
                             float deltaY  /**< [in] the measured y delta, in pixels */
    ) const;

    /// Write diagnostic FITS images for the most recent measurement.
    /** Writes the poke image, the last raw WFS image, the dark, the poke DM channel, and the combined DM stream to
     * `m_diagnosticsDir/YYYY_MM_DD/` as `<configName>_<timestamp>_<image>.fits`.  The configuration, timing, stream
     * counters, and mean combined DM command at the poke actuators for each poke sign of the most recent measurement
     * are appended to \p head, which is written to every file.
     *
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    int writeDiagnostics( std::string &prefix, /**< [out] the path and file name prefix, to which `_<image>.fits` is
                                                          appended */
                          mx::fits::fitsHeader<> &head /**< [in.out] header cards from the caller, to which this
                                                                     function appends its own */
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

    pcf::IndiProperty m_indiP_nSettleImages;
    INDI_NEWCALLBACK_DECL( derivedT, m_indiP_nSettleImages );

    pcf::IndiProperty m_indiP_wfsFps; ///< Property to get the FPS from the WFS camera
    INDI_SETCALLBACK_DECL(derivedT, m_indiP_wfsFps);

    pcf::IndiProperty m_indiP_single; ///< Switch to start a single measurement
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_single);

    pcf::IndiProperty m_indiP_continuous; ///< Switch to start continuous measurement
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_continuous);

    pcf::IndiProperty m_indiP_stop; ///< Switch to request that measurement stop
    INDI_NEWCALLBACK_DECL(derivedT, m_indiP_stop);

    pcf::IndiProperty m_indiP_saveDiagnostics; ///< Switch to request diagnostics after the next completed measurement
    INDI_NEWCALLBACK_DECL( derivedT, m_indiP_saveDiagnostics );

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
    config.add("pokecen.dmTriggerChannel", "", "pokecen.dmTriggerChannel", argType::Required, "pokecen", "dmTriggerChannel", false, "string", "The DM channel whose semaphore indicates that the combined DM command has updated. Defaults to dmChannel with a two digit channel suffix removed.");
    config.add("pokecen.pokeX", "", "pokecen.pokeX", argType::Required, "pokecen", "pokeX", false, "vector<int>", "The x-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeY", "", "pokecen.pokeY", argType::Required, "pokecen", "pokeY", false, "vector<int>", "The y-coordinates of the actuators to poke. ");
    config.add("pokecen.pokeAmp", "", "pokecen.pokeAmp", argType::Required, "pokecen", "pokeAmp", false, "float", "The poke amplitude, in DM command units. Default is 0.");
    config.add("pokecen.dmSleep", "", "pokecen.dmSleep", argType::Required, "pokecen", "dmSleep", false, "float", "The time to sleep for the DM command to be applied, in microseconds. Default is 10000.");
    config.add( "pokecen.nSettleImages",
                "",
                "pokecen.nSettleImages",
                argType::Required,
                "pokecen",
                "nSettleImages",
                false,
                "int",
                "The number of WFS images to discard after the DM trigger. Default 1." );
    config.add("pokecen.nPokeImages", "", "pokecen.nPokeImages", argType::Required, "pokecen", "nPokeImages", false, "int", "The number of poke images to average.  Default 5.");
    config.add("pokecen.nPokeAverage", "", "pokecen.nPokeAverage", argType::Required, "pokecen", "nPokeAverage", false, "int", "The number of poke sequences to average.  Default 10.");
    config.add(
        "pokecen.maxDelta",
        "",
        "pokecen.maxDelta",
        argType::Required,
        "pokecen",
        "maxDelta",
        false,
        "float",
        "Maximum magnitude of either measured delta component, in pixels.  Larger measurements are rejected and "
        "diagnostics are written.  <= 0 disables the check.  Default 10." );
    config.add( "pokecen.diagnosticsDir",
                "",
                "pokecen.diagnosticsDir",
                argType::Required,
                "pokecen",
                "diagnosticsDir",
                false,
                "string",
                "Directory for diagnostics written when a measurement is rejected.  Default is "
                "<basePath>/rawimages/<configName>." );

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

    m_dmTriggerChan = m_dmChan;
    if( m_dmTriggerChan.size() > 2 && m_dmTriggerChan[m_dmTriggerChan.size() - 1] >= '0' &&
        m_dmTriggerChan[m_dmTriggerChan.size() - 1] <= '9' && m_dmTriggerChan[m_dmTriggerChan.size() - 2] >= '0' &&
        m_dmTriggerChan[m_dmTriggerChan.size() - 2] <= '9' )
    {
        m_dmTriggerChan.resize( m_dmTriggerChan.size() - 2 );
    }
    config( m_dmTriggerChan, "pokecen.dmTriggerChannel" );

    config(m_poke_x, "pokecen.pokeX");

    config(m_poke_y, "pokecen.pokeY");

    if(m_poke_x.size() == 0 || (m_poke_x.size() != m_poke_y.size()))
    {
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "invalid poke specification"});
    }

    config(m_poke_amp, "pokecen.pokeAmp");

    config(m_dmSleep, "pokecen.dmSleep");

    config( m_nSettleImages, "pokecen.nSettleImages" );

    config(m_nPokeImages, "pokecen.nPokeImages");

    config(m_nPokeAverage, "pokecen.nPokeAverage");

    config( m_maxDelta, "pokecen.maxDelta" );

    std::string rawimageDir = mx::sys::getEnv( MAGAOX_env_rawimage );
    if( rawimageDir == "" )
    {
        rawimageDir = MAGAOX_rawimageRelPath;
    }
    m_diagnosticsDir = derived().basePath() + "/" + rawimageDir + "/" + derived().m_configName;
    config( m_diagnosticsDir, "pokecen.diagnosticsDir" );

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

    CREATE_REG_INDI_NEW_NUMBERI_DERIVED( m_indiP_nSettleImages, "nSettleImages", 0, 1000, 1, "%d", "", "" );
    m_indiP_nSettleImages["current"].setValue( m_nSettleImages );
    m_indiP_nSettleImages["target"].setValue( m_nSettleImages );

    REG_INDI_SETPROP_DERIVED(m_indiP_wfsFps, m_wfsCamDevName, std::string("fps"));

    CREATE_REG_INDI_NEW_TOGGLESWITCH_DERIVED( m_indiP_single, "single");

    CREATE_REG_INDI_NEW_TOGGLESWITCH_DERIVED( m_indiP_continuous, "continuous");

    CREATE_REG_INDI_NEW_REQUESTSWITCH_DERIVED( m_indiP_stop, "stop");

    CREATE_REG_INDI_NEW_REQUESTSWITCH_DERIVED( m_indiP_saveDiagnostics, "save_diagnostics" );

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
    derived().template updateIfChanged( m_indiP_nSettleImages, "current", m_nSettleImages );
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

    if( m_dmTriggerStreamOpen )
    {
        if( m_dmTriggerSemaphoreNumber >= 0 )
        {
            m_dmTriggerStream.semReadPID[m_dmTriggerSemaphoreNumber] = 0;
        }
        ImageStreamIO_closeIm( &m_dmTriggerStream );
        m_dmTriggerStreamOpen      = false;
        m_dmTriggerSemaphore       = nullptr;
        m_dmTriggerSemaphoreNumber = 0;
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

    m_dmStream.passive(true);

    m_dmImage.resize(m_dmStream.rows(), m_dmStream.cols());

    if( m_dmTriggerStreamOpen )
    {
        if( m_dmTriggerSemaphoreNumber >= 0 )
        {
            m_dmTriggerStream.semReadPID[m_dmTriggerSemaphoreNumber] = 0;
        }
        ImageStreamIO_closeIm( &m_dmTriggerStream );
        m_dmTriggerStreamOpen      = false;
        m_dmTriggerSemaphore       = nullptr;
        m_dmTriggerSemaphoreNumber = 0;
    }

    if( ImageStreamIO_openIm( &m_dmTriggerStream, m_dmTriggerChan.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, "error opening DM trigger channel " + m_dmTriggerChan } );
    }

    if( m_dmTriggerStream.md[0].sem < SEMAPHORE_MAXVAL )
    {
        ImageStreamIO_closeIm( &m_dmTriggerStream );
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, "DM trigger channel has no valid semaphore " + m_dmTriggerChan } );
    }

    m_dmTriggerSemaphoreNumber = ImageStreamIO_getsemwaitindex( &m_dmTriggerStream, m_dmTriggerSemaphoreNumber );
    if( m_dmTriggerSemaphoreNumber < 0 )
    {
        ImageStreamIO_closeIm( &m_dmTriggerStream );
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, "no valid semaphore found for DM trigger channel " + m_dmTriggerChan } );
    }

    ImageStreamIO_semflush( &m_dmTriggerStream, m_dmTriggerSemaphoreNumber );
    m_dmTriggerSemaphore  = m_dmTriggerStream.semptr[m_dmTriggerSemaphoreNumber];
    m_dmTriggerStreamOpen = true;

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

    ++m_wfsFrames;

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
                derivedT::template log<software_error>( { __FILE__, __LINE__, "analyzeSensor returned error" } );
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
    ImageStreamIO_semflush( &m_dmTriggerStream, m_dmTriggerSemaphoreNumber );
    m_dmStream = m_dmImage;

    int rv = waitForDmTrigger();
    if(rv < 0)
    {
        m_dmImage.setZero();
        m_dmStream = m_dmImage;
        return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "error waiting for DM trigger"});
    }
    else if(rv > 0)
    {
        m_dmImage.setZero();
        m_dmStream = m_dmImage;
        return rv;
    }

    mx::sys::microSleep(m_dmSleep);

    //flush semaphore so we take the _next_ good image
    XWC_SEM_FLUSH_DERIVED(m_imageSemaphore);

    //** And wait for settle images to be sure we are on a whole poke**//
    uint32_t nSettle = 0;
    while( nSettle < m_nSettleImages && !( m_stopMeasurement || derived().m_shutdown ) )
    {
        XWC_SEM_WAIT_TS_DERIVED( ts, m_imageSemWait_sec, m_imageSemWait_nsec );
        XWC_SEM_TIMEDWAIT_LOOP_DERIVED( m_imageSemaphore, ts )

        ++nSettle;
    }

    // Record the combined DM command at the poke actuators as accumulation starts
    if( m_dmTriggerStreamOpen && m_dmTriggerStream.md->datatype == _DATATYPE_FLOAT &&
        m_dmTriggerStream.md->size[0] == m_dmImage.rows() && m_dmTriggerStream.md->size[1] == m_dmImage.cols() )
    {
        double dmVal = 0;
        for( size_t nn = 0; nn < m_poke_x.size(); ++nn )
        {
            dmVal += m_dmTriggerStream.array.F[m_poke_x[nn] + m_poke_y[nn] * m_dmTriggerStream.md->size[0]];
        }
        dmVal /= m_poke_x.size();

        if( sign > 0 )
        {
            m_dmPlusSum += dmVal;
            ++m_dmPlusN;
        }
        else
        {
            m_dmMinusSum += dmVal;
            ++m_dmMinusN;
        }
    }

    uint32_t n = 0;
    while(n < m_nPokeImages && !(m_stopMeasurement || derived().m_shutdown))
    {
        //** Now we record the poke image **//
        XWC_SEM_WAIT_TS_DERIVED(ts, m_imageSemWait_sec, m_imageSemWait_nsec);
        XWC_SEM_TIMEDWAIT_LOOP_DERIVED( m_imageSemaphore, ts )

        // If here, we got an image.  m_rawImage will have been updated
        { // mutex scope
            std::unique_lock<std::mutex> lock( m_wfsImageMutex );
            m_pokeLocal += sign * m_rawImage();
        }

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

    clock_gettime( CLOCK_REALTIME, &m_measStart );
    { // mutex scope
        std::unique_lock<std::mutex> lock( m_wfsImageMutex );
        m_wfsFramesStart = m_wfsFrames;
    }
    m_dmTriggerCnt0Start = m_dmTriggerStreamOpen ? m_dmTriggerStream.md->cnt0 : 0;

    m_dmPlusSum  = 0;
    m_dmPlusN    = 0;
    m_dmMinusSum = 0;
    m_dmMinusN   = 0;

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

    clock_gettime( CLOCK_REALTIME, &m_measEnd );
    { // mutex scope
        std::unique_lock<std::mutex> lock( m_wfsImageMutex );
        m_wfsFramesEnd = m_wfsFrames;
    }
    m_dmTriggerCnt0End = m_dmTriggerStreamOpen ? m_dmTriggerStream.md->cnt0 : 0;

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

template <class derivedT>
int dmPokeWFS<derivedT>::waitForDmTrigger()
{
    if( !m_dmTriggerStreamOpen || m_dmTriggerSemaphore == nullptr )
    {
        return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "DM trigger stream is not open" } );
    }

    bool ready = false;
    while( !ready && !( m_stopMeasurement || derived().m_shutdown ) )
    {
        timespec ts;
        XWC_SEM_WAIT_TS_DERIVED( ts, m_imageSemWait_sec, m_imageSemWait_nsec );

        if( sem_timedwait( m_dmTriggerSemaphore, &ts ) != 0 )
        {
            if( errno == EINTR || errno == ETIMEDOUT )
            {
                continue;
            }

            return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
        }

        ready = true;
    }

    if(m_stopMeasurement || derived().m_shutdown)
    {
        return 1;
    }

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

template <class derivedT>
bool dmPokeWFS<derivedT>::measurementInRange( float deltaX, float deltaY ) const
{
    if( m_maxDelta <= 0 )
    {
        return true;
    }

    return ( fabs( deltaX ) <= m_maxDelta && fabs( deltaY ) <= m_maxDelta );
}

template <class derivedT>
int dmPokeWFS<derivedT>::writeDiagnostics( std::string &prefix, mx::fits::fitsHeader<> &head )
{
    try
    {
        timespec now;
        clock_gettime( CLOCK_REALTIME, &now );

        std::string tstamp, relPath;
        if( !!file::fileTimeRelPath( tstamp, relPath, now.tv_sec, now.tv_nsec ) )
        {
            return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "error making timestamp" } );
        }

        std::string dir = m_diagnosticsDir + "/" + relPath;
        if( !!mx::ioutils::createDirectories( dir ) )
        {
            return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "error creating " + dir } );
        }

        prefix = dir + "/" + derived().m_configName + "_" + tstamp;

        // Copy images first so the header and files describe one consistent snapshot
        mx::improc::eigenImage<float> raw, dark;
        { // mutex scope
            std::unique_lock<std::mutex> lock( m_wfsImageMutex );
            raw  = m_rawImage();
            dark = m_darkImage;
        }

        mx::improc::eigenImage<float> poke   = m_pokeImage();
        mx::improc::eigenImage<float> dmChan = m_dmStream();

        mx::improc::eigenImage<float> dmComb;
        uint64_t                      dmTriggerCnt0 = 0;
        if( m_dmTriggerStreamOpen && m_dmTriggerStream.md->datatype == _DATATYPE_FLOAT )
        {
            dmTriggerCnt0 = m_dmTriggerStream.md->cnt0;
            dmComb        = mx::improc::eigenMap<float>(
                m_dmTriggerStream.array.F, m_dmTriggerStream.md->size[0], m_dmTriggerStream.md->size[1] );
        }

        head.append( "COUNTER", m_counter, "number of completed measurements" );
        head.append( "POKEAMP", m_poke_amp, "poke amplitude [DM units]" );
        head.append( "NPOKEIM", m_nPokeImages, "images accumulated per poke" );
        head.append( "NPOKEAVG", m_nPokeAverage, "poke pairs averaged" );
        head.append( "NSETTLE", m_nSettleImages, "WFS images discarded after DM trigger" );
        head.append( "DMSLEEP", m_dmSleep, "sleep after DM trigger [us]" );
        head.append( "WFSFPS", m_wfsFps, "WFS camera frame rate [Hz]" );
        head.append( "MAXDELTA", m_maxDelta, "maximum allowed delta [pix]" );
        head.append( "WFSSHMIM", derived().shmimMonitor().shmimName(), "WFS stream" );
        head.append( "DRKSHMIM", derived().darkShmimMonitor().shmimName(), "WFS dark stream" );
        head.append( "DARKVLD", static_cast<int>( m_darkValid ), "dark subtraction applied" );
        head.append( "DMCHAN", m_dmChan, "DM poke channel stream" );
        head.append( "DMCHCNT0", m_dmStream.cnt0(), "DM poke channel cnt0" );
        head.append( "DMTRIG", m_dmTriggerChan, "DM trigger stream" );
        head.append( "DMTRCNT0", dmTriggerCnt0, "DM trigger stream cnt0 at write" );
        head.append( "DMTRUPD", m_dmTriggerCnt0End - m_dmTriggerCnt0Start, "DM trigger updates during measurement" );
        head.append( "WFSFRMS", m_wfsFramesEnd - m_wfsFramesStart, "WFS frames received during measurement" );
        head.append( "WFSUSED",
                     2 * m_nPokeAverage * ( m_nSettleImages + m_nPokeImages ),
                     "WFS frames waited on in measurement" );
        float dmPlus  = ( m_dmPlusN > 0 ) ? m_dmPlusSum / m_dmPlusN : 0;
        float dmMinus = ( m_dmMinusN > 0 ) ? m_dmMinusSum / m_dmMinusN : 0;
        head.append( "DMPLUS", dmPlus, "DM trigger at poke acts, + windows" );
        head.append( "DMMINUS", dmMinus, "DM trigger at poke acts, - windows" );
        head.append( "DMHALFDF", 0.5 * ( dmPlus - dmMinus ), "(DMPLUS-DMMINUS)/2, expect POKEAMP" );
        head.append( "TSTART", mx::sys::ISO8601DateTimeStr( m_measStart ), "measurement start [UTC]" );
        head.append( "TEND", mx::sys::ISO8601DateTimeStr( m_measEnd ), "measurement end [UTC]" );

        const std::vector<std::pair<std::string, const mx::improc::eigenImage<float> *>> images = {
            { "poke", &poke }, { "raw", &raw }, { "dark", &dark }, { "dmchan", &dmChan }, { "dmcomb", &dmComb } };

        mx::fits::fitsFile<float> ff;
        for( auto &im : images )
        {
            if( im.second->size() == 0 )
            {
                continue;
            }

            std::string fname = prefix + "_" + im.first + ".fits";
            if( !!ff.write( fname, *im.second, head ) )
            {
                return derivedT::template log<software_error, -1>( { __FILE__, __LINE__, "error writing " + fname } );
            }
        }
    }
    catch( const std::exception &e )
    {
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, std::string( "exception writing diagnostics: " ) + e.what() } );
    }
    catch( ... )
    {
        return derivedT::template log<software_error, -1>(
            { __FILE__, __LINE__, "unknown exception writing diagnostics" } );
    }

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

template <class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_nSettleImages )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED( m_indiP_nSettleImages, ipRecv )

    float target;

    if( derived().template indiTargetUpdate( m_indiP_nSettleImages, target, ipRecv, false ) < 0 )
    {
        return derivedT::template log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    m_nSettleImages = target;

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

template <class derivedT>
INDI_NEWCALLBACK_DEFN( dmPokeWFS<derivedT>, m_indiP_saveDiagnostics )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED( m_indiP_saveDiagnostics, ipRecv )

    if( ipRecv.find( "request" ) != true ) // this isn't valid
    {
        return -1;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_saveDiagnostics = true;
        derivedT::template log<text_log>( "diagnostics will be saved after the next completed measurement" );
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
