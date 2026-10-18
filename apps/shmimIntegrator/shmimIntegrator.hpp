/** \file shmimIntegrator.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef shmimIntegrator_hpp
#define shmimIntegrator_hpp

#include <chrono>
#include <condition_variable>
#include <limits>

#include <mx/math/floatUtils.hpp>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/// Configuration and INDI tag for the first optional dark monitor.
struct darkShmimT
{
    /// Return the configuration section for this monitor.
    static std::string configSection()
    {
        return "darkShmim";
    };

    /// Return the prefix for this monitor's INDI properties.
    static std::string indiPrefix()
    {
        return "dark";
    };
};

/// Configuration and INDI tag for the second optional dark monitor.
struct dark2ShmimT
{
    /// Return the configuration section for this monitor.
    static std::string configSection()
    {
        return "dark2Shmim";
    };

    /// Return the prefix for this monitor's INDI properties.
    static std::string indiPrefix()
    {
        return "dark2";
    };
};

/** \defgroup shmimIntegrator ImageStreamIO Stream Integrator
 * \brief Integrates (i.e. averages) an ImageStreamIO image stream.
 *
 * <a href="../handbook/operating/software/apps/shmimIntegrator.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup shmimIntegrator_files ImageStreamIO Stream Integrator
 * \ingroup shmimIntegrator
 */

/** MagAO-X application for averaging an ImageStreamIO stream.
 * Shared image/configuration state is protected by m_imageMutex. INDI operations
 * take m_indiMutex first, snapshot image state, then release m_imageMutex before
 * sending properties. Workers never acquire m_indiMutex while holding m_imageMutex.
 * The frame-grabber alone owns its base configuration and claimed output image.
 * \ingroup shmimIntegrator
 */
class shmimIntegrator : public MagAOXApp<true>,
                        public dev::shmimMonitor<shmimIntegrator>,
                        public dev::shmimMonitor<shmimIntegrator, darkShmimT>,
                        public dev::shmimMonitor<shmimIntegrator, dark2ShmimT>,
                        public dev::frameGrabber<shmimIntegrator>,
                        public dev::telemeter<shmimIntegrator>
{
    friend class shmimIntegrator_test;
    friend class dev::shmimMonitor<shmimIntegrator>;
    friend class dev::shmimMonitor<shmimIntegrator, darkShmimT>;
    friend class dev::shmimMonitor<shmimIntegrator, dark2ShmimT>;
    friend class dev::frameGrabber<shmimIntegrator>;
    friend class dev::telemeter<shmimIntegrator>;

    typedef dev::shmimMonitor<shmimIntegrator>              shmimMonitorT;
    typedef dev::shmimMonitor<shmimIntegrator, darkShmimT>  darkMonitorT;
    typedef dev::shmimMonitor<shmimIntegrator, dark2ShmimT> dark2MonitorT;
    typedef dev::frameGrabber<shmimIntegrator>              frameGrabberT;
    typedef dev::telemeter<shmimIntegrator>                 telemeterT;
    typedef float                                           realT;
    typedef mx::improc::eigenImage<realT>                   imageT;

  public:
    /// Averaged images are not flipped by the frame-grabber.
    static constexpr bool c_frameGrabber_flippable = false;

  protected:
    /** \name Configurable Parameters - Data
     * @{ */
    /// Fallback window length before a time-based average receives FPS.
    unsigned m_nAverageDefault{ 10 };

    /// Device providing the input FPS.
    std::string m_fpsSource;

    /// Requested averaging time; zero selects a frame-count window.
    float m_avgTime{ 0 };

    /// Requested update interval; zero selects block averaging.
    unsigned m_nUpdate{ 0 };

    /// Whether block averaging continues after a completed image.
    bool m_continuous{ true };

    /// Whether source frames are currently accepted.
    bool m_running{ true };

    /// Device providing the state string used to identify saved images.
    std::string m_stateSource;

    /// Enable saving and reloading state-matched averaged images.
    bool m_fileSaver{ false };
    ///@}

    /// Committed input geometry and converter, imported only by its monitor.
    struct imageInfo
    {
        uint32_t m_width{ 0 };                            ///< Committed width.
        uint32_t m_height{ 0 };                           ///< Committed height.
        size_t   m_pixels{ 0 };                           ///< Validated pixel count; zero means unavailable.
        realT ( *m_pixget )( void *, size_t ){ nullptr }; ///< Converter for this allocation.
    };

    /// One optional dark image, owned under m_imageMutex.
    struct darkFrame
    {
        imageInfo m_info;            ///< Descriptor for this monitor allocation.
        imageT    m_image;           ///< Pixels copied from the dark stream.
        bool      m_set{ false };    ///< A complete frame has been installed since allocation.
        bool      m_failed{ false }; ///< A failed allocation gates output until a successful allocation of this dark.
    };

    /// Protects all app image, configuration, readiness, and state-string data below.
    std::mutex m_imageMutex;

    /// Wakes the output worker for data, reconfiguration, or shutdown; never a data count.
    std::condition_variable m_imageCondition;

    imageInfo m_source;                         ///< Committed source descriptor; never synthesized from a saved file.
    bool      m_sourceFailed{ false };          ///< Invalid source allocation also gates saved-image publication.
    darkFrame m_dark;                           ///< First optional correction image.
    darkFrame m_dark2;                          ///< Second optional correction image.
    mx::improc::eigenCube<realT> m_accumImages; ///< Moving-average history for the prepared generation.
    imageT                       m_avgImage;    ///< Working or completed image protected by the readiness flags.

    unsigned m_nAverage{ 10 };            ///< Requested frame count, adjusted when time-based settings are prepared.
    float    m_fps{ 0 };                  ///< Latest requested source FPS.
    unsigned m_activeAverage{ 10 };       ///< Frame count used by the prepared buffers.
    unsigned m_activeUpdate{ 0 };         ///< Update interval used by the prepared buffers.
    unsigned m_nprocessed{ 0 };           ///< Burn-in count, saturated at the active window size.
    size_t   m_currImage{ 0 };            ///< Next moving-average history slot.
    size_t   m_sinceUpdate{ 0 };          ///< Frames accumulated since the last completed average.
    bool     m_updated{ false };          ///< One completed but unclaimed result is available.
    bool     m_inFlight{ false };         ///< A claimed result is being copied; preserves producer backpressure.
    bool     m_prepared{ false };         ///< Working buffers are valid for m_generation.
    bool     m_savedImage{ false };       ///< m_avgImage contains a complete saved image awaiting publication.
    bool     m_stopping{ false };         ///< Shutdown gate and condition-variable wake predicate.
    uint64_t m_generation{ 1 };           ///< Latest configuration request; every allocation advances it.
    uint64_t m_configuredGeneration{ 0 }; ///< Generation selected by configureAcquisition().
    uint64_t m_activeGeneration{ 0 };     ///< Generation acknowledged after output stream setup.

    bool        m_imageValid{ false };              ///< Published/saved image matches the current valid state string.
    std::string m_stateString;                      ///< Current state used in saved-image filenames.
    bool        m_stateStringValid{ false };        ///< State provider has declared the current state valid.
    bool        m_stateStringValidOnStart{ false }; ///< State validity at the start of one-shot averaging.
    bool        m_stateStringChanged{ false };      ///< Invalidates an integration or requests a saved-image lookup.
    uint64_t    m_stateRevision{ 0 };               ///< Rejects disk reads overtaken by state or start/stop changes.
    std::string m_fileSaveDir;                      ///< Startup-configured directory for saved averages.

    imageT m_claimedImage;     ///< Frame-grabber-only immutable snapshot through loadImageIntoStream().
    bool   m_claimed{ false }; ///< Frame-grabber owns one snapshot matching its configured output.

  public:
    /// Initialize the optional dark monitor behavior.
    shmimIntegrator();

    /// Destroy after appShutdown() has joined the workers.
    ~shmimIntegrator() noexcept = default;

    /// Register application and device-helper configuration.
    void setupConfig() override;

    /// Load configuration without starting workers.
    int loadConfigImpl( mx::app::appConfigurator &configurator /**< [in] configuration source */ );

    /// Load configuration and request shutdown if it is invalid.
    void loadConfig() override;

    /// Register INDI properties and start the workers.
    int appStartup() override;

    /// Publish synchronized status and perform pending saved-image lookups.
    int appLogic() override;

    /// Wake the output waiter and join every started worker.
    int appShutdown() override;

  protected:
    /// Validate geometry and select an input converter; invalid input returns an empty descriptor.
    static imageInfo describeImage( uint32_t width /**< [in] input width */,
                                    uint32_t height /**< [in] input height */,
                                    uint8_t  type /**< [in] ImageStreamIO datatype */ );

    /// Parse an entire finite numeric element, rejecting malformed INDI values.
    static bool readNumber( const pcf::IndiElement &element /**< [in] numeric element */,
                            double                 &value /**< [out] parsed value */ );

    /// Publish an image-state snapshot; caller holds m_indiMutex.
    void updateIntegratorINDI();

    /// Invalidate unclaimed work and advance the request generation; caller holds m_imageMutex.
    void requestReconfigure();

    /// Import a source descriptor and request a coordinated configuration transition.
    int allocate( const dev::shmimT &tag /**< [in] source monitor tag */ );

    /// Accumulate a source frame using the acknowledged generation.
    int processImage( void              *source /**< [in] current source frame */,
                      const dev::shmimT &tag /**< [in] source monitor tag */ );

    /// Allocate the first dark and reset its readiness.
    int allocate( const darkShmimT &tag /**< [in] first-dark monitor tag */ );

    /// Install a complete first dark atomically with respect to subtraction.
    int processImage( void             *source /**< [in] current dark frame */,
                      const darkShmimT &tag /**< [in] first-dark monitor tag */ );

    /// Allocate the second dark and reset its readiness.
    int allocate( const dark2ShmimT &tag /**< [in] second-dark monitor tag */ );

    /// Install a complete second dark atomically with respect to subtraction.
    int processImage( void              *source /**< [in] current dark frame */,
                      const dark2ShmimT &tag /**< [in] second-dark monitor tag */ );

    /// Common optional-dark allocation and transition.
    int allocateDark( darkFrame &dark /**< [in/out] dark storage */,
                      uint32_t   width /**< [in] monitor width */,
                      uint32_t   height /**< [in] monitor height */,
                      uint8_t    type /**< [in] monitor datatype */ );

    /// Copy an entire dark under the image-state lock.
    int processDark( darkFrame &dark /**< [in/out] dark storage */, void *source /**< [in] current frame */ );

    /// Read the newest matching saved image without holding either app mutex.
    int findMatchingDark();

    /// Install a disk-read snapshot only if its configuration and state are still current.
    int installSavedImage( imageT  &image /**< [in/out] candidate pixels, moved on success */,
                           uint64_t generation /**< [in] request generation at lookup start */,
                           uint64_t revision /**< [in] state revision at lookup start */ );

    /// Save an independently owned completed average outside the image-state lock.
    int saveImage( const imageT      &image /**< [in] completed average */,
                   const std::string &stateString /**< [in] state at completion */ );

    /** \name dev::frameGrabber interface
     * @{ */
    /// Prepare buffers and select a generation before the base configures its stream.
    int configureAcquisition();

    /// Return the synchronized estimated output frame rate.
    float fps();

    /// Acknowledge setup only if no newer generation arrived during stream configuration.
    int startAcquisition();

    /// Wait for and claim one output; return positive for no data/restart, negative for shutdown/error.
    int acquireAndCheckValid();

    /// Copy the frame-grabber-owned snapshot into a matching output frame.
    int loadImageIntoStream( void *destination /**< [out] configured output frame */ );

    /// Retire output acknowledgement after the base finishes any in-flight publication.
    int reconfig();
    ///@}

    pcf::IndiProperty m_indiP_nAverage;       ///< Requested/current frame count.
    pcf::IndiProperty m_indiP_avgTime;        ///< Requested/current averaging duration.
    pcf::IndiProperty m_indiP_nUpdate;        ///< Requested/current update cadence.
    pcf::IndiProperty m_indiP_startAveraging; ///< Start/stop switch.

    /// Request a frame-count change.
    INDI_NEWCALLBACK_DECL( shmimIntegrator, m_indiP_nAverage );

    /// Request an averaging-duration change.
    INDI_NEWCALLBACK_DECL( shmimIntegrator, m_indiP_avgTime );

    /// Request an update-cadence change.
    INDI_NEWCALLBACK_DECL( shmimIntegrator, m_indiP_nUpdate );

    /// Start or stop averaging and invalidate stale disk lookups.
    INDI_NEWCALLBACK_DECL( shmimIntegrator, m_indiP_startAveraging );

    pcf::IndiProperty m_indiP_fpsSource; ///< Subscription to the source FPS.

    /// Stage a changed input frame rate.
    INDI_SETCALLBACK_DECL( shmimIntegrator, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_fps;         ///< Estimated output FPS.
    pcf::IndiProperty m_indiP_stateSource; ///< Subscription to the state string and validity.

    /// Track state changes and invalidate obsolete images/lookups.
    INDI_SETCALLBACK_DECL( shmimIntegrator, m_indiP_stateSource );

    pcf::IndiProperty m_indiP_imageValid; ///< Whether the current saved average matches the state.

    /** \name Telemeter interface
     * @{ */
    /// Check whether frame timing telemetry is due.
    int checkRecordTimes();

    /// Record frame timing telemetry.
    int recordTelem( const telem_fgtimings *type /**< [in] telemetry dispatch tag */ );
    ///@}
};

inline shmimIntegrator::shmimIntegrator() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    darkMonitorT::m_getExistingFirst = true;
    return;
}

inline void shmimIntegrator::setupConfig()
{
    SHMIMMONITORT_SETUP_CONFIG( shmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( darkMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( dark2MonitorT, config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "integrator.nAverage",
                "",
                "integrator.nAverage",
                argType::Required,
                "integrator",
                "nAverage",
                false,
                "unsigned",
                "The default number of frames to average.  Default 10. Can be changed via INDI." );
    config.add(
        "integrator.fpsSource",
        "",
        "integrator.fpsSource",
        argType::Required,
        "integrator",
        "fpsSource",
        false,
        "string",
        "Device name for getting fps if time-based averaging is used.  This device should have *.fps.current." );

    config.add( "integrator.avgTime",
                "",
                "integrator.avgTime",
                argType::Required,
                "integrator",
                "avgTime",
                false,
                "float",
                "If non zero, then m_nAverage adjusts automatically to keep a constant averaging time [sec].  Default "
                "0. Can be changed via INDI." );

    config.add( "integrator.nUpdate",
                "",
                "integrator.nUpdate",
                argType::Required,
                "integrator",
                "nUpdate",
                false,
                "unsigned",
                "The rate at which to update the average.  If 0 < m_nUpdate < m_nAverage then this is a moving "
                "averager. Default 0.  If 0, then it is a simple average." );

    config.add( "integrator.continuous",
                "",
                "integrator.continuous",
                argType::Required,
                "integrator",
                "continuous",
                false,
                "bool",
                "Flag controlling whether averaging is continuous or only when triggered.  Default true." );
    config.add( "integrator.running",
                "",
                "integrator.running",
                argType::Required,
                "integrator",
                "running",
                false,
                "bool",
                "Flag controlling whether averaging is running at startup.  Default true." );

    config.add( "integrator.stateSource",
                "",
                "integrator.stateSource",
                argType::Required,
                "integrator",
                "stateSource",
                false,
                "string",
                "///< Device name for getting the state string for file management.  This device should have "
                "*.state_string.current." );
    config.add( "integrator.fileSaver",
                "",
                "integrator.fileSaver",
                argType::Required,
                "integrator",
                "fileSaver",
                false,
                "bool",
                "Flag controlling whether this saves and reloads files automatically.  Default false." );
}

inline int shmimIntegrator::loadConfigImpl( mx::app::appConfigurator &_config )
{

    SHMIMMONITORT_LOAD_CONFIG( shmimMonitorT, _config );
    SHMIMMONITORT_LOAD_CONFIG( darkMonitorT, _config );
    SHMIMMONITORT_LOAD_CONFIG( dark2MonitorT, _config );
    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_nAverageDefault, "integrator.nAverage" );
    m_nAverage = m_nAverageDefault;
    _config( m_fpsSource, "integrator.fpsSource" );
    _config( m_avgTime, "integrator.avgTime" );
    _config( m_nUpdate, "integrator.nUpdate" );

    _config( m_continuous, "integrator.continuous" );

    _config( m_running, "integrator.running" );

    _config( m_stateSource, "integrator.stateSource" );
    _config( m_fileSaver, "integrator.fileSaver" );

    if( !m_nAverageDefault || !mx::math::isFinite( m_avgTime ) || m_avgTime < 0 )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid integrator configuration" } );
    return 0;
}

inline void shmimIntegrator::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
        m_shutdown = 1;
}

inline int shmimIntegrator::appStartup()
{

    createStandardIndiNumber<unsigned>(
        m_indiP_nAverage, "nAverage", 1, std::numeric_limits<unsigned>::max(), 1, "%u" );
    m_indiP_nAverage["current"].set( m_nAverage );
    m_indiP_nAverage["target"].set( m_nAverage );

    if( registerIndiPropertyNew( m_indiP_nAverage, INDI_NEWCALLBACK( m_indiP_nAverage ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_avgTime, "avgTime", 0, std::numeric_limits<float>::max(), 0, "%0.1f" );
    m_indiP_avgTime["current"].set( m_avgTime );
    m_indiP_avgTime["target"].set( m_avgTime );

    if( registerIndiPropertyNew( m_indiP_avgTime, INDI_NEWCALLBACK( m_indiP_avgTime ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiNumber<unsigned>( m_indiP_nUpdate, "nUpdate", 0, std::numeric_limits<unsigned>::max(), 1, "%u" );
    m_indiP_nUpdate["current"].set( m_nUpdate );
    m_indiP_nUpdate["target"].set( m_nUpdate );

    if( registerIndiPropertyNew( m_indiP_nUpdate, INDI_NEWCALLBACK( m_indiP_nUpdate ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_startAveraging, "start" );
    if( registerIndiPropertyNew( m_indiP_startAveraging, INDI_NEWCALLBACK( m_indiP_startAveraging ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( m_fpsSource != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_fps, "fps", "", "" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    m_indiP_fps["current"].set( 0 );

    if( m_fileSaver == true && m_stateSource != "" )
    {
        REG_INDI_SETPROP( m_indiP_stateSource, m_stateSource, std::string( "state_string" ) );

        createROIndiText( m_indiP_imageValid, "image_valid", "flag", "Image Valid", "Image", "Valid" );
        if( !m_imageValid ) // making sure we stay up with default
        {
            m_indiP_imageValid["flag"] = "no";
        }
        else
        {
            m_indiP_imageValid["flag"] = "yes";
        }

        if( registerIndiPropertyReadOnly( m_indiP_imageValid ) < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            return -1;
        }

        m_fileSaveDir = m_calibDir + "/" + m_configName;

        // Create save directory.
        errno = 0;
        if( mkdir( m_fileSaveDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH ) < 0 )
        {
            if( errno != EEXIST )
            {
                std::stringstream logss;
                logss << "Failed to create image directory (" << m_fileSaveDir
                      << ").  Errno says: " << strerror( errno );
                log<software_critical>( { __FILE__, __LINE__, errno, 0, logss.str() } );

                return -1;
            }
        }
    }

    SHMIMMONITORT_APP_STARTUP( shmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( darkMonitorT );
    SHMIMMONITORT_APP_STARTUP( dark2MonitorT );
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    state( stateCodes::READY );

    return 0;
}

inline int shmimIntegrator::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( shmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( darkMonitorT );
    SHMIMMONITORT_APP_LOGIC( dark2MonitorT );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    if( findMatchingDark() < 0 )
        log<software_error>( { __FILE__, __LINE__, "saved-image lookup failed" } );

    std::lock_guard<std::mutex> lock( m_indiMutex );
    SHMIMMONITORT_UPDATE_INDI( shmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( darkMonitorT );
    SHMIMMONITORT_UPDATE_INDI( dark2MonitorT );
    FRAMEGRABBER_UPDATE_INDI;
    updateIntegratorINDI();
    return 0;
}

inline void shmimIntegrator::updateIntegratorINDI()
{
    bool     running, valid;
    unsigned average, update;
    float    time;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        running = m_running;
        valid   = m_imageValid;
        average = m_nAverage;
        update  = m_nUpdate;
        time    = m_avgTime;
    }
    state( running ? stateCodes::OPERATING : stateCodes::READY );
    updateSwitchIfChanged( m_indiP_startAveraging,
                           "toggle",
                           running ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           running ? INDI_BUSY : INDI_IDLE );
    if( m_fileSaver )
        updateIfChanged( m_indiP_imageValid, "flag", valid ? "yes" : "no" );
    updateIfChanged( m_indiP_nAverage, "current", average, INDI_IDLE );
    updateIfChanged( m_indiP_nAverage, "target", average, INDI_IDLE );
    updateIfChanged( m_indiP_avgTime, "current", time, INDI_IDLE );
    updateIfChanged( m_indiP_avgTime, "target", time, INDI_IDLE );
    updateIfChanged( m_indiP_nUpdate, "current", update, INDI_IDLE );
    updateIfChanged( m_indiP_nUpdate, "target", update, INDI_IDLE );
    updateIfChanged( m_indiP_fps, "current", fps(), INDI_OK );
}

inline int shmimIntegrator::appShutdown()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        m_stopping = true;
        m_updated  = false;
        m_imageCondition.notify_all();
    }
    // MagAOXApp sets its shutdown flag before invoking this, including startup failure.
    SHMIMMONITORT_APP_SHUTDOWN( shmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( darkMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( dark2MonitorT );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

inline shmimIntegrator::imageInfo shmimIntegrator::describeImage( uint32_t width, uint32_t height, uint8_t type )
{
    // eigenCube uses signed int products internally, including in resize().
    if( width == 0 || height == 0 || width > std::numeric_limits<int>::max() / height )
        return {};
    auto   converter = getPixPointer<realT>( type );
    size_t pixels    = size_t( width ) * height;
    if( !converter || pixels > size_t( std::numeric_limits<ptrdiff_t>::max() ) / sizeof( realT ) )
        return {};
    return { width, height, pixels, converter };
}

inline void shmimIntegrator::requestReconfigure()
{
    ++m_generation;
    m_prepared    = false;
    m_updated     = false;
    m_savedImage  = false;
    m_nprocessed  = 0;
    m_currImage   = 0;
    m_sinceUpdate = 0;
    m_imageValid  = false;
    if( m_fileSaver && !m_running )
        m_stateStringChanged = true;
    m_imageCondition.notify_all();
}

inline int shmimIntegrator::allocate( const dev::shmimT &tag )
{
    static_cast<void>( tag );
    std::lock_guard<std::mutex> lock( m_imageMutex );
    requestReconfigure();
    m_source       = describeImage( shmimMonitorT::m_width, shmimMonitorT::m_height, shmimMonitorT::m_dataType );
    m_sourceFailed = !m_source.m_pixels;
    if( m_sourceFailed )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid source geometry or datatype" } );
    return 0;
}

inline int shmimIntegrator::processImage( void *source, const dev::shmimT &tag )
{
    static_cast<void>( tag );
    imageT                       saved;
    std::string                  savedState;
    std::unique_lock<std::mutex> lock( m_imageMutex );
    if( m_stopping || !m_running || !m_prepared || m_activeGeneration != m_generation || !m_source.m_pixels )
        return 0;
    if( !source )
        return -1;

    if( m_activeUpdate == 0 )
    {
        if( m_updated || m_inFlight )
            return 0;
        if( m_sinceUpdate == 0 )
            m_avgImage.setZero();
        for( size_t n = 0; n < m_source.m_pixels; ++n )
            m_avgImage.data()[n] += m_source.m_pixget( source, n );
        if( ++m_sinceUpdate < m_activeAverage )
            return 0;
    }
    else
    {
        realT *data = m_accumImages.image( m_currImage ).data();
        for( size_t n = 0; n < m_source.m_pixels; ++n )
            data[n] = m_source.m_pixget( source, n );
        m_currImage = ( m_currImage + 1 ) % m_activeAverage;
        if( m_nprocessed < m_activeAverage )
            ++m_nprocessed;
        if( m_nprocessed < m_activeAverage )
            return 0;
        // Saturation preserves cadence when the output consumer is stalled.
        if( m_sinceUpdate < m_activeUpdate )
            ++m_sinceUpdate;
        if( m_sinceUpdate < m_activeUpdate || m_updated || m_inFlight )
            return 0;
        m_avgImage.setZero();
        for( unsigned plane = 0; plane < m_activeAverage; ++plane )
        {
            const realT *data = m_accumImages.image( plane ).data();
            for( size_t n = 0; n < m_source.m_pixels; ++n )
                m_avgImage.data()[n] += data[n];
        }
    }

    m_avgImage /= m_activeAverage;
    bool firstDark =
        m_dark.m_set && m_dark.m_info.m_width == m_source.m_width && m_dark.m_info.m_height == m_source.m_height;
    // Preserve the existing first-dark-only policy for moving averages.
    bool secondDark = m_activeUpdate == 0 && m_dark2.m_set && m_dark2.m_info.m_width == m_source.m_width &&
                      m_dark2.m_info.m_height == m_source.m_height;
    if( firstDark && secondDark )
        m_avgImage -= m_dark.m_image + m_dark2.m_image;
    else if( firstDark )
        m_avgImage -= m_dark.m_image;
    else if( secondDark )
        m_avgImage -= m_dark2.m_image;

    m_updated     = true;
    m_sinceUpdate = 0;
    m_imageCondition.notify_one();
    if( m_activeUpdate == 0 && !m_continuous )
    {
        m_running = false;
        if( m_fileSaver )
        {
            m_imageValid = !m_stateStringChanged && m_stateStringValid && m_stateStringValidOnStart;
            if( m_imageValid )
            {
                saved      = m_avgImage;
                savedState = m_stateString;
            }
        }
    }
    lock.unlock();
    if( saved.size() )
        return saveImage( saved, savedState );
    return 0;
}

inline int shmimIntegrator::allocateDark( darkFrame &dark, uint32_t width, uint32_t height, uint8_t type )
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    requestReconfigure();
    dark.m_failed  = true;
    dark.m_set     = false;
    dark.m_info    = {};
    imageInfo info = describeImage( width, height, type );
    if( !info.m_pixels )
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid dark geometry or datatype" } );
    try
    {
        imageT image( width, height );
        image.setZero();
        dark.m_image  = std::move( image );
        dark.m_info   = info;
        dark.m_failed = false;
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    return 0;
}

inline int shmimIntegrator::processDark( darkFrame &dark, void *source )
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( m_stopping )
        return 0;
    if( !source || !dark.m_info.m_pixels )
        return -1;
    for( size_t n = 0; n < dark.m_info.m_pixels; ++n )
        dark.m_image.data()[n] = dark.m_info.m_pixget( source, n );
    dark.m_set = true;
    return 0;
}

inline int shmimIntegrator::allocate( const darkShmimT &tag )
{
    static_cast<void>( tag );
    return allocateDark( m_dark, darkMonitorT::m_width, darkMonitorT::m_height, darkMonitorT::m_dataType );
}

inline int shmimIntegrator::processImage( void *source, const darkShmimT &tag )
{
    static_cast<void>( tag );
    return processDark( m_dark, source );
}

inline int shmimIntegrator::allocate( const dark2ShmimT &tag )
{
    static_cast<void>( tag );
    return allocateDark( m_dark2, dark2MonitorT::m_width, dark2MonitorT::m_height, dark2MonitorT::m_dataType );
}

inline int shmimIntegrator::processImage( void *source, const dark2ShmimT &tag )
{
    static_cast<void>( tag );
    return processDark( m_dark2, source );
}

inline int shmimIntegrator::findMatchingDark()
{
    std::string stateString;
    uint64_t    generation, revision;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        if( m_stopping || m_running || !m_fileSaver || !m_stateStringChanged )
            return 0;
        if( !m_stateStringValid )
        {
            m_imageValid         = false;
            m_stateStringChanged = false;
            return 0;
        }
        stateString = m_stateString;
        generation  = m_generation;
        revision    = m_stateRevision;
    }

    std::vector<std::string> names;
    mx::error_t              error = mx::ioutils::getFileNames( names, m_fileSaveDir, m_configName, "", ".fits" );
    if( error != mx::error_t::noerror )
        return log<text_log, -1>( std::string( "Could not list saved averages: " ) + mx::errorMessage( error ),
                                  logPrio::LOG_ERROR );

    // Filenames are sorted oldest first; keep the existing newest-match policy.
    for( auto name = names.rbegin(); name != names.rend(); ++name )
    {
        std::string stem  = mx::ioutils::pathStem( *name );
        size_t      begin = m_configName.size() + 1;
        size_t      end   = stem.find( "__T" );
        if( end == std::string::npos || end < begin + 2 || stem.substr( begin, end - begin ) != stateString )
            continue;
        imageT                    image;
        mx::fits::fitsFile<realT> file;
        if( file.read( image, *name ) != mx::error_t::noerror )
            return log<software_error, -1>( { __FILE__, __LINE__, "could not read saved average" } );
        return installSavedImage( image, generation, revision );
    }

    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( !m_running && m_generation == generation && m_stateRevision == revision )
    {
        m_imageValid         = false;
        m_stateStringChanged = false;
    }
    return 0;
}

inline int shmimIntegrator::installSavedImage( imageT &image, uint64_t generation, uint64_t revision )
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( m_stopping || m_running || m_sourceFailed || m_dark.m_failed || m_dark2.m_failed ||
        generation != m_generation || revision != m_stateRevision || !m_stateStringValid )
        return 0;
    if( image.rows() <= 0 || image.cols() <= 0 || image.rows() > std::numeric_limits<int>::max() / image.cols() )
        return -1;
    if( m_source.m_pixels && ( image.rows() != m_source.m_width || image.cols() != m_source.m_height ) )
    {
        m_imageValid         = false;
        m_stateStringChanged = true;
        return 0;
    }
    requestReconfigure();
    m_avgImage           = std::move( image );
    m_savedImage         = true;
    m_updated            = true;
    m_imageValid         = true;
    m_stateStringChanged = false;
    return 0;
}

inline int shmimIntegrator::saveImage( const imageT &image, const std::string &stateString )
{
    timespec timestamp;
    if( clock_gettime( CLOCK_REALTIME, &timestamp ) < 0 )
        return log<software_error, -1>( { __FILE__, __LINE__, errno, "clock_gettime" } );
    tm utc;
    if( !gmtime_r( &timestamp.tv_sec, &utc ) )
        return log<software_error, -1>( { __FILE__, __LINE__, errno, "gmtime_r" } );
    char timeString[32];
    if( strftime( timeString, sizeof( timeString ), "%Y%m%d%H%M%S", &utc ) == 0 )
        return -1;
    std::string name = m_fileSaveDir + "/" + m_configName + "_" + stateString + "__T" + timeString +
                       std::format( "{:09d}", timestamp.tv_nsec ) + ".fits";
    mx::fits::fitsFile<realT> file;
    if( file.write( name, image ) != mx::error_t::noerror )
        return log<software_error, -1>( { __FILE__, __LINE__, "could not write saved average" } );
    log<text_log>( "Wrote " + name );
    return 0;
}

inline int shmimIntegrator::configureAcquisition()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( m_stopping || m_claimed || m_sourceFailed || m_dark.m_failed || m_dark2.m_failed ||
        ( !m_source.m_pixels && !m_savedImage ) )
        return -1;
    if( !m_prepared )
    {
        unsigned average = m_nAverage;
        if( !mx::math::isFinite( m_avgTime ) || m_avgTime < 0 || !mx::math::isFinite( m_fps ) || m_fps < 0 )
            return -1;
        if( m_avgTime > 0 )
        {
            if( m_fps > 0 )
            {
                double count = double( m_avgTime ) * m_fps;
                if( count > std::numeric_limits<unsigned>::max() )
                    return -1;
                average = count < 1 ? 1 : static_cast<unsigned>( count );
            }
            else
                average = m_nAverageDefault;
        }
        if( average == 0 || ( m_nUpdate && m_source.m_pixels > size_t( std::numeric_limits<int>::max() ) / average ) )
            return log<software_error, -1>( { __FILE__, __LINE__, "invalid averaging buffer size" } );
        try
        {
            if( !m_savedImage )
            {
                mx::improc::eigenCube<realT> history;
                if( m_nUpdate )
                    history.resize( m_source.m_width, m_source.m_height, average );
                imageT image( m_source.m_width, m_source.m_height );
                image.setZero();
                // Commit only after every allocation succeeds.
                m_claimedImage.resize( image.rows(), image.cols() );
                m_accumImages = std::move( history );
                m_avgImage    = std::move( image );
            }
            else
                m_claimedImage.resize( m_avgImage.rows(), m_avgImage.cols() );
        }
        catch( const std::exception &error )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
        }
        m_nAverage      = average;
        m_activeAverage = average;
        m_activeUpdate  = m_nUpdate;
        m_prepared      = true;
    }
    m_configuredGeneration    = m_generation;
    frameGrabberT::m_width    = m_avgImage.rows();
    frameGrabberT::m_height   = m_avgImage.cols();
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;
    return 0;
}

inline float shmimIntegrator::fps()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( m_fps > 0 && m_nAverage > 0 && ( m_running || m_continuous ) )
        return m_fps / m_nAverage;
    return 1.0;
}

inline int shmimIntegrator::startAcquisition()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    if( m_stopping || !m_prepared || m_configuredGeneration != m_generation )
        return -1;
    m_activeGeneration = m_configuredGeneration;
    return 0;
}

inline int shmimIntegrator::acquireAndCheckValid()
{
    std::unique_lock<std::mutex> lock( m_imageMutex );
    m_imageCondition.wait_for( lock,
                               std::chrono::seconds( 1 ),
                               [this]() { return m_stopping || m_activeGeneration != m_generation || m_updated; } );
    if( m_stopping )
        return -1;
    if( m_activeGeneration != m_generation || !m_prepared )
    {
        // This base flag is written only by the output worker in this app.
        frameGrabberT::m_reconfig = true;
        return 1;
    }
    if( !m_updated || m_claimed )
        return 1;
    if( clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp ) < 0 )
        return -1;
    m_claimedImage = m_avgImage;
    m_claimed      = true;
    m_inFlight     = true;
    m_updated      = false;
    return 0;
}

inline int shmimIntegrator::loadImageIntoStream( void *destination )
{
    if( !destination || !m_claimed || m_claimedImage.rows() != frameGrabberT::m_width ||
        m_claimedImage.cols() != frameGrabberT::m_height || frameGrabberT::m_typeSize != sizeof( realT ) )
    {
        // The base sets write before calling us and does not clear it on failure.
        if( frameGrabberT::m_imageStream )
            frameGrabberT::m_imageStream->md->write = 0;
        std::lock_guard<std::mutex> lock( m_imageMutex );
        m_inFlight = false;
        m_claimed  = false;
        return -1;
    }
    memcpy( destination, m_claimedImage.data(), size_t( m_claimedImage.size() ) * sizeof( realT ) );
    std::lock_guard<std::mutex> lock( m_imageMutex );
    m_claimed  = false;
    m_inFlight = false;
    return 0;
}

inline int shmimIntegrator::reconfig()
{
    std::lock_guard<std::mutex> lock( m_imageMutex );
    m_activeGeneration = 0;
    return 0;
}

inline bool shmimIntegrator::readNumber( const pcf::IndiElement &element, double &value )
{
    std::istringstream input( element.get() );
    return ( input >> value ) && ( input >> std::ws ).eof() && mx::math::isFinite( value );
}

INDI_NEWCALLBACK_DEFN( shmimIntegrator, m_indiP_nAverage )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nAverage, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Number || !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
        return -1;
    double target;
    if( !readNumber( ipRecv[ipRecv.find( "target" ) ? "target" : "current"], target ) || target < 1 ||
        target > std::numeric_limits<unsigned>::max() || std::floor( target ) != target )
        return -1;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        m_nAverage = static_cast<unsigned>( target );
        if( m_avgTime > 0 && m_fps > 0 )
            m_avgTime = m_nAverage / m_fps;
        requestReconfigure();
    }
    updateIntegratorINDI();
    log<text_log>( "set nAverage to " + std::to_string( target ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( shmimIntegrator, m_indiP_avgTime )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_avgTime, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Number || !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
        return -1;
    double target;
    if( !readNumber( ipRecv[ipRecv.find( "target" ) ? "target" : "current"], target ) || target < 0 ||
        target > std::numeric_limits<float>::max() )
        return -1;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        m_avgTime = target;
        requestReconfigure();
    }
    updateIntegratorINDI();
    log<text_log>( "set avgTime to " + std::to_string( target ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( shmimIntegrator, m_indiP_nUpdate )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nUpdate, ipRecv );
    if( ipRecv.getType() != pcf::IndiProperty::Number || !( ipRecv.find( "target" ) || ipRecv.find( "current" ) ) )
        return -1;
    double target;
    if( !readNumber( ipRecv[ipRecv.find( "target" ) ? "target" : "current"], target ) || target < 0 ||
        target > std::numeric_limits<unsigned>::max() || std::floor( target ) != target )
        return -1;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        m_nUpdate = static_cast<unsigned>( target );
        requestReconfigure();
    }
    updateIntegratorINDI();
    log<text_log>( "set nUpdate to " + std::to_string( target ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( shmimIntegrator, m_indiP_startAveraging )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_startAveraging, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return 0;
    auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return 0;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        ++m_stateRevision;
        if( value == pcf::IndiElement::On )
        {
            if( !m_running )
                requestReconfigure();
            if( m_fileSaver && !m_continuous )
                m_stateStringChanged = false;
            m_stateStringValidOnStart = m_stateStringValid;
            m_running                 = true;
        }
        else
            m_running = false;
    }
    updateIntegratorINDI();
    return 0;
}

INDI_SETCALLBACK_DEFN( shmimIntegrator, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );
    if( !ipRecv.find( "current" ) )
        return 0;
    double value;
    if( !readNumber( ipRecv["current"], value ) || value < 0 || value > std::numeric_limits<float>::max() )
        return -1;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        if( static_cast<float>( value ) != m_fps )
        {
            m_fps = value;
            requestReconfigure();
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( shmimIntegrator, m_indiP_stateSource )( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stateSource, ipRecv );
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_imageMutex );
        bool                        changed = false;
        if( ipRecv.find( "valid" ) )
        {
            bool valid         = ipRecv["valid"].get<std::string>() == "yes";
            changed            = valid != m_stateStringValid;
            m_stateStringValid = valid;
        }
        if( ipRecv.find( "current" ) )
        {
            std::string value = ipRecv["current"].get<std::string>();
            changed           = changed || value != m_stateString;
            m_stateString     = value;
        }
        if( changed )
        {
            ++m_stateRevision;
            m_imageValid         = false;
            m_stateStringChanged = true;
            // An old saved image that has not been claimed must not be newly published.
            if( m_savedImage )
                requestReconfigure();
        }
    }
    updateIntegratorINDI();
    return 0;
}

inline int shmimIntegrator::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int shmimIntegrator::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // shmimIntegrator_hpp
