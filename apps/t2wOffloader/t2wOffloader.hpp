/** \file t2wOffloader.hpp
 * \brief The MagAO-X tweeter to woofer offloading manager
 *
 * \ingroup t2wOffloader_files
 */

#ifndef t2wOffloader_hpp
#define t2wOffloader_hpp

#include <algorithm>
#include <cmath>
#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/sigproc/gramSchmidt.hpp>
#include <mx/math/floatUtils.hpp>
#include <mx/math/templateBLAS.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup t2wOffloader Tweeter to Woofer Offloading
 * \brief Monitors the averaged tweeter shape, and sends it to the woofer.
 *
 * <a href="../handbook/operating/software/apps/t2wOffloader.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup t2wOffloader_files Tweeter to Woofer Offloading
 * \ingroup t2wOffloader
 */

/// A milkImage whose post sets the acquisition time to a supplied stamp instead of the write time.
/** milkImage::post() sets atime to the write time.  postAtime() keeps the given atime, so a reader can recover the time
 * of the event the frame describes.  Only for streams this process created (not passive).
 *
 * \ingroup t2wOffloader
 */
template <typename dataT>
class atimeMilkImage : public mx::improc::milkImage<dataT>
{
  public:
    /// Update the metadata with the given acquisition time and post all semaphores.
    /**
     * \throws mx::exception if the image is not open or ImageStreamIO_UpdateIm_atime fails
     */
    void postAtime( const timespec &atime /**< [in] the acquisition time to store in md->atime */ );
};

template <typename dataT>
void atimeMilkImage<dataT>::postAtime( const timespec &atime )
{
    if( this->m_image == nullptr )
    {
        throw( mx::exception( mx::error_t::invalidconfig, "Image is not open" ) );
    }

    timespec stamp = atime; // ImageStreamIO_UpdateIm_atime takes a non-const pointer

    if( ImageStreamIO_UpdateIm_atime( this->m_image, &stamp ) != IMAGESTREAMIO_SUCCESS )
    {
        throw( mx::exception( mx::error_t::liberr, "ImageStreamIO_UpdateIm_atime returned an error" ) );
    }
}

/** MagAO-X application to control offloading the tweeter to the woofer.
 *
 * \ingroup t2wOffloader
 *
 */
class t2wOffloader : public MagAOXApp<true>, public dev::shmimMonitor<t2wOffloader>, public dev::telemeter<t2wOffloader>
{

    // Give the test harness access.
    friend class t2wOffloader_test;

    friend class dev::shmimMonitor<t2wOffloader>;

    // The base shmimMonitor type
    typedef dev::shmimMonitor<t2wOffloader> shmimMonitorT;

    friend class dev::telemeter<t2wOffloader>;

    typedef dev::telemeter<t2wOffloader> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  protected:
    /** \name Configuration - Data
     *@{
     */

    std::string m_twRespMPath; ///< Path to the tweeter-to-woofer response matrix.

    std::string m_dmChannel; ///< Woofer command channel receiving the offload.

    std::string m_fpsSource{ "camwfs" };         ///< INDI device providing the input frame rate.
    std::string m_navgSource{ "dmtweeter-avg" }; ///< INDI device providing the number of averaged frames.
    float       m_gain{ 0.1 };                   ///< Gain applied to each new offload increment.
    float       m_leak{ 0.0 };                   ///< Fraction of the accumulated command leaked per update.

    float m_actLim{ 7.0 }; ///< the upper limit on woofer actuator commands.  default is 7.0.

    std::string m_tweeterModeFile; ///< File containing the tweeter modes to use for offloading
    std::string m_tweeterMaskFile; ///< File containing the tweeter pupil mask applied to the tweeter modes.

    std::string m_wooferMaskFile; ///< File containing the woofer actuator mask applied to the woofer modes.

    uint32_t m_maxModes{ 50 }; ///< Maximum number of modes available for modal offloading.

    uint32_t m_numModes{ 0 }; ///< Number of modes currently offloaded.  0 means actuator (zonal) offloading.

    int m_loopNumber{ 0 }; ///< The aol loop number used in shmim names.
    ///@}

    mx::improc::eigenImage<realT> m_twRespM; ///< Response matrix mapping a tweeter shape to a woofer command.

    /// The latest modal offload change back-projected onto the tweeter, published to m_tweeterOffload.
    mx::improc::eigenImage<realT> m_tweeter;

    mx::improc::eigenImage<realT> m_woofer; ///< The woofer command written to m_dmStream, held across offload toggles.

    mx::improc::eigenImage<realT> m_wooferDelta; ///< Woofer command change calculated for zonal offloading.

    mx::improc::eigenImage<realT> m_modeDeltaAmps; ///< Averaged tweeter shape projected onto m_tModesOrtho.

    /// Change in each modal offload command during the latest update, including any modevalDMf resync.
    /** Holds the previous m_modevalDM values until the change is calculated after the woofer is written.
     */
    mx::improc::eigenImage<realT> m_deltavalDM;

    mx::improc::milkImage<realT> m_modevalDM; ///< Stream aol0_modevalDM holding the current modal offload commands.

    mx::improc::milkImage<realT> m_modevalDMf; ///< Stream aol0_modevalDMf, written externally to resync m_modevalDM.

    /// Stream aol<loop>_tweeterOffload publishing m_tweeter after each modal offload.
    /** Its atime is the CLOCK_ISIO time of the woofer post the offload belongs to; its writetime is its own post.
     */
    atimeMilkImage<realT> m_tweeterOffload;

    uint64_t m_lastDMf_cnt0{ 0 }; ///< m_modevalDMf cnt0 at the last check, used to detect new resyncs.

    uint64_t m_updatedDMf{ 0 }; ///< Number of resyncs from m_modevalDMf, reported in appLogic().

    mx::improc::eigenImage<realT> m_tweeterMask; ///< Tweeter pupil mask applied to the tweeter modes.

    mx::improc::eigenCube<float> m_tModesOrtho; ///< Masked, unit-normalized tweeter modes used for offloading.

    mx::improc::milkImage<float> m_wooferMask; ///< Woofer actuator mask, also published as aol<loop>_dmmask.

    mx::improc::eigenCube<float> m_wModes; ///< Woofer modes corresponding to each of m_tModesOrtho.

    IMAGE *m_wModesStream{ nullptr }; ///< Stream aol<loop>_CMmodesDM publishing m_wModes.

    float    m_fps{ 0 };  ///< Current FPS from the FPS source.
    uint32_t m_navg{ 0 }; ///< Current navg from the averager

    /// INDI device of dmSyncopation, which changes the DM command rate.
    /** Fixed in code rather than configured, so the config stays compatible with versions without this feature.
     */
    const std::string m_syncoSource{ "dmtweeter-synco" };

    float m_syncoMainHz{ 0 }; ///< dmSyncopation's fitted main rate, from its timing.mainHz.

    uint32_t m_syncoSteps{ 0 }; ///< dmSyncopation steps actually running, from its timing.active.  0 when inactive.

    uint32_t m_navgRequested{ 0 };        ///< navg last sent to the averager and not yet reported back.  0 if none.
    double   m_navgApplied{ 0 };          ///< Frame count active in the averager's running generation.
    double   m_navgStride{ 0 };           ///< Frames between publications in the active averaging generation.
    double   m_syncoPrepareHz{ 0 };       ///< Future DM cadence to stage, or zero when no preparation is requested.
    double   m_syncoOffloadEpoch{ 0 };    ///< Restart identity of the pending preparation.
    double   m_syncoOffloadSequence{ 0 }; ///< Request identity within the synco instance.

    /// Requested offload rate in Hz, kept by setting the averager's navg.  0 = unmanaged.
    /** Set only through INDI (offloadRate), so the config stays compatible with versions without this feature.
     */
    float m_offloadRate{ 250 };

    float m_effFPS{ 0 }; ///< Effective offload rate, the DM command rate / m_navg.

    IMAGE    m_dmStream;      ///< The woofer DM channel stream, opened in allocate().
    uint32_t m_dmWidth{ 0 };  ///< The width of the image
    uint32_t m_dmHeight{ 0 }; ///< The height of the image.

    uint8_t m_dmDataType{ 0 }; ///< ImageStreamIO type code of the woofer command stream.
    size_t  m_dmTypeSize{ 0 }; ///< Size of a command pixel in bytes.

    bool m_dmOpened{ false };  ///< Whether the offload channel is open.
    bool m_dmRestart{ false }; ///< Woofer channel restart flag, cleared when reallocating.

    bool m_offloading{ false }; ///< Whether incoming tweeter frames drive the offload integrator.

    /// Mutex for locking shared memory access.
    std::mutex m_shmimMutex;

  public:
    /// Construct the tweeter-to-woofer offloader.
    t2wOffloader();

    /// Destroy the offloader.
    ~t2wOffloader() noexcept
    {
    }

    /** \name Configuration
     * @{
     */

    /// Register the application and base-class configuration options.
    virtual void setupConfig();

    /// Read the offloader settings from a configurator.
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] The application configuration. */ );

    /// Load the configuration.
    virtual void loadConfig();

    /// Startup function
    /** Registers INDI properties, prepares the modes, and starts the shmim monitor and telemeter.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int appStartup();

    /// Implementation of the FSM for t2wOffloader.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     * \returns 0 on success
     */
    virtual int appShutdown();

    /// Update the effective FPS, and the averager's navg if the offload rate is managed, after any rate input changes.
    /** The DM command rate feeding the averager is the camera fps, multiplied by dmSyncopation's active step
     * count while it is running.
     */
    int updateFPS();

    /// Check that the averager is running with the requested window and cadence.
    bool offloadPrepared( uint32_t navg /**< [in] Required frame count and publication stride. */ ) const;

    /// The DM command rate feeding the tweeter averager, in Hz.
    float dmRate() const;

    /// Calculate the averager navg that gives a requested offload rate.
    /**
     * \returns the nearest navg, at least 1
     * \returns 0 if the offload rate is unmanaged or the DM rate is unknown
     */
    static uint32_t navgForRate( float dmRateHz /**< [in] DM command rate feeding the averager, in Hz */,
                                 float rateHz /**< [in] requested offload rate in Hz, 0 for unmanaged */ );

    /// Allocate working memory and open or create the woofer and modal streams for the input shmim.
    /**
     * \returns 0 on success
     * \returns -1 if the woofer channel could not be opened
     */
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Offload the averaged tweeter shape to the woofer.
    /** In modal mode, also publishes the offload change as it appears on the tweeter to m_tweeterOffload, with the
     * CLOCK_ISIO time of the woofer post as its atime.
     *
     * \returns 0 always
     */
    int processImage( void              *curr_src, /**< [in] pointer to start of current frame.*/
                      const dev::shmimT &dummy     /**< [in] tag to differentiate shmimMonitor parents.*/
    );

    /// Zero the woofer command and the modal offload commands.
    /**
     * \returns 0 always
     */
    int zero();

    /// Read the mode, mask, and response matrix files, then calculate and publish the woofer modes.
    /**
     * \returns 0 on success
     * \returns -1 if the woofer modes stream could not be created
     */
    int prepareModes();

    /// Back-project the latest modal offload change onto the tweeter.
    /** Sets m_tweeter to the sum of m_deltavalDM(p) * m_tModesOrtho.image(p) over the offloaded modes.
     */
    void calcTweeterOffload();

  protected:
    /** \name INDI Interface
     *
     * @{
     */
    pcf::IndiProperty m_indiP_gain;   ///< Offload gain control.
    pcf::IndiProperty m_indiP_leak;   ///< Integrator leak control.
    pcf::IndiProperty m_indiP_actLim; ///< Woofer actuator limit control.

    pcf::IndiProperty m_indiP_zero; ///< Explicit request to zero the offloaded command.

    pcf::IndiProperty m_indiP_numModes; ///< Number of modes to offload.

    pcf::IndiProperty m_indiP_offloadToggle; ///< Offload switch with an optional link marker on incoming commands.

    /// Apply a new offload gain.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_gain );

    /// Apply a new integrator leak.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_leak );

    /// Apply a new woofer actuator limit.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_actLim );

    /// Zero the offloaded command on request.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_zero );

    /// Select the number of modes to offload.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_numModes );

    /// Start or stop offload updates and release the HO link on a separate state change.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_offloadToggle );

    pcf::IndiProperty m_indiP_fpsSource; ///< Property monitoring the fps of m_fpsSource.
    INDI_SETCALLBACK_DECL( t2wOffloader, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_navgSource; ///< Property monitoring the navg of m_navgSource.
    INDI_SETCALLBACK_DECL( t2wOffloader, m_indiP_navgSource );

    pcf::IndiProperty m_indiP_navgTarget; ///< Outbound property to request m_navgSource's nAverage.

    pcf::IndiProperty m_indiP_syncoSource; ///< Property monitoring the timing of m_syncoSource.
    INDI_SETCALLBACK_DECL( t2wOffloader, m_indiP_syncoSource );

    pcf::IndiProperty m_indiP_offloadReady; ///< Synco preparation identity and applied averaging confirmation.

    pcf::IndiProperty m_indiP_fps; ///< Read-only property reporting the effective offload rate.

    pcf::IndiProperty m_indiP_offloadRate; ///< Property to set the requested offload rate, Hz.
    INDI_NEWCALLBACK_DECL( t2wOffloader, m_indiP_offloadRate );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether offload or loop gain telemetry is due.
    int checkRecordTimes();

    /// Record the loop gain telemetry on demand.
    int recordTelem( const telem_loopgain *tag /**< [in] Telemetry type tag. */ );

    /// Record the offload state and integrator settings.
    int recordLoopGain( bool force = false /**< [in] Record even when the settings are unchanged. */ );

    /// Record the offloading telemetry on demand.
    int recordTelem( const telem_offloading *tag /**< [in] Telemetry type tag. */ );

    /// Record the mode count, averaging count, and effective offload rate.
    int recordOffloading( bool force = false /**< [in] Record even when the settings are unchanged. */ );

    ///@}
};

inline t2wOffloader::t2wOffloader() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void t2wOffloader::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );

    config.add( "integrator.fpsSource",
                "",
                "integrator.fpsSource",
                argType::Required,
                "integrator",
                "fpsSource",
                false,
                "string",
                "Device name for getting fps of the loop.  This device should have *.fps.current.  Default is camwfs" );

    config.add( "integrator.navgSource",
                "",
                "integrator.navgSource",
                argType::Required,
                "integrator",
                "navgSource",
                false,
                "string",
                "Device name for getting navg of tweeter-ave.  This device should have *.fps.current. Default is "
                "dmtweeter-avg." );

    config.add( "offload.respMPath",
                "",
                "offload.respMPath",
                argType::Required,
                "offload",
                "respMPath",
                false,
                "string",
                "The path to the response matrix." );

    config.add( "offload.channel",
                "",
                "offload.channel",
                argType::Required,
                "offload",
                "channel",
                false,
                "string",
                "The DM channel to offload to." );

    config.add( "offload.gain",
                "",
                "offload.gain",
                argType::Required,
                "offload",
                "gain",
                false,
                "float",
                "The starting offload gain.  Default is 0.1." );

    config.add( "offload.leak",
                "",
                "offload.leak",
                argType::Required,
                "offload",
                "leak",
                false,
                "float",
                "The starting offload leak.  Default is 0.0." );

    config.add( "offload.startupOffloading",
                "",
                "offload.startupOffloading",
                argType::Required,
                "offload",
                "startupOffloading",
                false,
                "bool",
                "Flag controlling whether offloading is on at startup.  Default is false." );

    config.add( "offload.actLim",
                "",
                "offload.actLim",
                argType::Required,
                "offload",
                "actLim",
                false,
                "float",
                "The woofer actuator command limit.  Default is 7.0." );

    config.add( "offload.tweeterModes",
                "",
                "offload.tweeterModes",
                argType::Required,
                "offload",
                "tweeterModes",
                false,
                "string",
                "File containing the tweeter modes to use for offloading" );

    config.add( "offload.tweeterMask",
                "",
                "offload.tweeterMask",
                argType::Required,
                "offload",
                "tweeterMask",
                false,
                "string",
                "File containing the tweeter mask." );

    config.add( "offload.wooferMask",
                "",
                "offload.wooferMask",
                argType::Required,
                "offload",
                "wooferMask",
                false,
                "string",
                "File containing the woofer mask." );

    config.add( "offload.maxModes",
                "",
                "offload.maxModes",
                argType::Required,
                "offload",
                "maxModes",
                false,
                "string",
                "Maximum number of modes for modal offloading." );

    config.add( "offload.numModes",
                "",
                "offload.numModes",
                argType::Required,
                "offload",
                "numModes",
                false,
                "string",
                "Number of modes to offload. 0 means use actuator offloading." );

    config.add( "offload.loopNumber",
                "",
                "offload.loopNUmber",
                argType::Required,
                "offload",
                "loopNumber",
                false,
                "string",
                "The aol loop number to use. Default is 0." );
}

inline int t2wOffloader::loadConfigImpl( mx::app::appConfigurator &_config )
{

    SHMIMMONITOR_LOAD_CONFIG( _config );

    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsSource, "integrator.fpsSource" );
    _config( m_navgSource, "integrator.navgSource" );

    _config( m_twRespMPath, "offload.respMPath" );
    _config( m_dmChannel, "offload.channel" );
    _config( m_gain, "offload.gain" );
    _config( m_leak, "offload.leak" );
    _config( m_actLim, "offload.actLim" );
    _config( m_tweeterModeFile, "offload.tweeterModes" );
    _config( m_tweeterMaskFile, "offload.tweeterMask" );
    _config( m_wooferMaskFile, "offload.wooferMask" );
    _config( m_maxModes, "offload.maxModes" );
    _config( m_numModes, "offload.numModes" );

    bool startupOffloading = false;

    if( _config.isSet( "offload.startupOffloading" ) )
    {
        _config( startupOffloading, "offload.startupOffloading" );
    }
    m_offloading = startupOffloading;

    return 0;
}

inline void t2wOffloader::loadConfig()
{
    loadConfigImpl( config );
}

inline int t2wOffloader::appStartup()
{

    createStandardIndiNumber<float>( m_indiP_gain, "gain", 0, 1, 0, "%0.2f" );
    m_indiP_gain["current"] = m_gain;
    m_indiP_gain["target"]  = m_gain;

    if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK( m_indiP_gain ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_leak, "leak", 0, 1, 0, "%0.2f" );
    m_indiP_leak["current"] = m_leak;
    m_indiP_leak["target"]  = m_leak;

    if( registerIndiPropertyNew( m_indiP_leak, INDI_NEWCALLBACK( m_indiP_leak ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_actLim, "actLim", 0, 8, 0, "%0.2f" );
    m_indiP_actLim["current"] = m_actLim;
    m_indiP_actLim["target"]  = m_actLim;

    if( registerIndiPropertyNew( m_indiP_actLim, INDI_NEWCALLBACK( m_indiP_actLim ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    if( prepareModes() < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    SHMIMMONITOR_APP_STARTUP;

    createStandardIndiRequestSw( m_indiP_zero, "zero", "zero loop" );
    if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK( m_indiP_zero ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    createStandardIndiNumber<int>( m_indiP_numModes, "numModes", 0, 97, 0, "%d" );
    m_indiP_numModes["current"] = m_numModes;
    m_indiP_numModes["target"]  = m_numModes;

    if( registerIndiPropertyNew( m_indiP_numModes, INDI_NEWCALLBACK( m_indiP_numModes ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_offloadToggle, "offload" );
    m_indiP_offloadToggle.setRule( pcf::IndiProperty::AnyOfMany );
    // Only commands from the HO link set this marker. It stays off in published state.
    m_indiP_offloadToggle.add( pcf::IndiElement( "linked_request", pcf::IndiElement::Off ) );

    if( registerIndiPropertyNew( m_indiP_offloadToggle, INDI_NEWCALLBACK( m_indiP_offloadToggle ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
    REG_INDI_SETPROP( m_indiP_navgSource, m_navgSource, std::string( "nAverage" ) );
    REG_INDI_SETPROP( m_indiP_syncoSource, m_syncoSource, std::string( "timing" ) );

    m_indiP_navgTarget = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_navgTarget.setDevice( m_navgSource );
    m_indiP_navgTarget.setName( "nAverage" );
    m_indiP_navgTarget.add( pcf::IndiElement( "target" ) );

    if( createROIndiNumber( m_indiP_offloadReady, "offloadReady" ) < 0 )
    {
        return -1;
    }

    for( const auto *name : { "epoch", "sequence", "frames" } )
    {
        m_indiP_offloadReady.add( pcf::IndiElement( name ) );
        m_indiP_offloadReady[name] = 0.0;
    }

    if( registerIndiPropertyReadOnly( m_indiP_offloadReady ) < 0 )
    {
        return -1;
    }

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_offloadRate, "offloadRate", 0, 10000, 0, "%0.1f" );
    m_indiP_offloadRate["current"] = m_offloadRate;
    m_indiP_offloadRate["target"]  = m_offloadRate;

    if( registerIndiPropertyNew( m_indiP_offloadRate, INDI_NEWCALLBACK( m_indiP_offloadRate ) ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    TELEMETER_APP_STARTUP;

    state( stateCodes::OPERATING );

    return 0;
}

inline int t2wOffloader::appLogic()
{
    static uint64_t lastUpdateDMf = 0;

    SHMIMMONITOR_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    if( lastUpdateDMf != m_updatedDMf )
    {
        std::cerr << std::format( "DMf updated {} times\n", m_updatedDMf - lastUpdateDMf );
        lastUpdateDMf = m_updatedDMf;
    }

    return 0;
}

inline int t2wOffloader::appShutdown()
{
    SHMIMMONITOR_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int t2wOffloader::updateFPS()
{
    float effFPS;

    if( m_navg < 1 )
    {
        effFPS = 0;
    }
    else
    {
        effFPS = dmRate() / m_navg;
    }

    if( effFPS != m_effFPS )
    {
        recordOffloading( true );
        m_effFPS = effFPS;
        recordOffloading();
    }

    updateIfChanged( m_indiP_fps, "current", m_effFPS );

    const auto targetRate = m_syncoPrepareHz > 0 ? m_syncoPrepareHz : dmRate();
    const auto navg       = navgForRate( targetRate, m_offloadRate );
    const bool ready      = offloadPrepared( navg );

    if( m_indiP_offloadReady.find( "frames" ) )
    {
        updateIfChanged<double>( m_indiP_offloadReady,
                                 { "epoch", "sequence", "frames" },
                                 { m_syncoOffloadEpoch, m_syncoOffloadSequence, ready ? double( navg ) : 0.0 },
                                 ready ? INDI_OK : INDI_IDLE );
    }

    // Keep only one request in flight. A pending rollback must be acknowledged before rearming synco.
    if( navg > 0 && navg != m_navg && m_navgRequested == 0 )
    {
        if( sendNewProperty( m_indiP_navgTarget, "target", navg ) == 0 )
        {
            m_navgRequested = navg;
            log<text_log>( std::format( "requested {}.nAverage = {} for {} Hz offloading at a {} Hz DM rate",
                                        m_navgSource,
                                        navg,
                                        m_offloadRate,
                                        targetRate ),
                           logPrio::LOG_NOTICE );
        }
    }

    return 0;
}

inline bool t2wOffloader::offloadPrepared( uint32_t navg ) const
{
    return m_syncoPrepareHz > 0 && navg > 0 && navg == m_navg && navg == m_navgApplied && navg == m_navgStride &&
           m_navgRequested == 0;
}

inline float t2wOffloader::dmRate() const
{
    if( m_syncoSteps > 0 && m_syncoMainHz > 0 )
    {
        return m_syncoSteps * m_syncoMainHz;
    }

    return m_fps;
}

inline uint32_t t2wOffloader::navgForRate( float dmRateHz, float rateHz )
{
    // mx::math::isFinite is bitwise, so it still rejects NaN/inf under the fast-math build flags.
    if( !mx::math::isFinite( rateHz ) || !mx::math::isFinite( dmRateHz ) || rateHz <= 0 || dmRateHz <= 0 )
    {
        return 0;
    }

    return std::max<uint32_t>( 1, static_cast<uint32_t>( std::lround( dmRateHz / rateHz ) ) );
}

inline int t2wOffloader::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    m_tweeter.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );

    if( m_dmOpened )
    {
        ImageStreamIO_closeIm( &m_dmStream );
    }

    m_dmOpened  = false;
    m_dmRestart = false; // Set this up front, since we're about to restart.

    if( ImageStreamIO_openIm( &m_dmStream, m_dmChannel.c_str() ) == 0 )
    {
        if( m_dmStream.md[0].sem < 10 )
        {
            ImageStreamIO_closeIm( &m_dmStream );
        }
        else
        {
            m_dmOpened = true;
        }
    }

    if( !m_dmOpened )
    {
        log<software_error>( { m_dmChannel + " not opened." } );
        return -1;
    }
    else
    {
        m_dmWidth  = m_dmStream.md->size[0];
        m_dmHeight = m_dmStream.md->size[1];

        m_dmDataType = m_dmStream.md->datatype;
        m_dmTypeSize = ImageStreamIO_typesize( m_dataType );

        log<text_log>( "Opened " + m_dmChannel + " " + std::to_string( m_dmWidth ) + " x " +
                       std::to_string( m_dmHeight ) + " with data type: " + std::to_string( m_dmDataType ) );

        m_woofer.resize( m_dmWidth, m_dmHeight );
        m_woofer.setZero();
    }

    m_modeDeltaAmps.resize( 1, m_tModesOrtho.planes() );

    m_modevalDM.create( "aol0_modevalDM", m_tModesOrtho.planes(), 1 );
    m_modevalDMf.create( "aol0_modevalDMf", m_tModesOrtho.planes(), 1 );

    m_lastDMf_cnt0 = m_modevalDMf.cnt0();

    m_deltavalDM.resize( m_tModesOrtho.planes(), 1 );
    m_deltavalDM.setZero();

    m_tweeterOffload.create(
        std::format( "aol{}_tweeterOffload", m_loopNumber ), shmimMonitorT::m_width, shmimMonitorT::m_height );

    ///\todo size checks here.

    return 0;
}

inline int t2wOffloader::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    if( !m_offloading )
    {
        return 0;
    }

    if( m_numModes == 0 )
    {
        m_wooferDelta =
            m_twRespM.matrix() * Eigen::Map<Eigen::Matrix<float, -1, -1>>( (float *)curr_src, m_width * m_height, 1 );

        std::lock_guard<std::mutex> guard( m_shmimMutex );

        size_t n = 0;
        while( m_dmStream.md[0].write == 1 && n < 10000 ) // Check if zero() is running
        {
            ++n;
            mx::sys::microSleep( 1 );
        }

        if( m_dmStream.md[0].write == 1 || n > 10000 - 1 )
        {
            log<software_warning>( { "timed out with write==1" } );
            return 0;
        }

        m_woofer = m_gain * Eigen::Map<Eigen::Array<float, -1, -1>>( m_wooferDelta.data(), m_dmWidth, m_dmHeight ) +
                   ( 1.0 - m_leak ) * m_woofer;

        for( int jj = 0; jj < m_woofer.cols(); ++jj )
        {
            for( int ii = 0; ii < m_woofer.rows(); ++ii )
            {
                float val = m_woofer( ii, jj );
                if( fabs( val ) > m_actLim )
                {
                    if( val > 0 )
                    {
                        m_woofer( ii, jj ) = m_actLim;
                    }
                    else
                    {
                        m_woofer( ii, jj ) = -m_actLim;
                    }
                }
            }
        }

        m_dmStream.md[0].write = 1;

        memcpy( m_dmStream.array.raw, m_woofer.data(), m_woofer.rows() * m_woofer.cols() * m_typeSize );

        m_dmStream.md[0].cnt0++;

        m_dmStream.md->write = 0;
        ImageStreamIO_sempost( &m_dmStream, -1 );
    }
    else // modal offloading
    {
        m_modeDeltaAmps = Eigen::Map<Eigen::Matrix<float, -1, -1>>( (float *)curr_src, 1, m_width * m_height ) *
                          Eigen::Map<Eigen::Matrix<float, -1, -1>>( m_tModesOrtho.data(),
                                                                    m_tModesOrtho.rows() * m_tModesOrtho.cols(),
                                                                    m_tModesOrtho.planes() );

        m_modevalDM.setWrite( true );

        // Save the commands the woofer currently has, before any modevalDMf resync, for the tweeter offload
        for( int p = 0; p < m_deltavalDM.rows(); ++p )
        {
            m_deltavalDM( p, 0 ) = m_modevalDM( p, 0 );
        }

        if( m_modevalDMf.cnt0() > m_lastDMf_cnt0 )
        {
            for( uint32_t p = 0; p < m_modevalDM.rows() && p < m_modevalDMf.rows(); ++p )
            {
                m_modevalDM( p, 0 ) = m_modevalDMf( p, 0 );
            }

            ++m_updatedDMf;
        }

        m_lastDMf_cnt0 = m_modevalDMf.cnt0();

        uint32_t p = 0;
        for( ; p < m_numModes && p < m_maxModes; ++p )
        {
            m_modevalDM( p, 0 ) = m_gain * m_modeDeltaAmps( 0, p ) + ( 1.0 - m_leak ) * m_modevalDM( p, 0 );
        }
        for( ; p < m_numModes; ++p )
        {
            m_modevalDM( p, 0 ) = 0;
        }

        m_modevalDM.post();

        std::lock_guard<std::mutex> guard( m_shmimMutex );

        size_t n = 0;
        while( m_dmStream.md[0].write == 1 && n < 10000 ) // Check if zero() is running
        {
            ++n;
            mx::sys::microSleep( 1 );
        }

        if( m_dmStream.md[0].write == 1 || n > 10000 - 1 )
        {
            log<software_warning>( { "timed out with write==1" } );
            return 0;
        }

        m_woofer = m_modevalDM( 0, 0 ) * m_wModes.image( 0 );
        for( uint32_t p = 1; p < m_numModes && p < m_maxModes; ++p )
        {
            m_woofer += m_modevalDM( p, 0 ) * m_wModes.image( p );
        }

        for( int jj = 0; jj < m_woofer.cols(); ++jj )
        {
            for( int ii = 0; ii < m_woofer.rows(); ++ii )
            {
                float val = m_woofer( ii, jj );
                if( fabs( val ) > m_actLim )
                {
                    if( val > 0 )
                    {
                        m_woofer( ii, jj ) = m_actLim;
                    }
                    else
                    {
                        m_woofer( ii, jj ) = -m_actLim;
                    }
                }
            }
        }

        m_dmStream.md[0].write = 1;

        memcpy( m_dmStream.array.raw, m_woofer.data(), m_woofer.rows() * m_woofer.cols() * m_typeSize );

        m_dmStream.md[0].cnt0++;

        m_dmStream.md->write = 0;
        ImageStreamIO_sempost( &m_dmStream, -1 );

        // Stamp the woofer post, carried to counterOffload as the tweeter offload's atime for latency measurement
        timespec wooferStamp;
        clock_gettime( CLOCK_ISIO, &wooferStamp );

        // Publish the offload as it appears on the tweeter, after the woofer so its timing is unaffected
        for( int p = 0; p < m_deltavalDM.rows(); ++p )
        {
            m_deltavalDM( p, 0 ) = m_modevalDM( p, 0 ) - m_deltavalDM( p, 0 );
        }

        calcTweeterOffload();

        m_tweeterOffload.setWrite( true );
        m_tweeterOffload() = m_tweeter;
        m_tweeterOffload.postAtime( wooferStamp );
    }

    return 0;
}

int t2wOffloader::zero()
{
    std::lock_guard<std::mutex> guard( m_shmimMutex );

    size_t n = 0;
    while( m_dmStream.md[0].write == 1 && n < 10000 ) // Check if processImage() is running
    {
        ++n;
        mx::sys::microSleep( 1 );
    }

    if( m_dmStream.md[0].write == 1 || n > 10000 - 1 )
    {
        log<software_warning>( { "timed out with write==1, processImage() might be stuck" } );
        return 0;
    }

    m_dmStream.md[0].write = 1;

    m_modevalDM.setWrite( true );
    m_modevalDM().setZero();
    m_modevalDM.post();

    m_woofer.setZero();

    memcpy( m_dmStream.array.raw, m_woofer.data(), m_woofer.rows() * m_woofer.cols() * m_typeSize );

    m_dmStream.md[0].cnt0++;

    m_dmStream.md->write = 0;
    ImageStreamIO_sempost( &m_dmStream, -1 );

    log<text_log>( "zeroed", logPrio::LOG_NOTICE );

    return 0;
}

int t2wOffloader::prepareModes()
{
    mx::improc::eigenCube<float> tmodes;

    mx::fits::fitsFile<float> ff;

    ff.read( tmodes, m_tweeterModeFile );

    ff.read( m_tweeterMask, m_tweeterMaskFile );

    ff.read( m_twRespM, m_twRespMPath );

    eigenImage<float> wm;
    ff.read( wm, m_wooferMaskFile );
    m_wooferMask.create( std::format( "aol{}_dmmask", m_loopNumber ), wm );

    for( int p = 0; p < tmodes.planes(); ++p )
    {
        tmodes.image( p ) *= m_tweeterMask;
        float norm = ( tmodes.image( p ) ).square().sum();
        tmodes.image( p ) /= sqrt( norm );
    }

    m_tModesOrtho.resize( tmodes.rows(), tmodes.cols(), m_maxModes );

    for( int p = 0; p < m_tModesOrtho.planes(); ++p )
    {
        m_tModesOrtho.image( p ) = tmodes.image( p );
    }

    ff.write( "/tmp/tModesOrtho.fits", m_tModesOrtho );

    m_wModes.resize( 11, 11, m_tModesOrtho.planes() );
    mx::improc::eigenImage<realT> win, wout;

    // win.resize( 11, 11 );
    wout.resize( 11, 11 );

    // Calculate the woofer modes corresponding to the tweeter modes
    for( int p = 0; p < m_tModesOrtho.planes(); ++p )
    {
        win = m_tModesOrtho.image( p );

        Eigen::Map<Eigen::Matrix<float, -1, -1>>( wout.data(), wout.rows() * wout.cols(), 1 ) =
            m_twRespM.matrix() * Eigen::Map<Eigen::Matrix<float, -1, -1>>( win.data(), win.rows() * win.cols(), 1 );

        m_wModes.image( p ) = wout * m_wooferMask();
    }

    ff.write( "/tmp/wModes.fits", m_wModes );

    m_wModesStream = new IMAGE;

    uint32_t imsize[3];
    imsize[0] = m_wModes.rows();
    imsize[1] = m_wModes.cols();
    imsize[2] = m_wModes.planes();

    std::string imname = std::format( "aol{}_CMmodesDM", m_loopNumber );
    errno_t     rv     = ImageStreamIO_createIm_gpu( m_wModesStream,
                                             imname.c_str(),
                                             3,
                                             imsize,
                                             _DATATYPE_FLOAT,
                                             -1,
                                             1,
                                             IMAGE_NB_SEMAPHORE,
                                             0,
                                             CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                             0 );

    if( rv != IMAGESTREAMIO_SUCCESS )
    {
        delete m_wModesStream;
        m_wModesStream = nullptr;
        return log<software_error, -1>( "failed to create " + imname );
    }

    memcpy( m_wModesStream->array.raw,
            m_wModes.data(),
            m_wModes.rows() * m_wModes.cols() * m_wModes.planes() * sizeof( float ) );

    return 0;
}

inline void t2wOffloader::calcTweeterOffload()
{
    m_tweeter.setZero();

    for( uint32_t p = 0; p < m_numModes && p < m_maxModes && p < m_deltavalDM.rows(); ++p )
    {
        m_tweeter += m_deltavalDM( p, 0 ) * m_tModesOrtho.image( p );
    }
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_gain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    recordLoopGain( true );

    m_gain = target;

    recordLoopGain();

    updateIfChanged( m_indiP_gain, "current", m_gain );
    updateIfChanged( m_indiP_gain, "target", m_gain );

    log<text_log>( "set gain to " + std::to_string( m_gain ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_leak )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_leak, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_leak, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    recordLoopGain( true );
    m_leak = target;
    recordLoopGain();

    updateIfChanged( m_indiP_leak, "current", m_leak );
    updateIfChanged( m_indiP_leak, "target", m_leak );

    log<text_log>( "set leak to " + std::to_string( m_leak ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_actLim )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_actLim, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_actLim, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    recordLoopGain( true );

    m_actLim = target;

    recordLoopGain();

    updateIfChanged( m_indiP_actLim, "current", m_actLim );
    updateIfChanged( m_indiP_actLim, "target", m_actLim );

    log<text_log>( "set actuator limit to " + std::to_string( m_actLim ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_zero )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_zero, ipRecv );

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        return zero();
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_numModes )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numModes, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_numModes, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    recordOffloading( true );
    m_numModes = target;
    recordOffloading();

    if( m_numModes > m_maxModes )
    {
        log<text_log>( std::format( "maximum number of offloadings modes is {}", m_maxModes ), logPrio::LOG_WARNING );
        m_numModes = m_maxModes;
    }

    updateIfChanged( m_indiP_numModes, "current", m_numModes );
    updateIfChanged( m_indiP_numModes, "target", m_numModes );

    log<text_log>( "set number of modes to " + std::to_string( m_numModes ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_offloadToggle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadToggle, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    auto toggle = ipRecv["toggle"].getSwitchState();

    if( toggle != pcf::IndiElement::On && toggle != pcf::IndiElement::Off )
    {
        return 0;
    }

    bool enabled = toggle == pcf::IndiElement::On;

    if( enabled == m_offloading )
    {
        return 0;
    }

    recordLoopGain( true );
    m_offloading = enabled;
    recordLoopGain();

    bool linked = ipRecv.find( "linked_request" ) && ipRecv["linked_request"].getSwitchState() == pcf::IndiElement::On;

    if( !linked )
    {
        pcf::IndiProperty overrideLink( pcf::IndiProperty::Switch );
        overrideLink.setDevice( "holoop" );
        overrideLink.setName( "offload_override" );
        overrideLink.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
        sendNewProperty( overrideLink );
    }

    log<text_log>( enabled ? "started offloading" : "stopped offloading", logPrio::LOG_NOTICE );
    updateSwitchIfChanged( m_indiP_offloadToggle, "toggle", toggle, enabled ? INDI_OK : INDI_IDLE );

    return 0;
}

INDI_SETCALLBACK_DEFN( t2wOffloader, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( "current" ) != true ) // this isn't valie
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        updateFPS();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( t2wOffloader, m_indiP_navgSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_navgSource, ipRecv );

    if( ipRecv.find( "current" ) != true ) // this isn't valie
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    const auto navg    = ipRecv["current"].get<double>();
    const auto applied = ipRecv.find( "applied" ) ? ipRecv["applied"].get<double>() : 0.0;
    const auto stride  = ipRecv.find( "stride" ) ? ipRecv["stride"].get<double>() : 0.0;

    if( !mx::math::isFinite( navg ) || navg < 1 || navg > std::numeric_limits<uint32_t>::max() ||
        std::floor( navg ) != navg )
    {
        return 0;
    }

    if( navg != m_navg || applied != m_navgApplied || stride != m_navgStride || navg == m_navgRequested )
    {
        m_navg        = static_cast<uint32_t>( navg );
        m_navgApplied = applied;
        m_navgStride  = stride;

        if( navg == m_navgRequested )
        {
            m_navgRequested = 0;
        }

        updateFPS();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( t2wOffloader, m_indiP_syncoSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_syncoSource, ipRecv );

    if( !ipRecv.find( "active" ) || !ipRecv.find( "mainHz" ) )
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    float      mainHz   = ipRecv["mainHz"].get<float>();
    uint32_t   steps    = static_cast<uint32_t>( std::max( 0.0f, ipRecv["active"].get<float>() ) );
    const auto rate     = ipRecv.find( "offloadPrepareHz" ) ? ipRecv["offloadPrepareHz"].get<double>() : 0.0;
    const auto epoch    = ipRecv.find( "offloadEpoch" ) ? ipRecv["offloadEpoch"].get<double>() : 0.0;
    const auto sequence = ipRecv.find( "offloadSequence" ) ? ipRecv["offloadSequence"].get<double>() : 0.0;
    const bool prepare  = mx::math::isFinite( rate ) && rate > 0 && mx::math::isFinite( epoch ) && epoch > 0 &&
                         mx::math::isFinite( sequence ) && sequence > 0;
    const auto prepareHz       = prepare ? rate : 0.0;
    const auto prepareEpoch    = prepare ? epoch : 0.0;
    const auto prepareSequence = prepare ? sequence : 0.0;

    if( steps != m_syncoSteps || ( steps > 0 && mainHz != m_syncoMainHz ) || prepareHz != m_syncoPrepareHz ||
        prepareEpoch != m_syncoOffloadEpoch || prepareSequence != m_syncoOffloadSequence )
    {
        m_syncoSteps           = steps;
        m_syncoMainHz          = mainHz;
        m_syncoPrepareHz       = prepareHz;
        m_syncoOffloadEpoch    = prepareEpoch;
        m_syncoOffloadSequence = prepareSequence;
        updateFPS();
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( t2wOffloader, m_indiP_offloadRate )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadRate, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_offloadRate, target, ipRecv, true ) < 0 || !mx::math::isFinite( target ) ||
        target < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid offloadRate" } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_offloadRate = target;

    updateIfChanged( m_indiP_offloadRate, "current", m_offloadRate );
    updateIfChanged( m_indiP_offloadRate, "target", m_offloadRate );

    log<text_log>( "set offload rate to " + std::to_string( m_offloadRate ) + " Hz", logPrio::LOG_NOTICE );

    updateFPS();

    return 0;
}

int t2wOffloader::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_loopgain(), telem_offloading() );
}

int t2wOffloader::recordTelem( const telem_loopgain * )
{
    return recordLoopGain( true );
}

int t2wOffloader::recordLoopGain( bool force )
{
    static uint8_t state{ 0 };
    static float   gain{ -1000 };
    static float   leak{ 0 };
    static float   limit{ 0 };

    if( state != m_offloading || gain != m_gain || leak != m_leak || limit != m_actLim || force )
    {
        state = m_offloading;
        gain  = m_gain;
        leak  = m_leak;
        limit = m_actLim;

        telem<telem_loopgain>( { state, m_gain, 1 - leak, limit } );
    }

    return 0;
}

int t2wOffloader::recordTelem( const telem_offloading * )
{
    return recordOffloading( true );
}

int t2wOffloader::recordOffloading( bool force )
{
    static uint32_t num_modes{ 0 };
    static uint32_t num_average{ 0 };
    float           fps{ 0 };

    if( num_modes != m_numModes || num_average != m_navg || fps != m_effFPS || force )
    {
        num_modes   = m_numModes;
        num_average = m_navg;
        fps         = m_effFPS;

        telem<telem_offloading>( { num_modes, num_average, fps } );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // t2wOffloader_hpp
