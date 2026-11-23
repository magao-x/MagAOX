/** \file t2wOffloader.hpp
 * \brief The MagAO-X tweeter to woofer offloading manager
 *
 * \ingroup t2wOffloader_files
 */

#ifndef t2wOffloader_hpp
#define t2wOffloader_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/sigproc/gramSchmidt.hpp>
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
    std::string m_tweeterMaskFile; ///< File containing the tweeter actuator mask.

    std::string m_wooferMaskFile; ///< File containing the woofer actuator mask.

    uint32_t m_maxModes{ 50 }; ///< Maximum number of modes supported by the offloader.

    uint32_t m_numModes{ 0 }; ///< Number of modes to offload, with zero selecting actuator offloading.

    int m_loopNumber{ 0 }; ///< CACAO loop number used to name the woofer mode and mask streams.
    ///@}

    mx::improc::eigenImage<realT> m_twRespM;       ///< Response matrix mapping tweeter commands to woofer commands.
    mx::improc::eigenImage<realT> m_tweeter;       ///< Buffer sized to the tweeter input image.
    mx::improc::eigenImage<realT> m_woofer;        ///< Accumulated actuator command, held across offload toggles.
    mx::improc::eigenImage<realT> m_wooferDelta;   ///< Woofer actuator increment for the current frame.
    mx::improc::eigenImage<realT> m_modeDeltaAmps; ///< Modal increments for the current frame.

    mx::improc::milkImage<realT> m_modevalDM;  ///< Shared stream holding the accumulated modal command.
    mx::improc::milkImage<realT> m_modevalDMf; ///< Shared stream supplying reconstructed woofer modal feedback.

    uint64_t m_lastDMf_cnt0{ 0 }; ///< Feedback stream counter last observed by the modal integrator.
    uint64_t m_updatedDMf{ 0 };   ///< Number of feedback updates incorporated by the modal integrator.

    mx::improc::eigenImage<realT> m_tweeterMask; ///< Mask used to select valid tweeter actuators.

    mx::improc::eigenCube<float> m_tModesOrtho; ///< Orthonormal tweeter modes used for projection.

    mx::improc::milkImage<float> m_wooferMask; ///< Published mask of valid woofer actuators.

    mx::improc::eigenCube<float> m_wModes; ///< Woofer shapes corresponding to the tweeter modes.

    IMAGE *m_wModesStream{ nullptr }; ///< Published stream of woofer mode shapes.

    float    m_fps{ 0 };  ///< Current FPS from the FPS source.
    uint32_t m_navg{ 0 }; ///< Current navg from the averager

    float m_effFPS{ 0 }; ///< Effective offload update rate after frame averaging.

    IMAGE    m_dmStream;      ///< Open woofer command stream.
    uint32_t m_dmWidth{ 0 };  ///< Width of the woofer command image.
    uint32_t m_dmHeight{ 0 }; ///< Height of the woofer command image.

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

    /// Load the configuration into the application.
    virtual void loadConfig();

    ///@}

    /// Prepare the mode basis and start the stream monitors, INDI properties, and telemetry.
    virtual int appStartup();

    /// Implementation of the FSM for t2wOffloader.
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

    /// Update the effective offload rate after a frame rate or averaging change.
    int updateFPS();

    /// Allocate workspace and connect the woofer command channel for the current tweeter stream.
    int allocate( const dev::shmimT &dummy /**< [in] Stream monitor tag. */ );

    /// Integrate an averaged tweeter frame into the woofer command when offloading is active.
    int processImage( void              *curr_src /**< [in] Start of the current tweeter frame. */,
                      const dev::shmimT &dummy /**< [in] Stream monitor tag. */ );

    /// Clear the integrator and publish a zero woofer command.
    int zero();

    /// Load the offload calibration and prepare the tweeter and woofer mode bases.
    int prepareModes();

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

    pcf::IndiProperty m_indiP_fpsSource; ///< Subscription to the input frame rate.

    /// Update the input frame rate from its INDI source.
    INDI_SETCALLBACK_DECL( t2wOffloader, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_navgSource; ///< Subscription to the input averaging count.

    /// Update the averaging count from its INDI source.
    INDI_SETCALLBACK_DECL( t2wOffloader, m_indiP_navgSource );

    pcf::IndiProperty m_indiP_fps; ///< Published effective offload rate.

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

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
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

    if(lastUpdateDMf != m_updatedDMf)
    {
        std::cerr << std::format("DMf updated {} times\n", m_updatedDMf - lastUpdateDMf);
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
        effFPS = m_fps / m_navg;

    if( effFPS != m_effFPS )
    {
        recordOffloading( true );
        m_effFPS = effFPS;
        recordOffloading();
    }

    updateIfChanged( m_indiP_fps, "current", m_effFPS );

    return 0;
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

        if( m_modevalDMf.cnt0() > m_lastDMf_cnt0 )
        {
            for( uint32_t p = 0; p < m_modevalDM.rows() && p < m_modevalDMf.rows(); ++p )
            {
                m_modevalDM( p, 0 ) = m_modevalDMf(p,0);
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

    realT navg = ipRecv["current"].get<float>();

    if( navg != m_navg )
    {
        m_navg = navg;
        updateFPS();
    }

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
