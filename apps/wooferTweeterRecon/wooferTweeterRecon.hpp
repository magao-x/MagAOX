/** \file wooferTweeterRecon.hpp
 * \brief The MagAO-X woofer-tweeter pseudo-open-loop reconstructor
 *
 * \ingroup wooferTweeterRecon_files
 */

#ifndef wooferTweeterRecon_hpp
#define wooferTweeterRecon_hpp

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/sigproc/gramSchmidt.hpp>
#include <mx/math/templateBLAS.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup wooferTweeterRecon Woofer Tweeter Pseudo-Open-Loop Reconstructor
 * \brief Reconstruct the open-loop wavefront from the woofer and tweeter surfaces
 *
 * Reconstructs the tweeter shape corresponding to the woofer shape, and combines the woofer and tweeter shapes
 * and the measured delta.
 *
 * <a href="../handbook/operating/software/apps/wooferTweeterRecon.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup wooferTweeterRecon_files Woofer Tweeter Pseudo-Open-Loop Reconstructor Files
 * \ingroup wooferTweeterRecon
 */

struct wooferModesShmimT
{
    static std::string configSection()
    {
        return "wooferModes";
    };

    static std::string indiPrefix()
    {
        return "wooferModes";
    };
};

struct tweeterModesShmimT
{
    static std::string configSection()
    {
        return "tweeterModes";
    };

    static std::string indiPrefix()
    {
        return "tweeterModes";
    };
};

struct wfsModesShmimT
{
    static std::string configSection()
    {
        return "wfsModes";
    };

    static std::string indiPrefix()
    {
        return "wfsModes";
    };
};

/** MagAO-X application to perform pseudo-open-loop reconstruction of an offloading woofer-tweeter system
 *
 * \ingroup wooferTweeterRecon
 *
 */
class wooferTweeterRecon : public MagAOXApp<true>,
                           public dev::shmimMonitor<wooferTweeterRecon, wooferModesShmimT>,
                           public dev::shmimMonitor<wooferTweeterRecon, tweeterModesShmimT>,
                           public dev::shmimMonitor<wooferTweeterRecon, wfsModesShmimT> //,
// public dev::frameGrabber<wooferTweeterRecon>,
// public dev::telemeter<wooferTweeterRecon>
{
    // Give the test harness access.
    friend class wooferTweeterRecon_test;

    friend class dev::shmimMonitor<wooferTweeterRecon, wooferModesShmimT>;
    typedef dev::shmimMonitor<wooferTweeterRecon, wooferModesShmimT> wooferModesSMT;

    friend class dev::shmimMonitor<wooferTweeterRecon, tweeterModesShmimT>;
    typedef dev::shmimMonitor<wooferTweeterRecon, tweeterModesShmimT> tweeterModesSMT;

    friend class dev::shmimMonitor<wooferTweeterRecon, wfsModesShmimT>;
    typedef dev::shmimMonitor<wooferTweeterRecon, wfsModesShmimT> wfsModesSMT;

    // friend class dev::telemeter<wooferTweeterRecon>;

    // typedef dev::telemeter<wooferTweeterRecon> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    /// Device name providing the loop frame rate.
    std::string m_fpsSource{ "camwfs" };

    /// Device name providing the telescope elevation.
    std::string m_elSource{ "tcsi" };

    /// Number of samples retained in each mode-value circular buffer.
    uint32_t m_modevalCircBuffLen{ 5000 };

    /// Time offset applied to woofer command timestamps.
    double m_wooferOffset{ 500e-6 };

    /// Time offset applied to tweeter command timestamps.
    double m_tweeterOffset{ 50e-6 };

    /// Time offset applied to WFS timestamps after the loop-latency correction.
    double m_wfsOffset{ -10e-6 };

    ///@}

    /** \name Reconstruction State - Data
     *
     * @{
     */
    /// True once the woofer mode ring buffer is sized for the current stream.
    bool m_wooferModesReady{ false };

    /// True once the tweeter mode ring buffer and seeing buffers are sized for the current stream.
    bool m_tweeterModesReady{ false };

    /// True once the WFS mode ring buffer is sized for the current stream.
    bool m_wfsModesReady{ false };

    /// Stores one timestamped modal sample from a monitored stream.
    struct modevals
    {
        /// Absolute sample time used to align the stream with the other inputs.
        double t{ 0 };

        /// Modal amplitudes copied from the current frame.
        std::vector<float> vals;

        /// Tracks whether the aligned WFS sample has already been reconstructed.
        bool reconstructed{ false };
    };

    /// Guards stream buffers, readiness flags, timestamps, and seeing history.
    mutable std::mutex m_reconMutex;

    /// Circular buffer of woofer mode samples keyed by timestamp.
    std::vector<modevals> m_wooferVals;

    /// Index of the most recently written woofer sample.
    size_t m_lastWooferVal{ 0 };

    /// Circular buffer of tweeter mode samples keyed by timestamp.
    std::vector<modevals> m_tweeterVals;

    /// Index of the most recently written tweeter sample.
    size_t m_lastTweeterVal{ 0 };

    /// Circular buffer of WFS mode samples keyed by timestamp.
    std::vector<modevals> m_wfsVals;

    /// Index of the most recently written WFS sample.
    size_t m_lastWfsVal{ 0 };

    /// Current FPS from the FPS source.
    float m_fps{ 0 };

    /// Cached inverse FPS used to align the WFS timestamp with the command streams.
    float m_invFps{ 0 };

    /// Current telescope elevation in degrees.
    float m_el{ 90 };

    /// Optical gain used to convert residual WFS modes to physical modal amplitudes.
    float m_opticalGain{ 0.8 };

    /// Reconstructed pseudo-open-loop modal vectors awaiting downstream consumption.
    mx::improc::eigenImage<float> m_outputVal;

    /// Number of reconstructed modal vectors retained in the output image.
    int m_nvals{ 3600 * 2 };

    /// Number of reconstructed modal vectors currently loaded in the output image.
    int m_nloaded{ 0 };

    /// Number of seeing samples retained in the circular statistics buffers.
    size_t m_seeingCircBuffLen{ 3600 * 30 };

    /// Circular buffer of reconstructed r0 estimates.
    std::vector<float> m_r0;

    /// Circular buffer of residual variances used for future seeing products.
    std::vector<float> m_sig;

    /// Index of the most recently stored seeing estimate.
    size_t m_lastr0{ 0 };

    /// Number of valid seeing samples currently stored in the circular buffers.
    size_t m_r0Count{ 0 };

    ///@}

  public:
    /// Default c'tor.
    wooferTweeterRecon();

    /// D'tor, declared and defined for noexcept.
    ~wooferTweeterRecon() noexcept
    {
    }

    /// Configure application and shmimMonitor parameters.
    virtual void setupConfig();

    /// Load configuration values into the application state.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                        from which to load values*/
    );

    /// Load configuration values from the application configurator.
    virtual void loadConfig();

    /// Register INDI properties and start the shmimMonitor threads.
    virtual int appStartup();

    /// Implementation of the FSM for wooferTweeterRecon.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shut down the shmimMonitor threads.
    virtual int appShutdown();

    /// Prepare storage for a newly connected woofer mode stream.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const wooferModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Copy one woofer mode frame into the circular buffer.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int processImage( void *curr_src,           /**< [in] pointer to start of current frame. */
                      const wooferModesShmimT & /**< [in] tag to differentiate shmimMonitor parents. */
    );

    /// Reconstruct pseudo-open-loop modal vectors for newly aligned samples.
    int recon();

    /// Prepare storage for a newly connected tweeter mode stream.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const tweeterModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Copy one tweeter mode frame into the circular buffer.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int processImage( void *curr_src,            /**< [in] pointer to start of current frame. */
                      const tweeterModesShmimT & /**< [in] tag to differentiate shmimMonitor parents. */
    );

    /// Prepare storage for a newly connected WFS mode stream.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const wfsModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Copy one WFS mode frame into the circular buffer.
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int processImage( void *curr_src,        /**< [in] pointer to start of current frame. */
                      const wfsModesShmimT & /**< [in] tag to differentiate shmimMonitor parents. */
    );

    /// Reserved for future mode-preparation helpers.
    int prepareModes();

  private:
    /// Reset one timestamped mode-value circular buffer to the current mode count.
    void resetModevalBuffer( std::vector<modevals> &buffer /**< [in,out] buffer to resize and clear */,
                             size_t &lastVal /**< [in,out] index of the most recent valid sample in the buffer */,
                             size_t  modeCount /**< [in] number of modes to store in each entry */ );

    /// Reset the circular seeing buffers after a tweeter-stream restart.
    void resetSeeingState();

    /// Copy the seeing samples needed for INDI updates into local buffers.
    bool snapshotSeeing( std::vector<float> &oneSecond /**< [out] recent samples for the short averaging window */,
                         std::vector<float> &tenSecond /**< [out] recent samples for the long averaging window */,
                         float              &fps /**< [out] current loop rate in Hz */,
                         float              &el /**< [out] current telescope elevation in degrees */ );

  protected:
    /** \name INDI Interface
     *
     * @{
     */
    /// Subscription property used to receive loop-FPS updates.
    pcf::IndiProperty m_indiP_fpsSource;

    /// Handle updates from the configured FPS source.
    INDI_SETCALLBACK_DECL( wooferTweeterRecon, m_indiP_fpsSource );

    /// Published current loop FPS.
    pcf::IndiProperty m_indiP_fps;

    /// Subscription property used to receive telescope elevation updates.
    pcf::IndiProperty m_indiP_elSource;

    /// Handle updates from the configured elevation source.
    INDI_SETCALLBACK_DECL( wooferTweeterRecon, m_indiP_elSource );

    /// Published seeing summary derived from reconstructed pseudo-open-loop modes.
    pcf::IndiProperty m_indiP_seeing;

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether loop-gain and offloading telemetry should be recorded.
    int checkRecordTimes();

    /// Record loop-gain telemetry when the helper dispatches this telemetry type.
    int recordTelem( const telem_loopgain * /**< [in] telemetry tag used by the telemeter helper */ );

    /// Publish the current loop-gain telemetry packet when needed.
    int recordLoopGain( bool force = false /**< [in] force emission even when the state is unchanged */ );

    /// Record offloading telemetry when the helper dispatches this telemetry type.
    int recordTelem( const telem_offloading * /**< [in] telemetry tag used by the telemeter helper */ );

    /// Publish the current offloading telemetry packet when needed.
    int recordOffloading( bool force = false /**< [in] force emission even when the state is unchanged */ );

    ///@}
};

inline wooferTweeterRecon::wooferTweeterRecon() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void wooferTweeterRecon::setupConfig()
{

    SHMIMMONITORT_SETUP_CONFIG( wooferModesSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( tweeterModesSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( wfsModesSMT, config );

    // TELEMETER_SETUP_CONFIG( config );

    config.add( "integrator.fpsSource",
                "",
                "integrator.fpsSource",
                argType::Required,
                "integrator",
                "fpsSource",
                false,
                "string",
                "Device name for getting fps of the loop.  This device should have *.fps.current.  Default is camwfs" );

    config.add( "woofer.offset",
                "",
                "woofer.offset",
                argType::Required,
                "woofer",
                "offset",
                false,
                "float",
                "Offset, in seconds, for the woofer command from its write time" );

    config.add( "tweeter.offset",
                "",
                "tweeter.offset",
                argType::Required,
                "tweeter",
                "offset",
                false,
                "float",
                "Offset, in seconds, for the tweeter command from its write time" );

    config.add( "wfs.offset",
                "",
                "wfs.offset",
                argType::Required,
                "wfs",
                "offset",
                false,
                "float",
                "Offset, in seconds, for the wfs from its acquisition time and 1/fps" );
}

inline int wooferTweeterRecon::loadConfigImpl( mx::app::appConfigurator &_config )
{

    wooferModesSMT::m_shmimName        = "aol0_modevalDMf_mon";
    wooferModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( wooferModesSMT, _config );

    tweeterModesSMT::m_shmimName        = "aol1_modevalDMf_mon";
    tweeterModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( tweeterModesSMT, _config );

    wfsModesSMT::m_shmimName        = "aol1_modevalWFS";
    wfsModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( wfsModesSMT, _config );

    // TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsSource, "integrator.fpsSource" );

    _config( m_wooferOffset, "woofer.offset" );
    _config( m_tweeterOffset, "tweeter.offset" );
    _config( m_wfsOffset, "wfs.offset" );

    return 0;
}

inline void wooferTweeterRecon::loadConfig()
{
    loadConfigImpl( config );
}

inline int wooferTweeterRecon::appStartup()
{

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    REG_INDI_SETPROP( m_indiP_elSource, m_elSource, std::string( "telpos" ) );

    createROIndiNumber( m_indiP_seeing, "seeing" );
    m_indiP_seeing.add( pcf::IndiElement( "r0_1sec" ) );
    m_indiP_seeing.add( pcf::IndiElement( "r0_1sec_std" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_1sec" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_1sec_std" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_1sec_zenith" ) );
    m_indiP_seeing.add( pcf::IndiElement( "r0_10sec" ) );
    m_indiP_seeing.add( pcf::IndiElement( "r0_10sec_std" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_10sec" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_10sec_std" ) );
    m_indiP_seeing.add( pcf::IndiElement( "fwhm_10sec_zenith" ) );

    if( registerIndiPropertyReadOnly( m_indiP_seeing ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    SHMIMMONITORT_APP_STARTUP( wooferModesSMT );
    SHMIMMONITORT_APP_STARTUP( tweeterModesSMT );
    SHMIMMONITORT_APP_STARTUP( wfsModesSMT );

    // TELEMETER_APP_STARTUP;

    state( stateCodes::OPERATING );

    return 0;
}

inline void wooferTweeterRecon::resetModevalBuffer( std::vector<modevals> &buffer, size_t &lastVal, size_t modeCount )
{
    lastVal = 0;
    buffer.resize( m_modevalCircBuffLen );

    for( auto &val : buffer )
    {
        val.t = 0;
        val.vals.assign( modeCount, 0 );
        val.reconstructed = false;
    }
}

inline void wooferTweeterRecon::resetSeeingState()
{
    m_lastr0  = 0;
    m_r0Count = 0;
    m_r0.assign( m_seeingCircBuffLen, 0 );
    m_sig.assign( m_seeingCircBuffLen, 0 );
}

inline bool wooferTweeterRecon::snapshotSeeing( std::vector<float> &oneSecond,
                                                std::vector<float> &tenSecond,
                                                float              &fps,
                                                float              &el )
{
    std::lock_guard<std::mutex> guard( m_reconMutex );

    fps = m_fps;
    el  = m_el;

    if( fps <= 0 || m_r0.empty() || m_r0Count == 0 )
    {
        return false;
    }

    size_t n1sec  = std::min( m_r0Count, std::max<size_t>( 1, static_cast<size_t>( std::ceil( fps ) ) ) );
    size_t n10sec = std::min( m_r0Count, std::max<size_t>( 1, static_cast<size_t>( std::ceil( 10 * fps ) ) ) );

    if( n1sec == 0 || n10sec == 0 )
    {
        return false;
    }

    oneSecond.resize( n1sec );
    tenSecond.resize( n10sec );

    for( size_t n = 0; n < n10sec; ++n )
    {
        size_t idx = ( m_lastr0 + m_r0.size() - n ) % m_r0.size();

        tenSecond[n] = m_r0[idx];

        if( n < n1sec )
        {
            oneSecond[n] = tenSecond[n];
        }
    }

    return true;
}

int wooferTweeterRecon::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( wooferModesSMT );
    SHMIMMONITORT_APP_LOGIC( tweeterModesSMT );
    SHMIMMONITORT_APP_LOGIC( wfsModesSMT );

    // TELEMETER_APP_LOGIC;

    std::vector<float> r0_1sec;
    std::vector<float> r0_10sec;
    float              fps{ 0 };
    float              el{ 90 };
    bool               haveSeeing{ false };
    float              r01{ 0 };
    float              vr01{ 0 };
    float              fwhm1{ 0 };
    float              vfw1{ 0 };
    float              fwhm1cz{ 0 };
    float              r010{ 0 };
    float              vr010{ 0 };
    float              fwhm10{ 0 };
    float              vfw10{ 0 };
    float              fwhm10cz{ 0 };

    if( snapshotSeeing( r0_1sec, r0_10sec, fps, el ) )
    {
        float cz = std::pow( std::cos( 3.14159 / 180. * ( 90 - el ) ), 3. / 5. );

        r01  = mx::math::vectorMean( r0_1sec );
        vr01 = std::sqrt( mx::math::vectorVariance( r0_1sec, r01 ) );

        r010  = mx::math::vectorMean( r0_10sec );
        vr010 = std::sqrt( mx::math::vectorVariance( r0_10sec, r010 ) );

        if( r01 > 0 && r010 > 0 )
        {
            fwhm1   = 0.2063 * 0.5 / r01;
            vfw1    = fwhm1 * ( vr01 / r01 );
            fwhm1cz = fwhm1 * cz;

            fwhm10   = 0.2063 * 0.5 / r010;
            vfw10    = fwhm10 * ( vr010 / r010 );
            fwhm10cz = fwhm10 * cz;

            haveSeeing = true;
        }
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( haveSeeing )
    {
        updatesIfChanged<float>( m_indiP_seeing,
                                 { "r0_1sec",
                                   "r0_1sec_std",
                                   "fwhm_1sec",
                                   "fwhm_1sec_std",
                                   "fwhm_1sec_zenith",
                                   "r0_10sec",
                                   "r0_10sec_std",
                                   "fwhm_10sec",
                                   "fwhm_10sec_std",
                                   "fwhm_10sec_zenith" },
                                 { r01, vr01, fwhm1, vfw1, fwhm1cz, r010, vr010, fwhm10, vfw10, fwhm10cz } );
    }

    SHMIMMONITORT_UPDATE_INDI( wooferModesSMT );
    SHMIMMONITORT_UPDATE_INDI( tweeterModesSMT );
    SHMIMMONITORT_UPDATE_INDI( wfsModesSMT );

    return 0;
}

inline int wooferTweeterRecon::appShutdown()
{
    SHMIMMONITORT_APP_SHUTDOWN( wooferModesSMT );
    SHMIMMONITORT_APP_SHUTDOWN( tweeterModesSMT );
    SHMIMMONITORT_APP_SHUTDOWN( wfsModesSMT );

    // TELEMETER_APP_SHUTDOWN;

    return 0;
}

int wooferTweeterRecon::allocate( const wooferModesShmimT & )
{
    bool restartWfs{ false };
    bool restartWoofer{ false };

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_reconMutex );

        m_wooferModesReady = false;

        std::cerr << "woofer modes not ready\n";

        if( !m_wfsModesReady || wooferModesSMT::m_width > wfsModesSMT::m_width )
        {
            restartWfs    = m_wfsModesReady;
            restartWoofer = true;
        }
        else
        {
            resetModevalBuffer( m_wooferVals, m_lastWooferVal, wooferModesSMT::m_width );
            m_nloaded          = 0;
            m_wooferModesReady = true;

            std::cerr << "woofer modes ready\n";
        }
    }

    if( restartWfs )
    {
        wfsModesSMT::m_restart = true;
    }

    if( restartWoofer )
    {
        wooferModesSMT::m_restart = true;

        mx::sys::milliSleep( 1000 );
    }

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const wooferModesShmimT & )
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_reconMutex );

        if( !m_wooferModesReady || m_wooferVals.empty() )
        {
            return 0;
        }

        size_t next = m_lastWooferVal + 1;
        if( next >= m_wooferVals.size() )
        {
            next = 0;
        }

        for( size_t n = 0; n < m_wooferVals[next].vals.size(); ++n )
        {
            m_wooferVals[next].vals[n] = reinterpret_cast<float *>( curr_src )[n];
        }

        m_wooferVals[next].t = wooferModesSMT::m_imageStream.md->atime.tv_sec +
                               wooferModesSMT::m_imageStream.md->atime.tv_nsec / 1e9 + m_wooferOffset;
        m_wooferVals[next].reconstructed = false;

        m_lastWooferVal = next;
    }

    return recon();
}

int wooferTweeterRecon::allocate( const tweeterModesShmimT & )
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_reconMutex );

        m_tweeterModesReady = false;
        m_wooferModesReady  = false;
        m_wfsModesReady     = false;

        std::cerr << "tweeter modes not ready\n";

        resetModevalBuffer( m_tweeterVals, m_lastTweeterVal, tweeterModesSMT::m_width );
        resetSeeingState();
        m_outputVal.resize( tweeterModesSMT::m_width, m_nvals );
        m_nloaded           = 0;
        m_tweeterModesReady = true;

        std::cerr << "tweeter modes ready\n";
    }

    wfsModesSMT::m_restart    = true;
    wooferModesSMT::m_restart = true;

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const tweeterModesShmimT & )
{
    std::lock_guard<std::mutex> guard( m_reconMutex );

    if( !m_tweeterModesReady || m_tweeterVals.empty() )
    {
        return 0;
    }

    size_t next = m_lastTweeterVal + 1;
    if( next >= m_tweeterVals.size() )
    {
        next = 0;
    }

    for( size_t n = 0; n < m_tweeterVals[next].vals.size(); ++n )
    {
        m_tweeterVals[next].vals[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    m_tweeterVals[next].t = tweeterModesSMT::m_imageStream.md->atime.tv_sec +
                            tweeterModesSMT::m_imageStream.md->atime.tv_nsec / 1e9 + m_tweeterOffset;
    m_tweeterVals[next].reconstructed = false;

    m_lastTweeterVal = next;

    return 0;
}

int wooferTweeterRecon::allocate( const wfsModesShmimT & )
{
    bool restartTweeter{ false };
    bool restartWfs{ false };

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_reconMutex );

        m_wfsModesReady = false;

        std::cerr << "wfs modes not ready\n";

        if( !m_tweeterModesReady || wfsModesSMT::m_width != tweeterModesSMT::m_width )
        {
            restartTweeter = m_tweeterModesReady;
            restartWfs     = true;
        }
        else
        {
            resetModevalBuffer( m_wfsVals, m_lastWfsVal, wfsModesSMT::m_width );
            m_nloaded       = 0;
            m_wfsModesReady = true;

            std::cerr << "wfs modes ready\n";
        }
    }

    if( restartTweeter )
    {
        tweeterModesSMT::m_restart = true;
    }

    if( restartWfs )
    {
        wfsModesSMT::m_restart = true;
        mx::sys::milliSleep( 1000 );
    }

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const wfsModesShmimT & )
{
    std::lock_guard<std::mutex> guard( m_reconMutex );

    if( !m_wfsModesReady || m_wfsVals.empty() )
    {
        return 0;
    }

    size_t next = m_lastWfsVal + 1;
    if( next >= m_wfsVals.size() )
    {
        next = 0;
    }

    for( size_t n = 0; n < m_wfsVals[next].vals.size(); ++n )
    {
        m_wfsVals[next].vals[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    m_wfsVals[next].t = wfsModesSMT::m_imageStream.md->writetime.tv_sec +
                        wfsModesSMT::m_imageStream.md->writetime.tv_nsec / 1e9 - m_invFps + m_wfsOffset;
    m_wfsVals[next].reconstructed = false;

    m_lastWfsVal = next;

    return 0;
}

#define decst                                                                                                          \
    if( st == 0 )                                                                                                      \
    {                                                                                                                  \
        st = m_wfsVals.size();                                                                                         \
    }                                                                                                                  \
    --st;

int wooferTweeterRecon::recon()
{
    std::lock_guard<std::mutex> guard( m_reconMutex );

    if( !m_wooferModesReady || !m_tweeterModesReady || !m_wfsModesReady || m_wooferVals.empty() ||
        m_tweeterVals.empty() || m_wfsVals.empty() || m_r0.empty() || m_outputVal.rows() == 0 ||
        m_outputVal.cols() == 0 )
    {
        return 0;
    }

    if( m_nloaded != 0 )
    {
        std::cerr << "we're behind!\n";
    }

    size_t st = m_lastWfsVal;

    size_t wst = m_lastWooferVal;

    size_t tst = m_lastTweeterVal;

    size_t nChecked = 0;
    while( nChecked < m_wfsVals.size() && m_wfsVals[st].reconstructed == false && m_wfsVals[st].t > 0 )
    {
        ++nChecked;

        // Find starting woofer value
        if( m_wooferVals[wst].t < m_wfsVals[st].t )
        {
            // Starting woofer value is not later than current WFS val
            decst;
            continue;
        }

        while( m_wooferVals[wst].t > m_wfsVals[st].t && m_wooferVals[wst].t > 0 )
        {
            if( wst == 0 )
            {
                wst = m_wooferVals.size();
            }
            --wst;

            if( wst == m_lastWooferVal || m_wooferVals[wst].t == 0 )
            {
                // Starting woofer value is not later than current WFS val
                break;
            }
        }

        // Have to check this again so we continue the right loop
        if( wst == m_lastWooferVal || m_wooferVals[wst].t == 0 )
        {
            // Starting woofer value is not later than current WFS val
            decst;
            continue;
        }

        size_t wnxt = wst + 1;
        if( wnxt >= m_wooferVals.size() )
        {
            wnxt = 0;
        }

        if( m_wooferVals[wnxt].t == 0 )
        {
            decst;
            continue;
        }

        if( !( m_wooferVals[wst].t <= m_wfsVals[st].t && m_wooferVals[wnxt].t >= m_wfsVals[st].t ) )
        {
            std::cerr << __LINE__ << '\n';
            // an error!
            return -1;
        }

        // std::cerr << "Found woofer: " << m_wfsVals[st].t - m_wooferVals[wst].t << ' '
        //           << m_wooferVals[wnxt].t - m_wfsVals[st].t << '\n';

        // Find starting tweeter value
        if( m_tweeterVals[tst].t < m_wfsVals[st].t )
        {
            // Starting tweeter value is not later than current WFS val
            decst;
            continue;
        }

        while( m_tweeterVals[tst].t > m_wfsVals[st].t && m_tweeterVals[tst].t > 0 )
        {
            if( tst == 0 )
            {
                tst = m_tweeterVals.size();
            }
            --tst;

            if( tst == m_lastTweeterVal || m_tweeterVals[tst].t == 0 )
            {
                // Starting tweeter value is not later than current WFS val
                break;
            }
        }

        // Have to check this again so we continue the right loop
        if( tst == m_lastTweeterVal || m_tweeterVals[tst].t == 0 )
        {
            // Starting tweeter value is not later than current WFS val
            decst;
            continue;
        }

        size_t tnxt = tst + 1;
        if( tnxt >= m_tweeterVals.size() )
        {
            tnxt = 0;
        }

        if( m_tweeterVals[tnxt].t == 0 )
        {
            decst;
            continue;
        }

        if( !( m_tweeterVals[tst].t <= m_wfsVals[st].t && m_tweeterVals[tnxt].t >= m_wfsVals[st].t ) )
        {
            std::cerr << __LINE__ << '\n';
            // an error!
            return -1;
        }

        // std::cerr << "\tFound tweeter: " << m_wfsVals[st].t - m_tweeterVals[tst].t << ' '
        //           << m_tweeterVals[tnxt].t - m_wfsVals[st].t << '\n';

        double wdt = ( m_wfsVals[st].t - m_wooferVals[wst].t ) / ( m_wooferVals[wnxt].t - m_wooferVals[wst].t );
        double tdt = ( m_wfsVals[st].t - m_tweeterVals[tst].t ) / ( m_tweeterVals[tnxt].t - m_tweeterVals[tst].t );

        float  s2     = 0;
        size_t nModes = std::min( { m_wfsVals[st].vals.size(),
                                    m_tweeterVals[tst].vals.size(),
                                    m_tweeterVals[tnxt].vals.size(),
                                    static_cast<size_t>( m_outputVal.rows() ) } );

        m_outputVal.col( m_nloaded ).setZero();

        if( nModes == 0 )
        {
            decst;
            continue;
        }

        for( size_t n = 0; n < nModes; ++n )
        {
            float wval = 0;
            if( n < m_wooferVals[wst].vals.size() && n < m_wooferVals[wnxt].vals.size() )
            {
                wval = m_wooferVals[wst].vals[n] + ( m_wooferVals[wnxt].vals[n] - m_wooferVals[wst].vals[n] ) * wdt;
            }

            float tval =
                m_tweeterVals[tst].vals[n] + ( m_tweeterVals[tnxt].vals[n] - m_tweeterVals[tst].vals[n] ) * tdt;
            float wfsval = m_wfsVals[st].vals[n] / m_opticalGain;

            s2 += wfsval * wfsval;

            m_outputVal( n, m_nloaded ) = 0.04 * wval + tval + wfsval; // wval;// + tval + wfsval;
        }

        float var = m_outputVal.col( m_nloaded ).square().sum();

        float r0 =
            std::pow( 1.0299 * std::pow( 6.5, 5. / 3. ) / ( 4 * var * std::pow( 2 * 3.14159 / 0.5, 2 ) ), 3. / 5. );

        size_t nr0 = m_lastr0 + 1;
        if( nr0 >= m_r0.size() )
        {
            nr0 = 0;
        }

        m_r0[nr0]  = r0;
        m_sig[nr0] = s2;

        m_lastr0  = nr0;
        m_r0Count = std::min( m_r0Count + 1, m_r0.size() );

        m_wfsVals[st].reconstructed = true;

        ++m_nloaded;

        if( m_nloaded >= m_outputVal.cols() )
        {
            std::cerr << "we're behind more\n";
            break;
        }

        decst;
    }

    m_nloaded = 0; // resetting until fg implemented
    // the fg will load the last m_nloaded into the c-buff shmim, from which PSDs will be calculated

    return 0;
}

INDI_SETCALLBACK_DEFN( wooferTweeterRecon, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fpsSource.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid INDI property." } );
        return -1;
    }

    if( ipRecv.find( "current" ) != true ) // this isn't valie
    {
        return 0;
    }

    realT fps = ipRecv["current"].get<float>();
    bool  fpsChanged{ false };

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_reconMutex );

        if( fps != m_fps )
        {
            m_fps = fps;
            if( m_fps <= 0 )
            {
                m_invFps = 0;
            }
            else
            {
                m_invFps = 1.0 / m_fps;
            }

            fpsChanged = true;
        }
    }

    if( fpsChanged )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );
        updateIfChanged( m_indiP_fps, "current", fps );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( wooferTweeterRecon, m_indiP_elSource )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_elSource.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid INDI property." } );
        return -1;
    }

    if( ipRecv.find( "el" ) != true ) // this isn't valid
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_reconMutex );

    m_el = ipRecv["el"].get<float>();

    return 0;
}

/*
int wooferTweeterRecon::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_loopgain(), telem_offloading() );
}

int wooferTweeterRecon::recordTelem( const telem_loopgain * )
{
    return recordLoopGain( true );
}

int wooferTweeterRecon::recordLoopGain( bool force )
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

int wooferTweeterRecon::recordTelem( const telem_offloading * )
{
    return recordOffloading( true );
}

int wooferTweeterRecon::recordOffloading( bool force )
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
*/

} // namespace app
} // namespace MagAOX

#endif // wooferTweeterRecon_hpp
