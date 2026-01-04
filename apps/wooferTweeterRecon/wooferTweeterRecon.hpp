/** \file wooferTweeterRecon.hpp
 * \brief The MagAO-X woofer-tweeter pseudo-open-loop reconstructor
 *
 * \ingroup wooferTweeterRecon_files
 */

#ifndef wooferTweeterRecon_hpp
#define wooferTweeterRecon_hpp

#include <limits>

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
                           public dev::shmimMonitor<wooferTweeterRecon, wfsModesShmimT>//,
                           //public dev::frameGrabber<wooferTweeterRecon>,
                           //public dev::telemeter<wooferTweeterRecon>
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

    std::string m_fpsSource{ "camwfs" };

    std::string m_elSource{ "tcsi" };

    uint32_t m_modevalCircBuffLen{ 5000 };

    double m_wooferOffset{ 500e-6 };

    double m_tweeterOffset{ 50e-6 };

    double m_wfsOffset{ -10e-6 };

    ///@}

    bool m_wooferModesReady{ false };
    bool m_tweeterModesReady{ false };
    bool m_wfsModesReady{ false };

    struct modevals
    {
        double             t{ 0 };
        std::vector<float> vals;
        bool               reconstructed{ false };
    };

    std::vector<modevals> m_wooferVals;
    size_t                m_lastWooferVal{ 0 };

    std::vector<modevals> m_tweeterVals;
    size_t                m_lastTweeterVal{ 0 };

    std::vector<modevals> m_wfsVals;
    size_t                m_lastWfsVal{ 0 };

    float m_fps{ 0 }; ///< Current FPS from the FPS source.

    float m_invFps{ 0 }; ///< The inverse of FPS

    float m_el {90}; ///< The current elevation

    float m_opticalGain{ 0.8 };

    /// Mutex for locking shared memory access.
    // std::mutex m_shmimMutex;

    mx::improc::eigenImage<float> m_outputVal;
    int                           m_nvals{ 3600*2 }; //2 sec of data at max speed
    int                           m_nloaded{ 0 };

    std::vector<float> m_r0;
    std::vector<float> m_sig;
    size_t             m_lastr0;

  public:
    /// Default c'tor.
    wooferTweeterRecon();

    /// D'tor, declared and defined for noexcept.
    ~wooferTweeterRecon() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                        from which to load values*/
    );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for wooferTweeterRecon.
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

    /// Allocate method for the woofer command shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const wooferModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the woofer command shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,           ///< [in] pointer to start of current frame.
                      const wooferModesShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int recon();

    /// Allocate method for the tweeter command shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const tweeterModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the tweeter command shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,            ///< [in] pointer to start of current frame.
                      const tweeterModesShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /// Allocate method for the wfs modes shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const wfsModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the wfs modes shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,        ///< [in] pointer to start of current frame.
                      const wfsModesShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int prepareModes();

  protected:
    /** \name INDI Interface
     *
     * @{
     */
    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( wooferTweeterRecon, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_fps;

    pcf::IndiProperty m_indiP_elSource;
    INDI_SETCALLBACK_DECL( wooferTweeterRecon, m_indiP_elSource );

    pcf::IndiProperty m_indiP_seeing;

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_loopgain * );

    int recordLoopGain( bool force = false );

    int recordTelem( const telem_offloading * );

    int recordOffloading( bool force = false );

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

int wooferTweeterRecon::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( wooferModesSMT );
    SHMIMMONITORT_APP_LOGIC( tweeterModesSMT );
    SHMIMMONITORT_APP_LOGIC( wfsModesSMT );

    // TELEMETER_APP_LOGIC;

    if( m_fps > 0 )
    {
        float cz = pow( cos( 3.14159 / 180. * ( 90 - m_el ) ), 3 / 5. );

        size_t n1sec = m_fps;

        std::vector<float> cr0( n1sec );

        for( size_t n = 0; n < n1sec; ++n )
        {
            ssize_t m = m_lastr0 - n;
            if( m < 0 )
            {
                m = m_r0.size() - 1;
            }
            cr0[n] = m_r0[m];
        }

        float r01  = mx::math::vectorMean( cr0 );
        float vr01 = sqrt( mx::math::vectorVariance( cr0, r01 ) );

        float fwhm1   = 0.2063 * 0.5 / r01;
        float vfw1    = fwhm1 * ( vr01 / r01 );
        float fwhm1cz = fwhm1 * cz;

        size_t n10sec = 10 * m_fps;

        cr0.resize( n10sec );

        for( size_t n = 0; n < n10sec; ++n )
        {
            ssize_t m = m_lastr0 - n;
            if( m < 0 )
            {
                m = m_r0.size() - 1;
            }
            cr0[n] = m_r0[m];
        }

        float r010  = mx::math::vectorMean( cr0 );
        float vr010 = sqrt( mx::math::vectorVariance( cr0, r010 ) );

        float fwhm10   = 0.2063 * 0.5 / r010;
        float vfw10    = fwhm10 * ( vr010 / r010 );
        float fwhm10cz = fwhm10 * cz;

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

        // float s2 = mx::math::vectorMean(m_sig);
        // float S = exp(-s2*pow(2*3.14159/0.9,2) - 0.28*pow(0.135/r0, 5./3.));

        // std::cerr << std::format("r0 = {} +/- {} fwhm = {}\" +/- {}\" at zenith = {}\" SR = {}", r0, vr0, fwhm, vfw,
        // fwhm*cz, S) << '\n';
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

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
    m_wooferModesReady = false;

    std::cerr << "woofer modes not ready\n";

    if( !m_wfsModesReady || wooferModesSMT::m_width > wfsModesSMT::m_width )
    {
        if( m_wfsModesReady )
        {
            wfsModesSMT::m_restart = true;
        }

        wooferModesSMT::m_restart = true;

        mx::sys::milliSleep( 1000 );

        return 0; // This won't log an error, but setting m_restart will cause it to loop again until sizes match
    }

    m_wooferVals.resize( m_modevalCircBuffLen );

    for( auto &val : m_wooferVals )
    {
        val.t = 0;
        val.vals.resize( tweeterModesSMT::m_width, 0 );
        val.reconstructed = false;
    }

    m_wooferModesReady = true;

    std::cerr << "woofer modes ready\n";

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const wooferModesShmimT & )
{
    size_t next = m_lastWooferVal + 1;
    if( next >= m_wooferVals.size() )
    {
        next = 0;
    }

    for( size_t n = 0; n < wooferModesSMT::m_width; ++n )
    {
        m_wooferVals[next].vals[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    m_wooferVals[next].t = wooferModesSMT::m_imageStream.md->atime.tv_sec +
                           wooferModesSMT::m_imageStream.md->atime.tv_nsec / 1e9 + m_wooferOffset;
    m_wooferVals[next].reconstructed = false;

    m_lastWooferVal = next;

    recon();

    return 0;
}

int wooferTweeterRecon::allocate( const tweeterModesShmimT & )
{
    m_tweeterModesReady = false;

    wfsModesSMT::m_restart    = true;
    wooferModesSMT::m_restart = true;

    std::cerr << "tweeter modes not ready\n";

    m_tweeterVals.resize( m_modevalCircBuffLen );

    m_r0.resize( 3600*30, 0 );
    m_sig.resize( 3600*30,0 );
    for( auto &val : m_tweeterVals )
    {
        val.t = 0;
        val.vals.resize( tweeterModesSMT::m_width, 0 );
        val.reconstructed = false;
    }

    m_outputVal.resize( tweeterModesSMT::m_width, m_nvals );
    m_tweeterModesReady = true;

    std::cerr << "tweeter modes ready\n";

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const tweeterModesShmimT & )
{
    size_t next = m_lastTweeterVal + 1;
    if( next >= m_tweeterVals.size() )
    {
        next = 0;
    }

    for( size_t n = 0; n < tweeterModesSMT::m_width; ++n )
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
    m_wfsModesReady = false;

    std::cerr << "wfs modes not ready\n";

    if( !m_tweeterModesReady || wfsModesSMT::m_width != tweeterModesSMT::m_width )
    {
        if( m_tweeterModesReady )
        {
            tweeterModesSMT::m_restart = true;
        }

        wfsModesSMT::m_restart = true;
        mx::sys::milliSleep( 1000 );

        return 0; // This won't log an error, but setting m_restart will cause it to loop again until sizes match
    }

    m_wfsVals.resize( m_modevalCircBuffLen );

    for( auto &val : m_wfsVals )
    {
        val.t = 0;
        val.vals.resize( wfsModesSMT::m_width, 0 );
        val.reconstructed = false;
    }

    m_wfsModesReady = true;

    std::cerr << "wfs modes ready\n";

    return 0;
}

int wooferTweeterRecon::processImage( void *curr_src, const wfsModesShmimT & )
{
    size_t next = m_lastWfsVal + 1;
    if( next >= m_wfsVals.size() )
    {
        next = 0;
    }

    for( size_t n = 0; n < wooferModesSMT::m_width; ++n )
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
    if( m_nloaded != 0 )
    {
        std::cerr << "we're behind!\n";
    }

    size_t st = m_lastWfsVal;

    size_t wst = m_lastWooferVal;

    size_t tst = m_lastTweeterVal;

    while( m_wfsVals[st].reconstructed == false && m_wfsVals[st].t > 0 )
    {
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

        float s2 = 0;
        for( size_t n = 0; n < m_wfsVals[st].vals.size(); ++n )
        {
            float wval = m_wooferVals[wst].vals[n] + ( m_wooferVals[wnxt].vals[n] - m_wooferVals[wst].vals[n] ) * wdt;
            float tval =
                m_tweeterVals[tst].vals[n] + ( m_tweeterVals[tnxt].vals[n] - m_tweeterVals[tst].vals[n] ) * tdt;
            float wfsval = m_wfsVals[st].vals[n] / m_opticalGain;

            s2 += wfsval * wfsval;

            m_outputVal( n, m_nloaded ) = 0.04 * wval + tval + wfsval; // wval;// + tval + wfsval;
        }

        float var = m_outputVal.col( m_nloaded ).square().sum();

        float r0 = pow( 1.0299 * pow( 6.5, 5. / 3. ) / ( 4 * var * pow( 2 * 3.14159 / 0.5, 2 ) ), 3. / 5. );

        size_t nr0 = m_lastr0 + 1;
        if( nr0 >= m_r0.size() )
        {
            nr0 = 0;
        }

        m_r0[nr0]  = r0;
        m_sig[nr0] = s2;

        m_lastr0 = nr0;

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

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

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

        updateIfChanged( m_indiP_fps, "current", m_fps );
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

    std::lock_guard<std::mutex> guard( m_indiMutex );

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
