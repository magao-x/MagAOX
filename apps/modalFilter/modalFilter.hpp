/** \file modalFilter.hpp
 * \brief The MagAO-X modal filter header file
 *
 * \ingroup modalFilter_files
 */

#ifndef modalFilter_hpp
#define modalFilter_hpp

#include <mutex>
#include <shared_mutex>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/sigproc/circularBuffer.hpp>

/** \defgroup modalFilter
 * \brief The MagAO-X application to perform modal filtering
 *
 * <a href="../handbook/operating/software/apps/modalFilter.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup modalFilter_files
 * \ingroup modalFilter
 */

namespace MagAOX
{
namespace app
{

struct gainFactShmimT
{
    static std::string configSection()
    {
        return "gainFactShmim";
    };

    static std::string indiPrefix()
    {
        return "gainFact";
    };
};

struct multFactShmimT
{
    static std::string configSection()
    {
        return "multFactShmim";
    };

    static std::string indiPrefix()
    {
        return "multFact";
    };
};

struct pcGainFactShmimT
{
    static std::string configSection()
    {
        return "pcGainFactShmim";
    };

    static std::string indiPrefix()
    {
        return "pcGainFact";
    };
};

struct pcMultFactShmimT
{
    static std::string configSection()
    {
        return "pcMultFactShmim";
    };

    static std::string indiPrefix()
    {
        return "pcMultFact";
    };
};

struct acoeffShmimT
{
    static std::string configSection()
    {
        return "acoeffShmim";
    };

    static std::string indiPrefix()
    {
        return "acoeff";
    };
};

struct bcoeffShmimT
{
    static std::string configSection()
    {
        return "bcoeffShmim";
    };

    static std::string indiPrefix()
    {
        return "bcoeff";
    };
};

struct modevalShmimT
{
    static std::string configSection()
    {
        return "modevalShmim";
    };

    static std::string indiPrefix()
    {
        return "modeval";
    };
};

/// The MagAO-X modal filter
/**
 * \ingroup modalFilter
 */
class modalFilter : public MagAOXApp<true>,
                    dev::shmimMonitor<modalFilter, gainFactShmimT>,
                    dev::shmimMonitor<modalFilter, multFactShmimT>,
                    dev::shmimMonitor<modalFilter, pcGainFactShmimT>,
                    dev::shmimMonitor<modalFilter, pcMultFactShmimT>,
                    dev::shmimMonitor<modalFilter, acoeffShmimT>,
                    dev::shmimMonitor<modalFilter, bcoeffShmimT>,
                    dev::shmimMonitor<modalFilter, modevalShmimT>,
                    dev::frameGrabber<modalFilter>,
                    dev::telemeter<modalFilter>
{

    // Give the test harness access.
    friend class modalFilter_test;

    friend class dev::shmimMonitor<modalFilter, gainFactShmimT>;
    friend class dev::shmimMonitor<modalFilter, multFactShmimT>;
    friend class dev::shmimMonitor<modalFilter, pcGainFactShmimT>;
    friend class dev::shmimMonitor<modalFilter, pcMultFactShmimT>;
    friend class dev::shmimMonitor<modalFilter, acoeffShmimT>;
    friend class dev::shmimMonitor<modalFilter, bcoeffShmimT>;
    friend class dev::shmimMonitor<modalFilter, modevalShmimT>;
    friend class dev::frameGrabber<modalFilter>;
    friend class dev::telemeter<modalFilter>;

    typedef dev::shmimMonitor<modalFilter, gainFactShmimT>   gainFactShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, multFactShmimT>   multFactShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, pcGainFactShmimT> pcGainFactShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, pcMultFactShmimT> pcMultFactShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, acoeffShmimT>     acoeffShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, bcoeffShmimT>     bcoeffShmimMonitorT;
    typedef dev::shmimMonitor<modalFilter, modevalShmimT>    modevalShmimMonitorT;
    typedef dev::frameGrabber<modalFilter>                   frameGrabberT;
    typedef dev::telemeter<modalFilter>                      telemeterT;

    static constexpr bool c_frameGrabber_flippable = false; /**< app:dev config to tell framegrabber these images
                                                                 can not be flipped*/

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_fpsDevice;               ///< Device name for getting fps to set circular buffer length.
    std::string m_fpsProperty{ "fps" };    ///< Property name for getting fps to set circular buffer length.
    std::string m_fpsElement{ "current" }; ///< Element name for getting fps to set circular buffer length.

    float m_fpsTol{ 0 }; ///< The tolerance for detecting a change in FPS.

    int m_loopNum{ 1 }; ///< The number of the loop. Used to set shmim names, as in aolN_mgainfact.  Default is 1.

    std::string m_loopName{ "ho" }; ///< The name of the loop control INDI device name. Defalt is "ho".

    int32_t m_modevalCBLength{ 1000 }; ///< The length of the modeval circular buffers.  Default is 1000 entries.

    ///@}

    float m_fps{ 0 };

    bool m_loop{ false };

    float m_gain{ 0 };

    float m_mc{ 1 };

    bool m_pcOn{ false };

    float m_pcGain{ 0 };

    float m_pcMc{ 1 };

    std::vector<float> m_gainfacts;

    std::vector<float> m_multfacts;

    std::vector<float> m_pcGainfacts;

    std::vector<float> m_pcMultfacts;

    std::vector<int> m_Na;

    std::vector<int> m_Nb;

    eigenImage<float> m_as;

    eigenImage<float> m_bs;

    uint32_t m_modevalSz{ 0 };

    mx::sigproc::circularBufferIndex<std::vector<float>, int32_t> m_modevalWFS;

    mx::sigproc::circularBufferIndex<std::vector<float>, int32_t> m_modevalDM;

    sem_t m_filtSem; ///< Semaphore used to signal that fresh modevals are waiting to be filtered

    bool m_sizesMatch{ false }; ///< Flag indicating that all sizes are consistent

    bool m_pcSizesMatch{ false }; ///< Flag indicating that all sizes are consistent

    std::shared_mutex m_filtMutex; ///< Mutex for locking access to filter parameters

    timespec m_atime; ///< The acq time of the WFS modevals
  public:
    /// Default c'tor.
    modalFilter();

    /// D'tor, declared and defined for noexcept.
    ~modalFilter() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                                                                    from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for modalFilter.
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

    int allocate( const gainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,        ///< [in] pointer to the start of the current frame
                      const gainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const multFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,        ///< [in] pointer to the start of the current frame
                      const multFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const pcGainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,          ///< [in] pointer to the start of the current frame
                      const pcGainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const pcMultFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,          ///< [in] pointer to the start of the current frame
                      const pcMultFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const acoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,      ///< [in] pointer to the start of the current frame
                      const acoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const bcoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,      ///< [in] pointer to the start of the current frame
                      const bcoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const modevalShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,       ///< [in] pointer to the start of the current frame
                      const modevalShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /** \name frameGrabber interface
     * @{
     */

    /// Implementation of the framegrabber configureAcquisition interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Implementation of the frameGrabber fps interface
    /** Just returns the value of m_fps
     */
    float fps();

    /// Implementation of the framegrabber startAcquisition interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Implementation of the framegrabber acquireAndCheckValid interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Implementation of the framegrabber loadImageIntoStream interface
    /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadImageIntoStream( void *dest /**< [in] */ );

    /// Implementation of the framegrabber reconfig interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int reconfig();

    ///@}

    void checkSizes();

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );

    ///@}

    /** \name INDI
     * @{
     */

    pcf::IndiProperty m_indiP_fpsSource;

    pcf::IndiProperty m_indiP_loop;

    pcf::IndiProperty m_indiP_gain;

    pcf::IndiProperty m_indiP_mult;

    pcf::IndiProperty m_indiP_pcGain;

    pcf::IndiProperty m_indiP_pcMult;

    pcf::IndiProperty m_indiP_pcOn;

    INDI_SETCALLBACK_DECL( modalFilter, m_indiP_fpsSource );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_loop );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_gain );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_mult );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcGain );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcMult );

    INDI_NEWCALLBACK_DECL( modalFilter, m_indiP_pcOn );

    ///@}
};

modalFilter::modalFilter() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{

    gainFactShmimMonitorT::m_getExistingFirst   = true;
    multFactShmimMonitorT::m_getExistingFirst   = true;
    pcGainFactShmimMonitorT::m_getExistingFirst = true;
    pcMultFactShmimMonitorT::m_getExistingFirst = true;
    acoeffShmimMonitorT::m_getExistingFirst     = true;
    bcoeffShmimMonitorT::m_getExistingFirst     = true;
    modevalShmimMonitorT::m_getExistingFirst    = true;
    return;
}

void modalFilter::setupConfig()
{
    config.add( "circBuff.fpsDevice",
                "",
                "circBuff.fpsDevice",
                argType::Required,
                "circBuff",
                "fpsDevice",
                false,
                "string",
                "Device name for getting fps to set circular buffer length." );
    config.add( "circBuff.fpsProperty",
                "",
                "circBuff.fpsProperty",
                argType::Required,
                "circBuff",
                "fpsProperty",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'fps'." );
    config.add( "circBuff.fpsElement",
                "",
                "circBuff.fpsElement",
                argType::Required,
                "circBuff",
                "fpsElement",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'current'." );
    config.add( "circBuff.fpsTol",
                "",
                "circBuff.fpsTol",
                argType::Required,
                "circBuff",
                "fpsTol",
                false,
                "float",
                "Tolerance for detecting a change in FPS.  Default is 0." );

    config.add( "loop.number",
                "",
                "loop.number",
                argType::Required,
                "loop",
                "number",
                false,
                "int",
                "The number of the loop. Used to set shmim names, as in aolN_mgainfact.  Default is 1." );

    config.add( "loop.name",
                "",
                "loop.name",
                argType::Required,
                "loop",
                "name",
                false,
                "string",
                "The name of the loop control INDI device name. Default is \"ho\"." );

    SHMIMMONITORT_SETUP_CONFIG( gainFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( multFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( pcGainFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( pcMultFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( acoeffShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( bcoeffShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( modevalShmimMonitorT, config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
}

int modalFilter::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_fpsDevice, "circBuff.fpsDevice" );
    _config( m_fpsProperty, "circBuff.fpsProperty" );
    _config( m_fpsElement, "circBuff.fpsElement" );
    _config( m_fpsTol, "circBuff.fpsTol" );

    _config( m_loopNum, "loop.number" );
    _config( m_loopName, "loop.name" );

    char shmim[1024];

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainfact", m_loopNum );
    gainFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( gainFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mmultfact", m_loopNum );
    multFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( multFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mpcgainfact", m_loopNum );
    pcGainFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( pcGainFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mpcmultfact", m_loopNum );
    pcMultFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( pcMultFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_acoeff", m_loopNum );
    acoeffShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( acoeffShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_bcoeff", m_loopNum );
    bcoeffShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( bcoeffShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_modevalWFS", m_loopNum );
    modevalShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( modevalShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_modevalDM", m_loopNum );
    frameGrabberT::m_shmimName = shmim;
    FRAMEGRABBER_LOAD_CONFIG( _config );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void modalFilter::loadConfig()
{
    loadConfigImpl( config );
}

int modalFilter::appStartup()
{

    if( sem_init( &m_filtSem, 0, 0 ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Initializing filter semaphore" } );
    }

    SHMIMMONITORT_APP_STARTUP( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( multFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( modevalShmimMonitorT );
    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    if( m_fpsDevice == "" )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "FPS source is not configurated (circBuff.fpsDevice)" } );
    }

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_loop, "loop_state" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_gain, "loop_gain", 0, 1, 0.01, "%0.01f", "Gain", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_mult, "loop_multcoeff", 0, 1, 0.01, "%0.01f", "Mult. Coeff.", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_pcGain, "loop_pcgain", 0, 1, 0.01, "%0.01f", "PC Gain", "Loop" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_pcMult, "loop_pcmultcoeff", 0, 1, 0.01, "%0.01f", "PC Mult. Coeff.", "Loop" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_pcOn, "loop_pcon" );

    state( stateCodes::OPERATING );

    return 0;
}

int modalFilter::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( multFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( modevalShmimMonitorT );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    SHMIMMONITORT_UPDATE_INDI( gainFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( multFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( pcGainFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( pcMultFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( acoeffShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( bcoeffShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( modevalShmimMonitorT );
    FRAMEGRABBER_UPDATE_INDI;

    if( m_loop )
    {
        updateSwitchIfChanged( m_indiP_loop, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_loop, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( m_pcOn )
    {
        updateSwitchIfChanged( m_indiP_pcOn, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_pcOn, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    updatesIfChanged<float>( m_indiP_gain, { "current", "target" }, { m_gain, m_gain } );
    updatesIfChanged<float>( m_indiP_mult, { "current", "target" }, { m_mc, m_mc } );

    updatesIfChanged<float>( m_indiP_pcGain, { "current", "target" }, { m_pcGain, m_pcGain } );
    updatesIfChanged<float>( m_indiP_pcMult, { "current", "target" }, { m_pcMc, m_pcMc } );

    return 0;
}

int modalFilter::appShutdown()
{
    SHMIMMONITORT_APP_SHUTDOWN( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( multFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( modevalShmimMonitorT );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int modalFilter::allocate( const gainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( gainFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "got gainFacts with height not 1" } );
    }

    // If there's a size change we have to lock
    if( gainFactShmimMonitorT::m_width != m_gainfacts.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_gainfacts.resize( gainFactShmimMonitorT::m_width );

        checkSizes();
    }

    return 0;
}

int modalFilter::processImage( void *curr_src, const gainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    // We just update without a lock.  Size change handled in allocate.
    for( uint32_t n = 0; n < gainFactShmimMonitorT::m_width; ++n )
    {
        m_gainfacts[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    return 0;
}

int modalFilter::allocate( const multFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( multFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "got multFacts with height not 1" } );
    }

    // If there's a size change we have to lock
    if( multFactShmimMonitorT::m_width != m_multfacts.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_multfacts.resize( multFactShmimMonitorT::m_width );

        checkSizes();
    }

    return 0;
}

int modalFilter::processImage( void *curr_src, const multFactShmimT &dummy )
{
    static_cast<void>( dummy );

    // We just update without a lock.  Size change handled in allocate.
    for( uint32_t n = 0; n < multFactShmimMonitorT::m_width; ++n )
    {
        m_multfacts[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    return 0;
}

int modalFilter::allocate( const pcGainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( pcGainFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "got pcGainFacts with height not 1" } );
    }

    // If there's a size change we have to lock
    if( pcGainFactShmimMonitorT::m_width != m_pcGainfacts.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_pcGainfacts.resize( pcGainFactShmimMonitorT::m_width );

        checkSizes();
    }

    return 0;
}

int modalFilter::processImage( void *curr_src, const pcGainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    // We just update without a lock.  Size change handled in allocate.
    for( uint32_t n = 0; n < pcGainFactShmimMonitorT::m_width; ++n )
    {
        m_pcGainfacts[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    return 0;
}

int modalFilter::allocate( const pcMultFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( pcMultFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "got pcMultFacts with height not 1" } );
    }

    // If there's a size change we have to lock
    if( pcMultFactShmimMonitorT::m_width != m_pcMultfacts.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_pcMultfacts.resize( pcMultFactShmimMonitorT::m_width );

        checkSizes();
    }

    return 0;
}

int modalFilter::processImage( void *curr_src, const pcMultFactShmimT &dummy )
{
    static_cast<void>( dummy );

    // We just update without a lock.  Size change handled in allocate.
    for( uint32_t n = 0; n < pcMultFactShmimMonitorT::m_width; ++n )
    {
        m_pcMultfacts[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    return 0;
}

int modalFilter::allocate( const acoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = acoeffShmimMonitorT::m_width;
    uint32_t h = acoeffShmimMonitorT::m_height;

    // If there's a size change we lock
    if( w - 1 != m_as.rows() || h != m_as.cols() || h != m_Na.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_Na.resize( h );
        m_as.resize( w - 1, h );

        checkSizes();
    }

    return 0;
}

int modalFilter::processImage( void *curr_src, const acoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = acoeffShmimMonitorT::m_width;
    uint32_t h = acoeffShmimMonitorT::m_height;

    eigenMap<float> ac( reinterpret_cast<float *>( curr_src ), w, h );

    for( uint32_t cc = 0; cc < h; ++cc )
    {
        m_Na[cc] = ac( 0, cc );
        for( uint32_t rr = 1; rr < w; ++rr )
        {
            m_as( rr - 1, cc ) = ac( rr, cc );
        }
    }

    std::cerr << "Got pc a-coeffs.  Mode 0 has " << m_Na[0] << '\n';

    return 0;
}

int modalFilter::allocate( const bcoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = bcoeffShmimMonitorT::m_width;
    uint32_t h = bcoeffShmimMonitorT::m_height;

    // If there's a size change we lock
    if( w - 1 != m_bs.rows() || h != m_bs.cols() || h != m_Nb.size() )
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_Nb.resize( h );
        m_bs.resize( w - 1, h );

        checkSizes();
    }

    std::cerr << "Got pc b-coeffs.  Mode 0 has " << m_Nb[0] << '\n';

    return 0;
}

int modalFilter::processImage( void *curr_src, const bcoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = bcoeffShmimMonitorT::m_width;
    uint32_t h = bcoeffShmimMonitorT::m_height;

    eigenMap<float> bc( reinterpret_cast<float *>( curr_src ), w, h );

    for( uint32_t cc = 0; cc < h; ++cc )
    {
        m_Nb[cc] = bc( 0, cc );
        for( uint32_t rr = 1; rr < w; ++rr )
        {
            m_bs( rr - 1, cc ) = bc( rr, cc );
        }
    }
    return 0;
}

int modalFilter::allocate( const modevalShmimT &dummy )
{
    static_cast<void>( dummy );

    if( modevalShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "got modevals with height not 1" } );
    }

    if( modevalShmimMonitorT::m_width != m_modevalSz ) // This invalidates the whole c.b.
    {
        std::shared_lock<std::shared_mutex> lock( m_filtMutex );

        m_modevalSz = modevalShmimMonitorT::m_width;
        m_modevalWFS.maxEntries( m_modevalCBLength );
        m_modevalDM.maxEntries( m_modevalCBLength );

        checkSizes();
    }

    std::cerr << "Allocated modevals: " << modevalShmimMonitorT::m_width << " x " << modevalShmimMonitorT::m_height
              << '\n';
    return 0;
}

int modalFilter::processImage( void *curr_src, const modevalShmimT &dummy )
{
    static_cast<void>( dummy );

    clock_gettime( CLOCK_ISIO, &m_atime );

    float *F = reinterpret_cast<float *>( curr_src );

    m_modevalWFS.nextEntry( std::vector<float>( F, F + m_modevalSz ) );

    // Now tell the writer to get going
    if( sem_post( &m_filtSem ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Error posting to filter semaphore" } );
    }

    return 0;
}

int modalFilter::configureAcquisition()
{

    while( ( modevalShmimMonitorT::m_width < 1 || modevalShmimMonitorT::m_height != 1 ) && !m_shutdown )
    {
        sleep( 1 );
    }

    frameGrabberT::m_width    = modevalShmimMonitorT::m_width;
    frameGrabberT::m_height   = modevalShmimMonitorT::m_height;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    return 0;
}

float modalFilter::fps()
{
    return m_fps;
}

int modalFilter::startAcquisition()
{
    return 0;
}

int modalFilter::acquireAndCheckValid()
{
    timespec ts;

    XWC_SEM_WAIT_TS( ts, 1, 0 );

    if( sem_timedwait( &m_filtSem, &ts ) != 0 )
    {
        /* Check for why we timed out */
        /* EINTER probably indicates time to shutdown, loop will exit if m_shutdown is set */
        /* ETIMEDOUT just means keep waiting */
        if( errno == EINTR || errno == ETIMEDOUT )
        {
            return 1;
        }

        /*Otherwise, report an error.*/
        return log<software_error, -1>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
    }
    else
    {
        std::unique_lock<std::shared_mutex> lock( m_filtMutex ); // gotta be the only one

        if( !m_sizesMatch )
        {
            return 1;
        }

        m_modevalDM.nextEntry();               // move to the next entry without modifying what is there
        m_modevalDM[-1].resize( m_modevalSz ); // needed while growing, no-op once grown
        if( !m_loop )
        {
            for( uint32_t mode = 0; mode < m_modevalSz; ++mode )
            {
                m_modevalDM[-1][mode] = 0;
            }
            return 1; // we update the cbuff but we don't push to the DM
        }
        else
        {
            if( !m_pcOn )
            {
                for( uint32_t mode = 0; mode < m_modevalSz; ++mode )
                {
                    // index [0] is 1the latest entry in the circular buffer
                    m_modevalDM[-1][mode] = -1.0 * m_gain * m_gainfacts[mode] * m_modevalWFS[-1][mode] +
                                            m_mc * m_multfacts[mode] * m_modevalDM[-2][mode];
                }
            }
            else
            {
                for( uint32_t mode = 0; mode < m_modevalSz; ++mode )
                {
                    float a = 0;
                    float b = 0;
                    for(int c = 0; c < m_Na[mode]; ++c)
                    {
                        a += m_as(c,mode)*m_modevalWFS[-1-c][mode];
                    }

                    for(int c = 0; c < m_Nb[mode]; ++c)
                    {
                        b += m_bs(c,mode)*m_modevalDM[-2-c][mode];
                    }

                    a = m_modevalWFS[-1][mode];
                    b = m_modevalDM[-2][mode];
                    m_modevalDM[-1][mode] = -1.0 * m_pcGain * m_pcGainfacts[mode] * a +
                                            m_pcMc * m_pcMultfacts[mode] * b;
                }
            }

            frameGrabberT::m_currImageTimestamp = m_atime;
        }
    }

    return 0;
}

int modalFilter::loadImageIntoStream( void *dest )
{

    memcpy( dest, m_modevalDM[-1].data(), m_modevalDM[-1].size() * sizeof( float ) );

    return 0;
}

int modalFilter::reconfig()
{
    return 0;
}

void modalFilter::checkSizes()
{
    m_sizesMatch   = true;
    m_pcSizesMatch = true;

    if( m_gainfacts.size() != m_modevalSz )
    {
        m_sizesMatch = false;
    }

    if( m_multfacts.size() != m_modevalSz )
    {
        m_sizesMatch = false;
    }

    if( static_cast<size_t>( m_as.cols() ) != m_modevalSz )
    {
        m_pcSizesMatch = false;
    }

    if( static_cast<size_t>( m_bs.cols() ) != m_modevalSz )
    {
        m_pcSizesMatch = false;
    }

    if( m_modevalWFS.size() != m_modevalCBLength )
    {
        m_pcSizesMatch = false;
    }

    if( m_modevalDM.size() != m_modevalCBLength )
    {
        m_pcSizesMatch = false;
    }
}

int modalFilter::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int modalFilter::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

INDI_SETCALLBACK_DEFN( modalFilter, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        log<software_error>( { __FILE__, __LINE__, "No current property in fps source." } );
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    float fps = ipRecv[m_fpsElement].get<float>();

    if( fabs( fps - m_fps ) > m_fpsTol )
    {
        m_fps = fps;
        log<text_log>( "set fps to " + std::to_string( m_fps ), logPrio::LOG_NOTICE );
        frameGrabberT::m_reconfig = true;
    }

    return 0;

} // INDI_SETCALLBACK_DEFN(modalFilter, m_indiP_fpsSource)

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_loop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loop, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {
            if( !m_loop )
            {
                log<loop_closed>();
            }
            m_loop = true;
        }
        else
        {
            if( m_loop )
            {
                log<loop_open>();
            }
            m_loop = false;
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_gain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_gain = target;

    std::cerr << "Got global gain: " << m_gain << '\n';

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_mult )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mult, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_mult, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_mc = target;

    std::cerr << "Got global mc: " << m_mc << '\n';

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcGain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcGain, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_pcGain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_pcGain = target;

    std::cerr << "Got global pc gain: " << m_pcGain << '\n';
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcMult )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcMult, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_pcMult, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_pcMc = target;

    std::cerr << "Got global pc mc: " << m_mc << '\n';

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalFilter, m_indiP_pcOn )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcOn, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {

            m_pcOn = true;
        }
        else
        {
            m_pcOn = false;
        }
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // modalFilter_hpp
