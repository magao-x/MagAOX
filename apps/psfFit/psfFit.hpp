/** \file psfFit.hpp
 * \brief The MagAO-X PSF Fitter application header
 *
 * \ingroup psfFit_files
 */

#ifndef psfFit_hpp
#define psfFit_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup psfFit
 * \brief The MagAO-X PSF fitter.
 *
 * <a href="../handbook/operating/software/apps/psfFit.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup psfFit_files
 * \ingroup psfFit
 */

namespace MagAOX
{
namespace app
{

struct darkShmimT
{
    static std::string configSection()
    {
        return "darkShmim";
    };

    static std::string indiPrefix()
    {
        return "dark";
    };
};

struct refShmimT
{
    static std::string configSection()
    {
        return "refShmim";
    };

    static std::string indiPrefix()
    {
        return "ref";
    };
};

/// The MagAO-X PSF Fitter
/**
 * \ingroup psfFit
 */
class psfFit : public MagAOXApp<true>,
               public dev::shmimMonitor<psfFit>,
               public dev::shmimMonitor<psfFit, darkShmimT>,
               public dev::shmimMonitor<psfFit, refShmimT>,
               public dev::frameGrabber<psfFit>,
               public dev::telemeter<psfFit>
{
    // Give the test harness access.
    friend class psfFit_test;

    friend class dev::shmimMonitor<psfFit>;
    friend class dev::shmimMonitor<psfFit, darkShmimT>;
    friend class dev::shmimMonitor<psfFit, refShmimT>;
    friend class dev::frameGrabber<psfFit>;

    friend class dev::telemeter<psfFit>;

  public:
    /// The base shmimMonitor type
    typedef dev::shmimMonitor<psfFit> shmimMonitorT;

    /// The dark shmimMonitor type
    typedef dev::shmimMonitor<psfFit, darkShmimT> darkShmimMonitorT;

    /// The reference shmimMonitor type
    typedef dev::shmimMonitor<psfFit, refShmimT> refShmimMonitorT;

    // The base frameGrabber type
    typedef dev::frameGrabber<psfFit> frameGrabberT;

    // The base telemeter type
    typedef dev::telemeter<psfFit> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

    /** \name app::dev Configurations
     *@{
     */

    static constexpr bool c_frameGrabber_flippable =
        false; ///< app:dev config to tell framegrabber these images can not be flipped

    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_fpsDevice;               ///< Device name for getting fps to set circular buffer length.
    std::string m_fpsProperty{ "fps" };    ///< Property name for getting fps to set circular buffer length.
    std::string m_fpsElement{ "current" }; ///< Element name for getting fps to set circular buffer length.

    float m_fpsTol{ 0 }; ///< The tolerance for detecting a change in FPS.

    // shutter a change in shutter state resets stats
    std::string m_shutterDevice;                ///< Device name for getting shutter state
    std::string m_shutterProperty{ "shutter" }; ///< Property name for getting shutter state
    std::string m_shutterElement{ "toggle" };   ///< Element name for getting shutter state

    uint16_t m_fitCircBuffMaxLength{ 5 * 10000 }; ///< Maximum length of the latency measurement circular buffers

    float m_fitCircBuffMaxTime{ 5 }; ///< Maximum time of the latency meaurement circular buffers

    float m_deltaPixThresh{ 8 }; /**< Threshold in pixels for skipping frame due to mismatch between max and c.o.l.
                                      Default 2.*/

    float m_sigmaMaxThreshUp{ 5 }; /**< Threshold in rms for skipping frame due to max positive difference from mean
                                      max. Default 5.*/

    float m_fractionMaxThreshDown{ 0.1 }; /**< Threshold in fraction of the mean max for skipping frame due to
                                              drop from mean max. Default 0.1*/

    float m_sigmaPixThresh{ 10 }; /**< Threshold in rms for skipping frame due to max difference from last value.
                                     Example: if this is set to 10, then the pixel postion has to change from -5 sigma
                                     to + 5 sigma to be rejected. Default 10.*/

    ///@}

    mx::improc::eigenImage<float> m_image; ///< Holds the raw image

    mx::improc::eigenImage<float> m_dark; ///< Holds the dark image

    mx::improc::eigenImage<float> m_ref; ///< Holds the reference image

    bool m_updated{ false }; ///< Indicates that the coordinates were updated was updated

    bool m_skipped{ false }; ///< Indicates that the image failed quality control and this is a skip frame

    float m_x{ 0 };      ///< The current x coordinate
    float m_last_x{ 0 }; ///< The previous x coordinate

    float m_y{ 0 };      ///< The current y coordinate
    float m_last_y{ 0 }; ///< The previous y coordinate

    float m_dx{ 0 }; ///< The offset in x to apply to non-skipped measurements
    float m_dy{ 0 }; ///< The offset in y to apply to non-skipped measurements

    float m_fps{ 0 }; ///< The frame rate from the source camera

    bool m_shutter{ false };

    mx::sigproc::circularBufferIndex<float, cbIndexT> m_pcb; ///< Circular buffer for max pixel (p=peak)
    mx::sigproc::circularBufferIndex<float, cbIndexT> m_xcb; ///< Circular buffer for x COL coords
    mx::sigproc::circularBufferIndex<float, cbIndexT> m_ycb; ///< Circular buffer for y COL coords

    std::vector<float> m_pcbD; ///< Vector for doing calcs on max pixel
    std::vector<float> m_xcbD; ///< Vector for doing calcs on x COL coords
    std::vector<float> m_ycbD; ///< Vector for doing calcs on y COL coords

    float m_mnp{ 0 };  ///< The mean max pixel over the stats time
    float m_rmsp{ 0 }; ///< The rms max pixel over the stats time

    float m_mnx{ 0 };  ///< The mean x coord over the stats time
    float m_rmsx{ 0 }; ///< The rms x coord over the stats time
    float m_mny{ 0 };  ///< The mean y coord over the stats time
    float m_rmsy{ 0 }; ///< The rms y coord over the stats time

    uint64_t m_skipped_updating{ 0 };
    uint64_t m_skipped_updating_last{ 0 };

    uint64_t m_skipped_DeltaFromMax{ 0 };
    uint64_t m_skipped_DeltaFromMax_last{ 0 };

    uint64_t m_skipped_MaxRmsUp{ 0 };
    uint64_t m_skipped_MaxRmsUp_last{ 0 };

    uint64_t m_skipped_MaxRmsDown{ 0 };
    uint64_t m_skipped_MaxRmsDown_last{ 0 };

    uint64_t m_skipped_XRms{ 0 };
    uint64_t m_skipped_XRms_last{ 0 };

    uint64_t m_skipped_YRms{ 0 };
    uint64_t m_skipped_YRms_last{ 0 };

  public:
    /// Default c'tor.
    psfFit();

    /// D'tor, declared and defined for noexcept.
    ~psfFit() noexcept;

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for psfFit.
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

    // shmimMonitor interface:
    int allocate( const dev::shmimT & );

    int processImage( void *curr_src, const dev::shmimT & );

    // shmimMonitor interface for dark:
    int allocate( const darkShmimT & );

    int processImage( void *curr_src, const darkShmimT & );

    // shmimMonitor interface for reference:
    int allocate( const refShmimT & );

    int processImage( void *curr_src, const refShmimT & );

  protected:
    std::mutex m_imageMutex;

    sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the fg thread and the sm thread.

  public:
    /** \name dev::frameGrabber interface
     *
     * @{
     */

    /// Implementation of the framegrabber configureAcquisition interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int configureAcquisition();

    /// Implementation of the framegrabber fps interface
    /**
     * \todo this needs to infer the stream fps and return it
     */
    float fps()
    {
        return m_fps;
    }

    /// Implementation of the framegrabber startAcquisition interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int startAcquisition();

    /// Implementation of the framegrabber acquireAndCheckValid interface
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int acquireAndCheckValid();

    /// Implementation of the framegrabber loadImageIntoStream interface
    /**
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

  protected:
    /** \name INDI
     * @{
     */

    pcf::IndiProperty m_indiP_values;

    pcf::IndiProperty m_indiP_reset;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_reset );

    pcf::IndiProperty m_indiP_statsTime;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_statsTime );

    pcf::IndiProperty m_indiP_deltaPixThresh;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_deltaPixThresh );

    pcf::IndiProperty m_indiP_sigmaMaxThreshUp;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_sigmaMaxThreshUp );

    pcf::IndiProperty m_indiP_fractionMaxThreshDown;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_fractionMaxThreshDown );

    pcf::IndiProperty m_indiP_sigmaPixThresh;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_sigmaPixThresh );

    pcf::IndiProperty m_indiP_dx;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_dx );

    pcf::IndiProperty m_indiP_dy;
    INDI_NEWCALLBACK_DECL( psfFit, m_indiP_dy );

    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( psfFit, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_shutter;
    INDI_SETCALLBACK_DECL( psfFit, m_indiP_shutter );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );

    ///@}
};

inline psfFit::psfFit() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    darkShmimMonitorT::m_getExistingFirst = true;
    refShmimMonitorT::m_getExistingFirst  = true;

    return;
}

inline psfFit::~psfFit() noexcept
{
}

inline void psfFit::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    SHMIMMONITORT_SETUP_CONFIG( darkShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( refShmimMonitorT, config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "fitter.fpsDevice",
                "",
                "fitter.fpsDevice",
                argType::Required,
                "fitter",
                "fpsDevice",
                false,
                "string",
                "Device name for getting fps to set circular buffer length." );
    config.add( "fitter.fpsProperty",
                "",
                "fitter.fpsProperty",
                argType::Required,
                "fitter",
                "fpsProperty",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'fps'." );
    config.add( "fitter.fpsElement",
                "",
                "fitter.fpsElement",
                argType::Required,
                "fitter",
                "fpsElement",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'current'." );
    config.add( "fitter.fpsTol",
                "",
                "fitter.fpsTol",
                argType::Required,
                "fitter",
                "fpsTol",
                false,
                "float",
                "Tolerance for detecting a change in FPS.  Default is 0." );
    config.add( "fitter.defaultFPS",
                "",
                "fitter.defaultFPS",
                argType::Required,
                "fitter",
                "defaultFPS",
                false,
                "realT",
                "Default FPS at startup, will enable changing average length with psdTime before INDI available." );

    config.add( "fitter.shutterDevice",
                "",
                "fitter.shutterDevice",
                argType::Required,
                "fitter",
                "shutterDevice",
                false,
                "string",
                "Device name for getting shutter state to reset circular buffers" );
    config.add( "fitter.shutterProperty",
                "",
                "fitter.shutterProperty",
                argType::Required,
                "fitter",
                "shutterProperty",
                false,
                "string",
                "Property name for getting shutter state to reset circular buffers. Default is 'shutter'." );
    config.add( "fitter.shutterElement",
                "",
                "fitter.shutterElement",
                argType::Required,
                "fitter",
                "shutterElement",
                false,
                "string",
                "Property name for getting shutter state to reset circular buffers. Default is 'toggle'." );

    config.add( "fitter.deltaPixThresh",
                "",
                "fitter.deltaPixThresh",
                argType::Required,
                "fitter",
                "deltaPixThresh",
                false,
                "float",
                "Threshold in pixels for skipping frame due to mismatch between max and c.o.l.  Default 8." );
    config.add( "fitter.sigmaMaxThreshUp",
                "",
                "fitter.sigmaMaxThreshUp",
                argType::Required,
                "fitter",
                "sigmaMaxThreshUp",
                false,
                "float",
                "Threshold in rms for skipping frame due to max positive difference from mean max. Default 5." );
    config.add( "fitter.fractionMaxThreshDown",
                "",
                "fitter.fractionMaxThreshDown",
                argType::Required,
                "fitter",
                "fractionMaxThreshDown",
                false,
                "float",
                "Threshold in fraction of the mean max for skipping frame due to drop from mean max. Default 0.1" );

    config.add( "fitter.sigmaPixThresh",
                "",
                "fitter.sigmaPixThresh",
                argType::Required,
                "fitter",
                "sigmaPixThresh",
                false,
                "float",
                "Threshold in rms for skipping frame due to max difference from last value.  Example: if this is set "
                "to 10, then the pixel postion has to change from -5 sigma to + 5 sigma to be rejected. Default 10." );
}

inline int psfFit::loadConfigImpl( mx::app::appConfigurator &_config )
{
    SHMIMMONITOR_LOAD_CONFIG( _config );
    SHMIMMONITORT_LOAD_CONFIG( darkShmimMonitorT, _config );
    SHMIMMONITORT_LOAD_CONFIG( refShmimMonitorT, _config );

    FRAMEGRABBER_LOAD_CONFIG( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsDevice, "fitter.fpsDevice" );
    _config( m_fpsProperty, "fitter.fpsProperty" );
    _config( m_fpsElement, "fitter.fpsElement" );
    _config( m_fpsTol, "fitter.fpsTol" );
    _config( m_fps, "fitter.defaultFPS" );

    _config( m_shutterDevice, "fitter.shutterDevice" );
    _config( m_shutterProperty, "fitter.shutterProperty" );
    _config( m_shutterElement, "fitter.shutterElement" );

    _config( m_deltaPixThresh, "fitter.deltaPixThresh" );
    _config( m_sigmaMaxThreshUp, "fitter.sigmaMaxThreshUp" );
    _config( m_fractionMaxThreshDown, "fitter.fractionMaxThreshDown" );
    _config( m_sigmaPixThresh, "fitter.sigmaPixThresh" );

    return 0;
}

inline void psfFit::loadConfig()
{
    loadConfigImpl( config );
}

inline int psfFit::appStartup()
{
    SHMIMMONITOR_APP_STARTUP;
    SHMIMMONITORT_APP_STARTUP( darkShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( refShmimMonitorT );

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Initializing S.M. semaphore" } );
        return -1;
    }

    FRAMEGRABBER_APP_STARTUP;
    TELEMETER_APP_STARTUP;

    if( m_fpsDevice != "" && m_fpsProperty != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );
    }

    if( m_shutterDevice != "" && m_shutterProperty != "" )
    {
        REG_INDI_SETPROP( m_indiP_shutter, m_shutterDevice, m_shutterProperty );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_values, "values", "", "" );
    m_indiP_values.add( pcf::IndiElement( "max_mean" ) );
    m_indiP_values.add( pcf::IndiElement( "max_rms" ) );
    m_indiP_values.add( pcf::IndiElement( "x_mean" ) );
    m_indiP_values.add( pcf::IndiElement( "x_rms" ) );
    m_indiP_values.add( pcf::IndiElement( "y_mean" ) );
    m_indiP_values.add( pcf::IndiElement( "y_rms" ) );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_reset, "reset" );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_statsTime, "statsTime", 0, 5, 1, "%0.1f", "", "" );
    m_indiP_statsTime["current"].setValue( m_fitCircBuffMaxTime );
    m_indiP_statsTime["target"].setValue( m_fitCircBuffMaxTime );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_deltaPixThresh, "deltaPixThresh", 0, 512, 1, "%0.1f", "", "" );
    m_indiP_deltaPixThresh["current"].setValue( m_deltaPixThresh );
    m_indiP_deltaPixThresh["target"].setValue( m_deltaPixThresh );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_sigmaMaxThreshUp, "sigmaMaxThreshUp", 0, 512, 1, "%0.1f", "", "" );
    m_indiP_sigmaMaxThreshUp["current"].setValue( m_sigmaMaxThreshUp );
    m_indiP_sigmaMaxThreshUp["target"].setValue( m_sigmaMaxThreshUp );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_fractionMaxThreshDown, "fractionMaxThreshDown", 0, 512, 1, "%0.1f", "", "" );
    m_indiP_fractionMaxThreshDown["current"].setValue( m_fractionMaxThreshDown );
    m_indiP_fractionMaxThreshDown["target"].setValue( m_fractionMaxThreshDown );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_sigmaPixThresh, "sigmaPixThresh", 0, 512, 1, "%0.1f", "", "" );
    m_indiP_sigmaPixThresh["current"].setValue( m_sigmaPixThresh );
    m_indiP_sigmaPixThresh["target"].setValue( m_sigmaPixThresh );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_dx, "dx", -100, 100, 1e-2, "%0.02f", "", "" );
    m_indiP_dx["current"].setValue( m_dx );
    m_indiP_dx["target"].setValue( m_dx );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_dy, "dy", -100, 100, 1e-2, "%0.02f", "", "" );
    m_indiP_dy["current"].setValue( m_dy );
    m_indiP_dy["target"].setValue( m_dy );

    state( stateCodes::OPERATING );

    return 0;
}

inline int psfFit::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    SHMIMMONITORT_APP_LOGIC( darkShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( refShmimMonitorT );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    if( state() == stateCodes::OPERATING && m_xcb.size() > 0 )
    {
        if( m_xcb.maxEntries() > 2 && m_xcb.size() >= m_xcb.maxEntries() )
        {
            cbIndexT refEntry = m_xcb.earliest();

            m_pcbD.resize( m_pcb.maxEntries() - 1 );
            m_xcbD.resize( m_xcb.maxEntries() - 1 );
            m_ycbD.resize( m_xcb.maxEntries() - 1 );

            for( size_t n = 0; n < m_xcbD.size(); ++n )
            {
                m_pcbD[n] = m_pcb.at( refEntry, n );
                m_xcbD[n] = m_xcb.at( refEntry, n );
                m_ycbD[n] = m_ycb.at( refEntry, n );
            }

            m_mnp  = mx::math::vectorMean( m_pcbD );
            m_rmsp = sqrt( mx::math::vectorVariance( m_pcbD, m_mnp ) );

            m_mnx = mx::math::vectorMean( m_xcbD );

            m_rmsx = sqrt( mx::math::vectorVariance( m_xcbD, m_mnx ) );

            m_mny  = mx::math::vectorMean( m_ycbD );
            m_rmsy = sqrt( mx::math::vectorVariance( m_ycbD, m_mny ) );
        }
        else
        {
            m_mnp  = 0;
            m_rmsp = 0;

            m_mnx  = 0;
            m_rmsx = 0;

            m_mny  = 0;
            m_rmsy = 0;
        }

        int skipped_updating     = m_skipped_updating - m_skipped_updating_last;
        int skipped_DeltaFromMax = m_skipped_DeltaFromMax - m_skipped_DeltaFromMax_last;
        int skipped_MaxRmsUp     = m_skipped_MaxRmsUp - m_skipped_MaxRmsUp_last;
        int skipped_MaxRmsDown   = m_skipped_MaxRmsDown - m_skipped_MaxRmsDown_last;
        int skipped_XRms         = m_skipped_XRms - m_skipped_XRms_last;
        int skipped_YRms         = m_skipped_YRms - m_skipped_YRms_last;

        if( skipped_updating || skipped_DeltaFromMax || skipped_MaxRmsUp || skipped_MaxRmsDown || skipped_XRms ||
            skipped_YRms )
        {
            std::cerr << "skipping frames: \n";
            std::cerr << "          updating: " << skipped_updating << '\n';
            std::cerr << "    delta-from-max: " << skipped_DeltaFromMax << '\n';
            std::cerr << "        max-rms-up: " << skipped_MaxRmsUp << '\n';
            std::cerr << "      max-rms-down: " << skipped_MaxRmsDown << '\n';
            std::cerr << "             x-rms: " << skipped_XRms << '\n';
            std::cerr << "             y-rms: " << skipped_YRms << '\n';
        }
        m_skipped_updating          = m_skipped_updating_last;
        m_skipped_DeltaFromMax_last = m_skipped_DeltaFromMax;
        m_skipped_MaxRmsUp_last     = m_skipped_MaxRmsUp;
        m_skipped_MaxRmsDown_last   = m_skipped_MaxRmsDown;
        m_skipped_XRms_last         = m_skipped_XRms;
        m_skipped_YRms_last         = m_skipped_YRms;
    }
    else
    {
        m_mnp  = 0;
        m_rmsp = 0;

        m_mnx  = 0;
        m_rmsx = 0;

        m_mny  = 0;
        m_rmsy = 0;
    }

    SHMIMMONITOR_UPDATE_INDI;
    SHMIMMONITORT_UPDATE_INDI( darkShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( refShmimMonitorT );
    FRAMEGRABBER_UPDATE_INDI;

    updateIfChanged( m_indiP_statsTime, "current", m_fitCircBuffMaxTime );

    updatesIfChanged<float>( m_indiP_values,
                             { "max_mean", "max_rms", "x_mean", "x_rms", "y_mean", "y_rms" },
                             { m_mnp, m_rmsp, m_mnx, m_rmsx, m_mny, m_rmsy } );

    updateIfChanged( m_indiP_dx, "current", m_dx );
    updateIfChanged( m_indiP_dy, "current", m_dy );


    return 0;
}

inline int psfFit::appShutdown()
{
    SHMIMMONITOR_APP_SHUTDOWN;
    SHMIMMONITORT_APP_SHUTDOWN( darkShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( refShmimMonitorT );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int psfFit::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    m_image.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );
    m_image.setZero();

    if( m_fitCircBuffMaxLength == 0 || m_fitCircBuffMaxTime == 0 || m_fps <= 0 )
    {
        m_pcb.maxEntries( 0 );
        m_xcb.maxEntries( 0 );
        m_ycb.maxEntries( 0 );

        m_mnp  = 0;
        m_rmsp = 0;

        m_mnx  = 0;
        m_rmsx = 0;

        m_mny  = 0;
        m_rmsy = 0;
    }
    else
    {
        // Set up the fit circ. buffs
        cbIndexT cbSz = m_fitCircBuffMaxTime * m_fps+1;
        if( cbSz > m_fitCircBuffMaxLength )
        {
            cbSz = m_fitCircBuffMaxLength;
        }
        if( cbSz < 3 )
        {
            cbSz = 3; // Make variance meaningful
        }

        std::cerr << "Fit circ. buff size: " << cbSz << ' ' << m_fitCircBuffMaxTime << ' ' << m_fps << '\n';
        m_pcb.maxEntries( cbSz );
        m_xcb.maxEntries( cbSz );
        m_ycb.maxEntries( cbSz );

        m_mnp  = 0;
        m_rmsp = 0;

        m_mnx  = 0;
        m_rmsx = 0;

        m_mny  = 0;
        m_rmsy = 0;
    }

    m_updated = false;
    return 0;
}

inline int psfFit::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    // counters for managing printing of delta-pix skips when shutter closed
    // static int skip_interval = 10;
    // static int last_passed = skip_interval + 1;

    std::unique_lock<std::mutex> lock( m_imageMutex );

    if( m_dark.rows() == m_image.rows() && m_dark.cols() == m_image.cols() )
    {
        if( shmimMonitorT::m_dataType == _DATATYPE_UINT16 )
        {
            for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
            {
                m_image.data()[nn] = ( reinterpret_cast<uint16_t *>( curr_src ) )[nn] - m_dark.data()[nn];
            }
        }
        else if( shmimMonitorT::m_dataType == _DATATYPE_FLOAT )
        {
            for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
            {
                m_image.data()[nn] = ( reinterpret_cast<float *>( curr_src ) )[nn] - m_dark.data()[nn];
            }
        }
    }
    else
    {
        if( shmimMonitorT::m_dataType == _DATATYPE_UINT16 )
        {
            for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
            {
                m_image.data()[nn] = ( reinterpret_cast<uint16_t *>( curr_src ) )[nn];
            }
        }
        else if( shmimMonitorT::m_dataType == _DATATYPE_FLOAT )
        {
            for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
            {
                m_image.data()[nn] = ( reinterpret_cast<float *>( curr_src ) )[nn];
            }
        }
    }

    lock.unlock();

    float max;
    realT local_x = 0;
    realT local_y = 0;
    int   x       = 0;
    int   y       = 0;

    max = m_image.maxCoeff( &x, &y );

    mx::improc::imageCenterOfLight( local_x, local_y, m_image );

    bool local_skipped = false;

    if( m_shutter )
    {
        local_skipped = true; // silently skip when shutter closed
    }

    if( !local_skipped && ( fabs( local_x - x ) > m_deltaPixThresh || fabs( local_y - y ) > m_deltaPixThresh ) )
    {
        ++m_skipped_DeltaFromMax;
        local_skipped = true;

        // We do not add these measurements to stats b/c this means bad PSF
    }

    // still filling circular buffer
    if( !local_skipped && ( m_rmsx == 0 || m_rmsy == 0 || m_rmsp == 0 ) )
    {
        if( m_xcb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( local_x );
            m_ycb.nextEntry( local_y );
        }

        m_last_x = local_x;
        m_last_y = local_y;

        local_skipped = true;
    }

    // The remaining checks are for wild motions but otherwise valid fits (good PSF)

    if( !local_skipped && ( ( max - m_mnp ) / m_rmsp > m_sigmaMaxThreshUp ) )
    {
        if( m_pcb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( local_x );
            m_ycb.nextEntry( local_y );
        }

        m_last_x = local_x;
        m_last_y = local_y;

        ++m_skipped_MaxRmsUp;
        local_skipped = true;
    }

    if( !local_skipped && ( ( m_mnp - max ) / m_mnp > ( 1.0 - m_fractionMaxThreshDown ) ) )
    {
        if( m_pcb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( local_x );
            m_ycb.nextEntry( local_y );
        }

        m_last_x = local_x;
        m_last_y = local_y;

        ++m_skipped_MaxRmsDown;
        local_skipped = true;
    }

    if( !local_skipped && ( fabs( local_x - m_last_x ) / m_rmsx > m_sigmaPixThresh ) )
    {
        if( m_xcb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( local_x );
            m_ycb.nextEntry( local_y );
        }

        m_last_x = local_x;
        m_last_y = local_y;

        ++m_skipped_XRms;
        local_skipped = true;
    }

    if( !local_skipped && ( fabs( local_y - m_last_y ) / m_rmsy > m_sigmaPixThresh ) )
    {
        if( m_ycb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( local_x );
            m_ycb.nextEntry( local_y );
        }

        ++m_skipped_YRms;
        m_last_x = local_x;
        m_last_y = local_y;

        local_skipped = true;
    }

    if( local_skipped )
    {
        if( m_ref.rows() == 2 && m_ref.cols() == 1 )
        {
            local_x = m_ref( 0, 0 );
            local_y = m_ref( 1, 0 );
        }
        else
        {
            local_x = 0;
            local_y = 0;
        }
    }

    if( m_updated == true ) // means the framegrabber hasn't posted the last one yet
    {
        ++m_skipped_updating;
        return 0;
    }

    m_skipped = local_skipped;
    m_x       = local_x;
    m_y       = local_y;

    m_updated = true;

    // signal framegrabber
    // Now tell the f.g. to get going
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
        return -1;
    }

    if( !m_skipped )
    {
        if( m_xcb.maxEntries() > 0 )
        {
            m_pcb.nextEntry( max );
            m_xcb.nextEntry( m_x );
            m_ycb.nextEntry( m_y );
        }

        m_last_x = m_x;
        m_last_y = m_y;
    }

    return 0;
}

int psfFit::allocate( const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    if( darkShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "dark is not float" } );
    }

    m_dark.resize( darkShmimMonitorT::m_width, darkShmimMonitorT::m_height );
    m_dark.setZero();

    return 0;
}

int psfFit::processImage( void *curr_src, const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    std::unique_lock<std::mutex> lock( m_imageMutex );

    for( unsigned nn = 0; nn < darkShmimMonitorT::m_width * darkShmimMonitorT::m_height; ++nn )
    {
        m_dark.data()[nn] = ( reinterpret_cast<float *>( curr_src ) )[nn];
    }

    lock.unlock();

    log<text_log>( "dark updated", logPrio::LOG_INFO );

    return 0;
}

int psfFit::allocate( const refShmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    if( refShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "ref is not float" } );
    }

    m_ref.resize( refShmimMonitorT::m_width, refShmimMonitorT::m_height );
    m_ref.setZero();

    return 0;
}

int psfFit::processImage( void *curr_src, const refShmimT &dummy )
{
    static_cast<void>( dummy );

    std::unique_lock<std::mutex> lock( m_imageMutex );

    for( unsigned nn = 0; nn < refShmimMonitorT::m_width * refShmimMonitorT::m_height; ++nn )
    {
        m_ref.data()[nn] = ( reinterpret_cast<float *>( curr_src ) )[nn];
    }

    lock.unlock();

    log<text_log>( "reference updated", logPrio::LOG_INFO );

    return 0;
}

inline int psfFit::configureAcquisition()
{

    frameGrabberT::m_width    = 2;
    frameGrabberT::m_height   = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    return 0;
}

inline int psfFit::startAcquisition()
{
    return 0;
}

inline int psfFit::acquireAndCheckValid()
{
    timespec ts;

    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
        return -1;
    }

    ts.tv_sec += 1;

    if( sem_timedwait( &m_smSemaphore, &ts ) == 0 )
    {
        if( m_updated )
        {
            clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp );
            return 0;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        return 1;
    }
}

inline int psfFit::loadImageIntoStream( void *dest )
{
    if( !m_skipped )
    {
        ( reinterpret_cast<float *>( dest ) )[0] = m_x - m_dx;
        ( reinterpret_cast<float *>( dest ) )[1] = m_y - m_dy;
    }
    else
    {
        ( reinterpret_cast<float *>( dest ) )[0] = m_x;
        ( reinterpret_cast<float *>( dest ) )[1] = m_y;
    }

    m_updated = false;

    return 0;
}

inline int psfFit::reconfig()
{
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_reset )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reset, ipRecv );

    if( ipRecv.find( "request" ) != true ) // this isn't valid
    {
        return -1;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        log<text_log>( "reset requested", logPrio::LOG_NOTICE );
        shmimMonitorT::m_restart = true;
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_statsTime )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_statsTime, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_statsTime, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_fitCircBuffMaxTime = target;

    shmimMonitorT::m_restart = true;

    log<text_log>( "set statsTime = " + std::to_string( m_fitCircBuffMaxTime ), logPrio::LOG_NOTICE );

    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_deltaPixThresh )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_deltaPixThresh, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_deltaPixThresh, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_deltaPixThresh = target;
    updatesIfChanged<float>( m_indiP_deltaPixThresh, { "current", "target" }, { m_deltaPixThresh, m_deltaPixThresh } );
    log<text_log>( "set deltaPixThresh = " + std::to_string( m_deltaPixThresh ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_sigmaMaxThreshUp )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sigmaMaxThreshUp, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_sigmaMaxThreshUp, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_sigmaMaxThreshUp = target;
    updatesIfChanged<float>(
        m_indiP_sigmaMaxThreshUp, { "current", "target" }, { m_sigmaMaxThreshUp, m_sigmaMaxThreshUp } );

    log<text_log>( "set sigmaMaxThreshUp = " + std::to_string( m_sigmaMaxThreshUp ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_fractionMaxThreshDown )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fractionMaxThreshDown, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_fractionMaxThreshDown, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_fractionMaxThreshDown = target;
    updatesIfChanged<float>(
        m_indiP_fractionMaxThreshDown, { "current", "target" }, { m_fractionMaxThreshDown, m_fractionMaxThreshDown } );

    log<text_log>( "set fractionMaxThreshDown = " + std::to_string( m_fractionMaxThreshDown ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_sigmaPixThresh )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sigmaPixThresh, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_sigmaPixThresh, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_sigmaPixThresh = target;
    updatesIfChanged<float>( m_indiP_sigmaPixThresh, { "current", "target" }, { m_sigmaPixThresh, m_sigmaPixThresh } );

    log<text_log>( "set fractionMaxThreshDown = " + std::to_string( m_sigmaPixThresh ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_dx )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dx, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_dx, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_dx = target;

    log<text_log>( "set dx = " + std::to_string( m_dx ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_NEWCALLBACK_DEFN( psfFit, m_indiP_dy )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dy, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_dy, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_dy = target;

    log<text_log>( "set dy = " + std::to_string( m_dy ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_SETCALLBACK_DEFN( psfFit, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv[m_fpsElement].get<float>();

    if( fabs( fps - m_fps ) > m_fpsTol )
    {
        m_fps = fps;

        shmimMonitorT::m_restart = true;
        frameGrabberT::m_reconfig = true;
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( psfFit, m_indiP_shutter )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_shutter, ipRecv );

    if( ipRecv.find( m_shutterElement ) != true ) // this isn't valid
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    bool shutter;
    if( ipRecv[m_shutterElement].getSwitchState() == pcf::IndiElement::On )
    {
        shutter = true;
    }
    else
    {
        shutter = false;
    }

    if( shutter != m_shutter )
    {
        m_shutter = shutter;

        shmimMonitorT::m_restart = true;
    }

    return 0;
}

inline int psfFit::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int psfFit::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // psfFit_hpp
