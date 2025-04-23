/** \file cameraSim.hpp
 * \brief The MagAO-X camera simulator.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cameraSim_files
 */

#ifndef cameraSim_hpp
#define cameraSim_hpp

// #include <ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <mx/math/func/gaussian.hpp>
#include <mx/math/randomT.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup cameraSim Camera Simulator
 * \brief A camera simulator
 *
 * <a href="../handbook/operating/software/apps/cameraSim.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup cameraSim_files Camera Simulator Files
 * \ingroup cameraSim
 */

/** MagAO-X application to simulate a camera
 *
 * \ingroup cameraSim
 *
 */
class cameraSim : public MagAOXApp<>,
                  public dev::stdCamera<cameraSim>,
                  public dev::frameGrabber<cameraSim>,
                  public dev::telemeter<cameraSim>
{

    friend class dev::stdCamera<cameraSim>;
    friend class dev::frameGrabber<cameraSim>;
    friend class dev::telemeter<cameraSim>;

  public:
    typedef dev::stdCamera<cameraSim> stdCameraT;
    typedef dev::frameGrabber<cameraSim> frameGrabberT;
    typedef dev::telemeter<cameraSim> telemeterT;

    /** \name app::dev Configurations
     *@{
     */
    static constexpr bool c_stdCamera_tempControl =
        true; ///< app::dev config to tell stdCamera to not expose temperature controls

    static constexpr bool c_stdCamera_temp = true; ///< app::dev config to tell stdCamera to expose temperature

    static constexpr bool c_stdCamera_readoutSpeed =
        true; ///< app::dev config to tell stdCamera not to  expose readout speed controls

    static constexpr bool c_stdCamera_vShiftSpeed =
        true; ///< app:dev config to tell stdCamera not to expose vertical shift speed control

    static constexpr bool c_stdCamera_emGain =
        true; ///< app::dev config to tell stdCamera to not expose EM gain controls

    static constexpr bool c_stdCamera_exptimeCtrl =
        true; ///< app::dev config to tell stdCamera to expose exposure time controls

    static constexpr bool c_stdCamera_fpsCtrl = true; ///< app::dev config to tell stdCamera to expose FPS controls

    static constexpr bool c_stdCamera_fps =
        true; ///< app::dev config to tell stdCamera not to expose FPS status (ignored since fpsCtrl=true)

    static constexpr bool c_stdCamera_synchro =
        true; ///< app::dev config to tell stdCamera to not expose synchro mode controls

    static constexpr bool c_stdCamera_usesModes =
        false; ///< app:dev config to tell stdCamera not to expose mode controls

    static constexpr bool c_stdCamera_usesROI = true; ///< app:dev config to tell stdCamera to expose ROI controls

    static constexpr bool c_stdCamera_cropMode =
        true; ///< app:dev config to tell stdCamera not to expose Crop Mode controls

    static constexpr bool c_stdCamera_hasShutter =
        true; ///< app:dev config to tell stdCamera to expose shutter controls

    static constexpr bool c_stdCamera_usesStateString =
        true; ///< app::dev confg to tell stdCamera to expose the state string property

    static constexpr bool c_frameGrabber_flippable =
        true; ///< app:dev config to tell framegrabber that this camera can be flipped

    ///@}

  protected:
    mx::improc::eigenImage<int16_t> m_fgimage;

    double m_lastTime{ 0 };
    double m_offset = { 0 };

    float m_bias{ 500 }; ///< the simulated bias level. default 500.
    float m_ron{ 5 };    ///< the simulated readout noise, in counts/read. default 5.

    std::vector<float> m_xcen{ 0 }; /**< the simulated star x center coordinate, relative to image center.
                                         one per star. default 0*/
    std::vector<float> m_ycen{ 0 }; /**< the simulated star y center coordinate, relative to image center.
                                         one per star. default 0.*/
    std::vector<float> m_peak{ 2000 }; ///< the simulated star peak, in counts/second. one per star. default 2000.
    float m_fwhm{ 2 };                 ///< the simulated star FWHM in pixels. the same for all stars. default 2.

    float m_jitter{ 0.1 }; ///< the simulated jitter, in pixels/read. the same for all stars. default 0.1.

    mx::math::normDistT<float> m_norm;

  public:
    /// Default c'tor
    cameraSim();

    /// Destructor
    ~cameraSim() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual int loadConfigImpl( mx::app::appConfigurator &cfg );

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Startup functions
    /** Sets up the INDI vars.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for the Siglent SDG
    virtual int appLogic();

    /// Do any needed shutdown tasks.  Currently nothing in this app.
    virtual int appShutdown();

    int configureAcquisition();
    int startAcquisition();
    int acquireAndCheckValid();
    int loadImageIntoStream( void *dest );
    int reconfig();

  protected:
    float fps();

    /** \name stdCamera Interface
     *
     * @{
     */

    /// Set defaults for a power on state.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int powerOnDefaults();

    int setTempControl();

    int setTempSetPt();

    int setReadoutSpeed();

    int setVShiftSpeed();

    /// Set the Exposure Time. [stdCamera interface]
    /** Sets the frame rate to m_expTimeSet.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setExpTime();

    /// Set the framerate.
    /**
     * \returns 0 always
     */
    int setFPS();

    int setSynchro();

    int setEMGain();

    /// Check the next ROI
    /** Checks if the target values are valid and adjusts them to the closest valid values if needed.
     *
     * \returns 0 always
     */
    int checkNextROI();

    int setCropMode();

    /// Set the next ROI
    /**
     * \returns 0 always
     */
    int setNextROI();

    int setShutter( int ss );

    std::string stateString();

    bool stateStringValid();

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */

    int checkRecordTimes();

    int recordTelem( const telem_stdcam * );

    ///@}
};

inline cameraSim::cameraSim() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = true;

    m_defaultReadoutSpeed = "1";
    m_defaultVShiftSpeed = "1";

    m_default_x = 511.5;
    m_default_y = 511.5;
    m_default_w = 1024;
    m_default_h = 1024;

    m_nextROI.x = m_default_x;
    m_nextROI.y = m_default_y;
    m_nextROI.w = m_default_w;
    m_nextROI.h = m_default_h;
    m_nextROI.bin_x = 1;
    m_nextROI.bin_y = 1;

    m_full_x = 511.5;
    m_full_y = 511.5;
    m_full_w = 1024;
    m_full_h = 1024;

    return;
}

inline cameraSim::~cameraSim() noexcept
{
    return;
}

inline void cameraSim::setupConfig()
{
    dev::stdCamera<cameraSim>::setupConfig( config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );

    config.add( "camsim.fullW",
                "",
                "camsim.fullW",
                argType::Required,
                "camsim",
                "fullW",
                false,
                "int",
                "Full (maximum) width of the simulated detector" );

    config.add( "camsim.fullH",
                "",
                "camsim.fullH",
                argType::Required,
                "camsim",
                "fullH",
                false,
                "int",
                "Full (maximum) height of the simulated detector" );

    config.add( "camsim.defaultFPS",
                "",
                "camsim.defaultFPS",
                argType::Required,
                "camsim",
                "defaultFPS",
                false,
                "float",
                "the camera default FPS, set at startup.  Default is 10" );

    config.add( "camsim.bias",
                "",
                "camsim.bias",
                argType::Required,
                "camsim",
                "bias",
                false,
                "float",
                "the simulated bias level. default 500." );

    config.add( "camsim.ron",
                "",
                "camsim.ron",
                argType::Required,
                "camsim",
                "ron",
                false,
                "float",
                "the simulated readout noise, in counts/read. default 5." );

    config.add( "camsim.xcen",
                "",
                "camsim.xcen",
                argType::Required,
                "camsim",
                "xcen",
                false,
                "vector<float>",
                "the simulated star x center coordinate, relative to image center. one per star. default 0." );

    config.add( "camsim.ycen",
                "",
                "camsim.ycen",
                argType::Required,
                "camsim",
                "ycen",
                false,
                "vector<float>",
                "the simulated star y center coordinate, relative to image center. one per star. default 0." );

    config.add( "camsim.peak",
                "",
                "camsim.peak",
                argType::Required,
                "camsim",
                "peak",
                false,
                "vector<float>",
                "the simulated star peak, in counts/second. one per star. default 2000." );

    config.add( "camsim.fwhm",
                "",
                "camsim.fwhm",
                argType::Required,
                "camsim",
                "fwhm",
                false,
                "float",
                "the simulated star FWHM in pixels. the same for all stars. default 2." );

    config.add( "camsim.jitter",
                "",
                "camsim.jitter",
                argType::Required,
                "camsim",
                "jitter",
                false,
                "float",
                "the simulated jitter, in pixels/read. the same for all stars. default 0.1." );
}

inline int cameraSim::loadConfigImpl( mx::app::appConfigurator &cfg )
{
    dev::stdCamera<cameraSim>::loadConfig( cfg );

    FRAMEGRABBER_LOAD_CONFIG( cfg );

    TELEMETER_LOAD_CONFIG( cfg );

    cfg( m_full_w, "camsim.fullW" );
    cfg( m_full_h, "camsim.fullH" );

    m_full_x = 0.5 * ( m_full_w - 1.0 );
    m_full_y = 0.5 * ( m_full_h - 1.0 );

    if( m_default_w > m_full_w || m_default_x > m_full_w )
    {
        m_default_w = m_full_w;
        m_default_x = m_full_x;
    }

    if( m_default_h > m_full_h || m_default_y > m_full_h )
    {
        m_default_h = m_full_h;
        m_default_y = m_full_y;
    }

    m_fps = 10;
    cfg( m_fps, "camsim.defaultFPS" );
    m_fpsSet = m_fps;

    config( m_bias, "camsim.bias" );
    config( m_ron, "camsim.ron" );

    config( m_xcen, "camsim.xcen" );
    config( m_ycen, "camsim.ycen" );
    config( m_peak, "camsim.peak" );

    if( m_xcen.size() != m_ycen.size() || m_xcen.size() != m_peak.size() )
    {
        log<software_error>( { __FILE__, __LINE__, "cameraSim: xcen, ycen, and peak must be the same size." } );
    }

    config( m_fwhm, "camsim.fwhm" );

    config( m_jitter, "camsim.jitter" );
    return 0;
}

inline void cameraSim::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
    }
}

inline int cameraSim::appStartup()
{

    //=================================
    // Do camera configuration here

    m_ccdTemp = -40;
    m_ccdTempSetpt = -40;

    m_readoutSpeedNames = { "one", "two", "three" };
    m_readoutSpeedNameLabels = { "One", "Two", "Three" };
    m_readoutSpeedName = m_readoutSpeedNames[0];

    m_vShiftSpeedNames = { "0.1", "0.2", "0.3" };
    m_vShiftSpeedNameLabels = { "0.1 Hz", "0.2 kHz", "0.4 MhZ" };
    m_vShiftSpeedName = m_vShiftSpeedNames[0];

    m_shutterStatus = "READY";
    m_shutterState = 0;

    if( dev::stdCamera<cameraSim>::appStartup() < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    FRAMEGRABBER_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    m_currentROI.x = m_default_x;
    m_currentROI.y = m_default_y;
    m_currentROI.w = m_default_w;
    m_currentROI.h = m_default_h;
    m_currentROI.bin_x = 1;
    m_currentROI.bin_y = 1;
    m_nextROI = m_currentROI;

    m_expTime = 1.0 / m_fps;
    m_expTimeSet = m_expTime;

    m_lastTime = mx::sys::get_curr_time();
    m_offset = 0;

    state( stateCodes::OPERATING );

    return 0;
}

inline int cameraSim::appLogic()
{
    state( stateCodes::OPERATING );

    // and run stdCamera's appLogic
    if( dev::stdCamera<cameraSim>::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // and run frameGrabber's appLogic to see if the f.g. thread has exited.
    FRAMEGRABBER_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
    {
        // Get a lock if we can
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        // but don't wait for it, just go back around.
        if( !lock.owns_lock() )
            return 0;

        if( stdCamera<cameraSim>::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
            state( stateCodes::ERROR );
            return 0;
        }

        FRAMEGRABBER_UPDATE_INDI;
    }

    ///\todo Fall through check?

    return 0;
}

inline int cameraSim::appShutdown()
{

    dev::stdCamera<cameraSim>::appShutdown();

    FRAMEGRABBER_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int cameraSim::configureAcquisition()
{
    try
    {
        recordCamera( true );

        m_currentROI.x = m_nextROI.x;
        m_currentROI.y = m_nextROI.y;
        m_currentROI.w = m_nextROI.w;
        m_currentROI.h = m_nextROI.h;
        m_currentROI.bin_x = m_nextROI.bin_x;
        m_currentROI.bin_y = m_nextROI.bin_y;

        m_width = m_currentROI.w / m_currentROI.bin_x;
        m_height = m_currentROI.h / m_currentROI.bin_y;
        m_xbinning = m_currentROI.bin_x;
        m_ybinning = m_currentROI.bin_y;

        m_fgimage.resize( m_width, m_height );

        m_dataType = IMAGESTRUCT_UINT16;
        m_typeSize = imageStructDataType<IMAGESTRUCT_UINT16>::size;

        recordCamera( true );
    }
    catch( ... )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid ROI specifications" } );
        state( stateCodes::NOTCONNECTED );
        return -1;
    }

    return 0;
}

int cameraSim::startAcquisition()
{

    m_offset = 0;
    m_lastTime = mx::sys::get_curr_time();

    state( stateCodes::OPERATING );

    return 0;
}

int cameraSim::acquireAndCheckValid()
{
    double et = mx::sys::get_curr_time() - m_lastTime;
    while( et <= m_expTime - m_offset )
    {
        mx::sys::nanoSleep( et * 1e6 );
        et = mx::sys::get_curr_time() - m_lastTime;
    }

    double dt = mx::sys::get_curr_time( m_currImageTimestamp );

    m_offset += 0.1 * ( ( dt - m_lastTime ) - m_expTime );

    m_lastTime = dt;

    for( uint32_t cc = 0; cc < m_height; ++cc )
    {
        float y = ( cc - 0.5 * ( 1.0 * m_height - 1.0 ) ) +
                  ( m_currentROI.y - 0.5 * ( 1.0 * m_full_h - 1.0 ) ); // Y position relative to the full array

        for( uint32_t rr = 0; rr < m_width; ++rr )
        {
            float x = ( rr - 0.5 * ( 1.0 * m_width - 1.0 ) ) +
                      ( m_currentROI.x - 0.5 * ( 1.0 * m_full_w - 1.0 ) ); // X position realtive to the full array

            m_fgimage( rr, cc ) = m_bias + m_norm * m_ron;

            if( m_shutterState == 1 )
            {
                for( size_t s = 0; s < m_xcen.size(); ++s )
                {
                    if( ( fabs( y - m_ycen[s] ) < 4 * m_fwhm ) && ( fabs( x - m_xcen[s] ) < 4 * m_fwhm ) )
                    {
                        float xcen = m_xcen[s] + m_norm * m_jitter;
                        float ycen = m_ycen[s] + m_norm * m_jitter;

                        float flux = mx::math::func::gaussian2D<float>(
                            x, y, 0.0, m_peak[s] * m_expTime, xcen, ycen, mx::math::func::fwhm2sigma( m_fwhm ) );
                        flux += m_norm * sqrt( flux );

                        m_fgimage( rr, cc ) += flux;
                    }
                }
            }
        }
    }

    return 0;
}

int cameraSim::loadImageIntoStream( void *dest )
{

    if( frameGrabber<cameraSim>::loadImageIntoStreamCopy(
            dest, m_fgimage.data(), m_width, m_height, sizeof( uint16_t ) ) == nullptr )
    {
        return -1;
    }

    m_imageStream->md->atime = m_imageStream->md->writetime;

    return 0;
}

int cameraSim::reconfig()
{

    return 0;
}

inline float cameraSim::fps()
{
    return m_fps;
}

inline int cameraSim::powerOnDefaults()
{
    m_nextROI.x = m_default_x;
    m_nextROI.y = m_default_y;
    m_nextROI.w = m_default_w;
    m_nextROI.h = m_default_h;
    m_nextROI.bin_x = m_default_bin_x;
    m_nextROI.bin_y = m_default_bin_y;

    return 0;
}

inline int cameraSim::setTempControl()
{
    m_tempControlStatus = m_tempControlStatusSet;
    return 0;
}

inline int cameraSim::setTempSetPt()
{
    m_ccdTemp = m_ccdTempSetpt;
    return 0;
}

inline int cameraSim::setReadoutSpeed()
{
    m_readoutSpeedName = m_readoutSpeedNameSet;
    return 0;
}

inline int cameraSim::setVShiftSpeed()
{
    m_vShiftSpeedName = m_vShiftSpeedNameSet;
    return 0;
}

inline int cameraSim::setExpTime()
{

    m_expTime = m_expTimeSet;
    m_fps = 1. / m_fps;
    m_fpsSet = m_fps;

    log<text_log>( "Set exposure time: " + std::to_string( m_expTimeSet ) + " sec" );

    m_reconfig = true;

    return 0;
}

inline int cameraSim::setFPS()
{
    recordCamera( true );

    m_fps = m_fpsSet;
    m_expTime = 1.0 / m_fps;
    m_expTimeSet = m_expTime;

    log<text_log>( "Set frame rate: " + std::to_string( m_fps ) + " fps" );

    m_reconfig = true;

    return 0;
}

inline int cameraSim::setSynchro()
{
    m_synchro = m_synchroSet;
    return 0;
}

inline int cameraSim::setEMGain()
{
    m_emGain = m_emGainSet;
    return 0;
}

inline int cameraSim::checkNextROI()
{

    updateIfChanged( m_indiP_roi_x, "target", m_nextROI.x, INDI_OK );
    updateIfChanged( m_indiP_roi_y, "target", m_nextROI.y, INDI_OK );
    updateIfChanged( m_indiP_roi_w, "target", m_nextROI.w, INDI_OK );
    updateIfChanged( m_indiP_roi_h, "target", m_nextROI.h, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_x, "target", m_nextROI.bin_x, INDI_OK );
    updateIfChanged( m_indiP_roi_bin_y, "target", m_nextROI.bin_y, INDI_OK );

    return 0;
}

inline int cameraSim::setNextROI()
{
    m_reconfig = true;

    updateSwitchIfChanged( m_indiP_roi_set, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_full, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_last, "request", pcf::IndiElement::Off, INDI_IDLE );
    updateSwitchIfChanged( m_indiP_roi_default, "request", pcf::IndiElement::Off, INDI_IDLE );
    return 0;
}

inline int cameraSim::setCropMode()
{
    m_cropMode = m_cropModeSet;
    return 0;
}

inline int cameraSim::setShutter( int ss )
{
    m_shutterState = ss;

    return 0;
}

inline std::string cameraSim::stateString()
{
    return "stateString";
}

inline bool cameraSim::stateStringValid()
{
    return true;
}

inline int cameraSim::checkRecordTimes()
{
    return telemeter<cameraSim>::checkRecordTimes( telem_stdcam() );
}

inline int cameraSim::recordTelem( const telem_stdcam * )
{
    return recordCamera( true );
}

} // namespace app
} // namespace MagAOX
#endif
