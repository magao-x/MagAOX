/** \file w2tcsOffloader.hpp
 * \brief The MagAO-X Woofer To Telescope Control System (TCS) offloading manager.
 *
 * \ingroup app_files
 */

#ifndef w2tcsOffloader_hpp
#define w2tcsOffloader_hpp

#include <format>
#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup w2tcsOffloader Woofer to TCS Offloading
 * \brief Monitors the averaged woofer shape, fits Zernikes, and sends it to INDI.
 *
 * <a href="../handbook/operating/software/apps/w2tcsOffloader.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup w2tcsOffloader_files Woofer to TCS Offloading
 * \ingroup w2tcsOffloader
 */

/** MagAO-X application to control offloading the woofer to the TCS.
 *
 * \ingroup w2tcsOffloader
 *
 */
class w2tcsOffloader : public MagAOXApp<true>,
                       public dev::shmimMonitor<w2tcsOffloader>,
                       public dev::telemeter<w2tcsOffloader>
{

    // Give the test harness access.
    friend class w2tcsOffloader_test;

    friend class dev::shmimMonitor<w2tcsOffloader>;
    friend class dev::telemeter<w2tcsOffloader>;

    // The base helper types.
    typedef dev::shmimMonitor<w2tcsOffloader> shmimMonitorT;
    typedef dev::telemeter<w2tcsOffloader>    telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_wZModesPath; ///< Filesystem path to the woofer Zernike basis cube.

    std::string m_wMaskPath; ///< Filesystem path to the mask used for coefficient projection.

    std::vector<std::string>
        m_elNames; ///< INDI element names corresponding to the coefficient vector, formatted as `00` through `99`.

    std::vector<realT> m_zCoeffs; ///< Current coefficient vector sent to INDI and telemetry.

    unsigned m_nModes{
        5 }; ///< Number of low-order modes to retain when offloading, clamped to the loaded cube size at startup.

    float m_norm{ 1.0 }; ///< Mask normalization applied to each coefficient measurement.

    ///@}

    /** \name Offloading State - Data
     * @{
     */
    mx::improc::eigenCube<realT> m_wZModes; ///< Basis cube used to project the incoming woofer image.

    mx::improc::eigenImage<realT> m_woofer; ///< Copy of the most recently processed woofer image.

    mx::improc::eigenImage<realT> m_wMask; ///< Mask selecting valid pixels for the coefficient projection.

    std::vector<realT> m_lastZCoeffs; ///< Last coefficient vector recorded to telemetry.
                                      ///@}

  public:
    /// Default constructor.
    w2tcsOffloader();

    /// Destructor, declared and defined for noexcept.
    ~w2tcsOffloader() noexcept
    {
    }

    /// Set up the application configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load the application configuration.
    virtual void loadConfig();

    /// Start the application and validate the loaded mode cube against the configured mode count.
    virtual int appStartup();

    /// Implementation of the FSM for w2tcsOffloader.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shut down the application.
    virtual int appShutdown();

    /// Allocate image buffers for a new shared-memory image stream.
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process a new woofer image and update offload outputs using the currently allowed mode count.
    int processImage( void              *curr_src, ///< [in] pointer to start of current frame.
                      const dev::shmimT &dummy     ///< [in] tag to differentiate shmimMonitor parents.
    );

    /** \name Telemeter Interface
     * @{
     */
    /// Check whether the telemetry max-interval requires a record.
    int checkRecordTimes();

    /// Record the current coefficient vector for telemetry when requested by the telemeter.
    int recordTelem( const logger::telem_w2tcsoffloader * /**< [in] telemetry tag used for overload resolution */ );

    /// Record the current coefficient vector when it changes or when forced.
    int recordZCoeffs( bool force = false /**< [in] set true to record even if unchanged */ );
    ///@}

  protected:
    /** \name Offloading State
     * @{
     */
    pcf::IndiProperty m_indiP_nModes; ///< INDI property publishing the number of active offload modes.

    pcf::IndiProperty m_indiP_zCoeffs; ///< INDI property publishing the current offload coefficients.
    ///@}
};

inline w2tcsOffloader::w2tcsOffloader() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void w2tcsOffloader::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add( "offload.wZModesPath",
                "",
                "offload.wZModesPath",
                argType::Required,
                "offload",
                "wZModesPath",
                false,
                "string",
                "The path to the woofer Zernike modes." );
    config.add( "offload.wMaskPath",
                "",
                "offload.wMaskPath",
                argType::Required,
                "offload",
                "wMaskPath",
                false,
                "string",
                "Path to the woofer Zernike mode mask." );
    config.add( "offload.nModes",
                "",
                "offload.nModes",
                argType::Required,
                "offload",
                "nModes",
                false,
                "int",
                "Number of modes to offload to the TCS." );
}

inline int w2tcsOffloader::loadConfigImpl( mx::app::appConfigurator &_config )
{

    shmimMonitorT::loadConfig( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_wZModesPath, "offload.wZModesPath" );
    _config( m_wMaskPath, "offload.wMaskPath" );
    _config( m_nModes, "offload.nModes" );

    return 0;
}

inline void w2tcsOffloader::loadConfig()
{
    loadConfigImpl( config );
}

inline int w2tcsOffloader::appStartup()
{

    mx::fits::fitsFile<float> ff;
    mx::error_t               errc = ff.read( m_wZModes, m_wZModesPath );
    if( errc != mx::error_t::noerror )
    {
        return log<text_log, -1>( "Could not open mode cube file", logPrio::LOG_ERROR );
    }

    m_zCoeffs.resize( m_wZModes.planes(), 0 );
    m_lastZCoeffs.resize( m_zCoeffs.size(), std::numeric_limits<realT>::max() );

    if( m_nModes > m_zCoeffs.size() )
    {
        m_nModes = m_zCoeffs.size();
    }

    if( m_zCoeffs.size() > 100 )
    {
        m_shutdown = true;
        return log<text_log, -1>( "w2tcsOffloader supports at most 100 offload modes because INDI element names are "
                                  "formatted with two digits.",
                                  logPrio::LOG_CRITICAL );
    }

    errc = ff.read( m_wMask, m_wMaskPath );
    if( errc != mx::error_t::noerror )
    {
        return log<text_log, -1>( "Could not open mode mask file", logPrio::LOG_ERROR );
    }

    m_norm = m_wMask.sum();

    createROIndiNumber( m_indiP_nModes, "nModes", "number of modes calculated" );
    indi::addNumberElement<unsigned>( m_indiP_nModes, "current", 0, m_zCoeffs.size(), 1, "%d" );
    m_indiP_nModes["current"] = m_nModes;

    registerIndiPropertyReadOnly( m_indiP_nModes );

    createROIndiNumber( m_indiP_zCoeffs, "zCoeffs", "offload coefficients" );

    m_elNames.resize( m_zCoeffs.size() );
    for( size_t n = 0; n < m_zCoeffs.size(); ++n )
    {
        m_elNames[n] = std::format( "{:02}", n );

        indi::addNumberElement<realT>( m_indiP_zCoeffs,
                                       m_elNames[n],
                                       -std::numeric_limits<realT>::max(),
                                       std::numeric_limits<realT>::max(),
                                       0,
                                       "%0.6f" );
        m_indiP_zCoeffs[m_elNames[n]] = 0;
    }

    registerIndiPropertyReadOnly( m_indiP_zCoeffs );

    TELEMETER_APP_STARTUP;

    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::OPERATING );

    return 0;
}

inline int w2tcsOffloader::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    {
        std::unique_lock<std::mutex> lock( m_indiMutex ); // mutex scope

        if( shmimMonitorT::updateINDI() < 0 )
        {
            log<software_error>( { __FILE__, __LINE__ } );
        }
    }

    TELEMETER_APP_LOGIC;

    return 0;
}

inline int w2tcsOffloader::appShutdown()
{
    shmimMonitorT::appShutdown();
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int w2tcsOffloader::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    m_woofer.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );

    return 0;
}

inline int w2tcsOffloader::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    Eigen::Map<mx::improc::eigenImage<realT>> wooferImage(
        static_cast<realT *>( curr_src ), shmimMonitorT::m_width, shmimMonitorT::m_height );

    {
        std::unique_lock<std::mutex> lock( m_indiMutex ); // mutex scope

        m_woofer = wooferImage;

        for( size_t i = 0; i < m_zCoeffs.size(); ++i )
        {
            if( i < m_nModes )
            {
                m_zCoeffs[i] = ( wooferImage * m_wZModes.image( i ) * m_wMask ).sum() / m_norm;
            }
            else
            {
                m_zCoeffs[i] = 0;
            }

            m_indiP_zCoeffs[m_elNames[i]] = m_zCoeffs[i];
        }

        m_indiP_zCoeffs.setState( pcf::IndiProperty::Ok );

        if( m_indiDriver )
        {
            m_indiDriver->sendSetProperty( m_indiP_zCoeffs );
        }
    }

    recordZCoeffs();

    return 0;
}

inline int w2tcsOffloader::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( logger::telem_w2tcsoffloader() );
}

inline int w2tcsOffloader::recordTelem( const logger::telem_w2tcsoffloader * )
{
    return recordZCoeffs( true );
}

inline int w2tcsOffloader::recordZCoeffs( bool force )
{
    std::vector<float> coeffs;
    bool               changed{ false };

    {
        std::unique_lock<std::mutex> lock( m_indiMutex ); // mutex scope

        if( m_lastZCoeffs.size() != m_zCoeffs.size() )
        {
            m_lastZCoeffs.resize( m_zCoeffs.size(), std::numeric_limits<realT>::max() );
        }

        coeffs.resize( m_zCoeffs.size() );

        for( size_t n = 0; n < m_zCoeffs.size(); ++n )
        {
            coeffs[n] = m_zCoeffs[n];

            if( m_lastZCoeffs[n] != m_zCoeffs[n] )
            {
                changed = true;
            }
        }

        if( force || changed )
        {
            for( size_t n = 0; n < m_lastZCoeffs.size(); ++n )
            {
                m_lastZCoeffs[n] = m_zCoeffs[n];
            }
        }
    }

    if( force || changed )
    {
        telem<logger::telem_w2tcsoffloader>( logger::telem_w2tcsoffloader::messageT( coeffs ) );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // w2tcsOffloader_hpp
