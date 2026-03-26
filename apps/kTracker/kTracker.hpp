/** \file kTracker.hpp
 * \brief The MagAO-X K-mirror rotation tracker header file
 *
 * \ingroup kTracker_files
 */

#ifndef kTracker_hpp
#define kTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <cmath>

#include <mx/math/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup kTracker
 * \brief The MagAO-X application to track pupil rotation with the k-mirror.
 *
 * <a href="../handbook/operating/software/apps/kTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup kTracker_files
 * \ingroup kTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X K-mirror tracker.
/**
 * \ingroup kTracker
 */
class kTracker : public MagAOXApp<true>
{

    // Give the test harness access.
    friend class kTracker_test;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    float m_zero{ 0 }; ///< The starting point for the K-mirror at zd = 0.

    int m_sign{ 1 }; ///< The sign to apply to the zenith distance to rotate the K-mirror.

    std::string m_devName{ "stagek" }; ///< The device name of the K-mirror stage.  Default is 'stagek'.
    std::string m_tcsDevName{
        "tcsi" }; ///< The device name of the TCS interface providing 'teldata.zd'.  Default is 'tcsi'.

    float m_updateInterval{ 10 }; ///< The interval at which to update positions, in seconds.  Default is 10 secs.

    ///@}

    bool m_tracking{ false }; ///< True when automatic K-mirror updates are enabled.

    float m_zd{ 0 }; ///< The most recent finite zenith distance received from the TCS interface.

    bool m_haveZD{ false }; ///< True once at least one valid zenith distance has been received.

    double m_lastUpdate{ 0 }; ///< Timestamp of the last stage command dispatched by the tracker.

  public:
    /// Default c'tor.
    kTracker();

    /// D'tor, declared and defined for noexcept.
    ~kTracker() noexcept
    {
    }

    /// Set up configuration entries.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    /// Load configuration values.
    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for kTracker.
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

    /** @name INDI
     *
     * @{
     */
  protected:
    pcf::IndiProperty m_indiP_tracking; ///< The INDI toggle used to enable or disable tracking.

    pcf::IndiProperty m_indiP_teldata; ///< The subscribed TCS property providing zenith distance updates.

    pcf::IndiProperty m_indiP_kpos; ///< The outbound K-mirror stage position command property.

  public:
    /// Handle new tracking toggle requests.
    INDI_NEWCALLBACK_DECL( kTracker, m_indiP_tracking );

    /// Handle incoming telescope data updates.
    INDI_SETCALLBACK_DECL( kTracker, m_indiP_teldata );

    ///@}
};

kTracker::kTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{

    return;
}

void kTracker::setupConfig()
{
    config.add( "k.zero",
                "",
                "k.zero",
                argType::Required,
                "k",
                "zero",
                false,
                "float",
                "The k-mirror zero position.  Default is -40.0." );

    config.add( "k.sign",
                "",
                "k.sign",
                argType::Required,
                "k",
                "sign",
                false,
                "int",
                "The k-mirror rotation sign. Default is +1." );

    config.add( "k.devName",
                "",
                "k.devName",
                argType::Required,
                "k",
                "devName",
                false,
                "string",
                "The device name of the k-mirrorstage.  Default is 'stagek'" );

    config.add( "tcs.devName",
                "",
                "tcs.devName",
                argType::Required,
                "tcs",
                "devName",
                false,
                "string",
                "The device name of the TCS Interface providing 'teldata.zd'.  Default is 'tcsi'" );

    config.add( "tracking.updateInterval",
                "",
                "tracking.updateInterval",
                argType::Required,
                "tracking",
                "updateInterval",
                false,
                "float",
                "The interval at which to update positions, in seconds.  Default is 10 secs." );
}

int kTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_zero, "k.zero" );
    _config( m_sign, "k.sign" );
    _config( m_devName, "k.devName" );

    _config( m_tcsDevName, "tcs.devName" );

    _config( m_updateInterval, "tracking.updateInterval" );

    return 0;
}

void kTracker::loadConfig()
{
    loadConfigImpl( config );
}

int kTracker::appStartup()
{

    createStandardIndiToggleSw( m_indiP_tracking, "tracking" );
    registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK( m_indiP_tracking ) );

    REG_INDI_SETPROP( m_indiP_teldata, m_tcsDevName, "teldata" );

    m_indiP_kpos = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_kpos.setDevice( m_devName );
    m_indiP_kpos.setName( "position" );
    m_indiP_kpos.add( pcf::IndiElement( "target" ) );

    state( stateCodes::READY );

    return 0;
}

int kTracker::appLogic()
{
    const double now = mx::sys::get_curr_time();

    float k = 0;

    { // mutex scope
        std::unique_lock<std::mutex> lock( m_indiMutex, std::try_to_lock );

        if( !lock.owns_lock() )
        {
            return 0;
        }

        if( !m_tracking )
        {
            m_lastUpdate = 0;
            return 0;
        }

        if( !m_haveZD || now - m_lastUpdate <= m_updateInterval )
        {
            return 0;
        }

        k            = m_zero + m_sign * 0.5f * m_zd;
        m_lastUpdate = now;
    } // mutex scope

    if( !std::isfinite( k ) )
    {
        log<software_error>( { __FILE__, __LINE__, "computed non-finite K-mirror target" } );
        return 0;
    }

    std::cerr << "Sending k-mirror to: " << k << "\n";

    if( sendNewProperty( m_indiP_kpos, "target", k ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "failed to send K-mirror target" } );
    }

    return 0;
}

int kTracker::appShutdown()
{
    return 0;
}

INDI_NEWCALLBACK_DEFN( kTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tracking, ipRecv );

    if( !ipRecv.find( "toggle" ) )
        return 0;

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_IDLE );

        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );
            m_tracking   = true;
            m_lastUpdate = 0;
        }

        log<text_log>( "started K-mirror rotation tracking" );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );
            m_tracking   = false;
            m_lastUpdate = 0;
        }

        log<text_log>( "stopped K-mirror rotation tracking" );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( kTracker, m_indiP_teldata )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_teldata, ipRecv );

    if( !ipRecv.find( "zd" ) )
        return 0;

    float zd = 0;

    try
    {
        zd = ipRecv["zd"].get<float>();
    }
    catch( const std::exception &e )
    {
        log<software_error>( { __FILE__, __LINE__, std::string( "exception reading teldata.zd: " ) + e.what() } );
        return 0;
    }
    catch( ... )
    {
        log<software_error>( { __FILE__, __LINE__, "unknown exception reading teldata.zd" } );
        return 0;
    }

    if( !std::isfinite( zd ) )
    {
        log<software_error>( { __FILE__, __LINE__, "received non-finite teldata.zd" } );
        return 0;
    }

    { // mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        m_zd     = zd;
        m_haveZD = true;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // kTracker_hpp
