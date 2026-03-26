/** \file adcTracker.hpp
 * \brief The MagAO-X ADC Tracker header file
 *
 * \ingroup adcTracker_files
 */

#ifndef adcTracker_hpp
#define adcTracker_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <cmath>

#include <mx/math/gslInterpolator.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup adcTracker
 * \brief The MagAO-X application to track sky rotation with the atmospheric dispersion corrector.
 *
 * <a href="../handbook/operating/software/apps/adcTracker.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup adcTracker_files
 * \ingroup adcTracker
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/**
 * \ingroup adcTracker
 */
class adcTracker : public MagAOXApp<true>
{

    // Give the test harness access.
    friend class adcTracker_test;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    // here add parameters which will be config-able at runtime
    std::string m_lookupFile{ "adc_lookup_table.txt" }; ///< The name of the file, in the calib directory, containing
                                                        ///< the ADC lookup table.  Default is 'adc_lookup_table.txt'.

    float m_adc1zero{ 0 }; ///< The starting point for ADC 1. Default is 0.

    int m_adc1lupsign{ 1 }; ///< The sign to apply to the lookup table value for ADC 1.

    float m_adc2zero{ 0 }; ///< The starting point for ADC 2. Default is 0.

    int m_adc2lupsign{ 1 }; ///< The sign to apply to the lookup table value for ADC 2.

    float m_deltaAngle{ 0 }; ///< The offset angle to apply to the looked-up values, applied to both.  Default is 0.

    float m_adc1delta{ 0 }; ///< The offset angle to apply to the looked-up value for ADC 1, applied in addition to
                            ///< deltaAngle.  Default is 0.

    float m_adc2delta{ 0 }; ///< The offset angle to apply to the looked-up value for ADC 2, applied in addition to
                            ///< deltaAngle.  Default is 0.

    float m_minZD{ 5.1 }; ///< The minimum zenith distance at which to interpolate and move the ADCs.  Default is 5.1.

    std::string m_adc1DevName{ "stageadc1" }; ///< The device name of the ADC 1 stage.  Default is 'stageadc1'.
    std::string m_adc2DevName{ "stageadc2" }; ///< The device name of the ADC 2 stage.  Default is 'stageadc2'.

    std::string m_tcsDevName{
        "tcsi" }; ///< The device name of the TCS interface providing 'teldata.zd'.  Default is 'tcsi'.

    float m_updateInterval{ 10 }; ///< The interval at which to update positions, in seconds.  Default is 10 secs.

    ///@}

    float m_maxZD{ 0 }; ///< The maximum zenith distance represented by the loaded lookup table.

    std::vector<double> m_lupZD;   ///< Lookup-table zenith-distance samples.
    std::vector<double> m_lupADC1; ///< Lookup-table ADC 1 offsets corresponding to m_lupZD.
    std::vector<double> m_lupADC2; ///< Lookup-table ADC 2 offsets corresponding to m_lupZD.

    mx::math::gslInterpolator<mx::math::gsl_interp_linear<double>>
        m_terpADC1; ///< ADC 1 interpolator built from the lookup table.
    mx::math::gslInterpolator<mx::math::gsl_interp_linear<double>>
        m_terpADC2; ///< ADC 2 interpolator built from the lookup table.

    bool m_lookupReady{ false }; ///< True once the ADC lookup table has been validated and the interpolators are ready.

    bool m_tracking{ false }; ///< True when automatic ADC updates are enabled.

    float m_zd{ 0 }; ///< The most recent finite zenith distance received from the TCS interface.

    bool m_haveZD{ false }; ///< True once at least one valid zenith distance has been received.

    double m_lastUpdate{ 0 }; ///< Timestamp of the last ADC command dispatched by the tracker.

  public:
    /// Default c'tor.
    adcTracker();

    /// D'tor, declared and defined for noexcept.
    ~adcTracker() noexcept
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

    /// Implementation of the FSM for adcTracker.
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

    pcf::IndiProperty m_indiP_deltaAngle; ///< The shared user offset applied to both ADC targets.
    pcf::IndiProperty m_indiP_deltaADC1;  ///< The user offset applied only to ADC 1.
    pcf::IndiProperty m_indiP_deltaADC2;  ///< The user offset applied only to ADC 2.

    pcf::IndiProperty m_indiP_minZD; ///< The user-configurable minimum zenith distance for interpolation.

    pcf::IndiProperty m_indiP_teldata; ///< The subscribed TCS property providing zenith distance updates.

    pcf::IndiProperty m_indiP_adc1pos; ///< The outbound ADC 1 stage position command property.
    pcf::IndiProperty m_indiP_adc2pos; ///< The outbound ADC 2 stage position command property.

  public:
    /// Handle new tracking toggle requests.
    INDI_NEWCALLBACK_DECL( adcTracker, m_indiP_tracking );

    /// Handle new shared delta-angle requests.
    INDI_NEWCALLBACK_DECL( adcTracker, m_indiP_deltaAngle );
    /// Handle new ADC 1 delta-angle requests.
    INDI_NEWCALLBACK_DECL( adcTracker, m_indiP_deltaADC1 );
    /// Handle new ADC 2 delta-angle requests.
    INDI_NEWCALLBACK_DECL( adcTracker, m_indiP_deltaADC2 );

    /// Handle new minimum-zenith-distance requests.
    INDI_NEWCALLBACK_DECL( adcTracker, m_indiP_minZD );

    /// Handle incoming telescope data updates.
    INDI_SETCALLBACK_DECL( adcTracker, m_indiP_teldata );

    ///@}
};

adcTracker::adcTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{

    return;
}

void adcTracker::setupConfig()
{
    config.add( "adcs.lookupFile",
                "",
                "adcs.lookupFile",
                argType::Required,
                "adcs",
                "lookupFile",
                false,
                "string",
                "The name of the file, in the calib directory, containing the adc lookup table.  Default is "
                "'adc_lookup_table.txt'." );

    config.add( "adcs.adc1zero",
                "",
                "adcs.adc1zero",
                argType::Required,
                "adcs",
                "adc1zero",
                false,
                "float",
                "The starting point for ADC 1. Default is 0." );

    config.add( "adcs.adc1lupsign",
                "",
                "adcs.adc1lupsign",
                argType::Required,
                "adcs",
                "adc1lupsign",
                false,
                "int",
                "The sign to apply for the LUP values for ADC 1. Default is +1." );

    config.add( "adcs.adc2zero",
                "",
                "adcs.adc2zero",
                argType::Required,
                "adcs",
                "adc2zero",
                false,
                "float",
                "The starting point for ADC 2. Default is 0." );

    config.add( "adcs.adc2lupsign",
                "",
                "adcs.adc2lupsign",
                argType::Required,
                "adcs",
                "adc2lupsign",
                false,
                "int",
                "The sign to apply for the LUP values for ADC 2. Default is +1." );

    config.add( "adcs.deltaAngle",
                "",
                "adcs.deltaAngle",
                argType::Required,
                "adcs",
                "deltaAngle",
                false,
                "float",
                "The offset angle to apply to the looked-up values, applied to both.  Default is 0." );

    config.add( "adcs.adc1delta",
                "",
                "adcs.adc1delta",
                argType::Required,
                "adcs",
                "adc1delta",
                false,
                "float",
                "The offset angle to apply to the looked-up value for ADC 1, applied in addition to deltaAngle.  "
                "Default is 0." );

    config.add( "adcs.adc2delta",
                "",
                "adcs.adc2delta",
                argType::Required,
                "adcs",
                "adc2delta",
                false,
                "float",
                "The offset angle to apply to the looked-up value for ADC 2, applied in addition to deltaAngle.  "
                "Default is 0." );

    config.add( "adcs.minZD",
                "",
                "adcs.minZD",
                argType::Required,
                "adcs",
                "minZD",
                false,
                "float",
                "The minimum zenith distance at which to interpolate and move the ADCs.  Default is 5.1" );

    config.add( "adcs.adc1DevName",
                "",
                "adcs.adc1devName",
                argType::Required,
                "adcs",
                "adc1DevName",
                false,
                "string",
                "The device name of the ADC 1 stage.  Default is 'stageadc1'" );

    config.add( "adcs.adc2DevName",
                "",
                "adcs.adc2devName",
                argType::Required,
                "adcs",
                "adc2DevName",
                false,
                "string",
                "The device name of the ADC 2 stage.  Default is 'stageadc2'" );

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

int adcTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_lookupFile, "adcs.lookupFile" );
    _config( m_adc1zero, "adcs.adc1zero" );
    _config( m_adc1lupsign, "adcs.adc1lupsign" );
    _config( m_adc2zero, "adcs.adc2zero" );
    _config( m_adc2lupsign, "adcs.adc2lupsign" );
    _config( m_deltaAngle, "adcs.deltaAngle" );
    _config( m_adc1delta, "adcs.adc1delta" );
    _config( m_adc2delta, "adcs.adc2delta" );
    _config( m_minZD, "adcs.minZD" );
    _config( m_adc1DevName, "adcs.adc1DevName" );
    _config( m_adc2DevName, "adcs.adc2DevName" );

    _config( m_tcsDevName, "tcs.devName" );

    _config( m_updateInterval, "tracking.updateInterval" );

    return 0;
}

void adcTracker::loadConfig()
{
    loadConfigImpl( config );
}

int adcTracker::appStartup()
{

    std::string luppath = m_calibDir + "/" + m_lookupFile;

    std::cerr << "Reading " << luppath << "\n";

    if( mx::ioutils::readColumns<mx::ioutils::readColCommaDelim>( luppath, m_lupZD, m_lupADC1, m_lupADC2 ) !=
        mx::error_t::noerror )
    {
        log<software_critical>( { __FILE__, __LINE__, "error reading lookup table from " + luppath } );
        return -1;
    }

    if( m_lupZD.size() != m_lupADC1.size() || m_lupZD.size() != m_lupADC2.size() )
    {
        log<software_critical>( { __FILE__, __LINE__, "inconsistent sizes in " + luppath } );
        return -1;
    }

    if( m_lupZD.size() < 2 )
    {
        log<software_critical>( { __FILE__, __LINE__, "lookup table must contain at least two rows in " + luppath } );
        return -1;
    }

    for( size_t n = 0; n < m_lupZD.size(); ++n )
    {
        if( !std::isfinite( m_lupZD[n] ) || !std::isfinite( m_lupADC1[n] ) || !std::isfinite( m_lupADC2[n] ) )
        {
            log<software_critical>(
                { __FILE__,
                  __LINE__,
                  "non-finite lookup table value at row " + std::to_string( n ) + " in " + luppath } );
            return -1;
        }

        if( n > 0 && m_lupZD[n] <= m_lupZD[n - 1] )
        {
            log<software_critical>(
                { __FILE__, __LINE__, "lookup table zenith distances must be strictly increasing in " + luppath } );
            return -1;
        }
    }

    log<text_log>( "Read " + std::to_string( m_lupZD.size() ) + " points from " + m_lookupFile );

    try
    {
        m_terpADC1.setup( m_lupZD, m_lupADC1 );
        m_terpADC2.setup( m_lupZD, m_lupADC2 );
    }
    catch( const std::exception &e )
    {
        log<software_critical>(
            { __FILE__,
              __LINE__,
              std::string( "exception setting up ADC interpolators from " ) + luppath + ": " + e.what() } );
        return -1;
    }
    catch( ... )
    {
        log<software_critical>(
            { __FILE__, __LINE__, "unknown exception setting up ADC interpolators from " + luppath } );
        return -1;
    }

    m_maxZD       = static_cast<float>( m_lupZD.back() );
    m_lookupReady = true;

    createStandardIndiToggleSw( m_indiP_tracking, "tracking" );
    registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK( m_indiP_tracking ) );

    createStandardIndiNumber<float>( m_indiP_deltaAngle, "deltaAngle", -180.0, 180.0, 0, "%0.2f" );
    m_indiP_deltaAngle["target"].set( m_deltaAngle );
    m_indiP_deltaAngle["current"].set( m_deltaAngle );
    registerIndiPropertyNew( m_indiP_deltaAngle, INDI_NEWCALLBACK( m_indiP_deltaAngle ) );

    createStandardIndiNumber<float>( m_indiP_deltaADC1, "deltaADC1", -180.0, 180.0, 0, "%0.2f" );
    m_indiP_deltaADC1["target"].set( m_adc1delta );
    m_indiP_deltaADC1["current"].set( m_adc1delta );
    registerIndiPropertyNew( m_indiP_deltaADC1, INDI_NEWCALLBACK( m_indiP_deltaADC1 ) );

    createStandardIndiNumber<float>( m_indiP_deltaADC2, "deltaADC2", -180.0, 180.0, 0, "%0.2f" );
    m_indiP_deltaADC2["target"].set( m_adc2delta );
    m_indiP_deltaADC2["current"].set( m_adc2delta );
    registerIndiPropertyNew( m_indiP_deltaADC2, INDI_NEWCALLBACK( m_indiP_deltaADC2 ) );

    createStandardIndiNumber<float>( m_indiP_minZD, "minZD", 0.0, 90.0, 0, "%0.2f" );
    m_indiP_minZD["target"].set( m_minZD );
    m_indiP_minZD["current"].set( m_minZD );
    registerIndiPropertyNew( m_indiP_minZD, INDI_NEWCALLBACK( m_indiP_minZD ) );

    REG_INDI_SETPROP( m_indiP_teldata, m_tcsDevName, "teldata" );

    m_indiP_adc1pos = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_adc1pos.setDevice( m_adc1DevName );
    m_indiP_adc1pos.setName( "position" );
    m_indiP_adc1pos.add( pcf::IndiElement( "target" ) );

    m_indiP_adc2pos = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_adc2pos.setDevice( m_adc2DevName );
    m_indiP_adc2pos.setName( "position" );
    m_indiP_adc2pos.add( pcf::IndiElement( "target" ) );

    state( stateCodes::READY );

    return 0;
}

int adcTracker::appLogic()
{
    const double now = mx::sys::get_curr_time();

    float zd          = 0;
    float minZD       = 0;
    float deltaAngle  = 0;
    float adc1delta   = 0;
    float adc2delta   = 0;
    float adc1zero    = 0;
    float adc2zero    = 0;
    float maxZD       = 0;
    int   adc1lupsign = 1;
    int   adc2lupsign = 1;

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

        if( !m_lookupReady || !m_haveZD || now - m_lastUpdate <= m_updateInterval )
        {
            return 0;
        }

        zd          = m_zd;
        minZD       = m_minZD;
        deltaAngle  = m_deltaAngle;
        adc1delta   = m_adc1delta;
        adc2delta   = m_adc2delta;
        adc1zero    = m_adc1zero;
        adc2zero    = m_adc2zero;
        adc1lupsign = m_adc1lupsign;
        adc2lupsign = m_adc2lupsign;
        maxZD       = m_maxZD;

        m_lastUpdate = now;
    } // mutex scope

    if( !std::isfinite( zd ) )
    {
        log<software_error>( { __FILE__, __LINE__, "received non-finite zenith distance in ADC tracker" } );
        return 0;
    }

    float dadc1 = 0.0;
    float dadc2 = 0.0;

    if( zd > maxZD )
    {
        std::cerr << "end of lup\n";
        dadc1 = static_cast<float>( m_lupADC1.back() );
        dadc2 = static_cast<float>( m_lupADC2.back() );
    }
    else if( zd >= minZD )
    {
        try
        {
            dadc1 = static_cast<float>( std::fabs( m_terpADC1( zd ) ) );
            dadc2 = static_cast<float>( std::fabs( m_terpADC2( zd ) ) );
        }
        catch( const std::exception &e )
        {
            log<software_error>(
                { __FILE__, __LINE__, std::string( "exception interpolating ADC targets: " ) + e.what() } );
            return 0;
        }
        catch( ... )
        {
            log<software_error>( { __FILE__, __LINE__, "unknown exception interpolating ADC targets" } );
            return 0;
        }
    }
    else
    {
        std::cerr << "zenith limit\n";
    }

    float adc1 = adc1zero + adc1lupsign * ( dadc1 + adc1delta + deltaAngle );
    float adc2 = adc2zero + adc2lupsign * ( dadc2 + adc2delta + deltaAngle );

    if( !std::isfinite( adc1 ) || !std::isfinite( adc2 ) )
    {
        log<software_error>( { __FILE__, __LINE__, "computed non-finite ADC target" } );
        return 0;
    }

    std::cerr << "Sending adcs to: " << adc1 << " " << adc2 << "\n";

    if( sendNewProperty( m_indiP_adc1pos, "target", adc1 ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "failed to send ADC 1 target" } );
    }

    if( sendNewProperty( m_indiP_adc2pos, "target", adc2 ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "failed to send ADC 2 target" } );
    }

    return 0;
}

int adcTracker::appShutdown()
{
    return 0;
}

INDI_NEWCALLBACK_DEFN( adcTracker, m_indiP_tracking )( const pcf::IndiProperty &ipRecv )
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

        log<text_log>( "started ADC rotation tracking" );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE );

        { // mutex scope
            std::lock_guard<std::mutex> guard( m_indiMutex );
            m_tracking   = false;
            m_lastUpdate = 0;
        }

        log<text_log>( "stopped ADC rotation tracking" );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( adcTracker, m_indiP_deltaAngle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_deltaAngle, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_deltaAngle, target, ipRecv ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_deltaAngle = target;
    updateIfChanged( m_indiP_deltaAngle, "current", m_deltaAngle );

    log<text_log>( "set deltaAngle to " + std::to_string( m_deltaAngle ) );

    return 0;
}

INDI_NEWCALLBACK_DEFN( adcTracker, m_indiP_deltaADC1 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_deltaADC1, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_deltaADC1, target, ipRecv ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_adc1delta = target;
    updateIfChanged( m_indiP_deltaADC1, "current", m_adc1delta );

    log<text_log>( "set deltaADC1 to " + std::to_string( m_adc1delta ) );

    return 0;
}

INDI_NEWCALLBACK_DEFN( adcTracker, m_indiP_deltaADC2 )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_deltaADC2, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_deltaADC2, target, ipRecv ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_adc2delta = target;
    updateIfChanged( m_indiP_deltaADC2, "current", m_adc2delta );

    log<text_log>( "set deltaADC2 to " + std::to_string( m_adc2delta ) );

    return 0;
}

INDI_NEWCALLBACK_DEFN( adcTracker, m_indiP_minZD )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_minZD, ipRecv );

    float target;

    if( indiTargetUpdate( m_indiP_minZD, target, ipRecv ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    m_minZD = target;
    updateIfChanged( m_indiP_minZD, "current", m_minZD );

    log<text_log>( "set minZD to " + std::to_string( m_minZD ) );

    return 0;
}

INDI_SETCALLBACK_DEFN( adcTracker, m_indiP_teldata )( const pcf::IndiProperty &ipRecv )
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

#endif // adcTracker_hpp
