/** \file adcTracker_test.cpp
 * \brief Catch2 tests for the adcTracker app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup adcTracker_files
 */

/** \defgroup adcTracker_unit_test adcTracker Unit Tests
 * \brief Unit tests for the adcTracker application.
 *
 * \ingroup app_unit_test
 */

#include "../../../tests/catch2/catch.hpp"
#include "../adcTracker.hpp"
#include "../../../tests/testMacrosINDI.hpp"
#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>
#include <limits>

namespace libXWCTest
{
namespace adcTrackerTest
{

namespace
{
/// Build a per-test scratch directory under /tmp.
std::string testRoot( const std::string &name )
{
    std::string root = "/tmp/adcTracker_test/" + name;
    std::filesystem::create_directories( root );
    return root;
}

/// Write an ADC lookup table for a test case.
void writeLookupTable( const std::string &calibDir, const std::string &contents )
{
    std::filesystem::create_directories( calibDir );

    std::ofstream out( calibDir + "/adc_lookup_table.txt" );
    out << contents;
}

} // namespace

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness exposing adcTracker internals needed by the unit tests.
class adcTracker_test : public MagAOX::app::adcTracker
{
  public:
    /// Exception injection modes for the narrow test seams.
    enum class exceptionMode
    {
        none,            ///< No injected exception.
        stdException,    ///< Throw a standard exception.
        nonStdException, ///< Throw a non-standard exception.
        nonFinite        ///< Return a non-finite value.
    };

    /// Construct a testable ADC tracker instance in the default callback-test scratch area.
    adcTracker_test( const std::string &device ) : adcTracker_test( device, testRoot( "indi_callbacks" ) )
    {
    }

    /// Construct a testable ADC tracker instance.
    adcTracker_test( const std::string &device, const std::string &root )
    {
        m_configName = device;
        m_basePath   = root;
        m_calibDir   = root + "/calib";

        std::filesystem::create_directories( m_calibDir );
        std::filesystem::create_directories( root + "/telem" );

        m_tel.logPath( root + "/telem" );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;

        XWCTEST_SETUP_INDI_NEW_PROP( tracking );
        XWCTEST_SETUP_INDI_NEW_PROP( deltaAngle );
        XWCTEST_SETUP_INDI_NEW_PROP( deltaADC1 );
        XWCTEST_SETUP_INDI_NEW_PROP( deltaADC2 );
        XWCTEST_SETUP_INDI_NEW_PROP( minZD );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_teldata, tcsi, teldata );
    }

    /// Run adcTracker startup.
    int startup()
    {
        return appStartup();
    }

    /// Run adcTracker logic.
    int logic()
    {
        return appLogic();
    }

    /// Run adcTracker shutdown.
    int shutdown()
    {
        return appShutdown();
    }

    /// Invoke the telemeter callback directly.
    int recordTelemDirect()
    {
        return recordTelem( nullptr );
    }

    /// Invoke the ADC 1 send helper directly.
    int sendADC1PositionDirect( float adc1 )
    {
        return MagAOX::app::adcTracker::sendADC1Position( adc1 );
    }

    /// Invoke the ADC 2 send helper directly.
    int sendADC2PositionDirect( float adc2 )
    {
        return MagAOX::app::adcTracker::sendADC2Position( adc2 );
    }

    /// Return whether the lookup table was initialized successfully.
    bool lookupReady() const
    {
        return m_lookupReady;
    }

    /// Return the configured maximum lookup zenith distance.
    float maxZD() const
    {
        return m_maxZD;
    }

    /// Return the configured lookup-table filename.
    std::string lookupFile() const
    {
        return m_lookupFile;
    }

    /// Return the configured ADC 1 zero position.
    float adc1zero() const
    {
        return m_adc1zero;
    }

    /// Return the configured ADC 1 lookup sign.
    int adc1lupsign() const
    {
        return m_adc1lupsign;
    }

    /// Return the configured ADC 2 zero position.
    float adc2zero() const
    {
        return m_adc2zero;
    }

    /// Return the configured ADC 2 lookup sign.
    int adc2lupsign() const
    {
        return m_adc2lupsign;
    }

    /// Return the shared ADC delta angle.
    float deltaAngle() const
    {
        return m_deltaAngle;
    }

    /// Return the ADC 1 delta angle.
    float adc1delta() const
    {
        return m_adc1delta;
    }

    /// Return the ADC 2 delta angle.
    float adc2delta() const
    {
        return m_adc2delta;
    }

    /// Return the configured minimum zenith distance threshold.
    float minZD() const
    {
        return m_minZD;
    }

    /// Return the configured ADC 1 device name.
    std::string adc1DevName() const
    {
        return m_adc1DevName;
    }

    /// Return the configured ADC 2 device name.
    std::string adc2DevName() const
    {
        return m_adc2DevName;
    }

    /// Return the configured TCS device name.
    std::string tcsDevName() const
    {
        return m_tcsDevName;
    }

    /// Return the configured update interval.
    float updateInterval() const
    {
        return m_updateInterval;
    }

    /// Return whether tracking is currently enabled.
    bool tracking() const
    {
        return m_tracking;
    }

    /// Return the most recent received zenith distance.
    float zd() const
    {
        return m_zd;
    }

    /// Return whether a valid zenith distance has been received.
    bool haveZD() const
    {
        return m_haveZD;
    }

    /// Return whether the below-minZD status switch is on.
    bool belowMinZDStatus() const
    {
        return m_indiP_belowMinZD["state"].getSwitchState() == pcf::IndiElement::On;
    }

    /// Return whether the above-maxZD status switch is on.
    bool aboveMaxZDStatus() const
    {
        return m_indiP_aboveMaxZD["state"].getSwitchState() == pcf::IndiElement::On;
    }

    /// Return the last ADC update timestamp.
    double lastUpdate() const
    {
        return m_lastUpdate;
    }

    /// Return the current FSM state.
    MagAOX::app::stateCodes::stateCodeT fsmState()
    {
        return state();
    }

    /// Set whether tracking is enabled.
    void setTracking( bool tracking )
    {
        m_tracking = tracking;
    }

    /// Set the current test zenith distance and validity flag.
    void setZD( float zd, bool haveZD = true )
    {
        m_zd     = zd;
        m_haveZD = haveZD;
    }

    /// Set the last-update timestamp used by appLogic.
    void setLastUpdate( double lastUpdate )
    {
        m_lastUpdate = lastUpdate;
    }

    /// Set the update interval used by appLogic.
    void setUpdateInterval( float updateInterval )
    {
        m_updateInterval = updateInterval;
    }

    /// Set the shared and per-ADC tracking offsets.
    void setOffsets( float deltaAngle, float adc1delta, float adc2delta )
    {
        m_deltaAngle = deltaAngle;
        m_adc1delta  = adc1delta;
        m_adc2delta  = adc2delta;
    }

    /// Set the ADC zero positions used by appLogic.
    void setZeros( float adc1zero, float adc2zero )
    {
        m_adc1zero = adc1zero;
        m_adc2zero = adc2zero;
    }

    /// Set the ADC lookup-table signs used by appLogic.
    void setSigns( int adc1sign, int adc2sign )
    {
        m_adc1lupsign = adc1sign;
        m_adc2lupsign = adc2sign;
    }

    /// Set the minimum zenith distance threshold.
    void setMinZDValue( float minZD )
    {
        m_minZD = minZD;
    }

    /// Set up the standard config entries on the test configurator.
    void setupTestConfig()
    {
        setupConfig();
    }

    /// Read a config file into the app's internal configurator.
    void readConfigFile( const std::string &path )
    {
        config.readConfig( path );
    }

    /// Load configuration values from the app's internal configurator.
    void loadOwnConfig()
    {
        loadConfig();
    }

    /// Apply a tracking toggle callback payload.
    int applyTracking( bool enabled )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( m_configName );
        ip.setName( "tracking" );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"].setSwitchState( enabled ? pcf::IndiElement::On : pcf::IndiElement::Off );

        return newCallBack_m_indiP_tracking( ip );
    }

    /// Apply a shared delta-angle callback payload.
    int applyDeltaAngle( float value )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_configName );
        ip.setName( "deltaAngle" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( value );

        return newCallBack_m_indiP_deltaAngle( ip );
    }

    /// Apply an ADC 1 delta-angle callback payload.
    int applyDeltaADC1( float value )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_configName );
        ip.setName( "deltaADC1" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( value );

        return newCallBack_m_indiP_deltaADC1( ip );
    }

    /// Apply an ADC 2 delta-angle callback payload.
    int applyDeltaADC2( float value )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_configName );
        ip.setName( "deltaADC2" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( value );

        return newCallBack_m_indiP_deltaADC2( ip );
    }

    /// Apply a minimum-ZD callback payload.
    int applyMinZD( float value )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( m_configName );
        ip.setName( "minZD" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( value );

        return newCallBack_m_indiP_minZD( ip );
    }

    /// Apply a teldata callback payload.
    int applyTeldataZD( float value )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "teldata" );
        ip.add( pcf::IndiElement( "zd" ) );
        ip["zd"].set( value );

        return setCallBack_m_indiP_teldata( ip );
    }

    /// Invoke the static tracking callback wrapper.
    int staticTrackingCallback( const pcf::IndiProperty &ip )
    {
        return st_newCallBack_m_indiP_tracking( this, ip );
    }

    /// Invoke the static deltaAngle callback wrapper.
    int staticDeltaAngleCallback( const pcf::IndiProperty &ip )
    {
        return st_newCallBack_m_indiP_deltaAngle( this, ip );
    }

    /// Invoke the static deltaADC1 callback wrapper.
    int staticDeltaADC1Callback( const pcf::IndiProperty &ip )
    {
        return st_newCallBack_m_indiP_deltaADC1( this, ip );
    }

    /// Invoke the static deltaADC2 callback wrapper.
    int staticDeltaADC2Callback( const pcf::IndiProperty &ip )
    {
        return st_newCallBack_m_indiP_deltaADC2( this, ip );
    }

    /// Invoke the static minZD callback wrapper.
    int staticMinZDCallback( const pcf::IndiProperty &ip )
    {
        return st_newCallBack_m_indiP_minZD( this, ip );
    }

    /// Invoke the static teldata callback wrapper.
    int staticTeldataCallback( const pcf::IndiProperty &ip )
    {
        return st_setCallBack_m_indiP_teldata( this, ip );
    }

    /// Clear any previously captured outbound ADC command.
    void resetSent()
    {
        m_sendCount = 0;
        m_lastADC1  = 0;
        m_lastADC2  = 0;
    }

    /// Return how many outbound ADC commands were captured.
    int sendCount() const
    {
        return m_sendCount;
    }

    /// Return the last ADC 1 command captured by the test harness.
    float lastADC1() const
    {
        return m_lastADC1;
    }

    /// Return the last ADC 2 command captured by the test harness.
    float lastADC2() const
    {
        return m_lastADC2;
    }

    /// Configure exception injection for startup interpolator setup.
    void setSetupInterpolatorExceptionMode( exceptionMode mode )
    {
        m_setupInterpolatorExceptionMode = mode;
    }

    /// Configure exception injection for appLogic interpolation.
    void setInterpolationExceptionMode( exceptionMode mode )
    {
        m_interpolationExceptionMode = mode;
    }

    /// Configure exception injection for teldata ZD extraction.
    void setExtractZDExceptionMode( exceptionMode mode )
    {
        m_extractZDExceptionMode = mode;
    }

    /// Configure failure injection for ADC 1 sends.
    void setADC1SendFailure( bool shouldFail )
    {
        m_failADC1Send = shouldFail;
    }

    /// Configure failure injection for ADC 2 sends.
    void setADC2SendFailure( bool shouldFail )
    {
        m_failADC2Send = shouldFail;
    }

    /// Lock the INDI mutex for a test scope.
    std::unique_lock<std::mutex> lockIndiMutex()
    {
        return std::unique_lock<std::mutex>( m_indiMutex );
    }

  protected:
    /// Capture outbound ADC 1 commands instead of sending them over INDI.
    int sendADC1Position( float adc1 ) override
    {
        if( m_failADC1Send )
        {
            return -1;
        }

        m_lastADC1 = adc1;
        ++m_sendCount;

        return 0;
    }

    /// Capture outbound ADC 2 commands instead of sending them over INDI.
    int sendADC2Position( float adc2 ) override
    {
        if( m_failADC2Send )
        {
            return -1;
        }

        m_lastADC2 = adc2;
        ++m_sendCount;

        return 0;
    }

    /// Optionally throw while setting up interpolators to exercise startup exception handling.
    void setupInterpolators() override
    {
        if( m_setupInterpolatorExceptionMode == exceptionMode::stdException )
        {
            throw std::runtime_error( "test setupInterpolators std::exception" );
        }

        if( m_setupInterpolatorExceptionMode == exceptionMode::nonStdException )
        {
            throw 1;
        }

        MagAOX::app::adcTracker::setupInterpolators();
    }

    /// Optionally throw while interpolating ADC 1 to exercise appLogic exception handling.
    float interpolateADC1( float zd ) override
    {
        if( m_interpolationExceptionMode == exceptionMode::stdException )
        {
            throw std::runtime_error( "test interpolateADC1 std::exception" );
        }

        if( m_interpolationExceptionMode == exceptionMode::nonStdException )
        {
            throw 1;
        }

        return MagAOX::app::adcTracker::interpolateADC1( zd );
    }

    /// Preserve the production ADC 2 interpolation logic in the test harness.
    float interpolateADC2( float zd ) override
    {
        return MagAOX::app::adcTracker::interpolateADC2( zd );
    }

    /// Optionally throw while extracting teldata.zd to exercise callback exception handling.
    float extractZD( const pcf::IndiProperty &ipRecv ) override
    {
        if( m_extractZDExceptionMode == exceptionMode::stdException )
        {
            throw std::runtime_error( "test extractZD std::exception" );
        }

        if( m_extractZDExceptionMode == exceptionMode::nonStdException )
        {
            throw 1;
        }

        if( m_extractZDExceptionMode == exceptionMode::nonFinite )
        {
            return std::numeric_limits<float>::infinity();
        }

        return MagAOX::app::adcTracker::extractZD( ipRecv );
    }

  private:
    float         m_lastADC1{ 0 };         ///< Last ADC 1 target captured by the test harness.
    float         m_lastADC2{ 0 };         ///< Last ADC 2 target captured by the test harness.
    int           m_sendCount{ 0 };        ///< Number of outbound ADC commands captured by the test harness.
    bool          m_failADC1Send{ false }; ///< True to force ADC 1 send failure in appLogic tests.
    bool          m_failADC2Send{ false }; ///< True to force ADC 2 send failure in appLogic tests.
    exceptionMode m_setupInterpolatorExceptionMode{ exceptionMode::none }; ///< Startup exception injection state.
    exceptionMode m_interpolationExceptionMode{ exceptionMode::none }; ///< appLogic interpolation exception injection.
    exceptionMode m_extractZDExceptionMode{ exceptionMode::none }; ///< teldata extraction exception injection state.
};
/// \endcond

/// Validate the adcTracker INDI callback wiring.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "INDI callbacks validate their source properties", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty( pcf::IndiProperty::Switch ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaAngle( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaADC1( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaADC2( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_minZD( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty( pcf::IndiProperty::Number ) );
#endif
    // clang-format on

    adcTracker_test tracker( "adcTracker_validation" );

    WHEN( "the tracking callback receives the wrong device" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "wrong" );
        ip.setName( "tracking" );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( tracker.newCallBack_m_indiP_tracking( ip ) == -1 );
    }

    WHEN( "the deltaAngle callback receives the wrong property name" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_validation" );
        ip.setName( "wrong" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 1.0f );

        REQUIRE( tracker.newCallBack_m_indiP_deltaAngle( ip ) == -1 );
    }

    WHEN( "the deltaADC1 callback receives a valid payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_validation" );
        ip.setName( "deltaADC1" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 2.0f );

        REQUIRE( tracker.newCallBack_m_indiP_deltaADC1( ip ) == 0 );
    }

    WHEN( "the deltaADC2 callback receives a valid payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_validation" );
        ip.setName( "deltaADC2" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 3.0f );

        REQUIRE( tracker.newCallBack_m_indiP_deltaADC2( ip ) == 0 );
    }

    WHEN( "the minZD callback receives a valid payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_validation" );
        ip.setName( "minZD" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 4.0f );

        REQUIRE( tracker.newCallBack_m_indiP_minZD( ip ) == 0 );
    }

    WHEN( "the teldata callback receives the wrong device" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "wrong" );
        ip.setName( "teldata" );
        ip.add( pcf::IndiElement( "zd" ) );
        ip["zd"].set( 5.0f );

        REQUIRE( tracker.setCallBack_m_indiP_teldata( ip ) == -1 );
    }

    WHEN( "the teldata callback receives a valid payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "teldata" );
        ip.add( pcf::IndiElement( "zd" ) );
        ip["zd"].set( 6.0f );

        REQUIRE( tracker.setCallBack_m_indiP_teldata( ip ) == 0 );
    }
}

/// Validate adcTracker configuration loading for defaults and explicit overrides.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker configuration loading preserves defaults and applies overrides", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::setupConfig();
    MagAOX::app::adcTracker::loadConfig();
#endif
    // clang-format on

    GIVEN( "an empty config file after setupConfig" )
    {
        std::string root = testRoot( "config_defaults" );
        mx::app::writeConfigFile( root + "/adcTracker_test.conf", {}, {}, {} );

        adcTracker_test tracker( "adcTracker_config_defaults", root );
        tracker.setupTestConfig();
        tracker.readConfigFile( root + "/adcTracker_test.conf" );

        tracker.loadOwnConfig();

        REQUIRE( tracker.lookupFile() == "adc_lookup_table.txt" );
        REQUIRE( tracker.adc1zero() == Approx( 0.0f ) );
        REQUIRE( tracker.adc1lupsign() == 1 );
        REQUIRE( tracker.adc2zero() == Approx( 0.0f ) );
        REQUIRE( tracker.adc2lupsign() == 1 );
        REQUIRE( tracker.deltaAngle() == Approx( 0.0f ) );
        REQUIRE( tracker.adc1delta() == Approx( 0.0f ) );
        REQUIRE( tracker.adc2delta() == Approx( 0.0f ) );
        REQUIRE( tracker.minZD() == Approx( 5.1f ) );
        REQUIRE( tracker.adc1DevName() == "stageadc1" );
        REQUIRE( tracker.adc2DevName() == "stageadc2" );
        REQUIRE( tracker.tcsDevName() == "tcsi" );
        REQUIRE( tracker.updateInterval() == Approx( 10.0f ) );
    }

    GIVEN( "a config file overriding the ADC tracker parameters" )
    {
        std::string root = testRoot( "config_overrides" );
        mx::app::writeConfigFile( root + "/adcTracker_test.conf",
                                  { "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "adcs",
                                    "tcs",
                                    "tracking" },
                                  { "lookupFile",
                                    "adc1zero",
                                    "adc1lupsign",
                                    "adc2zero",
                                    "adc2lupsign",
                                    "deltaAngle",
                                    "adc1delta",
                                    "adc2delta",
                                    "minZD",
                                    "adc1DevName",
                                    "adc2DevName",
                                    "devName",
                                    "updateInterval" },
                                  { "custom_lookup.txt",
                                    "12.5",
                                    "-1",
                                    "-7.5",
                                    "-1",
                                    "4.5",
                                    "1.5",
                                    "-2.5",
                                    "8.2",
                                    "adc1custom",
                                    "adc2custom",
                                    "tcsi_custom",
                                    "3.5" } );

        adcTracker_test tracker( "adcTracker_config_overrides", root );
        tracker.setupTestConfig();
        tracker.readConfigFile( root + "/adcTracker_test.conf" );

        tracker.loadOwnConfig();

        REQUIRE( tracker.lookupFile() == "custom_lookup.txt" );
        REQUIRE( tracker.adc1zero() == Approx( 12.5f ) );
        REQUIRE( tracker.adc1lupsign() == -1 );
        REQUIRE( tracker.adc2zero() == Approx( -7.5f ) );
        REQUIRE( tracker.adc2lupsign() == -1 );
        REQUIRE( tracker.deltaAngle() == Approx( 4.5f ) );
        REQUIRE( tracker.adc1delta() == Approx( 1.5f ) );
        REQUIRE( tracker.adc2delta() == Approx( -2.5f ) );
        REQUIRE( tracker.minZD() == Approx( 8.2f ) );
        REQUIRE( tracker.adc1DevName() == "adc1custom" );
        REQUIRE( tracker.adc2DevName() == "adc2custom" );
        REQUIRE( tracker.tcsDevName() == "tcsi_custom" );
        REQUIRE( tracker.updateInterval() == Approx( 3.5f ) );
    }
}

/// Validate adcTracker callback behavior for valid payloads.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker callbacks update runtime state from valid INDI payloads", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty( pcf::IndiProperty::Switch ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaAngle( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaADC1( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_deltaADC2( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::newCallBack_m_indiP_minZD( pcf::IndiProperty( pcf::IndiProperty::Number ) );
    MagAOX::app::adcTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty( pcf::IndiProperty::Number ) );
#endif
    // clang-format on

    std::string     root = testRoot( "callback_behavior" );
    adcTracker_test tracker( "adcTracker_callback_behavior", root );

    GIVEN( "a tracking callback payload" )
    {
        tracker.setLastUpdate( 42.0 );

        REQUIRE( tracker.applyTracking( true ) == 0 );
        REQUIRE( tracker.tracking() == true );
        REQUIRE( tracker.lastUpdate() == Approx( 0.0 ) );

        tracker.setLastUpdate( 84.0 );

        REQUIRE( tracker.applyTracking( false ) == 0 );
        REQUIRE( tracker.tracking() == false );
        REQUIRE( tracker.lastUpdate() == Approx( 0.0 ) );
    }

    GIVEN( "numeric offset callback payloads" )
    {
        REQUIRE( tracker.applyDeltaAngle( 7.25f ) == 0 );
        REQUIRE( tracker.deltaAngle() == Approx( 7.25f ) );

        REQUIRE( tracker.applyDeltaADC1( -1.5f ) == 0 );
        REQUIRE( tracker.adc1delta() == Approx( -1.5f ) );

        REQUIRE( tracker.applyDeltaADC2( 2.75f ) == 0 );
        REQUIRE( tracker.adc2delta() == Approx( 2.75f ) );

        REQUIRE( tracker.applyMinZD( 11.0f ) == 0 );
        REQUIRE( tracker.minZD() == Approx( 11.0f ) );
    }

    GIVEN( "numeric callbacks missing their target element" )
    {
        pcf::IndiProperty deltaAngleIP( pcf::IndiProperty::Number );
        deltaAngleIP.setDevice( "adcTracker_callback_behavior" );
        deltaAngleIP.setName( "deltaAngle" );
        REQUIRE( tracker.newCallBack_m_indiP_deltaAngle( deltaAngleIP ) == -1 );

        pcf::IndiProperty deltaADC1IP( pcf::IndiProperty::Number );
        deltaADC1IP.setDevice( "adcTracker_callback_behavior" );
        deltaADC1IP.setName( "deltaADC1" );
        REQUIRE( tracker.newCallBack_m_indiP_deltaADC1( deltaADC1IP ) == -1 );

        pcf::IndiProperty deltaADC2IP( pcf::IndiProperty::Number );
        deltaADC2IP.setDevice( "adcTracker_callback_behavior" );
        deltaADC2IP.setName( "deltaADC2" );
        REQUIRE( tracker.newCallBack_m_indiP_deltaADC2( deltaADC2IP ) == -1 );

        pcf::IndiProperty minZDIP( pcf::IndiProperty::Number );
        minZDIP.setDevice( "adcTracker_callback_behavior" );
        minZDIP.setName( "minZD" );
        REQUIRE( tracker.newCallBack_m_indiP_minZD( minZDIP ) == -1 );
    }

    GIVEN( "a tracking callback payload missing the toggle element" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "adcTracker_callback_behavior" );
        ip.setName( "tracking" );

        REQUIRE( tracker.newCallBack_m_indiP_tracking( ip ) == 0 );
    }

    GIVEN( "a valid teldata.zd payload" )
    {
        REQUIRE( tracker.applyTeldataZD( 33.5f ) == 0 );
        REQUIRE( tracker.zd() == Approx( 33.5f ) );
        REQUIRE( tracker.haveZD() == true );
    }

    GIVEN( "a teldata payload missing the zd element" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "teldata" );

        REQUIRE( tracker.setCallBack_m_indiP_teldata( ip ) == 0 );
        REQUIRE( tracker.haveZD() == false );
    }
}

/// Validate adcTracker teldata callback exception handling.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker teldata callback handles extraction exceptions", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty( pcf::IndiProperty::Number ) );
#endif
    // clang-format on

    GIVEN( "a valid teldata payload and an injected std::exception" )
    {
        adcTracker_test tracker( "adcTracker_teldata_std_exception", testRoot( "teldata_std_exception" ) );
        tracker.setExtractZDExceptionMode( adcTracker_test::exceptionMode::stdException );

        REQUIRE( tracker.applyTeldataZD( 11.0f ) == 0 );
        REQUIRE( tracker.haveZD() == false );
    }

    GIVEN( "a valid teldata payload and an injected non-standard exception" )
    {
        adcTracker_test tracker( "adcTracker_teldata_unknown_exception", testRoot( "teldata_unknown_exception" ) );
        tracker.setExtractZDExceptionMode( adcTracker_test::exceptionMode::nonStdException );

        REQUIRE( tracker.applyTeldataZD( 11.0f ) == 0 );
        REQUIRE( tracker.haveZD() == false );
    }

    GIVEN( "a valid teldata payload and an injected non-finite zenith distance" )
    {
        adcTracker_test tracker( "adcTracker_teldata_nonfinite", testRoot( "teldata_nonfinite" ) );
        tracker.setExtractZDExceptionMode( adcTracker_test::exceptionMode::nonFinite );

        REQUIRE( tracker.applyTeldataZD( 11.0f ) == 0 );
    }
}

/// Validate adcTracker startup against valid and invalid lookup tables.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker startup validates the lookup table before running", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::appStartup();
#endif
    // clang-format on

    GIVEN( "a valid lookup table" )
    {
        std::string root = testRoot( "startup_valid" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_test_valid", root );

        REQUIRE( tracker.startup() == 0 );
        REQUIRE( tracker.lookupReady() );
        REQUIRE( tracker.maxZD() == Approx( 20.0f ) );
        REQUIRE( tracker.belowMinZDStatus() == false );
        REQUIRE( tracker.aboveMaxZDStatus() == false );
        REQUIRE( tracker.fsmState() == MagAOX::app::stateCodes::READY );
    }

    GIVEN( "a missing lookup table" )
    {
        std::string     root = testRoot( "startup_missing" );
        adcTracker_test tracker( "adcTracker_test_missing", root );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }

    GIVEN( "a lookup table with too few rows" )
    {
        std::string root = testRoot( "startup_too_few_rows" );
        writeLookupTable( root + "/calib", "0,0,0\n" );

        adcTracker_test tracker( "adcTracker_test_too_few_rows", root );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }

    GIVEN( "a lookup table with a malformed row" )
    {
        std::string root = testRoot( "startup_malformed_row" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_test_malformed_row", root );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }

    GIVEN( "a lookup table with non-monotonic zenith-distance samples" )
    {
        std::string root = testRoot( "startup_non_monotonic" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n10,20,40\n" );

        adcTracker_test tracker( "adcTracker_test_non_monotonic", root );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }

    GIVEN( "a valid lookup table and an injected std::exception during interpolator setup" )
    {
        std::string root = testRoot( "startup_interp_std_exception" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_test_interp_std_exception", root );
        tracker.setSetupInterpolatorExceptionMode( adcTracker_test::exceptionMode::stdException );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }

    GIVEN( "a valid lookup table and an injected non-standard exception during interpolator setup" )
    {
        std::string root = testRoot( "startup_interp_unknown_exception" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_test_interp_unknown_exception", root );
        tracker.setSetupInterpolatorExceptionMode( adcTracker_test::exceptionMode::nonStdException );

        REQUIRE( tracker.startup() == -1 );
        REQUIRE( tracker.lookupReady() == false );
    }
}

/// Validate adcTracker appLogic dispatch decisions and target calculations.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker appLogic sends ADC targets only when tracking data are ready", "[adcTracker]" )
{
    // clang-format off
#ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::appLogic();
#endif
    // clang-format on

    WHEN( "tracking is disabled" )
    {
        std::string root = testRoot( "logic_tracking_disabled" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_tracking_disabled", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.resetSent();

        tracker.setTracking( false );
        tracker.setZD( 15.0f );

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "tracking is enabled before any valid zenith distance is received" )
    {
        std::string root = testRoot( "logic_no_zd" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_no_zd", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.resetSent();

        tracker.setTracking( true );
        tracker.setZD( 0.0f, false );

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "the INDI mutex is already locked by another scope" )
    {
        std::string root = testRoot( "logic_mutex_locked" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_mutex_locked", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setTracking( true );
        tracker.setZD( 15.0f );
        tracker.resetSent();

        auto lock = tracker.lockIndiMutex();

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "the zenith distance is within the lookup table range" )
    {
        std::string root = testRoot( "logic_interpolate" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_interpolate", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.resetSent();

        tracker.setTracking( true );
        tracker.setZD( 15.0f );

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 2 );
        REQUIRE( tracker.lastADC1() == Approx( 24.0f ) );
        REQUIRE( tracker.lastADC2() == Approx( 41.0f ) );
        REQUIRE( tracker.belowMinZDStatus() == false );
        REQUIRE( tracker.aboveMaxZDStatus() == false );
    }

    WHEN( "the zenith distance is below minZD" )
    {
        std::string root = testRoot( "logic_below_min_zd" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_below_min_zd", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.resetSent();

        tracker.setTracking( true );
        tracker.setZD( 2.0f );

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 2 );
        REQUIRE( tracker.lastADC1() == Approx( 9.0f ) );
        REQUIRE( tracker.lastADC2() == Approx( 11.0f ) );
        REQUIRE( tracker.belowMinZDStatus() == true );
        REQUIRE( tracker.aboveMaxZDStatus() == false );
    }

    WHEN( "the zenith distance is above the lookup-table maximum" )
    {
        std::string root = testRoot( "logic_above_max_zd" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_above_max_zd", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.resetSent();

        tracker.setTracking( true );
        tracker.setZD( 30.0f );

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 2 );
        REQUIRE( tracker.lastADC1() == Approx( 29.0f ) );
        REQUIRE( tracker.lastADC2() == Approx( 51.0f ) );
        REQUIRE( tracker.belowMinZDStatus() == false );
        REQUIRE( tracker.aboveMaxZDStatus() == true );
    }

    WHEN( "interpolation throws a std::exception" )
    {
        std::string root = testRoot( "logic_interpolation_std_exception" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_interpolation_std_exception", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.setTracking( true );
        tracker.setZD( 15.0f );
        tracker.setInterpolationExceptionMode( adcTracker_test::exceptionMode::stdException );
        tracker.resetSent();

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "interpolation throws a non-standard exception" )
    {
        std::string root = testRoot( "logic_interpolation_unknown_exception" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_interpolation_unknown_exception", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.setTracking( true );
        tracker.setZD( 15.0f );
        tracker.setInterpolationExceptionMode( adcTracker_test::exceptionMode::nonStdException );
        tracker.resetSent();

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "sending ADC 1 fails during appLogic" )
    {
        std::string root = testRoot( "logic_send_adc1_failure" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_send_adc1_failure", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.setTracking( true );
        tracker.setZD( 15.0f );
        tracker.setADC1SendFailure( true );
        tracker.resetSent();

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 0 );
    }

    WHEN( "sending ADC 2 fails during appLogic" )
    {
        std::string root = testRoot( "logic_send_adc2_failure" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_logic_send_adc2_failure", root );
        REQUIRE( tracker.startup() == 0 );

        tracker.setUpdateInterval( 0.0f );
        tracker.setLastUpdate( 0.0 );
        tracker.setZeros( 1.0f, 2.0f );
        tracker.setSigns( 1, 1 );
        tracker.setOffsets( 5.0f, 3.0f, 4.0f );
        tracker.setMinZDValue( 5.1f );
        tracker.setTracking( true );
        tracker.setZD( 15.0f );
        tracker.setADC2SendFailure( true );
        tracker.resetSent();

        REQUIRE( tracker.logic() == 0 );
        REQUIRE( tracker.sendCount() == 1 );
        REQUIRE( tracker.belowMinZDStatus() == false );
        REQUIRE( tracker.aboveMaxZDStatus() == false );
    }
}

/// Validate adcTracker shutdown and telemetry callback direct entry points.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker direct helper entry points behave as expected", "[adcTracker]" )
{
    // clang-format off
    #ifdef ADCTRACK_TEST_DOXYGEN_REF
    MagAOX::app::adcTracker::appShutdown();
    MagAOX::app::adcTracker::recordTelem( static_cast<const telem_adctrack *>( nullptr ) );
    MagAOX::app::adcTracker::sendADC1Position( 0.0f );
    MagAOX::app::adcTracker::sendADC2Position( 0.0f );
    #endif
    // clang-format on

    GIVEN( "a tracker with its startup completed" )
    {
        std::string root = testRoot( "direct_helper_entry_points" );
        writeLookupTable( root + "/calib", "0,0,0\n10,10,20\n20,20,40\n" );

        adcTracker_test tracker( "adcTracker_direct_helper_entry_points", root );
        REQUIRE( tracker.startup() == 0 );

        WHEN( "appShutdown is called directly" )
        {
            REQUIRE( tracker.shutdown() == 0 );
        }

        WHEN( "recordTelem is called directly" )
        {
            tracker.setTracking( true );
            tracker.setOffsets( 1.5f, -2.0f, 3.0f );
            tracker.setMinZDValue( 7.5f );

            REQUIRE( tracker.recordTelemDirect() == 0 );
        }

        WHEN( "the direct ADC send helpers are called without an initialized INDI driver" )
        {
            REQUIRE( tracker.sendADC1PositionDirect( 12.5f ) == -1 );
            REQUIRE( tracker.sendADC2PositionDirect( -7.25f ) == -1 );
        }
    }
}

/// Validate adcTracker static callback wrappers forward to the instance callbacks.
/**
 * \ingroup adcTracker_unit_test
 */
SCENARIO( "adcTracker static callback wrappers forward correctly", "[adcTracker]" )
{
    adcTracker_test tracker( "adcTracker_static_wrappers", testRoot( "static_wrappers" ) );

    GIVEN( "a valid tracking callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( "adcTracker_static_wrappers" );
        ip.setName( "tracking" );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"].setSwitchState( pcf::IndiElement::On );

        REQUIRE( tracker.staticTrackingCallback( ip ) == 0 );
        REQUIRE( tracker.tracking() == true );
    }

    GIVEN( "a valid deltaAngle callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_static_wrappers" );
        ip.setName( "deltaAngle" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 4.0f );

        REQUIRE( tracker.staticDeltaAngleCallback( ip ) == 0 );
        REQUIRE( tracker.deltaAngle() == Approx( 4.0f ) );
    }

    GIVEN( "a valid deltaADC1 callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_static_wrappers" );
        ip.setName( "deltaADC1" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( -1.0f );

        REQUIRE( tracker.staticDeltaADC1Callback( ip ) == 0 );
        REQUIRE( tracker.adc1delta() == Approx( -1.0f ) );
    }

    GIVEN( "a valid deltaADC2 callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_static_wrappers" );
        ip.setName( "deltaADC2" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 2.5f );

        REQUIRE( tracker.staticDeltaADC2Callback( ip ) == 0 );
        REQUIRE( tracker.adc2delta() == Approx( 2.5f ) );
    }

    GIVEN( "a valid minZD callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "adcTracker_static_wrappers" );
        ip.setName( "minZD" );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].set( 9.5f );

        REQUIRE( tracker.staticMinZDCallback( ip ) == 0 );
        REQUIRE( tracker.minZD() == Approx( 9.5f ) );
    }

    GIVEN( "a valid teldata callback payload" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "teldata" );
        ip.add( pcf::IndiElement( "zd" ) );
        ip["zd"].set( 22.0f );

        REQUIRE( tracker.staticTeldataCallback( ip ) == 0 );
        REQUIRE( tracker.zd() == Approx( 22.0f ) );
        REQUIRE( tracker.haveZD() == true );
    }
}

} // namespace adcTrackerTest
} // namespace libXWCTest
