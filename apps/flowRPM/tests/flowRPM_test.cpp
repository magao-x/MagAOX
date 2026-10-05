/** \file flowRPM_test.cpp
 * \brief Catch2 tests for the flowRPM app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup flowRPM_files
 */

#include "../../../tests/testXWC.hpp"

#include <filesystem>

#define protected public
#include "../flowRPM.hpp"
#undef protected

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup flowRPM_unit_test flowRPM Unit Tests
 * \brief Unit tests for the flowRPM application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `flowRPM` unit tests.
/** \ingroup flowRPM_unit_test
 */
namespace flowRPMTest
{

namespace
{

/// Convert a numeric INDI element value to `double` for assertions.
double indiNumberValue( const pcf::IndiProperty &property, const std::string &element )
{
    return std::stod( property[element].getValue() );
}

/// Test harness that forces `loadConfig()` to see a configuration failure.
class flowRPMLoadConfigFailure : public flowRPM
{
  public:
    /// Return a failure so the inherited `loadConfig()` path sets `m_shutdown`.
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] unused configuration object */ ) override
    {
        static_cast<void>( _config );
        return -1;
    }
};

/// Test harness that can inject failures into the `appLogic()` call chain.
class flowRPMFaultInject : public flowRPM
{
  public:
    /// Failure sites that can be forced during `appLogic()`.
    enum class faultMode
    {
        none,
        readAndParse,
        publishResult,
        recordFlow
    };

    /// Select which call inside `appLogic()` should fail.
    void fault( faultMode mode /**< [in] injected failure site */ )
    {
        m_faultMode = mode;
    }

    /// Return an injected failure from `readAndParse()` or provide a valid sample.
    int readAndParse( parseResult    &result, /**< [out] parse result */
                      const timespec &now     /**< [in] current time */
    ) const override
    {
        if( m_faultMode == faultMode::readAndParse )
        {
            return -1;
        }

        result.m_status   = parseStatus::success;
        result.m_flowRate = 1.9;
        result.m_age      = 0.0;
        result.m_sourceTs = now;

        return 0;
    }

    /// Return an injected failure from `publishResult()`.
    int publishResult( const parseResult &result /**< [in] parse result to publish */ ) override
    {
        if( m_faultMode == faultMode::publishResult )
        {
            return -1;
        }

        return flowRPM::publishResult( result );
    }

    /// Return an injected failure from `recordFlow()`.
    int recordFlow( bool force = false /**< [in] whether telemetry is forced */ ) override
    {
        if( m_faultMode == faultMode::recordFlow )
        {
            return -1;
        }

        return flowRPM::recordFlow( force );
    }

  private:
    faultMode m_faultMode{ faultMode::none }; ///< Current fault injected into the `appLogic()` flow.
};

} // namespace

/// Verify default `flowRPM` configuration values are loaded.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM configuration defaults load correctly", "[flowRPM]" )
{
    flowRPM app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/flowRPM_test.conf", { "none" }, { "nada" }, { "0" } );
    app.config.readConfig( "/tmp/flowRPM_test.conf" );

    app.loadConfig();
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::setupConfig();
    flowRPM::loadConfig();
    #endif
    // clang-format on

    REQUIRE( app.m_shutdown == 0 );
    REQUIRE( app.inputPath() == "/tmp/fac_flow.txt" );
    REQUIRE( app.maxAge() == Approx( 60.0 ) );
    REQUIRE( app.fanDescriptor() == "CHA_FAN1" );
    REQUIRE( app.badValue() == Approx( -999.0 ) );
    REQUIRE( app.errorLogInterval() == Approx( 60.0 ) );
}

/// Verify configured `flowRPM` overrides are loaded.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM configuration overrides load correctly", "[flowRPM]" )
{
    flowRPM app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/flowRPM_test_override.conf",
                              { "input", "input", "input", "input", "input" },
                              { "path", "maxAge", "fanDescriptor", "badValue", "errorLogInterval" },
                              { "/tmp/custom_flow.txt", "12.5", "SYS_FAN9", "-321", "17" } );
    app.config.readConfig( "/tmp/flowRPM_test_override.conf" );

    app.loadConfig();
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::setupConfig();
    flowRPM::loadConfig();
    #endif
    // clang-format on

    REQUIRE( app.m_shutdown == 0 );
    REQUIRE( app.inputPath() == "/tmp/custom_flow.txt" );
    REQUIRE( app.maxAge() == Approx( 12.5 ) );
    REQUIRE( app.fanDescriptor() == "SYS_FAN9" );
    REQUIRE( app.badValue() == Approx( -321.0 ) );
    REQUIRE( app.errorLogInterval() == Approx( 17.0 ) );
}

/// Verify `loadConfig()` requests shutdown when configuration loading fails.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM loadConfig sets shutdown on configuration failure", "[flowRPM]" )
{
    flowRPMLoadConfigFailure app;

    app.setupConfig();
    app.loadConfig();
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::setupConfig();
    flowRPM::loadConfig();
    #endif
    // clang-format on

    REQUIRE( app.m_shutdown == 1 );
}

/// Verify `trimToken()` removes surrounding whitespace and handles blank tokens.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM helper token parsing handles whitespace-only input", "[flowRPM]" )
{
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPMDetail::trimToken( "" );
    #endif
    // clang-format on
    REQUIRE( flowRPMDetail::trimToken( "  CHA_FAN1 \t\r" ) == "CHA_FAN1" );
    REQUIRE( flowRPMDetail::trimToken( " \t\r " ).empty() );
}

/// Verify `splitLogicalLines()` removes CRLF suffixes and ignores blank lines.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM helper logical-line splitting handles CRLF and blank lines", "[flowRPM]" )
{
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPMDetail::splitLogicalLines( "" );
    #endif
    // clang-format on
    const std::vector<std::string> lines =
        flowRPMDetail::splitLogicalLines( "\r\n \t \r\n1775430287 145131374\r\n"
                                          "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\r\n" );

    REQUIRE( lines.size() == 2 );
    REQUIRE( lines[0] == "1775430287 145131374" );
    REQUIRE( lines[1] == "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'" );
}

/// Verify the default-initialized `parseResult` sentinel state.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM parseResult defaults to the invalid sentinel state", "[flowRPM]" )
{
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::parseResult();
    #endif
    // clang-format on
    flowRPM::parseResult result;

    REQUIRE( result.m_status == flowRPM::parseStatus::fileReadError );
    REQUIRE( result.m_flowRate == Approx( -999.0 ) );
    REQUIRE( result.m_age == Approx( -999.0 ) );
    REQUIRE( result.m_sourceTs.tv_sec == 0 );
    REQUIRE( result.m_sourceTs.tv_nsec == 0 );
}

/// Verify `parseTimestamp()` accepts valid input and rejects malformed timestamps.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM timestamp parsing covers valid and malformed inputs", "[flowRPM]" )
{
    flowRPM  app;
    timespec ts;
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::parseTimestamp( ts, "" );
    #endif
    // clang-format on

    SECTION( "a valid timestamp parses successfully" )
    {
        REQUIRE( app.parseTimestamp( ts, "1775430287 145131374" ) == 0 );
        REQUIRE( ts.tv_sec == 1775430287 );
        REQUIRE( ts.tv_nsec == 145131374 );
    }

    SECTION( "a trailing token is rejected" )
    {
        REQUIRE( app.parseTimestamp( ts, "1775430287 145131374 extra" ) == -1 );
    }

    SECTION( "a negative nanosecond field is rejected" )
    {
        REQUIRE( app.parseTimestamp( ts, "1775430287 -1" ) == -1 );
    }

    SECTION( "an overflowing nanosecond field is rejected" )
    {
        REQUIRE( app.parseTimestamp( ts, "1775430287 1000000000" ) == -1 );
    }
}

/// Verify `parseRecordLine()` covers the accepted and rejected record formats.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM record-line parsing covers all parse branches", "[flowRPM]" )
{
    flowRPM app;
    double  flowRate = app.badValue();
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::parseRecordLine( flowRate, "" );
    #endif
    // clang-format on

    SECTION( "a valid record converts RPM to LPM" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | CHA_FAN1 | Fan | 1900.00 | RPM | 'OK'" ) ==
                 flowRPM::parseStatus::success );
        REQUIRE( flowRate == Approx( 1.9 ) );
    }

    SECTION( "the zero-flow threshold status is accepted as a valid reading" )
    {
        REQUIRE(
            app.parseRecordLine(
                flowRate, "36 | CHA_FAN1 | Fan | 0.00 | RPM | 'At or Below (<=) Lower Non-Recoverable Threshold'" ) ==
            flowRPM::parseStatus::success );
        REQUIRE( flowRate == Approx( 0.0 ) );
    }

    SECTION( "the wrong field count is rejected" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | CHA_FAN1 | Fan | 1900.00 | RPM" ) ==
                 flowRPM::parseStatus::malformedRecord );
    }

    SECTION( "the wrong descriptor is rejected" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | OTHER_FAN | Fan | 1900.00 | RPM | 'OK'" ) ==
                 flowRPM::parseStatus::wrongDescriptor );
    }

    SECTION( "the wrong units are rejected" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | CHA_FAN1 | Fan | 1900.00 | LPM | 'OK'" ) ==
                 flowRPM::parseStatus::wrongUnits );
    }

    SECTION( "a non-OK status is rejected" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | CHA_FAN1 | Fan | 1900.00 | RPM | 'BAD'" ) ==
                 flowRPM::parseStatus::badStatus );
    }

    SECTION( "a malformed numeric field is rejected" )
    {
        REQUIRE( app.parseRecordLine( flowRate, "36 | CHA_FAN1 | Fan | 19x | RPM | 'OK'" ) ==
                 flowRPM::parseStatus::badValue );
    }
}

/// Verify `parseFileContents()` handles valid, invalid, and stale two-line inputs.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM file parsing", "[flowRPM]" )
{
    flowRPM              app;
    timespec             now{ 1775430290, 145131374 };
    flowRPM::parseResult result;
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::parseFileContents( result, "", now );
    #endif
    // clang-format on

    SECTION( "empty contents are reported as missing timestamps" )
    {
        REQUIRE( app.parseFileContents( result, "", now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::missingTimestamp );
    }

    SECTION( "whitespace-only contents are reported as missing timestamps" )
    {
        REQUIRE( app.parseFileContents( result, " \t\r\n\r\n", now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::missingTimestamp );
    }

    SECTION( "valid two-line input parses successfully" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::success );
        REQUIRE( result.m_flowRate == Approx( 1.9 ) );
        REQUIRE( result.m_age == Approx( 3.0 ) );
    }

    SECTION( "zero-flow threshold status parses as a valid zero reading" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 0.00       | RPM   | "
                                     "'At or Below (<=) Lower Non-Recoverable Threshold'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::success );
        REQUIRE( result.m_flowRate == Approx( 0.0 ) );
        REQUIRE( result.m_age == Approx( 3.0 ) );
    }

    SECTION( "wrong descriptor is rejected" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | OTHER_FAN        | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::wrongDescriptor );
        REQUIRE( result.m_flowRate == app.badValue() );
    }

    SECTION( "wrong units are rejected" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | LPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::wrongUnits );
    }

    SECTION( "bad status is rejected" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'BAD'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::badStatus );
    }

    SECTION( "partial writes are reported as missing records" )
    {
        const std::string contents = "1775430287 145131374\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::missingRecord );
        REQUIRE( result.m_age == app.badValue() );
    }

    SECTION( "malformed timestamps are rejected" )
    {
        const std::string contents = "1775430287 nope\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::malformedTimestamp );
    }

    SECTION( "timestamps with trailing tokens are rejected" )
    {
        const std::string contents = "1775430287 145131374 extra\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::malformedTimestamp );
    }

    SECTION( "malformed numeric values are rejected" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | nineteen    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::badValue );
    }

    SECTION( "stale readings are marked invalid while retaining age" )
    {
        const timespec    nowStale{ 1775430400, 145131374 };
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, nowStale ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::staleReading );
        REQUIRE( result.m_flowRate == app.badValue() );
        REQUIRE( result.m_age == Approx( 113.0 ) );
    }

    SECTION( "extra non-empty lines are rejected when only one sensor row is expected" )
    {
        const std::string contents = "1775430287 145131374\n"
                                     "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n"
                                     "37 | CHA_FAN2         | Fan          | 1900.00    | RPM   | 'OK'\n";

        REQUIRE( app.parseFileContents( result, contents, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::malformedRecord );
    }
}

/// Verify `readAndParse()` reads through the configured path and handles missing files.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM reads and parses configured files", "[flowRPM]" )
{
    flowRPM              app;
    flowRPM::parseResult result;
    const timespec       now{ 1775430290, 145131374 };
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::readAndParse( result, now );
    #endif
    // clang-format on

    SECTION( "a missing file reports fileReadError without crashing" )
    {
        app.m_inputPath = "/tmp/flowRPM_missing_file.txt";

        REQUIRE( app.readAndParse( result, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::fileReadError );
        REQUIRE( result.m_flowRate == Approx( app.badValue() ) );
    }

    SECTION( "a present file is parsed through the configured path" )
    {
        const std::string path = "/tmp/flowRPM_read_and_parse.txt";
        std::ofstream     ofs( path.c_str() );

        ofs << "1775430287 145131374\n";
        ofs << "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";
        ofs.close();

        app.m_inputPath = path;

        REQUIRE( app.readAndParse( result, now ) == 0 );
        REQUIRE( result.m_status == flowRPM::parseStatus::success );
        REQUIRE( result.m_flowRate == Approx( 1.9 ) );
        REQUIRE( result.m_age == Approx( 3.0 ) );
    }
}

/// Verify `appStartup()` initializes the published state and transitions to READY.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM appStartup initializes state and published status", "[flowRPM]" )
{
    flowRPM app;

    REQUIRE( app.appStartup() == 0 );
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::appStartup();
    #endif
    // clang-format on
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( indiNumberValue( app.m_indiP_status, "flow_rate" ) == Approx( app.badValue() ) );
    REQUIRE( indiNumberValue( app.m_indiP_status, "age" ) == Approx( app.badValue() ) );
}

/// Verify `appShutdown()` completes cleanly.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM appShutdown completes cleanly", "[flowRPM]" )
{
    flowRPM app;

    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.appShutdown() == 0 );
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::appStartup();
    flowRPM::appShutdown();
    #endif
    // clang-format on
}

/// Verify repeated error logging is rate-limited per status key.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM log backoff is per error key and interval", "[flowRPM]" )
{
    flowRPM app;
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::shouldLogError( "", timespec{ 0, 0 } );
    #endif
    // clang-format on

    REQUIRE( app.shouldLogError( "open_failed", timespec{ 10, 0 } ) == true );
    REQUIRE( app.shouldLogError( "open_failed", timespec{ 20, 0 } ) == false );
    REQUIRE( app.shouldLogError( "timestamp_parse_failed", timespec{ 21, 0 } ) == true );
    REQUIRE( app.shouldLogError( "timestamp_parse_failed", timespec{ 82, 0 } ) == true );
}

/// Verify `recordTelem()` forces the telemeter bookkeeping update.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM recordTelem forces telemetry bookkeeping refresh", "[flowRPM]" )
{
    flowRPM app;

    app.m_flowRate          = 2.5;
    app.m_age               = 7.0;
    app.m_haveValidReading  = true;
    app.m_lastTelemFlowRate = std::numeric_limits<double>::quiet_NaN();
    app.m_lastTelemValid    = false;

    REQUIRE( app.recordTelem( static_cast<const MagAOX::logger::telem_flowrpm *>( nullptr ) ) == 0 );
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::recordTelem( static_cast<const MagAOX::logger::telem_flowrpm *>( nullptr ) );
    #endif
    // clang-format on
    REQUIRE( app.m_lastTelemFlowRate == Approx( 2.5 ) );
    REQUIRE( app.m_lastTelemValid == true );
}

/// Verify `appLogic()` executes the nominal runtime control flow.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM appLogic drives end-to-end display-state updates", "[flowRPM]" )
{
    flowRPM app;

    REQUIRE( app.appStartup() == 0 );
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::appStartup();
    flowRPM::appLogic();
    #endif
    // clang-format on

    SECTION( "a valid file publishes a fresh good reading" )
    {
        timespec now;
        REQUIRE( clock_gettime( CLOCK_REALTIME, &now ) == 0 );

        const std::string path = "/tmp/flowRPM_app_logic_valid.txt";
        std::ofstream     ofs( path.c_str() );

        ofs << now.tv_sec << ' ' << now.tv_nsec << '\n';
        ofs << "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";
        ofs.close();

        app.m_inputPath = path;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.haveValidReading() == true );
        REQUIRE( app.flowRate() == Approx( 1.9 ) );
        REQUIRE( app.age() >= 0.0 );
        REQUIRE( app.age() < app.maxAge() );
        REQUIRE( app.m_lastErrorKey.empty() );
    }

    SECTION( "a missing file publishes the sentinel and records the error key" )
    {
        app.m_inputPath = "/tmp/flowRPM_app_logic_missing.txt";
        std::filesystem::remove( app.m_inputPath );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.haveValidReading() == false );
        REQUIRE( app.flowRate() == Approx( app.badValue() ) );
        REQUIRE( app.age() == Approx( app.badValue() ) );
        REQUIRE( app.m_lastErrorKey == "open_failed" );
        REQUIRE( indiNumberValue( app.m_indiP_status, "flow_rate" ) == Approx( app.badValue() ) );
        REQUIRE( indiNumberValue( app.m_indiP_status, "age" ) == Approx( app.badValue() ) );
    }

    SECTION( "a valid file clears a previously latched error key" )
    {
        timespec now;
        REQUIRE( clock_gettime( CLOCK_REALTIME, &now ) == 0 );

        const std::string path = "/tmp/flowRPM_app_logic_recovery.txt";
        std::ofstream     ofs;

        app.m_inputPath = path;
        std::filesystem::remove( path );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_lastErrorKey == "open_failed" );

        ofs.open( path.c_str() );
        ofs << now.tv_sec << ' ' << now.tv_nsec << '\n';
        ofs << "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";
        ofs.close();

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.haveValidReading() == true );
        REQUIRE( app.flowRate() == Approx( 1.9 ) );
        REQUIRE( app.m_lastErrorKey.empty() );
    }

    SECTION( "a transient partial write preserves the last good display state" )
    {
        timespec now;
        REQUIRE( clock_gettime( CLOCK_REALTIME, &now ) == 0 );

        const std::string path = "/tmp/flowRPM_app_logic_partial.txt";
        std::ofstream     ofs( path.c_str() );

        ofs << now.tv_sec << ' ' << now.tv_nsec << '\n';
        ofs << "36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'\n";
        ofs.close();

        app.m_inputPath = path;

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.haveValidReading() == true );
        REQUIRE( app.flowRate() == Approx( 1.9 ) );

        REQUIRE( clock_gettime( CLOCK_REALTIME, &now ) == 0 );
        ofs.open( path.c_str() );
        ofs << now.tv_sec << ' ' << now.tv_nsec << '\n';
        ofs.close();

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.haveValidReading() == true );
        REQUIRE( app.flowRate() == Approx( 1.9 ) );
        REQUIRE( app.age() >= 0.0 );
        REQUIRE( app.age() < app.maxAge() );
        REQUIRE( app.m_lastErrorKey.empty() );
    }
}

/// Verify `appLogic()` returns errors when internal runtime steps fail.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM appLogic returns errors when internal steps fail", "[flowRPM]" )
{
    flowRPMFaultInject app;

    REQUIRE( app.appStartup() == 0 );
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::appStartup();
    flowRPM::appLogic();
    #endif
    // clang-format on

    SECTION( "readAndParse failures propagate as appLogic errors" )
    {
        app.fault( flowRPMFaultInject::faultMode::readAndParse );

        REQUIRE( app.appLogic() == -1 );
    }

    SECTION( "publishResult failures propagate as appLogic errors" )
    {
        app.fault( flowRPMFaultInject::faultMode::publishResult );

        REQUIRE( app.appLogic() == -1 );
    }

    SECTION( "recordFlow failures propagate as appLogic errors" )
    {
        app.fault( flowRPMFaultInject::faultMode::recordFlow );

        REQUIRE( app.appLogic() == -1 );
    }
}

/// Verify `reconcileResult()` and `publishResult()` manage held and invalid display state.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM display-state reconciliation", "[flowRPM]" )
{
    flowRPM::parseResult result;
    flowRPM::parseResult lastGood;
    flowRPM::parseResult partialWrite;
    flowRPM              app;
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::publishResult( result );
    flowRPM::reconcileResult( result, timespec{ 0, 0 } );
    #endif
    // clang-format on

    SECTION( "publishResult updates valid and invalid states" )
    {
        result.m_status   = flowRPM::parseStatus::success;
        result.m_flowRate = 1.9;
        result.m_age      = 3.0;
        result.m_sourceTs = timespec{ 1, 2 };

        REQUIRE( app.publishResult( result ) == 0 );
        REQUIRE( app.haveValidReading() == true );
        REQUIRE( app.flowRate() == Approx( 1.9 ) );
        REQUIRE( app.age() == Approx( 3.0 ) );

        result.m_status   = flowRPM::parseStatus::missingRecord;
        result.m_flowRate = app.badValue();
        result.m_age      = app.badValue();

        REQUIRE( app.publishResult( result ) == 0 );
        REQUIRE( app.haveValidReading() == false );
        REQUIRE( app.flowRate() == Approx( app.badValue() ) );
    }

    SECTION( "last good value is held through transient partial-write failures" )
    {
        lastGood.m_status   = flowRPM::parseStatus::success;
        lastGood.m_flowRate = 1.9;
        lastGood.m_age      = 2.0;
        lastGood.m_sourceTs = timespec{ 100, 0 };

        REQUIRE( app.publishResult( lastGood ) == 0 );

        partialWrite.m_status   = flowRPM::parseStatus::missingRecord;
        partialWrite.m_flowRate = app.badValue();
        partialWrite.m_age      = app.badValue();

        flowRPM::parseResult display = app.reconcileResult( partialWrite, timespec{ 120, 0 } );

        REQUIRE( display.m_status == flowRPM::parseStatus::success );
        REQUIRE( display.m_flowRate == Approx( 1.9 ) );
        REQUIRE( display.m_age == Approx( 20.0 ) );
    }

    SECTION( "sentinel is published once the last good value ages past maxAge" )
    {
        lastGood.m_status   = flowRPM::parseStatus::success;
        lastGood.m_flowRate = 1.9;
        lastGood.m_age      = 2.0;
        lastGood.m_sourceTs = timespec{ 100, 0 };

        REQUIRE( app.publishResult( lastGood ) == 0 );

        partialWrite.m_status   = flowRPM::parseStatus::missingRecord;
        partialWrite.m_flowRate = app.badValue();
        partialWrite.m_age      = app.badValue();

        flowRPM::parseResult display = app.reconcileResult( partialWrite, timespec{ 161, 0 } );

        REQUIRE( display.m_status == flowRPM::parseStatus::staleReading );
        REQUIRE( display.m_flowRate == Approx( app.badValue() ) );
        REQUIRE( display.m_age == Approx( 61.0 ) );
    }
}

/// Verify `statusKey()` maps every parser status to a stable log key.
/**
 * \ingroup flowRPM_unit_test
 */
TEST_CASE( "flowRPM statusKey maps parse statuses consistently", "[flowRPM]" )
{
    // clang-format off
    #ifdef FLOWRPM_TEST_DOXYGEN_REF
    flowRPM::statusKey( flowRPM::parseStatus::success );
    #endif
    // clang-format on
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::success ) == "success" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::fileReadError ) == "open_failed" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::missingTimestamp ) == "missing_timestamp" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::malformedTimestamp ) == "timestamp_parse_failed" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::missingRecord ) == "missing_record" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::malformedRecord ) == "record_parse_failed" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::wrongDescriptor ) == "wrong_descriptor" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::wrongUnits ) == "wrong_units" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::badStatus ) == "bad_status" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::badValue ) == "bad_value" );
    REQUIRE( flowRPM::statusKey( flowRPM::parseStatus::staleReading ) == "stale" );
    REQUIRE( flowRPM::statusKey( static_cast<flowRPM::parseStatus>( 999 ) ) == "unknown" );
}

} // namespace flowRPMTest
} // namespace libXWCTest
