/** \file flowRPM.hpp
 * \brief The MagAO-X flowRPM application header.
 *
 * \ingroup flowRPM_files
 */

#ifndef flowRPM_hpp
#define flowRPM_hpp

#include <cerrno>
#include <cmath>
#include <fstream>
#include <sstream>

#include "../../libMagAOX/libMagAOX.hpp" // Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup flowRPM
 * \brief The MagAO-X application to convert a file-based fan reading into flow telemetry.
 *
 * <a href="../handbook/operating/software/apps/flowRPM.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup flowRPM_files
 * \ingroup flowRPM
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X flow-from-RPM monitor.
/**
 * \ingroup flowRPM
 */
class flowRPM : public MagAOXApp<true>, public dev::telemeter<flowRPM>
{
    // Give the test harness access.
    friend class flowRPM_test;

    friend class dev::telemeter<flowRPM>;

  public:
    /// Status returned by the file parser.
    enum class parseStatus
    {
        success,
        fileReadError,
        missingTimestamp,
        malformedTimestamp,
        missingRecord,
        malformedRecord,
        wrongDescriptor,
        wrongUnits,
        badStatus,
        badValue,
        staleReading
    };

    /// Parsed result of the current file contents.
    struct parseResult
    {
        parseStatus m_status{ parseStatus::fileReadError }; ///< Outcome of the parse attempt.
        double      m_flowRate{ -999.0 };                   ///< Displayed flow rate in LPM or the bad-value sentinel.
        double      m_age{ -999.0 };                        ///< Displayed age in seconds or the bad-value sentinel.
        timespec    m_sourceTs{ 0, 0 };                     ///< Parsed source timestamp when available.
    };

    /// The telemeter base type.
    typedef dev::telemeter<flowRPM> telemeterT;

  protected:
    /** \name Configurable Parameters - Data
     *
     * @{
     */

    /// Path to the file written by the systemd producer.
    std::string m_inputPath{ "/tmp/fac_flow.txt" };

    /// Maximum allowed source age in seconds before the reading is treated as stale.
    double m_maxAge{ 60.0 };

    /// Expected fan descriptor in the pipe-delimited record.
    std::string m_fanDescriptor{ "CHA_FAN1" };

    /// Bad-value sentinel published on parse or availability failures.
    double m_badValue{ -999.0 };

    /// Minimum interval between repeated logs for the same persistent error.
    double m_errorLogInterval{ 60.0 };

    ///@}

    /** \name Runtime State - Data
     *
     * @{
     */

    /// Current published flow rate in LPM or the bad-value sentinel.
    double m_flowRate{ -999.0 };

    /// Current published age in seconds or the bad-value sentinel.
    double m_age{ -999.0 };

    /// Timestamp parsed from the input file for the currently displayed value.
    timespec m_sourceTs{ 0, 0 };

    /// Whether the current displayed value is valid.
    bool m_haveValidReading{ false };

    /// Last flow value written to telemetry.
    double m_lastTelemFlowRate{ std::numeric_limits<double>::quiet_NaN() };

    /// Last validity state written to telemetry.
    bool m_lastTelemValid{ false };

    /// Time of the most recent error log emission.
    timespec m_lastErrorLogTs{ 0, 0 };

    /// Key for the most recently logged error class.
    std::string m_lastErrorKey;

    /// Read-only status property exposing flow rate and age.
    pcf::IndiProperty m_indiP_status;

    ///@}

  public:
    /// Default c'tor.
    flowRPM();

    /// D'tor, declared and defined for noexcept.
    ~flowRPM() noexcept;

    /// Set up the application configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    virtual int
    loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration from which to load */ );

    /// Load the application configuration.
    virtual void loadConfig();

    /// Perform application startup.
    virtual int appStartup();

    /// Implementation of the FSM for flowRPM.
    virtual int appLogic();

    /// Shut the application down.
    virtual int appShutdown();

    /** \name Parser Helpers
     *
     * @{
     */

    /// Parse a source timestamp line into a timespec.
    int parseTimestamp( timespec          &ts,  /**< [out] parsed timestamp */
                        const std::string &line /**< [in] source timestamp line */
    ) const;

    /// Parse the sensor record line into a flow rate in LPM.
    parseStatus parseRecordLine( double            &flowRate, /**< [out] parsed flow rate in LPM */
                                 const std::string &line      /**< [in] sensor record line */
    ) const;

    /// Parse the two-line file contents using a supplied current time.
    int parseFileContents( parseResult       &result,   /**< [out] parse result */
                           const std::string &contents, /**< [in] raw file contents */
                           const timespec    &now       /**< [in] current time for age calculation */
    ) const;

    /// Read the configured file and parse its current contents.
    virtual int readAndParse( parseResult    &result, /**< [out] parse result */
                              const timespec &now     /**< [in] current time for age calculation */
    ) const;

    /// Get the string key used for a parse status in log rate limiting.
    static std::string statusKey( parseStatus status /**< [in] parse status to stringify */ );

    /// Get the configured input path.
    const std::string &inputPath() const;

    /// Get the configured maximum reading age in seconds.
    double maxAge() const;

    /// Get the configured fan descriptor.
    const std::string &fanDescriptor() const;

    /// Get the configured bad-value sentinel.
    double badValue() const;

    /// Get the configured repeated-error log interval.
    double errorLogInterval() const;

    /// Get the currently published flow rate.
    double flowRate() const;

    /// Get the currently published age.
    double age() const;

    /// Get whether the current published value is valid.
    bool haveValidReading() const;

    ///@}

    /** \name Telemeter
     *
     * @{
     */

    /// Check whether telemetry records need to be forced.
    int checkRecordTimes();

    /// Record telemetry when requested by the telemeter helper.
    int recordTelem( const logger::telem_flowrpm * /**< [in] telemetry tag used for overload resolution */ );

    /// Record the currently displayed flow state to telemetry.
    virtual int recordFlow( bool force = false /**< [in] force a telemetry record even if unchanged */ );

    /// Reconcile a newly parsed result against the currently displayed state.
    parseResult reconcileResult( const parseResult &result, /**< [in] newly parsed file result */
                                 const timespec    &now     /**< [in] current time */
    ) const;

    /// Publish a parse result to INDI and runtime state.
    virtual int publishResult( const parseResult &result /**< [in] parse result to publish */ );

    /// Determine whether a repeated error should be logged now.
    bool shouldLogError( const std::string &key, /**< [in] error key under consideration */
                         const timespec    &now  /**< [in] current time */
    );

    ///@}
};

namespace flowRPMDetail
{

/// Trim leading and trailing ASCII whitespace from a token.
inline std::string trimToken( const std::string &token )
{
    const std::string::size_type first = token.find_first_not_of( " \t\r" );

    if( first == std::string::npos )
    {
        return "";
    }

    const std::string::size_type last = token.find_last_not_of( " \t\r" );
    return token.substr( first, last - first + 1 );
}

/// Convert a timespec to fractional seconds.
inline double timespecToDouble( const timespec &ts )
{
    return static_cast<double>( ts.tv_sec ) + 1e-9 * static_cast<double>( ts.tv_nsec );
}

/// Measure elapsed seconds between two timestamps.
inline double elapsedSeconds( const timespec &start, const timespec &end )
{
    return timespecToDouble( end ) - timespecToDouble( start );
}

/// Split a pipe-delimited sensor line into trimmed fields.
inline std::vector<std::string> splitPipeDelimited( const std::string &line )
{
    std::vector<std::string> fields;
    std::stringstream        ss( line );
    std::string              field;

    while( std::getline( ss, field, '|' ) )
    {
        fields.push_back( trimToken( field ) );
    }

    return fields;
}

/// Split file contents into non-empty logical lines.
inline std::vector<std::string> splitLogicalLines( const std::string &contents )
{
    std::vector<std::string> lines;
    std::stringstream        ss( contents );
    std::string              line;

    while( std::getline( ss, line ) )
    {
        if( !line.empty() && line.back() == '\r' )
        {
            line.pop_back();
        }

        if( trimToken( line ).empty() )
        {
            continue;
        }

        lines.push_back( line );
    }

    return lines;
}

} // namespace flowRPMDetail

inline flowRPM::flowRPM() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline flowRPM::~flowRPM() noexcept
{
}

inline void flowRPM::setupConfig()
{
    config.add( "input.path",
                "",
                "input.path",
                argType::Required,
                "input",
                "path",
                false,
                "string",
                "Path to the file containing the two-line flow RPM record." );
    config.add( "input.maxAge",
                "",
                "input.maxAge",
                argType::Required,
                "input",
                "maxAge",
                false,
                "double",
                "Maximum source age in seconds before the reading is treated as stale. Default is 60." );
    config.add( "input.fanDescriptor",
                "",
                "input.fanDescriptor",
                argType::Required,
                "input",
                "fanDescriptor",
                false,
                "string",
                "Expected descriptor token in the source file. Default is CHA_FAN1." );
    config.add( "input.badValue",
                "",
                "input.badValue",
                argType::Required,
                "input",
                "badValue",
                false,
                "double",
                "Bad-value sentinel published when the file can not be read or parsed. Default is -999." );
    config.add( "input.errorLogInterval",
                "",
                "input.errorLogInterval",
                argType::Required,
                "input",
                "errorLogInterval",
                false,
                "double",
                "Minimum interval in seconds between repeated logs of the same persistent error. Default is 60." );

    TELEMETER_SETUP_CONFIG( config );
}

inline int flowRPM::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_inputPath, "input.path" );
    _config( m_maxAge, "input.maxAge" );
    _config( m_fanDescriptor, "input.fanDescriptor" );
    _config( m_badValue, "input.badValue" );
    _config( m_errorLogInterval, "input.errorLogInterval" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

inline void flowRPM::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
    }
}

inline int flowRPM::appStartup()
{
    createROIndiNumber( m_indiP_status, "status", "Flow Status" );
    indi::addNumberElement<double>( m_indiP_status, "flow_rate", -1e9, 1e9, 0.0, "%0.3f", "Flow Rate [LPM]" );
    indi::addNumberElement<double>( m_indiP_status, "age", -1e9, 1e9, 0.0, "%0.3f", "Age [s]" );
    m_indiP_status["flow_rate"] = m_badValue;
    m_indiP_status["age"]       = m_badValue;
    registerIndiPropertyReadOnly( m_indiP_status );

    TELEMETER_APP_STARTUP;

    state( stateCodes::READY );

    return 0;
}

inline int flowRPM::appLogic()
{
    timespec    now;
    parseResult result;
    parseResult displayResult;

    clock_gettime( CLOCK_REALTIME, &now );

    if( readAndParse( result, now ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "unexpected failure while reading flow file" } );
    }

    displayResult = reconcileResult( result, now );

    const bool wasValid = m_haveValidReading;
    const bool isValid  = ( displayResult.m_status == parseStatus::success );

    if( publishResult( displayResult ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "failed to publish flow result" } );
    }

    if( isValid )
    {
        if( !wasValid )
        {
            log<text_log>( "Recovered valid flow reading from " + m_inputPath + ".", logPrio::LOG_NOTICE );
        }

        m_lastErrorKey.clear();
    }
    else
    {
        const std::string key = statusKey( result.m_status );

        if( shouldLogError( key, now ) )
        {
            log<software_error>( { __FILE__, __LINE__, "flowRPM " + key + " for " + m_inputPath } );
        }
    }

    if( recordFlow() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error recording flow telemetry" } );
    }

    TELEMETER_APP_LOGIC;

    return 0;
}

inline int flowRPM::appShutdown()
{
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int flowRPM::parseTimestamp( timespec &ts, const std::string &line ) const
{
    std::stringstream ss( line );
    long long         sec  = 0;
    long long         nsec = 0;

    if( !( ss >> sec >> nsec ) )
    {
        return -1;
    }

    char trailing = '\0';
    if( ss >> trailing )
    {
        return -1;
    }

    if( nsec < 0 || nsec >= 1000000000LL )
    {
        return -1;
    }

    ts.tv_sec  = static_cast<time_t>( sec );
    ts.tv_nsec = static_cast<long>( nsec );

    return 0;
}

inline flowRPM::parseStatus flowRPM::parseRecordLine( double &flowRate, const std::string &line ) const
{
    const std::vector<std::string> fields = flowRPMDetail::splitPipeDelimited( line );

    if( fields.size() != 6 )
    {
        return parseStatus::malformedRecord;
    }

    if( fields[1] != m_fanDescriptor )
    {
        return parseStatus::wrongDescriptor;
    }

    if( fields[4] != "RPM" )
    {
        return parseStatus::wrongUnits;
    }

    if( fields[5] != "'OK'" )
    {
        return parseStatus::badStatus;
    }

    char *end    = nullptr;
    errno        = 0;
    double value = std::strtod( fields[3].c_str(), &end );

    if( errno != 0 || end == fields[3].c_str() || *end != '\0' )
    {
        return parseStatus::badValue;
    }

    flowRate = value / 1000.0;

    return parseStatus::success;
}

inline int flowRPM::parseFileContents( parseResult &result, const std::string &contents, const timespec &now ) const
{
    result.m_status   = parseStatus::fileReadError;
    result.m_flowRate = m_badValue;
    result.m_age      = m_badValue;
    result.m_sourceTs = { 0, 0 };

    const std::vector<std::string> lines = flowRPMDetail::splitLogicalLines( contents );

    if( lines.empty() )
    {
        result.m_status = parseStatus::missingTimestamp;
        return 0;
    }

    if( parseTimestamp( result.m_sourceTs, lines[0] ) < 0 )
    {
        result.m_status = parseStatus::malformedTimestamp;
        return 0;
    }

    if( lines.size() < 2 )
    {
        result.m_status = parseStatus::missingRecord;
        return 0;
    }

    if( lines.size() != 2 )
    {
        result.m_status = parseStatus::malformedRecord;
        result.m_age    = std::max( 0.0, flowRPMDetail::elapsedSeconds( result.m_sourceTs, now ) );
        return 0;
    }

    double parsedFlowRate = m_badValue;

    result.m_status = parseRecordLine( parsedFlowRate, lines[1] );
    result.m_age    = std::max( 0.0, flowRPMDetail::elapsedSeconds( result.m_sourceTs, now ) );

    if( result.m_status != parseStatus::success )
    {
        return 0;
    }

    if( result.m_age > m_maxAge )
    {
        result.m_status = parseStatus::staleReading;
        return 0;
    }

    result.m_flowRate = parsedFlowRate;
    return 0;
}

inline int flowRPM::readAndParse( parseResult &result, const timespec &now ) const
{
    std::ifstream ifs( m_inputPath.c_str() );

    result.m_status   = parseStatus::fileReadError;
    result.m_flowRate = m_badValue;
    result.m_age      = m_badValue;
    result.m_sourceTs = { 0, 0 };

    if( !ifs )
    {
        return 0;
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();

    return parseFileContents( result, buffer.str(), now );
}

inline std::string flowRPM::statusKey( parseStatus status )
{
    switch( status )
    {
    case parseStatus::success:
        return "success";
    case parseStatus::fileReadError:
        return "open_failed";
    case parseStatus::missingTimestamp:
        return "missing_timestamp";
    case parseStatus::malformedTimestamp:
        return "timestamp_parse_failed";
    case parseStatus::missingRecord:
        return "missing_record";
    case parseStatus::malformedRecord:
        return "record_parse_failed";
    case parseStatus::wrongDescriptor:
        return "wrong_descriptor";
    case parseStatus::wrongUnits:
        return "wrong_units";
    case parseStatus::badStatus:
        return "bad_status";
    case parseStatus::badValue:
        return "bad_value";
    case parseStatus::staleReading:
        return "stale";
    default:
        return "unknown";
    }
}

inline const std::string &flowRPM::inputPath() const
{
    return m_inputPath;
}

inline double flowRPM::maxAge() const
{
    return m_maxAge;
}

inline const std::string &flowRPM::fanDescriptor() const
{
    return m_fanDescriptor;
}

inline double flowRPM::badValue() const
{
    return m_badValue;
}

inline double flowRPM::errorLogInterval() const
{
    return m_errorLogInterval;
}

inline double flowRPM::flowRate() const
{
    return m_flowRate;
}

inline double flowRPM::age() const
{
    return m_age;
}

inline bool flowRPM::haveValidReading() const
{
    return m_haveValidReading;
}

inline int flowRPM::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( logger::telem_flowrpm() );
}

inline int flowRPM::recordTelem( const logger::telem_flowrpm * )
{
    return recordFlow( true );
}

inline int flowRPM::recordFlow( bool force )
{
    if( force || m_flowRate != m_lastTelemFlowRate || m_haveValidReading != m_lastTelemValid )
    {
        telem<logger::telem_flowrpm>( logger::telem_flowrpm::messageT( m_flowRate, m_age, m_haveValidReading ) );

        m_lastTelemFlowRate = m_flowRate;
        m_lastTelemValid    = m_haveValidReading;
    }

    return 0;
}

inline flowRPM::parseResult flowRPM::reconcileResult( const parseResult &result, const timespec &now ) const
{
    parseResult displayResult = result;

    if( result.m_status == parseStatus::success )
    {
        return displayResult;
    }

    if( !m_haveValidReading )
    {
        return displayResult;
    }

    const double lastGoodAge = std::max( 0.0, flowRPMDetail::elapsedSeconds( m_sourceTs, now ) );

    if( lastGoodAge > m_maxAge )
    {
        displayResult.m_status   = parseStatus::staleReading;
        displayResult.m_flowRate = m_badValue;
        displayResult.m_age      = lastGoodAge;
        displayResult.m_sourceTs = m_sourceTs;
        return displayResult;
    }

    displayResult.m_status   = parseStatus::success;
    displayResult.m_flowRate = m_flowRate;
    displayResult.m_age      = lastGoodAge;
    displayResult.m_sourceTs = m_sourceTs;

    return displayResult;
}

inline int flowRPM::publishResult( const parseResult &result )
{
    m_flowRate         = result.m_flowRate;
    m_age              = result.m_age;
    m_sourceTs         = result.m_sourceTs;
    m_haveValidReading = ( result.m_status == parseStatus::success );

    updateIfChanged( m_indiP_status,
                     std::vector<std::string>( { "flow_rate", "age" } ),
                     std::vector<double>( { m_flowRate, m_age } ),
                     m_haveValidReading ? INDI_OK : INDI_ALERT );

    return 0;
}

inline bool flowRPM::shouldLogError( const std::string &key, const timespec &now )
{
    if( key != m_lastErrorKey )
    {
        m_lastErrorKey   = key;
        m_lastErrorLogTs = now;
        return true;
    }

    if( flowRPMDetail::elapsedSeconds( m_lastErrorLogTs, now ) >= m_errorLogInterval )
    {
        m_lastErrorLogTs = now;
        return true;
    }

    return false;
}

} // namespace app
} // namespace MagAOX

#endif // flowRPM_hpp
