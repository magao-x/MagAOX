/** \file sysMonitorDiagnostics_test.cpp
 * \brief Offline lifecycle, INDI and telemetry integration tests for system diagnostics.
 * \ingroup sysMonitor_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../sysMonitor.hpp"
#include "../../stateRuleEngine/indiCompRuleConfig.hpp"

#include <filesystem>
#include <limits>
#include <sys/stat.h>

namespace libXWCTest
{
namespace sysMonitorTest
{

/// \cond SYSMONITOR_TEST_HARNESS
using namespace MagAOX::app;
using namespace MagAOX::logger;

const std::string gpuUUID = "GPU-12345678-1234-1234-1234-123456789abc";
const std::string gpuCSV  = gpuUUID + ",00000000:01:00.0,42,35,12,100,1000,1500,1500,5000\n";

/// Serialize INDI updates only into memory, without activating a connection.
class diagnosticDriver : public indiDriver<MagAOXApp<true>>
{
    int m_input{ -1 }; ///< Input FIFO owned by the inactive test driver.

  public:
    mutable std::vector<std::string> m_xml; ///< Captured production XML messages.

    /// Open private fixture FIFOs and enable serialization into the capture vector.
    explicit diagnosticDriver( MagAOXApp<true> *parent /**< [in] owner of private FIFO names */ );

    /// Release the inactive driver's input descriptor.
    ~diagnosticDriver();

    /// Capture serialized messages without writing to a server or FIFO.
    void sendXml( const std::string &xml /**< [in] serialized message */ ) const override;
};

diagnosticDriver::diagnosticDriver( MagAOXApp<true> *parent )
    : indiDriver<MagAOXApp<true>>( parent, "sysmon-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( stat( parent->driverInName().c_str(), &input ) == 0 );
    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        const int   fd = std::stoi( entry.path().filename() );
        struct stat candidate{};
        if( fstat( fd, &candidate ) == 0 && candidate.st_dev == input.st_dev && candidate.st_ino == input.st_ino )
        {
            m_input = fd;
        }
    }

    REQUIRE( m_input >= 0 );
}

diagnosticDriver::~diagnosticDriver()
{
    if( m_input >= 0 )
    {
        close( m_input );
    }

    setInputFd( -1 );
}

void diagnosticDriver::sendXml( const std::string &xml ) const
{
    m_xml.push_back( xml );
}

/// Run only the new diagnostic lifecycle with injected collectors and private logging.
class diagnosticHarness : public sysMonitor
{
  public:
    std::string                         m_root;                               ///< Owned temporary fixture directory.
    size_t                              m_gpuQueries{ 0 }, m_irqQueries{ 0 }; ///< Number of fake collector invocations.
    MagAOX::sys::boundedCommand::result m_result;                             ///< Next fake command result.
    std::vector<telem_irq::groupT>      m_groups;                             ///< Next fake IRQ snapshot.

    using sysMonitor::acceptGPUResult;
    using sysMonitor::config;
    using sysMonitor::loadDiagnosticsConfig;
    using sysMonitor::m_gpuEnabled;
    using sysMonitor::m_gpuExpected;
    using sysMonitor::m_gpuInfoProperties;
    using sysMonitor::m_gpuInterval;
    using sysMonitor::m_gpuProperties;
    using sysMonitor::m_gpus;
    using sysMonitor::m_gpuStatus;
    using sysMonitor::m_gpuTimeout;
    using sysMonitor::m_indiP_irqStatus;
    using sysMonitor::m_indiP_irqWarning;
    using sysMonitor::m_irqInterval;
    using sysMonitor::m_irqRules;
    using sysMonitor::m_irqs;
    using sysMonitor::m_lastGPUs;
    using sysMonitor::m_lastIRQs;
    using sysMonitor::pollDiagnostics;
    using sysMonitor::publishDiagnostics;
    using sysMonitor::recordGPUs;
    using sysMonitor::recordIRQs;
    using sysMonitor::startupDiagnostics;

    /// Prepare private logging and a serialization-only INDI driver.
    diagnosticHarness();

    /// Flush local logs and remove the private fixture directory.
    ~diagnosticHarness();

    /// Supply the fake GPU result without invoking a command.
    MagAOX::sys::boundedCommand::result queryGPUs() override;

    /// Supply fake IRQ results without reading proc/sys files.
    std::vector<telem_irq::groupT> queryIRQs( double time /**< [in] wall-clock check time */ ) override;

    /// Return the local message capture driver.
    diagnosticDriver &driver();

    /// Load a tracked diagnostic configuration fragment.
    int loadExample( const std::string &file /**< [in] basename under the app's config directory */ );
};

diagnosticHarness::diagnosticHarness()
{
    char directory[] = "/tmp/sysmon-diagnostics-XXXXXX";
    REQUIRE( mkdtemp( directory ) != nullptr );
    m_root       = directory;
    m_configName = "sysmon-test";
    m_tel.logPath( m_root );
    m_tel.logName( "telemetry" );
    m_tel.logExt( "bintel" );
    m_tel.logLevel( logPrio::LOG_TELEM );
    m_log.logPath( m_root );
    m_log.logName( "logs" );
    m_driverInName   = m_root + "/in";
    m_driverOutName  = m_root + "/out";
    m_driverCtrlName = m_root + "/ctrl";
    for( const auto &file : { m_driverInName, m_driverOutName, m_driverCtrlName } )
    {
        REQUIRE( mkfifo( file.c_str(), 0600 ) == 0 );
    }

    m_indiDriver = new diagnosticDriver( this );
    REQUIRE( m_indiDriver->good() );
    m_result.m_status = MagAOX::sys::boundedCommand::status::success;
    m_result.m_output = gpuCSV;
}

diagnosticHarness::~diagnosticHarness()
{
    delete m_indiDriver;
    m_indiDriver = nullptr;
    m_tel.logShutdown( true );
    m_tel.logThreadExec();
    m_log.logShutdown( true );
    m_log.logThreadExec();
    std::filesystem::remove_all( m_root );
}

MagAOX::sys::boundedCommand::result diagnosticHarness::queryGPUs()
{
    ++m_gpuQueries;
    return m_result;
}

std::vector<telem_irq::groupT> diagnosticHarness::queryIRQs( double time )
{
    ++m_irqQueries;
    for( auto &group : m_groups )
    {
        group.m_checkTime = time;
    }

    return m_groups;
}

diagnosticDriver &diagnosticHarness::driver()
{
    return *static_cast<diagnosticDriver *>( m_indiDriver );
}

int diagnosticHarness::loadExample( const std::string &file )
{
    setupConfig();
    const auto app = std::filesystem::read_symlink( "/proc/self/exe" ).parent_path().parent_path();
    config.readConfig( ( app / "config" / file ).string() );
    return loadDiagnosticsConfig( config );
}

/// Construct a singleton fake IRQ result.
telem_irq::groupT irqGroup( bool warning /**< [in] whether to inject a mismatch */ )
{
    telem_irq::groupT group;
    group.m_name     = "camera";
    group.m_expected = "45";
    group.m_status   = warning ? "requested affinity mismatch" : "ok";
    group.m_warning  = warning;
    group.m_irqs.push_back( { 123, warning ? "0-63" : "45", "45" } );
    return group;
}
/// \endcond

/// Verify disabled collectors and independent monotonic interval scheduling.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor diagnostics poll only when enabled and due", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupDiagnostics();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    #endif
    // clang-format on
    diagnosticHarness app;
    REQUIRE( app.startupDiagnostics() == 0 );
    const auto epoch = std::chrono::steady_clock::time_point{};
    REQUIRE( app.pollDiagnostics( epoch ) == 0 );
    REQUIRE( app.m_gpuQueries == 0 );
    REQUIRE( app.m_irqQueries == 0 );

    app.m_gpuEnabled = true;
    REQUIRE( app.pollDiagnostics( epoch ) == 0 );
    REQUIRE( app.m_gpuQueries == 1 );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 4 ) ) == 0 );
    REQUIRE( app.m_gpuQueries == 1 );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 5 ) ) == 0 );
    REQUIRE( app.m_gpuQueries == 2 );
    REQUIRE( app.m_irqQueries == 0 );
}

/// Verify failed/partial/missing GPU reads preserve identity and publish explicit invalidity.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor GPU failures and recovery publish validity", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptGPUResult(*(MagAOX::sys::boundedCommand::result *)nullptr, 0);
    sysMonitor::publishDiagnostics();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_gpuEnabled  = true;
    app.m_gpuExpected = { gpuUUID };
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptGPUResult( app.m_result, 100 );
    REQUIRE( app.publishDiagnostics() == 0 );
    const auto key = sysMonitorDetail::gpuKey( gpuUUID );
    REQUIRE( app.m_gpuProperties.at( key ).getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_gpuProperties.at( key )["clock_memory"].get<double>() == 5000 );
    REQUIRE( app.m_gpuProperties.at( key )["valid"].get<unsigned>() == 255 );
    REQUIRE( app.m_gpuInfoProperties.at( key )["uuid"].get<std::string>() == gpuUUID );
    app.driver().m_xml.clear();
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.driver().m_xml.empty() );

    app.m_result.m_status = MagAOX::sys::boundedCommand::status::timeout;
    app.acceptGPUResult( app.m_result, 200 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpus.size() == 1 );
    REQUIRE( app.m_gpus[0].m_temperature == 42 );
    REQUIRE( app.m_gpus[0].m_sampleTime == 100 );
    REQUIRE( app.m_gpuProperties.at( key )["valid"].get<unsigned>() == 0 );
    REQUIRE( app.m_gpuProperties.at( key ).getState() == INDI_ALERT );

    app.m_result.m_status = MagAOX::sys::boundedCommand::status::success;
    app.m_result.m_output.clear();
    app.acceptGPUResult( app.m_result, 300 );
    REQUIRE( app.m_gpus.size() == 1 );
    REQUIRE( app.m_gpus[0].m_valid == 0 );
    app.m_result.m_output = gpuCSV;
    app.acceptGPUResult( app.m_result, 400 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpuProperties.at( key ).getState() == INDI_OK );
    REQUIRE( app.m_gpus[0].m_sampleTime == 400 );
}

/// Verify IRQ warnings start unchecked, include failures, and clear without repeated property traffic.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor IRQ warnings recover independently of GPU errors", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupDiagnostics();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    sysMonitor::publishDiagnostics();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_gpuEnabled = true;
    app.m_irqRules.push_back( { "camera", "edt", "camera", "", true, true } );
    app.m_groups = { irqGroup( true ) };
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.m_indiP_irqWarning["active"].getSwitchState() == pcf::IndiElement::On );
    const auto epoch      = std::chrono::steady_clock::time_point{};
    app.m_result.m_status = MagAOX::sys::boundedCommand::status::timeout;
    REQUIRE( app.pollDiagnostics( epoch ) == 0 );
    REQUIRE( app.m_gpuQueries == 1 );
    REQUIRE( app.m_irqQueries == 1 );
    REQUIRE( app.m_indiP_irqWarning.getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_indiP_irqStatus["summary"].get<std::string>() == "warning" );
    app.driver().m_xml.clear();
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.driver().m_xml.empty() );

    app.m_groups = { irqGroup( false ) };
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 6 ) ) == 0 );
    REQUIRE( app.m_indiP_irqWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_irqStatus["summary"].get<std::string>() == "ok" );
    REQUIRE( app.m_gpus.empty() );
}

/// Verify periodic records retain actual sample times and changed-state suppression is per instance.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor telemetry records first changes and forced snapshots", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::recordGPUs(false);
    sysMonitor::recordIRQs(false);
    sysMonitor::recordTelem((telem_gpu *)nullptr);
    sysMonitor::recordTelem((telem_irq *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_gpuEnabled = true;
    app.m_irqRules.push_back( { "camera", "edt", "camera", "", true, true } );
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptGPUResult( app.m_result, 100 );
    app.m_irqs                = { irqGroup( false ) };
    app.m_irqs[0].m_checkTime = 100;
    REQUIRE( app.recordGPUs() == 0 );
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( app.m_lastGPUs.size() == 1 );
    REQUIRE( app.m_lastIRQs.size() == 1 );
    REQUIRE( telem_gpu::lastRecord.tv_sec != 0 );
    REQUIRE( telem_irq::lastRecord.tv_sec != 0 );

    telem_gpu::lastRecord = {};
    telem_irq::lastRecord = {};
    app.acceptGPUResult( app.m_result, 200 );
    app.m_irqs[0].m_checkTime = 200;
    REQUIRE( app.recordGPUs() == 0 );
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( telem_gpu::lastRecord.tv_sec == 0 );
    REQUIRE( telem_irq::lastRecord.tv_sec == 0 );
    REQUIRE( app.recordTelem( static_cast<const telem_gpu *>( nullptr ) ) == 0 );
    REQUIRE( app.recordTelem( static_cast<const telem_irq *>( nullptr ) ) == 0 );
    REQUIRE( telem_gpu::lastRecord.tv_sec != 0 );
    REQUIRE( telem_irq::lastRecord.tv_sec != 0 );
    REQUIRE( app.m_lastGPUs[0].m_sampleTime == 200 );
    REQUIRE( app.m_lastIRQs[0].m_checkTime == 200 );
}

/// Verify shipped host policies match the merged dedicated IRQ structure.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor example configs omit NVIDIA pinning and retain PVCAM identities", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::loadDiagnosticsConfig(*(mx::app::appConfigurator *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    const auto        file = GENERATE( "sysMonRTC-diagnostics.conf", "sysMonICC-diagnostics.conf" );
    REQUIRE( app.loadExample( file ) == 0 );
    REQUIRE( app.m_gpuEnabled );
    REQUIRE( app.m_gpuInterval == 5 );
    REQUIRE( app.m_gpuTimeout == 2 );
    REQUIRE( app.m_irqRules.size() == ( std::string( file ).find( "RTC" ) != std::string::npos ? 3 : 5 ) );
    for( const auto &rule : app.m_irqRules )
    {
        REQUIRE( rule.m_match != "nvidia" );
        REQUIRE( rule.m_required );
        REQUIRE( rule.m_requireEffective );
        if( rule.m_name == "camflowfs" )
        {
            REQUIRE( rule.m_match == "pvcamPCIE_1" );
        }
        if( rule.m_name == "camllowfs" )
        {
            REQUIRE( rule.m_match == "pvcamPCIE_0" );
        }
    }
}

/// Verify invalid intervals cannot silently disable monitoring under production fast-math flags.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor rejects invalid diagnostic intervals", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::loadDiagnosticsConfig(*(mx::app::appConfigurator *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.setupConfig();
    app.m_gpuInterval =
        GENERATE( 0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() );
    REQUIRE( app.loadDiagnosticsConfig( app.config ) < 0 );
}

/// Verify the deployed warning rules evaluate the real published switch and clear after recovery.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor warning examples raise and clear via stateRuleEngine", "[sysMonitor][diagnostics]" )
{
    diagnosticHarness app;
    app.m_irqRules.push_back( { "camera", "edt", "camera", "", true, true } );
    REQUIRE( app.startupDiagnostics() == 0 );
    const auto               directory = std::filesystem::read_symlink( "/proc/self/exe" ).parent_path().parent_path();
    mx::app::appConfigurator cfg;
    cfg.readConfig( ( directory / "config/sysmon-warning.conf" ).string() );
    indiRuleMaps                        rules;
    std::map<std::string, ruleRuleKeys> keys;
    loadRuleConfig( rules, keys, cfg );
    REQUIRE( rules.rules.size() == 2 );
    for( const auto &device : { "sysMonRTC", "sysMonICC" } )
    {
        auto &property = *rules.props.at( std::string( device ) + ".irq_warning" );
        property       = app.m_indiP_irqWarning;
        property.setDevice( device );
        auto *rule = rules.rules.at( std::string( device ) + "-irq-warning" );
        REQUIRE( rule->priority() == rulePriority::warning );
        REQUIRE( rule->value() );
        property["active"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE_FALSE( rule->value() );
    }
}

/// Verify optional-only startup is visibly unchecked without raising a missing-device alarm.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor optional IRQ startup remains unchecked", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupDiagnostics();
    sysMonitor::publishDiagnostics();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_irqRules.push_back( { "camera", "edt", "camera", "", false, true } );
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.m_indiP_irqStatus["summary"].get<std::string>() == "unchecked" );
    REQUIRE( app.m_indiP_irqStatus.getState() == INDI_IDLE );
    REQUIRE( app.m_indiP_irqWarning["active"].getSwitchState() == pcf::IndiElement::Off );
}

/// Verify malformed policy booleans fail configuration instead of silently suppressing warnings.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor rejects malformed policy booleans", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::loadDiagnosticsConfig(*(mx::app::appConfigurator *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.setupConfig();
    const auto        setting = GENERATE( "gpu.enabled", "irq.camera.required", "irq.camera.requireEffective" );
    const std::string key     = setting;
    const auto        dot     = key.rfind( '.' );
    const std::string path    = app.m_root + "/invalid.conf";
    std::ofstream     file( path );
    file << "[irq.camera]\nmatch=edt\ncpuset=camera\n";
    file << '[' << key.substr( 0, dot ) << "]\n" << key.substr( dot + 1 ) << "=garbage\n";
    file.close();
    REQUIRE( app.config.readConfig( path ) == 0 );
    REQUIRE( app.loadDiagnosticsConfig( app.config ) < 0 );
}

} // namespace sysMonitorTest
} // namespace libXWCTest
