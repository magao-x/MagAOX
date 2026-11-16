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
const std::string gpuCSV  = "0," + gpuUUID + ",00000000:01:00.0,42,35,12,100,1000,1500,1500,5000\n";

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
    size_t                              m_raidQueries{ 0 };                   ///< Number of fake MD reads.
    MagAOX::sys::boundedCommand::result m_result;                             ///< Next fake command result.
    std::vector<telem_irq::groupT>      m_groups;                             ///< Next fake IRQ snapshot.
    sysMonitorDetail::raidSnapshotT     m_raidResult;                         ///< Next fake MD snapshot.

    using sysMonitor::acceptGPUResult;
    using sysMonitor::acceptRAIDResult;
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
    using sysMonitor::m_indiNewCallBacks;
    using sysMonitor::m_indiP_irqStatus;
    using sysMonitor::m_indiP_irqWarning;
    using sysMonitor::m_indiP_raidStatus;
    using sysMonitor::m_indiP_raidWarning;
    using sysMonitor::m_irqInterval;
    using sysMonitor::m_irqRules;
    using sysMonitor::m_irqs;
    using sysMonitor::m_lastGPUs;
    using sysMonitor::m_lastIRQOK;
    using sysMonitor::m_raidEnabled;
    using sysMonitor::m_raidExpected;
    using sysMonitor::m_raidInfoProperties;
    using sysMonitor::m_raidInterval;
    using sysMonitor::m_raidProperties;
    using sysMonitor::m_raids;
    using sysMonitor::pollDiagnostics;
    using sysMonitor::publishDiagnostics;
    using sysMonitor::recordGPUs;
    using sysMonitor::recordIRQs;
    using sysMonitor::recordRAID;
    using sysMonitor::startupDiagnostics;

    /// Prepare private logging and a serialization-only INDI driver.
    diagnosticHarness();

    /// Flush local logs and remove the private fixture directory.
    ~diagnosticHarness();

    /// Supply the fake GPU result without invoking a command.
    MagAOX::sys::boundedCommand::result queryGPUs() override;

    /// Supply fake IRQ results without reading proc/sys files.
    std::vector<telem_irq::groupT> queryIRQs( double time /**< [in] wall-clock check time */ ) override;

    /// Supply a fake MD snapshot without reading the host's arrays.
    sysMonitorDetail::raidSnapshotT queryRAID( double time /**< [in] wall-clock check time */ ) override;

    /// Return the local message capture driver.
    diagnosticDriver &driver();

    /// Load a tracked diagnostic configuration fragment.
    int loadExample( const std::string &file /**< [in] basename under the app's config directory */ );

    /// Flush private logs and return their text messages with priorities in record order.
    std::vector<std::pair<flatlogs::logPrioT, std::string>> textLogs();

    /// Flush and read private records in file order.
    std::vector<flatlogs::bufferPtrT> records( bool telemetry /**< [in] select telemetry instead of process logs */ );
};

diagnosticHarness::diagnosticHarness()
{
    char directory[] = "/tmp/sysmon-diagnostics-XXXXXX";
    REQUIRE( mkdtemp( directory ) != nullptr );
    m_root        = directory;
    m_configName  = "sysmon-test";
    m_raidEnabled = false;
    m_tel.logPath( m_root );
    m_tel.logName( "telemetry" );
    m_tel.logExt( "bintel" );
    m_tel.logLevel( logPrio::LOG_TELEM );
    m_log.logPath( m_root );
    m_log.logName( "logs" );
    m_log.logExt( "binlog" );
    m_log.requestRotation(); // The process logger is shared across fixture instances.
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

sysMonitorDetail::raidSnapshotT diagnosticHarness::queryRAID( double time )
{
    ++m_raidQueries;
    m_raidResult.m_checkTime = time;
    return m_raidResult;
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

std::vector<flatlogs::bufferPtrT> diagnosticHarness::records( bool telemetry )
{
    if( telemetry )
    {
        m_tel.logShutdown( true );
        m_tel.logThreadExec();
    }
    else
    {
        m_log.logShutdown( true );
        m_log.logThreadExec();
    }

    std::vector<std::filesystem::path> files;

    for( const auto &entry : std::filesystem::recursive_directory_iterator( m_root ) )
    {
        if( entry.path().extension() == ( telemetry ? ".bintel" : ".binlog" ) )
        {
            files.push_back( entry.path() );
        }
    }

    std::sort( files.begin(), files.end() );
    std::vector<flatlogs::bufferPtrT> result;

    for( const auto &path : files )
    {
        std::ifstream     input( path, std::ios::binary );
        std::vector<char> bytes( ( std::istreambuf_iterator<char>( input ) ), {} );
        size_t            offset = 0;

        while( offset < bytes.size() )
        {
            REQUIRE( bytes.size() - offset >= flatlogs::logHeader::minHeadSize );
            auto      *record = bytes.data() + offset;
            const auto size   = flatlogs::logHeader::totalSize( record );
            REQUIRE( size >= flatlogs::logHeader::minHeadSize );
            REQUIRE( size <= bytes.size() - offset );

            flatlogs::bufferPtrT buffer( static_cast<char *>( ::operator new( size ) ) );
            std::memcpy( buffer.get(), record, size );
            result.push_back( buffer );
            offset += size;
        }
    }

    return result;
}

std::vector<std::pair<flatlogs::logPrioT, std::string>> diagnosticHarness::textLogs()
{
    std::vector<std::pair<flatlogs::logPrioT, std::string>> result;

    for( const auto &record : records( false ) )
    {
        if( flatlogs::logHeader::eventCode( record.get() ) == text_log::eventCode )
        {
            result.emplace_back( flatlogs::logHeader::logLevel( record.get() ),
                                 text_log::message( flatlogs::logHeader::messageBuffer( record.get() ) ) );
        }
    }

    return result;
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

/// Parse a complete synthetic MD inventory for lifecycle tests.
sysMonitorDetail::raidSnapshotT raidSnapshot( const std::string &body /**< [in] array blocks, or empty */,
                                              double             time = 100 /**< [in] actual snapshot time */ )
{
    auto result =
        sysMonitorDetail::parseMdstat( "Personalities : [raid1]\n" + body + "unused devices: <none>\n", time );
    REQUIRE( result.m_result == telem_raid::Ready );
    return result;
}

/// Return a two-member RAID1 block with either full or incomplete redundancy.
std::string raidBlock( const std::string &name /**< [in] kernel array name */,
                       bool               degraded = false /**< [in] whether the second member is unavailable */ )
{
    return name + " : active raid1 sda1[0] sdb1[1]" + ( degraded ? "(F)" : "" ) + "\n 1000 blocks super 1.2 " +
           ( degraded ? "[2/1] [U_]\n" : "[2/2] [UU]\n" );
}

/// Provide healthy arrays with verified synthetic identities, independent of kernel-name ordering.
sysMonitorDetail::raidSnapshotT
identifiedRAIDs( const std::vector<std::pair<std::string, std::string>> &identities /**< [in] kernel names and UUIDs */,
                 double                                                  time /**< [in] actual collection time */ )
{
    std::string body;

    for( const auto &identity : identities )
    {
        body += raidBlock( identity.first );
    }

    auto result = raidSnapshot( body, time );

    for( auto &array : result.m_arrays )
    {
        const auto match = std::find_if( identities.begin(),
                                         identities.end(),
                                         [&]( const auto &value )
                                         {
                                             return value.first == array.m_name;
                                         } );
        array.m_uuid     = match->second;
    }

    return result;
}

/// Native MD UUID fixtures in the kernel's byte order, never obtained from real devices.
const std::string raidUUIDA = "01234567-89ab-cdef-0123-456789abcdef";
const std::string raidUUIDB = "fedcba98-7654-3210-fedc-ba9876543210";
const std::string raidUUIDC = "11111111-2222-3333-4444-555555555555";
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
    REQUIRE( app.m_raidQueries == 0 );

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
    REQUIRE( app.m_gpuProperties.empty() );
    app.acceptGPUResult( app.m_result, 100 );
    REQUIRE( app.publishDiagnostics() == 0 );
    const std::string key = "gpu_0";
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

/// Verify NVIDIA indices control property names through reordering, disappearance, and replacement.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor GPU properties follow NVIDIA indices", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptGPUResult(*(MagAOX::sys::boundedCommand::result *)nullptr, 0);
    sysMonitor::publishDiagnostics();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_gpuEnabled = true;
    REQUIRE( app.startupDiagnostics() == 0 );
    const std::string secondUUID = "GPU-abcdefab-1234-1234-1234-123456789abc";
    const std::string secondCSV  = secondUUID + ",00000000:02:00.0,60,70,15,200,2000,1600,1600,6000\n";
    app.m_result.m_output        = "2," + secondCSV + gpuCSV;
    app.acceptGPUResult( app.m_result, 100 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpuProperties.size() == 2 );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_0" )["uuid"].get<std::string>() == gpuUUID );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_2" )["uuid"].get<std::string>() == secondUUID );
    REQUIRE( app.m_gpuProperties.at( "gpu_2" )["temperature"].get<double>() == 60 );

    app.m_result.m_output = gpuCSV + "2," + secondCSV;
    app.acceptGPUResult( app.m_result, 100 );
    app.driver().m_xml.clear();
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.driver().m_xml.empty() );

    app.m_result.m_output = gpuCSV;
    app.acceptGPUResult( app.m_result, 200 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpuProperties.at( "gpu_2" )["valid"].get<unsigned>() == 0 );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_2" )["status"].get<std::string>() == "missing" );

    app.m_result.m_output = "0," + secondCSV;
    app.acceptGPUResult( app.m_result, 300 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpus.size() == 2 );
    REQUIRE( app.m_gpus[0].m_uuid == gpuUUID );
    REQUIRE( app.m_gpus[0].m_index == telem_gpu::UnknownIndex );
    REQUIRE( app.m_gpus[0].m_valid == 0 );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_0" )["uuid"].get<std::string>() == secondUUID );
    REQUIRE( app.m_gpuProperties.at( "gpu_0" )["temperature"].get<double>() == 60 );
    REQUIRE( app.m_gpuProperties.at( "gpu_0" )["valid"].get<unsigned>() == 255 );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_2" )["status"].get<std::string>() == "unavailable" );
    REQUIRE( app.m_gpuProperties.at( "gpu_2" )["valid"].get<unsigned>() == 0 );
    REQUIRE( app.m_gpuProperties.at( "gpu_2" ).getState() == INDI_ALERT );
    REQUIRE( app.m_gpuStatus == "partial" );
    REQUIRE( app.recordGPUs() == 0 );

    app.m_result.m_output = "1," + gpuCSV.substr( 2 ) + "0," + secondCSV;
    app.acceptGPUResult( app.m_result, 400 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_gpuInfoProperties.at( "gpu_1" )["uuid"].get<std::string>() == gpuUUID );
    REQUIRE( app.m_gpuProperties.at( "gpu_1" )["valid"].get<unsigned>() == 255 );
    REQUIRE( app.m_gpuStatus == "ok" );
    REQUIRE( app.recordGPUs() == 0 );
    const auto records = app.records( true );
    REQUIRE( records.size() == 2 );
    const auto replaced = telem_gpu::gpus( flatlogs::logHeader::messageBuffer( records[0].get() ) );
    REQUIRE( replaced.size() == 1 );
    REQUIRE( replaced[0].m_index == 0 );
    REQUIRE( replaced[0].m_temperature == 60 );
    REQUIRE( replaced[0].m_uuid.empty() );
    const auto recovered = telem_gpu::gpus( flatlogs::logHeader::messageBuffer( records[1].get() ) );
    REQUIRE( recovered.size() == 2 );
    REQUIRE( recovered[0].m_index == 1 );
    REQUIRE( recovered[1].m_index == 0 );

    std::vector<std::string> mappings;

    for( const auto &[priority, message] : app.textLogs() )
    {
        if( message.find( "GPU mapping: " ) == 0 )
        {
            REQUIRE( priority == logPrio::LOG_NOTICE );
            mappings.push_back( message );
        }
    }

    REQUIRE( mappings == std::vector<std::string>{ "GPU mapping: index=0 uuid=" + gpuUUID + " pci=00000000:01:00.0",
                                                   "GPU mapping: index=2 uuid=" + secondUUID + " pci=00000000:02:00.0",
                                                   "GPU mapping: index=0 uuid=" + secondUUID + " pci=00000000:02:00.0",
                                                   "GPU mapping: index=1 uuid=" + gpuUUID + " pci=00000000:01:00.0" } );
}

/// Verify IRQ detail is absent from INDI and logged only on changed fault snapshots.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor logs IRQ details only for changed problems", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupDiagnostics();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_irqRules.push_back( { "camera", "edt", "camera", "", true, true } );
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.m_indiNewCallBacks.count( "sysmon-test.irq_status" ) == 1 );
    REQUIRE( app.m_indiNewCallBacks.count( "sysmon-test.irq_details" ) == 0 );
    const auto epoch = std::chrono::steady_clock::time_point{};
    app.m_groups     = { irqGroup( false ) };
    REQUIRE( app.pollDiagnostics( epoch ) == 0 );
    app.m_groups = { irqGroup( true ) };
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 6 ) ) == 0 );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 12 ) ) == 0 );
    app.m_groups[0].m_irqs[0].m_requested = "0-31";
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 18 ) ) == 0 );
    app.m_groups = { irqGroup( false ) };
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 24 ) ) == 0 );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 30 ) ) == 0 );
    app.m_groups[0].m_expected = "46";
    app.m_groups[0].m_irqs     = { { 125, "46", "46" } };
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 36 ) ) == 0 );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 42 ) ) == 0 );
    const auto logs = app.textLogs();
    REQUIRE( logs.size() == 7 );
    REQUIRE( logs[0].first == logPrio::LOG_NOTICE );
    REQUIRE( logs[0].second == "IRQ policy 0: camera match=edt cpuset=camera required=1 requireEffective=1" );
    REQUIRE( logs[1].second == "IRQ expectation camera: expected=45" );
    REQUIRE( logs[2].second == "IRQ camera: ok" );
    REQUIRE( logs[3].first == logPrio::LOG_WARNING );
    REQUIRE( logs[3].second ==
             "IRQ camera: requested affinity mismatch; expected=45; IRQ 123 requested=0-63 effective=45" );
    REQUIRE( logs[4].first == logPrio::LOG_WARNING );
    REQUIRE( logs[4].second ==
             "IRQ camera: requested affinity mismatch; expected=45; IRQ 123 requested=0-31 effective=45" );
    REQUIRE( logs[5].first == logPrio::LOG_NOTICE );
    REQUIRE( logs[5].second == "IRQ camera: ok" );
    REQUIRE( logs[6].second == "IRQ expectation camera: expected=46" );
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
    REQUIRE( app.m_lastIRQOK == std::vector<uint8_t>{ 1 } );
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
    const auto records = app.records( true );
    REQUIRE( records.size() == 4 );

    for( size_t n = 0; n < records.size(); n += 2 )
    {
        const auto gpu = telem_gpu::gpus( flatlogs::logHeader::messageBuffer( records[n].get() ) );
        REQUIRE( gpu.size() == 1 );
        REQUIRE( gpu[0].m_index == 0 );
        REQUIRE( gpu[0].m_uuid.empty() );
        REQUIRE( gpu[0].m_pci.empty() );
        REQUIRE( gpu[0].m_sampleTime == ( n == 0 ? 100 : 200 ) );
        auto *irq = flatlogs::logHeader::messageBuffer( records[n + 1].get() );
        REQUIRE( telem_irq::ok( irq ) == std::vector<unsigned>{ 1 } );
        REQUIRE( telem_irq::groups( irq ).empty() );
        REQUIRE( telem_irq::checkTime( irq ) == ( n == 0 ? 100 : 200 ) );
    }
}

/// Verify compact flags use the logged group order and ignore placement-only changes.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor IRQ telemetry records only policy OK changes", "[sysMonitor][diagnostics]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupDiagnostics();
    sysMonitor::recordIRQs(false);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_irqRules = { { "network", "nic", "", "46", false, false }, { "camera", "edt", "camera", "", true, true } };
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( app.m_lastIRQOK == std::vector<uint8_t>{ 0, 0 } );
    auto camera        = irqGroup( false );
    camera.m_checkTime = 100;
    telem_irq::groupT network{ "network", "46", "optional device absent", false, 100, {} };
    app.m_irqs = { network, camera };
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( app.m_lastIRQOK == std::vector<uint8_t>{ 1, 1 } );
    app.m_irqs[1].m_irqs[0].m_number = 456;
    app.m_irqs[1].m_expected         = "47";
    app.m_irqs[1].m_status           = "requested affinity verified; effective affinity unverified";
    telem_irq::lastRecord            = {};
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( telem_irq::lastRecord.tv_sec == 0 );
    app.m_irqs[1].m_warning = true;
    app.m_irqs[1].m_status  = "requested affinity mismatch";
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( app.m_lastIRQOK == std::vector<uint8_t>{ 0, 1 } );
    app.m_irqs[1].m_irqs[0].m_requested = "0-31";
    app.m_irqs[0].m_checkTime           = 200;
    app.m_irqs[1].m_checkTime           = 200;
    telem_irq::lastRecord               = {};
    REQUIRE( app.recordIRQs() == 0 );
    REQUIRE( telem_irq::lastRecord.tv_sec == 0 );
    REQUIRE( app.recordIRQs( true ) == 0 );
    const auto records = app.records( true );
    REQUIRE( records.size() == 4 );
    const std::vector<std::vector<unsigned>> flags{ { 0, 0 }, { 1, 1 }, { 0, 1 }, { 0, 1 } };
    const std::vector<double>                times{ 0, 100, 100, 200 };

    for( size_t n = 0; n < records.size(); ++n )
    {
        auto *payload = flatlogs::logHeader::messageBuffer( records[n].get() );
        REQUIRE( telem_irq::ok( payload ) == flags[n] );
        REQUIRE( telem_irq::checkTime( payload ) == times[n] );
        REQUIRE( telem_irq::groups( payload ).empty() );
    }

    const auto logs = app.textLogs();
    REQUIRE( logs.size() == 2 );
    REQUIRE( logs[0].second == "IRQ policy 0: camera match=edt cpuset=camera required=1 requireEffective=1" );
    REQUIRE( logs[1].second == "IRQ policy 1: network match=nic cpus=46 required=0 requireEffective=0" );
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
    REQUIRE( app.m_raidEnabled );
    REQUIRE( app.m_raidInterval == 5 );
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
    const auto interval =
        GENERATE( 0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() );
    const bool raid = GENERATE( false, true );

    if( raid )
    {
        app.m_raidInterval = interval;
    }
    else
    {
        app.m_gpuInterval = interval;
    }

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
    REQUIRE( rules.rules.size() == 5 );

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
    const auto setting =
        GENERATE( "gpu.enabled", "raid.enabled", "irq.camera.required", "irq.camera.requireEffective" );
    const std::string key  = setting;
    const auto        dot  = key.rfind( '.' );
    const std::string path = app.m_root + "/invalid.conf";
    std::ofstream     file( path );
    file << "[irq.camera]\nmatch=edt\ncpuset=camera\n";
    file << '[' << key.substr( 0, dot ) << "]\n" << key.substr( dot + 1 ) << "=garbage\n";
    file.close();
    REQUIRE( app.config.readConfig( path ) == 0 );
    REQUIRE( app.loadDiagnosticsConfig( app.config ) < 0 );
}

/// Verify RAID discovery, warning recovery, and independent cadence through the real INDI publisher.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor discovers RAID degradation and lost arrays", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupRAID();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::publishRAID();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled     = true;
    app.m_gpuEnabled      = true;
    app.m_result.m_status = MagAOX::sys::boundedCommand::status::timeout;
    app.m_raidResult      = raidSnapshot( raidBlock( "md0" ) );
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.m_indiP_raidStatus["summary"].get<std::string>() == "unchecked" );
    const auto epoch = std::chrono::steady_clock::time_point{};
    REQUIRE( app.pollDiagnostics( epoch ) == 0 );
    REQUIRE( app.m_raidQueries == 1 );
    REQUIRE( app.m_gpuQueries == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" ).getPerm() == pcf::IndiProperty::ReadOnly );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get<std::string>() == "md0" );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["active"].get<unsigned>() == 2 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 4 ) ) == 0 );
    REQUIRE( app.m_raidQueries == 1 );
    app.driver().m_xml.clear();
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.driver().m_xml.empty() );

    app.m_raidResult = raidSnapshot( raidBlock( "md0", true ) );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 6 ) ) == 0 );
    REQUIRE( app.m_raidQueries == 2 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["degraded"].get<unsigned>() == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["valid"].get<unsigned>() == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" ).getState() == INDI_ALERT );
    app.m_raidResult = raidSnapshot( raidBlock( "md0" ) );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 12 ) ) == 0 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );

    app.m_raidResult = raidSnapshot( raidBlock( "md1" ) + raidBlock( "md0" ) );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 18 ) ) == 0 );
    REQUIRE( app.m_raids.size() == 2 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_1" )["name"].get<std::string>() == "md1" );
    app.m_raidResult = raidSnapshot( raidBlock( "md1" ) );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 24 ) ) == 0 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["status"].get<std::string>() == "missing" );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["valid"].get<unsigned>() == 0 );
    REQUIRE( app.m_raidProperties.at( "raid_1" )["ok"].get<unsigned>() == 1 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );

    app.m_raidResult = sysMonitorDetail::parseMdstat( "Personalities : [raid1]\n" + raidBlock( "md1" ), 200 );
    REQUIRE( app.m_raidResult.m_result == telem_raid::ParseError );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 30 ) ) == 0 );
    REQUIRE( app.m_raids.size() == 2 );
    REQUIRE( app.m_raidProperties.at( "raid_1" )["valid"].get<unsigned>() == 0 );
    app.m_raidResult = raidSnapshot( raidBlock( "md0" ) + raidBlock( "md1" ) );
    REQUIRE( app.pollDiagnostics( epoch + std::chrono::seconds( 36 ) ) == 0 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
}

/// Verify absence before discovery is benign while failed reads and loss of an observed array warn.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor RAID warnings distinguish no arrays from unknown health", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::raidWarning();
    sysMonitor::publishRAID();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptRAIDResult( raidSnapshot( "" ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_indiP_raidStatus["summary"].get() == "no arrays" );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    sysMonitorDetail::raidSnapshotT missing;
    missing.m_result    = telem_raid::NotPresent;
    missing.m_status    = "MD not present";
    missing.m_checkTime = 101;
    app.acceptRAIDResult( missing );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    auto error     = missing;
    error.m_result = telem_raid::ReadError;
    error.m_status = "read failed";
    app.acceptRAIDResult( error );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0" ) ) );
    app.acceptRAIDResult( missing );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );

    const auto               directory = std::filesystem::read_symlink( "/proc/self/exe" ).parent_path().parent_path();
    mx::app::appConfigurator cfg;
    cfg.readConfig( ( directory / "config/sysmon-warning.conf" ).string() );
    indiRuleMaps                        rules;
    std::map<std::string, ruleRuleKeys> keys;
    loadRuleConfig( rules, keys, cfg );
    REQUIRE( rules.rules.size() == 5 );

    for( const auto &device : { "sysMonRTC", "sysMonICC", "sysMonAOC" } )
    {
        auto &property = *rules.props.at( std::string( device ) + ".raid_warning" );
        property       = app.m_indiP_raidWarning;
        property.setDevice( device );
        auto *rule = rules.rules.at( std::string( device ) + "-raid-warning" );
        REQUIRE( rule->priority() == rulePriority::warning );
        REQUIRE( rule->value() );
    }

    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0" ) ) );
    REQUIRE( app.publishDiagnostics() == 0 );

    for( const auto &device : { "sysMonRTC", "sysMonICC", "sysMonAOC" } )
    {
        auto &property = *rules.props.at( std::string( device ) + ".raid_warning" );
        property       = app.m_indiP_raidWarning;
        property.setDevice( device );
        REQUIRE_FALSE( rules.rules.at( std::string( device ) + "-raid-warning" )->value() );
    }
}

/// Verify compact RAID records, mapping logs, actual times, and suppression of unchanged fault details.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor records compact RAID state and changed fault logs", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::recordRAID(false);
    sysMonitor::recordTelem((telem_raid *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0" ), 100 ) );
    REQUIRE( app.recordRAID() == 0 );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0" ), 200 ) );
    telem_raid::lastRecord = {};
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( telem_raid::lastRecord.tv_sec == 0 );
    REQUIRE( app.recordTelem( static_cast<const telem_raid *>( nullptr ) ) == 0 );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0", true ), 300 ) );
    REQUIRE( app.recordRAID() == 0 );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0", true ), 400 ) );
    telem_raid::lastRecord = {};
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( telem_raid::lastRecord.tv_sec == 0 );
    app.acceptRAIDResult( raidSnapshot( raidBlock( "md0" ), 500 ) );
    REQUIRE( app.recordRAID() == 0 );
    const auto records = app.records( true );
    REQUIRE( records.size() == 4 );
    const std::vector<double> times{ 100, 200, 300, 500 };

    for( size_t n = 0; n < records.size(); ++n )
    {
        auto      *payload = flatlogs::logHeader::messageBuffer( records[n].get() );
        const auto arrays  = telem_raid::arrays( payload );
        REQUIRE( arrays.size() == 1 );
        REQUIRE( arrays[0].m_index == 0 );
        REQUIRE( arrays[0].m_expected == 2 );
        REQUIRE( arrays[0].m_active == ( n == 2 ? 1 : 2 ) );
        REQUIRE( telem_raid::checkTime( payload ) == times[n] );
        REQUIRE( telem_raid::status( payload ) == telem_raid::Ready );
    }

    size_t mappings = 0, faults = 0, healthy = 0;

    for( const auto &[priority, message] : app.textLogs() )
    {
        if( message.find( "RAID mapping: " ) == 0 )
        {
            REQUIRE( message == "RAID mapping: index=0 name=md0 level=raid1" );
            ++mappings;
        }
        else if( message.find( "RAID md0: " ) == 0 )
        {
            if( priority == logPrio::LOG_WARNING )
            {
                REQUIRE( message.find( "active=1/2" ) != std::string::npos );
                REQUIRE( message.find( "sdb1[1](F)" ) != std::string::npos );
                ++faults;
            }
            else
            {
                REQUIRE( message.find( "members=" ) == std::string::npos );
                ++healthy;
            }
        }
    }

    REQUIRE( mappings == 1 );
    REQUIRE( faults == 1 );
    REQUIRE( healthy == 2 );
}

/// Record changing fault details only in logs while ignoring maintenance progress percentages.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor RAID maintenance progress does not repeat fault telemetry", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::recordRAID(false);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    REQUIRE( app.startupDiagnostics() == 0 );
    const auto block = raidBlock( "md0", true );
    app.acceptRAIDResult( raidSnapshot( block + "  [=>...................] recovery = 12.3%\n", 100 ) );
    REQUIRE( app.recordRAID() == 0 );
    app.acceptRAIDResult( raidSnapshot( block + "  [=========>...........] recovery = 45.6%\n", 200 ) );
    REQUIRE( app.recordRAID() == 0 );
    auto changed = block;
    changed.replace( changed.find( "sdb1" ), 4, "sdc1" );
    app.acceptRAIDResult( raidSnapshot( changed + "  [=========>...........] recovery = 45.6%\n", 300 ) );
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( app.records( true ).size() == 1 );
    std::vector<std::string> faults;

    for( const auto &[priority, message] : app.textLogs() )
    {
        if( priority == logPrio::LOG_WARNING && message.find( "RAID md0: " ) == 0 )
        {
            REQUIRE( message.find( '%' ) == std::string::npos );
            faults.push_back( message );
        }
    }

    REQUIRE( faults.size() == 2 );
    REQUIRE( faults[0].find( "sdb1[1](F)" ) != std::string::npos );
    REQUIRE( faults[1].find( "sdc1[1](F)" ) != std::string::npos );
}

/// Load named expectations, normalizing kernel UUIDs and rejecting ambiguous or incomplete configuration.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor validates named expected RAID UUIDs", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::loadDiagnosticsConfig(*(mx::app::appConfigurator *)nullptr);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.setupConfig();
    const std::string body = GENERATE(
        "[raid.data]\nuuid=0123456789ABCDEF0123456789ABCDEF\n[raid.root]\nuuid=fedcba98-7654-3210-fedc-ba9876543210\n",
        "[raid.data]\nuuid=\n",
        "[raid.data]\nlabel=data\n",
        "[raid.data]\nuuid=00000000-0000-0000-0000-000000000000\n",
        "[raid.data]\nuuid=01234567:89abcdef:01234567:89abcdef\n",
        "[raid.]\nuuid=0123456789abcdef0123456789abcdef\n",
        "[raid.bad.label]\nuuid=0123456789abcdef0123456789abcdef\n",
        "[raid.data]\nuuid=0123456789abcdef0123456789abcdef\n[raid.root]\nuuid=01234567-89AB-CDEF-0123-"
        "456789ABCDEF\n" );
    std::ofstream file( app.m_root + "/raid.conf" );
    file << "[raid]\nenabled=true\n" << body;
    file.close();
    REQUIRE( app.config.readConfig( app.m_root + "/raid.conf" ) == 0 );

    if( body.find( raidUUIDB ) != std::string::npos )
    {
        REQUIRE( app.loadDiagnosticsConfig( app.config ) == 0 );
        REQUIRE( app.m_raidExpected.size() == 2 );
        REQUIRE( app.m_raidExpected.at( "data" ) == raidUUIDA );
        REQUIRE( app.m_raidExpected.at( "root" ) == raidUUIDB );
    }
    else
    {
        REQUIRE( app.loadDiagnosticsConfig( app.config ) < 0 );
    }
}

/// Required identities warn on the first empty or missing-interface result without waiting for the poll interval.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor reports expected RAID absence on the first poll", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupRAID();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    sysMonitor::publishRAID();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    app.m_raidExpected.emplace( "data", raidUUIDA );
    app.m_raidResult           = raidSnapshot( "" );
    const bool absentInterface = GENERATE( false, true );

    if( absentInterface )
    {
        app.m_raidResult.m_result = telem_raid::NotPresent;
        app.m_raidResult.m_status = "mdstat not present";
    }

    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_raids.size() == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" ).getState() == INDI_IDLE );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["configured"].get<unsigned>() == 1 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get().empty() );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["label"].get() == "data" );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["uuid"].get() == raidUUIDA );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.pollDiagnostics( std::chrono::steady_clock::time_point{} ) == 0 );
    REQUIRE( app.m_raidQueries == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["flags"].get<unsigned>() == telem_raid::Missing );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["status"].get() == "missing" );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );
    const auto records = app.records( true );
    REQUIRE( records.size() == 1 );
    const auto arrays = telem_raid::arrays( flatlogs::logHeader::messageBuffer( records[0].get() ) );
    REQUIRE( arrays.size() == 1 );
    REQUIRE( arrays[0].m_index == 0 );
    REQUIRE( arrays[0].m_flags == telem_raid::Missing );
    bool expectationLogged = false;

    for( const auto &[priority, message] : app.textLogs() )
    {
        if( message == "RAID expectation: index=0 label=data uuid=" + raidUUIDA )
        {
            REQUIRE( priority == logPrio::LOG_NOTICE );
            expectationLogged = true;
        }
    }

    REQUIRE( expectationLogged );
}

/// Disabling collection suppresses required-array warnings and leaves configured identities visibly disabled.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor can disable configured RAID expectations", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::startupRAID();
    sysMonitor::pollDiagnostics(std::chrono::steady_clock::time_point{});
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidExpected.emplace( "data", raidUUIDA );
    REQUIRE( app.startupDiagnostics() == 0 );
    REQUIRE( app.pollDiagnostics( std::chrono::steady_clock::time_point{} ) == 0 );
    REQUIRE( app.m_raidQueries == 0 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["status"].get() == "disabled" );
    REQUIRE( app.m_raidProperties.at( "raid_0" ).getState() == INDI_IDLE );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.records( true ).empty() );
}

/// UUIDs retain indices through name swaps while replacement arrays cannot satisfy an absent expectation.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor tracks configured and discovered RAIDs by UUID", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::publishRAID();
    sysMonitor::recordRAID(false);
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    app.m_raidExpected.emplace( "data", raidUUIDA );
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptRAIDResult( identifiedRAIDs( { { "md0", raidUUIDA }, { "md1", raidUUIDB } }, 100 ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( app.m_raids.size() == 2 );
    REQUIRE( app.m_raidProperties.at( "raid_1" )["configured"].get<unsigned>() == 0 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_1" )["uuid"].get() == raidUUIDB );
    app.acceptRAIDResult( identifiedRAIDs( { { "md0", raidUUIDB }, { "md1", raidUUIDA } }, 200 ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( app.m_raids.size() == 2 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get() == "md1" );
    REQUIRE( app.m_raidInfoProperties.at( "raid_1" )["name"].get() == "md0" );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    app.acceptRAIDResult( identifiedRAIDs( { { "md0", raidUUIDB }, { "md1", raidUUIDC } }, 300 ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( app.m_raids.size() == 3 );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["flags"].get<unsigned>() == telem_raid::Missing );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get().empty() );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["level"].get().empty() );
    REQUIRE( app.m_raidInfoProperties.at( "raid_2" )["uuid"].get() == raidUUIDC );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );
    app.acceptRAIDResult(
        identifiedRAIDs( { { "md0", raidUUIDB }, { "md1", raidUUIDC }, { "md127", raidUUIDA } }, 400 ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.recordRAID() == 0 );
    REQUIRE( app.m_raids.size() == 3 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get() == "md127" );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
    const auto records = app.records( true );
    REQUIRE( records.size() == 3 );
    size_t renamed = 0;

    for( const auto &[priority, message] : app.textLogs() )
    {
        static_cast<void>( priority );

        if( message == "RAID mapping: index=0 name=md1 level=raid1 uuid=" + raidUUIDA + " label=data" ||
            message == "RAID mapping: index=0 name=md127 level=raid1 uuid=" + raidUUIDA + " label=data" )
        {
            ++renamed;
        }
    }

    REQUIRE( renamed == 2 );
}

/// Incomplete identity collection invalidates known results and cannot match an expected array by kernel name.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "sysMonitor retains RAID expectations across identity errors", "[sysMonitor][diagnostics][raid]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    sysMonitor::acceptRAIDResult(*(sysMonitorDetail::raidSnapshotT *)nullptr);
    sysMonitor::publishRAID();
    #endif
    // clang-format on
    diagnosticHarness app;
    app.m_raidEnabled = true;
    app.m_raidExpected.emplace( "data", raidUUIDA );
    REQUIRE( app.startupDiagnostics() == 0 );
    app.acceptRAIDResult( identifiedRAIDs( { { "md0", raidUUIDA } }, 100 ) );
    auto failed     = identifiedRAIDs( { { "md0", raidUUIDC } }, 200 );
    failed.m_result = GENERATE( telem_raid::ReadError, telem_raid::ParseError );
    failed.m_status = "UUID unavailable or ambiguous";
    app.acceptRAIDResult( failed );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_raids.size() == 1 );
    REQUIRE( app.m_raidProperties.at( "raid_0" )["flags"].get<unsigned>() == 0 );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["uuid"].get() == raidUUIDA );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["status"].get() == "unavailable" );
    REQUIRE( app.m_raidInfoProperties.at( "raid_0" )["name"].get().empty() );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::On );
    app.acceptRAIDResult( identifiedRAIDs( { { "md127", raidUUIDA } }, 300 ) );
    REQUIRE( app.publishDiagnostics() == 0 );
    REQUIRE( app.m_raids.size() == 1 );
    REQUIRE( app.m_indiP_raidWarning["active"].getSwitchState() == pcf::IndiElement::Off );
}

} // namespace sysMonitorTest
} // namespace libXWCTest
