/** \file irqMonitor_test.cpp
 * \brief Offline IRQ discovery, affinity policy and recovery tests.
 * \ingroup sysMonitor_unit_test
 */
#include <cstdlib>

#include "../../../tests/testXWC.hpp"
#include "../irqMonitor.hpp"

namespace libXWCTest
{
namespace sysMonitorTest
{
namespace monitor = MagAOX::app::sysMonitorDetail;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Own a synthetic proc/cpuset tree so no test reads or changes real machine state.
class irqFixture
{
  public:
    std::filesystem::path m_root; ///< Unique temporary tree owned by this fixture.

    /// Create an isolated diagnostic-file tree.
    irqFixture();

    /// Remove only this fixture's temporary files.
    ~irqFixture();

    /// Write a synthetic file under the fixture root.
    void put( const std::string &path /**< [in] relative diagnostic path */,
              const std::string &text /**< [in] complete synthetic contents */ );

    /// Populate one IRQ's requested and optional effective affinity.
    void irq( unsigned           number /**< [in] synthetic IRQ number */,
              const std::string &requested /**< [in] requested CPU-list text */,
              const char        *effective /**< [in] effective text, or null to omit the interface */ );

    /// Run the real collector against this fixture only.
    std::vector<MagAOX::logger::telem_irq::groupT>
    check( const std::vector<monitor::ruleT> &rules /**< [in] logical groups to check */ );
};

irqFixture::irqFixture()
{
    char path[]  = "/tmp/magaox-irq-test-XXXXXX";
    auto created = mkdtemp( path );
    REQUIRE( created != nullptr );
    m_root = created;
    put( "proc/interrupts", " CPU0 CPU34\n 405: 0 0 IR-PCI-MSI 343-edge edt\n" );
    put( "cpuset/camwfs_irq/cpuset.cpus", "34\n" );
    irq( 405, "34\n", "34\n" );
}

irqFixture::~irqFixture()
{
    std::error_code error;
    std::filesystem::remove_all( m_root, error );
}

void irqFixture::put( const std::string &path, const std::string &text )
{
    auto target = m_root / path;
    std::filesystem::create_directories( target.parent_path() );
    std::ofstream file( target );
    file << text;
    REQUIRE( file.good() );
}

void irqFixture::irq( unsigned number, const std::string &requested, const char *effective )
{
    std::string path = "proc/irq/" + std::to_string( number ) + "/";
    put( path + "smp_affinity_list", requested );

    if( effective )
    {
        put( path + "effective_affinity_list", effective );
    }
    else
    {
        std::filesystem::remove( m_root / path / "effective_affinity_list" );
    }
}

std::vector<MagAOX::logger::telem_irq::groupT> irqFixture::check( const std::vector<monitor::ruleT> &rules )
{
    return monitor::checkIRQs( rules, ( m_root / "proc" ).string(), ( m_root / "cpuset" ).string(), 1234.5 );
}
/// \endcond

/// Rule validation rejects ambiguous CPU sources, invalid selectors and unsafe cpuset names.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "IRQ rules require explicit consistent configuration", "[sysMonitor][irqMonitor]" )
{
    std::string    error;
    monitor::ruleT rule{ "camera", "edt", "camwfs_irq", "" };
    REQUIRE( monitor::validateIRQRules( { rule }, error ) );
    REQUIRE( error.empty() );
    REQUIRE_FALSE( monitor::validateIRQRules( { rule, rule }, error ) );
    rule.m_cpus = "34";
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_cpuset.clear();
    REQUIRE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_cpus = "34junk";
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_cpus.clear();
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_cpuset = "../camwfs_irq";
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_cpuset = "camwfs_irq";
    rule.m_match.clear();
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_match = "edt";
    rule.m_name  = "summary";
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    rule.m_name = "bad.name";
    REQUIRE_FALSE( monitor::validateIRQRules( { rule }, error ) );
    REQUIRE( monitor::checkIRQs( {}, "/nonexistent-fixture", "/nonexistent-fixture", 1 ).empty() );
}

/// Only action names participate in discovery; counters, controller metadata and local vectors cannot match.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "IRQ discovery isolates action names", "[sysMonitor][irqMonitor]" )
{
    std::map<unsigned, std::string> actions;
    REQUIRE( monitor::irqDetail::parseInterrupts(
        actions,
        " CPU0 CPU34 CPU128\n"
        " 405: 0 0 99999999999999999999999 IR-PCI-MSI 300-edge edt, shared_handler\n"
        " 406: 0 0 0 edt 301-edge other\n"
        " 407: 0 0 0 PCI-MSI 302 Edge-fasteoi enx-test-TxRx-0\n"
        " NMI: 0 0 0 edt\n" ) );
    REQUIRE( actions.size() == 3 );
    REQUIRE( actions.at( 405 ) == "edt, shared_handler" );
    REQUIRE( actions.at( 406 ) == "other" );
    REQUIRE( actions.at( 407 ) == "enx-test-TxRx-0" );
    REQUIRE_FALSE( monitor::irqDetail::parseInterrupts( actions, "" ) );
    REQUIRE_FALSE( monitor::irqDetail::parseInterrupts( actions, " CPU0 CPU1\n405: 0 broken PCI-MSI 2-edge edt\n" ) );
    REQUIRE_FALSE( monitor::irqDetail::parseInterrupts(
        actions, " CPU0\n405: 0 PCI-MSI 1-edge edt\n405: 0 PCI-MSI 1-edge edt\n" ) );
}

/// Repeated checks rediscover every vector and reread cpuset policy without caching numeric IRQs.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "IRQ groups follow renumbering and cpuset updates", "[sysMonitor][irqMonitor]" )
{
    // clang-format off
#ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::checkIRQs( {}, "", "", 0 );
#endif
    // clang-format on
    irqFixture     fixture;
    monitor::ruleT rule{ "camera", "edt", "camwfs_irq", "" };
    auto           result = fixture.check( { rule } );
    REQUIRE_FALSE( result[0].m_warning );
    REQUIRE( result[0].m_status == "ok" );
    REQUIRE( result[0].m_expected == "34" );
    REQUIRE( result[0].m_irqs[0].m_number == 405 );
    REQUIRE( result[0].m_checkTime == 1234.5 );

    fixture.put( "proc/interrupts",
                 " CPU0 CPU34\n 705: 0 0 IR-PCI-MSI 343-edge edt\n 704: 0 0 IR-PCI-MSI 342-edge edt\n" );
    std::filesystem::remove_all( fixture.m_root / "proc/irq/405" );
    fixture.put( "cpuset/camwfs_irq/cpuset.cpus", "129-130,128\n" );
    fixture.irq( 704, "128-130", "129" );
    fixture.irq( 705, "130,128-129", "128,130" );
    result = fixture.check( { rule } );
    REQUIRE_FALSE( result[0].m_warning );
    REQUIRE( result[0].m_expected == "128-130" );
    REQUIRE( result[0].m_irqs.size() == 2 );
    REQUIRE( result[0].m_irqs[0].m_number == 704 );
    REQUIRE( result[0].m_irqs[1].m_number == 705 );

    fixture.irq( 705, "128-131", "128" );
    result = fixture.check( { rule } );
    REQUIRE( result[0].m_warning );
    REQUIRE( result[0].m_status.find( "IRQ705: requested affinity mismatch" ) != std::string::npos );
    fixture.irq( 705, "128-130", "127" );
    result = fixture.check( { rule } );
    REQUIRE( result[0].m_status.find( "effective affinity mismatch" ) != std::string::npos );
    fixture.irq( 705, "128-130", "130" );
    REQUIRE_FALSE( fixture.check( { rule } )[0].m_warning );
}

/// Missing devices and explicitly unsupported effective routing retain distinct operator-visible states.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "IRQ absence and effective support honor configured policy", "[sysMonitor][irqMonitor]" )
{
    // clang-format off
#ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::checkIRQs( {}, "", "", 0 );
#endif
    // clang-format on
    irqFixture     fixture;
    monitor::ruleT rule{ "camera", "edt", "camwfs_irq", "" };
    fixture.irq( 405, "34", nullptr );
    REQUIRE( fixture.check( { rule } )[0].m_warning );
    rule.m_requireEffective = false;
    auto result             = fixture.check( { rule } );
    REQUIRE_FALSE( result[0].m_warning );
    REQUIRE( result[0].m_status.find( "effective affinity unverified" ) != std::string::npos );
    fixture.irq( 405, "34", "" );
    REQUIRE( fixture.check( { rule } )[0].m_warning );

    fixture.put( "proc/interrupts", " CPU0\n NMI: 0 edt\n" );
    result = fixture.check( { rule } );
    REQUIRE( result[0].m_warning );
    REQUIRE( result[0].m_status == "required device missing" );
    rule.m_required = false;
    result          = fixture.check( { rule } );
    REQUIRE_FALSE( result[0].m_warning );
    REQUIRE( result[0].m_status == "optional device absent" );
    REQUIRE( result[0].m_irqs.empty() );
}

/// Incomplete reads and overlapping policies cannot produce a successful placement check.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "IRQ read failures and conflicting rules warn", "[sysMonitor][irqMonitor]" )
{
    // clang-format off
#ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::checkIRQs( {}, "", "", 0 );
#endif
    // clang-format on
    irqFixture     fixture;
    monitor::ruleT rule{ "camera", "edt", "camwfs_irq", "" };

    SECTION( "affinity disappears after interrupt discovery" )
    {
        std::filesystem::remove_all( fixture.m_root / "proc/irq/405" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
    }

    SECTION( "malformed requested and effective values" )
    {
        fixture.irq( 405, "34garbage", "34-" );
        auto result = fixture.check( { rule } );
        REQUIRE( result[0].m_warning );
        REQUIRE( result[0].m_status.find( "requested affinity unavailable" ) != std::string::npos );
        REQUIRE( result[0].m_status.find( "effective affinity unavailable" ) != std::string::npos );
    }

    SECTION( "missing cpuset" )
    {
        std::filesystem::remove( fixture.m_root / "cpuset/camwfs_irq/cpuset.cpus" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
    }

    SECTION( "effective interface is unreadable rather than absent" )
    {
        rule.m_requireEffective = false;
        std::filesystem::remove( fixture.m_root / "proc/irq/405/effective_affinity_list" );
        std::filesystem::create_directory( fixture.m_root / "proc/irq/405/effective_affinity_list" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
    }

    SECTION( "malformed cpuset" )
    {
        fixture.put( "cpuset/camwfs_irq/cpuset.cpus", "" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
    }

    SECTION( "interrupt table unreadable or malformed" )
    {
        std::filesystem::remove( fixture.m_root / "proc/interrupts" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
        fixture.put( "proc/interrupts", "garbage" );
        REQUIRE( fixture.check( { rule } )[0].m_warning );
    }

    SECTION( "overlapping action selectors" )
    {
        auto second    = rule;
        second.m_name  = "another";
        second.m_match = "ed";
        auto result    = fixture.check( { rule, second } );
        REQUIRE( result.size() == 2 );
        REQUIRE( result[0].m_name == "another" );
        REQUIRE( result[0].m_warning );
        REQUIRE( result[1].m_warning );
        REQUIRE( result[1].m_status.find( "overlapping rules" ) != std::string::npos );
    }
}

} // namespace sysMonitorTest
} // namespace libXWCTest
