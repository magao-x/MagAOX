/** \file raidMonitor_test.cpp
 * \brief Offline MD discovery, degradation policy and bounded-read tests.
 * \ingroup sysMonitor_unit_test
 */
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "../../../tests/testXWC.hpp"
#include "../raidMonitor.hpp"

namespace libXWCTest
{
namespace sysMonitorTest
{
namespace monitor = MagAOX::app::sysMonitorDetail;
using raidLog     = MagAOX::logger::telem_raid;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Add the kernel's complete discovery framing to synthetic array blocks.
std::string mdstat( const std::string &arrays /**< [in] array header and continuation lines */ )
{
    return "Personalities : [raid0] [raid1] [raid4] [raid5] [raid6] [raid10] [linear]\n" + arrays +
           "unused devices: <none>\n";
}

/// Own a temporary mdstat file, ensuring tests never inspect the running host's arrays.
class raidFixture
{
  public:
    std::filesystem::path m_root; ///< Temporary directory owned exclusively by this fixture.

    /// Create an isolated source directory.
    raidFixture();

    /// Remove this fixture's synthetic files.
    ~raidFixture();

    /// Replace the synthetic mdstat source.
    void put( const std::string &text /**< [in] complete synthetic source contents */ );

    /// Collect only the synthetic source.
    monitor::raidSnapshotT check( double time /**< [in] collection timestamp */ );
};

raidFixture::raidFixture()
{
    char path[]  = "/tmp/magaox-raid-test-XXXXXX";
    auto created = mkdtemp( path );
    REQUIRE( created != nullptr );
    m_root = created;
}

raidFixture::~raidFixture()
{
    std::error_code error;
    std::filesystem::remove_all( m_root, error );
}

void raidFixture::put( const std::string &text )
{
    std::ofstream file( m_root / "mdstat" );
    file << text;
    REQUIRE( file.good() );
}

monitor::raidSnapshotT raidFixture::check( double time )
{
    return monitor::checkRAID( ( m_root / "mdstat" ).string(), time );
}
/// \endcond

/// Discover every supported redundant personality and preserve counts independently of read-only mode.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID discovery identifies and sorts healthy redundant arrays", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    #endif
    // clang-format on
    const auto sample =
        monitor::parseMdstat( mdstat( "md9 : active raid6 sda1[0] sdb1[1] sdc1[2] sdd1[3]\n"
                                      "      999 blocks super 1.2 level 6, 512k chunk, algorithm 2 [4/4] [UUUU]\n"
                                      "md4 : active (read-only) raid4 sda2[0] sdb2[1] sdc2[2]\n"
                                      "      999 blocks super 1.2 level 4, 512k chunk, algorithm 0 [3/3] [UUU]\n"
                                      "md2 : active (auto-read-only) raid1 nvme0n1p2[0] nvme1n1p2[1](W)\n"
                                      "      999 blocks super 1.2 [2/2] [UU]\n"
                                      "md5 : active raid5 sda3[0] sdb3[1] sdc3[2] sdd3[4](J)\n"
                                      "      999 blocks level 5, 64k chunk, algorithm 2 [3/3] [UUU]\n"
                                      "md10 : active raid10 sda4[0] sdb4[1] sdc4[2] sdd4[3]\n"
                                      "      999 blocks super 1.2 512K chunks 2 near-copies [4/4] [UUUU]\n" ),
                              1234.5 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_checkTime == 1234.5 );
    REQUIRE( sample.m_arrays.size() == 5 );
    const std::vector<std::string> names{ "md10", "md2", "md4", "md5", "md9" };
    const std::vector<unsigned>    counts{ 4, 2, 3, 3, 4 };

    for( size_t n = 0; n < names.size(); ++n )
    {
        const auto &array = sample.m_arrays[n];
        REQUIRE( array.m_name == names[n] );
        REQUIRE( array.m_sample.m_index == raidLog::UnknownIndex );
        REQUIRE( array.m_sample.m_flags == ( raidLog::Valid | raidLog::OK | raidLog::Redundant ) );
        REQUIRE( array.m_sample.m_expected == counts[n] );
        REQUIRE( array.m_sample.m_active == counts[n] );
    }

    REQUIRE( sample.m_arrays[1].m_status == "ok (auto-read-only)" );
    REQUIRE( sample.m_arrays[2].m_status == "ok (read-only)" );
    REQUIRE( sample.m_arrays[1].m_members == "nvme0n1p2[0] nvme1n1p2[1](W) [UU]" );
}

/// A rebuilding array remains degraded, and a failed spare remains a fault even with full redundancy.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID faults include multiline members failed spares and rebuilding arrays", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    #endif
    // clang-format on
    auto sample = monitor::parseMdstat(
        mdstat( "md0 : active raid1 sda[0]\n"
                "      sdb[1](F) sdc[2](S)\n"
                "      1000 blocks super 1.2 [2/1] [U_]\n"
                "      [=>...................] recovery = 12.3% (123/1000) finish=1.0min speed=10K/sec\n" ),
        10 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays.size() == 1 );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags ==
             ( raidLog::Valid | raidLog::Redundant | raidLog::Degraded | raidLog::Faulty ) );
    REQUIRE( sample.m_arrays[0].m_sample.m_expected == 2 );
    REQUIRE( sample.m_arrays[0].m_sample.m_active == 1 );
    REQUIRE( sample.m_arrays[0].m_status == "degraded; faulty member; recovery" );
    REQUIRE( sample.m_arrays[0].m_members == "sda[0] sdb[1](F) sdc[2](S) [U_]" );

    sample = monitor::parseMdstat( mdstat( "md0 : active raid1 sda[0] sdb[1] sdc[2](F) sdd[3](R)\n"
                                           "      1000 blocks super 1.2 [2/2] [UU]\n" ),
                                   20 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::Redundant | raidLog::Faulty ) );
    REQUIRE( sample.m_arrays[0].m_sample.m_active == 2 );
    REQUIRE( sample.m_arrays[0].m_status == "faulty member" );

    sample =
        monitor::parseMdstat( mdstat( "md0 : active raid1 sda[0]\n      1000 blocks super 1.2 [2/1] [_U]\n" ), 30 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::Redundant | raidLog::Degraded ) );
}

/// Maintenance and bitmap page counters must not masquerade as degraded membership or noisy progress state.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID maintenance does not alarm when membership is intact", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    #endif
    // clang-format on
    const std::vector<std::string> actions{ "recovery", "resync", "reshape", "check", "repair" };

    for( const auto &action : actions )
    {
        INFO( action );
        auto sample = monitor::parseMdstat( mdstat( "md0 : active raid1 sda[0] sdb[1]\n"
                                                    "      1000 blocks super 1.2 [2/2] [UU]\n"
                                                    "      [====>................] " +
                                                    action +
                                                    " = 22.0% (220/1000) finish=1.0min speed=10K/sec\n"
                                                    "      bitmap: 0/1 pages [0KB], 65536KB chunk\n" ),
                                            1 );
        REQUIRE( sample.m_result == raidLog::Ready );
        REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::OK | raidLog::Redundant ) );
        REQUIRE( sample.m_arrays[0].m_status == "ok; " + action );
        REQUIRE( sample.m_arrays[0].m_members == "sda[0] sdb[1] [UU]" );
    }

    for( const auto &action :
         { "resync=DELAYED", "resync=PENDING", "resync=REMOTE", "recover=REMOTE", "reshape=REMOTE" } )
    {
        INFO( action );
        const auto sample = monitor::parseMdstat(
            mdstat( std::string( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\n      " ) + action +
                    "\n" ),
            2 );
        REQUIRE( sample.m_result == raidLog::Ready );
        REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::OK | raidLog::Redundant ) );
    }
}

/// Nonredundant devices and recognized containers are described honestly without requiring nonexistent bitmaps.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID distinguishes containers nonredundant arrays and inactive failures", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    #endif
    // clang-format on
    for( const auto &level : { "raid0", "linear" } )
    {
        const auto sample = monitor::parseMdstat( mdstat( std::string( "md0 : active " ) + level +
                                                          " sda[0] sdb[1]\n      1000 blocks super 1.2 512k chunks\n" ),
                                                  1 );
        REQUIRE( sample.m_result == raidLog::Ready );
        REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::OK ) );
        REQUIRE( sample.m_arrays[0].m_sample.m_expected == 0 );
        REQUIRE( sample.m_arrays[0].m_sample.m_active == 0 );
        REQUIRE( sample.m_arrays[0].m_status == "active; non-redundant" );
    }

    for( const auto &metadata : { "imsm", "ddf" } )
    {
        const auto sample = monitor::parseMdstat(
            mdstat( std::string( "md127 : inactive sda[0](S) sdb[1](S)\n      1000 blocks super external:" ) +
                    metadata + "\n" ),
            2 );
        REQUIRE( sample.m_result == raidLog::Ready );
        REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::OK | raidLog::Container ) );
        REQUIRE( sample.m_arrays[0].m_status == "metadata container" );
    }

    for( const auto &metadata : { "1.2", "external:/md127/0", "external:-md127/0", "external:unknown" } )
    {
        const auto sample = monitor::parseMdstat(
            mdstat( std::string( "md0 : inactive sda[0](S)\n      1000 blocks super " ) + metadata + "\n" ), 3 );
        REQUIRE( sample.m_result == raidLog::Ready );
        REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::Inactive ) );
        REQUIRE( sample.m_arrays[0].m_status == "inactive" );
    }

    const auto container = monitor::parseMdstat(
        mdstat( "md127 : inactive sda[0](S) sdb[1](F)\n      1000 blocks super external:imsm\n" ), 4 );
    REQUIRE( container.m_result == raidLog::Ready );
    REQUIRE( container.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::Container | raidLog::Faulty ) );

    auto sample = monitor::parseMdstat(
        mdstat( "md0 : broken raid0 sda[0] sdb[1](F)\n      1000 blocks super 1.2 512k chunks\n" ), 4 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags == ( raidLog::Valid | raidLog::Faulty ) );
    sample =
        monitor::parseMdstat( mdstat( "md0 : active multipath sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\n" ), 5 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags == 0 );
    REQUIRE( sample.m_arrays[0].m_status == "unsupported RAID personality: multipath" );
}

/// Inconsistent counts, partial snapshots and malformed membership cannot clear an earlier failure.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID parsing rejects incomplete and contradictory health evidence", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    #endif
    // clang-format on
    const std::vector<std::string> malformed{
        "",
        "Personalities : [raid1]\n",
        "unused devices: <none>\n",
        "Personalities : [raid1]\nunused devices:\n",
        "Personalities : [raid1]\nunused devices: sda <none>\n",
        "Personalities : [raid1]\nunused devices: <none>\ntrailing junk\n",
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks super 1.2\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [U_]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/1] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UUU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UX]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/3] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [0/0] []\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [65537/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [4294967296/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [+2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU] [2/2]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0]\n      1000 blocks [2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1](S)\n      1000 blocks [2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[broken]\n      1000 blocks [2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1](Z)\n      1000 blocks [2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sda[1]\n      1000 blocks [2/2] [UU]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks super [2/2] [UU]\n" ),
        mdstat( "md0 : active raid0 sda[0]\n      1000 blocks [2/1] [U_]\n" ),
        mdstat( "md0 : active linear sda[0]\n      1000 blocks [2/1] [U_]\n" ),
        mdstat( "md0 : inactive sda[0]\n      1000 blocks super external:imsm [2/1] [U_]\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\n"
                "md0 : inactive sdc[0](S)\n      1000 blocks super 1.2\n" ),
        mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\nmd1 : active raid1 sdc[0]\n" ),
        mdstat( "md0 : unknown raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\n" ) };

    for( const auto &text : malformed )
    {
        INFO( text );
        const auto sample = monitor::parseMdstat( text, 123 );
        REQUIRE( sample.m_result == raidLog::ParseError );
        REQUIRE( sample.m_checkTime == 123 );
    }

    auto text = mdstat( "" );
    text.insert( 10, 1, '\0' );
    REQUIRE( monitor::parseMdstat( text, 1 ).m_result == raidLog::ParseError );
    REQUIRE( monitor::parseMdstat( std::string( monitor::raidDetail::textLimit + 1, ' ' ), 1 ).m_result ==
             raidLog::ParseError );
}

/// An intact empty snapshot differs from a missing interface, and each read reflects the newest source contents.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "RAID collector distinguishes no arrays read errors and repeated changes", "[sysMonitor][raidMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseMdstat("", 0);
    MagAOX::app::sysMonitorDetail::checkRAID("", 0);
    #endif
    // clang-format on
    auto sample = monitor::parseMdstat( "Personalities :\nunused devices: <none>\n", 1 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays.empty() );
    REQUIRE( sample.m_status == "no arrays" );
    sample = monitor::parseMdstat( "Personalities : [raid1]\nunused devices: sda sdb\n", 2 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays.empty() );

    raidFixture fixture;
    sample = fixture.check( 3 );
    REQUIRE( sample.m_result == raidLog::NotPresent );
    REQUIRE( sample.m_checkTime == 3 );
    REQUIRE( sample.m_arrays.empty() );
    fixture.put( mdstat( "md0 : active raid1 sda[0]\n      1000 blocks [2/1] [U_]\n" ) );
    sample = fixture.check( 4 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags & raidLog::Degraded );
    fixture.put( mdstat( "md0 : active raid1 sda[0] sdb[1]\n      1000 blocks [2/2] [UU]\n" ) );
    sample = fixture.check( 5 );
    REQUIRE( sample.m_result == raidLog::Ready );
    REQUIRE( sample.m_arrays[0].m_sample.m_flags & raidLog::OK );
    REQUIRE( sample.m_checkTime == 5 );
    fixture.put( "Personalities : [raid1]\nmd0 : active raid1 sda[0]\n" );
    REQUIRE( fixture.check( 6 ).m_result == raidLog::ParseError );
    fixture.put( std::string( monitor::raidDetail::textLimit + 1, ' ' ) );
    REQUIRE( fixture.check( 7 ).m_result == raidLog::ReadError );
    REQUIRE( monitor::checkRAID( fixture.m_root.string(), 8 ).m_result == raidLog::ReadError );
}

} // namespace sysMonitorTest
} // namespace libXWCTest
