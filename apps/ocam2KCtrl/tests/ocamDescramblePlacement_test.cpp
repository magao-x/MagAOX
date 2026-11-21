/** \file ocamDescramblePlacement_test.cpp
 * \brief Offline topology and local-thread admission checks for parallel OCAM descrambling.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../ocamDescramblePlacement.hpp"

#include <array>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <vector>

namespace libXWCTest
{
namespace ocam2KCtrlTest
{

using MagAOX::app::ocamDescramblePlacement;

/// \cond OCAM2KCTRL_PLACEMENT_TEST_SUPPORT

/// Own synthetic cpuset and sysfs files without changing any system configuration.
class placementTree
{
  public:
    std::string m_path; ///< Unique temporary root removed after each test section.

    /// Create an isolated temporary directory.
    placementTree();

    /// Remove only the fixture's temporary tree.
    ~placementTree();

    /// Write one synthetic field after creating its parent directories.
    void write( const std::string &path /**< [in] Relative field path. */,
                const std::string &value /**< [in] Complete synthetic field contents. */ );

    /// Describe one physical core and its unified L3 cache.
    void cpu( int                id /**< [in] Logical CPU number. */,
              int                core /**< [in] Physical core number within package zero. */,
              const std::string &shared /**< [in] Canonical shared CPU list. */ );
};

placementTree::placementTree()
{
    char        path[]  = "/tmp/ocam-placement-XXXXXX";
    const char *created = mkdtemp( path );

    if( !created )
    {
        throw std::runtime_error( "could not create placement fixture" );
    }

    m_path = created;
}

placementTree::~placementTree()
{
    std::error_code ignored;
    std::filesystem::remove_all( m_path, ignored );
}

void placementTree::write( const std::string &path, const std::string &value )
{
    const std::filesystem::path target = m_path + '/' + path;
    std::filesystem::create_directories( target.parent_path() );
    std::ofstream file( target );
    file << value;

    if( !file )
    {
        throw std::runtime_error( "could not write placement fixture" );
    }
}

void placementTree::cpu( int id, int core, const std::string &shared )
{
    const std::string path = "cpu" + std::to_string( id );
    write( path + "/topology/physical_package_id", "0\n" );
    write( path + "/topology/core_id", std::to_string( core ) + '\n' );
    write( path + "/cache/index3/level", "3\n" );
    write( path + "/cache/index3/type", "Unified\n" );
    write( path + "/cache/index3/shared_cpu_list", shared + '\n' );
}

/// Own an ordinary sleeping local thread whose lifetime spans readback checks.
class placementThread
{
    std::promise<void> m_stop;    ///< Destructor releases the thread before joining it.
    std::future<void>  m_release; ///< One-shot blocking wait; no busy loop or real-time scheduling.

  public:
    std::thread m_thread; ///< Test-owned native thread; only its local affinity is changed.

    /// Start a thread blocked on the destructor's release signal.
    placementThread();

    /// Release and join the local thread even when an assertion fails.
    ~placementThread();
};

placementThread::placementThread()
    : m_release( m_stop.get_future() ), m_thread(
                                            [this]
                                            {
                                                m_release.wait();
                                            } )
{
}

placementThread::~placementThread()
{
    m_stop.set_value();
    m_thread.join();
}

/// \endcond

/** \brief Admit distinct physical cores in one verified L3 domain and reject ambiguous or incomplete topology.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */
TEST_CASE( "OCAM parallel placement validates physical cores and L3", "[ocam2KCtrl][descramble][placement]" )
{
    placementTree tree;
    tree.cpu( 31, 7, "24-39,152-167" );
    tree.cpu( 33, 9, "24-39,152-167" );
    std::string reason = "previous failure";

    REQUIRE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    REQUIRE( reason.empty() );

    SECTION( "same CPU is rejected" )
    {
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 31, reason, tree.m_path ) );
    }

    SECTION( "SMT siblings are rejected" )
    {
        tree.write( "cpu33/topology/core_id", "7\n" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "different packages are rejected" )
    {
        tree.write( "cpu33/topology/physical_package_id", "1\n" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "different L3 domains are rejected" )
    {
        tree.write( "cpu33/cache/index3/shared_cpu_list", "32-39,160-167\n" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "unknown core IDs are rejected" )
    {
        tree.write( "cpu33/topology/core_id", "-1\n" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "missing L3 fields are rejected" )
    {
        std::filesystem::remove( tree.m_path + "/cpu31/cache/index3/shared_cpu_list" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "multiple L3 entries are rejected" )
    {
        tree.write( "cpu31/cache/index4/level", "3\n" );
        tree.write( "cpu31/cache/index4/type", "Unified\n" );
        tree.write( "cpu31/cache/index4/shared_cpu_list", "24-39,152-167\n" );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
    }

    SECTION( "invalid or inconsistent shared CPU lists are rejected" )
    {
        for( const char *list : { "", "-", "33-31", "31,,33", "31-33,", "31-33,33", "0-3", "31", "31x" } )
        {
            tree.write( "cpu31/cache/index3/shared_cpu_list", list );
            tree.write( "cpu33/cache/index3/shared_cpu_list", list );
            REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 33, reason, tree.m_path ) );
        }
    }

    SECTION( "invalid CPUs and absent topology are rejected" )
    {
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( -1, 33, reason, tree.m_path ) );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( CPU_SETSIZE, 33, reason, tree.m_path ) );
        REQUIRE_FALSE( ocamDescramblePlacement::verifyTopology( 31, 35, reason, tree.m_path ) );
    }

    REQUIRE_FALSE( reason.empty() );
}

/** \brief Read exact policy and priority from a live ordinary thread without requiring RT privileges.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */
TEST_CASE( "OCAM parallel scheduling requires exact readback", "[ocam2KCtrl][descramble][placement]" )
{
    placementThread live;
    std::thread     absent;
    std::string     reason = "previous failure";

    REQUIRE( ocamDescramblePlacement::verifyScheduling( live.m_thread, 0, SCHED_OTHER, reason ) );
    REQUIRE( reason.empty() );
    REQUIRE_FALSE( ocamDescramblePlacement::verifyScheduling( live.m_thread, 1, SCHED_OTHER, reason ) );
    REQUIRE_FALSE( reason.empty() );
    REQUIRE_FALSE( ocamDescramblePlacement::verifyScheduling( live.m_thread, 0, SCHED_FIFO, reason ) );
    REQUIRE_FALSE( ocamDescramblePlacement::verifyScheduling( absent, 0, SCHED_OTHER, reason ) );
}

/** \brief Require explicit single-CPU cpusets and exact local-thread affinities before returning CPU IDs.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */
TEST_CASE( "OCAM parallel placement matches configured and actual affinity", "[ocam2KCtrl][descramble][placement]" )
{
    cpu_set_t allowed;
    CPU_ZERO( &allowed );
    REQUIRE( pthread_getaffinity_np( pthread_self(), sizeof( allowed ), &allowed ) == 0 );
    std::vector<int> cpus;

    for( int cpu = 0; cpu < CPU_SETSIZE; ++cpu )
    {
        if( CPU_ISSET( cpu, &allowed ) )
        {
            cpus.push_back( cpu );
        }
    }

    REQUIRE( cpus.size() >= 2 );
    placementThread   primary, worker;
    placementTree     tree;
    const std::string list = std::to_string( cpus[0] ) + ',' + std::to_string( cpus[1] );
    tree.cpu( cpus[0], 0, list );
    tree.cpu( cpus[1], 1, list );
    tree.write( "primary/cpuset.cpus", std::to_string( cpus[0] ) + '\n' );
    tree.write( "worker/cpuset.cpus", std::to_string( cpus[1] ) + '\n' );
    std::array<std::thread *, 2> threads{ &primary.m_thread, &worker.m_thread };

    for( size_t index = 0; index < threads.size(); ++index )
    {
        cpu_set_t selected;
        CPU_ZERO( &selected );
        CPU_SET( cpus[index], &selected );
        REQUIRE( pthread_setaffinity_np( threads[index]->native_handle(), sizeof( selected ), &selected ) == 0 );
    }

    std::string reason          = "previous failure";
    int         observedPrimary = -1, observedWorker = -1;

    /// Verify both live local threads against this section's synthetic configuration.
    const auto verify = [&]( const std::string &primarySet = "primary" /**< [in] Framegrabber cpuset name. */,
                             const std::string &workerSet  = "worker" /**< [in] Worker cpuset name. */ )
    {
        return ocamDescramblePlacement::verifyPlacement( primary.m_thread,
                                                         worker.m_thread,
                                                         tree.m_path,
                                                         primarySet,
                                                         workerSet,
                                                         observedPrimary,
                                                         observedWorker,
                                                         reason,
                                                         tree.m_path );
    };

    REQUIRE( verify() );
    REQUIRE( reason.empty() );
    REQUIRE( observedPrimary == cpus[0] );
    REQUIRE( observedWorker == cpus[1] );

    SECTION( "both cpusets must be explicit" )
    {
        REQUIRE_FALSE( verify( "" ) );
        REQUIRE_FALSE( verify( "primary", "" ) );
    }

    SECTION( "cpuset fields must be single strict decimal CPU values" )
    {
        for( const char *field : { "", "1-1", "1,2", "-1", "+1", "1x", "1 2", "999999999999999999" } )
        {
            tree.write( "worker/cpuset.cpus", field );
            REQUIRE_FALSE( verify() );
            REQUIRE( observedPrimary == -1 );
            REQUIRE( observedWorker == -1 );
        }
    }

    SECTION( "one CPU different from the configured CPU is rejected" )
    {
        tree.write( "worker/cpuset.cpus", std::to_string( cpus[0] ) );
        REQUIRE_FALSE( verify() );
    }

    SECTION( "missing or unsupported cpuset CPU values are rejected" )
    {
        tree.write( "worker/cpuset.cpus", std::to_string( CPU_SETSIZE ) );
        REQUIRE_FALSE( verify() );
        std::filesystem::remove( tree.m_path + "/worker/cpuset.cpus" );
        REQUIRE_FALSE( verify() );
    }

    SECTION( "a wider actual affinity is rejected" )
    {
        REQUIRE( pthread_setaffinity_np( worker.m_thread.native_handle(), sizeof( allowed ), &allowed ) == 0 );
        REQUIRE_FALSE( verify() );
    }

    SECTION( "a topology failure clears both returned CPU IDs" )
    {
        tree.write( "cpu" + std::to_string( cpus[1] ) + "/topology/core_id", "0\n" );
        REQUIRE_FALSE( verify() );
    }

    REQUIRE( observedPrimary == -1 );
    REQUIRE( observedWorker == -1 );
    REQUIRE_FALSE( reason.empty() );
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
