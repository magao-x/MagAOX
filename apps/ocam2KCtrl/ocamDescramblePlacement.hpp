/** \file ocamDescramblePlacement.hpp
 * \brief Read-only scheduling and CPU-placement admission checks for parallel OCAM descrambling.
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamDescramblePlacement_hpp
#define ocamDescramblePlacement_hpp

#include <charconv>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <sched.h>

namespace MagAOX
{
namespace app
{

/// Verify two configured threads without changing their scheduling, affinity, or cpusets.
/** Check before admitting parallel work, while both threads are alive and the worker is parked. The operator must
 * reserve the worker's physical core and keep scheduling/cpuset configuration unchanged while it is active. These
 * checks establish observed placement, not exclusive ownership of either core or protection against later changes.
 * Definitions remain inline to match the application's header-only testable implementation.
 */
class ocamDescramblePlacement
{
  private:
    /// Read exactly one whitespace-delimited field, rejecting empty files and additional fields.
    static bool readField( const std::string &path /**< [in] File to read. */,
                           std::string       &value /**< [out] Sole field, or empty on failure. */ );

    /// Parse an unsigned decimal field into a nonnegative int without accepting signs or suffixes.
    static bool decimal( const std::string &text /**< [in] Complete decimal field. */,
                         int               &value /**< [out] Parsed value on success. */ );

    /// Match one configured single-CPU cpuset against the thread's observed affinity.
    static bool verifyAffinity( std::thread &thread /**< [in] Started thread kept alive throughout verification. */,
                                const std::string &cpusetPath /**< [in] Cpuset filesystem root. */,
                                const std::string &cpuset /**< [in] Nonempty configured cpuset name. */,
                                int         &cpu /**< [out] Sole observed and configured CPU, or -1 on failure. */,
                                std::string &reason /**< [out] Failure explanation. */ );

    /// Read the canonical kernel CPU-list field for the sole unified L3 cache.
    static bool readL3( const std::string &cpuPath /**< [in] CPU directory below the topology root. */,
                        std::string       &shared /**< [out] Canonical shared CPU list. */ );

    /// Require a valid ordered kernel CPU list containing the specified CPU.
    static bool containsCpu( const std::string &list /**< [in] Canonical comma-separated CPUs and ranges. */,
                             int                cpu /**< [in] CPU that must belong to the list. */ );

  public:
    /// Require exact observed policy and priority; zero with SCHED_OTHER supports unprivileged local tests.
    static bool
    verifyScheduling( std::thread &thread /**< [in] Started thread kept alive throughout verification. */,
                      int          expectedPriority /**< [in] Requested priority without clamping. */,
                      int          expectedPolicy /**< [in] Requested POSIX scheduling policy. */,
                      std::string &reason /**< [out] Empty on success, otherwise the failure explanation. */ );

    /// Require separate physical cores and the same canonical unified-L3 sharing list.
    /** Unknown, missing, or inconsistent topology fails admission. This does not infer CCD membership from CPU IDs. */
    static bool verifyTopology( int          primaryCpu /**< [in] Sole framegrabber CPU. */,
                                int          workerCpu /**< [in] Sole worker CPU. */,
                                std::string &reason /**< [out] Empty on success, otherwise the failure explanation. */,
                                const std::string &topologyRoot = "/sys/devices/system/cpu"
                                /**< [in] CPU topology root; injectable for offline tests. */ );

    /// Require explicit single-decimal cpusets, exact thread affinities, separate cores, and shared L3.
    /** Call again after reconfiguration before admitting work. Both CPU outputs remain -1 on any failure. */
    static bool verifyPlacement( std::thread &primary /**< [in] Started framegrabber kept alive during verification. */,
                                 std::thread &worker /**< [in] Started worker kept parked during verification. */,
                                 const std::string &cpusetPath /**< [in] Cpuset filesystem root. */,
                                 const std::string &primaryCpuset /**< [in] Explicit single-CPU framegrabber cpuset. */,
                                 const std::string &workerCpuset /**< [in] Explicit single-CPU worker cpuset. */,
                                 int         &primaryCpu /**< [out] Verified framegrabber CPU, or -1 on failure. */,
                                 int         &workerCpu /**< [out] Verified worker CPU, or -1 on failure. */,
                                 std::string &reason /**< [out] Empty on success, otherwise the failure explanation. */,
                                 const std::string &topologyRoot = "/sys/devices/system/cpu"
                                 /**< [in] CPU topology root; injectable for offline tests. */ );
};

inline bool ocamDescramblePlacement::readField( const std::string &path, std::string &value )
{
    std::ifstream file( path );
    std::string   extra;
    value.clear();

    if( !( file >> value ) || file >> extra || file.bad() )
    {
        value.clear();
        return false;
    }

    return true;
}

inline bool ocamDescramblePlacement::decimal( const std::string &text, int &value )
{
    if( text.empty() || text.find_first_not_of( "0123456789" ) != std::string::npos )
    {
        return false;
    }

    const auto parsed = std::from_chars( text.data(), text.data() + text.size(), value );
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

inline bool ocamDescramblePlacement::verifyScheduling( std::thread &thread,
                                                       int          expectedPriority,
                                                       int          expectedPolicy,
                                                       std::string &reason )
{
    reason.clear();
    sched_param actual{};
    int         policy = -1;

    if( !thread.joinable() || pthread_getschedparam( thread.native_handle(), &policy, &actual ) != 0 )
    {
        reason = "thread scheduling could not be read";
        return false;
    }

    if( policy != expectedPolicy || actual.sched_priority != expectedPriority )
    {
        reason = "thread scheduling does not match requested policy/priority";
        return false;
    }

    return true;
}

inline bool ocamDescramblePlacement::verifyAffinity(
    std::thread &thread, const std::string &cpusetPath, const std::string &cpuset, int &cpu, std::string &reason )
{
    cpu = -1;
    std::string field;
    int         expected = -1;

    if( cpuset.empty() || !readField( cpusetPath + '/' + cpuset + "/cpuset.cpus", field ) ||
        !decimal( field, expected ) || expected >= CPU_SETSIZE )
    {
        reason = "cpuset must name exactly one decimal CPU below CPU_SETSIZE";
        return false;
    }

    cpu_set_t actual;
    CPU_ZERO( &actual );

    if( !thread.joinable() || pthread_getaffinity_np( thread.native_handle(), sizeof( actual ), &actual ) != 0 ||
        CPU_COUNT( &actual ) != 1 || !CPU_ISSET( expected, &actual ) )
    {
        reason = "thread affinity does not match its configured single-CPU cpuset";
        return false;
    }

    cpu = expected;
    return true;
}

inline bool ocamDescramblePlacement::readL3( const std::string &cpuPath, std::string &shared )
{
    shared.clear();
    std::error_code                     error;
    std::filesystem::directory_iterator entry( cpuPath + "/cache", error ), end;

    while( !error && entry != end )
    {
        const std::string path = entry->path().string();
        std::string       level, type, list;

        if( entry->path().filename().string().rfind( "index", 0 ) == 0 && readField( path + "/level", level ) &&
            level == "3" && readField( path + "/type", type ) && type == "Unified" )
        {
            if( !shared.empty() || !readField( path + "/shared_cpu_list", list ) ||
                list.find_first_not_of( "0123456789,-" ) != std::string::npos )
            {
                return false;
            }

            shared = list;
        }

        entry.increment( error );
    }

    return !error && !shared.empty();
}

inline bool ocamDescramblePlacement::containsCpu( const std::string &list, int cpu )
{
    size_t begin    = 0;
    int    previous = -1;
    bool   found    = false;

    while( begin < list.size() )
    {
        const size_t      comma = list.find( ',', begin );
        const std::string range = list.substr( begin, comma == std::string::npos ? comma : comma - begin );
        const size_t      dash  = range.find( '-' );
        int               first = -1, last = -1;

        if( !decimal( range.substr( 0, dash ), first ) ||
            ( dash != std::string::npos && !decimal( range.substr( dash + 1 ), last ) ) )
        {
            return false;
        }

        last = dash == std::string::npos ? first : last;

        if( first <= previous || last < first || last >= CPU_SETSIZE )
        {
            return false;
        }

        found    = found || ( first <= cpu && cpu <= last );
        previous = last;

        if( comma == std::string::npos )
        {
            return found;
        }

        begin = comma + 1;
    }

    return false;
}

inline bool ocamDescramblePlacement::verifyTopology( int                primaryCpu,
                                                     int                workerCpu,
                                                     std::string       &reason,
                                                     const std::string &topologyRoot )
{
    reason.clear();

    if( primaryCpu < 0 || workerCpu < 0 || primaryCpu >= CPU_SETSIZE || workerCpu >= CPU_SETSIZE ||
        primaryCpu == workerCpu )
    {
        reason = "framegrabber and worker require distinct supported CPUs";
        return false;
    }

    int         package[2], core[2];
    std::string shared[2];
    const int   cpus[2] = { primaryCpu, workerCpu };

    for( int index = 0; index < 2; ++index )
    {
        const std::string path = topologyRoot + "/cpu" + std::to_string( cpus[index] );
        std::string       field;

        if( !readField( path + "/topology/physical_package_id", field ) || !decimal( field, package[index] ) ||
            !readField( path + "/topology/core_id", field ) || !decimal( field, core[index] ) ||
            !readL3( path, shared[index] ) )
        {
            reason = "CPU physical-core or unified-L3 topology could not be verified";
            return false;
        }
    }

    if( package[0] == package[1] && core[0] == core[1] )
    {
        reason = "framegrabber and worker are SMT siblings on the same physical core";
        return false;
    }

    if( package[0] != package[1] || shared[0] != shared[1] || !containsCpu( shared[0], primaryCpu ) ||
        !containsCpu( shared[0], workerCpu ) )
    {
        reason = "framegrabber and worker do not share the same unified L3 cache";
        return false;
    }

    return true;
}

inline bool ocamDescramblePlacement::verifyPlacement( std::thread       &primary,
                                                      std::thread       &worker,
                                                      const std::string &cpusetPath,
                                                      const std::string &primaryCpuset,
                                                      const std::string &workerCpuset,
                                                      int               &primaryCpu,
                                                      int               &workerCpu,
                                                      std::string       &reason,
                                                      const std::string &topologyRoot )
{
    primaryCpu = workerCpu = -1;
    reason.clear();
    int observedPrimary = -1, observedWorker = -1;

    if( !verifyAffinity( primary, cpusetPath, primaryCpuset, observedPrimary, reason ) )
    {
        reason = "framegrabber: " + reason;
        return false;
    }

    if( !verifyAffinity( worker, cpusetPath, workerCpuset, observedWorker, reason ) )
    {
        reason = "descramble worker: " + reason;
        return false;
    }

    if( !verifyTopology( observedPrimary, observedWorker, reason, topologyRoot ) )
    {
        return false;
    }

    primaryCpu = observedPrimary;
    workerCpu  = observedWorker;
    return true;
}

} // namespace app
} // namespace MagAOX

#endif // ocamDescramblePlacement_hpp
