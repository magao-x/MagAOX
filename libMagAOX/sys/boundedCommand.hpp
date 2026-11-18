/** \file boundedCommand.hpp
 * \brief Execute an unprivileged command with bounded output and elapsed time.
 * \ingroup sys_files
 */

#ifndef sys_boundedCommand_hpp
#define sys_boundedCommand_hpp

#include <cstddef>
#include <string>
#include <vector>
#include <sys/types.h>

namespace MagAOX
{
namespace sys
{

/// Own one command and prevent relaunch while a timed-out child remains unreaped.
/** Commands execute directly, without a shell, in their own process group with real user/group IDs.
 * Calls on an instance must be serialized by its owner. The timeout bounds collection and normal cleanup;
 * an uninterruptible kernel operation can leave a killed child pending. Destruction makes a short bounded
 * reap attempt, then delegates a remaining child to a detached reaper. If that thread cannot be created,
 * final reaping is left to process exit rather than blocking destruction.
 * \ingroup sys
 */
class boundedCommand
{
  public:
    /// Classify command completion without conflating unavailable output with success.
    enum class status
    {
        success,
        spawnError,
        exitError,
        signalError,
        timeout,
        outputLimit,
        ioError,
        busy
    };

    /// Captured streams and the child outcome for one attempted command.
    struct result
    {
        status m_status{ status::spawnError }; ///< Completion category, including failures before spawning.

        std::string m_output; ///< Captured standard output, bounded together with standard error.

        std::string m_error; ///< Captured standard error, bounded together with standard output.

        int m_exitCode{ -1 }; ///< Exit status, terminating signal for signalError, or -1 when unavailable.
    };

    /// Construct an idle command owner.
    boundedCommand() = default;

    /// Kill and reap the owned command without indefinitely blocking destruction.
    ~boundedCommand() noexcept;

    /// Prevent two owners from reaping the same child.
    boundedCommand( const boundedCommand &other /**< [in] owner that cannot be copied */ ) = delete;

    /// Prevent copying ownership of a pending child.
    boundedCommand &operator=( const boundedCommand &other /**< [in] owner that cannot be copied */ ) = delete;

    /// Execute an absolute executable path and capture both streams until completion or a bound is reached.
    /** Invalid arguments return spawnError without launching a child. maxOutput bounds the combined captured
     * streams; exceeding it returns outputLimit. A pending previous child returns busy. The executable must
     * be an ordinary unprivileged program, not a set-ID executable. Spawn itself is subject to kernel latency.
     */
    result run( const std::vector<std::string> &argv, /**< [in] executable path followed by individual arguments */
                double                          timeoutSec, /**< [in] positive finite elapsed-time limit in seconds */
                size_t                          maxOutput = 65536 /**< [in] maximum combined captured output bytes */ );

    /// Try a nonblocking reap and report whether a previous child still prevents another launch.
    bool pending() noexcept;

    /// Kill the owned process group and attempt a nonblocking reap.
    void shutdown() noexcept;

  private:
    pid_t m_pid{ -1 }; ///< Owned child and process-group ID, retained until a specific-child reap succeeds.
};

} // namespace sys
} // namespace MagAOX

#endif // sys_boundedCommand_hpp
