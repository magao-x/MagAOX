/** \file sdNotify.hpp
 * \brief Notify systemd of service state, without depending on libsystemd
 *
 * \ingroup sys_files
 */

#ifndef sys_sdNotify_hpp
#define sys_sdNotify_hpp

namespace MagAOX
{
namespace sys
{

/// Send a state notification to systemd, as sd_notify(3) does
/** Writes \p state, e.g. "READY=1", "WATCHDOG=1" or "STOPPING=1", as one datagram to the socket named by the
 * NOTIFY_SOCKET environment variable, which systemd sets for services with Type=notify.  Does nothing when the
 * process was not started by systemd.  Never blocks.
 *
 * \returns 1 if the notification was sent
 * \returns 0 if NOTIFY_SOCKET is not set, i.e. not running under systemd
 * \returns -1 on error, and errno is set
 *
 * \ingroup sys
 */
int sdNotify( const char *state /**< [in] newline-separated VARIABLE=value assignments */ );

} // namespace sys
} // namespace MagAOX

#endif // sys_sdNotify_hpp
