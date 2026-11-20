/** \file sdNotify.cpp
 * \brief Notify systemd of service state, without depending on libsystemd
 *
 * \ingroup sys_files
 */

#include "sdNotify.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace MagAOX
{
namespace sys
{

int sdNotify( const char *state )
{
    const char *path = getenv( "NOTIFY_SOCKET" );
    if( path == nullptr || path[0] == '\0' )
    {
        return 0;
    }

    // Only filesystem and abstract (leading @) AF_UNIX sockets are used by systemd for services
    if( path[0] != '/' && path[0] != '@' )
    {
        errno = EAFNOSUPPORT;
        return -1;
    }

    struct sockaddr_un addr;
    memset( &addr, 0, sizeof( addr ) );
    addr.sun_family = AF_UNIX;

    size_t len = strlen( path );
    if( len >= sizeof( addr.sun_path ) )
    {
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy( addr.sun_path, path, len );
    if( addr.sun_path[0] == '@' )
    {
        addr.sun_path[0] = '\0';
    }

    int fd = socket( AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0 );
    if( fd < 0 )
    {
        return -1;
    }

    ssize_t nw = sendto( fd,
                         state,
                         strlen( state ),
                         MSG_NOSIGNAL | MSG_DONTWAIT,
                         reinterpret_cast<struct sockaddr *>( &addr ),
                         offsetof( struct sockaddr_un, sun_path ) + len );

    int en = errno;
    close( fd );
    errno = en;

    return nw < 0 ? -1 : 1;
}

} // namespace sys
} // namespace MagAOX
