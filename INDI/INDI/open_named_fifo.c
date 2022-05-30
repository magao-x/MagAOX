#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "strcat_varargs.h"
#include "open_named_fifo.h"

/* Open a named FIFO:  return file descriptor for success, else -1
 */
int open_named_fifo(int O_rwoption, ...)
{
#define LFULLPATH 2048
char fullpath[LFULLPATH];
char* pfp;
int fdrd = -1;
int fdrtn = -1;
int istat;
struct stat fdstat;

    // Ensure read/write option is one of the three valid values
    switch ((O_RDONLY|O_WRONLY|O_RDWR) & O_rwoption) {
    case O_RDONLY:
    case O_WRONLY:
    case O_RDWR:
        break;
    default:
        errno = EINVAL;
        return -1;
    }

    ////////////////////////////////////////////////////////////////////
    do { // Poor-man's exception handling via break
    ////////////////////////////////////////////////////////////////////
        va_list ap;

        // Build path
        va_start(ap, O_rwoption);
        pfp = vstrcat_varargs(fullpath, LFULLPATH, ap);
        va_end(ap);

        // Break (throw "exception") if path could not be built
        if (!pfp) { errno = ENAMETOOLONG; break; }

        // Open FIFO read-only and non-blocking; create FIFO if needed
        fdrd = open(fullpath,O_RDONLY | O_NONBLOCK);

        if (fdrd < 0 && errno == ENOENT) {
            // File does not exist: create FIFO; open read-only
            mode_t prev_mode;
            errno = 0;
            prev_mode = umask(0);
            istat = mkfifo(fullpath, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP);
            prev_mode = umask(prev_mode);
            if (istat < 0) { break; }
            fdrd = open(fullpath, O_RDONLY | O_NONBLOCK);
        }
        if (fdrd < 0) { break; }

        //  Ensure opened file is a FIFO
        istat = fstat(fdrd,&fdstat);
        if (istat < 0) { break; }
        if (!S_ISFIFO(fdstat.st_mode)) { errno = EEXIST; break; }

        //  Open file with desired read/write option
        fdrtn = open(fullpath,O_rwoption | (O_RDONLY==O_rwoption ? O_NONBLOCK : 0));
        if (fdrtn < 0) { break; }

        istat = close(fdrd);
        if (istat < 0) { break; }

        return fdrtn;  // Success
    ////////////////////////////////////////////////////////////////////
    } while (0);  // End of poor-man's exception handling
    ////////////////////////////////////////////////////////////////////

    {
    int icleanup = errno;
        close(fdrd);
        close(fdrtn);
        errno = icleanup;
    }
    return -1;
}
