#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char** argv) {
int pipe_wrapper(int[2], char*, char*, char*);
char* argv0 = *argv;
char* path;
int xp[2];
int istat;
struct stat fdstat;
int exit_on_error = 0;

    while (--argc) {
        path = *(++argv);
        xp[0] = xp[1] = -1;

        if (!strcmp("--exit-on-error",path)) {
            exit_on_error = 1;
            continue;
        }
        if (!strncmp("--",path,2)) { continue; }

        errno = 0;
        fprintf(stdout,"\n%s:  About to open FIFO pipes", path);
        fflush(stdout);
        istat = pipe_wrapper(xp, strlen(path) ? path : NULL, "", "");
        fprintf(stdout,"; pipe_wrapper([%d,%d],%s,%s,%s)=%d ERRNO=%d", xp[0], xp[1], path, "", "", istat, errno);
        fflush(stdout);
        if (istat < 0) {
            perror("# pipe_wrapper");
            if (!exit_on_error) continue;
            return -1;
        }

        errno = 0;
        istat = fcntl(xp[0], F_GETFL);
        fprintf(stdout,"; fcntl(%d,F_GETFL)=0x%x ERRNO=%d", xp[0], istat, errno);
        fflush(stdout);
        if (errno != 0) { perror("# fcntl read pipe"); }

        errno = 0;
        istat = fcntl(xp[1], F_GETFL);
        fprintf(stdout,"; fcntl(%d,F_GETFL)=0x%x ERRNO=%d", xp[1], istat, errno);
        fflush(stdout);
        if (errno != 0) { perror("# fcntl write pipe"); }

        do {
        char buf1k[1024] = { "<<<empty init>>>" };

            errno = 0;
            fprintf(stdout,"; about to write ...");
            fflush(stdout);
            istat = write(xp[1],path,strlen(path)+1);
            fprintf(stdout,"; write(%d,'%s')=%d, ERRNO=%d", xp[1], path, istat, errno);
            if (istat < 0) {
                perror("# write");
                if (!exit_on_error) break;;
                return -1;
            }

            errno = 0;
            fprintf(stdout,"; about to read(%d,...,1023)", xp[0]);
            fflush(stdout);
            istat = read(xp[0],buf1k,1023);
            fprintf(stdout,"; read(%d,'%s',1023)=%d, ERRNO=%d", xp[0], buf1k, istat, errno);
            fflush(stdout);
            if (istat < 0) {
                perror("# read");
                if (!exit_on_error) break;
                return -1;
            }

        } while(0);

        errno = 0;
        fprintf(stdout,"; about to read-only close");
        fflush(stdout);
        istat = close(xp[0]);
        fprintf(stdout,"; post read-only close(%d)=%d ERRNO=%d", xp[0], istat, errno);
        fflush(stdout);
        if (istat < 0) {
            perror("# final close read");
            if (exit_on_error) return -1;;
        }

        errno = 0;
        fprintf(stdout,"; about to write-only close");
        fflush(stdout);
        istat = close(xp[1]);
        fprintf(stdout,"; post write-only close(%d)=%d ERRNO=%d", xp[1], istat, errno);
        fflush(stdout);
        if (istat < 0) {
            perror("# final close write");
            if (exit_on_error) return -1;
        }

        if (!strcmp("tmpfifo",path)) {
            errno = 0;
            fprintf(stdout,"; about to unlink tmpfifo");
            fflush(stdout);
            istat = unlink(path);
            fprintf(stdout,"; post unlink('tmpfifo')=%d ERRNO=%d", istat, errno);
            fflush(stdout);
            if (istat < 0) {
                perror("# unlink tmpfifo");
                if (exit_on_error) return -1;;
            }
        }

        fprintf(stdout,"\n");
    }
    fprintf(stdout,"\nO_NONBLOCK=0x%x; __O_LARGEFILE=0x%x; O_RDONLY=0x%x; O_WRONLY=0x%x; O_RDWR=0x%x; O_ACCMODE=0x%x\n"
                  , O_NONBLOCK, __O_LARGEFILE, O_RDONLY, O_WRONLY, O_RDWR, O_ACCMODE
                  );
    return 0;
}
