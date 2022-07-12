#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

////////////////////////////////////////////////////////////////////////
// De-reference possible char* ptr to environment variable (envvar) name
// - Returns pointer to either initial char* or to dereferenced envvar
//
// Protocol:  use leading asterisk to indicate presence of envvar name
// - e.g. *ENVNAME*default_dereference\0
// - **... if first char of target is *; leading * will be stripped
////////////////////////////////////////////////////////////////////////
#define __TMP_MAX_ENVVAR_LEN 256
char*
str_deref_envvar(char* input_ptr) {
char envvar_name[__TMP_MAX_ENVVAR_LEN];
char* default_ptr;
char* output_ptr;
size_t L;
    if (!input_ptr) { return input_ptr; }
    if ('*' != *input_ptr) { return input_ptr; }
    if ('*' == *(++input_ptr)) { return input_ptr; }
    default_ptr = strchr(input_ptr,'*');
    if (default_ptr) { L = (default_ptr++) - input_ptr; }
    if (__TMP_MAX_ENVVAR_LEN <= L) { return NULL; }
    strncpy(envvar_name, input_ptr, L);
    envvar_name[L] = '\0';
    output_ptr = getenv(envvar_name);
    return output_ptr ? output_ptr : default_ptr;
}
#undef __TMP_MAX_ENVVAR_LEN

char*
strcat_varargs(char* output_ptr0, size_t output_length, ...) {
va_list ap;
char* output_ptr = output_ptr0;
char* output_ptr_last = output_ptr + output_length;
char* nextarg_ptr;
    va_start(ap,output_length);
    while (output_ptr < output_ptr_last) {
        nextarg_ptr = va_arg(ap, char*);
        if (!nextarg_ptr) break;
        if (!nextarg_ptr) { output_ptr = output_ptr_last; break; }
        strncpy(output_ptr, nextarg_ptr, output_ptr_last-output_ptr);
        output_ptr += strlen(nextarg_ptr);
    }
    va_end(ap);
    if (output_ptr >= output_ptr_last) { return NULL; }
    *output_ptr = '\0';
    return output_ptr0;
}

int
pipe_wrapper(int xp[2], char* prefix_path, char* suffix_path) {
#define LFULLPATH 2048
char fullpath[LFULLPATH];
int fdrd = -1; /* => xp[0] */
int fdwr = -1; /* => xp[1] */
int istat;
size_t Lpfx;
size_t Lsfx;
struct stat fdstat;

    ////////////////////////////////////////////////////////////////////
    do { // Poor-man's exception handling
    ////////////////////////////////////////////////////////////////////

        // If no prefix or suffix, then use unnamed pipe
        if (!prefix_path || !suffix_path) { return pipe(xp); }

        // Build path name for named pipe (FIFO)
        Lpfx = strnlen(prefix_path,LFULLPATH);

        // If prefix too long, or empty prefix, then error
        if (Lpfx > LFULLPATH) { errno = ENAMETOOLONG; break; }
        if (!Lpfx) { errno = ENOENT; break; }

        Lsfx = strnlen(suffix_path, LFULLPATH-Lpfx);
        if ((Lpfx+Lsfx) >= LFULLPATH) { errno = ENAMETOOLONG; break; }

        //  Build path name for named pipe (FIFO)
        strcpy(fullpath,prefix_path);
        strcpy(fullpath+Lpfx,suffix_path);
        fullpath[Lpfx+Lsfx] = fullpath[LFULLPATH-1] = '\0';

        //  Open FIFO; create if needed
        fdrd = open(fullpath,O_RDONLY | O_NONBLOCK);
        if (fdrd < 0) {
        mode_t prev_mode;
            if (errno != ENOENT) { break; }
            errno = 0;
            prev_mode = umask(0);
            istat = mkfifo(fullpath, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
            prev_mode = umask(prev_mode);
            if (istat < 0) { break; }
            fdrd = open(fullpath, O_RDONLY | O_NONBLOCK);
            if (fdrd < 0) { break; }
        }

        //  Ensure opened file is a FIFO
        istat = fstat(fdrd,&fdstat);
        if (istat < 0) { break; }
        if (!S_ISFIFO(fdstat.st_mode)) { errno = EEXIST; break; }

        fdwr = open(fullpath,O_WRONLY);
        if (fdwr < 0) { break; }

        istat = close(fdrd);
        if (istat < 0) { break; }

        fdrd = open(fullpath,O_RDONLY);
        if (fdrd < 0) { break; }

        xp[0] = fdrd;
        xp[1] = fdwr;

        return 0;
    ////////////////////////////////////////////////////////////////////
    } while (0);  // End of poor-man's exception handling
    ////////////////////////////////////////////////////////////////////

    {
    int icleanup = errno;
        close(fdrd);
        close(fdwr);
        errno = icleanup;
    }
    return -1;
}

int main(int argc, char** argv) {
char* argv0 = *argv;
char* path;
int xp[2];
int istat;
struct stat fdstat;

    while (--argc) {
        path = *(++argv);
        xp[0] = xp[1] = -1;

        errno = 0;
        fprintf(stdout,"\n%s:  About to open FIFO pipes", path);
        fflush(stdout);
        istat = pipe_wrapper(xp, strlen(path) ? path : NULL, "");
        fprintf(stdout,"; pipe_wrapper([%d,%d],%s,%s)=%d ERRNO=%d", xp[0], xp[1], path, "", istat, errno);
        fflush(stdout);
        if (istat < 0) {
            perror("# pipe_wrapper");
            continue;
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
            if (istat < 0) { perror("# write"); break; }

            errno = 0;
            fprintf(stdout,"; about to read(%d,...,1023)", xp[0]);
            fflush(stdout);
            istat = read(xp[0],buf1k,1023);
            fprintf(stdout,"; read(%d,'%s',1023)=%d, ERRNO=%d", xp[0], buf1k, istat, errno);
            fflush(stdout);
            if (istat < 0) { perror("# read"); break; }

        } while(0);

        errno = 0;
        fprintf(stdout,"; about to read-only close");
        fflush(stdout);
        istat = close(xp[0]);
        fprintf(stdout,"; post read-only close(%d)=%d ERRNO=%d", xp[0], istat, errno);
        fflush(stdout);
        if (istat < 0) { perror("# final close read"); }

        errno = 0;
        fprintf(stdout,"; about to write-only close");
        fflush(stdout);
        istat = close(xp[1]);
        fprintf(stdout,"; post write-only close(%d)=%d ERRNO=%d", xp[1], istat, errno);
        fflush(stdout);
        if (istat < 0) { perror("# final close write"); }

        if (!strcmp("tmpfifo",path)) {
          errno = 0;
          fprintf(stdout,"; about to unlink tmpfifo");
          fflush(stdout);
          istat = unlink(path);
          fprintf(stdout,"; post unlink('tmpfifo')=%d ERRNO=%d", istat, errno);
          fflush(stdout);
          if (istat < 0) { perror("# unlink tmpfifo"); }
        }

        fprintf(stdout,"\n");
    }
    fprintf(stdout,"\nO_NONBLOCK=0x%x; __O_LARGEFILE=0x%x; O_RDONLY=0x%x; O_WRONLY=0x%x; O_RDWR=0x%x; O_ACCMODE=0x%x\n"
                  , O_NONBLOCK, __O_LARGEFILE, O_RDONLY, O_WRONLY, O_RDWR, O_ACCMODE
                  );
    return 0;
}
