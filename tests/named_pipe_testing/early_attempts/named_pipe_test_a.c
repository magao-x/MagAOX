#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char** argv) {
char* argv0 = *argv;
char* path;
int fdrd;
int istat;
struct stat fdstat;

  while (--argc) {
    path = *(++argv);
    errno = 0;
    fprintf(stdout,"; about to open '%s'", path);
    fflush(stdout);
    fdrd = open(path,O_RDONLY | O_NONBLOCK);
    fprintf(stdout,"\n%s:  post-open ERRNO=%d", path, errno);
    if (fdrd > 0) {
      errno = 0;
      istat = fstat(fdrd,&fdstat);
      fprintf(stdout,"; post-fstat ERRNO=%d", errno);
      fprintf(stdout,"; fstat(...)=%d:  %s (fdrd=%d) is%s a FIFO",istat,path,fdrd,S_ISFIFO(fdstat.st_mode) ? "" : " not");
      if (S_ISFIFO(fdstat.st_mode)) {
      int fdwrt;
      int iint;
      int forkedpid;
        fflush(stdout);
        errno = 0;
        if ((forkedpid = fork()) < 0) {
          fprintf(stdout,"; fork=%d, ERRNO=%d", forkedpid, errno);
          fflush(stdout);
          perror(path);
        } else if (0 == forkedpid) {
          sleep(1.0);
          errno = 0;
          fdwrt = open(path,O_WRONLY);
          fprintf(stdout,"; forked write-only open(%s) ERRNO=%d", path, errno);
          errno = 0;
          iint = write(fdwrt,path,strlen(path)+1);
          fprintf(stdout,"; forked write(%d,'%s')=%d, ERRNO=%d", fdwrt, path, iint, errno);
          errno = 0;
          iint = close(fdwrt);
          fprintf(stdout,"; forked close(%d) ERRNO=%d", fdwrt, errno);
          fflush(stdout);
          exit(0);
        } else {
        char buf1k[1024] = { "<<<empty init>>>" };
        int iarg;

          errno = 0;
          istat = close(fdrd);
          fprintf(stdout,"; close(%d,F_GETFL)=0x%x, ERRNO=%d", fdrd, istat, errno);
          fflush(stdout);
          if (istat < 0) { perror("close"); }

          errno = 0;
          fprintf(stdout,"; about to re-open '%s'", path);
          fflush(stdout);
          fdrd = open(path,O_RDONLY);
          fprintf(stdout,"; re-open(%s,O_RDONLY)=0x%x, ERRNO=%d", path, iint, fdrd, errno);
          fflush(stdout);
          if (fdrd < 0) { perror("re-open(,F_GETFL)"); }

          errno = 0;
          fprintf(stdout,"; about to read(%d,...,1023)", fdrd);
          fflush(stdout);
          iint = read(fdrd,buf1k,1023);
          if (iint < 0) { perror("read"); }
          fprintf(stdout,"; read(%d,'%s',1023)=%d, ERRNO=%d", fdrd, buf1k, iint, errno);
          fflush(stdout);
        }
      }
    } else {
      perror(path);
    }
    errno = 0;
    istat = close(fdrd);
    fprintf(stdout,"; post-close ERRNO=%d", errno);
    fprintf(stdout,"; close(%d) = %d\n",fdrd,istat);
    fflush(stdout);
    fflush(stderr);
  }
  
  return 0;
}
