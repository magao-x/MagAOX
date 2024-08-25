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
int fdwr;
int istat;
struct stat fdstat;

  while (--argc) {
    path = *(++argv);
    fdrd = fdwr = -1;

    errno = 0;
    fprintf(stdout,"\n%s:  About to open for reading, non-blocking", path);
    fflush(stdout);
    fdrd = open(path,O_RDONLY | O_NONBLOCK);
    fprintf(stdout,"; read-only open(%s,O_RDONLY|O_NONBLOCK)=%d ERRNO=%d", path, fdrd, errno);
    fflush(stdout);
    if (fdrd < 0) {
      perror("# read-only open");
      continue;
    }

    while (fdrd > 2) {
    char buf1k[1024] = { "<<<empty init>>>" };

      errno = 0;
      fprintf(stdout,"; about to fstat(%d)", fdrd);
      fflush(stdout);
      istat = fstat(fdrd,&fdstat);
      fprintf(stdout,"; fstat(...)=0x%x:  %s (fdrd=%d) is%s a FIFO", istat, path, fdrd, S_ISFIFO(fdstat.st_mode) ? "" : " not");
      fflush(stdout);
      if (fdrd < 0) { perror("# read-only fstat"); break; }
      if (!S_ISFIFO(fdstat.st_mode)) { break; }

      fprintf(stdout,"; about to write-only open '%s'", path);
      fflush(stdout);
      fdwr = open(path,O_WRONLY);
      fprintf(stdout,"; write-only open(%s)=%d ERRNO=%d", path, fdwr, errno);
      fflush(stdout);
      if (fdwr < 0) { perror("# write-only open"); break; }

      errno = 0;
      fprintf(stdout,"; about to write ...");
      fflush(stdout);
      istat = write(fdwr,path,strlen(path)+1);
      fprintf(stdout,"; write(%d,'%s')=%d, ERRNO=%d", fdwr, path, istat, errno);
      if (fdrd < 0) { perror("# write"); }

      errno = 0;
      fprintf(stdout,"; about to close(%d) '%s'", fdrd, path);
      fflush(stdout);
      istat = close(fdrd);
      fprintf(stdout,"; close(%d)=%d, ERRNO=%d", fdrd, istat, errno);
      fflush(stdout);
      if (istat < 0) { perror("# close read"); }

      errno = 0;
      fprintf(stdout,"; about to re-open '%s'", path);
      fflush(stdout);
      fdrd = open(path,O_RDONLY);
      fprintf(stdout,"; re-open(%s,O_RDONLY)=%d, ERRNO=%d", path, fdrd, errno);
      fflush(stdout);
      if (fdrd < 0) { perror("# re-open for read"); }

      errno = 0;
      fprintf(stdout,"; about to read(%d,...,1023)", fdrd);
      fflush(stdout);
      istat = read(fdrd,buf1k,1023);
      fprintf(stdout,"; read(%d,'%s',1023)=%d, ERRNO=%d", fdrd, buf1k, istat, errno);
      fflush(stdout);
      if (istat < 0) { perror("# read"); }

      break;
    }

    errno = 0;
    fprintf(stdout,"; about to read-only close");
    fflush(stdout);
    istat = close(fdrd);
    fprintf(stdout,"; post read-only close(%d)=%d ERRNO=%d", fdrd, istat, errno);
    fflush(stdout);
    if (istat < 0) { perror("# final close read"); }

    errno = 0;
    fprintf(stdout,"; about to write-only close");
    fflush(stdout);
    istat = close(fdwr);
    fprintf(stdout,"; post write-only close(%d)=%d ERRNO=%d", fdwr, istat, errno);
    fflush(stdout);
    if (istat < 0) { perror("# final close write"); }
    fprintf(stdout,"\n");
  }
  
  return 0;
}
