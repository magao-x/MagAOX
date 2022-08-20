#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#include "usbtemp.h"

#define TIMEOUT 1

int ut_errno;

int owReset(int fd)
{
  int rv;
  int wbytes;
  unsigned char wbuff, rbuff;
  fd_set readset;
  struct timeval timeout_tv;
  struct termios term;

  tcflush(fd, TCIOFLUSH);

  if (tcgetattr(fd, &term) < 0) {
    ut_errno = 1;
    return -1;
  }
  term.c_cflag &= ~CSIZE | CS8;
  cfsetispeed(&term, B9600);
  cfsetospeed(&term, B9600);
  tcsetattr(fd, TCSANOW, &term);

  /* Send the reset pulse. */
  wbuff = 0xf0;
  wbytes = write(fd, &wbuff, 1);
  if (wbytes != 1) {
    ut_errno = 9;
    return -1;
  }

  timeout_tv.tv_usec = 0;
  timeout_tv.tv_sec = TIMEOUT;

  FD_ZERO(&readset);
  FD_SET(fd, &readset);

  if (select(fd + 1, &readset, NULL, NULL, &timeout_tv) > 0) {

    if (FD_ISSET(fd, &readset)) {
      int rbytes = read(fd, &rbuff, 1);
      if (rbytes != 1) {
        return -1;
      }
      switch (rbuff) {
        case 0:
          /* Ground. */
        case 0xf0:
          /* No response. */
          rv = -1;
          break;
        default:
          /* Got a response */
          rv = 0;
      }
    }
    else {
      rv = -1;
    }
  }
  else {
    rv = -1; /* Timed out or interrupt. */
  }

  term.c_cflag &= ~CSIZE | CS6;
  cfsetispeed(&term, B115200);
  cfsetospeed(&term, B115200);

  tcsetattr(fd, TCSANOW, &term);

  return rv;
}

static unsigned char owWriteByte(int fd, unsigned char wbuff)
{
  char buf[8];
  int wbytes;
  unsigned char rbuff, i;
  size_t remaining, rbytes;
  fd_set readset;
  struct timeval timeout_tv;

  tcflush(fd, TCIOFLUSH);

  for (i = 0; i < 8; i++) {
    buf[i] = (wbuff & (1 << (i & 0x7))) ? 0xff : 0x00;
  }
  wbytes = write(fd, buf, 8);
  if (wbytes != 8) {
    ut_errno = 9;
    return -1;
  }

  timeout_tv.tv_usec = 0;
  timeout_tv.tv_sec = TIMEOUT;

  FD_ZERO(&readset);
  FD_SET(fd, &readset);

  rbuff = 0;
  remaining = 8;
  while (remaining > 0) {

    if (select(fd + 1, &readset, NULL, NULL, &timeout_tv) > 0) {

      if (FD_ISSET(fd, &readset)) {
        rbytes = read(fd, &buf, remaining);
        for (i = 0; i < rbytes; i++) {
          rbuff >>= 1;
          rbuff |= (buf[i] & 0x01) ? 0x80 : 0x00;
          remaining--;
        }
      }
      else {
        return 0xff;
      }
    }
    else {
      return 0xff;
    }
  }
  return rbuff;
}

unsigned char owRead(int fd)
{
  return owWriteByte(fd, 0xff);
}

int owWrite(int fd, unsigned char wbuff)
{
  return (owWriteByte(fd, wbuff) == wbuff) ? 0 : -1;
}

static int file_exists(const char *filename)
{
  struct stat st;

  return (stat(filename, &st) == 0);
}

int owOpen(const char *serial_port)
{
  int fd;
  struct termios term;

  if (!file_exists(serial_port)) {
    ut_errno = 3;
    return -1;
  }

  /*if (access(serial_port, R_OK|W_OK) < 0) {
    ut_errno = 4;
    return -1;
  }*/

  fd = open(serial_port, O_RDWR);
  if (fd < 0) {
    ut_errno = 5;
    return -1;
  }

  memset(&term, 0, sizeof(term));

  term.c_cc[VMIN] = 1;
  term.c_cc[VTIME] = 0;
  term.c_cflag |= CS6 | CREAD | HUPCL | CLOCAL;

  cfsetispeed(&term, B115200);
  cfsetospeed(&term, B115200);

  if (tcsetattr(fd, TCSANOW, &term) < 0) {
    close(fd);
    ut_errno = 2;
    return -1;
  }

  tcflush(fd, TCIOFLUSH);

  return fd;
}

void owClose(int fd)
{
  close(fd);
}
