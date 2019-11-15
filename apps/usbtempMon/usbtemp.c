#include "usbtemp.h"

#ifdef __linux__
#include "linux.c"
void wait_1s(void)
{
  struct timeval wait_tv;

  wait_tv.tv_usec = 0;
  wait_tv.tv_sec = 1;
  select(0, NULL, NULL, NULL, &wait_tv);
}

int is_fd_valid(HANDLE fd)
{
  return (fd > 0);
}
#else
#include "windows.c"
void wait_1s(void)
{
  Sleep(1000);
}

int is_fd_valid(HANDLE fd)
{
  return (fd != INVALID_HANDLE_VALUE);
}
#endif

extern int ut_errno;

static char* ut_msgs[] = {
  "",
  "Error, could not get baudrate!",
  "Error, could not set baudrate!",
  "Error, serial port does not exist!",
  "Error, you don't have rw permission to access serial port!",
  "Error, failed to open serial port device!",
  "Error, sensor not found!", /* 6 */
  "Error, sensor CRC mismatch!",
  "Warining, not expected sensor response!",
  "Error, could not send data!"
};

static unsigned char lsb_crc8(unsigned char *data_in, unsigned int len, const unsigned char generator)
{
  unsigned char i, bit_counter;
  unsigned char crc = 0;

  for (i = 0; i < len; i++) {
    crc ^= *(data_in + i);
    bit_counter = 8;
    do {
      if (crc & 0x01) {
        crc = (((crc >> 1) & 0x7f) ^ generator);
      }
      else {
        crc = (crc >> 1) & 0x7f;
      }
      bit_counter--;
    } while (bit_counter > 0);
  }
  return crc;
}

char *DS18B20_errmsg(void)
{
  return ut_msgs[ut_errno];
}

static int DS18B20_start(HANDLE fd)
{
  if (owReset(fd) < 0) {
    ut_errno = 6;
    return -1;
  }
  if (owWrite(fd, 0xcc) < 0) {
    ut_errno = 8;
    return -1;
  }
  return 0;
}

static int DS18B20_sp(HANDLE fd, unsigned char *sp)
{
  unsigned char i, crc;

  if (DS18B20_start(fd) < 0) {
    return -1;
  }
  if (owWrite(fd, 0xbe) < 0) {
    ut_errno = 8;
    return -1;
  }
  for (i = 0; i < DS18X20_SP_SIZE; i++) {
    *(sp + i) = owRead(fd);
  }

  if ((*(sp + 4) & 0x9f) != 0x1f) {
    ut_errno = 6;
    return -1;
  }

  crc = lsb_crc8(sp, DS18X20_SP_SIZE - 1, DS18X20_GENERATOR);
  if (*(sp + DS18X20_SP_SIZE - 1) != crc) {
    ut_errno = 7;
    return -1;
  }

  return 0;
}

int DS18B20_measure(HANDLE fd)
{
  if (DS18B20_start(fd) < 0) {
    return -1;
  }
  if (owWrite(fd, 0x44) < 0) {
    ut_errno = 8;
    return -1;
  }
  return 0;
}

int DS18B20_setprecision(HANDLE fd, int precision)
{
  int i, rv;
  unsigned char cfg_old, cfg[4], sp_sensor[DS18X20_SP_SIZE];
  unsigned char *p;

  p = cfg + 3;
  *p = 0x1f | ((unsigned char)(precision - 9) << 5);

  rv = DS18B20_sp(fd, sp_sensor);
  if (rv < 0) {
    return rv;
  }

  cfg_old = sp_sensor[DS18B20_SP_CONFIG];
  if (cfg_old == *p) {
    return 0;
  }

  p--;
  *p-- = sp_sensor[DS18B20_SP_TL];
  *p-- = sp_sensor[DS18B20_SP_TH];
  *p = DS18B20_SP_WRITE;

  if (DS18B20_start(fd) < 0) {
    return -1;
  }
  for (i = 0; i < 4; i++) {
    if (owWrite(fd, *p++) < 0) {
      ut_errno = 8;
      return -1;
    }
  }

  if (DS18B20_start(fd) < 0) {
    return -1;
  }
  if (owWrite(fd, DS18B20_SP_SAVE) < 0) {
    ut_errno = 8;
    return -1;
  }

  return 0;
}

int DS18B20_acquire(HANDLE fd, float *temperature)
{
  int rv;
  unsigned short T;
  unsigned char sp_sensor[DS18X20_SP_SIZE];

  rv = DS18B20_sp(fd, sp_sensor);
  if (rv < 0) {
    return rv;
  }

  T = (sp_sensor[1] << 8) + (sp_sensor[0] & 0xff);
  if ((T >> 15) & 0x01) {
    T--;
    T ^= 0xffff;
    T *= -1;
  }
  *temperature = (float)T / 16;

  return 0;
}

int DS18B20_rom(HANDLE fd, unsigned char *rom)
{
  unsigned char i, crc;

  if (owReset(fd) < 0) {
    ut_errno = 6;
    return -1;
  }
  if (owWrite(fd, 0x33) < 0) {
    ut_errno = 8;
    return -1;
  }

  for (i = 0; i < DS18X20_ROM_SIZE; i++) {
    rom[i] = owRead(fd);
  }

  crc = lsb_crc8(rom, DS18X20_ROM_SIZE - 1, DS18X20_GENERATOR);
  if (*(rom + DS18X20_ROM_SIZE - 1) != crc) {
    ut_errno = 7;
    return -1;
  }

  return 0;
}

HANDLE DS18B20_open(const char *serial_port)
{
  return owOpen(serial_port);
}

void DS18B20_close(HANDLE fd)
{
  return owClose(fd);
}
