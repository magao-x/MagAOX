#ifndef _USBTEMP
#define _USBTEMP

#include "platform.h"

#if WINDOWS
 #define DEFAULT_SERIAL_PORT "COM3"
#else
 #define DEFAULT_SERIAL_PORT "/dev/ttyUSB0"
#endif

#define DS18X20_GENERATOR 0x8c
#define DS18X20_ROM_SIZE 8
#define DS18X20_SP_SIZE 9
#define DS18B20_SP_TH 2
#define DS18B20_SP_TL 3
#define DS18B20_SP_CONFIG 4

#define DS18B20_SP_WRITE 0x4e
#define DS18B20_SP_SAVE 0x48

char *DS18B20_errmsg(void);
HANDLE DS18B20_open(const char *);
int DS18B20_measure(HANDLE);
int DS18B20_acquire(HANDLE, float *);
int DS18B20_rom(HANDLE, unsigned char *);
int DS18B20_setprecision(HANDLE, int);
void DS18B20_close(HANDLE);

void wait_1s(void);
int is_fd_valid(HANDLE);

#endif
