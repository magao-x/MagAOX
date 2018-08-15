/*
 * \file zb_serial.c
 * \author Eric Dand
 * \version 1.0 
 * \date 28 November 2014
 * \copyright Apache Software License Version 2.0
 *
 * Implementation file for binary portion of the Zaber Serial API in C.
 * See zb_serial.h for documentation.
 */
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

#include "zb_serial.h"

static int zb_verbose = 1;

void zb_set_verbose(int value)
{
	zb_verbose = value;
}

int zb_encode(uint8_t *destination, uint8_t device_number, 
		uint8_t command_number, int32_t data)
{
	unsigned int i;
	uint32_t udata = (uint32_t)data;

	if (destination == NULL)
	{
		return Z_ERROR_NULL_PARAMETER;
	}

	destination[0] = device_number;
	destination[1] = command_number;

	for (i = 2; i < 6; i++) 
	{
		destination[i] = (uint8_t)udata;
		udata >>= 8;
	}

	return Z_SUCCESS;
}

int zb_decode(int32_t *destination, const uint8_t *reply)
{
	unsigned int i;
	uint32_t data = 0;
	
	if (destination == NULL)
	{
		return Z_ERROR_NULL_PARAMETER;
	}

	for (i = 5; i > 1; i--)
	{
		data <<= 8;
		data |= reply[i];
	}

	if ((data & 0x80000000UL) == 0)
	{
		*destination = (int32_t)data;
	}
	else
	{
		*destination = -(int32_t)(UINT32_MAX - data) - 1;
	}

	return Z_SUCCESS;
}

#if defined(_WIN32)

/* These macros save us a lot of repetition. Call the specified function,
 * and complain and return early with -1 if things go badly. */
#if defined(NDEBUG)
#define PRINT_ERROR(M)
#define PRINT_SYSCALL_ERROR(M)
#else
#define PRINT_ERROR(M) do { if (zb_verbose) { fprintf(stderr, "(%s: %d)" M\
		"\n", __FILE__, __LINE__); } } while(0)
#define PRINT_SYSCALL_ERROR(M) do { if (zb_verbose) { fprintf(stderr,\
		"(%s: %d) [ERROR] " M " failed with error code %d.\n",\
		__FILE__, __LINE__, GetLastError()); } } while(0)
#endif
#define SYSCALL(F) do { if ((F) == 0) {\
		PRINT_SYSCALL_ERROR(#F); return Z_ERROR_SYSTEM_ERROR; } } while (0)

int zb_connect(z_port *port, const char *port_name)
{
	DCB dcb = { 0 };
	COMMTIMEOUTS timeouts;
	
	if (port_name == NULL)
	{
		PRINT_ERROR("[ERROR] port name cannot be NULL.");
		return Z_ERROR_NULL_PARAMETER;
	}
	
	*port = CreateFileA(port_name,
			GENERIC_READ | GENERIC_WRITE,
			0,
			NULL,
			OPEN_EXISTING,
			0,
			NULL);
	if (*port == INVALID_HANDLE_VALUE)
	{
		PRINT_SYSCALL_ERROR("CreateFileA");
		return Z_ERROR_NULL_PARAMETER;
	}

	SYSCALL(GetCommState(*port, &dcb));
	dcb.DCBlength = sizeof(DCB);
	dcb.BaudRate = 9600;
	dcb.fBinary = TRUE;  /* Binary Mode (skip EOF check) */
	dcb.fParity = FALSE;  /* Disable parity checking */
	dcb.fOutxCtsFlow = FALSE; /* No CTS handshaking on output */
	dcb.fOutxDsrFlow = FALSE; /* No DSR handshaking on output */
	dcb.fDtrControl = DTR_CONTROL_DISABLE;  /* Disable DTR Flow control */
	dcb.fDsrSensitivity = FALSE; /* No DSR Sensitivity */
	dcb.fTXContinueOnXoff = TRUE; /* Continue TX when Xoff sent */
	dcb.fOutX = FALSE; /* Disable output X-ON/X-OFF */
	dcb.fInX = FALSE;  /* Disable input X-ON/X-OFF */
	dcb.fErrorChar = FALSE;  /* Disable Err Replacement */
	dcb.fNull = FALSE; /* Disable Null stripping */
	dcb.fRtsControl = RTS_CONTROL_DISABLE;  /* Disable Rts Flow control */
	dcb.fAbortOnError = FALSE; /* Do not abort all reads and writes on Error */
	dcb.wReserved = 0; /* Not currently used, but must be set to 0 */
	dcb.XonLim = 0; /* Transmit X-ON threshold */
	dcb.XoffLim = 0;   /* Transmit X-OFF threshold */
	dcb.ByteSize = 8;  /* Number of bits/byte, 4-8 */
	dcb.Parity = NOPARITY; /* 0-4=None,Odd,Even,Mark,Space */
	dcb.StopBits = ONESTOPBIT;  /* 0,1,2 = 1, 1.5, 2 */
	SYSCALL(SetCommState(*port, &dcb));

	timeouts.ReadIntervalTimeout = MAXDWORD;
	timeouts.ReadTotalTimeoutMultiplier = MAXDWORD;
	timeouts.ReadTotalTimeoutConstant = READ_TIMEOUT; /* #defined in header */
	timeouts.WriteTotalTimeoutMultiplier = 0;
	timeouts.WriteTotalTimeoutConstant = 100;
	SYSCALL(SetCommTimeouts(*port, &timeouts));

	return Z_SUCCESS;
}

int zb_disconnect(z_port port)
{
	SYSCALL(CloseHandle(port));
	return Z_SUCCESS;
}

int zb_send(z_port port, const uint8_t *command)
{
	DWORD nbytes;

	if (command == NULL)
	{
		PRINT_ERROR("[ERROR] command cannot be NULL.");
		return Z_ERROR_NULL_PARAMETER;
	}
	
	SYSCALL(WriteFile(port, command, 6, &nbytes, NULL));
	if (nbytes == 6)
	{
		return (int) nbytes;
	}
	return Z_ERROR_SYSTEM_ERROR;
}

/* We read bytes one-at-a-time as ReadFile(port, destination, 6, &nread, NULL)
 * often enough will "read" bytes of \0 in the middle of a message when it 
 * should instead be waiting for a real byte of data down the line.
 * Worse, it reports afterwards that it has read a full 6 bytes, making this 
 * behaviour hard to track and harder to debug and compensate for. */
int zb_receive(z_port port, uint8_t *destination)
{
	DWORD nread;
	int i;
	char c;

	for (i = 0; i < 6; i++)
	{
		SYSCALL(ReadFile(port, &c, 1, &nread, NULL));

		if (nread == 0) /* timed out */
		{
			PRINT_ERROR("[INFO] Read timed out.");
			break;
		}

		if (destination != NULL) destination[i] = c;
	}

	if (i == 6)
	{
		return i;
	} 
	/* if we didn't read a whole 6 bytes, we count that as an error. */
	return Z_ERROR_SYSTEM_ERROR;
}

int zb_drain(z_port port)
{
	char c;
	DWORD nread,
		  old_timeout;
	COMMTIMEOUTS timeouts;

	SYSCALL(PurgeComm(port, PURGE_RXCLEAR));
	SYSCALL(GetCommTimeouts(port, &timeouts));
	old_timeout = timeouts.ReadTotalTimeoutConstant;
	timeouts.ReadTotalTimeoutConstant = 100;
	SYSCALL(SetCommTimeouts(port, &timeouts));

	do
	{
		SYSCALL(ReadFile(port, &c, 1, &nread, NULL));
	}
	while (nread == 1);

	timeouts.ReadTotalTimeoutConstant = old_timeout;
	SYSCALL(SetCommTimeouts(port, &timeouts));

	return Z_SUCCESS;
}

int zb_set_timeout(z_port port, int milliseconds)
{
	COMMTIMEOUTS timeouts;

	SYSCALL(GetCommTimeouts(port, &timeouts));
	timeouts.ReadTotalTimeoutConstant = milliseconds;
	SYSCALL(SetCommTimeouts(port, &timeouts));

	return milliseconds;
}

#elif defined(__unix__) || defined(__APPLE__) /* end of if defined(_WIN32) */
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

/* A little sugar for checking return values from system calls.
 * I would have liked to use GNU/GCC's "statement expressions" so that one 
 * do something like "z_port port = SYSCALL(open([parameters]))", but they're
 * a GNU extension, and therefore unfriendly to non-GCC compilers.
 * Workaround to avoid dependence on statement expressions for SYSCALL macro:
 * use SYSCALL on result of assignment instead. */
#if defined(NDEBUG)
#define PRINT_ERROR(M) 
#define PRINT_SYSCALL_ERROR(M)
#else
#include <errno.h>
#define PRINT_ERROR(M) do { if (zb_verbose) { fprintf(stderr, "(%s: %d)" M\
		"\n", __FILE__, __LINE__); } } while(0)
#define PRINT_SYSCALL_ERROR(M) do { if (zb_verbose) {\
		fprintf(stderr, "(%s: %d) [ERROR] " M " failed: %s.\n",\
		__FILE__, __LINE__, strerror(errno)); } } while(0)
#endif
/* A little sugar for checking return values from system calls.
 * I would have liked to use GNU/GCC's "statement expressions" so that one 
 * do something like "z_port port = SYSCALL(open([parameters]))", but they're
 * a GNU extension, and therefore unfriendly to non-GCC compilers.
 * Workaround to avoid dependence on statement expressions for SYSCALL macro:
 * use SYSCALL on result of assignment instead. */
#define SYSCALL(F) do { if ((F) < 0) { PRINT_SYSCALL_ERROR(#F);\
	return Z_ERROR_SYSTEM_ERROR; } } while(0)

int zb_connect(z_port *port, const char *port_name)
{
	struct termios tio, orig_tio;

	if (port == NULL || port_name == NULL)
	{
		PRINT_ERROR("[ERROR] port and port_name cannot be NULL.");
		return Z_ERROR_NULL_PARAMETER;
	}

    /* blocking read/write */
	SYSCALL(*port = open(port_name, O_RDWR | O_NOCTTY));
	SYSCALL(tcgetattr(*port, &orig_tio));
	memcpy(&tio, &orig_tio, sizeof(struct termios)); /* copy padding too */

	/* cfmakeraw() without cfmakeraw() for cygwin compatibility */
	tio.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR
			| IGNCR | ICRNL | IXON);
	tio.c_oflag &= ~OPOST;
	tio.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
	tio.c_cflag &= ~(CSIZE | PARENB);
	/* end cfmakeraw() */
	tio.c_cflag = CS8|CREAD|CLOCAL;

	/* READ_TIMEOUT is defined in zb_serial.h */
	if (READ_TIMEOUT % 100 != 0)
	{
		tio.c_cc[VTIME] = READ_TIMEOUT / 100 + 1; /* Round up */
	}
	else
	{
		tio.c_cc[VTIME] = READ_TIMEOUT / 100;
	}
	tio.c_cc[VMIN] = 0;

	SYSCALL(cfsetospeed(&tio, B9600) & cfsetispeed(&tio, B9600));

	while(memcmp(&orig_tio, &tio, sizeof(struct termios)) != 0)
	{ /* memcmp is only OK here because we used memcpy above */
		SYSCALL(tcsetattr(*port, TCSAFLUSH, &tio));
		SYSCALL(tcgetattr(*port, &orig_tio));
	}

	return Z_SUCCESS;
}

int zb_disconnect(z_port port) 
{
	SYSCALL(close(port));
	return Z_SUCCESS;
}

int zb_send(z_port port, const uint8_t *command) 
{
	int nbytes;
	
	SYSCALL(nbytes = write(port, command, 6));
	if (nbytes == 6)
	{
		return nbytes;
	}
	return Z_ERROR_SYSTEM_ERROR;
}

/* More struggles with termios: we're forced to read one byte at a time on
 * *NIX because of how termios.c_cc[VTIME] and [VMIN] work. From the termios
 * man page: 
 * 
 * * if MIN == 0; TIME > 0: "read(2) returns either when at least one
 * byte of data is available, or when the timer expires."
 * * if MIN > 0; TIME > 0: "Because the timer is started only after the
 * initial byte becomes available, at least one byte will be read."
 *
 * Neither of these cases are what we want, namely to start the timer
 * immediately, and to only return when all 6 requested/MIN bytes are received
 * or the timer times out. As a result, we instead set MIN to 0 (the first
 * case above), then read 1 byte at a time to get the behaviour we want. */
int zb_receive(z_port port, uint8_t *destination)
{
	int nread,
		i;
	char c;

    for (i = 0; i < 6; i++)
    {
        SYSCALL(nread = (int) read(port, &c, 1));

		if (nread == 0) /* timed out */
		{
			PRINT_ERROR("[INFO] Read timed out.");
			break;
		}

		if (destination != NULL) destination[i] = c;
	}

	if (i == 6)
	{
		return i;
	}
	/* if we didn't read a whole 6 bytes, we count that as an error. */
	return Z_ERROR_SYSTEM_ERROR;
}

int zb_drain(z_port port)
{
	struct termios tio;
	int old_timeout;
	char c;

	/* set timeout to 0.1s */
	SYSCALL(tcgetattr(port, &tio));
	old_timeout = tio.c_cc[VTIME];
	tio.c_cc[VTIME] = 1;
	SYSCALL(tcsetattr(port, TCSANOW, &tio));

	/* flush and read whatever else comes in */
	SYSCALL(tcflush(port, TCIFLUSH));
	while(read(port, &c, 1) > 0);

	/* set timeout back to what it was */
	tio.c_cc[VTIME] = old_timeout;
	SYSCALL(tcsetattr(port, TCSANOW, &tio));

	return Z_SUCCESS;
}

int zb_set_timeout(z_port port, int milliseconds)
{
	struct termios tio;
	int new_time;

	if (milliseconds % 100 != 0)
	{
		new_time = milliseconds / 100 + 1;
	}
	else
	{
		new_time = milliseconds / 100; /* VTIME is in increments of 0.1s */
	}

	SYSCALL(tcgetattr(port, &tio));
	tio.c_cc[VTIME] = new_time;
	SYSCALL(tcsetattr(port, TCSANOW, &tio));

	return new_time * 100;
}

#endif

