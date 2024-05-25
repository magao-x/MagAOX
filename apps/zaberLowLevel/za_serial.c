/*
 * \file za_serial.c
 * \author Eric Dand
 * \version 1.0
 * \date 28 November 2014
 * \copyright Apache Software License Version 2.0
 *
 * Implementation file for ASCII portion of the Zaber Serial API in C.
 * See za_serial.h for documentation.
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "za_serial.h"

static int za_verbose = 1;

void za_set_verbose(int value)
{
	za_verbose = value;
}


#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

#if defined(NDEBUG)
#define PRINT_ERROR(M)
#define PRINTF_ERROR(M, ...)
#define PRINT_SYSCALL_ERROR(M)
#else
#define PRINT_ERROR(M) do { if (za_verbose) { fprintf(stderr, "(%s: %d) " M\
		"\n", __FILE__, __LINE__); } } while(0)
#define PRINTF_ERROR(M, ...) do { if (za_verbose) { fprintf(stderr,\
		"(%s: %d) " M "\n", __FILE__, __LINE__,  __VA_ARGS__); } } while(0)
#include <errno.h>
#define PRINT_SYSCALL_ERROR(M) do { if (za_verbose) {\
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

int za_connect(z_port *port, const char *port_name)
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

	/* READ_TIMEOUT is defined in za_serial.h */
	if (READ_TIMEOUT % 100 != 0)
	{
		tio.c_cc[VTIME] = READ_TIMEOUT / 100 + 1; /* Round up */
	}
	else
	{
		tio.c_cc[VTIME] = READ_TIMEOUT / 100;
	}
	tio.c_cc[VMIN] = 0;

	SYSCALL(cfsetospeed(&tio, B115200) & cfsetispeed(&tio, B115200));

	while(memcmp(&orig_tio, &tio, sizeof(struct termios)) != 0)
	{ /* memcmp is only OK here because we used memcpy above */
		SYSCALL(tcsetattr(*port, TCSAFLUSH, &tio));
		SYSCALL(tcgetattr(*port, &orig_tio));
	}

	return Z_SUCCESS;
}

int za_disconnect(z_port port)
{
	SYSCALL(close(port));
	return Z_SUCCESS;
}

/* a subtle feature of za_send() that is implied but not explicitly documented:
 * if you pass a pointer to a 0-length string (ie. just a \0) for command,
 * za_send() will send the minimal command of "/\n" automagically, as it will
 * assume you are sending a command without content, and that you have
 * forgotten the leading '/' and trailing '\n'. Mention of this is probably
 * best left out of the official docs since it's a hacky way to use za_send().
 */
int za_send( z_port port, 
             const char *command,
             size_t sMaxSz
           )
{
	int nlast,
		length,
		written = 0;

	if (command == NULL)
	{
		PRINT_ERROR("[ERROR] command cannot be NULL.");
		return Z_ERROR_NULL_PARAMETER;
	}

	if (command[0] != '/')
	{ /* send a '/' if they forgot it */
		SYSCALL(nlast = write(port, "/", 1));
		written += nlast;
	}
	length = strnlen(command, sMaxSz);
	SYSCALL(nlast = write(port, command, length));
	written += nlast;
	if (nlast != length)
	{ /* write can return a short write: test for it */
		PRINTF_ERROR("[ERROR] write did not write entire message: "
				"could only write %d bytes of %d.", nlast, length);
		return Z_ERROR_SYSTEM_ERROR;
	}

	if (length == 0 || command[length-1] != '\n')
	{ /* send a newline if they forgot it */
		SYSCALL(nlast = write(port, "\n", 1));
		written += nlast;
	}

	return written;
}

int za_receive(z_port port, char *destination, int length)
{
	int nread = 0,
		nlast;
	char c;

	for (;;)
	{
		SYSCALL(nlast = (int) read(port, &c, 1));

		if (nlast == 0) /* timed out */
		{
			//PRINTF_ERROR("[INFO] Read timed out after reading %d "
			//		"bytes.", nread);
			return Z_ERROR_TIMEOUT;
		}

		if (destination != NULL)
		{
			destination[nread] = c;
		}
		nread += nlast;

		if (nread == length)
		{
			PRINTF_ERROR("[ERROR] Read destination buffer not large "
					"enough. Recommended size: 256B. Your size: %dB.",
					length);
			return Z_ERROR_BUFFER_TOO_SMALL;
		}

		if (c == '\n')
		{
			nread -= 2; /* prepare to cut off the "\r\n" */
			if (nread < 0)
			{
				PRINT_ERROR("[ERROR] Reply too short. It is likely that "
						"only a partial reply was read.");
				return Z_ERROR_SYSTEM_ERROR;
			}
			if (destination != NULL)
			{
				destination[nread] = '\0'; /* chomp the "\r\n" */
			}
			return nread;
		}
	}
}

int za_setbaud(z_port port, int baud)
{
	struct termios tio;
	speed_t tbaud;
	switch (baud)
	{
		case 9600:
			tbaud = B9600;
			break;
		case 19200:
			tbaud = B19200;
			break;
		case 38400:
			tbaud = B38400;
			break;
		case 57600:
			tbaud = B57600;
			break;
		case 115200:
			tbaud = B115200;
			break;
		default:
			PRINTF_ERROR("[ERROR] Invalid baud rate. Valid rates are "
						"9600, 19200, 38400, 57600, and 115200 (default).\n"
						"Your rate: %d.", baud);
			return Z_ERROR_INVALID_BAUDRATE;
	}

	SYSCALL(tcgetattr(port, &tio));
	SYSCALL(cfsetospeed(&tio, tbaud) & cfsetispeed(&tio, tbaud));
	SYSCALL(tcsetattr(port, TCSAFLUSH, &tio));

	return Z_SUCCESS;
}

int za_drain(z_port port)
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


/* A helper for za_decode. Copies from source to destination, until delim or
 * a '\0' is found, then null-terminates destination.
 *
 * Returns the number of bytes copied, including the added null-terminator. */
static size_t copy_until_delim(char *destination, const char *source,
		const char delim, size_t destination_length)
{
	size_t i;

	for(i = 0; source[i] != delim && source[i] != '\0'
			&& i < destination_length - 1; i++)
	{
		destination[i] = source[i];
	}

	destination[i] = '\0'; /* null-terminate instead of copying delim */

	return i + 1;
}

static int decode_reply( struct za_reply *destination, 
                         const char *reply,
                         size_t sMaxSz
                       )
{
	char buffer[8]; /* needs to be at least 5B: set to 8 because
					   it will be padded to 8 from 5 anyway. */
	size_t offset,
		   length;

	if (strnlen(reply, sMaxSz) < 18)
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: shorter than expected. "
				"It is likely that only a partial reply was read.\n"
				"Reply value: %s", reply);
		return Z_ERROR_COULD_NOT_DECODE;
	}

	destination->message_type = '@';

	/* device address: 2 digits, 00-99
	 *
	 * The device address is part of the same "token" as the message type,
	 * so we call strtol on the same token, skipping the first char.
	 *
	 * We don't check the length here for future-proofing:
	 * if we add support for more device addresses (ie. more digits),
	 * this should handle it gracefully. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->device_address = (int) strtol(buffer + 1, NULL, 10);
	reply += offset;

	/* axis number: 1 digit, 0-9
	 *
	 * The axis number may be 2 digits (or more) in the future, but it's
	 * unlikely to go over 2^31 - 1 any time soon, so we convert to
	 * standard int and use that. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->axis_number = (int) strtol(buffer, NULL, 10);
	reply += offset;

	/* reply flags: 2 letters
	 *
	 * Only replies have reply flags. Value is either "OK" or "RJ". */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	if (offset > sizeof(destination->reply_flags))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: reply flags too "
				"long. Maximum length: %zu. Your length: %zu. Reply flags "
				"value: %s\n.", sizeof(destination->reply_flags), offset,
				buffer);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->reply_flags, buffer);
	reply += offset;

	/* device status: 4 letters
	 *
	 * Replies and alerts have a "status" field. Value is either "IDLE" or
	 * "BUSY", depending on what the device is doing at the time. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	if (offset > sizeof(destination->device_status))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: device status too "
				"long. Expected length: %zu. Your length: %zu. Device status "
				"value: %s\n.", sizeof(destination->device_status), offset,
				buffer);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->device_status, buffer);
	reply += offset;

	/* warning flags: 2 letters
	 *
	 * Replies and alerts have warning flags. Warning flags are typically
	 * "--". All other possible warning flag values should be two
	 * consecutive capital letters. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	if (offset > sizeof(destination->warning_flags))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: warning flags too "
				"long. Expected length: %zu. Your length: %zu. Warning flags "
				"value: %s\n.", sizeof(destination->warning_flags), offset,
				buffer);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->warning_flags, buffer);
	reply += offset;

	/* data: 1+ characters, probably less than 128 characters
	 *
	 * Replies and info have data. This can be anything, including spaces,
	 * numbers, and human-readable text. */
	length = strnlen(reply, sMaxSz); /* get length of remaining data */
	if (length > sizeof(destination->response_data))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: response data too "
				"long. Maximum length: %zu. Your length: %zu. Data: %s.\n",
				sizeof(destination->response_data), length, reply);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->response_data, reply);

	return Z_SUCCESS;

}

static int decode_alert( struct za_reply *destination, 
                         const char *reply,
                         size_t sMaxSz
                       )
{
	size_t offset;
	char buffer[8]; /* needs to be at least 5B: set to 8 because
					   it will be padded to 8 from 5 anyway. */

	if (strnlen(reply, sMaxSz) < 13)
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: shorter than expected. "
				"It is likely that only a partial reply was read.\n"
				"Reply value: %s", reply);
		return Z_ERROR_COULD_NOT_DECODE;
	}

	destination->message_type = '!';

	/* device address: 2 digits, 00-99
	 *
	 * The device address is part of the same "token" as the message type,
	 * so we call strtol on the same token, skipping the first char.
	 *
	 * We don't check the length here for future-proofing:
	 * if we add support for more device addresses (ie. more digits),
	 * this should handle it gracefully. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->device_address = (int) strtol(buffer + 1, NULL, 10);
	reply += offset;

	/* axis number: 1 digit, 0-9
	 *
	 * The axis number may be 2 digits (or more) in the future, but it's
	 * unlikely to go over 2^31 - 1 any time soon, so we convert to
	 * standard int and use that. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->axis_number = (int) strtol(buffer, NULL, 10);
	reply += offset;

	destination->reply_flags[0] = '\0';

	/* device status: 4 letters
	 *
	 * Replies and alerts have a "status" field. Value is either "IDLE" or
	 * "BUSY", depending on what the device is doing at the time. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	if (offset > sizeof(destination->device_status))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: device status too "
				"long. Expected length: %zu. Your length: %zu. Device status "
				"value: %s\n.", sizeof(destination->device_status), offset,
				buffer);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->device_status, buffer);
	reply += offset;

	/* warning flags: 2 letters
	 *
	 * Replies and alerts have warning flags. Warning flags are typically
	 * "--". All other possible warning flag values should be two
	 * consecutive capital letters. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	if (offset > sizeof(destination->warning_flags))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: warning flags too "
				"long. Expected length: %zu. Your length: %zu. Warning flags "
				"value: %s\n.", sizeof(destination->warning_flags), offset,
				buffer);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->warning_flags, buffer);

	destination->response_data[0] = '\0';

	return Z_SUCCESS;
}

static int decode_info( struct za_reply *destination, 
                        const char *reply,
                        size_t sMaxSz
                      )
{
	size_t length,
		   offset;
	char buffer[8]; /* needs to be at least 5B: set to 8 because
					   it will be padded to 8 from 5 anyway. */

	if (strnlen(reply, sMaxSz) < 7)
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: shorter than expected. "
				"It is likely that only a partial reply was read.\n"
				"Reply value: %s", reply);
		return Z_ERROR_COULD_NOT_DECODE;
	}

	destination->message_type = '#';

	/* device address: 2 digits, 00-99
	 *
	 * The device address is part of the same "token" as the message type,
	 * so we call strtol on the same token, skipping the first char.
	 *
	 * We don't check the length here for future-proofing:
	 * if we add support for more device addresses (ie. more digits),
	 * this should handle it gracefully. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->device_address = (int) strtol(buffer + 1, NULL, 10);
	reply += offset;

	/* axis number: 1 digit, 0-9
	 *
	 * The axis number may be 2 digits (or more) in the future, but it's
	 * unlikely to go over 2^31 - 1 any time soon, so we convert to
	 * standard int and use that. */
	offset = copy_until_delim(buffer, reply, ' ', sizeof(buffer));
	destination->axis_number = (int) strtol(buffer, NULL, 10);
	reply += offset;

	destination->reply_flags[0] = '\0';
	destination->device_status[0] = '\0';
	destination->warning_flags[0] = '\0';

	/* data: 1+ characters, probably less than 128 characters
	 *
	 * Replies and info have data. This can be anything, including spaces,
	 * numbers, and human-readable text. */
	length = strnlen(reply,sMaxSz); /* get length of remaining data */
	if (length > sizeof(destination->response_data))
	{
		PRINTF_ERROR("[ERROR] Reply could not be decoded: response data too "
				"long. Maximum length: %zu. Your length: %zu. Data: %s.\n",
				sizeof(destination->response_data), length, reply);
		return Z_ERROR_COULD_NOT_DECODE;
	}
	strcpy(destination->response_data, reply);

	return Z_SUCCESS;
}

/* This function tokenizes using the above helper function to copy
 * token-by-token from reply into buffer, or directly into destination,
 * depending on the field in destination we are populating. It could probably
 * be prettier, more clever, or more efficient, but it's written to be
 * readable, reliable, and robust first.
 *
 * Note that we can safely use strcpy here because the above helper function is
 * guaranteed to write a NUL at the end of its destination string.
 *
 * See http://www.zaber.com/wiki/Manuals/ASCII_Protocol_Manual#Replies for
 * more info on replies.
 */
int za_decode( struct za_reply *destination, 
               const char *reply,
               size_t sMaxSz
             )
{
	char message_type;

	if (destination == NULL || reply == NULL)
	{
		PRINT_ERROR("[ERROR] decoding requires both a non-NULL destination "
				"and reply to decode.");
		return Z_ERROR_NULL_PARAMETER;
	}

	if (strnlen(reply, sMaxSz) == 0)
	{
		PRINT_ERROR("[ERROR] Reply could not be decoded: no data.");
		return Z_ERROR_COULD_NOT_DECODE;
	}
	message_type = reply[0];

	/* most replies are 18 chars or longer: info and alerts can be shorter */
	switch(message_type)
	{
		case '@':
			return decode_reply(destination, reply, sMaxSz);
		case '!':
			return decode_alert(destination, reply, sMaxSz);
		case '#':
			return decode_info(destination, reply, sMaxSz);
		default:
			PRINTF_ERROR("[ERROR] Reply could not be decoded: unexpected "
					"message type. Valid types are '@' (reply), '!' (alert), "
					"and '#' (info). Your type: '%c'. \n", message_type);
			return Z_ERROR_COULD_NOT_DECODE;
	}
}
