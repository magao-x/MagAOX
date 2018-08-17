/**
 * \file z_common.h
 * \author Eric Dand
 * \version 1.0 
 * \date 28 November 2014
 * \copyright Apache Software License Version 2.0
 *
 * \brief Defines a few things that all of the serial API has in common.
 * 
 * This file should not be included directly: only include either za_serial.h
 * or zb_serial.h, which will in turn include this file. The purpose of this
 * file is to avoid code duplication and to enable a user to include both
 * halves of the API in one source file without too many include guards and 
 * other preprocessor mess.
 */
#if !defined(Z_COMMON_H)
#define Z_COMMON_H

/** Allows for programmatic access to the library's version number. */
#define VERSION 1.0

/** \typedef z_port
 *
 * A type to represent a port connected to one or more Zaber devices.
 * Essentially a wrapper, this type is a `HANDLE` on Windows, and a file
 * descriptor (`int`) on *NIX. za_connect() and zb_connect() will properly
 * set and configure a `z_port`.
 */
#if defined(_WIN32)
#include <windows.h>
typedef HANDLE z_port;
#elif defined(__unix__) || defined(__APPLE__)
typedef int z_port;
#endif /* if defined(_WIN32) and other OS checks */

/** Defines how long, in milliseconds, za_receive() and zb_receive() should 
 * wait for input before returning without a full message. 
 *
 * This number acts as an upper bound on how long the receive functions will
 * take: they will return immediately once a full message is received.
 *
 * A note about the read timeout on *NIX operating systems: because of the way
 * the POSIX "termios" functions work, this value will be rounded down to the 
 * nearest tenth of a second (eg. 200ms = 246ms = 0.2s). A value between 0 and
 * 100 will be rounded up to 100 instead of down to 0 to give slightly more 
 * consistent behaviour between Windows and *NIX systems.
 *
 * Change this value with caution. It is set to two seconds by default, 
 * but a shorter time may be desired if many operations in your program 
 * depend on reading until a timeout. See zb_set_timeout() for more 
 * info on how this value affects the behaviour of zb_serial.h.
 */
#define READ_TIMEOUT 2000

/** \enum z_returns Defines a set of return values in case things go wrong.
 *
 * All errors are negative values in order to not be confused with the 
 * 0-or-greater regular return values. This was done so that a user can check
 * whether a return value is < 0 to check for all error codes simultaneously.
 *
 * Remember to check your return values! It's good for you.
 */
enum z_returns { 
	/** Everything is OK! */
	Z_SUCCESS = 0,
	/** Something went wrong in system code */
	Z_ERROR_SYSTEM_ERROR = -1,
	/** Tried to write to a buffer that wasn't long enough */
	Z_ERROR_BUFFER_TOO_SMALL = -2,
	/** Was passed NULL when not expecting it */
	Z_ERROR_NULL_PARAMETER = -3,
	/** Tried to set an unsupported baudrate */
	Z_ERROR_INVALID_BAUDRATE = -4,
	/** Tried to decode a partial reply, 
	 * or a string that wasn't a reply at all */
	Z_ERROR_COULD_NOT_DECODE = -5
};

#endif /* if !defined(Z_COMMON_H) */

