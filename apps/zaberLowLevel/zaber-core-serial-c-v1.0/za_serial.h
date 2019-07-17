/** 
 * \file za_serial.h
 * \author Eric Dand
 * \version 1.0
 * \date 28 November 2014
 * \copyright Apache Software License Version 2.0
 * 
 * \brief Provides a set of functions for interacting with Zaber devices in 
 * the ASCII protocol.
 */

#if !defined(ZA_SERIAL_H)
#define ZA_SERIAL_H

#if defined(__cplusplus)
extern "C"{
#endif /* if defined(__cplusplus) */

#include "z_common.h"

/** Provides programmatic access to reply data.
 *
 * This struct is provided as a convenience to allow for easier interaction
 * with ASCII replies. The names of the fields are taken straight from the
 * ASCII protocol manual.
 *
 * Note that the response data may contain more than one piece of data: 
 * because many replies will return multiple pieces of data, the response_data
 * field will simply contain the end of the message, without the newline
 * terminators. (ie. "\r\n")
 */
struct za_reply {
	/** The message type will always be '@' for replies. 
	 * Info will have the type '#', and alerts '!'. */
	char message_type;
	/** The address of the device, an integer between 1 and 99. */
	int device_address;
	/** The axis number: 0-9, where 0 refers to the whole device, 
	 * and 1-9 refer only to that specific axis. */
	int axis_number;
	/** Whether a command was accepted or rejected: either "OK" or "RJ". */
	char reply_flags[3];
	/** Either "BUSY" when the axis is moving, or "IDLE" when it isn't. */
	char device_status[5];
	/** The highest priority warning for the device,
	 * or -- under normal conditions. */
	char warning_flags[3];
	/** The response for the command executed. See the protocol manual entry
	 * for your command of choice to know what to expect here. */
	char response_data[128];
};

/** Connect to a serial port specified by \a port_name. 
 *
 * Configures the port to the ASCII protocol defaults (115200 baud, 8N1). 
 * If you have set your device to run at a different baud rate, use 
 * za_setbaud() to change it after connecting using this function.
 *
 * On Linux the port name will likely be something like "/dev/ttyUSB0", 
 * on Windows "COM1", and on OS X and BSD systems "/dev/cu.usbserial-A4017CQX".
 * It is important that OS X/BSD systems use the "callout" (cu.* or cua.*) port
 * as opposed to the "dial in" ports with names starting with tty.*, 
 * which will not work with Zaber devices.
 *
 * If you are re-using a `z_port`, make sure to use za_disconnect() to 
 * disconnect the old port before overwriting it with a new one.
 *
 * \param[out] port a pointer to a `z_port` to be written-over with the 
 * newly-connected port.
 * \param[in] port_name a string containing the name of the port to be opened. 
 *
 * \return Z_SUCCESS on success, Z_ERROR_NULL_PARAMETER if \a port or 
 * \a port_name is NULL, or Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int za_connect(z_port *port, const char *port_name);

/** Gracefully closes a connection.
 *
 * \param[in] port the port to be disconnected.
 *
 * \return Z_SUCCESS on success, Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int za_disconnect(z_port port);

/** Sends a command to a serial port.
 *
 * Automatically adds a '/' to begin the command and a '\\n' to end it if
 * these characters are omitted from \a command. 
 *
 * \param[in] port the port to which to send the command.
 * \param[in] command a string containing the command to be sent.
 *
 * \return the number of bytes written, Z_ERROR_NULL_PARAMETER if \a command is
 * NULL, or Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int za_send(z_port port, const char *command);

/** Reads a message from the serial port. 
 * 
 * Blocks while reading until it encounters a newline character or its 
 * timeout has elapsed.
 *
 * Note: It is recommended that your \a destination buffer be 256B long. 
 * This is long enough to hold any reply from a Zaber device using the 
 * ASCII protocol.
 *
 * Note that this function will simply read the first message on the input
 * buffer, whatever it may be. If you have sent many commands without
 * receiving their corresponding replies, sorting them all out may be a real
 * headache. Note also that the input buffer is finite, and allowing it to
 * fill up will result in undefined behaviour. Try to receive responses to 
 * every command you send, or use za_drain() when necessary.
 *
 * Passing NULL for \a destination and 0 for \a length will read a single 
 * reply, discarding it as it is read. This is useful for keeping your 
 * commands and replies straight without using zb_drain() when you don't care
 * about the contents of most of the replies.
 *
 * \param[in] port the port to receive a message from.
 * \param[out] destination a pointer to which to write the reply read.
 * \param[in] length the length of the buffer pointed to by \a destination.
 *
 * \return the number of bytes read, Z_ERROR_SYSTEM_ERROR in case of system 
 * error, or Z_ERROR_BUFFER_TOO_SMALL if \a length is insufficient to store 
 * the reply.
 */
int za_receive(z_port port, char *destination, int length);

/** Build a #za_reply struct from a string pointed-to by reply.
 *
 * The #za_reply struct can then be used to gain easier access to the parts
 * of an ASCII reply.
 *
 * \param[out] destination a pointer to a za_reply struct to be populated 
 * with the data found in \a reply.
 * \param[in] reply a pointer to a string containing a full reply from a Zaber
 * device, as specified by the ASCII protocol manual.
 *
 * \return Z_SUCCESS on success, Z_ERROR_NULL_PARAMETER if \a destination or 
 * \a reply is NULL, or Z_ERROR_COULD_NOT_DECODE if the reply is malformed. 
 */
int za_decode(struct za_reply *destination, char *reply);

/** Changes the baud rate of both input and output. 
 *
 * This function is unlikely to be necessary for typical use, as za_connect()
 * already sets the baud rate to 115200, the recommended rate for the 
 * ASCII protocol.
 *
 * Note: za_setbaud() flushes both input and output buffers to ensure a
 * "clean slate" after the baud rate has been changed.
 *
 * Also note that this sets the baud rate at which the program tries to talk
 * to the device. It does not change the baud rate of the device itself. See
 * the ASCII Protocol Manual for info on how to change the device's baud rate.
 *
 * Valid baid rates are 9600, 19200, 38400, 57600, and 115200.
 *
 * \param[in] port the port to change.
 * \param[in] baud the new desired baud rate.
 *
 * \return Z_SUCCESS on success, Z_ERROR_INVALID_BAUDRATE if the baud rate is not one
 * specified above, or Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int za_setbaud(z_port port, int baud);

/** Flushes all input received but not yet read, and attempts to drain all
 * input that would be incoming. 
 *
 * This function is intended to be used when many commands have been sent 
 * without reading any responses, and there is a need to read a response from
 * a certain command. In other words, call this function before sending a 
 * command whose response you would like to read if you have not been 
 * consistently reading the responses to previous commands.
 *
 * This function will always take at least 100ms to complete, as it tries to
 * read input until it is certain no more is arriving by waiting 100ms before
 * deciding there is no more input incoming. Do not use it for
 * time-sensitive operations, or in any kind of "chatty" setup, eg. multiple 
 * devices daisy-chained together with move tracking enabled. In such a setup,
 * the only reliable way to retrieve reply data is to always follow
 * calls to za_send() with calls to za_receive().
 *
 * \param[in] port the port to drain.
 *
 * \return Z_SUCCESS on success, or Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int za_drain(z_port port);

/** Sets whether errors and extra info are reported to `stderr`. 
 * 
 * Set \a value to 0 to disable all output. Additionally, you can compile this
 * library with the macro NDEBUG defined, which will disable all output and
 * skip checks to "verbose" altogether in the compiled code.
 *
 * \param[in] value whether or not the program should output error messages 
 * and info to `stderr`.
 */
void za_set_verbose(int value);

#if defined(__cplusplus)
}
#endif /* if defined(__cplusplus) */

#endif /* if !defined(ZA_SERIAL_H) */
