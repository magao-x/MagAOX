/**
 * \file zb_serial.h
 * \author Eric Dand
 * \version 1.0
 * \date 28 November 2014
 * \copyright Apache Software License Version 2.0
 *
 * \brief Provides a set of functions for interacting with Zaber devices in 
 * the binary protocol.
 *
 * Before using this library, it is recommended to read about the message
 * format of the binary protocol. The manual can be found at the link below,
 * as of November 2014:
 *
 * http://www.zaber.com/wiki/Manuals/Binary_Protocol_Manual#Message_Format
 *
 * Note especially that binary replies are sent when a command has *finished*
 * as opposed to when the command was received. It is up to you to wait for
 * appropriate responses to commands. 
 */

#if !defined(ZB_SERIAL_H)
#define ZB_SERIAL_H

#if defined(__cplusplus)
extern "C"{
#endif /* if defined(__cplusplus) */

#include "z_common.h"

/** Connect to a serial port specified by \a port_name.
 *
 * Configures the port to the binary protocol defaults (9600 baud, 8N1). 
 * It is not recommended to use a device in binary at any other baud rate.
 *
 * On Linux the port name will likely be something like "/dev/ttyUSB0", 
 * on Windows "COM1", and on OS X and BSD systems "/dev/cu.usbserial-A4017CQX".
 * It is important that OS X/BSD systems use the "callout" (cu.* or cua.*) port
 * as opposed to the "dial in" ports with names starting with tty.*, 
 * which will not work with Zaber devices.
 *
 * If you are re-using a `z_port`, make sure to use zb_disconnect() to 
 * disconnect the old port before overwriting it with a new one.
 *
 * \param[out] port a pointer to a `z_port` to be written-over with the 
 * newly-connected port.
 * \param[in] port_name a string containing the name of the port to be opened. 
 *
 * \return Z_SUCCESS on success, Z_ERROR_NULL_PARAMETER if \a port or
 * \a port_name is NULL, or Z_ERROR_SYSTEM_ERROR in case of system error.
 */
int zb_connect(z_port *port, const char *port_name);

/** Gracefully closes a connection.
 *
 * \param[in] port the port to be disconnected.
 *
 * \return Z_SUCCESS on success, Z_ERROR_SYSTEM_ERROR on failure.
 */
int zb_disconnect(z_port port);

/** Sends a command to a serial port. 
 *
 * It is recommended that before sending any commands, you read the Binary 
 * Protocol Manual to understand how these commands should be formatted 
 * (least-significant byte first). See zb_encode() for help encoding commands.
 *
 * Note that this function does not typically crash if \a command is not six
 * (or greater) bytes long: it will simply write the first six bytes found at
 * the address \a command, whatever they may be. It is up to you to make sure
 * that \a command is long enough.
 *
 * \param[in] port the port to which to send the command.
 * \param[in] command a string of bytes containing the command to be sent.
 *
 * \return the number of bytes written (will always be 6 if successful), or
 * Z_ERROR_SYSTEM_ERROR on failure.
 */
int zb_send(z_port port, const uint8_t *command);

/** Receives a message from the serial port. 
 *
 * Blocks while reading until 6 bytes have been read, or its timeout has
 * elapsed. The default timeout is defined by #READ_TIMEOUT, and can also be
 * changed programmatically by zb_set_timeout(). 
 * 
 * The reply received will follow the message format specified 
 * in Zaber's Binary Protocol Manual. This means that the final 4 bytes 
 * received will be ordered from least to most significant. It is recommended 
 * to use zb_decode() to get the last 4 bytes of data as a convenient, 
 * well-formed 32-bit integer that matches your system.
 *
 * Note that this function will simply read the first six bytes on the input
 * buffer, whatever they may be. If you have sent many commands without
 * receiving their corresponding replies, sorting them all out may be a real
 * headache. Note also that this buffer is finite, and allowing it to
 * fill up will result in undefined behaviour. See zb_drain() for a solution
 * to this problem.
 *
 * \param[in] port the port to receive a message from.
 * \param[out] destination a pointer to which to write the reply read. Must be
 * at least 6 bytes long.
 *
 * \return the number of bytes received (will always be 6 if successful), or 
 * Z_ERROR_SYSTEM_ERROR on failure.
 */
int zb_receive(z_port port, uint8_t *destination);

/** Encodes a command according to Zaber's Binary Protocol Manual.
 *
 * This function performs the necessary transposition of data bytes for
 * transmission to Zaber devices using the binary protocol.
 *
 * This function does not support the encoding of Message IDs. 
 *
 * \param[out] destination a pointer to an array to which to write the encoded
 * command. Assumed to be at least 6 bytes long.
 * \param[in] device_number the number of the device to which to send the 
 * command.
 * \param[in] command_number the number of the command to be sent.
 * \param[in] data the data to be sent along with the command. The contents of
 * this field depend on the command number being sent. See the Binary Protocol
 * Manual for info on specific commands.
 *
 * \return Z_SUCCESS on success, or Z_ERROR_NULL_PARAMETER if \a destination
 * is NULL.
 */
int zb_encode(uint8_t *destination, uint8_t device_number, 
		uint8_t command_number, int32_t data);

/** Decodes the last 4 bytes of a 6-byte (complete) reply.
 *
 * This function wraps the last four bytes of the reply into a 32-bit integer
 * matching the architecture of the compilation target (typically the machine
 * which compiled the code, unless otherwise specified).
 *
 * This function only writes the "data" portion of a reply (the last four 
 * bytes) into \a destination. You will still need to manually read the first 
 * two bytes of your reply to get the device and command IDs.
 *
 * This function does not support the decoding of Message IDs. If your device
 * does have Message IDs enabled, the data decoded by this function will be
 * incorrect and should be considered undefined.
 *
 * \param[out] destination a pointer to a 32-bit integer to which to write the
 * reply's data.
 * \param[in] reply a pointer to an array containing a 6-byte reply message.
 *
 * \return Z_SUCCESS on success, or Z_ERROR_NULL_PARAMETER if \a destination
 * or \a reply is NULL.
 */
int zb_decode(int32_t *destination, const uint8_t *reply);

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
 * the only reliable way to retrieve any reply data is to always follow
 * calls to zb_send() with calls to zb_receive().
 *
 * \param[in] port the port to drain.
 *
 * \return Z_SUCCESS on success, or Z_ERROR_SYSTEM_ERROR on failure.
 */
int zb_drain(z_port port);

/** Change the duration zb_receive() will wait before timing out and returning
 * without having read anything.
 *
 * A Zaber device using the binary protocol will only respond once a command 
 * has been completed. Though we recommend users implement their own "wait" 
 * behaviour for commands they expect to take a long time, this function can 
 * be used to wait for such commands to complete before continuing execution.
 *
 * A value of 0 for milliseconds will block indefinitely until a reply is
 * received. Use with caution.
 *
 * On *NIX systems, the value of milliseconds will be rounded up to the
 * nearest tenth of a second (0.1s). This function returns the new timeout 
 * value to aid with compatibility between Windows and *NIX.
 *
 * \param[in] port the port whose timeout shall be changed.
 * \param[in] milliseconds the duration of the new timeout, in milliseconds.
 *
 * \return the new timeout value on success, Z_ERROR_SYSTEM_ERROR on failure.
 */
int zb_set_timeout(z_port port, int milliseconds);

/** Sets whether errors and extra info are reported to stderr. 
 *
 * Set \a value to 0 to disable all output. Additionally, you can compile this
 * library with the macro NDEBUG defined, which will disable all output and
 * skip checks to "verbose" altogether in the compiled code.
 *
 * \param[in] value whether (1) or not (0) the program should output 
 * error messages and info to stderr.
 */
void zb_set_verbose(int value);

#if defined(__cplusplus)
}
#endif /* if defined(__cplusplus) */

#endif /* if !defined(ZB_SERIAL_H) */

