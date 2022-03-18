/** \file telnetConn.hpp
  * \brief Managing a connection to a telnet device.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup tty_files
  * History:
  * - 2018-08-24 created by JRM
  */
#ifndef telnet_telnetConn_hpp
#define telnet_telnetConn_hpp


/* Much of the code in this file was taken from telnet-client.c in
 * libtelnet (https://github.com/seanmiddleditch/libtelnet), with modifications for our needs.
 *
 * That code was placed in the public domain:
 *
 * libtelnet - TELNET protocol handling library
 *
 * Sean Middleditch
 * sean@sourcemud.org
 *
 * The author or authors of [the libtelnet] code dedicate any and all copyright interest
 * in [the libtelnet] to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and successors. We
 * intend this dedication to be an overt act of relinquishment in perpetuity of
 * all present and future rights to this code under copyright law.
 */


#include <mx/sys/timeUtils.hpp>

#include "../../libs/libtelnet/libtelnet.h"
#include "ttyErrors.hpp"
#include "ttyIOUtils.hpp"

namespace MagAOX
{
namespace tty
{

#ifndef TELNET_BUFFSIZE
   #define TELNET_BUFFSIZE (1024)
#endif

/// libtelnet option table.
/** \ingroup tty
  */
static const telnet_telopt_t telopts[] = {
            { TELNET_TELOPT_ECHO,       TELNET_WONT, TELNET_DO   },
            { TELNET_TELOPT_TTYPE,      TELNET_WILL, TELNET_DONT },
            { TELNET_TELOPT_COMPRESS2,  TELNET_WONT, TELNET_DO   },
            { TELNET_TELOPT_MSSP,       TELNET_WONT, TELNET_DO   },
            { -1, 0, 0 }                    };

#define TELNET_WAITING_USER (0)
#define TELNET_GOT_USER (1)
#define TELNET_WAITING_PASS (2)
#define TELNET_GOT_PASS (3)
#define TELNET_WAITING_PROMPT (4)
#define TELNET_LOGGED_IN (5)

/// A Telnet connection manager, wrapping \p libtelnet.
/**
  * Establishes the connection to the server, and initializes the
  * \p libtelnet structure, including registering the event handler callback.
  *
  * Errors encountered during telnet event handling are indicated by an internal flag,
  * which must be checked each time a libtelnet function is called.  If it is nonzero an
  * error has occurred.
  *
  * Responses from the server are accumulated in the \p m_strRead member.  It is typically
  * cleared before reading, but this can be suppressed when desired.
  *
  * Because of the way event handling is managed, and the class-global error and response accumulation
  * this is not thread-safe.  Any calls to this class methods should be mutex-ed.
  *
  * \ingroup tty
  */
struct telnetConn
{
   int m_sock {0}; ///< The socket file descriptor.

   telnet_t * m_telnet {nullptr}; ///< libtelnet telnet_t structure

   ///The device's username entry prompt, used for managing login.
   std::string m_usernamePrompt {"Username:"};

   ///The device's password entry prompt, used for managing login.
   std::string m_passwordPrompt {"Password:"};

   std::string m_prompt {"$> "}; ///< The device's prompt, used for detecting end of transmission.

   ///Flag denoting the login state.
   /** Used to manage different behaviors in the libtelnet event handler.
     *
     * - TELNET_WAITING_USER: waiting on m_usernamePrompt
     * - TELNET_GOT_USER: got m_usernamePrompt
     * - TELNET_WAITING_PASS: waiting on m_passwordPrompt
     * - TELNET_GOT_PASS: got m_passwordPrompt
     * - TELNET_WAITING_PROMPT: waiting on m_prompt
     * - TELNET_LOGGED_IN: logged in
     */
   int m_loggedin {0};

   /// Used to indicate an error occurred in the event handler callback.
   int m_EHError {0};

   /// The accumulated string read from the device.
   /** This needs to be clear()-ed when expecting a new response to start.
     * \warning This makes telnetConn NOT threadsafe.
     */
   std::string m_strRead;

   /// D'tor, conducts connection cleanup.
   ~telnetConn();

   /// Connect to the device
   int connect( const std::string & host, ///< [in] The host specification (i.p. address)
                const std::string & port  ///< [in] the port on the host.
              );

   /// Manage the login process on this device.
   int login( const std::string & username, /// [in] The username
              const std::string & password  /// [in] The password.
            );

   /// Set flags as if we're logged in, used when device doesn't require it.
   int noLogin();

   /// Write to a telnet connection
   /**
     *
     * \returns TTY_E_NOERROR on success
     * \returns TTY_E_TIMEOUTONWRITEPOLL if the poll times out.
     * \returns TTY_E_ERRORONWRITEPOLL if an error is returned by poll.
     * \returns TTY_E_TIMEOUTONWRITE if a timeout occurs during the write.
     * \returns TTY_E_ERRORONWRITE if an error occurs writing to the file.
    */
   int write( const std::string & buffWrite, ///< [in] The characters to write to the telnet.
              int timeoutWrite               ///< [in] The timeout in milliseconds.
            );

   /// Read from a telnet connection, until end-of-transmission string is read.
   /**
     * \returns TTY_E_NOERROR on success
     * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
     * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
     * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
     * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
     */
   int read( const std::string & eot, ///< [in] the end-of-transmission indicator
             int timeoutRead, ///< [in] The timeout in milliseconds.
             bool clear=true  ///< [in] [optional] whether or not to clear the strRead buffer
           );

   /// Read from a telnet connection, until m_prompt is read.
   /**
     * \returns TTY_E_NOERROR on success
     * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
     * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
     * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
     * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
     */
   int read( int timeoutRead, ///< [in] The timeout in milliseconds.
             bool clear=true  ///< [in] [optional] whether or not to clear the strRead buffer
           );

   /// Write to a telnet connection, then get the reply.
   /** The read is conducted until the m_prompt string is received.
     * Echo characters are swallowed if desired.
     *
     * \returns TTY_E_NOERROR on success
     * \returns TTY_E_TIMEOUTONWRITEPOLL if the poll times out.
     * \returns TTY_E_ERRORONWRITEPOLL if an error is returned by poll.
     * \returns TTY_E_TIMEOUTONWRITE if a timeout occurs during the write.
     * \returns TTY_E_ERRORONWRITE if an error occurs writing to the file.
     * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
     * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
     * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
     * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
     */
   int writeRead( const std::string & strWrite, ///< [in] The characters to write to the telnet.
                  bool swallowEcho,             ///< [in] If true, strWrite.size() characters are read after the write
                  int timeoutWrite,             ///< [in] The write timeout in milliseconds.
                  int timeoutRead               ///< [in] The read timeout in milliseconds.
                );

   /// Internal send for use by event_handler.
   static int send( int sock,
                    const char *buffer,
                    size_t size
                  );

   /// Event handler callback for libtelnet processing.
   /** Resets the internal m_EHError value to TTY_E_NOERROR on entry.
     * Will set it to an error flag if an error is encountered, so this
     * flag should be checked after any call to a libtelnet function.
     * \warning this makes telnetConn not thread safe
     */
   static void event_handler( telnet_t *telnet,
                              telnet_event_t *ev,
                              void *user_data
                            );
};

} //namespace tty
} //namespace MagAOX

#endif //telnet_telnetConn_hpp
