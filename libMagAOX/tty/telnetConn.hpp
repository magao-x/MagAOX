/** \file telnetConn.hpp
  * \brief Managing a connection to a telnet device.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
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


#include <mx/timeUtils.hpp>

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
  */ 
struct telnetConn
{
   int m_sock {0}; ///< The socket file descriptor.
   
   telnet_t * m_telnet {nullptr}; ///< libtelnet telnet_t structure

   ///The device's username entry prompt, used for managing login.
   std::string m_usernamePrompt {"Username:"};
   
   ///The device's password entry prompt, used for managing login.
   std::string m_passwordPrompt {"Password:"};
   
   std::string m_prompt {"$>"}; ///< The device's prompt, used for detecting end of transmission.
   
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

   /// Read from a telnet connection, until the m_prompt is read.
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

inline
telnetConn::~telnetConn()
{   
   /* clean up */
   if(m_telnet) telnet_free(m_telnet);
   if(m_sock) close(m_sock);
}
   
inline
int telnetConn::connect( const std::string & host,
                         const std::string & port
                       )
{
   int rs;
   
   struct sockaddr_in addr;
   struct addrinfo *ai;
   struct addrinfo hints;
    
   /* look up server host */
   memset(&hints, 0, sizeof(hints));
   hints.ai_family = AF_UNSPEC;
   hints.ai_socktype = SOCK_STREAM;
   if ((rs = getaddrinfo(host.c_str(), port.c_str(), &hints, &ai)) != 0) 
   {
      fprintf(stderr, "getaddrinfo() failed for %s: %s\n", host.c_str(),
      gai_strerror(rs));
      return TELNET_E_GETADDR;
   }
   
   /* create server m_socket */
   if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) 
   {
      fprintf(stderr, "socket() failed: %s\n", strerror(errno));
      return TELNET_E_SOCKET;
   }

   /* bind server socket */
   memset(&addr, 0, sizeof(addr));
   addr.sin_family = AF_INET;
   if (bind(m_sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) 
   {
      fprintf(stderr, "bind() failed: %s\n", strerror(errno));
      return TELNET_E_BIND;
   }

   /* connect */
   if (::connect(m_sock, ai->ai_addr, ai->ai_addrlen) == -1) 
   {
      fprintf(stderr, "connect() failed: %s\n", strerror(errno));
      return TELNET_E_CONNECT;
   }

   /* free address lookup info */
   freeaddrinfo(ai);
   
   /* initialize the telnet box */
   m_telnet = telnet_init(telopts, telnetConn::event_handler, 0, this);
   
   if(m_telnet == nullptr)
   {
      return TELNET_E_TELNETINIT;
   }
   
   return TTY_E_NOERROR;
}

inline
int telnetConn::login( const std::string & username,
                       const std::string & password
                     )
{
   char buffer[512];
   int rs;

   struct pollfd pfd[1];

   /* initialize poll descriptors */
   memset(pfd, 0, sizeof(pfd));
   pfd[0].fd = m_sock;
   pfd[0].events = POLLIN;

   //Loop while waiting on the login process to complete.
   while (poll(pfd, 1, -1) != -1) 
   {
      /* read from client */
      if (pfd[0].revents & POLLIN) 
      {
         if ((rs = recv(m_sock, buffer, sizeof(buffer), 0)) > 0) 
         {
            telnet_recv(m_telnet, buffer, rs);
            if(m_EHError != TTY_E_NOERROR) return m_EHError;
         } 
         else if (rs == 0) 
         {
            break;
         } 
         else 
         {
            fprintf(stderr, "recv(client) failed: %s\n",
            strerror(errno));
            return TTY_E_ERRORONREAD;
         }
      }
      
      if(m_loggedin == TELNET_GOT_USER)
      {
         int rv = write(username + "\n", 1000);
         if(rv != TTY_E_NOERROR) return rv;
         
         m_loggedin = TELNET_WAITING_PASS;
      }
      
      if(m_loggedin == TELNET_GOT_PASS)
      {
         int rv = write(password + "\n", 1000);
         if(rv != TTY_E_NOERROR) return rv;
         
         m_loggedin = TELNET_WAITING_PROMPT;
      }
      
      if(m_loggedin == TELNET_LOGGED_IN)
      {
         break;
      }
   }   
}

inline
int telnetConn::write( const std::string & buffWrite,
                        int timeoutWrite  
                     )
{
   double t0;
   struct pollfd pfd;

   errno = 0;
   pfd.fd = m_sock;
   pfd.events = POLLOUT;

   t0 = mx::get_curr_time();

   size_t totWritten = 0;
   while( totWritten < buffWrite.size())
   {
      int timeoutCurrent = timeoutWrite - (mx::get_curr_time()-t0)*1000;
      if(timeoutCurrent < 0) return TTY_E_TIMEOUTONWRITE;

      int rv = poll( &pfd, 1, timeoutCurrent);
      if( rv == 0 ) return TTY_E_TIMEOUTONWRITEPOLL;
      else if( rv < 0 ) return TTY_E_ERRORONWRITEPOLL;
   
      /* if we got a CR or LF, replace with CRLF
       * NOTE that usually you'd get a CR in UNIX, but in raw
       * mode we get LF instead (not sure why)
       */
      if (buffWrite[totWritten] == '\r' || buffWrite[totWritten] == '\n') 
      {
         static char crlf[] = { '\r', '\n' };

         telnet_send(m_telnet, crlf, 2);
         if(m_EHError != TTY_E_NOERROR) return m_EHError;
         totWritten+=1; //though we wrote 2
      } 
      else 
      {
         telnet_send(m_telnet, &buffWrite[totWritten], 1);
         if(m_EHError != TTY_E_NOERROR) return m_EHError;
         totWritten+=1;
      }
   
      #ifdef TELNET_DEBUG
      std::cerr << "Wrote " << totWritten << " chars of " << buffWrite.size() << "\n";
      #endif


      if( ( mx::get_curr_time()-t0)*1000 > timeoutWrite ) return TTY_E_TIMEOUTONWRITE;
   }

   return TTY_E_NOERROR;
}

inline
int telnetConn::read( int timeoutRead,
                      bool clear
                    )
{
   int rv;
   int timeoutCurrent;
   double t0;

   struct pollfd pfd;

   errno = 0;

   pfd.fd = m_sock;
   pfd.events = POLLIN;

   char buffRead[TELNET_BUFFSIZE];

   //Start timeout clock for reading.
   t0 = mx::get_curr_time();
   timeoutCurrent = timeoutRead;

   //Now read the response up to the eot.
   if(clear) m_strRead.clear();

   rv = poll( &pfd, 1, timeoutCurrent);
   if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
   if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

   rv = ::read(m_sock, buffRead, TELNET_BUFFSIZE);
   if( rv < 0 ) return TTY_E_ERRORONREAD;

   telnet_recv(m_telnet, buffRead, rv);
   if(m_EHError != TTY_E_NOERROR) return m_EHError;

   while( !isEndOfTrans(m_strRead, m_prompt) )
   {
      timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
      if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

      rv = poll( &pfd, 1, timeoutCurrent);
      if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
      if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

      rv = ::read(m_sock, buffRead, TELNET_BUFFSIZE);
      if( rv < 0 ) return TTY_E_ERRORONREAD;
      buffRead[rv] ='\0';

      telnet_recv(m_telnet, buffRead, rv);
      if(m_EHError != TTY_E_NOERROR) return m_EHError;
   
      #ifdef TELNET_DEBUG
      std::cerr << "telnetRead: read " << rv << " bytes. buffRead=" << buffRead << "\n";
      #endif
   }


   return TTY_E_NOERROR;


}

inline
int telnetConn::writeRead( const std::string & strWrite, 
                           bool swallowEcho,             
                           int timeoutWrite,                                        
                           int timeoutRead 
                         )
{
   m_strRead.clear();

   int rv;

   //Write First
   rv = write( strWrite, timeoutWrite);
   if(rv != TTY_E_NOERROR) return rv;

   //Now read response from console
   int timeoutCurrent;
   double t0;

   struct pollfd pfd;
   pfd.fd = m_sock;
   pfd.events = POLLIN;


   //Start timeout clock for reading.
   t0 = mx::get_curr_time();;

   if(swallowEcho)
   {
      char buffRead[TELNET_BUFFSIZE];

      //First swallow the echo.
      while( m_strRead.size() <= strWrite.size() )
      {
         timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
         if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

         rv = poll( &pfd, 1, timeoutCurrent);
         if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
         if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

         rv = ::read(m_sock, buffRead, TELNET_BUFFSIZE);
         if( rv < 0 ) return TTY_E_ERRORONREAD;

         telnet_recv(m_telnet, buffRead, rv);
         if(m_EHError != TTY_E_NOERROR) return m_EHError;
      }
      
      m_strRead.erase(0, strWrite.size());
   }

   timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
   if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

   //Now read the response up to the eot.
   return read(timeoutCurrent, false);
}

inline
int telnetConn::send(int sock, const char *buffer, size_t size) 
{
   /* send data */
   while (size > 0) 
   {
      int rs;

      if ((rs = ::send(sock, buffer, size, 0)) == -1) 
      {
         fprintf(stderr, "send() failed: %s\n", strerror(errno));
         return TTY_E_ERRORONWRITE;
      } 
      else if (rs == 0) 
      {
         fprintf(stderr, "send() unexpectedly returned 0\n");
         TTY_E_ERRORONWRITE;
      }
      /* update pointer and size to see if we've got more to send */
      buffer += rs;
      size -= rs;
   }
   
   return TTY_E_NOERROR;
}

inline
void telnetConn::event_handler( telnet_t *telnet, 
                                telnet_event_t *ev, 
                                void *user_data
                              ) 
{
   telnetConn * cs = static_cast<telnetConn*>(user_data); 
   int sock = cs->m_sock;

   //Always reset the error at beginning.
   cs->m_EHError = 0;
   
   switch (ev->type) 
   {
      /* data received */
      case TELNET_EV_DATA:
      {
         //First we remove the various control chars from the front.
         if(ev->data.size == 0) break;
         
         char * buf = const_cast<char *>(ev->data.buffer);
         buf[ev->data.size] = 0;
         for(int i=0; i<ev->data.size; ++i)
         {
            if(ev->data.buffer[i] < 20)
            {
               ++buf;
               continue;
            }
            break;
         }
         
         //Now make it a string so we can make use of it.
         std::string sbuf(buf);
   
         if(sbuf.size() == 0) break;
         
         if(cs->m_loggedin < TELNET_LOGGED_IN) //we aren't logged in yet
         {
            if(cs->m_loggedin == TELNET_WAITING_USER)
            {
               if(sbuf.find(cs->m_usernamePrompt) != std::string::npos)
               {
                  cs->m_loggedin = TELNET_GOT_USER;
               }
               break;
            }
         
            if(cs->m_loggedin == TELNET_WAITING_PASS)
            {
               if( sbuf.find(cs->m_passwordPrompt) != std::string::npos)
               {
                  cs->m_loggedin = TELNET_GOT_PASS;
               }
               break;
            }
         
            if(cs->m_loggedin == TELNET_WAITING_PROMPT)
            {
               if( sbuf.find(cs->m_prompt) != std::string::npos)
               {
                  cs->m_loggedin = TELNET_LOGGED_IN;
               }
               break;
            }
         }
         
         //Always append
         cs->m_strRead += sbuf;
         break;
      }
      /* data must be sent */
      case TELNET_EV_SEND:
      {
         send(sock, ev->data.buffer, ev->data.size);
         break;
      }
      /* request to enable remote feature (or receipt) */
      case TELNET_EV_WILL:
      {
         /* we'll agree to turn off our echo if server wants us to stop */
         //if (ev->neg.telopt == TELNET_TELOPT_ECHO) do_echo = 0;
         break;
      }
      /* notification of disabling remote feature (or receipt) */
      case TELNET_EV_WONT:
      {
         break;
      }
      /* request to enable local feature (or receipt) */
      case TELNET_EV_DO:
      {
         break;
      }
      /* demand to disable local feature (or receipt) */
      case TELNET_EV_DONT:
      {
         break;
      }
      /* respond to TTYPE commands */
      case TELNET_EV_TTYPE:
      {
         /* respond with our terminal type, if requested */
         if (ev->ttype.cmd == TELNET_TTYPE_SEND) 
         {
            telnet_ttype_is(telnet, getenv("TERM"));
         }
         break;
      }
      /* respond to particular subnegotiations */
      case TELNET_EV_SUBNEGOTIATION:
      {
         break;
      }
      /* error */
      case TELNET_EV_ERROR:
      {
         fprintf(stderr, "ERROR: %s\n", ev->error.msg);
         cs->m_EHError = TELNET_E_EHERROR;
         break;
      }
      default:
      {
         /* ignore */
         break;
      }
   }
}

} //namespace tty 
} //namespace MagAOX

#endif //telnet_telnetConn_hpp
