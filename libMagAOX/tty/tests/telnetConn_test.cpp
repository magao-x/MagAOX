/** \file telnetConn_test.cpp
  * \brief Catch2 tests for the telnetConn class in libMagAOX/tty/telnetConn.hpp.
  *
  * No mocks are used. Each test starts a real in-process TCP server on loopback with a port
  * chosen by the operating system. See FakeServer below. A background thread accepts one
  * connection and plays the device side of the conversation. telnetConn connects to it with
  * real libtelnet, so login, write, read, and the protocol negotiation handler all run for
  * real. Error paths use real operating system behavior, for example a refused port, a
  * lowered open file limit, a reset connection, or a full socket send buffer.
  */
#include "../../../tests/catch2/catch.hpp"

#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <future>

#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "../telnetConn.hpp"

namespace libXWCTest
{
namespace telnetConnTest
{

/// A real listening socket on loopback with a port chosen by the operating system.
/** A background thread accepts exactly one connection and delivers its descriptor through
  * the connFd future. The destructor joins the thread and closes the listening socket.
  */
struct FakeServer
{
   int listenSock;
   uint16_t port;
   std::future<int> connFd;
   std::thread acceptThread;

   FakeServer()
   {
      listenSock = ::socket(AF_INET, SOCK_STREAM, 0);
      struct sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = inet_addr("127.0.0.1");
      addr.sin_port = 0;

      ::bind(listenSock, (struct sockaddr*)&addr, sizeof(addr));
      ::listen(listenSock, 1);

      socklen_t len = sizeof(addr);
      ::getsockname(listenSock, (struct sockaddr*)&addr, &len);
      port = ntohs(addr.sin_port);

      std::promise<int> prom;
      connFd = prom.get_future();
      acceptThread = std::thread([this, p = std::move(prom)]() mutable
      {
         int fd = ::accept(listenSock, nullptr, nullptr);
         p.set_value(fd);
      });
   }

   ~FakeServer()
   {
      if(acceptThread.joinable()) acceptThread.join();
      ::close(listenSock);
   }
};

/// Fills the kernel send buffer on fd with junk so later POLLOUT waits are not ready.
/** The descriptor is put in non-blocking mode for the fill and restored afterward. */
static void fillSendBuffer( int fd )
{
   int flags = fcntl(fd, F_GETFL, 0);
   fcntl(fd, F_SETFL, flags | O_NONBLOCK);

   char junk[65536];
   memset(junk, 'x', sizeof(junk));
   while( ::write(fd, junk, sizeof(junk)) > 0 ) {}

   fcntl(fd, F_SETFL, flags);
}

// connect() resolves the host and port, creates a socket, and connects. Each section makes
// one real condition and checks the matching return code.
TEST_CASE( "telnetConn::connect", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("succeeds against a real listening server")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      int rv = tc.connect("127.0.0.1", std::to_string(server.port));

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_sock > 0);
      REQUIRE(tc.m_telnet != nullptr);

      ::close(connFd);
   }

   SECTION("fails with TELNET_E_GETADDR for an unresolvable host/port combination")
   {
      MagAOX::tty::telnetConn tc;
      // An empty host with a numeric service and no AI_PASSIVE flag makes getaddrinfo() fail
      // at once with EAI_NONAME. No network access is attempted.
      int rv = tc.connect("", "12345");

      REQUIRE(rv == TELNET_E_GETADDR);
   }

   SECTION("fails with TELNET_E_SOCKET when socket() itself fails")
   {
      struct rlimit orig;
      REQUIRE( getrlimit(RLIMIT_NOFILE, &orig) == 0 );

      struct rlimit tiny;
      tiny.rlim_cur = 3; // Only stdin, stdout, and stderr fit. socket() cannot get a new descriptor.
      tiny.rlim_max = orig.rlim_max;
      REQUIRE( setrlimit(RLIMIT_NOFILE, &tiny) == 0 );

      MagAOX::tty::telnetConn tc;
      int rv = tc.connect("127.0.0.1", "12345"); // arbitrary port. socket() fails before any connection is attempted

      REQUIRE( setrlimit(RLIMIT_NOFILE, &orig) == 0 );

      REQUIRE(rv == TELNET_E_SOCKET);
   }

   SECTION("fails with TELNET_E_CONNECT when the connection is refused")
   {
      // Bind to an ephemeral port and close it at once. Nothing is listening on it, so the
      // loopback connect is refused instead of hanging.
      int probe = ::socket(AF_INET, SOCK_STREAM, 0);
      REQUIRE(probe >= 0);

      struct sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = inet_addr("127.0.0.1");
      addr.sin_port = 0;

      REQUIRE( ::bind(probe, (struct sockaddr*)&addr, sizeof(addr)) == 0 );

      socklen_t len = sizeof(addr);
      REQUIRE( ::getsockname(probe, (struct sockaddr*)&addr, &len) == 0 );
      uint16_t port = ntohs(addr.sin_port);

      REQUIRE( ::close(probe) == 0 );

      MagAOX::tty::telnetConn tc;
      int rv = tc.connect("127.0.0.1", std::to_string(port));

      REQUIRE(rv == TELNET_E_CONNECT);
   }
}

// The destructor frees the libtelnet state and closes the socket. The server side of the
// connection sees a clean end of file when it does.
TEST_CASE( "telnetConn destructor cleans up an open connection", "[libMagAOX::tty::telnetConn]" )
{
   FakeServer server;
   int connFd;

   {
      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      connFd = server.connFd.get();
   } // tc is destroyed here. The destructor calls telnet_free() and closes m_sock.

   char buff[8];
   REQUIRE( ::recv(connFd, buff, sizeof(buff), 0) == 0 ); // The peer sees a clean end of file.
   ::close(connFd);
}

// noLogin() skips the login handshake by setting the state to logged in.
TEST_CASE( "telnetConn::noLogin marks the connection as logged in", "[libMagAOX::tty::telnetConn]" )
{
   MagAOX::tty::telnetConn tc;
   REQUIRE( tc.noLogin() == TTY_E_NOERROR );
   REQUIRE( tc.m_loggedin == 5 ); // The value of TELNET_LOGGED_IN.
}

// login() waits for a username prompt, sends the username, waits for a password prompt,
// sends the password, and waits for the shell prompt. The device thread below plays the
// server side of that conversation.
TEST_CASE( "telnetConn::login walks the username/password/prompt handshake", "[libMagAOX::tty::telnetConn]" )
{
   FakeServer server;

   MagAOX::tty::telnetConn tc;
   REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

   int connFd = server.connFd.get();
   REQUIRE(connFd >= 0);

   std::thread device([connFd]()
   {
      char buff[256];

      // Prompt for username, then wait for the client to send it.
      std::string prompt1 = "Username:";
      ::send(connFd, prompt1.c_str(), prompt1.size(), 0);
      REQUIRE( ::recv(connFd, buff, sizeof(buff), 0) > 0 );

      // Prompt for password, then wait for the client to send it.
      std::string prompt2 = "Password:";
      ::send(connFd, prompt2.c_str(), prompt2.size(), 0);
      REQUIRE( ::recv(connFd, buff, sizeof(buff), 0) > 0 );

      // Finally send the shell prompt to complete the login.
      std::string prompt3 = "$> ";
      ::send(connFd, prompt3.c_str(), prompt3.size(), 0);
   });

   int rv = tc.login("theuser", "thepass");

   device.join();
   ::close(connFd);

   REQUIRE(rv == TTY_E_NOERROR);
   REQUIRE(tc.m_loggedin == 5); // The value of TELNET_LOGGED_IN.
}

// login() must cope with a server that goes away before the handshake completes. A clean
// close and a reset connection take different paths and return different codes.
TEST_CASE( "telnetConn::login handles a peer that disconnects", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("a clean close (recv returns 0) ends the loop and reports success")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);
      ::close(connFd); // The server closes without ever sending the username prompt.

      // In the current implementation an end of file here only breaks out of the poll loop.
      // login() then falls through to a success return. This does not mean the login
      // completed. m_loggedin never advances past TELNET_WAITING_USER.
      int rv = tc.login("theuser", "thepass");

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_loggedin == 0); // Still the value of TELNET_WAITING_USER.
   }

   SECTION("a reset connection (recv error) is reported as TTY_E_ERRORONREAD")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      // SO_LINGER with a zero timeout makes close() send a TCP reset instead of a normal
      // FIN. The next recv() in the client then fails with ECONNRESET instead of seeing a
      // clean end of file.
      struct linger lo{1, 0};
      ::setsockopt(connFd, SOL_SOCKET, SO_LINGER, &lo, sizeof(lo));
      ::close(connFd);

      int rv = tc.login("theuser", "thepass");

      REQUIRE(rv == TTY_E_ERRORONREAD);
   }
}

// write() sends a string through libtelnet with a timeout. The sections check the bytes
// arrive at the server, that a zero timeout fails at once, and that a full send buffer
// produces a poll timeout.
TEST_CASE( "telnetConn::write", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("sends the (telnet-CRLF-ified) buffer to the peer")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      int rv = tc.write("hello", 1000);
      REQUIRE(rv == TTY_E_NOERROR);

      char buff[256];
      ssize_t n = ::recv(connFd, buff, sizeof(buff), 0);
      REQUIRE(n == 5);
      REQUIRE( std::string(buff, n) == "hello" );

      ::close(connFd);
   }

   SECTION("returns TTY_E_TIMEOUTONWRITE when the elapsed time exceeds a zero timeout")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      int rv = tc.write("data", 0);
      REQUIRE(rv == TTY_E_TIMEOUTONWRITE);

      ::close(connFd);
   }

   SECTION("returns TTY_E_TIMEOUTONWRITEPOLL when the send buffer stays full")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      // A TCP loopback connection differs from a Unix domain socket pair. The kernel keeps
      // draining the sender's buffer into the peer's receive buffer on its own, whether or
      // not anyone calls recv(). So the peer's receive buffer must be shrunk as well.
      // Otherwise the full send buffer drains again before write() gets a chance to poll it.
      int rcvbuf = 1024;
      ::setsockopt(connFd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

      int sndbuf = 1024;
      ::setsockopt(tc.m_sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
      fillSendBuffer(tc.m_sock);

      int rv = tc.write("more data that will not fit", 50);
      REQUIRE(rv == TTY_E_TIMEOUTONWRITEPOLL);

      ::close(connFd);
   }
}

// read(eot) collects incoming data into m_strRead until the end of transmission string
// arrives. The sections cover the clear and append modes, a timeout with no data, a closed
// socket, data split across chunks, and the cleanup of control bytes and embedded NULs.
TEST_CASE( "telnetConn::read(eot)", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("accumulates m_strRead until the eot is seen, clearing by default")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      // The event handler only collects data into m_strRead once logged in. Before that it
      // treats every chunk as a possible username, password, or prompt match instead.
      tc.noLogin();
      tc.m_strRead = "stale data that should be cleared";

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      std::thread device([connFd]()
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         std::string msg = "hello world> ";
         ::send(connFd, msg.c_str(), msg.size(), 0);
      });

      int rv = tc.read("> ", 2000);

      device.join();
      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == "hello world> ");
   }

   SECTION("appends instead of clearing when clear=false")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();
      tc.m_strRead = "prefix:";

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      std::thread device([connFd]()
      {
         std::string msg = "suffix> ";
         ::send(connFd, msg.c_str(), msg.size(), 0);
      });

      int rv = tc.read("> ", 2000, false);

      device.join();
      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == "prefix:suffix> ");
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when nothing arrives")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      int rv = tc.read("> ", 50);
      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(connFd);
   }

   SECTION("returns TTY_E_ERRORONREAD when the socket is closed out from under it")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);
      ::close(connFd);

      ::close(tc.m_sock); // Leave tc.m_sock holding a stale, closed descriptor number.
      int rv = tc.read("> ", 50);
      REQUIRE(rv == TTY_E_ERRORONREAD);

      tc.m_sock = 0; // Stop the destructor from closing an unrelated descriptor that may have reused the number.
   }

   SECTION("accumulates across multiple chunks before the eot arrives")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      std::thread device([connFd]()
      {
         ::send(connFd, "hello ", 6, 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(connFd, "world> ", 7, 0);
      });

      int rv = tc.read("> ", 2000);

      device.join();
      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == "hello world> ");
   }

   SECTION("strips leading control bytes and turns embedded NULs into newlines")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      unsigned char raw[] = { 5, 5, 'h', 'e', 'l', 'l', 'o', 0, 'w', 'o', 'r', 'l', 'd', '>', ' ' };
      ::send(connFd, raw, sizeof(raw), 0);

      int rv = tc.read("> ", 2000);

      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == std::string("hello\nworld> "));
   }
}

// The single argument read(timeout) overload uses the stored m_prompt as the end of
// transmission string. The test sets a custom prompt and checks the read stops at it.
TEST_CASE( "telnetConn::read(timeout) uses m_prompt as the eot", "[libMagAOX::tty::telnetConn]" )
{
   FakeServer server;

   MagAOX::tty::telnetConn tc;
   REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
   tc.noLogin();
   tc.m_prompt = "custom% ";

   int connFd = server.connFd.get();
   REQUIRE(connFd >= 0);

   std::thread device([connFd]()
   {
      std::string msg = "response custom% ";
      ::send(connFd, msg.c_str(), msg.size(), 0);
   });

   int rv = tc.read(2000);

   device.join();
   ::close(connFd);

   REQUIRE(rv == TTY_E_NOERROR);
   REQUIRE(tc.m_strRead == "response custom% ");
}

// writeRead() sends a command and then reads the reply up to m_prompt. It can optionally
// discard the echo of the command that a real console sends back. A write failure must
// return before any read is attempted.
TEST_CASE( "telnetConn::writeRead", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("succeeds with echo swallowing")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();
      tc.m_prompt = "> "; // writeRead() always reads up to m_prompt. It takes no eot argument.

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      std::string strWrite = "cmd";

      std::thread device([connFd, strWrite]()
      {
         char buff[256];
         ssize_t n = ::recv(connFd, buff, sizeof(buff), 0);
         REQUIRE(n == (ssize_t) strWrite.size());

         // This swallow loop differs from the one in ttyIOUtils::ttyWriteRead. Every chunk
         // goes straight into m_strRead through telnet_recv() and the event handler. Once
         // m_strRead holds more than strWrite.size() characters, the loop erases that many
         // from the front. So the device sends the echo first, then waits, then sends the
         // whole reply in one piece. The delay makes the reply land in a separate read().
         // After the erase exactly the reply is left.
         ::send(connFd, strWrite.c_str(), strWrite.size(), 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         std::string reply = "result> ";
         ::send(connFd, reply.c_str(), reply.size(), 0);
      });

      int rv = tc.writeRead(strWrite, true, 1000, 1000);

      device.join();
      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == "result> ");
   }

   SECTION("succeeds without echo swallowing")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );
      tc.noLogin();
      tc.m_prompt = "> ";

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      std::thread device([connFd]()
      {
         char buff[256];
         ::recv(connFd, buff, sizeof(buff), 0);

         std::string reply = "result> ";
         ::send(connFd, reply.c_str(), reply.size(), 0);
      });

      int rv = tc.writeRead("cmd", false, 1000, 1000);

      device.join();
      ::close(connFd);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(tc.m_strRead == "result> ");
   }

   SECTION("propagates a write timeout without attempting the read")
   {
      FakeServer server;

      MagAOX::tty::telnetConn tc;
      REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

      int connFd = server.connFd.get();
      REQUIRE(connFd >= 0);

      int rv = tc.writeRead("cmd", false, 0, 1000);
      REQUIRE(rv == TTY_E_TIMEOUTONWRITE);

      ::close(connFd);
   }
}

// The static send() helper writes raw bytes to a descriptor. It is used by libtelnet's
// send callback. A socket pair checks every byte arrives. A closed descriptor must fail.
TEST_CASE( "telnetConn::send (static)", "[libMagAOX::tty::telnetConn]" )
{
   SECTION("sends all the requested bytes")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      int rv = MagAOX::tty::telnetConn::send(sp[0], "abc", 3);
      REQUIRE(rv == TTY_E_NOERROR);

      char buff[8];
      REQUIRE( ::recv(sp[1], buff, sizeof(buff), 0) == 3 );

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_ERRORONWRITE when the descriptor is closed")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      int rv = MagAOX::tty::telnetConn::send(fd, "abc", 3);
      REQUIRE(rv == TTY_E_ERRORONWRITE);
   }
}

// The server sends raw telnet option negotiation bytes. This drives the WILL, WONT, DO,
// DONT, and terminal type branches of the libtelnet event handler. The checks are that
// nothing crashes and that the read times out as expected. Branch coverage is confirmed
// through the coverage report, not through assertions.
TEST_CASE( "telnetConn's libtelnet event handler reacts to raw protocol negotiation", "[libMagAOX::tty::telnetConn]" )
{
   FakeServer server;

   MagAOX::tty::telnetConn tc;
   REQUIRE( tc.connect("127.0.0.1", std::to_string(server.port)) == TTY_E_NOERROR );

   int connFd = server.connFd.get();
   REQUIRE(connFd >= 0);

   // The terminal type reply uses getenv("TERM"). Set it so the handler cannot dereference
   // a null pointer, whatever environment this test runs in.
   setenv("TERM", "xterm", 1);

   std::thread device([connFd]()
   {
      // libtelnet only fires a negotiate event when the option's RFC 1143 state changes.
      // So each WONT or DONT needs an earlier WILL or DO on the same option to revoke.
      //   WILL COMPRESS2 sets him to YES. This fires EV_WILL because telopts lists the option as WONT and DO.
      //   WONT COMPRESS2 revokes it. This fires EV_WONT.
      //   DO TTYPE sets us to YES. This fires EV_DO because telopts lists the option as WILL and DONT.
      //   DONT TTYPE revokes it. This fires EV_DONT.
      unsigned char bytes[] = {
         255, 251, 86,  // IAC WILL COMPRESS2
         255, 252, 86,  // IAC WONT COMPRESS2
         255, 253, 24,  // IAC DO TTYPE
         255, 254, 24,  // IAC DONT TTYPE
         255, 250, 24, 1, 255, 240, // IAC SB TTYPE SEND IAC SE
      };
      ::send(connFd, bytes, sizeof(bytes), 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
   });

   // Drive telnet_recv() through a real read. The eot never arrives, so the read times out.
   int rv = tc.read("this eot will not arrive", 100);
   REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

   device.join();
   ::close(connFd);
}

} // namespace telnetConnTest
} // namespace libXWCTest
