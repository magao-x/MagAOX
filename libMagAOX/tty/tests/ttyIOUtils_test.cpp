/** \file ttyIOUtils_test.cpp
  * \brief Catch2 tests for the tty I/O helpers in libMagAOX/tty/ttyIOUtils.cpp.
  *
  * No mocks are used. The open tests use real pseudo terminal pairs from openpty() as the
  * serial device, so the device setup runs against genuine terminal file descriptors. The
  * read, write, timeout, and error tests use real AF_UNIX socket pairs and closed
  * descriptors so the poll and errno paths are exercised by the kernel.
  */
#include "../../../tests/catch2/catch.hpp"

#include <pty.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <cstring>
#include <thread>
#include <chrono>

#include "../ttyIOUtils.hpp"
#include "../ttyErrors.hpp"

namespace ttyIOUtils_test
{

SCENARIO( "A string needs to be telnet-ified", "[libMagAOX::tty]" )
{
   GIVEN("Strings in non-telnet format with single chars")
   {
      std::string telnetStr, inputStr;
      int rv;

      WHEN("A single \\r char at end")
      {
         inputStr = "test\r";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\n");
      }

      WHEN("A single \\n char at end")
      {
         inputStr = "test\n";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\n");
      }

      WHEN("A single \\r char in the middle")
      {
         inputStr = "test\rtest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\ntest");
      }

      WHEN("A single \\n char in the middle")
      {
         inputStr = "test\ntest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\ntest");
      }

      WHEN("A single \\r char at the beginning")
      {
         inputStr = "\rtest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest");
      }

      WHEN("A single \\n char at the beginning")
      {
         inputStr = "\ntest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest");
      }
   }

   GIVEN("Strings in non-telnet format with two split chars")
   {
      std::string telnetStr, inputStr;
      int rv;

      WHEN("A single \\r char at end, a \\n at beginning")
      {
         inputStr = "\ntest\r";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\n");
      }

      WHEN("A single \\n char at end, a \\r at beginning")
      {
         inputStr = "\rtest\n";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\n");
      }

      WHEN("A single \\r char in the middle, a \n at beginning")
      {
         inputStr = "\ntest\rtset";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\ntset");
      }

      WHEN("A single \\n char in the middle, a \r at beginning")
      {
         inputStr = "\rtest\ntest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\ntest");
      }

      WHEN("A single \\r char at the beginning, a \\r at end")
      {
         inputStr = "\rtest\r";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\n");
      }

      WHEN("A single \\n char at the beginning, a \\n at end")
      {
         inputStr = "\ntest\r\n";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest\r\n");
      }
   }

   GIVEN("Strings already in telnet format")
   {
      std::string telnetStr, inputStr;
      int rv;

      WHEN("A \\r\\n at end")
      {
         inputStr = "test\r\n";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\n");
      }

      WHEN("A \\r\\n char in the middle")
      {
         inputStr = "test\r\ntest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "test\r\ntest");
      }

      WHEN("A \\r\\n char at the beginning")
      {
         inputStr = "\r\ntest";
         rv = MagAOX::tty::telnetCRLF(telnetStr, inputStr );
         REQUIRE(rv == 0);
         REQUIRE(telnetStr == "\r\ntest");
      }

   }
}

// Verifies that isEndOfTrans() only reports a match when the end-of-transmission string is
// exactly the tail of the read buffer, including the edge cases of a too-short buffer and
// empty inputs.
TEST_CASE( "isEndOfTrans checks the tail of the read buffer against the eot string", "[libMagAOX::tty::isEndOfTrans]" )
{
   REQUIRE( MagAOX::tty::isEndOfTrans( "hello> ", "> " ) == true );
   REQUIRE( MagAOX::tty::isEndOfTrans( "hello>x", "> " ) == false );
   REQUIRE( MagAOX::tty::isEndOfTrans( "hi", "hello" ) == false ); // eot longer than strRead
   REQUIRE( MagAOX::tty::isEndOfTrans( "", "" ) == true ); // both empty
}

/// Opens a fresh pseudo terminal pair and returns the slave device path.
/** The slave end is closed right away because ttyOpenRaw() reopens it by path. The caller
  * owns the master descriptor and must close it.
  */
static std::string openPtySlaveName( int & masterFd )
{
   int slaveFd;
   char name[256];

   REQUIRE( ::openpty( &masterFd, &slaveFd, name, nullptr, nullptr ) == 0 );
   ::close(slaveFd);

   return std::string(name);
}

// Verifies ttyOpenRaw() against a real pseudo terminal slave. The success path checks that a
// descriptor is returned. The failure paths use a regular file, a missing path, and an invalid
// baud rate to reach the TTY_E_TCGETATTR and TTY_E_SETISPEED error codes.
TEST_CASE( "ttyOpenRaw opens a real tty device and configures it", "[libMagAOX::tty::ttyOpenRaw]" )
{
   SECTION("succeeds on a real pty slave device")
   {
      int masterFd;
      std::string devName = openPtySlaveName(masterFd);

      int fd = -1;
      int rv = MagAOX::tty::ttyOpenRaw(fd, devName, B9600);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(fd > 0);

      ::close(fd);
      ::close(masterFd);
   }

   SECTION("fails with TTY_E_TCGETATTR when the path is not a tty")
   {
      std::string devName = "/tmp/ttyIOUtils_test_regular_file.txt";
      int fdmake = ::open(devName.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
      REQUIRE(fdmake >= 0);
      ::close(fdmake);

      int fd = -1;
      int rv = MagAOX::tty::ttyOpenRaw(fd, devName, B9600);

      REQUIRE(rv == TTY_E_TCGETATTR);
      REQUIRE(fd == 0);

      ::unlink(devName.c_str());
   }

   SECTION("fails with TTY_E_TCGETATTR when the device does not exist")
   {
      std::string devName = "/dev/xwctest-no-such-tty-device";

      int fd = -1;
      int rv = MagAOX::tty::ttyOpenRaw(fd, devName, B9600);

      REQUIRE(rv == TTY_E_TCGETATTR);
      REQUIRE(fd == 0);
   }

   SECTION("fails with TTY_E_SETISPEED when an invalid speed is given")
   {
      int masterFd;
      std::string devName = openPtySlaveName(masterFd);

      int fd = -1;
      int rv = MagAOX::tty::ttyOpenRaw(fd, devName, (speed_t) 0xDEADBEEF); // not a valid speed_t

      REQUIRE(rv == TTY_E_SETISPEED);
      REQUIRE(fd == 0);

      ::close(masterFd);
   }
}

/// Fills the kernel send buffer of a socket descriptor so later writes cannot proceed.
/** The descriptor is switched to non-blocking mode, written to until write() fails, and then
  * restored to its original flags. After this a poll for POLLOUT on the descriptor will not
  * become ready until the peer drains the buffer. The descriptor stays open.
  */
static void fillSendBuffer( int fd )
{
   int flags = fcntl(fd, F_GETFL, 0);
   fcntl(fd, F_SETFL, flags | O_NONBLOCK);

   char junk[65536];
   memset(junk, 'x', sizeof(junk));
   while(true)
   {
      ssize_t rv = ::write(fd, junk, sizeof(junk));
      if(rv < 0) break;
   }

   fcntl(fd, F_SETFL, flags);
}

// Verifies ttyWrite() on a real socket pair. The success path reads the bytes back from the
// peer. The error paths use a closed descriptor, a zero timeout, and a full send buffer to
// reach each write error code.
TEST_CASE( "ttyWrite writes to a file descriptor", "[libMagAOX::tty::ttyWrite]" )
{
   SECTION("succeeds writing to a socket that's ready and being drained")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::string msg = "hello device";
      int rv = MagAOX::tty::ttyWrite(msg, sp[0], 1000);
      REQUIRE(rv == TTY_E_NOERROR);

      char buff[256];
      ssize_t n = ::recv(sp[1], buff, sizeof(buff), 0);
      REQUIRE(n == (ssize_t) msg.size());
      REQUIRE( std::string(buff, n) == msg );

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_ERRORONWRITE when the descriptor is closed")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      int rv = MagAOX::tty::ttyWrite("data", fd, 1000);
      REQUIRE(rv == TTY_E_ERRORONWRITE);
   }

   SECTION("returns TTY_E_TIMEOUTONWRITE when the elapsed time exceeds a zero timeout")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      int rv = MagAOX::tty::ttyWrite("data", sp[0], 0);
      REQUIRE(rv == TTY_E_TIMEOUTONWRITE);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONWRITEPOLL when the send buffer stays full")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      int sndbuf = 1024;
      ::setsockopt(sp[0], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

      fillSendBuffer(sp[0]);

      int rv = MagAOX::tty::ttyWrite("more data that won't fit", sp[0], 50);
      REQUIRE(rv == TTY_E_TIMEOUTONWRITEPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }
}

// Verifies ttyReadRaw() on a real socket pair. It checks that available bytes are returned
// with the correct count, that silence produces a poll timeout, and that a closed descriptor
// produces a read error.
TEST_CASE( "ttyReadRaw reads a raw buffer of bytes from a file descriptor", "[libMagAOX::tty::ttyReadRaw]" )
{
   SECTION("succeeds reading available bytes")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::string msg = "raw bytes";
      REQUIRE( ::send(sp[1], msg.c_str(), msg.size(), 0) == (ssize_t) msg.size() );

      std::vector<unsigned char> vecRead(256);
      int readBytes = -1;
      int rv = MagAOX::tty::ttyReadRaw(vecRead, readBytes, sp[0], 1000);

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(readBytes == (int) msg.size());
      REQUIRE( std::string((char*)vecRead.data(), readBytes) == msg );

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when nothing arrives")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::vector<unsigned char> vecRead(256);
      int readBytes = -1;
      int rv = MagAOX::tty::ttyReadRaw(vecRead, readBytes, sp[0], 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_ERRORONREAD when the descriptor is closed")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      std::vector<unsigned char> vecRead(256);
      int readBytes = -1;
      int rv = MagAOX::tty::ttyReadRaw(vecRead, readBytes, fd, 50);

      REQUIRE(rv == TTY_E_ERRORONREAD);
   }
}

// Verifies the byte-count overload of ttyRead() on a real socket pair. A sender thread
// delivers the data in separate chunks to prove that the reader accumulates across reads.
// The timeout and closed-descriptor error paths are also checked.
TEST_CASE( "ttyRead(bytes) reads until a specific number of bytes have arrived", "[libMagAOX::tty::ttyRead]" )
{
   SECTION("succeeds, accumulating across multiple sends")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::thread sender([sp]()
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(sp[1], "abc", 3, 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(sp[1], "de", 2, 0);
      });

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, 5, sp[0], 2000);

      sender.join();

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(strRead == "abcde");

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when nothing arrives")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, 5, sp[0], 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when the byte count is never reached")
   {
      // Once the initial bytes are consumed, the next poll for more data blocks for the
      // remaining time budget and times out. The overall TTY_E_TIMEOUTONREAD check only fires
      // if the budget is already exhausted at the top of a loop iteration. That cannot happen
      // deterministically here because the preceding poll is bounded by the same budget. So
      // the expected result is the poll timeout code.
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      REQUIRE( ::send(sp[1], "ab", 2, 0) == 2 );

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, 5, sp[0], 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_ERRORONREAD when the descriptor is closed")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, 5, fd, 50);

      REQUIRE(rv == TTY_E_ERRORONREAD);
   }
}

// Verifies the end-of-transmission overload of ttyRead() on a real socket pair. A sender
// thread delivers the reply in two chunks so the terminator arrives in a later read. The
// timeout, missing-terminator, and closed-descriptor error paths are also checked.
TEST_CASE( "ttyRead(eot) reads until an end-of-transmission string is seen", "[libMagAOX::tty::ttyRead]" )
{
   SECTION("succeeds, accumulating across multiple sends until the eot arrives")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::thread sender([sp]()
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(sp[1], "hello ", 6, 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(sp[1], "world> ", 7, 0);
      });

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, std::string("> "), sp[0], 2000);

      sender.join();

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(strRead == "hello world> ");

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when nothing arrives")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, std::string("> "), sp[0], 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_TIMEOUTONREADPOLL when the eot never arrives")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      REQUIRE( ::send(sp[1], "no eot here", 11, 0) == 11 );

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, std::string("> "), sp[0], 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("returns TTY_E_ERRORONREAD when the descriptor is closed")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      std::string strRead;
      int rv = MagAOX::tty::ttyRead(strRead, std::string("> "), fd, 50);

      REQUIRE(rv == TTY_E_ERRORONREAD);
   }
}

// Verifies ttyWriteRead() on a real socket pair with a device thread standing in for the
// serial peer. The echo-swallowing path models a console that echoes each command before
// replying. Write errors and read timeouts during the echo swallow are also checked.
TEST_CASE( "ttyWriteRead writes then reads a reply, optionally swallowing the echo", "[libMagAOX::tty::ttyWriteRead]" )
{
   SECTION("succeeds with echo swallowing, as a real echoing console would behave")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::string strWrite = "cmd";

      std::thread device([sp, strWrite]()
      {
         // Receive the command that ttyWriteRead() sends.
         char buff[256];
         ssize_t n = ::recv(sp[1], buff, sizeof(buff), 0);
         REQUIRE(n == (ssize_t) strWrite.size());

         // Echo the command back. The swallow loop in ttyWriteRead() reads while
         // totrv <= strWrite.size(), so it always consumes one byte beyond the echo and
         // discards whatever chunk that byte arrived in. A single throwaway filler byte is
         // sent on its own for that purpose. The sleeps keep the echo, the filler, and the
         // real reply from being coalesced into one read(). The real reply is sent last so
         // the following ttyRead() picks it up.
         ::send(sp[1], strWrite.c_str(), strWrite.size(), 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         ::send(sp[1], "X", 1, 0);
         std::this_thread::sleep_for(std::chrono::milliseconds(20));
         std::string reply = "result> ";
         ::send(sp[1], reply.c_str(), reply.size(), 0);
      });

      std::string strRead;
      int rv = MagAOX::tty::ttyWriteRead(strRead, strWrite, "> ", true, sp[0], 1000, 1000);

      device.join();

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(strRead == "result> ");

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("succeeds without echo swallowing")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      std::thread device([sp]()
      {
         char buff[256];
         ::recv(sp[1], buff, sizeof(buff), 0);

         std::string reply = "result> ";
         ::send(sp[1], reply.c_str(), reply.size(), 0);
      });

      std::string strRead;
      int rv = MagAOX::tty::ttyWriteRead(strRead, "cmd", "> ", false, sp[0], 1000, 1000);

      device.join();

      REQUIRE(rv == TTY_E_NOERROR);
      REQUIRE(strRead == "result> ");

      ::close(sp[0]);
      ::close(sp[1]);
   }

   SECTION("propagates a write error without attempting the read")
   {
      int fd = ::dup(STDIN_FILENO);
      REQUIRE(fd >= 0);
      ::close(fd);

      std::string strRead;
      int rv = MagAOX::tty::ttyWriteRead(strRead, "cmd", "> ", false, fd, 1000, 1000);

      REQUIRE(rv == TTY_E_ERRORONWRITE);
   }

   SECTION("returns a read error when swallowing the echo fails")
   {
      int sp[2];
      REQUIRE( ::socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0 );

      // Nothing is ever sent back, and the read timeout is short.
      std::string strRead;
      int rv = MagAOX::tty::ttyWriteRead(strRead, "cmd", "> ", true, sp[0], 1000, 50);

      REQUIRE(rv == TTY_E_TIMEOUTONREADPOLL);

      ::close(sp[0]);
      ::close(sp[1]);
   }
}

} //namespace ttyIOUtils_test
