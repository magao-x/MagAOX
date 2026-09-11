/** \file modbus_test.cpp
 * \brief Catch2 tests for the MagAO-X Modbus transport.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 */

#include "../../../tests/catch2/catch.hpp"

#include <sys/socket.h>
#include <sys/resource.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <cstring>

#define private public
#include "../modbus.hpp"
#undef private
#include "../modbus_exception.hpp"

// Technique notes for this file. The tests never talk to real Modbus hardware. Most of
// them connect the modbus object to one end of a socketpair and play the server on the
// other end from a background thread. The private members are reached by defining
// private as public before the include. Fault injection: each block below compiles
// modbus.hpp and modbus.cpp a second time inside a test namespace with one XWCTEST_
// fault macro defined. The macro turns on one production error branch that a real
// socket will not produce on demand. The include guard is undefined first so the header
// is really re-read.

// Makes modbus_set_timeouts() see the SO_SNDTIMEO setsockopt call fail after the real
// call succeeded.
#undef MODBUSPP_MODBUS_H
#define XWCTEST_NAMESPACE XWCTEST_MODBUS_SET_TIMEOUTS_SNDTIMEO_FAIL_ns
#define XWCTEST_MODBUS_SET_TIMEOUTS_SNDTIMEO_FAIL
#define private public
#include "../modbus.hpp"
#include "../modbus.cpp"
#undef private
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MODBUS_SET_TIMEOUTS_SNDTIMEO_FAIL

// Makes modbus_send() see one byte fewer sent than requested after a real, fully
// successful send.
#undef MODBUSPP_MODBUS_H
#define XWCTEST_NAMESPACE XWCTEST_MODBUS_SEND_PARTIAL_ns
#define XWCTEST_MODBUS_SEND_PARTIAL
#define private public
#include "../modbus.hpp"
#include "../modbus.cpp"
#undef private
#undef XWCTEST_NAMESPACE
#undef XWCTEST_MODBUS_SEND_PARTIAL

namespace libXWCTest
{
namespace modbusTest
{

/// Connects mb to one end of a fresh socketpair and runs callModbus. While it runs, a
/// background thread drains whatever request bytes arrive on the other end. Unless
/// response is empty, the thread then sends response back. The function blocks until both
/// callModbus and the responder thread finish. If req is not null it is filled with the
/// bytes that were received.
void runAgainstFakeServer( modbus &mb,
                            const std::vector<uint8_t> &response,
                            const std::function<void()> &callModbus,
                            std::vector<uint8_t> *req = nullptr )
{
    int sp[2];
    REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

    mb._socket    = sp[0];
    mb._connected = true;

    std::thread responder(
        [peer = sp[1], &response, req]()
        {
            uint8_t buf[MAX_MSG_LENGTH];
            ssize_t n = ::recv( peer, buf, sizeof( buf ), 0 );

            if( req != nullptr && n > 0 )
            {
                req->assign( buf, buf + n );
            }

            if( !response.empty() )
            {
                ::send( peer, response.data(), response.size(), 0 );
            }

            ::close( peer );
        } );

    // callModbus() may throw. That is the whole point of the error response tests. So the
    // join has to happen no matter what. A std::thread that is still joinable calls
    // std::terminate() when it is destroyed while unwinding.
    try
    {
        callModbus();
    }
    catch( ... )
    {
        responder.join();
        throw;
    }

    responder.join();
}

/// The constructors store the host and port and start in the disconnected state.
TEST_CASE( "modbus construction sets the host, port, and initial state", "[modbus]" )
{
    SECTION( "host and port constructor" )
    {
        modbus mb( "10.0.0.5", 1502 );

        REQUIRE( mb.HOST == "10.0.0.5" );
        REQUIRE( mb.PORT == 1502 );
        REQUIRE( mb._connected == false );
        REQUIRE( mb._socket == -1 );
        REQUIRE( mb._slaveid == 1 );
    }

    SECTION( "host-only constructor defaults the port to 502" )
    {
        modbus mb( "10.0.0.5" );

        REQUIRE( mb.HOST == "10.0.0.5" );
        REQUIRE( mb.PORT == 502 );
    }
}

/// modbus_set_slave_id stores the new slave id.
TEST_CASE( "modbus_set_slave_id updates the slave id", "[modbus]" )
{
    modbus mb( "127.0.0.1", 502 );

    mb.modbus_set_slave_id( 7 );

    REQUIRE( mb._slaveid == 7 );
}

/// modbus_connect is exercised against an empty host, an exhausted file descriptor
/// table, a refused loopback port, and a real listening socket on loopback.
TEST_CASE( "modbus_connect", "[modbus]" )
{
    SECTION( "fails immediately when the host is empty" )
    {
        modbus mb( "" );

        REQUIRE( mb.modbus_connect() == false );
    }

    SECTION( "fails when socket() itself fails" )
    {
        struct rlimit orig;
        REQUIRE( getrlimit( RLIMIT_NOFILE, &orig ) == 0 );

        struct rlimit tiny;
        tiny.rlim_cur = 3; // This leaves room for stdin, stdout, and stderr only, so no new socket fd can be created.
        tiny.rlim_max = orig.rlim_max;
        REQUIRE( setrlimit( RLIMIT_NOFILE, &tiny ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        bool   rv = mb.modbus_connect();

        REQUIRE( setrlimit( RLIMIT_NOFILE, &orig ) == 0 );

        REQUIRE( rv == false );
    }

    SECTION( "fails when the connection is refused" )
    {
        // Bind to an ephemeral port, then close it immediately so nothing is listening.
        // Connecting to it on loopback fails fast with ECONNREFUSED rather than hanging.
        int probe = ::socket( AF_INET, SOCK_STREAM, 0 );
        REQUIRE( probe >= 0 );

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = inet_addr( "127.0.0.1" );
        addr.sin_port        = 0;

        REQUIRE( ::bind( probe, (struct sockaddr *)&addr, sizeof( addr ) ) == 0 );

        socklen_t len = sizeof( addr );
        REQUIRE( ::getsockname( probe, (struct sockaddr *)&addr, &len ) == 0 );
        uint16_t port = ntohs( addr.sin_port );

        REQUIRE( ::close( probe ) == 0 );

        modbus mb( "127.0.0.1", port );

        REQUIRE( mb.modbus_connect() == false );
        REQUIRE( mb._socket == -1 );
    }

    SECTION( "succeeds against a real listening server" )
    {
        int listenSock = ::socket( AF_INET, SOCK_STREAM, 0 );
        REQUIRE( listenSock >= 0 );

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = inet_addr( "127.0.0.1" );
        addr.sin_port        = 0;

        REQUIRE( ::bind( listenSock, (struct sockaddr *)&addr, sizeof( addr ) ) == 0 );
        REQUIRE( ::listen( listenSock, 1 ) == 0 );

        socklen_t len = sizeof( addr );
        REQUIRE( ::getsockname( listenSock, (struct sockaddr *)&addr, &len ) == 0 );
        uint16_t port = ntohs( addr.sin_port );

        std::thread acceptThread(
            [listenSock]()
            {
                int connFd = ::accept( listenSock, nullptr, nullptr );
                if( connFd >= 0 )
                {
                    ::close( connFd );
                }
            } );

        modbus mb( "127.0.0.1", port );
        bool   rv = mb.modbus_connect();

        acceptThread.join();
        ::close( listenSock );

        REQUIRE( rv == true );
        REQUIRE( mb._connected == true );

        mb.modbus_close();
    }
}

/// modbus_close closes the socket and resets the connection state. It is also safe to
/// call when nothing was ever connected.
TEST_CASE( "modbus_close releases the socket and resets state", "[modbus]" )
{
    SECTION( "closes an open socket" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket    = sp[0];
        mb._connected = true;

        mb.modbus_close();

        REQUIRE( mb._socket == -1 );
        REQUIRE( mb._connected == false );

        // sp[0] should now be closed. Writing to its peer should eventually see it gone.
        ::close( sp[1] );
    }

    SECTION( "is safe to call when never connected" )
    {
        modbus mb( "127.0.0.1", 502 );

        mb.modbus_close();

        REQUIRE( mb._socket == -1 );
        REQUIRE( mb._connected == false );
    }
}

/// modbus_set_timeouts is exercised on a disconnected object, a live socketpair, a
/// closed descriptor, and the fault-injected build where only the send timeout call
/// fails.
TEST_CASE( "modbus_set_timeouts", "[modbus]" )
{
    SECTION( "fails when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );

        REQUIRE( mb.modbus_set_timeouts( 1 ) == false );
    }

    SECTION( "succeeds on a connected socket" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket    = sp[0];
        mb._connected = true;

        REQUIRE( mb.modbus_set_timeouts( 1, 500 ) == true );

        ::close( sp[0] );
        ::close( sp[1] );
    }

    SECTION( "fails when setsockopt fails" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );
        REQUIRE( ::close( sp[0] ) == 0 );
        REQUIRE( ::close( sp[1] ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket    = sp[0]; // This is a stale descriptor that is now closed.
        mb._connected = true;

        REQUIRE( mb.modbus_set_timeouts( 1 ) == false );
    }

    SECTION( "fails when only the SO_SNDTIMEO call fails" )
    {
        // SO_RCVTIMEO and SO_SNDTIMEO act on the same fd with the same timeval. So there
        // is no natural way to make only the second setsockopt() call fail. This uses the
        // XWCTEST_MODBUS_SET_TIMEOUTS_SNDTIMEO_FAIL build variant, which forces the result
        // after the real successful call, rather than mocking setsockopt() itself.
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        XWCTEST_MODBUS_SET_TIMEOUTS_SNDTIMEO_FAIL_ns::modbus mb( "127.0.0.1", 502 );
        mb._socket    = sp[0];
        mb._connected = true;

        REQUIRE( mb.modbus_set_timeouts( 1 ) == false );

        ::close( sp[0] );
        ::close( sp[1] );
    }
}

/// modbus_build_request encodes the header fields into a raw buffer. Each byte is
/// checked.
TEST_CASE( "modbus_build_request encodes the header fields", "[modbus]" )
{
    modbus mb( "127.0.0.1", 502 );
    mb.modbus_set_slave_id( 7 );

    uint8_t buf[10]{};
    mb.modbus_build_request( buf, 0x1234, READ_REGS );

    // The transaction id starts at 1, so its high byte is 0 here.
    REQUIRE( buf[0] == (uint8_t)( mb._msg_id >> 8 ) );
    REQUIRE( buf[1] == (uint8_t)( mb._msg_id & 0x00FF ) );
    REQUIRE( buf[2] == 0 );
    REQUIRE( buf[3] == 0 );
    REQUIRE( buf[4] == 0 );
    REQUIRE( buf[6] == 7 );
    REQUIRE( buf[7] == READ_REGS );
    REQUIRE( buf[8] == 0x12 );
    REQUIRE( buf[9] == 0x34 );
}

/// modbus_read_holding_registers argument checks, response decoding, and error responses
/// are exercised through the fake server helper.
TEST_CASE( "modbus_read_holding_registers", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[1]{};

        REQUIRE_THROWS_AS( mb.modbus_read_holding_registers( 0, 1, buffer ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when amount is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        uint16_t buffer[1]{};
        REQUIRE_THROWS_AS( mb.modbus_read_holding_registers( 0, 65536, buffer ), modbus_amount_exception );
    }

    SECTION( "throws modbus_amount_exception when address is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        uint16_t buffer[1]{};
        REQUIRE_THROWS_AS( mb.modbus_read_holding_registers( 65536, 1, buffer ), modbus_amount_exception );
    }

    SECTION( "decodes two registers from a successful response" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[2]{};

        std::vector<uint8_t> response( 13, 0 );
        response[7]  = READ_REGS;
        response[9]  = 0x00;
        response[10] = 0x2A; // Register 0 is 42.
        response[11] = 0x00;
        response[12] = 0x63; // Register 1 is 99.

        std::vector<uint8_t> req;
        runAgainstFakeServer(
            mb, response, [&]() { mb.modbus_read_holding_registers( 0, 2, buffer ); }, &req );

        REQUIRE( buffer[0] == 42 );
        REQUIRE( buffer[1] == 99 );
        REQUIRE( req.size() == 12 );
        REQUIRE( req[7] == READ_REGS );
    }

    SECTION( "throws the specific exception for an illegal-address error response" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[1]{};

        std::vector<uint8_t> response( 9, 0 );
        response[7] = READ_REGS + 0x80;
        response[8] = EX_ILLEGAL_ADDRESS;

        REQUIRE_THROWS_AS(
            runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_holding_registers( 0, 1, buffer ); } ),
            modbus_illegal_address_exception );
    }
}

/// modbus_read_input_registers argument checks, response decoding, and error responses
/// are exercised through the fake server helper.
TEST_CASE( "modbus_read_input_registers", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[1]{};

        REQUIRE_THROWS_AS( mb.modbus_read_input_registers( 0, 1, buffer ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when amount is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        uint16_t buffer[1]{};
        REQUIRE_THROWS_AS( mb.modbus_read_input_registers( 0, 65536, buffer ), modbus_amount_exception );
    }

    SECTION( "decodes a register from a successful response" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[1]{};

        std::vector<uint8_t> response( 11, 0 );
        response[7] = READ_INPUT_REGS;
        response[9] = 0x01;
        response[10] = 0x00; // Register 0 is 256.

        runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_input_registers( 0, 1, buffer ); } );

        REQUIRE( buffer[0] == 256 );
    }

    SECTION( "throws the specific exception for a server-failure error response" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t buffer[1]{};

        std::vector<uint8_t> response( 9, 0 );
        response[7] = READ_INPUT_REGS + 0x80;
        response[8] = EX_SERVER_FAILURE;

        REQUIRE_THROWS_AS(
            runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_input_registers( 0, 1, buffer ); } ),
            modbus_server_failure_exception );
    }
}

/// modbus_read_coils argument checks, packed bit decoding, and error responses are
/// exercised through the fake server helper.
TEST_CASE( "modbus_read_coils", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[1]{};

        REQUIRE_THROWS_AS( mb.modbus_read_coils( 0, 1, buffer ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when amount exceeds the coil limit" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        bool buffer[1]{};
        REQUIRE_THROWS_AS( mb.modbus_read_coils( 0, 2041, buffer ), modbus_amount_exception );
    }

    SECTION( "decodes packed bits from a successful response" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[10]{};

        std::vector<uint8_t> response( 11, 0 );
        response[7] = READ_COILS;
        response[9] = 0xB5;  // The bit pattern 1011 0101 sets bits 0, 2, 4, 5, and 7.
        response[10] = 0x01; // Bit 8 is set and bit 9 is clear.

        runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_coils( 0, 10, buffer ); } );

        REQUIRE( buffer[0] == true );
        REQUIRE( buffer[1] == false );
        REQUIRE( buffer[2] == true );
        REQUIRE( buffer[7] == true );
        REQUIRE( buffer[8] == true );
        REQUIRE( buffer[9] == false );
    }

    SECTION( "throws the specific exception for an illegal-function error response" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[1]{};

        std::vector<uint8_t> response( 9, 0 );
        response[7] = READ_COILS + 0x80;
        response[8] = EX_ILLEGAL_FUNCTION;

        REQUIRE_THROWS_AS( runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_coils( 0, 1, buffer ); } ),
                            modbus_illegal_function_exception );
    }
}

/// modbus_read_input_bits argument checks, bit decoding, and error responses are
/// exercised through the fake server helper.
TEST_CASE( "modbus_read_input_bits", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[1]{};

        REQUIRE_THROWS_AS( mb.modbus_read_input_bits( 0, 1, buffer ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when amount exceeds the limit" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        bool buffer[1]{};
        REQUIRE_THROWS_AS( mb.modbus_read_input_bits( 0, 2041, buffer ), modbus_amount_exception );
    }

    SECTION( "decodes a single bit from a successful response" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[1]{};

        std::vector<uint8_t> response( 10, 0 );
        response[7] = READ_INPUT_BITS;
        response[9] = 0x01;

        runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_input_bits( 0, 1, buffer ); } );

        REQUIRE( buffer[0] == true );
    }

    SECTION( "throws the specific exception for an illegal-data-value error response" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   buffer[1]{};

        std::vector<uint8_t> response( 9, 0 );
        response[7] = READ_INPUT_BITS + 0x80;
        response[8] = EX_ILLEGAL_VALUE;

        REQUIRE_THROWS_AS(
            runAgainstFakeServer( mb, response, [&]() { mb.modbus_read_input_bits( 0, 1, buffer ); } ),
            modbus_illegal_data_value_exception );
    }
}

/// modbus_write_coil argument checks, the encoded request bytes, and error responses are
/// exercised through the fake server helper.
TEST_CASE( "modbus_write_coil", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );

        REQUIRE_THROWS_AS( mb.modbus_write_coil( 0, true ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when the address is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        REQUIRE_THROWS_AS( mb.modbus_write_coil( 65536, true ), modbus_amount_exception );
    }

    SECTION( "completes without throwing on a successful acknowledgement" )
    {
        modbus mb( "127.0.0.1", 502 );

        std::vector<uint8_t> response( 12, 0 );
        response[7] = WRITE_COIL;

        std::vector<uint8_t> req;
        runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_coil( 0, true ); }, &req );

        REQUIRE( req.size() == 12 );
        // The value 0xFF00 is written as the high and low bytes at the end of the request.
        REQUIRE( req[10] == 0xFF );
        REQUIRE( req[11] == 0x00 );
    }

    SECTION( "throws the specific exception for a server-busy error response" )
    {
        modbus mb( "127.0.0.1", 502 );

        std::vector<uint8_t> response( 9, 0 );
        response[7] = WRITE_COIL + 0x80;
        response[8] = EX_SERVER_BUSY;

        REQUIRE_THROWS_AS( runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_coil( 0, true ); } ),
                            modbus_server_busy_exception );
    }
}

/// modbus_write_register argument checks, a successful acknowledgement, and an error
/// response are exercised through the fake server helper.
TEST_CASE( "modbus_write_register", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );

        REQUIRE_THROWS_AS( mb.modbus_write_register( 0, 123 ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when the address is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        REQUIRE_THROWS_AS( mb.modbus_write_register( 65536, 123 ), modbus_amount_exception );
    }

    SECTION( "completes without throwing on a successful acknowledgement" )
    {
        modbus mb( "127.0.0.1", 502 );

        std::vector<uint8_t> response( 12, 0 );
        response[7] = WRITE_REG;

        runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_register( 0, 4660 ); } );
    }

    SECTION( "throws the specific exception for an acknowledge error response" )
    {
        // The error bit is set on the function code that was sent, WRITE_REG.
        modbus mb( "127.0.0.1", 502 );

        std::vector<uint8_t> response( 9, 0 );
        response[7] = WRITE_REG + 0x80;
        response[8] = EX_ACKNOWLEDGE;

        REQUIRE_THROWS_AS( runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_register( 0, 4660 ); } ),
                            modbus_acknowledge_exception );
    }
}

/// modbus_write_coils argument checks, a successful acknowledgement, and an error
/// response are exercised through the fake server helper.
TEST_CASE( "modbus_write_coils", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   value[4]{ true, false, true, false };

        REQUIRE_THROWS_AS( mb.modbus_write_coils( 0, 4, value ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when the address is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        bool value[4]{ true, false, true, false };
        REQUIRE_THROWS_AS( mb.modbus_write_coils( 65536, 4, value ), modbus_amount_exception );
    }

    SECTION( "completes without throwing on a successful acknowledgement" )
    {
        // modbus_write_coils always copies exactly 4 source values regardless of amount.
        // So amount is kept at 4 here to stay within the bounds of value.
        modbus mb( "127.0.0.1", 502 );
        bool   value[4]{ true, false, true, false };

        std::vector<uint8_t> response( 12, 0 );
        response[7] = WRITE_COILS;

        runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_coils( 0, 4, value ); } );
    }

    SECTION( "throws the specific exception for a gateway-problem error response" )
    {
        modbus mb( "127.0.0.1", 502 );
        bool   value[4]{ true, false, true, false };

        std::vector<uint8_t> response( 9, 0 );
        response[7] = WRITE_COILS + 0x80;
        response[8] = EX_GATEWAY_PROBLEMP;

        REQUIRE_THROWS_AS( runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_coils( 0, 4, value ); } ),
                            modbus_gateway_exception );
    }
}

/// modbus_write_registers argument checks, the request size, and an error response are
/// exercised through the fake server helper.
TEST_CASE( "modbus_write_registers", "[modbus]" )
{
    SECTION( "throws modbus_connect_exception when not connected" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t value[2]{ 1, 2 };

        REQUIRE_THROWS_AS( mb.modbus_write_registers( 0, 2, value ), modbus_connect_exception );
    }

    SECTION( "throws modbus_amount_exception when amount is too large" )
    {
        modbus mb( "127.0.0.1", 502 );
        mb._connected = true;

        uint16_t value[2]{ 1, 2 };
        REQUIRE_THROWS_AS( mb.modbus_write_registers( 0, 65536, value ), modbus_amount_exception );
    }

    SECTION( "completes without throwing on a successful acknowledgement" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t value[2]{ 1, 2 };

        std::vector<uint8_t> response( 12, 0 );
        response[7] = WRITE_REGS;

        std::vector<uint8_t> req;
        runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_registers( 0, 2, value ); }, &req );

        REQUIRE( req.size() == 17 ); // The size is 13 + 2*amount.
        REQUIRE( req[7] == WRITE_REGS );
    }

    SECTION( "throws the specific exception for a gateway-problem (variant) error response" )
    {
        modbus   mb( "127.0.0.1", 502 );
        uint16_t value[2]{ 1, 2 };

        std::vector<uint8_t> response( 9, 0 );
        response[7] = WRITE_REGS + 0x80;
        response[8] = EX_GATEWYA_PROBLEMF;

        REQUIRE_THROWS_AS( runAgainstFakeServer( mb, response, [&]() { mb.modbus_write_registers( 0, 2, value ); } ),
                            modbus_gateway_exception );
    }
}

/// modbus_error_handle maps each known Modbus exception code to its C++ exception type
/// and ignores everything else.
TEST_CASE( "modbus_error_handle", "[modbus]" )
{
    modbus mb( "127.0.0.1", 502 );

    SECTION( "does nothing when the function code doesn't carry the error bit" )
    {
        uint8_t msg[9]{};
        msg[7] = READ_REGS;

        REQUIRE_NOTHROW( mb.modbus_error_handle( msg, READ_REGS ) );
    }

    SECTION( "does nothing for an unrecognized exception code" )
    {
        uint8_t msg[9]{};
        msg[7] = READ_REGS + 0x80;
        msg[8] = 0x7F; // This is not one of the known EX_* codes.

        REQUIRE_NOTHROW( mb.modbus_error_handle( msg, READ_REGS ) );
    }

    SECTION( "throws the exception matching each known exception code" )
    {
        uint8_t msg[9]{};
        msg[7] = READ_REGS + 0x80;

        msg[8] = EX_ILLEGAL_FUNCTION;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_illegal_function_exception );

        msg[8] = EX_ILLEGAL_ADDRESS;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_illegal_address_exception );

        msg[8] = EX_ILLEGAL_VALUE;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_illegal_data_value_exception );

        msg[8] = EX_SERVER_FAILURE;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_server_failure_exception );

        msg[8] = EX_ACKNOWLEDGE;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_acknowledge_exception );

        msg[8] = EX_SERVER_BUSY;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_server_busy_exception );

        msg[8] = EX_GATEWAY_PROBLEMP;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_gateway_exception );

        msg[8] = EX_GATEWYA_PROBLEMF;
        REQUIRE_THROWS_AS( mb.modbus_error_handle( msg, READ_REGS ), modbus_gateway_exception );
    }
}

/// modbus_send is exercised over a real socketpair, a closed descriptor, and the
/// fault-injected partial send build.
TEST_CASE( "modbus_send", "[modbus]" )
{
    SECTION( "returns the number of bytes sent on success" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        modbus  mb( "127.0.0.1", 502 );
        mb._socket = sp[0];

        uint8_t data[4]{ 1, 2, 3, 4 };
        ssize_t rv = mb.modbus_send( data, sizeof( data ) );

        REQUIRE( rv == 4 );

        uint8_t received[4]{};
        REQUIRE( ::recv( sp[1], received, sizeof( received ), 0 ) == 4 );
        REQUIRE( std::memcmp( received, data, 4 ) == 0 );

        ::close( sp[0] );
        ::close( sp[1] );
    }

    SECTION( "throws modbus_connect_exception when send() fails" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );
        REQUIRE( ::close( sp[0] ) == 0 );
        REQUIRE( ::close( sp[1] ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket = sp[0]; // This is a stale descriptor that is now closed.

        uint8_t data[1]{ 42 };
        REQUIRE_THROWS_AS( mb.modbus_send( data, sizeof( data ) ), modbus_connect_exception );
    }

    SECTION( "throws modbus_connect_exception on a partial send" )
    {
        // A blocking send() of a small buffer does not produce short writes under normal
        // conditions. So this uses the XWCTEST_MODBUS_SEND_PARTIAL build variant. It
        // forces the byte count to be one less than requested after the real, fully
        // successful send completes, rather than mocking send() itself.
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        XWCTEST_MODBUS_SEND_PARTIAL_ns::modbus mb( "127.0.0.1", 502 );
        mb._socket = sp[0];

        uint8_t data[4]{ 1, 2, 3, 4 };
        REQUIRE_THROWS_AS( mb.modbus_send( data, sizeof( data ) ), modbus_connect_exception );

        ::close( sp[0] );
        ::close( sp[1] );
    }
}

/// modbus_receive is exercised over a real socketpair, a closed descriptor, and a peer
/// that shut down its write side.
TEST_CASE( "modbus_receive", "[modbus]" )
{
    SECTION( "returns the received bytes on success" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket = sp[0];

        uint8_t data[3]{ 9, 8, 7 };
        REQUIRE( ::send( sp[1], data, sizeof( data ), 0 ) == 3 );

        uint8_t buffer[MAX_MSG_LENGTH]{};
        ssize_t rv = mb.modbus_receive( buffer );

        REQUIRE( rv == 3 );
        REQUIRE( std::memcmp( buffer, data, 3 ) == 0 );

        ::close( sp[0] );
        ::close( sp[1] );
    }

    SECTION( "throws modbus_connect_exception when the socket itself is invalid" )
    {
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );
        REQUIRE( ::close( sp[0] ) == 0 );
        REQUIRE( ::close( sp[1] ) == 0 );

        modbus  mb( "127.0.0.1", 502 );
        mb._socket = sp[0];

        uint8_t buffer[MAX_MSG_LENGTH]{};
        REQUIRE_THROWS_AS( mb.modbus_receive( buffer ), modbus_connect_exception );
    }

    SECTION( "throws modbus_connect_exception on a clean peer shutdown (recv returns 0)" )
    {
        // Shut down only the write side of the peer rather than closing it outright. A
        // full close() tends to make our own send() fail with EPIPE first, which is a
        // different branch. Shutting down writes lets recv() itself observe the clean EOF.
        int sp[2];
        REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sp ) == 0 );
        REQUIRE( ::shutdown( sp[1], SHUT_WR ) == 0 );

        modbus mb( "127.0.0.1", 502 );
        mb._socket = sp[0];

        uint8_t buffer[MAX_MSG_LENGTH]{};
        REQUIRE_THROWS_AS( mb.modbus_receive( buffer ), modbus_connect_exception );

        ::close( sp[0] );
        ::close( sp[1] );
    }
}

/// The Modbus client throws instead of dying when the peer disappears. The peer end of a
/// socketpair is closed before a read is attempted.
TEST_CASE( "modbus reports a dropped peer as a connection exception", "[modbus]" )
{
    int socketPair[2];
    REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, socketPair ) == 0 );

    modbus mb( "127.0.0.1", 502 );
    mb._socket    = socketPair[0];
    mb._connected = true;

    REQUIRE( ::close( socketPair[1] ) == 0 );

    uint16_t inputRegs[1]{ 0 };

    REQUIRE_THROWS_AS( mb.modbus_read_input_registers( 0, 1, inputRegs ), modbus_connect_exception );
    REQUIRE( mb._connected == false );
}

} // namespace modbusTest
} // namespace libXWCTest
