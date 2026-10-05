/** \file modbus_test.cpp
 * \brief Catch2 tests for the MagAO-X Modbus transport.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 */

#include "../../../tests/catch2/catch.hpp"

#include <sys/socket.h>
#include <unistd.h>

#define private public
#include "../modbus.hpp"
#undef private
#include "../modbus_exception.hpp"

namespace libXWCTest
{
namespace modbusTest
{

/// Verify the Modbus client throws instead of dying when the peer disappears.
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
