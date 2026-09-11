// #define CATCH_CONFIG_MAIN
/** \file indiDriver_test.cpp
  * \brief Catch2 tests for the MagAOX::app::indiDriver INDI driver wrapper and its indiClient helper.
  *
  * The driver is constructed against a small stand-in parent from indiDriver_test.hpp instead of a
  * full MagAOXApp. Each scenario creates real FIFOs under /tmp/indiDriver_test for the driver's
  * input, output, and control channels. Outgoing connections are tested against a real TCP
  * listener on 127.0.0.1 that never accepts. A completed TCP handshake is enough for the client
  * connect() call to succeed.
  *
  * Failure branches that no external condition can trigger are reached through the
  * XWCTEST_INDIDRIVER_ fault injection macros. This file compiles indiDriver.hpp once per
  * hook inside its own namespace with that one hook defined. See the includes below.
  *
  * \ingroup indiDriver_tests
  */
#include "../../../tests/catch2/catch.hpp"

#include <filesystem>
#include <thread>
#include <chrono>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// indiDriver.hpp and MagAOXApp.hpp include each other. indiDriver.hpp needs the logger
// types from MagAOXApp, and MagAOXApp needs indiDriver<MagAOXApp> as a member type.
// MagAOXApp.hpp must be included first. It pulls in indiDriver.hpp, so indiDriver.hpp is
// fully processed by the time MagAOXApp.hpp declares its indiDriver<MagAOXApp> member.
#include "../MagAOXApp.hpp"
#include "../indiDriver.hpp"

// Fault injection copies of indiDriver. Each block compiles indiDriver.hpp again inside
// its own namespace with one XWCTEST_INDIDRIVER_ name defined. The matching XWCTEST_IF_
// macro in that copy runs its fault for real. The production copy above stays untouched.
// MagAOXApp.hpp keeps its include guard, so only indiDriver.hpp is recompiled.
#undef app_indiDriver_hpp
#define XWCTEST_NAMESPACE XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL_ns
#define XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL
#include "../indiDriver.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL

#undef app_indiDriver_hpp
#define XWCTEST_NAMESPACE XWCTEST_INDIDRIVER_OUTGOING_NULL_ns
#define XWCTEST_INDIDRIVER_OUTGOING_NULL
#include "../indiDriver.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_INDIDRIVER_OUTGOING_NULL

#undef app_indiDriver_hpp
#define XWCTEST_NAMESPACE XWCTEST_INDIDRIVER_SEND_NONSTD_THROW_ns
#define XWCTEST_INDIDRIVER_SEND_NONSTD_THROW
#include "../indiDriver.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_INDIDRIVER_SEND_NONSTD_THROW

#undef app_indiDriver_hpp
#define XWCTEST_NAMESPACE XWCTEST_INDIDRIVER_QUIT_AFTER_SEND_ns
#define XWCTEST_INDIDRIVER_QUIT_AFTER_SEND
#include "../indiDriver.hpp"
#undef XWCTEST_NAMESPACE
#undef XWCTEST_INDIDRIVER_QUIT_AFTER_SEND

#include "indiDriver_test.hpp"

using namespace indiDriver_tests;

namespace
{

/// Make sure a fresh FIFO exists at path, removing anything already there first.
void makeFreshFifo( const std::string &path )
{
    std::filesystem::remove( path );
    REQUIRE( ::mkfifo( path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP ) == 0 );
}

/// Build a simple, well-formed Number property with one element.
/// It is suitable for use as a NEW property message because its type is not Unknown.
pcf::IndiProperty makeNumberProperty( const std::string &device, const std::string &name )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( "value" ) );
    ip["value"].setValue( 1.0 );
    return ip;
}

/// A minimal TCP listener on 127.0.0.1, used only so that the IndiClient connect() call succeeds.
/// A real INDI server on the other end is not required. accept() is never called. The completed
/// TCP handshake sitting in the kernel backlog is enough for connect() to return success.
/// The constructor binds to port 0 so the OS picks a free port, then records the chosen port.
struct TcpListener
{
    int fd{ -1 };
    int port{ 0 };

    TcpListener()
    {
        fd = ::socket( AF_INET, SOCK_STREAM, 0 );
        REQUIRE( fd >= 0 );

        int opt = 1;
        ::setsockopt( fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
        addr.sin_port        = 0; // let the OS choose a free port

        REQUIRE( ::bind( fd, reinterpret_cast<sockaddr *>( &addr ), sizeof( addr ) ) == 0 );
        REQUIRE( ::listen( fd, 8 ) == 0 );

        socklen_t len = sizeof( addr );
        REQUIRE( ::getsockname( fd, reinterpret_cast<sockaddr *>( &addr ), &len ) == 0 );
        port = ntohs( addr.sin_port );
    }

    ~TcpListener()
    {
        if( fd >= 0 )
        {
            ::close( fd );
        }
    }
};

} // namespace

/** \defgroup indiDriver_tests libXWC::app::indiDriver Unit Tests
 * \ingroup app_unit_test
 */

/// Verify construction of indiDriver, including every FIFO-related failure mode.
/// Each GIVEN block points the parent at real FIFOs under /tmp, or at paths that can not be
/// opened, and checks good() after construction.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "Constructing an indiDriver", "[app::indiDriver]" )
{
    std::filesystem::create_directories( "/tmp/indiDriver_test" );

    GIVEN( "all three FIFOs can be opened and written to" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/good.in" );
        makeFreshFifo( "/tmp/indiDriver_test/good.out" );
        makeFreshFifo( "/tmp/indiDriver_test/good.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/good.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/good.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/good.ctrl";

        WHEN( "constructed" )
        {
            indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );

            THEN( "good() is true" )
            {
                REQUIRE( drv.good() == true );
            }
        }
    }

    GIVEN( "the input FIFO can not be opened" )
    {
        indiDriverTestParent parent;
        parent.m_driverInName   = "/tmp/indiDriver_test/does_not_exist/bad.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/unused.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/unused.ctrl";

        WHEN( "constructed" )
        {
            indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );

            THEN( "good() is false" )
            {
                REQUIRE( drv.good() == false );
            }
        }
    }

    GIVEN( "the input FIFO opens but the output FIFO can not be opened" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/in2.in" );
        parent.m_driverInName   = "/tmp/indiDriver_test/in2.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/does_not_exist/bad.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/unused2.ctrl";

        WHEN( "constructed" )
        {
            indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );

            THEN( "good() is false" )
            {
                REQUIRE( drv.good() == false );
            }
        }
    }

    GIVEN( "input and output FIFOs open but the ctrl FIFO can not be opened" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/in3.in" );
        makeFreshFifo( "/tmp/indiDriver_test/out3.out" );
        parent.m_driverInName   = "/tmp/indiDriver_test/in3.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/out3.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/does_not_exist/bad.ctrl";

        WHEN( "constructed" )
        {
            indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );

            THEN( "good() is false" )
            {
                REQUIRE( drv.good() == false );
            }
        }
    }

    GIVEN( "all three FIFOs open but the write to the ctrl FIFO fails" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/in4.in" );
        makeFreshFifo( "/tmp/indiDriver_test/out4.out" );
        makeFreshFifo( "/tmp/indiDriver_test/ctrl4.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/in4.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/out4.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/ctrl4.ctrl";

        WHEN( "constructed with the ctrl-write-fail hook engaged" )
        {
            indiDriverExposed<indiDriverTestParent, MagAOX::app::XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL_ns::indiDriver> drv(
                &parent, "test", "0", "0" );

            THEN( "good() is false" )
            {
                REQUIRE( drv.good() == false );
            }
        }
    }
}

/// Verify that the handleXXXProperty overrides forward to the parent, and that they are
/// safe no-ops when the parent pointer has been cleared. The stand-in parent counts each call.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "indiDriver property handlers", "[app::indiDriver]" )
{
    std::filesystem::create_directories( "/tmp/indiDriver_test" );

    indiDriverTestParent parent;
    makeFreshFifo( "/tmp/indiDriver_test/hp.in" );
    makeFreshFifo( "/tmp/indiDriver_test/hp.out" );
    makeFreshFifo( "/tmp/indiDriver_test/hp.ctrl" );
    parent.m_driverInName   = "/tmp/indiDriver_test/hp.in";
    parent.m_driverOutName  = "/tmp/indiDriver_test/hp.out";
    parent.m_driverCtrlName = "/tmp/indiDriver_test/hp.ctrl";

    indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );
    REQUIRE( drv.good() == true );

    pcf::IndiProperty ip;

    GIVEN( "a parent is set" )
    {
        WHEN( "each handler is called" )
        {
            drv.handleDefProperty( ip );
            drv.handleGetProperties( ip );
            drv.handleNewProperty( ip );
            drv.handleSetProperty( ip );

            THEN( "the parent's counters are incremented" )
            {
                REQUIRE( parent.defCount == 1 );
                REQUIRE( parent.getPropCount == 1 );
                REQUIRE( parent.newPropCount == 1 );
                REQUIRE( parent.setPropCount == 1 );
            }
        }
    }

    GIVEN( "the parent has been cleared" )
    {
        drv.clearParent();

        WHEN( "each handler is called" )
        {
            drv.handleDefProperty( ip );
            drv.handleGetProperties( ip );
            drv.handleNewProperty( ip );
            drv.handleSetProperty( ip );

            THEN( "nothing happens (no crash, no counters incremented)" )
            {
                REQUIRE( parent.defCount == 0 );
                REQUIRE( parent.getPropCount == 0 );
                REQUIRE( parent.newPropCount == 0 );
                REQUIRE( parent.setPropCount == 0 );
            }
        }
    }
}

/// Verify that execute() and update() run the underlying INDI processing loop and can be
/// stopped cleanly. execute() is run on a separate thread and stopped with quitProcess().
/// Also verify that deleting a heap-allocated driver of the exact base type does not crash.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "indiDriver execute and update", "[app::indiDriver]" )
{
    std::filesystem::create_directories( "/tmp/indiDriver_test" );

    GIVEN( "a valid indiDriver" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/ex.in" );
        makeFreshFifo( "/tmp/indiDriver_test/ex.out" );
        makeFreshFifo( "/tmp/indiDriver_test/ex.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/ex.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/ex.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/ex.ctrl";

        indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        WHEN( "execute() is run and then stopped" )
        {
            std::thread t( [&drv]() { drv.execute(); } );

            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
            drv.quitProcess();

            t.join();

            THEN( "it returns cleanly" )
            {
                REQUIRE( drv.getQuitProcess() == true );
            }
        }
    }

    GIVEN( "a heap-allocated indiDriver of the exact (non-derived) type" )
    {
        // Exercises deletion through a pointer to the exact MagAOX::app::indiDriver<parentT>
        // type. This matches how MagAOXApp owns and deletes its m_indiDriver member.
        // Every other test in this file uses automatic destruction of a stack object.
        // Deleting through the derived indiDriverExposed helper would not do the same thing.
        // That would enter the generated destructor of indiDriverExposed instead of the
        // destructor of indiDriver<parentT>.
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/heap.in" );
        makeFreshFifo( "/tmp/indiDriver_test/heap.out" );
        makeFreshFifo( "/tmp/indiDriver_test/heap.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/heap.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/heap.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/heap.ctrl";

        auto *drv = new MagAOX::app::indiDriver<indiDriverTestParent>( &parent, "test", "0", "0" );

        WHEN( "it is deleted" )
        {
            REQUIRE( drv->good() == true );
            delete drv;

            THEN( "no crash occurs" )
            {
                REQUIRE( true );
            }
        }
    }
}

/// Verify the execute() override of indiClient directly. indiClient is the small helper
/// class that indiDriver uses internally to send outgoing NEW property commands.
/// The client is connected to a local listener, activated, left to run once, and torn down.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "indiClient execute", "[app::indiDriver]" )
{
    TcpListener listener;

    MagAOX::app::indiClient client( "indiClientTest", "127.0.0.1", listener.port );

    client.activate();
    // Thread::runLoop() sleeps for its 1000 ms interval before its first call to execute().
    // The wait here must be longer than that so execute() runs at least once before the
    // thread is torn down.
    std::this_thread::sleep_for( std::chrono::milliseconds( 1200 ) );
    client.quitProcess();
    client.deactivate();

    REQUIRE( client.getQuitProcess() == true );
}

/// Verify the connect, reuse, and reconnect logic of sendNewProperty. The driver is pointed
/// at a real local TCP listener that never calls accept(). That is enough for the IndiClient
/// connect() call to succeed. The three GIVEN blocks cover a first send, a second send on a
/// good connection, and a send after the existing connection has quit.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "indiDriver sendNewProperty connects, reuses, and reconnects", "[app::indiDriver]" )
{
    std::filesystem::create_directories( "/tmp/indiDriver_test" );

    indiDriverTestParent parent;
    makeFreshFifo( "/tmp/indiDriver_test/snp.in" );
    makeFreshFifo( "/tmp/indiDriver_test/snp.out" );
    makeFreshFifo( "/tmp/indiDriver_test/snp.ctrl" );
    parent.m_driverInName   = "/tmp/indiDriver_test/snp.in";
    parent.m_driverOutName  = "/tmp/indiDriver_test/snp.out";
    parent.m_driverCtrlName = "/tmp/indiDriver_test/snp.ctrl";

    indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );
    REQUIRE( drv.good() == true );

    TcpListener listener;
    drv.setServer( "127.0.0.1", listener.port );

    pcf::IndiProperty ip = makeNumberProperty( "test", "nprop" );

    GIVEN( "no existing outgoing connection" )
    {
        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "a client is created and the send succeeds" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( drv.hasOutGoing() == true );
            }
        }
    }

    GIVEN( "an existing, still-good outgoing connection" )
    {
        REQUIRE( drv.sendNewProperty( ip ) == 0 );

        WHEN( "sendNewProperty is called again" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the existing connection is reused" )
            {
                REQUIRE( rv == 0 );
            }
        }
    }

    GIVEN( "an existing outgoing connection which has quit" )
    {
        REQUIRE( drv.sendNewProperty( ip ) == 0 );
        drv.forceOutGoingQuit();

        WHEN( "sendNewProperty is called again" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the old connection is torn down and a new one is created" )
            {
                REQUIRE( rv == 0 );
                REQUIRE( drv.hasOutGoing() == true );
            }
        }
    }
}

/// Verify the exception and disconnect handling paths in sendNewProperty. Each GIVEN block
/// builds a fresh driver. A closed port, a property of Unknown type, and the XWCTEST_INDIDRIVER_
/// fault-injection flags are used to reach each catch block and each early return.
/**
 * \ingroup indiDriver_tests
 */
SCENARIO( "indiDriver sendNewProperty exception handling", "[app::indiDriver]" )
{
    std::filesystem::create_directories( "/tmp/indiDriver_test" );

    GIVEN( "no listener at all on the configured server address" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/exc0.in" );
        makeFreshFifo( "/tmp/indiDriver_test/exc0.out" );
        makeFreshFifo( "/tmp/indiDriver_test/exc0.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/exc0.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/exc0.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/exc0.ctrl";

        indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        // Nothing is listening on port 1. The connection attempt made while creating the
        // outgoing IndiClient fails. The generic catch(...) around client creation handles it.
        drv.setServer( "127.0.0.1", 1 );

        pcf::IndiProperty ip = makeNumberProperty( "test", "nprop" );

        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the exception thrown while connecting is caught and -1 is returned" )
            {
                REQUIRE( rv == -1 );
                REQUIRE( drv.hasOutGoing() == false );
            }
        }
    }

    GIVEN( "connection creation succeeds but is then forced to null" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/excnull.in" );
        makeFreshFifo( "/tmp/indiDriver_test/excnull.out" );
        makeFreshFifo( "/tmp/indiDriver_test/excnull.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/excnull.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/excnull.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/excnull.ctrl";

        indiDriverExposed<indiDriverTestParent, MagAOX::app::XWCTEST_INDIDRIVER_OUTGOING_NULL_ns::indiDriver> drv(
            &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        TcpListener listener;
        drv.setServer( "127.0.0.1", listener.port );

        pcf::IndiProperty ip = makeNumberProperty( "test", "nprop" );

        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the defensive null check fires and -1 is returned" )
            {
                REQUIRE( rv == -1 );
                REQUIRE( drv.hasOutGoing() == false );
            }
        }
    }

    GIVEN( "a property of Unknown type, which the underlying INDI library rejects" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/exc.in" );
        makeFreshFifo( "/tmp/indiDriver_test/exc.out" );
        makeFreshFifo( "/tmp/indiDriver_test/exc.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/exc.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/exc.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/exc.ctrl";

        indiDriverExposed<indiDriverTestParent> drv( &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        TcpListener listener;
        drv.setServer( "127.0.0.1", listener.port );

        pcf::IndiProperty badProp; // A default-constructed property has type Unknown.

        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( badProp );

            THEN( "the std::exception thrown by the INDI library is caught and -1 is returned" )
            {
                REQUIRE( rv == -1 );
            }
        }
    }

    GIVEN( "a forced non-std::exception thrown from the send call" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/exc3.in" );
        makeFreshFifo( "/tmp/indiDriver_test/exc3.out" );
        makeFreshFifo( "/tmp/indiDriver_test/exc3.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/exc3.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/exc3.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/exc3.ctrl";

        indiDriverExposed<indiDriverTestParent, MagAOX::app::XWCTEST_INDIDRIVER_SEND_NONSTD_THROW_ns::indiDriver> drv(
            &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        TcpListener listener;
        drv.setServer( "127.0.0.1", listener.port );

        pcf::IndiProperty ip = makeNumberProperty( "test", "nprop" );

        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the catch(...) branch is hit and -1 is returned" )
            {
                REQUIRE( rv == -1 );
            }
        }
    }

    GIVEN( "the outgoing connection quits immediately after the send call" )
    {
        indiDriverTestParent parent;
        makeFreshFifo( "/tmp/indiDriver_test/exc4.in" );
        makeFreshFifo( "/tmp/indiDriver_test/exc4.out" );
        makeFreshFifo( "/tmp/indiDriver_test/exc4.ctrl" );
        parent.m_driverInName   = "/tmp/indiDriver_test/exc4.in";
        parent.m_driverOutName  = "/tmp/indiDriver_test/exc4.out";
        parent.m_driverCtrlName = "/tmp/indiDriver_test/exc4.ctrl";

        indiDriverExposed<indiDriverTestParent, MagAOX::app::XWCTEST_INDIDRIVER_QUIT_AFTER_SEND_ns::indiDriver> drv(
            &parent, "test", "0", "0" );
        REQUIRE( drv.good() == true );

        TcpListener listener;
        drv.setServer( "127.0.0.1", listener.port );

        pcf::IndiProperty ip = makeNumberProperty( "test", "nprop" );

        WHEN( "sendNewProperty is called" )
        {
            int rv = drv.sendNewProperty( ip );

            THEN( "the disconnect is detected and -1 is returned" )
            {
                REQUIRE( rv == -1 );
            }
        }
    }
}
