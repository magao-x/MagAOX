/** \file koolanceCtrl_test.cpp
 * \brief Catch2 tests for the koolanceCtrl app.
 *
 * \ingroup koolanceCtrl_files
 */

#include <sys/socket.h>

#include "../../../tests/testXWC.hpp"

#include "../koolanceCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup koolanceCtrl_unit_test koolanceCtrl Unit Tests
 * \brief Unit tests for the koolanceCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `koolanceCtrl` unit tests.
/** \ingroup koolanceCtrl_unit_test
 */
namespace koolanceCtrlTest
{

/// \cond KOOLANCECTRL_TEST_HARNESS
namespace
{

/// Exercise cooler status reads and configuration with a software-only serial peer.
class Controller : public koolanceCtrl
{
  public:
    /// Name the simulated INDI device.
    Controller();

    /// Release both ends of the simulated serial connection.
    ~Controller();

    /// Set up options and load the averaging window from a command line.
    int configure( const std::string &seconds /**< [in] value of the averaging window option */ );

    /// Attach a local socket as the serial transport for the selected protocol.
    void connectPeer( size_t protocolChars /**< [in] expected controller response length */ );

    /// Queue a synthetic response and run the normal status read.
    int readFlow( int    tenths /**< [in] flow in tenths of a liter per minute */,
                  size_t replyChars /**< [in] response length to simulate */ );

    using koolanceCtrl::m_flowAverageSeconds;
    using koolanceCtrl::m_flowRate;
    using koolanceCtrl::m_indiNewCallBacks;
    using koolanceCtrl::m_indiP_status;
    using koolanceCtrl::updateFlowAverage;

  private:
    /// Peer descriptor used to inject responses into the controller's socket.
    int m_peer{ -1 };
};

Controller::Controller()
{
    m_configName = "simCool";
}

Controller::~Controller()
{
    if( m_fileDescrip > 0 )
    {
        ::close( m_fileDescrip );
    }

    if( m_peer >= 0 )
    {
        ::close( m_peer );
    }
}

int Controller::configure( const std::string &seconds )
{
    setupConfig();
    std::string name   = "koolanceCtrl";
    std::string option = "--flow.averageSeconds=" + seconds;
    char       *argv[]{ name.data(), option.data(), nullptr };
    config.parseCommandLine( 2, argv );

    return loadConfigImpl( config );
}

void Controller::connectPeer( size_t protocolChars )
{
    int sockets[2];
    REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sockets ) == 0 );
    m_fileDescrip   = sockets[0];
    m_peer          = sockets[1];
    m_protocolChars = protocolChars;
}

int Controller::readFlow( int tenths, size_t replyChars )
{
    std::string reply( replyChars, '\0' );
    reply[2]  = 0x08;
    reply[3]  = static_cast<char>( 0xAC ); // 2220 encodes a liquid temperature of 22 C.
    reply[8]  = 0x04;
    reply[10] = 0x08;
    reply[12] = static_cast<char>( tenths >> 8 );
    reply[13] = static_cast<char>( tenths & 0xFF );
    REQUIRE( ::write( m_peer, reply.data(), reply.size() ) == static_cast<ssize_t>( reply.size() ) );

    return getStatus();
}

} // namespace
/// \endcond

/// Verify elapsed-time averaging suppresses one low sample and exposes sustained low flow.
/**
 * \ingroup koolanceCtrl_unit_test
 */
TEST_CASE( "koolanceCtrl averages flow over the preceding 60 seconds by default", "[koolanceCtrl]" )
{
    // clang-format off
    #ifdef KOOLANCECTRL_TEST_DOXYGEN_REF
    koolanceCtrl::updateFlowAverage( flowTenths, now );
    #endif
    // clang-format on

    using namespace std::chrono;
    const steady_clock::time_point start{ seconds{ 100 } };
    Controller                     app;
    REQUIRE( app.m_flowAverageSeconds == 60 );

    for( int i = 0; i < 60; ++i )
    {
        REQUIRE( app.updateFlowAverage( 19, start + seconds{ i } ) == Approx( 1.9 ) );
    }

    CHECK( app.updateFlowAverage( 0, start + seconds{ 60 } ) == Approx( 59 * 1.9 / 60 ) );

    double average = 0;

    for( int i = 61; i < 120; ++i )
    {
        average = app.updateFlowAverage( 0, start + seconds{ i } );
    }

    CHECK( average == 0 );
    CHECK( app.updateFlowAverage( 21, start + seconds{ 180 } ) == Approx( 2.1 ) );
}

/// Verify configuration, expiration at the elapsed-time boundary, and raw passthrough.
/**
 * \ingroup koolanceCtrl_unit_test
 */
TEST_CASE( "koolanceCtrl flow averaging duration can be configured", "[koolanceCtrl]" )
{
    // clang-format off
    #ifdef KOOLANCECTRL_TEST_DOXYGEN_REF
    koolanceCtrl::setupConfig();
    koolanceCtrl::loadConfigImpl( config );
    koolanceCtrl::updateFlowAverage( flowTenths, now );
    #endif
    // clang-format on

    using namespace std::chrono;
    const steady_clock::time_point start{ seconds{ 100 } };
    Controller                     app;

    SECTION( "samples expire by age even when they arrive at irregular intervals" )
    {
        REQUIRE( app.configure( "10" ) == 0 );
        CHECK( app.updateFlowAverage( 20, start ) == Approx( 2.0 ) );
        CHECK( app.updateFlowAverage( 10, start + seconds{ 9 } ) == Approx( 1.5 ) );
        CHECK( app.updateFlowAverage( 30, start + seconds{ 10 } ) == Approx( 2.0 ) );
        CHECK( app.updateFlowAverage( 40, start + seconds{ 20 } ) == Approx( 4.0 ) );
    }

    SECTION( "zero seconds uses the latest sample even at the same timestamp" )
    {
        REQUIRE( app.configure( "0" ) == 0 );
        CHECK( app.updateFlowAverage( 19, start ) == Approx( 1.9 ) );
        CHECK( app.updateFlowAverage( 12, start ) == Approx( 1.2 ) );
    }

    SECTION( "negative, nonnumeric, fractional, and overflowing values are rejected" )
    {
        const auto option = GENERATE( "-1", "bad", "1.5", "99999999999999999999" );
        REQUIRE( app.configure( option ) < 0 );
    }
}

/// Verify both flow elements and the other status values are defined from a valid controller reply.
/**
 * \ingroup koolanceCtrl_unit_test
 */
TEST_CASE( "koolanceCtrl status exposes raw and averaged flow", "[koolanceCtrl]" )
{
    // clang-format off
    #ifdef KOOLANCECTRL_TEST_DOXYGEN_REF
    koolanceCtrl::getStatus();
    koolanceCtrl::publishStatus( rawFlowRate, flowRateAverage );
    #endif
    // clang-format on

    const size_t protocolChars = GENERATE( 43, 51 );
    Controller   app;
    app.connectPeer( protocolChars );
    REQUIRE( app.m_indiNewCallBacks.count( "simCool.status" ) == 0 );

    SECTION( "the first status definition has both values and the raw telemetry reading" )
    {
        app.updateFlowAverage( 20, std::chrono::steady_clock::now() );

        REQUIRE( app.readFlow( 12, protocolChars ) == 0 );
        REQUIRE( app.m_indiNewCallBacks.count( "simCool.status" ) == 1 );
        CHECK( app.m_indiP_status["flow_rate"].get<double>() == Approx( 1.6 ) );
        CHECK( app.m_indiP_status["flow_rate_raw"].get<double>() == Approx( 1.2 ) );
        CHECK( app.m_indiP_status["liquid_temp"].get<double>() == Approx( 22 ) );
        CHECK( app.m_indiP_status["pump_rpm"].get<int>() == 2048 );
        CHECK( app.m_indiP_status["fan_rpm"].get<int>() == 1024 );
        CHECK( app.m_flowRate == Approx( 1.2 ) );
    }

    SECTION( "an invalid response never supplies a flow sample or defines the status" )
    {
        CHECK( app.readFlow( 12, protocolChars + 1 ) < 0 );
        CHECK( app.m_indiNewCallBacks.count( "simCool.status" ) == 0 );
        CHECK( app.updateFlowAverage( 20, std::chrono::steady_clock::now() ) == Approx( 2.0 ) );
    }
}

} // namespace koolanceCtrlTest

} // namespace libXWCTest
