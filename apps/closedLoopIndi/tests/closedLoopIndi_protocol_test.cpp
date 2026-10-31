/** \file closedLoopIndi_protocol_test.cpp
 * \brief Offline guarded-input protocol tests using real callbacks and a capturing INDI driver.
 * \ingroup closedLoopIndi_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"

#include <filesystem>
#include <fstream>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace closedLoopProtocol
{
/// Deterministic Unix time for arming and input-expiration tests.
inline double now = 1800000000;

/// Replace only the app header's clock reads with the test clock.
inline int clock( clockid_t id /**< [in] Clock requested by production code. */,
                  timespec *value /**< [out] Current test time. */ )
{
    if( id != CLOCK_REALTIME )
        return ::clock_gettime( id, value );
    value->tv_sec  = static_cast<time_t>( now );
    value->tv_nsec = static_cast<long>( ( now - value->tv_sec ) * 1e9 );
    return 0;
}
} // namespace closedLoopProtocol
#define clock_gettime closedLoopProtocol::clock
#include "../closedLoopIndi.hpp"
#undef clock_gettime

namespace closedLoopProtocol
{
/// Unactivated INDI driver that captures actual outgoing commands and publications.
class Driver : public MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>
{
  public:
    /// Construct using private fixture files without starting an outgoing client.
    Driver( MagAOX::app::MagAOXApp<true> *parent /**< [in] Fixture application. */ );

    /// Release the input descriptor that the unactivated base does not close.
    ~Driver();

    /// Capture production New properties, optionally failing a selected send.
    int sendNewProperty( const pcf::IndiProperty &property /**< [in] Actual command property. */ ) override;

    /// Capture production XML for property publications.
    void sendXml( const std::string &xml /**< [in] Serialized publication. */ ) const override;

    std::vector<pcf::IndiProperty> m_commands; ///< Ordered command attempts from the real app.
    mutable std::string            m_xml;      ///< Serialized INDI output from production publication code.
    int m_failSend{ 0 };                       ///< One-based command attempt to fail; zero disables failure injection.
    int m_input{ -1 };                         ///< Owned input descriptor for the unactivated driver.
};

Driver::Driver( MagAOX::app::MagAOXApp<true> *parent ) : indiDriver( parent, "closedloop-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( ::stat( parent->driverInName().c_str(), &input ) == 0 );
    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        int         descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};
        if( ::fstat( descriptor, &candidate ) == 0 && candidate.st_ino == input.st_ino &&
            candidate.st_dev == input.st_dev )
            m_input = descriptor;
    }
}

Driver::~Driver()
{
    if( m_input >= 0 )
        ::close( m_input );
    setInputFd( -1 );
}

int Driver::sendNewProperty( const pcf::IndiProperty &property )
{
    m_commands.push_back( property );
    return static_cast<int>( m_commands.size() ) == m_failSend ? -1 : 0;
}

void Driver::sendXml( const std::string &xml ) const
{
    m_xml += xml;
}

/// Fixture running production startup and callbacks with temporary files and captured commands.
class Fixture : public MagAOX::app::closedLoopIndi
{
  public:
    /// Configure guarded or legacy operation through the actual configuration loader.
    Fixture( bool               guarded = true, /**< [in] Enable the opt-in protocol. */
             const std::string &matrix =
                 "intMat00=2\nintMat01=1\nintMat10=-1\nintMat11=3\n" /**< [in] Matrix configuration. */ );

    /// Release private transport before removing fixture files.
    ~Fixture();

    /// Construct a complete measurement at the current test time.
    pcf::IndiProperty sample( double counter,        /**< [in] Sequence number. */
                              double session = 1000, /**< [in] Producer epoch. */
                              double x       = 2,    /**< [in] First residual. */
                              double y       = -4 /**< [in] Second residual. */ );

    /// Deliver a finite or malformed axis-current report through the real callback.
    void current( unsigned           axis, /**< [in] Zero-based axis. */
                  const std::string &value /**< [in] Reported current text. */ );

    /// Deliver controller FSM state through the real callback.
    void controller( const std::string &state /**< [in] Controller FSM state. */ );

    /// Deliver upstream readiness through the real callback.
    void upstream( bool ready /**< [in] Upstream toggle state. */ );

    /// Deliver an actual operator loop-state request.
    void close( bool value = true /**< [in] Requested loop state. */ );

    /// Return whether production code considers the loop closed.
    bool closed() const;

    /// Return a field from the most recent production acknowledgment.
    double ack( const std::string &name /**< [in] Acknowledgment element. */ ) const;

    /// Parse the captured production XML publications.
    std::vector<pcf::IndiProperty> publications();

    int         m_loadResult{ 0 };    ///< Result of the actual configuration validator.
    Driver     *m_capture{ nullptr }; ///< Driver owned through the application pointer.
    std::string m_directory;          ///< Private temporary configuration and transport files.
};

Fixture::Fixture( bool guarded, const std::string &matrix )
{
    now         = 1800000000;
    char path[] = "/tmp/closedloop-protocol-XXXXXX";
    REQUIRE( ::mkdtemp( path ) != nullptr );
    m_directory  = path;
    m_configName = "test-ctrl";
    m_basePath   = m_directory;
    m_configDir  = m_directory;
    setupConfig();
    std::string   configPath = m_directory + "/test.conf";
    std::ofstream output( configPath );
    output << "[input]\ndevice=sensor\nproperty=measurement\nelements=delta_x,delta_y\nguarded="
           << ( guarded ? "true" : "false" )
           << "\nmaxAge=5\n[ctrl]\ndevices=ttmperi\nproperties=axis1_voltage,axis2_voltage\noperatingOK=true\n"
              "[loop]\ngain=0.1\ngains=1,1\n"
           << matrix << "upstream=sensor\nupstreamProperty=ready\nupstreamFollowClosed=true\n";
    output.close();
    REQUIRE( config.readConfig( configPath ) == 0 );
    m_loadResult = loadConfigImpl( config );
    if( m_loadResult < 0 )
        return;
    REQUIRE( appStartup() == 0 );
    m_indiP_state = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_state.setDevice( m_configName );
    m_indiP_state.setName( "fsm" );
    m_indiP_state.add( pcf::IndiElement( "state" ) );
    m_driverInName   = m_directory + "/input";
    m_driverOutName  = m_directory + "/output";
    m_driverCtrlName = m_directory + "/control";
    for( const auto &file : { m_driverInName, m_driverOutName, m_driverCtrlName } )
        std::ofstream( file ).close();
    m_capture    = new Driver( this );
    m_indiDriver = m_capture;
    REQUIRE( m_capture->good() );
    current( 0, "75" );
    current( 1, "80" );
    controller( "OPERATING" );
}

Fixture::~Fixture()
{
    delete m_indiDriver;
    m_indiDriver = nullptr;
    std::filesystem::remove_all( m_directory );
}

pcf::IndiProperty Fixture::sample( double counter, double session, double x, double y )
{
    pcf::IndiProperty value( pcf::IndiProperty::Number, "sensor", "measurement" );
    for( const auto &name : { "delta_x", "delta_y", "counter", "session", "timestamp", "valid" } )
        value.add( pcf::IndiElement( name ) );
    value["delta_x"]   = x;
    value["delta_y"]   = y;
    value["counter"]   = counter;
    value["session"]   = session;
    value["timestamp"] = now;
    value["valid"]     = 1;
    return value;
}

void Fixture::current( unsigned axis, const std::string &value )
{
    pcf::IndiProperty report( pcf::IndiProperty::Number, "ttmperi", axis ? "axis2_voltage" : "axis1_voltage" );
    report.add( pcf::IndiElement( "current", value ) );
    REQUIRE( ( axis ? setCallBack_m_indiP_ctrl1( report ) : setCallBack_m_indiP_ctrl0( report ) ) == 0 );
}

void Fixture::controller( const std::string &state )
{
    pcf::IndiProperty report( pcf::IndiProperty::Text, "ttmperi", "fsm" );
    report.add( pcf::IndiElement( "state", state ) );
    REQUIRE( setCallBack_m_indiP_ctrl0_fsm( report ) == 0 );
}

void Fixture::upstream( bool ready )
{
    pcf::IndiProperty report( pcf::IndiProperty::Switch, "sensor", "ready" );
    report.add( pcf::IndiElement( "toggle" ) );
    report["toggle"].setSwitchState( ready ? pcf::IndiElement::On : pcf::IndiElement::Off );
    REQUIRE( setCallBack_m_indiP_upstream( report ) == 0 );
}

void Fixture::close( bool value )
{
    pcf::IndiProperty request( pcf::IndiProperty::Switch, "test-ctrl", "loop_state" );
    request.add( pcf::IndiElement( "toggle" ) );
    request["toggle"].setSwitchState( value ? pcf::IndiElement::On : pcf::IndiElement::Off );
    REQUIRE( newCallBack_m_indiP_ctrlEnabled( request ) == 0 );
}

bool Fixture::closed() const
{
    return m_loopClosed;
}

double Fixture::ack( const std::string &name ) const
{
    return m_indiP_processed[name].get<double>();
}

std::vector<pcf::IndiProperty> Fixture::publications()
{
    pcf::IndiXmlParser             parser( "1.7" );
    std::string                    error;
    std::vector<pcf::IndiProperty> result;
    for( char byte : m_capture->m_xml )
    {
        parser.parseXml( &byte, 1, error );
        REQUIRE( error.empty() );
        if( parser.getState() == pcf::IndiXmlParser::CompleteState )
        {
            result.push_back( parser.createIndiMessage().getProperty() );
            parser.clear();
        }
    }
    m_capture->m_xml.clear();
    return result;
}
} // namespace closedLoopProtocol
/// \endcond

namespace libXWCTest
{
namespace closedLoopIndiTest
{
using closedLoopProtocol::Fixture;
using closedLoopProtocol::now;

/// Verify arming never applies retained input and a new sample produces the actual configured command pair.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi requires a new post-arm sample", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::setCallBack_m_indiP_inputs( pcf::IndiProperty() );
    MagAOX::app::closedLoopIndi::toggleLoop( true );
    MagAOX::app::closedLoopIndi::sendCommands( std::vector<float>() );
    #endif
    // clang-format on
    Fixture app;
    app.close();
    REQUIRE_FALSE( app.closed() );
    auto retained = app.sample( 1 );
    REQUIRE( app.setCallBack_m_indiP_inputs( retained ) == 0 );
    REQUIRE( app.ack( "status" ) == 4 );
    app.upstream( true );
    REQUIRE_FALSE( app.closed() );
    now += 1;
    app.close();
    REQUIRE( app.closed() );
    REQUIRE( app.m_capture->m_commands.empty() );
    REQUIRE( app.setCallBack_m_indiP_inputs( retained ) == 0 );
    REQUIRE( app.m_capture->m_commands.empty() );

    auto delayed         = app.sample( 2 );
    delayed["timestamp"] = now - 0.5;
    REQUIRE( app.setCallBack_m_indiP_inputs( delayed ) == 0 );
    REQUIRE( app.ack( "status" ) == 5 );
    REQUIRE( app.m_capture->m_commands.empty() );
    now += 1;
    auto fresh = app.sample( 3 );
    REQUIRE( app.setCallBack_m_indiP_inputs( fresh ) == 0 );
    REQUIRE( app.m_capture->m_commands.size() == 2 );
    REQUIRE( app.m_capture->m_commands[0].getDevice() == "ttmperi" );
    REQUIRE( app.m_capture->m_commands[0].getName() == "axis1_voltage" );
    REQUIRE( app.m_capture->m_commands[0]["target"].get<float>() == Approx( 75 ) );
    REQUIRE( app.m_capture->m_commands[1].getName() == "axis2_voltage" );
    REQUIRE( app.m_capture->m_commands[1]["target"].get<float>() == Approx( 81.4 ) );
    REQUIRE( app.ack( "session" ) == 1000 );
    REQUIRE( app.ack( "counter" ) == 3 );
    REQUIRE( app.ack( "applied" ) == 1 );
    REQUIRE( app.ack( "target1" ) == Approx( 81.4 ) );
    REQUIRE( app.setCallBack_m_indiP_inputs( fresh ) == 0 );
    REQUIRE( app.m_capture->m_commands.size() == 2 );
    REQUIRE( app.ack( "applied" ) == 1 );

    auto   publications = app.publications();
    size_t acks         = 0;
    for( const auto &property : publications )
    {
        if( property.getName() != "processed" )
            continue;
        ++acks;
        for( const auto &name : { "counter", "session", "target0", "target1", "applied", "status" } )
            REQUIRE( property.find( name ) );
    }
    REQUIRE( acks == 5 );
}

/// Verify invalid metadata and data open the loop and never reach actual command transmission.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi rejects invalid coherent inputs", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::guardedInput( pcf::IndiProperty() );
    MagAOX::app::closedLoopIndi::readNumber( double(), pcf::IndiProperty(), "valid" );
    #endif
    // clang-format on
    Fixture app;
    REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 1 ) ) == 0 );
    app.close();
    now += 1;
    auto invalid = app.sample( 2 );
    SECTION( "explicit invalid uses the previous counter" )
    {
        invalid["valid"]   = 0;
        invalid["counter"] = 1;
    }
    SECTION( "not a number residual" )
    {
        invalid["delta_x"] = "nan";
    }
    SECTION( "infinite residual" )
    {
        invalid["delta_y"] = "inf";
    }
    SECTION( "overflowing finite float residual" )
    {
        invalid["delta_y"] = "1e100";
    }
    SECTION( "malformed value" )
    {
        invalid["counter"] = "broken";
    }
    SECTION( "trailing numeric junk" )
    {
        invalid["session"] = "1000junk";
    }
    SECTION( "fractional counter" )
    {
        invalid["counter"] = 1.5;
    }
    SECTION( "negative counter" )
    {
        invalid["counter"] = -1;
    }
    SECTION( "stale timestamp" )
    {
        invalid["timestamp"] = now - 6;
    }
    SECTION( "future timestamp" )
    {
        invalid["timestamp"] = now + 1;
    }
    SECTION( "missing metadata" )
    {
        invalid = pcf::IndiProperty( pcf::IndiProperty::Number, "sensor", "measurement" );
    }
    REQUIRE( app.setCallBack_m_indiP_inputs( invalid ) == 0 );
    REQUIRE_FALSE( app.closed() );
    REQUIRE( app.m_capture->m_commands.empty() );
    app.close();
    REQUIRE_FALSE( app.closed() );
}

/// Verify session changes, reordered input, and expired input cannot rearm or reuse a previous command.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi isolates sessions and expires readiness", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::guardedInput( pcf::IndiProperty() );
    MagAOX::app::closedLoopIndi::appLogic();
    MagAOX::app::closedLoopIndi::newCallBack_m_indiP_counterReset( pcf::IndiProperty() );
    #endif
    // clang-format on
    Fixture app;
    REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 10 ) ) == 0 );
    app.close();
    now += 1;
    SECTION( "new session requires manual rearm" )
    {
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 1, 1001 ) ) == 0 );
        REQUIRE_FALSE( app.closed() );
        REQUIRE( app.ack( "status" ) == 4 );
        app.close();
        REQUIRE( app.closed() );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 11, 1000 ) ) == 0 );
        REQUIRE_FALSE( app.closed() );
    }
    SECTION( "older counter does not command" )
    {
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 9 ) ) == 0 );
        REQUIRE_FALSE( app.closed() );
    }
    SECTION( "higher counter cannot reuse old measurement time" )
    {
        auto reordered         = app.sample( 11 );
        reordered["timestamp"] = now - 1;
        REQUIRE( app.setCallBack_m_indiP_inputs( reordered ) == 0 );
        REQUIRE_FALSE( app.closed() );
    }
    SECTION( "counter reset preserves guarded replay protection" )
    {
        pcf::IndiProperty reset( pcf::IndiProperty::Switch, "test-ctrl", "counter_reset" );
        reset.add( pcf::IndiElement( "request" ) );
        reset["request"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_counterReset( reset ) == 0 );
        REQUIRE_FALSE( app.closed() );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 10 ) ) == 0 );
        app.close();
        REQUIRE_FALSE( app.closed() );
    }
    SECTION( "watchdog expires readiness even without more messages" )
    {
        now += 5;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE_FALSE( app.closed() );
        app.close();
        REQUIRE_FALSE( app.closed() );
    }
    REQUIRE( app.m_capture->m_commands.empty() );
}

/// Verify both readiness/sample orders keep invalidity open without automatic recovery commands.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi readiness ordering cannot auto close", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::setCallBack_m_indiP_upstream( pcf::IndiProperty() );
    #endif
    // clang-format on
    Fixture app;
    REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 1 ) ) == 0 );
    app.close();
    now += 1;
    SECTION( "Off before a delayed valid sample" )
    {
        app.upstream( false );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2 ) ) == 0 );
    }
    SECTION( "invalid sample before Off" )
    {
        auto invalid     = app.sample( 1 );
        invalid["valid"] = 0;
        REQUIRE( app.setCallBack_m_indiP_inputs( invalid ) == 0 );
        app.upstream( false );
    }
    app.upstream( true );
    REQUIRE_FALSE( app.closed() );
    REQUIRE( app.m_capture->m_commands.empty() );
}

/// Verify all no-command outcomes acknowledge the input and a failed send opens the loop.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi acknowledges processing outcomes", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::updateLoop();
    MagAOX::app::closedLoopIndi::publishProcessed( 0, 0, 0, std::vector<float>() );
    #endif
    // clang-format on
    Fixture app;
    REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 1 ) ) == 0 );
    app.close();
    now += 1;
    SECTION( "open loop acknowledges without applying" )
    {
        app.close( false );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2 ) ) == 0 );
        REQUIRE( app.ack( "status" ) == 0 );
    }
    SECTION( "unready controller acknowledges without applying" )
    {
        app.controller( "NOTCONNECTED" );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2 ) ) == 0 );
        REQUIRE( app.ack( "status" ) == 2 );
        app.controller( "OPERATING" );
        app.close();
        REQUIRE_FALSE( app.closed() );
        app.current( 0, "75" );
        app.current( 1, "80" );
        app.close();
        REQUIRE( app.closed() );
    }
    SECTION( "nonfinite current cannot become a command" )
    {
        app.current( 1, "nan" );
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2 ) ) == 0 );
        REQUIRE( app.ack( "status" ) == 2 );
    }
    SECTION( "zero residual acknowledges with unchanged targets" )
    {
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2, 1000, 0, 0 ) ) == 0 );
        REQUIRE( app.ack( "status" ) == 3 );
        REQUIRE( app.ack( "target0" ) == 75 );
        REQUIRE( app.ack( "target1" ) == 80 );
    }
    SECTION( "overflowing computed command is rejected before either axis is sent" )
    {
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2, 1000, 0.75 * std::numeric_limits<float>::max(), 0 ) ) ==
                 -1 );
        REQUIRE( app.ack( "status" ) == -1 );
        REQUIRE_FALSE( app.closed() );
    }
    SECTION( "failed second-axis send reports partial transmission failure" )
    {
        app.m_capture->m_failSend = 2;
        REQUIRE( app.setCallBack_m_indiP_inputs( app.sample( 2 ) ) == -1 );
        REQUIRE( app.ack( "status" ) == -2 );
        REQUIRE_FALSE( app.closed() );
        REQUIRE( app.m_capture->m_commands.size() == 2 );
        return;
    }
    REQUIRE( app.ack( "counter" ) == 2 );
    REQUIRE( app.ack( "applied" ) == 0 );
    REQUIRE( app.m_capture->m_commands.empty() );
}

/// Verify guarded operation requires an explicit finite nonsingular matrix while legacy defaults remain available.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "guarded closedLoopIndi requires a calibrated matrix", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::loadConfigImpl( mx::app::appConfigurator() );
    #endif
    // clang-format on
    SECTION( "all entries absent" )
    {
        Fixture app( true, "" );
        REQUIRE( app.m_loadResult == -1 );
    }
    SECTION( "one entry absent" )
    {
        Fixture app( true, "intMat00=1\nintMat01=0\nintMat10=0\n" );
        REQUIRE( app.m_loadResult == -1 );
    }
    SECTION( "singular matrix" )
    {
        Fixture app( true, "intMat00=1\nintMat01=2\nintMat10=2\nintMat11=4\n" );
        REQUIRE( app.m_loadResult == -1 );
    }
    SECTION( "nonfinite matrix" )
    {
        Fixture app( true, "intMat00=1\nintMat01=nan\nintMat10=0\nintMat11=1\n" );
        REQUIRE( app.m_loadResult == -1 );
    }
    SECTION( "malformed matrix" )
    {
        Fixture app( true, "intMat00=1\nintMat01=oops\nintMat10=0\nintMat11=1\n" );
        REQUIRE( app.m_loadResult == -1 );
    }
    SECTION( "legacy identity default" )
    {
        Fixture app( false, "" );
        REQUIRE( app.m_loadResult == 0 );
    }
}

/// Verify the default legacy protocol still accepts its original three-field input and upstream auto-close.
/** \ingroup closedLoopIndi_unit_test */
TEST_CASE( "closedLoopIndi legacy input behavior is unchanged", "[closedLoopIndi][protocol]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    MagAOX::app::closedLoopIndi::setCallBack_m_indiP_inputs( pcf::IndiProperty() );
    MagAOX::app::closedLoopIndi::setCallBack_m_indiP_upstream( pcf::IndiProperty() );
    #endif
    // clang-format on
    Fixture app( false );
    app.upstream( true );
    REQUIRE( app.closed() );
    pcf::IndiProperty legacy( pcf::IndiProperty::Number, "sensor", "measurement" );
    for( const auto &name : { "delta_x", "delta_y", "counter" } )
        legacy.add( pcf::IndiElement( name ) );
    legacy["delta_x"] = 2;
    legacy["delta_y"] = -4;
    legacy["counter"] = 1;
    REQUIRE( app.setCallBack_m_indiP_inputs( legacy ) == 0 );
    REQUIRE( app.m_capture->m_commands.size() == 2 );
    REQUIRE( app.setCallBack_m_indiP_inputs( legacy ) == 0 );
    REQUIRE( app.m_capture->m_commands.size() == 2 );
}
} // namespace closedLoopIndiTest
} // namespace libXWCTest
