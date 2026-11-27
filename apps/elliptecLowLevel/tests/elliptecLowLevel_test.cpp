/** \file elliptecLowLevel_test.cpp
 * \brief Catch2 bus-scenario tests for elliptecLowLevel against simulated Elliptec devices.
 * \author Miles Lucas
 *
 * The simulated bus follows the ELLx protocol manual (Issue 12) and can reproduce the ELL21 behavior observed on
 * hardware: motion commands not acknowledged, queries mostly unanswered while moving, and false idle replies.
 *
 * \ingroup elliptecLowLevel_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <sys/file.h>

#include "../elliptecLowLevel.hpp"
#include "fakeElliptecBus.hpp"

namespace libXWCTest
{
namespace elliptecLowLevelTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace
{

/// Test harness exposing what the bus scenarios need from elliptecLowLevel.
class elliptecLowLevel_test : public MagAOX::app::elliptecLowLevel
{
  public:
    /// Construct an app on the simulated bus, with no stages yet.
    elliptecLowLevel_test( const std::string &name, /**< [in] app name and scratch directory */
                           const std::string &port /**< [in] PTY path */ )
    {
        const std::string root = "/tmp/elliptecLowLevel_test/" + name;
        std::filesystem::remove_all( root );
        std::filesystem::create_directories( root );

        m_configName       = name;
        m_basePath         = root;
        m_sysPath          = root + "/sys";
        m_deviceName       = port; // no USB ids: use the PTY directly
        m_readTimeoutMs    = 100;
        m_busyTimeoutMs    = 30;
        m_powerState       = 1;
        m_powerTargetState = 1;
        m_powerOnWait      = 0;
    }

    /// Close the port.
    ~elliptecLowLevel_test()
    {
        closePort_();
    }

    /// Add a stage on the bus.
    void addStage( const std::string &name, char addr, const std::string &serial = "", int vel = 80 )
    {
        MagAOX::app::elliptecDevice dev;
        dev.name( name );
        dev.address( addr );
        dev.expectedSerial( serial );
        dev.velPercent( vel );
        m_deviceByName[name] = m_devices.size();
        m_devices.push_back( dev );
    }

    /// Run appStartup from INITIALIZED.
    int startup()
    {
        state( MagAOX::app::stateCodes::INITIALIZED );
        return appStartup();
    }

    /// Run one loop and record each stage's published state.
    int logic()
    {
        int rv = appLogic();
        for( auto &dev : m_devices )
        {
            auto &h = m_history[dev.name()];
            auto  s = curr( dev.name() );
            if( h.empty() || h.back() != s )
            {
                h.push_back( s );
            }
        }
        return rv;
    }

    /// Run loops until a predicate holds.
    bool runUntil( const std::function<bool()> &done, int timeoutMs )
    {
        auto start = std::chrono::steady_clock::now();
        while( std::chrono::steady_clock::now() - start < std::chrono::milliseconds( timeoutMs ) )
        {
            logic();
            if( done() )
            {
                return true;
            }
            std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
        }
        return false;
    }

    /// Published state of a stage.
    std::string curr( const std::string &stage )
    {
        return m_indiP_curr_state[stage].getValue();
    }

    /// Published position of a stage.
    double pos( const std::string &stage )
    {
        return m_indiP_curr_pos[stage].get<double>();
    }

    /// Published error of a stage.
    std::string lastError( const std::string &stage )
    {
        return m_indiP_last_error[stage].getValue();
    }

    /// Whether a stage ever published a state.
    bool visited( const std::string &stage, const std::string &st )
    {
        for( auto &s : m_history[stage] )
        {
            if( s == st )
            {
                return true;
            }
        }
        return false;
    }

    /// Forget recorded state history.
    void clearHistory()
    {
        m_history.clear();
    }

    /// Device state for a stage.
    MagAOX::app::elliptecDevice &device( const std::string &stage )
    {
        return m_devices[m_deviceByName[stage]];
    }

    /// Send `tgt_pos` for one stage.
    int target( const std::string &stage, double deg )
    {
        pcf::IndiProperty ip = m_indiP_tgt_pos;
        ip[stage].set( deg );
        return newCallBack_m_indiP_tgt_pos( ip );
    }

    /// Send an `On` request on a switch property for one stage.
    int request( pcf::IndiProperty &prop,
                 const std::string &stage,
                 int ( elliptecLowLevel_test::*cb )( const pcf::IndiProperty & ) )
    {
        pcf::IndiProperty ip = prop;
        ip[stage].setSwitchState( pcf::IndiElement::On );
        return ( this->*cb )( ip );
    }

    /// Request a home.
    int home( const std::string &stage )
    {
        return request( m_indiP_req_home, stage, &elliptecLowLevel_test::newCallBack_m_indiP_req_home );
    }

    /// Request a halt.
    int halt( const std::string &stage )
    {
        return request( m_indiP_req_halt, stage, &elliptecLowLevel_test::newCallBack_m_indiP_req_halt );
    }

    /// Request an optimize.
    int optimize( const std::string &stage )
    {
        return request( m_indiP_req_optimize, stage, &elliptecLowLevel_test::newCallBack_m_indiP_req_optimize );
    }

    /// Request a save.
    int save( const std::string &stage )
    {
        return request( m_indiP_req_save, stage, &elliptecLowLevel_test::newCallBack_m_indiP_req_save );
    }

    /// Set a stage's velocity.
    int velocity( const std::string &stage, int pct )
    {
        pcf::IndiProperty ip = m_indiP_velocity;
        ip[stage].set( pct );
        return newCallBack_m_indiP_velocity( ip );
    }

    /// Enable or disable the maintenance commands.
    void maintenance( bool on )
    {
        m_enableMaintenance = on;
    }

    /// Set the motion timeout.
    void moveTimeout( double sec )
    {
        m_moveTimeoutSec = sec;
    }

    /// Simulate the framework's power-off.
    int powerOffNow()
    {
        m_powerState       = 0;
        m_powerTargetState = 0;
        state( MagAOX::app::stateCodes::POWEROFF );
        return onPowerOff();
    }

    /// Simulate the framework's power-on.
    void powerOnNow()
    {
        m_powerState       = 1;
        m_powerTargetState = 1;
        state( MagAOX::app::stateCodes::POWERON );
    }

    /// Read a configuration file and load it.
    int loadConfigFile( const std::string &path /**< [in] configuration file */ )
    {
        config.readConfig( path );
        return loadConfigImpl( config );
    }

    /// Whether the port is open.
    bool portOpen() const
    {
        return m_fd >= 0;
    }

  private:
    std::map<std::string, std::vector<std::string>> m_history; ///< Published states per stage.
};

/// Start the app and wait until the given stages are identified.
void connectStages( elliptecLowLevel_test &app, const std::vector<std::string> &stages )
{
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            for( auto &s : stages )
            {
                if( !app.device( s ).identified() )
                {
                    return false;
                }
            }
            return true;
        },
        3000 ) );
    app.clearHistory();
}

/// Home a stage and wait until it is READY.
void homeStage( elliptecLowLevel_test &app, const std::string &stage )
{
    REQUIRE( app.home( stage ) == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( stage ) == "READY";
        },
        5000 ) );
}

} // namespace
/// \endcond

/// Two mounts on one bus are identified, homed, and moved independently.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel drives two mounts on one bus", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::appLogic();
    MagAOX::app::elliptecLowLevel::identify_( dev );
    MagAOX::app::elliptecLowLevel::poll_( dev );
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_tgt_pos( ip );
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_req_home( ip );
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '1', "12100017" );
    bus.addDevice( '2', "12100070" );
    bus.configure( '2',
                   []( simElliptec &d )
                   {
                       d.pos = 9000;
                   } );

    elliptecLowLevel_test app( "ell-two", bus.slavePath() );
    app.addStage( "qwp1", '1', "12100017" );
    app.addStage( "qwp2", '2', "12100070" );
    connectStages( app, { "qwp1", "qwp2" } );

    REQUIRE( app.curr( "qwp1" ) == "NOTHOMED" );
    REQUIRE( app.device( "qwp2" ).model() == 21 );
    REQUIRE( app.pos( "qwp2" ) == Approx( 9000.0 * 360 / 65536 ) );
    REQUIRE( bus.count( '1', "sv" ) == 1 );

    homeStage( app, "qwp1" );
    homeStage( app, "qwp2" );

    REQUIRE( app.target( "qwp1", 115.0 ) == 0 );
    REQUIRE( app.target( "qwp2", 250.0 ) == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp1" ) == "READY" && app.curr( "qwp2" ) == "READY";
        },
        5000 ) );

    REQUIRE( app.pos( "qwp1" ) == Approx( 115.0 ).margin( 0.01 ) );
    REQUIRE( app.pos( "qwp2" ) == Approx( 250.0 ).margin( 0.01 ) );
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
}

/// The ELL21's observed motion behavior: silent while moving, with occasional false idle replies.
/** A long home and move must neither drop the device nor complete early.
 *
 * \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel tolerates silent, falsely idle motion", "[elliptecLowLevel]" )
{
    fakeElliptecBus bus;
    bus.addDevice( '2', "12100070" );
    bus.configure( '2',
                   []( simElliptec &d )
                   {
                       d.pos      = 30000;
                       d.motionMs = 1500;
                       d.hardwareQuirks( 3 );
                   } );

    elliptecLowLevel_test app( "ell-quirks", bus.slavePath() );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp2" } );

    REQUIRE( app.home( "qwp2" ) == 0 );
    bool early = false;
    REQUIRE( app.runUntil(
        [&]()
        {
            early = early || ( app.device( "qwp2" ).homed() && bus.busy( '2' ) );
            return app.curr( "qwp2" ) == "READY";
        },
        8000 ) );
    REQUIRE_FALSE( early );

    REQUIRE( app.target( "qwp2", 180.0 ) == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp2" ) == "OPERATING";
        },
        1000 ) );
    REQUIRE( app.runUntil(
        [&]()
        {
            early = early || ( app.curr( "qwp2" ) == "READY" && bus.busy( '2' ) );
            return app.curr( "qwp2" ) == "READY";
        },
        8000 ) );
    REQUIRE_FALSE( early );
    REQUIRE( app.pos( "qwp2" ) == Approx( 180.0 ).margin( 0.01 ) );
    REQUIRE_FALSE( app.visited( "qwp2", "NOTCONNECTED" ) );
}

/// A silent, moving mount does not stall polling of the other, which moves only once the first has stopped.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel moves one mount at a time and keeps polling the other", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::requestMotion_( dev, op, deg );
    MagAOX::app::elliptecLowLevel::dispatch_();
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '1', "12100017" );
    bus.addDevice( '2', "12100070" );
    bus.configure( '1',
                   []( simElliptec &d )
                   {
                       d.motionMs = 1500;
                       d.hardwareQuirks( 0 );
                   } );
    bus.configure( '2',
                   []( simElliptec &d )
                   {
                       d.motionMs = 200;
                   } );

    elliptecLowLevel_test app( "ell-one-at-a-time", bus.slavePath() );
    app.addStage( "qwp1", '1' );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp1", "qwp2" } );

    REQUIRE( app.target( "qwp1", 90.0 ) == 0 );
    REQUIRE( app.target( "qwp2", 90.0 ) == 0 );
    REQUIRE( bus.count( '1', "ma" ) == 1 );
    REQUIRE( bus.count( '2', "ma" ) == 0 );
    REQUIRE( app.curr( "qwp2" ) == "OPERATING" ); // held, so clients wait for it
    REQUIRE( app.device( "qwp2" ).queued() == MagAOX::app::elliptecDevice::Pending::Move );

    // QWP2 is polled, but not moved, while QWP1 moves silently.
    size_t polls      = bus.count( '2', "gp" );
    bool   overlapped = false;
    REQUIRE( app.runUntil(
        [&]()
        {
            overlapped = overlapped || ( bus.busy( '1' ) && bus.busy( '2' ) );
            return app.device( "qwp1" ).moving() == 0;
        },
        5000 ) );
    REQUIRE( bus.count( '2', "gp" ) > polls + 5 );
    REQUIRE( app.pos( "qwp1" ) == Approx( 90.0 ).margin( 0.01 ) );

    REQUIRE( app.runUntil(
        [&]()
        {
            overlapped = overlapped || ( bus.busy( '1' ) && bus.busy( '2' ) );
            return app.device( "qwp2" ).moving() == 0 && app.curr( "qwp2" ) != "OPERATING";
        },
        3000 ) );
    REQUIRE_FALSE( overlapped );
    REQUIRE( bus.count( '2', "ma" ) == 1 );
    REQUIRE( app.pos( "qwp2" ) == Approx( 90.0 ).margin( 0.01 ) );
}

/// Held requests: the latest replaces an older one, devices take turns, and a halt drops the request.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel holds one request per mount and serves them in turn", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::requestMotion_( dev, op, deg );
    MagAOX::app::elliptecLowLevel::dispatch_();
    MagAOX::app::elliptecLowLevel::halt_( dev );
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '1', "12100017" );
    bus.addDevice( '2', "12100070" );
    for( char a : { '1', '2' } )
    {
        bus.configure( a,
                       []( simElliptec &d )
                       {
                           d.motionMs = 600;
                       } );
    }

    elliptecLowLevel_test app( "ell-held", bus.slavePath() );
    app.addStage( "qwp1", '1' );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp1", "qwp2" } );

    auto idle = [&]()
    {
        return app.device( "qwp1" ).moving() == 0 && app.device( "qwp2" ).moving() == 0 &&
               app.device( "qwp1" ).queued() == MagAOX::app::elliptecDevice::Pending::None &&
               app.device( "qwp2" ).queued() == MagAOX::app::elliptecDevice::Pending::None;
    };

    SECTION( "the latest request replaces a held one" )
    {
        REQUIRE( app.target( "qwp1", 30.0 ) == 0 );
        REQUIRE( app.target( "qwp2", 50.0 ) == 0 );
        REQUIRE( app.target( "qwp2", 70.0 ) == 0 );
        REQUIRE( app.runUntil( idle, 5000 ) );

        REQUIRE( bus.count( '2', "ma" ) == 1 );
        REQUIRE( app.pos( "qwp2" ) == Approx( 70.0 ).margin( 0.01 ) );
    }

    SECTION( "a moving mount is not retargeted; its new target waits its turn after the other mount" )
    {
        REQUIRE( app.target( "qwp1", 30.0 ) == 0 );
        REQUIRE( app.target( "qwp2", 50.0 ) == 0 );
        REQUIRE( app.target( "qwp1", 60.0 ) == 0 ); // held: QWP1 is moving
        REQUIRE( bus.count( '1', "ma" ) == 1 );

        // QWP2 waited longer, so it moves before QWP1's second target.
        REQUIRE( app.runUntil(
            [&]()
            {
                return bus.count( '2', "ma" ) == 1;
            },
            3000 ) );
        REQUIRE( bus.count( '1', "ma" ) == 1 );

        REQUIRE( app.runUntil( idle, 5000 ) );
        REQUIRE( bus.count( '1', "ma" ) == 2 );
        REQUIRE( app.pos( "qwp1" ) == Approx( 60.0 ).margin( 0.01 ) );
        REQUIRE( app.pos( "qwp2" ) == Approx( 50.0 ).margin( 0.01 ) );
    }

    SECTION( "a halt drops a held request" )
    {
        REQUIRE( app.target( "qwp1", 30.0 ) == 0 );
        REQUIRE( app.target( "qwp2", 50.0 ) == 0 );
        REQUIRE( app.halt( "qwp2" ) == 0 );
        REQUIRE( app.curr( "qwp2" ) == "NOTHOMED" );

        REQUIRE( app.runUntil( idle, 5000 ) );
        REQUIRE( bus.count( '2', "ma" ) == 0 );
    }

    SECTION( "a held home reports HOMING" )
    {
        REQUIRE( app.target( "qwp1", 30.0 ) == 0 );
        REQUIRE( app.home( "qwp2" ) == 0 );
        REQUIRE( app.curr( "qwp2" ) == "HOMING" );
        REQUIRE( bus.count( '2', "ho" ) == 0 );

        REQUIRE( app.runUntil(
            [&]()
            {
                return app.curr( "qwp2" ) == "READY";
            },
            5000 ) );
        REQUIRE( bus.count( '2', "ho" ) == 1 );
    }
}

/// One mount dropping out leaves the other working; it is re-identified when it answers again.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel isolates a mount that stops answering", "[elliptecLowLevel]" )
{
    fakeElliptecBus bus;
    bus.addDevice( '1', "12100017" );
    bus.addDevice( '2', "12100070" );

    elliptecLowLevel_test app( "ell-dropout", bus.slavePath() );
    app.addStage( "qwp1", '1' );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp1", "qwp2" } );
    homeStage( app, "qwp1" );

    bus.configure( '1',
                   []( simElliptec &d )
                   {
                       d.bootSilent = 12;
                   } );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp1" ) == "NOTCONNECTED";
        },
        5000 ) );
    REQUIRE( app.curr( "qwp2" ) == "NOTHOMED" );
    REQUIRE( app.portOpen() );

    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp1" ) == "READY";
        },
        5000 ) );
    REQUIRE( app.device( "qwp1" ).homed() ); // a communication loss does not require a re-home
    REQUIRE( bus.count( '1', "ho" ) == 1 );
}

/// A mount with the wrong serial is reported NODEVICE and never commanded.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel never commands a device with the wrong serial", "[elliptecLowLevel]" )
{
    fakeElliptecBus bus;
    bus.addDevice( '1', "99999999" );

    elliptecLowLevel_test app( "ell-serial", bus.slavePath() );
    app.addStage( "qwp1", '1', "12100017" );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp1" ) == "NODEVICE";
        },
        3000 ) );

    REQUIRE( app.home( "qwp1" ) == 0 );
    REQUIRE( app.target( "qwp1", 90.0 ) == 0 );
    for( int n = 0; n < 5; ++n )
    {
        app.logic();
    }

    REQUIRE( bus.count( '1', "ho" ) == 0 );
    REQUIRE( bus.count( '1', "ma" ) == 0 );
    REQUIRE( bus.count( '1', "in" ) == 1 );
}

/// A device that is slow to answer after power-on, or answers with a truncated info reply, is retried.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel retries identification", "[elliptecLowLevel]" )
{
    fakeElliptecBus bus;
    bus.addDevice( '2', "12100070" );
    bus.configure( '2',
                   []( simElliptec &d )
                   {
                       d.bootSilent   = 3;
                       d.truncateInfo = true;
                   } );

    elliptecLowLevel_test app( "ell-boot", bus.slavePath() );
    app.addStage( "qwp2", '2' );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.device( "qwp2" ).identified();
        },
        5000 ) );

    REQUIRE( app.device( "qwp2" ).pulsesPerRev() == 65536 );
    REQUIRE( bus.count( '2', "in" ) >= 5 );
}

/// Errors and timeouts are reported per stage without closing the port.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel reports device errors and motion timeouts", "[elliptecLowLevel]" )
{
    SECTION( "a mechanical timeout during home" )
    {
        fakeElliptecBus bus;
        bus.addDevice( '2', "12100070" );
        bus.configure( '2',
                       []( simElliptec &d )
                       {
                           d.failNextCode = 2;
                       } );

        elliptecLowLevel_test app( "ell-error", bus.slavePath() );
        app.addStage( "qwp2", '2' );
        connectStages( app, { "qwp2" } );

        REQUIRE( app.home( "qwp2" ) == 0 );
        REQUIRE( app.runUntil(
            [&]()
            {
                return !app.lastError( "qwp2" ).empty();
            },
            3000 ) );
        REQUIRE( app.lastError( "qwp2" ) == "home failed (mechanical time out)" );
        REQUIRE( app.curr( "qwp2" ) == "NOTHOMED" );
        REQUIRE( app.portOpen() );

        homeStage( app, "qwp2" );
        REQUIRE( app.lastError( "qwp2" ).empty() );
    }

    SECTION( "silence past the motion timeout" )
    {
        fakeElliptecBus bus;
        bus.addDevice( '2', "12100070" );
        bus.configure( '2',
                       []( simElliptec &d )
                       {
                           d.motionMs = 5000;
                           d.hardwareQuirks( 0 );
                       } );

        elliptecLowLevel_test app( "ell-timeout", bus.slavePath() );
        app.addStage( "qwp2", '2' );
        app.moveTimeout( 0.5 );
        connectStages( app, { "qwp2" } );

        REQUIRE( app.target( "qwp2", 90.0 ) == 0 );
        REQUIRE( app.runUntil(
            [&]()
            {
                return !app.lastError( "qwp2" ).empty();
            },
            3000 ) );
        REQUIRE( app.lastError( "qwp2" ) == "move failed (no response from the device during motion)" );
    }
}

/// Unsupported halt, velocity, and maintenance commands.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel halt, velocity, and maintenance", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_req_halt( ip );
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_velocity( ip );
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_req_optimize( ip );
    MagAOX::app::elliptecLowLevel::newCallBack_m_indiP_req_save( ip );
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '2', "12100070" );
    bus.configure( '2',
                   []( simElliptec &d )
                   {
                       d.motionMs   = 600;
                       d.optimizeMs = 800;
                   } );

    elliptecLowLevel_test app( "ell-maint", bus.slavePath() );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp2" } );
    homeStage( app, "qwp2" );

    // An unsupported halt leaves the move to complete.
    REQUIRE( app.target( "qwp2", 180.0 ) == 0 );
    REQUIRE( app.halt( "qwp2" ) == 0 );
    REQUIRE( bus.count( '2', "st" ) == 1 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp2" ) == "READY";
        },
        3000 ) );
    REQUIRE( app.pos( "qwp2" ) == Approx( 180.0 ).margin( 0.01 ) );
    REQUIRE( app.lastError( "qwp2" ).empty() );

    // Velocity requests are floored at the ELL21 minimum.
    REQUIRE( app.velocity( "qwp2", 10 ) == 0 );
    REQUIRE( app.device( "qwp2" ).velPercent() == 50 );
    REQUIRE( bus.count( '2', "sv" ) == 2 );

    // Maintenance is rejected unless enabled.
    REQUIRE( app.optimize( "qwp2" ) == 0 );
    REQUIRE( app.save( "qwp2" ) == 0 );
    REQUIRE( bus.count( '2', "om" ) == 0 );
    REQUIRE( bus.count( '2', "us" ) == 0 );

    app.maintenance( true );
    REQUIRE( app.optimize( "qwp2" ) == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return bus.count( '2', "om" ) == 1;
        },
        1000 ) );
    REQUIRE( app.target( "qwp2", 90.0 ) == 0 ); // rejected while optimizing
    REQUIRE( app.save( "qwp2" ) == 0 );         // rejected while optimizing
    REQUIRE( bus.count( '2', "ma" ) == 1 );
    REQUIRE( bus.count( '2', "us" ) == 0 );

    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp2" ) == "NOTHOMED";
        },
        3000 ) );
    REQUIRE( app.save( "qwp2" ) == 0 );
    REQUIRE( bus.count( '2', "us" ) == 1 );
}

/// Power-off publishes POWEROFF for every stage, and a power-on requires a new home.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel power cycle", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::onPowerOff();
    MagAOX::app::elliptecLowLevel::whilePowerOff();
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '1', "12100017" );
    bus.addDevice( '2', "12100070" );

    elliptecLowLevel_test app( "ell-power", bus.slavePath() );
    app.addStage( "qwp1", '1' );
    app.addStage( "qwp2", '2' );
    connectStages( app, { "qwp1", "qwp2" } );
    homeStage( app, "qwp1" );

    REQUIRE( app.powerOffNow() == 0 );
    REQUIRE( app.curr( "qwp1" ) == "POWEROFF" );
    REQUIRE( app.curr( "qwp2" ) == "POWEROFF" );
    REQUIRE_FALSE( app.portOpen() );
    REQUIRE( app.whilePowerOff() == 0 );
    REQUIRE( app.logic() == 0 ); // power off: no communication

    app.powerOnNow();
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.curr( "qwp1" ) == "NOTHOMED";
        },
        3000 ) );
    REQUIRE_FALSE( app.device( "qwp1" ).homed() );
}

/// The port is locked so a second process cannot share the bus.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel refuses a port locked by another process", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::openPort_();
    // clang-format on
#endif

    fakeElliptecBus bus;
    bus.addDevice( '2', "12100070" );

    int other = ::open( bus.slavePath().c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK );
    REQUIRE( other >= 0 );
    REQUIRE( ::flock( other, LOCK_EX | LOCK_NB ) == 0 );

    elliptecLowLevel_test app( "ell-lock", bus.slavePath() );
    app.addStage( "qwp2", '2' );
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.logic() == 0 );
    REQUIRE_FALSE( app.portOpen() );
    REQUIRE( app.state() == MagAOX::app::stateCodes::NOTCONNECTED );

    ::close( other );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.device( "qwp2" ).identified();
        },
        3000 ) );
}

/// Configuration: stage sections, address validation, and duplicate addresses.
/** \ingroup elliptecLowLevel_unit_test
 */
TEST_CASE( "elliptecLowLevel configuration", "[elliptecLowLevel]" )
{
#ifdef ELLIPTECLOWLEVEL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecLowLevel::setupConfig();
    MagAOX::app::elliptecLowLevel::loadConfigImpl( config );
    // clang-format on
#endif

    SECTION( "stage sections are found by their address key" )
    {
        elliptecLowLevel_test app( "ell-config", "/dev/null" );
        app.setupConfig();
        mx::app::writeConfigFile(
            "/tmp/elliptecLowLevel_test.conf",
            { "usb", "qwp1", "qwp1", "qwp1", "qwp2", "qwp2", "serial", "other" },
            { "baud", "address", "serial", "velPercent", "address", "velPercent", "trace", "key" },
            { "9600", "\"1\"", "12100017", "60", "2", "90", "true", "value" } );
        REQUIRE( app.loadConfigFile( "/tmp/elliptecLowLevel_test.conf" ) == 0 );
        REQUIRE( app.device( "qwp1" ).address() == '1' );
        REQUIRE( app.device( "qwp1" ).expectedSerial() == "12100017" );
        REQUIRE( app.device( "qwp1" ).velPercent() == 60 );
        REQUIRE( app.device( "qwp2" ).address() == '2' );
        REQUIRE( app.device( "qwp2" ).velPercent() == 90 );
    }

    SECTION( "invalid and duplicate addresses are rejected" )
    {
        {
            elliptecLowLevel_test app( "ell-config-bad", "/dev/null" );
            app.setupConfig();
            mx::app::writeConfigFile( "/tmp/elliptecLowLevel_test_bad.conf", { "qwp1" }, { "address" }, { "12" } );
            REQUIRE( app.loadConfigFile( "/tmp/elliptecLowLevel_test_bad.conf" ) == -1 );
        }
        {
            elliptecLowLevel_test app( "ell-config-dup", "/dev/null" );
            app.setupConfig();
            mx::app::writeConfigFile(
                "/tmp/elliptecLowLevel_test_dup.conf", { "qwp1", "qwp2" }, { "address", "address" }, { "1", "1" } );
            REQUIRE( app.loadConfigFile( "/tmp/elliptecLowLevel_test_dup.conf" ) == -1 );
        }
        {
            elliptecLowLevel_test app( "ell-config-none", "/dev/null" );
            app.setupConfig();
            mx::app::writeConfigFile( "/tmp/elliptecLowLevel_test_none.conf", { "serial" }, { "trace" }, { "true" } );
            REQUIRE( app.loadConfigFile( "/tmp/elliptecLowLevel_test_none.conf" ) == -1 );
        }
    }
}

} // namespace elliptecLowLevelTest
} // namespace libXWCTest
