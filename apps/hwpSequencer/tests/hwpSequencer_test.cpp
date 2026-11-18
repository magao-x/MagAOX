/** \file hwpSequencer_test.cpp
 * \brief Catch2 tests for the hwpSequencer app.
 *
 * \ingroup hwpSequencer_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../hwpSequencer.hpp"
#include "../../../tests/testMacrosINDI.hpp"
#include "../../../tests/testXWC.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

namespace libXWCTest
{
namespace hwpSequencerTest
{

namespace
{
/// Build a per-test scratch directory under /tmp.
std::string testRoot( const std::string &name )
{
    std::string root = "/tmp/hwpSequencer_test/" + name;
    std::filesystem::create_directories( root );
    return root;
}

/// Format a number without trailing zeros, e.g. 45 or 22.5.
std::string num( double val )
{
    std::ostringstream str;
    str << val;
    return str.str();
}

} // namespace

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness recording the device-facing actions of hwpSequencer and returning scripted results.
class hwpSequencer_test : public MagAOX::app::hwpSequencer
{
  public:
    typedef MagAOX::app::hwpSequencer::waitResult waitResult;

    typedef MagAOX::app::hwpSequencer::endReason endReason;

    std::vector<std::string> m_actions; ///< Recorded actions, in order.

    std::mutex m_actionMutex; ///< Protects m_actions for the threaded test.

    waitResult m_frameResult{ waitResult::done }; ///< Result returned by waitFrame.

    int m_hwpTimeoutAt{ 0 }; ///< 1-based waitHwp call that times out, 0 for never.

    std::atomic<int> m_hwpWaits{ 0 }; ///< Number of waitHwp calls.

    std::atomic<int> m_dwells{ 0 }; ///< Number of dwell calls, written by the sequencer thread.

    bool m_realDwell{ false }; ///< Use the real dwell instead of returning done.

    std::function<waitResult( int )> m_dwellHook; ///< Optional dwell behavior, given the 1-based dwell count.

    using MagAOX::app::hwpSequencer::m_abortRequested;
    using MagAOX::app::hwpSequencer::m_cycle;
    using MagAOX::app::hwpSequencer::m_frameMargin;
    using MagAOX::app::hwpSequencer::m_frameMaxTimeout;
    using MagAOX::app::hwpSequencer::m_fxngenFreq;
    using MagAOX::app::hwpSequencer::m_fxngenOutp;
    using MagAOX::app::hwpSequencer::m_fxngenTimeout;
    using MagAOX::app::hwpSequencer::m_haveFxngenFreq;
    using MagAOX::app::hwpSequencer::m_haveFxngenOutp;
    using MagAOX::app::hwpSequencer::m_haveHwpPos;
    using MagAOX::app::hwpSequencer::m_haveObsOn;
    using MagAOX::app::hwpSequencer::m_hwpPositions;
    using MagAOX::app::hwpSequencer::m_hwpTimeout;
    using MagAOX::app::hwpSequencer::m_hwpTolerance;
    using MagAOX::app::hwpSequencer::m_hwpTracker;
    using MagAOX::app::hwpSequencer::m_hwpWait;
    using MagAOX::app::hwpSequencer::m_lastCycle;
    using MagAOX::app::hwpSequencer::m_numCycles;
    using MagAOX::app::hwpSequencer::m_obsOn;
    using MagAOX::app::hwpSequencer::m_posIndex;
    using MagAOX::app::hwpSequencer::m_reportedHwpPos;
    using MagAOX::app::hwpSequencer::m_sequencerThreadInit;
    using MagAOX::app::hwpSequencer::m_sequencing;
    using MagAOX::app::hwpSequencer::m_smSemaphore;
    using MagAOX::app::hwpSequencer::m_startRequested;
    using MagAOX::app::hwpSequencer::m_stopRequested;
    using MagAOX::app::hwpSequencer::m_timePerPos;
    using MagAOX::app::hwpSequencer::m_wakeSemaphore;
    using MagAOX::app::hwpSequencer::m_weStartedObs;

    using MagAOX::app::hwpSequencer::canStart;
    using MagAOX::app::hwpSequencer::finishSequence;
    using MagAOX::app::hwpSequencer::frameTimeout;
    using MagAOX::app::hwpSequencer::runSequence;
    using MagAOX::app::hwpSequencer::sequencerThreadExec;
    using MagAOX::app::hwpSequencer::stepTo;
    using MagAOX::app::hwpSequencer::waitFor;

    /// Construct a testable hwpSequencer with the given device name.
    explicit hwpSequencer_test( const std::string &device )
    {
        m_configName = device;

        sem_init( &m_smSemaphore, 0, 0 );
        sem_init( &m_wakeSemaphore, 0, 0 );

        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_sequence, sequence );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_lastCycle, last_cycle );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_abort, abort );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_timePerPos, time_per_pos );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_numCycles, num_cycles );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_hwpPosition, hwptrack, hwp_position );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_fxngenOutpStatus, fxngensync, C2outp );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_fxngenFreq, fxngensync, C2freq );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_obsOn, observers, obs_on );
    }

    /// Destroy the semaphores.
    ~hwpSequencer_test() noexcept
    {
        sem_destroy( &m_smSemaphore );
        sem_destroy( &m_wakeSemaphore );
    }

    /// Put the app in a state where a sequence can start.
    void makeReady( const std::vector<float> &positions )
    {
        m_hwpPositions   = positions;
        m_timePerPos     = 1.0;
        m_numCycles      = 0;
        m_fxngenFreq     = 10.0;
        m_haveFxngenFreq = true;
        m_haveFxngenOutp = true;
        m_haveHwpPos     = true;
        m_haveObsOn      = true;
        m_obsOn          = false;
    }

    /// Set the shutdown flag.
    void setShutdown( int sd )
    {
        m_shutdown = sd;
    }

    /// Run setupConfig.
    void setupTestConfig()
    {
        setupConfig();
    }

    /// Read a config file into the app's configurator.
    void readConfigFile( const std::string &path )
    {
        config.readConfig( path );
    }

    /// Run loadConfigImpl on the app's configurator.
    int loadOwnConfig()
    {
        return loadConfigImpl( config );
    }

    /// Call the real waitFrame.
    waitResult realWaitFrame( double timeout )
    {
        return MagAOX::app::hwpSequencer::waitFrame( timeout );
    }

    /// Call the real dwell.
    waitResult realDwell( double seconds )
    {
        return MagAOX::app::hwpSequencer::dwell( seconds );
    }

    /// Call the real waitHwp.
    waitResult realWaitHwp( float target )
    {
        return MagAOX::app::hwpSequencer::waitHwp( target );
    }

    /// Get a copy of the recorded actions.
    std::vector<std::string> actions()
    {
        std::lock_guard<std::mutex> guard( m_actionMutex );
        return m_actions;
    }

    /// Count the recorded actions equal to \p act.
    size_t count( const std::string &act )
    {
        std::vector<std::string> acts = actions();
        return std::count( acts.begin(), acts.end(), act );
    }

    /// Clear the recorded actions.
    void clearActions()
    {
        std::lock_guard<std::mutex> guard( m_actionMutex );
        m_actions.clear();
    }

  protected:
    /// Record an action.
    void record( const std::string &act )
    {
        std::lock_guard<std::mutex> guard( m_actionMutex );
        m_actions.push_back( act );
    }

    int sendTrigger( bool on ) override
    {
        record( on ? "trigger on" : "trigger off" );
        m_fxngenOutp = on;
        return 0;
    }

    int sendHwpTarget( float target ) override
    {
        record( "hwp " + num( target ) );
        return 0;
    }

    int sendObsOn( bool on ) override
    {
        record( on ? "obs on" : "obs off" );
        return 0;
    }

    waitResult waitTrigger( bool on ) override
    {
        record( on ? "wait trigger on" : "wait trigger off" );
        return interrupted() ? waitResult::interrupted : waitResult::done;
    }

    waitResult waitHwp( float target ) override
    {
        static_cast<void>( target );

        record( "wait hwp" );
        ++m_hwpWaits;

        if( m_hwpWaits == m_hwpTimeoutAt )
        {
            return waitResult::timeout;
        }

        return waitResult::done;
    }

    waitResult waitFrame( double timeout ) override
    {
        static_cast<void>( timeout );

        record( "wait frame" );
        return m_frameResult;
    }

    waitResult dwell( double seconds ) override
    {
        record( "dwell " + num( seconds ) );
        ++m_dwells;

        if( m_dwellHook )
        {
            return m_dwellHook( m_dwells.load() );
        }

        if( m_realDwell )
        {
            return MagAOX::app::hwpSequencer::dwell( seconds );
        }

        return waitResult::done;
    }
};
/// \endcond

/// Build a toggle switch property for a callback.
pcf::IndiProperty toggleProp( const std::string &device, const std::string &name, bool on )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return ip;
}

/// Build a number property with a single element for a callback.
pcf::IndiProperty numberProp( const std::string &device, const std::string &name, const std::string &el, double val )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( el ) );
    ip[el].set( val );
    return ip;
}

/// Validate the hwpSequencer INDI callbacks reject wrong devices and names.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer INDI callbacks validate their source properties", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_sequence( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_lastCycle( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_abort( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_timePerPos( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_numCycles( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_hwpPosition( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_fxngenOutpStatus( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_fxngenFreq( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_obsOn( pcf::IndiProperty() );
#endif
    // clang-format on

    XWCTEST_INDI_ARBNEW_CALLBACK( hwpSequencer, newCallBack_m_indiP_sequence, sequence );
    XWCTEST_INDI_ARBNEW_CALLBACK( hwpSequencer, newCallBack_m_indiP_lastCycle, last_cycle );
    XWCTEST_INDI_ARBNEW_CALLBACK( hwpSequencer, newCallBack_m_indiP_abort, abort );
    XWCTEST_INDI_ARBNEW_CALLBACK( hwpSequencer, newCallBack_m_indiP_timePerPos, time_per_pos );
    XWCTEST_INDI_ARBNEW_CALLBACK( hwpSequencer, newCallBack_m_indiP_numCycles, num_cycles );
    XWCTEST_INDI_SET_CALLBACK( hwpSequencer, m_indiP_hwpPosition, hwptrack, hwp_position );
    XWCTEST_INDI_SET_CALLBACK( hwpSequencer, m_indiP_fxngenOutpStatus, fxngensync, C2outp );
    XWCTEST_INDI_SET_CALLBACK( hwpSequencer, m_indiP_fxngenFreq, fxngensync, C2freq );
    XWCTEST_INDI_SET_CALLBACK( hwpSequencer, m_indiP_obsOn, observers, obs_on );
}

/// Validate that the set callbacks record remote device state and the parameter callbacks apply their rules.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer callbacks record remote state and validate parameters", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_hwpPosition( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_fxngenFreq( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::setCallBack_m_indiP_obsOn( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_timePerPos( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_numCycles( pcf::IndiProperty() );
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_lastCycle( pcf::IndiProperty() );
#endif
    // clang-format on

    hwpSequencer_test app( "hwpseq" );

    SECTION( "hwp_position, C2outp, C2freq and obs_on are recorded" )
    {
        REQUIRE( app.setCallBack_m_indiP_hwpPosition( numberProp( "hwptrack", "hwp_position", "current", 22.5 ) ) ==
                 0 );
        REQUIRE( app.m_haveHwpPos.load() );
        REQUIRE( app.m_reportedHwpPos.load() == Approx( 22.5 ) );

        pcf::IndiProperty outp( pcf::IndiProperty::Text );
        outp.setDevice( "fxngensync" );
        outp.setName( "C2outp" );
        outp.add( pcf::IndiElement( "value" ) );
        outp["value"].set( std::string( "On" ) );
        REQUIRE( app.setCallBack_m_indiP_fxngenOutpStatus( outp ) == 0 );
        REQUIRE( app.m_haveFxngenOutp.load() );
        REQUIRE( app.m_fxngenOutp.load() );

        REQUIRE( app.setCallBack_m_indiP_fxngenFreq( numberProp( "fxngensync", "C2freq", "current", 5.0 ) ) == 0 );
        REQUIRE( app.m_haveFxngenFreq.load() );
        REQUIRE( app.m_fxngenFreq.load() == Approx( 5.0 ) );

        REQUIRE( app.setCallBack_m_indiP_obsOn( toggleProp( "observers", "obs_on", true ) ) == 0 );
        REQUIRE( app.m_haveObsOn.load() );
        REQUIRE( app.m_obsOn.load() );
    }

    SECTION( "obs_on turning off mid-sequence releases ownership" )
    {
        app.m_sequencing   = true;
        app.m_weStartedObs = true;

        REQUIRE( app.setCallBack_m_indiP_obsOn( toggleProp( "observers", "obs_on", false ) ) == 0 );
        REQUIRE_FALSE( app.m_weStartedObs.load() );
    }

    SECTION( "time_per_pos below one trigger period is rejected while sequencing" )
    {
        app.makeReady( { 0, 45 } );
        app.m_timePerPos = 2.0;
        app.m_fxngenFreq = 0.1; // 10 s period
        app.m_sequencing = true;

        REQUIRE( app.newCallBack_m_indiP_timePerPos( numberProp( "hwpseq", "time_per_pos", "target", 5.0 ) ) == 0 );
        REQUIRE( app.m_timePerPos.load() == Approx( 2.0 ) );

        REQUIRE( app.newCallBack_m_indiP_timePerPos( numberProp( "hwpseq", "time_per_pos", "target", 20.0 ) ) == 0 );
        REQUIRE( app.m_timePerPos.load() == Approx( 20.0 ) );
    }

    SECTION( "num_cycles accepts 0 and rejects negative values" )
    {
        REQUIRE( app.newCallBack_m_indiP_numCycles( numberProp( "hwpseq", "num_cycles", "target", 3 ) ) == 0 );
        REQUIRE( app.m_numCycles.load() == 3 );

        REQUIRE( app.newCallBack_m_indiP_numCycles( numberProp( "hwpseq", "num_cycles", "target", 0 ) ) == 0 );
        REQUIRE( app.m_numCycles.load() == 0 );

        REQUIRE( app.newCallBack_m_indiP_numCycles( numberProp( "hwpseq", "num_cycles", "target", -1 ) ) == 0 );
        REQUIRE( app.m_numCycles.load() == 0 );
    }

    SECTION( "last_cycle is ignored while idle and set while sequencing" )
    {
        REQUIRE( app.newCallBack_m_indiP_lastCycle( toggleProp( "hwpseq", "last_cycle", true ) ) == 0 );
        REQUIRE_FALSE( app.m_lastCycle.load() );

        app.m_sequencing = true;

        REQUIRE( app.newCallBack_m_indiP_lastCycle( toggleProp( "hwpseq", "last_cycle", true ) ) == 0 );
        REQUIRE( app.m_lastCycle.load() );

        REQUIRE( app.newCallBack_m_indiP_lastCycle( toggleProp( "hwpseq", "last_cycle", false ) ) == 0 );
        REQUIRE_FALSE( app.m_lastCycle.load() );
    }
}

/// Validate hwpSequencer configuration defaults, overrides and validation.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer configuration loading", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::setupConfig();
    MagAOX::app::hwpSequencer::loadConfigImpl( mx::app::appConfigurator() );
#endif
    // clang-format on

    SECTION( "defaults" )
    {
        std::string root = testRoot( "config_defaults" );
        mx::app::writeConfigFile( root + "/hwpseq.conf", {}, {}, {} );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == 0 );
        REQUIRE( app.m_hwpTracker == "hwptrack" );
        REQUIRE( app.m_hwpPositions == std::vector<float>( { 0, 45, 22.5, 67.5 } ) );
        REQUIRE( app.m_hwpTolerance == Approx( 0.5 ) );
        REQUIRE( app.m_hwpTimeout == 5000 );
        REQUIRE( app.m_hwpWait == 100 );
        REQUIRE( app.m_fxngenTimeout == 5000 );
        REQUIRE( app.m_frameMargin == Approx( 0.5 ) );
        REQUIRE( app.m_frameMaxTimeout == Approx( 610 ) );
    }

    SECTION( "overrides" )
    {
        std::string root = testRoot( "config_overrides" );
        mx::app::writeConfigFile( root + "/hwpseq.conf",
                                  { "hwp", "hwp", "hwp", "hwp", "fxngen", "frame", "frame" },
                                  { "positions", "tolerance", "timeout", "wait", "timeout", "margin", "maxTimeout" },
                                  { "0,45", "0.25", "3000", "50", "2000", "1.5", "120" } );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == 0 );
        REQUIRE( app.m_hwpPositions == std::vector<float>( { 0, 45 } ) );
        REQUIRE( app.m_hwpTolerance == Approx( 0.25 ) );
        REQUIRE( app.m_hwpTimeout == 3000 );
        REQUIRE( app.m_hwpWait == 50 );
        REQUIRE( app.m_fxngenTimeout == 2000 );
        REQUIRE( app.m_frameMargin == Approx( 1.5 ) );
        REQUIRE( app.m_frameMaxTimeout == Approx( 120 ) );
    }

    SECTION( "a single position is allowed" )
    {
        std::string root = testRoot( "config_single" );
        mx::app::writeConfigFile( root + "/hwpseq.conf", { "hwp" }, { "positions" }, { "45" } );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == 0 );
        REQUIRE( app.m_hwpPositions == std::vector<float>( { 45 } ) );
    }

    SECTION( "repeated positions are rejected" )
    {
        std::string root = testRoot( "config_repeat" );
        mx::app::writeConfigFile( root + "/hwpseq.conf", { "hwp" }, { "positions" }, { "0,45,0,45" } );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == -1 );
    }

    SECTION( "out-of-range positions are rejected" )
    {
        std::string root = testRoot( "config_range" );
        mx::app::writeConfigFile( root + "/hwpseq.conf", { "hwp" }, { "positions" }, { "0,400" } );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == -1 );
    }

    SECTION( "a non-positive tolerance is rejected" )
    {
        std::string root = testRoot( "config_tolerance" );
        mx::app::writeConfigFile( root + "/hwpseq.conf", { "hwp" }, { "tolerance" }, { "0" } );

        hwpSequencer_test app( "hwpseq" );
        app.setupTestConfig();
        app.readConfigFile( root + "/hwpseq.conf" );

        REQUIRE( app.loadOwnConfig() == -1 );
    }
}

/// Validate the bounded waits: frame, condition polling, HWP position and dwell.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer waits are bounded and interruptible", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::waitFrame( 0 );
    MagAOX::app::hwpSequencer::waitFor( 0, 0, 0, 0 );
    MagAOX::app::hwpSequencer::waitHwp( 0 );
    MagAOX::app::hwpSequencer::dwell( 0 );
    MagAOX::app::hwpSequencer::frameTimeout();
#endif
    // clang-format on

    typedef hwpSequencer_test::waitResult waitResult;

    hwpSequencer_test app( "hwpseq" );

    SECTION( "waitFrame returns done for a posted frame" )
    {
        sem_post( &app.m_smSemaphore );
        REQUIRE( app.realWaitFrame( 1.0 ) == waitResult::done );
    }

    SECTION( "waitFrame times out when no frame arrives" )
    {
        auto t0 = std::chrono::steady_clock::now();
        REQUIRE( app.realWaitFrame( 0.1 ) == waitResult::timeout );
        REQUIRE( std::chrono::steady_clock::now() - t0 < std::chrono::seconds( 2 ) );
    }

    SECTION( "waitFrame treats an abort wake-up as interrupted, not a frame" )
    {
        app.m_abortRequested = true;
        sem_post( &app.m_smSemaphore );
        REQUIRE( app.realWaitFrame( 1.0 ) == waitResult::interrupted );
    }

    SECTION( "waitFrame returns promptly on shutdown" )
    {
        app.setShutdown( 1 );
        REQUIRE( app.realWaitFrame( 10.0 ) == waitResult::interrupted );
    }

    SECTION( "waitFor returns done, timeout, or interrupted" )
    {
        REQUIRE( hwpSequencer_test::waitFor(
                     []()
                     {
                         return true;
                     },
                     []()
                     {
                         return false;
                     },
                     1.0,
                     1 ) == waitResult::done );
        REQUIRE( hwpSequencer_test::waitFor(
                     []()
                     {
                         return false;
                     },
                     []()
                     {
                         return false;
                     },
                     0.05,
                     1 ) == waitResult::timeout );
        REQUIRE( hwpSequencer_test::waitFor(
                     []()
                     {
                         return false;
                     },
                     []()
                     {
                         return true;
                     },
                     10.0,
                     1 ) == waitResult::interrupted );
    }

    SECTION( "waitHwp uses the tolerance" )
    {
        app.m_hwpTolerance   = 0.5;
        app.m_hwpTimeout     = 50;
        app.m_hwpWait        = 1;
        app.m_reportedHwpPos = 45.3;

        REQUIRE( app.realWaitHwp( 45 ) == waitResult::done );
        REQUIRE( app.realWaitHwp( 46 ) == waitResult::timeout );
    }

    SECTION( "dwell reaches its deadline, or ends early on stop, abort or shutdown" )
    {
        REQUIRE( app.realDwell( 0.05 ) == waitResult::done );

        app.m_stopRequested = true;
        REQUIRE( app.realDwell( 10.0 ) == waitResult::stopped );
        app.m_stopRequested = false;

        app.m_abortRequested = true;
        REQUIRE( app.realDwell( 10.0 ) == waitResult::interrupted );
        app.m_abortRequested = false;

        app.setShutdown( 1 );
        REQUIRE( app.realDwell( 10.0 ) == waitResult::interrupted );
    }

    SECTION( "frameTimeout follows the trigger period, capped, with a fallback" )
    {
        app.m_frameMargin     = 0.5;
        app.m_frameMaxTimeout = 610;

        app.m_fxngenFreq = 0;
        REQUIRE( app.frameTimeout() == Approx( 610 ) );

        app.m_fxngenFreq = 2;
        REQUIRE( app.frameTimeout() == Approx( 1.0 ) );

        app.m_fxngenFreq = 0.001;
        REQUIRE( app.frameTimeout() == Approx( 610 ) );
    }
}

/// Validate the start conditions checked before a sequence.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer refuses to start unless all start conditions hold", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::canStart( std::string() );
#endif
    // clang-format on

    hwpSequencer_test app( "hwpseq" );
    app.makeReady( { 0, 45 } );

    std::string why;

    REQUIRE( app.canStart( why ) );

    SECTION( "no hwp_position" )
    {
        app.m_haveHwpPos = false;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "no trigger output state" )
    {
        app.m_haveFxngenOutp = false;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "no trigger frequency" )
    {
        app.m_haveFxngenFreq = false;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "trigger frequency not positive" )
    {
        app.m_fxngenFreq = 0;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "no obs_on" )
    {
        app.m_haveObsOn = false;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "already observing" )
    {
        app.m_obsOn = true;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "time_per_pos not positive" )
    {
        app.m_timePerPos = 0;
        REQUIRE_FALSE( app.canStart( why ) );
    }

    SECTION( "time_per_pos shorter than one trigger period" )
    {
        app.m_fxngenFreq = 0.5; // 2 s period
        app.m_timePerPos = 1.0;
        REQUIRE_FALSE( app.canStart( why ) );
    }
}

/// Validate the order of actions in one step.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer steps pause the trigger, wait for the frame, move, and resume", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::stepTo( 0, false );
#endif
    // clang-format on

    typedef hwpSequencer_test::waitResult waitResult;

    hwpSequencer_test app( "hwpseq" );
    app.makeReady( { 0, 45 } );

    SECTION( "first step turns obs_on on after the trigger resumes" )
    {
        REQUIRE( app.stepTo( 45, true ) == 0 );
        REQUIRE( app.actions() == std::vector<std::string>( { "trigger off",
                                                              "wait trigger off",
                                                              "wait frame",
                                                              "hwp 45",
                                                              "wait hwp",
                                                              "trigger on",
                                                              "wait trigger on",
                                                              "obs on" } ) );
        REQUIRE( app.m_weStartedObs.load() );
    }

    SECTION( "later steps do not touch obs_on" )
    {
        REQUIRE( app.stepTo( 45, false ) == 0 );
        REQUIRE( app.count( "obs on" ) == 0 );
    }

    SECTION( "no frame in flight is not an error" )
    {
        app.m_frameResult = waitResult::timeout;
        REQUIRE( app.stepTo( 45, false ) == 0 );
    }

    SECTION( "an HWP timeout is an error" )
    {
        app.m_hwpTimeoutAt = 1;
        REQUIRE( app.stepTo( 45, false ) == -1 );
        REQUIRE( app.count( "trigger on" ) == 0 );
    }
}

/// Validate how sequences end: cycle counts, live parameter changes, stop, abort and errors.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer sequences end as requested", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::runSequence( std::vector<float>() );
#endif
    // clang-format on

    typedef hwpSequencer_test::waitResult waitResult;
    typedef hwpSequencer_test::endReason  endReason;

    hwpSequencer_test app( "hwpseq" );
    app.makeReady( { 0, 45 } );

    std::vector<float> positions = app.m_hwpPositions;

    SECTION( "num_cycles reached" )
    {
        app.m_numCycles = 2;
        REQUIRE( app.runSequence( positions ) == endReason::completed );
        REQUIRE( app.m_dwells.load() == 4 );
        REQUIRE( app.m_cycle.load() == 2 );
    }

    SECTION( "num_cycles changed mid-run applies at the end of the cycle" )
    {
        app.m_dwellHook = [&app]( int n )
        {
            if( n == 3 )
            {
                app.m_numCycles = 1;
            }
            return waitResult::done;
        };

        REQUIRE( app.runSequence( positions ) == endReason::completed );
        REQUIRE( app.m_dwells.load() == 4 );
    }

    SECTION( "last_cycle finishes the current cycle" )
    {
        app.m_dwellHook = [&app]( int n )
        {
            if( n == 1 )
            {
                app.m_lastCycle = true;
            }
            return waitResult::done;
        };

        REQUIRE( app.runSequence( positions ) == endReason::completed );
        REQUIRE( app.m_dwells.load() == 2 );
    }

    SECTION( "time_per_pos changed mid-run applies at the next dwell" )
    {
        app.m_numCycles = 1;
        app.m_dwellHook = [&app]( int n )
        {
            if( n == 1 )
            {
                app.m_timePerPos = 2.0;
            }
            return waitResult::done;
        };

        REQUIRE( app.runSequence( positions ) == endReason::completed );
        REQUIRE( app.count( "dwell 1" ) == 1 );
        REQUIRE( app.count( "dwell 2" ) == 1 );
    }

    SECTION( "stop during a dwell ends the sequence" )
    {
        app.m_dwellHook = []( int n )
        {
            return n == 2 ? waitResult::stopped : waitResult::done;
        };

        REQUIRE( app.runSequence( positions ) == endReason::stopped );
        REQUIRE( app.m_dwells.load() == 2 );
    }

    SECTION( "stop is honoured at the next step boundary" )
    {
        app.m_dwellHook = [&app]( int n )
        {
            static_cast<void>( n );
            app.m_stopRequested = true;
            return waitResult::done;
        };

        REQUIRE( app.runSequence( positions ) == endReason::stopped );
        REQUIRE( app.count( "trigger off" ) == 1 );
    }

    SECTION( "abort during a dwell ends the sequence as aborted" )
    {
        app.m_dwellHook = [&app]( int n )
        {
            static_cast<void>( n );
            app.m_abortRequested = true;
            return waitResult::interrupted;
        };

        REQUIRE( app.runSequence( positions ) == endReason::aborted );
    }

    SECTION( "a failed step ends the sequence with an error" )
    {
        app.m_hwpTimeoutAt = 2;
        REQUIRE( app.runSequence( positions ) == endReason::error );
        REQUIRE( app.m_dwells.load() == 1 );
    }
}

/// Validate the cleanup performed for each way a sequence ends.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer cleanup depends on how the sequence ended", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::finishSequence( MagAOX::app::hwpSequencer::endReason::completed, std::vector<float>() );
#endif
    // clang-format on

    typedef hwpSequencer_test::endReason endReason;

    hwpSequencer_test app( "hwpseq" );
    app.makeReady( { 22.5, 45 } );

    std::vector<float> positions = app.m_hwpPositions;

    app.m_sequencing   = true;
    app.m_weStartedObs = true;

    SECTION( "a stop waits for the final frame, stops observing and returns the HWP to the first position" )
    {
        REQUIRE( app.finishSequence( endReason::stopped, positions ) == 0 );
        REQUIRE( app.actions() ==
                 std::vector<std::string>( { "trigger on", "wait trigger on", "wait frame", "obs off", "hwp 22.5" } ) );
        REQUIRE_FALSE( app.m_sequencing.load() );
        REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
        REQUIRE_FALSE( app.stateAlert() );
    }

    SECTION( "an abort skips the final frame but still returns the HWP" )
    {
        app.m_abortRequested = true;

        REQUIRE( app.finishSequence( endReason::aborted, positions ) == 0 );
        REQUIRE( app.count( "wait frame" ) == 0 );
        REQUIRE( app.count( "obs off" ) == 1 );
        REQUIRE( app.count( "hwp 22.5" ) == 1 );
        REQUIRE_FALSE( app.m_abortRequested.load() );
        REQUIRE_FALSE( app.stateAlert() );
    }

    SECTION( "an error skips the final frame, leaves the HWP in place and raises an alert" )
    {
        REQUIRE( app.finishSequence( endReason::error, positions ) == 0 );
        REQUIRE( app.count( "wait frame" ) == 0 );
        REQUIRE( app.count( "obs off" ) == 1 );
        REQUIRE( app.count( "hwp 22.5" ) == 0 );
        REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
        REQUIRE( app.stateAlert() );
    }

    SECTION( "obs_on is left alone if the sequencer does not own it" )
    {
        app.m_weStartedObs = false;

        REQUIRE( app.finishSequence( endReason::completed, positions ) == 0 );
        REQUIRE( app.count( "obs off" ) == 0 );
    }
}

/// Validate a full start and stop through the real callbacks and sequencer thread.
/**
 * \ingroup hwpSequencer_unit_test
 */
TEST_CASE( "hwpSequencer starts and stops through its callbacks without blocking", "[hwpSequencer]" )
{
    // clang-format off
#ifdef HWPSEQUENCER_TEST_DOXYGEN_REF
    MagAOX::app::hwpSequencer::sequencerThreadExec();
    MagAOX::app::hwpSequencer::newCallBack_m_indiP_sequence( pcf::IndiProperty() );
#endif
    // clang-format on

    hwpSequencer_test app( "hwpseq" );
    app.makeReady( { 0, 45 } );
    app.m_timePerPos          = 30.0;
    app.m_realDwell           = true;
    app.m_sequencerThreadInit = false;

    std::thread thrd( &hwpSequencer_test::sequencerThreadExec, &app );

    auto waitUntil = []( std::function<bool()> cond )
    {
        auto t0 = std::chrono::steady_clock::now();

        while( !cond() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds( 5 ) )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }

        return cond();
    };

    auto t0 = std::chrono::steady_clock::now();
    REQUIRE( app.newCallBack_m_indiP_sequence( toggleProp( "hwpseq", "sequence", true ) ) == 0 );
    REQUIRE( std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds( 100 ) );

    REQUIRE( waitUntil(
        [&app]()
        {
            return app.m_dwells.load() > 0;
        } ) );
    REQUIRE( app.m_sequencing.load() );

    t0 = std::chrono::steady_clock::now();
    REQUIRE( app.newCallBack_m_indiP_sequence( toggleProp( "hwpseq", "sequence", false ) ) == 0 );
    REQUIRE( std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds( 100 ) );

    REQUIRE( waitUntil(
        [&app]()
        {
            return !app.m_sequencing.load();
        } ) );

    app.setShutdown( 1 );
    sem_post( &app.m_wakeSemaphore );
    thrd.join();

    REQUIRE( app.count( "obs on" ) == 1 );
    REQUIRE( app.count( "obs off" ) == 1 );
    REQUIRE( app.count( "hwp 0" ) == 2 ); // first step, then the return after the stop
    REQUIRE_FALSE( app.stateAlert() );
}

} // namespace hwpSequencerTest
} // namespace libXWCTest
