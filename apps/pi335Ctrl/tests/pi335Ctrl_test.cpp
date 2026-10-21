/** \file pi335Ctrl_test.cpp
 * \brief Offline PI controller command-origin, serial outcome and timing endpoint tests.
 * \ingroup pi335Ctrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"

#include <array>

/// \cond TEST_HARNESS
namespace pi335TimingTest
{
/// Scripted serial writes, including the endpoint bracket measured by the real timing collector.
struct Script
{
    std::array<int, 3>       m_results{};       ///< Return codes for the configured axis writes.
    std::vector<std::string> m_commands;        ///< Actual commands passed by the production move helpers.
    int64_t                  m_firstEntry{ 0 }; ///< First transport entry, after the production SDK-start marker.
    int64_t                  m_lastReturn{ 0 }; ///< Last transport completion, before the production SDK-end marker.
};

Script script; ///< Current test's software-only serial endpoint.
} // namespace pi335TimingTest

namespace MagAOX
{
namespace tty
{
/// Replace only the app's serial-write boundary; no file descriptor or hardware is accessed.
int pi335TestWrite( const std::string &command /**< [in] Actual position command. */,
                    int                fd /**< [in] Unused production descriptor. */,
                    int                timeout /**< [in] Unused production write timeout. */ )
{
    static_cast<void>( fd );
    static_cast<void>( timeout );
    auto      &script = pi335TimingTest::script;
    const auto index  = script.m_commands.size();
    if( index == 0 )
        script.m_firstEntry = app::dev::dmTiming::monotonicNowNs();
    script.m_commands.push_back( command );
    script.m_lastReturn = app::dev::dmTiming::monotonicNowNs();
    return script.m_results[index % script.m_results.size()];
}
} // namespace tty
} // namespace MagAOX

#define ttyWrite pi335TestWrite
#include "../pi335Ctrl.hpp"
#undef ttyWrite

namespace MagAOX
{
namespace app
{
/// Access the existing friend seam without starting the app, serial connection or monitor threads.
class pi335Ctrl_test : public pi335Ctrl
{
  public:
    using dev::dm<pi335Ctrl, float>::m_dmTiming;
    using pi335Ctrl::m_dmWritesAttempted;
    using pi335Ctrl::m_dmWritesSuccessful;

    /// Configure software axes and an ordinary collector without starting any helper threads.
    pi335Ctrl_test();

    /// Select the two-axis or three-axis production command path.
    void axes( int count /**< [in] Number of controlled axes. */ );

    /// Open a paired timing event at the current monotonic receipt time.
    void begin();

    /// Complete the event and refresh the consumer-owned timing summary.
    void finish();
};

pi335Ctrl_test::pi335Ctrl_test()
{
    m_configName  = "pi335-offline-test";
    m_imageStream = {};
    m_max3        = 35;
    m_dmTiming.configure( 2.0, 32 );
    state( stateCodes::OPERATING );
    pi335TimingTest::script = {};
}

void pi335Ctrl_test::axes( int count )
{
    m_naxes = count;
}

void pi335Ctrl_test::begin()
{
    m_dmTiming.beginEventAt( true, dev::dmTiming::monotonicNowNs() );
}

void pi335Ctrl_test::finish()
{
    m_dmTiming.mark( MagAOX::dmTiming::Marker::finish );
    m_dmTiming.finishEvent();
    m_dmTiming.update();
}
} // namespace app
} // namespace MagAOX
/// \endcond

namespace libXWCTest
{
namespace pi335CtrlTest
{
using namespace MagAOX::app;
using MagAOX::dmTiming::Metric;

/** \defgroup pi335Ctrl_unit_test pi335Ctrl Unit Tests
 * \brief Unit tests for the PI controller application.
 * \ingroup application_unit_test
 */

/// Qualify one complete multi-axis command and bracket every serial operation with the SDK timing endpoints.
/** \ingroup pi335Ctrl_unit_test */
TEST_CASE( "pi335Ctrl records the complete DM-origin serial submission", "[pi335Ctrl][timing]" )
{
    // clang-format off
    #ifdef PI335CTRL_TEST_DOXYGEN_REF
    pi335Ctrl::commandDM(nullptr);
    pi335Ctrl::move_1(0, true);
    pi335Ctrl::move_2(0, true);
    pi335Ctrl::move_3(0, true);
    #endif
    // clang-format on
    pi335Ctrl_test app;
    const int      axes = GENERATE( 2, 3 );
    app.axes( axes );
    float command[]{ 3, 4, 5 };
    app.begin();
    REQUIRE( app.commandDM( command ) == 0 );
    app.finish();
    const auto &summary = app.dmTimingSummary();
    REQUIRE( app.m_dmWritesAttempted == unsigned( axes ) );
    REQUIRE( app.m_dmWritesSuccessful == unsigned( axes ) );
    REQUIRE( summary.total.submitted == 1 );
    REQUIRE( summary.total.partial == 0 );
    REQUIRE( summary.total.failed == 0 );
    REQUIRE( pi335TimingTest::script.m_commands.size() == size_t( axes ) );
    REQUIRE( pi335TimingTest::script.m_commands[0] == "MOV 1 3.000000\n" );
    REQUIRE( pi335TimingTest::script.m_commands[1] == "MOV 2 4.000000\n" );
    if( axes == 3 )
        REQUIRE( pi335TimingTest::script.m_commands[2] == "MOV 3 5.000000\n" );
    const auto &sdk = summary.metrics[MagAOX::dmTiming::metricIndex( Metric::sdk )];
    REQUIRE( sdk.count == 1 );
    REQUIRE( sdk.mean >= ( pi335TimingTest::script.m_lastReturn - pi335TimingTest::script.m_firstEntry ) * 1e-9 );
}

/// Preserve historical move return values while distinguishing partial and wholly failed transport attempts.
/** \ingroup pi335Ctrl_unit_test */
TEST_CASE( "pi335Ctrl timing uses actual serial results without changing command returns", "[pi335Ctrl][timing]" )
{
    // clang-format off
    #ifdef PI335CTRL_TEST_DOXYGEN_REF
    pi335Ctrl::commandDM(nullptr);
    pi335Ctrl::recordDMOutcome();
    #endif
    // clang-format on
    pi335Ctrl_test app;
    float          command[]{ 3, 4, 5 };
    const bool     partial            = GENERATE( false, true );
    pi335TimingTest::script.m_results = partial ? std::array<int, 3>{ 0, -9, 0 } : std::array<int, 3>{ -9, -9, 0 };
    app.begin();
    REQUIRE( app.commandDM( command ) == 0 ); // Existing move helpers log transport errors and return success.
    app.finish();
    const auto &summary = app.dmTimingSummary();
    REQUIRE( app.m_dmWritesAttempted == 2 );
    REQUIRE( app.m_dmWritesSuccessful == ( partial ? 1 : 0 ) );
    REQUIRE( summary.total.submitted == 0 );
    REQUIRE( summary.total.partial == ( partial ? 1 : 0 ) );
    REQUIRE( summary.total.failed == ( partial ? 0 : 1 ) );
    REQUIRE( summary.metrics[MagAOX::dmTiming::metricIndex( Metric::sdk )].count == 1 );
}

/// Keep non-operating and rejected moves out of successful-send populations while preserving partial writes.
/** \ingroup pi335Ctrl_unit_test */
TEST_CASE( "pi335Ctrl timing distinguishes unsent and range-rejected commands", "[pi335Ctrl][timing]" )
{
    // clang-format off
    #ifdef PI335CTRL_TEST_DOXYGEN_REF
    pi335Ctrl::commandDM(nullptr);
    #endif
    // clang-format on
    pi335Ctrl_test app;
    float          command[]{ 3, 4, 5 };
    int            expected    = -1;
    bool           intentional = false;
    unsigned       writes      = 0;
    SECTION( "a non-operating controller does not send" )
    {
        app.state( stateCodes::READY );
        expected    = 0;
        intentional = true;
    }
    SECTION( "the first axis is rejected before any write" )
    {
        command[0] = 100;
    }
    SECTION( "the second axis is rejected after the first was sent" )
    {
        command[1] = 100;
        writes     = 1;
    }
    app.begin();
    REQUIRE( app.commandDM( command ) == expected );
    app.finish();
    const auto &summary = app.dmTimingSummary();
    REQUIRE( pi335TimingTest::script.m_commands.size() == writes );
    REQUIRE( summary.total.submitted == 0 );
    REQUIRE( summary.total.noSend == ( intentional ? 1 : 0 ) );
    REQUIRE( summary.total.partial == ( writes ? 1 : 0 ) );
    REQUIRE( summary.total.failed == ( !intentional && !writes ? 1 : 0 ) );
    REQUIRE( summary.metrics[MagAOX::dmTiming::metricIndex( Metric::sdk )].count == writes );
}

/// Exclude manual position writes even when an unrelated DM receipt is pending.
/** \ingroup pi335Ctrl_unit_test */
TEST_CASE( "pi335Ctrl manual moves cannot populate DM timing endpoints", "[pi335Ctrl][timing]" )
{
    // clang-format off
    #ifdef PI335CTRL_TEST_DOXYGEN_REF
    pi335Ctrl::move_1(0);
    pi335Ctrl::move_2(0);
    pi335Ctrl::move_3(0);
    #endif
    // clang-format on
    pi335Ctrl_test app;
    app.axes( 3 );
    app.begin();
    REQUIRE( app.move_1( 1 ) == 0 );
    REQUIRE( app.move_2( 2 ) == 0 );
    REQUIRE( app.move_3( 3 ) == 0 );
    app.finish();
    const auto &summary = app.dmTimingSummary();
    REQUIRE( pi335TimingTest::script.m_commands.size() == 3 );
    REQUIRE( app.m_dmWritesAttempted == 0 );
    REQUIRE( app.m_dmWritesSuccessful == 0 );
    REQUIRE( summary.total.submitted == 0 );
    REQUIRE( summary.metrics[MagAOX::dmTiming::metricIndex( Metric::sdk )].count == 0 );
    REQUIRE( summary.metrics[MagAOX::dmTiming::metricIndex( Metric::receiptSend )].count == 0 );
}
} // namespace pi335CtrlTest
} // namespace libXWCTest
