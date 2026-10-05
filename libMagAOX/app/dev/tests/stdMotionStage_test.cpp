/** \file stdMotionStage_test.cpp
 * \brief Catch2 tests for the stdMotionStage helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup testing
 */

#include "../../../../tests/testXWC.hpp"

#include "../../MagAOXApp.hpp"
#include "../stdMotionStage.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{
namespace appTest
{
namespace devTest
{

/** \defgroup stdMotionStage_tests libXWC::app::dev::stdMotionStage Unit Tests
 * \ingroup app_dev_unit_tests
 */

/// Test harness for exercising stdMotionStage preset-name callbacks without the INDI validation short-circuit.
/** \ingroup stdMotionStage_tests
 */
class stdMotionStageHarness : public MagAOXApp<false>, public dev::stdMotionStage<stdMotionStageHarness>
{
    friend class dev::stdMotionStage<stdMotionStageHarness>;

  protected:
    float m_lastMoveTarget{ -1.0f }; ///< Last target passed to moveTo by the helper.

    int m_moveCalls{ 0 }; ///< Number of motion requests issued by the helper.

    static std::string s_lastLogMessage; ///< Most recent text log message captured from stdMotionStage.

    static logPrioT s_lastLogLevel; ///< Most recent log priority captured from stdMotionStage.

    static int s_logCount; ///< Number of captured stdMotionStage log messages.

  public:
    /// Construct a stdMotionStage test harness with a presetName callback property.
    stdMotionStageHarness();

    /// Destroy the stdMotionStage test harness.
    ~stdMotionStageHarness() noexcept override;

    /// Reset the captured stdMotionStage logging state shared across harness instances.
    static void resetLogState();

    /// Configure the preset-name list and notation used by stdMotionStage.
    void
    configurePresets( const std::vector<std::string> &presetNames /**< [in] configured preset names */,
                      const std::string &presetNotation /**< [in] singular preset notation such as preset or filter */
    );

    /// Apply a presetName request property to the stdMotionStage callback under test.
    int applyPresetNameRequest(
        const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> &elements /**< [in] requested
                                                                                                        switch
                                                                                                        elements and
                                                                                                        states */
    );

    /// Get the number of move requests accepted by stdMotionStage.
    int moveCalls() const;

    /// Get the last move target accepted by stdMotionStage.
    float lastMoveTarget() const;

    /// Get the most recent stdMotionStage text log message captured by the harness.
    static const std::string &lastLogMessage();

    /// Get the most recent stdMotionStage log priority captured by the harness.
    static logPrioT lastLogLevel();

    /// Get the number of stdMotionStage log messages captured by the harness.
    static int logCount();

    /// Capture stdMotionStage log messages instead of sending them to the normal logger.
    template <typename logT, int retval = 0>
    static int log( const typename logT::messageT &msg /**< [in] the logged message */,
                    logPrioT                       level = logPrio::LOG_DEFAULT /**< [in] the logged priority */
    );

    /// No-op startup implementation required by MagAOXApp for testing.
    int appStartup() override;

    /// No-op logic implementation required by MagAOXApp for testing.
    int appLogic() override;

    /// No-op shutdown implementation required by MagAOXApp for testing.
    int appShutdown() override;

    /// No-op stop implementation required by stdMotionStage for testing.
    int stop();

    /// No-op homing implementation required by stdMotionStage for testing.
    int startHoming();

    /// Return a fixed preset number for testing paths that query the current preset.
    float presetNumber();

    /// Record a requested move target when stdMotionStage accepts a motion request.
    int moveTo( float target /**< [in] the accepted move target */ );
};

std::string stdMotionStageHarness::s_lastLogMessage;

logPrioT stdMotionStageHarness::s_lastLogLevel = logPrio::LOG_DEFAULT;

int stdMotionStageHarness::s_logCount = 0;

stdMotionStageHarness::stdMotionStageHarness() : MagAOXApp<false>( "", false )
{
    m_configName = "stest";

    m_indiP_presetName = pcf::IndiProperty( pcf::IndiProperty::Switch );
    m_indiP_presetName.setDevice( m_configName );
    m_indiP_presetName.setName( "presetName" );

    resetLogState();
}

stdMotionStageHarness::~stdMotionStageHarness() noexcept = default;

void stdMotionStageHarness::resetLogState()
{
    s_lastLogMessage.clear();
    s_lastLogLevel = logPrio::LOG_DEFAULT;
    s_logCount     = 0;
}

void stdMotionStageHarness::configurePresets( const std::vector<std::string> &presetNames,
                                              const std::string              &presetNotation )
{
    m_presetNames    = presetNames;
    m_presetNotation = presetNotation;
}

int stdMotionStageHarness::applyPresetNameRequest(
    const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> &elements )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( m_configName );
    ip.setName( "presetName" );

    for( const auto &element : elements )
    {
        ip.add( pcf::IndiElement( element.first ) );
        ip[element.first].setSwitchState( element.second );
    }

    return newCallBack_m_indiP_presetName( ip );
}

int stdMotionStageHarness::moveCalls() const
{
    return m_moveCalls;
}

float stdMotionStageHarness::lastMoveTarget() const
{
    return m_lastMoveTarget;
}

const std::string &stdMotionStageHarness::lastLogMessage()
{
    return s_lastLogMessage;
}

logPrioT stdMotionStageHarness::lastLogLevel()
{
    return s_lastLogLevel;
}

int stdMotionStageHarness::logCount()
{
    return s_logCount;
}

template <typename logT, int retval>
int stdMotionStageHarness::log( const typename logT::messageT &msg, logPrioT level )
{
    s_lastLogMessage =
        logT::msgString( const_cast<uint8_t *>( msg.builder.GetBufferPointer() ), msg.builder.GetSize() );
    s_lastLogLevel = level;
    ++s_logCount;

    return retval;
}

int stdMotionStageHarness::appStartup()
{
    return 0;
}

int stdMotionStageHarness::appLogic()
{
    return 0;
}

int stdMotionStageHarness::appShutdown()
{
    return 0;
}

int stdMotionStageHarness::stop()
{
    return 0;
}

int stdMotionStageHarness::startHoming()
{
    return 0;
}

float stdMotionStageHarness::presetNumber()
{
    return 0.0f;
}

int stdMotionStageHarness::moveTo( float target )
{
    m_lastMoveTarget = target;
    ++m_moveCalls;

    return 0;
}

/// Verify stdMotionStage logs and rejects invalid preset-name selections before issuing motion requests.
/**
 * \ingroup stdMotionStage_tests
 */
TEST_CASE( "stdMotionStage rejects invalid preset-name selections", "[dev::stdMotionStage]" )
{
    // clang-format off
    #ifdef STDMOTIONSTAGE_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdMotionStage<libXWCTest::appTest::devTest::stdMotionStageHarness>::newCallBack_m_indiP_presetName( pcf::IndiProperty() );
    #endif
    // clang-format on

    SECTION( "quoted preset names are logged and rejected" )
    {
        stdMotionStageHarness app;

        app.configurePresets( { "open", "focus" }, "preset" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest( { { "\"focus\"", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "Unknown presetName selected: \"focus\"" );
        REQUIRE( app.moveCalls() == 0 );
        REQUIRE( app.lastMoveTarget() == -1.0f );
    }

    SECTION( "invalid names are rejected even when a valid preset is also selected" )
    {
        stdMotionStageHarness app;

        app.configurePresets( { "open", "focus" }, "filter" );
        stdMotionStageHarness::resetLogState();

        REQUIRE( app.applyPresetNameRequest(
                     { { "open", pcf::IndiElement::On }, { "bogus", pcf::IndiElement::On } } ) == -1 );
        REQUIRE( stdMotionStageHarness::logCount() == 1 );
        REQUIRE( stdMotionStageHarness::lastLogLevel() == logPrio::LOG_ERROR );
        REQUIRE( stdMotionStageHarness::lastLogMessage() == "Unknown filterName selected: bogus" );
        REQUIRE( app.moveCalls() == 0 );
        REQUIRE( app.lastMoveTarget() == -1.0f );
    }
}

} // namespace devTest
} // namespace appTest
} // namespace libXWCTest
