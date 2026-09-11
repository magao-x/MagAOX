/** \file stdMotionStage_test.hpp
 * \brief Test harness for the MagAOX::app::dev::stdMotionStage device mixin.
 *
 * One harness class, stdMotionStageHarness, drives the real stdMotionStage code without a real
 * INDI server or hardware. The derived class interface a real stage app implements is stubbed.
 * The harness counts stop, homing, and move calls. It captures log and telemetry messages in
 * static counters instead of sending them to the real loggers. It can also fail INDI property
 * registration for one named property and install a FIFO-less INDI driver. The common parts of
 * every dev:: harness, such as the FIFO-less indiDriver and the public power management state,
 * come from appHarnessBase in testHarnessCommon.hpp.
 *
 * \ingroup testing
 */

#include <cmath>

#include "../../MagAOXApp.hpp"
#include "../stdMotionStage.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace libXWCTest
{
namespace appTest
{
namespace devTest
{

/// Test harness for exercising stdMotionStage without a real INDI server or hardware.
/** \ingroup stdMotionStage_tests
 *
 * The harness counts stop, homing, and move calls. It captures log and telemetry
 * messages in static counters instead of sending them to the real loggers. It can
 * also inject INDI property registration failures and install a FIFO-less INDI driver.
 */
class stdMotionStageHarness : public MagAOX::app::dev::testHarness::appHarnessBase,
                              public MagAOX::app::dev::stdMotionStage<stdMotionStageHarness>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBase           baseT;
    typedef MagAOX::app::dev::stdMotionStage<stdMotionStageHarness> stdMotionStageT;

    friend class MagAOX::app::dev::stdMotionStage<stdMotionStageHarness>;

  protected:
    float m_lastMoveTarget{ -1.0f }; ///< Last target passed to moveTo by the helper.

    int m_moveCalls{ 0 }; ///< Number of motion requests issued by the helper.

    int m_stopCalls{ 0 }; ///< Number of times stop() was invoked.

    int m_homingCalls{ 0 }; ///< Number of times startHoming() was invoked.

    float m_presetNumberValue{ 0.0f }; ///< Value returned by presetNumber().

    std::string m_failRegisterName; ///< If non-empty, registerIndiPropertyNew fails for the property with this name.

    static inline std::string s_lastLogMessage; ///< Most recent text log message captured from stdMotionStage.

    static inline flatlogs::logPrioT s_lastLogLevel =
        flatlogs::logPrio::LOG_DEFAULT; ///< Most recent log priority captured from stdMotionStage.

    static inline int s_logCount = 0; ///< Number of captured stdMotionStage log messages.

    static inline int s_telemCount = 0; ///< Number of captured stdMotionStage telemetry records.

  public:
    /// Construct a stdMotionStage test harness with a presetName callback property.
    stdMotionStageHarness() : baseT( "stest" ) // short for stage test
    {
        m_indiP_presetName = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_presetName.setDevice( m_configName );
        m_indiP_presetName.setName( "presetName" );

        resetLogState();
    }

    /// Reset the captured stdMotionStage logging state shared across harness instances.
    static void resetLogState()
    {
        s_lastLogMessage.clear();
        s_lastLogLevel = flatlogs::logPrio::LOG_DEFAULT;
        s_logCount     = 0;
    }

    /// Reset the captured stdMotionStage telemetry count shared across harness instances.
    static void resetTelemCount()
    {
        s_telemCount = 0;
    }

    /// Configure the preset-name list and notation used by stdMotionStage.
    void
    configurePresets( const std::vector<std::string> &presetNames /**< [in] configured preset names */,
                      const std::string &presetNotation /**< [in] singular preset notation such as preset or filter */
    )
    {
        m_presetNames    = presetNames;
        m_presetNotation = presetNotation;
    }

    /// Directly set the preset positions used to resolve preset-name aliases.
    void setPresetPositions( const std::vector<float> &positions /**< [in] the preset positions to use */ )
    {
        m_presetPositions = positions;
    }

    /// Override the configured device name. Used to probe INDI unique-key edge cases.
    void setConfigName( const std::string &name /**< [in] the new device/config name */ )
    {
        m_configName = name;
    }

    /// Control the m_defaultPositions flag exercised by loadConfig().
    void setDefaultPositions( bool defaultPositions /**< [in] the new flag value */ )
    {
        m_defaultPositions = defaultPositions;
    }

    /// Control the m_fractionalPresets flag exercised by appStartup().
    void setFractionalPresets( bool fractionalPresets /**< [in] the new flag value */ )
    {
        m_fractionalPresets = fractionalPresets;
    }

    /// Directly set m_moving. In a real app the derived class maintains this motion state.
    void setMoving( int8_t moving /**< [in] the new m_moving value */ )
    {
        m_moving = moving;
    }

    /// Directly set the value returned by presetNumber().
    void setPresetNumberValue( float presetNumber /**< [in] the new preset number to report */ )
    {
        m_presetNumberValue = presetNumber;
    }

    /// Force registerIndiPropertyNew to fail for the property with this name, simulating a registration failure.
    void setFailRegisterName( const std::string &name /**< [in] the property name that should fail to register */ )
    {
        m_failRegisterName = name;
    }

    /// Install or remove a real but never-activated indiDriver. This exercises the code paths that need m_indiDriver to be non-null.
    void setFakeIndiDriver( bool present /**< [in] whether a fake driver should be installed */ )
    {
        if( present )
        {
            if( !m_indiDriver )
            {
                // See dev::testHarness::makeFifolessIndiDriver() for why this never needs a live,
                // connected INDI server.
                setupRealDriver();
            }
        }
        else if( m_indiDriver )
        {
            delete m_indiDriver;
            m_indiDriver = nullptr;
        }
    }

    /// Apply a presetName request property to the stdMotionStage callback under test.
    int applyPresetNameRequest(
        const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> &elements /**< [in] requested
                                                                                                        switch
                                                                                                        elements and
                                                                                                        states */
    )
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

    /// Apply a numeric preset-position request property to the stdMotionStage callback under test.
    int applyPresetRequest( float target /**< [in] the requested target position */,
                            bool includeTargetElement = true /**< [in] whether to include a target element */,
                            bool matchProperty = true /**< [in] whether the request should match the real property */
    )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( matchProperty ? m_configName : m_configName + "_other" );
        ip.setName( matchProperty ? m_presetNotation : m_presetNotation + "_other" );

        if( includeTargetElement )
        {
            ip.add( pcf::IndiElement( "target" ) );
            ip["target"].set( target );
        }

        return newCallBack_m_indiP_preset( ip );
    }

    /// Apply a home request property to the stdMotionStage callback under test.
    int applyHomeRequest(
        pcf::IndiElement::SwitchStateType state /**< [in] the requested switch state */,
        bool includeRequestElement = true /**< [in] whether to include the request element */,
        bool matchProperty = true /**< [in] whether the request should match the real property */
    )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( matchProperty ? m_configName : m_configName + "_other" );
        ip.setName( matchProperty ? "home" : "home_other" );

        if( includeRequestElement )
        {
            ip.add( pcf::IndiElement( "request" ) );
            ip["request"].setSwitchState( state );
        }

        return newCallBack_m_indiP_home( ip );
    }

    /// Apply a stop request property to the stdMotionStage callback under test, with optional key overrides.
    int applyStopRequest(
        pcf::IndiElement::SwitchStateType state /**< [in] the requested switch state */,
        bool includeRequestElement = true /**< [in] whether to include the request element */,
        const std::string &deviceOverride = "" /**< [in] if non-empty, overrides the device name used */,
        const std::string &nameOverride = "" /**< [in] if non-empty, overrides the property name used */
    )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( deviceOverride.empty() ? m_configName : deviceOverride );
        ip.setName( nameOverride.empty() ? "stop" : nameOverride );

        if( includeRequestElement )
        {
            ip.add( pcf::IndiElement( "request" ) );
            ip["request"].setSwitchState( state );
        }

        return newCallBack_m_indiP_stop( ip );
    }

    /// Get the number of move requests accepted by stdMotionStage.
    int moveCalls() const
    {
        return m_moveCalls;
    }

    /// Get the last move target accepted by stdMotionStage.
    float lastMoveTarget() const
    {
        return m_lastMoveTarget;
    }

    /// Get the number of times stop() was invoked.
    int stopCalls() const
    {
        return m_stopCalls;
    }

    /// Get the number of times startHoming() was invoked.
    int homingCalls() const
    {
        return m_homingCalls;
    }

    /// Get the currently tracked movingState value.
    int8_t movingStateValue() const
    {
        return m_movingState;
    }

    /// Get the currently tracked m_moving value.
    int8_t movingValue() const
    {
        return m_moving;
    }

    /// Get the currently tracked preset-name alias index.
    int presetNameIndexValue() const
    {
        return m_presetNameIndex;
    }

    /// Get the currently tracked numerical preset target.
    float presetTargetValue() const
    {
        return m_preset_target;
    }

    /// Get the currently tracked numerical preset current value.
    float presetCurrentValue() const
    {
        return m_preset;
    }

    /// Get the configured preset names.
    const std::vector<std::string> &presetNamesValue() const
    {
        return m_presetNames;
    }

    /// Get the configured/resolved preset positions.
    const std::vector<float> &presetPositionsValue() const
    {
        return m_presetPositions;
    }

    /// Get the configured powerOnHome flag.
    bool powerOnHomeValue() const
    {
        return m_powerOnHome;
    }

    /// Get the configured homePreset value.
    int homePresetValue() const
    {
        return m_homePreset;
    }

    /// Get the name of the built numeric preset position INDI property.
    std::string presetPropertyName() const
    {
        return m_indiP_preset.getName();
    }

    /// Get the name of the built presetName INDI property.
    std::string presetNamePropertyName() const
    {
        return m_indiP_presetName.getName();
    }

    /// Get the name of the built home INDI property.
    std::string homePropertyName() const
    {
        return m_indiP_home.getName();
    }

    /// Get the name of the built stop INDI property.
    std::string stopPropertyName() const
    {
        return m_indiP_stop.getName();
    }

    /// Get the most recent stdMotionStage text log message captured by the harness.
    static const std::string &lastLogMessage()
    {
        return s_lastLogMessage;
    }

    /// Get the most recent stdMotionStage log priority captured by the harness.
    static flatlogs::logPrioT lastLogLevel()
    {
        return s_lastLogLevel;
    }

    /// Get the number of stdMotionStage log messages captured by the harness.
    static int logCount()
    {
        return s_logCount;
    }

    /// Get the number of stdMotionStage telemetry records captured by the harness.
    static int telemCount()
    {
        return s_telemCount;
    }

    /// Capture stdMotionStage log messages instead of sending them to the normal logger.
    template <typename logT, int retval = 0>
    static int log( const typename logT::messageT &msg /**< [in] the logged message */,
                    flatlogs::logPrioT level = flatlogs::logPrio::LOG_DEFAULT /**< [in] the logged priority */
    )
    {
        s_lastLogMessage =
            logT::msgString( const_cast<uint8_t *>( msg.builder.GetBufferPointer() ), msg.builder.GetSize() );
        s_lastLogLevel = level;
        ++s_logCount;

        return retval;
    }

    /// Capture stdMotionStage telemetry records instead of sending them to a real telemetry logger.
    template <typename telT, int retval = 0>
    int telem( const typename telT::messageT &msg /**< [in] the telemetry message */ )
    {
        static_cast<void>( msg );
        ++s_telemCount;

        return retval;
    }

    /// Delegate to the stdMotionStage base setupConfig, resolving the multiple-inheritance ambiguity.
    int setupConfig( mx::app::appConfigurator &config /**< [out] the configurator to use */ )
    {
        return stdMotionStageT::setupConfig( config );
    }

    /// Delegate to the stdMotionStage base loadConfig, resolving the multiple-inheritance ambiguity.
    int loadConfig( mx::app::appConfigurator &config /**< [in] the configurator to use */ )
    {
        return stdMotionStageT::loadConfig( config );
    }

    /// No-op startup implementation required by MagAOXApp for testing.
    int appStartup() override
    {
        return 0;
    }

    /// No-op logic implementation required by MagAOXApp for testing.
    int appLogic() override
    {
        return 0;
    }

    /// No-op shutdown implementation required by MagAOXApp for testing.
    int appShutdown() override
    {
        return 0;
    }

    /// Simulate stop, tracking how many times it was called.
    int stop()
    {
        ++m_stopCalls;
        return 0;
    }

    /// Simulate homing, tracking how many times it was called.
    int startHoming()
    {
        ++m_homingCalls;
        return 0;
    }

    /// Return the currently configured preset number for testing paths that query the current preset.
    float presetNumber()
    {
        return m_presetNumberValue;
    }

    /// Record a requested move target when stdMotionStage accepts a motion request.
    int moveTo( float target /**< [in] the accepted move target */ )
    {
        m_lastMoveTarget = target;
        ++m_moveCalls;
        m_preset = target;

        // Simulate a derived class that immediately reports its new preset number.
        // Search the configured preset positions for one that matches the target.
        // If none matches, report -1.
        m_presetNumberValue = -1.0f;
        for( size_t n = 0; n < m_presetPositions.size(); ++n )
        {
            if( std::fabs( m_presetPositions[n] - target ) < 1e-4f )
            {
                m_presetNumberValue = static_cast<float>( n );
                break;
            }
        }

        return 0;
    }

    /// Test-only override that lets appStartup() registration failures be simulated one property at a time.
    /** The tests pick the failing property by name, so this sits on top of the count based
     * injection in appHarnessBase. Registrations that are not forced to fail still go through the base.
     */
    int registerIndiPropertyNew( pcf::IndiProperty &prop /**< [in,out] the property to register */,
                                 int ( *callBack )( void *, const pcf::IndiProperty & ) /**< [in] the callback */
    )
    {
        if( !m_failRegisterName.empty() && prop.getName() == m_failRegisterName )
        {
            return -1;
        }

        return baseT::registerIndiPropertyNew( prop, callBack );
    }

    /// Test-only wrapper exposing the protected setPresetNameTracking() so its out-of-range guard can be tested.
    int callSetPresetNameTracking( int presetNameIndex /**< [in] the index to record, which may be invalid */ )
    {
        return setPresetNameTracking( presetNameIndex );
    }
};

} // namespace devTest
} // namespace appTest
} // namespace libXWCTest

// LCOV_EXCL_STOP
