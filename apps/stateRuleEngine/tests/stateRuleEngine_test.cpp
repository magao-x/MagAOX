/** \file stateRuleEngine_test.cpp
 * \brief Catch2 tests for the stateRuleEngine app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup stateRuleEngine_files
 */

#include "../../../tests/testXWC.hpp"

#include <stdexcept>

#include "../stateRuleEngine.hpp"

using namespace MagAOX::app;

namespace MagAOX
{
namespace app
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class stateRuleEngine_test : public stateRuleEngine
{
  public:
    using stateRuleEngine::notificationLabel;
    using stateRuleEngine::notificationMessage;
    using stateRuleEngine::ruleIsOn;
    using stateRuleEngine::ruleStateProperty;

    /// Add one rule to the harness and provision its published switch element.
    void addRule( const std::string &name, indiCompRule *rule /**< [in] rule instance owned by the app rule map */ )
    {
        m_ruleMaps.rules[name] = rule;

        pcf::IndiProperty *property = ruleStateProperty( rule->priority() );
        if( property == nullptr )
        {
            return;
        }

        if( property->getType() != pcf::IndiProperty::Switch )
        {
            *property = pcf::IndiProperty( pcf::IndiProperty::Switch );
        }

        if( !property->find( name ) )
        {
            property->add( pcf::IndiElement( name, pcf::IndiElement::Off ) );
        }
    }

    /// Force the published switch state for one rule element.
    void setPublishedRuleState( const rulePriority &priority, /**< [in] priority whose published property is updated */
                                const std::string  &name,     /**< [in] rule element name to update */
                                pcf::IndiElement::SwitchStateType state /**< [in] switch state to write */ )
    {
        pcf::IndiProperty *property = ruleStateProperty( priority );
        if( property == nullptr )
        {
            throw std::runtime_error( "published rule property is not available" );
        }

        if( property->getType() != pcf::IndiProperty::Switch )
        {
            *property = pcf::IndiProperty( pcf::IndiProperty::Switch );
        }

        if( !property->find( name ) )
        {
            property->add( pcf::IndiElement( name, pcf::IndiElement::Off ) );
        }

        ( *property )[name].setSwitchState( state );
    }

    /// Read the notification messages captured during `appLogic()`.
    const std::vector<std::string> &notifications() const
    {
        return m_notifications;
    }

    /// Clear the captured notification list.
    void clearNotifications()
    {
        m_notifications.clear();
    }

  protected:
    /// Capture the notification text instead of sending it through an INDI driver.
    int sendNotification( const std::string &message /**< [in] formatted notification text */ ) override
    {
        m_notifications.push_back( message );
        return 0;
    }

    /// Notification messages observed by the test harness.
    std::vector<std::string> m_notifications;
};

/// Fixed-value rule used to exercise `stateRuleEngine` notification handling.
class fixedRule : public indiCompRule
{
  public:
    /// Construct a rule with an initial boolean value.
    fixedRule( bool value = false /**< [in] initial rule value */ ) : m_value( value )
    {
    }

    /// Update the value returned by `value()`.
    void value( bool value /**< [in] new boolean result */ )
    {
        m_value = value;
    }

    /// Report that this test rule is always valid.
    boolorerr_t valid() override
    {
        return true;
    }

    /// Return the configured boolean result.
    bool value() override
    {
        return m_value;
    }

  private:
    /// Boolean result returned by the harness rule.
    bool m_value{ false };
};
/// \endcond

} // namespace app
} // namespace MagAOX

namespace libXWCTest
{

/** \defgroup stateRuleEngine_unit_test stateRuleEngine Unit Tests
 * \brief Unit tests for the stateRuleEngine application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `stateRuleEngine` unit tests.
/** \ingroup stateRuleEngine_unit_test
 */
namespace stateRuleEngineTest
{

/// Verify the placeholder stateRuleEngine test harness instantiates the app cleanly.
/**
 * \ingroup stateRuleEngine_unit_test
 */
TEST_CASE( "stateRuleEngine placeholder harness instantiates the app", "[stateRuleEngine]" )
{
    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    stateRuleEngine();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        stateRuleEngine app;

        REQUIRE( true );
    }
}

/// Verify notification helpers preserve active severity labels and clear formatting.
/**
 * \ingroup stateRuleEngine_unit_test
 */
TEST_CASE( "stateRuleEngine notification helpers format active and clear messages", "[stateRuleEngine]" )
{
    fixedRule rule;
    rule.message( std::string( "camera combo mismatch" ) );

    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    stateRuleEngine::notificationLabel( rulePriority::warning );
    stateRuleEngine::notificationMessage( std::string(), rule, std::string() );
    #endif
    // clang-format on

    SECTION( "active notifications use the rule priority label" )
    {
        REQUIRE( stateRuleEngine_test::notificationLabel( rulePriority::warning ) == "WARNING" );
        REQUIRE( stateRuleEngine_test::notificationMessage(
                     "combo-mismatch", rule, stateRuleEngine_test::notificationLabel( rulePriority::warning ) ) ==
                 "WARNING: camera combo mismatch" );
    }

    SECTION( "clear notifications prefix the configured rule message" )
    {
        REQUIRE( stateRuleEngine_test::notificationMessage(
                     "combo-mismatch", rule, stateRuleEngine_test::notificationLabel( rulePriority::info ), true ) ==
                 "INFO: Cleared: camera combo mismatch" );
    }

    SECTION( "clear notifications fall back to the rule name when message is empty" )
    {
        rule.message( std::string() );

        REQUIRE( stateRuleEngine_test::notificationMessage(
                     "combo-mismatch", rule, stateRuleEngine_test::notificationLabel( rulePriority::info ), true ) ==
                 "INFO: Cleared: combo-mismatch" );
    }
}

/// Verify the published-state helpers select the right property and detect switch state.
/**
 * \ingroup stateRuleEngine_unit_test
 */
TEST_CASE( "stateRuleEngine published-state helpers select properties and detect On state", "[stateRuleEngine]" )
{
    stateRuleEngine_test app;

    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    stateRuleEngine::ruleStateProperty( rulePriority::info );
    stateRuleEngine::ruleIsOn( pcf::IndiProperty(), std::string() );
    #endif
    // clang-format on

    app.m_indiP_info    = pcf::IndiProperty( pcf::IndiProperty::Switch );
    app.m_indiP_caution = pcf::IndiProperty( pcf::IndiProperty::Switch );
    app.m_indiP_warning = pcf::IndiProperty( pcf::IndiProperty::Switch );
    app.m_indiP_alert   = pcf::IndiProperty( pcf::IndiProperty::Switch );

    REQUIRE( app.ruleStateProperty( rulePriority::info ) == &app.m_indiP_info );
    REQUIRE( app.ruleStateProperty( rulePriority::caution ) == &app.m_indiP_caution );
    REQUIRE( app.ruleStateProperty( rulePriority::warning ) == &app.m_indiP_warning );
    REQUIRE( app.ruleStateProperty( rulePriority::alert ) == &app.m_indiP_alert );
    REQUIRE( app.ruleStateProperty( rulePriority::none ) == nullptr );

    app.m_indiP_warning.add( pcf::IndiElement( "combo-mismatch", pcf::IndiElement::Off ) );

    REQUIRE( stateRuleEngine_test::ruleIsOn( app.m_indiP_warning, "combo-mismatch" ) == false );

    app.m_indiP_warning["combo-mismatch"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( stateRuleEngine_test::ruleIsOn( app.m_indiP_warning, "combo-mismatch" ) == true );

    REQUIRE( stateRuleEngine_test::ruleIsOn( app.m_indiP_warning, "missing-rule" ) == false );
}

/// Verify `appLogic()` emits one clear notification for an observed `On -> Off` transition.
/**
 * \ingroup stateRuleEngine_unit_test
 */
TEST_CASE( "stateRuleEngine appLogic reports one clear notification per observed On-to-Off transition",
           "[stateRuleEngine]" )
{
    stateRuleEngine_test app;
    auto                *rule = new fixedRule( false );

    rule->priority( rulePriority::warning );
    rule->message( std::string( "camera combo mismatch" ) );
    rule->messageCount( 2 );
    app.addRule( "combo-mismatch", rule );

    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    stateRuleEngine::appLogic();
    #endif
    // clang-format on

    SECTION( "configured rule message is used for the clear notification" )
    {
        app.setPublishedRuleState( rulePriority::warning, "combo-mismatch", pcf::IndiElement::On );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.notifications().size() == 1 );
        REQUIRE( app.notifications().front() == "INFO: Cleared: camera combo mismatch" );
        REQUIRE( rule->messageCount() == 0 );

        app.clearNotifications();
        app.setPublishedRuleState( rulePriority::warning, "combo-mismatch", pcf::IndiElement::Off );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.notifications().empty() );
    }

    SECTION( "rule name is used when no explicit message is configured" )
    {
        rule->message( std::string() );
        app.setPublishedRuleState( rulePriority::warning, "combo-mismatch", pcf::IndiElement::On );

        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.notifications().size() == 1 );
        REQUIRE( app.notifications().front() == "INFO: Cleared: combo-mismatch" );
        REQUIRE( rule->messageCount() == 0 );
    }
}

} // namespace stateRuleEngineTest

} // namespace libXWCTest
