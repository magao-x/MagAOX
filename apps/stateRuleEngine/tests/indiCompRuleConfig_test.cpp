/** \file indiCompRuleConfig_test.cpp
 * \brief Catch2 tests for stateRuleEngine rule-configuration helpers.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup stateRuleEngine_files
 */

#include "../../../tests/testXWC.hpp"

#include "../indiCompRuleConfig.hpp"

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

SCENARIO( "configuring basic rules", "[stateRuleEngine::ruleConfig]" )
{
    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    loadRuleConfig( *(indiRuleMaps *)nullptr, *(std::map<std::string, ruleRuleKeys> *)nullptr, *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    GIVEN( "single rules in a config file" )
    {
        WHEN( "a numValRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property", "element", "target" },
                                      { "numVal", "dev.prop", "elem", "1.234" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->target() == 1.234 );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->tol() == 1e-6 );
        }

        WHEN( "a numValRule changing defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "priority", "comp", "property", "element", "target", "tol" },
                                      { "numVal", "warning", "GtEq", "dev.prop", "elem", "1.234", "1e-8" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::warning );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::GtEq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->target() == 1.234 );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->tol() == 1e-8 );
        }

        WHEN( "a txtValRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property", "element", "target" },
                                      { "txtVal", "dev.prop", "elem", "xxx" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<txtValRule *>( maps.rules["rule1"] )->target() == "xxx" );
        }

        WHEN( "a txtValRule changing defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "priority", "comp", "property", "element", "target" },
                                      { "txtVal", "alert", "Neq", "dev.prop", "elem", "xxx" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::alert );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Neq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<txtValRule *>( maps.rules["rule1"] )->target() == "xxx" );
        }

        WHEN( "a swValRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1" },
                                      { "ruleType", "property", "element" },
                                      { "swVal", "dev.prop", "elem" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<swValRule *>( maps.rules["rule1"] )->target() == pcf::IndiElement::On );
        }

        WHEN( "a swValRule changing defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "priority", "comp", "property", "element", "target" },
                                      { "swVal", "info", "Neq", "dev.prop", "elem", "Off" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::info );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Neq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<swValRule *>( maps.rules["rule1"] )->target() == pcf::IndiElement::Off );
        }

        WHEN( "a timeDiffRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property", "element", "target" },
                                      { "timeDiff", "dev.prop", "elem", "1.234" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->target() == 1.234 );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->tol() == 1e-6 );
        }

        WHEN( "a timeDiffRule changing defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "priority", "comp", "property", "element", "target", "tol" },
                                      { "timeDiff", "warning", "GtEq", "dev.prop", "elem", "1.234", "1e-8" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::warning );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::GtEq );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->property() == maps.props["dev.prop"] );
            REQUIRE( static_cast<onePropRule *>( maps.rules["rule1"] )->element() == "elem" );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->target() == 1.234 );
            REQUIRE( static_cast<numValRule *>( maps.rules["rule1"] )->tol() == 1e-8 );
        }

        WHEN( "an elCompNumRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property1", "element1", "property2", "element2" },
                                      { "elCompNum", "dev1.prop1", "elem1", "dev2.prop2", "elem2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property1() == maps.props["dev1.prop1"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element1() == "elem1" );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property2() == maps.props["dev2.prop2"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element2() == "elem2" );
        }

        WHEN( "an elCompTxtRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property1", "element1", "property2", "element2" },
                                      { "elCompTxt", "dev1.prop1", "elem1", "dev2.prop2", "elem2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property1() == maps.props["dev1.prop1"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element1() == "elem1" );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property2() == maps.props["dev2.prop2"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element2() == "elem2" );
        }

        WHEN( "an elCompSwRule using defaults" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property1", "element1", "property2", "element2" },
                                      { "elCompSw", "dev1.prop1", "elem1", "dev2.prop2", "elem2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property1() == maps.props["dev1.prop1"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element1() == "elem1" );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->property2() == maps.props["dev2.prop2"] );
            REQUIRE( static_cast<twoPropRule *>( maps.rules["rule1"] )->element2() == "elem2" );
        }

        WHEN( "a ruleCompRule using defaults" )
        {
            // This requires configuring the other rules too
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleA", "ruleA", "ruleA", "rule3", "rule3", "rule3", "rule3", "rule4", "rule4", "rule4", "rule4" },
                { "ruleType",
                  "rule1",
                  "rule2",
                  "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "property",
                  "element",
                  "target" },
                { "ruleComp",
                  "rule3",
                  "rule4",
                  "txtVal",
                  "dev3.propQ",
                  "elem",
                  "xxx",
                  "txtVal",
                  "dev4.propR",
                  "mele",
                  "yyy" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );
            finalizeRuleValRules( maps, rrkMap );

            REQUIRE( maps.rules["ruleA"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["ruleA"]->comparison() == ruleComparison::And );

            REQUIRE( static_cast<ruleCompRule *>( maps.rules["ruleA"] )->rule1() == maps.rules["rule3"] );
            REQUIRE( static_cast<ruleCompRule *>( maps.rules["ruleA"] )->rule2() == maps.rules["rule4"] );
        }

        WHEN( "a multiSwitchComboRule using defaults" )
        {
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                { "ruleType", "numSwitches", "property1", "property2", "format", "targetProperty" },
                { "multiSwitchCombo", "2", "dev1.prop1", "dev2.prop2", "{}-{}", "dev3.prop3" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            multiSwitchComboRule *mscr = dynamic_cast<multiSwitchComboRule *>( maps.rules["rule1"] );

            REQUIRE( mscr != nullptr );
            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::none );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Neq );
            REQUIRE( mscr->ruleName() == "rule1" );
            REQUIRE( mscr->numSwitches() == 2 );
            REQUIRE( mscr->property( 0 ) == maps.props["dev1.prop1"] );
            REQUIRE( mscr->propertyKey( 0 ) == "dev1.prop1" );
            REQUIRE( mscr->property( 1 ) == maps.props["dev2.prop2"] );
            REQUIRE( mscr->propertyKey( 1 ) == "dev2.prop2" );
            REQUIRE( mscr->format() == "{}-{}" );
            REQUIRE( mscr->targetProperty() == maps.props["dev3.prop3"] );
            REQUIRE( mscr->targetPropertyKey() == "dev3.prop3" );
        }

        WHEN( "a multiSwitchComboRule changing defaults" )
        {
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                { "ruleType", "priority", "comp", "numSwitches", "property1", "property2", "format", "targetProperty" },
                { "multiSwitchCombo", "warning", "Eq", "2", "dev1.prop1", "dev2.prop2", "\"{}:{}\"", "dev3.prop3" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );

            multiSwitchComboRule *mscr = dynamic_cast<multiSwitchComboRule *>( maps.rules["rule1"] );

            REQUIRE( mscr != nullptr );
            REQUIRE( maps.rules["rule1"]->priority() == rulePriority::warning );
            REQUIRE( maps.rules["rule1"]->comparison() == ruleComparison::Eq );
            REQUIRE( mscr->format() == "{}:{}" );
        }
    }
}

SCENARIO( "configuring the demo", "[stateRuleEngine::ruleConfig]" )
{
    GIVEN( "the demo" )
    {
        WHEN( "the demo as writen" )
        {
            std::ofstream fout;
            fout.open( "/tmp/ruleConfig_test.conf" );
            fout << "[fwfpm-fpm]\n";
            fout << "ruleType=swVal\n";
            fout << "priority=none\n";
            fout << "comp=Eq\n";
            fout << "property=fwfpm.filterName\n";
            fout << "element=fpm\n";
            fout << "target=On\n";
            fout << "\n";
            fout << "[fwfpm-READY]\n";
            fout << "ruleType=txtVal\n";
            fout << "property=fwfpm.fsm_state\n";
            fout << "element=state\n";
            fout << "target=READY\n";
            fout << "\n";
            fout << "[fwfpm-fpm-READY]\n";
            fout << "ruleType=ruleComp\n";
            fout << "comp=And\n";
            fout << "rule1=fwfpm-READY\n";
            fout << "rule2=fwfpm-fpm\n";
            fout << "\n";
            fout << "[fwfpm-stagesci1-neq]\n";
            fout << "ruleType=elCompSw\n";
            fout << "property1=fwfpm.filterName\n";
            fout << "element1=fpm\n";
            fout << "property2=stagesci1.presetName\n";
            fout << "element2=fpm\n";
            fout << "comp=Neq\n";
            fout << "\n";
            fout << "[fwfpm-fpm-stagesci-fpm]\n";
            fout << "ruleType=ruleComp\n";
            fout << "priority=caution\n";
            fout << "rule1=fwfpm-fpm-READY\n";
            fout << "rule2=fwfpm-stagesci1-neq\n";
            fout << "comp=And\n";
            fout.close();

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            loadRuleConfig( maps, rrkMap, config );
            finalizeRuleValRules( maps, rrkMap );

            ruleCompRule *rcr = dynamic_cast<ruleCompRule *>( maps.rules["fwfpm-fpm-stagesci-fpm"] );

            const indiCompRule *r1 = rcr->rule1();
            const indiCompRule *r2 = rcr->rule2();

            REQUIRE( r1 == maps.rules["fwfpm-fpm-READY"] );
            REQUIRE( r2 == maps.rules["fwfpm-stagesci1-neq"] );
        }
    }
}

SCENARIO( "rule configurations with errors", "[stateRuleEngine::ruleConfig]" )
{
    GIVEN( "single rules in a config file" )
    {
        WHEN( "no rule sections given" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", { "rule1" }, { "property" }, { "dev.prop" } );
            mx::app::appConfigurator config;
            // By adding this to the config list we remove if from the "unused" so it won't get detected by
            // loadRuleConfig
            config.add( "rule1.prop", "", "", argType::Required, "rule1", "property", false, "string", "" );
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( const mx::exception<mx::verbose::d> &e )
            {
                caught = true;
            }
            catch( ... )
            {
            }

            REQUIRE( caught == true );
        }

        WHEN( "an invalid rule" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "property", "element", "target" },
                                      { "badRule", "dev.prop", "elem", "1.234" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( const mx::exception<mx::verbose::d> &e )
            {
                caught = true;
            }
            catch( ... )
            {
            }

            REQUIRE( caught == true );
        }
    }
    GIVEN( "ruleComp rules with errors" )
    {
        WHEN( "a ruleCompRule with rule1 not found" )
        {
            // This requires configuring the other rules too
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleA", "ruleA", "ruleA", "rule3", "rule3", "rule3", "rule3", "rule4", "rule4", "rule4", "rule4" },
                { "ruleType",
                  "rule1",
                  "rule2",
                  "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "property",
                  "element",
                  "target" },
                { "ruleComp",
                  "rule6",
                  "rule4",
                  "txtVal",
                  "dev3.propQ",
                  "elem",
                  "xxx",
                  "txtVal",
                  "dev4.propR",
                  "mele",
                  "yyy" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
                finalizeRuleValRules( maps, rrkMap );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "a ruleCompRule with rule1 self-referencing" )
        {
            // This requires configuring the other rules too
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleA", "ruleA", "ruleA", "rule3", "rule3", "rule3", "rule3", "rule4", "rule4", "rule4", "rule4" },
                { "ruleType",
                  "rule1",
                  "rule2",
                  "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "property",
                  "element",
                  "target" },
                { "ruleComp",
                  "ruleA",
                  "rule4",
                  "txtVal",
                  "dev3.propQ",
                  "elem",
                  "xxx",
                  "txtVal",
                  "dev4.propR",
                  "mele",
                  "yyy" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }
        WHEN( "a ruleCompRule with rule2 not found" )
        {
            // This requires configuring the other rules too
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleA", "ruleA", "ruleA", "rule3", "rule3", "rule3", "rule3", "rule4", "rule4", "rule4", "rule4" },
                { "ruleType",
                  "rule1",
                  "rule2",
                  "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "property",
                  "element",
                  "target" },
                { "ruleComp",
                  "rule3",
                  "rule5",
                  "txtVal",
                  "dev3.propQ",
                  "elem",
                  "xxx",
                  "txtVal",
                  "dev4.propR",
                  "mele",
                  "yyy" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
                finalizeRuleValRules( maps, rrkMap );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "a ruleCompRule with rule2 self-referencing" )
        {
            // This requires configuring the other rules too
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleA", "ruleA", "ruleA", "rule3", "rule3", "rule3", "rule3", "rule4", "rule4", "rule4", "rule4" },
                { "ruleType",
                  "rule1",
                  "rule2",
                  "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "property",
                  "element",
                  "target" },
                { "ruleComp",
                  "rule3",
                  "ruleA",
                  "txtVal",
                  "dev3.propQ",
                  "elem",
                  "xxx",
                  "txtVal",
                  "dev4.propR",
                  "mele",
                  "yyy" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
                finalizeRuleValRules( maps, rrkMap );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }
    }

    GIVEN( "multiSwitchCombo rules with errors" )
    {
        WHEN( "numSwitches is zero" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "numSwitches", "property1", "format", "targetProperty" },
                                      { "multiSwitchCombo", "0", "dev1.prop1", "{}", "dev2.prop2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "a required propertyK is missing" )
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf",
                                      { "rule1", "rule1", "rule1", "rule1", "rule1" },
                                      { "ruleType", "numSwitches", "property1", "format", "targetProperty" },
                                      { "multiSwitchCombo", "2", "dev1.prop1", "{}-{}", "dev2.prop2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "a source property conflicts with a non-switch rule type" )
        {
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "ruleText",
                  "ruleText",
                  "ruleText",
                  "ruleText",
                  "ruleCombo",
                  "ruleCombo",
                  "ruleCombo",
                  "ruleCombo",
                  "ruleCombo" },
                { "ruleType",
                  "property",
                  "element",
                  "target",
                  "ruleType",
                  "numSwitches",
                  "property1",
                  "format",
                  "targetProperty" },
                { "txtVal", "dev1.prop1", "elem", "xxx", "multiSwitchCombo", "1", "dev1.prop1", "{}", "dev2.prop2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "the comparison operator is not valid" )
        {
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                { "ruleType", "comp", "numSwitches", "property1", "property2", "format", "targetProperty" },
                { "multiSwitchCombo", "And", "2", "dev1.prop1", "dev2.prop2", "{}-{}", "dev3.prop3" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }

        WHEN( "the format placeholder count does not match numSwitches" )
        {
            mx::app::writeConfigFile(
                "/tmp/ruleConfig_test.conf",
                { "rule1", "rule1", "rule1", "rule1", "rule1", "rule1" },
                { "ruleType", "numSwitches", "property1", "property2", "format", "targetProperty" },
                { "multiSwitchCombo", "2", "dev1.prop1", "dev2.prop2", "{}", "dev3.prop3" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/ruleConfig_test.conf" );

            indiRuleMaps                        maps;
            std::map<std::string, ruleRuleKeys> rrkMap;

            bool caught = false;
            try
            {
                loadRuleConfig( maps, rrkMap, config );
            }
            catch( ... )
            {
                caught = true;
            }

            REQUIRE( caught == true );
        }
    }
}

} // namespace stateRuleEngineTest

} // namespace libXWCTest
