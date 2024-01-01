#include "../../../tests/catch2/catch.hpp"

#include "../indiCompRuleConfig.hpp"

SCENARIO( "configuring basic rules", "[stateRuleEngine::ruleConfig]" ) 
{
    GIVEN("single rules in a config file")
    {
        WHEN("a numValRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1",   "rule1" },
                                                                   {"ruleType", "property", "element", "target" },
                                                                   {"numVal",   "dev.prop",     "elem",    "1.234"  } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<numValRule*>(maps.rules["rule1"])->target() == 1.234);
            REQUIRE(static_cast<numValRule*>(maps.rules["rule1"])->tol() == 1e-6);
        }

        WHEN("a numValRule changing defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1", "rule1",    "rule1",   "rule1",  "rule1" },
                                                                   {"ruleType", "priority", "comp",  "property", "element", "target", "tol" },
                                                                   {"numVal",   "warning",  "GtEq",   "dev.prop",  "elem",    "1.234", "1e-8"  } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::warning);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::GtEq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<numValRule*>(maps.rules["rule1"])->target() == 1.234);
            REQUIRE(static_cast<numValRule*>(maps.rules["rule1"])->tol() == 1e-8);
        }

        WHEN("a txtValRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1",   "rule1" },
                                                                   {"ruleType", "property", "element", "target" },
                                                                   {"txtVal",   "dev.prop", "elem",    "xxx"  } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<txtValRule*>(maps.rules["rule1"])->target() == "xxx");
        }

        WHEN("a txtValRule changing defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1", "rule1",    "rule1",   "rule1"  },
                                                                   {"ruleType", "priority", "comp",  "property", "element", "target" },
                                                                   {"txtVal",   "alert",    "Neq",   "dev.prop", "elem",   "xxx" } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::alert);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Neq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<txtValRule*>(maps.rules["rule1"])->target() == "xxx");

        }

        WHEN("a swValRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1" },
                                                                   {"ruleType", "property", "element" },
                                                                   {"swVal",    "dev.prop", "elem"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<swValRule*>(maps.rules["rule1"])->target() == pcf::IndiElement::On);
        }

        WHEN("a swValRule changing defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1", "rule1",    "rule1",   "rule1"  },
                                                                   {"ruleType", "priority", "comp",  "property", "element", "target" },
                                                                   {"swVal",    "info",     "Neq",   "dev.prop", "elem",    "Off" } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::info);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Neq);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->property() == maps.props["dev.prop"]);
            REQUIRE(static_cast<onePropRule*>(maps.rules["rule1"])->element() == "elem");
            REQUIRE(static_cast<swValRule*>(maps.rules["rule1"])->target() == pcf::IndiElement::Off);

        }

        WHEN("an elCompNumRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",        "rule1",     "rule1",    "rule1",      "rule1"    },
                                                                   {"ruleType",     "property1", "element1", "property2",  "element2" },
                                                                   {"elCompNum", "dev1.prop1", "elem1",   "dev2.prop2", "elem2"   } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property1() == maps.props["dev1.prop1"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element1() == "elem1");
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property2() == maps.props["dev2.prop2"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element2() == "elem2");

        }

        WHEN("an elCompTxtRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",        "rule1",     "rule1",    "rule1",      "rule1"    },
                                                                   {"ruleType",     "property1", "element1", "property2",  "element2" },
                                                                   {"elCompTxt", "dev1.prop1", "elem1",   "dev2.prop2", "elem2"   } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property1() == maps.props["dev1.prop1"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element1() == "elem1");
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property2() == maps.props["dev2.prop2"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element2() == "elem2");

        }

        WHEN("an elCompSwRule using defaults")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",        "rule1",     "rule1",    "rule1",      "rule1"    },
                                                                   {"ruleType",     "property1", "element1", "property2",  "element2" },
                                                                   {"elCompSw", "dev1.prop1", "elem1",   "dev2.prop2", "elem2"   } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            REQUIRE(maps.rules["rule1"]->priority() == rulePriority::none);
            REQUIRE(maps.rules["rule1"]->comparison() == ruleComparison::Eq);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property1() == maps.props["dev1.prop1"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element1() == "elem1");
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->property2() == maps.props["dev2.prop2"]);
            REQUIRE(static_cast<twoPropRule*>(maps.rules["rule1"])->element2() == "elem2");

        }
    }
}

SCENARIO( "configuring the demo", "[stateRuleEngine::ruleConfig]" ) 
{
    GIVEN("the demo")
    {
        WHEN("the demo as writen")
        {
            std::ofstream fout;
            fout.open("/tmp/ruleConfig_test.conf");
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
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            loadRuleConfig(maps, config);

            ruleCompRule * rcr = dynamic_cast<ruleCompRule *>(maps.rules["fwfpm-fpm-stagesci-fpm"]);

            const indiCompRule * r1 = rcr->rule1();
            const indiCompRule * r2 = rcr->rule2();

            REQUIRE(r1 == maps.rules["fwfpm-fpm-READY"]);
            REQUIRE(r2 == maps.rules["fwfpm-stagesci1-neq"]);

        }
    }
}

SCENARIO( "rule configurations with errors", "[stateRuleEngine::ruleConfig]" ) 
{
    GIVEN("single rules in a config file")
    {
        WHEN("no rule sections given")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1" },
                                                                   {"property"},
                                                                   {"dev.prop"} );
            mx::app::appConfigurator config;
            //By adding this to the config list we remove if from the "unused" so it won't get detected by loadRuleConfig
            config.add("rule1.prop", "", "", argType::Required, "rule1", "property", false, "string", "");
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            bool caught = false;
            try
            {
                loadRuleConfig(maps, config);
            }
            catch(const mx::err::invalidconfig & e)
            {
                caught = true;
            }
            catch(...)
            {
            }

            REQUIRE(caught==true);
        }

        WHEN("an invalid rule")
        {
            mx::app::writeConfigFile( "/tmp/ruleConfig_test.conf", {"rule1",    "rule1",    "rule1",   "rule1" },
                                                                   {"ruleType", "property", "element", "target" },
                                                                   {"badRule",   "dev.prop",     "elem",    "1.234"  } );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/ruleConfig_test.conf");

            indiRuleMaps maps;

            bool caught = false;
            try
            {
                loadRuleConfig(maps, config);
            }
            catch(const mx::err::notimpl & e)
            {
                caught = true;
            }
            catch(...)
            {
            }

            REQUIRE(caught==true);
        }
    }
}
