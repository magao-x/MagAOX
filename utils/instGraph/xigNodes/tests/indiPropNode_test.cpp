//#define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../indiPropNode.hpp"

void writeXML()
{
    std::ofstream fout("/tmp/xigNode_test.xml");
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:telescope\">\n";
    fout <<                 "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

SCENARIO( "Creating and configuring an indiPropNode", "[instGraph::indiPropNode]" )
{
    GIVEN("a valid XML file, a valid config file")
    {
        WHEN("node is in file, default config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);

            //check config-ed values
            REQUIRE(tsn->propKey() == "tel.dome");
            REQUIRE(tsn->propEl() == "status");
            REQUIRE(tsn->propValStr() == "on");
            REQUIRE(tsn->propValNum() == std::numeric_limits<double>::lowest());
            REQUIRE(tsn->propValSw() == pcf::IndiElement::SwitchStateType::UnknownSwitchState);
            REQUIRE(tsn->type() == pcf::IndiProperty::Type::Unknown);
            REQUIRE(tsn->tol() == 1e-7);
            REQUIRE(tsn->state() == false);

        }
        WHEN("node is in file, config setting tol")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal", "tol"},
                                                                      {"indiProp", "tel.dome", "status", "on", "1e-8"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);

            //check config-ed values
            REQUIRE(tsn->propKey() == "tel.dome");
            REQUIRE(tsn->propEl() == "status");
            REQUIRE(tsn->propValStr() == "on");
            REQUIRE(tsn->propValNum() == std::numeric_limits<double>::lowest());
            REQUIRE(tsn->propValSw() == pcf::IndiElement::SwitchStateType::UnknownSwitchState);
            REQUIRE(tsn->type() == pcf::IndiProperty::Type::Unknown);
            REQUIRE(tsn->tol() == 1e-8);
            REQUIRE(tsn->state() == false);

        }

        WHEN("node is in file, handling a text property")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            indiPropNode * tsn = new indiPropNode("telescope", &parentGraph);
            tsn->loadConfig(config);

            pcf::IndiProperty ip(pcf::IndiProperty::Text);
            ip.setDevice("tel");
            ip.setName("dome");
            ip.add(pcf::IndiElement("status"));
            ip["status"]="on";

            tsn->handleSetProperty(ip);
            REQUIRE(tsn->type() == pcf::IndiProperty::Text);
            REQUIRE(tsn->state() == true);

            ip["status"]="off";
            tsn->handleSetProperty(ip);
            REQUIRE(tsn->state() == false);
        }

        WHEN("node is in file, handling a number property")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "1.5"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            indiPropNode * tsn = new indiPropNode("telescope", &parentGraph);
            tsn->loadConfig(config);

            pcf::IndiProperty ip(pcf::IndiProperty::Number);
            ip.setDevice("tel");
            ip.setName("dome");
            ip.add(pcf::IndiElement("status"));
            ip["status"]="1.5";

            tsn->handleSetProperty(ip);
            REQUIRE(tsn->type() == pcf::IndiProperty::Number);
            REQUIRE(tsn->propValNum() == 1.5);
            REQUIRE(tsn->state() == true);


            ip["status"]="1.6";
            tsn->handleSetProperty(ip);
            REQUIRE(tsn->state() == false);
        }

        WHEN("node is in file, handling a switch property")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            indiPropNode * tsn = new indiPropNode("telescope", &parentGraph);
            tsn->loadConfig(config);

            pcf::IndiProperty ip(pcf::IndiProperty::Switch);
            ip.setDevice("tel");
            ip.setName("dome");
            ip.add(pcf::IndiElement("status"));
            ip["status"].setSwitchState(pcf::IndiElement::SwitchStateType::On);

            tsn->handleSetProperty(ip);
            REQUIRE(tsn->type() == pcf::IndiProperty::Switch);
            REQUIRE(tsn->propValSw() == pcf::IndiElement::SwitchStateType::On);
            REQUIRE(tsn->state() == true);


            ip["status"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);
            tsn->handleSetProperty(ip);
            REQUIRE(tsn->state() == false);
        }
    }

    GIVEN("invalid configs")
    {
        WHEN("parent graph is null, default config")
        {
            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", nullptr);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);
            REQUIRE(tsn == nullptr);
        }
        WHEN("node not in file, default config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope2", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);
            REQUIRE(tsn == nullptr);
        }

        WHEN("node is in file, default config, wrong node type")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"fakeProp", "tel.dome", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);

        }
        WHEN("node is in file, default config, propKey empty")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "", "status", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);

        }
        WHEN("node is in file, default config, propEl empty")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "", "on"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);

        }
        WHEN("node is in file, default config, propVal empty")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.come", "status", ""} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            indiPropNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new indiPropNode("telescope", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "telescope");
            REQUIRE( tsn->node()->name() == "telescope");

            pass = false;
            try
            {
                tsn->loadConfig(config);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == false);

        }
        WHEN("a number property with invalid propVal")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "abcde"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            indiPropNode * tsn = new indiPropNode("telescope", &parentGraph);
            tsn->loadConfig(config);

            pcf::IndiProperty ip(pcf::IndiProperty::Number);
            ip.setDevice("tel");
            ip.setName("dome");
            ip.add(pcf::IndiElement("status"));
            ip["status"]="1.5";

            bool pass = false;
            try
            {
                tsn->handleSetProperty(ip);
                pass = true;
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }

            REQUIRE(pass == false);
        }
        WHEN("invalid switch property")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf", {"telescope","telescope","telescope","telescope"},
                                                                      {"type", "propKey", "propEl", "propVal"},
                                                                      {"indiProp", "tel.dome", "status", "qq"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/indiPropNode_test.conf");

            std::string emsg;

            parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            indiPropNode * tsn = new indiPropNode("telescope", &parentGraph);
            tsn->loadConfig(config);

            pcf::IndiProperty ip(pcf::IndiProperty::Switch);
            ip.setDevice("tel");
            ip.setName("dome");
            ip.add(pcf::IndiElement("status"));
            ip["status"].setSwitchState(pcf::IndiElement::SwitchStateType::On);

            bool pass = false;
            try
            {
                tsn->handleSetProperty(ip);
                pass = true;
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }

            REQUIRE(pass == false);
        }
    }
}
