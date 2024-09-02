//#define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../stdMotionNode.hpp"

void writeXML()
{
    std::ofstream fout("/tmp/xigNode_test.xml");
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:fwtelsim\">\n";
    fout <<                 "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

SCENARIO( "Creating and configuring a stdMotionNode", "[instGraph::stdMotionNode]" )
{
    GIVEN("a valid XML file, a valid config file")
    {
        WHEN("node is in file, default config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim"},
                                                                      {"type"},
                                                                      {"stdMotionNode"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

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

            //check defaults
            REQUIRE(tsn->device() == "fwtelsim");
            REQUIRE(tsn->presetPrefix() == "preset");
            REQUIRE(tsn->presetDir() == ingr::ioDir::output);
            REQUIRE(tsn->presetPutName().size() == 1);
            REQUIRE(tsn->presetPutName()[0] == "out");
            REQUIRE(tsn->trackerKey() == "");
            REQUIRE(tsn->trackerElement() == "");
        }

        WHEN("node is in file, full config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",     "fwtelsim",  "fwtelsim",     "fwtelsim",  "fwtelsim",      "fwtelsim",   "fwtelsim",},
                                                                      {"type",         "device",    "presetPrefix", "presetDir", "presetPutName", "trackerKey", "trackerElement"},
                                                                      {"stdMotionNode","devtelsim", "filter",       "input",     "filt1,filt2",          "adc.track",  "toggle"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

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
            REQUIRE(tsn->device() == "devtelsim");
            REQUIRE(tsn->presetPrefix() == "filter");
            REQUIRE(tsn->presetDir() == ingr::ioDir::input);
            REQUIRE(tsn->presetPutName().size() == 2);
            REQUIRE(tsn->presetPutName()[0] == "filt1");
            REQUIRE(tsn->presetPutName()[1] == "filt2");
            REQUIRE(tsn->trackerKey() == "adc.track");
            REQUIRE(tsn->trackerElement() == "toggle");
        }
    }
    GIVEN("an invalid parent graph")
    {
        WHEN("parent graph is null on construction")
        {
            ingr::instGraphXML * parentGraph = nullptr;

            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            // pass should be false b/c parentGraph being nullptr causes and exception
            REQUIRE(pass == false);
            REQUIRE(tsn == nullptr);
        }

        WHEN("valid xml, parent graph becomes null somehow")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim"},
                                                                      {"type"},
                                                                      {"stdMotionNode"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Set it to null for testing
            tsn->setParentGraphNull();

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
    }

    GIVEN("an invalid config file")
    {
        WHEN("node is in xml file, does not have type set in config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim"},
                                                                      {""},
                                                                      {""} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c type isn't set, so pass should stay false
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

        WHEN("node is in xml file, has wrong type in config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim"},
                                                                      {"type"},
                                                                      {"xigNode"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c type is wrong, so pass should stay false
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

        WHEN("node is in xml file, is not in config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"nonode"},
                                                                      {"type"},
                                                                      {"stdMotionNode"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c it doesn't have fwtelsim, so pass should stay false
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

        WHEN("config invalid: changing device")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "device"},
                                                                      {"stdMotionNode", "device2"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            tsn->device("device1");

            //Now we load the config, which should fail b/c device is already set, so pass should stay false
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

        WHEN("config invalid: changing presetName")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "presetPrefix"},
                                                                      {"stdMotionNode", "preset2"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            tsn->presetPrefix("preset1");

            //Now we load the config, which should fail b/c presetName is already set, so pass should stay false
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

        WHEN("config invalid: invalid presetDir")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "presetDir"},
                                                                      {"stdMotionNode", "wrongput"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c presetDir is neither input nor output, so pass should stay false
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

        WHEN("config invalid: presetPutName empty")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "presetPutName"},
                                                                      {"stdMotionNode", ""} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c presetPutName is empty, so pass should stay false
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

        WHEN("config invalid: only trackerKey provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "trackerKey"},
                                                                      {"stdMotionNode", "adctrack.tracking"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c trackerElement is empty, so pass should stay false
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

        WHEN("config invalid: only trackerElement provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "trackerElement"},
                                                                      {"stdMotionNode", "toggle"} );
            mx::app::appConfigurator config;
            config.readConfig("/tmp/stdMotionNode_test.conf");

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            // First we load the XML file which has fwtelsim
            stdMotionNode * tsn = nullptr;
            bool pass = false;
            try
            {
                tsn = new stdMotionNode("fwtelsim", &parentGraph);
                pass = true;
            }
            catch(const std::exception & e)
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE(pass == true);
            REQUIRE(tsn != nullptr);

            REQUIRE( tsn->name() == "fwtelsim");
            REQUIRE( tsn->node()->name() == "fwtelsim");

            //Now we load the config, which should fail b/c trackerKey is empty, so pass should stay false
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
    }
}


