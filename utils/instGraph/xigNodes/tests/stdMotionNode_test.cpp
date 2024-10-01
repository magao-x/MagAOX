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
                                                                      {"stdMotion"} );
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
            REQUIRE(tsn->trackingReqKey() == "");
            REQUIRE(tsn->trackingReqElement() == "");
            REQUIRE(tsn->trackerKey() == "");
            REQUIRE(tsn->trackerElement() == "");
        }

        WHEN("node is in file, full config")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",  "fwtelsim",  "fwtelsim",     "fwtelsim",  "fwtelsim",      "fwtelsim",      "fwtelsim",          "fwtelsim",   "fwtelsim",},
                                                                      {"type",      "device",    "presetPrefix", "presetDir", "presetPutName", "trackingReqKey", "trackingReqElement", "trackerKey", "trackerElement"},
                                                                      {"stdMotion", "devtelsim", "filter",       "input",     "filt1,filt2",   "labrules.info",  "trackreq",          "adc.track",  "toggle"} );
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
                                                                      {"stdMotion"} );
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
                                                                      {"stdMotion"} );
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
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",  "fwtelsim"},
                                                                      {"type",      "device"},
                                                                      {"stdMotion", "device2"} );
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
                                                                      {"stdMotion", "preset2"} );
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
                                                                      {"stdMotion", "wrongput"} );
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
                                                                      {"stdMotion", ""} );
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

        WHEN("config invalid: only trackingReqKey provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",  "fwtelsim"},
                                                                      {"type",      "trackingReqKey"},
                                                                      {"stdMotion", "labrules.info"} );
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

        WHEN("config invalid: only trackingReqElement provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "trackingReqElement"},
                                                                      {"stdMotion", "toggle"} );
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

        WHEN("config invalid: only trackerKey provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim", "fwtelsim"},
                                                                      {"type", "trackerKey"},
                                                                      {"stdMotion", "adctrack.tracking"} );
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
                                                                      {"stdMotion", "toggle"} );
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

        WHEN("config invalid: only trackingReqKey and trackingReqElement provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",  "fwtelsim",       "fwtelsim"},
                                                                      {"type",      "trackingReqKey", "trackingReqElement"},
                                                                      {"stdMotion", "labrules.info",  "adcTrackingReq"} );
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

        WHEN("config invalid: only trackerKey and trackerElement provided")
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", {"fwtelsim",  "fwtelsim",          "fwtelsim"},
                                                                      {"type",      "trackerKey",        "trackerElement"},
                                                                      {"stdMotion", "adctrack.tracking", "toggle"} );
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
    }
}

SCENARIO( "Sending Properties to a stdMotionNode", "[instGraph::stdMotionNode]" )
{
    GIVEN("a configured stdMotionNode with tracking")
    {
        //First configure the node
        ingr::instGraphXML parentGraph;
        writeXML();

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

        tsn->device("fwtelsim");
        tsn->presetPrefix("filter");
        tsn->trackingReqKey("labrules.info");
        tsn->trackingReqElement("adcTrackReq");
        tsn->trackerKey("adctrack.tracking");
        tsn->trackerElement("toggle");

        WHEN("tracking off, tracking not rquired")
        {
            pcf::IndiProperty ipSend;
            ipSend.setDevice("adctrack");
            ipSend.setName("tracking");
            ipSend.add(pcf::IndiElement("toggle"));
            ipSend["toggle"].setSwitchState(pcf::IndiElement::Off);

            tsn->handleSetProperty(ipSend);

            pcf::IndiProperty ipSend2;
            ipSend2.setDevice("labrules");
            ipSend2.setName("info");
            ipSend2.add(pcf::IndiElement("adcTrackReq"));
            ipSend2["adcTrackReq"].setSwitchState(pcf::IndiElement::Off);

            tsn->handleSetProperty(ipSend2);

            pcf::IndiProperty ipSend3;
            ipSend3.setDevice("fwtelsim");
            ipSend3.setName("filterName");
            ipSend3.add(pcf::IndiElement("filt1"));
            ipSend3["filt1"].setSwitchState(pcf::IndiElement::On);
            ipSend3.add(pcf::IndiElement("none"));
            ipSend3["none"].setSwitchState(pcf::IndiElement::Off);

            tsn->handleSetProperty(ipSend3);

            REQUIRE(tsn->curLabel() == "off");

            pcf::IndiProperty ipSend4;
            ipSend4.setDevice("fwtelsim");
            ipSend4.setName("fsm");
            ipSend4.add(pcf::IndiElement("state"));
            ipSend4["state"].set("READY");

            tsn->handleSetProperty(ipSend4);

            REQUIRE(tsn->curLabel() == "filt1");

            ipSend2["adcTrackReq"].setSwitchState(pcf::IndiElement::On);
            tsn->handleSetProperty(ipSend2);

            REQUIRE(tsn->curLabel() == "not tracking");

            ipSend["toggle"].setSwitchState(pcf::IndiElement::On);
            tsn->handleSetProperty(ipSend);

            REQUIRE(tsn->curLabel() == "tracking");

            ipSend4["state"].set("OPERATING");
            tsn->handleSetProperty(ipSend4);

            REQUIRE(tsn->curLabel() == "tracking");

            ipSend2["adcTrackReq"].setSwitchState(pcf::IndiElement::Off);
            tsn->handleSetProperty(ipSend2);

            //Now we're still tracking and OPERATING, but shouldn't be
            REQUIRE(tsn->curLabel() == "tracking");

            ipSend["toggle"].setSwitchState(pcf::IndiElement::Off);
            tsn->handleSetProperty(ipSend);

            //Now we're not tracking, but still OPERATING and in filt1
            REQUIRE(tsn->curLabel() == "off");

            ipSend3["filt1"].setSwitchState(pcf::IndiElement::Off);
            tsn->handleSetProperty(ipSend3);

            ipSend4["state"].set("READY");
            tsn->handleSetProperty(ipSend4);

            //Now we're in READY but nothing is on
            REQUIRE(tsn->curLabel() == "off");

            ipSend3["filt1"].setSwitchState(pcf::IndiElement::On);
            tsn->handleSetProperty(ipSend3);

            //Now filt1 is on
            REQUIRE(tsn->curLabel() == "filt1");

            ipSend3["filt1"].setSwitchState(pcf::IndiElement::Off);
            ipSend3["none"].setSwitchState(pcf::IndiElement::On);

            tsn->handleSetProperty(ipSend3);

            //Now none is on
            REQUIRE(tsn->curLabel() == "off");

            ipSend3["filt1"].setSwitchState(pcf::IndiElement::On);
            ipSend3["none"].setSwitchState(pcf::IndiElement::Off);

            tsn->handleSetProperty(ipSend3);

            //Now filt1 is back on
            REQUIRE(tsn->curLabel() == "filt1");
        }
    }
}
