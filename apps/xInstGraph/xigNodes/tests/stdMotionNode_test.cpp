/** \file stdMotionNode_test.cpp
 * \brief Catch2 tests for the xInstGraph `stdMotionNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup instGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../stdMotionNode.hpp"

namespace libXWCTest
{

/** \addtogroup xInstGraph_unit_test
 * \brief Additional unit tests for the xInstGraph application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xInstGraph` node unit tests.
/** \ingroup xInstGraph_unit_test
 */
namespace xInstGraphTest
{

/// Write the minimal motion-stage graph used by configuration tests.
void writeXML()
{
    std::ofstream fout( "/tmp/xigNode_test.xml" );
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:fwtelsim\"/>\n";
    for( const auto &put : { "in", "filt1", "filt2" } )
    {
        fout << "               <mxCell id=\"input:fwtelsim:" << put << "\" style=\"strokeColor=#FF0000;\"/>\n";
    }
    fout << "               <mxCell id=\"output:fwtelsim:out\" style=\"strokeColor=#FF0000;\"/>\n";
    fout << "               <mxCell id=\"state:fwtelsim\" value=\"state\"/>\n";
    fout << "               <mxCell id=\"fsmstate:fwtelsim\" value=\"fsmstate\"/>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

/// Verify default, explicit, and invalid motion-stage configuration.
/** \ingroup xInstGraph_unit_test
 */
SCENARIO( "Creating and configuring a stdMotionNode", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::device();
    #endif
    // clang-format on

    GIVEN( "a valid XML file, a valid config file" )
    {
        WHEN( "node is in file, default config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", { "fwtelsim" }, { "type" }, { "stdMotion" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );

            // check defaults
            REQUIRE( tsn->device() == "fwtelsim" );
            REQUIRE( tsn->presetPrefix() == "preset" );
            REQUIRE( tsn->presetDir() == ingr::ioDir::output );
            REQUIRE( tsn->presetPutName().size() == 1 );
            REQUIRE( tsn->presetPutName()[0] == "out" );
            REQUIRE( tsn->trackingReqKey() == "" );
            REQUIRE( tsn->trackingReqElement() == "" );
            REQUIRE( tsn->trackerKey() == "" );
            REQUIRE( tsn->trackerElement() == "" );
        }

        WHEN( "node is in file, full config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      {
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                          "fwtelsim",
                                      },
                                      { "type",
                                        "device",
                                        "presetPrefix",
                                        "presetDir",
                                        "presetPutName",
                                        "trackingReqKey",
                                        "trackingReqElement",
                                        "trackerKey",
                                        "trackerElement" },
                                      { "stdMotion",
                                        "devtelsim",
                                        "filter",
                                        "input",
                                        "filt1,filt2",
                                        "labrules.info",
                                        "trackreq",
                                        "adc.track",
                                        "toggle" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn->device() == "devtelsim" );
            REQUIRE( tsn->presetPrefix() == "filter" );
            REQUIRE( tsn->presetDir() == ingr::ioDir::input );
            REQUIRE( tsn->presetPutName().size() == 2 );
            REQUIRE( tsn->presetPutName()[0] == "filt1" );
            REQUIRE( tsn->presetPutName()[1] == "filt2" );
            REQUIRE( tsn->trackerKey() == "adc.track" );
            REQUIRE( tsn->trackerElement() == "toggle" );
        }
    }
    GIVEN( "an invalid parent graph" )
    {
        WHEN( "parent graph is null on construction" )
        {
            ingr::instGraphXML *parentGraph = nullptr;

            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            // pass should be false b/c parentGraph being nullptr causes and exception
            REQUIRE( pass == false );
            REQUIRE( tsn == nullptr );
        }

        WHEN( "valid xml, parent graph becomes null somehow" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", { "fwtelsim" }, { "type" }, { "stdMotion" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Set it to null for testing
            tsn->setParentGraphNull();

            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }
    }

    GIVEN( "an invalid config file" )
    {
        WHEN( "node is in xml file, does not have type set in config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", { "fwtelsim" }, { "" }, { "" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c type isn't set, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "node is in xml file, has wrong type in config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", { "fwtelsim" }, { "type" }, { "xigNode" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c type is wrong, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "node is in xml file, is not in config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf", { "nonode" }, { "type" }, { "stdMotion" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c it doesn't have fwtelsim, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: changing device" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "device" },
                                      { "stdMotion", "device2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            tsn->device( "device1" );

            // Now we load the config, which should fail b/c device is already set, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: changing presetName" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "presetPrefix" },
                                      { "stdMotion", "preset2" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            tsn->presetPrefix( "preset1" );

            // Now we load the config, which should fail b/c presetName is already set, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: invalid presetDir" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "presetDir" },
                                      { "stdMotion", "wrongput" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c presetDir is neither input nor output, so pass should stay
            // false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: presetPutName empty" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "presetPutName" },
                                      { "stdMotion", "" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c presetPutName is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackingReqKey provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "trackingReqKey" },
                                      { "stdMotion", "labrules.info" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerElement is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackingReqElement provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "trackingReqElement" },
                                      { "stdMotion", "toggle" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerKey is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackerKey provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "trackerKey" },
                                      { "stdMotion", "adctrack.tracking" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerElement is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackerElement provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim" },
                                      { "type", "trackerElement" },
                                      { "stdMotion", "toggle" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerKey is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackingReqKey and trackingReqElement provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim", "fwtelsim" },
                                      { "type", "trackingReqKey", "trackingReqElement" },
                                      { "stdMotion", "labrules.info", "adcTrackingReq" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerElement is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }

        WHEN( "config invalid: only trackerKey and trackerElement provided" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/stdMotionNode_test.conf",
                                      { "fwtelsim", "fwtelsim", "fwtelsim" },
                                      { "type", "trackerKey", "trackerElement" },
                                      { "stdMotion", "adctrack.tracking", "toggle" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/stdMotionNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            // First we load the XML file which has fwtelsim
            stdMotionNode *tsn  = nullptr;
            bool           pass = false;
            try
            {
                tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "fwtelsim" );
            REQUIRE( tsn->node()->name() == "fwtelsim" );

            // Now we load the config, which should fail b/c trackerElement is empty, so pass should stay false
            pass = false;
            try
            {
                tsn->loadConfig( config );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
        }
    }
}

/// Write a motion-stage graph with configurable input and output puts.
void writeMotionXML( const std::string              &path,   /**< [in] graph path */
                     const std::vector<std::string> &inputs, /**< [in] input put names */
                     const std::vector<std::string> &outputs /**< [in] output put names */ )
{
    std::ofstream out( path );
    out << "<mxfile><diagram><mxGraphModel><root>\n"
           "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>\n"
           "<mxCell id=\"node:fwtelsim\"/>\n";
    for( const auto &name : inputs )
    {
        out << "<mxCell id=\"input:fwtelsim:" << name << "\" style=\"strokeColor=#FF0000;\"/>\n";
    }
    for( const auto &name : outputs )
    {
        out << "<mxCell id=\"output:fwtelsim:" << name << "\" style=\"strokeColor=#FF0000;\"/>\n";
    }
    out << "<mxCell id=\"state:fwtelsim\" value=\"state\"/>\n"
           "<mxCell id=\"fsmstate:fwtelsim\" value=\"fsmstate\"/>\n"
           "</root></mxGraphModel></diagram></mxfile>\n";
}

/// Turn off alwaysOn puts when a stage leaves READY.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode clears alwaysOn puts after leaving READY", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::handleSetProperty( *(pcf::IndiProperty *)nullptr );
    stdMotionNode::togglePutsOff();
    #endif
    // clang-format on

    const std::string xmlPath    = "/tmp/stdMotionNode_F06_test.drawio";
    const std::string configPath = "/tmp/stdMotionNode_F06_test.conf";
    writeMotionXML( xmlPath, { "in" }, { "out", "ref" } );
    mx::app::writeConfigFile( configPath,
                              { "fwtelsim", "fwtelsim", "fwtelsim" },
                              { "type", "presetPutName", "alwaysOn" },
                              { "stdMotion", "out,ref", "ref" } );

    ingr::instGraphXML graph;
    graph.autoSave( false );
    std::string error;
    REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );
    mx::app::appConfigurator config;
    REQUIRE( config.readConfig( configPath ) == 0 );
    stdMotionNode node( "fwtelsim", &graph );
    REQUIRE_NOTHROW( node.loadConfig( config ) );

    pcf::IndiProperty fsm;
    fsm.setDevice( "fwtelsim" );
    fsm.setName( "fsm" );
    fsm.add( pcf::IndiElement( "state" ) );
    fsm["state"] = "READY";
    REQUIRE( node.handleSetProperty( fsm ) == 0 );

    pcf::IndiProperty preset( pcf::IndiProperty::Switch );
    preset.setDevice( "fwtelsim" );
    preset.setName( "presetName" );
    preset.add( pcf::IndiElement( "out" ) );
    preset["out"].setSwitchState( pcf::IndiElement::On );
    preset.add( pcf::IndiElement( "ref" ) );
    preset["ref"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( node.handleSetProperty( preset ) == 0 );
    REQUIRE( graph.node( "fwtelsim" )->output( "out" )->state() == ingr::putState::on );
    REQUIRE( graph.node( "fwtelsim" )->output( "ref" )->state() == ingr::putState::on );

    fsm["state"] = "OPERATING";
    REQUIRE( node.handleSetProperty( fsm ) == 0 );
    REQUIRE( graph.node( "fwtelsim" )->output( "out" )->state() == ingr::putState::off );
    REQUIRE( graph.node( "fwtelsim" )->output( "ref" )->state() == ingr::putState::off );
}

/// Select a default put name from the configured preset direction.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode defaults the preset put by direction", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::presetDir();
    stdMotionNode::presetPutName();
    #endif
    // clang-format on

    const std::string xmlPath    = "/tmp/stdMotionNode_direction_default.drawio";
    const std::string configPath = "/tmp/stdMotionNode_direction_default.conf";
    writeMotionXML( xmlPath, { "in", "custom" }, { "out" } );

    std::vector<std::string> keys{ "type" };
    std::vector<std::string> values{ "stdMotion" };
    ingr::ioDir              expectedDirection = ingr::ioDir::output;
    std::string              expectedPut       = "out";

    SECTION( "default output direction selects out" )
    {
        expectedDirection = ingr::ioDir::output;
        expectedPut       = "out";
    }
    SECTION( "input direction selects in" )
    {
        keys.push_back( "presetDir" );
        values.push_back( "input" );
        expectedDirection = ingr::ioDir::input;
        expectedPut       = "in";
    }
    SECTION( "explicit input put overrides the directional default" )
    {
        keys.insert( keys.end(), { "presetDir", "presetPutName" } );
        values.insert( values.end(), { "input", "custom" } );
        expectedDirection = ingr::ioDir::input;
        expectedPut       = "custom";
    }

    mx::app::writeConfigFile( configPath, std::vector<std::string>( keys.size(), "fwtelsim" ), keys, values );
    ingr::instGraphXML graph;
    graph.autoSave( false );
    std::string error;
    REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );
    mx::app::appConfigurator config;
    REQUIRE( config.readConfig( configPath ) == 0 );
    stdMotionNode node( "fwtelsim", &graph );
    REQUIRE_NOTHROW( node.loadConfig( config ) );
    REQUIRE( node.presetDir() == expectedDirection );
    REQUIRE( node.presetPutName() == std::vector<std::string>{ expectedPut } );
}

/// Reject missing opposite-side puts and configured put names absent from the graph.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode validates configured put topology", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    const std::string        xmlPath    = "/tmp/stdMotionNode_F07_test.drawio";
    const std::string        configPath = "/tmp/stdMotionNode_F07_test.conf";
    std::vector<std::string> inputs{ "in" };
    std::vector<std::string> outputs{ "out", "ref" };
    std::vector<std::string> keys{ "type", "presetPutName" };
    std::vector<std::string> values{ "stdMotion", "out,ref" };
    std::string              expected;

    SECTION( "multi-output stage without an input" )
    {
        inputs.clear();
        expected = "opposite-side put";
    }
    SECTION( "multi-input stage without an output" )
    {
        inputs = { "a", "b" };
        outputs.clear();
        keys.push_back( "presetDir" );
        values   = { "stdMotion", "a,b", "input" };
        expected = "opposite-side put";
    }
    SECTION( "missing selected put" )
    {
        values[1] = "out,missing";
        expected  = "presetPutName 'missing'";
    }
    SECTION( "missing alwaysOn put" )
    {
        keys.push_back( "alwaysOn" );
        values.push_back( "missing" );
        expected = "alwaysOn put 'missing'";
    }
    SECTION( "missing noAutoOn output" )
    {
        keys.push_back( "noAutoOn" );
        values.push_back( "missing" );
        expected = "noAutoOn output 'missing'";
    }

    writeMotionXML( xmlPath, inputs, outputs );
    mx::app::writeConfigFile( configPath, std::vector<std::string>( keys.size(), "fwtelsim" ), keys, values );
    ingr::instGraphXML graph;
    graph.autoSave( false );
    std::string error;
    REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );
    mx::app::appConfigurator config;
    REQUIRE( config.readConfig( configPath ) == 0 );
    stdMotionNode node( "fwtelsim", &graph );
    REQUIRE_THROWS_WITH( node.loadConfig( config ), Catch::Matchers::Contains( expected ) );
}

/// Guard a multi-put runtime update even if configuration validation was bypassed.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode guards missing opposite-side puts at runtime", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::togglePutsOn();
    #endif
    // clang-format on

    const std::string        xmlPath = "/tmp/stdMotionNode_F07_runtime.drawio";
    std::vector<std::string> inputs;
    std::vector<std::string> outputs{ "out", "ref" };
    std::vector<std::string> presetNames{ "out", "ref" };
    ingr::ioDir              direction = ingr::ioDir::output;
    std::string              expected  = "no input";

    SECTION( "multi-input without output" )
    {
        inputs = { "a", "b" };
        outputs.clear();
        presetNames = { "a", "b" };
        direction   = ingr::ioDir::input;
        expected    = "no output";
    }

    writeMotionXML( xmlPath, inputs, outputs );
    ingr::instGraphXML graph;
    graph.autoSave( false );
    std::string error;
    REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );
    stdMotionNode node( "fwtelsim", &graph );
    node.device( "fwtelsim" );
    node.presetPrefix( "preset" );
    node.presetDir( direction );
    node.presetPutName( presetNames );

    pcf::IndiProperty fsm;
    fsm.setDevice( "fwtelsim" );
    fsm.setName( "fsm" );
    fsm.add( pcf::IndiElement( "state" ) );
    fsm["state"] = "READY";
    REQUIRE( node.handleSetProperty( fsm ) == 0 );

    pcf::IndiProperty preset( pcf::IndiProperty::Switch );
    preset.setDevice( "fwtelsim" );
    preset.setName( "presetName" );
    for( const auto &name : presetNames )
    {
        preset.add( pcf::IndiElement( name ) );
        preset[name].setSwitchState( name == presetNames.front() ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }
    REQUIRE_THROWS_WITH( node.handleSetProperty( preset ), Catch::Matchers::Contains( expected ) );
}

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Isolated graph and configuration for parked motion-node tests.
struct parkedMotionFixture
{
    /// Directory owned by this fixture and removed at destruction.
    std::filesystem::path m_root;

    /// Parent graph, kept alive until the node is destroyed.
    ingr::instGraphXML m_graph;

    /// Configured handler owned by this fixture.
    std::unique_ptr<stdMotionNode> m_node;

    /// Configure a single-put or multi-put stage in either direction.
    parkedMotionFixture( ingr::ioDir        dir,              /**< [in] selected put direction */
                         bool               multi,            /**< [in] whether names select multiple puts */
                         const std::string &prefix,           /**< [in] preset or filter notation */
                         bool               tracking = false, /**< [in] configure tracking subscriptions */
                         const std::string &parkable = "true" /**< [in] parking option value; empty to omit */ );

    /// Remove the fixture's files.
    ~parkedMotionFixture();
};

parkedMotionFixture::parkedMotionFixture(
    ingr::ioDir dir, bool multi, const std::string &prefix, bool tracking, const std::string &parkable )
{
    char        name[] = "/tmp/parkedMotionNode_XXXXXX";
    const char *root   = ::mkdtemp( name );
    if( !root )
    {
        throw std::runtime_error( "could not create parked motion fixture" );
    }
    m_root = root;
    const std::vector<std::string> selected =
        multi ? std::vector<std::string>{ "routeA", "routeB", "ref" }
              : std::vector<std::string>{ dir == ingr::ioDir::input ? "in" : "out" };
    writeMotionXML( ( m_root / "graph.drawio" ).string(),
                    dir == ingr::ioDir::input ? selected : std::vector<std::string>{ "in" },
                    dir == ingr::ioDir::output ? selected : std::vector<std::string>{ "out" } );
    std::vector<std::string> keys{ "type", "presetDir", "presetPrefix", "presetPutName" };
    std::vector<std::string> values{ "stdMotion",
                                     dir == ingr::ioDir::input ? "input" : "output",
                                     prefix,
                                     multi ? "routeA,routeB,ref" : selected.front() };
    if( !parkable.empty() )
    {
        keys.push_back( "parkable" );
        values.push_back( parkable );
    }
    if( multi )
    {
        keys.insert( keys.end(), { "alwaysOn", "noAutoOn" } );
        values.insert( values.end(), { "ref", dir == ingr::ioDir::output ? "routeB" : "out" } );
    }
    if( tracking )
    {
        keys.insert( keys.end(), { "trackingReqKey", "trackingReqElement", "trackerKey", "trackerElement" } );
        values.insert( values.end(), { "labrules.info", "trackReq", "adctrack.tracking", "toggle" } );
    }
    mx::app::writeConfigFile(
        ( m_root / "config.conf" ).string(), std::vector<std::string>( keys.size(), "fwtelsim" ), keys, values );
    m_graph.autoSave( false );
    std::string              error;
    mx::app::appConfigurator config;
    if( m_graph.loadXMLFile( error, ( m_root / "graph.drawio" ).string() ) != 0 ||
        config.readConfig( ( m_root / "config.conf" ).string() ) != 0 )
    {
        throw std::runtime_error( "could not load parked motion fixture: " + error );
    }
    m_node = std::make_unique<stdMotionNode>( "fwtelsim", &m_graph );
    m_node->loadConfig( config );
}

parkedMotionFixture::~parkedMotionFixture()
{
    std::error_code error;
    std::filesystem::remove_all( m_root, error );
}

/// Build a stage FSM update.
pcf::IndiProperty motionFSM( const std::string &state /**< [in] reported FSM state */ )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text );
    property.setDevice( "fwtelsim" );
    property.setName( "fsm" );
    property.add( pcf::IndiElement( "state", state ) );
    return property;
}

/// Build a numeric parked update, including malformed numeric strings for rejection tests.
pcf::IndiProperty motionParked( const std::string &current /**< [in] numeric parking value */ )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number );
    property.setDevice( "fwtelsim" );
    property.setName( "parked" );
    property.add( pcf::IndiElement( "current", current ) );
    return property;
}

/// Build a current-position update, including malformed strings and alternate property types.
pcf::IndiProperty
motionPosition( const std::string      &current,                   /**< [in] numerical current position */
                const std::string      &propertyName = "position", /**< [in] numeric property name */
                pcf::IndiProperty::Type type         = pcf::IndiProperty::Number /**< [in] received property type */ )
{
    pcf::IndiProperty property( type );
    property.setDevice( "fwtelsim" );
    property.setName( propertyName );
    property.add( pcf::IndiElement( "current", current ) );
    return property;
}

/// Require identical effective states and enablement with and without numeric telemetry.
void requireSameMotionPuts( stdMotionNode &actual, /**< [in] handler receiving numeric telemetry */
                            stdMotionNode &expected /**< [in] handler receiving only original routing telemetry */ )
{
    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        const auto &puts = dir == ingr::ioDir::input ? actual.node()->inputs() : actual.node()->outputs();
        for( const auto &put : puts )
        {
            auto *reference =
                dir == ingr::ioDir::input ? expected.node()->input( put.first ) : expected.node()->output( put.first );
            REQUIRE( put.second->state() == reference->state() );
            REQUIRE( put.second->enabled() == reference->enabled() );
        }
    }
}

/// Build a named-position snapshot with the requested names selected.
pcf::IndiProperty motionPreset( const std::vector<std::string> &selected, /**< [in] On element names */
                                const std::string              &prefix = "preset" /**< [in] preset property prefix */ )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch );
    property.setDevice( "fwtelsim" );
    property.setName( prefix + "Name" );
    for( const char *name : { "routeA", "routeB", "ref", "none" } )
    {
        property.add( pcf::IndiElement( name, pcf::IndiElement::Off ) );
    }
    for( const auto &name : selected )
    {
        if( !property.find( name ) )
        {
            property.add( pcf::IndiElement( name ) );
        }
        property[name].setSwitchState( pcf::IndiElement::On );
    }
    return property;
}

/// Require all puts, including any alwaysOn put, to be inactive.
void requireMotionOff( stdMotionNode &node /**< [in] stage handler */ )
{
    for( const auto &put : node.node()->inputs() )
    {
        REQUIRE( put.second->state() == ingr::putState::off );
    }
    for( const auto &put : node.node()->outputs() )
    {
        REQUIRE( put.second->state() == ingr::putState::off );
    }
}
/// \endcond

/// Parked startup snapshots converge and route retained presets in both directions.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode routes parked positions independently of message order",
           "[instGraph::stdMotionNode][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::device( "fwtelsim" );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::togglePutsOn();
    #endif
    // clang-format on

    for( const std::string state : { "POWEROFF", "POWERON", "NOTCONNECTED", "CONNECTED" } )
        for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
            for( const bool multi : { false, true } )
                for( const std::string prefix : { "preset", "filter" } )
                {
                    std::array<int, 3> order{ 0, 1, 2 };
                    do
                    {
                        CAPTURE( state, dir, multi, prefix, order );
                        parkedMotionFixture fixture( dir, multi, prefix );
                        auto               &node = *fixture.m_node;
                        REQUIRE( node.keys().count( "fwtelsim.parked" ) == 1 );
                        const std::array<pcf::IndiProperty, 3> snapshot{
                            motionFSM( state ), motionParked( "1" ), motionPreset( { "routeA" }, prefix ) };
                        for( int index : order )
                        {
                            REQUIRE( node.handleSetProperty( snapshot[index] ) == 0 );
                        }
                        REQUIRE( node.curLabel() == "routeA" );
                        auto stage = fixture.m_graph.node( "fwtelsim" );
                        if( multi )
                        {
                            auto selected =
                                dir == ingr::ioDir::input ? stage->input( "routeA" ) : stage->output( "routeA" );
                            auto other =
                                dir == ingr::ioDir::input ? stage->input( "routeB" ) : stage->output( "routeB" );
                            auto ref = dir == ingr::ioDir::input ? stage->input( "ref" ) : stage->output( "ref" );
                            REQUIRE( selected->state() == ingr::putState::on );
                            REQUIRE( other->state() == ingr::putState::off );
                            REQUIRE( ref->state() == ingr::putState::on );
                            REQUIRE( node.handleSetProperty( motionPreset( { "routeB" }, prefix ) ) == 0 );
                            REQUIRE( node.curLabel() == "routeB" );
                            REQUIRE( selected->state() == ingr::putState::off );
                            REQUIRE( other->state() == ingr::putState::on );
                            REQUIRE( other->enabled() );
                        }
                        else
                        {
                            REQUIRE( stage->input( "in" )->state() == ingr::putState::on );
                            REQUIRE( stage->output( "out" )->state() == ingr::putState::on );
                        }
                        std::string xml, error;
                        REQUIRE( fixture.m_graph.serializeXML( xml, error ) == 0 );
                        REQUIRE( xml.find( "value=\"" + state + "\"" ) != std::string::npos );
                        REQUIRE( node.handleSetProperty( motionParked( "0" ) ) == 0 );
                        requireMotionOff( node );
                        REQUIRE( node.curLabel() == "off" );
                        if( multi )
                        {
                            REQUIRE_FALSE( stage->output( dir == ingr::ioDir::output ? "routeB" : "out" )->enabled() );
                        }
                        REQUIRE( node.handleSetProperty( motionParked( "1.0" ) ) == 0 );
                        REQUIRE( node.curLabel() == ( multi ? "routeB" : "routeA" ) );
                        if( multi )
                        {
                            auto route =
                                dir == ingr::ioDir::input ? stage->input( "routeB" ) : stage->output( "routeB" );
                            auto common = dir == ingr::ioDir::input ? stage->output( "out" ) : stage->input( "in" );
                            REQUIRE( route->state() == ingr::putState::on );
                            REQUIRE( route->enabled() );
                            REQUIRE( common->state() == ingr::putState::on );
                            REQUIRE( common->enabled() );
                        }
                        else
                        {
                            REQUIRE( stage->input( "in" )->state() == ingr::putState::on );
                            REQUIRE( stage->output( "out" )->state() == ingr::putState::on );
                        }
                    } while( std::next_permutation( order.begin(), order.end() ) );
                }
}

/// Parking is requested and used only when the configuration explicitly enables that capability.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode subscribes to parking only when parkable", "[instGraph::stdMotionNode][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::device( "fwtelsim" );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    for( const std::string option : { "", "false", "true" } )
    {
        CAPTURE( option );
        const bool          enabled = option == "true";
        parkedMotionFixture fixture( ingr::ioDir::output, true, "preset", false, option );
        auto               &node = *fixture.m_node;
        REQUIRE( node.keys().count( "fwtelsim.parked" ) == ( enabled ? 1 : 0 ) );
        REQUIRE( node.keys().count( "fwtelsim.fsm" ) == 1 );
        REQUIRE( node.keys().count( "fwtelsim.presetName" ) == 1 );
        REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
        REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
        REQUIRE( node.curLabel() == "routeA" );
        REQUIRE( node.node()->output( "routeA" )->state() == ingr::putState::on );
        REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
        REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
        if( enabled )
        {
            REQUIRE( node.curLabel() == "routeA" );
            REQUIRE( node.node()->output( "routeA" )->state() == ingr::putState::on );
        }
        else
        {
            requireMotionOff( node );
            REQUIRE( node.curLabel() == "off" );
            REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
            requireMotionOff( node );
        }
    }
}

/// Parked routes require valid parking, one named selection, and a supported FSM state.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode rejects unusable parked positions", "[instGraph::stdMotionNode][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::togglePutsOff();
    #endif
    // clang-format on

    parkedMotionFixture fixture( ingr::ioDir::output, true, "preset" );
    auto               &node = *fixture.m_node;
    REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
    REQUIRE( node.curLabel() == "routeA" );
    REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
    requireMotionOff( node ); // Parking has not been published.
    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
    REQUIRE( node.curLabel() == "routeA" );

    SECTION( "Malformed or unavailable numeric parking clears affirmative parking" )
    {
        for( const std::string value : { "", "bad", "1 garbage", "nan", "inf", "1e999", "0" } )
        {
            CAPTURE( value );
            REQUIRE( node.handleSetProperty( motionParked( value ) ) == 0 );
            requireMotionOff( node );
            REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
            REQUIRE( node.curLabel() == "routeA" );
        }
        auto missing = motionParked( "1" );
        missing.remove( "current" );
        REQUIRE( node.handleSetProperty( missing ) == 0 );
        requireMotionOff( node );
        REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
        pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
        wrongType.setDevice( "fwtelsim" );
        wrongType.setName( "parked" );
        wrongType.add( pcf::IndiElement( "current", "1" ) );
        REQUIRE( node.handleSetProperty( wrongType ) == 0 );
        requireMotionOff( node );
    }

    SECTION( "Absent, none, ambiguous, or unmatched selections clear every route" )
    {
        for( const std::vector<std::string> &selected :
             { std::vector<std::string>{}, { "none" }, { "unknown" }, { "routeA", "routeB" }, { "none", "routeA" } } )
        {
            CAPTURE( selected );
            REQUIRE( node.handleSetProperty( motionPreset( selected ) ) == 0 );
            requireMotionOff( node );
            node.togglePutsOn(); // Direct toggles must enforce the same usability decision.
            requireMotionOff( node );
            REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
            REQUIRE( node.curLabel() == "routeA" );
        }
        pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
        wrongType.setDevice( "fwtelsim" );
        wrongType.setName( "presetName" );
        wrongType.add( pcf::IndiElement( "routeA", pcf::IndiElement::On ) );
        REQUIRE( node.handleSetProperty( wrongType ) == 0 );
        requireMotionOff( node );
    }

    SECTION( "Parking does not enable other unavailable states" )
    {
        for( const std::string state : { "HOMING", "NOTHOMED", "CONFIGURING", "LOGGEDIN", "ERROR", "invalid" } )
        {
            CAPTURE( state );
            REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
            requireMotionOff( node );
        }
    }
}

/// Tracking flags cannot override a parked preset, but normal tracking resumes with power.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode gives parked positions priority over tracking", "[instGraph::stdMotionNode][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::togglePutsOn();
    stdMotionNode::togglePutsOff();
    #endif
    // clang-format on

    parkedMotionFixture fixture( ingr::ioDir::output, false, "preset", true );
    auto               &node = *fixture.m_node;
    pcf::IndiProperty   requested( pcf::IndiProperty::Switch );
    requested.setDevice( "labrules" );
    requested.setName( "info" );
    requested.add( pcf::IndiElement( "trackReq", pcf::IndiElement::On ) );
    pcf::IndiProperty tracker( pcf::IndiProperty::Switch );
    tracker.setDevice( "adctrack" );
    tracker.setName( "tracking" );
    tracker.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
    REQUIRE( node.handleSetProperty( requested ) == 0 );
    REQUIRE( node.handleSetProperty( tracker ) == 0 );
    REQUIRE( node.curLabel() == "tracking" );
    REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
    REQUIRE( node.curLabel() == "routeA" );
    REQUIRE( node.handleSetProperty( motionPreset( { "routeB" } ) ) == 0 );
    REQUIRE( node.curLabel() == "routeB" );
    tracker["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( node.handleSetProperty( tracker ) == 0 );
    REQUIRE( node.curLabel() == "routeB" );
    requested["trackReq"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( node.handleSetProperty( requested ) == 0 );
    REQUIRE( node.curLabel() == "routeB" );
    tracker["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( node.handleSetProperty( tracker ) == 0 );
    REQUIRE( node.curLabel() == "routeB" );
    REQUIRE( node.handleSetProperty( motionPreset( { "none" } ) ) == 0 );
    REQUIRE( node.curLabel() == "off" );
    requireMotionOff( node );
    REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionParked( "0" ) ) == 0 );
    REQUIRE( node.curLabel() == "off" );
    requireMotionOff( node );
    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
    requested["trackReq"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( node.handleSetProperty( requested ) == 0 );
    REQUIRE( node.handleSetProperty( motionFSM( "OPERATING" ) ) == 0 );
    REQUIRE( node.curLabel() == "tracking" );
    tracker["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( node.handleSetProperty( tracker ) == 0 );
    REQUIRE( node.curLabel() == "not tracking" );
    requireMotionOff( node );
    REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
    requested["trackReq"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( node.handleSetProperty( requested ) == 0 );
    REQUIRE( node.curLabel() == "routeA" );
}

/// Verify a motion stage handles preset and tracking property updates.
/** \ingroup xInstGraph_unit_test
 */
SCENARIO( "Sending Properties to a stdMotionNode", "[instGraph::stdMotionNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::togglePutsOn();
    stdMotionNode::togglePutsOff();
    #endif
    // clang-format on

    GIVEN( "a configured stdMotionNode with tracking" )
    {
        // First configure the node
        ingr::instGraphXML parentGraph;
        parentGraph.autoSave( false );
        libXWCTest::xInstGraphTest::writeXML();

        std::string emsg;
        int         rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

        REQUIRE( rv == 0 );
        REQUIRE( emsg == "" );

        stdMotionNode *tsn  = nullptr;
        bool           pass = false;
        try
        {
            tsn  = new stdMotionNode( "fwtelsim", &parentGraph );
            pass = true;
        }
        catch( const std::exception &e )
        {
            std::cerr << e.what() << "\n";
        }

        REQUIRE( pass == true );
        REQUIRE( tsn != nullptr );

        REQUIRE( tsn->name() == "fwtelsim" );
        REQUIRE( tsn->node()->name() == "fwtelsim" );

        tsn->device( "fwtelsim" );
        tsn->presetPrefix( "filter" );
        tsn->trackingReqKey( "labrules.info" );
        tsn->trackingReqElement( "adcTrackReq" );
        tsn->trackerKey( "adctrack.tracking" );
        tsn->trackerElement( "toggle" );

        WHEN( "tracking off, tracking not rquired" )
        {
            pcf::IndiProperty ipSend;
            ipSend.setDevice( "adctrack" );
            ipSend.setName( "tracking" );
            ipSend.add( pcf::IndiElement( "toggle" ) );
            ipSend["toggle"].setSwitchState( pcf::IndiElement::Off );

            tsn->handleSetProperty( ipSend );

            pcf::IndiProperty ipSend2;
            ipSend2.setDevice( "labrules" );
            ipSend2.setName( "info" );
            ipSend2.add( pcf::IndiElement( "adcTrackReq" ) );
            ipSend2["adcTrackReq"].setSwitchState( pcf::IndiElement::Off );

            tsn->handleSetProperty( ipSend2 );

            pcf::IndiProperty ipSend3;
            ipSend3.setDevice( "fwtelsim" );
            ipSend3.setName( "filterName" );
            ipSend3.add( pcf::IndiElement( "filt1" ) );
            ipSend3["filt1"].setSwitchState( pcf::IndiElement::On );
            ipSend3.add( pcf::IndiElement( "none" ) );
            ipSend3["none"].setSwitchState( pcf::IndiElement::Off );

            tsn->handleSetProperty( ipSend3 );

            REQUIRE( tsn->curLabel() == "off" );

            pcf::IndiProperty ipSend4;
            ipSend4.setDevice( "fwtelsim" );
            ipSend4.setName( "fsm" );
            ipSend4.add( pcf::IndiElement( "state" ) );
            ipSend4["state"].set( "READY" );

            tsn->handleSetProperty( ipSend4 );

            REQUIRE( tsn->curLabel() == "filt1" );

            ipSend2["adcTrackReq"].setSwitchState( pcf::IndiElement::On );
            tsn->handleSetProperty( ipSend2 );

            REQUIRE( tsn->curLabel() == "not tracking" );

            ipSend["toggle"].setSwitchState( pcf::IndiElement::On );
            tsn->handleSetProperty( ipSend );

            REQUIRE( tsn->curLabel() == "tracking" );

            ipSend4["state"].set( "OPERATING" );
            tsn->handleSetProperty( ipSend4 );

            REQUIRE( tsn->curLabel() == "tracking" );

            ipSend2["adcTrackReq"].setSwitchState( pcf::IndiElement::Off );
            tsn->handleSetProperty( ipSend2 );

            // Now we're still tracking and OPERATING, but shouldn't be
            REQUIRE( tsn->curLabel() == "tracking" );

            ipSend["toggle"].setSwitchState( pcf::IndiElement::Off );
            tsn->handleSetProperty( ipSend );

            // Now we're not tracking, but still OPERATING and in filt1
            REQUIRE( tsn->curLabel() == "off" );

            ipSend3["filt1"].setSwitchState( pcf::IndiElement::Off );
            tsn->handleSetProperty( ipSend3 );

            ipSend4["state"].set( "READY" );
            tsn->handleSetProperty( ipSend4 );

            // Now we're in READY but nothing is on
            REQUIRE( tsn->curLabel() == "off" );

            ipSend3["filt1"].setSwitchState( pcf::IndiElement::On );
            tsn->handleSetProperty( ipSend3 );

            // Now filt1 is on
            REQUIRE( tsn->curLabel() == "filt1" );

            ipSend3["filt1"].setSwitchState( pcf::IndiElement::Off );
            ipSend3["none"].setSwitchState( pcf::IndiElement::On );

            tsn->handleSetProperty( ipSend3 );

            // Now none is on
            REQUIRE( tsn->curLabel() == "off" );

            ipSend3["filt1"].setSwitchState( pcf::IndiElement::On );
            ipSend3["none"].setSwitchState( pcf::IndiElement::Off );

            tsn->handleSetProperty( ipSend3 );

            // Now filt1 is back on
            REQUIRE( tsn->curLabel() == "filt1" );
        }
    }
}

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// A mapped stage with real internal links and one independent upstream source per input.
struct mappedMotionFixture
{
    /// Isolated files owned by the fixture.
    std::filesystem::path m_root;

    /// Parent graph retained until the handler is destroyed.
    ingr::instGraphXML m_graph;

    /// Configurator retained for assertions about consumed route keys.
    mx::app::appConfigurator m_config;

    /// Configured stage handler owned by the fixture.
    std::unique_ptr<stdMotionNode> m_node;

    /// Direction of the controlled branch puts.
    ingr::ioDir m_dir;

    /// Build a graph without configuring its stage, allowing configuration-failure checks.
    mappedMotionFixture( ingr::ioDir                     dir,           /**< [in] selected branch direction */
                         const std::vector<std::string> &puts,          /**< [in] selected-side graph puts */
                         bool                            linked = true, /**< [in] include required internal links */
                         size_t                          commonCount = 1 /**< [in] number of opposite-side puts */ );

    /// Remove the fixture's files.
    ~mappedMotionFixture();

    /// Load route rows and optional extra node settings.
    void load( const std::string &rows, /**< [in] route row text */
               const std::string &extra = "" /**< [in] additional config text */ );

    /// Change all upstream sources without delivering any stage telemetry.
    void sources( bool on /**< [in] whether incoming light is available */ );
};

mappedMotionFixture::mappedMotionFixture( ingr::ioDir                     dir,
                                          const std::vector<std::string> &puts,
                                          bool                            linked,
                                          size_t                          commonCount )
    : m_dir( dir )
{
    char        name[] = "/tmp/mappedMotionNode_XXXXXX";
    const char *root   = ::mkdtemp( name );
    if( !root )
    {
        throw std::runtime_error( "could not create mapped motion fixture" );
    }
    m_root = root;
    std::vector<std::string> common;
    for( size_t i = 0; i < commonCount; ++i )
    {
        common.push_back( ( dir == ingr::ioDir::output ? "in" : "out" ) + ( i ? std::to_string( i ) : "" ) );
    }
    const auto &inputs  = dir == ingr::ioDir::input ? puts : common;
    const auto &outputs = dir == ingr::ioDir::output ? puts : common;
    {
        std::ofstream xml( m_root / "graph.drawio" );
        xml << "<mxfile><diagram><mxGraphModel><root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
               "<mxCell id=\"node:fwtelsim\"/><mxCell id=\"state:fwtelsim\" value=\"before\"/>"
               "<mxCell id=\"fsmstate:fwtelsim\" value=\"before\"/>";
        for( const auto &put : inputs )
        {
            xml << "<mxCell id=\"input:fwtelsim:" << put << "\" value=\"" << put
                << "\" style=\"strokeColor=#FF0000;\"/>"
                << "<mxCell id=\"node:source_" << put << "\"/>"
                << "<mxCell id=\"output:source_" << put << ":out\" style=\"strokeColor=#FF0000;\"/>"
                << "<mxCell id=\"beam:source_" << put << "2stage\" source=\"output:source_" << put
                << ":out\" target=\"input:fwtelsim:" << put << "\" style=\"strokeColor=#FF0000;\"/>";
        }
        for( const auto &put : outputs )
        {
            xml << "<mxCell id=\"output:fwtelsim:" << put << "\" value=\"" << put
                << "\" style=\"strokeColor=#FF0000;\"/>";
        }
        if( linked )
        {
            for( const auto &input : inputs )
                for( const auto &output : outputs )
                {
                    xml << "<mxCell id=\"link:fwtelsim:" << input << "2" << output
                        << "\" source=\"input:fwtelsim:" << input << "\" target=\"output:fwtelsim:" << output
                        << "\" style=\"strokeColor=#FF0000;\"/>";
                }
        }
        xml << "</root></mxGraphModel></diagram></mxfile>";
    }
    m_graph.autoSave( false );
    std::string error;
    if( m_graph.loadXMLFile( error, ( m_root / "graph.drawio" ).string() ) != 0 )
    {
        throw std::runtime_error( "could not load mapped motion graph: " + error );
    }
}

mappedMotionFixture::~mappedMotionFixture()
{
    std::error_code error;
    std::filesystem::remove_all( m_root, error );
}

void mappedMotionFixture::load( const std::string &rows, const std::string &extra )
{
    {
        std::ofstream config( m_root / "config.conf" );
        config << "[fwtelsim]\ntype=stdMotion\npresetDir=" << ( m_dir == ingr::ioDir::input ? "input" : "output" )
               << '\n'
               << rows << extra;
    }
    REQUIRE( m_config.readConfig( ( m_root / "config.conf" ).string() ) == 0 );
    m_node = std::make_unique<stdMotionNode>( "fwtelsim", &m_graph );
    m_node->loadConfig( m_config );
}

void mappedMotionFixture::sources( bool on )
{
    for( const auto &input : m_graph.node( "fwtelsim" )->inputs() )
    {
        m_graph.node( "source_" + input.first )
            ->output( "out" )
            ->state( on ? ingr::putState::on : ingr::putState::off );
    }
}

/// Check the complete route mask and its propagated effective states.
void requireMappedRoute( stdMotionNode               &node,     /**< [in] configured stage */
                         const std::set<std::string> &selected, /**< [in] expected active branch names */
                         ingr::putState state = ingr::putState::on /**< [in] effective state of permitted puts */ )
{
    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        const auto &puts = dir == ingr::ioDir::input ? node.node()->inputs() : node.node()->outputs();
        for( const auto &put : puts )
        {
            const bool enabled = dir == node.presetDir() ? selected.count( put.first ) != 0 : !selected.empty();
            REQUIRE( put.second->enabled() == enabled );
            REQUIRE( put.second->state() == ( enabled ? state : ingr::putState::off ) );
        }
    }
}
/// \endcond

/// The supplied beamsplitter matrices work in both directions and follow upstream light changes.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode maps beamsplitter presets and propagates incoming light",
           "[instGraph::stdMotionNode][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::applyPresetRoute( {} );
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool filter : { false, true } )
        {
            CAPTURE( dir, filter );
            const std::string              prefix = filter ? "filter" : "preset";
            const std::vector<std::string> puts   = filter ? std::vector<std::string>{ "out", "refl", "unused" }
                                                           : std::vector<std::string>{ "wfs", "sci", "unused" };
            const std::string              rows =
                filter ? "presetRoute.open=out\npresetRoute.lyotlg=out\npresetRoute.mirror=refl\n"
                                    : "presetRoute.out=sci\npresetRoute.65-35= wfs , sci\npresetRoute.ha-ir=wfs,sci\n";
            const std::vector<std::pair<std::string, std::set<std::string>>> routes =
                filter ? std::vector<std::pair<std::string, std::set<std::string>>>{ { "open", { "out" } },
                                                                                     { "lyotlg", { "out" } },
                                                                                     { "mirror", { "refl" } },
                                                                                     { "closed", {} } }
                       : std::vector<std::pair<std::string, std::set<std::string>>>{ { "out", { "sci" } },
                                                                                     { "65-35", { "wfs", "sci" } },
                                                                                     { "ha-ir", { "wfs", "sci" } },
                                                                                     { "closed", {} } };
            mappedMotionFixture fixture( dir, puts );
            fixture.load( rows + "presetRoute.closed=\n",
                          "presetPrefix=" + prefix + "\n[other]\npresetRoute.ignore=missing\n" );
            auto &node = *fixture.m_node;
            for( const auto &entry : fixture.m_config.m_unusedConfigs )
            {
                if( entry.second.keyword.find( "presetRoute." ) == 0 )
                {
                    REQUIRE( entry.second.used == ( entry.second.section == "fwtelsim" ) );
                }
            }
            fixture.sources( true );
            requireMappedRoute( node, {} );
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            requireMappedRoute( node, {} );
            for( const auto &before : routes )
                for( const auto &after : routes )
                {
                    CAPTURE( before.first, after.first );
                    REQUIRE( node.handleSetProperty( motionPreset( { before.first }, prefix ) ) == 0 );
                    requireMappedRoute( node, before.second );
                    REQUIRE( node.handleSetProperty( motionPreset( { after.first }, prefix ) ) == 0 );
                    requireMappedRoute( node, after.second );
                    REQUIRE( node.curLabel() == after.first );
                    fixture.sources( false );
                    requireMappedRoute( node, after.second, ingr::putState::waiting );
                    REQUIRE( node.handleSetProperty( motionPreset( { before.first }, prefix ) ) == 0 );
                    requireMappedRoute( node, before.second, ingr::putState::waiting );
                    fixture.sources( true );
                    requireMappedRoute( node, before.second );
                }
        }
}

/// Mapping never falls back to a legacy route for unavailable or malformed telemetry.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode blocks mapped routes for unusable telemetry", "[instGraph::stdMotionNode][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::selectedPresetRoute();
    stdMotionNode::putsShouldBeOn();
    stdMotionNode::togglePutsOff();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        mappedMotionFixture fixture( dir, { "wfs", "sci" } );
        fixture.load( "presetRoute.alpha=wfs\npresetRoute.beta=sci\n" );
        auto &node = *fixture.m_node;
        fixture.sources( true );
        for( const std::string state :
             { "POWEROFF", "OPERATING", "HOMING", "NOTHOMED", "POWERON", "NOTCONNECTED", "ERROR", "invalid" } )
        {
            CAPTURE( dir, state );
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPreset( { "alpha" } ) ) == 0 );
            requireMappedRoute( node, { "wfs" } );
            REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
            fixture.sources( false );
            fixture.sources( true );
            requireMappedRoute( node, {} );
            REQUIRE( node.curLabel() == "off" );
        }
        pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
        wrongType.setDevice( "fwtelsim" );
        wrongType.setName( "presetName" );
        wrongType.add( pcf::IndiElement( "alpha", pcf::IndiElement::On ) );
        const std::vector<pcf::IndiProperty> invalid{ motionPreset( {} ),
                                                      motionPreset( { "none" } ),
                                                      motionPreset( { "missing" } ),
                                                      motionPreset( { "alpha", "beta" } ),
                                                      wrongType };
        for( const auto &property : invalid )
        {
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPreset( { "alpha" } ) ) == 0 );
            REQUIRE( node.handleSetProperty( property ) == 0 );
            fixture.sources( false );
            fixture.sources( true );
            requireMappedRoute( node, {} );
            REQUIRE( node.curLabel() == "off" );
        }
    }
}

/// Parking opt-in and all initial message orders preserve mapped positions in supported startup states.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode maps parked presets independently of message order",
           "[instGraph::stdMotionNode][mapping][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::parkedState();
    stdMotionNode::selectedPresetRoute();
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    for( const std::string state : { "POWEROFF", "POWERON", "NOTCONNECTED", "CONNECTED" } )
        for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
            for( const bool parkable : { false, true } )
            {
                std::array<int, 3> order{ 0, 1, 2 };
                do
                {
                    CAPTURE( state, dir, parkable, order );
                    mappedMotionFixture fixture( dir, { "wfs", "sci" } );
                    fixture.load( "presetRoute.65-35=wfs,sci\npresetRoute.closed=\n",
                                  parkable ? "parkable=true\n" : "" );
                    auto &node = *fixture.m_node;
                    fixture.sources( true );
                    REQUIRE( node.keys().count( "fwtelsim.parked" ) == ( parkable ? 1 : 0 ) );
                    const std::array<pcf::IndiProperty, 3> snapshot{
                        motionFSM( state ), motionParked( "1" ), motionPreset( { "65-35" } ) };
                    for( const auto index : order )
                    {
                        REQUIRE( node.handleSetProperty( snapshot[index] ) == 0 );
                    }
                    requireMappedRoute( node,
                                        parkable ? std::set<std::string>{ "wfs", "sci" } : std::set<std::string>{} );
                    if( parkable )
                    {
                        REQUIRE( node.curLabel() == "65-35" );
                        REQUIRE( node.handleSetProperty( motionPreset( { "closed" } ) ) == 0 );
                        requireMappedRoute( node, {} );
                        REQUIRE( node.curLabel() == "closed" );
                        REQUIRE( node.handleSetProperty( motionPreset( { "65-35" } ) ) == 0 );
                        REQUIRE( node.handleSetProperty( motionParked( "garbage" ) ) == 0 );
                        requireMappedRoute( node, {} );
                    }
                    REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
                    requireMappedRoute( node, { "wfs", "sci" } );
                } while( std::next_permutation( order.begin(), order.end() ) );
            }
}

/// Bad route rows, conflicting settings, and incomplete graph topology fail during configuration.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode rejects invalid preset route configuration", "[instGraph::stdMotionNode][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        const std::vector<std::pair<std::string, std::string>> invalid{
            { "presetRoute.=wfs\n", "empty or reserved" },
            { "presetRoute.none=wfs\n", "empty or reserved" },
            { "presetRoute.alpha=missing\n", "put 'missing'" },
            { "presetRoute.alpha=wfs,,sci\n", "empty put" },
            { "presetRoute.alpha=wfs,\n", "empty put" },
            { "presetRoute.alpha=wfs, wfs\n", "duplicate put 'wfs'" },
            { "presetRoute.alpha=" + std::string( dir == ingr::ioDir::output ? "in" : "out" ) + "\n", "is not a" } };
        for( const auto &row : invalid )
        {
            CAPTURE( dir, row.first );
            mappedMotionFixture fixture( dir, { "wfs", "sci" } );
            REQUIRE_THROWS_WITH( fixture.load( row.first ),
                                 Catch::Matchers::Contains( "[fwtelsim]" ) && Catch::Matchers::Contains( row.second ) );
        }
        for( const std::string option : { "presetPutName",
                                          "alwaysOn",
                                          "noAutoOn",
                                          "trackingReqKey",
                                          "trackingReqElement",
                                          "trackerKey",
                                          "trackerElement" } )
        {
            CAPTURE( dir, option );
            mappedMotionFixture fixture( dir, { "wfs", "sci" } );
            REQUIRE_THROWS_WITH( fixture.load( "presetRoute.alpha=wfs\n", option + "=\n" ),
                                 Catch::Matchers::Contains( "cannot be combined with '" + option + "'" ) );
        }
        for( const size_t commonCount : { 0, 2 } )
        {
            CAPTURE( dir, commonCount );
            mappedMotionFixture fixture( dir, { "wfs", "sci" }, true, commonCount );
            REQUIRE_THROWS_WITH( fixture.load( "presetRoute.alpha=wfs\n" ),
                                 Catch::Matchers::Contains( "exactly one opposite-side put" ) );
        }
        mappedMotionFixture missingLink( dir, { "wfs", "sci" }, false );
        REQUIRE_THROWS_WITH( missingLink.load( "presetRoute.alpha=wfs\n" ),
                             Catch::Matchers::Contains( "requires internal link 'input:fwtelsim:" ) &&
                                 Catch::Matchers::Contains( "' -> 'output:fwtelsim:" ) );
        mappedMotionFixture noBranch( dir, {} );
        REQUIRE_THROWS_WITH( noBranch.load( "presetRoute.closed=\n" ),
                             Catch::Matchers::Contains( "requires at least one" ) );
    }
}

/// One branch still requires an explicit mapped name rather than accepting every non-none preset.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode validates mapped names even for a single branch", "[instGraph::stdMotionNode][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::selectedPresetRoute();
    stdMotionNode::togglePutsOn();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        mappedMotionFixture fixture( dir, { "branch" } );
        fixture.load( "presetRoute.alpha=branch\npresetRoute.closed=\n" );
        auto &node = *fixture.m_node;
        fixture.sources( true );
        REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
        REQUIRE( node.handleSetProperty( motionPreset( { "alpha" } ) ) == 0 );
        requireMappedRoute( node, { "branch" } );
        REQUIRE( node.handleSetProperty( motionPreset( { "unknown" } ) ) == 0 );
        requireMappedRoute( node, {} );
        REQUIRE( node.handleSetProperty( motionPreset( { "closed" } ) ) == 0 );
        requireMappedRoute( node, {} );
        REQUIRE( node.curLabel() == "closed" );
    }
}

/// Explicit rows override fallback routing, which still requires one usable selection and an available FSM.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode uses defaultRoute for unmatched valid presets",
           "[instGraph::stdMotionNode][mapping][defaultRoute]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::selectedPresetRoute();
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        mappedMotionFixture fixture( dir, { "out", "refl" } );
        fixture.load( "presetRoute.mirror=refl\npresetRoute.closed=\n",
                      "defaultRoute= out \nparkable=true\n[other]\ndefaultRoute=missing\n" );
        auto &node = *fixture.m_node;
        REQUIRE( fixture.m_config.m_unusedConfigs.at( mx::app::iniFile::makeKey( "fwtelsim", "defaultRoute" ) ).used );
        REQUIRE_FALSE(
            fixture.m_config.m_unusedConfigs.at( mx::app::iniFile::makeKey( "other", "defaultRoute" ) ).used );
        fixture.sources( true );
        REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
        requireMappedRoute( node, {} );
        for( const std::string preset : { "open", "cmc", "new-filter" } )
        {
            REQUIRE( node.handleSetProperty( motionPreset( { preset } ) ) == 0 );
            requireMappedRoute( node, { "out" } );
            REQUIRE( node.curLabel() == preset );
        }
        REQUIRE( node.handleSetProperty( motionPreset( { "mirror" } ) ) == 0 );
        requireMappedRoute( node, { "refl" } );
        REQUIRE( node.handleSetProperty( motionPreset( { "closed" } ) ) == 0 );
        requireMappedRoute( node, {} );
        REQUIRE( node.curLabel() == "closed" );
        REQUIRE( node.handleSetProperty( motionPreset( { "open" } ) ) == 0 );
        fixture.sources( false );
        requireMappedRoute( node, { "out" }, ingr::putState::waiting );
        fixture.sources( true );
        requireMappedRoute( node, { "out" } );
        REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
        requireMappedRoute( node, {} );
        REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
        requireMappedRoute( node, { "out" } );
        REQUIRE( node.handleSetProperty( motionPreset( { "mirror" } ) ) == 0 );
        requireMappedRoute( node, { "refl" } );
        REQUIRE( node.handleSetProperty( motionPreset( { "closed" } ) ) == 0 );
        requireMappedRoute( node, {} );
        REQUIRE( node.curLabel() == "closed" );
        REQUIRE( node.handleSetProperty( motionPreset( { "new-filter" } ) ) == 0 );
        requireMappedRoute( node, { "out" } );
        REQUIRE( node.handleSetProperty( motionParked( "0" ) ) == 0 );
        requireMappedRoute( node, {} );

        pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
        wrongType.setDevice( "fwtelsim" );
        wrongType.setName( "presetName" );
        wrongType.add( pcf::IndiElement( "open", pcf::IndiElement::On ) );
        for( const auto &invalid :
             { motionPreset( {} ), motionPreset( { "none" } ), motionPreset( { "open", "mirror" } ), wrongType } )
        {
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPreset( { "open" } ) ) == 0 );
            REQUIRE( node.handleSetProperty( invalid ) == 0 );
            fixture.sources( false );
            fixture.sources( true );
            requireMappedRoute( node, {} );
        }
        for( const std::string state : { "OPERATING", "HOMING", "NOTHOMED", "NOTCONNECTED", "ERROR" } )
        {
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPreset( { "open" } ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
            fixture.sources( false );
            fixture.sources( true );
            requireMappedRoute( node, {} );
        }
    }
}

/// A configured fallback alone enables mapping, including an intentionally empty fallback.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode supports defaultRoute without explicit rows",
           "[instGraph::stdMotionNode][mapping][defaultRoute]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::presetRoutingConfigured();
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::selectedPresetRoute();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool empty : { false, true } )
        {
            mappedMotionFixture fixture( dir, { "out", "refl" } );
            fixture.load( "", std::string( "defaultRoute=" ) + ( empty ? "" : "out" ) + "\nparkable=true\n" );
            auto &node = *fixture.m_node;
            fixture.sources( true );
            requireMappedRoute( node, {} );
            REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPreset( { "anything" } ) ) == 0 );
            const auto selected = empty ? std::set<std::string>{} : std::set<std::string>{ "out" };
            requireMappedRoute( node, selected );
            REQUIRE( node.curLabel() == "anything" );
            REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
            requireMappedRoute( node, selected );
            REQUIRE( node.curLabel() == "anything" );
            REQUIRE( node.handleSetProperty( motionPreset( { "none" } ) ) == 0 );
            requireMappedRoute( node, {} );
            REQUIRE( node.curLabel() == "off" );
        }
}

/// Default routes use the same put, topology, and conflicting-option validation as explicit routes.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode rejects invalid defaultRoute configuration",
           "[instGraph::stdMotionNode][mapping][defaultRoute]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
    {
        for( const std::string value : { "missing", "wfs,,sci", "wfs,", "wfs,wfs" } )
        {
            mappedMotionFixture fixture( dir, { "wfs", "sci" } );
            REQUIRE_THROWS_WITH( fixture.load( "", "defaultRoute=" + value + "\n" ),
                                 Catch::Matchers::Contains( "row 'defaultRoute'" ) );
        }
        for( const std::string option : { "presetPutName",
                                          "alwaysOn",
                                          "noAutoOn",
                                          "trackingReqKey",
                                          "trackingReqElement",
                                          "trackerKey",
                                          "trackerElement" } )
        {
            mappedMotionFixture fixture( dir, { "wfs", "sci" } );
            REQUIRE_THROWS_WITH( fixture.load( "", "defaultRoute=wfs\n" + option + "=\n" ),
                                 Catch::Matchers::Contains( "cannot be combined with '" + option + "'" ) );
        }
        mappedMotionFixture missingLink( dir, { "wfs", "sci" }, false );
        REQUIRE_THROWS_WITH( missingLink.load( "", "defaultRoute=wfs\n" ),
                             Catch::Matchers::Contains( "requires internal link" ) );
        mappedMotionFixture missingCommon( dir, { "wfs", "sci" }, true, 0 );
        REQUIRE_THROWS_WITH( missingCommon.load( "", "defaultRoute=wfs\n" ),
                             Catch::Matchers::Contains( "exactly one opposite-side put" ) );
    }
}

/// Numerical fallback tracks live current values and display availability without changing legacy routing.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode displays numerical position without changing legacy puts",
           "[instGraph::stdMotionNode][position]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::device( "fwtelsim" );
    stdMotionNode::presetPrefix( "preset" );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::updatePositionLabel();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool multi : { false, true } )
            for( const std::string prefix : { "preset", "filter" } )
            {
                CAPTURE( dir, multi, prefix );
                parkedMotionFixture fixture( dir, multi, prefix );
                parkedMotionFixture reference( dir, multi, prefix );
                auto               &node     = *fixture.m_node;
                auto               &baseline = *reference.m_node;
                const std::string   property = prefix == "filter" ? "filter" : "position";
                REQUIRE( node.keys().count( "fwtelsim." + property ) == 1 );
                REQUIRE( node.keys().count( std::string( "fwtelsim." ) +
                                            ( prefix == "filter" ? "position" : "filter" ) ) == 0 );
                auto deliver = [&]( const pcf::IndiProperty &update )
                {
                    REQUIRE( node.handleSetProperty( update ) == 0 );
                    REQUIRE( baseline.handleSetProperty( update ) == 0 );
                    requireSameMotionPuts( node, baseline );
                };
                REQUIRE( node.handleSetProperty( motionPosition( "-12.34567", property ) ) == 0 );
                requireSameMotionPuts( node, baseline );
                deliver( motionFSM( "READY" ) );
                REQUIRE( node.curLabel() == "-12.3457" );
                deliver( motionPreset( { "none" }, prefix ) );
                REQUIRE( node.curLabel() == "-12.3457" );
                deliver( motionPreset( { "routeA" }, prefix ) );
                REQUIRE( node.curLabel() == "routeA" );
                REQUIRE( node.handleSetProperty( motionPosition( "9.5", property ) ) == 0 );
                REQUIRE( node.curLabel() == "routeA" );
                requireSameMotionPuts( node, baseline );
                deliver( motionPreset( {}, prefix ) );
                REQUIRE( node.curLabel() == "9.5000" );
                auto target = motionPosition( "100", property );
                target.remove( "current" );
                target.add( pcf::IndiElement( "target", "100" ) );
                REQUIRE( node.handleSetProperty( target ) == 0 );
                REQUIRE( node.curLabel() == "9.5000" );
                auto foreign = motionPosition( "100", property );
                foreign.setDevice( "otherstage" );
                REQUIRE( node.handleSetProperty( foreign ) == 0 );
                REQUIRE( node.curLabel() == "9.5000" );
                for( const std::string state : { "READY", "OPERATING", "HOMING", "CONFIGURING", "NOTHOMED" } )
                {
                    deliver( motionFSM( state ) );
                    REQUIRE( node.curLabel() == "9.5000" );
                }
                for( const std::string state : { "POWEROFF", "NOTCONNECTED", "ERROR", "POWERON" } )
                {
                    deliver( motionFSM( state ) );
                    REQUIRE( node.curLabel() == "off" );
                }
                deliver( motionFSM( "POWEROFF" ) );
                deliver( motionParked( "1" ) );
                REQUIRE( node.curLabel() == "9.5000" );
                requireMotionOff( node );
                deliver( motionPreset( { "routeA" }, prefix ) );
                REQUIRE( node.curLabel() == "routeA" );
                REQUIRE( node.handleSetProperty( motionPosition( "5.25", property ) ) == 0 );
                requireSameMotionPuts( node, baseline );
                deliver( motionPreset( { "none" }, prefix ) );
                REQUIRE( node.curLabel() == "5.2500" );
                deliver( motionParked( "0" ) );
                REQUIRE( node.curLabel() == "off" );
                deliver( motionFSM( "READY" ) );
                for( const std::string value : { "", "garbage", "12junk", "nan", "inf", "1e309" } )
                {
                    REQUIRE( node.handleSetProperty( motionPosition( value, property ) ) == 0 );
                    REQUIRE( node.curLabel() == "off" );
                    requireSameMotionPuts( node, baseline );
                    REQUIRE( node.handleSetProperty( motionPosition( " 2.5e1 ", property ) ) == 0 );
                    REQUIRE( node.curLabel() == "25.0000" );
                }
                REQUIRE( node.handleSetProperty( motionPosition( "40", property, pcf::IndiProperty::Text ) ) == 0 );
                REQUIRE( node.curLabel() == "off" );
                REQUIRE( node.handleSetProperty( motionPosition( "0", property ) ) == 0 );
                for( const auto &selection : { std::vector<std::string>{},
                                               std::vector<std::string>{ "none" },
                                               std::vector<std::string>{ "routeA", "routeB" } } )
                {
                    deliver( motionPreset( selection, prefix ) );
                    REQUIRE( node.curLabel() == "0.0000" );
                }
                pcf::IndiProperty wrongType( pcf::IndiProperty::Text );
                wrongType.setDevice( "fwtelsim" );
                wrongType.setName( prefix + "Name" );
                wrongType.add( pcf::IndiElement( "routeA", pcf::IndiElement::On ) );
                deliver( wrongType );
                REQUIRE( node.curLabel() == "0.0000" );
            }
}

/// Mapped and tracking nodes retain their routing masks and priority labels when numeric telemetry arrives.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode numerical display preserves mapped and tracking behavior",
           "[instGraph::stdMotionNode][position]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::updatePositionLabel();
    stdMotionNode::putsShouldBeOn();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool fallback : { false, true } )
        {
            mappedMotionFixture fixture( dir, { "out", "refl" } );
            mappedMotionFixture reference( dir, { "out", "refl" } );
            for( auto *f : { &fixture, &reference } )
            {
                f->load( "presetRoute.alpha=refl\npresetRoute.closed=\n",
                         "parkable=true\n" + std::string( fallback ? "defaultRoute=out\n" : "" ) );
                f->sources( true );
            }
            auto &node     = *fixture.m_node;
            auto &baseline = *reference.m_node;
            for( const auto &selection : { std::vector<std::string>{ "alpha" },
                                           std::vector<std::string>{ "closed" },
                                           std::vector<std::string>{ "unmapped" },
                                           std::vector<std::string>{ "none" },
                                           std::vector<std::string>{ "alpha", "unmapped" },
                                           std::vector<std::string>{} } )
            {
                for( auto *n : { &node, &baseline } )
                {
                    REQUIRE( n->handleSetProperty( motionFSM( "READY" ) ) == 0 );
                    REQUIRE( n->handleSetProperty( motionPreset( selection ) ) == 0 );
                }
                const auto label = node.curLabel();
                REQUIRE( node.handleSetProperty( motionPosition( "3.125" ) ) == 0 );
                requireSameMotionPuts( node, baseline );
                REQUIRE( node.curLabel() ==
                         ( selection.size() != 1 || selection.front() == "none" ? "3.1250" : label ) );
                fixture.sources( false );
                reference.sources( false );
                requireSameMotionPuts( node, baseline );
                fixture.sources( true );
                reference.sources( true );
                requireSameMotionPuts( node, baseline );
            }
            requireMappedRoute( node, {} );
            REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
            REQUIRE( baseline.handleSetProperty( motionParked( "1" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
            REQUIRE( baseline.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
            REQUIRE( node.curLabel() == "3.1250" );
            requireSameMotionPuts( node, baseline );
        }
    parkedMotionFixture fixture( ingr::ioDir::output, false, "preset", true );
    auto               &node = *fixture.m_node;
    REQUIRE( node.handleSetProperty( motionFSM( "READY" ) ) == 0 );
    auto requested = motionPreset( { "none" } );
    requested.setDevice( "labrules" );
    requested.setName( "info" );
    requested.add( pcf::IndiElement( "trackReq", pcf::IndiElement::On ) );
    auto tracker = motionPreset( { "none" } );
    tracker.setDevice( "adctrack" );
    tracker.setName( "tracking" );
    tracker.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    REQUIRE( node.handleSetProperty( requested ) == 0 );
    REQUIRE( node.curLabel() == "not tracking" );
    REQUIRE( node.handleSetProperty( motionPosition( "12.5" ) ) == 0 );
    REQUIRE( node.curLabel() == "not tracking" );
    requireMotionOff( node );
    REQUIRE( node.handleSetProperty( tracker ) == 0 );
    REQUIRE( node.curLabel() == "tracking" );
    const auto state = node.node()->output( "out" )->state();
    REQUIRE( node.handleSetProperty( motionPosition( "13.5" ) ) == 0 );
    REQUIRE( node.curLabel() == "tracking" );
    REQUIRE( node.node()->output( "out" )->state() == state );
    REQUIRE( node.handleSetProperty( motionPreset( { "none" } ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
    REQUIRE( node.handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
    REQUIRE( node.curLabel() == "13.5000" );
    requireMotionOff( node );
}

/// Retained routes and numeric labels stay steady through startup while parking remains affirmative.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "stdMotionNode retains parked routing through power-on sequences",
           "[instGraph::stdMotionNode][parked][startup]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    stdMotionNode::parkedFSMState();
    stdMotionNode::parkedState();
    stdMotionNode::putsShouldBeOn();
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::togglePutsOn();
    stdMotionNode::togglePutsOff();
    stdMotionNode::updatePositionLabel();
    #endif
    // clang-format on

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool multi : { false, true } )
            for( const bool tracking : { false, true } )
            {
                parkedMotionFixture fixture( dir, multi, "preset", tracking );
                parkedMotionFixture reference( dir, multi, "preset", tracking );
                auto               &node     = *fixture.m_node;
                auto               &baseline = *reference.m_node;
                for( auto *n : { &node, &baseline } )
                {
                    REQUIRE( n->handleSetProperty( motionFSM( "READY" ) ) == 0 );
                    if( tracking )
                    {
                        pcf::IndiProperty requested( pcf::IndiProperty::Switch );
                        requested.setDevice( "labrules" );
                        requested.setName( "info" );
                        requested.add( pcf::IndiElement( "trackReq", pcf::IndiElement::On ) );
                        pcf::IndiProperty tracker( pcf::IndiProperty::Switch );
                        tracker.setDevice( "adctrack" );
                        tracker.setName( "tracking" );
                        tracker.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
                        REQUIRE( n->handleSetProperty( requested ) == 0 );
                        REQUIRE( n->handleSetProperty( tracker ) == 0 );
                    }
                    REQUIRE( n->handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
                    REQUIRE( n->handleSetProperty( motionParked( "1" ) ) == 0 );
                    REQUIRE( n->handleSetProperty( motionFSM( "POWEROFF" ) ) == 0 );
                }
                for( const std::string state : { "POWEROFF", "POWERON", "NOTCONNECTED", "CONNECTED" } )
                {
                    CAPTURE( dir, multi, tracking, state );
                    REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
                    REQUIRE( node.curLabel() == "routeA" );
                    requireSameMotionPuts( node, baseline );
                    std::string xml, error;
                    REQUIRE( fixture.m_graph.serializeXML( xml, error ) == 0 );
                    REQUIRE( xml.find( "value=\"" + state + "\"" ) != std::string::npos );
                    for( auto *n : { &node, &baseline } )
                    {
                        REQUIRE( n->handleSetProperty( motionPreset( { "routeB" } ) ) == 0 );
                    }
                    REQUIRE( node.curLabel() == "routeB" );
                    requireSameMotionPuts( node, baseline );
                    REQUIRE( node.handleSetProperty( motionParked( "0" ) ) == 0 );
                    requireMotionOff( node );
                    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
                    requireSameMotionPuts( node, baseline );
                    for( auto *n : { &node, &baseline } )
                    {
                        REQUIRE( n->handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
                    }
                }
                REQUIRE( node.handleSetProperty( motionPosition( "2.5" ) ) == 0 );
                REQUIRE( node.handleSetProperty( motionPreset( { "none" } ) ) == 0 );
                for( const std::string state : { "POWEROFF", "POWERON", "NOTCONNECTED", "CONNECTED" } )
                {
                    REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
                    REQUIRE( node.curLabel() == "2.5000" );
                    requireMotionOff( node );
                }
                REQUIRE( node.handleSetProperty( motionPreset( { "routeA" } ) ) == 0 );
                for( const std::string state : { "HOMING", "NOTHOMED", "CONFIGURING", "LOGGEDIN", "ERROR" } )
                {
                    REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
                    requireMotionOff( node );
                }
            }

    for( const auto dir : { ingr::ioDir::input, ingr::ioDir::output } )
        for( const bool fallback : { false, true } )
        {
            mappedMotionFixture fixture( dir, { "wfs", "sci" } );
            fixture.load( "presetRoute.split=wfs,sci\npresetRoute.closed=\n",
                          "parkable=true\n" + std::string( fallback ? "defaultRoute=sci\n" : "" ) );
            auto &node = *fixture.m_node;
            fixture.sources( true );
            REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
            REQUIRE( node.handleSetProperty( motionPosition( "3.125" ) ) == 0 );
            for( const std::string preset : { "split", "unlisted", "closed", "none" } )
            {
                REQUIRE( node.handleSetProperty( motionPreset( { preset } ) ) == 0 );
                const std::set<std::string> selected = preset == "split" ? std::set<std::string>{ "wfs", "sci" }
                                                       : preset == "unlisted" && fallback
                                                           ? std::set<std::string>{ "sci" }
                                                           : std::set<std::string>{};
                const std::string           label    = preset == "none"                    ? "3.1250"
                                                       : preset == "unlisted" && !fallback ? "off"
                                                                                           : preset;
                for( const std::string state : { "POWEROFF", "POWERON", "NOTCONNECTED", "CONNECTED" } )
                {
                    CAPTURE( dir, fallback, preset, state );
                    REQUIRE( node.handleSetProperty( motionFSM( state ) ) == 0 );
                    requireMappedRoute( node, selected );
                    REQUIRE( node.curLabel() == label );
                    fixture.sources( false );
                    requireMappedRoute( node, selected, ingr::putState::waiting );
                    fixture.sources( true );
                    requireMappedRoute( node, selected );
                    REQUIRE( node.handleSetProperty( motionParked( "bad" ) ) == 0 );
                    requireMappedRoute( node, {} );
                    REQUIRE( node.handleSetProperty( motionParked( "1" ) ) == 0 );
                    requireMappedRoute( node, selected );
                }
            }
        }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
