/** \file stdMotionNode_test.cpp
 * \brief Catch2 tests for the xInstGraph `stdMotionNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <fstream>

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

} // namespace xInstGraphTest

} // namespace libXWCTest

/// Verify a motion stage handles preset and tracking property updates.
/** \ingroup xInstGraph_unit_test
 */
SCENARIO( "Sending Properties to a stdMotionNode", "[instGraph::stdMotionNode]" )
{
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
