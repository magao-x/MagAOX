/** \file fsmNode_test.cpp
 * \brief Catch2 tests for the xInstGraph `fsmNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../fsmNode.hpp"

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

void writeXML()
{
    std::ofstream fout( "/tmp/xigNode_test.xml" );
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:ttmpupil\">\n";
    fout << "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

SCENARIO( "Creating and configuring an fsmNode", "[instGraph::fsmNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    fsmNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    fsmNode::fsmKey();
    #endif
    // clang-format on

    GIVEN( "a valid XML file, a valid config file" )
    {
        WHEN( "node is in file, default config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/fsmNode_test.conf", { "ttmpupil" }, { "type" }, { "fsm" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/fsmNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            fsmNode *tsn  = nullptr;
            bool     pass = false;
            try
            {
                tsn  = new fsmNode( "ttmpupil", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "ttmpupil" );
            REQUIRE( tsn->node()->name() == "ttmpupil" );

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

            // check config-ed values
            REQUIRE( tsn->device() == "ttmpupil" );
            REQUIRE( tsn->fsmKey() == "ttmpupil.fsm" );
            REQUIRE( tsn->fsmAction() == fsmNodeActionT::passive );
            REQUIRE( tsn->targetStates().size() == 0 );
        }
        WHEN( "node is in file, setting action to threshOff for two states" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/fsmNode_test.conf",
                                      { "ttmpupil", "ttmpupil", "ttmpupil" },
                                      { "type", "fsmAction", "targetStates" },
                                      { "fsm", "threshOff", "READY,OPERATING" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/fsmNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            fsmNode *tsn  = nullptr;
            bool     pass = false;
            try
            {
                tsn  = new fsmNode( "ttmpupil", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "ttmpupil" );
            REQUIRE( tsn->node()->name() == "ttmpupil" );

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

            // check config-ed values
            REQUIRE( tsn->device() == "ttmpupil" );
            REQUIRE( tsn->fsmKey() == "ttmpupil.fsm" );
            REQUIRE( tsn->fsmAction() == fsmNodeActionT::threshOff );
            REQUIRE( tsn->targetStates().size() == 2 );
            REQUIRE( tsn->targetStates()[0] == MagAOX::app::stateCodes::READY );
            REQUIRE( tsn->targetStates()[1] == MagAOX::app::stateCodes::OPERATING );
        }
        WHEN( "node is in file, setting action to active for one state" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/fsmNode_test.conf",
                                      { "ttmpupil", "ttmpupil", "ttmpupil" },
                                      { "type", "fsmAction", "targetStates" },
                                      { "fsm", "active", "OPERATING" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/fsmNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            fsmNode *tsn  = nullptr;
            bool     pass = false;
            try
            {
                tsn  = new fsmNode( "ttmpupil", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "ttmpupil" );
            REQUIRE( tsn->node()->name() == "ttmpupil" );

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

            // check config-ed values
            REQUIRE( tsn->device() == "ttmpupil" );
            REQUIRE( tsn->fsmKey() == "ttmpupil.fsm" );
            REQUIRE( tsn->fsmAction() == fsmNodeActionT::active );
            REQUIRE( tsn->targetStates().size() == 1 );
            REQUIRE( tsn->targetStates()[0] == MagAOX::app::stateCodes::OPERATING );
        }
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
