/** \file pwrOnOffNode_test.cpp
 * \brief Catch2 tests for the xInstGraph `pwrOnOffNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../pwrOnOffNode.hpp"

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

/// Write the minimal power-node graph used by configuration tests.
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

/// Verify the required power property key is loaded.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "Creating and configuring an pwrOnOffNode", "[instGraph::pwrOnOffNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    pwrOnOffNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    pwrOnOffNode::pwrKey();
    #endif
    // clang-format on

    SECTION( "node is in file, setting pwr key" )
    {
        ingr::instGraphXML parentGraph;
        writeXML();
        mx::app::writeConfigFile( "/tmp/pwrOnOffNode_test.conf",
                                  { "ttmpupil", "ttmpupil" },
                                  { "type", "pwrKey" },
                                  { "pwrOnOff", "test.pwr" } );
        mx::app::appConfigurator config;
        config.readConfig( "/tmp/pwrOnOffNode_test.conf" );

        std::string emsg;

        int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

        REQUIRE( rv == 0 );
        REQUIRE( emsg == "" );

        pwrOnOffNode *tsn  = nullptr;
        bool          pass = false;
        try
        {
            tsn  = new pwrOnOffNode( "ttmpupil", &parentGraph );
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
        REQUIRE( tsn->pwrKey() == "test.pwr" );
    }

    SECTION( "node is in file, error: not setting pwr key" )
    {
        ingr::instGraphXML parentGraph;
        writeXML();
        mx::app::writeConfigFile( "/tmp/pwrOnOffNode_test.conf", { "ttmpupil" }, { "type" }, { "pwrOnOff" } );
        mx::app::appConfigurator config;
        config.readConfig( "/tmp/pwrOnOffNode_test.conf" );

        std::string emsg;

        int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

        REQUIRE( rv == 0 );
        REQUIRE( emsg == "" );

        pwrOnOffNode *tsn  = nullptr;
        bool          pass = false;
        try
        {
            tsn  = new pwrOnOffNode( "ttmpupil", &parentGraph );
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

        REQUIRE( pass == false );
    }
}

/// Show intermediate and unknown power states without reporting OFF.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "pwrOnOffNode distinguishes power states", "[instGraph::pwrOnOffNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    pwrOnOffNode::handleSetProperty( *(pcf::IndiProperty *)nullptr );
    pwrOnOffNode::toggleUnknown( *(std::string *)nullptr );
    #endif
    // clang-format on

    const std::string xmlPath = "/tmp/pwrOnOffNode_F09_test.drawio";
    {
        std::ofstream out( xmlPath );
        out << "<mxfile><diagram><mxGraphModel><root>\n"
               "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>\n"
               "<mxCell id=\"node:ttmpupil\"/>\n"
               "<mxCell id=\"output:ttmpupil:out\" style=\"strokeColor=#FF0000;\"/>\n"
               "<mxCell id=\"fsmstate:ttmpupil\" value=\"---\"/>\n"
               "<mxCell id=\"state:ttmpupil\" value=\"state\"/>\n"
               "</root></mxGraphModel></diagram></mxfile>\n";
    }
    ingr::instGraphXML graph;
    graph.autoSave( false );
    std::string error;
    REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );

    const std::string configPath = "/tmp/pwrOnOffNode_F09_test.conf";
    mx::app::writeConfigFile(
        configPath, { "ttmpupil", "ttmpupil" }, { "type", "pwrKey" }, { "pwrOnOff", "test.pwr" } );
    mx::app::appConfigurator config;
    REQUIRE( config.readConfig( configPath ) == 0 );
    pwrOnOffNode node( "ttmpupil", &graph );
    REQUIRE_NOTHROW( node.loadConfig( config ) );

    pcf::IndiProperty property( pcf::IndiProperty::Text );
    property.setDevice( "test" );
    property.setName( "pwr" );
    property.add( pcf::IndiElement( "state" ) );

    for( const auto &entry : { std::pair{ "On", "ON" },
                               std::pair{ "Int", "INT" },
                               std::pair{ "On", "ON" },
                               std::pair{ "Unk", "UNK" },
                               std::pair{ "On", "ON" },
                               std::pair{ "on", "UNK" },
                               std::pair{ "Off", "OFF" } } )
    {
        CAPTURE( entry.first );
        property["state"] = entry.first;
        REQUIRE( node.handleSetProperty( property ) == 0 );
        REQUIRE( graph.node( "ttmpupil" )->output( "out" )->state() ==
                 ( std::string( entry.second ) == "ON" ? ingr::putState::on : ingr::putState::off ) );
        std::string xml;
        REQUIRE( graph.serializeXML( xml, error ) == 0 );
        const auto start = xml.find( "id=\"fsmstate:ttmpupil\"" );
        REQUIRE( start != std::string::npos );
        REQUIRE( xml.substr( start, xml.find( '>', start ) - start )
                     .find( std::string( "value=\"" ) + entry.second + "\"" ) != std::string::npos );
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
