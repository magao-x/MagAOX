// #define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../pwrOnOffNode.hpp"

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

TEST_CASE( "Creating and configuring an pwrOnOffNode", "[instGraph::pwrOnOffNode]" )
{
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
        mx::app::writeConfigFile( "/tmp/pwrOnOffNode_test.conf",
                                  { "ttmpupil" },
                                  { "type" },
                                  { "pwrOnOff" } );
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
