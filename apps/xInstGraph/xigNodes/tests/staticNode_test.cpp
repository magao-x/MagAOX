// #define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../staticNode.hpp"

void writeXML()
{
    std::ofstream fout( "/tmp/xigNode_test.xml" );
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:ttmpupil\" />\n";
    fout << "                   <mxCell id=\"input:ttmpupil:in1\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"input:ttmpupil:in2\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"input:ttmpupil:in3\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"input:ttmpupil:in4\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"output:ttmpupil:out1\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"output:ttmpupil:out2\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"output:ttmpupil:out3\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "                   <mxCell id=\"output:ttmpupil:out4\" parent=\"node:ttmpupil\" style=\"fontSize=17;\" />\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

TEST_CASE( "Creating and configuring an staticNode", "[instGraph::staticNode]" )
{
    SECTION( "node is in file, setting pwr key" )
    {
        ingr::instGraphXML parentGraph;
        writeXML();
        mx::app::writeConfigFile( "/tmp/staticNode_test.conf",
                                  { "ttmpupil", "ttmpupil","ttmpupil","ttmpupil","ttmpupil" },
                                  { "type", "inputsOn", "inputsOff", "outputsOn", "outputsOff" },
                                  { "static", "in1,in2", "in3,in4", "out1,out2", "out3,out4" } );
        mx::app::appConfigurator config;
        config.readConfig( "/tmp/staticNode_test.conf" );

        std::string emsg;

        int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

        REQUIRE( rv == 0 );
        REQUIRE( emsg == "" );

        staticNode *tsn  = nullptr;
        bool        pass = false;
        try
        {
            tsn  = new staticNode( "ttmpupil", &parentGraph );
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
        REQUIRE( tsn->inputsOn().size() == 2 );
        REQUIRE( tsn->inputsOn().count( "in1" ) == 1 );
        REQUIRE( tsn->inputsOn().count( "in2" ) == 1 );
        REQUIRE( tsn->inputsOff().size() == 2 );
        REQUIRE( tsn->inputsOff().count( "in3" ) == 1 );
        REQUIRE( tsn->inputsOff().count( "in4" ) == 1 );
        REQUIRE( tsn->outputsOn().size() == 2 );
        REQUIRE( tsn->outputsOn().count( "out1" ) == 1 );
        REQUIRE( tsn->outputsOn().count( "out2" ) == 1 );
        REQUIRE( tsn->outputsOff().size() == 2 );
        REQUIRE( tsn->outputsOff().count( "out3" ) == 1 );
        REQUIRE( tsn->outputsOff().count( "out4" ) == 1 );
    }
}
