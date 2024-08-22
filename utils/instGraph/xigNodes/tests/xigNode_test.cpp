//#define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <fstream>

#include "../xigNode.hpp"

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

class test_xigNode : public xigNode
{
public:
    test_xigNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode(name,parentGraph)
    {}

    void handleSetProperty( const pcf::IndiProperty &ipRecv )
    {
        static_cast<void>(ipRecv);
    }
};

SCENARIO( "Creating an xigNode", "[instGraph::xigNode]" )
{
    GIVEN("a valid XML file")
    {
        WHEN("node is in file")
        {
            ingr::instGraphXML parentGraph;
            writeXML();

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            test_xigNode * txn = nullptr;
            bool pass = false;
            try
            {
                txn = new test_xigNode("telescope", &parentGraph);
                pass = true;
            }
            catch(...)
            {
            }

            REQUIRE(pass == true);
            REQUIRE(txn != nullptr);


            REQUIRE( txn->name() == "telescope");
            REQUIRE( txn->node()->name() == "telescope");
            REQUIRE( pass == true );

            txn->key("tkey");
            REQUIRE( txn->keys().count("tkey") == 1);

        }

        WHEN("node is not in file")
        {
            ingr::instGraphXML parentGraph;
            writeXML();

            std::string emsg;

            int rv = parentGraph.loadXMLFile(emsg, "/tmp/xigNode_test.xml");

            REQUIRE(rv == 0);
            REQUIRE(emsg == "");

            test_xigNode * txn = nullptr;

            bool pass = false;
            try
            {
                txn = new test_xigNode("epocselet", &parentGraph);
                pass = true;
            }
            catch(...)
            {
            }

            REQUIRE(pass == false);
            REQUIRE(txn == nullptr);
        }
    }
}


