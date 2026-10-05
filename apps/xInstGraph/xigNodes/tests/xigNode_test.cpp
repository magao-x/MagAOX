/** \file xigNode_test.cpp
 * \brief Catch2 tests for the xInstGraph base `xigNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <fstream>

#include "../xigNode.hpp"

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
    fout << "               <mxCell id=\"node:telescope\">\n";
    fout << "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

class test_xigNode : public xigNode
{
  public:
    test_xigNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
    {
    }

    int handleSetProperty( const pcf::IndiProperty &ipRecv )
    {
        static_cast<void>( ipRecv );
        return 0;
    }
};

SCENARIO( "Creating an xigNode", "[instGraph::xigNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    xigNode::key( "" );
    xigNode::keys();
    #endif
    // clang-format on

    GIVEN( "a valid XML file" )
    {
        WHEN( "node is in file" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            test_xigNode *txn  = nullptr;
            bool          pass = false;
            try
            {
                txn  = new test_xigNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( ... )
            {
            }

            REQUIRE( pass == true );
            REQUIRE( txn != nullptr );

            REQUIRE( txn->name() == "telescope" );
            REQUIRE( txn->node()->name() == "telescope" );
            REQUIRE( pass == true );

            txn->key( "tkey" );
            REQUIRE( txn->keys().count( "tkey" ) == 1 );
        }

        WHEN( "node is not in file" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            test_xigNode *txn = nullptr;

            bool pass = false;
            try
            {
                txn  = new test_xigNode( "epocselet", &parentGraph );
                pass = true;
            }
            catch( ... )
            {
            }

            REQUIRE( pass == false );
            REQUIRE( txn == nullptr );
        }
    }

    GIVEN( "an invalid XML file" )
    {
        WHEN( "parent graph is null on construction" )
        {
            ingr::instGraphXML *parentGraph = nullptr;

            test_xigNode *txn  = nullptr;
            bool          pass = false;
            try
            {
                txn  = new test_xigNode( "telescope", parentGraph );
                pass = true;
            }
            catch( ... )
            {
            }

            REQUIRE( pass == false );
            REQUIRE( txn == nullptr );
        }
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
