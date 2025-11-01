/** \file xInstGraph_test.cpp
 * \brief Catch2 tests for the xInstGraph app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../xInstGraph.hpp"

using namespace MagAOX::app;

namespace xInstGraph_test
{

void writeXML()
{
    std::ofstream fout("/tmp/xInstGraph_test/config/instgraph_test.drawio");
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:fsmNode\">\n";
    fout <<                 "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

TEST_CASE( "Configuring xInstGraph with no errors", "[xInstGraph]" )
{
    mx::ioutils::createDirectories("/tmp/xInstGraph_test/config");

    SECTION( "A valid configuration with each node type" )
    {
        std::vector<std::string> sections;
        std::vector<std::string> keys;
        std::vector<std::string> values;

        sections.insert(sections.end(), {"graph", "graph"});
        keys.insert(keys.end(), {"file", "outputPath"});
        values.insert(values.end(), {"instgraph_test.drawio", "/tmp/xInstGraph_test/instgraph_test_out.drawio"});

        sections.insert(sections.end(), "fsmNode"
        // this adds unknown=value
        mx::app::writeConfigFile( "/tmp/xInstGraph_test/config/instgraph_test.conf", sections, keys, values);
    }
}

} // namespace xInstGraph_test
