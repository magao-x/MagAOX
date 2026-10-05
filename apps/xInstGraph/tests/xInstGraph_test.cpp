/** \file xInstGraph_test.cpp
 * \brief Catch2 tests for the xInstGraph app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../xInstGraph.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup xInstGraph_unit_test xInstGraph Unit Tests
 * \brief Unit tests for the xInstGraph application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xInstGraph` unit tests.
/** \ingroup xInstGraph_unit_test
 */
namespace xInstGraphTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class xInstGraph : public MagAOX::app::xInstGraph
{
  public:
    void configDir( const std::string &cp )
    {
        m_configDir = cp;
    }

    mx::app::appConfigurator &config()
    {
        return MagAOX::app::xInstGraph::config;
    }
};
/// \endcond

/// Write a minimal draw.io graph containing one of each supported node type.
void writeXML()
{
    std::ofstream fout( "/tmp/xInstGraph_test/config/instgraph_test.drawio" );
    fout << "<mxfile host=\"test\">\n";
    fout << "    <diagram id=\"test\" name=\"test\">\n";
    fout << "        <mxGraphModel>\n";
    fout << "            <root>\n";
    fout << "               <mxCell id=\"0\"/>\n";
    fout << "               <mxCell id=\"1\" parent=\"0\"/>\n";
    fout << "               <mxCell id=\"node:fsmNode\">\n";
    fout << "</mxCell>\n";
    fout << "               <mxCell id=\"node:indiPropNode\">\n";
    fout << "</mxCell>\n";
    fout << "               <mxCell id=\"node:pwrOnOffNode\">\n";
    fout << "</mxCell>\n";
    fout << "               <mxCell id=\"node:stdMotionNode\">\n";
    fout << "</mxCell>\n";
    fout << "               <mxCell id=\"node:staticNode\">\n";
    fout << "</mxCell>\n";
    fout << "            </root>\n";
    fout << "       </mxGraphModel>\n";
    fout << "   </diagram>\n";
    fout << "</mxfile>\n";
    fout.close();
}

/// Verify xInstGraph configures and runs when given a minimal graph with all supported node types.
/**
 * \ingroup xInstGraph_unit_test
 */
TEST_CASE( "Configuring xInstGraph with no errors", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    xInstGraph::setupConfig();
    xInstGraph::loadConfig();
    xInstGraph::appStartup();
    xInstGraph::appLogic();
    xInstGraph::appShutdown();
    xInstGraph::st_igHandleSetProperty( nullptr, pcf::IndiProperty() );
    #endif
    // clang-format on

    mx::ioutils::createDirectories( "/tmp/xInstGraph_test/config" );

    writeXML();

    SECTION( "A valid configuration with each node type" )
    {
        std::vector<std::string> sections;
        std::vector<std::string> keys;
        std::vector<std::string> values;

        sections.insert( sections.end(), { "graph", "graph" } );
        keys.insert( keys.end(), { "file", "outputPath" } );
        values.insert( values.end(), { "instgraph_test.drawio", "/tmp/xInstGraph_test/instgraph_test_out.drawio" } );

        sections.insert( sections.end(), { "indiPropNode", "indiPropNode", "indiPropNode", "indiPropNode" } );
        keys.insert( keys.end(), { "type", "propKey", "propEl", "propVal" } );
        values.insert( values.end(), { "indiProp", "test.test", "test", "test" } );

        sections.insert( sections.end(), { "pwrOnOffNode", "pwrOnOffNode" } );
        keys.insert( keys.end(), { "type", "pwrKey" } );
        values.insert( values.end(), { "pwrOnOff", "testpwr.test" } );

        sections.insert( sections.end(), { "fsmNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "fsm" } );

        sections.insert( sections.end(), { "stdMotionNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "stdMotion" } );

        sections.insert( sections.end(), { "staticNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "static" } );

        mx::app::writeConfigFile( "/tmp/xInstGraph_test/config/instgraph_test.conf", sections, keys, values );

        xInstGraph xig;
        xig.configDir( "/tmp/xInstGraph_test/config" );

        REQUIRE( xig.shutdown() == 0 );

        xig.setupConfig();

        REQUIRE( xig.shutdown() == 0 );

        xig.config().readConfig( "/tmp/xInstGraph_test/config/instgraph_test.conf" );

        xig.loadConfig();
        REQUIRE( xig.shutdown() == 0 );

        REQUIRE( xig.appStartup() == 0 );

        pcf::IndiProperty ip;
        ip.setDevice( "testpwr" );
        ip.setName( "test" );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"] = "On";

        MagAOX::app::xInstGraph::st_igHandleSetProperty( &xig, ip );

        bool ex = std::filesystem::exists( "/tmp/xInstGraph_test/instgraph_test_out.drawio" );
        REQUIRE( ex == true );

        REQUIRE( xig.appLogic() == 0 );

        REQUIRE( xig.appShutdown() == 0 );

        ex = std::filesystem::exists( "/tmp/xInstGraph_test/instgraph_test_out.drawio" );
        REQUIRE( ex == false );
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
