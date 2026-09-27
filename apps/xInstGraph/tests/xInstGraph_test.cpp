/** \file xInstGraph_test.cpp
 * \brief Catch2 tests for xInstGraph output publication and ownership.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xInstGraph.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup xInstGraph_unit_test xInstGraph Unit Tests
 * \brief Unit tests for the xInstGraph application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for xInstGraph unit tests.
/** \ingroup xInstGraph_unit_test
 */
namespace xInstGraphTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class xInstGraph : public MagAOX::app::xInstGraph
{
  public:
    /// Set the application config directory for a test.
    void configDir( const std::string &cp /**< [in] config directory */ )
    {
        m_configDir = cp;
    }

    /// Access the app's configurator.
    mx::app::appConfigurator &config()
    {
        return MagAOX::app::xInstGraph::config;
    }
};

struct temporaryDirectory
{
    /// Isolated test root, removed on destruction.
    std::filesystem::path root;

    /// Create a unique temporary test directory.
    temporaryDirectory()
    {
        char  name[] = "/tmp/xInstGraph_test_XXXXXX";
        char *dir    = ::mkdtemp( name );
        if( dir == nullptr )
        {
            throw std::runtime_error( "could not create xInstGraph test directory" );
        }

        root = dir;
        std::filesystem::create_directories( root / "config" );
    }

    /// Remove this test's files.
    ~temporaryDirectory()
    {
        std::error_code ec;
        std::filesystem::remove_all( root, ec );
    }

    temporaryDirectory( const temporaryDirectory & )            = delete;
    temporaryDirectory &operator=( const temporaryDirectory & ) = delete;
};
/// \endcond

/// Read all bytes from a test file.
std::string readFile( const std::filesystem::path &path /**< [in] file to read */ )
{
    std::ifstream in( path );
    return { std::istreambuf_iterator<char>( in ), std::istreambuf_iterator<char>() };
}

/// Write either a minimal five-node graph or a static node with a put and internal link.
void writeXML( const std::filesystem::path &path /**< [in] source graph path */,
               bool                         connected /**< [in] include a static node with linked puts */ )
{
    std::ofstream out( path );
    out << "<mxfile host=\"test\"><diagram id=\"test\" name=\"test\"><mxGraphModel><root>\n";
    out << "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>\n";

    if( connected )
    {
        out << "<mxCell id=\"node:staticNode\" style=\"rounded=0;strokeColor=#FF0000;\"/>\n";
        out << "<mxCell id=\"input:staticNode:in\" style=\"rounded=0;strokeColor=#FF0000;\"/>\n";
        out << "<mxCell id=\"output:staticNode:out\" style=\"rounded=0;strokeColor=#FF0000;\"/>\n";
        out << "<mxCell id=\"link:staticNode:in2out\" style=\"strokeColor=#00FF00;\" "
               "source=\"input:staticNode:in\" target=\"output:staticNode:out\"/>\n";
    }
    else
    {
        out << "<mxCell id=\"node:fsmNode\"/>\n";
        out << "<mxCell id=\"node:indiPropNode\"/>\n";
        out << "<mxCell id=\"node:pwrOnOffNode\"/>\n";
        out << "<mxCell id=\"node:stdMotionNode\"/>\n";
        out << "<mxCell id=\"node:staticNode\"/>\n";
    }

    out << "</root></mxGraphModel></diagram></mxfile>\n";
}

/// Write an application config matching the selected graph fixture.
void writeConfig( const std::filesystem::path &path,      /**< [in] config file path */
                  const std::filesystem::path &output,    /**< [in] output graph path */
                  bool                         connected, /**< [in] whether to configure linked puts */
                  std::optional<bool>          clobber,   /**< [in] explicit clobber setting, if any */
                  const std::string           &pwrKey = "testpwr.test" /**< [in] power property key */ )
{
    std::vector<std::string> sections{ "graph", "graph" };
    std::vector<std::string> keys{ "file", "outputPath" };
    std::vector<std::string> values{ "instgraph_test.drawio", output.string() };

    if( clobber.has_value() )
    {
        sections.push_back( "graph" );
        keys.push_back( "clobberOutput" );
        values.push_back( *clobber ? "true" : "false" );
    }

    if( connected )
    {
        sections.insert( sections.end(), { "staticNode", "staticNode" } );
        keys.insert( keys.end(), { "type", "inputsOn" } );
        values.insert( values.end(), { "static", "in" } );
    }
    else
    {
        sections.insert( sections.end(), { "indiPropNode", "indiPropNode", "indiPropNode", "indiPropNode" } );
        keys.insert( keys.end(), { "type", "propKey", "propEl", "propVal" } );
        values.insert( values.end(), { "indiProp", "test.test", "test", "test" } );

        sections.insert( sections.end(), { "pwrOnOffNode", "pwrOnOffNode" } );
        keys.insert( keys.end(), { "type", "pwrKey" } );
        values.insert( values.end(), { "pwrOnOff", pwrKey } );

        sections.insert( sections.end(), { "fsmNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "fsm" } );

        sections.insert( sections.end(), { "stdMotionNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "stdMotion" } );

        sections.insert( sections.end(), { "staticNode" } );
        keys.insert( keys.end(), { "type" } );
        values.insert( values.end(), { "static" } );
    }

    mx::app::writeConfigFile( path.string(), sections, keys, values );
}

/// Load a fixture config into the app under test.
void loadFixture( xInstGraph                  &app, /**< [in,out] app to configure */
                  const std::filesystem::path &root /**< [in] temporary test root */ )
{
    app.configDir( ( root / "config" ).string() );
    app.setupConfig();
    app.config().readConfig( ( root / "config" / "instgraph_test.conf" ).string() );
    app.loadConfig();
}

/// Return the mxCell opening tag with the given ID.
std::string cellTag( const std::string &xml, /**< [in] graph XML */
                     const std::string &id /**< [in] cell ID */ )
{
    size_t start = xml.find( "id=\"" + id + "\"" );
    if( start == std::string::npos )
    {
        return "";
    }

    size_t end = xml.find( '>', start );
    if( end == std::string::npos )
    {
        return "";
    }

    return xml.substr( start, end - start );
}

/// Verify a graph is published before the first property update.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes an initial graph", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::setupConfig();
    MagAOX::app::xInstGraph::loadConfig();
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    MagAOX::app::xInstGraph::appShutdown();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, false );
    writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, std::nullopt );
    const std::string source = readFile( input );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE_FALSE( std::filesystem::exists( output ) );

    REQUIRE( app.appStartup() == 0 );
    REQUIRE( std::filesystem::exists( output ) );

    ingr::instGraphXML parsed;
    std::string        emsg;
    REQUIRE( parsed.loadXMLFile( emsg, output.string() ) == 0 );

    pcf::IndiProperty property;
    property.setDevice( "testpwr" );
    property.setName( "test" );
    property.add( pcf::IndiElement( "state" ) );
    property["state"] = "On";
    REQUIRE( MagAOX::app::xInstGraph::st_igHandleSetProperty( &app, property ) == 0 );
    REQUIRE( app.appLogic() == 0 );

    REQUIRE( app.appShutdown() == 0 );
    REQUIRE_FALSE( std::filesystem::exists( output ) );
    REQUIRE( readFile( input ) == source );
}

/// Verify configuration-time writes do not publish stale link or put visibility.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes hidden puts and links", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfigImpl( *(mx::app::appConfigurator *)nullptr );
    MagAOX::app::xInstGraph::appStartup();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, true );
    writeConfig( temp.root / "config" / "instgraph_test.conf", output, true, std::nullopt );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE_FALSE( std::filesystem::exists( output ) );

    REQUIRE( app.appStartup() == 0 );
    ingr::instGraphXML parsed;
    std::string        emsg;
    REQUIRE( parsed.loadXMLFile( emsg, output.string() ) == 0 );

    std::string xml = readFile( output );
    REQUIRE( cellTag( xml, "link:staticNode:in2out" ).find( "opacity=0;" ) != std::string::npos );
    for( const char *id : { "input:staticNode:in", "output:staticNode:out" } )
    {
        REQUIRE( cellTag( xml, id ).find( "opacity=0;" ) != std::string::npos );
        REQUIRE( cellTag( xml, id ).find( "textOpacity=0;" ) != std::string::npos );
    }

    REQUIRE( app.appShutdown() == 0 );
}

/// Verify clobber requires an explicit option and only an owned output is removed.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph controls output replacement and cleanup", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfig();
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::appShutdown();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, false );
    const std::string source = readFile( input );
    {
        std::ofstream old( output );
        old << "previous output";
    }

    SECTION( "existing output is preserved by default" )
    {
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, std::nullopt );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() != 0 );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( readFile( output ) == "previous output" );
    }

    SECTION( "explicit false also preserves an existing output" )
    {
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, false );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() != 0 );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( readFile( output ) == "previous output" );
    }

    SECTION( "explicit true replaces an existing regular output" )
    {
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, true );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() == 0 );
        REQUIRE( readFile( output ) == "previous output" );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( readFile( output ).find( "<mxfile" ) != std::string::npos );
        REQUIRE( readFile( input ) == source );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE_FALSE( std::filesystem::exists( output ) );
    }

    SECTION( "a later replacement is not removed at shutdown" )
    {
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, true );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.appStartup() == 0 );
        std::filesystem::remove( output );
        {
            std::ofstream replacement( output );
            replacement << "replacement";
        }
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( readFile( output ) == "replacement" );
    }
}

/// Verify an output path can never refer to the input graph.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph rejects input output aliases", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfig();
    MagAOX::app::xInstGraph::appShutdown();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "alias.drawio";
    writeXML( input, false );
    const std::string   source  = readFile( input );
    std::optional<bool> clobber = true;

    SECTION( "exact input path with default policy" )
    {
        output  = input;
        clobber = std::nullopt;
    }

    SECTION( "exact input path with clobber enabled" )
    {
        output = input;
    }

    SECTION( "relative spelling of the input path" )
    {
        output = temp.root / "config" / ".." / "config" / "instgraph_test.drawio";
    }

    SECTION( "symbolic link to input" )
    {
        std::filesystem::create_symlink( input, output );
    }

    SECTION( "hard link to input" )
    {
        std::filesystem::create_hard_link( input, output );
    }

    writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, clobber );
    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() != 0 );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( readFile( input ) == source );
}

/// Verify clobber rejects a symlink and a failed startup preserves the prior output.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph preserves outputs before publication", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::appShutdown();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, false );

    SECTION( "symlink destination is rejected even with clobber" )
    {
        auto other = temp.root / "other.drawio";
        {
            std::ofstream old( other );
            old << "other file";
        }
        std::filesystem::create_symlink( other, output );
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, true );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() != 0 );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( std::filesystem::is_symlink( output ) );
        REQUIRE( readFile( other ) == "other file" );
    }

    SECTION( "startup failure before publication keeps an existing output" )
    {
        {
            std::ofstream old( output );
            old << "previous output";
        }
        writeConfig( temp.root / "config" / "instgraph_test.conf", output, false, true, "invalid-key" );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() == 0 );
        REQUIRE( app.appStartup() < 0 );
        REQUIRE( readFile( output ) == "previous output" );

        for( const auto &entry : std::filesystem::directory_iterator( temp.root ) )
        {
            REQUIRE( entry.path().filename().string().find( ".xInstGraph-" ) == std::string::npos );
        }
        REQUIRE( app.appShutdown() == 0 );
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
