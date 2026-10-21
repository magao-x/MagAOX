/** \file xInstGraph_test.cpp
 * \brief Catch2 tests for xInstGraph publication, ownership, and motion routing.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup instGraph_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
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
    void configDir( const std::string &cp /**< [in] config directory */ );

    /// Access the app's configurator.
    mx::app::appConfigurator &config();

    /// Return a diagnostic from the graph-node validation pass.
    std::string nodeValidationError();

    /// Check whether a staging path or descriptor remains open.
    bool hasStage() const;

    /// Return the number of configured graph-node handlers.
    size_t handlerCount() const;

    /// Attach a driver backed by /dev/null so base Def/Set dispatch runs without an event thread.
    bool enableIndiDispatch();

    /// Check whether startup registered a callback for a device property.
    bool subscribed( const std::string &key /**< [in] device.property key */ ) const;

    /// Failure or short-write behavior injected at a publication step.
    enum class failurePoint
    {
        none,
        serialize,
        write,
        sync,
        rename,
        shortWrite,
        interruptedWrite
    };

    /// Selected failure or write behavior for the next callback.
    failurePoint failure{ failurePoint::none };

    /// True after the selected short or interrupted write was injected.
    bool injected{ false };

    /// Capture the published bytes immediately before replacement.
    bool observeRename{ false };

    /// Published bytes observed by the rename hook.
    std::string previousAtRename;

  protected:
    /// Optionally fail serialization for a test.
    int serializeGraph( std::string &xml, /**< [out] serialized XML */
                        std::string &error /**< [out] failure reason */ ) override;

    /// Optionally fail, shorten, or interrupt a staging write.
    ssize_t writeStageBytes( int         fd,   /**< [in] staging descriptor */
                             const void *data, /**< [in] bytes to write */
                             size_t      size /**< [in] byte count */ ) override;

    /// Optionally fail staging synchronization.
    int syncStage( int fd /**< [in] staging descriptor */ ) override;

    /// Observe or fail an output replacement.
    int renameStage( const std::filesystem::path &from, /**< [in] staging path */
                     const std::filesystem::path &to /**< [in] published path */ ) override;
};

void xInstGraph::configDir( const std::string &cp )
{
    m_configDir = cp;
}

mx::app::appConfigurator &xInstGraph::config()
{
    return MagAOX::app::xInstGraph::config;
}

std::string xInstGraph::nodeValidationError()
{
    std::vector<std::pair<std::string, std::string>> nodeTypes;
    std::string                                      error;
    validateNodeConfig( config(), nodeTypes, error );
    return error;
}

bool xInstGraph::hasStage() const
{
    return m_stageFd >= 0 || !m_stagePath.empty();
}

size_t xInstGraph::handlerCount() const
{
    return m_nodes.size();
}

bool xInstGraph::enableIndiDispatch()
{
    m_driverInName   = "/dev/null";
    m_driverOutName  = "/dev/null";
    m_driverCtrlName = "/dev/null";
    m_indiDriver     = new indiDriver<MagAOXApp<true>>( this, "test", "0", "0" );
    return m_indiDriver->good();
}

bool xInstGraph::subscribed( const std::string &key ) const
{
    return m_indiSetCallBacks.contains( key );
}

int xInstGraph::serializeGraph( std::string &xml, std::string &error )
{
    if( failure == failurePoint::serialize )
    {
        error = "injected serialization failure";
        return -1;
    }
    return MagAOX::app::xInstGraph::serializeGraph( xml, error );
}

ssize_t xInstGraph::writeStageBytes( int fd, const void *data, size_t size )
{
    if( failure == failurePoint::write )
    {
        errno = EIO;
        return -1;
    }
    if( !injected && failure == failurePoint::interruptedWrite )
    {
        injected = true;
        errno    = EINTR;
        return -1;
    }
    if( !injected && failure == failurePoint::shortWrite )
    {
        injected = true;
        return MagAOX::app::xInstGraph::writeStageBytes( fd, data, std::min<size_t>( size, 7 ) );
    }
    return MagAOX::app::xInstGraph::writeStageBytes( fd, data, size );
}

int xInstGraph::syncStage( int fd )
{
    if( failure == failurePoint::sync )
    {
        errno = EIO;
        return -1;
    }
    return MagAOX::app::xInstGraph::syncStage( fd );
}

int xInstGraph::renameStage( const std::filesystem::path &from, const std::filesystem::path &to )
{
    if( observeRename )
    {
        std::ifstream in( to );
        previousAtRename = { std::istreambuf_iterator<char>( in ), std::istreambuf_iterator<char>() };
    }
    if( failure == failurePoint::rename )
    {
        errno = EIO;
        return -1;
    }
    return MagAOX::app::xInstGraph::renameStage( from, to );
}

struct temporaryDirectory
{
    /// Isolated test root, removed on destruction.
    std::filesystem::path root;

    /// Create a unique temporary test directory.
    temporaryDirectory();

    /// Remove this test's files.
    ~temporaryDirectory();

    /// Prevent accidental sharing of a test directory.
    temporaryDirectory( const temporaryDirectory &other /**< [in] source directory ownership */ ) = delete;

    /// Prevent accidental sharing of a test directory.
    temporaryDirectory &operator=( const temporaryDirectory &other /**< [in] source directory ownership */ ) = delete;
};

temporaryDirectory::temporaryDirectory()
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

temporaryDirectory::~temporaryDirectory()
{
    std::error_code ec;
    std::filesystem::remove_all( root, ec );
}
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
        out << "<mxCell id=\"output:stdMotionNode:out\" style=\"strokeColor=#FF0000;\"/>\n";
        out << "<mxCell id=\"node:staticNode\"/>\n";
    }

    out << "</root></mxGraphModel></diagram></mxfile>\n";
}

/// Write a power node with a put, internal link, and status label.
void writePowerXML( const std::filesystem::path &path /**< [in] source graph path */ )
{
    std::ofstream out( path );
    out << "<mxfile><diagram><mxGraphModel><root>\n";
    out << "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>\n";
    out << "<mxCell id=\"node:pwrOnOffNode\" value=\"power\" style=\"strokeColor=#FF0000;\"/>\n";
    out << "<mxCell id=\"input:pwrOnOffNode:in\" value=\"in\" style=\"strokeColor=#FF0000;\"/>\n";
    out << "<mxCell id=\"output:pwrOnOffNode:out\" value=\"out\" style=\"strokeColor=#FF0000;\"/>\n";
    out << "<mxCell id=\"link:pwrOnOffNode:in2out\" style=\"strokeColor=#FF0000;\" "
           "source=\"input:pwrOnOffNode:in\" target=\"output:pwrOnOffNode:out\"/>\n";
    out << "<mxCell id=\"fsmstate:pwrOnOffNode:label\" value=\"before\"/>\n";
    out << "</root></mxGraphModel></diagram></mxfile>\n";
}

/// Write a config that registers the power node's INDI property.
void writePowerConfig( const std::filesystem::path &path, /**< [in] config file path */
                       const std::filesystem::path &output /**< [in] output graph path */ )
{
    mx::app::writeConfigFile( path.string(),
                              { "graph", "graph", "pwrOnOffNode", "pwrOnOffNode" },
                              { "file", "outputPath", "type", "pwrKey" },
                              { "instgraph_test.drawio", output.string(), "pwrOnOff", "testpwr.test" } );
}

/// Return a power INDI property for the requested state.
pcf::IndiProperty powerProperty( const std::string &state /**< [in] power state */ )
{
    pcf::IndiProperty property;
    property.setDevice( "testpwr" );
    property.setName( "test" );
    property.add( pcf::IndiElement( "state" ) );
    property["state"] = state;
    return property;
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

/// Write graph options followed by node and unrelated configuration sections.
void writeNodeSections( const std::filesystem::path &path,     /**< [in] config file path */
                        const std::filesystem::path &output,   /**< [in] output graph path */
                        const std::string           &sections, /**< [in] section text after graph options */
                        bool                         clobber = false /**< [in] permit an existing output */ )
{
    std::ofstream out( path );
    out << "[graph]\nfile=instgraph_test.drawio\noutputPath=" << output.string() << '\n';
    if( clobber )
    {
        out << "clobberOutput=true\n";
    }
    out << sections;
}

/// Check whether a test output has a sibling staging file.
bool hasStageFile( const std::filesystem::path &output /**< [in] graph output path */ )
{
    for( const auto &entry : std::filesystem::directory_iterator( output.parent_path() ) )
    {
        if( entry.path().filename().string().find( output.filename().string() + ".xInstGraph-" ) == 0 )
        {
            return true;
        }
    }
    return false;
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

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class destructionProbe : public xigNode
{
    /// Receives the derived destructor notification; owned by the enclosing test.
    bool &m_destroyed;

  public:
    /// Construct a node that records destruction.
    destructionProbe( ingr::instGraphXML *graph, /**< [in] parent graph */
                      bool               &destroyed /**< [out] destruction flag */ );

    /// Record destruction of the derived node.
    ~destructionProbe() override;

    /// Ignore property updates in this ownership probe.
    int handleSetProperty( const pcf::IndiProperty &property /**< [in] ignored property */ ) override;
};

destructionProbe::destructionProbe( ingr::instGraphXML *graph, bool &destroyed )
    : xigNode( "aGood", graph ), m_destroyed( destroyed )
{
}

destructionProbe::~destructionProbe()
{
    m_destroyed = true;
}

int destructionProbe::handleSetProperty( const pcf::IndiProperty &property )
{
    static_cast<void>( property );
    return 0;
}
/// \endcond

/// Owned base pointers destroy the complete derived node.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph node ownership destroys derived handlers", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    xigNode::~xigNode();
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         input = temp.root / "config" / "instgraph_test.drawio";
    std::ofstream( input ) << "<mxfile><diagram><mxGraphModel><root>"
                              "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                              "<mxCell id=\"node:aGood\"/>"
                              "</root></mxGraphModel></diagram></mxfile>";
    ingr::instGraphXML graph;
    std::string        error;
    REQUIRE( graph.loadXMLFile( error, input.string() ) == 0 );
    bool destroyed = false;
    {
        std::unique_ptr<xigNode> handler = std::make_unique<destructionProbe>( &graph, destroyed );
        REQUIRE_FALSE( destroyed );
    }
    REQUIRE( destroyed );
}

/// Failed later node configuration releases earlier handlers and staging output.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph releases handlers after partial configuration failure", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfig();
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         input  = temp.root / "config" / "instgraph_test.drawio";
    const auto         output = temp.root / "output.drawio";
    std::ofstream( input ) << "<mxfile><diagram><mxGraphModel><root>"
                              "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                              "<mxCell id=\"node:aGood\"/><mxCell id=\"node:zBad\"/>"
                              "</root></mxGraphModel></diagram></mxfile>";
    writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                       output,
                       "[aGood]\ntype=static\n[zBad]\ntype=static\noutputsOn=missing\n" );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() != 0 );
    REQUIRE( app.handlerCount() == 0 );
    REQUIRE_FALSE( app.hasStage() );
    REQUIRE_FALSE( hasStageFile( output ) );
    REQUIRE_FALSE( std::filesystem::exists( output ) );
}

/// xInstGraph rejects unsupported node types.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph rejects unsupported node types", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfigImpl( *(mx::app::appConfigurator *)nullptr );
    MagAOX::app::xInstGraph::validateNodeConfig( *(mx::app::appConfigurator *)nullptr,
        *(std::vector<std::pair<std::string, std::string>> *)nullptr, *(std::string *)nullptr );
    #endif
    // clang-format on

    for( const std::string type : { "statc", "Static", "" } )
    {
        CAPTURE( type );
        temporaryDirectory temp;
        auto               input  = temp.root / "config" / "instgraph_test.drawio";
        auto               output = temp.root / "output.drawio";
        writeXML( input, true );
        const std::string source = readFile( input );
        {
            std::ofstream existing( output );
            existing << "pre-existing output";
        }
        writeNodeSections(
            temp.root / "config" / "instgraph_test.conf", output, "[staticNode]\ntype=" + type + "\n", true );

        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() != 0 );
        REQUIRE( app.state() != stateCodes::READY );
        REQUIRE( app.handlerCount() == 0 );
        REQUIRE_FALSE( app.hasStage() );
        REQUIRE_FALSE( hasStageFile( output ) );
        REQUIRE( readFile( input ) == source );
        REQUIRE( readFile( output ) == "pre-existing output" );

        const std::string error = app.nodeValidationError();
        REQUIRE( error.find( "[staticNode]" ) != std::string::npos );
        if( type.empty() )
        {
            REQUIRE( error.find( "empty type" ) != std::string::npos );
        }
        else
        {
            REQUIRE( error.find( type ) != std::string::npos );
        }
    }
}

/// xInstGraph requires a handler for every graph node.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph requires a handler for every graph node", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfigImpl( *(mx::app::appConfigurator *)nullptr );
    MagAOX::app::xInstGraph::validateNodeConfig( *(mx::app::appConfigurator *)nullptr,
        *(std::vector<std::pair<std::string, std::string>> *)nullptr, *(std::string *)nullptr );
    #endif
    // clang-format on

    for( const std::string sections : { "[metadata]\nowner=test\n", "[staticNode]\ninputsOn=in\n" } )
    {
        CAPTURE( sections );
        temporaryDirectory temp;
        auto               input  = temp.root / "config" / "instgraph_test.drawio";
        auto               output = temp.root / "output.drawio";
        writeXML( input, true );
        const std::string source = readFile( input );
        writeNodeSections( temp.root / "config" / "instgraph_test.conf", output, sections );

        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() != 0 );
        REQUIRE( app.state() != stateCodes::READY );
        REQUIRE( app.handlerCount() == 0 );
        REQUIRE_FALSE( app.hasStage() );
        REQUIRE_FALSE( hasStageFile( output ) );
        REQUIRE_FALSE( std::filesystem::exists( output ) );
        REQUIRE( readFile( input ) == source );
        const std::string error = app.nodeValidationError();
        REQUIRE( error.find( "staticNode" ) != std::string::npos );
        REQUIRE( error.find( sections.find( "[staticNode]" ) == std::string::npos
                                 ? "no configuration section"
                                 : "without required type" ) != std::string::npos );
    }
}

/// xInstGraph rejects typed sections outside the graph.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph rejects typed sections outside the graph", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfigImpl( *(mx::app::appConfigurator *)nullptr );
    MagAOX::app::xInstGraph::validateNodeConfig( *(mx::app::appConfigurator *)nullptr,
        *(std::vector<std::pair<std::string, std::string>> *)nullptr, *(std::string *)nullptr );
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, true );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                       output,
                       "[staticNode]\ntype=static\ninputsOn=in\n[typoNode]\ntype=static\n" );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() != 0 );
    REQUIRE_FALSE( app.hasStage() );
    REQUIRE_FALSE( hasStageFile( output ) );
    REQUIRE_FALSE( std::filesystem::exists( output ) );
    REQUIRE( app.handlerCount() == 0 );
    const std::string error = app.nodeValidationError();
    REQUIRE( error.find( "[typoNode]" ) != std::string::npos );
    REQUIRE( error.find( "static" ) != std::string::npos );
    REQUIRE( error.find( "no graph node" ) != std::string::npos );
}

/// xInstGraph accepts unrelated sections alongside complete graph handlers.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph accepts unrelated sections alongside complete graph handlers", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfigImpl( *(mx::app::appConfigurator *)nullptr );
    MagAOX::app::xInstGraph::validateNodeConfig( *(mx::app::appConfigurator *)nullptr,
        *(std::vector<std::pair<std::string, std::string>> *)nullptr, *(std::string *)nullptr );
    MagAOX::app::xInstGraph::appStartup();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writeXML( input, true );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                       output,
                       "[staticNode]\ntype=static\ninputsOn=in\n[metadata]\nowner=test\n" );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.handlerCount() == 1 );
    REQUIRE( app.nodeValidationError().empty() );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( std::filesystem::exists( output ) );
    REQUIRE( app.appShutdown() == 0 );
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
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.handlerCount() == 5 );
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

/// A callback publishes its final power graph as one complete snapshot.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes one complete callback snapshot", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    MagAOX::app::xInstGraph::appLogic();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writePowerXML( input );
    writePowerConfig( temp.root / "config" / "instgraph_test.conf", output );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.appStartup() == 0 );
    std::string before = readFile( output );
    REQUIRE( before.find( "value=\"---\"" ) != std::string::npos );

    pcf::IndiProperty unrelated = powerProperty( "On" );
    unrelated.setDevice( "other" );
    struct stat beforeInfo;
    struct stat afterInfo;
    REQUIRE( ::stat( output.c_str(), &beforeInfo ) == 0 );
    REQUIRE( app.igHandleSetProperty( unrelated ) == 0 );
    REQUIRE( ::stat( output.c_str(), &afterInfo ) == 0 );
    REQUIRE( afterInfo.st_ino == beforeInfo.st_ino );
    REQUIRE( readFile( output ) == before );

    app.observeRename = true;
    REQUIRE( app.igHandleSetProperty( powerProperty( "On" ) ) == 0 );
    REQUIRE( app.previousAtRename == before );
    REQUIRE( app.appLogic() == 0 );
    std::string after = readFile( output );
    REQUIRE( after != before );
    REQUIRE( after.find( "value=\"ON\"" ) != std::string::npos );
    REQUIRE( cellTag( after, "input:pwrOnOffNode:in" ).find( "strokeColor=#00FF00;" ) != std::string::npos );
    REQUIRE( cellTag( after, "output:pwrOnOffNode:out" ).find( "strokeColor=#00FF00;" ) != std::string::npos );
    ingr::instGraphXML parsed;
    std::string        error;
    REQUIRE( parsed.loadXMLFile( error, output.string() ) == 0 );

    app.previousAtRename.clear();
    REQUIRE( app.igHandleSetProperty( powerProperty( "Off" ) ) == 0 );
    REQUIRE( app.previousAtRename == after );
    REQUIRE( readFile( output ).find( "value=\"OFF\"" ) != std::string::npos );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE_FALSE( std::filesystem::exists( output ) );
}

/// One INDI property update reaches every node subscribed to its key.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph dispatches a shared property to all handlers", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         input  = temp.root / "config" / "instgraph_test.drawio";
    const auto         output = temp.root / "output.drawio";
    {
        std::ofstream out( input );
        out << "<mxfile><diagram><mxGraphModel><root>"
               "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
               "<mxCell id=\"node:powerA\"/><mxCell id=\"output:powerA:out\" "
               "style=\"strokeColor=#FF0000;\"/>"
               "<mxCell id=\"node:powerB\"/><mxCell id=\"output:powerB:out\" "
               "style=\"strokeColor=#FF0000;\"/>"
               "</root></mxGraphModel></diagram></mxfile>";
    }
    writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                       output,
                       "[powerA]\ntype=pwrOnOff\npwrKey=testpwr.test\n"
                       "[powerB]\ntype=pwrOnOff\npwrKey=testpwr.test\n" );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.handlerCount() == 2 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.igHandleSetProperty( powerProperty( "On" ) ) == 0 );
    const std::string published = readFile( output );
    for( const char *node : { "powerA", "powerB" } )
    {
        CAPTURE( node );
        REQUIRE( cellTag( published, std::string( "output:" ) + node + ":out" ).find( "strokeColor=#00FF00;" ) !=
                 std::string::npos );
    }
    REQUIRE( app.appShutdown() == 0 );
}

/// Callback publication errors retain the previous snapshot and stop the app.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph reports callback publication failures", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    MagAOX::app::xInstGraph::appLogic();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writePowerXML( input );
    writePowerConfig( temp.root / "config" / "instgraph_test.conf", output );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.appStartup() == 0 );
    std::string before = readFile( output );

    SECTION( "serialization failure" )
    {
        app.failure = xInstGraph::failurePoint::serialize;
    }
    SECTION( "write failure" )
    {
        app.failure = xInstGraph::failurePoint::write;
    }
    SECTION( "sync failure" )
    {
        app.failure = xInstGraph::failurePoint::sync;
    }
    SECTION( "rename failure" )
    {
        app.failure = xInstGraph::failurePoint::rename;
    }

    REQUIRE( app.igHandleSetProperty( powerProperty( "On" ) ) < 0 );
    REQUIRE( app.appLogic() < 0 );
    REQUIRE( readFile( output ) == before );
    REQUIRE( app.igHandleSetProperty( powerProperty( "Off" ) ) < 0 );
    REQUIRE( readFile( output ) == before );
    for( const auto &entry : std::filesystem::directory_iterator( temp.root ) )
    {
        REQUIRE( entry.path().filename().string().find( ".xInstGraph-" ) == std::string::npos );
    }
    REQUIRE( app.appShutdown() == 0 );
}

/// Short and interrupted writes still produce a complete snapshot.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph completes short and interrupted writes", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writePowerXML( input );
    writePowerConfig( temp.root / "config" / "instgraph_test.conf", output );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.appStartup() == 0 );

    SECTION( "short write" )
    {
        app.failure = xInstGraph::failurePoint::shortWrite;
    }
    SECTION( "interrupted write" )
    {
        app.failure = xInstGraph::failurePoint::interruptedWrite;
    }

    REQUIRE( app.igHandleSetProperty( powerProperty( "On" ) ) == 0 );
    REQUIRE( app.injected );
    REQUIRE( readFile( output ).find( "value=\"ON\"" ) != std::string::npos );
    ingr::instGraphXML parsed;
    std::string        error;
    REQUIRE( parsed.loadXMLFile( error, output.string() ) == 0 );
    REQUIRE( app.appShutdown() == 0 );
}

/// An externally replaced output is left untouched during updates and shutdown.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph rejects replaced callback destinations", "[xInstGraph]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    MagAOX::app::xInstGraph::appShutdown();
    #endif
    // clang-format on

    temporaryDirectory temp;
    auto               input  = temp.root / "config" / "instgraph_test.drawio";
    auto               output = temp.root / "output.drawio";
    writePowerXML( input );
    writePowerConfig( temp.root / "config" / "instgraph_test.conf", output );
    std::string source = readFile( input );

    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.appStartup() == 0 );
    std::filesystem::remove( output );

    bool alias = false;
    SECTION( "regular replacement" )
    {
        std::ofstream replacement( output );
        replacement << "external replacement";
    }
    SECTION( "input alias" )
    {
        alias = true;
        std::filesystem::create_symlink( input, output );
    }

    REQUIRE( app.igHandleSetProperty( powerProperty( "On" ) ) < 0 );
    REQUIRE( app.appLogic() < 0 );
    REQUIRE( app.appShutdown() == 0 );
    REQUIRE( readFile( input ) == source );
    if( alias )
    {
        REQUIRE( std::filesystem::is_symlink( output ) );
    }
    else
    {
        REQUIRE( readFile( output ) == "external replacement" );
    }
}

/// Published parked positions remain active while the FSM label continues to report POWEROFF.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes parked power-off positions", "[xInstGraph][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    MagAOX::app::MagAOXApp<true>::handleDefProperty( pcf::IndiProperty() );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    #endif
    // clang-format on

    std::array<int, 3> order{ 0, 1, 2 };
    do
    {
        CAPTURE( order );
        temporaryDirectory temp;
        const auto         input  = temp.root / "config" / "instgraph_test.drawio";
        const auto         output = temp.root / "output.drawio";
        {
            std::ofstream xml( input );
            xml << "<mxfile><diagram><mxGraphModel><root>"
                   "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                   "<mxCell id=\"node:motionStage\"/>"
                   "<mxCell id=\"input:motionStage:in\" value=\"in\" style=\"strokeColor=#FF0000;\"/>"
                   "<mxCell id=\"output:motionStage:out\" value=\"out\" style=\"strokeColor=#FF0000;\"/>"
                   "<mxCell id=\"state:motionStage\" value=\"before\"/>"
                   "<mxCell id=\"fsmstate:motionStage\" value=\"before\"/>"
                   "</root></mxGraphModel></diagram></mxfile>";
        }
        const std::string source = readFile( input );
        writeNodeSections(
            temp.root / "config" / "instgraph_test.conf",
            output,
            "[motionStage]\ntype=stdMotion\ndevice=teststage\npresetPrefix=filter\npresetDir=input\nparkable=true\n" );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() == 0 );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.enableIndiDispatch() );
        for( const char *key : { "teststage.fsm", "teststage.parked", "teststage.filterName" } )
        {
            REQUIRE( app.subscribed( key ) );
        }
        pcf::IndiProperty fsm( pcf::IndiProperty::Text );
        fsm.setDevice( "teststage" );
        fsm.setName( "fsm" );
        fsm.add( pcf::IndiElement( "state", "POWEROFF" ) );
        pcf::IndiProperty parked( pcf::IndiProperty::Number );
        parked.setDevice( "teststage" );
        parked.setName( "parked" );
        parked.add( pcf::IndiElement( "current", "1" ) );
        pcf::IndiProperty preset( pcf::IndiProperty::Switch );
        preset.setDevice( "teststage" );
        preset.setName( "filterName" );
        preset.add( pcf::IndiElement( "routeA", pcf::IndiElement::On ) );
        preset.add( pcf::IndiElement( "routeB", pcf::IndiElement::Off ) );
        const std::array<pcf::IndiProperty, 3> snapshot{ fsm, parked, preset };
        for( int index : order )
        {
            app.handleDefProperty( snapshot[index] );
            REQUIRE( app.appLogic() == 0 );
        }
        std::string published = readFile( output );
        REQUIRE( cellTag( published, "fsmstate:motionStage" ).find( "value=\"POWEROFF\"" ) != std::string::npos );
        REQUIRE( cellTag( published, "state:motionStage" ).find( "value=\"routeA\"" ) != std::string::npos );
        REQUIRE( cellTag( published, "input:motionStage:in" ).find( "value=\"routeA\"" ) != std::string::npos );
        for( const char *id : { "input:motionStage:in", "output:motionStage:out" } )
        {
            REQUIRE( cellTag( published, id ).find( "strokeColor=#00FF00;" ) != std::string::npos );
        }
        preset["routeA"].setSwitchState( pcf::IndiElement::Off );
        preset["routeB"].setSwitchState( pcf::IndiElement::On );
        app.handleSetProperty( preset );
        REQUIRE( app.appLogic() == 0 );
        published = readFile( output );
        REQUIRE( cellTag( published, "input:motionStage:in" ).find( "value=\"routeB\"" ) != std::string::npos );
        REQUIRE( cellTag( published, "fsmstate:motionStage" ).find( "value=\"POWEROFF\"" ) != std::string::npos );
        parked["current"] = "0";
        app.handleSetProperty( parked );
        REQUIRE( app.appLogic() == 0 );
        published = readFile( output );
        for( const char *id : { "input:motionStage:in", "output:motionStage:out" } )
        {
            REQUIRE( cellTag( published, id ).find( "strokeColor=#FF0000;" ) != std::string::npos );
        }
        REQUIRE( cellTag( published, "fsmstate:motionStage" ).find( "value=\"POWEROFF\"" ) != std::string::npos );
        ingr::instGraphXML parsed;
        std::string        error;
        REQUIRE( parsed.loadXMLFile( error, output.string() ) == 0 );
        REQUIRE( readFile( input ) == source );
        REQUIRE_FALSE( app.hasStage() );
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE_FALSE( std::filesystem::exists( output ) );
    } while( std::next_permutation( order.begin(), order.end() ) );
}

/// Disabled parking has no callback registration and cannot change a powered-off graph.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph registers parked callbacks only for parkable stages", "[xInstGraph][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    for( const std::string option : { "", "false", "true" } )
    {
        CAPTURE( option );
        const bool         enabled = option == "true";
        temporaryDirectory temp;
        const auto         input  = temp.root / "config" / "instgraph_test.drawio";
        const auto         output = temp.root / "output.drawio";
        {
            std::ofstream xml( input );
            xml << "<mxfile><diagram><mxGraphModel><root>"
                   "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                   "<mxCell id=\"node:motionStage\"/>"
                   "<mxCell id=\"output:motionStage:out\" value=\"out\" style=\"strokeColor=#FF0000;\"/>"
                   "<mxCell id=\"state:motionStage\" value=\"before\"/>"
                   "<mxCell id=\"fsmstate:motionStage\" value=\"before\"/>"
                   "</root></mxGraphModel></diagram></mxfile>";
        }
        writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                           output,
                           "[motionStage]\ntype=stdMotion\n" +
                               ( option.empty() ? std::string{} : "parkable=" + option + "\n" ) );
        xInstGraph app;
        loadFixture( app, temp.root );
        REQUIRE( app.shutdown() == 0 );
        REQUIRE( app.appStartup() == 0 );
        REQUIRE( app.enableIndiDispatch() );
        REQUIRE( app.subscribed( "motionStage.parked" ) == enabled );
        REQUIRE( app.subscribed( "motionStage.fsm" ) );
        REQUIRE( app.subscribed( "motionStage.presetName" ) );
        pcf::IndiProperty fsm( pcf::IndiProperty::Text );
        fsm.setDevice( "motionStage" );
        fsm.setName( "fsm" );
        fsm.add( pcf::IndiElement( "state", "READY" ) );
        pcf::IndiProperty preset( pcf::IndiProperty::Switch );
        preset.setDevice( "motionStage" );
        preset.setName( "presetName" );
        preset.add( pcf::IndiElement( "open", pcf::IndiElement::On ) );
        app.handleDefProperty( fsm );
        app.handleDefProperty( preset );
        REQUIRE( cellTag( readFile( output ), "output:motionStage:out" ).find( "strokeColor=#00FF00;" ) !=
                 std::string::npos );
        fsm["state"] = "POWEROFF";
        app.handleSetProperty( fsm );
        const auto beforeParked = readFile( output );
        REQUIRE( cellTag( beforeParked, "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                 std::string::npos );
        pcf::IndiProperty parked( pcf::IndiProperty::Number );
        parked.setDevice( "motionStage" );
        parked.setName( "parked" );
        parked.add( pcf::IndiElement( "current", "1" ) );
        app.handleDefProperty( parked );
        REQUIRE( app.appLogic() == 0 );
        const auto afterParked = readFile( output );
        REQUIRE( cellTag( afterParked, "fsmstate:motionStage" ).find( "value=\"POWEROFF\"" ) != std::string::npos );
        if( enabled )
        {
            REQUIRE( cellTag( afterParked, "output:motionStage:out" ).find( "strokeColor=#00FF00;" ) !=
                     std::string::npos );
        }
        else
        {
            REQUIRE( afterParked == beforeParked );
        }
        REQUIRE( app.appShutdown() == 0 );
    }
}

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Write two beamsplitters in series with a controllable upstream source.
void writeBeamsplitterXML( const std::filesystem::path &path, /**< [in] graph file */
                           bool                         linked = true /**< [in] include the stage's internal links */ )
{
    std::ofstream xml( path );
    xml << "<mxfile><diagram><mxGraphModel><root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
           "<mxCell id=\"node:lamp\"/><mxCell id=\"output:lamp:out\" style=\"strokeColor=#FF0000;\"/>";
    for( const std::string device : { "stagebs", "fwfpm" } )
    {
        xml << "<mxCell id=\"node:" << device << "\"/><mxCell id=\"input:" << device
            << ":in\" value=\"in\" style=\"strokeColor=#FF0000;\"/>"
            << "<mxCell id=\"state:" << device << "\" value=\"before\"/>"
            << "<mxCell id=\"fsmstate:" << device << "\" value=\"before\"/>";
        for( const auto &put : device == "stagebs" ? std::vector<std::string>{ "wfs", "sci" }
                                                   : std::vector<std::string>{ "out", "refl" } )
        {
            xml << "<mxCell id=\"output:" << device << ':' << put << "\" value=\"" << put
                << "\" style=\"strokeColor=#FF0000;\"/>";
            if( linked )
            {
                xml << "<mxCell id=\"link:" << device << ":in2" << put << "\" source=\"input:" << device
                    << ":in\" target=\"output:" << device << ':' << put << "\" style=\"strokeColor=#FF0000;\"/>";
            }
        }
    }
    xml << "<mxCell id=\"beam:lamp2stagebs\" source=\"output:lamp:out\" target=\"input:stagebs:in\" "
           "style=\"strokeColor=#FF0000;\"/>"
           "<mxCell id=\"beam:stagebs2fwfpm\" source=\"output:stagebs:sci\" target=\"input:fwfpm:in\" "
           "style=\"strokeColor=#FF0000;\"/>"
           "</root></mxGraphModel></diagram></mxfile>";
}

/// Return the routing sections for the two example stages and their upstream source.
std::string beamsplitterSections( bool parkable /**< [in] enable parking on both simulated stages */ )
{
    const std::string parking = parkable ? "parkable=true\n" : "";
    return "[lamp]\ntype=pwrOnOff\npwrKey=lamp.power\n"
           "[stagebs]\ntype=stdMotion\npresetRoute.out=sci\npresetRoute.65-35=wfs,sci\n"
           "presetRoute.ha-ir=wfs,sci\npresetRoute.closed=\n" +
           parking +
           "[fwfpm]\ntype=stdMotion\npresetPrefix=filter\npresetRoute.open=out\n"
           "presetRoute.lyotlg=out\npresetRoute.mirror=refl\npresetRoute.closed=\n" +
           parking;
}

/// Build a stage FSM or parking property with one named value.
pcf::IndiProperty
beamsplitterValue( const std::string      &device,   /**< [in] property publisher */
                   const std::string      &property, /**< [in] property name */
                   const std::string      &element,  /**< [in] element name */
                   const std::string      &value,    /**< [in] reported value */
                   pcf::IndiProperty::Type type = pcf::IndiProperty::Text /**< [in] INDI property type */ )
{
    pcf::IndiProperty result( type );
    result.setDevice( device );
    result.setName( property );
    result.add( pcf::IndiElement( element, value ) );
    return result;
}

/// Build a stage preset selection using the published preset or filter names.
pcf::IndiProperty beamsplitterPreset( const std::string              &device, /**< [in] stage name */
                                      const std::vector<std::string> &selected /**< [in] selected names */ )
{
    pcf::IndiProperty result( pcf::IndiProperty::Switch );
    result.setDevice( device );
    result.setName( device == "fwfpm" ? "filterName" : "presetName" );
    for( const auto &name : selected )
    {
        result.add( pcf::IndiElement( name, pcf::IndiElement::On ) );
    }
    return result;
}

/// Require a complete published route and verify that put labels remain port names.
void requireBeamsplitterGraph( const std::string           &xml,      /**< [in] published graph */
                               const std::string           &device,   /**< [in] stage name */
                               const std::set<std::string> &selected, /**< [in] active outputs */
                               const std::string           &color = "#00FF00" /**< [in] effective active-path color */ )
{
    const auto puts =
        device == "stagebs" ? std::vector<std::string>{ "wfs", "sci" } : std::vector<std::string>{ "out", "refl" };
    for( const auto &put : puts )
    {
        const auto tag = cellTag( xml, "output:" + device + ':' + put );
        REQUIRE( tag.find( "strokeColor=" + ( selected.count( put ) ? color : "#FF0000" ) + ';' ) !=
                 std::string::npos );
        REQUIRE( tag.find( "value=\"" + put + "\"" ) != std::string::npos );
    }
    REQUIRE( cellTag( xml, "input:" + device + ":in" )
                 .find( "strokeColor=" + ( selected.empty() ? "#FF0000" : color ) + ';' ) != std::string::npos );
}
/// \endcond

/// Published routes match both matrices and follow upstream changes without fresh stage telemetry.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes beamsplitter routes and upstream changes", "[xInstGraph][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::applyPresetRoute( {} );
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         output = temp.root / "output.drawio";
    writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio" );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf", output, beamsplitterSections( false ) );
    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.enableIndiDispatch() );
    REQUIRE_FALSE( app.subscribed( "stagebs.parked" ) );
    REQUIRE_FALSE( app.subscribed( "fwfpm.parked" ) );
    auto lamp = beamsplitterValue( "lamp", "power", "state", "On" );
    app.handleDefProperty( lamp );
    requireBeamsplitterGraph( readFile( output ), "stagebs", {} );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
    for( const std::string device : { "stagebs", "fwfpm" } )
    {
        app.handleDefProperty( beamsplitterValue( device, "fsm", "state", "READY" ) );
    }
    const std::vector<std::pair<std::string, std::set<std::string>>> stageRoutes{
        { "out", { "sci" } }, { "65-35", { "wfs", "sci" } }, { "ha-ir", { "wfs", "sci" } } };
    const std::vector<std::pair<std::string, std::set<std::string>>> fpmRoutes{
        { "open", { "out" } }, { "lyotlg", { "out" } }, { "mirror", { "refl" } } };
    for( const auto &stage : stageRoutes )
        for( const auto &fpm : fpmRoutes )
        {
            CAPTURE( stage.first, fpm.first );
            app.handleSetProperty( beamsplitterPreset( "stagebs", { stage.first } ) );
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { fpm.first } ) );
            auto xml = readFile( output );
            requireBeamsplitterGraph( xml, "stagebs", stage.second );
            requireBeamsplitterGraph( xml, "fwfpm", fpm.second );
            REQUIRE( cellTag( xml, "state:stagebs" ).find( "value=\"" + stage.first + "\"" ) != std::string::npos );
            REQUIRE( cellTag( xml, "state:fwfpm" ).find( "value=\"" + fpm.first + "\"" ) != std::string::npos );
            lamp["state"] = "Off";
            app.handleSetProperty( lamp );
            xml = readFile( output );
            requireBeamsplitterGraph( xml, "stagebs", stage.second, "#FFFF00" );
            requireBeamsplitterGraph( xml, "fwfpm", fpm.second, "#FFFF00" );
            lamp["state"] = "On";
            app.handleSetProperty( lamp );
            xml = readFile( output );
            requireBeamsplitterGraph( xml, "stagebs", stage.second );
            requireBeamsplitterGraph( xml, "fwfpm", fpm.second );
        }
    app.handleSetProperty( beamsplitterPreset( "stagebs", { "65-35", "ha-ir" } ) );
    requireBeamsplitterGraph( readFile( output ), "stagebs", {} );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", { "refl" }, "#FFFF00" );
    app.handleSetProperty( beamsplitterPreset( "fwfpm", { "missing" } ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
    app.handleSetProperty( beamsplitterPreset( "stagebs", { "closed" } ) );
    REQUIRE( cellTag( readFile( output ), "state:stagebs" ).find( "value=\"closed\"" ) != std::string::npos );
    for( const std::string device : { "stagebs", "fwfpm" } )
    {
        app.handleSetProperty( beamsplitterValue( device, "fsm", "state", "POWEROFF" ) );
        app.handleDefProperty( beamsplitterValue( device, "parked", "current", "1", pcf::IndiProperty::Number ) );
        requireBeamsplitterGraph( readFile( output ), device, {} );
        REQUIRE( cellTag( readFile( output ), "fsmstate:" + device ).find( "value=\"POWEROFF\"" ) !=
                 std::string::npos );
    }
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.appShutdown() == 0 );
}

/// All initial DefProperty orders converge on mapped parked routes while preserving the real FSM.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes mapped parked presets in any initial order", "[xInstGraph][mapping][parked]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::selectedPresetRoute();
    #endif
    // clang-format on

    for( const bool parkable : { false, true } )
    {
        std::array<int, 3> order{ 0, 1, 2 };
        do
        {
            CAPTURE( parkable, order );
            temporaryDirectory temp;
            const auto         output = temp.root / "output.drawio";
            writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio" );
            writeNodeSections( temp.root / "config" / "instgraph_test.conf", output, beamsplitterSections( parkable ) );
            xInstGraph app;
            loadFixture( app, temp.root );
            REQUIRE( app.shutdown() == 0 );
            REQUIRE( app.appStartup() == 0 );
            REQUIRE( app.enableIndiDispatch() );
            app.handleDefProperty( beamsplitterValue( "lamp", "power", "state", "On" ) );
            for( const std::string device : { "stagebs", "fwfpm" } )
            {
                REQUIRE( app.subscribed( device + ".parked" ) == parkable );
                requireBeamsplitterGraph( readFile( output ), device, {} );
            }
            for( const int index : order )
                for( const std::string device : { "stagebs", "fwfpm" } )
                {
                    const std::array<pcf::IndiProperty, 3> properties{
                        beamsplitterValue( device, "fsm", "state", "POWEROFF" ),
                        beamsplitterValue( device, "parked", "current", "1", pcf::IndiProperty::Number ),
                        beamsplitterPreset( device, { device == "stagebs" ? "65-35" : "mirror" } ) };
                    app.handleDefProperty( properties[index] );
                }
            auto xml = readFile( output );
            requireBeamsplitterGraph(
                xml, "stagebs", parkable ? std::set<std::string>{ "wfs", "sci" } : std::set<std::string>{} );
            requireBeamsplitterGraph(
                xml, "fwfpm", parkable ? std::set<std::string>{ "refl" } : std::set<std::string>{} );
            for( const std::string device : { "stagebs", "fwfpm" } )
            {
                REQUIRE( cellTag( xml, "fsmstate:" + device ).find( "value=\"POWEROFF\"" ) != std::string::npos );
            }
            if( parkable )
            {
                app.handleSetProperty(
                    beamsplitterValue( "stagebs", "parked", "current", "0", pcf::IndiProperty::Number ) );
                requireBeamsplitterGraph( readFile( output ), "stagebs", {} );
                requireBeamsplitterGraph( readFile( output ), "fwfpm", { "refl" }, "#FFFF00" );
                app.handleSetProperty(
                    beamsplitterValue( "stagebs", "parked", "current", "1", pcf::IndiProperty::Number ) );
                app.handleSetProperty( beamsplitterPreset( "fwfpm", { "closed" } ) );
                requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
                REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"closed\"" ) != std::string::npos );
            }
            else
            {
                for( const std::string device : { "stagebs", "fwfpm" } )
                {
                    app.handleSetProperty( beamsplitterValue( device, "fsm", "state", "READY" ) );
                }
                requireBeamsplitterGraph( readFile( output ), "stagebs", { "wfs", "sci" } );
                requireBeamsplitterGraph( readFile( output ), "fwfpm", { "refl" } );
            }
            REQUIRE( app.appLogic() == 0 );
            REQUIRE( app.appShutdown() == 0 );
        } while( std::next_permutation( order.begin(), order.end() ) );
    }
}

/// Missing internal propagation links fail configuration before any graph output is published.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph rejects beamsplitters without internal links", "[xInstGraph][mapping]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::loadConfig();
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         output = temp.root / "output.drawio";
    writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio", false );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf", output, beamsplitterSections( false ) );
    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 1 );
    REQUIRE_FALSE( std::filesystem::exists( output ) );
    REQUIRE_FALSE( app.hasStage() );
    REQUIRE( app.appShutdown() == 0 );
}

/// Default FPM transmission applies to unlisted names while explicit reflected and blocked positions take precedence.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes default routes with explicit overrides", "[xInstGraph][mapping][defaultRoute]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::loadPresetRoutes( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::selectedPresetRoute();
    #endif
    // clang-format on

    for( const bool powerOff : { false, true } )
    {
        std::array<int, 3> order{ 0, 1, 2 };
        do
        {
            CAPTURE( powerOff, order );
            temporaryDirectory temp;
            const auto         output = temp.root / "output.drawio";
            writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio" );
            writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                               output,
                               "[lamp]\ntype=pwrOnOff\npwrKey=lamp.power\n"
                               "[stagebs]\ntype=stdMotion\npresetRoute.out=sci\n"
                               "[fwfpm]\ntype=stdMotion\npresetPrefix=filter\ndefaultRoute=out\n"
                               "presetRoute.lyotlg=out,refl\npresetRoute.mirror=refl\npresetRoute.closed=\n" +
                                   std::string( powerOff ? "parkable=true\n" : "" ) );
            xInstGraph app;
            loadFixture( app, temp.root );
            REQUIRE( app.shutdown() == 0 );
            REQUIRE( app.config().m_unusedConfigs.at( mx::app::iniFile::makeKey( "fwfpm", "defaultRoute" ) ).used );
            REQUIRE( app.appStartup() == 0 );
            REQUIRE( app.enableIndiDispatch() );
            auto lamp = beamsplitterValue( "lamp", "power", "state", "On" );
            app.handleDefProperty( lamp );
            app.handleDefProperty( beamsplitterValue( "stagebs", "fsm", "state", "READY" ) );
            app.handleDefProperty( beamsplitterPreset( "stagebs", { "out" } ) );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
            const std::array<pcf::IndiProperty, 3> properties{
                beamsplitterValue( "fwfpm", "fsm", "state", powerOff ? "POWEROFF" : "READY" ),
                beamsplitterValue( "fwfpm", "parked", "current", "1", pcf::IndiProperty::Number ),
                beamsplitterPreset( "fwfpm", { "cmc3" } ) };
            for( const auto index : order )
            {
                app.handleDefProperty( properties[index] );
            }
            requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
            for( const std::string name : { "open", "cmc", "new-filter" } )
            {
                app.handleSetProperty( beamsplitterPreset( "fwfpm", { name } ) );
                requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
                REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"" + name + "\"" ) !=
                         std::string::npos );
            }
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { "lyotlg" } ) );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out", "refl" } );
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { "mirror" } ) );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", { "refl" } );
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { "closed" } ) );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
            REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"closed\"" ) != std::string::npos );
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { "new-filter" } ) );
            lamp["state"] = "Off";
            app.handleSetProperty( lamp );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" }, "#FFFF00" );
            lamp["state"] = "On";
            app.handleSetProperty( lamp );
            requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
            for( const auto &selection : { std::vector<std::string>{},
                                           std::vector<std::string>{ "none" },
                                           std::vector<std::string>{ "open", "lyotlg" } } )
            {
                app.handleSetProperty( beamsplitterPreset( "fwfpm", selection ) );
                requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
            }
            app.handleSetProperty( beamsplitterPreset( "fwfpm", { "open" } ) );
            if( powerOff )
            {
                app.handleSetProperty(
                    beamsplitterValue( "fwfpm", "parked", "current", "0", pcf::IndiProperty::Number ) );
            }
            else
            {
                app.handleSetProperty( beamsplitterValue( "fwfpm", "fsm", "state", "POWEROFF" ) );
            }
            requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
            REQUIRE( cellTag( readFile( output ), "fsmstate:fwfpm" ).find( "value=\"POWEROFF\"" ) !=
                     std::string::npos );
            REQUIRE( app.appLogic() == 0 );
            REQUIRE( app.appShutdown() == 0 );
        } while( std::next_permutation( order.begin(), order.end() ) );
    }
}

/// Numeric-only callbacks publish live positions while preserving put colors, preset labels, and the reported FSM.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph publishes numerical positions without changing put routing", "[xInstGraph][position]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::updatePositionLabel();
    #endif
    // clang-format on

    for( const std::string prefix : { "preset", "filter" } )
        for( const std::string state : { "READY", "OPERATING", "POWEROFF" } )
        {
            std::array<int, 3> order{ 0, 1, 2 };
            do
            {
                CAPTURE( prefix, state, order );
                temporaryDirectory temp;
                const auto         output = temp.root / "output.drawio";
                {
                    std::ofstream xml( temp.root / "config" / "instgraph_test.drawio" );
                    xml << "<mxfile><diagram><mxGraphModel><root>"
                           "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                           "<mxCell id=\"node:motionStage\"/>"
                           "<mxCell id=\"output:motionStage:out\" value=\"out\" style=\"strokeColor=#FF0000;\"/>"
                           "<mxCell id=\"state:motionStage\" value=\"before\"/>"
                           "<mxCell id=\"fsmstate:motionStage\" value=\"before\"/>"
                           "</root></mxGraphModel></diagram></mxfile>";
                }
                writeNodeSections(
                    temp.root / "config" / "instgraph_test.conf",
                    output,
                    "[motionStage]\ntype=stdMotion\ndevice=teststage\nparkable=true\npresetPrefix=" + prefix + "\n" );
                xInstGraph app;
                loadFixture( app, temp.root );
                REQUIRE( app.shutdown() == 0 );
                REQUIRE( app.appStartup() == 0 );
                REQUIRE( app.enableIndiDispatch() );
                const std::string property = prefix == "filter" ? "filter" : "position";
                REQUIRE( app.subscribed( "teststage." + property ) );
                REQUIRE_FALSE(
                    app.subscribed( std::string( "teststage." ) + ( prefix == "filter" ? "position" : "filter" ) ) );
                app.handleDefProperty(
                    beamsplitterValue( "teststage", "parked", "current", "1", pcf::IndiProperty::Number ) );
                pcf::IndiProperty preset( pcf::IndiProperty::Switch );
                preset.setDevice( "teststage" );
                preset.setName( prefix + "Name" );
                preset.add( pcf::IndiElement( "none", pcf::IndiElement::On ) );
                const std::array<pcf::IndiProperty, 3> properties{
                    beamsplitterValue( "teststage", property, "current", "-12.34567", pcf::IndiProperty::Number ),
                    beamsplitterValue( "teststage", "fsm", "state", state ),
                    preset };
                for( const auto index : order )
                {
                    app.handleDefProperty( properties[index] );
                }
                auto published = readFile( output );
                REQUIRE( cellTag( published, "state:motionStage" ).find( "value=\"-12.3457\"" ) != std::string::npos );
                REQUIRE( cellTag( published, "output:motionStage:out" ).find( "value=\"-12.3457\"" ) !=
                         std::string::npos );
                REQUIRE( cellTag( published, "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                         std::string::npos );
                REQUIRE( cellTag( published, "fsmstate:motionStage" ).find( "value=\"" + state + "\"" ) !=
                         std::string::npos );
                app.handleSetProperty(
                    beamsplitterValue( "teststage", property, "current", "8.5", pcf::IndiProperty::Number ) );
                published = readFile( output );
                REQUIRE( cellTag( published, "state:motionStage" ).find( "value=\"8.5000\"" ) != std::string::npos );
                REQUIRE( cellTag( published, "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                         std::string::npos );
                app.handleSetProperty(
                    beamsplitterValue( "teststage", property, "target", "100", pcf::IndiProperty::Number ) );
                REQUIRE( readFile( output ) == published );
                preset["none"].setSwitchState( pcf::IndiElement::Off );
                preset.add( pcf::IndiElement( "open", pcf::IndiElement::On ) );
                app.handleSetProperty( preset );
                const auto beforePosition = cellTag( readFile( output ), "output:motionStage:out" );
                REQUIRE( beforePosition.find( state == "OPERATING" ? "strokeColor=#FF0000;"
                                                                   : "strokeColor=#00FF00;" ) != std::string::npos );
                app.handleSetProperty(
                    beamsplitterValue( "teststage", property, "current", "11.125", pcf::IndiProperty::Number ) );
                REQUIRE( cellTag( readFile( output ), "output:motionStage:out" ) == beforePosition );
                preset["open"].setSwitchState( pcf::IndiElement::Off );
                app.handleSetProperty( preset );
                published = readFile( output );
                REQUIRE( cellTag( published, "state:motionStage" ).find( "value=\"11.1250\"" ) != std::string::npos );
                REQUIRE( cellTag( published, "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                         std::string::npos );
                app.handleSetProperty(
                    beamsplitterValue( "teststage", property, "current", "bad", pcf::IndiProperty::Number ) );
                published = readFile( output );
                REQUIRE( cellTag( published, "state:motionStage" ).find( "value=\"---\"" ) != std::string::npos );
                REQUIRE( cellTag( published, "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                         std::string::npos );
                REQUIRE( app.appLogic() == 0 );
                REQUIRE( app.appShutdown() == 0 );
            } while( std::next_permutation( order.begin(), order.end() ) );
        }

    temporaryDirectory temp;
    const auto         output = temp.root / "output.drawio";
    writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio" );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf", output, beamsplitterSections( false ) );
    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.enableIndiDispatch() );
    app.handleDefProperty( beamsplitterValue( "lamp", "power", "state", "On" ) );
    app.handleDefProperty( beamsplitterValue( "stagebs", "fsm", "state", "READY" ) );
    app.handleDefProperty( beamsplitterPreset( "stagebs", { "out" } ) );
    app.handleDefProperty( beamsplitterValue( "fwfpm", "fsm", "state", "READY" ) );
    app.handleDefProperty( beamsplitterPreset( "fwfpm", { "none" } ) );
    app.handleDefProperty( beamsplitterValue( "fwfpm", "filter", "current", "3.2", pcf::IndiProperty::Number ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
    REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"3.2000\"" ) != std::string::npos );
    app.handleSetProperty( beamsplitterPreset( "fwfpm", { "open" } ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
    const auto beforePosition = readFile( output );
    app.handleSetProperty( beamsplitterValue( "fwfpm", "filter", "current", "7.4", pcf::IndiProperty::Number ) );
    REQUIRE( readFile( output ) == beforePosition );
    app.handleSetProperty( beamsplitterPreset( "fwfpm", {} ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
    REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"7.4000\"" ) != std::string::npos );
    REQUIRE( app.appShutdown() == 0 );
}

/// Parked mapped/default routes retain their colors and position labels while the real FSM advances through startup.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph preserves parked routes through startup", "[xInstGraph][parked][startup]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::parkedState();
    stdMotionNode::putsShouldBeOn();
    stdMotionNode::updatePositionLabel();
    #endif
    // clang-format on

    temporaryDirectory temp;
    const auto         output = temp.root / "output.drawio";
    writeBeamsplitterXML( temp.root / "config" / "instgraph_test.drawio" );
    writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                       output,
                       "[lamp]\ntype=pwrOnOff\npwrKey=lamp.power\n"
                       "[stagebs]\ntype=stdMotion\nparkable=true\npresetRoute.65-35=wfs,sci\n"
                       "[fwfpm]\ntype=stdMotion\nparkable=true\npresetPrefix=filter\ndefaultRoute=out\n"
                       "presetRoute.closed=\n" );
    xInstGraph app;
    loadFixture( app, temp.root );
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.enableIndiDispatch() );
    auto lamp = beamsplitterValue( "lamp", "power", "state", "On" );
    app.handleDefProperty( lamp );
    for( const std::string device : { "stagebs", "fwfpm" } )
    {
        app.handleDefProperty( beamsplitterValue( device, "fsm", "state", "POWEROFF" ) );
        app.handleDefProperty( beamsplitterValue( device, "parked", "current", "1", pcf::IndiProperty::Number ) );
        app.handleDefProperty( beamsplitterPreset( device, { device == "stagebs" ? "65-35" : "open" } ) );
    }
    for( const std::string state : { "POWEROFF", "POWERON", "NODEVICE", "NOTCONNECTED", "CONNECTED" } )
        for( const std::string device : { "stagebs", "fwfpm" } )
        {
            const auto before = readFile( output );
            app.handleSetProperty( beamsplitterValue( device, "fsm", "state", state ) );
            const auto published = readFile( output );
            requireBeamsplitterGraph( published, "stagebs", { "wfs", "sci" } );
            requireBeamsplitterGraph( published, "fwfpm", { "out" } );
            REQUIRE( cellTag( published, "fsmstate:" + device ).find( "value=\"" + state + "\"" ) !=
                     std::string::npos );
            for( const char *cell : { "output:stagebs:wfs",
                                      "output:stagebs:sci",
                                      "output:fwfpm:out",
                                      "output:fwfpm:refl",
                                      "state:stagebs",
                                      "state:fwfpm" } )
            {
                REQUIRE( cellTag( published, cell ) == cellTag( before, cell ) );
            }
        }
    lamp["state"] = "Off";
    app.handleSetProperty( lamp );
    requireBeamsplitterGraph( readFile( output ), "stagebs", { "wfs", "sci" }, "#FFFF00" );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" }, "#FFFF00" );
    lamp["state"] = "On";
    app.handleSetProperty( lamp );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
    app.handleSetProperty( beamsplitterPreset( "fwfpm", { "closed" } ) );
    for( const std::string state : { "POWEROFF", "POWERON", "NODEVICE", "NOTCONNECTED", "CONNECTED" } )
    {
        app.handleSetProperty( beamsplitterValue( "fwfpm", "fsm", "state", state ) );
        requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
        REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"closed\"" ) != std::string::npos );
    }
    app.handleDefProperty( beamsplitterValue( "fwfpm", "filter", "current", "4.25", pcf::IndiProperty::Number ) );
    app.handleSetProperty( beamsplitterPreset( "fwfpm", { "none" } ) );
    for( const std::string state : { "POWEROFF", "POWERON", "NODEVICE", "NOTCONNECTED", "CONNECTED" } )
    {
        app.handleSetProperty( beamsplitterValue( "fwfpm", "fsm", "state", state ) );
        requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
        REQUIRE( cellTag( readFile( output ), "state:fwfpm" ).find( "value=\"4.2500\"" ) != std::string::npos );
        REQUIRE( cellTag( readFile( output ), "fsmstate:fwfpm" ).find( "value=\"" + state + "\"" ) !=
                 std::string::npos );
    }
    app.handleSetProperty( beamsplitterPreset( "fwfpm", { "open" } ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", { "out" } );
    app.handleSetProperty( beamsplitterValue( "fwfpm", "parked", "current", "0", pcf::IndiProperty::Number ) );
    requireBeamsplitterGraph( readFile( output ), "fwfpm", {} );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.appShutdown() == 0 );
}

/// Controllers without numeric telemetry retain preset and parking callbacks and ignore unsolicited position
/// properties.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "xInstGraph registers numerical callbacks only when hasPosition", "[xInstGraph][position]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    MagAOX::app::xInstGraph::appStartup();
    MagAOX::app::xInstGraph::igHandleSetProperty( pcf::IndiProperty() );
    stdMotionNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    stdMotionNode::handleSetProperty( pcf::IndiProperty() );
    stdMotionNode::updatePositionLabel();
    #endif
    // clang-format on

    for( const std::string prefix : { "preset", "filter" } )
        for( const std::string option : { "", "true", "false" } )
        {
            CAPTURE( prefix, option );
            const bool         enabled = option != "false";
            temporaryDirectory temp;
            const auto         output = temp.root / "output.drawio";
            {
                std::ofstream xml( temp.root / "config" / "instgraph_test.drawio" );
                xml << "<mxfile><diagram><mxGraphModel><root>"
                       "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>"
                       "<mxCell id=\"node:motionStage\"/>"
                       "<mxCell id=\"output:motionStage:out\" value=\"out\" style=\"strokeColor=#FF0000;\"/>"
                       "<mxCell id=\"state:motionStage\" value=\"before\"/>"
                       "<mxCell id=\"fsmstate:motionStage\" value=\"before\"/>"
                       "</root></mxGraphModel></diagram></mxfile>";
            }
            writeNodeSections( temp.root / "config" / "instgraph_test.conf",
                               output,
                               "[motionStage]\ntype=stdMotion\ndevice=teststage\nparkable=true\npresetPrefix=" +
                                   prefix + "\n" + ( option.empty() ? "" : "hasPosition=" + option + "\n" ) );
            xInstGraph app;
            loadFixture( app, temp.root );
            REQUIRE( app.shutdown() == 0 );
            if( !option.empty() )
            {
                REQUIRE(
                    app.config().m_unusedConfigs.at( mx::app::iniFile::makeKey( "motionStage", "hasPosition" ) ).used );
            }
            REQUIRE( app.appStartup() == 0 );
            REQUIRE( app.enableIndiDispatch() );
            const std::string property = prefix == "filter" ? "filter" : "position";
            REQUIRE( app.subscribed( "teststage." + property ) == enabled );
            REQUIRE( app.subscribed( "teststage.fsm" ) );
            REQUIRE( app.subscribed( "teststage." + prefix + "Name" ) );
            REQUIRE( app.subscribed( "teststage.parked" ) );
            app.handleDefProperty( beamsplitterValue( "teststage", "fsm", "state", "READY" ) );
            pcf::IndiProperty preset( pcf::IndiProperty::Switch );
            preset.setDevice( "teststage" );
            preset.setName( prefix + "Name" );
            preset.add( pcf::IndiElement( "out", pcf::IndiElement::On ) );
            app.handleDefProperty( preset );
            REQUIRE( cellTag( readFile( output ), "state:motionStage" ).find( "value=\"out\"" ) != std::string::npos );
            REQUIRE( cellTag( readFile( output ), "output:motionStage:out" ).find( "strokeColor=#00FF00;" ) !=
                     std::string::npos );
            app.handleDefProperty(
                beamsplitterValue( "teststage", property, "current", "23.75", pcf::IndiProperty::Number ) );
            preset["out"].setSwitchState( pcf::IndiElement::Off );
            preset.add( pcf::IndiElement( "none", pcf::IndiElement::On ) );
            app.handleSetProperty( preset );
            const auto beforePosition = readFile( output );
            REQUIRE( cellTag( beforePosition, "state:motionStage" )
                         .find( enabled ? "value=\"23.7500\"" : "value=\"---\"" ) != std::string::npos );
            app.handleSetProperty(
                beamsplitterValue( "teststage", property, "current", "40.25", pcf::IndiProperty::Number ) );
            if( enabled )
            {
                REQUIRE( cellTag( readFile( output ), "state:motionStage" ).find( "value=\"40.2500\"" ) !=
                         std::string::npos );
            }
            else
            {
                REQUIRE( readFile( output ) == beforePosition );
            }
            REQUIRE( cellTag( readFile( output ), "output:motionStage:out" ).find( "strokeColor=#FF0000;" ) !=
                     std::string::npos );
            app.handleDefProperty(
                beamsplitterValue( "teststage", "parked", "current", "1", pcf::IndiProperty::Number ) );
            for( const std::string state : { "POWEROFF", "POWERON", "NODEVICE", "NOTCONNECTED", "CONNECTED" } )
            {
                app.handleSetProperty( beamsplitterValue( "teststage", "fsm", "state", state ) );
                REQUIRE( cellTag( readFile( output ), "state:motionStage" )
                             .find( enabled ? "value=\"40.2500\"" : "value=\"---\"" ) != std::string::npos );
            }
            preset["none"].setSwitchState( pcf::IndiElement::Off );
            preset["out"].setSwitchState( pcf::IndiElement::On );
            app.handleSetProperty( preset );
            REQUIRE( cellTag( readFile( output ), "state:motionStage" ).find( "value=\"out\"" ) != std::string::npos );
            REQUIRE( cellTag( readFile( output ), "output:motionStage:out" ).find( "strokeColor=#00FF00;" ) !=
                     std::string::npos );
            REQUIRE( app.appShutdown() == 0 );
        }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
