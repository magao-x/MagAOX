/** \file indiPropNode_test.cpp
 * \brief Catch2 tests for the xInstGraph `indiPropNode` helper.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xInstGraph_files
 */

#include "../../../../tests/testXWC.hpp"

#include <fstream>

#include "../../../../libMagAOX/libMagAOX.hpp"

#define XWC_XIGNODE_TEST
#include "../indiPropNode.hpp"

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

/// Write the minimal property-node graph used by configuration tests.
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

/// Verify property-node configuration and supported INDI property types.
/** \ingroup xInstGraph_unit_test
 */
SCENARIO( "Creating and configuring an indiPropNode", "[instGraph::indiPropNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    indiPropNode::loadConfig( *(mx::app::appConfigurator *)nullptr );
    indiPropNode::propKey();
    #endif
    // clang-format on

    GIVEN( "a valid XML file, a valid config file" )
    {
        WHEN( "node is in file, default config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
            REQUIRE( tsn->propKey() == "tel.dome" );
            REQUIRE( tsn->propEl() == "status" );
            REQUIRE( tsn->propValStr() == "on" );
            REQUIRE( tsn->propValNum() == std::numeric_limits<double>::lowest() );
            REQUIRE( tsn->propValSw() == pcf::IndiElement::SwitchStateType::UnknownSwitchState );
            REQUIRE( tsn->type() == pcf::IndiProperty::Type::Unknown );
            REQUIRE( tsn->tol() == 1e-7 );
            REQUIRE( tsn->state() == false );
        }
        WHEN( "node is in file, config setting tol" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal", "tol" },
                                      { "indiProp", "tel.dome", "status", "on", "1e-8" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
            REQUIRE( tsn->propKey() == "tel.dome" );
            REQUIRE( tsn->propEl() == "status" );
            REQUIRE( tsn->propValStr() == "on" );
            REQUIRE( tsn->propValNum() == std::numeric_limits<double>::lowest() );
            REQUIRE( tsn->propValSw() == pcf::IndiElement::SwitchStateType::UnknownSwitchState );
            REQUIRE( tsn->type() == pcf::IndiProperty::Type::Unknown );
            REQUIRE( tsn->tol() == 1e-8 );
            REQUIRE( tsn->state() == false );
        }

        WHEN( "node is in file, handling a text property" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            indiPropNode *tsn = new indiPropNode( "telescope", &parentGraph );
            tsn->loadConfig( config );

            pcf::IndiProperty ip( pcf::IndiProperty::Text );
            ip.setDevice( "tel" );
            ip.setName( "dome" );
            ip.add( pcf::IndiElement( "status" ) );
            ip["status"] = "on";

            tsn->handleSetProperty( ip );
            REQUIRE( tsn->type() == pcf::IndiProperty::Text );
            REQUIRE( tsn->state() == true );

            ip["status"] = "off";
            tsn->handleSetProperty( ip );
            REQUIRE( tsn->state() == false );
        }

        WHEN( "node is in file, handling a number property" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "1.5" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            indiPropNode *tsn = new indiPropNode( "telescope", &parentGraph );
            tsn->loadConfig( config );

            pcf::IndiProperty ip( pcf::IndiProperty::Number );
            ip.setDevice( "tel" );
            ip.setName( "dome" );
            ip.add( pcf::IndiElement( "status" ) );
            ip["status"] = "1.5";

            tsn->handleSetProperty( ip );
            REQUIRE( tsn->type() == pcf::IndiProperty::Number );
            REQUIRE( tsn->propValNum() == 1.5 );
            REQUIRE( tsn->state() == true );

            ip["status"] = "1.6";
            tsn->handleSetProperty( ip );
            REQUIRE( tsn->state() == false );
        }

        WHEN( "node is in file, handling a switch property" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            indiPropNode *tsn = new indiPropNode( "telescope", &parentGraph );
            tsn->loadConfig( config );

            pcf::IndiProperty ip( pcf::IndiProperty::Switch );
            ip.setDevice( "tel" );
            ip.setName( "dome" );
            ip.add( pcf::IndiElement( "status" ) );
            ip["status"].setSwitchState( pcf::IndiElement::SwitchStateType::On );

            tsn->handleSetProperty( ip );
            REQUIRE( tsn->type() == pcf::IndiProperty::Switch );
            REQUIRE( tsn->propValSw() == pcf::IndiElement::SwitchStateType::On );
            REQUIRE( tsn->state() == true );

            ip["status"].setSwitchState( pcf::IndiElement::SwitchStateType::Off );
            tsn->handleSetProperty( ip );
            REQUIRE( tsn->state() == false );
        }
    }

    GIVEN( "invalid configs" )
    {
        WHEN( "parent graph is null, default config" )
        {
            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", nullptr );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
            REQUIRE( tsn == nullptr );
        }
        WHEN( "node not in file, default config" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope2", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == false );
            REQUIRE( tsn == nullptr );
        }

        WHEN( "node is in file, default config, wrong node type" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "fakeProp", "tel.dome", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
        WHEN( "node is in file, default config, propKey empty" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "", "status", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
        WHEN( "node is in file, default config, propEl empty" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "", "on" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
        WHEN( "node is in file, default config, propVal empty" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.come", "status", "" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            int rv = parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            REQUIRE( rv == 0 );
            REQUIRE( emsg == "" );

            indiPropNode *tsn  = nullptr;
            bool          pass = false;
            try
            {
                tsn  = new indiPropNode( "telescope", &parentGraph );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << "\n";
            }

            REQUIRE( pass == true );
            REQUIRE( tsn != nullptr );

            REQUIRE( tsn->name() == "telescope" );
            REQUIRE( tsn->node()->name() == "telescope" );

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
        WHEN( "a number property with invalid propVal" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "abcde" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            indiPropNode *tsn = new indiPropNode( "telescope", &parentGraph );
            tsn->loadConfig( config );

            pcf::IndiProperty ip( pcf::IndiProperty::Number );
            ip.setDevice( "tel" );
            ip.setName( "dome" );
            ip.add( pcf::IndiElement( "status" ) );
            ip["status"] = "1.5";

            bool pass = false;
            try
            {
                tsn->handleSetProperty( ip );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << '\n';
            }

            REQUIRE( pass == false );
        }
        WHEN( "invalid switch property" )
        {
            ingr::instGraphXML parentGraph;
            writeXML();
            mx::app::writeConfigFile( "/tmp/indiPropNode_test.conf",
                                      { "telescope", "telescope", "telescope", "telescope" },
                                      { "type", "propKey", "propEl", "propVal" },
                                      { "indiProp", "tel.dome", "status", "qq" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/indiPropNode_test.conf" );

            std::string emsg;

            parentGraph.loadXMLFile( emsg, "/tmp/xigNode_test.xml" );

            indiPropNode *tsn = new indiPropNode( "telescope", &parentGraph );
            tsn->loadConfig( config );

            pcf::IndiProperty ip( pcf::IndiProperty::Switch );
            ip.setDevice( "tel" );
            ip.setName( "dome" );
            ip.add( pcf::IndiElement( "status" ) );
            ip["status"].setSwitchState( pcf::IndiElement::SwitchStateType::On );

            bool pass = false;
            try
            {
                tsn->handleSetProperty( ip );
                pass = true;
            }
            catch( const std::exception &e )
            {
                std::cerr << e.what() << '\n';
            }

            REQUIRE( pass == false );
        }
    }
}

/// Restore a true property path when its FSM gate reopens.
/** \ingroup xInstGraph_unit_test
 */
TEST_CASE( "indiPropNode recomputes state after FSM changes", "[instGraph::indiPropNode]" )
{
    // clang-format off
    #ifdef XINSTGRAPH_TEST_DOXYGEN_REF
    indiPropNode::handleSetProperty( *(pcf::IndiProperty *)nullptr );
    indiPropNode::updateEffectiveState( false );
    #endif
    // clang-format on

    const std::string xmlPath = "/tmp/indiPropNode_F08_test.drawio";
    {
        std::ofstream out( xmlPath );
        out << "<mxfile><diagram><mxGraphModel><root>\n"
               "<mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>\n"
               "<mxCell id=\"node:telescope\"/>\n"
               "<mxCell id=\"output:telescope:out\" style=\"strokeColor=#FF0000;\"/>\n"
               "<mxCell id=\"state:telescope\" value=\"state\"/>\n"
               "<mxCell id=\"fsmstate:telescope\" value=\"fsmstate\"/>\n"
               "</root></mxGraphModel></diagram></mxfile>\n";
    }

    for( const std::string action : { "threshOff", "active" } )
    {
        CAPTURE( action );
        ingr::instGraphXML graph;
        graph.autoSave( false );
        std::string error;
        REQUIRE( graph.loadXMLFile( error, xmlPath ) == 0 );

        const std::string configPath = "/tmp/indiPropNode_F08_test.conf";
        mx::app::writeConfigFile( configPath,
                                  { "telescope", "telescope", "telescope", "telescope", "telescope", "telescope" },
                                  { "type", "propKey", "propEl", "propVal", "fsmAction", "targetStates" },
                                  { "indiProp", "tel.dome", "status", "open", action, "READY" } );
        mx::app::appConfigurator config;
        REQUIRE( config.readConfig( configPath ) == 0 );
        indiPropNode node( "telescope", &graph );
        REQUIRE_NOTHROW( node.loadConfig( config ) );

        pcf::IndiProperty property( pcf::IndiProperty::Text );
        property.setDevice( "tel" );
        property.setName( "dome" );
        property.add( pcf::IndiElement( "status" ) );
        property["status"] = "open";
        REQUIRE( node.handleSetProperty( property ) == 0 );
        REQUIRE( node.state() );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::off );

        pcf::IndiProperty fsm;
        fsm.setDevice( "telescope" );
        fsm.setName( "fsm" );
        fsm.add( pcf::IndiElement( "state" ) );
        fsm["state"] = "READY";
        REQUIRE( node.handleSetProperty( fsm ) == 0 );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::on );

        fsm["state"] = "OPERATING";
        REQUIRE( node.handleSetProperty( fsm ) == 0 );
        REQUIRE( node.state() );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::off );

        REQUIRE( node.handleSetProperty( property ) == 0 );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::off );

        fsm["state"] = "READY";
        REQUIRE( node.handleSetProperty( fsm ) == 0 );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::on );

        property["status"] = "closed";
        REQUIRE( node.handleSetProperty( property ) == 0 );
        REQUIRE_FALSE( node.state() );
        REQUIRE( graph.node( "telescope" )->output( "out" )->state() == ingr::putState::off );
    }
}

} // namespace xInstGraphTest

} // namespace libXWCTest
