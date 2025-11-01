/** \file xInstGraph.hpp
 * \brief The MagAO-X Instrument Graph header file
 *
 * \ingroup instGraph_files
 */

#ifndef xInstGraph_hpp
#define xInstGraph_hpp

#include <instGraph/instGraphXML.hpp>
using namespace ingr;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "xigNodes/indiPropNode.hpp"
#include "xigNodes/fsmNode.hpp"
#include "xigNodes/pwrOnOffNode.hpp"
#include "xigNodes/stdMotionNode.hpp"
#include "xigNodes/staticNode.hpp"

/** \defgroup instGraph
 * \brief The XXXXXX application to do YYYYYYY
 *
 * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup instGraph_files
 * \ingroup instGraph
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/**
 * \ingroup instGraph
 */
class xInstGraph : public MagAOXApp<true>
{
    // Give the test harness access.
    friend class instGraph_test;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    // here add parameters which will be config-able at runtime

    ///@}

    ingr::instGraphXML m_graph;

    std::map<std::string, xigNode *> m_nodes;

    std::vector<pcf::IndiProperty *> m_nodeProps; ///< The node INDI properties to register for SetProperty

    std::multimap<std::string, xigNode *> m_nodeHandleSets; /**< Map from propery keys to nodes which
                                                                 have registered for them*/

  public:
    /// Default c'tor.
    xInstGraph();

    /// D'tor
    ~xInstGraph() noexcept;

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration from
                        which to load values*/
    );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for xInstGraph.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    static int st_igHandleSetProperty( void                    *igapp, /**< [in] this pointer */
                                       const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                       the the set property message.*/
    );

    int igHandleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                      the the set property message.*/
    );
};

xInstGraph::xInstGraph() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

xInstGraph::~xInstGraph()
{
    for( auto p : m_nodeProps )
    {
        delete p;
    }
}

void xInstGraph::setupConfig()
{
    config.add( "graph.file",
                "",
                "graph.file",
                argType::Required,
                "graph",
                "file",
                false,
                "string",
                "name of input graph drawio file, including extension, in the config directory" );

    config.add( "graph.outputPath",
                "",
                "graph.outputPath",
                argType::Required,
                "graph",
                "outputPath",
                false,
                "string",
                "path to the output graph .drawio file" );
}

int xInstGraph::loadConfigImpl( mx::app::appConfigurator &_config )
{
    ///\todo this should be relative to config path
    std::string file;
    config( file, "graph.file" );

    if( file == "" )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "no graph file in configuration (graph.file)" } );
    }

    file = m_configDir + '/' + file;


    std::string outputPath = m_graph.outputPath();
    config( outputPath, "graph.outputPath" );
    m_graph.outputPath( outputPath );

    std::string emsg;
    if( m_graph.loadXMLFile( emsg, file ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error loading graph file: " + emsg } );
    }

    std::vector<std::string> sections;

    _config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "no nodes found in configuration" } );
    }

    for( size_t i = 0; i < sections.size(); ++i )
    {
        bool isNode = config.isSetUnused( mx::app::iniFile::makeKey( sections[i], "type" ) );

        if( !isNode )
        {
            continue;
        }

        std::string type;
        config.configUnused( type, mx::app::iniFile::makeKey( sections[i], "type" ) );

        // std::cerr << "found node " << sections[i] << ": " << type << "\n";

        xigNode *xn = nullptr;

        if( type == "indiProp" )
        {
            indiPropNode *ip = nullptr;
            try
            {
                ip = new indiPropNode( sections[i], &m_graph );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            if( ip == nullptr )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "failed to allocate node" );
                throw std::runtime_error( msg );
            }

            try
            {
                ip->loadConfig( _config );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            xn = ip;
        }
        else if( type == "pwrOnOff" )
        {
            pwrOnOffNode *nn = nullptr;

            try
            {
                nn = new pwrOnOffNode( sections[i], &m_graph );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            if( nn == nullptr )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "failed to allocate node" );
                throw std::runtime_error( msg );
            }

            try
            {
                nn->loadConfig( _config );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            xn = nn;
        }
        else if( type == "fsm" )
        {
            fsmNode *nn = nullptr;

            try
            {
                nn = new fsmNode( sections[i], &m_graph );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            if( nn == nullptr )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "failed to allocate node" );
                throw std::runtime_error( msg );
            }

            try
            {
                nn->loadConfig( _config );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            xn = nn;
        }
        else if( type == "stdMotion" )
        {
            stdMotionNode *nn = nullptr;

            try
            {
                nn = new stdMotionNode( sections[i], &m_graph );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            if( nn == nullptr )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "failed to allocate node" );
                throw std::runtime_error( msg );
            }

            try
            {
                nn->loadConfig( _config );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            xn = nn;
        }
        else if( type == "static" )
        {
            staticNode *nn = nullptr;

            try
            {
                nn = new staticNode( sections[i], &m_graph );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            if( nn == nullptr )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "failed to allocate node" );
                throw std::runtime_error( msg );
            }

            try
            {
                nn->loadConfig( _config );
            }
            catch( const std::exception &e )
            {
                std::string msg = XIGN_EXCEPTION( "indiGraph::loadConfigImpl", "exception caught" );
                msg += ": ";
                msg += e.what();
                throw std::runtime_error( msg );
            }

            xn = nn;
        }

        if( xn != nullptr )
        {
            try
            {
                m_nodes.insert( { xn->node()->name(), xn } );
            }
            catch( const std::exception &e )
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string( __LINE__ );
                throw std::runtime_error( msg );
            }
        }
    }

    m_graph.hideLinks();
    m_graph.hidePuts();

    return 0;
}

void xInstGraph::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "error loading configuation" } );
        m_shutdown = true;
    }
}

std::string deviceFromKey( const std::string &key )
{
    size_t dot = key.find( '.' );

    if( dot == std::string::npos )
    {
        return "";
    }

    return key.substr( 0, dot );
}

std::string nameFromKey( const std::string &key )
{
    size_t dot = key.find( '.' );
    if( dot == std::string::npos )
    {
        return "";
    }

    return key.substr( dot + 1 );
}

int xInstGraph::appStartup()
{
    for( auto it = m_nodes.begin(); it != m_nodes.end(); ++it )
    {
        for( auto kit = it->second->keys().begin(); kit != it->second->keys().end(); ++kit )
        {
            try
            {
                std::string devName  = deviceFromKey( *kit );
                std::string propName = nameFromKey( *kit );

                if( devName == "" )
                {
                    return log<software_error, -1>(
                        { __FILE__, __LINE__, "bad devName from key: " + it->second->name() } );
                }

                if( propName == "" )
                {
                    return log<software_error, -1>(
                        { __FILE__, __LINE__, "bad propName from key: " + it->second->name() } );
                }

                m_nodeHandleSets.insert( { *kit, it->second } );

                pcf::IndiProperty *p = new pcf::IndiProperty;

                p->setDevice( devName );
                p->setName( propName );

                m_nodeProps.push_back( p );

                if( !m_indiSetCallBacks.contains( *kit ) )
                {
                    callBackInsertResult result =
                        m_indiSetCallBacks.insert( callBackValueType( *kit, { p, &st_igHandleSetProperty } ) );

                    if( !result.second )
                    {
                        return log<software_error, -1>(
                            { __FILE__, __LINE__, "failed to insert INDI property: " + p->createUniqueKey() } );
                    }
                }
            }
            catch( std::exception &e )
            {
                return log<software_error, -1>(
                    { __FILE__, __LINE__, std::string( "Exception caught: " ) + e.what() } );
            }
            catch( ... )
            {
                return log<software_error, -1>( { __FILE__, __LINE__, "Unknown exception caught." } );
            }
        }
    }

    state( stateCodes::READY );

    return 0;
}

int xInstGraph::appLogic()
{
    return 0;
}

int xInstGraph::appShutdown()
{
    //remove the output file so that it is clear there is no valid graph
    std::filesystem::remove(m_graph.outputPath());

    return 0;
}

int xInstGraph::st_igHandleSetProperty( void *igapp, const pcf::IndiProperty &ipRecv )
{
    if( igapp == nullptr )
    {
        return -1;
    }

    return reinterpret_cast<xInstGraph *>( igapp )->igHandleSetProperty( ipRecv );
}

int xInstGraph::igHandleSetProperty( const pcf::IndiProperty &ipRecv )
{
    try
    {
        auto range = m_nodeHandleSets.equal_range( ipRecv.createUniqueKey() );

        for( auto it = range.first; it != range.second; ++it )
        {
            int rv = it->second->handleSetProperty( ipRecv );
            if( rv != 0 )
            {
                return log<software_error, -1>(
                    { __FILE__, __LINE__, "error from handleSetProperty for " + it->second->name() } );
            }
        }

        return 0;
    }
    catch( std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "Exception caught: " ) + e.what() } );
    }
    catch( ... )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "Unknown exception caught." } );
    }
}

} // namespace app
} // namespace MagAOX

#endif // xInstGraph_hpp
