/** \file instGraph.hpp
 * \brief The MagAO-X Instrument Graph header file
 *
 * \ingroup instGraph_files
 */

#ifndef xInstGraph_hpp
#define xInstGraph_hpp

#include "/usr/local/include/instGraph/instGraphXML.hpp"
using namespace ingr;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch

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

#include "xigNodes/fsmNode.hpp"
#include "xigNodes/pwrOnOffNode.hpp"
#include "xigNodes/stdMotionNode.hpp"

class xInstGraph;

class xInstGraphIndiClient : public pcf::IndiClient
{
  public:
    xInstGraph *m_parent{ nullptr };

    xInstGraphIndiClient( const std::string &name,
                          const std::string &version,
                          const std::string &something,
                          const std::string &ipAddr,
                          int port,
                          xInstGraph *parent )
        : IndiClient( name, version, something, ipAddr, port ), m_parent( parent )
    {
    }

    void execute()
    {
        processIndiRequests( false );
    }

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
};

/// The MagAO-X xxxxxxxx
/**
 * \ingroup instGraph
 */
class xInstGraph : public mx::app::application
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
    std::multimap<std::string, xigNode *> m_nodeHandleSets;

    xInstGraphIndiClient *m_client{ nullptr };

  public:
    /// Default c'tor.
    xInstGraph();

    /// D'tor, declared and defined for noexcept.
    ~xInstGraph() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for instGraph.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int execute();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    virtual void handleSetProperty(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the set property message.*/ );
};

xInstGraph::xInstGraph()
{

    return;
}

void xInstGraph::setupConfig()
{
    config.add("graph.file", "", "graph.file", argType::Required, "graph", "file", false, "string", "path to the graph .drawio file");

}

int xInstGraph::loadConfigImpl( mx::app::appConfigurator &_config )
{
    ///\todo this should be relative to config path
    std::string file;
    config(file, "graph.file");

    std::string emsg;
    if( m_graph.loadXMLFile( emsg, file ) < 0 )
    {
        std::cerr << emsg << "\n";
        return -1;
    }

    std::vector<std::string> sections;

    _config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        std::cerr << "no nodes found in config\n";
        return -1;
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

        std::cerr << "found node " << sections[i] << ": " << type << "\n";

        if( type == "pwrOnOff" )
        {
            pwrOnOffNode *nn;

            try
            {
                nn = new pwrOnOffNode(sections[i], &m_graph);
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }

            try
            {
                nn->loadConfig(_config);
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }

            try
            {
                m_nodes.insert( { nn->node()->name(), nn } );
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }

        }
        else if(type == "stdMotion")
        {
            stdMotionNode *nn;
            try
            {
                 nn = new stdMotionNode(sections[i], &m_graph);
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }

            try
            {
                nn->loadConfig(_config);
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }

            try
            {
                m_nodes.insert( { nn->node()->name(), nn } );
            }
            catch(const std::exception& e)
            {
                std::string msg = e.what();
                msg += "\ncaught at ";
                msg += __FILE__;
                msg += " " + std::to_string(__LINE__);
                throw std::runtime_error(msg);
            }
        }
    }

    return 0;
}

void xInstGraph::loadConfig()
{
    loadConfigImpl( config );
}

std::string deviceFromKey( const std::string &key )
{
    size_t dot = key.find( '.' );
    if( dot == std::string::npos )
        return "";

    return key.substr( 0, dot );
}

std::string nameFromKey( const std::string &key )
{
    size_t dot = key.find( '.' );
    if( dot == std::string::npos )
        return "";

    return key.substr( dot + 1 );
}

int xInstGraph::appStartup()
{

    return 0;
}

int xInstGraph::execute()
{
    appStartup();

    // std::cout << ingr::beamState2String(m_beam_source2fwtelsim->state()) << "\n";

    bool m_shutdown = false;
    bool m_connected = false;

    while( !m_shutdown )
    {
        while( !m_connected && !m_shutdown )
        {
            try
            {
                m_client = new xInstGraphIndiClient( "instGraph", "1.7", "1,7", "127.0.0.1", 7624, this );
            }
            catch( ... )
            {
                sleep( 1 );
                continue;
            }

            m_client->activate();

            sleep( 2 );

            if( m_client->getQuitProcess() )
            {
                m_client->quitProcess();
                m_client->deactivate();
                sleep( 2 );
                delete m_client;
                m_client = nullptr;
                continue;
            }

            for( auto it = m_nodes.begin(); it != m_nodes.end(); ++it )
            {
                for( auto kit = it->second->keys().begin(); kit != it->second->keys().end(); ++kit )
                {
                    m_nodeHandleSets.insert( { *kit, it->second } );
                    std::string devname = deviceFromKey( *kit );
                    std::string propname = nameFromKey( *kit );

                    pcf::IndiProperty ip;
                    ip.setDevice( devname );
                    ip.setName( propname );
                    m_client->sendGetProperties( ip );

                    if( m_client->getQuitProcess() )
                    {
                        break;
                    }
                }

                if( m_client->getQuitProcess() )
                {
                    break;
                }
            }

            if( m_client->getQuitProcess() )
            {
                m_client->quitProcess();
                m_client->deactivate();
                sleep( 2 );
                delete m_client;
                m_client = nullptr;
                continue;
            }

            m_connected = true;
        }

        while( m_connected && !m_shutdown )
        {
            mx::sys::milliSleep( 1000 );

            if( m_client->getQuitProcess() )
            {
                m_connected = false;
                m_client->quitProcess();
                m_client->deactivate();
                sleep( 2 );
                delete m_client;
                m_client = nullptr;
            }
        }
    }

    return 0;
}

int xInstGraph::appShutdown()
{
    return 0;
}

void xInstGraph::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    auto range = m_nodeHandleSets.equal_range( ipRecv.createUniqueKey() );

    for( auto it = range.first; it != range.second; ++it )
    {
        it->second->handleSetProperty( ipRecv );
    }
}

void xInstGraphIndiClient::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( m_parent )
    {
        m_parent->handleSetProperty( ipRecv );
    }
}

void xInstGraphIndiClient::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    handleSetProperty( ipRecv );
}

#endif // xInstGraph_hpp
