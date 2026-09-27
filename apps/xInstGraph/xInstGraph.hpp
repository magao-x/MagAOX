/** \file xInstGraph.hpp
 * \brief The MagAO-X Instrument Graph header file
 *
 * \ingroup instGraph_files
 */

#ifndef xInstGraph_hpp
#define xInstGraph_hpp

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

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
 * \brief The MagAO-X instrument graph publisher.
 *
 * \ingroup apps
 */

/** \defgroup instGraph_files
 * \ingroup instGraph
 */

// forward for test harness
namespace xInstGraph_test
{
class xInstGraph;
}

namespace MagAOX
{
namespace app
{

/// The MagAO-X instrument graph application.
/**
 * \ingroup instGraph
 */
class xInstGraph : public MagAOXApp<true>
{
    // Give the test harness access.
    friend class xInstGraph_test::xInstGraph;

  protected:
    /** \name Output Configuration - Data
     *@{
     */
    /// Input diagram path, resolved against the application's config directory.
    std::filesystem::path m_inputPath;

    /// Requested output diagram path, resolved against the current directory.
    std::filesystem::path m_outputPath;

    /// Permit replacement of an existing regular output file at startup.
    bool m_clobberOutput{ false };

    ///@}

    /** \name Output Ownership - Data
     *@{
     */
    /// Filesystem identity used to avoid removing a path replaced by another process.
    struct fileIdentity
    {
        dev_t device{ 0 }; ///< Device containing the file.
        ino_t inode{ 0 };  ///< Inode of the file.
    };

    /// Private staging path used while the graph is configured.
    std::filesystem::path m_stagePath;

    /// Identity of the staging file created by this run.
    fileIdentity m_stageIdentity;

    /// Open descriptor retaining the staging inode until publication or cleanup.
    int m_stageFd{ -1 };

    /// Identity of the output file published by this run.
    fileIdentity m_outputIdentity;

    /// Open descriptor retaining the published inode until shutdown.
    int m_outputFd{ -1 };

    /// True after this run successfully publishes its output.
    bool m_outputPublished{ false };

    ///@}

    /// The in-memory graph and its draw.io XML representation.
    ingr::instGraphXML m_graph;

    /// Node handlers allocated during configuration and retained for the app lifetime.
    std::map<std::string, xigNode *> m_nodes;

    /// Node INDI properties owned by this app for SetProperty registration.
    std::vector<pcf::IndiProperty *> m_nodeProps;

    /// Property keys mapped to each node that consumes their updates.
    std::multimap<std::string, xigNode *> m_nodeHandleSets;

    /// Validate that the output is distinct from the input and may be published.
    int checkOutputPath( std::string &error /**< [out] reason for a rejected path */ ) const;

    /// Create a private staging file in the output directory.
    int createStage( std::string &error /**< [out] reason staging failed */ );

    /// Publish the staged initial snapshot according to the clobber policy.
    int publishOutput( std::string &error /**< [out] reason publication failed */ );

    /// Remove only staging and output files still owned by this run.
    void cleanupOwnedFiles() noexcept;

  public:
    /// Construct the instrument graph app.
    xInstGraph();

    /// Release registered properties and any output owned by this run.
    ~xInstGraph() noexcept;

    /// Register the graph input, output, and clobber settings.
    virtual void setupConfig();

    /// Load graph configuration; exposed separately for the test harness.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration to load */
    );

    /// Load the graph, configure nodes, and prepare a private output.
    virtual void loadConfig();

    /// Register INDI callbacks and publish the initial graph snapshot.
    /**
     * \returns 0 on success or -1 if startup cannot publish the output.
     */
    virtual int appStartup();

    /// Implementation of the FSM for xInstGraph.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Remove output files created by this run.
    virtual int appShutdown();

    /// Forward an INDI SetProperty callback to this app.
    static int st_igHandleSetProperty( void                    *igapp, /**< [in] application instance */
                                       const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                       the set property message */
    );

    /// Dispatch a received INDI property to interested nodes.
    int igHandleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                      the set property message */
    );
};

xInstGraph::xInstGraph() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

xInstGraph::~xInstGraph()
{
    cleanupOwnedFiles();

    for( auto p : m_nodeProps )
    {
        delete p;
    }
}

int xInstGraph::checkOutputPath( std::string &error ) const
{
    if( m_inputPath == m_outputPath )
    {
        error = "graph output path is the input graph path";
        return -1;
    }

    struct stat inputInfo;
    if( ::stat( m_inputPath.c_str(), &inputInfo ) < 0 )
    {
        error = "cannot stat input graph " + m_inputPath.string() + ": " + std::strerror( errno );
        return -1;
    }

    struct stat outputInfo;
    if( ::lstat( m_outputPath.c_str(), &outputInfo ) < 0 )
    {
        if( errno == ENOENT )
        {
            return 0;
        }

        error = "cannot inspect graph output " + m_outputPath.string() + ": " + std::strerror( errno );
        return -1;
    }

    struct stat outputTarget;
    if( ::stat( m_outputPath.c_str(), &outputTarget ) == 0 && inputInfo.st_dev == outputTarget.st_dev &&
        inputInfo.st_ino == outputTarget.st_ino )
    {
        error = "graph output path refers to the input graph";
        return -1;
    }

    if( !m_clobberOutput )
    {
        error = "graph output already exists (set graph.clobberOutput=true to replace it): " + m_outputPath.string();
        return -1;
    }

    if( !S_ISREG( outputInfo.st_mode ) )
    {
        error = "graph output is not a regular file: " + m_outputPath.string();
        return -1;
    }

    return 0;
}

int xInstGraph::createStage( std::string &error )
{
    std::string       stageTemplate = m_outputPath.string() + ".xInstGraph-XXXXXX";
    std::vector<char> name( stageTemplate.begin(), stageTemplate.end() );
    name.push_back( '\0' );

    int fd = ::mkstemp( name.data() );
    if( fd < 0 )
    {
        error = "cannot create graph staging file: " + std::string( std::strerror( errno ) );
        return -1;
    }

    struct stat info;
    if( ::fstat( fd, &info ) < 0 )
    {
        error = "cannot inspect graph staging file: " + std::string( std::strerror( errno ) );
        ::close( fd );
        ::unlink( name.data() );
        return -1;
    }

    m_stagePath     = name.data();
    m_stageIdentity = { info.st_dev, info.st_ino };
    m_stageFd       = fd;
    m_graph.outputPath( m_stagePath.string() );

    return 0;
}

int xInstGraph::publishOutput( std::string &error )
{
    if( m_stagePath.empty() )
    {
        error = "graph staging file was not created";
        return -1;
    }

    struct stat stageInfo;
    if( ::lstat( m_stagePath.c_str(), &stageInfo ) < 0 || stageInfo.st_dev != m_stageIdentity.device ||
        stageInfo.st_ino != m_stageIdentity.inode || !S_ISREG( stageInfo.st_mode ) || stageInfo.st_size == 0 )
    {
        error = "graph staging file is missing, empty, or was replaced";
        return -1;
    }

    if( checkOutputPath( error ) < 0 )
    {
        return -1;
    }

    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    if( m_clobberOutput )
    {
        struct stat oldOutput;
        if( ::lstat( m_outputPath.c_str(), &oldOutput ) == 0 )
        {
            mode = oldOutput.st_mode & 0777;
        }
    }

    if( ::chmod( m_stagePath.c_str(), mode ) < 0 )
    {
        error = "cannot set graph output permissions: " + std::string( std::strerror( errno ) );
        return -1;
    }

    if( m_clobberOutput )
    {
        if( ::rename( m_stagePath.c_str(), m_outputPath.c_str() ) < 0 )
        {
            error = "cannot publish graph output: " + std::string( std::strerror( errno ) );
            return -1;
        }
        m_stagePath.clear();
    }
    else
    {
        if( ::link( m_stagePath.c_str(), m_outputPath.c_str() ) < 0 )
        {
            error = "cannot publish graph output without replacing an existing file: " +
                    std::string( std::strerror( errno ) );
            return -1;
        }
    }

    m_outputIdentity  = m_stageIdentity;
    m_outputFd        = m_stageFd;
    m_stageFd         = -1;
    m_outputPublished = true;

    if( !m_stagePath.empty() )
    {
        if( ::unlink( m_stagePath.c_str() ) < 0 )
        {
            error = "cannot remove graph staging link: " + std::string( std::strerror( errno ) );
            return -1;
        }
        m_stagePath.clear();
    }

    m_graph.outputPath( m_outputPath.string() );
    return 0;
}

void xInstGraph::cleanupOwnedFiles() noexcept
{
    if( !m_stagePath.empty() )
    {
        struct stat info;
        if( ::lstat( m_stagePath.c_str(), &info ) == 0 && info.st_dev == m_stageIdentity.device &&
            info.st_ino == m_stageIdentity.inode )
        {
            ::unlink( m_stagePath.c_str() );
        }
        m_stagePath.clear();
    }

    if( m_stageFd >= 0 )
    {
        ::close( m_stageFd );
        m_stageFd = -1;
    }

    if( m_outputPublished )
    {
        struct stat info;
        if( ::lstat( m_outputPath.c_str(), &info ) == 0 && info.st_dev == m_outputIdentity.device &&
            info.st_ino == m_outputIdentity.inode )
        {
            ::unlink( m_outputPath.c_str() );
        }
        m_outputPublished = false;
    }

    if( m_outputFd >= 0 )
    {
        ::close( m_outputFd );
        m_outputFd = -1;
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

    config.add( "graph.clobberOutput",
                "",
                "graph.clobberOutput",
                argType::Required,
                "graph",
                "clobberOutput",
                false,
                "bool",
                "replace an existing regular output file at startup (default false)" );
}

int xInstGraph::loadConfigImpl( mx::app::appConfigurator &_config )
{
    std::string file;
    _config( file, "graph.file" );

    if( file == "" )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "no graph file in configuration (graph.file)" } );
    }

    m_inputPath = std::filesystem::absolute( m_configDir + '/' + file ).lexically_normal();

    std::string outputPath;
    _config( outputPath, "graph.outputPath" );
    if( outputPath.empty() )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "no graph output path in configuration" } );
    }

    m_outputPath = std::filesystem::absolute( outputPath ).lexically_normal();
    _config( m_clobberOutput, "graph.clobberOutput" );

    std::string emsg;
    if( m_graph.loadXMLFile( emsg, m_inputPath.string() ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error loading graph file: " + emsg } );
    }

    if( checkOutputPath( emsg ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, emsg } );
    }

    std::vector<std::string> sections;

    _config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "no nodes found in configuration" } );
    }

    if( createStage( emsg ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, emsg } );
    }

    for( size_t i = 0; i < sections.size(); ++i )
    {
        bool isNode = config.isSetUnused( mx::app::iniFile::makeKey( sections[i], "type" ) );

        if( !isNode )
        {
            continue;
        }

        std::string type;
        _config.configUnused( type, mx::app::iniFile::makeKey( sections[i], "type" ) );

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
    try
    {
        if( loadConfigImpl( config ) < 0 )
        {
            cleanupOwnedFiles();
            log<software_error>( { __FILE__, __LINE__, "error loading configuration" } );
            m_shutdown = true;
        }
    }
    catch( const std::exception &e )
    {
        cleanupOwnedFiles();
        log<software_error>( { __FILE__, __LINE__, std::string( "error loading configuration: " ) + e.what() } );
        m_shutdown = true;
    }
}

/// Return the device portion of a device.property INDI key.
std::string deviceFromKey( const std::string &key /**< [in] INDI property key */ )
{
    size_t dot = key.find( '.' );

    if( dot == std::string::npos )
    {
        return "";
    }

    return key.substr( 0, dot );
}

/// Return the property portion of a device.property INDI key.
std::string nameFromKey( const std::string &key /**< [in] INDI property key */ )
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
                    cleanupOwnedFiles();
                    return log<software_error, -1>(
                        { __FILE__, __LINE__, "bad devName from key: " + it->second->name() } );
                }

                if( propName == "" )
                {
                    cleanupOwnedFiles();
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
                        cleanupOwnedFiles();
                        return log<software_error, -1>(
                            { __FILE__, __LINE__, "failed to insert INDI property: " + p->createUniqueKey() } );
                    }
                }
            }
            catch( std::exception &e )
            {
                cleanupOwnedFiles();
                return log<software_error, -1>(
                    { __FILE__, __LINE__, std::string( "Exception caught: " ) + e.what() } );
            }
            catch( ... )
            {
                cleanupOwnedFiles();
                return log<software_error, -1>( { __FILE__, __LINE__, "Unknown exception caught." } );
            }
        }
    }

    try
    {
        m_graph.stateChange();

        std::string emsg;
        if( publishOutput( emsg ) < 0 )
        {
            cleanupOwnedFiles();
            return log<software_error, -1>( { __FILE__, __LINE__, emsg } );
        }
    }
    catch( const std::exception &e )
    {
        cleanupOwnedFiles();
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "error publishing graph: " ) + e.what() } );
    }
    catch( ... )
    {
        cleanupOwnedFiles();
        return log<software_error, -1>( { __FILE__, __LINE__, "unknown error publishing graph" } );
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
    cleanupOwnedFiles();
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
    std::cerr << ipRecv.createUniqueKey() << '\n';
    try
    {
        auto range = m_nodeHandleSets.equal_range( ipRecv.createUniqueKey() );

        for( auto it = range.first; it != range.second; ++it )
        {
            std::cerr << it->second->name() << '\n';

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
