/** \file mzmqServer.hpp
 * \brief The MagAO-X milkzmqServer wrapper
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mzmqServer_files
 */

#ifndef mzmqServer_hpp
#define mzmqServer_hpp

// #include <ImageStruct.h>
// #include <ImageStreamIO.h>

#include <milkzmqServer.hpp>

// #include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup mzmqServer ImageStreamIO Stream Server
 * \brief Writes the contents of an ImageStreamIO image stream over a zeroMQ channel
 *
 * <a href="../handbook/operating/software/apps/mzmqServer.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup mzmqServer_files ImageStreamIO Stream Synchronization
 * \ingroup mzmqServer
 */

/// MagAO-X application to control writing ImageStreamIO streams to a zeroMQ channel
/** \todo document this better
 * \ingroup mzmqServer
 *
 */
class mzmqServer : public MagAOXApp<>, public milkzmq::milkzmqServer
{

  public:
    /// Default c'tor
    mzmqServer();

    /// Destructor
    ~mzmqServer() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Startup functions
    /** Sets up the INDI vars.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for the Siglent SDG
    virtual int appLogic();

    /// Do any needed shutdown tasks.  Currently nothing in this app.
    virtual int appShutdown();

  protected:
    bool                     m_compress{ false };
    std::vector<std::string> m_shMemImNames;

    /** \name milkzmq Status and Error Handling
     * Implementation of status updates, warnings, and errors from milkzmq using logs.
     *
     * @{
     */

    /// Log status (with LOG_INFO level of priority).
    virtual void reportInfo( const std::string &msg /**< [in] the status message */ );

    /// Log status (with LOG_NOTICE level of priority).
    virtual void reportNotice( const std::string &msg /**< [in] the status message */ );

    /// Log a warning.
    virtual void reportWarning( const std::string &msg /**< [in] the warning message */ );

    /// Log an error.
    virtual void reportError( const std::string &msg,  ///< [in] the error message
                              const std::string &file, ///< [in] the name of the file where the error occurred
                              int                line  ///< [in] the line number of the error
    );
    ///@}
};

inline mzmqServer::mzmqServer() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = false;

    return;
}

inline mzmqServer::~mzmqServer() noexcept
{
    return;
}

inline void mzmqServer::setupConfig()
{
    config.add(
        "server.imagePort", "", "server.imagePort", argType::Required, "server", "imagePort", false, "int", "" );

    config.add(
        "server.shmimNames", "", "server.shmimNames", argType::Required, "server", "shmimNames", false, "string", "" );

    config.add(
        "server.usecSleep", "", "server.usecSleep", argType::Required, "server", "usecSleep", false, "int", "" );

    config.add( "server.fpsTgt", "", "server.fpsTgt", argType::Required, "server", "fpsTgt", false, "float", "" );

    config.add( "server.fpsGain", "", "server.fpsGain", argType::Required, "server", "fpsGain", false, "float", "" );

    config.add( "server.compress",
                "",
                "server.compress",
                argType::Required,
                "server",
                "compress",
                false,
                "bool",
                "Flag to turn on compression for INT16 and UINT16." );
}

inline void mzmqServer::loadConfig()
{
    m_argv0 = m_configName;

    config( m_imagePort, "server.imagePort" );

    config( m_shMemImNames, "server.shmimNames" );
    config( m_usecSleep, "server.usecSleep" );
    config( m_fpsTgt, "server.fpsTgt" );

    config( m_fpsGain, "server.fpsGain" );

    config( m_compress, "server.compress" );
}

#include <sys/syscall.h>

inline int mzmqServer::appStartup()
{
    if( m_compress )
        defaultCompression();

    for( size_t n = 0; n < m_shMemImNames.size(); ++n )
    {
        shMemImName( m_shMemImNames[n] );
    }

    if( serverThreadStart() < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    for( size_t n = 0; n < m_imageThreads.size(); ++n )
    {
        if( imageThreadStart( n ) > 0 )
        {
            log<software_critical>( { __FILE__, __LINE__, "Starting image thread " + m_imageThreads[n].m_imageName } );
            return -1;
        }
    }

    std::cerr << "Main Thread: " << syscall( SYS_gettid ) << "\n";

    return 0;
}

inline int mzmqServer::appLogic()
{
    // first do a join check to see if other threads have exited.

    if( pthread_tryjoin_np( m_serverThread.native_handle(), 0 ) == 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "server thread has exited" } );

        return -1;
    }

    for( size_t n = 0; n < m_imageThreads.size(); ++n )
    {
        if( pthread_tryjoin_np( m_imageThreads[n].m_thread->native_handle(), 0 ) == 0 )
        {
            log<software_error>(
                { __FILE__, __LINE__, "image thread " + m_imageThreads[n].m_imageName + " has exited" } );

            return -1;
        }
    }

    return 0;
}

inline int mzmqServer::appShutdown()
{
    m_timeToDie.store( true, std::memory_order_relaxed );

    serverThreadKill();

    if( m_serverThread.joinable() )
    {
        m_serverThread.join();
    }

    for( size_t n = 0; n < m_imageThreads.size(); ++n )
    {
        imageThreadKill( n );
    }

    for( size_t n = 0; n < m_imageThreads.size(); ++n )
    {
        if( m_imageThreads[n].m_thread->joinable() )
        {
            m_imageThreads[n].m_thread->join();
        }
    }

    return 0;
}

inline void mzmqServer::reportInfo( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_INFO );
}

inline void mzmqServer::reportNotice( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_NOTICE );
}

inline void mzmqServer::reportWarning( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_WARNING );
}

inline void mzmqServer::reportError( const std::string &msg, const std::string &file, int line )
{
    log<software_error>( { file.c_str(), (uint32_t)line, msg } );
}

} // namespace app
} // namespace MagAOX
#endif
