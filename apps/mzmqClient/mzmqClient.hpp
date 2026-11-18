/** \file mzmqClient.hpp
 * \brief The MagAO-X milkzmqClient wrapper
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup mzmqClient_files
 */

#ifndef mzmqClient_hpp
#define mzmqClient_hpp

// #include <ImageStreamIO/ImageStruct.h>
// #include <ImageStreamIO/ImageStreamIO.h>

#include <milkzmqClient.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup mzmqClient ImageStreamIO 0mq Stream Client
 * \brief Reads the contents of an ImageStreamIO image stream over a zeroMQ channel
 *
 * <a href="../handbook/operating/software/apps/mzmqClient.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup mzmqClient_files ImageStreamIO Stream Synchronization
 * \ingroup mzmqClient
 */

/// MagAO-X application to control reading ImageStreamIO streams from a zeroMQ channel
/** Contents are published to a local ImageStreamIO shmem buffer.
 *
 * \todo handle the alternate local name option as in the base milkzmqClient
 * \todo md docs for this.
 *
 * \ingroup mzmqClient
 *
 */
class mzmqClient : public MagAOXApp<>, public milkzmq::milkzmqClient
{

  public:
    /// Default c'tor
    mzmqClient();

    /// Destructor
    ~mzmqClient() noexcept;

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

inline mzmqClient::mzmqClient() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = false;

    return;
}

inline mzmqClient::~mzmqClient() noexcept
{
    return;
}

inline void mzmqClient::setupConfig()
{
    config.add( "server.address",
                "",
                "server.address",
                argType::Required,
                "server",
                "address",
                false,
                "string",
                "The server's remote address. Usually localhost if using a tunnel." );
    config.add( "server.imagePort",
                "",
                "server.imagePort",
                argType::Required,
                "server",
                "imagePort",
                false,
                "int",
                "The server's port.  Usually the port on localhost forwarded to the host." );

    config.add( "server.shmimNames",
                "",
                "server.shmimNames",
                argType::Required,
                "server",
                "shmimNames",
                false,
                "string",
                "List of names of the remote shmim streams to get." );
}

inline void mzmqClient::loadConfig()
{
    m_argv0 = m_configName;

    config( m_address, "server.address" );
    config( m_imagePort, "server.imagePort" );

    config( m_shMemImNames, "server.shmimNames" );

    std::cerr << "m_imagePort = " << m_imagePort << "\n";
}

inline int mzmqClient::appStartup()
{
    for( size_t n = 0; n < m_shMemImNames.size(); ++n )
    {
        shMemImName( m_shMemImNames[n] );
    }

    for( size_t n = 0; n < m_imageThreads.size(); ++n )
    {
        if( imageThreadStart( n ) > 0 )
        {
            log<software_critical>( { __FILE__, __LINE__, "Starting image thread " + m_imageThreads[n].m_imageName } );
            return -1;
        }
    }

    return 0;
}

inline int mzmqClient::appLogic()
{
    // first do a join check to see if other threads have exited.

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

inline int mzmqClient::appShutdown()
{
    m_timeToDie.store( true, std::memory_order_relaxed );

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

inline void mzmqClient::reportInfo( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_INFO );
}

inline void mzmqClient::reportNotice( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_NOTICE );
}

inline void mzmqClient::reportWarning( const std::string &msg )
{
    log<text_log>( msg, logPrio::LOG_WARNING );
}

inline void mzmqClient::reportError( const std::string &msg, const std::string &file, int line )
{
    log<software_error>( { file.c_str(), (uint32_t)line, msg } );
}

} // namespace app
} // namespace MagAOX
#endif
