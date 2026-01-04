/** \file git_state.hpp
 * \brief The MagAO-X logger git_state log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 * History:
 * - 2018-08-18 created by JRM
 */
#ifndef logger_types_git_state_hpp
#define logger_types_git_state_hpp

#include "generated/git_state_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the build-time git state.
/** \ingroup logger_types
 */
struct git_state : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::GIT_STATE;

    /// The default level
    static const flatlogs::logPrio defaultLevel = flatlogs::logPrio::LOG_INFO;

    /// The type of the input message
    struct messageT : public fbMessage
    {
        // Construct from components of early versions
        messageT( const std::string &repoName, ///< [in] the name of the repo
                  const std::string &sha1,     ///< [in] the SHA1 hash of the repo
                  const bool         modified  ///< [in] the modified status (true or false)
        )
        {
            auto _repoName = builder.CreateString( repoName );
            auto _sha1     = builder.CreateString( sha1 );

            uint8_t _modified = modified;

            auto gs = CreateGit_state_fb( builder, _repoName, _sha1, _modified );
            builder.Finish( gs );
        }

        /// Construct from components of latest versions
        messageT( const std::string &repoName, ///< [in] the name of the repo
                  const std::string &sha1,     ///< [in] the SHA1 hash of the repo
                  const bool         modified, ///< [in] the modified status (true or false)
                  const std::string &url,      ///< [in] the url of the repo
                  const std::string &branch,   ///< [in] the branch of the repo
                  const std::string &path,     ///< [in] the source path of the repo
                  const bool         untracked ///< [in] whether or not untracked files are present (true or false)
        )
        {
            auto _repoName = builder.CreateString( repoName );
            auto _sha1     = builder.CreateString( sha1 );

            uint8_t _modified = modified;

            auto _url    = builder.CreateString( url );
            auto _branch = builder.CreateString( branch );
            auto _path   = builder.CreateString( path );

            uint8_t _untracked = untracked;

            auto gs = CreateGit_state_fb( builder, _repoName, _sha1, _modified, _url, _branch, _path, _untracked );
            builder.Finish( gs );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyGit_state_fbBuffer( verifier );
    }

    /// Get the message formatte for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto rgs = GetGit_state_fb( msgBuffer );

        std::string str;
        if( rgs->repo() )
        {
            str = rgs->repo()->c_str();
        }

        str += " GIT: ";

        if( rgs->sha1() )
        {
            str += rgs->sha1()->c_str();
        }

        if( rgs->url() )
        {
            str += std::format( " url: {}", rgs->url()->c_str() );
        }

        if( rgs->branch() )
        {
            str += std::format( " branch: {}", rgs->branch()->c_str() );
        }

        if( rgs->path() )
        {
            str += std::format( " path: {}", rgs->path()->c_str() );
        }

        if( rgs->modified() > 0 )
        {
            str += " MODIFIED";
        }

        if( rgs->untracked() )
        {
            str += " UNTRACKED";
        }

        return str;
    }

    /// Access the repo name field
    static std::string repoName( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->repo() )
            return std::string( rgs->repo()->c_str() );
        else
            return "";
    }

    /// Access the sha1 field
    static std::string sha1( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->sha1() )
        {
            return std::string( rgs->sha1()->c_str() );
        }
        else
        {
            return "";
        }
    }

    /// Access the modified field
    static bool modified( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->modified() > 0 )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Access the url field
    static std::string url( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->url() )
        {
            return std::string( rgs->url()->c_str() );
        }
        else
        {
            return "";
        }
    }

    /// Access the branch field
    static std::string branch( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->branch() )
        {
            return std::string( rgs->branch()->c_str() );
        }
        else
        {
            return "";
        }
    }

    /// Access the path field
    static std::string path( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->path() )
        {
            return std::string( rgs->path()->c_str() );
        }
        else
        {
            return "";
        }
    }

    /// Access the untracked field
    static bool untracked( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto rgs = GetGit_state_fb( msgBuffer );

        if( rgs->untracked() > 0 )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "repoName" )
        {
            return logMetaDetail( { "GIT REPO NAME",
                                    "git repository name",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &repoName ),
                                    false } );
        }
        else if( member == "sha1" )
        {
            return logMetaDetail( { "GIT REPO SHA1",
                                    "git repo sha1 hash",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &sha1 ),
                                    false } );
        }
        else if( member == "modified" )
        {
            return logMetaDetail( { "GIT REPO MODIFIED",
                                    "git repo modified state",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &modified ),
                                    false } );
        }
        else if( member == "url" )
        {
            return logMetaDetail( { "GIT REPO URL",
                                    "git repo remote url",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &url ),
                                    false } );
        }
        else if( member == "branch" )
        {
            return logMetaDetail( { "GIT REPO BRANCH",
                                    "git repo branch name",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &branch ),
                                    false } );
        }
        else if( member == "path" )
        {
            return logMetaDetail( { "GIT REPO PATH",
                                    "git repo source path",
                                    logMeta::valTypes::String,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &path ),
                                    false } );
        }
        else if( member == "untracked" )
        {
            return logMetaDetail( { "GIT REPO UNTRACKED",
                                    "git repo untracked files present",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &untracked ),
                                    false } );
        }
        else
        {
            std::cerr << "No member " << member << " in git_state\n";
            return logMetaDetail();
        }
    }

}; // git_state

} // namespace logger
} // namespace MagAOX

#endif // logger_types_git_state_hpp
