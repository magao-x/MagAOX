/** \file raidMonitor.hpp
 * \brief Bounded, read-only discovery of Linux MD array health from mdstat.
 * \ingroup sysMonitor_files
 */
#ifndef sysMonitor_raidMonitor_hpp
#define sysMonitor_raidMonitor_hpp

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <fcntl.h>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../libMagAOX/logger/types/telem_raid.hpp"

namespace MagAOX
{
namespace app
{
namespace sysMonitorDetail
{

/// One discovered array, with human-readable context retained outside compact telemetry.
struct raidArrayT
{
    std::string m_name;    ///< Kernel array name; the app assigns its stable telemetry index.
    std::string m_level;   ///< MD personality, or empty for an inactive metadata container.
    std::string m_status;  ///< Concise assessment without changing maintenance percentages.
    std::string m_members; ///< Device descriptors and flags, followed by the redundancy bitmap when present.

    logger::telem_raid::arrayT m_sample; ///< Current numeric assessment; index is unassigned by the collector.
};

/// One complete collection attempt; parse failures must not replace previously known arrays with healthy state.
struct raidSnapshotT
{
    std::vector<raidArrayT> m_arrays; ///< Discovered arrays, sorted by kernel name on a successful parse.
    std::string             m_status; ///< Collection outcome, independent of individual array health.
    uint8_t                 m_result{ logger::telem_raid::ParseError }; ///< Whether discovery completed or failed.
    double                  m_checkTime{ 0 }; ///< Unix time of this attempt, including failed reads.
};

/// Parse complete mdstat framing and array summaries without accessing the host.
inline raidSnapshotT parseMdstat( const std::string &text /**< [in] complete synthetic or collected mdstat text */,
                                  double             checkTime /**< [in] Unix time of this collection attempt */ );

/// Read at most one MiB from the supplied mdstat path, without invoking a command or writing to MD.
inline raidSnapshotT checkRAID( const std::string &path /**< [in] mdstat file path, normally /proc/mdstat */,
                                double             checkTime /**< [in] Unix time of this collection attempt */ );

namespace raidDetail
{

/// Bound memory use and parsing work even when the configured source is not procfs.
constexpr size_t textLimit = 1024 * 1024;

/// Bound array cardinality and redundancy maps independently of the input-byte limit.
constexpr unsigned countLimit = 65536;

/// Split one line on whitespace, keeping member flags and bracketed summaries intact.
inline std::vector<std::string> words( const std::string &line /**< [in] source line */ );

/// Check a kernel device/personality token without interpreting it as a filesystem path.
inline bool identifier( const std::string &text /**< [in] token to validate */ );

/// Parse an unsigned decimal token without signs, overflow or trailing characters.
inline bool number( unsigned          &value /**< [out] parsed number */,
                    const std::string &text /**< [in] complete decimal token */ );

/// Validate one member descriptor and identify failed and normally active components.
inline bool member( const std::string &text /**< [in] device[descriptor](flags) token */,
                    std::string       &name /**< [out] component device name */,
                    bool              &faulty /**< [out] whether the member has the kernel's faulty flag */,
                    bool              &active /**< [out] whether this is a nonfaulty nonspare nonjournal component */ );

/// Recognize maintenance lines while discarding percentages and throughput.
inline std::string maintenance( const std::vector<std::string> &tokens /**< [in] continuation-line tokens */ );

/// Parse one framed array block, returning false for incomplete or inconsistent health evidence.
inline bool array( raidArrayT                                  &result /**< [out] array assessment */,
                   const std::vector<std::vector<std::string>> &lines /**< [in] header and continuation tokens */ );

inline std::vector<std::string> words( const std::string &line )
{
    std::istringstream       input( line );
    std::vector<std::string> result;
    std::string              token;

    while( input >> token )
    {
        result.push_back( token );
    }

    return result;
}

inline bool identifier( const std::string &text )
{
    return !text.empty() && text.size() <= 255 &&
           text.find_first_not_of( "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-" ) ==
               std::string::npos;
}

inline bool number( unsigned &value, const std::string &text )
{
    if( text.empty() || text.find_first_not_of( "0123456789" ) != std::string::npos )
    {
        return false;
    }

    auto parsed = std::from_chars( text.data(), text.data() + text.size(), value );
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

inline bool member( const std::string &text, std::string &name, bool &faulty, bool &active )
{
    auto begin = text.find( '[' );
    auto end   = text.find( ']', begin );

    if( begin == std::string::npos || end == std::string::npos || !identifier( text.substr( 0, begin ) ) )
    {
        return false;
    }

    unsigned   descriptor;
    const auto index = text.substr( begin + 1, end - begin - 1 );

    if( index != "-1" && !number( descriptor, index ) )
    {
        return false;
    }

    faulty = false;
    active = true;
    std::set<char> flags;

    for( size_t pos = end + 1; pos < text.size(); pos += 3 )
    {
        if( pos + 3 > text.size() || text[pos] != '(' || text[pos + 2] != ')' ||
            std::string( "WJFSR" ).find( text[pos + 1] ) == std::string::npos || !flags.insert( text[pos + 1] ).second )
        {
            return false;
        }

        faulty = faulty || text[pos + 1] == 'F';
        active = active && text[pos + 1] != 'F' && text[pos + 1] != 'S' && text[pos + 1] != 'J';
    }

    name = text.substr( 0, begin );
    return true;
}

inline std::string maintenance( const std::vector<std::string> &tokens )
{
    size_t pos = 0;

    if( !tokens.empty() && tokens[0].size() >= 3 && tokens[0].front() == '[' && tokens[0].back() == ']' &&
        tokens[0].substr( 1, tokens[0].size() - 2 ).find_first_not_of( "=.>" ) == std::string::npos )
    {
        pos = 1;
    }

    if( pos >= tokens.size() )
    {
        return {};
    }

    const auto equals = tokens[pos].find( '=' );
    const auto action = tokens[pos].substr( 0, equals );

    if( action != "resync" && action != "recovery" && action != "recover" && action != "reshape" && action != "check" &&
        action != "repair" )
    {
        return {};
    }

    if( equals == std::string::npos &&
        ( pos + 1 >= tokens.size() || tokens[pos + 1].empty() || tokens[pos + 1].front() != '=' ) )
    {
        return {};
    }

    return action;
}

inline bool array( raidArrayT &result, const std::vector<std::vector<std::string>> &lines )
{
    using telem        = logger::telem_raid;
    const auto &header = lines.front();

    if( header.size() < 3 || !identifier( header[0] ) || header[1] != ":" )
    {
        return false;
    }

    result.m_name    = header[0];
    const auto state = header[2];

    if( state != "active" && state != "inactive" && state != "broken" )
    {
        return false;
    }

    size_t      pos = 3;
    std::string readOnly;

    if( pos < header.size() && ( header[pos] == "(read-only)" || header[pos] == "(auto-read-only)" ) )
    {
        readOnly = header[pos++];
    }

    if( state != "inactive" )
    {
        if( pos >= header.size() || !identifier( header[pos] ) )
        {
            return false;
        }

        result.m_level = header[pos++];
    }

    std::vector<std::string> summary;
    std::set<std::string>    devices;
    std::string              action;
    bool                     haveBlocks    = false;
    bool                     haveSummary   = false;
    bool                     faulty        = false;
    unsigned                 activeMembers = 0;

    for( size_t row = 0; row < lines.size(); ++row )
    {
        const auto &tokens = lines[row];
        size_t      col    = row == 0 ? pos : 0;

        if( row > 0 && haveSummary )
        {
            if( tokens[0] == "bitmap:" )
            {
                if( tokens.size() < 2 )
                {
                    return false;
                }

                continue;
            }

            const auto currentAction = maintenance( tokens );

            if( !currentAction.empty() )
            {
                if( !action.empty() )
                {
                    return false;
                }

                action = currentAction;
                continue;
            }
        }

        for( ; col < tokens.size(); ++col )
        {
            if( !haveSummary )
            {
                if( col + 1 < tokens.size() && tokens[col + 1] == "blocks" )
                {
                    if( tokens[col].empty() || tokens[col].find_first_not_of( "0123456789" ) != std::string::npos )
                    {
                        return false;
                    }

                    haveBlocks  = true;
                    haveSummary = true;
                    ++col;
                    continue;
                }

                if( state == "inactive" && tokens[col] == "super" )
                {
                    haveSummary = true;
                }
                else
                {
                    std::string name;
                    bool        memberFaulty, memberActive;

                    if( !member( tokens[col], name, memberFaulty, memberActive ) || !devices.insert( name ).second )
                    {
                        return false;
                    }

                    if( !result.m_members.empty() )
                    {
                        result.m_members += ' ';
                    }

                    result.m_members += tokens[col];
                    faulty = faulty || memberFaulty;
                    activeMembers += memberActive;
                    continue;
                }
            }

            summary.push_back( tokens[col] );
        }
    }

    if( !haveSummary || ( state != "inactive" && ( !haveBlocks || devices.empty() ) ) )
    {
        return false;
    }

    std::string metadata, bitmap;
    unsigned    expected = 0, active = 0;
    bool        haveCounts = false;

    for( size_t n = 0; n < summary.size(); ++n )
    {
        const auto &token = summary[n];

        if( token == "super" )
        {
            if( !metadata.empty() || n + 1 == summary.size() )
            {
                return false;
            }

            metadata = summary[++n];
        }
        else if( token == "blocks" || token.find( '(' ) != std::string::npos )
        {
            return false;
        }
        else if( token.find( '[' ) != std::string::npos || token.find( ']' ) != std::string::npos )
        {
            if( token.size() < 3 || token.front() != '[' || token.back() != ']' )
            {
                return false;
            }

            const auto content = token.substr( 1, token.size() - 2 );
            const auto slash   = content.find( '/' );

            if( slash != std::string::npos )
            {
                if( haveCounts || !number( expected, content.substr( 0, slash ) ) ||
                    !number( active, content.substr( slash + 1 ) ) || expected == 0 || expected > countLimit ||
                    active > expected )
                {
                    return false;
                }

                haveCounts = true;
            }
            else
            {
                if( !bitmap.empty() || content.find_first_not_of( "U_" ) != std::string::npos )
                {
                    return false;
                }

                bitmap = content;
            }
        }
    }

    if( haveCounts != !bitmap.empty() ||
        ( haveCounts && ( bitmap.size() != expected || std::count( bitmap.begin(), bitmap.end(), 'U' ) != active ||
                          active > activeMembers ) ) )
    {
        return false;
    }

    if( !bitmap.empty() )
    {
        result.m_members += " [" + bitmap + "]";
    }

    const bool redundant = result.m_level == "raid1" || result.m_level == "raid4" || result.m_level == "raid5" ||
                           result.m_level == "raid6" || result.m_level == "raid10";
    const bool container = state == "inactive" && ( metadata == "external:imsm" || metadata == "external:ddf" );
    auto      &sample    = result.m_sample;

    if( ( redundant && !haveCounts ) ||
        ( haveCounts && ( state == "inactive" || result.m_level == "raid0" || result.m_level == "linear" ) ) )
    {
        return false;
    }

    if( state == "inactive" )
    {
        sample.m_flags  = telem::Valid | ( container ? telem::Container | telem::OK : telem::Inactive );
        result.m_status = container ? "metadata container" : "inactive";
    }
    else if( state == "broken" )
    {
        sample.m_flags  = telem::Valid | telem::Faulty;
        result.m_status = "broken";
    }
    else if( redundant )
    {
        sample.m_flags    = telem::Valid | telem::Redundant;
        sample.m_expected = expected;
        sample.m_active   = active;

        if( active < expected )
        {
            sample.m_flags |= telem::Degraded;
            result.m_status = "degraded";
        }
        else
        {
            sample.m_flags |= telem::OK;
            result.m_status = "ok";
        }
    }
    else if( result.m_level == "raid0" || result.m_level == "linear" )
    {
        sample.m_flags  = telem::Valid | telem::OK;
        result.m_status = "active; non-redundant";
    }
    else
    {
        sample.m_flags  = 0;
        result.m_status = "unsupported RAID personality: " + result.m_level;
    }

    if( faulty )
    {
        if( sample.m_flags & telem::Valid )
        {
            sample.m_flags = ( sample.m_flags | telem::Faulty ) & ~telem::OK;
        }

        result.m_status = result.m_status == "ok" ? "faulty member" : result.m_status + "; faulty member";
    }

    if( !readOnly.empty() )
    {
        result.m_status += " " + readOnly;
    }

    if( !action.empty() )
    {
        result.m_status += "; " + action;
    }

    return true;
}

} // namespace raidDetail

inline raidSnapshotT parseMdstat( const std::string &text, double checkTime )
{
    raidSnapshotT result;
    result.m_checkTime = checkTime;
    result.m_status    = "malformed or incomplete mdstat";

    if( text.size() > raidDetail::textLimit || text.find( '\0' ) != std::string::npos )
    {
        return result;
    }

    std::istringstream                    input( text );
    std::string                           line;
    std::vector<std::vector<std::string>> block;
    std::set<std::string>                 names;
    bool                                  header = false, footer = false;

    while( std::getline( input, line ) )
    {
        auto tokens = raidDetail::words( line );

        if( tokens.empty() )
        {
            continue;
        }

        if( footer )
        {
            return result;
        }

        if( !header )
        {
            if( tokens.size() < 2 || tokens[0] != "Personalities" || tokens[1] != ":" )
            {
                return result;
            }

            for( size_t n = 2; n < tokens.size(); ++n )
            {
                if( tokens[n].size() < 3 || tokens[n].front() != '[' || tokens[n].back() != ']' ||
                    !raidDetail::identifier( tokens[n].substr( 1, tokens[n].size() - 2 ) ) )
                {
                    return result;
                }
            }

            header = true;
            continue;
        }

        const bool end  = tokens.size() >= 2 && tokens[0] == "unused" && tokens[1] == "devices:";
        const bool next = tokens.size() >= 2 && tokens[1] == ":";

        if( end || next )
        {
            if( !block.empty() )
            {
                raidArrayT array;

                if( !raidDetail::array( array, block ) || !names.insert( array.m_name ).second || names.size() > 4096 )
                {
                    return result;
                }

                result.m_arrays.push_back( std::move( array ) );
                block.clear();
            }

            if( end )
            {
                if( tokens.size() < 3 || ( tokens[2] == "<none>" && tokens.size() != 3 ) )
                {
                    return result;
                }

                for( size_t n = 2; n < tokens.size(); ++n )
                {
                    if( ( tokens[n] == "<none>" && tokens.size() != 3 ) ||
                        ( tokens[n] != "<none>" && !raidDetail::identifier( tokens[n] ) ) )
                    {
                        return result;
                    }
                }

                footer = true;
                continue;
            }
        }
        else if( block.empty() || ( line[0] != ' ' && line[0] != '\t' ) )
        {
            return result;
        }

        block.push_back( std::move( tokens ) );
    }

    if( !header || !footer )
    {
        return result;
    }

    std::sort( result.m_arrays.begin(),
               result.m_arrays.end(),
               []( const auto &a, const auto &b )
               {
                   return a.m_name < b.m_name;
               } );
    result.m_result = logger::telem_raid::Ready;
    result.m_status = result.m_arrays.empty() ? "no arrays" : "ok";
    return result;
}

inline raidSnapshotT checkRAID( const std::string &path, double checkTime )
{
    raidSnapshotT result;
    result.m_checkTime = checkTime;
    result.m_result    = logger::telem_raid::ReadError;
    result.m_status    = "cannot read mdstat";
    const int fd       = open( path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK );

    if( fd < 0 )
    {
        if( errno == ENOENT )
        {
            result.m_result = logger::telem_raid::NotPresent;
            result.m_status = "mdstat not present";
        }

        return result;
    }

    std::string text;
    char        buffer[4096];

    for( ;; )
    {
        const auto count = read( fd, buffer, sizeof( buffer ) );

        if( count < 0 && errno == EINTR )
        {
            continue;
        }

        if( count < 0 || static_cast<size_t>( count ) > raidDetail::textLimit - text.size() )
        {
            close( fd );
            return result;
        }

        if( count == 0 )
        {
            close( fd );
            return parseMdstat( text, checkTime );
        }

        text.append( buffer, count );
    }
}

} // namespace sysMonitorDetail
} // namespace app
} // namespace MagAOX
#endif // sysMonitor_raidMonitor_hpp
