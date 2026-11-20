/** \file irqMonitor.hpp
 * \brief Read-only discovery and verification of dedicated IRQ placement.
 * \ingroup sysMonitor_files
 */
#ifndef sysMonitor_irqMonitor_hpp
#define sysMonitor_irqMonitor_hpp

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "../../libMagAOX/sys/cpuList.hpp"
#include "../../libMagAOX/logger/types/telem_irq.hpp"

namespace MagAOX
{
namespace app
{
namespace sysMonitorDetail
{

/// Stable device matching and expected CPU placement for one IRQ group.
struct ruleT
{
    std::string m_name;                     ///< Unique operator-facing group name.
    std::string m_match;                    ///< Literal substring of an IRQ action name, never a numeric IRQ.
    std::string m_cpuset;                   ///< Expected cpuset name, mutually exclusive with m_cpus.
    std::string m_cpus;                     ///< Explicit expected CPU list, mutually exclusive with m_cpuset.
    bool        m_required{ true };         ///< Warn when the device has no matching IRQ vectors.
    bool        m_requireEffective{ true }; ///< Warn when effective-affinity support is unavailable.
};

/// Validate rule identity, selectors and expected CPU specifications before enabling checks.
inline bool validateIRQRules( const std::vector<ruleT> &rules /**< [in] configured IRQ groups */,
                              std::string              &error /**< [out] error description, empty on success */ );

/// Collect fresh IRQ identities and affinity, using only files beneath the supplied roots.
inline std::vector<logger::telem_irq::groupT>
checkIRQs( const std::vector<ruleT> &rules /**< [in] configured logical device groups */,
           const std::string        &procRoot /**< [in] proc directory, normally /proc */,
           const std::string        &cpusetRoot /**< [in] configured cpuset mount directory */,
           double                    checkTime /**< [in] Unix time of this collection attempt */ );

namespace irqDetail
{

/// File-read result distinguishes absent optional interfaces from other I/O errors.
enum class readResult
{
    success,
    missing,
    error
};

/// Read bounded diagnostic text without following a shell command path.
inline readResult readText( std::string       &text /**< [out] file contents */,
                            const std::string &path /**< [in] diagnostic file path */,
                            size_t             limit /**< [in] maximum permitted bytes */ );

/// Test a nonempty decimal token without imposing a counter-width limit.
inline bool decimal( const std::string &token /**< [in] token to inspect */ );

/// Parse numeric IRQ rows and retain only their action-name fields.
inline bool parseInterrupts( std::map<unsigned, std::string> &actions /**< [out] IRQ numbers and action names */,
                             const std::string               &text /**< [in] complete /proc/interrupts contents */ );

/// Append one explicit verification failure and raise this group's warning.
inline void fail( logger::telem_irq::groupT &group /**< [in,out] check result */,
                  const std::string         &reason /**< [in] failure description */ );

inline readResult readText( std::string &text, const std::string &path, size_t limit )
{
    text.clear();
    std::ifstream file( path );

    if( !file )
    {
        std::error_code error;
        bool            exists = std::filesystem::exists( path, error );
        return !exists && !error ? readResult::missing : readResult::error;
    }

    char buffer[4096];

    while( file )
    {
        file.read( buffer, sizeof( buffer ) );
        size_t count = file.gcount();

        if( count > limit - text.size() )
        {
            return readResult::error;
        }

        text.append( buffer, count );
    }

    return file.eof() && !file.bad() ? readResult::success : readResult::error;
}

inline bool decimal( const std::string &token )
{
    return !token.empty() && token.find_first_not_of( "0123456789" ) == std::string::npos;
}

inline bool parseInterrupts( std::map<unsigned, std::string> &actions, const std::string &text )
{
    actions.clear();
    std::istringstream input( text );
    std::string        line;
    size_t             cpuCount = 0;

    if( !std::getline( input, line ) )
    {
        return false;
    }

    std::istringstream header( line );
    std::string        token;

    while( header >> token )
    {
        if( token.compare( 0, 3, "CPU" ) != 0 || !decimal( token.substr( 3 ) ) )
        {
            return false;
        }

        ++cpuCount;
    }

    if( cpuCount == 0 )
    {
        return false;
    }

    while( std::getline( input, line ) )
    {
        std::istringstream row( line );

        if( !( row >> token ) || token.back() != ':' )
        {
            continue;
        }

        token.pop_back();

        if( !decimal( token ) )
        {
            continue;
        }

        unsigned number     = 0;
        auto     conversion = std::from_chars( token.data(), token.data() + token.size(), number );

        if( conversion.ec != std::errc{} || !actions.emplace( number, "" ).second )
        {
            return false;
        }

        for( size_t cpu = 0; cpu < cpuCount; ++cpu )
        {
            if( !( row >> token ) || !decimal( token ) )
            {
                return false;
            }
        }

        // Chip/controller metadata is not an action name, even if it matches a selector.
        if( !( row >> token ) )
        {
            return false;
        }

        if( !( row >> token ) )
        {
            continue;
        }

        // Linux may print a hardware number, a joined number/flow name, and/or a level marker.
        size_t dash = token.find( '-' );

        if( decimal( token ) || ( dash != std::string::npos && decimal( token.substr( 0, dash ) ) ) )
        {
            if( !( row >> token ) )
            {
                continue;
            }
        }

        if( token == "Edge" || token == "Level" || token.compare( 0, 5, "Edge-" ) == 0 ||
            token.compare( 0, 6, "Level-" ) == 0 || token.front() == '-' )
        {
            if( !( row >> token ) )
            {
                continue;
            }
        }

        std::string remaining;
        std::getline( row, remaining );
        actions[number] = token + remaining;
    }

    return true;
}

inline void fail( logger::telem_irq::groupT &group, const std::string &reason )
{
    if( !group.m_status.empty() )
    {
        group.m_status += "; ";
    }

    group.m_status += reason;
    group.m_warning = true;
}

} // namespace irqDetail

inline bool validateIRQRules( const std::vector<ruleT> &rules, std::string &error )
{
    error.clear();
    std::set<std::string> names;

    for( const auto &rule : rules )
    {
        std::vector<unsigned> cpus;

        if( rule.m_name.empty() || rule.m_name == "summary" ||
            rule.m_name.find_first_not_of( "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_" ) !=
                std::string::npos ||
            !names.insert( rule.m_name ).second || rule.m_match.find_first_not_of( " \t\r\n" ) == std::string::npos ||
            rule.m_match.find_first_of( "\r\n" ) != std::string::npos || rule.m_cpuset.empty() == rule.m_cpus.empty() ||
            ( !rule.m_cpuset.empty() &&
              ( rule.m_cpuset == "." || rule.m_cpuset == ".." || rule.m_cpuset.find( '/' ) != std::string::npos ) ) ||
            ( !rule.m_cpus.empty() && !sys::parseCpuList( cpus, rule.m_cpus ) ) )
        {
            error = "invalid IRQ rule: " + rule.m_name;
            return false;
        }
    }

    return true;
}

inline std::vector<logger::telem_irq::groupT> checkIRQs( const std::vector<ruleT> &rules,
                                                         const std::string        &procRoot,
                                                         const std::string        &cpusetRoot,
                                                         double                    checkTime )
{
    std::vector<logger::telem_irq::groupT> groups;
    std::string                            configError;
    bool                                   validConfig = validateIRQRules( rules, configError );
    std::string                            interrupts;
    std::map<unsigned, std::string>        actions;
    bool                                   validInterrupts =
        rules.empty() || ( irqDetail::readText( interrupts, procRoot + "/interrupts", 16 * 1024 * 1024 ) ==
                               irqDetail::readResult::success &&
                           irqDetail::parseInterrupts( actions, interrupts ) );

    for( const auto &rule : rules )
    {
        logger::telem_irq::groupT group;
        group.m_name      = rule.m_name;
        group.m_checkTime = checkTime;
        group.m_warning   = false;

        if( !validConfig )
        {
            irqDetail::fail( group, configError );
            groups.push_back( std::move( group ) );
            continue;
        }

        if( !validInterrupts )
        {
            irqDetail::fail( group, "interrupt discovery unavailable" );
            groups.push_back( std::move( group ) );
            continue;
        }

        std::vector<unsigned> expected;
        std::string           expectedText = rule.m_cpus;
        bool                  expectedRead =
            rule.m_cpuset.empty() ||
            irqDetail::readText( expectedText, cpusetRoot + "/" + rule.m_cpuset + "/cpuset.cpus", 1024 * 1024 ) ==
                irqDetail::readResult::success;
        bool expectedValid = expectedRead && sys::parseCpuList( expected, expectedText );

        if( expectedValid )
        {
            group.m_expected = sys::cpuListString( expected );
        }

        bool effectiveUnverified = false;

        for( const auto &[number, action] : actions )
        {
            if( action.find( rule.m_match ) == std::string::npos )
            {
                continue;
            }

            std::string prefix  = "IRQ" + std::to_string( number ) + ": ";
            size_t      matches = std::count_if( rules.begin(),
                                            rules.end(),
                                            [&]( const ruleT &candidate )
                                            {
                                                return action.find( candidate.m_match ) != std::string::npos;
                                            } );

            if( matches > 1 )
            {
                irqDetail::fail( group, prefix + "overlapping rules" );
            }

            logger::telem_irq::irqT irq;
            irq.m_number               = number;
            std::string           base = procRoot + "/irq/" + std::to_string( number ) + "/";
            std::vector<unsigned> requested;
            auto requestedRead = irqDetail::readText( irq.m_requested, base + "smp_affinity_list", 1024 * 1024 );

            if( requestedRead != irqDetail::readResult::success || !sys::parseCpuList( requested, irq.m_requested ) )
            {
                irqDetail::fail( group, prefix + "requested affinity unavailable" );
            }
            else
            {
                irq.m_requested = sys::cpuListString( requested );

                if( expectedValid && requested != expected )
                {
                    irqDetail::fail( group, prefix + "requested affinity mismatch" );
                }
            }

            std::vector<unsigned> effective;
            auto effectiveRead = irqDetail::readText( irq.m_effective, base + "effective_affinity_list", 1024 * 1024 );

            if( effectiveRead == irqDetail::readResult::missing )
            {
                effectiveUnverified = true;

                if( rule.m_requireEffective )
                {
                    irqDetail::fail( group, prefix + "effective affinity unverified" );
                }
                else
                {
                    // A disappearing IRQ directory must not masquerade as an unsupported interface.
                    std::string           confirmedText;
                    std::vector<unsigned> confirmed;

                    if( irqDetail::readText( confirmedText, base + "smp_affinity_list", 1024 * 1024 ) !=
                            irqDetail::readResult::success ||
                        !sys::parseCpuList( confirmed, confirmedText ) || confirmed != requested )
                    {
                        irqDetail::fail( group, prefix + "requested affinity changed or unavailable during check" );
                    }
                }
            }
            else if( effectiveRead != irqDetail::readResult::success ||
                     !sys::parseCpuList( effective, irq.m_effective ) )
            {
                irqDetail::fail( group, prefix + "effective affinity unavailable" );
            }
            else
            {
                irq.m_effective = sys::cpuListString( effective );

                if( expectedValid &&
                    !std::includes( expected.begin(), expected.end(), effective.begin(), effective.end() ) )
                {
                    irqDetail::fail( group, prefix + "effective affinity mismatch" );
                }
            }

            group.m_irqs.push_back( std::move( irq ) );
        }

        if( group.m_irqs.empty() )
        {
            group.m_status  = rule.m_required ? "required device missing" : "optional device absent";
            group.m_warning = rule.m_required;
        }
        else if( !expectedValid )
        {
            irqDetail::fail( group, "expected CPU list unavailable" );
        }

        if( group.m_status.empty() )
        {
            group.m_status = effectiveUnverified ? "requested affinity verified; effective affinity unverified" : "ok";
        }

        groups.push_back( std::move( group ) );
    }

    std::sort( groups.begin(),
               groups.end(),
               []( const auto &left, const auto &right )
               {
                   return left.m_name < right.m_name;
               } );

    return groups;
}

} // namespace sysMonitorDetail
} // namespace app
} // namespace MagAOX
#endif // sysMonitor_irqMonitor_hpp
