/** \file cpuList.cpp
 * \brief Strict normalization of Linux CPU-list strings.
 * \ingroup sys_files
 */
#include "cpuList.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace MagAOX
{
namespace sys
{

bool parseCpuList( std::vector<unsigned> &cpus, const std::string &text )
{
    constexpr unsigned    maxCpu     = 1048575;
    constexpr size_t      maxEntries = 65536;
    std::vector<unsigned> parsed;
    size_t                pos = 0;

    auto skipSpace = [&]()
    {
        while( pos < text.size() && std::isspace( static_cast<unsigned char>( text[pos] ) ) )
        {
            ++pos;
        }
    };

    auto number = [&]( unsigned &value )
    {
        skipSpace();
        const char *first  = text.data() + pos;
        const char *last   = text.data() + text.size();
        auto        result = std::from_chars( first, last, value );

        if( result.ec != std::errc{} || result.ptr == first || value > maxCpu )
        {
            return false;
        }

        pos = result.ptr - text.data();
        skipSpace();
        return true;
    };

    while( true )
    {
        unsigned first = 0;

        if( !number( first ) )
        {
            return false;
        }

        unsigned last = first;

        if( pos < text.size() && text[pos] == '-' )
        {
            ++pos;

            if( !number( last ) || last < first )
            {
                return false;
            }
        }

        if( last - first + 1 > maxEntries - parsed.size() )
        {
            return false;
        }

        for( unsigned cpu = first; cpu <= last; ++cpu )
        {
            parsed.push_back( cpu );
        }

        if( pos == text.size() )
        {
            break;
        }

        if( text[pos++] != ',' )
        {
            return false;
        }
    }

    std::sort( parsed.begin(), parsed.end() );
    parsed.erase( std::unique( parsed.begin(), parsed.end() ), parsed.end() );
    cpus = std::move( parsed );
    return true;
}

std::string cpuListString( const std::vector<unsigned> &cpus )
{
    std::string text;

    for( size_t begin = 0; begin < cpus.size(); )
    {
        size_t end = begin;

        while( end + 1 < cpus.size() && cpus[end + 1] - cpus[end] == 1 )
        {
            ++end;
        }

        if( !text.empty() )
        {
            text += ',';
        }

        text += std::to_string( cpus[begin] );

        if( end > begin )
        {
            text += '-' + std::to_string( cpus[end] );
        }

        begin = end + 1;
    }

    return text;
}

} // namespace sys
} // namespace MagAOX
