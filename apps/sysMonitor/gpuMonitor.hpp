/** \file gpuMonitor.hpp
 * \brief Parse identified GPU query snapshots without accessing hardware.
 * \ingroup sysMonitor_files
 */
#ifndef sysMonitor_gpuMonitor_hpp
#define sysMonitor_gpuMonitor_hpp

#include <algorithm>
#include <array>
#include <charconv>
#include <set>
#include <string_view>

#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/logger/types/telem_gpu.hpp"

namespace MagAOX
{
namespace app
{
namespace sysMonitorDetail
{

/// Return a canonical internal UUID key, or an empty string for an invalid physical GPU UUID.
/** Accepts GPU- followed by hexadecimal groups of lengths 8,4,4,4,12; hexadecimal case is normalized. */
inline std::string gpuKey( const std::string &uuid /**< [in] physical NVIDIA GPU UUID */ )
{
    if( uuid.size() != 40 || uuid.substr( 0, 4 ) != "GPU-" )
    {
        return {};
    }

    std::string key = "gpu_";

    for( size_t n = 4; n < uuid.size(); ++n )
    {
        const char ch = uuid[n];

        if( n == 12 || n == 17 || n == 22 || n == 27 )
        {
            if( ch != '-' )
            {
                return {};
            }

            key += '_';
        }
        else if( ( ch >= '0' && ch <= '9' ) || ( ch >= 'a' && ch <= 'f' ) )
        {
            key += ch;
        }
        else if( ch >= 'A' && ch <= 'F' )
        {
            key += char( ch - 'A' + 'a' );
        }
        else
        {
            return {};
        }
    }

    return key;
}

/// \cond SYSMONITOR_GPU_PARSER_INTERNALS
namespace gpuDetail
{

/// Trim the ASCII whitespace emitted around CSV fields.
inline std::string_view trim( std::string_view value /**< [in] field or line to trim */ )
{
    const auto first = value.find_first_not_of( " \t\r\n" );

    if( first == std::string_view::npos )
    {
        return {};
    }

    return value.substr( first, value.find_last_not_of( " \t\r\n" ) - first + 1 );
}

/// Validate and lowercase a 4- or 8-digit domain PCI address.
inline bool pciAddress( std::string &address /**< [in,out] address to validate and normalize */ )
{
    if( address.size() != 12 && address.size() != 16 )
    {
        return false;
    }

    const size_t domain = address.size() - 8;

    for( size_t n = 0; n < address.size(); ++n )
    {
        char &ch = address[n];

        if( n == domain || n == domain + 3 )
        {
            if( ch != ':' )
            {
                return false;
            }
        }
        else if( n == domain + 6 )
        {
            if( ch != '.' )
            {
                return false;
            }
        }
        else if( ch >= 'A' && ch <= 'F' )
        {
            ch = char( ch - 'A' + 'a' );
        }
        else if( !( ( ch >= '0' && ch <= '9' ) || ( ch >= 'a' && ch <= 'f' ) ) )
        {
            return false;
        }
    }

    return address.back() >= '0' && address.back() <= '7' &&
           ( address[domain + 4] == '0' || address[domain + 4] == '1' );
}

/// Parse a nonnegative finite number or an explicitly unavailable field.
/** Returns 0 for unavailable fields, 1 for measurements, and -1 for malformed input. */
inline int number( double          &value /**< [out] parsed measurement, unchanged when unavailable */,
                   std::string_view field /**< [in] trimmed CSV field */ )
{
    if( field == "N/A" || field == "[N/A]" || field == "Not Supported" || field == "[Not Supported]" )
    {
        return 0;
    }

    double parsed     = 0;
    auto   conversion = std::from_chars( field.data(), field.data() + field.size(), parsed );

    if( conversion.ec != std::errc{} || conversion.ptr != field.data() + field.size() ||
        !mx::math::isFinite( parsed ) || parsed < 0 )
    {
        return -1;
    }

    value = parsed;
    return 1;
}

} // namespace gpuDetail
/// \endcond

/// Parse an entire GPU CSV response atomically, ordered by stable UUID.
/** Columns are NVIDIA index, UUID, PCI address, Celsius, GPU %, memory activity %, used MiB, total MiB,
 * graphics MHz, SM MHz, and memory MHz. Unsupported fields clear their validity bit.
 * Indices must be decimal unsigned integers smaller than telem_gpu::UnknownIndex.
 * Malformed rows or duplicate UUIDs/indices return -1 and leave the previous output untouched.
 * Empty output is a successful empty inventory. Sample times are Unix UTC seconds.
 */
inline int parseGPUs( std::vector<logger::telem_gpu::gpuT> &out /**< [out] parsed device snapshot on success */,
                      const std::string                    &csv /**< [in] complete unquoted nvidia-smi CSV response */,
                      double sampleTime /**< [in] acquisition time in Unix UTC seconds */ )
{
    if( !mx::math::isFinite( sampleTime ) || sampleTime < 0 )
    {
        return -1;
    }

    using gpuT = logger::telem_gpu::gpuT;
    std::vector<gpuT>     parsed;
    std::set<std::string> keys;
    std::set<uint32_t>    indices;
    size_t                start = 0;

    while( start < csv.size() )
    {
        auto end = csv.find( '\n', start );

        if( end == std::string::npos )
        {
            end = csv.size();
        }

        auto line = gpuDetail::trim( std::string_view( csv ).substr( start, end - start ) );
        start     = end + 1;

        if( line.empty() )
        {
            continue;
        }

        std::array<std::string_view, 11> fields;
        size_t                           position = 0;

        for( size_t n = 0; n < fields.size(); ++n )
        {
            auto comma = line.find( ',', position );

            if( ( n + 1 == fields.size() ) != ( comma == std::string_view::npos ) )
            {
                return -1;
            }

            fields[n] = gpuDetail::trim( line.substr( position, comma - position ) );

            if( fields[n].empty() )
            {
                return -1;
            }

            position = comma + 1;
        }

        gpuT       gpu;
        const auto index = std::from_chars( fields[0].data(), fields[0].data() + fields[0].size(), gpu.m_index );

        if( index.ec != std::errc{} || index.ptr != fields[0].data() + fields[0].size() ||
            gpu.m_index == logger::telem_gpu::UnknownIndex || !indices.insert( gpu.m_index ).second )
        {
            return -1;
        }

        const auto key = gpuKey( std::string( fields[1] ) );

        if( key.empty() || !keys.insert( key ).second )
        {
            return -1;
        }

        gpu.m_uuid = "GPU-" + key.substr( 4 );
        std::replace( gpu.m_uuid.begin(), gpu.m_uuid.end(), '_', '-' );
        gpu.m_pci = fields[2];

        if( !gpuDetail::pciAddress( gpu.m_pci ) )
        {
            return -1;
        }

        const std::array<double gpuT::*, 8> members{ &gpuT::m_temperature,
                                                     &gpuT::m_load,
                                                     &gpuT::m_memoryLoad,
                                                     &gpuT::m_memoryUsed,
                                                     &gpuT::m_memoryTotal,
                                                     &gpuT::m_clockGraphics,
                                                     &gpuT::m_clockSM,
                                                     &gpuT::m_clockMemory };

        for( size_t n = 0; n < members.size(); ++n )
        {
            int result = gpuDetail::number( gpu.*members[n], fields[n + 3] );

            if( result < 0 )
            {
                return -1;
            }

            if( result > 0 )
            {
                gpu.m_valid |= 1U << n;
            }
        }

        if( gpu.m_load > 100 || gpu.m_memoryLoad > 100 ||
            ( ( gpu.m_valid & logger::telem_gpu::MemoryTotal ) && gpu.m_memoryTotal == 0 ) ||
            ( ( gpu.m_valid & logger::telem_gpu::MemoryUsed ) && ( gpu.m_valid & logger::telem_gpu::MemoryTotal ) &&
              gpu.m_memoryUsed > gpu.m_memoryTotal ) )
        {
            return -1;
        }

        gpu.m_load /= 100;
        gpu.m_memoryLoad /= 100;
        gpu.m_status     = gpu.m_valid == logger::telem_gpu::AllValid ? "ok" : "partial";
        gpu.m_sampleTime = sampleTime;
        parsed.push_back( std::move( gpu ) );
    }

    std::sort( parsed.begin(),
               parsed.end(),
               []( const gpuT &a, const gpuT &b )
               {
                   return a.m_uuid < b.m_uuid;
               } );
    out = std::move( parsed );
    return 0;
}

} // namespace sysMonitorDetail
} // namespace app
} // namespace MagAOX

#endif // sysMonitor_gpuMonitor_hpp
