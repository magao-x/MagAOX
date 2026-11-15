/** \file cpuList.hpp
 * \brief Strict normalization of Linux CPU-list strings.
 * \ingroup sys_files
 */
#ifndef sys_cpuList_hpp
#define sys_cpuList_hpp

#include <string>
#include <vector>

namespace MagAOX
{
namespace sys
{

/// Parse a nonempty CPU list into sorted unique IDs, leaving the output unchanged on failure.
/** Accepts decimal IDs and inclusive ranges separated by commas, with surrounding whitespace.
 * IDs are limited to 1048575 and expansion to 65536 entries to bound malformed input.
 * \ingroup sys
 */
bool parseCpuList( std::vector<unsigned> &cpus /**< [out] normalized CPU IDs on success */,
                   const std::string     &text /**< [in] Linux CPU-list text */ );

/// Format sorted unique CPU IDs as canonical singleton/range entries.
std::string cpuListString( const std::vector<unsigned> &cpus /**< [in] sorted unique CPU IDs */ );

} // namespace sys
} // namespace MagAOX
#endif // sys_cpuList_hpp
