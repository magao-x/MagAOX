/** \file logExtensions.hpp
 * \brief Declares how log file extensions and flatlogs versions are linked.
 *
 * \ingroup logger_files
 */

#ifndef logger_logExtensions_hpp
#define logger_logExtensions_hpp

#include <string>
#include <vector>

#include <flatlogs/flatlogs.hpp>

namespace MagAOX
{
namespace logger
{

/// Given a base extension, get the file extensions to search for.
/** \c X is the base extension.
 * Assuming that files named \c X correspond to v1 entries and files named \c X + "2"
 * correspond to v2 entries, both are searched.
 * If \c X already ends in "2", only \c X is searched and the found entries are assumed to be v2.
 * An empty base matches files of any extension.
 *
 * \returns the extensions to search for, matching form of \p baseExt (with or without the leading '.').
 */
inline std::vector<std::string>
logExtensionsFor( const std::string &baseExt /**< [in] the base extension */ )
{
    if( baseExt.empty() || baseExt.back() == '2' )
    {
        return { baseExt };
    }

    return { baseExt, baseExt + "2" };
}

/// Get the version of a log file from its extension.
/** Same assumption as \ref logExtensionsFor: an extension ending in "2" means v2, any other means v1.
 *
 * \returns flatlogs::logVersion2 if \p ext ends in "2", flatlogs::logVersion1 otherwise.
 */
inline flatlogs::logVersionT logVersionForExtension( const std::string &ext /**< [in] the file's extension */ )
{
    if( !ext.empty() && ext.back() == '2' )
    {
        return flatlogs::logVersion2;
    }

    return flatlogs::logVersion1;
}

} // namespace logger
} // namespace MagAOX

#endif // logger_logExtensions_hpp
