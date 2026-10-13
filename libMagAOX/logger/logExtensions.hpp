/** \file logExtensions.hpp
 * \brief Declares how log file extensions and flatlogs versions are linked.
 *
 * \ingroup logger_files
 */

#ifndef logger_logExtensions_hpp
#define logger_logExtensions_hpp

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <mx/ioutils/fileUtils.hpp>

#include <flatlogs/flatlogs.hpp>

#include "../common/defaults.hpp"

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
 * The Python tools use the same rule in log_extensions_for (magaox-python/magaox/utils.py).

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

/// Given a base extension and a version, build the file extension.
/** v1 files are named with \p baseExt, and v2 files with \p baseExt + "2".
 * A base that already ends in "2" is returned unchanged.
 *
 * \returns the extension to use for new files, in the same form as \p baseExt (with or without the leading '.').
 */
inline std::string logExtensionForVersion( const std::string    &baseExt, /**< [in] the base extension */
                                           flatlogs::logVersionT version  /**< [in] the version of the entries */
)
{
    if( version >= flatlogs::logVersion2 && !baseExt.empty() && baseExt.back() != '2' )
    {
        return baseExt + "2";
    }

    return baseExt;
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

/// Get a log file's version from its path.
/** Wrapper for logVersionForExtension
 *
 * \returns flatlogs::logVersion2 if the file's extension ends in "2", flatlogs::logVersion1 otherwise.
 */
inline flatlogs::logVersionT logVersionForFile( const std::string &path /**< [in] the file's path or name */ )
{
    return logVersionForExtension( std::filesystem::path( path ).extension().string() );
}

/// List an app's log files from one directory, for every extension implied by a base extension.
/** The files are listed, merged, and sorted by name (i.e. in time order).
 *
 * \returns mx::error_t::noerror on success
 * \returns errors from mx::ioutils::getFileNames
 */
template <class verboseT = XWC_DEFAULT_VERBOSITY>
mx::error_t listLogFiles( std::vector<std::string> &flist,  /**< [out] the sorted list of file paths */
                          const std::string        &dir,    /**< [in] the directory to list */
                          const std::string        &prefix, /**< [in] the device name, a prefix of the file names */
                          const std::string        &baseExt /**< [in] the base extension */
)
{
    flist.clear();

    for( const std::string &ext : logExtensionsFor( baseExt ) )
    {
        std::vector<std::string> extList;
        mx_error_check( mx::ioutils::getFileNames( extList, dir, prefix, "", ext ) );
        flist.insert( flist.end(), extList.begin(), extList.end() );
    }

    std::sort( flist.begin(), flist.end() );

    return mx::error_t::noerror;
}

} // namespace logger
} // namespace MagAOX

#endif // logger_logExtensions_hpp
