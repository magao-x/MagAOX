/// System.hpp
///
/// System utility functions.
///
/// @author Mark Milton
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_SYSTEM_HPP
#define PCF_SYSTEM_HPP

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{

class System
{
  // Prevent these from being invoked.
  private:
    System();
    System( const System &sysRhs );
    const System &operator=( const System &sysRhs );

  // Utility functions
  public:
    /// Creates a path as a string based on an environment variable. If the
    /// environment variable doesn't exist or is empty, the cwd() is returned.
    static std::string createEnvPath( const std::string &szEnvName );
    /// Creates a vector of file names in szPath that match szSuffix and szPrefix.
    static std::vector<std::string> createFileList( const std::string& szPath = "",
                                                    const std::string& szSuffix = "",
                                                    const std::string& szPrefix = "" );
    /// Return formatted filename with current timestamp.
    static std::string createFileNameWithTimeStamp( const std::string& szPrefix = "",
                                                    const std::string& szSuffix = "" );
    /// Check if specified file exists in the file system.
    static bool doesFileExist( const std::string& szFileName );
    /// Ensures that a path exists. If it does not, it is created.
    static std::string makePath( const std::string &szPath );
    /// Ensures that a symlink exists. If it does not, it is created.
    static void makeSymLink( const std::string &szFilename,
                             const std::string &szLinkname );
    /// Sets the cwd and the rlimit to ensure that core files go to
    /// a place specified by the user as well as creating the specified path.
    static void setCoreFilePath( const std::string &szCoreFilePath );

}; // class System

} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_SYSTEM_HPP
