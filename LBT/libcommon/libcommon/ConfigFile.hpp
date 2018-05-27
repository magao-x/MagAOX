/// ConfigFile.hpp
///
/// @author Paul Grenz
///
/// The "ConfigFile" class wraps the interface to a configuration file. the
/// config file must consist of lines which are either blank or of the format:
///
/// Name  Value # optional comment.
///
/// Lines may contain only comments as well (they start with a "#" ).
/// Anything on a line after a "#" will be ignored. If any one line is badly
/// formed, the entire read is aborted with the throwing of an exception,
/// and no values are stored.
///
/// Naming conventions follow the same rules as C-variable naming conventions.
/// A list of valid names is as follows:
///
/// xyz_running 1
/// active true
/// SomeValue 23.89
/// this_is_a_name "hello"
/// engine 15
///
/// To read the values, you can use the templated methods in this class, or use
/// a "Config" object (see the "Config" class for more information). Example:
///
/// int num = cf.get<int>( "engine" );
///
/// num is now 15.
///
/// "Value" may also be delimited with double quotes """ to allow strings with
/// spaces in it. For example:
///
/// xyz.greeting  "hello there - how are you?"
///
/// If you need to show a quote, escape it with a backslash ("\"). You may escape
/// a slash with another slash.
///
/// Update: "groups" are now supported, these are a way of using a short name
/// instead of specifying a longer, full name and allowing related values to
/// be listed together under a common root name. For example:
///
/// [group1]
/// engine_number 23
/// num_valves    6
/// name          "Thomas"
/// [group1]
///
/// In this case, the full names of the entries in this group are:
///
/// group1.engine_number
/// group1.num_valves
/// group1.name
///
/// Note that the group name is separated from the variable name by a ".". The
/// group name must be surrounded by square brackets and must not contain
/// spaces. The group must be closed by listing the group name again at the
/// end of the list of variables, and it must match the first group name.
///
/// The file is read using the "readFile" method which is automatically invoked
/// from any constructor which contains a path & filename as an argument.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_CONFIG_FILE_HPP
#define PCF_CONFIG_FILE_HPP

#include <errno.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <map>
#include <vector>
#include "Logger.hpp"
#include "MutexLock.hpp"

namespace pcf
{
class ConfigFile
{
  public:
    // Error constants.
    enum Error
    {
      /// This is a standard for "no error".
      ErrNone =                        0,
    };

  // Constructor/destructor.
  public:
    ConfigFile();
    ConfigFile( const std::string &szPath,
                const std::string &szFilename,
                const bool &oLogMode = true );
    virtual ~ConfigFile();
    ConfigFile( const ConfigFile &cf );
    const ConfigFile &operator =( const ConfigFile &cf );

  // Methods
  public:
    /// Sets whether we should log the settings when the 'get' methods are used.
    int enableLogMode( const bool &oEnable );
    /// Returns true if this variable is in the map, false otherwise.
    bool find( const std::string &szFullName ) const;
    /// Returns true if this variable is in the map, false otherwise.
    bool find( const std::string &szSection, const std::string &szName ) const;
    /// Return an item of type TT from the list.
    //template <class TT> TT get( const char *pcFullName,
    //                            const TT &tDefault ) const;
    /// Return an item of type TT from the list.
    template <class TT> TT get( const std::string &szFullName,
                                const TT &tDefault ) const;
    /// Return the message concerning the error.
    static std::string getErrorMsg( const int &nErr );
    /// Get a file list from a specific directory.
    /// An extension filter can be specified.
    static std::vector<std::string> getFileList( std::string &szPath,
                                                 std::string &szFilter );
    /// Return the name of the file.
    std::string getFilename() const;
    /// Return the path to the file.
    std::string getPath() const;
    /// Set all at the same time.
    int init( const std::string &szPath,
              const std::string &szFilename,
              const bool &oLogMode = true );
    // This returns a reference to the map of vars. Use sparingly!
    std::map<std::string, std::string> &getVars();
    /// Return whether we should log the settings when the 'get' methods are used.
    bool isLogModeEnabled() const;
    /// Prints all entries to std::out.
    virtual std::string printAll() const;
    /// Load the config file - all entries will be loaded.
    virtual int readFile();
    /// Set a named value of type TT. If it does not exist, it will be added.
    template <class TT> void set( const std::string &szFullName,
                                  const TT &tValue );
    /// This sets the filename to be read.
    int setFilename( const std::string &szFilename );
    /// This sets the path to be read from.
    int setPath( const std::string &szPath );
    /// This returns the number of entries read out of the file.
    int size() const;

  // Variables.
  private:
    /// This is the map of all the entries.
    std::map<std::string, std::string> m_mapVars;
    /// What file have we read?
    std::string m_szFilename;
    /// Should the 'get' methods log the settings when they are called?
    bool m_oLogMode;
    /// What path is this file on?
    std::string m_szPath;
    /// A mutex to protect the configuration data.
    mutable pcf::MutexLock m_mutConfig;

}; // Class ConfigFile
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////
/// Get a named value of type TT.
/// @param pcFullName The name of the variable.
/// @param tDefault The default value if the requested one is not found.
/// @return The templated value to be read.
/*
namespace pcf
{
template <class TT> TT pcf::ConfigFile::get( const char *pcFullName,
                                             const TT &tDefault ) const
{
  TT tValue;
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  //  does this variable exist (exact name match)?
  std::map<std::string, std::string>::const_iterator itr = m_mapVars.end();
  if ( ( itr = m_mapVars.find( string( pcFullName ) ) ) != m_mapVars.end() )
  {
    //  stream the data into the variable.
    std::stringstream ssItem( itr->second );
    ssItem >> std::boolalpha >> tValue;
  }
  //  we know nothing about this variable.
  else
  {
    tValue = tDefault;
  }

  return tValue;
}
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Get a named value of type TT.
/// @param szFullName The name of the variable.
/// @param tDefault The default value if the requested one is not found.
/// @return The templated value to be read.

namespace pcf
{
template <class TT> TT pcf::ConfigFile::get( const std::string &szFullName,
                                             const TT &tDefault ) const
{
  TT tValue;
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  //  does this variable exist (exact name match)?
  std::map<std::string, std::string>::const_iterator itr = m_mapVars.end();
  if ( ( itr = m_mapVars.find( szFullName ) ) != m_mapVars.end() )
  {
    //  stream the data into the variable.
    std::stringstream ssItem( itr->second );
    ssItem >> std::boolalpha >> tValue;
  }
  //  we know nothing about this variable.
  else
  {
    tValue = tDefault;
  }

  // Do we want to log this?
  if ( m_oLogMode == true )
  {
    Logger logMsg;
    logMsg.logSetting( Logger::Info, szFullName, tValue );
  }

  return tValue;
}
}
////////////////////////////////////////////////////////////////////////////////
/// Set a named value of type TT. If it does not exist, it will be added.
/// @param szFullName The name of the variable.
/// @param tValue The value to set the variable to.

namespace pcf
{
template <class TT> void pcf::ConfigFile::set( const std::string &szFullName,
                                               const TT &tValue )
{
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  //  does this variable exist (exact name match)?
  std::map<std::string, std::string>::iterator itr = m_mapVars.end();
  if ( ( itr = m_mapVars.find( szFullName ) ) != m_mapVars.end() )
  {
    //  stream the data into the variable.
    std::stringstream ssItem;
    ssItem.precision( 20 );
    ssItem << /*std::boolalpha <<*/ tValue;
    itr->second = ssItem.str();
  }
  // Add it to the map.
  else
  {
    //  stream the data into the variable.
    std::stringstream ssItem;
    ssItem.precision( 20 );
    ssItem << /*std::boolalpha <<*/ tValue;
    // Avriable will be added if it does not exist.
    m_mapVars[szFullName] = ssItem.str();
  }
}
}
////////////////////////////////////////////////////////////////////////////////

#endif // PCF_CONFIG_FILE_HPP

