/// Config.hpp
///
///  @author Paul Grenz
///
/// This object supports and interface which writes entries to and from a
/// ConfigFile object. The actual ConfigFile is a static object in this
/// class, meaning that only one instance will be created
/// on a per-process basis. In this way, any number of these "Config" objects
/// can be created, but the same ConfigFile object will be accessed for
/// the whole process.
///
/// Example:
///
///    Config cfgReader;
///    ...
///    int iCount = cfgReader.get<int>( "count", 10 );
///    ...
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_CONFIG_HPP
#define PCF_CONFIG_HPP

#include <string>
#include "IndiProperty.hpp"
#include "ConfigFile.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Config
{
  // Constructor/destructor/copy constructor.
  public:
    Config();
    virtual ~Config();

  private:
    Config( const Config &copy );
    const Config &operator=( const Config &copy );

  // Global static methods.
  public:
    /// Sets whether we should log the settings when the 'get' methods are used.
    static void enableLogMode( const bool &oEnable );
    /// Return the message concerning the error.
    static std::string getErrorMsg( const int &nErr );
    /// Return the path the config file is on.
    static std::string getPath();
    /// Return the filename of the config file.
    static std::string getFilename();
    /// Returns whether we should log the settings when the 'get' methods are used.
    static bool isLogModeEnabled();
    /// Set the path & filename and read in the settings in the underlying
    /// config file.
    static int init( const std::string &szPath,
                     const std::string &szFilename,
                     const bool &oLogMode = true );
    /// (re) Read the config file.
    static int readFile();

  // Methods.
  public:
    /// Get a named entry of type TT. If it does not exist, the default will
    /// be returned.
    template <class TT> TT get( const std::string &szFullName,
                                const TT &tDefault ) const;
    /// Get a named entry then convert into an INDI property. If the entry
    /// named "szName" does not exist, the value "szDefault" will be used.
    pcf::IndiProperty get( const IndiProperty::Type &tType,
                           const std::string &szDevice,
                           const std::string &szName,
                           const std::string &szDefault );
    /// Get a collection of entries named "szSection". If there are no entries
    /// in this section, or it doesn't exist, the property will not have any
    /// elements.
    pcf::IndiProperty get( const IndiProperty::Type &tType,
                           const std::string &szDevice,
                           const std::string &szSection );
    /// Get a collection of entries named "szSection". If there are no entries
    /// in this section, or it doesn't exist, the property will have just the
    /// default elements & values.
    pcf::IndiProperty get( const std::string &szSection,
                           pcf::IndiProperty &propDefault );
    /// Set a named entry of type TT. If it does not exist, it will be added.
    template <class TT> void set( const std::string &szFullName,
                                  const TT &tValue );
    /// This returns the number of entries read out of the file.
    int size() const;

  // Variables.
  private:
    /// This is the underlying config file object this
    /// class uses to actually access the settings.
    static pcf::ConfigFile sm_cfSettings;

}; // Class Config
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////
/// Get a named value of type TT.
/// @param szFullName The name of the entry.
/// @param tDefault The default value if the requested one is not found.
/// @return The templated value to be read.

namespace pcf
{
template <class TT> TT pcf::Config::get( const std::string &szFullName,
                                         const TT &tDefault ) const
{
  return sm_cfSettings.get<TT>( szFullName, tDefault );
}
}
////////////////////////////////////////////////////////////////////////////////
/// Set a named entry of type TT. If it does not exist, it will be added.
/// @param szFullName The name of the entry.
/// @param tValue The value to set the entry to.

namespace pcf
{
template <class TT> void pcf::Config::set( const std::string &szFullName,
                                           const TT &tValue )
{
  sm_cfSettings.set<TT>( szFullName, tValue );
}
}
////////////////////////////////////////////////////////////////////////////////

#endif // PCF_CONFIG_HPP
