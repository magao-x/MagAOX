/// Config.cpp
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
#include "Config.hpp"

using std::string;
using std::map;
using pcf::IndiElement;
using pcf::IndiProperty;
using pcf::Config;
using pcf::ConfigFile;

////////////////////////////////////////////////////////////////////////////////
/// This is the static member we are using to write to the log file.

pcf::ConfigFile pcf::Config::sm_cfSettings;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - sets the Severity to a default of 'info'.

Config::Config()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

Config::~Config()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

Config::Config( const Config &copy )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

const Config &Config::operator=( const Config &copy )
{
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the path & filename, then read the file.
/// @param szPath Sets the path to be read from.
/// @param szFilename Sets the filename.

int Config::init( const string &szPath,
                  const string &szFilename,
                  const bool &oLogMode )
{
  return sm_cfSettings.init( szPath, szFilename, oLogMode );
}

////////////////////////////////////////////////////////////////////////////////
/// (re) Read the config file.

int Config::readFile()
{
  return sm_cfSettings.readFile();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the path the config file is on.

std::string Config::getPath()
{
  return sm_cfSettings.getPath();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the filename of the config file.

std::string Config::getFilename()
{
  return sm_cfSettings.getFilename();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether we should log the settings when the 'get' methods are used.

bool Config::isLogModeEnabled()
{
  return sm_cfSettings.isLogModeEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets whether we should log the settings when the 'get' methods are used.

void Config::enableLogMode( const bool &oEnable )
{
  sm_cfSettings.enableLogMode( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Return the message associated with the error.

string Config::getErrorMsg( const int &nErr )
{
  return ConfigFile::getErrorMsg( nErr );
}

////////////////////////////////////////////////////////////////////////////////
/// Get a named entry then convert into an INDI property. If the entry
/// named "szName" does not exist, the value "szDefault" will be used.
/// If the name contains a ".", this will be considered a Section-name separator,
/// and the property name will be set to the part before the ".", and the element name
/// name will be set to the part after the dot.

IndiProperty Config::get( const IndiProperty::Type &tType,
                          const string &szDevice,
                          const string &szName,
                          const string &szDefault )
{
  string szSectionName;
  string szElementName;

  // Pull out the section (property name) and element names.
  size_t tPos = szName.find( '.' );
  if ( tPos != string::npos )
  {
    szSectionName = szName.substr( 0, tPos );
    szElementName = szName.substr( tPos+1, szName.length() );
  }

  // Make sure these names are valid.
  if ( szSectionName.length() == 0 )
  {
    szSectionName = szName;
  }
  if ( szElementName.length() == 0 )
  {
    szElementName = "value";
  }

  // Finally create the property.
  IndiProperty propReturn( tType, szDevice, szSectionName );
  propReturn.add( IndiElement( szElementName,
      sm_cfSettings.get<string>( szName, szDefault ) ) );

  return propReturn;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a collection of entries named "szSection". If there are no entries
/// in this section, or it doesn't exist, the property will not have any
/// elements.

IndiProperty Config::get( const IndiProperty::Type &tType,
                          const string &szDevice,
                          const string &szSection )
{
  IndiProperty propReturn( tType, szDevice, szSection,
                           IndiProperty::Ok, IndiProperty::ReadOnly );

  map<string, string>::iterator itr = sm_cfSettings.getVars().begin();
  map<string, string>::iterator itrEnd = sm_cfSettings.getVars().end();

  string szName;
  string szSectionName;
  string szElementName;
  for ( ; itr != itrEnd; ++itr )
  {
    szName = itr->first;
    // Do we have a section name?
    size_t tPos = szName.find( '.' );
    if ( tPos != string::npos )
    {
      szSectionName = szName.substr( 0, tPos );
      // Does the section name match the one we are looking for?
      if ( szSectionName == szSection )
      {
        szElementName = szName.substr( tPos+1, szName.length() );
        // There are special element names which are actually ATTRIBUTE names.
        if ( szElementName == "label" )
        {
          propReturn.setLabel( sm_cfSettings.get<string>( szName, "" ) );
        }
        else
        {
          propReturn.add( IndiElement( szElementName,
              sm_cfSettings.get<string>( szName, "" ) ) );
        }
      }
    }
  }

  // If we do not have a label attribute that has been specified,
  // make the un-modified name the label.
  if ( propReturn.getLabel().length() == 0 )
  {
    propReturn.setLabel( szSectionName );
  }

  return propReturn;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a collection of entries named "szSection". If there are no entries
/// in this section, or it doesn't exist, the property will have just the
/// default elements & vales.

IndiProperty Config::get( const string &szSection,
                          IndiProperty &propDefault )
{
  map<string, string>::iterator itr = sm_cfSettings.getVars().begin();
  map<string, string>::iterator itrEnd = sm_cfSettings.getVars().end();

  // Take the section name as the default name for the property.
  string szPropertyName( szSection );
  string szEntryName;
  string szSectionName;
  string szElementName;
  for ( ; itr != itrEnd; ++itr )
  {
    szEntryName = itr->first;
    // Do we have a section name?
    size_t tPos = szEntryName.find( '.' );
    if ( tPos != string::npos )
    {
      szSectionName = szEntryName.substr( 0, tPos );
      // Does the section name match the one we are looking for?
      if ( szSectionName == szSection )
      {
        // This is the name of the entry we want.
        szElementName = szEntryName.substr( tPos+1, szEntryName.length() );
        // There are special element names which are actually ATTRIBUTE names.
        if ( szElementName == "label" )
        {
          propDefault.setLabel( sm_cfSettings.get<string>( szEntryName, "" ) );
        }
        else if ( szElementName == "name" )
        {
          // We actually have a name, so save it for later.
          szPropertyName = sm_cfSettings.get<string>( szEntryName, szSectionName );
          propDefault.setName( IndiProperty::scrubName( szPropertyName ) );
        }
        else
        {
          propDefault.update( IndiElement( szElementName,
              sm_cfSettings.get<string>( szEntryName, "" ) ) );
        }
      }
    }
  }

  // If we do not have a label attribute that has been specified,
  // make the un-modified name the label.
  if ( propDefault.getLabel().length() == 0 )
  {
    propDefault.setLabel( szPropertyName );
  }

  return propDefault;
}

////////////////////////////////////////////////////////////////////////////////
