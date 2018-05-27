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
/// If the name contains a ".", this will be considered a group-name separator,
/// and the property name will be set to the part before the ".", and the element name
/// name will be set to the part after the dot.

IndiProperty Config::get( const IndiProperty::Type &tType,
                          const string &szDevice,
                          const string &szName,
                          const string &szDefault )
{
  string szGroupName;
  string szElementName;

  // Pull out the group and element names.
  size_t tPos = szName.find( '.' );
  if ( tPos != string::npos )
  {
    szGroupName = szName.substr( 0, tPos );
    szElementName = szName.substr( tPos+1, szName.length() );
  }

  // Make sure these names are valid.
  if ( szGroupName.length() == 0 )
  {
    szGroupName = szName;
  }
  if ( szElementName.length() == 0 )
  {
    szElementName = "value";
  }

  // Finally create the property.
  IndiProperty propReturn( tType, szDevice, szGroupName );
  propReturn.add( IndiElement( szElementName,
      sm_cfSettings.get<string>( szName, szDefault ) ) );

  return propReturn;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a group of entries named "szGroup". If there are no entries in this
/// group, or it doesn't exist, the group will not have any elements.

IndiProperty Config::get( const IndiProperty::Type &tType,
                          const string &szDevice,
                          const string &szGroup )
{
  IndiProperty propReturn( tType, szDevice, szGroup,
                           IndiProperty::Ok, IndiProperty::ReadOnly );

  map<string, string>::iterator itr = sm_cfSettings.getVars().begin();
  map<string, string>::iterator itrEnd = sm_cfSettings.getVars().end();

  string szName;
  string szGroupName;
  string szElementName;
  for ( ; itr != itrEnd; itr++ )
  {
    szName = itr->first;
    // Do we have a group name?
    size_t tPos = szName.find( '.' );
    if ( tPos != string::npos )
    {
      szGroupName = szName.substr( 0, tPos );
      // Does the group name match the one we are looking for?
      if ( szGroupName == szGroup )
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
  // make the config group name the label.
  if ( propReturn.getLabel().length() == 0 )
  {
    propReturn.setLabel( szGroupName );
  }

  return propReturn;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a group of entries named "szGroup". If there are no entries in this
/// group, or it doesn't exist, the group will have just the default elements.

IndiProperty Config::get( const string &szGroup,
                          IndiProperty &propDefault )
{
  map<string, string>::iterator itr = sm_cfSettings.getVars().begin();
  map<string, string>::iterator itrEnd = sm_cfSettings.getVars().end();

  string szName;
  string szGroupName;
  string szElementName;
  for ( ; itr != itrEnd; itr++ )
  {
    szName = itr->first;
    // Do we have a group name?
    size_t tPos = szName.find( '.' );
    if ( tPos != string::npos )
    {
      szGroupName = szName.substr( 0, tPos );
      // Does the group name match the one we are looking for?
      if ( szGroupName == szGroup )
      {
        szElementName = szName.substr( tPos+1, szName.length() );
        // There are special element names which are actually ATTRIBUTE names.
        if ( szElementName == "label" )
        {
          propDefault.setLabel( sm_cfSettings.get<string>( szName, "" ) );
        }
        else
        {
          propDefault.update( IndiElement( szElementName,
              sm_cfSettings.get<string>( szName, "" ) ) );
        }
      }
    }
  }

  // If we do not have a label attribute that has been specified,
  // make the config group name the label.
  if ( propDefault.getLabel().length() == 0 )
  {
    propDefault.setLabel( szGroupName );
  }

  return propDefault;
}

////////////////////////////////////////////////////////////////////////////////
