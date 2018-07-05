/// $Id: ConfigFile.cpp 7408 2009-12-16 18:16:26Z pgrenz $
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>
#include <dirent.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Cmd.hpp"
#include "ConfigFile.hpp"

using std::runtime_error;
using std::string;
using std::map;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::vector;
using pcf::Cmd;
using pcf::Logger;
using pcf::ConfigFile;
using pcf::MutexLock;

////////////////////////////////////////////////////////////////////////////////
/// constructor

ConfigFile::ConfigFile()
{
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

ConfigFile::ConfigFile( const string &szPath,
                        const string &szFilename,
                        const bool &oLogMode )
{
  init( szPath, szFilename, oLogMode );
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

ConfigFile::~ConfigFile()
{
}

////////////////////////////////////////////////////////////////////////////////
///  copy constructor.

ConfigFile::ConfigFile( const ConfigFile &cf )
{
  m_mapVars = cf.m_mapVars;
  m_szPath = cf.m_szPath;
  m_szFilename = cf.m_szFilename;
  m_oLogMode = cf.m_oLogMode;
}

////////////////////////////////////////////////////////////////////////////////
///  assignment operator.

const ConfigFile &ConfigFile::operator=( const ConfigFile &cf )
{
  if ( this != &cf )
  {
    m_mapVars = cf.m_mapVars;
    m_szPath = cf.m_szPath;
    m_szFilename = cf.m_szFilename;
    m_oLogMode = cf.m_oLogMode;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
///  initialize the class with the filename.

int ConfigFile::init( const string &szPath,
                      const string &szFilename,
                      const bool &oLogMode )
{
  m_szPath = szPath;
  m_szFilename = szFilename;
  m_oLogMode = oLogMode;

  // Make sure there is something in the path.
  m_szPath = ( m_szPath.length() == 0 ) ? ( "." ) : ( m_szPath );

  //  create the map.
  return readFile();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the path to the file.

int ConfigFile::setPath( const string &szPath )
{
  return init( szPath, m_szFilename, m_oLogMode );
}

////////////////////////////////////////////////////////////////////////////////
/// Return the path to the file.

string ConfigFile::getPath() const
{
  return m_szPath;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the name of the file.

int ConfigFile::setFilename( const string &szFilename )
{
  return init( m_szPath, szFilename, m_oLogMode );
}

////////////////////////////////////////////////////////////////////////////////
/// Return the name of the file.

string ConfigFile::getFilename() const
{
  return m_szFilename;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets whether we should log the settings when the 'get' methods are used.

int ConfigFile::enableLogMode(const bool &oEnable )
{
  return init( m_szPath, m_szFilename, oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether we should log the settings when the 'get' methods are used.

bool ConfigFile::isLogModeEnabled() const
{
  return m_oLogMode;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the number of entries.

int ConfigFile::size() const
{
  return m_mapVars.size();
}

////////////////////////////////////////////////////////////////////////////////
///  return the message concerning the error.

string ConfigFile::getErrorMsg( const int &nErr )
{
  return string( "No Error" );
}

////////////////////////////////////////////////////////////////////////////////
// This returns a reference to the map of vars. Use sparingly!

map<string, string> &ConfigFile::getVars()
{
  return m_mapVars;
}

////////////////////////////////////////////////////////////////////////////////
/// Load the config file. All entries will be loaded. Returns the number of
/// lines processed.

int ConfigFile::readFile()
{
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  string szSection;
  bool oInSection = false;
  char pcLine[1024];
  unsigned int uiCurrLine = 0;
  //  create an object to parse the line.
  vector<string> vecEntry;

  //  clear out the map.
  m_mapVars.clear();

  if ( m_szFilename.empty() )
  {
    throw runtime_error( string( "Config: Filename not specified." ) );
  }

  // Open the file.
  string szFullFilename = m_szPath + "/" + m_szFilename;
  ifstream ifs( szFullFilename.c_str(), ifstream::in );

  if ( ifs.good() != true )
  {
    throw runtime_error( string( "Config: Could not open file '" ) +
        szFullFilename + "'" );
  }

  // for every line....
  while ( ifs.getline( pcLine, 1024 ).fail() == false )
  {
    // We have a line...
    uiCurrLine++;
    stringstream ssCurrLine;
    ssCurrLine << uiCurrLine;

    //  we need to parse the line...
    string szLine = Cmd::mergeWhitespace( string( pcLine ) );

    if ( Cmd::parse( szLine, vecEntry ) != ( unsigned int )( string::npos ) )
    {
      // The only error returned will be an unclosed quote.
      throw runtime_error( string( "Config: Improperly closed quote" ) +
          " (line #" + ssCurrLine.str() + ")" );
    }

    // Do we have a section? This may be an entry with an empty value, though.
    else if ( vecEntry.size() == 1 )
    {
      int nLength = vecEntry[0].length();

      // If the name is less than two chars, it can't be a section.
      // Is it properly surrounded by square braces? If not, we have an
      // empty entry and not a section name.
      if ( nLength <= 2 ||
           vecEntry[0][0] != '[' || vecEntry[0][nLength-1] != ']')
      {
        m_mapVars[ vecEntry[0] ] = "";
      }
      else
      {
        string szNewSection = vecEntry[0];
        szNewSection[0] = szNewSection[nLength-1] = ' ';
        szNewSection = Cmd::trimWhitespace( szNewSection );

        // Do we have a valid name?
        if ( szNewSection.length() == 0 )
        {
          throw runtime_error( string( "Config: Empty section name" ) +
              " (line #" + ssCurrLine.str() + ")" );
        }

        // Is this the start or end of a section?
        oInSection = !oInSection;

        // Make sure the 'section close' matches the 'section open'
        if ( oInSection == false && szNewSection != szSection )
        {
          throw runtime_error( string( "Config: Wrong section name at close" ) +
              " (line #" + ssCurrLine.str() + ")" );
        }

        // This is our current section.
        szSection = szNewSection;
      }
    }

    //  is there a standard section entry (name-value pair)?
    else if ( vecEntry.size() >= 2 && oInSection == true )
    {
      m_mapVars[ szSection + "." + vecEntry[0] ] = vecEntry[1];
    }

    //  is there a standard non-section entry (name-value pair)?
    else if ( vecEntry.size() >= 2 && oInSection == false )
    {
      m_mapVars[ vecEntry[0] ] = vecEntry[1];
    }
  }

  //  now close the file.....
  ifs.close();

  return ErrNone; // uiCurrLine;
}

////////////////////////////////////////////////////////////////////////////////
///  returns true if this variable is in the map, false otherwise.
///  this function does not change the error condition of the class.

bool ConfigFile::find( const string &szFullName ) const
{
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  //  does this variable exist (exact name match)?
  return bool( m_mapVars.find( szFullName ) != m_mapVars.end() );
}

////////////////////////////////////////////////////////////////////////////////
///  prints all entries to one long string.

string ConfigFile::printAll() const
{
  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  stringstream ssList;

  ssList << "[ConfigFile::printAll] <-start-> " << endl;
  map<string, string>::const_iterator itr = m_mapVars.begin();
  for ( ; itr != m_mapVars.end(); itr++ )
  {
    ssList << " name: '" << itr->first << "'"
         << " value: '" << itr->second << "'" << endl;
  }
  ssList << "[ConfigFile::printAll] <-end-> " << endl;

  return ssList.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Get a named value of type string.
/// @param szFullName The name of the variable.
/// @param szDefault The default value if the requested one is not found.
/// @return The templated value to be read.

namespace pcf
{
template<> string ConfigFile::get<string>( const std::string &szFullName,
                                           const string &szDefault ) const
{
  string szValue;

  MutexLock::AutoLock autoConfigLock( &m_mutConfig );

  //  does this variable exist (exact name match)?
  map<string, string>::const_iterator itr = m_mapVars.end();
  if ( ( itr = m_mapVars.find( szFullName ) ) != m_mapVars.end() )
  {
    //  stream the data into the variable.
    szValue = itr->second;
  }
  //  we know nothing about this variable.
  else
  {
    szValue = szDefault;
  }

  // Do we want to log this?
  if ( m_oLogMode == true )
  {
    Logger logMsg;
    logMsg.logSetting( Logger::Info, szFullName, szValue );
  }

  return szValue;
}
}
////////////////////////////////////////////////////////////////////////////////
