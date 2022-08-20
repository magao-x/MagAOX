/// System.cpp
///
/// \author Mark Milton
/// \author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <unistd.h>             // provides 'getcwd', 'chdir'
#include <stdlib.h>             // provides 'getenv'
#include <sys/stat.h>           // provides 'mkdir', 'stat'
#include <stdexcept>            // provides 'std::runtime_error'
#include <string.h>             // provides 'strerror'
#include <sys/resource.h>
#include <cerrno>
#include <iostream>
#include <sys/types.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <fstream>
#include <dirent.h>
#include "TimeStamp.hpp"
#include "System.hpp"

using std::string;
using std::vector;
using std::sort;
using std::ifstream;
using std::endl;
using pcf::TimeStamp;
using pcf::System;
using std::runtime_error;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

System::System()
{
  // Empty because this is private.
}

////////////////////////////////////////////////////////////////////////////////
/// Standard copy constructor from another System object.

System::System( const System& ecRhs )
{
  static_cast<void>(ecRhs);
  // Empty because this is private.
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator. Assigns this object from another System object.

const System& System::operator=( const System& sysRhs )
{
  static_cast<void>(sysRhs);
  // Empty because this is private.
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief System::doesFileExist
/// Check if specified file exists in the file system.
/// \param szFileName the file name to look for.
/// \return

bool System::doesFileExist( const string& szFileName )
{
  bool oExists = false;

  ifstream ifs( szFileName.c_str(), ifstream::in );
  if ( ifs )
  {
    oExists = true;
    ifs.close();
  }

  return oExists;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief createEnvPath
/// Creates a path as a string based on an environment variable. If the
/// environment variable doesn't exist or is empty, the cwd() is returned.
/// \param szEnvName The environment variable name.
/// \return The complete path.

string System::createEnvPath( const std::string &szEnvName )
{
  string szPath = string( ::getcwd( NULL, 0 ) ) + "/";

  char *pcEnvPath = ::getenv( szEnvName.c_str() );
  if ( pcEnvPath != NULL )
  {
    szPath = string( pcEnvPath ) + "/";
  }

  return szPath;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief System::setCoreFilePath
/// Sets the cwd and the rlimit to ensure that core files go to
/// a place specified by the user as well as creating the specified path.
/// \param szCoreFilePath The path to create.

void System::setCoreFilePath( const string &szCoreFilePath )
{
  // make sure we can create a core file
  // this is not catastrophic if we can't.
  rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = RLIM_INFINITY;
  ::setrlimit( RLIMIT_CORE, &rlim );

  // change the file usage mask.
  ::umask( 0 );

  makePath( szCoreFilePath );

  // make the current working directory one that is set by the user.
  int rv = ::chdir( szCoreFilePath.c_str() );
  if(rv < 0) std::cerr << __FILE__ << " " << __LINE__ << " " << strerror(errno) << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// \brief System::createFileNameWithTimeStamp
/// Return formatted filename with current timestamp.
/// EXAMPLE: PREFIX_20171230.SUFFIX
/// \param szPrefix File name prefix (optional).
/// \param szSuffix File name suffix (optional).
/// \return The filename created.

string System::createFileNameWithTimeStamp( const string& szPrefix,
                                            const string& szSuffix )
{
  string szFileName = szPrefix;

  if ( szPrefix.empty() == false && szPrefix[szPrefix.size()-1] != '/' )
  {
    szFileName += "_";
  }

  szFileName += TimeStamp::now().getFormattedIsoDateStr();

  if ( szSuffix.empty() == false )
  {
    szFileName += ("." + szSuffix);
  }

  return szFileName;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief System::createFileList
/// Creates a vector of file names in szPath that match szSuffix and szPrefix.
/// \param szPath Path name to search.
/// \param szSuffix File name suffix (optional).
/// \param szPrefix File name prefix (optional).
/// \return String of comma separated list of file names.

vector<string> System::createFileList( const string& szPath,
                                       const string& szSuffix,
                                       const string& szPrefix )
{
  vector<string> szvecMatchingFiles;

  // Open dictory for searching.
  dirent* pdeDirEntry = NULL;
  string szFileName = "";
  DIR* pdirDirHandle;
  pdirDirHandle = opendir( szPath.empty() ? "." : szPath.c_str() );
  if ( pdirDirHandle )
  {
    while ( true )
    {
      // get the next directory entry
      pdeDirEntry = readdir( pdirDirHandle );
      if ( pdeDirEntry == NULL )
        break;

      // If file name passes the filters, add it to the list
      szFileName = string( pdeDirEntry->d_name );
      if ( !szSuffix.empty() )
        if ( szFileName.rfind( szSuffix ) !=
             ( szFileName.length() - szSuffix.length() ) )
          szFileName = "";
      if ( !szPrefix.empty() )
        if ( szFileName.find( szPrefix ) != 0 )
          szFileName = "";
      if ( !szFileName.empty() )
        if ( ( szFileName != "." ) && ( szFileName != ".." ) )
          szvecMatchingFiles.push_back( szFileName );
    }
    closedir( pdirDirHandle );

    if ( !szvecMatchingFiles.empty() )
      sort( szvecMatchingFiles.begin(), szvecMatchingFiles.end() );
  }

  return szvecMatchingFiles;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief System::makePath
/// Ensures that a path exists. If it does not, it is created.
/// \param szPath The path to create.
/// \return The path that was created.

string System::makePath( const string &szPath )
{
  // make the directory with maximum permissions.
  // this will be modified by the process permissions.
  for ( unsigned int ii = 0; ii < szPath.length(); ii++ )
  {
    if ( szPath[ii] == '/' )
    {
      string szDir = szPath.substr( 0, ii + 1 );

      struct stat stFile;
      if ( ::stat( szDir.c_str(), &stFile ) != 0 )
      {
        if ( ::mkdir( szDir.c_str(), 0777 ) != 0 && errno != EEXIST )
        {
          throw ( runtime_error( string( "Cannot create directory path '" ) +
              szDir + "', " + strerror( errno ) ) );
        }
      }
      else if ( !S_ISDIR( stFile.st_mode ) )
      {
        throw ( runtime_error(
            string( "Directory path not created; file name conflict: '") +
              szDir + "'" ) );
      }
    }
  }
  return szPath;
}

////////////////////////////////////////////////////////////////////////////////
/// Ensures that a symlink exists. If it does not, it is created.
/// @param szFilename The file to link to.
/// @param szLinkname The link to create.

void System::makeSymLink( const string &szFilename,
                          const string &szLinkname )
{
  // Try to remove the existing symlink. zero means it went okay.
  // ENOENT means it did not exist - but for us this is not an error.
  if ( ::remove( szLinkname.c_str() ) != 0 && errno != ENOENT )
  {
    throw ( runtime_error( "Could not remove symlink '" + szLinkname + "'." ) );
  }

  // Try to make the symlink. Zero means it went okay.
  if ( ::symlink( szFilename.c_str(), szLinkname.c_str() ) != 0 )
  {
    throw ( runtime_error( "Could not create symlink '" + szLinkname + "'." ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
