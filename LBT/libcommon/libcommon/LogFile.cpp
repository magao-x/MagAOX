/// $Id: LogFile.cpp 5109 2008-05-23 23:16:48Z pgrenz $
////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>  // provides 'std::runtime_error'
#include <sys/stat.h> // provides 'mkdir', 'stat'
#include <unistd.h>
#include <sys/types.h>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "System.hpp"
#include "LogFile.hpp"
#include "TimeStamp.hpp"

using std::runtime_error;
using std::string;
using std::stringstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::ios;
using std::setw;
using std::setfill;
using pcf::System;
using pcf::LogFile;
using pcf::MutexLock;
using pcf::TimeStamp;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - sets the Severity to a default of 'info'.

LogFile::LogFile()
{
  m_oEnableStdErr = false;
  m_oEnableTimeStamp = false;
  m_szPath = "";
  m_szBaseFilename = "";
  m_szName = "";
  m_oEnableDebug = true;
  m_oEnableError = true;
  m_oEnableInfo = true;
  m_oEnableNotice = true;
  m_oEnableWarning = true;
  m_oIsInit = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

LogFile::~LogFile()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Closes any open files.

void LogFile::close()
{
  // Doesn't do anything.
}

////////////////////////////////////////////////////////////////////////////////
/// Should we duplicate the messages to stderr?

void LogFile::enableStdErr( const bool &oEnable )
{
  m_oEnableStdErr = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Are we enabling the replication of the message to stderr?

bool LogFile::isStdErrEnabled()
{
  return m_oEnableStdErr;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we add a time stamp to each log message? The messages routed
/// to stderr do not get a timestamp regardless of this setting.

void LogFile::enableTimeStamp( const bool &oEnable )
{
  m_oEnableTimeStamp = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Are we enabling the time stamp in the log message?

bool LogFile::isTimeStampEnabled()
{
  return m_oEnableTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we add a severity prefix to each log message? This only is applied
/// if one of the severity log methods is called.

void LogFile::enableSeverityPrefix( const bool &oEnable )
{
  m_oEnableSeverityPrefix = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Are we enabling the severity prefix in the log message?

bool LogFile::isSeverityPrefixEnabled()
{
  return m_oEnableSeverityPrefix;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the name for this object.

string LogFile::getName() const
{
  return m_szName;
}

////////////////////////////////////////////////////////////////////////////////
/// This sets the name for this object.

void LogFile::setName( const std::string &szName )
{
  m_szName = szName;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable error Severity messages?

void LogFile::enableError( const bool &oEnable )
{
  m_oEnableError = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable warning Severity messages?

void LogFile::enableWarning( const bool &oEnable )
{
  m_oEnableWarning = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable notice Severity messages?

void LogFile::enableNotice( const bool &oEnable )
{
  m_oEnableNotice = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable info Severity messages?

void LogFile::enableInfo( const bool &oEnable )
{
  m_oEnableInfo = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable debug Severity messages?

void LogFile::enableDebug( const bool &oEnable )
{
  m_oEnableDebug = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the error Severity enabled?

bool LogFile::isErrorEnabled()
{
  return m_oEnableError;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the warning Severity enabled?

bool LogFile::isWarningEnabled()
{
  return m_oEnableWarning;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the notice Severity enabled?

bool LogFile::isNoticeEnabled()
{
  return m_oEnableNotice;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the info Severity enabled?

bool LogFile::isInfoEnabled()
{
  return m_oEnableInfo;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the debug Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool LogFile::isDebugEnabled()
{
  return m_oEnableDebug;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the needed info setup the file. For backwards compatibility.
/// @param szPath Sets the path where the file will be created..
/// @param szBaseFilename Sets the filename - usually the name of the process.
/// @param szSymLinkName Sets the sym link which is used to access the file.

void LogFile::open( const string &szPath,
                    const string &szBaseFilename,
                    const string &szSymLinkName )
{
  init( szPath, szBaseFilename );
}

////////////////////////////////////////////////////////////////////////////////
/// Set the needed info & store the filename.
/// @param szPath Sets the path where the file will be created..
/// @param szBaseFilename Sets the filename - usually the name of the process.

void LogFile::init( const string &szPath,
                    const string &szBaseFilename )
{
  // we need to ensure that the umask for this process is set correctly
  // so that the directory structure created here is readable by everyone.
  ::umask( 000 );

  m_szPath = szPath;
  m_szBaseFilename = szBaseFilename;

  // Make sure there is something in the path.
  m_szPath = ( m_szPath.length() == 0 ) ? ( "." ) : ( m_szPath );

  // Make the directory with maximum permissions.
  // This will be modified by the process permissions.
  System::makePath( m_szPath );

  // Make sure to recognize that we have been initialized and are ready
  // to write to the file.
  m_oIsInit = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with no severity information.
/// @Param szLogString The text that will be written on one line.
///
void LogFile::log( const string &szLogString )
{
  writeToFile( szLogString, "" );
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with a severity of 'debug'.
/// @Param szLogString The text that will be written on one line.

void LogFile::logDebug( const string &szLogString )
{
  if ( m_oEnableDebug == true )
  {
    writeToFile( szLogString, "[D]" );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with a severity of 'error'.
/// @Param szLogString The text that will be written on one line.

void LogFile::logError( const string &szLogString )
{
  if ( m_oEnableError == true )
  {
    writeToFile( szLogString, "[E]" );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with a severity of 'info'.
/// @Param szLogString The text that will be written on one line.

void LogFile::logInfo( const string &szLogString )
{
  if ( m_oEnableInfo == true )
  {
    writeToFile( szLogString, "[I]" );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with a severity of 'notice'.
/// @Param szLogString The text that will be written on one line.

void LogFile::logNotice( const string &szLogString )
{
  if ( m_oEnableNotice == true )
  {
    writeToFile( szLogString, "[N]" );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Function that writes to the log with a severity of 'warning'.
/// @Param szLogString The text that will be written on one line.

void LogFile::logWarning( const string &szLogString )
{
  if ( m_oEnableWarning == true )
  {
    writeToFile( szLogString, "[W]" );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Generic function that writes to the log.

void LogFile::writeToFile( const string &szLogString,
                           const string &szSeverityPrefix )
{
  // Has the file path & filename been set?
  if ( m_oIsInit == true )
  {
    // We need to know the current time to write to the correct file
    // and possibly create the time stamp.
    TimeStamp tsNow = TimeStamp::now();

    string szModLogString( szLogString );

    // Add a severity prefix to the front of the log message?
    if ( m_oEnableSeverityPrefix == true && szSeverityPrefix.length () > 0 )
    {
      szModLogString = szSeverityPrefix + " " + szModLogString;
    }

    //  should we send the message to stderr as well?
    if ( m_oEnableStdErr == true )
    {
      cerr << szModLogString << endl;
    }

    // Add a time stamp to the front of the log message?
    if ( m_oEnableTimeStamp == true )
    {
      szModLogString = tsNow.getFormattedIso8601Str() + " " + szModLogString;
    }

    // Create the filename.
    string szFilename = m_szPath + "/" + m_szBaseFilename;
    szFilename = System::createFileNameWithTimeStamp( szFilename, "log" );

    MutexLock::AutoLock mutFile( &m_mutFile );

    ///  open the log for this process.
    ofstream ofsLog;
    ofsLog.open( szFilename.c_str(), ios::out | ios::app );

    if ( ofsLog.good() == false )
    {
      cerr << "[LogFile] The logfile '" << szFilename
           << "' could not be opened. Message: '" << szModLogString << "'" << endl;
    }
    else
    {
      ofsLog << szModLogString << endl;
      ofsLog.flush();
      ofsLog.close();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a filename based on the date.
/*
string LogFile::createLogfileName( const string &szPath,
                                   const string &szBaseFilename,
                                   const TimeStamp &tsName )
{
  // we need the date prepended to the file name as a number.
  string szIsoDate = tsName.getFormattedIsoDateStr();

  if ( szBaseFilename.size() > 0 )
    return string( szPath + "/" + szBaseFilename  + "." + szIsoDate + ".log" );
  else
    return string( szPath + "/" + szIsoDate + ".log" );
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Ensures that a path exists. If it does not, it is created.
/// @param szPath The path to create.
/*
void LogFile::makePath( const string &szPath ) const
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
          throw ( runtime_error( string( "Cannot create directory '" ) +
              szDir + "', " + strerror( errno ) ) );
        }
      }
      else if ( !S_ISDIR( stFile.st_mode ) )
      {
        throw ( runtime_error(
            string( "Directory not created; file name conflict: '") + szDir + "'" ) );
      }
    }
  }
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Ensures that a symlink exists. If it does not, it is created.
/// @param szFilename The file to link to.
/// @param szLinkname The link to create.
/*
void LogFile::makeSymLink( const string &szFilename,
                           const string &szLinkname ) const
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
*/
////////////////////////////////////////////////////////////////////////////////
