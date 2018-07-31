/// Logger.cpp
////////////////////////////////////////////////////////////////////////////////

#include <exception>
#include "stdio.h"
#include "Logger.hpp"

using std::endl;
using std::cerr;
using std::string;
using std::stringstream;
using pcf::TimeStamp;
using pcf::IndiProperty;
using pcf::Logger;
using pcf::Message;

////////////////////////////////////////////////////////////////////////////////
/// This is the static member we are using to write to the log file.

pcf::LogFile pcf::Logger::sm_lfLog;

////////////////////////////////////////////////////////////////////////////////
// Used for the severity of the message.
// This is for backwards compatibility with old code. In new code,
// use the Severity enumerations defined in the header.

const pcf::Logger::Severity pcf::Logger::enumError = pcf::Logger::Error;
const pcf::Logger::Severity pcf::Logger::enumWarning = pcf::Logger::Warning;
const pcf::Logger::Severity pcf::Logger::enumNotice = pcf::Logger::Notice;
const pcf::Logger::Severity pcf::Logger::enumInfo = pcf::Logger::Info;
const pcf::Logger::Severity pcf::Logger::enumDebug = pcf::Logger::Debug;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - sets the Severity to a default of 'info'.

Logger::Logger()
{
  m_tSeverity = Logger::Info;
  m_oClearAfterLog = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
/// @param tSeverity The Severity associated with the message.

Logger::Logger( const Logger::Severity &tSeverity )
{
  m_tSeverity = tSeverity;
  m_oClearAfterLog = false;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

Logger::~Logger()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

Logger::Logger( const Logger &copy )
{
  m_tSeverity = copy.m_tSeverity;
  m_oClearAfterLog = copy.m_oClearAfterLog;
  m_ssMsg.str( copy.m_ssMsg.str() );
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

const Logger &Logger::operator=( const Logger &copy )
{
  if ( &copy != this )
  {
    m_tSeverity = copy.m_tSeverity;
    m_oClearAfterLog = copy.m_oClearAfterLog;
    m_ssMsg.str( copy.m_ssMsg.str() );
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the path & base filename, then opens the log.
/// By default, all Severity levels are enabled.
/// @param szPath The directory where the file will be created,
/// @param szBaseFilename Sets the base filename. This is
/// the filename we will write to, and it will have the date appended
/// on to it. If this is blank, this class will write to a file named
/// with only the date.

void Logger::init( const string &szPath,
                   const string &szBaseFilename )
{
  // Initialize the static logger object.
  sm_lfLog.init( szPath, szBaseFilename );

  // If we initialize this way, set the time stamp & severity options.
  sm_lfLog.enableTimeStamp( true );
  sm_lfLog.enableSeverityPrefix( true );
}

////////////////////////////////////////////////////////////////////////////////
/// @return Will the log message be cleared after it is logged?
/// @return True, if it is enabled, false otherwise.

bool Logger::isClearAfterLogEnabled() const
{
  return m_oClearAfterLog;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we duplicate the messages to stderr? (the default is not to)
/// @param oEnable If true, the messages are send to stderr as well as the log,
/// otherwise they are not.

void Logger::enableStdErr( const bool &oEnable )
{
  sm_lfLog.enableStdErr( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Is the error Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool Logger::isErrorEnabled()
{
  return sm_lfLog.isErrorEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Is the warning Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool Logger::isWarningEnabled()
{
  return sm_lfLog.isWarningEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Is the notice Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool Logger::isNoticeEnabled()
{
  return sm_lfLog.isNoticeEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Is the info Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool Logger::isInfoEnabled()
{
  return sm_lfLog.isInfoEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Is the debug Severity enabled?
/// @return True, if it is enabled, false otherwise.

bool Logger::isDebugEnabled()
{
  return sm_lfLog.isDebugEnabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable the clearing of messages after they have been logged?
/// @param oEnable If true, they will be cleared, otherwise they are not.

void Logger::enableClearAfterLog( const bool &oEnable )
{
  m_oClearAfterLog = oEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable error Severity messages?
/// @param oEnable If true, they will be enabled, otherwise they are not.

void Logger::enableError( const bool &oEnable )
{
  sm_lfLog.enableError( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable warning Severity messages?
/// @param oEnable If true, they will be enabled, otherwise they are not.

void Logger::enableWarning( const bool &oEnable )
{
  sm_lfLog.enableWarning( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable notice Severity messages?
/// @param oEnable If true, they will be enabled, otherwise they are not.

void Logger::enableNotice( const bool &oEnable )
{
  sm_lfLog.enableNotice( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable info Severity messages?
/// @param oEnable If true, they will be enabled, otherwise they are not.

void Logger::enableInfo( const bool &oEnable )
{
  sm_lfLog.enableInfo( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we enable debug Severity messages?
/// @param oEnable If true, they will be enabled, otherwise they are not.

void Logger::enableDebug( const bool &oEnable )
{
  sm_lfLog.enableDebug( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Clear out the message in this object, and reset the Severity back
/// to Logger::Info. No other settings are touched, however.

void Logger::clear()
{
  m_ssMsg.str( "" );
  //  make the Severity equivalent to "LOG_INFO".
  m_tSeverity = Logger::Info;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the current length of the string contained here.

unsigned int Logger::length() const
{
  return (unsigned int)( m_ssMsg.str().length() );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we add a one-letter prefix to the message showing the Severity?
/// @param oAdd If true, it will be added, otherwise it will not.

void Logger::enableSeverityPrefix( const bool &oEnable )
{
  sm_lfLog.enableSeverityPrefix( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Should we add a prefix to the message showing the milliseconds?
/// @param oAdd If true, it will be added, otherwise it will not.

void Logger::enableTimeStamp( const bool &oEnable )
{
  sm_lfLog.enableTimeStamp( oEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Generic function that wraps the actual log call.
/// This version WILL clear the message. If 'm_oClearAfterLog' is true, the
/// message will be cleared anyway, and this call will have no effect.

void Logger::logThenClear()
{
  log();
  clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Generic function that writes to the log.
/// This version does not actually clear the message unless 'm_oClearAfterLog'
/// is true. The message will be cleared only if the call is successful.

void Logger::log()
{
  switch ( m_tSeverity )
  {
    case Logger::Error:
      sm_lfLog.logError( m_ssMsg.str() );
      break;
    case Logger::Warning:
      sm_lfLog.logWarning( m_ssMsg.str() );
      break;
    case Logger::Notice:
      sm_lfLog.logNotice( m_ssMsg.str() );
      break;
    case Logger::Info:
      sm_lfLog.logInfo( m_ssMsg.str() );
      break;
    case Logger::Debug:
      sm_lfLog.logDebug( m_ssMsg.str() );
      break;
    default:
      sm_lfLog.log( m_ssMsg.str() );
      break;
  }

  if ( m_oClearAfterLog == true )
  {
    clear();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Use the stream operator to set the Severity.

Logger &Logger::operator <<( const Logger::Severity &tSeverity )
{
  setSeverity( tSeverity );
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// This function will handle "std::endl" among other things....
/// When a "std::endl" is received via the stream operator, the "log"
/// is called. This sends the message to the file.

Logger &Logger::operator <<( std::ostream & ( *fcn )( std::ostream & ) )
{
  //  check for 'std::endl'.
  if ( fcn == static_cast<std::ostream& ( * )( std::ostream & )>( std::endl ) )
  {
    log();
  }

  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// @return The Severity level set for this message.

Logger::Severity Logger::getSeverity() const
{
  return m_tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Severity level for this message.
/// @param tSeverity The Severity level for this message.

void Logger::setSeverity( const Logger::Severity &tSeverity )
{
  m_tSeverity = tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// @return The text set in this message.

string Logger::getText() const
{
  return m_ssMsg.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the text in this message.
/// @param szText The message text.

void Logger::setText( const string &szText )
{
  m_ssMsg.str( szText );
}

////////////////////////////////////////////////////////////////////////////////
/// @return True, if the Severity level set for this message  is "error",
/// false otherwise.

bool Logger::isError() const
{
  return bool( m_tSeverity == Logger::Error );
}

////////////////////////////////////////////////////////////////////////////////
/// @return True, if the Severity level set for this message  is "warning",
/// false otherwise.

bool Logger::isWarning() const
{
  return bool( m_tSeverity == Logger::Warning );
}

////////////////////////////////////////////////////////////////////////////////
/// @return True, if the Severity level set for this message  is "notice",
/// false otherwise.

bool Logger::isNotice() const
{
  return bool( m_tSeverity == Logger::Notice );
}

////////////////////////////////////////////////////////////////////////////////
/// @return True, if the Severity level set for this message  is "info",
/// false otherwise.

bool Logger::isInfo() const
{
  return bool( m_tSeverity == Logger::Info );
}

////////////////////////////////////////////////////////////////////////////////
/// @return True, if the Severity level set for this message  is "debug",
/// false otherwise.

bool Logger::isDebug() const
{
  return bool( m_tSeverity == Logger::Debug );
}

////////////////////////////////////////////////////////////////////////////////
/// Logs an INDI property to stderr, which ends up in the INDI log.

void Logger::logIndi( const Logger::Severity &tSeverity,
                      const IndiProperty &propLog )
{
  m_ssMsg.str( "" );
  setSeverity( tSeverity );
  m_ssMsg << "PROPERTY  " << propLog.createString();
  cerr << m_ssMsg.str() << endl;
  log();
}

////////////////////////////////////////////////////////////////////////////////
/// Logs an ALERT to the INDI log using a set format (ISO 8061). The timestamp
/// is optional and defaults to "now".
/// ALERT 2013-09-23T11:41:40.959453Z Any string you want

void Logger::logAlert( const Logger::Severity &tSeverity,
                       const string &szMessage,
                       const TimeStamp &tsNow )
{
  m_ssMsg.str( "" );
  setSeverity( tSeverity );
  m_ssMsg << "ALERT     " << tsNow.getFormattedIso8601Str() << " " << szMessage;
  cerr << m_ssMsg.str() << endl;
  log();
}

////////////////////////////////////////////////////////////////////////////////
/// Logs an EVENT to the INDI log using a set format (ISO 8061). The timestamp
/// is optional and defaults to "now".
/// EVENT 2013-09-23T11:41:40.959453Z Name Message

void Logger::logEvent( const Logger::Severity &tSeverity,
                       const string &szName,
                       const string &szMessage,
                       const TimeStamp &tsNow )
{
  m_ssMsg.str( "" );
  setSeverity( tSeverity );
  m_ssMsg << "EVENT     " << tsNow.getFormattedIso8601Str()
          << " " << szName << " " << szMessage;
  cerr << m_ssMsg.str() << endl;
  log();
}

////////////////////////////////////////////////////////////////////////////////
/// Logs an ALARM to the INDI log using a set format (ISO 8061). The timestamp
/// is optional and defaults to "now".
/// Also sends an email notification that there is a problem. The assigned
/// (comma-separated) recipiants are used.

void Logger::logAlarm( const Logger::Severity &tSeverity,
                       const string &szRecipients,
                       const string &szSubject,
                       const string &szMessage,
                       const TimeStamp &tsNow )
{
  m_ssMsg.str( "" );
  setSeverity( tSeverity );
  m_ssMsg << "ALARM     " << tsNow.getFormattedIso8601Str() << " " << szMessage;
  cerr << m_ssMsg.str() << endl;
  log();

  try
  {
    FILE *fpMail = NULL;
    if ( ( fpMail = ::popen( "/usr/sbin/sendmail -t", "w" ) ) != NULL )
    {
      fprintf( fpMail, "To: %s \n", szRecipients.c_str() );
      fprintf( fpMail, "Subject: %s\n", szSubject.c_str() );
      fprintf( fpMail, "\n" );
      fprintf( fpMail, "%s \n", m_ssMsg.str().c_str() );
      fprintf( fpMail, "\n.\n" );

      pclose( fpMail );
    }
  }
  catch ( std::exception & excep )
  {
    Logger msgLog;
    msgLog.clear();
    msgLog << Logger::enumError << "Exception occured when sending email alert: "
           << excep.what() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
