/// Event.cpp
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <syslog.h>
#include "TimeStamp.hpp"
#include "Logger.hpp"
#include "Event.hpp"

////////////////////////////////////////////////////////////////////////////////

using ::std::cerr;
using ::std::endl;
using ::std::string;
using ::std::stringstream;
using ::std::setw;
using ::std::setfill;
using ::std::fixed;
using ::std::setprecision;
using ::std::replace;
using pcf::Logger;
using pcf::TimeStamp;
using pcf::Event;

////////////////////////////////////////////////////////////////////////////////
/// Initialize global state variables for the overall syslog.
/// These are the defaults for these settings.

bool pcf::Event::sm_oAddSeverityPrefix = true;
bool pcf::Event::sm_oAddMillisPrefix = false;
bool pcf::Event::sm_oEnableStdErr = false;
bool pcf::Event::sm_oEnableError = true;
bool pcf::Event::sm_oEnableWarning = true;
bool pcf::Event::sm_oEnableNotice = true;
bool pcf::Event::sm_oEnableInfo = true;
bool pcf::Event::sm_oEnableDebug = true;
std::string pcf::Event::sm_szPrefix = "";
std::string pcf::Event::sm_szEmailList = "";
pcf::Event::Severity pcf::Event::sm_tSeverity = pcf::Event::enumDefaultSeverity;
pcf::Event::Facility pcf::Event::sm_tFacility = pcf::Event::enumDefaultFacility;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - sets the severity and facility to the defaults.

Event::Event()
{
  m_tFacility = Event::sm_tFacility;
  m_tSeverity = Event::sm_tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
/// @param tSeverity The severity associated with the event.

Event::Event( const Event::Severity &tSeverity )
{
  m_tFacility = Event::sm_tFacility;
  m_tSeverity = tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
/// @param tSeverity The severity associated with the event.
/// @param tFacility The facility associated with the eventf.

Event::Event( const Event::Severity &tSeverity, const Event::Facility &tFacility )
{
  m_tFacility = tFacility;
  m_tSeverity = tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

Event::~Event()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

Event::Event( const Event &rhs )
{
  m_tFacility = rhs.m_tFacility;
  m_tSeverity = rhs.m_tSeverity;
  m_ssMsg.str( rhs.m_ssMsg.str() );
  m_tsTimeStamp = rhs.m_tsTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

const Event &Event::operator=( const Event &rhs )
{
  if ( &rhs != this )
  {
    m_tFacility = rhs.m_tFacility;
    m_tSeverity = rhs.m_tSeverity;
    m_ssMsg.str( rhs.m_ssMsg.str() );
    m_tsTimeStamp = rhs.m_tsTimeStamp;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the prefix & facility, then opens the log.
/// By default, all severity levels are enabled.
/// @param szPrefix Sets the prefix appended to every event - usually the
/// name of the process.
/// @param tFacility The facility to send the messages to - see 'Facility'.
/// @param tSeverity The severity of the event - see 'Severity'.

void Event::init( const string &szPrefix,
                  const Facility &tFacility,
                  const Severity &tSeverity )
{
  //  make sure the log is closed, as this may be called more than once.
  ::closelog();

  string szFullPrefix( "EVENT" );
  if ( szPrefix.size() > 0 )
    szFullPrefix += string( "_" + szPrefix );

  Event::sm_szPrefix = szFullPrefix;
  Event::sm_tFacility = tFacility;
  Event::sm_tSeverity = tSeverity;

  //  open the syslog for this process.
  ::openlog( sm_szPrefix.c_str(), LOG_CONS, tFacility );
}

////////////////////////////////////////////////////////////////////////////////
/// Clear out all message in this object, and resets the severity & facility
/// back to the defaults. No other settings are touched, however.

void Event::clear()
{
  m_ssMsg.str( "" );
  m_tFacility = Event::sm_tFacility;
  m_tSeverity = Event::sm_tSeverity;
}

////////////////////////////////////////////////////////////////////////////////
/// Generic function that wraps the actual syslog call.
/// This version WILL clear the message.
/// @return enumNoError No error occurred. For now, this is the only return possible

int Event::logThenClear()
{
  int nError = enumNoError;
  if ( ( nError = log() ) == enumNoError )
  {
    //  clear the log string, the severity and facility.
    clear();
  }
  return nError;
}

////////////////////////////////////////////////////////////////////////////////
/// Generic function that wraps the actual syslog call.
/// This version does not actually clear the message.
/// @return enumNoError No error occurred. For now, this is the only return possible

int Event::log()
{
  if ( isSeverityEnabled() == true )
  {
    string szSeverity( "" );
    if ( sm_oAddSeverityPrefix == true )
    {
      switch ( m_tSeverity )
      {
        case Event::enumError:
          szSeverity = "[E] ";
          break;
        case Event::enumWarning:
          szSeverity = "[W] ";
          break;
        case Event::enumNotice:
          szSeverity = "[N] ";
          break;
        case Event::enumInfo:
          szSeverity = "[I] ";
          break;
        case Event::enumDebug:
          szSeverity = "[D] ";
          break;
        default:
          szSeverity = "[?] ";
          break;
      }
    }

    //  create a string which holds the number of milliseconds.
    stringstream ssMillis;
    if ( sm_oAddMillisPrefix == true )
    {
      timeval tvNow;
      gettimeofday( &tvNow, NULL );
      ssMillis << "<" << setw( 3 ) << setfill( '0' ) << fixed << setprecision( 3 )
               << tvNow.tv_usec / 1000 << "> ";
    }

    //  prepend the severity and ms string onto the message if the option is set.
    string szFullMsg( szSeverity + ssMillis.str() + m_ssMsg.str() );

    //  should we send the message to stderr as well?
    if ( sm_oEnableStdErr == true )
    {
      cerr << szFullMsg << endl;
    }

    //  translate the desired severity into one syslog understands.
    int nSysSeverity = Event::sm_tSeverity;
    switch ( m_tSeverity )
    {
      case Event::enumError:
        nSysSeverity = LOG_ERR;
        break;
      case Event::enumWarning:
        nSysSeverity = LOG_WARNING;
        break;
      case Event::enumNotice:
        nSysSeverity = LOG_NOTICE;
        break;
      case Event::enumInfo:
        nSysSeverity = LOG_INFO;
        break;
      case Event::enumDebug:
        nSysSeverity = LOG_DEBUG;
        break;
    }

    //  translate the desired facility into one syslog understands.
    int nSysFacility = Event::sm_tFacility;
    switch ( m_tFacility )
    {
      case Event::enumLocal0:
        nSysFacility = LOG_LOCAL0;
        break;
      case Event::enumLocal1:
        nSysFacility = LOG_LOCAL1;
        break;
      case Event::enumLocal2:
        nSysFacility = LOG_LOCAL2;
        break;
      case Event::enumLocal3:
        nSysFacility = LOG_LOCAL3;
        break;
      case Event::enumLocal4:
        nSysFacility = LOG_LOCAL4;
        break;
      case Event::enumLocal5:
        nSysFacility = LOG_LOCAL5;
        break;
      case Event::enumLocal6:
        nSysFacility = LOG_LOCAL6;
        break;
      case Event::enumLocal7:
        nSysFacility = LOG_LOCAL7;
        break;
      default:
      case Event::enumNoFacility:
        nSysFacility = LOG_LOCAL0;
        break;
    }

    //  actually write the message.
    syslog( nSysSeverity | nSysFacility, "%s", szFullMsg.c_str() );
  }

  return enumNoError;
}

////////////////////////////////////////////////////////////////////////////////
/// This sends an email notification that there is a problem. The assigned
/// recipiants are used.

bool Event::sendGenericEmailAlarm( const std::string &szSubject,
                                   const std::string &szMessage )
{
  bool oRetVal = false;

  try
  {
    FILE *fpMail = NULL;
    if ( ( fpMail = ::popen( "/usr/sbin/sendmail -t", "w" ) ) != NULL )
    {
      fprintf( fpMail, "To: %s \n", sm_szEmailList.c_str() );
      fprintf( fpMail, "Subject: %s\n", szSubject.c_str() );
      fprintf( fpMail, "\n" );
      fprintf( fpMail, "%s \n", szMessage.c_str() );
      fprintf( fpMail, "\n.\n" );

      pclose( fpMail );

      oRetVal = true;
    }
  }
  catch ( std::exception excep )
  {
    Logger logMsg;
    logMsg.clear();
    logMsg << Logger::enumError << "Exception occured when sending email alert: "
           << excep.what() << std::endl;
  }

  return oRetVal;
}

////////////////////////////////////////////////////////////////////////////////
/// This sends an email notification that there is a problem. The assigned
/// recipiants are used.

bool Event::sendTempSensorEmailAlarm( const std::string &szSensorName,
                                      const std::string &szChannelName,
                                      const float &eValue,
                                      const float &eMin,
                                      const float &eMax )
{
  string szSubject( "Sensor Temperature Alert Message" );

  // Send the alarm out via email.
  stringstream ssMsg;
  ssMsg << "The following sensor has detected a temperature out-of-range event"
        << endl << endl
        << "\tSensor:      " << szSensorName << endl
        << "\tChannel:     " << szChannelName << endl
        << "\tValue:       " << eValue << endl
        << "\tLower Bound: " << eMin << endl
        << "\tUpper Bound: " << eMax << endl
        << "\tDate:        " << TimeStamp::now().getFormattedStr() << endl;

  return sendGenericEmailAlarm( szSubject, ssMsg.str() );
}

////////////////////////////////////////////////////////////////////////////////
/// If the local objects severity label is enabled for this process, return
/// true, otherwise false.

bool Event::isSeverityEnabled()
{
  switch ( getSeverity() )
  {
    case Event::enumError:
      return isErrorEnabled();
      break;
    case Event::enumWarning:
      return isWarningEnabled();
      break;
    default:
    case Event::enumNotice:
      return isNoticeEnabled();
      break;
    case Event::enumInfo:
      return isInfoEnabled();
      break;
    case Event::enumDebug:
      return isDebugEnabled();
      break;

  }
}

////////////////////////////////////////////////////////////////////////////////
/// Use the stream operator to set the facility.

Event &Event::operator <<( const Event::Facility &tFacility )
{
  setFacility( tFacility );
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Use the stream operator to set the severity.

Event &Event::operator <<( const Event::Severity &tSeverity )
{
  setSeverity( tSeverity );
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Use the stream operator to set the time stamp.

Event &Event::operator<<( const TimeStamp &dtTimeStamp )
{
  setTimeStamp( dtTimeStamp );
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// This function will handle "std::endl" among other things....
/// When a "std::endl" is received via the stream operator, the "log" method
/// is called. This sends the message to syslog.

Event &Event::operator <<( std::ostream & ( *fcn )( std::ostream & ) )
{
  //  check for 'std::endl'.
  if ( fcn == static_cast<std::ostream& ( * )( std::ostream & )>( std::endl ) )
  {
    log();
  }

  return *this;
}

////////////////////////////////////////////////////////////////////////////////

