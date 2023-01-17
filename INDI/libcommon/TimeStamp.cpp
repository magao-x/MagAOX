/// $Id: TimeStamp.cpp,v 1.5 2007/09/20 18:06:48 pgrenz Exp $
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <stdio.h>
#include <iostream>

#include "TimeStamp.hpp"

using std::string;
using std::ostream;
using std::stringstream;
using std::transform;
using std::setfill;
using std::setw;
using std::setprecision;
using pcf::TimeStamp;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - initialize the internal timeval struct.

TimeStamp::TimeStamp()
{
  m_tvCurr = TimeStamp::now().getTimeVal();
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor from a timeval structure.
/// @param tvCurr A timeval struct which contains a valid time.

TimeStamp::TimeStamp( const timeval &tvCurr )
{
  m_tvCurr.tv_sec = tvCurr.tv_sec;
  m_tvCurr.tv_usec = tvCurr.tv_usec;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor from a number of milliseconds. The amount is
/// considered to be an absolute value.
/// @param nMillis The number of milliseconds since the epoch UTC.

TimeStamp::TimeStamp( const int &nMillis )
{
  m_tvCurr.tv_sec = nMillis / 1000;
  m_tvCurr.tv_usec = ( nMillis % 1000 ) * 1000;
}

////////////////////////////////////////////////////////////////////////////////
/// standard constructor from a year, month, day, hour (24), min, sec. The
/// time is considered to be in UTC, and no adjustment is done.
/// @param nYear The 4 digit year.
/// @param nMonth The month (between 1 and 12 inclusive).
/// @param nDay The day (between 1 and 31 inclusive).
/// @param nHour The hour (between 0 and 23 inclusive).
/// @param nMinute The minute (between 0 and 59 inclusive)
/// @param nSecond The second (between 0 and 60 inclusive).

TimeStamp::TimeStamp( const int &nYear, const int &nMonth, const int &nDay,
                      const int &nHour, const int &nMinute, const int &nSecond )
{
  // First get the current UTC. This will be modified to hold the
  // specified timestamp.
  ::time_t now = ::time( 0 );
  ::tm *gmtm = ::gmtime( &now );

  // Now fill in the rest of the struct with our desired values.
  gmtm->tm_sec = nSecond;
  gmtm->tm_min = nMinute;
  gmtm->tm_hour = nHour;
  gmtm->tm_mday = nDay;
  gmtm->tm_mon = nMonth - 1; // month is zero-based in tm struct.
  gmtm->tm_year = nYear - 1900; // 1900 is stored as 0 in tm struct.
  //gmtm->tm_wday;  /* day of week (Sunday = 0) */
  //gmtm->tm_yday;  /* day of year (0 - 365) */
  //gmtm->tm_isdst; /* is summer time in effect? */
  //gmtm->tm_zone;  /* abbreviation of timezone name */
  //gmtm->tm_gmtoff;  /* offset from UTC in seconds */

  // The object pointed by gmtm is modified, setting the tm_wday and tm_yday
  // to their appropiate values, and modifying the other members as necessary
  // to values within the normal range representing the specified time.
  m_tvCurr.tv_sec = local_timegm( gmtm );
  m_tvCurr.tv_usec = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

TimeStamp::~TimeStamp()
{
  //  nothing to do.
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the underlying timeval struct. This is the number of seconds since
/// January 1, 1970, UTC.

const timeval &TimeStamp::getTimeVal() const
{
  return m_tvCurr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the microsecond part of the time val struct.

/*int TimeStamp::getTimeValMicros() const
{
  return m_tvCurr.tv_usec;
}*/

////////////////////////////////////////////////////////////////////////////////
/// Returns the seconds part of the time val struct.

/*int TimeStamp::getTimeValSecs() const
{
  return m_tvCurr.tv_sec;
}*/

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator. Assigns this object from another TimeStamp object.
/// @param tsRhs The TimeStamp object to use to reset this one.

const TimeStamp &TimeStamp::operator= ( const TimeStamp &tsRhs )
{
  if ( &tsRhs != this )
  {
    m_tvCurr = tsRhs.m_tvCurr;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator from a timeval. Sets the internal data from a timeval
/// struct
/// @param tv The timeval struct to use.

const TimeStamp &TimeStamp::operator= ( const timeval &tv  )
{
  m_tvCurr.tv_sec = tv.tv_sec;
  m_tvCurr.tv_usec = tv.tv_usec;

  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator from a number of milliseconds. This is considered to be
/// an absolute value counted from the epoch, UTC.
/// @param nMillis The number of milliseconds that has elapsed since epoch, UTC.

const TimeStamp &TimeStamp::operator= ( const int &nMillis  )
{
  m_tvCurr.tv_sec = nMillis / 1000;
  m_tvCurr.tv_usec = ( nMillis % 1000 ) * 1000;

  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Subtraction operator. Subtracts two TimeStamps and returns the difference
/// as another TimeStamp. One must exercise caution, as the result is not an
/// absolute time, but rather a relative one.
/// @param tsRhs Another TimeStamp object.
/// @return The result: an absolute TimeStamp object.

TimeStamp TimeStamp::operator- ( const TimeStamp &tsRhs ) const
{
  //  subtracting two timeval structs is a little non-obvious.

  timeval tvDiff;

  tvDiff.tv_sec = m_tvCurr.tv_sec - tsRhs.m_tvCurr.tv_sec;
  tvDiff.tv_usec = m_tvCurr.tv_usec - tsRhs.m_tvCurr.tv_usec;
  if ( tvDiff.tv_usec < 0 )
  {
    tvDiff.tv_sec--;
    tvDiff.tv_usec += 1000000;
  }

  return TimeStamp( tvDiff );
}

////////////////////////////////////////////////////////////////////////////////
/// Addition operator. Adds two TimeStamps and returns the sum
/// as another TimeStamp. One must exercise caution, as the second may not
/// be a meaningful value.
/// @param tsRhs Another TimeStamp object.
/// @return The result: an absolute TimeStamp object.

TimeStamp TimeStamp::operator+ ( const TimeStamp &tsRhs ) const
{
  //  adding two timeval structs is a little non-obvious.

  timeval tvSum;
  tvSum.tv_sec = m_tvCurr.tv_sec + tsRhs.m_tvCurr.tv_sec;
  int nnSum = m_tvCurr.tv_usec + tsRhs.m_tvCurr.tv_usec;

  if ( nnSum < 1000000 )
  {
    tvSum.tv_usec = nnSum;
  }
  else
  {
    tvSum.tv_sec++;
    tvSum.tv_usec = nnSum - 1000000;
  }

  return TimeStamp( tvSum );
}

////////////////////////////////////////////////////////////////////////////////
/// Less than operator. Subtracts two TimeStamps and returns true if this
/// object is less than (further in the past) than tsRhs.
/// @param tsRhs Another TimeStamp object.
/// @return true or false.

bool TimeStamp::operator<( const TimeStamp &tsRhs ) const
{
  return ( ( getMicros() - tsRhs.getMicros() ) < 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Less than or equal to operator. Subtracts two TimeStamps and returns
/// true if this object is less than (further in the past) or equal to than tsRhs.
/// @param tsRhs Another TimeStamp object.
/// @return true or false.

bool TimeStamp::operator<=( const TimeStamp &tsRhs ) const
{
  return ( ( getMicros() - tsRhs.getMicros() ) <= 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Greater than operator. Subtracts two TimeStamps and returns true if this
/// object is greater than (more recent in the past) than tsRhs.
/// @param tsRhs Another TimeStamp object.
/// @return true or false.

bool TimeStamp::operator>( const TimeStamp &tsRhs ) const
{
  return ( ( getMicros() - tsRhs.getMicros() ) > 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Greater than or equal to operator. Subtracts two TimeStamps and returns
/// true if this object is greater than (more recent in the past) or equal to
/// the tsRhs.
/// @param tsRhs Another TimeStamp object.
/// @return true or false.

bool TimeStamp::operator>=( const TimeStamp &tsRhs ) const
{
  return ( ( getMicros() - tsRhs.getMicros() ) >= 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Initialize this TimeStamp from another one.
/// @param ts Another TimeStamp to use to initialize this one.

TimeStamp::TimeStamp( const TimeStamp &ts )
{
  m_tvCurr = ts.m_tvCurr;
}

////////////////////////////////////////////////////////////////////////////////
/// Equals operator. Returns true if the two TimeStamp objects are the same.
/// @param tsRhs Another TimeStamp object.
/// @return True if the two TimeStamps are the same, false otherwise.

bool TimeStamp::operator== ( const TimeStamp &tsRhs ) const
{
  return bool( m_tvCurr.tv_sec == tsRhs.m_tvCurr.tv_sec &&
               m_tvCurr.tv_usec == tsRhs.m_tvCurr.tv_usec );
}

////////////////////////////////////////////////////////////////////////////////
/// returns the month number (1 to 12) given the 3-letter name.
/// The name should be one of: jan feb mar apr may jun jul aug sep oct nov dec.
/// Capitalization will be corrected. It defaults to -1 if it is unknown.

int TimeStamp::getMonthNumber( const std::string &szMonth )
{
  string szMonthLC = szMonth.substr( 0, 3 );
  transform( szMonthLC.begin(), szMonthLC.end(), szMonthLC.begin(),
             ( int( * )( int ) )tolower );

  if ( szMonthLC == "jan" )
    return 1;
  if ( szMonthLC == "feb" )
    return 2;
  if ( szMonthLC == "mar" )
    return 3;
  if ( szMonthLC == "apr" )
    return 4;
  if ( szMonthLC == "may" )
    return 5;
  if ( szMonthLC == "jun" )
    return 6;
  if ( szMonthLC == "jul" )
    return 7;
  if ( szMonthLC == "aug" )
    return 8;
  if ( szMonthLC == "sep" )
    return 9;
  if ( szMonthLC == "oct" )
    return 10;
  if ( szMonthLC == "nov" )
    return 11;
  if ( szMonthLC == "dec" )
    return 12;
  // default is unknown.
  return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the 3-letter weekday name given the number (1 to 7) of the weekday.
/// The name will be one of: Sun Mon Tue Wed Thu Fri Sat.
/// Capitalization will be corrected. It defaults to "???" if it is unknown.

string TimeStamp::getWeekdayName( const int &nWeekdayNum )
{
  switch ( nWeekdayNum )
  {
    case 1:
      return "Sun";
      break;
    case 2:
      return "Mon";
      break;
    case 3:
      return "Tue";
      break;
    case 4:
      return "Wed";
      break;
    case 5:
      return "Thu";
      break;
    case 6:
      return "Fri";
      break;
    case 7:
      return "Sat";
      break;
    default:
      return "???";
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the 3-letter month name given the number (1 to 12) of the month.
/// The name will be one of: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec.
/// Capitalization will be corrected. It defaults to "???" if it is unknown.

string TimeStamp::getMonthName( const int &nMonthNum )
{
  switch ( nMonthNum )
  {
    case 1:
      return "Jan";
      break;
    case 2:
      return "Feb";
      break;
    case 3:
      return "Mar";
      break;
    case 4:
      return "Apr";
      break;
    case 5:
      return "May";
      break;
    case 6:
      return "Jun";
      break;
    case 7:
      return "Jul";
      break;
    case 8:
      return "Aug";
      break;
    case 9:
      return "Sep";
      break;
    case 10:
      return "Oct";
      break;
    case 11:
      return "Nov";
      break;
    case 12:
      return "Dec";
      break;
    default:
      return "???";
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Fetches the current time from the system. This can be used to initialize
/// an instance of this object: TimeStamp ts = TimeStamp::now();. It can
/// also be used when just the current itme is needed. UTC is assumed.
/// @return A TimeStamp object containing the current system time.

TimeStamp TimeStamp::now()
{
  timeval tvCurr;

#ifdef WIN32
  FILETIME  ftNow;
  GetSystemTimeAsFileTime ( &ftNow );
  long long nnTime = ( long long ) ftNow.dwHighDateTime << 32;
  nnTime |= ftNow.dwLowDateTime;
  ///  convert from 100 nanosec to 1 usec.
  nnTime /= 10;
  ///  Number of microsec between the beginning of the Windows epoch
  ///  (1 Jan 1601) and the Unix epoch (1 Jan 1970).
  nnTime -= 11644473600000000ULL;
  tvCurr.tv_sec  = ( nnTime / 1000000ULL );
  tvCurr.tv_usec = ( nnTime % 1000000ULL );
#else
  /**
  The  gettimeofday()  function shall obtain the current time, expressed
  as seconds and microseconds since the  Epoch, UTC, and  store  it  in  the
  timeval structure pointed to by tp. The resolution of the system clock
  is unspecified. The gettimeofday() function shall return  0  and  no
  value  shall  be reserved to indicate an error.
  If tzp is not a null pointer, the behavior is unspecified.
  **/
  ::gettimeofday( &tvCurr, NULL );
#endif

  return TimeStamp( tvCurr );
}

////////////////////////////////////////////////////////////////////////////////
/// Creates an iso-formatted time similar to: 193812 based on the currently
/// stored time.
/// @return the iso-formatted number.

string TimeStamp::getFormattedIsoTimeStr() const
{
  string szFormatted = "000000";

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  // transform date and time to broken-down time.
  if ( tmCurr != NULL )
  {
    stringstream ssFormatted;
    ssFormatted
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_hour
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_min
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_sec;
    szFormatted = ssFormatted.str();
  }

  return szFormatted;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the date and time formatted as "Sun Jun 24 19:38:12.234 2007".
/// @return A string formatted with the date and time information.

string TimeStamp::getFormattedStr() const
{
  string szFormatted = "??? ??? 00 00:00:00.000 0000";

  time_t tNumSecs = getTimeValSecs();
  int nNumMillis = getTimeValMicros() / 1000;
  tm *tmCurr = ::gmtime( &tNumSecs );

  // transform date and time to broken-down time.
  if ( tmCurr != NULL )
  {
    stringstream ssFormatted;
    ssFormatted
        << getWeekdayName( tmCurr->tm_wday + 1 )
        << " "
        << getMonthName( tmCurr->tm_mon + 1 )
        << " "
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_mday
        << " "
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_hour
        << ":"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_min
        << ":"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_sec
        << "."
        << setfill( '0' ) << setw( 3 ) << nNumMillis
        << " "
        << setfill( '0' ) << setw( 4 ) << 1900 + tmCurr->tm_year;
    szFormatted = ssFormatted.str();
  }

  return szFormatted;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates an iso-formatted date similar to:
/// YYYY-MM-DDThh:mm:ss.sTZD (eg 1997-07-16T19:20:30.451Z)
/// Based on the current time.
/// @return the iso-formatted string.

string TimeStamp::getFormattedIso8601Str() const
{
  string szFormatted = "0000-00-00T00:00:00.000000Z";

  // It is a bit roundabout, but this is the simpliest way I have found
  // to convert a timeval to a tm.
  time_t tNumSecs = getTimeValSecs();
  int nNumMicros = getTimeValMicros();
  tm *tmCurr = ::gmtime( &tNumSecs );

  // transform date and time to broken-down time.
  if ( tmCurr != NULL )
  {
    stringstream ssFormatted;
    ssFormatted
        << setfill( '0' ) << setw( 4 ) << 1900 + tmCurr->tm_year
        << "-"
        << setfill( '0' ) << setw( 2 ) << 1 + tmCurr->tm_mon
        << "-"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_mday
        << "T"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_hour
        << ":"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_min
        << ":"
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_sec
        << "."
        << setfill( '0' ) << setw( 6 ) << nNumMicros
        << "Z";
    szFormatted = ssFormatted.str();
  }

  return szFormatted;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates an iso-formatted date similar to: 20091230 based on the currently
/// stored time.
/// @return the iso-formatted number.

string TimeStamp::getFormattedIsoDateStr() const
{
  string szFormatted = "00000000";

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  // transform date and time to broken-down time.
  if ( tmCurr != NULL )
  {
    stringstream ssFormatted;
    ssFormatted
        << setfill( '0' ) << setw( 4 ) << 1900 + tmCurr->tm_year
        << setfill( '0' ) << setw( 2 ) << 1 + tmCurr->tm_mon
        << setfill( '0' ) << setw( 2 ) << tmCurr->tm_mday;
    szFormatted = ssFormatted.str();
  }

  return szFormatted;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of milliseconds that have elapsed since this object's
/// internal data was set and the time stamp ts.
/// @return The number of milliseconds that have elapsed since the object
/// was created or updated (see 'update').

double TimeStamp::elapsedMillis( const TimeStamp &ts ) const
{
  TimeStamp tsDiff = ts - *this;
  return tsDiff.getMillis();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of milliseconds that have elapsed since this object's
/// internal data was set and the time stamp ts. Resets the timestamp to
/// the current time if the interval has passsed.
/// @return The number of milliseconds that have elapsed since the object
/// was created or updated (see 'update').

bool TimeStamp::intervalElapsedMillis( const int iInterval )
{
  TimeStamp tsCurrent = TimeStamp::now();
  TimeStamp tsDiff = tsCurrent - *this;
  bool oIntervalElapsed = ( tsDiff.getMillis() >= iInterval );
  if ( oIntervalElapsed )
    *this = tsCurrent;
  return oIntervalElapsed;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of days that have elapsed since this object's
/// internal data was set and the time stamp ts.
/// @return The number of milliseconds that have elapsed since the object
/// was created or updated (see 'update').

double TimeStamp::elapsedDays( const TimeStamp &ts ) const
{
  TimeStamp tsDiff = ts - *this;
  return tsDiff.getDays();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of microseconds since epoch (1970 Jan 1), UTC.
/// @return The number of microseconds since the unix epoch.

double TimeStamp::getMicros() const
{
  return ( static_cast<double>( m_tvCurr.tv_sec ) * 1000000.0 )
         + ( static_cast<double>(  m_tvCurr.tv_usec ) );
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of milliseconds since epoch (1970 Jan 1), UTC.
/// @return The number of milliseconds since the unix epoch.

double TimeStamp::getMillis() const
{
  return ( static_cast<double>( m_tvCurr.tv_sec ) * 1000.0 )
         + ( static_cast<double>(  m_tvCurr.tv_usec ) / 1000.0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the number of days since epoch (1970 Jan 1), UTC.
/// @return The number of days since the unix epoch.

double TimeStamp::getDays() const
{
  // There are 86400 seconds in one day.
  return ( static_cast<double>( m_tvCurr.tv_sec )
           + static_cast<double>(  m_tvCurr.tv_usec ) / 1000000.0 ) / 86400.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the year (1900+) from the current time. If there is
/// an error, numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getYear() const
{
  unsigned int uiYear = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiYear = static_cast<unsigned int>( 1900 + tmCurr->tm_year );
  }

  return uiYear;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the year's month number (1-12) from the current time. If there is
/// an error, std::numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getYearMonth() const
{
  unsigned int uiMonth = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiMonth = static_cast<unsigned int>( tmCurr->tm_mon + 1 );
  }

  return uiMonth;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the day of the month (1-31) from the current time. If there is
/// an error, numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getMonthDay() const
{
  unsigned int uiDay = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiDay = static_cast<unsigned int>( tmCurr->tm_mday );
  }

  return uiDay;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the hour of the day (0-23) from the current time. If there is
/// an error, numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getDayHour() const
{
  unsigned int uiHour = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiHour = static_cast<unsigned int>( tmCurr->tm_hour );
  }

  return uiHour;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the minute of the hour (0-59) from the current time. If there is
/// an error, numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getHourMinute() const
{
  unsigned int uiMinute = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiMinute = static_cast<unsigned int>( tmCurr->tm_min );
  }

  return uiMinute;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the second of the minute (0-60) from the current time. If there is
/// an error, numeric_limits<unsigned int>::max() is returned.

unsigned int TimeStamp::getMinuteSecond() const
{
  unsigned int uiSecond = std::numeric_limits<unsigned int>::max();

  time_t tNumSecs = getTimeValSecs();
  tm *tmCurr = ::gmtime( &tNumSecs );

  if ( tmCurr != NULL )
  {
    uiSecond = static_cast<unsigned int>( tmCurr->tm_sec );
  }

  return uiSecond;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates the millisecond of the second (0-999) from the current time.

unsigned int TimeStamp::getSecondMillisecond() const
{
  unsigned int uiMillisecond = getTimeValMicros() / 1000;

  return uiMillisecond;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrement the time stamp one day.

void TimeStamp::decrementDay()
{
  // There are 86400 seconds in one day.
  m_tvCurr.tv_sec -= 86400;
}

////////////////////////////////////////////////////////////////////////////////
/// Increment the time stamp one day.

void TimeStamp::incrementDay()
{
  // There are 86400 seconds in one day.
  m_tvCurr.tv_sec += 86400;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a string filled with number of microseconds since epoch
/// (1970 Jan 1), UTC. See 'getMicros' for details about the returned value.
/// @return The number of microseconds since the unix epoch.

string TimeStamp::getMicrosStr() const
{
  stringstream ssValue;
  ssValue << getMicros();
  return ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a string filled with number of milliseconds since epoch
/// (1970 Jan 1), UTC. See 'getMillis' for details about the returned value.
/// @return The number of milliseconds since the unix epoch.

string TimeStamp::getMillisStr() const
{
  stringstream ssValue;
  ssValue << getMillis();
  return ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the internal timeval from an MJD number. See 'setMJD" to understand
/// the format of the MJD. The time is in UTC.
/// @param xMJD The Modified Julian Day number. The fractional part holds
/// partial days.

void TimeStamp::fromMJD( const double &xMJD )
{
  ///  adjust this day count to start on 1 Jan 1970.
  double xDays = xMJD - 40587.0;

  ///  convert to a number of seconds.
  long long nnSecs = static_cast<long long>( xDays * 86400.0 );

  ///  get the number of microsecs.
  m_tvCurr.tv_usec = static_cast<long long>(
                       ( xDays * 86400.0 - static_cast<double>( nnSecs ) ) * 1000000.0 );

  m_tvCurr.tv_sec = nnSecs;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the internal time from a ISO 8601 formatted string. UTC is assumed.
/// @param szIso8601 a string like: "2013-09-23T11:41:40.959453Z"

void TimeStamp::fromFormattedIso8601Str( const string &szIso8601 )
{
  /*
    string szIso8601Mod = szIso8601;

    // We need some speed here, so change this to happen in one loop.
    for ( unsigned int ii = 0; ii < szIso8601Mod.size(); ii++ )
    {
      if ( szIso8601Mod[ii] == '-' || szIso8601Mod[ii] == 'T' ||
           szIso8601Mod[ii] == ':' || szIso8601Mod[ii] == '.' ||
           szIso8601Mod[ii] == 'Z' )
        szIso8601Mod[ii] = ' ';
    }
    //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), '-', ' ' );
    //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), 'T', ' ' );
    //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), ':', ' ' );
    //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), '.', ' ' );
    //std::replace( szIso8601Mod.begin(), szIso8601Mod.end(), 'Z', ' ' );

    stringstream ssCurr;
    ssCurr.str( szIso8601Mod );

    tm tmCurr;
    int nYear, nMonth, nDay, nHour, nMinute, nSecond, nMicros;

    ssCurr >> nYear >> nMonth >> nDay >> nHour >> nMinute >> nSecond >> nMicros;
  */
  tm tmCurr;
  int nYear, nMonth, nDay, nHour, nMinute, nSecond, nMicros;
  ::sscanf( szIso8601.c_str(), "%d-%d-%dT%d:%d:%d.%dZ",
            &nYear, &nMonth, &nDay, &nHour, &nMinute, &nSecond, &nMicros );

  tmCurr.tm_year = nYear - 1900;
  tmCurr.tm_mon = nMonth - 1;
  tmCurr.tm_mday = nDay;
  tmCurr.tm_hour = nHour;
  tmCurr.tm_min = nMinute;
  tmCurr.tm_sec = nSecond;

  //JRM changed to timegm from local_timegm
  m_tvCurr.tv_sec = long( timegm( &tmCurr ) );
  m_tvCurr.tv_usec = nMicros;
}

////////////////////////////////////////////////////////////////////////////////
/// Generates a string representing the Modified Julian Day number (MJD).
/// This day count has been adjusted to start on 17 November 1858. This day
/// corresponds to MJD 2400000, and is generally accepted as 0, since the
/// "24" will not change for three centuries to "25".
/// @return A double which contains the day number and any fraction of a day
/// which has elapsed.

double TimeStamp::getMJD() const
{
  /// first, get the number of days (and fractions of a day) since unix epoch.
  double xDays = ( static_cast<double>( m_tvCurr.tv_sec ) +
                   ( static_cast<double>( m_tvCurr.tv_usec ) / 1000000.0 ) ) / 86400.0;

  ///  adjust this day count to start on 17 November 1858. This day corresponds
  ///  to MJD 2400000, and is generally accepted as 0, since the "24" will not
  ///  change for three centuries to "25".
  return xDays + 40587.0;
}

////////////////////////////////////////////////////////////////////////////////
/// 'timegm' is nonstandard at this time, so our own version is implemented

::time_t TimeStamp::local_timegm( ::tm *tmCurr )
{
  ::time_t tUtc;
  char *pcTz;

  // Get the current time zone. This will be null if it does not exist.
  pcTz = ::getenv( "TZ" );

  // Set the time zone to be UTC.
  ::setenv( "TZ", "UTC0", 1 );
  ::tzset();

  // Make a UTC time. (time in the UTC time zone).
  tUtc = ::mktime( tmCurr );

  // Set the time zone back, or delete it if it did not exist.
  if ( pcTz != NULL )
    ::setenv( "TZ", pcTz, 1 );
  else
    ::unsetenv( "TZ" );
  :: tzset();

  return tUtc;
}

////////////////////////////////////////////////////
////////////////////////////
/// Handles streaming the TimeStamp formatted string. See 'getFormattedStr'
/// for details about the format of the string.
/// @param strmOut The stream to be written to.
/// @param tsRhs The TimeStamp to be streamed.
/// @return The modified stream that has been written to.

ostream &operator<< ( ostream &strmOut, const TimeStamp &tsRhs )
{
  strmOut << tsRhs.getFormattedStr();
  return strmOut;
}

////////////////////////////////////////////////////////////////////////////////
