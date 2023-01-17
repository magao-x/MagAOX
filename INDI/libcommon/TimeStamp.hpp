/// TimeStamp.hpp
///
/// @author Paul Grenz
///
/// The TimeStamp class is the object which wraps a timeval struct. The class
/// facilitates subtracting two DataTimes to get an interval (a difference).
/// The class can also be used to transport a time and date in a platform
/// independant way.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_TIME_STAMP_HPP
#define PCF_TIME_STAMP_HPP

#include <string>
#ifdef WIN32
#include <winsock2.h>
#else
#include <iosfwd>
#include <sys/time.h>
#endif

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class TimeStamp
{
  // Constructor/destructor/copy constructor.
  public:
    TimeStamp();
    /// Standard constructor from a timeval.
    TimeStamp( const timeval &tv );
    /// Standard constructor from a TimeStamp.
    TimeStamp( const TimeStamp &ts );
    /// Standard constructor from a number of milliseconds.
    TimeStamp( const int &nMillis );
    /// Standard constructor from a year, month, day, hour (24), min, sec.
    TimeStamp( const int &nYear, const int &nMonth, const int &nDay,
               const int &nHour, const int &nMinute, const int &nSecond );
    virtual ~TimeStamp();

  // Operators
  public:
    const TimeStamp &operator= ( const TimeStamp &tsRhs );
    const TimeStamp &operator= ( const timeval &tv );
    const TimeStamp &operator= ( const int &nMillis );
    TimeStamp operator- ( const TimeStamp &tsRhs ) const;
    TimeStamp operator+ ( const TimeStamp &tsRhs ) const;
    bool operator> ( const TimeStamp &tsRhs ) const;
    bool operator>= ( const TimeStamp &tsRhs ) const;
    bool operator< ( const TimeStamp &tsRhs ) const;
    bool operator<= ( const TimeStamp &tsRhs ) const;
    bool operator==( const TimeStamp &tsRhs ) const;

  // Methods
  public:
    /// Decrement the time stamp one day.
    void decrementDay();
    /// Returns the number of days that have elapsed since this
    /// object was created, or between this time stamp and ts.
    double elapsedDays( const TimeStamp &ts = TimeStamp::now() ) const;
    /// Returns the number of milliseconds that have elapsed since this
    /// object was created, or between this time stamp and ts.
    double elapsedMillis( const TimeStamp &ts = TimeStamp::now() ) const;
    /// Returns a flag indicating if the specified number of milliseconds
    /// have elapsed since this timestamp was set.  Resets the timestamp to
    /// the current time if the interval has passsed.
    bool intervalElapsedMillis( const int iInterval );
    /// Sets the internal time from a ISO 8601 formatted string.
    void fromFormattedIso8601Str( const std::string &szIso8601 );
    /// Sets the internal time from a Modified Julian Day number.
    void fromMJD( const double &xMJD );
    /// Generates the hour of the day (0-23) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getDayHour() const;
    /// Returns the number of days since epoch (1970 Jan 1).
    double getDays() const;
    /// Returns the date and time formatted as "Sun Jun 24 19:38:12.234 2007".
    std::string getFormattedStr() const;
    /// Creates an iso-formatted date similar to:
    /// YYYY-MM-DDThh:mm:ss.sTZD (eg 1997-07-16T19:20:30.451243Z)
    /// Based on the current time.
    std::string getFormattedIso8601Str() const;
    /// Creates an iso-formatted date similar to: 20091230 based on the
    /// currently stored date.
    std::string getFormattedIsoDateStr() const;
    /// Creates an iso-formatted time similar to: 193812 based on the
    /// currently stored time.
    std::string getFormattedIsoTimeStr() const;
    /// Generates the minute of the hour (0-59) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getHourMinute() const;
    /// Returns the number of microseconds since epoch (1970 Jan 1).
    double getMicros() const;
    /// Returns, as a string, the number of microseconds since epoch (1970 Jan 1).
    std::string getMicrosStr() const;
    /// Returns the number of milliseconds since epoch (1970 Jan 1).
    double getMillis() const;
    /// Returns, as a string, the number of milliseconds since epoch (1970 Jan 1).
    std::string getMillisStr() const;
    /// Generates the second of the minute (0-60) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getMinuteSecond() const;
    /// Returns the Modified Julian Day number (days since 1858 Nov 17).
    double getMJD() const;
    /// Generates the year's month number (1-12) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getYearMonth() const;
    /// Generates the day of the month (1-31) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getMonthDay() const;
    /// Returns the month number given the name.
    static int getMonthNumber( const std::string &szMonth );
    /// Returns the 3-letter weekday name given the number (1 to 7) of the weekday.
    static std::string getWeekdayName( const int &nWeekdayNum );
    /// Returns the 3-letter month name given the number (1 to 12) of the month.
    static std::string getMonthName( const int &nMonthNum );
    /// Generates the millisecond of the second (0-999) from the current time.
    unsigned int getSecondMillisecond() const;
    /// Returns the microsecond part of the time val struct.
    inline int getTimeValMicros() const
    {
      return m_tvCurr.tv_usec;
    }
    /// Returns the seconds part of the time val struct.
    inline int getTimeValSecs() const
    {
      return m_tvCurr.tv_sec;
    }
    /// Returns the underlying timeval struct.
    const timeval &getTimeVal() const;
    /// Increment the time stamp one day.
    void incrementDay();
    /// Fetches the current time and date from the system.
    static TimeStamp now();
    /// Generates the year (1900+) from the current time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getYear() const;

  private:
    /// 'timegm' is nonstandard at this time, so our own version is implemented.
    ::time_t local_timegm( ::tm *tmCurr );

  // Variables
  private:
    /// The info about the stored time.
    timeval m_tvCurr;

}; // Class TimeStamp
} // Namespace pcf

std::ostream &operator<<( std::ostream &strmOut, const pcf::TimeStamp &tsRhs );

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_TIME_STAMP_HPP
