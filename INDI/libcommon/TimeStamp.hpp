/** \file TimeStamp.hpp
 * \brief Declare the timeval wrapper used for INDI timestamps and intervals.
 */
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
/// Wraps a timeval to represent a UTC time or an interval.
/** Subtracting two TimeStamps gives an interval. Formatting and calendar conversions use reentrant
 * UTC routines, so independent instances may be used from different threads.
 */
class TimeStamp
{
    // Constructor/destructor/copy constructor.
  public:
    /// Construct with the current time.
    TimeStamp();

    /// Standard constructor from a timeval.
    TimeStamp( const timeval &tv /**< [in] Time since the epoch. */ );

    /// Copy constructor.
    TimeStamp( const TimeStamp &ts /**< [in] TimeStamp to copy. */ );

    /// Standard constructor from a number of milliseconds.
    TimeStamp( const int &nMillis /**< [in] Milliseconds since the epoch, or an interval. */ );

    /// Construct from UTC calendar fields.
    TimeStamp( const int &nYear /**< [in] 4-digit year. */,
               const int &nMonth /**< [in] Month, 1 to 12. */,
               const int &nDay /**< [in] Day of the month, 1 to 31. */,
               const int &nHour /**< [in] Hour, 0 to 23. */,
               const int &nMinute /**< [in] Minute, 0 to 59. */,
               const int &nSecond /**< [in] Second, 0 to 60. */ );

    /// Destructor.
    virtual ~TimeStamp();

    // Operators
  public:
    /// Assign from another TimeStamp.
    const TimeStamp &operator=( const TimeStamp &tsRhs /**< [in] TimeStamp to copy. */ );

    /// Assign from a timeval.
    const TimeStamp &operator=( const timeval &tv /**< [in] Time since the epoch. */ );

    /// Assign from a number of milliseconds.
    const TimeStamp &operator=( const int &nMillis /**< [in] Milliseconds since the epoch, or an interval. */ );

    /// Return the interval from tsRhs to this time.
    TimeStamp operator-( const TimeStamp &tsRhs /**< [in] Earlier time. */ ) const;

    /// Return this time advanced by an interval.
    TimeStamp operator+( const TimeStamp &tsRhs /**< [in] Interval to add. */ ) const;

    /// Return true if this time is later than tsRhs.
    bool operator>( const TimeStamp &tsRhs /**< [in] Time to compare. */ ) const;

    /// Return true if this time is not earlier than tsRhs.
    bool operator>=( const TimeStamp &tsRhs /**< [in] Time to compare. */ ) const;

    /// Return true if this time is earlier than tsRhs.
    bool operator<( const TimeStamp &tsRhs /**< [in] Time to compare. */ ) const;

    /// Return true if this time is not later than tsRhs.
    bool operator<=( const TimeStamp &tsRhs /**< [in] Time to compare. */ ) const;

    /// Return true if both times are equal.
    bool operator==( const TimeStamp &tsRhs /**< [in] Time to compare. */ ) const;

    // Methods
  public:
    /// Decrement the time stamp one day.
    void decrementDay();

    /// Returns the number of days that have elapsed since this
    /// object was created, or between this time stamp and ts.
    double elapsedDays( const TimeStamp &ts = TimeStamp::now() /**< [in] End of the interval. */ ) const;

    /// Returns the number of milliseconds that have elapsed since this
    /// object was created, or between this time stamp and ts.
    double elapsedMillis( const TimeStamp &ts = TimeStamp::now() /**< [in] End of the interval. */ ) const;

    /// Returns a flag indicating if the specified number of milliseconds
    /// have elapsed since this timestamp was set.  Resets the timestamp to
    /// the current time if the interval has passed.
    bool intervalElapsedMillis( const int iInterval /**< [in] Interval in milliseconds. */ );

    /// Sets the internal time from a ISO 8601 formatted string.
    void
    fromFormattedIso8601Str( const std::string &szIso8601 /**< [in] UTC time, e.g. 1997-07-16T19:20:30.451243Z. */ );

    /// Sets the internal time from a Modified Julian Day number.
    void fromMJD( const double &xMJD /**< [in] Modified Julian Day number. */ );

    /// Generates the hour of the day (0-23) from the stored time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getDayHour() const;

    /// Returns the number of days since epoch (1970 Jan 1).
    double getDays() const;

    /// Returns the date and time formatted as "Sun Jun 24 19:38:12.234 2007".
    std::string getFormattedStr() const;

    /// Creates an iso-formatted date similar to:
    /// YYYY-MM-DDThh:mm:ss.sTZD (eg 1997-07-16T19:20:30.451243Z)
    /// Based on the stored time.
    std::string getFormattedIso8601Str() const;

    /// Creates an iso-formatted date similar to: 20091230 based on the
    /// currently stored date.
    std::string getFormattedIsoDateStr() const;

    /// Creates an iso-formatted time similar to: 193812 based on the
    /// currently stored time.
    std::string getFormattedIsoTimeStr() const;

    /// Generates the minute of the hour (0-59) from the stored time.
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

    /// Generates the second of the minute (0-60) from the stored time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getMinuteSecond() const;

    /// Returns the Modified Julian Day number (days since 1858 Nov 17).
    double getMJD() const;

    /// Generates the year's month number (1-12) from the stored time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getYearMonth() const;

    /// Generates the day of the month (1-31) from the stored time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getMonthDay() const;

    /// Returns the month number given the name.
    static int getMonthNumber( const std::string &szMonth /**< [in] Month name. */ );

    /// Returns the 3-letter weekday name given the number (1 to 7) of the weekday.
    static std::string getWeekdayName( const int &nWeekdayNum /**< [in] Weekday number, 1 to 7. */ );

    /// Returns the 3-letter month name given the number (1 to 12) of the month.
    static std::string getMonthName( const int &nMonthNum /**< [in] Month number, 1 to 12. */ );

    /// Generates the millisecond of the second (0-999) from the stored time.
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

    /// Generates the year (1900+) from the stored time.
    /// If there is an error, numeric_limits<unsigned int>::max() is returned.
    unsigned int getYear() const;

    // Variables
  private:
    /// The info about the stored time.
    timeval m_tvCurr;

}; // Class TimeStamp
} // Namespace pcf

/// Write the TimeStamp formatted by getFormattedStr() to a stream.
std::ostream &operator<<( std::ostream         &strmOut /**< [in,out] Destination stream. */,
                          const pcf::TimeStamp &tsRhs /**< [in] TimeStamp to write. */ );

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_TIME_STAMP_HPP
