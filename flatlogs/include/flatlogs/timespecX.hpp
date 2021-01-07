/** \file timespecX.hpp 
  * \brief A fixed-width timespec structure and utilities.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  * 
  * History:
  * - 2017-06-27 created by JRM
  * - 2018-08-17 moved to flatlogs
  */ 

#ifndef flatlogs_timespecX_hpp
#define flatlogs_timespecX_hpp

#include <cstdint>
#include <iostream>
#include <cmath>
#include <ctime>

#include "logDefs.hpp"

namespace flatlogs
{

///A fixed-width timespec structure.
/** To ensure that binary encoding of time is stable regardless of environment, we use a custom timespec
  * composed of fixed-width types.
  * 
  * \note This is NOT binary compatible with plain timespec.  Use the provided conversions.
  * 
  * \ingroup flatlogs_time
  * 
  */
struct timespecX 
{
   secT time_s {0}; ///< Time since the Unix epoch
   nanosecT time_ns {0}; ///< Nanoseconds.  

   ///Default c'tor
   timespecX()
   {
   }
   
   ///Construct with time values
   timespecX( secT s, nanosecT ns ) : time_s {s}, time_ns {ns}
   {
   }
   
   ///Construct from timespec
   timespecX( const timespec & ts)
   {
      operator=(ts);
   }
   
   ///Convert a native timespec to a timespecX.
   /**
     * \returns this reference, if values are 0 and 0 then the input was too big or negative. 
     */
   timespecX & operator=( const timespec & ts /**< [in] the native timespec from which to get values */)
   {
      if(ts.tv_sec < 0 || ts.tv_sec > 4294967295) ///\todo make this use minval and maxval
      {
         time_s = 0;
         time_ns = 0;
      }
      else 
      {
         time_s = ts.tv_sec;
         time_ns = ts.tv_nsec;
      }
      
      return *this;
   }
   
   ///Get a native timespec from this custom one.
   timespec getTimespec()
   {
      struct timespec ts;
      
      ts.tv_sec = time_s;
      ts.tv_nsec = time_ns;
      
      return ts;
   }
   
   ///Fill the the timespecX with the current time.
   /** This is based on the usual clock_gettime.  clockid_t is a template parameter 
     * since we probaby always want CLOCK_REALTIME, but if we don't for some reason
     * it will be passed in the same order as in clock_gettime.
     * 
     * \tparam clk_id specifies the type.
     */ 
   template<clockid_t clk_id=CLOCK_REALTIME>
   void gettime()
   {
      struct timespec ts;
      clock_gettime(clk_id, &ts);
      (*this) = ts; //see operator=
   }
   
   ///Get the filename timestamp for this timespecX.
   /** Fills in a string with the timestamp encoded as
     * \verbatim
       YYYYMMDDHHMMSSNNNNNNNNN
       \endverbatim
     *
     */ 
   int timeStamp(std::string & tstamp /**< [out] the string to hold the formatted time */)
   {
      tm uttime;//The broken down time.
   
      time_t t0 = time_s;

      if(gmtime_r(&t0, &uttime) == 0)
      {
         std::cerr << "Error getting UT time (gmtime_r returned 0). At: " <<  __FILE__ << " " << __LINE__ << "\n";
         return -1;
      }
   
      char buffer[48];

      snprintf(buffer, sizeof(buffer), "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year+1900, uttime.tm_mon+1, uttime.tm_mday, uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(time_ns)); //casting in case we switch type of time_ns.
   
      tstamp = buffer;

      return 0;   
   }
   
    ///Get the filname timestamp for this timespecX.
   /** Returns a string with the timestamp encoded as
     * \verbatim
       YYYYMMDDHHMMSSNNNNNNNNN
       \endverbatim
     *
     */ 
   std::string timeStamp()
   {
      std::string tstamp;
      timeStamp(tstamp);
      return tstamp;
   }
   
   /// Get a date-time string in ISO 8601 format for timespecX
   /** Returns a string in the ISO 8601 format:
     * \verbatim
       YYYY-MM-DDTHH:MM:SS.SSSSSSSSS
       \endverbatim
     *
     *
     * \retval std::string containing the formated date/time 
     * 
     */ 
   std::string ISO8601DateTimeStrX()
   {
      tm bdt; //broken down time
      time_t tt = time_s;
      gmtime_r( &tt, &bdt);
      
      char tstr1[25];
      
      strftime(tstr1, 25, "%FT%H:%M:%S", &bdt);
      
      char tstr2[11];
      
      snprintf(tstr2, 11, ".%09i", static_cast<int>(time_ns)); //casting in case we switch to int64_t
      
      return std::string(tstr1) + std::string(tstr2);
   }
   
   std::string ISO8601DateTimeStr2MinX()
   {
      tm bdt; //broken down time
      time_t tt = time_s;
      gmtime_r( &tt, &bdt);
      
      char tstr1[25];
      
      strftime(tstr1, 25, "%FT%H:%M", &bdt);
      
      return std::string(tstr1);
   }
   
   /// Get a date-time string with just the second for timespecX
   /** Returns a string in the format:
     * \verbatim
       SS.SSS
       \endverbatim
     * which is useful for real-time streams of log entries.
     *
     * \retval std::string containing the formated date/time 
     * 
     */ 
   std::string secondStrX()
   {
      tm bdt; //broken down time
      time_t tt = time_s;
      gmtime_r( &tt, &bdt);
      
      char tstr1[5];
      
      strftime(tstr1, sizeof(tstr1), "%S", &bdt);
      
      char tstr2[5];
      
      snprintf(tstr2, sizeof(tstr2), ".%02i", static_cast<int>(time_ns)); //casting in case we switch to int64_t
      
      return std::string(tstr1) + std::string(tstr2);
   }
   
   /// Get the minute from a timespecX
   /** 
     *
     * \returns the minute part. 
     * 
     */ 
   int minute()
   {
      tm bdt; //broken down time
      time_t tt = time_s;
      gmtime_r( &tt, &bdt);
      
      return bdt.tm_min;
   }
   
   /// Get the time as a double from a timespecX
   /** 
     *
     * \returns the time as a double. 
     * 
     */
   double asDouble()
   {
      return ((double) time_s) + ((double) time_ns)/1e9;
   }
   
} __attribute__((packed));



/// TimespecX comparison operator \< (see caveats)
/** Caveats:
  * - If the inputs are in UTC (or similar scale) this does not account for leap seconds
  * - Assumes that the `time_ns` field does not exceed 999999999 nanoseconds
  * 
  * \returns true if tsL is earlier than tsR
  * \returns false otherwise
  * 
  * \ingroup timeutils_tscomp
  */  
inline
bool operator<( timespecX const& tsL, ///< [in] the left hand side of the comparison
                timespecX const& tsR  ///< [in] the right hand side of the comparison 
              )
{
   return ( ((tsL.time_s == tsR.time_s) && (tsL.time_ns < tsR.time_ns)) || (tsL.time_s < tsR.time_s));   
}

/// TimespecX comparison operator \> (see caveats)
/** Caveats:
  * - If the inputs are in UTC (or similar scale) this does not account for leap seconds
  * - Assumes that the `time_ns` field does not exceed 999999999 nanoseconds
  * 
  * \returns true if tsL is later than tsR
  * \returns false otherwise
  * 
  * \ingroup timeutils_tscomp
  */
inline
bool operator>( timespecX const& tsL, ///< [in] the left hand side of the comparison
                timespecX const& tsR  ///< [in] the right hand side of the comparison 
              )
{
   return ( ((tsL.time_s == tsR.time_s) && (tsL.time_ns > tsR.time_ns)) || (tsL.time_s > tsR.time_s)); 
}

/// TimespecX comparison operator == (see caveats)
/** Caveats:
  * - If the inputs are in UTC (or similar scale) this does not account for leap seconds
  * - Assumes that the `time_ns` field does not exceed 999999999 nanoseconds
  * 
  * \returns true if tsL is exactly the same as tsR
  * \returns false otherwise
  * 
  * \ingroup timeutils_tscomp
  */
inline
bool operator==( timespecX const& tsL, ///< [in] the left hand side of the comparison
                 timespecX const& tsR  ///< [in] the right hand side of the comparison 
               )
{
   return ( (tsL.time_s == tsR.time_s)  &&  (tsL.time_ns == tsR.time_ns) );
}

/// TimespecX comparison operator \<= (see caveats)
/** Caveats:
  * - If the inputs are in UTC (or similar scale) this does not account for leap seconds
  * - Assumes that the `time_ns` field does not exceed 999999999 nanoseconds.  
  * 
  * \returns true if tsL is earlier than or exactly equal to tsR
  * \returns false otherwise
  * 
  * \ingroup timeutils_tscomp
  */
inline
bool operator<=( timespecX const& tsL, ///< [in] the left hand side of the comparison
                 timespecX const& tsR  ///< [in] the right hand side of the comparison 
               )
{
   return ( tsL < tsR || tsL == tsR );   
}

/// TimespecX comparison operator \>= (see caveats)
/** Caveats:
  * - If the inputs are in UTC (or similar scale) this does not account for leap seconds
  * - Assumes that the `time_ns` field does not exceed 999999999 nanoseconds
  * 
  * \returns true if tsL is exactly equal to or is later than tsR
  * \returns false otherwise
  * 
  * \ingroup timeutils_tscomp
  */
inline
bool operator>=( timespecX const& tsL, ///< [in] the left hand side of the comparison
                 timespecX const& tsR  ///< [in] the right hand side of the comparison 
               )
{
   return ( tsL > tsR || tsL == tsR );   
}

inline
timespecX meanTimespecX( timespecX ts1, timespecX ts2)
{
   double means = ((double)(ts1.time_s + ts2.time_s))/2.0;
   double meanns = ((double)(ts1.time_ns + ts2.time_ns))/2.0;
   
   ts1.time_s = std::floor(means);
   ts1.time_ns = std::round(meanns);
   
   if( means != floor(means) )
   {
      ts1.time_ns += 5e8;
      
      if(ts1.time_ns >= 1e9)
      {
         ts1.time_s += 1;
         ts1.time_ns -= 1e9;
      }
   }
   
   return ts1;
}

///Convert a timespecX to a native timespec
/**
  * \ingroup flatlogs_time
  */ 
inline
void timespecFromX ( timespec & ts, ///< [out] the native timespec to set
                     const timespecX & tsX ///< [in] the fixed-width timespec from which to get values
                   ) 
{
   ts.tv_sec = tsX.time_s;
   ts.tv_nsec = tsX.time_ns;
   
}

///Fill in a timespecX with the current time.
/** This is based on the usual clock_gettime.  clockid_t is a template parameter 
  * since we probaby always want CLOCK_REALTIME, but if we don't for some reason
  * it will be passed in the same order as in clock_gettime.
  * 
  * \tparam clk_id specifies the type.
  * 
  * \ingroup flatlogs_time
  * 
  */ 
template<clockid_t clk_id=CLOCK_REALTIME>
void clock_gettimeX( timespecX & tsX /**< [out] the fixed-width timespec to populate */)
{
   tsX.gettime<clk_id>();
}

}//namespace flatlogs


#endif //flatlogs_timespecX_hpp

