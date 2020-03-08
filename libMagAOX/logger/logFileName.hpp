/** \file logFileName.hpp
  * \brief Declares and defines the logFileName class
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2019-12-16 created by JRM
  */

#ifndef logger_logFileName_hpp
#define logger_logFileName_hpp

#include <map>
#include <set>

#include <mx/ioutils/fileUtils.hpp>

namespace MagAOX
{
namespace logger
{


/// Organize and analyze the name of a log or telemetry file.
class logFileName
{
   
protected:
   std::string m_fullName; ///< The full name of the file, including path
   
   std::string m_appName; ///< The name of the application which wrote the file
   int m_year {0}; ///< The year of the timestamp
   int m_month {0}; ///< The month of the timestamp
   int m_day {0}; ///< The day of the timestamp
   int m_hour {0}; ///< The hour of the timestamp
   int m_minute {0}; ///< The minute of the timestamp
   int m_second {0}; ///< The second of the timestamp
   int m_nsec {0}; ///< The nanosecond of the timestamp
   
   timespecX m_timestamp {0,0};  ///< The timestamp 
   
   std::string m_extension; ///< The extension of the file

   bool m_valid {false}; ///< Whether or not the file parsed correctly and the components are valid
   
public:

   /// Default c'tor
   logFileName();

   /// Construct from a full name
   /** This calls parseName, which parses the input and
     * populates all fields.
     * 
     * On success, sets `m_valid=true`
     * 
     * On error, sets `m_valid=false`
     */ 
   explicit logFileName(const std::string & fullName /**< [in] The new full name of the log (including the path)*/);

   /// Sets the full name
   /** Setting the full name is the only way to set any of the values.  This parses the input and
     * populates all fields.
     * 
     * \returns 0 on sucess, and sets `m_valid=true`
     * \returns -1 on an error, and sets `m_valid=false`
     */ 
   int fullName(const std::string & fullName /**< [in] The new full name of the log (including the path)*/);
   
   /// Assignment operator from string
   /** Sets the full name, which is the only way to set any of the values.  This parses the input and
     * populates all fields.
     * 
     * On success, sets `m_valid=true`
     * 
     * On error, sets `m_valid=false`
     *
     * \returns a reference the `this`
     */
   logFileName & operator=(const std::string & fullName /**< [in] The new full name of the log (including the path)*/);
   
   /// Get the current value of m_fullName
   /**
     * \returns the current value of m_fullName
     */ 
   std::string fullName() const;
   
   /// Get the current value of m_appName
   /**
     * \returns the current value of m_appName
     */ 
   std::string appName() const;
   
   /// Get the current value of m_year
   /**
     * \returns the current value of m_year
     */ 
   int year() const;
   
   /// Get the current value of m_month
   /**
     * \returns the current value of m_month
     */ 
   int month() const;
   
   /// Get the current value of m_day
   /**
     * \returns the current value of m_day
     */ 
   int day() const;
   
   /// Get the current value of m_hour
   /**
     * \returns the current value of m_hour
     */ 
   int hour() const;
   
   /// Get the current value of m_minute
   /**
     * \returns the current value of m_minute
     */ 
   int minute() const;
   
   /// Get the current value of m_second
   /**
     * \returns the current value of m_second
     */ 
   int second() const;
   
   /// Get the current value of m_nsec
   /**
     * \returns the current value of m_nsec
     */ 
   int nsec() const;
   
   /// Get the current value of m_valid
   /**
     * \returns the current value of m_valid
     */ 
   timespecX timestamp() const;
   
   /// Get the current value of
   /**
     * \returns the current value of
     */ 
   std::string extension() const;
   
   /// Get the current value of
   /**
     * \returns the current value of
     */ 
   bool valid() const;

protected:

   /// Parses the `m_fullName` and populates all fields.
   /** 
     * \returns 0 on sucess, and sets `m_valid=true`
     * \returns -1 on an error, and sets `m_valid=false`
     */
   int parseName();
     

};

inline
logFileName::logFileName()
{
   return;
}

inline
logFileName::logFileName(const std::string & fn) : m_fullName {fn}
{
   parseName();
}

inline
int logFileName::fullName(const std::string & fn)
{
   m_fullName = fn;
   return parseName();
}
   
inline
logFileName & logFileName::operator=(const std::string & fn)
{
   fullName(fn);
   
   return *this;
}
   
inline
std::string logFileName::fullName() const
{
   return m_fullName;
}
      
inline
std::string logFileName::appName() const
{
   return m_appName;
}

inline
int logFileName::year() const
{
   return m_year;
}

inline
int logFileName::month() const
{
   return m_month;
}

inline
int logFileName::day() const
{
   return m_day;
}

inline
int logFileName::hour() const
{
   return m_hour;
}

inline
int logFileName::minute() const
{
   return m_minute;
}

inline
int logFileName::second() const
{
   return m_second;
}

inline
int logFileName::nsec() const
{
   return m_nsec;
}

inline
timespecX logFileName::timestamp() const
{
   return m_timestamp;
}

inline
std::string logFileName::extension() const
{
   return m_extension;
}

inline
bool logFileName::valid() const
{
   return m_valid;
}

inline
int logFileName::parseName()
{
   size_t ext = m_fullName.rfind('.');
   
   if(ext == std::string::npos)
   {
      std::cerr << "No extension found in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
   
   m_extension = m_fullName.substr(ext+1);
   
   size_t ts = m_fullName.rfind('_', ext);
   
   if(ts == std::string::npos)
   {
      std::cerr << "No app name found in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
   
   size_t ps = m_fullName.rfind('/', ts);
   
   if(ps == std::string::npos) ps = 0;
   else ++ps;
   
   m_appName = m_fullName.substr(ps, ts-ps);
   
   ++ts;
   if(ext-ts != 23)
   {
      std::cerr << "Timestamp wrong size in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
      
   std::string tstamp = m_fullName.substr(ts, ext-ts);

   m_year = std::stoi(tstamp.substr(0,4));
   m_month = std::stoi(tstamp.substr(4,2));
   m_day = std::stoi(tstamp.substr(6,2));
   m_hour = std::stoi(tstamp.substr(8,2));
   m_minute = std::stoi(tstamp.substr(10,2));
   m_second = std::stoi(tstamp.substr(12,2));
   m_nsec = std::stoi(tstamp.substr(14,9));
 
   tm tmst;
   tmst.tm_year = m_year-1900;
   tmst.tm_mon = m_month - 1;
   tmst.tm_mday = m_day;
   tmst.tm_hour = m_hour;
   tmst.tm_min = m_minute;
   tmst.tm_sec = m_second;
   
   m_timestamp.time_s = timegm(&tmst);
   m_timestamp.time_ns = m_nsec;
   
   m_valid = true;
   
   return 0;
}


/// Sort predicate for logFileNames
/** Sorting is on 'fullName()'
  */
struct compLogFileName
{
   /// Comparison operator.
   /** \returns true if a < b
     * \returns false otherwise
     */ 
   bool operator()( const logFileName & a, 
                    const logFileName & b
                  )
   {
      return (a.fullName() < b.fullName());
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logFileName_hpp
