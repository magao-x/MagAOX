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

#include <flatlogs/flatlogs.hpp>

namespace MagAOX
{
namespace logger
{


/// Organize and analyze the name of a log or telemetry file.
class logFileName
{
   
protected:
   std::string m_fullName; ///< The full name of the file, including path
   std::string m_baseName; ///< The base name of the file, not including path

   std::string m_appName; ///< The name of the application which wrote the file
   int m_year {0}; ///< The year of the timestamp
   int m_month {0}; ///< The month of the timestamp
   int m_day {0}; ///< The day of the timestamp
   int m_hour {0}; ///< The hour of the timestamp
   int m_minute {0}; ///< The minute of the timestamp
   int m_second {0}; ///< The second of the timestamp
   int m_nsec {0}; ///< The nanosecond of the timestamp
   
   flatlogs::timespecX m_timestamp {0,0};  ///< The timestamp 
   
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
   
   /// Get the current value of m_baseName
   /**
     * \returns the current value of m_baseName
     */ 
   std::string baseName() const;

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
   flatlogs::timespecX timestamp() const;
   
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
                  ) const
   {
      return (a.baseName() < b.baseName());
   }
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logFileName_hpp
