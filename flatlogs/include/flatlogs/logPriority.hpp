/** \file logPriority.hpp
  * \brief The MagAO-X logger log priority levels.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  * 
  * History:
  * - 2017-07-22 created by JRM
  * - 2018-08-17 renamed and moved to flatlogs
  */
#ifndef flatlogs_logPriority_hpp
#define flatlogs_logPriority_hpp

#include <mx/ioutils/stringUtils.hpp>

#include "logDefs.hpp"

namespace flatlogs
{

/// The log priority codes.  These control if logs are stored on disk and how they are presented to users.
/** This is a scoping namespace for log priority codes.
  * We do not use the enum class feature since it does not have automatic integer conversion.
  * \ingroup logPrio
  */
namespace logPrio
{
    
  
  /// Normal operations of the entire system should be shut down immediately. 
  constexpr static logPrioT LOG_EMERGENCY = 0;  

  /// This should only be used if some action is required by operators to keep the system safe.
  constexpr static logPrioT LOG_ALERT = 1;
  
  /// The process can not continue and will shut down (fatal)
  constexpr static logPrioT LOG_CRITICAL = 2;
  
  /// An error has occured which the software will attempt to correct.
  constexpr static logPrioT LOG_ERROR = 3;
  
  /// A condition has occurred which may become an error, but the process continues.
  constexpr static logPrioT LOG_WARNING = 4;
  
  /// A normal but significant condition
  constexpr static logPrioT LOG_NOTICE = 5;
  
  /// Informational.  The info log level is the lowest level recorded during normal operations. 
  constexpr static logPrioT LOG_INFO = 6;
  
  /// Used for debugging
  constexpr static logPrioT LOG_DEBUG = 7;
  
  /// Used for debugging, providing a 2nd level.
  constexpr static logPrioT LOG_DEBUG2 = 8;
   
  /// A telemetry recording
  constexpr static logPrioT LOG_TELEM = 64;
  
  /// Used to denote "use the default level for this log type".
  constexpr static logPrioT LOG_DEFAULT = std::numeric_limits<logPrioT>::max() - 1;
  
  /// Used to denote an unkown log type for internal error handling.
  constexpr static logPrioT LOG_UNKNOWN = std::numeric_limits<logPrioT>::max();
  
};

///Get the string representation of a log priority
/** \ingroup logPrio
  */
inline
std::string priorityString( logPrioT & prio /**< [in] the log priority */)
{
   switch( prio )
   {
      case logPrio::LOG_EMERGENCY:
         return "EMER";
      case logPrio::LOG_ALERT:
         return "ALRT";
      case logPrio::LOG_CRITICAL:
         return "CRIT";
      case logPrio::LOG_ERROR:
         return "ERR ";
      case logPrio::LOG_WARNING:
         return "WARN";
      case logPrio::LOG_NOTICE:
         return "NOTE";
      case logPrio::LOG_INFO:
         return "INFO";
      case logPrio::LOG_DEBUG:
         return "DBG ";   
      case logPrio::LOG_DEBUG2:
         return "DBG2";   
      case logPrio::LOG_DEFAULT:
         return "DEF?";
      case logPrio::LOG_TELEM:
         return "TELM";   
      default:
         return "UNK?";
   }
}

///Get the log priority from a string, which might have the number or the name
/** Evaluates the input string to find the closest match to the log level names.
  * If the first non-whitespace character is a digit, then the string is treated as an integer
  * and converted.  If the first non-whitespace character is not a digit, then the string is
  * converted to upper case and the log level is
  * determined using the minimum number of characters.  That is
  * - EM = EMERGENCY
  * - A  = ALERT
  * - C  = CRITICAL
  * - ER = ERROR
  * - W = WARNING
  * - N = NOTICE
  * - I = INFO, returns 3
  * - D, D1, DBG, DBG1, DEBUG = DEBUG
  * - D2, DBG2, DEBUG2 = DEBUG2
  * - DEF = DEFAULT
  * - T = TELEM
  *
  * \returns the log priority value if parsing is successful.
  * \returns logPrio::LOG_UNKNOWN if parsing is not successful.
  *
  * \ingroup loglevels
  */
inline
logPrioT logLevelFromString( const std::string & str )
{
   std::string s = str;

   //Remove all whitespace
   s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

   if(s.size()==0) return logPrio::LOG_UNKNOWN;

   if( isdigit(s[0]) )
   {
      logPrioT l = atoi( s.c_str());

      return l;
   }

   //Convert to upper case
   for(size_t i=0; i< s.size(); ++i) s[i] = ::toupper(s[i]);
   
   if(s[0] == 'A') return logPrio::LOG_ALERT;
   if(s[0] == 'C') return logPrio::LOG_CRITICAL;
   if(s[0] == 'W') return logPrio::LOG_WARNING;
   if(s[0] == 'N') return logPrio::LOG_NOTICE;
   if(s[0] == 'I') return logPrio::LOG_INFO;

   if(s[0] == 'E') 
   {
      if(s.size() == 1) return logPrio::LOG_UNKNOWN;
      
      if(s[1] == 'M') return logPrio::LOG_EMERGENCY;
      if(s[2] == 'R') return logPrio::LOG_ERROR;
      
      return logPrio::LOG_UNKNOWN;
   }
      
   if(s[0] == 'D')
   {
      //Accept D by itself for DEBUG
      if(s.size()==1) return logPrio::LOG_DEBUG;

      //We accept D1 or D2
      if(s[1] == '1') return logPrio::LOG_DEBUG;
      if(s[1] == '2') return logPrio::LOG_DEBUG2;

      if(s.size()<3) return logPrio::LOG_UNKNOWN;

      //We accept DBG and DBG2
      if( s[1] == 'B' && s[2] == 'G' )
      {
         if(s.size() == 3) return logPrio::LOG_DEBUG;

         if(s.size() > 3)
         {
            if( s[3] == '2') return logPrio::LOG_DEBUG2;
         }

         return logPrio::LOG_UNKNOWN;
      }

      //Anything that starts with DEF
      if (s[1] == 'E' && s[2] == 'F') return logPrio::LOG_DEFAULT;

      //Now check for DEBUG and DEBUG2
      if( s.size() >=5 )
      {
         if( s.substr(0,5) != "DEBUG" ) return logPrio::LOG_UNKNOWN;

         if(s.size() > 5 )
         {
            if( s[5] == '2') return logPrio::LOG_DEBUG2;

            return logPrio::LOG_UNKNOWN;
         }

         return logPrio::LOG_DEBUG;
      }

   }

   if(s[0] == 'T')
   {
      return logPrio::LOG_TELEM;
   }
   
   return logPrio::LOG_UNKNOWN;
}

}// namespace flatlogs

#endif //flatlogs_logPriority_hpp
