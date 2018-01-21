/** \file logLevels.hpp 
  * \brief The MagAO-X logger log levels.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-07-22 created by JRM
  */ 
#ifndef logger_logLevels_hpp
#define logger_logLevels_hpp

#include <mx/stringUtils.hpp>

namespace MagAOX
{
namespace logger
{

///The type of the log level code.
typedef int8_t logLevelT;
   
///The log level codes.  These control if logs are stored on disk and how they are presented to users.
enum logLevels : logLevelT { MAXLEVEL = 8, // used only for error checking.
                             FATAL = 7, ///< A fatal error, this should only be used if the process will be shutdown as a result.
                             CRITICAL = 6, ///< A critical error, this should only be used if some action is required by operators.
                             ERROR = 5, ///< An error.
                             WARNING = 4, ///< A warning.  
                             INFO = 3, ///< The info log level is the lowest level recorded during normal operations.
                             DEBUG2 = 2,  ///< 2nd lowest priority log, used for debugging.
                             DEBUG  = 1,  ///< Lowest priority log, used for debugging.
                             DEFAULT = 0, ///< Uses the logType default.
                             TELEMETRY = -1, ///< Telemetry logs are handled differently.
                             UNKNOWN = -2 ///<For indicating errors
                           };
 
///Get the string representation of a log level
std::string levelString( logLevelT & lvl )
{
   switch( lvl )
   {
      case logLevels::FATAL:
         return "FAT ";
      case logLevels::CRITICAL:
         return "CRIT";
      case logLevels::ERROR:
         return "ERR ";
      case logLevels::WARNING:
         return "WARN";
      case logLevels::INFO:
         return "INFO";
      case logLevels::DEBUG2:
         return "DBG2";
      case logLevels::DEBUG:
         return "DBG ";
      case logLevels::DEFAULT:
         return "DEF?";
      case logLevels::TELEMETRY:
         return "TEL ";
      default:
         return "UNK?";
   }
}

///Get the log level from a string, which might have the number or the name 
/** Evaluates the input string to find the closest match to the log level names.
  * If the first non-whitespace character is a digit, then the string is treated as an integer
  * and converted.  If the first non-whitespace character is not a digit, then the string is
  * converted to upper case and the log level is
  * determined using the minimum number of characters.  That is
  * - F = FATAL, returns 7
  * - C = CRITICAL, returns 6
  * - E = ERROR, returns 5
  * - W = WARNING, returns 4
  * - I = INFO, returns 3
  * - D, D1, DBG, DBG1, DEBUG = DEBUG, returns 2
  * - D2, DBG2, DEBUG2 = DEBUG2, returns 1
  * - DEF = DEFAULT, returns 0
  * - T = TELEMETRY, returns -1
  * 
  * \returns the logLevels value if parsing is successful.
  * \returns logLevels::UNKNOWN (-2) if parsing is not successful.
  */
logLevelT logLevelFromString( const std::string & str )
{
   std::string s;
   
   mx::removeWhiteSpace(s, str);
   
   
   if(s.size()==0) return logLevels::UNKNOWN;
   
   if( isdigit(s[0]) )
   {
      logLevelT l = atoi( s.c_str());
      
      if( l > logLevels::UNKNOWN && l < logLevels::MAXLEVEL) return l;
      else return logLevels::UNKNOWN;
   }
   
   s = mx::toUpper(s);

   
   
   if(s[0] == 'F') return logLevels::FATAL;
   if(s[0] == 'C') return logLevels::CRITICAL;
   if(s[0] == 'E') return logLevels::ERROR;
   if(s[0] == 'W') return logLevels::WARNING;
   if(s[0] == 'I') return logLevels::INFO;
   if(s[0] == 'T') return logLevels::TELEMETRY;
   
   if(s[0] == 'D')
   {
      //Accept D by itself for DEBUG
      if(s.size()==1) return logLevels::DEBUG;
      
      //We accept D1 or D2
      if(s[1] == '1') return logLevels::DEBUG;
      if(s[1] == '2') return logLevels::DEBUG2;
      
      if(s.size()<3) return logLevels::UNKNOWN;
      
      //We accept DBG and DBG2
      if( s[1] == 'B' && s[2] == 'G' ) 
      {
         if(s.size() == 3) return logLevels::DEBUG;
         
         if(s.size() > 3)
         {
            if( s[3] == '2') return logLevels::DEBUG2;
         }
         
         return logLevels::UNKNOWN;
      }
      
      //Anything that stars with DEF 
      if (s[1] == 'E' && s[2] == 'F') return logLevels::DEFAULT;
      
      //Now check for DEBUG and DEBUG2
      if( s.size() >=5 )
      {
         if( s.substr(0,5) != "DEBUG" ) return logLevels::UNKNOWN;
         
         if(s.size() > 5 )
         {
            if( s[5] == '2') return logLevels::DEBUG2;
            
            return logLevels::UNKNOWN;
         }
         
         return logLevels::DEBUG;
      }
      
   }
   
   return logLevels::UNKNOWN;
}

}// namepace logger
}// namespace MagAOX

#endif //logger_logLevels_hpp
