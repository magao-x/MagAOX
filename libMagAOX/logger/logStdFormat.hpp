/** \file logStdFormat.hpp 
  * \brief Standard formating of log entries for readable output.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-12-24 created by JRM
  */ 

#ifndef logger_logStdFormat_hpp
#define logger_logStdFormat_hpp

#include "logTypes.hpp"

namespace MagAOX
{
namespace logger 
{
   
/// Worker function that formats a log into the standard text representation.
/** \todo change to using a std::ios as input instead of only using std::cout 
  */
template<typename logT>
void _stdFormat( bufferPtrT & logBuffer /**< [in] the binary log buffer */)
{
   logLevelT lvl;
   eventCodeT ec;
   time::timespecX ts;
   msgLenT len;

   extractBasicLog( lvl, ec, ts, len, logBuffer); 

   typename logT::messageT msg;

   logT::extract(msg, logBuffer.get()+messageOffset, len);

   std::cout << ts.ISO8601DateTimeStrX() << " " << levelString(lvl) << " " << logT::msgString(msg) << "\n";
}

/// Place the log in standard text format, with event code specific formatting.
inline
void logStdFormat(bufferPtrT & buffer /**< [in] the binary log buffer */ )
{
   eventCodeT ec;
   ec = eventCode(buffer);

   switch(ec)
   {
      case text_log::eventCode:
         return _stdFormat<text_log>(buffer);
      case user_log::eventCode:
         return _stdFormat<user_log>(buffer);
      case state_change::eventCode:
         return _stdFormat<state_change>(buffer);
      case software_debug::eventCode:
         return _stdFormat<software_debug>(buffer);
      case software_debug2::eventCode:
         return _stdFormat<software_debug2>(buffer);
      case software_info::eventCode:
         return _stdFormat<software_info>(buffer);
      case software_warning::eventCode:
         return _stdFormat<software_warning>(buffer);
      case software_error::eventCode:
         return _stdFormat<software_error>(buffer);
      case software_critical::eventCode:
         return _stdFormat<software_critical>(buffer);
      case software_fatal::eventCode:
         return _stdFormat<software_fatal>(buffer);
      case loop_closed::eventCode:
         return _stdFormat<loop_closed>(buffer);
      case loop_paused::eventCode:
         return _stdFormat<loop_paused>(buffer);
      case loop_open::eventCode:
         return _stdFormat<loop_open>(buffer);
      
      default:
         std::cout << "Unknown log type: " << ec << "\n";
   }
}

} //namespace logger 
} //namespace MagAOX 

#endif //logger_logStdFormat_hpp
