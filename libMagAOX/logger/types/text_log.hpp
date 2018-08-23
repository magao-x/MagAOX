/** \file text_log.hpp
  * \brief The MagAO-X logger text_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_text_log_hpp
#define logger_types_text_log_hpp

#include "string_log.hpp"

namespace MagAOX
{
namespace logger
{

///A simple text log, a string-type log.
/** \ingroup logger_types
  */
struct text_log : public string_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TEXT_LOG;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

};



} //namespace logger
} //namespace MagAOX

#endif //logger_types_text_log_hpp
