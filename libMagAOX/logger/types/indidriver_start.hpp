/** \file indidriver_start.hpp
  * \brief The MagAO-X logger INDI driver start type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_indidriver_start_hpp
#define logger_types_indidriver_start_hpp

#include "empty_log.hpp"

namespace MagAOX
{
namespace logger
{

///INDI Driver Start log entry
/** \ingroup logger_types
  */
struct indidriver_start : public empty_log<indidriver_start>
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::INDIDRIVER_START;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The message string
   static const char * msg() { return "INDI driver communications started"; }
   
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_indidriver_start_hpp
