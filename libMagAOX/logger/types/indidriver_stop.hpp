/** \file indidriver_stop.hpp
  * \brief The MagAO-X logger INDI driver stop type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_indidriver_stop_hpp
#define logger_types_indidriver_stop_hpp

#include "empty_log.hpp"

namespace MagAOX
{
namespace logger
{

///INDI Driver Start log entry
/** \ingroup logger_types
  */
struct indidriver_stop : public empty_log<indidriver_stop>
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::INDIDRIVER_STOP;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The message string
   static const char * msg() { return "INDI driver communications stopped"; }

};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_indidriver_stop_hpp
