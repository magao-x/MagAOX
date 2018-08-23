/** \file loop_closed.hpp
  * \brief The MagAO-X logger loop closed type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_loop_closed_hpp
#define logger_types_loop_closed_hpp

#include "empty_log.hpp"

namespace MagAOX
{
namespace logger
{

///Loop Closed event log
/** \ingroup logger_types
  */
struct loop_closed : public empty_log<loop_closed>
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::LOOP_CLOSED;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   ///The message string
   static const char * msg() {return "LOOP CLOSED";}
   
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_loop_closed_hpp
