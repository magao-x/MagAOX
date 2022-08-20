/** \file loop_open.hpp
  * \brief The MagAO-X logger loop open type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_loop_open_hpp
#define logger_types_loop_open_hpp

#include "empty_log.hpp"

namespace MagAOX
{
namespace logger
{

///Loop Open event log
/** \ingroup logger_types
  */
struct loop_open : public empty_log<loop_open>
{
   ///The event code
   constexpr static flatlogs::eventCodeT eventCode = eventCodes::LOOP_OPEN;

   ///The default level
   constexpr static flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   ///The message string
   static const char * msg() {return "LOOP OPEN";}
   
};


} //namespace logger
} //namespace MagAOX

#endif //logger_types_loop_open_hpp
