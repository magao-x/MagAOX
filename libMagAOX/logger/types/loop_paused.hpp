/** \file loop_paused.hpp
  * \brief The MagAO-X logger loop paused type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_loop_paused_hpp
#define logger_types_loop_paused_hpp

#include "empty_log.hpp"

namespace MagAOX
{
namespace logger
{

///Loop Paused event log
/** \ingroup logger_types
  */
struct loop_paused : public empty_log<loop_paused>
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::LOOP_PAUSED;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   ///The message string
   static const char * msg() {return "LOOP PAUSED";}
   
};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_loop_paused_hpp
