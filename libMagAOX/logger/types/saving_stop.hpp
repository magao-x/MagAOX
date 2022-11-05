/** \file saving_stop.hpp
  * \brief The MagAO-X logger saving_stop log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2019-05-04 created by JRM
  */
#ifndef logger_types_saving_stop_hpp
#define logger_types_saving_stop_hpp

#include "saving_state_change.hpp"

namespace MagAOX
{
namespace logger
{

///Saving started log
/** \ingroup logger_types
  */
struct saving_stop : public saving_state_change
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::SAVING_STOP;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

};



} //namespace logger
} //namespace MagAOX

#endif //logger_types_saving_stop_hpp
