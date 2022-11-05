/** \file telem_saving_state.hpp
  * \brief The MagAO-X logger telem_saving_state log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_telem_saving_state_hpp
#define logger_types_telem_saving_state_hpp

#include "saving_state_change.hpp"

namespace MagAOX
{
namespace logger
{

///Saving started log
/** \ingroup logger_types
  */
struct telem_saving_state : public saving_state_change
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_SAVING_STATE;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

};



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_saving_state_hpp
