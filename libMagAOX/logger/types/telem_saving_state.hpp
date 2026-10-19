/** \file telem_saving_state.hpp
  * \brief Saving-activity telemetry snapshots for the MagAO-X logger.
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

/// Snapshot of saving activity and the source-frame counter at the end of the latest queued save window.
/** \ingroup logger_types
  */
struct telem_saving_state : public saving_state_change
{
   /// Event code identifying a saving-state snapshot rather than a start or stop event.
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_SAVING_STATE;

   /// Telemetry priority used for periodic and per-chunk snapshots.
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// Format a state snapshot without implying that saving has just started or stopped.
    static std::string msgString( void             *msgBuffer /**< [in] serialized saving-state payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );
};



inline std::string telem_saving_state::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    auto *state = GetSaving_state_change_fb( msgBuffer );

    return std::string( "Saving state: " ) + ( state->state() == 0 ? "inactive" : "active" ) +
           "; save-window end frame: " + std::to_string( state->frameNo() );
}

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_saving_state_hpp
