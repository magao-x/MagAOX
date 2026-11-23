/** \file telem_tcsi_focus.hpp
 * \brief The MagAO-X logger telem_tcsi_focus log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_tcsi_focus_hpp
#define logger_types_telem_tcsi_focus_hpp

#include "telem_tcsi_offload.hpp"

namespace MagAOX
{
namespace logger
{

/// Focus offload-control telemetry.
/** \ingroup logger_types
 */
struct telem_tcsi_focus : public telem_tcsi_offload
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TCSI_FOCUS;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded. Used by the telemetry system.
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_tcsi_focus_hpp
