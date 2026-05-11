/** \file telem_tcsi_tiptilt.hpp
 * \brief The MagAO-X logger telem_tcsi_tiptilt log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_tcsi_tiptilt_hpp
#define logger_types_telem_tcsi_tiptilt_hpp

#include "telem_tcsi_offload.hpp"

namespace MagAOX
{
namespace logger
{

/// Tip/tilt offload-control telemetry.
/** \ingroup logger_types
 */
struct telem_tcsi_tiptilt : public telem_tcsi_offload
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TCSI_TIPTILT;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Format the message for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        return formatMsgString( "tcsi_tiptilt", msgBuffer, len );
    }

    static timespec lastRecord; ///< The time of the last time this log was recorded. Used by the telemetry system.
};

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_tcsi_tiptilt_hpp
