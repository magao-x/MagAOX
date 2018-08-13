/** \file logCodes.hpp
  * \brief The MagAO-X logger log event codes.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-06-27 created by JRM
  */
#ifndef logger_logCodes_hpp
#define logger_logCodes_hpp


namespace MagAOX
{
namespace logger
{

/** \addtogroup logcodes
  * A log entry type is uniquely identified by a code, which is a 16-bit unsigned integer.
  */

/// The type of an event code (16-bit unsigned int).
/** \ingroup logcodes
  */
typedef uint16_t eventCodeT;


/// Scoping struct for event codes
/** We do not use the enum class feature since it does not have automatic integer conversion.
  * \ingroup logcodes
  */
struct eventCodes
{

   ///The log event codes.  These are the unique identifiers for log entry types.
   /** These are in the eventCodes class scope, so must be referenced with, e.g., eventCodes::GIT_STATE.
     * \ingroup logcodes
     */
   enum : eventCodeT { GIT_STATE = 0,          ///< The git repository state at application build-time
                       
                       TEXT_LOG = 10,          ///< Denotes a simple text log.
                       
                       USER_LOG = 11,          ///< Denotes a log entered by the user.
                       
                       STATE_CHANGE = 20,      ///< Denotes an application state change
                       
                       SOFTWARE_DEBUG = 51,    ///< Denotes a software debug log entry
                       SOFTWARE_DEBUG2 = 52,   ///< Denotes a software debug-2 log entry
                       SOFTWARE_INFO = 53,     ///< Denotes a software info log entry
                       SOFTWARE_WARNING = 54,  ///< Denotes a software warning log entry
                       SOFTWARE_ERROR = 55,    ///< Denotes a software error log entry
                       SOFTWARE_CRITICAL = 56, ///< Denotes a software critical log entry
                       SOFTWARE_FATAL = 57,    ///< Denotes a software fatal log entry
                       
                       SOFTWARE_TRACE_DEBUG = 61,    ///< Denotes a software trace debug log entry
                       SOFTWARE_TRACE_DEBUG2 = 62,   ///< Denotes a software trace debug-2 log entry
                       SOFTWARE_TRACE_INFO = 63,     ///< Denotes a software trace info log entry
                       SOFTWARE_TRACE_WARNING = 64,  ///< Denotes a software trace warning log entry
                       SOFTWARE_TRACE_ERROR = 65,    ///< Denotes a software trace error log entry
                       SOFTWARE_TRACE_CRITICAL = 66, ///< Denotes a software trace critical log entry
                       SOFTWARE_TRACE_FATAL = 67,    ///< Denotes a software trace fatal log entry
                       
                       INDIDRIVER_START = 140, ///< The INDI driver has begun communications
                       INDIDRIVER_STOP = 141,  ///< The INDI driver has stopped communications
                       
                       LOOP_CLOSED = 1001,     ///< The loop is closed.
                       LOOP_PAUSED = 1002,     ///< The loop is paused.
                       LOOP_OPEN = 1003,        ///< The loop is open.
                       
                       TRIPPLITEPDU_OUTLET_OFF = 12001, ///< An outlet was turned off on a TrippLite PDU
                       TRIPPLITEPDU_OUTLET_ON = 12002 ///< An outlet was turned on on a TrippLite PDU
                     };
};

} //namespace logger
} //namespace MagAOX

#endif //logger_logCodes_hpp
