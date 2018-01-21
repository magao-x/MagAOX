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

/// The type of an event code (32-bit unsigned int).
typedef uint32_t eventCodeT;   

///Namespace containing the eventcodes enum.
namespace eventCodes 
{
   
///The log event codes.  These are the unique identifiers for log entry types.
enum : eventCodeT { TEXT_LOG = 1,           ///< Denotes a simple text log.
                    USER_LOG = 10,          ///< Denotes a log entered by the user.
                    STATE_CHANGE = 20,      ///< Denotes an application state change
                    SOFTWARE_DEBUG = 51,    ///< Denotes a software debug log entry
                    SOFTWARE_DEBUG2 = 52,   ///< Denotes a software debug-2 log entry
                    SOFTWARE_INFO = 53,     ///< Denotes a software info log entry
                    SOFTWARE_WARNING = 54,  ///< Denotes a software warning log entry
                    SOFTWARE_ERROR = 55,    ///< Denotes a software error log entry
                    SOFTWARE_CRITICAL = 56, ///< Denotes a software critical log entry
                    SOFTWARE_FATAL = 57,    ///< Denotes a software fatal log entry
                    LOOP_CLOSED = 1001,     ///< The loop is closed.
                    LOOP_PAUSED = 1002,     ///< The loop is paused.
                    LOOP_OPEN = 1003        ///< The loop is open.
                  };
   

} //namespace eventCodes 
} //namespace logger
} //namespace MagAOX

#endif //logger_logCodes_hpp

