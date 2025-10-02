/** \file saving_start.hpp
  * \brief The MagAO-X logger saving_start log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_saving_start_hpp
#define logger_types_saving_start_hpp

#include "saving_state_change.hpp"

namespace MagAOX
{
namespace logger
{

///Saving started log
/** \ingroup logger_types
  */
struct saving_start : public saving_state_change
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::SAVING_START;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_NOTICE;

   /// Get an empty logMetaDetail because meta data doesn't make sense for this log
   /**
     * \returns an empty logMetaDetail
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      static_cast<void>(member);

      std::cerr << "meta data doesn't make sense for saving_start.\n";
      return logMetaDetail();
   }

};



} //namespace logger
} //namespace MagAOX

#endif //logger_types_saving_start_hpp
