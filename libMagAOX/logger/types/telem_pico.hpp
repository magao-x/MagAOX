/** \file telem_pico.hpp
  * \brief The MagAO-X logger telem_pico log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_telem_pico_hpp
#define logger_types_telem_pico_hpp

#include "generated/telem_pico_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording CPU temperatures
/** \ingroup logger_types
  */
struct telem_pico : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_PICO;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

  static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<int64_t> & counts )
      {
         auto _countsVec = builder.CreateVector(counts);

         auto fp = CreateTelem_pico_fb(builder, _countsVec );

         builder.Finish(fp);

      }

   };

   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_pico_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types

      auto rgs = GetTelem_pico_fb(msgBuffer);

      std::string msg;

      if (rgs->counts() != nullptr)
      {
         msg+= "[pico pos] ";
         for(flatbuffers::Vector<int64_t>::const_iterator it = rgs->counts()->begin(); it != rgs->counts()->end(); ++it)
         {
            msg+= std::to_string(*it);
            msg+= " ";
         }
      }

      return msg;

   }

   static std::vector<long long int> counts( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      std::vector<long long int> countsvec;

      auto rgs = GetTelem_pico_fb(msgBuffer);

      if (rgs->counts() != nullptr)
      {
          for(auto it = rgs->counts()->begin(); it != rgs->counts()->end(); ++it)
          {
             countsvec.push_back(*it);
          }
      }

      return countsvec;
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if( member == "counts") return logMetaDetail({"COUNTS", "motor positions in counts", logMeta::valTypes::Vector_LongLong, logMeta::metaTypes::State, reinterpret_cast<void*>(&counts), false});
      else
      {
         std::cerr << "No member " << member << " in telem_pico\n";
         return logMetaDetail();
      }
   }
}; //telem_pico



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_pico_hpp
