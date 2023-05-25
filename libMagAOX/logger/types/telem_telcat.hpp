/** \file telem_telcat.hpp
  * \brief The MagAO-X logger telem_telcat log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telcat_hpp
#define logger_types_telem_telcat_hpp

#include "generated/telem_telcat_generated.h"
#include "flatbuffer_log.hpp"
#include "../logMeta.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telcat : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELCAT;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & catObj, ///< [in] Catalog object name
                const std::string & catRm,  ///< [in] Catalog rotator mode 
                const double & catRa,       ///< [in] Catalog right ascension [degrees]
                const double & catDec,      ///< [in] Catalog declination [degrees]
                const double & catEp,       ///< [in] Catalog epoch
                const double & catRo        ///< [in] Catalog rotator offset
              )
      {
         auto _catObj = builder.CreateString(catObj);
         auto _catRm = builder.CreateString(catRm);
         auto fp = CreateTelem_telcat_fb(builder, _catObj, _catRm, catRa, catDec, catEp, catRo);
         builder.Finish(fp);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_telcat_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telcat_fb(msgBuffer);

      std::string msg = "[telcat] ";
     
      if(fbs->catObj() != nullptr)
      {
         msg += "obj: ";
         msg += fbs->catObj()->c_str() ;
         msg += " ";
      }
      
      msg += "ra: ";
      msg += std::to_string(fbs->catRa()) + " ";
      
      msg += "dec: ";
      msg += std::to_string(fbs->catDec()) + " ";
      
      msg += "ep: ";
      msg += std::to_string(fbs->catEp()) + " ";
      
      if(fbs->catRm() != nullptr)
      {
         msg += "rm: ";
         msg += fbs->catRm()->c_str() ;
         msg += " ";
      }
      
      msg += "ro: ";
      msg += std::to_string(fbs->catRo()) + " ";
      
      
      
      return msg;
   
   }
   
   static std::string catObj(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      if(fbs->catObj() != nullptr)
      {
         return std::string(fbs->catObj()->c_str());
      }
      else
      {
         return std::string("");
      }
   }
   
   static std::string catRm(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      if(fbs->catRm() != nullptr)
      {
         return std::string(fbs->catRm()->c_str());
      }
      else
      {
         return std::string("");
      }
   }
   
   static double catRA(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catRa();
   }
   
   static double catDec(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catDec();
   }
   
   static double catEp(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catEp();
   }
   
   static double catRo(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catRo();
   }
   
   /// Get the logMetaDetail for a member by name
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "catObj") return logMetaDetail({"CATOBJ", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &catObj, false});
      else if(member == "catRm")  return logMetaDetail({"CATRM", logMeta::valTypes::String, logMeta::metaTypes::State, (void *) &catRm, false});
      else if(member == "catRA")  return logMetaDetail({"CATRA", logMeta::valTypes::Double, logMeta::metaTypes::State, (void *) &catRA, false});
      else if(member == "catDec") return logMetaDetail({"CATDEC", logMeta::valTypes::Double, logMeta::metaTypes::State, (void *) &catDec, false});
      else if(member == "catEp")  return logMetaDetail({"CATEP", logMeta::valTypes::Double, logMeta::metaTypes::State, (void *) &catEp, false});
      else if(member == "catRo")  return logMetaDetail({"CATRO", logMeta::valTypes::Double, logMeta::metaTypes::State, (void *) &catRo, false});
      else
      {
         std::cerr << "No string member " << member << " in telem_telcat\n";
         return logMetaDetail();
      }
   }
   
}; //telem_telcat



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telcat_hpp

