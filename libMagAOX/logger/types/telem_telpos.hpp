/** \file telem_telpos.hpp
  * \brief The MagAO-X logger telem_telpos log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telpos_hpp
#define logger_types_telem_telpos_hpp

#include "generated/telem_telpos_generated.h"
#include "flatbuffer_log.hpp"

#include <cmath>

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telpos : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELPOS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   enum member{ em_epoch, em_ra, em_dec, em_el, em_ha, em_am, em_rotoff};
   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const double & epoch, ///<[in] epoch
                const double & ra,    ///<[in] right ascension
                const double & dec,   ///<[in] declination
                const double & el,    ///<[in] elevation
                const double & ha,    ///<[in] hour angle
                const double & am,    ///<[in] air mass
                const double & rotoff ///<[in] rotoff
              )
      {
         auto fp = CreateTelem_telpos_fb(builder, epoch, ra, dec, el, ha, am, rotoff);
         builder.Finish(fp);
      }

   };
                 
 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_telpos_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telpos_fb(msgBuffer);

      std::string msg = "[telpos] ";
      
      msg += "ep: ";
      msg += std::to_string(fbs->epoch()) + " ";
      
      msg += "ra: ";
      msg += std::to_string(fbs->ra()) + " ";
      
      msg += "dec: ";
      msg += std::to_string(fbs->dec()) + " ";
      
      msg += "el: ";
      msg += std::to_string(fbs->el()) + " ";
      
      msg += "ha: ";
      msg += std::to_string(fbs->ha()) + " ";
      
      msg += "am: ";
      msg += std::to_string(fbs->am()) + " ";
      
      msg += "ro: ";
      msg += std::to_string(fbs->rotoff());
      
      return msg;
   
   }
   
   static double epoch( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->epoch();
   }

   static double ra( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->ra();
   }

   static double dec( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->dec();
   }

   static double el( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->el();
   }

   static double ha( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->ha();
   }

   static double am( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->am();
   }

   static double ro( void * msgBuffer )
   {
      auto fbs = GetTelem_telpos_fb(msgBuffer);
      return fbs->rotoff();
   }
   

   /// Get the logMetaDetail for a member by name
   /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "epoch") return logMetaDetail({"EPOCH", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&epoch), false});
      else if(member == "ra") return logMetaDetail({"RA", "right ascension [degrees]", "%0.8f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&ra), false}); 
      else if(member == "dec") return logMetaDetail({"DEC", "declination [degrees]", "%0.8f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&dec), false}); 
      else if(member == "el") return logMetaDetail({"EL", "elevation [degrees]", "%0.8f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&el), false}); 
      else if(member == "ha") return logMetaDetail({"HA", "hour angle [degrees]", "%0.8f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&ha), false}); 
      else if(member == "am") return logMetaDetail({"AIRMASS", "airmass", "%0.2f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&am), false});
      else if(member == "ro") return logMetaDetail({"RO", "rotator offset [degrees]", "%0.8f", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&ro), false});  
      else
      {
         std::cerr << "No string member " << member << " in telem_telpos\n";
         return logMetaDetail();
      }
   }

}; //telem_telpos



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telpos_hpp

