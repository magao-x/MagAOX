/** \file telem_fxngen.hpp
  * \brief The MagAO-X logger telem_fxngen log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_fxngen_hpp
#define logger_types_telem_fxngen_hpp

#include "generated/telem_fxngen_generated.h"
#include "flatbuffer_log.hpp"

#define TELEM_FXNGEN_WVTP_DC (0)
#define TELEM_FXNGEN_WVTP_SINE (1)
#define TELEM_FXNGEN_WVTP_PULSE (2)

namespace MagAOX
{
namespace logger
{


/// Log entry recording the function generator parameters
/** \ingroup logger_types
  */
struct telem_fxngen : public flatbuffer_log
{
    ///The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_FXNGEN;

    ///The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.
   
    


   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const uint8_t & C1outp, ///< [in] Channel 1 output status
                const double & C1freq,  ///< [in] Channel 1 frequency [Hz]
                const double & C1vpp,   ///< [in] Channel 1 P2P voltage [V]
                const double & C1ofst,  ///< [in] Channel 1 offset [V]
                const double & C1phse,  ///< [in] Channel 1 phase [deg]
                const uint8_t & C1wvtp, ///< [in] Channel 1 wavetype (SINE or DC)
                const uint8_t & C2outp, ///< [in] Channel 2 output status
                const double & C2freq,  ///< [in] Channel 2 frequency [Hz]
                const double & C2vpp,   ///< [in] Channel 2 P2P voltage [V]
                const double & C2ofst,  ///< [in] Channel 2 offset [V]
                const double & C2phse,  ///< [in] Channel 2 phase [deg]
                const uint8_t & C2wvtp, ///< [in] Channel 2 wavetype  (SINE or DC) 
                const uint8_t & C1sync, ///< [in] Channel 1 sync status
                const uint8_t & C2sync, ///< [in] Channel 2 sync status
                const double & C1wdth,  ///< [in] Channel 1 width [s]
                const double & C2wdth   ///< [in] Channel 2 width [s]
              )
      {
         auto fp = CreateTelem_fxngen_fb(builder, C1outp, C1freq, C1vpp, C1ofst, C1phse, C1wvtp, 
                                               C2outp, C2freq, C2vpp, C2ofst, C2phse, C2wvtp, C1sync, C2sync, C1wdth, C2wdth);
         builder.Finish(fp);
      }

   };
   
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_fxngen_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_fxngen_fb(msgBuffer);

      std::string msg = "Ch 1: ";
      
      if(fbs->C1wvtp() == TELEM_FXNGEN_WVTP_DC) msg += "DC ";
      else if(fbs->C1wvtp() == TELEM_FXNGEN_WVTP_SINE) msg += "SINE ";
      else if(fbs->C1wvtp() == TELEM_FXNGEN_WVTP_PULSE) msg += "PULSE ";
      else msg += "UNK ";
      
      if(fbs->C1outp() == 0) msg += "OFF ";
      else if(fbs->C1outp() == 1) msg += "ON ";
      else msg += "UNK ";
      
      msg += std::to_string(fbs->C1freq()) + " Hz ";
      msg += std::to_string(fbs->C1vpp()) + " Vp2p ";
      msg += std::to_string(fbs->C1ofst()) + " V ";
      msg += std::to_string(fbs->C1phse()) + " deg ";
      msg += std::to_string(fbs->C1wdth()) + " s ";
      msg += "SYNC ";
      if(fbs->C1sync()) msg += "ON ";
      else msg += "OFF ";

      msg += " | Ch 2: ";

      if(fbs->C2wvtp() == TELEM_FXNGEN_WVTP_DC) msg += "DC ";
      else if(fbs->C2wvtp() == TELEM_FXNGEN_WVTP_SINE) msg += "SINE ";
      else if(fbs->C2wvtp() == TELEM_FXNGEN_WVTP_PULSE) msg += "PULSE ";
      else msg += "UNK ";
      
      if(fbs->C2outp() == 0) msg += "OFF ";
      else if(fbs->C2outp() == 1) msg += "ON ";
      else msg += "UNK ";
      
      msg += std::to_string(fbs->C2freq()) + " Hz ";
      msg += std::to_string(fbs->C2vpp()) + " Vp2p ";
      msg += std::to_string(fbs->C2ofst()) + " V ";
      msg += std::to_string(fbs->C2phse()) + " deg ";
      msg += std::to_string(fbs->C2wdth()) + " s ";
      msg += "SYNC ";
      if(fbs->C2sync()) msg += "ON ";
      else msg += "OFF ";

      return msg;
   
   }

   static double C1freq( void * msgBuffer )
   {
      auto fbs = GetTelem_fxngen_fb(msgBuffer);
      return fbs->C1freq();
   }

   static double C2freq( void * msgBuffer )
   {
      auto fbs = GetTelem_fxngen_fb(msgBuffer);
      return fbs->C2freq();
   }

   /// Get the logMetaDetail for a member by name
   static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(     member == "C1freq") return logMetaDetail({"C1 FREQ", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&C1freq)});
      else if(member == "C2freq") return logMetaDetail({"C2 FREQ", logMeta::valTypes::Double, logMeta::metaTypes::Continuous, reinterpret_cast<void*>(&C2freq)});
      else
      {
         std::cerr << "No string member " << member << " in telem_fxngen\n";
         return logMetaDetail();
      }
   }
}; //telem_fxngen



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_fxngen_hpp

