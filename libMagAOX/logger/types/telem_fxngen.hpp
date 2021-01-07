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
      messageT( const uint8_t & C1outp,     ///< [in] Channel 1 output status
                const double & C1freq,      ///< [in] Channel 1 frequency [Hz]
                const double & C1vpp,       ///< [in] Channel 1 P2P voltage [V]
                const double & C1ofst,      ///< [in] Channel 1 offset [V]
                const double & C1phse,      ///< [in] Channel 1 phase [deg]
                const std::string & C1wvtp, ///< [in] Channel 1 wavetype (SINE or DC)
                const uint8_t & C2outp,     ///< [in] Channel 2 output status
                const double & C2freq,      ///< [in] Channel 2 frequency [Hz]
                const double & C2vpp,       ///< [in] Channel 2 P2P voltage [V]
                const double & C2ofst,      ///< [in] Channel 2 offset [V]
                const double & C2phse,      ///< [in] Channel 2 phase [deg]
                const std::string & C2wvtp  ///< [in] Channel 2 wavetype  (SINE or DC) 
              )
      {
         uint8_t  _C1wvtp = 3,  _C2wvtp = 3;
         
         
         if(C1wvtp == "DC") _C1wvtp = 0;
         else if(C1wvtp == "SINE") _C1wvtp = 1;
         
         if(C2wvtp == "DC") _C2wvtp = 0;
         else if(C2wvtp == "SINE") _C2wvtp = 1;
         
         
         auto fp = CreateTelem_fxngen_fb(builder, C1outp, C1freq, C1vpp, C1ofst, C1phse, _C1wvtp, C2outp, C2freq, C2vpp, C2ofst, C2phse, _C2wvtp);
         builder.Finish(fp);

      }

   };
   
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_fxngen_fb(msgBuffer);

      std::string msg = "Ch 1: ";
      
      if(fbs->C1wvtp() == 0) msg += "DC ";
      else if(fbs->C1wvtp() == 1) msg += "SINE ";
      else msg += "UNK ";
      
      if(fbs->C1outp() == 0) msg += "OFF ";
      else if(fbs->C1outp() == 1) msg += "ON ";
      else msg += "UNK ";
      
      msg += std::to_string(fbs->C1freq()) + " Hz ";
      msg += std::to_string(fbs->C1vpp()) + " Vp2p ";
      msg += std::to_string(fbs->C1ofst()) + " V ";
      msg += std::to_string(fbs->C1phse()) + " deg ";
      
      msg += " | Ch 2: ";

      if(fbs->C2wvtp() == 0) msg += "DC ";
      else if(fbs->C2wvtp() == 1) msg += "SINE ";
      else msg += "UNK ";
      
      if(fbs->C2outp() == 0) msg += "OFF ";
      else if(fbs->C2outp() == 1) msg += "ON ";
      else msg += "UNK ";
      
      msg += std::to_string(fbs->C2freq()) + " Hz ";
      msg += std::to_string(fbs->C2vpp()) + " Vp2p ";
      msg += std::to_string(fbs->C2ofst()) + " V ";
      msg += std::to_string(fbs->C2phse()) + " deg ";

      return msg;
   
   }
   
}; //telem_fxngen



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_fxngen_hpp

