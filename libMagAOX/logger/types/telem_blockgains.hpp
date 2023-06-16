/** \file telem_blockgains.hpp
  * \brief The MagAO-X logger telem_blockgains log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2022-11-27 created by JRM
  */
#ifndef logger_types_telem_blockgains_hpp
#define logger_types_telem_blockgains_hpp

#include "generated/telem_blockgains_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording electronics rack temperature
/** \ingroup logger_types
  */
struct telem_blockgains : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_BLOCKGAINS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::vector<float> & gains,            ///<[in] vector of gains
                const std::vector<uint8_t> & gains_constant, ///<[in] vector of gains constant flags
                const std::vector<float> & multcs,              ///<[in] vector of mult. coeffs.
                const std::vector<uint8_t> & multcs_constant,   ///<[in] vector of mult. coeff constant flags
                const std::vector<float> & lims,             ///<[in] vector of limits
                const std::vector<uint8_t> & lims_constant   ///<[in] vector of limits constant flags
              )
      {
         auto _gains = builder.CreateVector(gains);
         auto _gainsc = builder.CreateVector(gains_constant);
         auto _mcs = builder.CreateVector(multcs);
         auto _mcsc = builder.CreateVector(multcs_constant);         
         auto _lims = builder.CreateVector(lims);
         auto _limsc = builder.CreateVector(lims_constant);

         auto fb = CreateTelem_blockgains_fb(builder, _gains, _gainsc, _mcs, _mcsc, _lims, _limsc);

         builder.Finish(fb);
      }

   };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyTelem_blockgains_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_blockgains_fb(msgBuffer);

      std::string msg = "[gains] ";
      
      // being very paranoid about existence and length here
      if( fbs->gains() && fbs->gains_constant() )
      {
         if(fbs->gains()->Length() == fbs->gains_constant()->Length())
         {
            for(size_t i=0; i< fbs->gains()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->gains()->Get(i));
               msg += " (";
               msg += std::to_string(fbs->gains_constant()->Get(i));
               msg += ")";
            }
         }
         else
         {
            for(size_t i=0; i< fbs->gains()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->gains()->Get(i));
               msg += " (?)";
            }
         }
      }
      else if (fbs->gains())
      {
         for(size_t i=0; i< fbs->gains()->Length(); ++i)
         {
            msg += " ";
            msg += std::to_string(fbs->gains()->Get(i));
            msg += " (?)";
         }
      }

      msg += " [mcs] ";
      if( fbs->mcs() && fbs->mcs_constant() )
      {
         if(fbs->mcs()->Length() == fbs->mcs_constant()->Length())
         {
            for(size_t i=0; i< fbs->mcs()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->mcs()->Get(i));
               msg += " (";
               msg += std::to_string(fbs->mcs_constant()->Get(i));
               msg += ")";
            }
         }
         else
         {
            for(size_t i=0; i< fbs->mcs()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->mcs()->Get(i));
               msg += " (?)";
            }
         }
      }
      else if (fbs->mcs())
      {
         for(size_t i=0; i< fbs->mcs()->Length(); ++i)
         {
            msg += " ";
            msg += std::to_string(fbs->mcs()->Get(i));
            msg += " (?)";
         }
      }
      
      msg += " [lims] ";

      if( fbs->lims() && fbs->lims_constant() )
      {
         if(fbs->lims()->Length() == fbs->lims_constant()->Length())
         {
            for(size_t i=0; i< fbs->lims()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->lims()->Get(i));
               msg += " (";
               msg += std::to_string(fbs->lims_constant()->Get(i));
               msg += ")";
            }
         }
         else
         {
            for(size_t i=0; i< fbs->lims()->Length(); ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->lims()->Get(i));
               msg += " (?)";
            }
         }
      }
      else if (fbs->lims())
      {
         for(size_t i=0; i< fbs->lims()->Length(); ++i)
         {
            msg += " ";
            msg += std::to_string(fbs->lims()->Get(i));
            msg += " (?)";
         }
      }

      return msg;
   
   }
   
}; //telem_blockgains



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_blockgains_hpp

