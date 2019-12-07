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

#include "../logInterp.hpp"

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
   
   
   int epoch( double & epoch,
              timespec & tm,
              flatlogs::bufferPtrT & buffer0,
              flatlogs::bufferPtrT & buffer1 
            )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(epoch, tm, fbs0->epoch(), buffer0, fbs1->epoch(), buffer1); 
   }
   
   int ra( double & ra,
           timespec & tm,
           flatlogs::bufferPtrT & buffer0,
           flatlogs::bufferPtrT & buffer1 
         )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(ra, tm, fbs0->ra(), buffer0, fbs1->ra(), buffer1); 
   }
   
   int dec( double & dec,
            timespec & tm,
            flatlogs::bufferPtrT & buffer0,
            flatlogs::bufferPtrT & buffer1 
          )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(dec, tm, fbs0->dec(), buffer0, fbs1->dec(), buffer1); 
   }
   
   int el( double & el,
           timespec & tm,
           flatlogs::bufferPtrT & buffer0,
           flatlogs::bufferPtrT & buffer1 
         )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(el, tm, fbs0->el(), buffer0, fbs1->el(), buffer1); 
   }
   
   int ha( double & ha,
           timespec & tm,
           flatlogs::bufferPtrT & buffer0,
           flatlogs::bufferPtrT & buffer1 
         )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(ha, tm, fbs0->ha(), buffer0, fbs1->ha(), buffer1); 
   }
   
   int am( double & am,
           timespec & tm,
           flatlogs::bufferPtrT & buffer0,
           flatlogs::bufferPtrT & buffer1 
         )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(am, tm, fbs0->am(), buffer0, fbs1->am(), buffer1); 
   }
   
   int rotoff( double & ro,
               timespec & tm,
               flatlogs::bufferPtrT & buffer0,
               flatlogs::bufferPtrT & buffer1 
             )
   {

      auto fbs0 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer0));
      auto fbs1 = GetTelem_telpos_fb(logHeader::messageBuffer(buffer1));
      
      return interpLog(ro, tm, fbs0->rotoff(), buffer0, fbs1->rotoff(), buffer1); 
   }
   
}; //telem_telpos

timespec telem_telpos::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telpos_hpp

