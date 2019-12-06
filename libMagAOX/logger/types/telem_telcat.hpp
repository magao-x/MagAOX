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
      messageT( const std::string & catObj,     ///< [in] 
                const std::string & catRm,    ///< [in] 
                const double & catRa,    ///< [in] 
                const double & catDec,        ///< [in] 
                const double & catEp,     ///< [in] 
                const double & catRo   ///< [in] 
              )
      {
         auto _catObj = builder.CreateString(catObj);
         auto _catRm = builder.CreateString(catRm);
         auto fp = CreateTelem_telcat_fb(builder, _catObj, _catRm, catRa, catDec, catEp, catRo);
         builder.Finish(fp);
      }

   };
                 
 
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
   
}; //telem_telcat

timespec telem_telcat::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telcat_hpp

