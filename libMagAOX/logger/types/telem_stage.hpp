/** \file telem_stage.hpp
  * \brief The MagAO-X logger telem_stage log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_stage_hpp
#define logger_types_telem_stage_hpp

#include "generated/telem_stage_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording stdMotionStage status.
/** \ingroup logger_types
  */
struct telem_stage : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_STAGE;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded.  Used by the telemetry system.

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const int8_t & moving,           ///<[in] whether or not stage is in motion
                const double & preset,         ///<[in] current position of stage in preset units
                const std::string & presetName ///<[in] current prest name
              )
      {
         auto _name = builder.CreateString(presetName);
         
         auto fp = CreateTelem_stage_fb(builder, moving, preset, _name);
         builder.Finish(fp);
      }

   };
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_stage_fb(msgBuffer);

      std::string msg = "[stage] ";
      
      msg += "status: ";
      if(fbs->moving() == -2) msg += " OFF ";
      else if(fbs->moving() == -1) msg += " NOTHOMED ";
      else if(fbs->moving() == 0) msg += " STOPPED ";
      else if(fbs->moving() == 1) msg += " MOVING ";
      else if(fbs->moving() == 2) msg += " HOMING ";
      
      msg += "preset: ";
      msg += std::to_string(fbs->preset()) + " ";
      
      msg += "name: ";
      if(fbs->presetName())
      {
         msg += fbs->presetName()->c_str();
      }
      
      return msg;
   
   }
   
   static int8_t moving( void * msgBuffer )
   {
      auto fbs = GetTelem_stage_fb(msgBuffer);
      return fbs->moving();
   }
   
   static double preset( void * msgBuffer )
   {
      auto fbs = GetTelem_stage_fb(msgBuffer);
      return fbs->preset();
   }
   
   static std::string presetName( void * msgBuffer )
   {
      auto fbs = GetTelem_stage_fb(msgBuffer);
      if(fbs->presetName())
      {
         return std::string(fbs->presetName()->c_str());
      }
      else return std::string();
   }
   
   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static void * getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "moving") return (void *) &moving;
      else if(member == "preset") return (void *) &preset;
      else if(member == "presetName") return (void *) &presetName;
      else
      {
         std::cerr << "No string member " << member << " in telem_telcat\n";
         return 0;
      }
   }
   
}; //telem_stage

timespec telem_stage::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_stage_hpp

