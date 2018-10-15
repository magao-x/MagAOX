/** \file ttmmod_params.hpp
  * \brief The MagAO-X logger ttmmod_params log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_data_usage_hpp
#define logger_types_data_usage_hpp

#include "generated/data_usage_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct data_usage : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::DATA_USAGE;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const float & dataUsage
              )
      {
         
         auto fp = Createdata_usage_fb(builder, dataUsage);
         
         builder.Finish(fp);

      }

   };

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = Getdata_usage_fb(msgBuffer);  
      
      std::string msg = "";

      if (rgs->dataUsage() != 0) {
        msg+= "DATAUSAGE ";
        msg+= std::to_string(rgs->dataUsage());
        msg+= " ";
      }

      return msg;

   }

}; //sys_mon


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
