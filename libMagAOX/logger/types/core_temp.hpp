/** \file ttmmod_params.hpp
  * \brief The MagAO-X logger ttmmod_params log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_core_temp_hpp
#define logger_types_core_temp_hpp

#include "generated/core_temp_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct core_temp : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::CORE_TEMP;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<float> & coreTemps
              )
      {
         
         auto _coreTempsVec = builder.CreateVector(coreTemps.data(),coreTemps.size());

         auto fp = Createcore_temp_fb(builder, _coreTempsVec);
         
         builder.Finish(fp);

      }

   };

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = Getcore_temp_fb(msgBuffer);  
      
      std::string msg = "";

      if (rgs->coreTemps() != nullptr) {
        msg+= "CPUTEMPS: ";
        for(flatbuffers::Vector<float>::iterator it = rgs->coreTemps()->begin(); it != rgs->coreTemps()->end(); ++it) {
          msg+= std::to_string(*it);
          msg+= " ";
        }
      }

      return msg;

   }

}; //sys_mon


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
