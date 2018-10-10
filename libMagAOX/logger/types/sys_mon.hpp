/** \file ttmmod_params.hpp
  * \brief The MagAO-X logger ttmmod_params log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_sys_mon_hpp
#define logger_types_sys_mon_hpp

#include "generated/sys_mon_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct sys_mon : public flatbuffer_log
{

     static const flatlogs::eventCodeT eventCode = eventCodes::TTMMOD_PARAMS;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( std::vector<float> & coreTemps,
                std::vector<float> & cpu_core_loads,
                std::vector<float> & diskTemp,
                const float & rootUsage,
                const float & dataUsage,
                const float & bootUsage,
                const float & ramUsage
              )
      {
         
         auto _coreTempsVec = builder.CreateVector(coreTemps.data(),sizeof(coreTemps));
         auto _coreLoadsVec = builder.CreateVector(cpu_core_loads.data(),sizeof(cpu_core_loads));
         auto _driveTempsVec = builder.CreateVector(diskTemp.data(),sizeof(diskTemp));

         auto fp = Createsys_mon_fb(builder, _coreTempsVec, _coreLoadsVec, _driveTempsVec, rootUsage, dataUsage, bootUsage, ramUsage);
         
         builder.Finish(fp);

      }

   };

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      //auto rgs = sys_mon(msgBuffer); // EXAMPLE: how to work with a flatbuffer
      auto rgs = Getsys_mon_fb(msgBuffer);  

      std::string msg = "SYSTEM MONITOR: ";
      if (rgs->cpu_core_loads() != nullptr) {
        msg+= "CPU LOADS: ";
        for(flatbuffers::Vector<float>::iterator it = rgs->cpu_core_loads()->begin(); it != rgs->cpu_core_loads()->end(); ++it) {
          msg+= std::to_string(*it);
          msg+= " ";
        }
      }
      
      if (rgs->coreTemps() != nullptr) {
        msg+= "CPU TEMPS: ";
        for(flatbuffers::Vector<float>::iterator it = rgs->coreTemps()->begin(); it != rgs->coreTemps()->end(); ++it) {
          msg+= std::to_string(*it);
          msg+= " ";
        }
      }

      if (rgs->diskTemp() != nullptr) {
        msg+= "DRIVE TEMPS:  ";
        for(flatbuffers::Vector<float>::iterator it = rgs->diskTemp()->begin(); it != rgs->diskTemp()->end(); ++it) {
          msg+= std::to_string(*it);
          msg+= " ";
        }
      }

      if (rgs->rootUsage() != 0) {
        msg+= std::to_string(rgs->rootUsage());
        msg+= "/ROOT USAGE ";
        msg+= " ";
      }
      if (rgs->dataUsage() != 0) {
        msg+= std::to_string(rgs->dataUsage());
        msg+= "/DATA USAGE ";
        msg+= " ";
      }
      if (rgs->bootUsage() != 0){
        msg+= std::to_string(rgs->bootUsage());
        msg+= "/BOOT USAGE ";
        msg+= " ";
      }
      if (rgs->ramUsage() != 0) {
        msg+= std::to_string(rgs->ramUsage());
        msg+= "RAM USAGE ";
        msg+= " ";
      }

      return msg;

   }

}; //ttmmod_params


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
