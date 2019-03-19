/** \file drive_mon.hpp
  * \brief The MagAO-X logger drive_mon log type.
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup logger_types_files
  *
  * History:
  * - 2018-10-15 created by CJB
  */
#ifndef logger_types_drive_mon_hpp
#define logger_types_drive_mon_hpp

#include "generated/drive_mon_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct drive_mon : public flatbuffer_log
{

  static const flatlogs::eventCodeT eventCode = eventCodes::DRIVE_MON;
  static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( 
                std::vector<float> & diskTemp,
                const float & rootUsage,
                const float & dataUsage,
                const float & bootUsage
              )
      {

         auto _driveTempsVec = builder.CreateVector(diskTemp.data(),diskTemp.size());

         auto fp = Createdrive_mon_fb(builder, _driveTempsVec, rootUsage, dataUsage, bootUsage);
         
         builder.Finish(fp);

      }

   };

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {

      static_cast<void>(len); // unused by most log types
   
      auto rgs = Getdrive_mon_fb(msgBuffer);  
      
      std::string msg = "";

      if (rgs->diskTemp() != nullptr) {
        msg+= "DRIVETEMPS:  ";
        for(flatbuffers::Vector<float>::iterator it = rgs->diskTemp()->begin(); it != rgs->diskTemp()->end(); ++it) {
          msg+= std::to_string(*it);
          msg+= " ";
        }
      }

      if (rgs->rootUsage() != 0) {
        msg+= "/ROOTUSAGE ";
        msg+= std::to_string(rgs->rootUsage());      
        msg+= " ";
      }
      if (rgs->dataUsage() != 0) {
        msg+= "/DATAUSAGE ";
        msg+= std::to_string(rgs->dataUsage());       
        msg+= " ";
      }
      if (rgs->bootUsage() != 0){
        msg+= "/BOOTUSAGE ";
        msg+= std::to_string(rgs->bootUsage());      
        msg+= " ";
      }

      return msg;

   }

}; //sys_mon


} //namespace logger
} //namespace MagAOX

#endif //logger_types_ttmmod_params_hpp
