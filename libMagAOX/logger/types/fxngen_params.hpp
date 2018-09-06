/** \file fxngen_params.hpp
  * \brief The MagAO-X logger fxngen_params log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_fxngen_params_hpp
#define logger_types_fxngen_params_hpp

#include "generated/fxngen_params_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct fxngen_params : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::FXNGEN_PARAMS;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;


   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const uint8_t & C1outp,     ///< [in]
                const double & C1freq,      ///< [in]
                const double & C1vpp,       ///< [in]
                const double & C1ofst,      ///< [in]
                const std::string & C1wvtp, ///< [in]
                const uint8_t & C2outp,     ///< [in]
                const double & C2freq,      ///< [in]
                const double & C2vpp,       ///< [in]
                const double & C2ofst,      ///< [in]
                const std::string & C2wvtp  ///< [in]
              )
      {
         uint8_t  _C1wvtp = 3,  _C2wvtp = 3;
         
         
         if(C1wvtp == "DC") _C1wvtp = 0;
         else if(C1wvtp == "SINE") _C1wvtp = 1;
         
         if(C2wvtp == "DC") _C2wvtp = 0;
         else if(C2wvtp == "SINE") _C2wvtp = 1;
         
         
         auto fp = CreateFxngen_params_fb(builder, C1outp, C1freq, C1vpp, C1ofst, _C1wvtp, C2outp, C2freq, C2vpp, C2ofst, _C2wvtp);
         builder.Finish(fp);

      }

   };
   
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);
//       
//       auto rgs = GetGit_state_fb(msgBuffer);
//       
//       std::string str;
//       if( rgs->repo()) str = rgs->repo()->c_str();
//       
//       str += " GIT: ";
//       
//       if( rgs->sha1()) str += rgs->sha1()->c_str();
//       
//       if(rgs->modified() > 0) str+= " MODIFIED";

      static_cast<void>(msgBuffer);
      return "fxngen_params logged";
   
   }
   
}; //fxngen_params


} //namespace logger
} //namespace MagAOX

#endif //logger_types_fxngen_params_hpp

