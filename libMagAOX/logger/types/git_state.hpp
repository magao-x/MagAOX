/** \file git_state.hpp
  * \brief The MagAO-X logger git_state log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  */
#ifndef logger_types_git_state_hpp
#define logger_types_git_state_hpp

#include "generated/git_state_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct git_state : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::GIT_STATE;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;


   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & repoName, ///< [in] the name of the repo
                const std::string & sha1,     ///< [in] the SHA1 hash of the repo
                const bool modified           ///< [in] the modified status (true or false)
              )
      {
         auto _repoName = builder.CreateString(repoName);
         auto _sha1 = builder.CreateString(sha1);
                           
         uint8_t _modified = modified;
         
         auto gs = CreateGit_state_fb(builder, _repoName, _sha1, _modified);
         builder.Finish(gs);

      }

   };
   
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(logBuff), static_cast<size_t>(len));
      return VerifyGit_state_fbBuffer(verifier);
   }

   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);
      
      auto rgs = GetGit_state_fb(msgBuffer);
      
      std::string str;
      if( rgs->repo()) str = rgs->repo()->c_str();
      
      str += " GIT: ";
      
      if( rgs->sha1()) str += rgs->sha1()->c_str();
      
      if(rgs->modified() > 0) str+= " MODIFIED";

      return str;
   }
   
   /// Access the repo name field
   static std::string repoName( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto rgs = GetGit_state_fb(msgBuffer);
   
      if(rgs->repo()) return std::string(rgs->repo()->c_str());
      else return "";
   }
   
   /// Access the modified field
   static bool modified( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
   {
      auto rgs = GetGit_state_fb(msgBuffer);
   
      if(rgs->modified() > 0) return true;
      else return false;
   }
   
}; //git_state


} //namespace logger
} //namespace MagAOX

#endif //logger_types_git_state_hpp

