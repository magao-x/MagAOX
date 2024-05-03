/** \file user_log.hpp
  * \brief The MagAO-X logger user_log log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-08-18 created by JRM
  * - 2025-05-02 revised with added email field by JRM
  */
#ifndef logger_types_user_log_hpp
#define logger_types_user_log_hpp

#include "generated/user_log_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// User entered log
/** \ingroup logger_types
  */
struct user_log : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::USER_LOG;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_INFO;

    ///The type of the message
    struct messageT : public fbMessage
    {
        messageT( const std::string & email,
                  const std::string & message 
                )
        {
            auto _eml = builder.CreateString(email);
            auto _msg = builder.CreateString(message);

            auto gs = CreateUser_log_fb(builder, _eml, _msg);
            builder.Finish(gs);
        }
    };

    static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                      )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
        return VerifyString_log_fbBuffer(verifier);
    }

    ///Get the message formatte for human consumption.
    static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                                )
    {
        static_cast<void>(len);

        auto fbs = GetUser_log_fb(msgBuffer);

        std::string msg = "[USER] ";

        if(fbs->email())
        {
            msg += fbs->email()->c_str();
            msg += ": ";
        }
        else
        {
            msg += "unknown: ";
        }

        if(fbs->message())
        {
            msg += fbs->message()->c_str();
        }
      
        return msg;
    }

    ///Get the user email from a user_log
    static std::string email( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/)
    {
        auto fbs = GetUser_log_fb(msgBuffer);
        if(fbs->email() != nullptr)
        {
            return std::string(fbs->email()->c_str());
        }
        else return "";
   }

    ///Get the message from a user_log
    static std::string message( void * msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/)
    {
        auto fbs = GetUser_log_fb(msgBuffer);
        if(fbs->message() != nullptr)
        {
            return std::string(fbs->message()->c_str());
        }
        else return "";
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logmegaDetail if member not recognized
     */ 
    static logMetaDetail getAccessor( const std::string & member /**< [in] the name of the member */ )
    {
        if(     member == "email")   return logMetaDetail({"USER", logMeta::valTypes::String, logMeta::metaTypes::State, reinterpret_cast<void*>(&email), false});
        else if(member == "message") return logMetaDetail({"MESSAGE", logMeta::valTypes::String, logMeta::metaTypes::State, reinterpret_cast<void*>(&message), false});
        else
        {
            std::cerr << "No string member " << member << " in user_log\n";
            return logMetaDetail();
        }
   }

};


} //namespace logger
} //namespace MagAOX

#endif //logger_types_user_log_hpp
