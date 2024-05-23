/** \file telem_pi335.hpp
 * \brief The MagAO-X logger telem_pi335 log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_pi335_hpp
#define logger_types_telem_pi335_hpp

#include "generated/telem_pi335_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the build-time git state.
/** \ingroup logger_types
 */
struct telem_pi335 : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_PI335;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const float & pos1Set, ///< [in]
                  const float & pos1,    ///< [in]
                  const float & sva1,    ///< [in]
                  const float & pos2Set, ///< [in]
                  const float & pos2,    ///< [in]
                  const float & sva2,    ///< [in]
                  const float & pos3Set, ///< [in]
                  const float & pos3,    ///< [in]
                  const float & sva3     ///< [in]
        )
        {
            auto fp = CreateTelem_pi335_fb(builder, pos1Set, pos1, sva1, pos2Set, pos2, sva2, pos3Set, pos3, sva3);
            builder.Finish(fp);
        }
    };

    static bool verify(flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len          ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier(static_cast<uint8_t *>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
        return VerifyTelem_pi335_fbBuffer(verifier);
    }

    /// Get the message formatte for human consumption.
    static std::string msgString(void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>(len);

        auto fbs = GetTelem_pi335_fb(msgBuffer);

        std::string msg = "[pi335] ";

        msg += "pos1Set: ";
        msg += std::to_string(fbs->pos1Set()) + " ";

        msg += "pos1: ";
        msg += std::to_string(fbs->pos1()) + " ";

        msg += "sva1: ";
        msg += std::to_string(fbs->sva1()) + " ";

        msg += "pos2Set: ";
        msg += std::to_string(fbs->pos2Set()) + " ";

        msg += "pos2: ";
        msg += std::to_string(fbs->pos2()) + " ";

        msg += "sva2: ";
        msg += std::to_string(fbs->sva2()) + " ";

        msg += "pos3Set: ";
        msg += std::to_string(fbs->pos3Set()) + " ";

        msg += "pos3: ";
        msg += std::to_string(fbs->pos3()) + " ";

        msg += "sva3: ";
        msg += std::to_string(fbs->sva3());

        return msg;
    }

    static float pos1Set(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos1Set();
    }

    static float pos1(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos1();
    }

    static float sva1(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->sva1();
    }

    static float pos2Set(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos2Set();
    }

    static float pos2(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos2();
    }

    static float sva2(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->sva2();
    }

    static float pos3Set(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos3Set();
    }

    static float pos3(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->pos3();
    }

    static float sva3(void *msgBuffer)
    {
        auto fbs = GetTelem_pi335_fb(msgBuffer);
        return fbs->sva3();
    }

    /// Get pointer to the accessor for a member by name
    /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */
    static logMetaDetail getAccessor(const std::string &member /**< [in] the name of the member */)
    {
        if (member == "pos1Set")
            return logMetaDetail({"POS1 SETPT", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos1Set)});
        else if (member == "pos1")
            return logMetaDetail({"POS1", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos1)});
        else if (member == "sva1")
            return logMetaDetail({"SVA1", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&sva1)});
        else if (member == "pos2Set")
            return logMetaDetail({"POS2 SETPT", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos2Set)});
        else if (member == "pos2")
            return logMetaDetail({"POS2", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos2)});
        else if (member == "sva2")
            return logMetaDetail({"SVA2", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&sva2)});
        else if (member == "pos3Set")
            return logMetaDetail({"POS3 SETPT", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos3Set)});
        else if (member == "pos3")
            return logMetaDetail({"POS3", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&pos3)});
        else if (member == "sva3")
            return logMetaDetail({"SVA3", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>(&sva3)});
        else
        {
            std::cerr << "No string member " << member << " in telem_pi335\n";
            return logMetaDetail();
        }
    }

}; // telem_pi335

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_pi335_hpp
