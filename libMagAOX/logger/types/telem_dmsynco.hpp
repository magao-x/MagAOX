/** \file telem_dmsynco.hpp
 * \brief Aggregate syncopation timing, outcomes and per-step feedback telemetry.
 */
#ifndef logger_types_telem_dmsynco_hpp
#define logger_types_telem_dmsynco_hpp

#include "generated/telem_dmsynco_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Periodic diagnostics recorded outside the real-time publication path.
struct telem_dmsynco : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_DMSYNCO;    ///< Registered event code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static inline timespec                lastRecord{}; ///< Last aggregate publication, owned by the telemetry system.

    /// Serialized aggregate message.
    struct messageT : public fbMessage
    {
        /// Serialize a coherent snapshot of scheduler diagnostics.
        messageT( int                         state /**< [in] schedule state */,
                  int                         steps /**< [in] requested count */,
                  int64_t                     period /**< [in] measured period, ns */,
                  uint64_t                    delivered /**< [in] confirmed extra combinations */,
                  uint64_t                    missed /**< [in] failed timing events */,
                  uint64_t                    faults /**< [in] synchronization/resource faults */,
                  uint64_t                    cached /**< [in] source-cache fallback count */,
                  int64_t                     snapshot /**< [in] maximum whole-snapshot time, ns */,
                  int64_t                     copy /**< [in] maximum whole-copy time, ns */,
                  const std::vector<int64_t> &residuals /**< [in] latest per-step residuals, ns */,
                  const std::vector<int64_t> &advances /**< [in] per-step wait advances, ns */ );
    };

    /// Validate the serialized FlatBuffer.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] message length */ );

    /// Format the aggregate for log readers.
    static std::string msgString( void             *buffer /**< [in] message buffer */,
                                  flatlogs::msgLenT len /**< [in] message length, unused */ );

    /// This aggregate has no scalar logMeta projections; read its typed FlatBuffer fields.
    static logMetaDetail getAccessor( const std::string &member /**< [in] requested scalar member, unused */ );
};

inline telem_dmsynco::messageT::messageT( int                         state,
                                          int                         steps,
                                          int64_t                     period,
                                          uint64_t                    delivered,
                                          uint64_t                    missed,
                                          uint64_t                    faults,
                                          uint64_t                    cached,
                                          int64_t                     snapshot,
                                          int64_t                     copy,
                                          const std::vector<int64_t> &residuals,
                                          const std::vector<int64_t> &advances )
{
    const auto residualVector = builder.CreateVector( residuals );
    const auto advanceVector  = builder.CreateVector( advances );
    builder.Finish( CreateTelem_dmsynco_fb( builder,
                                            state,
                                            steps,
                                            period,
                                            delivered,
                                            missed,
                                            faults,
                                            cached,
                                            snapshot,
                                            copy,
                                            residualVector,
                                            advanceVector ) );
}

inline bool telem_dmsynco::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) ), len );
    return VerifyTelem_dmsynco_fbBuffer( verifier );
}

inline std::string telem_dmsynco::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    const auto *record = GetTelem_dmsynco_fb( buffer );
    return "[dmsynco] state=" + std::to_string( record->state() ) + " steps=" + std::to_string( record->steps() ) +
           " period_ns=" + std::to_string( record->period_ns() ) +
           " delivered=" + std::to_string( record->delivered() ) + " missed=" + std::to_string( record->missed() ) +
           " faults=" + std::to_string( record->faults() );
}

inline logMetaDetail telem_dmsynco::getAccessor( const std::string &member )
{
    static_cast<void>( member );
    return {};
}

} // namespace logger
} // namespace MagAOX
#endif
