/** \file telem_po4ao.hpp
 * \brief Typed PO4AO callback statistics, freshness and effective model identity.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_po4ao_hpp
#define logger_types_telem_po4ao_hpp
#include "generated/telem_po4ao_generated.h"
#include "flatbuffer_log.hpp"
namespace MagAOX::logger
{
/// PO4AO callback telemetry; backend codes are 0 fallback, 1 CPU and 2 TensorRT.
struct telem_po4ao : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_PO4AO;      ///< Registered telemetry code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry output stream.
    static timespec                       lastRecord;                                  ///< Most recent telemetry write.
    /// Serialized callback snapshot.
    struct messageT : public fbMessage
    {
        /// Encode one identified measurement window and cumulative counters.
        messageT( uint32_t p50_us /**< [in] p50 us. */,
                  uint32_t p95_us /**< [in] p95 us. */,
                  uint32_t p99_us /**< [in] p99 us. */,
                  uint32_t max_us /**< [in] max us. */,
                  uint64_t samples /**< [in] samples. */,
                  uint64_t sequence /**< [in] sequence. */,
                  uint64_t sample_time_ns /**< [in] sample time ns. */,
                  bool     fresh /**< [in] fresh. */,
                  bool     mixed_window /**< [in] mixed window. */,
                  uint64_t missed_frames /**< [in] missed frames. */,
                  uint64_t deadline_misses /**< [in] deadline misses. */,
                  uint64_t inference_failures /**< [in] inference failures. */,
                  uint64_t replay_drops /**< [in] replay drops. */,
                  uint32_t deadline_us /**< [in] deadline us. */,
                  uint64_t model_id /**< [in] model id. */,
                  uint32_t backend /**< [in] backend. */ );
    };
    /// Verify an encoded record before decoding.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] Complete log buffer. */,
                        flatlogs::msgLenT     length /**< [in] Message byte length. */ );
    /// Format a compact human-readable telemetry summary.
    static std::string msgString( void             *buffer /**< [in] Message bytes. */,
                                  flatlogs::msgLenT length /**< [in] Message byte length. */ );
    /// Decode p50 us.
    static uint32_t p50_us( void *buffer /**< [in] Message bytes. */ );
    /// Decode p95 us.
    static uint32_t p95_us( void *buffer /**< [in] Message bytes. */ );
    /// Decode p99 us.
    static uint32_t p99_us( void *buffer /**< [in] Message bytes. */ );
    /// Decode max us.
    static uint32_t max_us( void *buffer /**< [in] Message bytes. */ );
    /// Decode samples.
    static unsigned long long samples( void *buffer /**< [in] Message bytes. */ );
    /// Decode sequence.
    static unsigned long long sequence( void *buffer /**< [in] Message bytes. */ );
    /// Decode sample time ns.
    static unsigned long long sample_time_ns( void *buffer /**< [in] Message bytes. */ );
    /// Decode fresh.
    static bool fresh( void *buffer /**< [in] Message bytes. */ );
    /// Decode mixed window.
    static bool mixed_window( void *buffer /**< [in] Message bytes. */ );
    /// Decode missed frames.
    static unsigned long long missed_frames( void *buffer /**< [in] Message bytes. */ );
    /// Decode deadline misses.
    static unsigned long long deadline_misses( void *buffer /**< [in] Message bytes. */ );
    /// Decode inference failures.
    static unsigned long long inference_failures( void *buffer /**< [in] Message bytes. */ );
    /// Decode replay drops.
    static unsigned long long replay_drops( void *buffer /**< [in] Message bytes. */ );
    /// Decode deadline us.
    static uint32_t deadline_us( void *buffer /**< [in] Message bytes. */ );
    /// Decode model id.
    static unsigned long long model_id( void *buffer /**< [in] Message bytes. */ );
    /// Decode backend.
    static uint32_t backend( void *buffer /**< [in] Message bytes. */ );
    /// Resolve a field accessor for telemetry tools.
    static logMetaDetail getAccessor( const std::string &member /**< [in] Field name. */ );
};
inline telem_po4ao::messageT::messageT( uint32_t p50_us,
                                        uint32_t p95_us,
                                        uint32_t p99_us,
                                        uint32_t max_us,
                                        uint64_t samples,
                                        uint64_t sequence,
                                        uint64_t sample_time_ns,
                                        bool     fresh,
                                        bool     mixed_window,
                                        uint64_t missed_frames,
                                        uint64_t deadline_misses,
                                        uint64_t inference_failures,
                                        uint64_t replay_drops,
                                        uint32_t deadline_us,
                                        uint64_t model_id,
                                        uint32_t backend )
{
    builder.Finish( CreateTelem_po4ao_fb( builder,
                                          p50_us,
                                          p95_us,
                                          p99_us,
                                          max_us,
                                          samples,
                                          sequence,
                                          sample_time_ns,
                                          fresh,
                                          mixed_window,
                                          missed_frames,
                                          deadline_misses,
                                          inference_failures,
                                          replay_drops,
                                          deadline_us,
                                          model_id,
                                          backend ) );
}
inline bool telem_po4ao::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT length )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) ), length );
    return VerifyTelem_po4ao_fbBuffer( verifier );
}
inline std::string telem_po4ao::msgString( void *buffer, flatlogs::msgLenT length )
{
    (void)length;
    const auto *value = GetTelem_po4ao_fb( buffer );
    return "[po4ao callback us] p50=" + std::to_string( value->p50_us() ) +
           " p99=" + std::to_string( value->p99_us() ) + " samples=" + std::to_string( value->samples() ) +
           " fresh=" + std::to_string( value->fresh() );
}
inline uint32_t telem_po4ao::p50_us( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->p50_us();
}
inline uint32_t telem_po4ao::p95_us( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->p95_us();
}
inline uint32_t telem_po4ao::p99_us( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->p99_us();
}
inline uint32_t telem_po4ao::max_us( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->max_us();
}
inline unsigned long long telem_po4ao::samples( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->samples();
}
inline unsigned long long telem_po4ao::sequence( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->sequence();
}
inline unsigned long long telem_po4ao::sample_time_ns( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->sample_time_ns();
}
inline bool telem_po4ao::fresh( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->fresh();
}
inline bool telem_po4ao::mixed_window( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->mixed_window();
}
inline unsigned long long telem_po4ao::missed_frames( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->missed_frames();
}
inline unsigned long long telem_po4ao::deadline_misses( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->deadline_misses();
}
inline unsigned long long telem_po4ao::inference_failures( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->inference_failures();
}
inline unsigned long long telem_po4ao::replay_drops( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->replay_drops();
}
inline uint32_t telem_po4ao::deadline_us( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->deadline_us();
}
inline unsigned long long telem_po4ao::model_id( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->model_id();
}
inline uint32_t telem_po4ao::backend( void *buffer )
{
    return GetTelem_po4ao_fb( buffer )->backend();
}
inline logMetaDetail telem_po4ao::getAccessor( const std::string &member )
{
    if( member == "p50_us" )
    {
        return logMetaDetail( { "P50_US",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &p50_us ),
                                false } );
    }
    if( member == "p95_us" )
    {
        return logMetaDetail( { "P95_US",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &p95_us ),
                                false } );
    }
    if( member == "p99_us" )
    {
        return logMetaDetail( { "P99_US",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &p99_us ),
                                false } );
    }
    if( member == "max_us" )
    {
        return logMetaDetail( { "MAX_US",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &max_us ),
                                false } );
    }
    if( member == "samples" )
    {
        return logMetaDetail( { "SAMPLES",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &samples ),
                                false } );
    }
    if( member == "sequence" )
    {
        return logMetaDetail( { "SEQUENCE",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &sequence ),
                                false } );
    }
    if( member == "sample_time_ns" )
    {
        return logMetaDetail( { "SAMPLE_TIME_NS",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &sample_time_ns ),
                                false } );
    }
    if( member == "fresh" )
    {
        return logMetaDetail( { "FRESH",
                                "",
                                logMeta::valTypes::Bool,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &fresh ),
                                false } );
    }
    if( member == "mixed_window" )
    {
        return logMetaDetail( { "MIXED_WINDOW",
                                "",
                                logMeta::valTypes::Bool,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &mixed_window ),
                                false } );
    }
    if( member == "missed_frames" )
    {
        return logMetaDetail( { "MISSED_FRAMES",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &missed_frames ),
                                false } );
    }
    if( member == "deadline_misses" )
    {
        return logMetaDetail( { "DEADLINE_MISSES",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &deadline_misses ),
                                false } );
    }
    if( member == "inference_failures" )
    {
        return logMetaDetail( { "INFERENCE_FAILURES",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &inference_failures ),
                                false } );
    }
    if( member == "replay_drops" )
    {
        return logMetaDetail( { "REPLAY_DROPS",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &replay_drops ),
                                false } );
    }
    if( member == "deadline_us" )
    {
        return logMetaDetail( { "DEADLINE_US",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &deadline_us ),
                                false } );
    }
    if( member == "model_id" )
    {
        return logMetaDetail( { "MODEL_ID",
                                "",
                                logMeta::valTypes::ULongLong,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &model_id ),
                                false } );
    }
    if( member == "backend" )
    {
        return logMetaDetail( { "BACKEND",
                                "",
                                logMeta::valTypes::UInt,
                                logMeta::metaTypes::State,
                                reinterpret_cast<void *>( &backend ),
                                false } );
    }
    return logMetaDetail();
}
} // namespace MagAOX::logger
#endif
