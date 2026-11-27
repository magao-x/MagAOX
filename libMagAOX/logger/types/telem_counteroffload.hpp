/** \file telem_counteroffload.hpp
 * \brief The MagAO-X logger telem_counteroffload log type.
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_counteroffload_hpp
#define logger_types_telem_counteroffload_hpp

#include "generated/telem_counteroffload_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording the counterOffload state, settings, cumulative counters and latency summary.
/** Counters are cumulative since the app started, so rates come from differences between records.
 *
 * \ingroup logger_types
 */
struct telem_counteroffload : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_COUNTEROFFLOAD;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT(const bool     &enabled, ///< [in] whether new offloads are accepted
            const bool     &calibrating, ///< [in] whether a calibration run is in progress
            const float    &gain, ///< [in] scale applied to each new offload
            const float    &leak, ///< [in] fraction of the completed contribution removed after each published slot
            const float    &ramp_start_us, ///< [in] ramp start after the woofer command, us
            const float    &ramp_duration_us, ///< [in] ramp duration, us
            const float    &time_offset_us, ///< [in] offset added to the input receipt time to give the ramp reference, us
            const float    &max_command, ///< [in] command clip limit
            const float    &slot_us, ///< [in] dmSyncopation slot period, us; -1 until measured
            const uint64_t &events, ///< [in] offload updates accepted since startup
            const uint64_t &published, ///< [in] commands published since startup
            const uint64_t &failed, ///< [in] commands not published since startup
            const uint64_t &missed, ///< [in] offloads that waited at least a whole slot since startup
            const uint64_t &clipped, ///< [in] slots in which the command or the accumulator was clipped since startup
            const uint64_t &overflows, ///< [in] offloads folded without a ramp because the ramp pool was full, since startup
            const float    &pipeline_us, ///< [in] mean receipt-to-publication pipeline time over the latency window, us; -1 if no data
            const float    &pipeline_rms_us, ///< [in] RMS of the pipeline time over the latency window, us; -1 if no data
            const float    &compute_max_us ///< [in] largest compute time over the latency window, us; -1 if no data
        )
        {
            auto fp = CreateTelem_counteroffload_fb( builder,
                                                     enabled,
                                                     calibrating,
                                                     gain,
                                                     leak,
                                                     ramp_start_us,
                                                     ramp_duration_us,
                                                     time_offset_us,
                                                     max_command,
                                                     slot_us,
                                                     events,
                                                     published,
                                                     failed,
                                                     missed,
                                                     clipped,
                                                     overflows,
                                                     pipeline_us,
                                                     pipeline_rms_us,
                                                     compute_max_us );
            builder.Finish( fp );
        }
    };

    /// Verify the flatbuffer in a log entry.
    /**
     * \returns true if the message is a valid telem_counteroffload flatbuffer
     */
    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_counteroffload_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    /**
     * \returns the message as text
     */
    static std::string msgString( void             *msgBuffer, /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len        /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_counteroffload_fb( msgBuffer );

        std::string msg = "[counteroffload] ";
        msg += fbs->enabled() ? "enabled" : "disabled";
        msg += fbs->calibrating() ? " calibrating" : "";
        msg += " gain: " + std::to_string( fbs->gain() );
        msg += " leak: " + std::to_string( fbs->leak() );
        msg += " ramp: " + std::to_string( fbs->ramp_start_us() ) + "+" + std::to_string( fbs->ramp_duration_us() );
        msg += " us offset: " + std::to_string( fbs->time_offset_us() );
        msg += " us max: " + std::to_string( fbs->max_command() );
        msg += " slot: " + std::to_string( fbs->slot_us() );
        msg += " us events: " + std::to_string( fbs->events() );
        msg += " published: " + std::to_string( fbs->published() );
        msg += " failed: " + std::to_string( fbs->failed() );
        msg += " missed: " + std::to_string( fbs->missed() );
        msg += " clipped: " + std::to_string( fbs->clipped() );
        msg += " overflows: " + std::to_string( fbs->overflows() );
        msg += " pipeline: " + std::to_string( fbs->pipeline_us() ) + " +/- " + std::to_string( fbs->pipeline_rms_us() );
        msg += " us compute max: " + std::to_string( fbs->compute_max_us() ) + " us";

        return msg;
    }

    /// Get whether new offloads are accepted.
    /**
     * \returns the enabled field
     */
    static bool enabled( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->enabled();
    }

    /// Get whether a calibration run is in progress.
    /**
     * \returns the calibrating field
     */
    static bool calibrating( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->calibrating();
    }

    /// Get scale applied to each new offload.
    /**
     * \returns the gain field
     */
    static float gain( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->gain();
    }

    /// Get fraction of the completed contribution removed after each published slot.
    /**
     * \returns the leak field
     */
    static float leak( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->leak();
    }

    /// Get ramp start after the woofer command, us.
    /**
     * \returns the ramp_start_us field
     */
    static float ramp_start_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->ramp_start_us();
    }

    /// Get ramp duration, us.
    /**
     * \returns the ramp_duration_us field
     */
    static float ramp_duration_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->ramp_duration_us();
    }

    /// Get offset added to the input receipt time to give the ramp reference, us.
    /**
     * \returns the time_offset_us field
     */
    static float time_offset_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->time_offset_us();
    }

    /// Get command clip limit.
    /**
     * \returns the max_command field
     */
    static float max_command( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->max_command();
    }

    /// Get dmSyncopation slot period, us; -1 until measured.
    /**
     * \returns the slot_us field
     */
    static float slot_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->slot_us();
    }

    /// Get offload updates accepted since startup.
    /**
     * \returns the events field
     */
    static unsigned long long events( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->events();
    }

    /// Get commands published since startup.
    /**
     * \returns the published field
     */
    static unsigned long long published( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->published();
    }

    /// Get commands not published since startup.
    /**
     * \returns the failed field
     */
    static unsigned long long failed( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->failed();
    }

    /// Get offloads that waited at least a whole slot since startup.
    /**
     * \returns the missed field
     */
    static unsigned long long missed( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->missed();
    }

    /// Get slots in which the command or the accumulator was clipped since startup.
    /**
     * \returns the clipped field
     */
    static unsigned long long clipped( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->clipped();
    }

    /// Get offloads folded without a ramp because the ramp pool was full, since startup.
    /**
     * \returns the overflows field
     */
    static unsigned long long overflows( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->overflows();
    }

    /// Get mean receipt-to-publication pipeline time over the latency window, us; -1 if no data.
    /**
     * \returns the pipeline_us field
     */
    static float pipeline_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->pipeline_us();
    }

    /// Get RMS of the pipeline time over the latency window, us; -1 if no data.
    /**
     * \returns the pipeline_rms_us field
     */
    static float pipeline_rms_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->pipeline_rms_us();
    }

    /// Get largest compute time over the latency window, us; -1 if no data.
    /**
     * \returns the compute_max_us field
     */
    static float compute_max_us( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        return GetTelem_counteroffload_fb( msgBuffer )->compute_max_us();
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "enabled" )
            return logMetaDetail( { "ENABLED",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &enabled ) } );
        else if( member == "calibrating" )
            return logMetaDetail( { "CALIBRATING",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &calibrating ) } );
        else if( member == "gain" )
            return logMetaDetail( { "GAIN",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &gain ) } );
        else if( member == "leak" )
            return logMetaDetail( { "LEAK",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &leak ) } );
        else if( member == "ramp_start_us" )
            return logMetaDetail( { "RAMP START US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &ramp_start_us ) } );
        else if( member == "ramp_duration_us" )
            return logMetaDetail( { "RAMP DURATION US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &ramp_duration_us ) } );
        else if( member == "time_offset_us" )
            return logMetaDetail( { "TIME OFFSET US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &time_offset_us ) } );
        else if( member == "max_command" )
            return logMetaDetail( { "MAX COMMAND",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &max_command ) } );
        else if( member == "slot_us" )
            return logMetaDetail( { "SLOT US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &slot_us ) } );
        else if( member == "events" )
            return logMetaDetail( { "EVENTS",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &events ) } );
        else if( member == "published" )
            return logMetaDetail( { "PUBLISHED",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &published ) } );
        else if( member == "failed" )
            return logMetaDetail( { "FAILED",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &failed ) } );
        else if( member == "missed" )
            return logMetaDetail( { "MISSED",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &missed ) } );
        else if( member == "clipped" )
            return logMetaDetail( { "CLIPPED",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &clipped ) } );
        else if( member == "overflows" )
            return logMetaDetail( { "OVERFLOWS",
                                    logMeta::valTypes::ULongLong,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &overflows ) } );
        else if( member == "pipeline_us" )
            return logMetaDetail( { "PIPELINE US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &pipeline_us ) } );
        else if( member == "pipeline_rms_us" )
            return logMetaDetail( { "PIPELINE RMS US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &pipeline_rms_us ) } );
        else if( member == "compute_max_us" )
            return logMetaDetail( { "COMPUTE MAX US",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::Continuous,
                                    reinterpret_cast<void *>( &compute_max_us ) } );
        else
        {
            std::cerr << "No member " << member << " in telem_counteroffload\n";
            return logMetaDetail();
        }
    }

}; // telem_counteroffload

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_counteroffload_hpp
