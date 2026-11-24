/** \file telem_ocamdelay.hpp
 * \brief Telemetry for the OCAM frame-delay checks.
 */
#ifndef logger_types_telem_ocamdelay_hpp
#define logger_types_telem_ocamdelay_hpp

#include <array>
#include <sstream>

#include "generated/telem_ocamdelay_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX::logger
{
/// Store three single intervals and three FPS-sized rolling spans with their alarm state.
struct telem_ocamdelay : public flatbuffer_log
{
    static constexpr flatlogs::eventCodeT eventCode    = eventCodes::TELEM_OCAMDELAY;  ///< Logger dispatch code.
    static constexpr flatlogs::logPrioT   defaultLevel = flatlogs::logPrio::LOG_TELEM; ///< Telemetry priority.
    static timespec                       lastRecord; ///< Time of the previous telemetry emission.

    /// FlatBuffer payload with measured intervals and rate needed to derive the excess.
    struct messageT : public fbMessage
    {
        /// Construct the payload from a coherent consumer snapshot.
        messageT( double                        fps /**< [in] Effective rate in Hz, zero when unavailable. */,
                  uint64_t                      samples /**< [in] Number of intervals in a full rolling window. */,
                  double                        fraction /**< [in] Excess threshold in periods. */,
                  const std::array<int64_t, 6> &elapsed /**< [in] Measured spans in ns, -1 when unavailable. */,
                  const std::array<int64_t, 6> &excess /**< [in] Signed excess in ns when a span is available. */,
                  uint32_t enabled /**< [in] Enable mask, single checks followed by rolling checks. */,
                  uint32_t latched /**< [in] Warning latch mask. */,
                  uint64_t dropped /**< [in] Total diagnostic handoff losses. */ );
    };

    /// Verify an encoded telemetry message before reading it.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] Log envelope. */,
                        flatlogs::msgLenT     len /**< [in] Encoded payload length. */ );

    /// Render the measured spans and their nominal expectations for logdump.
    static std::string msgString( void             *buffer /**< [in] Verified FlatBuffer payload. */,
                                  flatlogs::msgLenT len /**< [in] Encoded payload length. */ );

    /// Expose any of the six measured spans to log metadata consumers, in ns.
    static logMetaDetail getAccessor( const std::string &name /**< [in] Name of the span to extract. */ );

  private:
    /// Read one measured span from a verified telemetry payload.
    template <int64_t ( OcamDelaySpans::*member )() const>
    static double spanNs( void *buffer /**< [in] Verified FlatBuffer payload. */ );
};

inline telem_ocamdelay::messageT::messageT( double                        fps,
                                            uint64_t                      samples,
                                            double                        fraction,
                                            const std::array<int64_t, 6> &elapsed,
                                            const std::array<int64_t, 6> &excess,
                                            uint32_t                      enabled,
                                            uint32_t                      latched,
                                            uint64_t                      dropped )
{
    const OcamDelaySpans spans( elapsed[0], elapsed[1], elapsed[2], elapsed[3], elapsed[4], elapsed[5] );
    const OcamDelaySpans signedExcess( excess[0], excess[1], excess[2], excess[3], excess[4], excess[5] );
    builder.Finish(
        CreateTelem_ocamdelay_fb( builder, fps, samples, fraction, &spans, &signedExcess, enabled, latched, dropped ) );
}

inline bool telem_ocamdelay::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) ), len );
    return VerifyTelem_ocamdelay_fbBuffer( verifier );
}

inline std::string telem_ocamdelay::msgString( void *buffer, flatlogs::msgLenT )
{
    const auto        *record = GetTelem_ocamdelay_fb( buffer );
    const auto        *spans  = record->elapsed();
    std::ostringstream out;
    out << "[ocamdelay] fps: " << record->fps() << " samples: " << record->samples()
        << " thresholdFrames: " << record->fraction() << " enabled: " << record->enabled()
        << " latched: " << record->latched() << " dropped: " << record->dropped() << " ns atime: " << spans->atime()
        << " arrival: " << spans->arrival() << " write: " << spans->write() << " atime_sum: " << spans->atime_sum()
        << " arrival_sum: " << spans->arrival_sum() << " write_sum: " << spans->write_sum();
    return out.str();
}

template <int64_t ( OcamDelaySpans::*member )() const>
inline double telem_ocamdelay::spanNs( void *buffer )
{
    return ( GetTelem_ocamdelay_fb( buffer )->elapsed()->*member )();
}

inline logMetaDetail telem_ocamdelay::getAccessor( const std::string &name )
{
    using Spans = OcamDelaySpans;
    static constexpr std::array<const char *, 6> names{
        "atime", "arrival", "write", "atime_sum", "arrival_sum", "write_sum" };
    static constexpr std::array accessors{ &spanNs<&Spans::atime>,
                                           &spanNs<&Spans::arrival>,
                                           &spanNs<&Spans::write>,
                                           &spanNs<&Spans::atime_sum>,
                                           &spanNs<&Spans::arrival_sum>,
                                           &spanNs<&Spans::write_sum> };

    for( size_t index = 0; index < names.size(); ++index )
    {
        if( name == names[index] )
        {
            return { names[index],
                     logMeta::valTypes::Double,
                     logMeta::metaTypes::Continuous,
                     reinterpret_cast<void *>( accessors[index] ) };
        }
    }

    return {};
}
} //namespace MagAOX::logger

#endif // logger_types_telem_ocamdelay_hpp
