/** \file telem_pupilxcfit.hpp
 * \brief Binary telemetry for cross-correlation pupil fits.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_pupilxcfit_hpp
#define logger_types_telem_pupilxcfit_hpp

#include <limits>
#include <sstream>

#include <mx/math/floatUtils.hpp>

#include "generated/telem_pupilxcfit_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Result of processing one live pupil image against the selected reference.
/** \ingroup logger_types */
struct telem_pupilxcfit : public flatbuffer_log
{
    /// Logger dispatch identifier.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_PUPILXCFIT;

    /// Priority used by the telemetry writer.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Most recent record time, maintained by the telemeter.
    static timespec lastRecord;

    /// Serialized input and correlation result.
    struct messageT : public fbMessage
    {
        /// Serialize one frame, including the measurement only when its components are finite.
        messageT( uint64_t           frame,     /**< [in] source stream frame counter */
                  const timespec    &acquired,  /**< [in] source acquisition timestamp */
                  const std::string &reference, /**< [in] selected reference stream name */
                  bool               valid,     /**< [in] true when the fitter accepted the measurement */
                  float              dx,        /**< [in] x displacement in pixels or NaN when unavailable */
                  float              dy,        /**< [in] y displacement in pixels or NaN when unavailable */
                  float              peak /**< [in] normalized correlation peak or NaN when unavailable */ );
    };

    /// Verify the buffer and require a finite measurement for an accepted fit.
    static bool verify( flatlogs::bufferPtrT &buffer, /**< [in] complete log record */
                        flatlogs::msgLenT     len /**< [in] serialized message length */ );

    /// Format the selected reference, frame identity and available measurements.
    static std::string msgString( void             *buffer, /**< [in] verified payload */
                                  flatlogs::msgLenT len /**< [in] serialized message length */ );

    /// Return metadata for the reference, acceptance flag and measurement values.
    static logMetaDetail getAccessor( const std::string &member /**< [in] metadata field name */ );

  private:
    /// Extract the selected reference stream name.
    static std::string reference( void *buffer /**< [in] verified payload */ );

    /// Extract whether the fitter accepted this measurement.
    static bool valid( void *buffer /**< [in] verified payload */ );

    /// Extract one measured quantity, returning NaN when no measurement was obtained.
    template <float ( PupilXCFitMeasurement::*field )() const>
    static float value( void *buffer /**< [in] verified payload */ );
};

inline telem_pupilxcfit::messageT::messageT(
    uint64_t frame, const timespec &acquired, const std::string &reference, bool valid, float dx, float dy, float peak )
{
    const auto                                 name = builder.CreateString( reference );
    flatbuffers::Offset<PupilXCFitMeasurement> measurement;

    if( mx::math::isFinite( dx ) && mx::math::isFinite( dy ) && mx::math::isFinite( peak ) )
    {
        measurement = CreatePupilXCFitMeasurement( builder, dx, dy, peak );
    }

    builder.Finish(
        CreateTelem_pupilxcfit_fb( builder, frame, acquired.tv_sec, acquired.tv_nsec, name, valid, measurement ) );
}

inline bool telem_pupilxcfit::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    const auto           *payload = static_cast<const uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_pupilxcfit_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_pupilxcfit_fb( payload );
    if( record->acq_nsec() < 0 || record->acq_nsec() >= 1000000000 )
    {
        return false;
    }

    const auto *measurement = record->measurement();
    return measurement ? mx::math::isFinite( measurement->dx() ) && mx::math::isFinite( measurement->dy() ) &&
                             mx::math::isFinite( measurement->peak() )
                       : !record->valid();
}

inline std::string telem_pupilxcfit::msgString( void *buffer, flatlogs::msgLenT )
{
    const auto        *record = GetTelem_pupilxcfit_fb( buffer );
    std::ostringstream out;
    out << "[pupilxcfit] frame=" << record->frame() << " acquired=(" << record->acq_sec() << ", " << record->acq_nsec()
        << ") reference=" << record->reference()->string_view() << " valid=" << std::boolalpha << record->valid();

    if( const auto *measurement = record->measurement() )
    {
        out << " dx=" << measurement->dx() << " dy=" << measurement->dy() << " peak=" << measurement->peak();
    }

    return out.str();
}

inline std::string telem_pupilxcfit::reference( void *buffer )
{
    return GetTelem_pupilxcfit_fb( buffer )->reference()->str();
}

inline bool telem_pupilxcfit::valid( void *buffer )
{
    return GetTelem_pupilxcfit_fb( buffer )->valid();
}

template <float ( PupilXCFitMeasurement::*field )() const>
inline float telem_pupilxcfit::value( void *buffer )
{
    const auto *measurement = GetTelem_pupilxcfit_fb( buffer )->measurement();
    return measurement ? ( measurement->*field )() : std::numeric_limits<float>::quiet_NaN();
}

inline logMetaDetail telem_pupilxcfit::getAccessor( const std::string &member )
{
    if( member == "reference" )
    {
        return { "REFERENCE", logMeta::String, logMeta::State, reinterpret_cast<void *>( &reference ) };
    }

    if( member == "valid" )
    {
        return { "VALID", logMeta::Bool, logMeta::State, reinterpret_cast<void *>( &valid ) };
    }

    if( member == "dx" )
    {
        return {
            "DX", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &value<&PupilXCFitMeasurement::dx> ) };
    }

    if( member == "dy" )
    {
        return {
            "DY", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &value<&PupilXCFitMeasurement::dy> ) };
    }

    if( member == "peak" )
    {
        return { "PEAK",
                 logMeta::Float,
                 logMeta::Continuous,
                 reinterpret_cast<void *>( &value<&PupilXCFitMeasurement::peak> ) };
    }

    return {};
}

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_pupilxcfit_hpp
