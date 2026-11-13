/** \file telem_pupilfit.hpp
 * \brief Per-image pupil measurements and derived averages.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_pupilfit_hpp
#define logger_types_telem_pupilfit_hpp

#include <mx/math/floatUtils.hpp>
#include <limits>
#include <iomanip>
#include <sstream>

#include "generated/telem_pupilfit_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{
/// One fit attempt for all active pupils; invalid attempts carry no measurements.
/** \ingroup logger_types */
struct telem_pupilfit : public flatbuffer_log
{
    /// Logger dispatch identifier.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_PUPILFIT;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Last emission time, maintained by the telemeter.
    static timespec lastRecord;

    /// Serialized fit input.
    struct messageT : public fbMessage
    {
        /// Build one record, storing measurements only for successful fits.
        messageT( uint64_t                           frame /**< [in] input stream counter */,
                  const timespec                    &acquired /**< [in] input acquisition timestamp */,
                  bool                               reference /**< [in] true for a reference-image fit */,
                  uint8_t                            numPupils /**< [in] configured active pupil count */,
                  bool                               valid /**< [in] whether all pupil measurements succeeded */,
                  const std::vector<PupilFitResult> &pupils /**< [in] results in quadrant order */ );
    };

    /// Verify both the FlatBuffer and the active-pupil measurement contract.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log record */,
                        flatlogs::msgLenT     len /**< [in] serialized message length */ );

    /// Format every pupil and its derived across-pupil average.
    static std::string msgString( void             *buffer /**< [in] serialized message */,
                                  flatlogs::msgLenT len /**< [in] serialized message length */ );

    /// Add derived averages to JSON without storing them in the binary schema.
    static std::string msgJSON( void             *buffer /**< [in] serialized message */,
                                flatlogs::msgLenT len /**< [in] serialized message length */,
                                const uint8_t    *schema /**< [in] binary schema */,
                                unsigned int      schemaLength /**< [in] binary schema size */ );

    /// Extract one measurement for all active pupils.
    template <float ( PupilFitResult::*Field )() const>
    static std::vector<float> values( void *buffer /**< [in] serialized message */ );

    /// Compute one across-pupil mean, or NaN for an invalid attempt.
    template <float ( PupilFitResult::*Field )() const>
    static float average( void *buffer /**< [in] serialized message */ );

    /// Return metadata for measured vectors or derived scalar averages.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );
};

inline telem_pupilfit::messageT::messageT( uint64_t                           frame,
                                           const timespec                    &acquired,
                                           bool                               reference,
                                           uint8_t                            numPupils,
                                           bool                               valid,
                                           const std::vector<PupilFitResult> &pupils )
{
    auto results = builder.CreateVectorOfStructs( valid ? pupils : std::vector<PupilFitResult>{} );
    builder.Finish( CreateTelem_pupilfit_fb(
        builder, frame, acquired.tv_sec, acquired.tv_nsec, reference, numPupils, valid, results ) );
}

inline bool telem_pupilfit::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );
    if( !VerifyTelem_pupilfit_fbBuffer( verifier ) )
        return false;
    auto *fb = GetTelem_pupilfit_fb( payload );
    if( fb->num_pupils() != 3 && fb->num_pupils() != 4 )
        return false;
    if( fb->acq_nsec() < 0 || fb->acq_nsec() >= 1000000000 )
        return false;
    if( !fb->valid() )
        return !fb->pupils() || fb->pupils()->size() == 0;
    if( !fb->pupils() || fb->pupils()->size() != fb->num_pupils() )
        return false;
    for( const auto *p : *fb->pupils() )
    {
        if( !mx::math::isFinite( p->x() ) || !mx::math::isFinite( p->y() ) || !mx::math::isFinite( p->D() ) ||
            !mx::math::isFinite( p->med() ) || !mx::math::isFinite( p->bg() ) || p->D() <= 0 || p->med() <= p->bg() )
            return false;
    }
    return true;
}

template <float ( PupilFitResult::*Field )() const>
inline std::vector<float> telem_pupilfit::values( void *buffer )
{
    auto              *fb = GetTelem_pupilfit_fb( buffer );
    std::vector<float> result;
    if( fb->valid() && fb->pupils() )
    {
        for( const auto *p : *fb->pupils() )
            result.push_back( ( p->*Field )() );
    }
    return result;
}

template <float ( PupilFitResult::*Field )() const>
inline float telem_pupilfit::average( void *buffer )
{
    auto *fb = GetTelem_pupilfit_fb( buffer );
    if( !fb->valid() || !fb->pupils() || fb->pupils()->size() == 0 )
        return std::numeric_limits<float>::quiet_NaN();
    double sum = 0;
    for( const auto *p : *fb->pupils() )
        sum += ( p->*Field )();
    return sum / fb->pupils()->size();
}

inline std::string telem_pupilfit::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    auto              *fb = GetTelem_pupilfit_fb( buffer );
    std::ostringstream out;
    out << "[pupilfit] frame: " << fb->frame() << " reference: " << fb->reference()
        << " pupils: " << unsigned( fb->num_pupils() ) << " valid: " << fb->valid();
    if( !fb->valid() || !fb->pupils() )
        return out.str();
    unsigned index = 0;
    for( const auto *p : *fb->pupils() )
    {
        out << " [pupil " << ++index << "] x: " << p->x() << " y: " << p->y() << " D: " << p->D()
            << " med: " << p->med() << " bg: " << p->bg();
    }
    out << " [average]";
    out << " x: " << average<&PupilFitResult::x>( buffer );
    out << " y: " << average<&PupilFitResult::y>( buffer );
    out << " D: " << average<&PupilFitResult::D>( buffer );
    out << " med: " << average<&PupilFitResult::med>( buffer );
    out << " bg: " << average<&PupilFitResult::bg>( buffer );
    return out.str();
}

inline std::string
telem_pupilfit::msgJSON( void *buffer, flatlogs::msgLenT len, const uint8_t *schema, unsigned int schemaLength )
{
    std::string json = flatbuffer_log::msgJSON( buffer, len, schema, schemaLength );
    auto        end  = json.find_last_of( '}' );
    if( end == std::string::npos )
        return json;
    std::ostringstream derived;
    derived << std::setprecision( std::numeric_limits<float>::max_digits10 );
    derived << ", \"average\": ";
    auto *fb = GetTelem_pupilfit_fb( buffer );
    if( !fb->valid() || !fb->pupils() || fb->pupils()->size() == 0 )
        derived << "null";
    else
    {
        derived << "{";
        derived << "\"x\": " << average<&PupilFitResult::x>( buffer );
        derived << ", \"y\": " << average<&PupilFitResult::y>( buffer );
        derived << ", \"D\": " << average<&PupilFitResult::D>( buffer );
        derived << ", \"med\": " << average<&PupilFitResult::med>( buffer );
        derived << ", \"bg\": " << average<&PupilFitResult::bg>( buffer );
        derived << "}";
    }
    json.insert( end, derived.str() );
    return json;
}

inline logMetaDetail telem_pupilfit::getAccessor( const std::string &member )
{
    if( member == "x" )
        return {
            "X", logMeta::Vector_Float, logMeta::Continuous, reinterpret_cast<void *>( &values<&PupilFitResult::x> ) };
    if( member == "avg_x" )
        return {
            "AVG X", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &average<&PupilFitResult::x> ) };
    if( member == "y" )
        return {
            "Y", logMeta::Vector_Float, logMeta::Continuous, reinterpret_cast<void *>( &values<&PupilFitResult::y> ) };
    if( member == "avg_y" )
        return {
            "AVG Y", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &average<&PupilFitResult::y> ) };
    if( member == "D" )
        return {
            "D", logMeta::Vector_Float, logMeta::Continuous, reinterpret_cast<void *>( &values<&PupilFitResult::D> ) };
    if( member == "avg_D" )
        return {
            "AVG D", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &average<&PupilFitResult::D> ) };
    if( member == "med" )
        return { "MED",
                 logMeta::Vector_Float,
                 logMeta::Continuous,
                 reinterpret_cast<void *>( &values<&PupilFitResult::med> ) };
    if( member == "avg_med" )
        return { "AVG MED",
                 logMeta::Float,
                 logMeta::Continuous,
                 reinterpret_cast<void *>( &average<&PupilFitResult::med> ) };
    if( member == "bg" )
        return { "BG",
                 logMeta::Vector_Float,
                 logMeta::Continuous,
                 reinterpret_cast<void *>( &values<&PupilFitResult::bg> ) };
    if( member == "avg_bg" )
        return {
            "AVG BG", logMeta::Float, logMeta::Continuous, reinterpret_cast<void *>( &average<&PupilFitResult::bg> ) };
    return {};
}

} // namespace logger
} // namespace MagAOX
#endif // logger_types_telem_pupilfit_hpp
