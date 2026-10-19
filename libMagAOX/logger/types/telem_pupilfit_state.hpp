/** \file telem_pupilfit_state.hpp
 * \brief Pupil fitting configuration, setpoints, and averaging boundaries.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_pupilfit_state_hpp
#define logger_types_telem_pupilfit_state_hpp

#include <sstream>
#include "generated/telem_pupilfit_state_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{
/// Settings applying to subsequent fits until the next state record.
/** Modes are 0=default, 1=reference image, 2=user; active mode can also be 3=current fit
 * or 255 before allocation.
 * \ingroup logger_types
 */
struct telem_pupilfit_state : public flatbuffer_log
{
    /// Logger dispatch identifier.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_PUPILFIT_STATE;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Last emission time, maintained by the telemeter.
    static timespec lastRecord;

    /// Serialized configuration input.
    struct messageT : public fbMessage
    {
        /// Build a configuration snapshot in active quadrant order.
        messageT( const std::vector<PupilFitSetpoint> &setpoints /**< [in] active per-pupil references */,
                  uint8_t requestedMode /**< [in] selected reference mode */,
                  uint8_t activeMode /**< [in] mode that supplied the active setpoints */,
                  float threshold /**< [in] normalized edge threshold */,
                  float pupIndex /**< [in] illuminated-pupil quantile */,
                  float bgIndex /**< [in] background quantile */,
                  uint32_t width /**< [in] input image width */,
                  uint32_t height /**< [in] input image height */,
                  bool averaging /**< [in] whether live fits enter temporal statistics */,
                  uint64_t window /**< [in] averaging window counter; incremented on every reset */ );
    };

    /// Verify a serialized state record and its supported pupil count.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log record */,
                        flatlogs::msgLenT len /**< [in] serialized message length */ );

    /// Format setpoints, fitting settings, and averaging boundaries.
    static std::string msgString( void *buffer /**< [in] serialized message */,
                                  flatlogs::msgLenT len /**< [in] serialized message length */ );

    /// Extract a setpoint coordinate for all active pupils.
    template <float (PupilFitSetpoint::*Field)() const>
    static std::vector<float> values( void *buffer /**< [in] serialized message */ );

    /// Get metadata accessors for the active setpoint vectors.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );
};

inline telem_pupilfit_state::messageT::messageT( const std::vector<PupilFitSetpoint> &setpoints,
                                               uint8_t requestedMode, uint8_t activeMode,
                                               float threshold, float pupIndex, float bgIndex,
                                               uint32_t width, uint32_t height, bool averaging, uint64_t window )
{
    auto points = builder.CreateVectorOfStructs( setpoints );
    builder.Finish( CreateTelem_pupilfit_state_fb( builder, points, requestedMode, activeMode, threshold,
                                                  pupIndex, bgIndex, width, height, averaging, window ) );
}

inline bool telem_pupilfit_state::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    auto *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );
    if( !VerifyTelem_pupilfit_state_fbBuffer( verifier ) ) return false;
    auto *fb = GetTelem_pupilfit_state_fb( payload );
    return fb->setpoints() && (fb->setpoints()->size() == 3 || fb->setpoints()->size() == 4);
}

inline std::string telem_pupilfit_state::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    auto *fb = GetTelem_pupilfit_state_fb( buffer );
    std::ostringstream out;
    out << "[pupilfit state] requested: " << unsigned(fb->requested_mode())
        << " active: " << unsigned(fb->active_mode()) << " threshold: " << fb->threshold()
        << " pup_index: " << fb->pup_index() << " bg_index: " << fb->bg_index()
        << " size: " << fb->width() << "x" << fb->height()
        << " averaging: " << fb->averaging() << " window: " << fb->window();
    unsigned index = 0;
    if( fb->setpoints() )
    {
        for( const auto *p : *fb->setpoints() )
            out << " [setpoint " << ++index << "] x: " << p->x() << " y: " << p->y() << " D: " << p->D();
    }
    return out.str();
}

template <float (PupilFitSetpoint::*Field)() const>
inline std::vector<float> telem_pupilfit_state::values( void *buffer )
{
    auto *fb = GetTelem_pupilfit_state_fb( buffer );
    std::vector<float> result;
    if( fb->setpoints() )
        for( const auto *p : *fb->setpoints() ) result.push_back( (p->*Field)() );
    return result;
}

inline logMetaDetail telem_pupilfit_state::getAccessor( const std::string &member )
{
    if( member == "set_x" ) return { "SET X", logMeta::Vector_Float, logMeta::State,
                                       reinterpret_cast<void *>( &values<&PupilFitSetpoint::x> ) };
    if( member == "set_y" ) return { "SET Y", logMeta::Vector_Float, logMeta::State,
                                       reinterpret_cast<void *>( &values<&PupilFitSetpoint::y> ) };
    if( member == "set_D" ) return { "SET D", logMeta::Vector_Float, logMeta::State,
                                       reinterpret_cast<void *>( &values<&PupilFitSetpoint::D> ) };
    return {};
}

} // namespace logger
} // namespace MagAOX
#endif // logger_types_telem_pupilfit_state_hpp
