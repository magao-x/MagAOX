/** \file dlObservation.hpp
 * \brief Checked position and camera readbacks for collection metadata.
 * \ingroup dlDataCollection_files
 */
#ifndef dlObservation_hpp
#define dlObservation_hpp

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../INDI/libcommon/IndiProperty.hpp"

namespace MagAOX
{
namespace app
{
namespace dlCollection
{
/// One settled named position, guarded by the application's INDI mutex.
struct PresetReadback
{
    /// Selected preset, or empty for unknown, ambiguous or moving positions.
    std::string m_value;

    /// Local monotonic receipt time; refreshed by read-only INDI requests.
    std::chrono::steady_clock::time_point m_received{};

    /// Controller status, refreshed separately to reject disconnected readbacks.
    std::string m_state;

    /// Receipt time of the controller status.
    std::chrono::steady_clock::time_point m_stateReceived{};

    /// Local Unix receipt time for the run's provenance record.
    uint64_t m_time{ 0 };

    /// Accept exactly one selected switch in an Idle or Ok property.
    void update( const pcf::IndiProperty &property /**< [in] Named-position readback. */ );

    /// Record the controller's fsm.state without issuing any hardware command.
    void updateState( const pcf::IndiProperty &property /**< [in] Controller status. */ );

    /// Require a known position received within the last five seconds.
    bool valid( std::chrono::steady_clock::time_point now /**< [in] Local monotonic time. */ ) const;
};

/// One positive camera setting, guarded by the application's INDI mutex.
struct CameraReadback
{
    /// Reported current value; zero means missing, invalid or changing.
    double m_value{ 0 };

    /// Local monotonic receipt time; refreshed by read-only INDI requests.
    std::chrono::steady_clock::time_point m_received{};

    /// Local Unix receipt time for the run's provenance record.
    uint64_t m_time{ 0 };

    /// Accept a finite positive current element in an Idle or Ok number property.
    void update( const pcf::IndiProperty &property /**< [in] Camera setting readback. */ );

    /// Require a positive current value received within the last five seconds.
    bool valid( std::chrono::steady_clock::time_point now /**< [in] Local monotonic time. */ ) const;
};

/// Serialize immutable positions and camera settings without relying on directory labels.
inline std::string observationContext( const PresetReadback &nd /**< [in] fwtelsim readback. */,
                                       const PresetReadback &flip /**< [in] flipwfsf readback. */,
                                       const CameraReadback &fps /**< [in] camwfs frame rate in Hz. */,
                                       const CameraReadback &emgain /**< [in] camwfs E.M. gain factor. */ );

inline void CameraReadback::update( const pcf::IndiProperty &property )
{
    m_value    = 0;
    m_received = std::chrono::steady_clock::now();
    m_time = std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
                 .count();
    if( property.getType() != pcf::IndiProperty::Number || !property.find( "current" ) ||
        ( property.getState() != pcf::IndiProperty::Idle && property.getState() != pcf::IndiProperty::Ok ) )
        return;
    // IndiElement::get<double>() does not detect failed numeric conversion.
    std::istringstream input( property["current"].get() );
    double             value = 0;
    if( input >> value && ( input >> std::ws ).eof() && std::isfinite( value ) && value > 0 )
        m_value = value;
}

inline bool CameraReadback::valid( std::chrono::steady_clock::time_point now ) const
{
    return std::isfinite( m_value ) && m_value > 0 && m_received != std::chrono::steady_clock::time_point{} &&
           now >= m_received && now - m_received < std::chrono::seconds( 5 );
}

inline void PresetReadback::update( const pcf::IndiProperty &property )
{
    m_value.clear();
    m_received = std::chrono::steady_clock::now();
    m_time = std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
                 .count();
    if( property.getType() != pcf::IndiProperty::Switch ||
        ( property.getState() != pcf::IndiProperty::Idle && property.getState() != pcf::IndiProperty::Ok ) )
        return;
    for( const auto &element : property.getElements() )
    {
        const auto state = element.second.getSwitchState();
        if( state == pcf::IndiElement::On )
        {
            if( !m_value.empty() || element.first.empty() ||
                element.first.find_first_of( "\r\n=" ) != std::string::npos )
            {
                m_value.clear();
                return;
            }
            m_value = element.first;
        }
        else if( state != pcf::IndiElement::Off )
        {
            m_value.clear();
            return;
        }
    }
}

inline bool PresetReadback::valid( std::chrono::steady_clock::time_point now ) const
{
    return !m_value.empty() && m_received != std::chrono::steady_clock::time_point{} &&
           m_stateReceived != std::chrono::steady_clock::time_point{} && now >= m_received &&
           now - m_received < std::chrono::seconds( 5 ) && now >= m_stateReceived &&
           now - m_stateReceived < std::chrono::seconds( 5 ) &&
           ( m_state == "READY" || m_state == "OPERATING" || m_state == "POWEROFF" );
}

inline void PresetReadback::updateState( const pcf::IndiProperty &property )
{
    m_stateReceived = std::chrono::steady_clock::now();
    m_state         = property.getType() == pcf::IndiProperty::Text && property.find( "state" )
                          ? property["state"].get<std::string>()
                          : "";
}

inline std::string observationContext( const PresetReadback &nd,
                                       const PresetReadback &flip,
                                       const CameraReadback &fps,
                                       const CameraReadback &emgain )
{
    const auto now = std::chrono::steady_clock::now();
    if( !nd.valid( now ) || !flip.valid( now ) || ( flip.m_value != "in" && flip.m_value != "out" ) )
        throw std::runtime_error( "missing, moving or stale fwtelsim/flipwfsf readback" );
    if( !fps.valid( now ) || !emgain.valid( now ) )
        throw std::runtime_error( "missing, changing or stale camwfs fps/emgain readback" );
    std::ostringstream camera;
    camera.precision( std::numeric_limits<double>::max_digits10 );
    camera << "camwfs_fps=" << fps.m_value << "\ncamwfs_emgain=" << emgain.m_value
           << "\ncamwfs_fps_property=fps.current\ncamwfs_emgain_property=emgain.current\ncamwfs_fps_readback_ns="
           << fps.m_time << "\ncamwfs_emgain_readback_ns=" << emgain.m_time << '\n';
    return "nd=" + nd.m_value + "\nflipwfsf=" + flip.m_value + "\nfwtelsim_state=" + nd.m_state +
           "\nflipwfsf_state=" + flip.m_state +
           "\nobservation_source=INDI\nfwtelsim_property=filterName\nflipwfsf_property=presetName\nfwtelsim_readback_"
           "ns=" +
           std::to_string( nd.m_time ) + "\nflipwfsf_readback_ns=" + std::to_string( flip.m_time ) + "\n" +
           camera.str();
}

} // namespace dlCollection
} // namespace app
} // namespace MagAOX
#endif // dlObservation_hpp
