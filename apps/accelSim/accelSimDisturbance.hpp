/** \file accelSimDisturbance.hpp
 * \brief Offline physical pointing conversion and explicitly calibrated actuator synthesis.
 * \ingroup accelSim_files
 */
#ifndef accelSimDisturbance_hpp
#define accelSimDisturbance_hpp

#include "accelSimModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

/// Immutable physical pointing in the simulator's X/Y coordinates, independent of any DM orientation.
/** Uses only sample::m_pointing, before sensor noise, ADC conversion, or legacy WFS attenuation.
 * Coordinates supplied to opd() and surface() are entrance-pupil metres in those same simulated axes.
 * These evaluations describe an ideal plane, not an actuator calibration or a synthetic WFS.
 */
class pointingDisturbance
{
    std::array<double, 2> m_radians; ///< Noise-free angular displacement in simulated X/Y axes [rad].

  public:
    /// Copy the existing physical pointing without advancing or modifying the simulation.
    explicit pointingDisturbance( const sample &value /**< [in] simultaneous physical and ADC sample */ );

    /// Return the physical angles in radians, with no assumed instrument-axis transformation.
    const std::array<double, 2> &radians() const;

    /// Evaluate the ideal optical path difference [m] at a supplied entrance-pupil coordinate.
    double opd( double x /**< [in] simulated X entrance-pupil coordinate [m] */,
                double y /**< [in] simulated Y entrance-pupil coordinate [m] */ ) const;

    /// Evaluate the equivalent reflective surface displacement [micron], including OPD/2 exactly once.
    double surface( double x /**< [in] simulated X entrance-pupil coordinate [m] */,
                    double y /**< [in] simulated Y entrance-pupil coordinate [m] */ ) const;

    /// Return the ideal surface peak-to-valley [micron] over a circular entrance pupil.
    double surfaceP2V( double diameter = 6.5 /**< [in] entrance-pupil diameter [m], not physical DM diameter */ ) const;
};

/// Numerical actuator synthesis from explicitly supplied angular calibration planes; never opens a stream.
/** Each plane must express native command units per radian of **simulated** X or Y pointing, in native
 * stream pixel order (linear index x + width*y). The caller must establish calibration provenance,
 * units, pupil registration, reflection physics, and axis signs/orientation; this class supplies none.
 * It neither interprets a FITS file nor assumes that native command values equal physical surface samples.
 * Finite dimensions/data checks are numerical checks only, not validation of an optical calibration.
 * Storage is allocated at construction. The owning thread alone calls synthesize() or zero().
 */
class tipTiltBasis
{
    size_t m_width, m_height; ///< Explicit native actuator-image geometry supplied with the calibration.

    std::array<std::vector<double>, 2> m_planes; ///< Owned X/Y calibration planes [command units/rad].

    std::vector<float> m_command; ///< Complete local contribution; reused on every synthesis and zero.

  public:
    /// Validate and copy both calibration planes; no default geometry, normalization, or basis is provided.
    tipTiltBasis( size_t                     width /**< [in] native stream size[0] */,
                  size_t                     height /**< [in] native stream size[1] */,
                  const std::vector<double> &x /**< [in] response to one simulated X radian */,
                  const std::vector<double> &y /**< [in] response to one simulated Y radian */ );

    /// Return the explicitly configured native image width.
    size_t width() const;

    /// Return the explicitly configured native image height.
    size_t height() const;

    /// Replace the local contribution with the calibrated sum; invalid/overflowed output is zeroed before throwing.
    /** No reflection factor or pupil-diameter scaling is added: both belong in the supplied calibration.
     * The returned reference remains owned by this object and changes on subsequent synthesis/zero calls.
     */
    const std::vector<float> &synthesize( const pointingDisturbance &pointing /**< [in] physical simulated angles */ );

    /// Zero the local contribution only; this does not publish or clear an applied DM command.
    const std::vector<float> &zero();
};

inline pointingDisturbance::pointingDisturbance( const sample &value )
{
    for( size_t axis = 0; axis < 2; ++axis )
    {
        if( !std::isfinite( value.m_pointing[axis] ) )
            throw std::invalid_argument( "nonfinite physical pointing" );
        m_radians[axis] = value.m_pointing[axis] * arcsecToRad;
    }
}

inline const std::array<double, 2> &pointingDisturbance::radians() const
{
    return m_radians;
}

inline double pointingDisturbance::opd( double x, double y ) const
{
    if( !std::isfinite( x ) || !std::isfinite( y ) )
        throw std::invalid_argument( "nonfinite entrance-pupil coordinate" );
    const double value = m_radians[0] * x + m_radians[1] * y;
    if( !std::isfinite( value ) )
        throw std::overflow_error( "optical path difference overflow" );
    return value;
}

inline double pointingDisturbance::surface( double x, double y ) const
{
    const double value = opd( x, y ) * 0.5e6;
    if( !std::isfinite( value ) )
        throw std::overflow_error( "reflective surface overflow" );
    return value;
}

inline double pointingDisturbance::surfaceP2V( double diameter ) const
{
    if( !std::isfinite( diameter ) || diameter <= 0 )
        throw std::invalid_argument( "entrance-pupil diameter must be finite and positive" );
    const double value = std::hypot( m_radians[0], m_radians[1] ) * diameter * 0.5e6;
    if( !std::isfinite( value ) )
        throw std::overflow_error( "surface peak-to-valley overflow" );
    return value;
}

inline tipTiltBasis::tipTiltBasis( size_t                     width,
                                   size_t                     height,
                                   const std::vector<double> &x,
                                   const std::vector<double> &y )
    : m_width( width ), m_height( height )
{
    if( !width || !height || width > std::numeric_limits<size_t>::max() / height || x.size() != width * height ||
        y.size() != width * height )
        throw std::invalid_argument( "invalid tip/tilt basis dimensions" );
    for( const auto *plane : { &x, &y } )
        for( double value : *plane )
            if( !std::isfinite( value ) )
                throw std::invalid_argument( "nonfinite tip/tilt calibration" );
    m_planes = { x, y };
    m_command.resize( x.size(), 0 );
}

inline size_t tipTiltBasis::width() const
{
    return m_width;
}

inline size_t tipTiltBasis::height() const
{
    return m_height;
}

inline const std::vector<float> &tipTiltBasis::synthesize( const pointingDisturbance &pointing )
{
    for( size_t pixel = 0; pixel < m_command.size(); ++pixel )
    {
        const double value = pointing.radians()[0] * m_planes[0][pixel] + pointing.radians()[1] * m_planes[1][pixel];
        if( !std::isfinite( value ) || std::abs( value ) > std::numeric_limits<float>::max() )
        {
            zero();
            throw std::overflow_error( "calibrated actuator command overflow" );
        }
        m_command[pixel] = static_cast<float>( value );
    }
    return m_command;
}

inline const std::vector<float> &tipTiltBasis::zero()
{
    std::fill( m_command.begin(), m_command.end(), 0 );
    return m_command;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
