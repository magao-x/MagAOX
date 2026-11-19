/** \file accelSimDisturbance.hpp
 * \brief Offline physical pointing conversion and explicitly calibrated actuator synthesis.
 * \ingroup accelSim_files
 */
#ifndef accelSimDisturbance_hpp
#define accelSimDisturbance_hpp

#include <Eigen/Dense>

#include "accelSimModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

/// Immutable physical pointing in the simulator's explicitly defined pupil axes.
/** Uses only sample::m_pointing, before sensor noise, ADC conversion, or legacy WFS attenuation.
 * Coordinates supplied to opd() and surface() are entrance-pupil metres in those same simulated axes.
 * The current accelSim convention is +X along increasing DM image column and +Y along increasing row.
 * This is a simulation definition, not a claim of telescope Az/El, sky, or WFS registration.
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
/** Each plane must express physical surface microns per radian of **simulated** X or Y pointing, in native
 * stream pixel order (linear index x + width*y). Tweeter dm01dispXX values are physical surface microns.
 * tipTiltProjection derives these angular responses from supplied modes and registered pupil coordinates.
 * This class neither interprets a FITS file nor supplies pupil registration or an instrument orientation.
 * Finite dimensions/data checks are numerical checks only, not validation of an optical calibration.
 * Storage is allocated at construction. The owning thread alone calls synthesize() or zero().
 */
class tipTiltBasis
{
    size_t m_width, m_height; ///< Explicit native actuator-image geometry supplied with the calibration.

    std::array<std::vector<double>, 2> m_planes; ///< Owned simulated-X/Y response planes [surface micron/rad].

    std::vector<float> m_command; ///< Complete local contribution; reused on every synthesis and zero.

  public:
    /// Validate and copy both calibration planes; no default geometry, normalization, or basis is provided.
    tipTiltBasis( size_t                     width /**< [in] native stream size[0] */,
                  size_t                     height /**< [in] native stream size[1] */,
                  const std::vector<double> &x /**< [in] surface microns per simulated X radian */,
                  const std::vector<double> &y /**< [in] surface microns per simulated Y radian */ );

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

/// Project physical pointing onto two supplied surface-micron TT modes without assuming their orientation.
/** Solve min ||sqrt(W) (B K - H)||, where B contains the modes and H = 0.5e6 * [x,y].
 * For radians theta, the modal coefficients are K theta and the surface is B K theta [micron].
 * The validated deployed CMmodesDM.00 modes have RMS = 1 micron of DM surface over dmmask, with
 * measured masked P2V of approximately 3.8492 and 3.8128 microns at coefficient one.
 * P2V = 4 * RMS is the nominal pure-tilt relation for a uniformly weighted continuous circular pupil,
 * not an exact sampled-grid requirement. D * |theta| * 1e6 / 8 is only an ideal-basis coefficient sanity
 * check. The fit derives physical scaling from supplied data without rescaling a simulated realization.
 * Input coordinates must already be registered to the simulator's axes and the entrance pupil in metres.
 * The TT mode rotation does not establish this registration or the physical pupil scale.
 * For the current DM-pupil simulation convention, the supplied nominal geometry is x = (column-24.5)*6.5/48
 * and y = (row-24.5)*6.5/48, in native order column + width*row. The diameter is 48 actuator pitches,
 * not the array width or the distance between the outermost fitting samples. Geometry remains caller supplied.
 * dmmask is appropriate fitting support for the validated calibration; current evidence does not establish
 * dmmap as either a coordinate map or fitting weights. Neither file is selected automatically here.
 * The offloader's actuator mask is different from dmmask. No telescope offload matrix or controller gain is used.
 * Pupil weights select/weight fit samples only; the full supplied modes determine the output everywhere.
 * No piston removal, mode renormalization, default pupil geometry, or FITS loading is performed.
 */
class tipTiltProjection
{
    /** \name Projection - Data
     * @{
     */
    Eigen::Matrix2d m_mapping; ///< Rows are supplied modes, columns are simulated axes [coefficient/rad].

    std::array<std::vector<double>, 2> m_response; ///< Full-image B K columns [surface micron/rad].

    std::array<double, 2> m_relativeResidual; ///< Weighted residual norm / target norm for each simulated axis.

    ///@}

  public:
    /** \name Projection
     * @{
     */
    /// Fit the two angular responses with pivoted QR and reject invalid geometry, rank, or residuals.
    /** All arrays have identical native pixel order and size. Both normalized design matrices must have
     * rank two at relative pivot threshold 1e-10. The residual tolerance is explicit; no deployed value
     * is assumed. Zero-weight coordinates may be nonfinite, but all mode pixels must remain finite.
     */
    tipTiltProjection( const std::vector<double> &mode0 /**< [in] first supplied mode [surface micron/coefficient] */,
                       const std::vector<double> &mode1 /**< [in] second supplied mode [surface micron/coefficient] */,
                       const std::vector<double> &x /**< [in] registered simulated-X pupil coordinates [m] */,
                       const std::vector<double> &y /**< [in] registered simulated-Y pupil coordinates [m] */,
                       const std::vector<double> &weights /**< [in] nonnegative pupil fitting weights */,
                       double maxRelativeResidual /**< [in] permitted relative residual per axis, in [0,1) */ );

    /// Return the supplied-mode coefficients for a physical pointing sample, without changing the sample.
    std::array<double, 2>
    coefficients( const pointingDisturbance &pointing /**< [in] simulated pointing angles */ ) const;

    /// Return angular-response planes for constructing tipTiltBasis with the supplied image dimensions.
    const std::array<std::vector<double>, 2> &response() const;

    /// Return the weighted relative surface-fit errors, without treating them as optical validation.
    const std::array<double, 2> &relativeResidual() const;
    ///@}
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

inline tipTiltProjection::tipTiltProjection( const std::vector<double> &mode0,
                                             const std::vector<double> &mode1,
                                             const std::vector<double> &x,
                                             const std::vector<double> &y,
                                             const std::vector<double> &weights,
                                             double                     maxRelativeResidual )
{
    const size_t count = mode0.size();
    if( count < 2 || count > size_t( std::numeric_limits<Eigen::Index>::max() ) || mode1.size() != count ||
        x.size() != count || y.size() != count || weights.size() != count || !std::isfinite( maxRelativeResidual ) ||
        maxRelativeResidual < 0 || maxRelativeResidual >= 1 )
        throw std::invalid_argument( "invalid TT projection dimensions or residual tolerance" );
    double maxWeight = 0;
    for( size_t pixel = 0; pixel < count; ++pixel )
    {
        if( !std::isfinite( mode0[pixel] ) || !std::isfinite( mode1[pixel] ) || !std::isfinite( weights[pixel] ) ||
            weights[pixel] < 0 )
            throw std::invalid_argument( "nonfinite TT modes or invalid pupil weights" );
        maxWeight = std::max( maxWeight, weights[pixel] );
    }
    if( maxWeight == 0 )
        throw std::invalid_argument( "empty TT fitting pupil" );

    Eigen::MatrixXd modes = Eigen::MatrixXd::Zero( count, 2 ), target = modes;
    for( size_t pixel = 0; pixel < count; ++pixel )
    {
        if( weights[pixel] == 0 )
            continue;
        if( !std::isfinite( x[pixel] ) || !std::isfinite( y[pixel] ) )
            throw std::invalid_argument( "nonfinite active pupil coordinates" );
        const double weight = std::sqrt( weights[pixel] / maxWeight );
        modes( pixel, 0 )   = weight * mode0[pixel];
        modes( pixel, 1 )   = weight * mode1[pixel];
        target( pixel, 0 )  = weight * x[pixel] * 0.5e6;
        target( pixel, 1 )  = weight * y[pixel] * 0.5e6;
    }
    if( !target.allFinite() )
        throw std::overflow_error( "TT target surface overflow" );
    Eigen::Vector2d modeNorm, targetNorm;
    for( size_t axis = 0; axis < 2; ++axis )
    {
        modeNorm[axis]   = modes.col( axis ).stableNorm();
        targetNorm[axis] = target.col( axis ).stableNorm();
        if( !std::isfinite( modeNorm[axis] ) || !std::isfinite( targetNorm[axis] ) || modeNorm[axis] <= 0 ||
            targetNorm[axis] <= 0 )
            throw std::invalid_argument( "degenerate TT mode or pupil coordinate" );
        modes.col( axis ) /= modeNorm[axis];
        target.col( axis ) /= targetNorm[axis];
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> fit( modes ), pupil( target );
    fit.setThreshold( 1e-10 );
    pupil.setThreshold( 1e-10 );
    if( fit.rank() != 2 || pupil.rank() != 2 )
        throw std::invalid_argument( "rank-deficient TT modes or pupil coordinates" );
    const Eigen::Matrix2d normalized = fit.solve( target );
    for( size_t axis = 0; axis < 2; ++axis )
    {
        m_relativeResidual[axis] = ( modes * normalized.col( axis ) - target.col( axis ) ).stableNorm();
        if( !std::isfinite( m_relativeResidual[axis] ) || m_relativeResidual[axis] > maxRelativeResidual )
            throw std::invalid_argument(
                "TT modes cannot reconstruct the requested physical surface within tolerance" );
        for( size_t mode = 0; mode < 2; ++mode )
            m_mapping( mode, axis ) = normalized( mode, axis ) * targetNorm[axis] / modeNorm[mode];
    }
    if( !m_mapping.allFinite() )
        throw std::overflow_error( "TT coefficient mapping overflow" );
    for( size_t axis = 0; axis < 2; ++axis )
    {
        m_response[axis].resize( count );
        for( size_t pixel = 0; pixel < count; ++pixel )
        {
            const double value = mode0[pixel] * m_mapping( 0, axis ) + mode1[pixel] * m_mapping( 1, axis );
            if( !std::isfinite( value ) )
                throw std::overflow_error( "TT angular response overflow" );
            m_response[axis][pixel] = value;
        }
    }
}

inline std::array<double, 2> tipTiltProjection::coefficients( const pointingDisturbance &pointing ) const
{
    const Eigen::Vector2d value = m_mapping * Eigen::Vector2d( pointing.radians()[0], pointing.radians()[1] );
    if( !value.allFinite() )
        throw std::overflow_error( "TT modal coefficient overflow" );
    return { value[0], value[1] };
}

inline const std::array<std::vector<double>, 2> &tipTiltProjection::response() const
{
    return m_response;
}

inline const std::array<double, 2> &tipTiltProjection::relativeResidual() const
{
    return m_relativeResidual;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
