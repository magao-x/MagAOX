/** \file accelSimDisturbance_test.cpp
 * \brief Hardware-free tests of physical disturbance conversion and explicit actuator bases.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimDisturbance.hpp"

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app::accelSimDetail;

/// \cond ACCELSIM_TT_FIXTURE
/// Synthetic circular-pupil quadrature with exact tilt RMS and four extra zero-weight boundary samples.
struct ttFixture
{
    std::vector<double> m_x, m_y;         ///< Entrance-pupil coordinates in the simulator's axes [m].
    std::vector<double> m_mode0, m_mode1; ///< Rotated RMS-one surface modes [micron/coefficient].
    std::vector<double> m_weights;        ///< Two radial Gauss nodes times eight azimuths, plus excluded boundaries.

    /// Build a 5-by-4 storage fixture; its sample locations are explicit rather than inferred from indices.
    ttFixture( double angle /**< [in] known synthetic rotation [rad] */,
               double diameter = 6.5 /**< [in] synthetic entrance-pupil diameter [m] */ );
};

ttFixture::ttFixture( double angle, double diameter )
    : m_x( 20 ), m_y( 20 ), m_mode0( 20 ), m_mode1( 20 ), m_weights( 20, 1 )
{
    for( size_t pixel = 0; pixel < 20; ++pixel )
    {
        double radius, azimuth;
        if( pixel < 16 )
        {
            radius  = diameter / 2 * std::sqrt( .5 + ( pixel < 8 ? -1 : 1 ) / ( 2 * std::sqrt( 3.0 ) ) );
            azimuth = ( pixel % 8 ) * pi / 4;
        }
        else
        {
            radius           = diameter / 2;
            azimuth          = angle + ( pixel - 16 ) * pi / 2;
            m_weights[pixel] = 0;
        }
        m_x[pixel]     = radius * std::cos( azimuth );
        m_y[pixel]     = radius * std::sin( azimuth );
        m_mode0[pixel] = 4 / diameter * ( std::cos( angle ) * m_x[pixel] + std::sin( angle ) * m_y[pixel] );
        m_mode1[pixel] = 4 / diameter * ( -std::sin( angle ) * m_x[pixel] + std::cos( angle ) * m_y[pixel] );
    }
}
/// \endcond

/** \brief Disturbances consume full physical pointing, ignoring sensor, ADC, and legacy WFS fields.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim disturbance uses only physical pointing", "[accelSim][disturbance]" )
{
    sample value;
    value.m_pointing = { 1, -2 };
    // Invalid unrelated fields deliberately catch accidental use of sensor or coupled pointing.
    value.m_wfsPointing = value.m_acceleration = { NAN, INFINITY };
    value.m_adc                                = { 0, 4095 };
    const pointingDisturbance pointing( value );
    REQUIRE( pointing.radians()[0] == Approx( pi / 648000 ) );
    REQUIRE( pointing.radians()[1] == Approx( -2 * pi / 648000 ) );
    REQUIRE( value.m_pointing == std::array<double, 2>{ 1, -2 } );
    value.m_pointing = {};
    REQUIRE( pointing.radians()[0] == Approx( pi / 648000 ) ); // Owns a snapshot, not model storage.
    const pointingDisturbance zero( value );
    REQUIRE( zero.surface( 3.25, -3.25 ) == 0 );
    REQUIRE( zero.surfaceP2V() == 0 );
}

/** \brief Positive/negative X/Y and mixed tilts satisfy the OPD plane and reflective factor of two.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim disturbance physical plane and optical scale", "[accelSim][disturbance]" )
{
    for( const auto angles : { std::array<double, 2>{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 3, -4 } } )
    {
        sample value;
        value.m_pointing = { angles[0] * 1e-6 / arcsecToRad, angles[1] * 1e-6 / arcsecToRad };
        const pointingDisturbance pointing( value );
        REQUIRE( pointing.opd( 2, -3 ) == Approx( ( 2 * angles[0] - 3 * angles[1] ) * 1e-6 ) );
        REQUIRE( pointing.surface( 2, -3 ) == Approx( ( 2 * angles[0] - 3 * angles[1] ) / 2 ) );
        REQUIRE( pointing.surface( 0, 0 ) == 0 );
        const double magnitude = std::hypot( angles[0], angles[1] );
        // Extremal points on a circle lie along the pointing vector, not the square's corners.
        const double x = 3.25 * angles[0] / magnitude, y = 3.25 * angles[1] / magnitude;
        REQUIRE( pointing.surface( x, y ) - pointing.surface( -x, -y ) == Approx( pointing.surfaceP2V() ) );
        REQUIRE( pointing.surfaceP2V() == Approx( 3.25 * magnitude ) );
        REQUIRE( pointing.surfaceP2V( 13 ) == Approx( 2 * pointing.surfaceP2V() ) );
    }
    sample diffraction;
    diffraction.m_pointing = { 900e-9 / 6.5 / arcsecToRad, 0 };
    const pointingDisturbance pointing( diffraction );
    REQUIRE( pointing.surfaceP2V() == Approx( .450 ) );
    REQUIRE( diffraction.m_pointing[0] * 1000 == Approx( 28.5597424 ) );
    REQUIRE( pointing.opd( 3.25, 0 ) - pointing.opd( -3.25, 0 ) == Approx( 900e-9 ) );
}

/** \brief Every retained mode preserves its signed acceleration-to-pointing relationship without integration.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim modal acceleration agrees with physical disturbance", "[accelSim][disturbance]" )
{
    parameters config;
    config.m_noisePSD = 0;
    REQUIRE( config.m_leverArm == 10 );
    for( const auto &spec : referenceModes( config.m_modeSeed ) )
    {
        model simulation;
        simulation.configure( config, { spec } );
        const double omega = 2 * pi * spec.m_frequency;
        for( size_t tick = 0; tick < 8; ++tick )
        {
            const auto               &value = simulation.step();
            const pointingDisturbance pointing( value );
            for( size_t axis = 0; axis < 2; ++axis )
                REQUIRE( value.m_acceleration[axis] ==
                         Approx( -omega * omega * config.m_leverArm * pointing.radians()[axis] ).margin( 1e-14 ) );
        }
    }
}

/** \brief The nominal stationary population matches the notebook scale without finite-record renormalization.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim nominal disturbance ensemble scale", "[accelSim][disturbance]" )
{
    parameters config;
    model      simulation;
    simulation.configure( config, referenceModes( config.m_modeSeed ) );
    double variance = 0;
    for( const auto &mode : simulation.m_modes )
        for( const auto &axis : mode.m_axes )
            variance += axis.m_rms * axis.m_rms;
    // Use radial ensemble RMS as a deterministic angular magnitude; this is not a generated record.
    sample rms;
    rms.m_pointing = { std::sqrt( variance ), 0 };
    const pointingDisturbance pointing( rms );
    REQUIRE( rms.m_pointing[0] * 1000 == Approx( 2.9 ).margin( .15 ) );
    REQUIRE( pointing.surfaceP2V() == Approx( .046 ).margin( .002 ) );
}

/** \brief Explicit synthetic calibration controls axes, signs, layout, and units without extra physical factors.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim explicit actuator calibration and local zero", "[accelSim][disturbance]" )
{
    // Deliberately non-square, with simulated X on reversed rows and simulated Y on columns.
    // These are synthetic surface microns per radian, not an instrument calibration.
    std::vector<double> x{ 2e6, 2e6, 2e6, -2e6, -2e6, -2e6 };
    std::vector<double> y{ -1e6, 0, 1e6, -1e6, 0, 1e6 };
    tipTiltBasis        basis( 3, 2, x, y );
    REQUIRE( basis.width() == 3 );
    REQUIRE( basis.height() == 2 );
    const auto &command = basis.zero();
    const auto *storage = command.data();
    REQUIRE( command == std::vector<float>( 6, 0 ) );
    for( const auto angles : { std::array<double, 2>{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 3, -4 }, { 0, 0 } } )
    {
        sample value;
        value.m_pointing = { angles[0] * 1e-6 / arcsecToRad, angles[1] * 1e-6 / arcsecToRad };
        const pointingDisturbance pointing( value );
        for( size_t repeat = 0; repeat < 3; ++repeat )
        {
            basis.synthesize( pointing );
            REQUIRE( command.data() == storage );
            for( size_t row = 0; row < 2; ++row )
                for( size_t column = 0; column < 3; ++column )
                {
                    const double expected = ( row ? -2 : 2 ) * angles[0] + ( double( column ) - 1 ) * angles[1];
                    REQUIRE( command[column + 3 * row] == Approx( expected ).margin( 1e-6 ) );
                }
        }
    }
    // Plane ownership is independent of caller lifetime/mutation.
    x.assign( 6, 0 );
    y.assign( 6, 0 );
    sample value;
    value.m_pointing = { 1e-6 / arcsecToRad, 0 };
    basis.synthesize( pointingDisturbance( value ) );
    REQUIRE( command[0] == Approx( 2 ) );
    REQUIRE( basis.zero() == std::vector<float>( 6, 0 ) );
    REQUIRE( basis.zero().data() == storage );
}

/** \brief Reading/synthesizing a disturbance cannot change RNG progression, ADC, catch-up, or rate changes.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim disturbance leaves existing model sequence unchanged", "[accelSim][disturbance]" )
{
    parameters config;
    const auto specs = referenceModes( config.m_modeSeed );
    model      baseline, observed;
    baseline.configure( config, specs );
    observed.configure( config, specs );
    const ttFixture         fixture( .37 );
    const tipTiltProjection projection(
        fixture.m_mode0, fixture.m_mode1, fixture.m_x, fixture.m_y, fixture.m_weights, 1e-12 );
    tipTiltBasis basis( 5, 4, projection.response()[0], projection.response()[1] );
    for( double rate : { 6000, 8000 } )
    {
        baseline.setRate( rate );
        observed.setRate( rate );
        for( size_t tick = 0; tick < 256; ++tick )
        {
            const uint64_t skipped = tick % 64 == 0 ? 9 : tick % 17 == 0 ? 3 : 0;
            REQUIRE( baseline.advance( skipped ) == observed.advance( skipped ) );
            const auto                ticks = observed.m_ticks;
            const pointingDisturbance pointing( observed.m_sample );
            REQUIRE( std::isfinite( projection.coefficients( pointing )[0] ) );
            basis.synthesize( pointing );
            basis.synthesize( pointing );
            REQUIRE( observed.m_ticks == ticks );
            REQUIRE( observed.m_ticks == baseline.m_ticks );
            REQUIRE( observed.m_sample.m_pointing == baseline.m_sample.m_pointing );
            REQUIRE( observed.m_sample.m_wfsPointing == baseline.m_sample.m_wfsPointing );
            REQUIRE( observed.m_sample.m_acceleration == baseline.m_sample.m_acceleration );
            REQUIRE( observed.m_sample.m_adc == baseline.m_sample.m_adc );
            REQUIRE( observed.m_clipLow == baseline.m_clipLow );
            REQUIRE( observed.m_clipHigh == baseline.m_clipHigh );
        }
    }
}

/** \brief Nonfinite physical inputs, invalid pupils, and overflowing optical quantities are rejected.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim disturbance rejects invalid physical inputs", "[accelSim][disturbance]" )
{
    sample value;
    for( double invalid : { NAN, INFINITY, -INFINITY } )
        for( size_t axis = 0; axis < 2; ++axis )
        {
            value.m_pointing       = {};
            value.m_pointing[axis] = invalid;
            REQUIRE_THROWS_AS( pointingDisturbance{ value }, std::invalid_argument );
        }
    value.m_pointing = { 1, 2 };
    const pointingDisturbance pointing( value );
    for( double invalid : { 0.0, -1.0, double( NAN ), double( INFINITY ) } )
        REQUIRE_THROWS_AS( pointing.surfaceP2V( invalid ), std::invalid_argument );
    REQUIRE_THROWS_AS( pointing.opd( NAN, 0 ), std::invalid_argument );
    REQUIRE_THROWS_AS( pointing.opd( 0, INFINITY ), std::invalid_argument );
    value.m_pointing = { std::numeric_limits<double>::max(), 0 };
    const pointingDisturbance huge( value );
    REQUIRE_THROWS_AS( huge.opd( 1e10, 0 ), std::overflow_error );
    REQUIRE_THROWS_AS( huge.surface( 1, 0 ), std::overflow_error );
    REQUIRE_THROWS_AS( huge.surfaceP2V(), std::overflow_error );
}

/** \brief Calibration geometry/data are validated; failed synthesis clears partial and previous local output.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim rejects invalid bases and command overflow", "[accelSim][disturbance]" )
{
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 0, 1, {}, {} } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 1, 0, {}, {} } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ std::numeric_limits<size_t>::max(), 2, {}, {} } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 2, 1, { 1 }, { 1, 2 } } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 2, 1, { 1, 2 }, { 1 } } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 1, 1, { NAN }, { 1 } } ), std::invalid_argument );
    REQUIRE_THROWS_AS( ( tipTiltBasis{ 1, 1, { 1 }, { INFINITY } } ), std::invalid_argument );
    for( double large : { double( std::numeric_limits<float>::max() ), std::numeric_limits<double>::max() } )
    {
        tipTiltBasis basis( 2, 1, { 1, large }, { 1, 0 } );
        sample       value;
        value.m_pointing    = { 0, 1 / arcsecToRad };
        const auto &command = basis.synthesize( pointingDisturbance( value ) );
        REQUIRE( command[0] == Approx( 1 ) );
        const auto *storage = command.data();
        for( double sign : { 1.0, -1.0 } )
        {
            value.m_pointing = { sign * 2 / arcsecToRad, 0 };
            REQUIRE_THROWS_AS( basis.synthesize( pointingDisturbance( value ) ), std::overflow_error );
            REQUIRE( command == std::vector<float>( 2, 0 ) );
            REQUIRE( command.data() == storage );
        }
        value.m_pointing = {};
        REQUIRE_NOTHROW( basis.synthesize( pointingDisturbance( value ) ) );
    }
}

/** \brief Arbitrary rotated TT modes reconstruct the desired physical micron surface with the calibrated scale.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim projects onto rotated RMS one TT modes", "[accelSim][disturbance][projection]" )
{
    for( double diameter : { 6.5, 8.0 } )
        for( double angle : { 0.0, .17, .71, pi / 2, 2.4 } )
        {
            const ttFixture f( angle, diameter );
            double          rms0 = 0, rms1 = 0;
            for( size_t pixel = 0; pixel < 16; ++pixel )
            {
                rms0 += f.m_mode0[pixel] * f.m_mode0[pixel] / 16;
                rms1 += f.m_mode1[pixel] * f.m_mode1[pixel] / 16;
            }
            REQUIRE( std::sqrt( rms0 ) == Approx( 1 ) );
            REQUIRE( std::sqrt( rms1 ) == Approx( 1 ) );
            REQUIRE( f.m_mode0[16] - f.m_mode0[18] == Approx( 4 ) );
            REQUIRE( f.m_mode1[17] - f.m_mode1[19] == Approx( 4 ) );
            const tipTiltProjection projection( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
            for( double residual : projection.relativeResidual() )
                REQUIRE( residual < 1e-14 );
            tipTiltBasis basis( 5, 4, projection.response()[0], projection.response()[1] );
            for( const auto xy :
                 { std::array<double, 2>{ 0, 0 }, { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 3, -4 } } )
            {
                sample value;
                value.m_pointing = { .001 * xy[0], .001 * xy[1] }; // arcsec, i.e. xy in mas
                const pointingDisturbance pointing( value );
                const auto                c     = projection.coefficients( pointing );
                const double              scale = diameter * 1e6 / 8;
                REQUIRE( c[0] == Approx( scale * ( std::cos( angle ) * pointing.radians()[0] +
                                                   std::sin( angle ) * pointing.radians()[1] ) )
                                     .margin( 1e-15 ) );
                REQUIRE( c[1] == Approx( scale * ( -std::sin( angle ) * pointing.radians()[0] +
                                                   std::cos( angle ) * pointing.radians()[1] ) )
                                     .margin( 1e-15 ) );
                REQUIRE( 4 * std::hypot( c[0], c[1] ) == Approx( pointing.surfaceP2V( diameter ) ).margin( 1e-15 ) );
                const auto &command         = basis.synthesize( pointing );
                double      surfaceVariance = 0;
                for( size_t pixel = 0; pixel < 20; ++pixel )
                {
                    const double physical = pointing.surface( f.m_x[pixel], f.m_y[pixel] );
                    surfaceVariance += f.m_weights[pixel] * physical * physical / 16;
                    REQUIRE( c[0] * f.m_mode0[pixel] + c[1] * f.m_mode1[pixel] == Approx( physical ).margin( 1e-15 ) );
                    REQUIRE( command[pixel] == Approx( physical ).epsilon( 1e-6 ).margin( 1e-9 ) );
                }
                REQUIRE( std::sqrt( surfaceVariance ) == Approx( std::hypot( c[0], c[1] ) ).margin( 1e-15 ) );
            }
        }
}

/** \brief Projection retains supplied mode scaling, sign, order, and nonorthogonality rather than assuming rotation
 * only.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim TT projection handles nonorthogonal rescaled modes", "[accelSim][disturbance][projection]" )
{
    ttFixture  f( .63 );
    const auto original0 = f.m_mode0, original1 = f.m_mode1;
    for( size_t pixel = 0; pixel < 20; ++pixel )
    {
        f.m_mode0[pixel] = -2 * original1[pixel];
        f.m_mode1[pixel] = 3 * original0[pixel] + .3 * original1[pixel];
    }
    const tipTiltProjection reference( original0, original1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    const tipTiltProjection projection( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    sample                  value;
    value.m_pointing = { .002, -.001 };
    const pointingDisturbance pointing( value );
    const auto                original = reference.coefficients( pointing ), c = projection.coefficients( pointing );
    REQUIRE( c[1] == Approx( original[0] / 3 ) );
    REQUIRE( c[0] == Approx( ( .3 * c[1] - original[1] ) / 2 ) );
    for( size_t pixel = 0; pixel < 20; ++pixel )
        REQUIRE( c[0] * f.m_mode0[pixel] + c[1] * f.m_mode1[pixel] ==
                 Approx( pointing.surface( f.m_x[pixel], f.m_y[pixel] ) ).margin( 1e-15 ) );
}

/** \brief Pupil weights select the fit, full modes supply the output, and residuals expose departures from pure TT.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim TT projection uses pupil weights and reports residuals", "[accelSim][disturbance][projection]" )
{
    ttFixture               f( .42 );
    const tipTiltProjection reference( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    f.m_mode0[19] = 100;
    f.m_mode1[19] = -50;
    f.m_x[19] = f.m_y[19] = NAN; // Excluded coordinates are not needed for fitting.
    for( size_t pixel = 0; pixel < 16; ++pixel )
        f.m_weights[pixel] = ( pixel + 1 ) * ( pixel + 1 );
    const tipTiltProjection weighted( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    sample                  value;
    value.m_pointing = { .001, .002 };
    const pointingDisturbance pointing( value );
    const auto                c = weighted.coefficients( pointing ), expected = reference.coefficients( pointing );
    REQUIRE( c[0] == Approx( expected[0] ) );
    REQUIRE( c[1] == Approx( expected[1] ) );
    tipTiltBasis basis( 5, 4, weighted.response()[0], weighted.response()[1] );
    REQUIRE( basis.synthesize( pointing )[19] == Approx( 100 * c[0] - 50 * c[1] ) );

    f.m_mode0[0] += .1; // A non-tilt defect cannot be silently treated as an exact physical plane.
    REQUIRE_THROWS_AS( ( tipTiltProjection{ f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 } ),
                       std::invalid_argument );
    const tipTiltProjection approximate( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, .1 );
    REQUIRE( approximate.relativeResidual()[0] > 1e-6 );
    f.m_weights[0] = 0;
    const tipTiltProjection masked( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    REQUIRE( masked.coefficients( pointing )[0] == Approx( expected[0] ) );
}

/** \brief Nominal modal-coefficient scale follows the unchanged physical ensemble, with no record-dependent gain.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim nominal modal TT coefficient scale", "[accelSim][disturbance][projection]" )
{
    const ttFixture         f( .91 );
    const tipTiltProjection projection( f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, 1e-12 );
    parameters              config;
    model                   simulation;
    simulation.configure( config, referenceModes( config.m_modeSeed ) );
    double variance = 0;
    for( const auto &mode : simulation.m_modes )
        for( const auto &axis : mode.m_axes )
            variance += axis.m_rms * axis.m_rms;
    sample rms;
    rms.m_pointing = { std::sqrt( variance ), 0 };
    const pointingDisturbance pointing( rms );
    const auto                c = projection.coefficients( pointing );
    REQUIRE( std::hypot( c[0], c[1] ) == Approx( .0115 ).margin( .0005 ) );
    REQUIRE( 4 * std::hypot( c[0], c[1] ) == Approx( .046 ).margin( .002 ) );
    REQUIRE( 4 * std::hypot( c[0], c[1] ) == Approx( pointing.surfaceP2V() ) );
    rms.m_pointing     = { .0029, 0 };
    const auto nominal = projection.coefficients( pointingDisturbance( rms ) );
    REQUIRE( std::hypot( nominal[0], nominal[1] ) == Approx( 6.5 * .0029 * arcsecToRad * 1e6 / 8 ) );
}

/** \brief The nominal 48-pitch pupil maps column/row pointing to reflective surface without an offload calibration.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim nominal DM pupil coordinates and sampled optical scale", "[accelSim][disturbance][projection]" )
{
    // Synthetic planes keep this regression independent of local/deployed FITS files.
    constexpr size_t    width = 50, height = 50;
    constexpr double    center = 24.5, diameter = 6.5, pupilPitches = 48;
    const double        pitch = diameter / pupilPitches;
    std::vector<double> x( width * height ), y( x.size() ), weights( x.size() );
    for( size_t row = 0; row < height; ++row )
        for( size_t column = 0; column < width; ++column )
        {
            const size_t pixel = column + width * row;
            x[pixel]           = ( double( column ) - center ) * pitch;
            y[pixel]           = ( double( row ) - center ) * pitch;
            weights[pixel]     = std::hypot( x[pixel], y[pixel] ) <= diameter / 2 ? 1 : 0;
        }
    REQUIRE( pitch == Approx( .13541666666666667 ) );
    REQUIRE( x[24 + width * 24] == Approx( -pitch / 2 ) );
    REQUIRE( y[24 + width * 24] == Approx( -pitch / 2 ) );
    REQUIRE( x[25 + width * 24] == Approx( pitch / 2 ) );
    REQUIRE( y[24 + width * 25] == Approx( pitch / 2 ) );

    for( double angle : { 0.0, .37, 2.4 } )
        for( int convention = 0; convention < 3; ++convention )
        {
            std::vector<double> mode0( x.size() ), mode1( x.size() );
            for( size_t pixel = 0; pixel < x.size(); ++pixel )
            {
                mode0[pixel] = 4 / diameter * ( std::cos( angle ) * x[pixel] + std::sin( angle ) * y[pixel] );
                mode1[pixel] = 4 / diameter * ( -std::sin( angle ) * x[pixel] + std::cos( angle ) * y[pixel] );
                if( convention == 1 )
                    std::swap( mode0[pixel], mode1[pixel] );
                if( convention == 2 )
                    mode0[pixel] = -mode0[pixel];
            }
            const tipTiltProjection projection( mode0, mode1, x, y, weights, 1e-12 );
            tipTiltBasis            basis( width, height, projection.response()[0], projection.response()[1] );
            for( const auto mas :
                 { std::array<double, 2>{ 1, 0 }, { 0, 1 }, { 2.9, 0 }, { 0, 2.9 }, { -1, 0 }, { 0, -1 }, { 1, -2 } } )
            {
                CAPTURE( angle, convention, mas );
                sample value;
                value.m_pointing = { mas[0] / 1000, mas[1] / 1000 };
                const pointingDisturbance pointing( value );
                const double              tx = mas[0] * pi / 648000000, ty = mas[1] * pi / 648000000;
                REQUIRE( pointing.radians()[0] == Approx( tx ).margin( 1e-24 ) );
                REQUIRE( pointing.radians()[1] == Approx( ty ).margin( 1e-24 ) );
                std::array<double, 2> expected{
                    diameter * 1e6 / 8 * ( std::cos( angle ) * tx + std::sin( angle ) * ty ),
                    diameter * 1e6 / 8 * ( -std::sin( angle ) * tx + std::cos( angle ) * ty ) };
                if( convention == 1 )
                    std::swap( expected[0], expected[1] );
                if( convention == 2 )
                    expected[0] = -expected[0];
                const auto coefficients = projection.coefficients( pointing );
                REQUIRE( coefficients[0] == Approx( expected[0] ).margin( 1e-15 ) );
                REQUIRE( coefficients[1] == Approx( expected[1] ).margin( 1e-15 ) );
                const auto &command = basis.synthesize( pointing );
                double      minimum = INFINITY, maximum = -INFINITY, maxError = 0;
                for( size_t pixel = 0; pixel < x.size(); ++pixel )
                {
                    const double surface = ( tx * x[pixel] + ty * y[pixel] ) * 1e6 / 2;
                    maxError             = std::max( maxError, std::abs( command[pixel] - surface ) );
                    if( weights[pixel] )
                    {
                        minimum = std::min( minimum, double( command[pixel] ) );
                        maximum = std::max( maximum, double( command[pixel] ) );
                    }
                }
                REQUIRE( maxError < 2e-9 );
                REQUIRE( command[25 + width * 24] - command[24 + width * 24] ==
                         Approx( tx * pitch * 1e6 / 2 ).margin( 1e-9 ) );
                REQUIRE( command[24 + width * 25] - command[24 + width * 24] ==
                         Approx( ty * pitch * 1e6 / 2 ).margin( 1e-9 ) );
                if( mas[0] == 0 || mas[1] == 0 )
                    REQUIRE( maximum - minimum == Approx( pointing.surfaceP2V() * 47 / 48 ).margin( 2e-9 ) );
                if( std::hypot( mas[0], mas[1] ) == 2.9 )
                    REQUIRE( pointing.surfaceP2V() == Approx( .04569368944457377 ).margin( 1e-15 ) );
            }
        }
}

/** \brief Missing/degenerate calibration geometry, invalid weights, and poorly conditioned fits are rejected.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim TT projection rejects invalid fits", "[accelSim][disturbance][projection]" )
{
    const ttFixture original( .29 );
    for( int fault = 0; fault < 14; ++fault )
    {
        CAPTURE( fault );
        auto   f         = original;
        double tolerance = 1e-12;
        switch( fault )
        {
        case 0:
            f.m_mode0.clear();
            break;
        case 1:
            f.m_mode1.pop_back();
            break;
        case 2:
            f.m_x.pop_back();
            break;
        case 3:
            f.m_y.pop_back();
            break;
        case 4:
            f.m_weights.pop_back();
            break;
        case 5:
            f.m_weights.assign( 20, 0 );
            break;
        case 6:
            f.m_weights[0] = -1;
            break;
        case 7:
            f.m_weights[0] = NAN;
            break;
        case 8:
            f.m_mode0[19] = INFINITY;
            break;
        case 9:
            f.m_x[0] = NAN;
            break;
        case 10:
            f.m_mode1 = f.m_mode0;
            break;
        case 11:
            f.m_y = f.m_x;
            break;
        case 12:
            tolerance = NAN;
            break;
        case 13:
            tolerance = 1;
            break;
        }
        REQUIRE_THROWS_AS( ( tipTiltProjection{ f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, tolerance } ),
                           std::invalid_argument );
    }
    auto f = original;
    for( size_t pixel = 0; pixel < 20; ++pixel )
        f.m_mode1[pixel] = f.m_mode0[pixel] + 1e-12 * f.m_mode1[pixel];
    REQUIRE_THROWS_AS( ( tipTiltProjection{ f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, .1 } ),
                       std::invalid_argument );
    f        = original;
    f.m_x[0] = std::numeric_limits<double>::max();
    REQUIRE_THROWS_AS( ( tipTiltProjection{ f.m_mode0, f.m_mode1, f.m_x, f.m_y, f.m_weights, .1 } ),
                       std::overflow_error );
    const tipTiltProjection projection(
        original.m_mode0, original.m_mode1, original.m_x, original.m_y, original.m_weights, 1e-12 );
    sample huge;
    huge.m_pointing = { std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };
    REQUIRE_THROWS_AS( projection.coefficients( pointingDisturbance( huge ) ), std::overflow_error );
}

} // namespace accelSimTest
} // namespace libXWCTest
