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
    // These are arbitrary fixture command units per radian, not an instrument calibration.
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
    tipTiltBasis basis( 2, 1, { 1e6, -1e6 }, { -2e6, 2e6 } );
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

} // namespace accelSimTest
} // namespace libXWCTest
