/** \file accelSimTimeline_test.cpp
 * \brief Shared timestamped realization: legacy equivalence, physics, interpolation, bounds and thread safety.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimTimeline.hpp"

#include <complex>
#include <random>
#include <thread>

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app::accelSimDetail;

/// \cond ACCELSIM_TIMELINE_FIXTURE
namespace
{
constexpr int64_t c_epoch  = 1000000000; ///< Arbitrary monotonic grid origin [ns].
constexpr int64_t c_period = 125000;     ///< 8 kHz node spacing [ns].

/// Configure a generator at 8 kHz from the reference population or explicit modes.
model makeModel( double                       noisePSD = 1e-8 /**< [in] sensor noise PSD */,
                 const std::vector<modeSpec> &specs    = referenceModes( 123 ) /**< [in] vibration modes */ )
{
    parameters config;
    config.m_fps      = 8000;
    config.m_noisePSD = noisePSD;
    model generator;
    generator.configure( config, specs );
    return generator;
}

/// Evaluate the documented cubic reconstruction from a complete node history, independently of the ring.
sample reference( const std::vector<sample> &nodes /**< [in] model::step() history from node zero */,
                  int64_t                    time /**< [in] monotonic evaluation time */ )
{
    const double x = double( time - c_epoch ) / c_period;
    const auto   k = size_t( std::floor( x ) );
    const auto   w = cubicWeights( x - double( k ) );
    sample       value{};
    for( size_t j = 0; j < 4; ++j )
        for( size_t axis = 0; axis < 2; ++axis )
        {
            value.m_pointing[axis] += w[j] * nodes.at( k - 1 + j ).m_pointing[axis];
            value.m_acceleration[axis] +=
                w[j] * ( nodes.at( k - 1 + j ).m_acceleration[axis] - nodes.at( k - 1 + j ).m_noise[axis] );
        }
    return value;
}

/// Generate the first count nodes of a fresh generator, matching a timeline anchored at node zero.
std::vector<sample> history( model generator /**< [in] configured generator copy */, size_t count /**< [in] nodes */ )
{
    std::vector<sample> nodes;
    for( size_t n = 0; n < count; ++n )
        nodes.push_back( generator.step() );
    return nodes;
}
} // namespace
/// \endcond

/** \brief Cubic weights are exact at nodes and preserve narrowband amplitude and phase across the mode population.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline cubic reconstruction preserves amplitude and phase", "[accelSim][timeline]" )
{
    REQUIRE( cubicWeights( 0 ) == std::array<double, 4>{ 0, 1, 0, 0 } );
    // Bounds are the measured worst case over mu, with margin; linear interpolation is 6.2% at 900 Hz.
    const std::array<std::array<double, 3>, 4> limits{
        { { 28, 1e-8, 1e-10 }, { 120, 3e-6, 5e-8 }, { 300, 1e-4, 5e-6 }, { 900, 6e-3, 7e-4 } } };
    for( const auto &[frequency, amplitude, phase] : limits )
    {
        const double omega = 2 * pi * frequency / 8000;
        for( int n = 0; n < 1000; ++n )
        {
            const double mu = n / 1000.;
            const auto   w  = cubicWeights( mu );
            REQUIRE( w[0] + w[1] + w[2] + w[3] == Approx( 1 ).margin( 1e-15 ) );
            std::complex<double> response = 0;
            for( int j = 0; j < 4; ++j )
                response += w[j] * std::polar( 1.0, omega * ( j - 1 - mu ) );
            REQUIRE( std::abs( std::abs( response ) - 1 ) <= amplitude );
            REQUIRE( std::abs( std::arg( response ) ) <= phase );
        }
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::cubicWeights;
#endif
    // clang-format on
}

/** \brief Node-aligned measurements reproduce the legacy one-step-per-tick model, including ADC counts and noise.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline reproduces the legacy model at node times", "[accelSim][timeline]" )
{
    model                   generator = makeModel(), legacy = makeModel();
    timeline                shared;
    std::array<uint64_t, 2> low{}, high{};
    shared.reset( c_epoch, generator );
    legacy.step(); // Node zero; the first acquisition is tick one.
    for( uint64_t tick = 1; tick <= 8000; ++tick )
    {
        const int64_t time = c_epoch + int64_t( tick ) * c_period;
        REQUIRE_FALSE( shared.advance( generator, time ) );
        sample value;
        REQUIRE( shared.measurement( time, tick, generator.m_config, value, low, high ) );
        const sample &expected = legacy.step();
        REQUIRE( value.m_pointing == expected.m_pointing );
        REQUIRE( value.m_wfsPointing == expected.m_wfsPointing );
        REQUIRE( value.m_noise == expected.m_noise );
        for( size_t axis = 0; axis < 2; ++axis )
            REQUIRE( value.m_acceleration[axis] == Approx( expected.m_acceleration[axis] ).epsilon( 1e-14 ) );
        REQUIRE( value.m_adc == expected.m_adc );
    }
    // The ring generated ahead of acquisition; only measured samples are counted as clamps.
    REQUIRE( generator.m_ticks > legacy.m_ticks );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::timeline::advance;
    MagAOX::app::accelSimDetail::timeline::measurement;
#endif
    // clang-format on
}

/** \brief Fixed seeds reproduce one realization, unchanged by acquisition jitter, gaps, or reader polling rate.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline realization is independent of acquisition timing and readers", "[accelSim][timeline]" )
{
    model    regularModel = makeModel(), jitteredModel = makeModel();
    timeline regular, jittered;
    regular.reset( c_epoch, regularModel );
    jittered.reset( c_epoch, jitteredModel );
    const auto                             nodes = history( makeModel(), 4000 );
    std::mt19937_64                        rng( 7 );
    std::uniform_int_distribution<int64_t> jitter( 0, 60000 ), probe( 0, 4000000 );
    std::bernoulli_distribution            stall( .01 ), poll( .5 );
    int64_t                                next = c_epoch + c_period;
    for( int64_t time = c_epoch + c_period; time < c_epoch + 3000 * c_period; time += c_period )
    {
        REQUIRE_FALSE( regular.advance( regularModel, time ) );
        if( time < next )
            continue;
        const int64_t now = time + jitter( rng ); // Late wake inside the tick, as in acquisition.
        REQUIRE_FALSE( jittered.advance( jitteredModel, now ) );
        next = time + ( stall( rng ) ? 24 * c_period : c_period ); // Occasional 3-ms gaps inside the horizon.
        for( int n = 0; n < 8; ++n )
        {
            const int64_t t = now + probe( rng );
            sample        a, b;
            REQUIRE( regular.physical( t, a ) );
            REQUIRE( jittered.physical( t, b ) );
            REQUIRE( a.m_pointing == b.m_pointing );
            REQUIRE( a.m_acceleration == b.m_acceleration );
            const sample expected = reference( nodes, t );
            for( size_t axis = 0; axis < 2; ++axis )
            {
                REQUIRE( a.m_pointing[axis] == Approx( expected.m_pointing[axis] ).epsilon( 1e-12 ) );
                REQUIRE( a.m_acceleration[axis] == Approx( expected.m_acceleration[axis] ).epsilon( 1e-12 ) );
            }
            if( poll( rng ) )
                break; // Readers at any rate draw no random numbers.
        }
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::timeline::physical;
#endif
    // clang-format on
}

/** \brief Sensor noise enters only accelerometer measurements, once per tick; physical pointing ignores noise PSD.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline separates sensor noise from physical disturbance", "[accelSim][timeline]" )
{
    model                   noisy = makeModel( 1e-8 ), quiet = makeModel( 0 );
    timeline                a, b;
    std::array<uint64_t, 2> low{}, high{};
    a.reset( c_epoch, noisy );
    b.reset( c_epoch, quiet );
    const auto            nodes = history( makeModel( 1e-8 ), 2100 );
    std::array<double, 2> previous{};
    REQUIRE_FALSE( a.advance( noisy, c_epoch + c_period ) ); // First acquisition at tick one anchors node zero.
    REQUIRE_FALSE( b.advance( quiet, c_epoch + c_period ) );
    for( uint64_t tick = 2; tick <= 2000; ++tick )
    {
        // Between nodes; odd ticks precede their own node, as a negative synchronized phase offset does.
        const int64_t time = c_epoch + int64_t( tick ) * c_period + ( tick % 2 ? -40000 : 37000 );
        REQUIRE_FALSE( a.advance( noisy, time ) );
        REQUIRE_FALSE( b.advance( quiet, time ) );
        sample physicalA, physicalB, measured;
        REQUIRE( a.physical( time, physicalA ) );
        REQUIRE( b.physical( time, physicalB ) );
        REQUIRE( physicalA.m_pointing == physicalB.m_pointing ); // DM input is unchanged by sensor noise.
        REQUIRE( physicalA.m_noise == std::array<double, 2>{} );
        REQUIRE( a.measurement( time, tick, noisy.m_config, measured, low, high ) );
        REQUIRE( measured.m_pointing == physicalA.m_pointing );
        REQUIRE( measured.m_noise == nodes[tick].m_noise ); // The consumed tick's draw, not the nearest node's.
        for( size_t axis = 0; axis < 2; ++axis )
        {
            REQUIRE( physicalA.m_acceleration[axis] == Approx( physicalB.m_acceleration[axis] ).margin( 1e-15 ) );
            REQUIRE( measured.m_acceleration[axis] == physicalA.m_acceleration[axis] + measured.m_noise[axis] );
            REQUIRE( measured.m_noise[axis] != previous[axis] ); // Each tick uses its own draw.
        }
        previous = measured.m_noise;
    }
}

/** \brief Every evaluated time preserves acceleration = -(2 pi f)^2 L theta, and a = L theta'' spectrally.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline preserves the pointing-to-acceleration relation", "[accelSim][timeline]" )
{
    for( const double frequency : { 28., 120. } )
    {
        model    generator = makeModel( 0, { { frequency, .002, .9999999, 1, 1 } } );
        timeline shared;
        shared.reset( c_epoch, generator );
        const double factor  = accelerationFactor( frequency, generator.m_config.m_leverArm );
        const double h       = c_period / 8 * 1e-9;
        double       crossed = 0, squared = 0;
        for( int64_t tick = 1; tick <= 16000; ++tick )
        {
            const int64_t node = c_epoch + tick * c_period;
            REQUIRE_FALSE( shared.advance( generator, node ) );
            std::array<sample, 3> v; // Mid-interval stencil inside one cubic piece.
            for( int j = 0; j < 3; ++j )
                REQUIRE( shared.physical( node + c_period / 2 + ( j - 1 ) * c_period / 8, v[j] ) );
            for( size_t axis = 0; axis < 2; ++axis )
            {
                // Exact modal relation, including sign and arcsec-to-radian units, at an off-node time.
                REQUIRE( v[1].m_acceleration[axis] == Approx( factor * v[1].m_pointing[axis] ).epsilon( 1e-12 ) );
                const double second =
                    ( v[0].m_pointing[axis] - 2 * v[1].m_pointing[axis] + v[2].m_pointing[axis] ) / ( h * h );
                crossed += generator.m_config.m_leverArm * second * arcsecToRad * v[1].m_acceleration[axis];
                squared += v[1].m_acceleration[axis] * v[1].m_acceleration[axis];
            }
        }
        // Regression slope of L theta'' on a: unity apart from cubic curvature error and AR(2) innovations.
        REQUIRE( crossed / squared == Approx( 1 ).margin( 1e-2 ) );
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::timeline::physical;
#endif
    // clang-format on
}

/** \brief Memory is fixed; edges, gaps within the horizon, long stalls and resets behave deterministically.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline bounds memory, catch-up and buffer edges", "[accelSim][timeline]" )
{
    static_assert( sizeof( timeline ) < 16384, "fixed storage independent of run time and polling" );
    model    generator = makeModel(), continuous = makeModel();
    timeline shared, reference;
    sample   value;
    REQUIRE_FALSE( shared.physical( c_epoch, value ) ); // Nothing exists before reset/advance.
    shared.reset( c_epoch, generator );
    reference.reset( c_epoch, continuous );
    REQUIRE_FALSE( shared.physical( c_epoch + c_period, value ) );
    int64_t now = c_epoch + c_period;
    REQUIRE_FALSE( shared.advance( generator, now ) ); // The first anchor is not a discontinuity.
    REQUIRE( shared.physical( now, value ) );
    REQUIRE( shared.physical( now + timeline::c_lookahead, value ) );
    REQUIRE_FALSE( shared.physical( now + timeline::c_lookahead + 2 * c_period, value ) );
    REQUIRE_FALSE( shared.physical( c_epoch + c_period / 2, value ) ); // Needs node -1.

    // A 4-ms gap stays inside the horizon: no restart, and values match continuous generation.
    for( int64_t t = c_epoch + c_period; t <= now + 32 * c_period; t += c_period )
        REQUIRE_FALSE( reference.advance( continuous, t ) );
    now += 32 * c_period;
    REQUIRE_FALSE( shared.advance( generator, now ) );
    sample expected;
    REQUIRE( reference.physical( now + 1000, expected ) );
    REQUIRE( shared.physical( now + 1000, value ) );
    REQUIRE( value.m_pointing == expected.m_pointing );

    // Long operation: storage remains bounded and old times leave the ring.
    const int64_t old = now;
    for( int n = 0; n < 100000; ++n )
    {
        now += c_period;
        REQUIRE_FALSE( shared.advance( generator, now ) );
    }
    REQUIRE_FALSE( shared.physical( old, value ) );
    REQUIRE( shared.physical( now - 8 * c_period, value ) );

    // A stall beyond the horizon and the legacy catch-up bound restarts once, then measures normally.
    now += timeline::c_lookahead + 2 * int64_t( maxCatchUp ) * c_period;
    REQUIRE( shared.advance( generator, now ) );
    std::array<uint64_t, 2> low{}, high{};
    const auto              tick = uint64_t( ( now - c_epoch ) / c_period );
    REQUIRE( shared.measurement( now, tick, generator.m_config, value, low, high ) );
    REQUIRE_FALSE( shared.measurement( now, tick + 1000, generator.m_config, value, low, high ) );

    // Reset (rate change or restart) empties the ring; the next acquisition anchors without a discontinuity.
    shared.reset( now, generator );
    REQUIRE_FALSE( shared.physical( now + c_period, value ) );
    REQUIRE_FALSE( shared.advance( generator, now + c_period ) );
    REQUIRE( shared.physical( now + 2 * c_period, value ) );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::timeline::reset;
    MagAOX::app::accelSimDetail::timeline::advance;
#endif
    // clang-format on
}

/** \brief Concurrent readers see only values of the deterministic realization while acquisition generates.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timeline concurrent readers observe the deterministic realization", "[accelSim][timeline]" )
{
    constexpr int                           ticks     = 20000;
    model                                   generator = makeModel();
    timeline                                shared;
    const auto                              nodes = history( makeModel(), ticks + 64 );
    std::atomic<int64_t>                    latest{ 0 };
    std::vector<std::pair<int64_t, sample>> observed;
    shared.reset( c_epoch, generator );
    std::thread reader(
        [&]
        {
            std::mt19937_64                        rng( 11 );
            std::uniform_int_distribution<int64_t> offset( -200000, 4000000 );
            while( latest.load() < c_epoch + ticks * c_period )
            {
                const int64_t t = latest.load() + offset( rng );
                sample        value;
                if( latest.load() > 0 && shared.physical( t, value ) )
                    observed.emplace_back( t, value );
            }
        } );
    for( int64_t tick = 1; tick <= ticks; ++tick )
    {
        const int64_t now = c_epoch + tick * c_period;
        REQUIRE_FALSE( shared.advance( generator, now ) );
        latest = now;
    }
    reader.join();
    REQUIRE( observed.size() > 1000 );
    for( const auto &[t, value] : observed )
    {
        const sample expected = reference( nodes, t );
        for( size_t axis = 0; axis < 2; ++axis )
            REQUIRE( value.m_pointing[axis] == Approx( expected.m_pointing[axis] ).epsilon( 1e-12 ) );
    }
}

} // namespace accelSimTest
} // namespace libXWCTest
