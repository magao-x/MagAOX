/** \file accelSim_test.cpp
 * \brief Numerical, control, pacing, and publication tests for accelSim.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimModel.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <type_traits>

#ifndef ACCELSIM_MODEL_ONLY
    #include "../accelSim.hpp"
    #include <filesystem>
    #include <thread>
    #include <unistd.h>

// The full app must preserve its virtual base's destructor contract, including all helper/member destructors.
static_assert( std::is_nothrow_destructible_v<MagAOX::app::accelSim> );
#endif

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app::accelSimDetail;

/** \brief Known AR coefficients, stable innovation scaling, and stationary covariance invariance.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim AR coefficients and innovation variance", "[accelSim][model]" )
{
    oscillator state;
    state.configure( 1500, .9, 6000, 2 );
    REQUIRE( state.m_phi1 == Approx( 0 ).margin( 1e-14 ) );
    REQUIRE( state.m_phi2 == Approx( -.81 ) );
    REQUIRE( state.m_sigma * state.m_sigma == Approx( 4 * ( 1 - std::pow( .9, 4 ) ) ) );
    state.configure( 11, 0, 6000, 2 );
    REQUIRE( state.m_sigma == 2 );
    REQUIRE( state.m_rho == 0 );
    REQUIRE( state.m_eta == 1 );
    for( double r : { .9, .99999, .9999999 } )
    {
        state.configure( 11, r, 6000, .3 );
        const long double a = state.m_phi1, b = state.m_phi2, variance = .3L * .3L;
        const long double covariance = a * variance / ( 1 - b );
        const long double innovation = variance * ( 1 + b ) * ( ( 1 - b ) * ( 1 - b ) - a * a ) / ( 1 - b );
        REQUIRE( state.m_sigma * state.m_sigma == Approx( double( innovation ) ).epsilon( 1e-8 ) );
        const long double propagated = ( a * a + b * b ) * variance + 2 * a * b * covariance + innovation;
        REQUIRE( double( propagated ) == Approx( double( variance ) ).epsilon( 1e-12 ) );
        REQUIRE( state.m_rho * state.m_rho + state.m_eta * state.m_eta == Approx( 1 ).epsilon( 1e-12 ) );
    }
}

/** \brief Ensemble stationary histories remain stationary after stepping, with independent axes.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim stationary initialization covariance and independence", "[accelSim][model]" )
{
    constexpr size_t                 count = 100000;
    std::mt19937_64                  rng( 42 );
    std::normal_distribution<double> normal;
    for( double r : { .9, .9999999 } )
    {
        oscillator x, y;
        x.configure( 11, r, 6000, 1 );
        y         = x;
        double xx = 0, yy = 0, lag = 0, cross = 0, after = 0, residual = 0, mean = 0;
        for( size_t n = 0; n < count; ++n )
        {
            const double z1 = normal( rng ), z2 = normal( rng ), z3 = normal( rng ), z4 = normal( rng );
            x.initialize( z1, z2 );
            y.initialize( z3, z4 );
            xx += x.m_previous1 * x.m_previous1;
            yy += x.m_previous2 * x.m_previous2;
            lag += x.m_previous1 * x.m_previous2;
            cross += x.m_previous1 * y.m_previous1;
            const double conditional = ( x.m_previous2 - x.m_rho * x.m_previous1 ) / x.m_eta;
            residual += conditional * conditional;
            mean += x.m_previous1;
            for( int k = 0; k < 4; ++k )
                x.step( normal( rng ) );
            after += x.m_previous1 * x.m_previous1;
        }
        const double varianceTolerance = 6 * std::sqrt( 2.0 / count );
        REQUIRE( xx / count == Approx( 1 ).margin( varianceTolerance ) );
        REQUIRE( yy / count == Approx( 1 ).margin( varianceTolerance ) );
        REQUIRE( after / count == Approx( 1 ).margin( varianceTolerance ) );
        REQUIRE( residual / count == Approx( 1 ).margin( varianceTolerance ) );
        REQUIRE( lag / count == Approx( x.m_rho ).margin( varianceTolerance ) );
        REQUIRE( std::abs( cross / count ) < 6 / std::sqrt( double( count ) ) );
        REQUIRE( std::abs( mean / count ) < 6 / std::sqrt( double( count ) ) );
    }
}

/** \brief Preserve every prominent row and the high-frequency population's envelope and damping.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim reference population and deterministic sequences", "[accelSim][model]" )
{
    const auto specs = referenceModes( 123 );
    REQUIRE( specs.size() == 214 );
    const std::array<double, 14> frequencies{ 11, 14, 17, 20, 25, 28, 54, 56, 58, 75, 85, 120, 220, 300 };
    const std::array<double, 14> amplitudes{
        .00025, .00025, .00025, .00025, .00025, .002, .002, .001, .002, .0008, .0008, .008, .015, .018 };
    const std::array<double, 14> radii{ .9999999,
                                        .9999999,
                                        .9999999,
                                        .9999999,
                                        .9999999,
                                        .99999,
                                        .99995,
                                        .99995,
                                        .99995,
                                        .9998,
                                        .9998,
                                        .998,
                                        .996,
                                        .995 };
    size_t                       weak     = 0;
    double                       previous = 0;
    for( const auto &spec : specs )
    {
        REQUIRE( spec.m_frequency >= previous );
        previous = spec.m_frequency;
        if( spec.m_couplingX == .01 )
        {
            ++weak;
            REQUIRE( spec.m_frequency >= 100 );
            REQUIRE( spec.m_frequency < 900 );
            REQUIRE( spec.m_r == Approx( .998 - ( spec.m_frequency - 100 ) / 800 * .006 ) );
            const double envelope = .5 + 2 * std::exp( -.5 * std::pow( ( spec.m_frequency - 250 ) / 250, 2 ) );
            REQUIRE( spec.m_accelRms / envelope >= .0005 );
            REQUIRE( spec.m_accelRms / envelope <= .002 );
        }
        else
        {
            const auto found = std::find( frequencies.begin(), frequencies.end(), spec.m_frequency );
            REQUIRE( found != frequencies.end() );
            const size_t i = found - frequencies.begin();
            REQUIRE( spec.m_accelRms == amplitudes[i] );
            REQUIRE( spec.m_r == radii[i] );
            REQUIRE( spec.m_couplingX == ( i < 5 ? .05 : 1 ) );
        }
        REQUIRE( spec.m_couplingX == spec.m_couplingY );
    }
    REQUIRE( weak == 200 );
    parameters config;
    model      a, b;
    a.configure( config, specs );
    b.configure( config, referenceModes( config.m_modeSeed ) );
    for( size_t n = 0; n < 1000; ++n )
        REQUIRE( a.step().m_acceleration == b.step().m_acceleration );
    a.configure( config, specs );
    b.configure( config, specs );
    REQUIRE( a.step().m_adc == b.step().m_adc );
    config.m_seed = 43;
    b.configure( config, specs );
    REQUIRE( a.step().m_acceleration != b.step().m_acceleration );
}

/** \brief Modewise frequency-squared conversion, lever-arm scaling, and noise-free WFS sums.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim physical units and separate WFS coupling", "[accelSim][model]" )
{
    REQUIRE( accelerationFactor( 100, 10 ) == Approx( -4 * pi * pi * 10000 * 10 * pi / ( 180 * 3600 ) ) );
    parameters config;
    config.m_noisePSD = 0;
    model value;
    value.configure( config, { { 11, .002, .9999999, .1, .2 }, { 300, .018, .995, .7, .8 } } );
    const auto output = value.step();
    for( size_t axis = 0; axis < 2; ++axis )
    {
        const auto  &low = value.m_modes[0], &high = value.m_modes[1];
        const double x = low.m_axes[axis].m_previous1, y = high.m_axes[axis].m_previous1;
        REQUIRE( output.m_pointing[axis] == Approx( x + y ) );
        REQUIRE( output.m_acceleration[axis] == Approx( x * low.m_accelPerArcsec + y * high.m_accelPerArcsec ) );
        REQUIRE( output.m_wfsPointing[axis] == Approx( x * ( axis ? .2 : .1 ) + y * ( axis ? .8 : .7 ) ) );
        REQUIRE( low.m_axes[axis].m_rms * -low.m_accelPerArcsec == Approx( .002 ) );
    }
    model longer;
    config.m_leverArm = 20;
    longer.configure( config, { { 11, .002, .9999999, .1, .2 }, { 300, .018, .995, .7, .8 } } );
    REQUIRE( longer.step().m_acceleration[0] == Approx( output.m_acceleration[0] ) );
    REQUIRE( longer.m_sample.m_pointing[0] == Approx( output.m_pointing[0] / 2 ) );
}

/** \brief White-noise variance and independent axes match the configured one-sided PSD.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim noise PSD and sample-rate reinitialization", "[accelSim][model]" )
{
    parameters config;
    model      value;
    value.configure( config, {} );
    double           xx = 0, yy = 0, xy = 0;
    constexpr size_t count = 100000;
    for( size_t n = 0; n < count; ++n )
    {
        const auto &s = value.step();
        xx += s.m_acceleration[0] * s.m_acceleration[0];
        yy += s.m_acceleration[1] * s.m_acceleration[1];
        xy += s.m_acceleration[0] * s.m_acceleration[1];
    }
    const double variance = config.m_noisePSD * config.m_fps / 2;
    REQUIRE( xx / count / variance == Approx( 1 ).margin( 6 * std::sqrt( 2.0 / count ) ) );
    REQUIRE( yy / count / variance == Approx( 1 ).margin( 6 * std::sqrt( 2.0 / count ) ) );
    REQUIRE( std::abs( xy / count / variance ) < 6 / std::sqrt( double( count ) ) );
    value.setRate( 3000 );
    REQUIRE( value.m_noiseSigma * value.m_noiseSigma == Approx( config.m_noisePSD * 1500 ) );
    config.m_noisePSD = 0;
    value.configure( config, {} );
    REQUIRE( value.step().m_adc == std::array<uint16_t, 2>{ 2048, 2048 } );
    value.configure( config, { { 11, .002, .9999999, 1, 1 } } );
    const double old = value.m_modes[0].m_axes[0].m_previous1;
    value.setRate( 4000 );
    REQUIRE( value.m_modes[0].m_axes[0].m_previous1 != old );
    REQUIRE( value.m_modes[0].m_axes[0].m_phi1 == Approx( 2 * .9999999 * std::cos( 2 * pi * 11 / 4000 ) ) );
}

/** \brief Calibration, midpoint, sign, rounding, rail clipping, and integer storage.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim ADC conversion and clipping", "[accelSim][model]" )
{
    parameters config;
    uint64_t   low = 0, high = 0;
    static_assert( std::is_same_v<decltype( adcCounts( 0, config, low, high ) ), uint16_t> );
    REQUIRE( adcCounts( 0, config, low, high ) == 2048 );
    REQUIRE( adcCounts( config.m_voltsPerBit / config.m_sensitivity * 10, config, low, high ) == 2058 );
    REQUIRE( adcCounts( -config.m_voltsPerBit / config.m_sensitivity * 10, config, low, high ) == 2038 );
    config.m_voltsPerBit = config.m_sensitivity = 1;
    REQUIRE( adcCounts( .4999, config, low, high ) == 2048 );
    REQUIRE( adcCounts( .5, config, low, high ) == 2049 );
    REQUIRE( adcCounts( -.5, config, low, high ) == 2048 ); // Round the ADC value, not the acceleration.
    REQUIRE( adcCounts( -.5001, config, low, high ) == 2047 );
    REQUIRE( adcCounts( -2048, config, low, high ) == 0 );
    REQUIRE( adcCounts( 2047, config, low, high ) == 4095 );
    REQUIRE( low == 0 );
    REQUIRE( high == 0 );
    REQUIRE( adcCounts( -2048.5, config, low, high ) == 0 );
    REQUIRE( adcCounts( 2047.5, config, low, high ) == 4095 );
    REQUIRE( low == 1 );
    REQUIRE( high == 1 );
    REQUIRE( adcCounts( 1e200, config, low, high ) == 4095 );
    REQUIRE( adcCounts( -1e200, config, low, high ) == 0 );
    REQUIRE_THROWS( adcCounts( std::numeric_limits<double>::infinity(), config, low, high ) );
    REQUIRE_THROWS( adcCounts( std::numeric_limits<double>::quiet_NaN(), config, low, high ) );
    config.m_zero = 100;
    REQUIRE( adcCounts( 0, config, low, high ) == 100 );
    model value;
    config.m_zero     = 5000;
    config.m_noisePSD = 0;
    value.configure( config, {} );
    value.step();
    REQUIRE( value.m_clipHigh == std::array<uint64_t, 2>{ 1, 1 } );
}

/** \brief Invalid physical inputs and aliased rates fail before numerical sampling.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim rejects invalid configuration", "[accelSim][model]" )
{
    oscillator state;
    REQUIRE_THROWS( state.configure( 11, 1, 6000, 1 ) );
    REQUIRE_THROWS( state.configure( 11, -.1, 6000, 1 ) );
    REQUIRE_THROWS( state.configure( 3000, .9, 6000, 1 ) );
    REQUIRE_THROWS( state.configure( 0, .9, 6000, 1 ) );
    REQUIRE_THROWS( state.configure( 11, .9, 6000, -1 ) );
    parameters config;
    model      value;
    const auto specs = referenceModes( 123 );
    for( double invalid : { 0.0,
                            -1.0,
                            10001.0,
                            100.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN() } )
    {
        config.m_fps = invalid;
        REQUIRE_THROWS( value.configure( config, specs ) );
    }
    config               = {};
    config.m_voltsPerBit = 0;
    REQUIRE_THROWS( value.configure( config, specs ) );
    config            = {};
    config.m_noisePSD = -1;
    REQUIRE_THROWS( value.configure( config, specs ) );
    config            = {};
    config.m_leverArm = 0;
    REQUIRE_THROWS( value.configure( config, specs ) );
    config = {};
    value.configure( config, specs );
    REQUIRE_THROWS( value.setRate( 100 ) );
    REQUIRE( value.m_config.m_fps == 6000 );
}

/** \brief Fractional deadlines do not drift, catch-up is bounded, and ordinary gaps preserve model history.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim pacing and bounded catch-up", "[accelSim][model]" )
{
    schedule clock;
    clock.reset( 123456789, 6000 );
    REQUIRE( clock.deadline() == 123623456 );
    REQUIRE_THROWS( clock.consume( clock.deadline() - 1 ) );
    for( size_t n = 0; n < 6000; ++n )
        REQUIRE( clock.consume( clock.deadline() ) == 0 );
    REQUIRE( clock.deadline() == 1123623456 );
    clock.reset( 0, 6000 );
    REQUIRE( clock.consume( 1000000 ) == 5 );
    REQUIRE( clock.deadline() == 1166667 );
    parameters config;
    model      a, b;
    const auto modes = referenceModes( 123 );
    a.configure( config, modes );
    b.configure( config, modes );
    REQUIRE_FALSE( a.advance( maxCatchUp ) );
    for( uint64_t n = 0; n <= maxCatchUp; ++n )
        b.step();
    REQUIRE( a.m_sample.m_acceleration == b.m_sample.m_acceleration );
    const auto ticks = a.m_ticks;
    REQUIRE( a.advance( maxCatchUp + 1 ) );
    REQUIRE( a.m_ticks == ticks + 1 );
    REQUIRE( a.m_sample.m_acceleration != b.m_sample.m_acceleration );
    clock.reset( 5000000, 4000 );
    REQUIRE( clock.deadline() == 5250000 );
}

/** \brief A deterministic event loop preserves accelerometer cadence for integer and fractional WFS ratios.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim phase reference preserves requested sampling rate", "[accelSim][model][sync-regression]" )
{
    for( const auto rates :
         { std::array<double, 2>{ 8000, 2000 }, { 6000, 2000 }, { 5500, 2000 }, { 6000, 2300 }, { 6000, 9000 } } )
    {
        CAPTURE( rates[0], rates[1] );
        phaseClock clock;
        clock.reset( 0, rates[0] );
        uint64_t source = 0, frames = 0;
        int64_t  nextSource = 0, previous = 0;
        while( std::min( nextSource, clock.deadline() ) <= 1000000000 )
        {
            if( nextSource <= clock.deadline() )
            {
                const auto deadline = clock.deadline();
                REQUIRE( clock.reference( ++source, nextSource, .01, 340600, 0, rates[1] ) );
                REQUIRE( clock.deadline() == deadline );
                nextSource = std::llround( source * ( 1e9 / rates[1] ) );
            }
            else
            {
                const auto now = clock.deadline();
                REQUIRE( clock.consume( now, .05, clock.m_period ) == 0 );
                if( previous )
                    REQUIRE( now - previous >= .95 * clock.m_period - 1 );
                previous = now;
                ++frames;
            }
        }
        REQUIRE( double( frames ) == Approx( rates[0] ).margin( 1 ) );
        REQUIRE( clock.m_sourcePeriod == Approx( 1e9 / rates[1] ).margin( 1 ) );
        REQUIRE( std::abs( clock.m_applied ) <= .5 * clock.m_period );
        REQUIRE( std::abs( clock.m_applied - clock.m_target ) < 1 );
    }
    schedule free;
    free.reset( 0, 6000 );
    for( size_t n = 0; n < 6000; ++n )
        REQUIRE( free.consume( free.deadline() ) == 0 );
    REQUIRE( free.deadline() == 1000166667 );
}

/** \brief WFS phase is computed from configured latencies and corrections cannot move an already scheduled sample.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim phase corrections are bounded and do not burst", "[accelSim][model][sync-regression]" )
{
    phaseClock clock;
    clock.reset( 0, 8000 );
    REQUIRE( clock.reference( 1, 0, .01, 340600, 0, 0 ) );
    REQUIRE_FALSE( clock.m_valid );
    while( clock.deadline() < 500000 )
        clock.consume( clock.deadline(), .05, clock.m_period );
    const auto pending = clock.deadline();
    REQUIRE( clock.reference( 2, 500000, .01, 340600, 0, 0 ) );
    REQUIRE( clock.m_sourcePeriod == 500000 );
    REQUIRE( clock.m_modelDelay == Approx( 409400 ) );
    REQUIRE( clock.m_delay == Approx( 409400 ) );
    REQUIRE( clock.deadline() == pending );
    for( int n = 0; n < 20; ++n )
    {
        const auto now = clock.deadline();
        REQUIRE( clock.consume( now, .05, clock.m_period ) == 0 );
        REQUIRE( clock.deadline() - now >= 118750 );
        REQUIRE( clock.deadline() - now <= 131250 );
    }
    REQUIRE( clock.m_applied == Approx( clock.m_target ) );
    REQUIRE( std::remainder( clock.deadline() - 909400., 125000. ) == Approx( 0 ).margin( 1 ) );
    // A large reference jump and signed phase-offset change can only slew future intervals.
    const auto frozen = clock.deadline();
    REQUIRE( clock.reference( 3, 1100000, 1, 340600, -100000, 2000 ) );
    REQUIRE( clock.deadline() == frozen );
    for( int n = 0; n < 20; ++n )
    {
        const auto now = clock.deadline();
        REQUIRE( clock.consume( now, .05, clock.m_period ) == 0 );
        REQUIRE( clock.deadline() - now >= 118750 );
    }
    const auto counter = clock.m_next;
    REQUIRE_FALSE( clock.reference( 3, 1100001, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.m_next == counter );
}

/** \brief Source loss, duplicate notifications, and source-counter gaps never stop or advance the accelerometer clock.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim synchronization loss preserves holdover sampling", "[accelSim][model][sync-regression]" )
{
    phaseClock clock;
    clock.reset( 0, 6000 );
    REQUIRE( clock.reference( 10, 0, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.reference( 14, 2000000, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.m_delta == 4 );
    REQUIRE( clock.m_sourcePeriod == 500000 ); // Gaps count WFS cycles, not accel ticks.
    REQUIRE( clock.m_next == 1 );
    REQUIRE( clock.consume( 2000000, .05, clock.m_period ) == 11 );
    REQUIRE( clock.expired( 12000000, 10000000 ) );
    const auto pending = clock.deadline();
    clock.invalidate();
    REQUIRE_FALSE( clock.m_valid );
    REQUIRE( clock.deadline() == pending );
    const double phase = clock.m_applied;
    const auto   next  = clock.m_next;
    for( int n = 0; n < 6000; ++n )
        REQUIRE( clock.consume( clock.deadline(), .05, clock.m_period ) == 0 );
    REQUIRE( clock.m_next == next + 6000 );
    REQUIRE( clock.m_applied == phase );
    REQUIRE_FALSE( clock.reference( 14, 1002000000, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.reference( 15, 1002500000, 1, 340600, 0, 2000 ) );
    REQUIRE_FALSE( clock.m_valid );
    REQUIRE( clock.reference( 16, 1003000000, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.m_valid );
    REQUIRE( clock.reference( 1, 1003500000, 1, 340600, 0, 2000 ) );
    REQUIRE_FALSE( clock.m_valid ); // Replaced/reset source needs fresh timing; physical model is unaffected.
    REQUIRE( clock.m_delta == 0 );
}

/** \brief Jitter, persistent reference shifts, and a missing-WFS interval cannot change the mean sampling rate.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim noisy reference and reacquisition do not bias rate", "[accelSim][model][sync-regression]" )
{
    phaseClock clock;
    clock.reset( 0, 8000 );
    uint64_t frames = 0, counter = 1;
    int64_t  nextSource = 500000, previous = 0;
    bool     sawHoldover = false, sawRecovery = false;
    while( std::min( nextSource, clock.deadline() ) <= 2000000000 )
    {
        if( nextSource <= clock.deadline() )
        {
            const auto deadline = clock.deadline();
            if( nextSource < 1000000000 || nextSource > 1050000000 )
            {
                clock.reference( counter, nextSource, .01, 340600, 0, 2000 );
                if( sawHoldover && clock.m_valid )
                    sawRecovery = true;
            }
            REQUIRE( clock.deadline() == deadline );
            ++counter;
            const int64_t nominal = counter * 500000;
            const int64_t shift   = nominal >= 800000000 ? -50000 : nominal >= 500000000 ? 70000 : 0;
            nextSource            = nominal + shift + ( counter % 2 ? 10000 : -10000 );
        }
        else
        {
            const auto now = clock.deadline();
            if( clock.expired( now, 10000000 ) )
            {
                clock.invalidate();
                sawHoldover = true;
            }
            REQUIRE( clock.consume( now, .05, clock.m_period ) == 0 );
            if( previous )
                REQUIRE( now - previous >= 118750 );
            previous = now;
            ++frames;
        }
    }
    REQUIRE( double( frames ) == Approx( 16000 ).margin( 1 ) );
    REQUIRE( sawHoldover );
    REQUIRE( sawRecovery );
    REQUIRE( clock.m_valid );
}

/** \brief Canceled waits retain source accounting; missed sampling periods advance physics without replay.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim phase clock missed samples retain model time", "[accelSim][model][sync-regression]" )
{
    phaseClock clock;
    clock.reset( 1000000000, 6000 );
    REQUIRE( clock.reference( 10, 1000000000, 1, 340600, 0, 2000 ) );
    const auto pending = clock.deadline();
    REQUIRE( clock.reference( 14, 1002000000, 1, 340600, 0, 2000 ) );
    REQUIRE( clock.m_producer == 14 ); // Accounting commits at receipt, not after a cancellable phase wait.
    REQUIRE( clock.m_delta == 4 );
    REQUIRE( clock.deadline() == pending );
    REQUIRE( clock.m_next == 1 );
    REQUIRE_FALSE( clock.reference( 14, 1002000001, 1, 340600, 0, 2000 ) );
    clock.reset( 2000000000, 6000 );
    REQUIRE( clock.m_producer == 14 );
    parameters config;
    model      a, b;
    const auto modes = referenceModes( 123 );
    a.configure( config, modes );
    b.configure( config, modes );
    const auto missed = clock.consume( 2000666667, .05, clock.m_period );
    REQUIRE( missed == 3 );
    REQUIRE_FALSE( a.advance( missed ) );
    for( uint64_t n = 0; n <= missed; ++n )
        b.step();
    REQUIRE( a.m_sample.m_acceleration == b.m_sample.m_acceleration );
    const auto stalled = clock.deadline() + 100000000;
    const auto skipped = clock.consume( stalled, .05, clock.m_period );
    REQUIRE( skipped > maxCatchUp );
    REQUIRE( a.advance( skipped ) );
    REQUIRE( clock.deadline() - stalled >= clock.m_period * .5 - 1 );
}

/** \brief Outage invalidation removes all lock/cadence measurements while retaining historical accounting.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim synchronization diagnostics invalidate stale lock", "[accelSim][model][sync-regression]" )
{
    diagnostics diag;
    diag.m_timing.fill( 10 );
    diag.m_timing[lock]     = 1;
    diag.m_timing[modeCode] = 1;
    diag.m_cadenceRatio     = 1;
    diag.m_skipped          = 7;
    diag.m_discontinuities  = 3;
    diag.m_low              = { 2, 4 };
    diag.m_high             = { 5, 6 };
    const auto before       = diag;
    diag.invalidateSync( -5 );
    for( size_t index = 0; index < frameSequence; ++index )
        REQUIRE( diag.m_timing[index] == ( index == delayTarget ? -5 : 0 ) );
    REQUIRE( diag.m_timing[producerDelta] == 0 );
    REQUIRE( diag.m_cadenceRatio == 0 );
    for( size_t index = frameSequence; index < timingCount; ++index )
        if( index != producerDelta )
            REQUIRE( diag.m_timing[index] == before.m_timing[index] );
    REQUIRE( diag.m_skipped == before.m_skipped );
    REQUIRE( diag.m_discontinuities == before.m_discontinuities );
    REQUIRE( diag.m_low == before.m_low );
    REQUIRE( diag.m_high == before.m_high );
    const auto invalid = diag;
    diag.invalidateSync( -5 );
    REQUIRE( diag.m_timing == invalid.m_timing );
    REQUIRE( diag.m_skipped == invalid.m_skipped );
}

/** \brief Optional offline export from the production numerical core, never from the RT app loop.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim export offline validation samples", "[.][accelSim-export]" )
{
    const char *directory = std::getenv( "ACCELSIM_EXPORT_DIR" );
    REQUIRE( directory != nullptr );
    const std::string prefix = std::string( directory ) + "/";
    std::ofstream     modes( prefix + "modes.csv" ), samples( prefix + "samples.csv" );
    REQUIRE( modes.good() );
    REQUIRE( samples.good() );
    modes << std::setprecision( 17 ) << "frequency,accel_rms,r,coupling_x,coupling_y\n";
    parameters       config;
    constexpr double duration = 60;
    const auto       count    = static_cast<size_t>( std::llround( duration * config.m_fps ) );
    const auto       specs    = referenceModes( config.m_modeSeed );
    for( const auto &s : specs )
        modes << s.m_frequency << ',' << s.m_accelRms << ',' << s.m_r << ',' << s.m_couplingX << ',' << s.m_couplingY
              << '\n';
    model value;
    value.configure( config, specs );
    samples << std::setprecision( 17 ) << "time,pointing_x,pointing_y,wfs_x,wfs_y,accel_x,accel_y,adc_x,adc_y\n";
    for( size_t n = 0; n < count; ++n )
    {
        const auto &s = value.step();
        samples << n / config.m_fps;
        for( const auto &pair : { s.m_pointing, s.m_wfsPointing, s.m_acceleration } )
            for( double x : pair )
                samples << ',' << x;
        samples << ',' << s.m_adc[0] << ',' << s.m_adc[1] << '\n';
    }
    samples.flush();
    modes.flush();
    REQUIRE( samples.good() );
    REQUIRE( modes.good() );
    std::ofstream metadata( prefix + "metadata.json" );
    metadata << std::setprecision( 17 ) << "{\n  \"fps\": " << config.m_fps << ",\n  \"duration\": " << duration
             << ",\n  \"seed\": " << config.m_seed << ",\n  \"mode_seed\": " << config.m_modeSeed
             << ",\n  \"lever_arm\": " << config.m_leverArm << ",\n  \"zero\": " << config.m_zero
             << ",\n  \"sensitivity\": " << config.m_sensitivity << ",\n  \"volts_per_bit\": " << config.m_voltsPerBit
             << ",\n  \"noise_psd\": " << config.m_noisePSD << ",\n  \"clip_low_xy\": [" << value.m_clipLow[0] << ", "
             << value.m_clipLow[1] << "],\n  \"clip_high_xy\": [" << value.m_clipHigh[0] << ", " << value.m_clipHigh[1]
             << "]\n}\n";
    metadata.flush();
    REQUIRE( metadata.good() );
}

/** \brief Signed phase error uses the hardware positive half-period boundary.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim phase error positive boundary", "[accelSim][model][sync-regression]" )
{
    diagnostics value;
    auto       &t      = value.m_timing;
    t[semaphorePeriod] = 500;
    t[delayModel]      = 409.4;
    t[delayApplied]    = t[delayModel] - 250;
    value.updatePhaseError();
    REQUIRE( t[phaseError] == Approx( 250 ) );
    for( double delta : { -750., -250., 250., 750., -250.001, -249.999, 249.999, 250.001, 0. } )
    {
        t[delayModel]   = 0;
        t[delayApplied] = delta;
        value.updatePhaseError();
        REQUIRE( t[phaseError] > -250 );
        REQUIRE( t[phaseError] <= 250 );
        if( std::abs( delta ) == 250 || std::abs( delta ) == 750 )
            REQUIRE( t[phaseError] == 250 );
        else if( delta < -250 )
            REQUIRE( t[phaseError] == Approx( delta + 500 ) );
        else if( delta > 250 )
            REQUIRE( t[phaseError] == Approx( delta - 500 ) );
        else
            REQUIRE( t[phaseError] == Approx( delta ) );
    }
    t[semaphorePeriod] = 0;
    value.updatePhaseError();
    REQUIRE( t[phaseError] == 0 );
}

/** \brief Startup and reconnection need a measured period and then a second valid prediction for an interval.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim trigger history requires measured timing", "[accelSim][model][sync-regression]" )
{
    diagnostics value;
    int64_t     previous = 0;
    auto       &t        = value.m_timing;
    t[delayTarget]       = 25;
    for( int64_t now : { 1000000000LL, 1000500000LL } )
    {
        value.updateTrigger( now, previous );
        REQUIRE( previous == 0 );
        REQUIRE( t[triggerTime] == 0 );
        REQUIRE( t[triggerInterval] == 0 );
    }
    t[semaphorePeriod] = 500;
    t[delayTarget]     = 409.4;
    value.updateTrigger( 1001000000, previous );
    REQUIRE( previous == 1001409400 );
    REQUIRE( t[triggerTime] == Approx( 409.4 ) );
    REQUIRE( t[triggerInterval] == 0 );
    value.updateTrigger( 1001500000, previous );
    REQUIRE( t[triggerInterval] == 500 );
    value.invalidateSync( 25 );
    previous = 0; // Acquisition lifecycle also resets the app-owned prediction history.
    value.updateTrigger( 2000000000, previous );
    REQUIRE( previous == 0 );
    REQUIRE( t[triggerTime] == 0 );
    REQUIRE( t[triggerInterval] == 0 );
    t[semaphorePeriod] = 500;
    value.updateTrigger( 2000500000, previous );
    REQUIRE( t[triggerInterval] == 0 );
}

#ifndef ACCELSIM_MODEL_ONLY
/// \cond TEST_HARNESS
/// Record real INDI serialization without a server or an activated input thread.
class recordingDriver : public MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>
{
  public:
    mutable std::vector<pcf::IndiProperty> m_messages; ///< Serialized property snapshots sent by the app.
    /// Attach only to test-owned local descriptors.
    recordingDriver( MagAOX::app::MagAOXApp<true> *parent /**< [in] local test app */ );
    /// Release the unactivated input; the shared destructor releases output.
    ~recordingDriver();
    /// Parse each complete outgoing XML message into a retained snapshot.
    void sendXml( const std::string &xml /**< [in] serialized property */ ) const override;

  private:
    int m_inputFd{ -1 }; ///< Input descriptor identified by the private file inode.
};

recordingDriver::recordingDriver( MagAOX::app::MagAOXApp<true> *parent )
    : indiDriver( parent, "accelSim-test", "0", "1.7" )
{
    REQUIRE( good() );
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( stat( parent->driverInName().c_str(), &input ) == 0 );
    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        const int   descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};
        if( fstat( descriptor, &candidate ) == 0 && candidate.st_dev == input.st_dev &&
            candidate.st_ino == input.st_ino )
        {
            REQUIRE( m_inputFd == -1 );
            m_inputFd = descriptor;
        }
    }
    REQUIRE( m_inputFd >= 0 );
}

recordingDriver::~recordingDriver()
{
    if( m_inputFd >= 0 )
        close( m_inputFd );
    setInputFd( -1 );
}

void recordingDriver::sendXml( const std::string &xml ) const
{
    pcf::IndiXmlParser parser( "1.7" );
    std::string        error;
    parser.parseXml( xml.c_str(), xml.size(), error );
    REQUIRE( error.empty() );
    REQUIRE( parser.getState() == pcf::IndiXmlParser::CompleteState );
    const auto message = parser.createIndiMessage();
    REQUIRE( message.getType() == pcf::IndiMessage::SetProperty );
    m_messages.push_back( message.getProperty() );
}

/// Expose only the protected surfaces exercised by application tests.
class appTest : public MagAOX::app::accelSim
{
  public:
    using accelSim::config;
    using accelSim::invalidateSync;
    using accelSim::m_activeFps;
    using accelSim::m_alpha;
    using accelSim::m_channelsDevice;
    using accelSim::m_circBuffLength;
    using accelSim::m_configName;
    using accelSim::m_currImageTimestamp;
    using accelSim::m_dataType;
    using accelSim::m_diagnostics;
    using accelSim::m_fpsDevice;
    using accelSim::m_height;
    using accelSim::m_indiP_alpha;
    using accelSim::m_indiP_fps;
    using accelSim::m_indiP_fpsSource;
    using accelSim::m_indiP_numChannels;
    using accelSim::m_indiP_numChannelsSource;
    using accelSim::m_indiP_synchroDelay;
    using accelSim::m_indiP_timingDiag;
    using accelSim::m_maxFrequency;
    using accelSim::m_maxSlew;
    using accelSim::m_model;
    using accelSim::m_nextSyncProbe;
    using accelSim::m_parameters;
    using accelSim::m_phaseClock;
    using accelSim::m_postDelay;
    using accelSim::m_rateApplied;
    using accelSim::m_rateRequest;
    using accelSim::m_reconfig;
    using accelSim::m_referenceFps;
    using accelSim::m_requestedFps;
    using accelSim::m_shmimName;
    using accelSim::m_shutdown;
    using accelSim::m_snapshot;
    using accelSim::m_syncImage;
    using accelSim::m_syncIndex;
    using accelSim::m_syncName;
    using accelSim::m_syncTimeout;
    using accelSim::m_width;
    using accelSim::setupProperties;
    using accelSim::syncTiming;
    using accelSim::updateINDI;
    using accelSim::updateSyncDiagnostics;
    using accelSim::waitSync;

    /// Install a recording driver using a private input file and /dev/null for unused endpoints.
    recordingDriver &captureINDI();
    std::string      m_testInput; ///< Temporary input file removed after driver destruction.

    /// Run the actual frameGrabber worker without an INDI server in stream unit tests.
    void startWorker();
    /// Join the worker and release its source on test exit.
    ~appTest();
};

recordingDriver &appTest::captureINDI()
{
    char      path[] = "/tmp/accelSim-indi-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    m_testInput = m_driverInName = path;
    m_driverOutName = m_driverCtrlName = "/dev/null";
    auto *driver                       = new recordingDriver( this );
    m_indiDriver                       = driver;
    return *driver;
}

void appTest::startWorker()
{
    state( MagAOX::app::stateCodes::OPERATING );
    m_fgThreadInit = false;
    m_fgThread     = std::thread( [this] { fgThreadExec(); } );
}

appTest::~appTest()
{
    m_shutdown = true;
    if( m_fgThread.joinable() )
        m_fgThread.join();
    reconfig();
    delete m_indiDriver;
    m_indiDriver = nullptr;
    if( !m_testInput.empty() )
        std::filesystem::remove( m_testInput );
}

/// Own a real local ImageStreamIO mapping for integration checks.
struct streamFixture
{
    IMAGE    m_image{};                                         ///< Mapping owned by the fixture.
    bool     m_open{ false }, m_owner{ false };                 ///< Mapping lifetime and destruction responsibility.
    int      m_index{ -1 };                                     ///< Reader semaphore to release on close.
    uint64_t m_counter{ 0 }, m_plane{ 0 }, m_planeCounter{ 0 }; ///< Coherent metadata from the last read.
    timespec m_acquired{}, m_written{}, m_planeAcquired{}, m_planeWritten{}; ///< Coherent image/plane timestamps.

    /// Close or destroy the mapping, without throwing during test unwinding.
    ~streamFixture();
    /// Open the named producer, retrying briefly for worker startup.
    void open( const char *name /**< [in] stream name */ );
    /// Create a one-pixel trigger source.
    void create( const char *name /**< [in] source name */ );
    /// Wait for a complete frame and return a coherent latest-plane ADC pair.
    std::array<uint16_t, 2> read();
};

streamFixture::~streamFixture()
{
    if( !m_open )
        return;
    if( m_index >= 0 )
        m_image.semReadPID[m_index] = 0;
    if( m_owner )
        ImageStreamIO_destroyIm( &m_image );
    else
        ImageStreamIO_closeIm( &m_image );
}

void streamFixture::open( const char *name )
{
    for( int retry = 0; retry < 200 && !m_open; ++retry )
    {
        m_open = ImageStreamIO_openIm( &m_image, name ) == 0;
        if( !m_open )
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }
    REQUIRE( m_open );
    m_index = ImageStreamIO_getsemwaitindex( &m_image, 9 );
    REQUIRE( m_index >= 0 );
    ImageStreamIO_semflush( &m_image, m_index );
}

void streamFixture::create( const char *name )
{
    uint32_t sizes[]{ 1, 1, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image,
                                         name,
                                         3,
                                         sizes,
                                         _DATATYPE_UINT16,
                                         -1,
                                         1,
                                         IMAGE_NB_SEMAPHORE,
                                         0,
                                         CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                         0 ) == 0 );
    m_open = m_owner = true;
}

std::array<uint16_t, 2> streamFixture::read()
{
    for( int attempt = 0; attempt < 100; ++attempt )
    {
        timespec timeout;
        clock_gettime( CLOCK_REALTIME, &timeout );
        ++timeout.tv_sec;
        REQUIRE( sem_timedwait( m_image.semptr[m_index], &timeout ) == 0 );
        const auto counter = m_image.md->cnt0;
        const auto plane   = m_image.md->cnt1;
        if( m_image.md->write || plane >= m_image.md->size[2] )
            continue;
        std::array<uint16_t, 2> pair{ m_image.array.UI16[2 * plane], m_image.array.UI16[2 * plane + 1] };
        m_counter       = counter;
        m_plane         = plane;
        m_planeCounter  = m_image.cntarray[plane];
        m_acquired      = m_image.md->atime;
        m_written       = m_image.md->writetime;
        m_planeAcquired = m_image.atimearray[plane];
        m_planeWritten  = m_image.writetimearray[plane];
        std::atomic_thread_fence( std::memory_order_acquire );
        if( !m_image.md->write && counter == m_image.md->cnt0 )
            return pair;
    }
    FAIL( "could not obtain a coherent frame" );
    return {};
}

/// Construct a compatible local property request.
pcf::IndiProperty update( const char *name /**< [in] property name */, double target /**< [in] requested value */ )
{
    pcf::IndiProperty result( pcf::IndiProperty::Number );
    result.setName( name );
    result.add( pcf::IndiElement( "target" ) );
    result["target"] = target;
    return result;
}
/// \endcond

/** \brief Defaults and INDI callbacks retain the approved accelerometer control contract.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim configuration and INDI controls", "[accelSim][app]" )
{
    appTest app;
    app.m_configName = "accelSim-test";
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_shmimName == "acc000" );
    REQUIRE( app.fps() == 6000 );
    REQUIRE( app.m_model.m_modes.size() == 214 );
    REQUIRE( app.setupProperties() == 0 );
    REQUIRE( app.m_indiP_fps.getPerm() == pcf::IndiProperty::ReadWrite );
    REQUIRE( app.m_indiP_fps["current"].get<double>() == 6000 );
    REQUIRE( app.m_indiP_timingDiag.find( "mode_code" ) );
    REQUIRE( app.m_indiP_timingDiag.find( "sync_producer_frame_delta" ) );
    REQUIRE( app.m_indiP_timingDiag.getNumElements() == 23 );
    REQUIRE( app.m_indiP_fps.getType() == pcf::IndiProperty::Number );
    REQUIRE( std::stod( app.m_indiP_fps["target"].getMin() ) == 0 );
    REQUIRE( std::stod( app.m_indiP_fps["target"].getMax() ) == 10000 );
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 0 ) ) < 0 );
    REQUIRE( app.m_requestedFps == 6000 );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Alert );
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 4000 ) ) == 0 );
    REQUIRE( app.m_requestedFps == 4000 );
    REQUIRE( app.m_activeFps == 6000 );
    REQUIRE( app.newCallBack_m_indiP_numChannels( update( "numChannels", 3 ) ) < 0 );
    REQUIRE( app.m_indiP_numChannels["current"].get<int>() == 2 );
    REQUIRE( app.newCallBack_m_indiP_numChannels( update( "numChannels", 2 ) ) == 0 );
    REQUIRE( app.newCallBack_m_indiP_alpha( update( "alpha", 2 ) ) == 0 );
    REQUIRE( app.m_alpha == 1 );
    REQUIRE( app.newCallBack_m_indiP_synchroDelay( update( "synchroDelay", -5 ) ) == 0 );
    REQUIRE( app.m_postDelay == -5 );
    app.m_fpsDevice = "source";
    app.m_indiP_fpsSource.setName( "fps" );
    app.m_indiP_fpsSource.setDevice( "source" );
    auto external = update( "fps", 5000 );
    external.setDevice( "source" );
    external.add( pcf::IndiElement( "current" ) );
    external["current"] = 5000;
    REQUIRE( app.setCallBack_m_indiP_fpsSource( external ) == 0 );
    REQUIRE( app.m_requestedFps == 5000 );
    app.m_indiP_numChannelsSource.setName( "numChannels" );
    app.m_indiP_numChannelsSource.setDevice( "source" );
    external.setName( "numChannels" );
    external["current"] = 3;
    REQUIRE( app.setCallBack_m_indiP_numChannelsSource( external ) < 0 );
    external["current"] = 2;
    REQUIRE( app.setCallBack_m_indiP_numChannelsSource( external ) == 0 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadConfigImpl;
    MagAOX::app::accelSim::setupProperties;
    MagAOX::app::accelSim::newCallBack_m_indiP_fps;
    MagAOX::app::accelSim::newCallBack_m_indiP_numChannels;
    MagAOX::app::accelSim::newCallBack_m_indiP_alpha;
    MagAOX::app::accelSim::newCallBack_m_indiP_synchroDelay;
    MagAOX::app::accelSim::setCallBack_m_indiP_fpsSource;
    MagAOX::app::accelSim::setCallBack_m_indiP_numChannelsSource;
#endif
    // clang-format on
}

/** \brief Source loss sends one coherent timingDiag message, never a cleared latency with an old lock.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim timing diagnostics serialize one snapshot", "[accelSim][app][indi-regression]" )
{
    appTest app;
    app.m_configName = "accelSim-test";
    REQUIRE( app.setupProperties() == 0 );
    auto &driver = app.captureINDI();
    app.m_diagnostics.m_timing.fill( 100 );
    app.m_diagnostics.m_timing[lock] = 1;
    app.updateINDI( app.m_diagnostics );
    driver.m_messages.clear();
    app.invalidateSync();
    app.updateINDI( app.m_snapshot );
    size_t timingMessages = 0;
    for( const auto &message : driver.m_messages )
    {
        if( message.getName() != "timingDiag" )
            continue;
        ++timingMessages;
        REQUIRE( message.getNumElements() == 23 );
        REQUIRE( message["avg_read_latency_us"].get<double>() == 0 );
        REQUIRE( message["delay_lock"].get<double>() == 0 );
        REQUIRE( message["avg_semaphore_period_us"].get<double>() == 0 );
        REQUIRE( message["local_frame_seq"].get<double>() == 100 );
    }
    REQUIRE( timingMessages == 1 );
    driver.m_messages.clear();
    app.updateINDI( app.m_snapshot );
    REQUIRE( driver.m_messages.empty() );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::updateINDI;
    MagAOX::app::accelSim::invalidateSync;
#endif
    // clang-format on
}

/** \brief Local/external requests stay Busy until the worker acknowledges their revision, including supersession.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim FPS acknowledgment follows worker application", "[accelSim][app][indi-regression]" )
{
    appTest app;
    app.m_configName = "accelSim-test";
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.setupProperties() == 0 );
    auto &driver = app.captureINDI();
    REQUIRE( app.startAcquisition() == 0 );
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 4000 ) ) == 0 );
    REQUIRE( driver.m_messages.back().getState() == pcf::IndiProperty::Busy );
    REQUIRE( driver.m_messages.back()["current"].get<double>() == 6000 );
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Busy );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Ok );
    REQUIRE( app.m_indiP_fps["current"].get<double>() == 4000 );
    const auto discontinuities = app.m_diagnostics.m_discontinuities;
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 5000 ) ) == 0 );
    const auto olderRevision = app.m_rateRequest.load();
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 4000 ) ) == 0 );
    app.m_rateApplied = olderRevision; // Older in-flight completion cannot acknowledge the newer request.
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Busy );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Ok );
    REQUIRE( app.m_diagnostics.m_discontinuities == discontinuities ); // No-op request does not restart physics.
    REQUIRE( app.newCallBack_m_indiP_fps( update( "fps", 0 ) ) < 0 );
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Alert );
    REQUIRE( app.m_requestedFps == 4000 );
    app.m_indiP_fpsSource.setName( "fps" );
    app.m_indiP_fpsSource.setDevice( "source" );
    auto external = update( "fps", 5000 );
    external.setDevice( "source" );
    external.add( pcf::IndiElement( "current", "5000" ) );
    REQUIRE( app.setCallBack_m_indiP_fpsSource( external ) == 0 );
    REQUIRE( driver.m_messages.back().getState() == pcf::IndiProperty::Busy );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    app.updateINDI( app.m_snapshot );
    REQUIRE( app.m_indiP_fps.getState() == pcf::IndiProperty::Ok );
    REQUIRE( app.m_indiP_fps["current"].get<double>() == 5000 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::newCallBack_m_indiP_fps;
    MagAOX::app::accelSim::setCallBack_m_indiP_fpsSource;
    MagAOX::app::accelSim::acquireAndCheckValid;
    MagAOX::app::accelSim::updateINDI;
#endif
    // clang-format on
}

/** \brief Parse the shipped example and reject an incompatible configured channel count.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim example configuration", "[accelSim][app]" )
{
    appTest app;
    app.setupConfig();
    const std::string filename = std::filesystem::exists( "apps/accelSim/config/example.conf" )
                                     ? "apps/accelSim/config/example.conf"
                                     : "../apps/accelSim/config/example.conf";
    REQUIRE( std::filesystem::exists( filename ) );
    app.config.readConfig( filename );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.m_parameters.m_zero == 2048 );
    REQUIRE( app.m_parameters.m_sensitivity == 1.02 );
    REQUIRE( app.m_parameters.m_voltsPerBit == .0083799 );
    REQUIRE( app.m_parameters.m_leverArm == 10 );
    REQUIRE( app.m_parameters.m_noisePSD == 1e-8 );
    REQUIRE( app.m_parameters.m_seed == 42 );
    REQUIRE( app.m_parameters.m_modeSeed == 123 );
    REQUIRE( app.m_syncTimeout == .01 );
    REQUIRE( app.m_maxSlew == .05 );
    REQUIRE( app.m_referenceFps == 0 );
    char      overrideFile[] = "/tmp/accelSim-config-XXXXXX";
    const int fd             = mkstemp( overrideFile );
    REQUIRE( fd >= 0 );
    close( fd );
    {
        std::ofstream file( overrideFile );
        file << "[sim]\nfps=4000\nseed=8\nleverArm=20\n[adc]\nzero=2000\n"
                "voltsPerBit=0.01\nsensitivity=2\n[sensor]\nnoisePSD=0\n";
    }
    app.config.readConfig( overrideFile );
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.fps() == 4000 );
    REQUIRE( app.m_parameters.m_seed == 8 );
    REQUIRE( app.m_parameters.m_leverArm == 20 );
    REQUIRE( app.m_parameters.m_zero == 2000 );
    REQUIRE( app.m_model.m_noiseSigma == 0 );
    {
        std::ofstream file( overrideFile );
        file << "[accel]\nnumChannels=3\n";
    }
    app.config.readConfig( overrideFile );
    REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    std::filesystem::remove( overrideFile );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadConfigImpl;
#endif
    // clang-format on
}

/** \brief Invalid phase-clock controls fail configuration before sampling begins.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim rejects invalid phase-clock controls", "[accelSim][app][sync-regression]" )
{
    char      path[] = "/tmp/accelSim-phase-config-XXXXXX";
    const int fd     = mkstemp( path );
    REQUIRE( fd >= 0 );
    close( fd );
    for( const std::string setting :
         { "timeout=0", "timeout=3601", "maxSlew=0", "maxSlew=0.26", "referenceFps=-1", "referenceFps=1e-300" } )
    {
        CAPTURE( setting );
        std::ofstream( path ) << "[synchro]\n" << setting << '\n';
        appTest app;
        app.setupConfig();
        app.config.readConfig( path );
        REQUIRE( app.loadConfigImpl( app.config ) < 0 );
    }
    std::filesystem::remove( path );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadConfigImpl;
#endif
    // clang-format on
}

/** \brief Actual acquisition stamps and two-value copies exclude shutdown replay.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim acquisition produces fresh timestamped ADC pairs", "[accelSim][app]" )
{
    appTest app;
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.configureAcquisition() == 0 );
    REQUIRE( app.m_width == 2 );
    REQUIRE( app.m_height == 1 );
    REQUIRE( app.m_dataType == _DATATYPE_UINT16 );
    REQUIRE( app.startAcquisition() == 0 );
    REQUIRE( app.acquireAndCheckValid() == 0 );
    REQUIRE( app.m_currImageTimestamp.tv_sec > 0 );
    app.m_model.m_sample.m_adc = { 123, 3456 }; // Deliberately asymmetric copy-order fixture.
    std::array<uint16_t, 2> values{};
    REQUIRE( app.loadImageIntoStream( values.data() ) == 0 );
    REQUIRE( values == app.m_model.m_sample.m_adc );
    REQUIRE( app.loadImageIntoStream( values.data() ) < 0 );
    app.m_shutdown = true;
    REQUIRE( app.acquireAndCheckValid() > 0 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::configureAcquisition;
    MagAOX::app::accelSim::startAcquisition;
    MagAOX::app::accelSim::acquireAndCheckValid;
    MagAOX::app::accelSim::loadImageIntoStream;
#endif
    // clang-format on
}

/** \brief The actual app reports phase correction, lock, and holdover consistently.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim synchronized delay model", "[accelSim][app][sync-regression]" )
{
    appTest app;
    app.m_syncName = "accelSim-test-source";
    app.m_phaseClock.reset( 1000000000, 6000 );
    app.m_alpha = 1;
    app.syncTiming( 1, 1000000000 );
    REQUIRE( app.m_diagnostics.m_timing[triggerTime] == 0 );
    app.syncTiming( 2, 1000500000 );
    const auto &t = app.m_diagnostics.m_timing;
    REQUIRE( t[semaphorePeriod] == 500 );
    REQUIRE( t[delayModel] == Approx( 409.4 ) );
    REQUIRE( t[delayTarget] == Approx( 409.4 ) );
    REQUIRE( t[triggerInterval] == 0 );
    for( int n = 0; n < 20; ++n )
        app.m_phaseClock.consume( app.m_phaseClock.deadline(), .05, 166666 );
    app.updateSyncDiagnostics();
    REQUIRE( t[phaseError] == Approx( 0 ).margin( 1e-9 ) );
    REQUIRE( t[lock] == 1 );
    REQUIRE( t[delayBudget] == Approx( 8.333333 ) );
    REQUIRE( t[capped] == 0 );
    REQUIRE( app.m_diagnostics.m_cadenceRatio == 3 );
    app.frameGrabberPostPublish( nullptr );
    const auto deadline = app.m_phaseClock.deadline();
    app.invalidateSync();
    REQUIRE( app.m_snapshot.m_timing[lock] == 0 );
    REQUIRE( app.m_snapshot.m_timing[semaphorePeriod] == 0 );
    REQUIRE( app.m_snapshot.m_timing[delayApplied] == 0 );
    REQUIRE( app.m_snapshot.m_cadenceRatio == 0 );
    REQUIRE( app.m_phaseClock.deadline() == deadline );
    app.syncTiming( 3, 2000000000 );
    REQUIRE( t[lock] == 0 );
    app.syncTiming( 4, 2000500000 );
    REQUIRE( t[triggerInterval] == 0 );
    REQUIRE( app.m_phaseClock.m_valid );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::syncTiming;
    MagAOX::app::accelSim::updateSyncDiagnostics;
    MagAOX::app::accelSim::invalidateSync;
#endif
    // clang-format on
}

/** \brief Real framework publication and trigger consumption in a private namespace.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim real local streams", "[.][accelSim-streams]" )
{
    char path[] = "/tmp/accelSim-streams-XXXXXX";
    REQUIRE( mkdtemp( path ) != nullptr );
    REQUIRE( setenv( "MILK_SHM_DIR", path, 1 ) == 0 );
    for( uint32_t depth : { 1u, 4u } )
    {
        appTest app;
        app.setupConfig();
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        app.m_circBuffLength = depth;
        app.startWorker();
        streamFixture reader;
        reader.open( "acc000" );
        REQUIRE( reader.m_image.md->naxis == 3 );
        REQUIRE( reader.m_image.md->size[0] == 2 );
        REQUIRE( reader.m_image.md->size[1] == 1 );
        REQUIRE( reader.m_image.md->size[2] == depth );
        REQUIRE( reader.m_image.md->datatype == _DATATYPE_UINT16 );
        REQUIRE( reader.m_image.md->sem == IMAGE_NB_SEMAPHORE );
        uint64_t previous = 0;
        for( size_t n = 0; n < 100; ++n )
        {
            const auto pair = reader.read();
            REQUIRE( pair[0] <= 4095 );
            REQUIRE( pair[1] <= 4095 );
            REQUIRE( reader.m_counter >= previous );
            previous = reader.m_counter;
            REQUIRE( reader.m_acquired.tv_sec > 0 );
            REQUIRE( reader.m_plane < depth );
            REQUIRE( reader.m_plane == ( reader.m_counter - 1 ) % depth );
            REQUIRE( reader.m_planeCounter == reader.m_counter );
            REQUIRE( reader.m_acquired.tv_sec == reader.m_planeAcquired.tv_sec );
            REQUIRE( reader.m_acquired.tv_nsec == reader.m_planeAcquired.tv_nsec );
            REQUIRE( reader.m_written.tv_sec == reader.m_planeWritten.tv_sec );
            REQUIRE( reader.m_written.tv_nsec == reader.m_planeWritten.tv_nsec );
            REQUIRE( reader.m_written.tv_sec > reader.m_acquired.tv_sec ||
                     ( reader.m_written.tv_sec == reader.m_acquired.tv_sec &&
                       reader.m_written.tv_nsec >= reader.m_acquired.tv_nsec ) );
        }
    }
    {
        appTest app;
        app.setupConfig();
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        app.m_syncName = "accelSim-absent-reference";
        app.startWorker();
        streamFixture reader;
        reader.open( "acc000" );
        for( int n = 0; n < 10; ++n )
        {
            const auto pair = reader.read();
            REQUIRE( pair[0] <= 4095 );
            REQUIRE( pair[1] <= 4095 );
        }
    }
    {
        streamFixture source;
        source.create( "accelSim-trigger" );
        appTest app;
        app.setupConfig();
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        app.m_syncName = "accelSim-trigger";
        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE( app.startAcquisition() == 0 );
        std::array<uint16_t, 2> pair{};
        for( uint64_t counter : { 1u, 4u } )
        {
            source.m_image.md->cnt0 = counter;
            REQUIRE( ImageStreamIO_sempost( &source.m_image, -1 ) == 0 );
            // A due accelerometer frame may precede processing the queued reference.
            for( int n = 0; n < 4; ++n )
            {
                REQUIRE( app.acquireAndCheckValid() == 0 );
                REQUIRE( app.loadImageIntoStream( pair.data() ) == 0 );
            }
        }
        REQUIRE( app.m_diagnostics.m_timing[dropped] == 2 );
        REQUIRE( app.m_diagnostics.m_timing[written] == 8 );
        REQUIRE( app.m_diagnostics.m_timing[received] == 2 );
        REQUIRE( ImageStreamIO_sempost( &source.m_image, -1 ) == 0 );
        for( int n = 0; n < 4; ++n )
            REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.m_diagnostics.m_timing[dropped] == 2 );
        REQUIRE( app.m_phaseClock.m_producer == 4 );
        const auto discontinuities = app.m_diagnostics.m_discontinuities;
        const int  slot            = app.m_syncIndex;
        REQUIRE( ImageStreamIO_destroyIm( &source.m_image ) == 0 );
        source.m_open       = false;
        app.m_nextSyncProbe = 0;
        REQUIRE( app.acquireAndCheckValid() == 0 ); // Source loss cannot stop the accelerometer clock.
        REQUIRE_FALSE( app.m_reconfig );
        REQUIRE( app.m_snapshot.m_timing[lock] == 0 );
        REQUIRE( app.m_diagnostics.m_discontinuities == discontinuities );
        REQUIRE( app.loadImageIntoStream( pair.data() ) == 0 );
        source.create( "accelSim-trigger" );
        app.m_nextSyncProbe = 0;
        REQUIRE( app.acquireAndCheckValid() == 0 ); // Reattach without recreating acc000 or restarting physics.
        REQUIRE( source.m_image.semReadPID[slot] != 0 );
        source.m_image.md->cnt0 = 1;
        REQUIRE( ImageStreamIO_sempost( &source.m_image, -1 ) == 0 );
        for( int n = 0; n < 4; ++n )
            REQUIRE( app.acquireAndCheckValid() == 0 );
        REQUIRE( app.m_phaseClock.m_producer == 1 );
        REQUIRE( app.reconfig() == 0 );
        REQUIRE( source.m_image.semReadPID[slot] == 0 );
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::configureAcquisition;
    MagAOX::app::accelSim::startAcquisition;
    MagAOX::app::accelSim::acquireAndCheckValid;
    MagAOX::app::accelSim::loadImageIntoStream;
    MagAOX::app::accelSim::frameGrabberPostPublish;
    MagAOX::app::accelSim::waitSync;
    MagAOX::app::accelSim::reconfig;
#endif
    // clang-format on
}

/** \brief Ten-minute read-only timing observation of a separately launched local producer.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim sustained local publication benchmark", "[.][accelSim-live]" )
{
    REQUIRE( std::getenv( "ACCELSIM_LOCAL_BENCHMARK" ) != nullptr );
    REQUIRE( std::getenv( "MILK_SHM_DIR" ) != nullptr );
    const char  *rateSetting  = std::getenv( "ACCELSIM_EXPECTED_FPS" );
    const double expectedRate = rateSetting ? std::stod( rateSetting ) : 6000;
    REQUIRE( std::isfinite( expectedRate ) );
    REQUIRE( expectedRate > 0 );
    streamFixture reader;
    reader.open( "acc000" );
    REQUIRE( reader.m_image.md->datatype == _DATATYPE_UINT16 );
    REQUIRE( reader.m_image.md->size[0] == 2 );
    REQUIRE( reader.m_image.md->size[1] == 1 );
    const auto                  begin    = std::chrono::steady_clock::now();
    const uint64_t              first    = reader.m_image.md->cnt0;
    uint64_t                    previous = first, gaps = 0, observed = 0;
    double                      previousWrite = 0, intervalSum = 0, intervalSquared = 0, maximum = 0;
    std::array<uint64_t, 8>     histogram{};
    const std::array<double, 7> limits{ 100, 150, 175, 200, 250, 500, 1000 };
    while( std::chrono::steady_clock::now() - begin < std::chrono::minutes( 10 ) )
    {
        const auto pair = reader.read();
        REQUIRE( pair[0] <= 4095 );
        REQUIRE( pair[1] <= 4095 );
        const auto counter = reader.m_counter;
        if( counter == previous )
            continue;
        REQUIRE( counter > previous );
        gaps += counter - previous - 1;
        const auto   stamp   = reader.m_written;
        const double written = double( stamp.tv_sec ) * 1e6 + stamp.tv_nsec / 1000.0;
        if( previousWrite && counter == previous + 1 )
        {
            const double interval = written - previousWrite;
            intervalSum += interval;
            intervalSquared += interval * interval;
            maximum = std::max( maximum, interval );
            ++observed;
            ++histogram[std::upper_bound( limits.begin(), limits.end(), interval ) - limits.begin()];
        }
        previousWrite = written;
        previous      = counter;
    }
    const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - begin ).count();
    const double rate    = ( previous - first ) / seconds;
    REQUIRE( observed > 0 );
    const double mean = intervalSum / observed;
    std::cout << "publication_hz=" << rate << " reader_counter_gaps=" << gaps << " interval_mean_us=" << mean
              << " interval_std_us=" << std::sqrt( std::max( 0.0, intervalSquared / observed - mean * mean ) )
              << " max_observed_interval_us=" << maximum << " interval_histogram=";
    for( auto bin : histogram )
        std::cout << bin << ',';
    std::cout
        << "\nCompare producer sim_diag/skipped_publications, late_samples, max_lateness_us and RSS separately.\n";
    REQUIRE( rate == Approx( expectedRate ).epsilon( .01 ) );
}
#endif

} // namespace accelSimTest
} // namespace libXWCTest
