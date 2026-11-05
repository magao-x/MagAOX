/** \file accelFFCtrlModel_test.cpp
 * \brief Deterministic raw-only harmonic estimation, timing, noise, and independent weighted-fit tests.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlModel.hpp"

#include <complex>
#include <iostream>
#include <random>
#include <vector>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_FIXTURES
constexpr double  pi    = 3.14159265358979323846;
constexpr int64_t epoch = 9000000000000000000; ///< Large monotonic epoch to expose loss of timestamp precision.

/// Encode analytic acceleration independently of the simulator, including the actual ADC quantizer.
adcSample encode( const parameters     &config /**< [in] decoder calibration */,
                  int64_t               time /**< [in] acquisition ns */,
                  uint64_t              sequence /**< [in] publication identity */,
                  std::array<double, 2> acceleration /**< [in] physical sensor-axis acceleration */ )
{
    adcSample value;
    value.m_timeNs   = time;
    value.m_sequence = sequence;
    for( size_t axis = 0; axis < 2; ++axis )
        value.m_adc[config.m_channel[axis]] = static_cast<uint16_t>( std::clamp(
            std::round( config.m_zero[axis] + acceleration[axis] * config.m_sensitivity[axis] /
                                                  ( config.m_voltsPerBit[axis] * config.m_polarity[axis] ) ),
            0.,
            4095. ) );
    return value;
}

/// Return a fine but representable ADC calibration for isolating estimator errors from weak-signal quantization.
parameters fineCalibration()
{
    parameters config;
    config.m_voltsPerBit = { 1e-6, 1e-6 };
    config.m_sensitivity = { 1, 1 };
    return config;
}

/// Captured predictions for independent harmonic regression and RMS scoring, never estimator input.
struct toneTrace
{
    std::vector<double> m_time, m_estimate,
        m_truth;       ///< Output-cadence time, acceleration prediction and analytic truth.
    prediction m_last; ///< Last complete numerical result.
    double     m_inputPower{ 0 }, m_errorPower{ 0 }, m_outputPower{ 0 }; ///< Accumulated scoring powers.
};

/// Drive every source sample while predicting at 2 kHz; retain only offline scoring data after settling.
toneTrace traceTone( const parameters &config /**< [in] numerical settings */,
                     double            frequency /**< [in] actual tone Hz */,
                     double            amplitude  = .001 /**< [in] peak acceleration */,
                     double            noiseSigma = 0 /**< [in] independent Gaussian acceleration noise */,
                     double            duration   = 4 /**< [in] simulated seconds */,
                     uint64_t          seed       = 42 /**< [in] fixed noise seed */ )
{
    harmonicEstimator                estimator( config );
    std::mt19937_64                  rng( seed );
    std::normal_distribution<double> normal;
    toneTrace                        trace;
    for( uint64_t n = 0; n < uint64_t( duration * 8000 ); ++n )
    {
        const double time  = n / 8000.;
        const double a     = amplitude * std::cos( 2 * pi * frequency * time + .37 );
        const auto   value = encode(
            config, epoch + n * 125000, n, { a + noiseSigma * normal( rng ), .7 * a + noiseSigma * normal( rng ) } );
        REQUIRE( estimator.consume( value, value.m_timeNs ) );
        if( n % 4 != 3 || time < 1 )
            continue;
        trace.m_last = estimator.predict( value.m_timeNs );
        REQUIRE( trace.m_last.m_state == estimatorState::ready );
        trace.m_time.push_back( time );
        trace.m_estimate.push_back( trace.m_last.m_acceleration[0] );
        trace.m_truth.push_back( a );
        trace.m_inputPower += a * a;
        trace.m_outputPower += trace.m_last.m_acceleration[0] * trace.m_last.m_acceleration[0];
        trace.m_errorPower += std::pow( trace.m_last.m_acceleration[0] - a, 2 );
    }
    REQUIRE( estimator.samples() == uint64_t( duration * 8000 ) );
    return trace;
}

/// Fit captured output at its actual signal frequency using an independent batch QR with a nuisance offset.
std::complex<double> harmonicResponse( const toneTrace &trace /**< [in] offline predictions */,
                                       double           frequency /**< [in] scoring frequency */,
                                       double           amplitude /**< [in] known input peak */ )
{
    Eigen::MatrixXd design( trace.m_time.size(), 3 );
    Eigen::VectorXd values( trace.m_time.size() );
    for( size_t n = 0; n < trace.m_time.size(); ++n )
    {
        const double phase = 2 * pi * frequency * trace.m_time[n] + .37;
        design.row( n ) << 1, std::cos( phase ), std::sin( phase );
        values[n] = trace.m_estimate[n];
    }
    const Eigen::Vector3d coefficients = design.colPivHouseholderQr().solve( values );
    return std::complex<double>( coefficients[1], -coefficients[2] ) / amplitude;
}
/// \endcond

/** \brief ADC offsets/scales, channel selection and polarity preserve signed physical units and legal rails.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl ADC calibration and independent channels", "[accelFFCtrl][numerical]" )
{
    parameters config;
    config.m_zero        = { 2048, 1024 };
    config.m_voltsPerBit = { .01, .02 };
    config.m_sensitivity = { 2, 4 };
    config.m_channel     = { 1, 0 };
    config.m_polarity    = { -1, 1 };
    harmonicEstimator estimator( config );
    adcSample         value;
    value.m_adc = { 1014, 2058 };
    for( uint64_t n = 0; n < 4000; ++n )
    {
        value.m_timeNs   = epoch + n * 125000;
        value.m_sequence = n;
        REQUIRE( estimator.consume( value, value.m_timeNs ) );
    }
    const auto result = estimator.predict( value.m_timeNs );
    REQUIRE( result.m_state == estimatorState::ready );
    for( size_t axis = 0; axis < 2; ++axis )
    {
        REQUIRE( result.m_offset[axis] == Approx( -.05 ).margin( 1e-13 ) );
        REQUIRE( result.m_amplitude[axis] < 1e-13 );
        REQUIRE( std::abs( result.m_pointing[axis] ) < 1e-17 );
    }
    estimator.reset();
    value.m_adc = { 0, 4095 };
    REQUIRE( estimator.consume( value, value.m_timeNs ) );
    value.m_adc[0] = 4096;
    ++value.m_sequence;
    value.m_timeNs += 125000;
    REQUIRE_FALSE( estimator.consume( value, value.m_timeNs ) );
    REQUIRE( estimator.state() == estimatorState::invalidSample );
}

/** \brief Matched sinusoids reconstruct both channel amplitudes, sample phases, offsets and future pointing.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl matched amplitude phase rotation and prediction", "[accelFFCtrl][numerical]" )
{
    for( double frequency : { 28., 28.5, 29. } )
        for( double rotation : { 0., 90., -37., 180. } )
        {
            auto config              = fineCalibration();
            config.m_frequency       = frequency;
            config.m_rotationDegrees = rotation;
            config.m_predictionNs    = 3000000;
            harmonicEstimator           estimator( config );
            const std::array<double, 2> amplitude{ .001, .0007 }, phase{ .31, -1.17 }, offset{ .0002, -.0001 };
            adcSample                   value;
            for( uint64_t n = 0; n < 12000; ++n )
            {
                const double t = n / 8000.;
                value          = encode( config,
                                         epoch + n * 125000,
                                         n,
                                         { offset[0] + amplitude[0] * std::cos( 2 * pi * frequency * t + phase[0] ),
                                           offset[1] + amplitude[1] * std::cos( 2 * pi * frequency * t + phase[1] ) } );
                REQUIRE( estimator.consume( value, value.m_timeNs + 100000 ) );
            }
            const int64_t now    = value.m_timeNs + 500000;
            const auto    result = estimator.predict( now );
            REQUIRE( result.m_state == estimatorState::ready );
            REQUIRE( result.m_targetTimeNs == now + config.m_predictionNs );
            REQUIRE( result.m_sequence == value.m_sequence );
            REQUIRE( result.m_generation == value.m_generation );
            const double          target     = ( result.m_targetTimeNs - epoch ) * 1e-9;
            const double          sampleTime = ( value.m_timeNs - epoch ) * 1e-9;
            std::array<double, 2> pointing;
            for( size_t axis = 0; axis < 2; ++axis )
            {
                REQUIRE( result.m_amplitude[axis] == Approx( amplitude[axis] ).epsilon( .001 ) );
                REQUIRE( result.m_offset[axis] == Approx( offset[axis] ).margin( 1e-7 ) );
                REQUIRE(
                    std::abs( std::remainder( result.m_phase[axis] - ( 2 * pi * frequency * sampleTime + phase[axis] ),
                                              2 * pi ) ) < .001 );
                const double desired = amplitude[axis] * std::cos( 2 * pi * frequency * target + phase[axis] );
                REQUIRE( result.m_acceleration[axis] == Approx( desired ).margin( 2e-7 ) );
                pointing[axis] = -desired / ( std::pow( 2 * pi * frequency, 2 ) * config.m_leverArm );
            }
            const double alpha = rotation * pi / 180;
            REQUIRE( result.m_pointing[0] ==
                     Approx( std::cos( alpha ) * pointing[0] - std::sin( alpha ) * pointing[1] ).margin( 1e-12 ) );
            REQUIRE( result.m_pointing[1] ==
                     Approx( std::sin( alpha ) * pointing[0] + std::cos( alpha ) * pointing[1] ).margin( 1e-12 ) );
            for( int64_t horizon : { 0LL, 1000000LL, 10000000LL, 20000000LL } )
            {
                const auto future = estimator.predict( now, now + horizon );
                REQUIRE( future.m_state == estimatorState::ready );
                REQUIRE( future.m_acceleration[0] ==
                         Approx( amplitude[0] *
                                 std::cos( 2 * pi * frequency * ( ( now + horizon - epoch ) * 1e-9 ) + phase[0] ) )
                             .margin( 2e-7 ) );
            }
        }
}

/** \brief Recursive window matches a separately weighted batch QR, including nonuniform acquisition intervals.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl irregular timestamps and weighted QR oracle", "[accelFFCtrl][numerical]" )
{
    const auto config = fineCalibration();
    for( bool irregular : { false, true } )
    {
        harmonicEstimator   estimator( config );
        constexpr size_t    count = 4000;
        Eigen::MatrixXd     design( count, 3 ), acceleration( count, 2 );
        std::vector<double> weight( count );
        int64_t             time = epoch;
        for( size_t n = 0; n < count; ++n )
        {
            const int64_t dt = irregular ? 100000 + ( n % 5 ) * 12500 : 125000;
            if( n )
                time += dt;
            const double t = ( time - epoch ) * 1e-9;
            const auto   value =
                encode( config,
                        time,
                        n,
                        { .0009 * std::cos( 2 * pi * 28.5 * t + .4 ) + .0002 * std::sin( 2 * pi * 54 * t ),
                          .0006 * std::sin( 2 * pi * 28.5 * t ) + .0001 } );
            REQUIRE( estimator.consume( value, time + 50000 ) );
            design.row( n ) << 1, std::cos( 2 * pi * config.m_frequency * t ),
                std::sin( 2 * pi * config.m_frequency * t );
            for( size_t axis = 0; axis < 2; ++axis )
                acceleration( n, axis ) = ( double( value.m_adc[axis] ) - config.m_zero[axis] ) *
                                          config.m_voltsPerBit[axis] / config.m_sensitivity[axis];
            weight[n] = -std::expm1( -2 * ( n ? dt * 1e-9 : 1 / config.m_sampleRate ) / config.m_memoryTime );
        }
        // The contribution of sample i is beta_i * sum(beta_i..beta_N) * product(lambda_(i+1)..lambda_N).
        double sum = 0, product = 1;
        for( size_t n = count; n-- > 0; )
        {
            sum += weight[n];
            const double factor = std::sqrt( weight[n] * sum * product );
            design.row( n ) *= factor;
            acceleration.row( n ) *= factor;
            product *= 1 - weight[n];
        }
        const Eigen::Matrix<double, 3, 2> coefficients = design.colPivHouseholderQr().solve( acceleration );
        const auto                        result       = estimator.predict( time + 50000, time + 1200000 );
        REQUIRE( result.m_state == estimatorState::ready );
        for( size_t axis = 0; axis < 2; ++axis )
        {
            REQUIRE( result.m_offset[axis] == Approx( coefficients( 0, axis ) ).margin( 1e-14 ) );
            REQUIRE( result.m_amplitude[axis] ==
                     Approx( std::hypot( coefficients( 1, axis ), coefficients( 2, axis ) ) ).epsilon( 1e-10 ) );
            const double phase = 2 * pi * config.m_frequency * ( ( result.m_targetTimeNs - epoch ) * 1e-9 );
            REQUIRE( result.m_acceleration[axis] ==
                     Approx( coefficients( 1, axis ) * std::cos( phase ) + coefficients( 2, axis ) * std::sin( phase ) )
                         .margin( 1e-14 ) );
        }
    }
}

/** \brief Every buffered sample contributes; duplicates cannot refresh a stopped source or conceal missing records.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl batching duplicates and freshness faults", "[accelFFCtrl][numerical]" )
{
    auto              config = fineCalibration();
    harmonicEstimator direct( config ), batched( config );
    adcSample         last;
    for( uint64_t base = 0; base < 8000; base += 4 )
    {
        const int64_t receipt = epoch + ( base + 3 ) * 125000 + 100000;
        for( uint64_t n = base; n < base + 4; ++n )
        {
            last = encode( config, epoch + n * 125000, n, { .001 * std::cos( 2 * pi * 28.5 * n / 8000. ), 0 } );
            REQUIRE( direct.consume( last, last.m_timeNs ) );
            REQUIRE( batched.consume( last, receipt ) );
            REQUIRE_FALSE( batched.consume( last, receipt ) );
        }
        const auto a = direct.predict( receipt ), b = batched.predict( receipt );
        REQUIRE( a.m_state == b.m_state );
        REQUIRE( a.m_acceleration == b.m_acceleration );
    }
    REQUIRE( batched.samples() == 8000 );
    REQUIRE_FALSE( batched.consume( last, last.m_timeNs + config.m_maxAgeNs + 1 ) );
    const auto stale = batched.predict( last.m_timeNs + config.m_maxAgeNs + 1 );
    REQUIRE( stale.m_state == estimatorState::stale );
    REQUIRE( stale.m_pointing == std::array<double, 2>{ 0, 0 } );
    ++last.m_sequence;
    last.m_timeNs += 125000;
    REQUIRE_FALSE( batched.consume( last, last.m_timeNs ) );
    batched.reset();
    REQUIRE( batched.consume( last, last.m_timeNs ) );
    last.m_sequence += 2;
    last.m_timeNs += 125000;
    REQUIRE_FALSE( batched.consume( last, last.m_timeNs ) );
    REQUIRE( batched.state() == estimatorState::sequenceFault );
    for( int failure = 0; failure < 7; ++failure )
    {
        batched.reset();
        last = encode( config, epoch, 5, {} );
        REQUIRE( batched.consume( last, epoch ) );
        ++last.m_sequence;
        last.m_timeNs += 125000;
        int64_t now = last.m_timeNs;
        if( failure == 0 )
            ++last.m_generation;
        if( failure == 1 )
            last.m_sequence = 4;
        if( failure == 2 )
            last.m_timeNs = epoch;
        if( failure == 3 )
            last.m_timeNs = epoch - 1;
        if( failure == 4 )
            --now;
        if( failure == 5 )
            last.m_timeNs = now = epoch + config.m_maxIntervalNs + 1;
        if( failure == 6 )
            now += config.m_maxAgeNs + 1;
        REQUIRE_FALSE( batched.consume( last, now ) );
        REQUIRE( batched.state() != estimatorState::ready );
        REQUIRE( batched.state() != estimatorState::warming );
        REQUIRE( batched.predict( now ).m_pointing == std::array<double, 2>{ 0, 0 } );
    }
}

/** \brief Startup configuration and target-time errors cannot leak held predictions or corrupt a valid configuration.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl configuration and prediction validity", "[accelFFCtrl][numerical]" )
{
    const auto        good = fineCalibration();
    harmonicEstimator estimator( good );
    for( int n = 0; n < 17; ++n )
    {
        auto bad = good;
        if( n == 0 )
            bad.m_frequency = 0;
        if( n == 1 )
            bad.m_frequency = NAN;
        if( n == 2 )
            bad.m_memoryTime = 0;
        if( n == 3 )
            bad.m_memoryTime = INFINITY;
        if( n == 4 )
            bad.m_sampleRate = 40;
        if( n == 5 )
            bad.m_leverArm = -1;
        if( n == 6 )
            bad.m_rotationDegrees = NAN;
        if( n == 7 )
            bad.m_voltsPerBit[0] = 0;
        if( n == 8 )
            bad.m_sensitivity[1] = INFINITY;
        if( n == 9 )
            bad.m_zero[0] = NAN;
        if( n == 10 )
            bad.m_channel[0] = 2;
        if( n == 11 )
            bad.m_polarity[1] = 0;
        if( n == 12 )
            bad.m_predictionNs = -1;
        if( n == 13 )
            bad.m_maxAgeNs = 0;
        if( n == 14 )
            bad.m_maxIntervalNs = 1000000000;
        if( n == 15 )
            bad.m_predictionNs = bad.m_maxPredictionNs + 1;
        if( n == 16 )
            bad.m_maxPredictionNs = 0;
        REQUIRE_THROWS_AS( estimator.configure( bad ), std::invalid_argument );
    }
    REQUIRE( estimator.predict( epoch ).m_state == estimatorState::warming );
    REQUIRE( estimator.predict( -1 ).m_state == estimatorState::invalidRequest );
    for( uint64_t n = 0; n < 4000; ++n )
        REQUIRE( estimator.consume( encode( good, epoch + n * 125000, n, {} ), epoch + n * 125000 ) );
    const int64_t now = epoch + 3999 * 125000;
    REQUIRE( estimator.predict( now ).m_state == estimatorState::ready );
    REQUIRE( estimator.predict( now, now - 1 ).m_state == estimatorState::invalidRequest );
    REQUIRE( estimator.predict( now, now + good.m_maxPredictionNs + 1 ).m_state == estimatorState::invalidRequest );
    REQUIRE( estimator.predict( now ).m_state == estimatorState::ready );
    estimator.configure( good );
    REQUIRE( estimator.samples() == 0 );
    REQUIRE( estimator.predict( now ).m_state == estimatorState::warming );
}

/** \brief Numeric overflow, unobservable fits and invalid clock/counter transitions return cleared fault results.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl numerical faults and clock boundaries", "[accelFFCtrl][numerical]" )
{
    parameters        config;
    harmonicEstimator estimator( config );
    adcSample         value;
    value.m_adc = { 2048, 2048 };
    REQUIRE( estimator.consume( value, 0 ) ); // A numerical monotonic epoch of zero is valid.
    REQUIRE_FALSE( estimator.consume( value, -1 ) );
    REQUIRE( estimator.state() == estimatorState::timingFault );
    estimator.reset();
    value.m_timeNs   = 100;
    value.m_sequence = std::numeric_limits<uint64_t>::max();
    REQUIRE( estimator.consume( value, 100 ) );
    value.m_timeNs   = 125100;
    value.m_sequence = 0;
    REQUIRE_FALSE( estimator.consume( value, value.m_timeNs ) );
    REQUIRE( estimator.state() == estimatorState::sequenceFault );
    config.m_zero[0]        = -std::numeric_limits<double>::max();
    config.m_voltsPerBit[0] = 2;
    config.m_sensitivity[0] = 1;
    estimator.configure( config );
    REQUIRE_FALSE( estimator.consume( value, value.m_timeNs ) );
    REQUIRE( estimator.state() == estimatorState::numericalFault );
    REQUIRE( estimator.predict( value.m_timeNs ).m_amplitude == std::array<double, 2>{ 0, 0 } );
    config              = fineCalibration();
    config.m_memoryTime = 1e-9; // Only one regressor row survives: conditioning must inhibit, not invent a fit.
    estimator.configure( config );
    for( uint64_t n = 0; n < 1000; ++n )
        REQUIRE( estimator.consume( encode( config, n * 125000, n, {} ), n * 125000 ) );
    REQUIRE( estimator.predict( 999 * 125000 ).m_state == estimatorState::numericalFault );
    config                = fineCalibration();
    config.m_predictionNs = 1;
    estimator.configure( config );
    REQUIRE( estimator.predict( std::numeric_limits<int64_t>::max() ).m_state == estimatorState::invalidRequest );
    REQUIRE( estimator.state() == estimatorState::warming );
    value = encode( config, 10, 0, {} );
    REQUIRE_FALSE( estimator.consume( value, 9 ) );
    REQUIRE( estimator.state() == estimatorState::timingFault );
}

/** \brief Detuning has measured gain and phase, not an assumed phase-free response or corrected frequency estimate.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl frequency mismatch transfer", "[accelFFCtrl][numerical]" )
{
    const auto config = fineCalibration();
    for( double delta : { -2., -1., -.5, -.1, 0., .1, .5, 1., 2. } )
    {
        const double               frequency   = config.m_frequency + delta;
        const auto                 trace       = traceTone( config, frequency );
        const auto                 response    = harmonicResponse( trace, frequency, .001 );
        const double               lambda      = std::exp( -2 / ( 8000 * config.m_memoryTime ) );
        const std::complex<double> z           = std::polar( 1., -2 * pi * delta / 8000 );
        const auto                 approximate = std::pow( ( 1 - lambda ) / ( 1. - lambda * z ), 2 );
        CAPTURE( delta, response, approximate );
        REQUIRE( std::abs( response - approximate ) < .012 );
        std::cout << "Detuning " << delta << " Hz: acceleration gain " << std::abs( response ) << ", phase "
                  << std::arg( response ) * 180 / pi << " deg\n";
    }
}

/** \brief Sensor noise/dither permits weak-tone reconstruction, whereas an undithered sub-count signal may vanish.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl sensor noise ADC quantization and broadband rejection", "[accelFFCtrl][numerical]" )
{
    const parameters config;
    const double     peak      = .002 * std::sqrt( 2. );
    const auto       invisible = traceTone( config, 28.5, peak, 0, 3 );
    REQUIRE( invisible.m_outputPower == 0 );
    for( uint64_t seed : { 42, 123, 701 } )
    {
        const auto   trace         = traceTone( config, 28.5, peak, std::sqrt( 1e-8 * 8000 / 2 ), 8, seed );
        const auto   response      = harmonicResponse( trace, 28.5, peak );
        const double relativeError = std::sqrt( trace.m_errorPower / trace.m_inputPower );
        CAPTURE( seed, response, relativeError );
        REQUIRE( std::abs( std::abs( response ) - 1 ) < .07 );
        REQUIRE( std::abs( std::arg( response ) ) < .06 );
        REQUIRE( relativeError < .20 );
        std::cout << "Quantized noisy weak tone seed " << seed << ": gain " << std::abs( response ) << ", phase "
                  << std::arg( response ) * 180 / pi << " deg, relative RMS error " << relativeError << '\n';
    }
    const double sigma     = std::sqrt( 1e-8 * 8000 / 2 );
    const auto   noise     = traceTone( config, 28.5, 0, sigma, 8 );
    const double outputRms = std::sqrt( noise.m_outputPower / noise.m_time.size() );
    REQUIRE( outputRms / sigma < .07 );
    std::cout << "White sensor noise: isolated acceleration RMS / input RMS " << outputRms / sigma << '\n';
}

/** \brief Far-off target acceleration is rejected even after the harmonic inverse is expressed in pointing units.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl unrelated vibration transmission", "[accelFFCtrl][numerical]" )
{
    const auto config = fineCalibration();
    for( double frequency : { 11., 14., 17., 20., 25., 54., 56., 58., 75., 85., 120., 220., 300., 900., 1900., 3500. } )
    {
        const auto   trace            = traceTone( config, frequency, .001, 0, 3 );
        const double accelerationGain = std::sqrt( trace.m_outputPower / trace.m_inputPower );
        const double pointingGain     = accelerationGain * std::pow( frequency / config.m_frequency, 2 );
        CAPTURE( frequency, accelerationGain, pointingGain );
        REQUIRE( pointingGain < .5 );
        if( frequency >= 54 )
            REQUIRE( pointingGain < .08 );
        double residual = 0;
        for( size_t n = 0; n < trace.m_time.size(); ++n )
            residual +=
                std::pow( trace.m_truth[n] - std::pow( frequency / config.m_frequency, 2 ) * trace.m_estimate[n], 2 );
        std::cout << "Unrelated " << frequency << " Hz: acceleration RMS gain " << accelerationGain
                  << ", pointing RMS gain " << pointingGain << ", hypothetical subtraction change "
                  << 10 * std::log10( residual / trace.m_inputPower ) << " dB\n";
    }
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
