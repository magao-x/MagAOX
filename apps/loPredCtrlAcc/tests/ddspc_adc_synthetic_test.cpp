/** \file ddspc_adc_synthetic_test.cpp
 * \brief Actual ADC conversion of independent SI waveforms, jitter, depth-one loss and zero acceleration.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "offline_adc.hpp"
#include <cmath>

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Zero physical acceleration encodes2048 on both axes and retains fresh, advancing original timestamps. */
TEST_CASE( "Offline ADC baseline uses calibrated zero codes at 8 kHz", "[adc][synthetic-adc]" )
{
    SyntheticAdcConfig config;
    config.m_lowAmplitude = config.m_highAmplitude = { 0, 0 };
    config.m_forceNoise = config.m_sensorNoise = 0;
    SyntheticAdcSource source( config );
    DDSPC::AdcCapture  capture( config.m_calibration );
    capture.attached( 1, 1 );
    for( int i = 0; i < 8000; ++i )
    {
        REQUIRE( source.advance() );
        REQUIRE( capture.drain( source ) == 1 );
        const auto sample = capture.status().m_latest;
        REQUIRE( sample.m_frame.m_raw == std::array<std::uint16_t, 2>{ 2048, 2048 } );
        REQUIRE( sample.m_acceleration == std::array<double, 2>{ 0, 0 } );
        REQUIRE( sample.m_frame.m_counter == std::uint64_t( i + 1 ) );
        REQUIRE( sample.m_acquiredNs == 1000000000LL + std::int64_t( i ) * 125000 );
        REQUIRE( sample.m_publishedNs == sample.m_acquiredNs + 1000 );
        REQUIRE( sample.m_receivedNs == sample.m_acquiredNs + 3000 );
    }
    REQUIRE( capture.status().m_rateHz == Approx( 8000 ) );
    REQUIRE( capture.status().m_retained == config.m_calibration.m_capacity );
    REQUIRE( capture.status().m_overwrites == 8000 - config.m_calibration.m_capacity );
}

/** \brief Quantized mixed28.5/56-Hz accelerations retain amplitudes, frequencies and clock evidence with noise/loss. */
TEST_CASE( "Offline synthetic ADC recovers SI tones with timestamp jitter and dropped reads", "[adc][synthetic-adc]" )
{
    const bool         faults = GENERATE( false, true );
    SyntheticAdcConfig config;
    config.m_jitterNs  = faults ? 5000 : 0;
    config.m_dropEvery = faults ? 997 : 0;
    SyntheticAdcSource source( config );
    DDSPC::AdcCapture  capture( config.m_calibration );
    capture.attached( 1, 1 );
    std::array<std::vector<double>, 2> values;
    std::vector<std::int64_t>          times;
    std::uint64_t                      missing = 0;
    for( int i = 0; i < 32000; ++i )
    {
        if( !source.advance() )
        {
            capture.tick( source.clock().m_monoNs );
            ++missing;
            continue;
        }
        REQUIRE( capture.drain( source ) == 1 );
        const auto sample = capture.status().m_latest;
        REQUIRE( sample.m_acquiredNs == sample.m_frame.m_acquiredRealNs - 1800000000000000000LL );
        REQUIRE( sample.m_frame.m_counter == std::uint64_t( i + 1 ) );
        REQUIRE( sample.m_acquiredNs < sample.m_publishedNs );
        REQUIRE( sample.m_publishedNs < sample.m_receivedNs );
        if( !times.empty() )
        {
            REQUIRE( sample.m_acquiredNs > times.back() );
        }
        times.push_back( sample.m_acquiredNs );
        for( int axis = 0; axis < 2; ++axis )
        {
            values[axis].push_back( sample.m_acceleration[axis] );
        }
    }
    REQUIRE( capture.status().m_dropped == missing );
    REQUIRE( capture.status().m_samples == 32000 - missing );
    REQUIRE( capture.status().m_rateHz == Approx( 8000 ).epsilon( .0025 ) );
    REQUIRE( capture.status().m_valid );
    for( int axis = 0; axis < 2; ++axis )
    {
        for( int tone = 0; tone < 2; ++tone )
        {
            const double expected  = tone ? config.m_highHz : config.m_lowHz;
            const double amplitude = tone ? config.m_highAmplitude[axis] : config.m_lowAmplitude[axis];
            REQUIRE( tone_amplitude( values[axis], times, expected ) == Approx( amplitude ).margin( .0015 ) );
            double       peak = 0, maximum = -1;
            const double start = tone ? 54 : 27, stop = tone ? 58 : 30;
            for( double f = start; f <= stop; f += .05 )
            {
                const double response = tone_amplitude( values[axis], times, f );
                if( response > maximum )
                {
                    maximum = response;
                    peak    = f;
                }
            }
            REQUIRE( std::abs( peak - expected ) <= .1 );
        }
    }
}

/** \brief Spectral density has a known sinusoid normalization independent of the ADC implementation. */
TEST_CASE( "Offline PSD integrates a known sinusoid variance", "[adc][synthetic-adc]" )
{
    std::vector<double> signal( 2000 );
    for( std::size_t i = 0; i < signal.size(); ++i )
    {
        signal[i] = 2 * std::sin( 2 * 3.14159265358979323846 * 56 * i / 2000 );
    }
    const auto spectrum = periodogram( signal, 2000, 1000 );
    double     variance = 0;
    for( double power : spectrum )
    {
        variance += power;
    } // df =1Hz.
    REQUIRE( variance == Approx( 2 ).epsilon( 1e-8 ) );
}
} // namespace libXWCTest::loPredCtrlAccTest
