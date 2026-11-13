/** \file offline_adc.cpp
 * \brief Deterministic 8-kHz raw ADC generation and test-only spectral estimators.
 */
#include "offline_adc.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

/// \cond OFFLINE_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
constexpr double       pi    = 3.14159265358979323846;
constexpr std::int64_t epoch = 1800000000000000000LL, origin = 1000000000;
} // namespace

SyntheticAdcSource::SyntheticAdcSource( const SyntheticAdcConfig &config )
    : m_config( config ), m_force( config.m_seed ), m_sensor( config.m_seed + 1 ), m_jitter( config.m_seed + 2 ),
      m_delivery( config.m_seed + 3 )
{
    DDSPC::validate_adc_config( config.m_calibration );
    if( config.m_jitterNs >= 60000 || config.m_dropEvery == 1 || !std::isfinite( config.m_lowHz ) ||
        config.m_lowHz <= 0 || config.m_lowHz >= 4000 || !std::isfinite( config.m_highHz ) || config.m_highHz <= 0 ||
        config.m_highHz >= 4000 || !std::isfinite( config.m_forceNoise ) || config.m_forceNoise < 0 ||
        !std::isfinite( config.m_sensorNoise ) || config.m_sensorNoise < 0 ||
        !std::isfinite( config.m_clockDriftPpm ) || std::abs( config.m_clockDriftPpm ) > 5000 ||
        config.m_latencyJitterNs < 0 || config.m_latencyNs < 2000 || config.m_latencyNs > 1000000000 ||
        config.m_latencyJitterNs > config.m_latencyNs - 2000 ||
        ( config.m_dropPeriod && ( !config.m_dropCount || config.m_dropCount >= config.m_dropPeriod ) ) )
    {
        throw std::invalid_argument( "Invalid synthetic ADC timing, frequency or noise" );
    }
    for( auto amplitudes : { config.m_lowAmplitude, config.m_highAmplitude } )
    {
        for( double amplitude : amplitudes )
        {
            if( !std::isfinite( amplitude ) || amplitude < 0 )
            {
                throw std::invalid_argument( "Invalid SI acceleration amplitude" );
            }
        }
    }
}

bool SyntheticAdcSource::advance()
{
    const auto tick   = m_tick++;
    const auto jitter = m_config.m_jitterNs ? std::int64_t( m_jitter() % ( 2 * m_config.m_jitterNs + 1 ) ) -
                                                  std::int64_t( m_config.m_jitterNs )
                                            : 0;
    const auto acquired =
        origin + std::llround( double( tick ) * 125000 * ( 1 + m_config.m_clockDriftPpm * 1e-6 ) ) + jitter;
    const double time = double( acquired - origin ) * 1e-9;
    for( std::size_t axis = 0; axis < 2; ++axis )
    {
        const double phase = .7 * axis;
        m_physical[axis] = m_config.m_lowAmplitude[axis] * std::sin( 2 * pi * m_config.m_lowHz * time + phase ) +
                           m_config.m_highAmplitude[axis] * std::sin( 2 * pi * m_config.m_highHz * time + 2 * phase ) +
                           m_config.m_forceNoise * m_forceNormal( m_force );
        const auto  &c   = m_config.m_calibration;
        const double measured = m_physical[axis] + m_config.m_sensorNoise * m_sensorNormal( m_sensor );
        const double code     = std::round( c.m_zero[axis] + measured * c.m_sensitivity[axis] /
                                                                 ( c.m_polarity[axis] * c.m_voltsPerCount[axis] ) );
        if( !std::isfinite( code ) || code < c.m_minCode || code > c.m_maxCode )
        {
            throw std::runtime_error( "Synthetic ADC waveform clips the declared encoding" );
        }
        m_frame.m_raw[axis] = static_cast<std::uint16_t>( code );
    }
    m_header.m_counter = m_frame.m_counter = m_tick;
    m_frame.m_acquiredRealNs               = epoch + acquired;
    m_frame.m_writtenRealNs                = epoch + acquired + 1000;
    const auto latencyJitter =
        m_config.m_latencyJitterNs
            ? std::int64_t( m_delivery() % ( 2 * m_config.m_latencyJitterNs + 1 ) ) - m_config.m_latencyJitterNs
            : 0;
    const auto received = acquired + m_config.m_latencyNs + latencyJitter;
    m_clock             = { epoch + received, received, 20 };
    const bool periodic = m_config.m_dropEvery && m_tick % m_config.m_dropEvery == 0;
    const auto elapsed  = tick >= m_config.m_dropStart ? tick - m_config.m_dropStart : 0;
    const bool burst    = tick >= m_config.m_dropStart &&
                          ( m_config.m_dropPeriod ? elapsed % m_config.m_dropPeriod : elapsed ) < m_config.m_dropCount;
    return !periodic && !burst;
}

DDSPC::AdcHeader SyntheticAdcSource::header() const
{
    return m_header;
}
DDSPC::AdcFrame SyntheticAdcSource::frame( std::size_t index ) const
{
    if( index != 0 )
    {
        throw std::out_of_range( "Synthetic ADC has one plane" );
    }
    return m_frame;
}
DDSPC::AdcClock SyntheticAdcSource::clock() const
{
    return m_clock;
}
std::array<double, 2> SyntheticAdcSource::physical() const
{
    return m_physical;
}

double tone_amplitude( const std::vector<double> &values, const std::vector<std::int64_t> &times, double frequency )
{
    if( values.size() < 4 || values.size() != times.size() || !std::isfinite( frequency ) || frequency <= 0 )
    {
        throw std::invalid_argument( "Invalid timestamped spectral record" );
    }
    const double mean = std::accumulate( values.begin(), values.end(), 0. ) / values.size();
    double       ss = 0, cc = 0, sc = 0, ys = 0, yc = 0;
    for( std::size_t i = 0; i < values.size(); ++i )
    {
        const double phase = 2 * pi * frequency * double( times[i] - times[0] ) * 1e-9;
        const double s = std::sin( phase ), c = std::cos( phase );
        ss += s * s;
        cc += c * c;
        sc += s * c;
        ys += ( values[i] - mean ) * s;
        yc += ( values[i] - mean ) * c;
    }
    const double det = ss * cc - sc * sc;
    if( !( det > 0 ) )
    {
        throw std::runtime_error( "Singular sinusoid fit" );
    }
    return std::hypot( ( ys * cc - yc * sc ) / det, ( yc * ss - ys * sc ) / det );
}

std::vector<double> periodogram( const std::vector<double> &values, double rate, double maxHz )
{
    if( values.size() < 4 || !std::isfinite( rate ) || rate <= 0 || !std::isfinite( maxHz ) || maxHz <= 0 )
    {
        throw std::invalid_argument( "Invalid PSD record" );
    }
    const auto          n    = values.size();
    const auto          bins = std::size_t( std::min( maxHz, rate / 2 ) * n / rate ) + 1;
    std::vector<double> windowed( n ), result( bins );
    const double        mean = std::accumulate( values.begin(), values.end(), 0. ) / n;
    double              norm = 0;
    for( std::size_t i = 0; i < n; ++i )
    {
        const double w = .5 - .5 * std::cos( 2 * pi * i / n );
        windowed[i]    = w * ( values[i] - mean );
        norm += w * w;
    }
    for( std::size_t k = 0; k < bins; ++k )
    {
        double       real = 0, imaginary = 0;
        const double angle = 2 * pi * k / n, c = std::cos( angle ), s = std::sin( angle );
        double       x = 1, y = 0;
        for( double value : windowed )
        {
            real += value * x;
            imaginary += value * y;
            const double next = x * c - y * s;
            y                 = y * c + x * s;
            x                 = next;
        }
        const double factor = k == 0 || 2 * k == n ? 1 : 2;
        result[k]           = factor * ( real * real + imaginary * imaginary ) / ( rate * norm );
    }
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
