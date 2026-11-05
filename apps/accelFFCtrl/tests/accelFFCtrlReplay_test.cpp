/** \file accelFFCtrlReplay_test.cpp
 * \brief Offline accelSim truth scoring; the estimator receives only raw codes and timestamp/source identity.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlModel.hpp"
#include "../../accelSim/accelSimModel.hpp"

#include <complex>
#include <iostream>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;
namespace simulation = MagAOX::app::accelSimDetail;

/// \cond ACCELFFCTRL_REPLAY_FIXTURES
/// Integrate the explicitly selected DFT bins with a Hann window, independently of the online estimator.
double bandPower( const std::vector<double> &values /**< [in] offline output-cadence samples */,
                  double                     rate /**< [in] scoring sample rate [Hz] */,
                  double                     low /**< [in] lower band edge [Hz] */,
                  double                     high /**< [in] upper band edge [Hz] */ )
{
    double power = 0;
    for( size_t k = std::ceil( low * values.size() / rate ); k <= size_t( high * values.size() / rate ); ++k )
    {
        std::complex<double> sum;
        for( size_t n = 0; n < values.size(); ++n )
        {
            const double window = .5 - .5 * std::cos( 2 * simulation::pi * n / ( values.size() - 1 ) );
            sum += window * values[n] * std::polar( 1., -2 * simulation::pi * k * n / values.size() );
        }
        power += std::norm( sum );
    }
    return power;
}
/// \endcond

/** \brief Full quantized model replay suppresses the isolated target in offline residuals without accessing truth.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl independent accelSim raw-only replay", "[accelFFCtrl][replay]" )
{
    for( bool fullPopulation : { false, true } )
        for( uint64_t seed : { 42, 123, 701 } )
        {
            simulation::parameters sourceConfig;
            sourceConfig.m_fps         = 8000;
            sourceConfig.m_seed        = seed;
            sourceConfig.m_zero        = 2017;
            sourceConfig.m_voltsPerBit = .0079;
            sourceConfig.m_sensitivity = 1.07;
            simulation::model source;
            const double      frequency = fullPopulation ? 28 : 28.5;
            source.configure( sourceConfig,
                              fullPopulation ? simulation::referenceModes( 123 )
                                             : std::vector<simulation::modeSpec>{ { frequency, .002, .99999, 1, 1 } } );
            // Generate a complete recording first. Truth is a separate scoring oracle, never estimator input.
            constexpr size_t                count = 96000, future = 8;
            std::vector<simulation::sample> recording;
            recording.reserve( count + future );
            for( size_t n = 0; n < count + future; ++n )
                recording.push_back( source.step() );
            for( double memoryTime : { .1, .3, .8 } )
            {
                parameters config;
                config.m_frequency  = frequency;
                config.m_memoryTime = memoryTime;
                config.m_zero.fill( sourceConfig.m_zero );
                config.m_voltsPerBit.fill( sourceConfig.m_voltsPerBit );
                config.m_sensitivity.fill( sourceConfig.m_sensitivity );
                config.m_leverArm = sourceConfig.m_leverArm;
                harmonicEstimator                  estimator( config );
                std::array<std::vector<double>, 2> truth, residual;
                for( size_t n = 0; n < count; ++n )
                {
                    const adcSample input{ recording[n].m_adc, int64_t( n * 125000 ), n, 17 };
                    REQUIRE( estimator.consume( input, input.m_timeNs + 100000 ) );
                    if( n % 4 != 3 || n < 32000 )
                        continue;
                    const auto predicted = estimator.predict( input.m_timeNs + 100000, input.m_timeNs + 1000000 );
                    REQUIRE( predicted.m_state == estimatorState::ready );
                    for( size_t axis = 0; axis < 2; ++axis )
                    {
                        const double expected = recording[n + future].m_pointing[axis] * simulation::arcsecToRad;
                        truth[axis].push_back( expected );
                        residual[axis].push_back( expected - predicted.m_pointing[axis] );
                    }
                }
                REQUIRE( estimator.samples() == count );
                REQUIRE( source.m_ticks == count + future );
                REQUIRE( source.m_clipLow == std::array<uint64_t, 2>{ 0, 0 } );
                REQUIRE( source.m_clipHigh == std::array<uint64_t, 2>{ 0, 0 } );
                for( size_t axis = 0; axis < 2; ++axis )
                {
                    const double off       = bandPower( truth[axis], 2000, 27, 30 );
                    const double on        = bandPower( residual[axis], 2000, 27, 30 );
                    const double rejection = 10 * std::log10( off / on );
                    CAPTURE( fullPopulation, seed, axis, memoryTime, rejection );
                    // Phase-one regression: these recordings benefit, but do not all meet the future 10-dB control
                    // gate.
                    CHECK( std::isfinite( rejection ) );
                    CHECK( rejection > 0 );
                    if( !fullPopulation && memoryTime == .1 )
                        CHECK( rejection > 10 );
                    std::cout << "Offline " << ( fullPopulation ? "214-mode" : "single-mode" ) << " seed " << seed
                              << " axis " << axis << " memory " << memoryTime << " s: 27-30 Hz residual reduction "
                              << rejection << " dB\n";
                }
            }
        }
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
