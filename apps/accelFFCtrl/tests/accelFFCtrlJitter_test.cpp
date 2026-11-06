/** \file accelFFCtrlJitter_test.cpp
 * \brief Production gap-policy cadence qualification using actual irregular analytic acquisition times.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "accelFFCtrlGap_test.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_JITTER_FIXTURES
namespace jitterValidation
{
using namespace MagAOX::app::accelFFCtrlDetail;
constexpr int64_t epoch = 1700000000000000000, origin = 1000000000;
constexpr double  pi = 3.14159265358979323846;

/// Return deterministic bounded timestamp offsets; slopes model valid phase slewing, not corrupt clocks.
int64_t offset( unsigned profile /**< [in] synthetic profile index */,
                int64_t  n /**< [in] physical acquisition index */,
                int64_t  period /**< [in] nominal acquisition ns */ )
{
    if( profile == 1 )
        return ( n % 3 - 1 ) * period / 50;
    if( profile == 2 )
        return ( n % 3 - 1 ) * period / 10;
    const int64_t cycle = profile == 3 ? 40 : 16;
    const int64_t q = n % cycle, triangle = std::min( q, cycle - q ) - cycle / 4;
    return profile >= 3 ? triangle * period / ( profile == 3 ? 50 : 20 ) : 0;
}

/// Encode a source epoch stamp with integer precision.
timespec stamp( int64_t mono /**< [in] actual acquisition ns */ )
{
    return { ( epoch + mono ) / 1000000000, ( epoch + mono ) % 1000000000 };
}

/// Commit a true analytic sample at its jittered acquisition time, with an independent later receipt clock.
void publish( gapValidation::snapshot &source /**< [in,out] coherent depth-one source */,
              inputClock              &clock /**< [out] receipt clock */,
              uint64_t                 counter /**< [in] physical publication counter */,
              int64_t                  index /**< [in] nominal acquisition index */,
              int64_t                  period /**< [in] nominal ns */,
              unsigned                 profile /**< [in] timestamp-offset profile */ )
{
    const int64_t acquired = origin + index * period + offset( profile, index, period );
    const int64_t received = origin + index * period + period / 5 + 50000;
    const double  phase    = 2 * pi * 28 * ( acquired - origin ) * 1e-9 + .31;
    source.m_header        = { 2, 2, 1, 1, counter, 0, true, true, true, false };
    source.m_frame         = { { uint16_t( 2048 + std::round( 1000 * std::cos( phase ) ) ),
                                 uint16_t( 2048 + std::round( 700 * std::cos( phase - .8 ) ) ) },
                               stamp( acquired ),
                               stamp( acquired + 10000 ),
                               counter };
    clock                  = { epoch + received, received, 0 };
}
} // namespace jitterValidation
/// \endcond

/** \brief Guard acceptance and phase remain explicit across valid jitter, phase slewing and one missing frame.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl production timestamp-guard qualification", "[accelFFCtrl][accelFFCtrl-jitter]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::predict;
#endif
    // clang-format on
    using namespace jitterValidation;
    std::ofstream csv;
    if( const char *path = std::getenv( "ACCELFFCTRL_VALIDATION_DIR" ) )
    {
        std::filesystem::create_directories( path );
        csv.open( std::filesystem::path( path ) / "timestamp_guard.csv" );
        csv << "rate,profile,trials,guard_pass,admitted,rejected,invalid_records,max_gain_error_pct,max_phase_error_"
               "deg\n"
            << std::setprecision( 17 );
    }
    const std::array<const char *, 5> names{ "uniform", "jitter_2pct", "jitter_10pct", "slew_2pct", "slew_5pct" };
    for( int rate : { 8000, 4000 } )
        for( unsigned profile = 0; profile < names.size(); ++profile )
        {
            unsigned accepted = 0, expectedCount = 0;
            double   maxGain = 0, maxPhase = 0;
            for( int phaseIndex = 0; phaseIndex < 48; ++phaseIndex )
            {
                parameters config;
                config.m_frequency    = 28;
                config.m_sampleRate   = rate;
                config.m_predictionNs = 1000000;
                config.m_voltsPerBit.fill( 1e-6 );
                config.m_sensitivity.fill( 1 );
                inputProcessor strict( config ), optional( config, 256, 1000000, true );
                strict.attached();
                optional.attached();
                gapValidation::snapshot source;
                inputClock              clock;
                const int64_t           period = 1000000000 / rate;
                for( int n = 0; n <= rate / 2; ++n )
                {
                    publish( source, clock, n + 1, n + phaseIndex, period, profile );
                    strict.drain( source, [&clock] { return clock; } );
                    optional.drain( source, [&clock] { return clock; } );
                    if( n % ( rate / 2000 ) == 0 )
                    {
                        strict.predict( clock.m_mono );
                        optional.predict( clock.m_mono );
                    }
                }
                const auto before = optional.diagnostics();
                REQUIRE( before.m_prediction.m_state == estimatorState::ready );
                REQUIRE( before.m_resets == 1 );
                REQUIRE( before.m_invalid == 0 );
                const int64_t previous = rate / 2 + phaseIndex, index = previous + 2;
                const bool    expected =
                    std::abs( offset( profile, index, period ) - offset( profile, previous, period ) ) <= .05 * period;
                expectedCount += expected;
                publish( source, clock, rate / 2 + 3, index, period, profile );
                strict.drain( source, [&clock] { return clock; } );
                optional.drain( source, [&clock] { return clock; } );
                strict.predict( clock.m_mono );
                optional.predict( clock.m_mono );
                const auto &d = optional.diagnostics();
                const auto &p = d.m_prediction;
                CAPTURE( rate, profile, phaseIndex );
                REQUIRE( d.m_processed == uint64_t( rate / 2 + 2 ) );
                REQUIRE( d.m_dropped == 1 );
                REQUIRE( d.m_invalid == 0 ); // A failed cadence guard never discards a valid frame.
                REQUIRE( d.m_admittedGaps == uint64_t( expected ) );
                REQUIRE( d.m_rejectedGaps == uint64_t( !expected ) );
                REQUIRE( d.m_resets == uint64_t( 1 + !expected ) );
                REQUIRE( ( p.m_state == estimatorState::ready ) == expected );
                REQUIRE( strict.diagnostics().m_resets == 2 );
                REQUIRE( strict.diagnostics().m_prediction.m_state != estimatorState::ready );
                if( expected )
                {
                    ++accepted;
                    REQUIRE( p.m_sequence == source.m_frame.m_counter );
                    REQUIRE( p.m_targetTimeNs == clock.m_mono + config.m_predictionNs );
                    REQUIRE( p.m_sampleTimeNs == origin + index * period + offset( profile, index, period ) );
                    for( unsigned axis = 0; axis < 2; ++axis )
                    {
                        const double amplitude = axis ? .0007 : .001;
                        const double phase     = 2 * pi * 28 * ( p.m_sampleTimeNs - origin ) * 1e-9 + .31 - axis * .8;
                        const double error = std::abs( std::remainder( p.m_phase[axis] - phase, 2 * pi ) ) * 180 / pi;
                        maxGain            = std::max( maxGain, 100 * std::abs( p.m_amplitude[axis] / amplitude - 1 ) );
                        maxPhase           = std::max( maxPhase, error );
                        REQUIRE( error < .12 );
                        REQUIRE( p.m_amplitude[axis] == Approx( amplitude ).margin( 1e-6 ) );
                        const double targetPhase = 2 * pi * 28 * ( p.m_targetTimeNs - origin ) * 1e-9 + .31 - axis * .8;
                        REQUIRE( p.m_acceleration[axis] ==
                                 Approx( amplitude * std::cos( targetPhase ) ).margin( 1e-6 ) );
                    }
                }
            }
            CHECK( accepted == expectedCount );
            CHECK( expectedCount == ( profile == 2 ? 0 : profile == 4 ? 6 : 48 ) );
            std::cout << "Synthetic timestamp guard " << rate << " Hz " << names[profile] << ": " << accepted
                      << "/48 retained\n";
            if( csv.is_open() )
                csv << rate << ',' << names[profile] << ",48," << expectedCount << ',' << accepted << ','
                    << 48 - accepted << ",0," << ( accepted ? maxGain : NAN ) << ',' << ( accepted ? maxPhase : NAN )
                    << '\n';
        }
    if( csv.is_open() )
    {
        csv.close();
        REQUIRE( csv.good() );
    }
}
} // namespace accelFFCtrlTest
} // namespace libXWCTest
