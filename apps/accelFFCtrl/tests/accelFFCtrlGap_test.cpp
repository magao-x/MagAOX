/** \file accelFFCtrlGap_test.cpp
 * \brief Offline gap admission, timestamp integrity, freshness and unchanged cold-start regression tests.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "accelFFCtrlGap_test.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_GAP_CASE_FIXTURES
namespace gapValidation
{
constexpr int64_t epoch = 1700000000000000000, origin = 1000000000;

/// Encode exact synthetic realtime nanoseconds.
timespec timestamp( int64_t mono /**< [in] monotonic ns */ )
{
    return { ( epoch + mono ) / 1000000000, ( epoch + mono ) % 1000000000 };
}

/// Prepare fine calibration for separating gap phase behavior from ADC noise.
parameters settings( int rate /**< [in] configured nominal sample rate [Hz] */ )
{
    parameters config;
    config.m_sampleRate   = rate;
    config.m_frequency    = 28;
    config.m_predictionNs = 1000000;
    config.m_voltsPerBit.fill( 1e-6 );
    config.m_sensitivity.fill( 1 );
    return config;
}

/// Owned, coherent source and explicitly independent receipt clock.
struct fixture
{
    snapshot   m_source{ { 2, 2, 1, 1, 0, 0, true, true, true, false }, {} }; ///< Depth-one physical source.
    inputClock m_clock; ///< Receipt clock, independent of source stamp fault injection.

    /// Commit an analytic tone without calling the simulator or estimator.
    void publish( uint64_t counter /**< [in] physical publication counter */, int64_t mono /**< [in] actual ns */ );
};

void fixture::publish( uint64_t counter, int64_t mono )
{
    const double phase          = 6.28318530717958647692 * 28 * ( mono - origin ) * 1e-9 + .31;
    m_source.m_header.m_counter = counter;
    m_source.m_frame            = { { uint16_t( 2048 + std::round( 1000 * std::cos( phase ) ) ),
                                      uint16_t( 2048 + std::round( 700 * std::cos( phase - .8 ) ) ) },
                                    timestamp( mono ),
                                    timestamp( mono + 10000 ),
                                    counter };
    m_clock                     = { epoch + mono + 50000, mono + 50000, 0 };
}

/// Feed half a second, with actual 2-kHz queries, to require the original warmup before any retained gap.
template <typename observerT>
void warm( observerT &app /**< [in,out] offline observer */,
           fixture   &source /**< [in,out] analytic source */,
           int        rate /**< [in] acquisition rate [Hz] */ )
{
    const int64_t period = 1000000000 / rate;
    app.attached();
    for( int n = 0; n <= rate / 2; ++n )
    {
        source.publish( n + 1, origin + n * period );
        app.drain( source.m_source, [&source] { return source.m_clock; } );
        if( n % ( rate / 2000 ) == 0 )
        {
            app.predict( source.m_clock.m_mono );
            if( n * period < 300000000 )
                REQUIRE( app.diagnostics().m_prediction.m_state != estimatorState::ready );
        }
    }
    REQUIRE( app.diagnostics().m_prediction.m_state == estimatorState::ready );
}

/// Verify bounded loss uses real elapsed time and retains cosine phase, without bypassing numerical checks.
template <unsigned limit, typename observerT = observer<limit>>
void gapProbe( int rate /**< [in] acquisition rate [Hz] */, unsigned missing /**< [in] overwritten frames */ )
{
    const auto config = settings( rate );
    observerT  app( config );
    fixture    source;
    warm( app, source, rate );
    const auto    before = app.diagnostics();
    const int64_t period = 1000000000 / rate, acquired = origin + 500000000 + ( missing + 1 ) * period;
    source.publish( rate / 2 + missing + 2, acquired );
    app.drain( source.m_source, [&source] { return source.m_clock; } );
    app.predict( source.m_clock.m_mono );
    const auto after  = app.diagnostics();
    const bool retain = missing <= limit && ( missing + 1 ) * period <= 1000000;
    CAPTURE( rate, missing, limit );
    REQUIRE( after.m_resets == before.m_resets + !retain );
    REQUIRE( after.m_dropped == missing );
    REQUIRE( after.m_counter == source.m_source.m_header.m_counter );
    REQUIRE( ( after.m_prediction.m_state == estimatorState::ready ) == retain );
    REQUIRE( app.retained() == uint64_t( retain ) );
    if( retain )
    {
        const auto  &p     = after.m_prediction;
        const double phase = 6.28318530717958647692 * 28 * ( p.m_targetTimeNs - origin ) * 1e-9 + .31;
        REQUIRE( p.m_sampleTimeNs == acquired );
        REQUIRE( p.m_targetTimeNs == source.m_clock.m_mono + config.m_predictionNs );
        REQUIRE( p.m_rcond > .1 );
        REQUIRE( p.m_amplitude[0] == Approx( .001 ).margin( 1e-6 ) );
        REQUIRE( p.m_acceleration[0] == Approx( .001 * std::cos( phase ) ).margin( 1e-6 ) );
        REQUIRE( std::abs( std::remainder( p.m_phase[0] - ( phase - 6.28318530717958647692 * 28 * .00105 ),
                                           6.28318530717958647692 ) ) < .002 );
    }
}
} // namespace gapValidation
/// \endcond

/** \brief Isolated and consecutive gaps preserve the real timestamp phase only inside the explicit admission bounds.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline bounded-gap phase and admission", "[accelFFCtrl][gap]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::consume;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::predict;
#endif
    // clang-format on
    for( int rate : { 8000, 4000 } )
        for( unsigned missing : { 1, 2, 3, 7, 8 } )
        {
            gapValidation::gapProbe<1>( rate, missing );
            gapValidation::gapProbe<7>( rate, missing );
            gapValidation::gapProbe<1, gapValidation::productionObserver>( rate, missing );
        }
}

/** \brief Source/time/ADC faults invalidate even a mature experimental fit; no stale numerical output remains ready.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline gap policy rejects integrity faults", "[accelFFCtrl][gap]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::attached;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::predict;
#endif
    // clang-format on
    using namespace gapValidation;
    for( int rate : { 8000, 4000 } )
        for( const std::string fault : { "backward",
                                         "forward",
                                         "rollback",
                                         "restart",
                                         "adc",
                                         "zero",
                                         "future",
                                         "write",
                                         "clock",
                                         "stale",
                                         "bracket",
                                         "gap_time",
                                         "backward_gap",
                                         "geometry" } )
        {
            const auto exercise = [rate, &fault]( auto &app )
            {
                fixture source;
                warm( app, source, rate );
                const auto    before = app.diagnostics();
                const int64_t period = 1000000000 / rate, acquired = origin + 500000000 + period;
                source.publish( rate / 2 + 2, acquired );
                if( fault == "backward" )
                    source.m_source.m_frame.m_acquired = timestamp( acquired - 2 * period );
                if( fault == "forward" )
                    source.m_source.m_frame.m_acquired = timestamp( acquired + period );
                if( fault == "rollback" || fault == "restart" )
                    source.m_source.m_header.m_counter = source.m_source.m_frame.m_counter = 1;
                if( fault == "restart" )
                    app.attached();
                if( fault == "adc" )
                    source.m_source.m_frame.m_adc[0] = 4096;
                if( fault == "zero" )
                    source.m_source.m_frame.m_acquired = {};
                if( fault == "future" )
                    source.m_source.m_frame.m_acquired = timestamp( acquired + 5000000 );
                if( fault == "write" )
                    source.m_source.m_frame.m_written = timestamp( acquired - 1 );
                if( fault == "clock" )
                    source.m_clock.m_real += 2000000;
                if( fault == "stale" )
                {
                    source.m_clock.m_real += 3000000;
                    source.m_clock.m_mono += 3000000;
                }
                if( fault == "bracket" )
                    source.m_clock.m_uncertainty = 1000001;
                if( fault == "geometry" )
                    source.m_source.m_header.m_uint16 = false;
                if( fault == "gap_time" || fault == "backward_gap" )
                {
                    ++source.m_source.m_header.m_counter;
                    ++source.m_source.m_frame.m_counter;
                    if( fault == "backward_gap" )
                        source.m_source.m_frame.m_acquired = timestamp( acquired - 2 * period );
                }
                app.drain( source.m_source, [&source] { return source.m_clock; } );
                REQUIRE( app.diagnostics().m_prediction.m_state != estimatorState::ready );
                app.predict( source.m_clock.m_mono );
                CAPTURE( rate, fault );
                const auto after = app.diagnostics();
                REQUIRE( after.m_resets > before.m_resets );
                REQUIRE( after.m_prediction.m_state != estimatorState::ready );
                REQUIRE( after.m_prediction.m_acceleration == std::array<double, 2>{ 0, 0 } );
                REQUIRE( after.m_prediction.m_pointing == std::array<double, 2>{ 0, 0 } );
                REQUIRE( app.retained() == 0 );
            };
            observer<7> prototype( settings( rate ) );
            exercise( prototype );
            productionObserver production( settings( rate ) );
            exercise( production );
        }
}

/** \brief Dense losses, silence and wrong rate assumptions cannot bypass cold start or freshness checks.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl offline gap density rate and freshness bounds", "[accelFFCtrl][gap]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::harmonicEstimator::predict;
#endif
    // clang-format on
    using namespace gapValidation;
    for( int rate : { 8000, 4000 } )
    {
        const int64_t period = 1000000000 / rate;
        fixture       source;
        observer<7>   app( settings( rate ) );
        warm( app, source, rate );
        for( int gap = 1; gap <= 2; ++gap )
        {
            source.publish( rate / 2 + 1 + 2 * gap, origin + 500000000 + 2 * gap * period );
            app.drain( source.m_source, [&source] { return source.m_clock; } );
            app.predict( source.m_clock.m_mono );
            REQUIRE( ( app.diagnostics().m_prediction.m_state == estimatorState::ready ) == ( gap == 1 ) );
        }
        REQUIRE( app.retained() == 1 );
        warm( app, source, rate );
        app.predict( origin + 500000000 + 2000001 );
        REQUIRE( app.diagnostics().m_prediction.m_state == estimatorState::stale );
        source.publish( rate / 2 + 31, origin + 500000000 + 30 * period );
        app.drain( source.m_source, [&source] { return source.m_clock; } );
        app.predict( source.m_clock.m_mono );
        REQUIRE( app.diagnostics().m_prediction.m_state != estimatorState::ready );

        observer<7> wrong( settings( rate == 8000 ? 4000 : 8000 ) );
        wrong.attached();
        for( int n = 0; n < rate; ++n )
        {
            source.publish( n + 1, origin + n * period );
            wrong.drain( source.m_source, [&source] { return source.m_clock; } );
            wrong.predict( source.m_clock.m_mono );
            REQUIRE( wrong.diagnostics().m_prediction.m_state != estimatorState::ready );
        }
        REQUIRE( wrong.diagnostics().m_invalid > 0 );
    }
    auto invalid      = settings( 8000 );
    invalid.m_zero[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS( observer<1>( invalid ), std::invalid_argument );
}

/** \brief Production opt-in retains physical identity, rejects dense gaps, and preserves configuration cold starts.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl production single-gap lifecycle and accounting", "[accelFFCtrl][gap]" )
{
    // clang-format off
#ifdef ACCELFFCTRL_TEST_DOXYGEN_REF
    MagAOX::app::accelFFCtrlDetail::inputProcessor::drain;
    MagAOX::app::accelFFCtrlDetail::inputProcessor::configure;
    MagAOX::app::accelFFCtrlDetail::singleFrameGapSetting;
#endif
    // clang-format on
    using namespace gapValidation;
    bool flag = false;
    REQUIRE( singleFrameGapSetting( flag, "true" ) );
    REQUIRE( flag );
    for( const std::string text : { "yes", "truse", "2", "", "TRUE" } )
    {
        REQUIRE_FALSE( singleFrameGapSetting( flag, text ) );
        REQUIRE( flag );
    }
    REQUIRE( singleFrameGapSetting( flag, "false" ) );
    REQUIRE_FALSE( flag );
    REQUIRE( singleFrameGapSetting( flag, "1" ) );
    REQUIRE( flag );
    REQUIRE( singleFrameGapSetting( flag, "0" ) );
    REQUIRE_FALSE( flag );
    for( int rate : { 8000, 4000 } )
    {
        const auto     config = settings( rate );
        inputProcessor defaults( config );
        REQUIRE_FALSE( defaults.diagnostics().m_allowSingleFrameGap );
        productionObserver app( config );
        fixture            source;
        warm( app, source, rate );
        REQUIRE( app.diagnostics().m_allowSingleFrameGap );
        const int64_t period = 1000000000 / rate;
        for( int gap = 1; gap <= 3; ++gap )
        {
            source.publish( rate / 2 + 1 + 2 * gap, origin + 500000000 + 2 * gap * period );
            app.drain( source.m_source, [&source] { return source.m_clock; } );
            REQUIRE( app.diagnostics().m_prediction.m_state != estimatorState::ready );
            app.predict( source.m_clock.m_mono );
            const auto &d = app.diagnostics();
            REQUIRE( d.m_dropped == uint64_t( gap ) );
            REQUIRE( d.m_admittedGaps == 1 );
            REQUIRE( d.m_rejectedGaps == uint64_t( gap - 1 ) );
            REQUIRE( d.m_resets == uint64_t( gap ) );
            REQUIRE( ( d.m_prediction.m_state == estimatorState::ready ) == ( gap == 1 ) );
            REQUIRE( d.m_prediction.m_sequence == source.m_source.m_frame.m_counter );
            if( gap == 1 )
                REQUIRE( d.m_sampleRate == Approx( rate ).epsilon( 1e-10 ) );
        }
        const auto before = app.diagnostics();
        app.configure( config, 37 );
        REQUIRE( app.diagnostics().m_resets == before.m_resets + 1 );
        REQUIRE( app.diagnostics().m_admittedGaps == 1 );
        REQUIRE( app.diagnostics().m_rejectedGaps == 2 );
        REQUIRE( app.diagnostics().m_allowSingleFrameGap );
        REQUIRE( app.diagnostics().m_prediction.m_state != estimatorState::ready );

        // A stale interval cannot be hidden merely because no prediction query ran during the delay.
        productionObserver delayed( config );
        warm( delayed, source, rate );
        source.publish( rate / 2 + 3, origin + 500000000 + 2 * period );
        source.m_clock.m_real += 2000000 - period - 50000;
        source.m_clock.m_mono += 2000000 - period - 50000;
        delayed.drain( source.m_source, [&source] { return source.m_clock; } );
        delayed.predict( source.m_clock.m_mono );
        REQUIRE( delayed.diagnostics().m_admittedGaps == 0 );
        REQUIRE( delayed.diagnostics().m_rejectedGaps == 1 );
        REQUIRE( delayed.diagnostics().m_invalid == 0 );
        REQUIRE( delayed.diagnostics().m_processed == uint64_t( rate / 2 + 2 ) );
        REQUIRE( delayed.diagnostics().m_prediction.m_state != estimatorState::ready );

        // Nominal rate is not a substitute for timestamps; a wrong value rejects gap retention only.
        productionObserver wrongRate( settings( rate == 8000 ? 4000 : 8000 ) );
        warm( wrongRate, source, rate );
        REQUIRE( wrongRate.diagnostics().m_prediction.m_amplitude[0] == Approx( .001 ).margin( 1e-6 ) );
        source.publish( rate / 2 + 3, origin + 500000000 + 2 * period );
        wrongRate.drain( source.m_source, [&source] { return source.m_clock; } );
        wrongRate.predict( source.m_clock.m_mono );
        REQUIRE( wrongRate.diagnostics().m_admittedGaps == 0 );
        REQUIRE( wrongRate.diagnostics().m_rejectedGaps == 1 );
        REQUIRE( wrongRate.diagnostics().m_invalid == 0 );
        REQUIRE( wrongRate.diagnostics().m_prediction.m_state != estimatorState::ready );
    }
}
} // namespace accelFFCtrlTest
} // namespace libXWCTest
