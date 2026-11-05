/** \file accelFFCtrlInput_test.cpp
 * \brief Deterministic retained-stream ingestion, time translation, discontinuity and configuration tests.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlInput.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_INPUT_FIXTURE
constexpr int64_t realEpoch = 1700000000000000000;

/// Build a source timestamp in the simulated realtime domain.
timespec sourceTime( int64_t mono /**< [in] simulated monotonic ns */ )
{
    const int64_t ns = realEpoch + mono;
    return { ns / 1000000000, ns % 1000000000 };
}

/// In-memory publication fixture; injects a write during copying without undefined concurrent C++ accesses.
struct retainedSource
{
    inputHeader             m_header;        ///< Mutable publication metadata.
    std::vector<inputFrame> m_frames;        ///< Producer temporal storage, deliberately overwritten on wrap.
    bool                    m_race{ false }; ///< Change publication state in the middle of a consumer copy.

    /// Prepare synthetic temporal geometry and storage.
    explicit retainedSource( size_t depth /**< [in] retained pairs */ );

    /// Commit one complete synthetic pair as frameGrabber does, with per-plane timestamps/counters.
    void publish( uint64_t                counter /**< [in] publication identity */,
                  int64_t                 mono /**< [in] acquisition ns */,
                  std::array<uint16_t, 2> codes = { 2048, 2048 } /**< [in] raw pair */ );

    /// Return a value snapshot of metadata.
    inputHeader header() const;

    /// Copy one retained pair and optionally simulate an overlapping producer update.
    inputFrame frame( size_t index /**< [in] source plane */ );
};

retainedSource::retainedSource( size_t depth ) : m_frames( depth )
{
    m_header = { 3, 2, 1, uint32_t( depth ), 0, depth - 1, true, true, true, false };
}

void retainedSource::publish( uint64_t counter, int64_t mono, std::array<uint16_t, 2> codes )
{
    m_header.m_counter         = counter;
    m_header.m_index           = ( counter - 1 ) % m_frames.size();
    m_frames[m_header.m_index] = { codes, sourceTime( mono ), sourceTime( mono + 10000 ), counter };
}

inputHeader retainedSource::header() const
{
    return m_header;
}

inputFrame retainedSource::frame( size_t index )
{
    const auto value = m_frames.at( index );
    if( m_race )
        m_header.m_write = true;
    return value;
}

/// Poll the production reader with an independent, explicitly timed receipt clock.
size_t drainAt( inputProcessor &processor /**< [in,out] observer */,
                retainedSource &source /**< [in,out] synthetic stream */,
                int64_t         mono /**< [in] actual receipt ns */ )
{
    return processor.drain( source, [mono] { return inputClock{ realEpoch + mono, mono, 10 }; } );
}
/// \endcond

/** \brief Every retained pair is ingested once, with raw decoding, per-plane times, prediction age and bounded queries.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl retained 8k stream decoding and timestamp translation", "[accelFFCtrl][input]" )
{
    parameters config;
    config.m_zero         = { 2017, 2031 };
    config.m_voltsPerBit  = { 1e-6, 1e-6 };
    config.m_sensitivity  = { 1, 1 };
    config.m_predictionNs = 1000000;
    inputProcessor processor( config );
    retainedSource source( 256 );
    processor.attached();
    for( uint64_t n = 1; n <= 12000; ++n )
    {
        const int64_t acquired = 1000000000 + n * 125000;
        const double  phase    = 6.28318530717958647692 * 28.5 * ( n / 8000. );
        source.publish( n,
                        acquired,
                        { uint16_t( 2017 + std::round( 1000 * std::cos( phase ) ) ),
                          uint16_t( 2031 + std::round( 700 * std::sin( phase ) ) ) } );
        // First frame binds the source. Later wakes intentionally batch four committed publications.
        if( n != 1 && n % 4 != 0 )
            continue;
        const int64_t receipt = acquired + 50000;
        REQUIRE( drainAt( processor, source, receipt ) > 0 );
        processor.predict( receipt );
        const auto &stats = processor.diagnostics();
        REQUIRE( stats.m_processed == n );
        REQUIRE( stats.m_dropped == 0 );
        if( n > 8000 )
        {
            REQUIRE( stats.m_prediction.m_state == estimatorState::ready );
            const double target = ( n * 125000 + 1050000 ) * 1e-9;
            REQUIRE( stats.m_prediction.m_acceleration[0] ==
                     Approx( .001 * std::cos( 6.28318530717958647692 * 28.5 * target ) ).margin( 2e-7 ) );
            REQUIRE( stats.m_ageUs == Approx( 50 ) );
            REQUIRE( stats.m_horizonUs == Approx( 1050 ) );
            REQUIRE( stats.m_writeDelayUs == Approx( 10 ) );
            REQUIRE( stats.m_rate == Approx( 8000 ) );
        }
    }
    const auto before = processor.diagnostics().m_processed;
    REQUIRE( drainAt( processor, source, 2500100000 ) == 0 );
    REQUIRE( processor.diagnostics().m_processed == before );
    processor.predict( 2503000000 );
    REQUIRE( processor.diagnostics().m_state == inputState::stale );
    REQUIRE( processor.diagnostics().m_prediction.m_pointing == std::array<double, 2>{ 0, 0 } );
}

/** \brief A one-frame producer loses overwritten samples; larger retention recovers them and counts finite overrun
 * exactly.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl depth one loss and ring recovery", "[accelFFCtrl][input]" )
{
    for( size_t depth : { 1, 4, 256 } )
    {
        inputProcessor processor;
        retainedSource source( depth );
        processor.attached();
        source.publish( 1, 1000000000 );
        REQUIRE( drainAt( processor, source, 1000050000 ) == 1 );
        for( uint64_t counter = 2; counter <= 5; ++counter )
            source.publish( counter, 1000000000 + ( counter - 1 ) * 125000 );
        REQUIRE( drainAt( processor, source, 1000550000 ) == std::min<size_t>( 4, depth ) );
        REQUIRE( processor.diagnostics().m_processed == 1 + std::min<size_t>( 4, depth ) );
        REQUIRE( processor.diagnostics().m_dropped == ( depth == 1 ? 3 : 0 ) );
        if( depth == 4 )
        {
            for( uint64_t counter = 6; counter <= 11; ++counter )
                source.publish( counter, 1000000000 + ( counter - 1 ) * 125000 );
            REQUIRE( drainAt( processor, source, 1001300000 ) == 4 );
            REQUIRE( processor.diagnostics().m_dropped == 2 );
        }
    }
    inputProcessor bounded( parameters{}, 2 );
    retainedSource source( 8 );
    bounded.attached();
    source.publish( 1, 1000000000 );
    drainAt( bounded, source, 1000050000 );
    for( uint64_t n = 2; n <= 6; ++n )
        source.publish( n, 1000000000 + ( n - 1 ) * 125000 );
    REQUIRE( drainAt( bounded, source, 1000700000 ) == 2 );
    REQUIRE( drainAt( bounded, source, 1000700000 ) == 2 );
    REQUIRE( drainAt( bounded, source, 1000700000 ) == 1 );
    REQUIRE( bounded.diagnostics().m_processed == 6 );
    REQUIRE( bounded.diagnostics().m_dropped == 0 );
}

/** \brief Busy/torn copies and inconsistent counter arrays never advance the consumer cursor or fit.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl source snapshot races and validation", "[accelFFCtrl][input]" )
{
    inputProcessor processor;
    retainedSource source( 8 );
    processor.attached();
    source.publish( 1, 1000000000 );
    source.m_race = true;
    REQUIRE( drainAt( processor, source, 1000050000 ) == 0 );
    REQUIRE( processor.diagnostics().m_races == 1 );
    REQUIRE( processor.diagnostics().m_processed == 0 );
    REQUIRE( drainAt( processor, source, 1000050000 ) == 0 );
    REQUIRE( processor.diagnostics().m_busy == 1 );
    source.m_race = source.m_header.m_write = false;
    REQUIRE( drainAt( processor, source, 1000050000 ) == 1 );
    source.publish( 2, 1000125000 );
    source.m_frames[source.m_header.m_index].m_counter = 0;
    REQUIRE( drainAt( processor, source, 1000175000 ) == 0 );
    source.m_frames[source.m_header.m_index].m_counter = 2;
    REQUIRE( drainAt( processor, source, 1000175000 ) == 1 );
    for( int invalid = 0; invalid < 5; ++invalid )
    {
        auto bad = source;
        if( invalid == 0 )
            bad.m_header.m_uint16 = false;
        if( invalid == 1 )
            bad.m_header.m_width = 3;
        if( invalid == 2 )
            bad.m_header.m_cpu = false;
        if( invalid == 3 )
            bad.m_header.m_index = 8;
        if( invalid == 4 )
            bad.m_header.m_valid = false;
        REQUIRE( drainAt( processor, bad, 1000200000 ) == 0 );
        REQUIRE( processor.diagnostics().m_state == inputState::incompatible );
        REQUIRE( processor.diagnostics().m_prediction.m_pointing == std::array<double, 2>{ 0, 0 } );
    }
}

/** \brief Restarts, clock steps, invalid/backwards/future/stale acquisition times invalidate and rewarm observation.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl stream generation and timestamp discontinuities", "[accelFFCtrl][input]" )
{
    inputProcessor processor;
    retainedSource source( 4 );
    processor.attached();
    source.publish( 10, 1000000000 );
    drainAt( processor, source, 1000050000 );
    source.publish( 1, 1000125000 );
    REQUIRE( drainAt( processor, source, 1000175000 ) == 1 );
    REQUIRE( processor.diagnostics().m_rollbacks == 1 );
    REQUIRE( processor.diagnostics().m_generation == 2 );
    processor.disconnected();
    REQUIRE( processor.diagnostics().m_state == inputState::disconnected );
    processor.attached();
    REQUIRE( processor.diagnostics().m_generation == 3 );
    for( int fault = 0; fault < 6; ++fault )
    {
        processor.attached();
        source.publish( 1, 1000000000 );
        drainAt( processor, source, 1000050000 );
        source.publish( 2, 1000125000 );
        auto   &frame   = source.m_frames[source.m_header.m_index];
        int64_t receipt = 1000175000;
        if( fault == 0 )
            frame.m_acquired = {};
        if( fault == 1 )
            frame.m_acquired.tv_nsec = 1000000000;
        if( fault == 2 )
            frame.m_acquired = sourceTime( 999999999 );
        if( fault == 3 )
        {
            frame.m_acquired = sourceTime( 1000300000 );
            frame.m_written  = sourceTime( 1000310000 );
        }
        if( fault == 4 )
            receipt += 3000000;
        if( fault == 5 )
            frame.m_adc[0] = 4096;
        drainAt( processor, source, receipt );
        REQUIRE( processor.diagnostics().m_invalid > 0 );
        REQUIRE( processor.diagnostics().m_prediction.m_pointing == std::array<double, 2>{ 0, 0 } );
    }
    processor.attached();
    source.publish( 1, 2000000000 );
    drainAt( processor, source, 2000050000 );
    source.publish( 2, 2000125000 );
    REQUIRE( processor.drain( source, [] { return inputClock{ realEpoch + 2000175000 + 2000000, 2000175000, 10 }; } ) ==
             0 );
    REQUIRE( processor.diagnostics().m_state == inputState::clockFault );
    REQUIRE( drainAt( processor, source, 2000175000 ) == 1 );
}

/** \brief Every exposed calibration/control change is validated as a whole and applied with a fit-reset revision.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl live configuration and explicit reset", "[accelFFCtrl][input]" )
{
    parameters config;
    for( size_t n = 0; n < parameterNames.size(); ++n )
    {
        const double current = parameterValue( config, n );
        REQUIRE( parameterUpdate( config, n, current ) );
        REQUIRE_FALSE( parameterUpdate( config, n, NAN ) );
        REQUIRE( parameterValue( config, n ) == current );
    }
    REQUIRE_FALSE( parameterUpdate( config, 0, -1 ) );
    REQUIRE_FALSE( parameterUpdate( config, 1, 0 ) );
    REQUIRE_FALSE( parameterUpdate( config, 2, 0 ) );
    REQUIRE_FALSE( parameterUpdate( config, 4, 51 ) );
    REQUIRE_FALSE( parameterUpdate( config, 7, -1 ) );
    for( const std::string text : { "nan", "inf", "", "bad", "23x", "1e999" } )
    {
        REQUIRE_FALSE( parameterUpdate( config, 5, text ) );
        REQUIRE( config.m_zero[0] == 2048 );
    }
    REQUIRE( parameterUpdate( config, 5, std::string( " +2048 \t" ) ) );
    REQUIRE( parameterUpdate( config, 0, 29 ) );
    REQUIRE( parameterUpdate( config, 4, 2.5 ) );
    inputProcessor processor;
    processor.attached();
    processor.configure( config, 17 );
    REQUIRE( processor.diagnostics().m_revision == 17 );
    REQUIRE( processor.diagnostics().m_config.m_frequency == 29 );
    REQUIRE( processor.diagnostics().m_config.m_predictionNs == 2500000 );
    const auto resets = processor.diagnostics().m_resets;
    processor.reset();
    REQUIRE( processor.diagnostics().m_resets == resets + 1 );
    REQUIRE( processor.diagnostics().m_prediction.m_state == estimatorState::warming );
    REQUIRE( inputNameValid( "acc000" ) );
    REQUIRE( inputNameValid( "accffctrl-private" ) );
    for( const std::string name : { "dm01disp10", "dm01disp", "", "../acc000", "/tmp/acc000" } )
        REQUIRE_FALSE( inputNameValid( name ) );
}

/** \brief Bad clock brackets never become timestamps; valid input recovers without reusing an invalid fit.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl clock bracket and timespec bounds", "[accelFFCtrl][input]" )
{
    int64_t converted;
    REQUIRE_FALSE( inputProcessor::timeNs( {}, converted ) );
    REQUIRE_FALSE( inputProcessor::timeNs( { -1, 0 }, converted ) );
    REQUIRE_FALSE( inputProcessor::timeNs( { 1, -1 }, converted ) );
    REQUIRE_FALSE( inputProcessor::timeNs( { std::numeric_limits<time_t>::max(), 0 }, converted ) );
    REQUIRE( inputProcessor::timeNs( { 1700000000, 123 }, converted ) );
    REQUIRE( converted == realEpoch + 123 );
    for( const inputClock clock : { inputClock{ 0, 1000050000, 0 },
                                    inputClock{ realEpoch, -1, 0 },
                                    inputClock{ realEpoch, 1000050000, -1 },
                                    inputClock{ realEpoch, 1000050000, 1000001 } } )
    {
        inputProcessor processor;
        retainedSource source( 8 );
        processor.attached();
        source.publish( 1, 1000000000 );
        REQUIRE( processor.drain( source, [clock] { return clock; } ) == 0 );
        REQUIRE( processor.diagnostics().m_state == inputState::clockFault );
        REQUIRE( processor.diagnostics().m_processed == 0 );
        REQUIRE( drainAt( processor, source, 1000050000 ) == 1 );
    }
}

/** \brief Accepted-sample throughput includes gaps even when every overwritten-frame event resets the fit.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl effective input throughput exposes loss", "[accelFFCtrl][input]" )
{
    inputProcessor processor;
    retainedSource source( 1 );
    processor.attached();
    for( uint64_t n = 1; n <= 8000; ++n )
    {
        const int64_t acquired = 1000000000 + n * 125000;
        source.publish( n, acquired );
        if( n != 1 && n % 2 )
            continue;
        drainAt( processor, source, acquired + 50000 );
        processor.predict( acquired + 50000 );
    }
    REQUIRE( processor.diagnostics().m_processed == 4001 );
    REQUIRE( processor.diagnostics().m_dropped == 3999 );
    REQUIRE( processor.diagnostics().m_rate == Approx( 4000 ).margin( 4 ) );
    processor.predict( 2300000000 );
    processor.predict( 2600000000 );
    REQUIRE( processor.diagnostics().m_rate == 0 );
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
