/** \file ddspc_adc_test.cpp
 * \brief Standalone actual ADC calibration, timestamp retention and gap recovery without native transport.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../adc_input.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

/// \cond ADC_CAPTURE_FIXTURE
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
/// Deterministic owned temporal source with controllable publication and clock faults.
class AdcFixture : public DDSPC::AdcSource
{
  public:
    DDSPC::AdcHeader m_header{ 3, 2, 1, 8, 0, 0, true, true, true, false }; ///< Fixed valid layout by default.
    DDSPC::AdcClock  m_clock;                     ///< Local clock observation, read after the stable copy.
    std::vector<DDSPC::AdcFrame> m_frames{ 8 };   ///< Fake producer retention, separate from consumer retention.
    bool                         m_race{ false }; ///< Change the final bracket only, without a concurrent data race.
    mutable unsigned             m_headers{ 0 }, m_reads{ 0 }; ///< Observe bounded capture operations.

    /// Copy the fake metadata, optionally invalidating the closing bracket.
    DDSPC::AdcHeader header() const override;

    /// Copy a fake producer plane.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Retained plane. */ ) const override;

    /// Return the declared local clock observation.
    DDSPC::AdcClock clock() const override;

    /// Publish one fake raw pair with explicit acquisition time.
    void publish( std::uint64_t                counter /**< [in] Producer identity. */,
                  std::int64_t                 acquired /**< [in] Monotonic acquisition ns. */,
                  std::array<std::uint16_t, 2> raw = { 2049, 2047 } /**< [in] Ordered channels. */ );
};

DDSPC::AdcHeader AdcFixture::header() const
{
    auto result = m_header;
    if( m_race && ++m_headers % 2 == 0 )
    {
        ++result.m_counter;
    }
    return result;
}

DDSPC::AdcFrame AdcFixture::frame( std::size_t index ) const
{
    ++m_reads;
    return m_frames.at( index );
}

DDSPC::AdcClock AdcFixture::clock() const
{
    return m_clock;
}

void AdcFixture::publish( std::uint64_t counter, std::int64_t acquired, std::array<std::uint16_t, 2> raw )
{
    m_header.m_counter         = counter;
    m_header.m_index           = ( counter - 1 ) % m_header.m_depth;
    m_frames[m_header.m_index] = { raw, counter, acquired + 10000000000LL, acquired + 10000001000LL };
    m_clock                    = { acquired + 10000010000LL, acquired + 10000, 10 };
}

} // namespace
/// \endcond

/** \brief Raw UINT16 decoding preserves channel order, unsigned subtraction, configurable ranges and polarity. */
TEST_CASE( "ADC capture decodes accelSim and explicit custom UINT16 profiles", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::validate_adc_config;
#endif
    // clang-format on
    DDSPC::AdcConfig  config;
    DDSPC::AdcCapture capture( config );
    capture.attached( 2, 3 );
    AdcFixture source;
    source.publish( 1, 1000000 );
    REQUIRE( capture.drain( source ) == 1 );
    const auto sample = capture.samples().at( 0 );
    REQUIRE( sample.m_frame.m_raw == std::array<std::uint16_t, 2>{ 2049, 2047 } );
    REQUIRE( sample.m_acceleration[0] == Approx( .0083799 / 1.02 ) );
    REQUIRE( sample.m_acceleration[1] == Approx( -.0083799 / 1.02 ) );
    REQUIRE( sample.m_frame.m_acquiredRealNs == 10001000000LL );
    REQUIRE( sample.m_acquiredNs == 1000000 );
    REQUIRE( sample.m_publishedNs == 1001000 );
    REQUIRE( sample.m_receivedNs == 1010000 );
    config.m_profile       = "custom";
    config.m_maxCode       = 65535;
    config.m_zero          = { 32768, 100 };
    config.m_voltsPerCount = { .01, .02 };
    config.m_sensitivity   = { 2, 4 };
    config.m_polarity      = { 1, -1 };
    DDSPC::AdcCapture custom( config );
    custom.attached( 1, 1 );
    source.publish( 1, 1000000, { 60000, 99 } );
    REQUIRE( custom.drain( source ) == 1 );
    REQUIRE( custom.status().m_latest.m_acceleration[0] == Approx( 136.16 ) );
    REQUIRE( custom.status().m_latest.m_acceleration[1] == Approx( .005 ) );
    for( const double invalid :
         { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        auto bad             = config;
        bad.m_sensitivity[0] = invalid;
        REQUIRE_THROWS_AS( DDSPC::AdcCapture( bad ), std::invalid_argument );
    }
    config.m_capacity = 0;
    REQUIRE_THROWS_AS( DDSPC::AdcCapture( config ), std::invalid_argument );
    config.m_capacity = 4;
    config.m_maxCode  = 65536;
    REQUIRE_THROWS_AS( DDSPC::AdcCapture( config ), std::invalid_argument );
    config.m_maxCode = 4095;
    config.m_stream  = "dm01disp10";
    REQUIRE_THROWS_AS( DDSPC::AdcCapture( config ), std::invalid_argument );
}

/** \brief Integer epoch mapping preserves individual nanoseconds at present-day realtime clock magnitudes. */
TEST_CASE( "ADC timestamp mapping retains nanoseconds beyond floating point epoch precision", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
#endif
    // clang-format on
    constexpr std::int64_t epoch = 1800000000000000000LL;
    DDSPC::AdcCapture      capture( DDSPC::AdcConfig{} );
    capture.attached( 1, 1 );
    AdcFixture source;
    for( std::uint64_t n = 1; n <= 2; ++n )
    {
        const std::int64_t acquired = 1000001 + std::int64_t( n - 1 ) * 125001;
        source.publish( n, acquired );
        auto &frame            = source.m_frames[source.m_header.m_index];
        frame.m_acquiredRealNs = epoch + acquired;
        frame.m_writtenRealNs  = epoch + acquired + 11;
        source.m_clock         = { epoch + acquired + 37, acquired + 37, 1 };
        REQUIRE( capture.drain( source ) == 1 );
        REQUIRE( capture.status().m_latest.m_acquiredNs == acquired );
        REQUIRE( capture.status().m_latest.m_publishedNs == acquired + 11 );
        REQUIRE( capture.status().m_ageUs == Approx( .037 ) );
    }
    REQUIRE( capture.status().m_rateHz == Approx( 1e9 / 125001 ) );
    REQUIRE( capture.samples()[1].m_acquiredNs - capture.samples()[0].m_acquiredNs == 125001 );
}

/** \brief Consumer retention wraps in order, recovers retained producer planes and counts only observed losses. */
TEST_CASE( "ADC capture retains bounded contiguous history and detects missing publications", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::samples;
    DDSPC::AdcCapture::drain;
#endif
    // clang-format on
    DDSPC::AdcConfig config;
    config.m_capacity = 3;
    DDSPC::AdcCapture capture( config );
    capture.attached( 1, 2 );
    AdcFixture source;
    source.m_header.m_depth = 4;
    for( std::uint64_t n = 1; n <= 6; ++n )
    {
        source.publish( n, n * 125000 );
        REQUIRE( capture.drain( source ) == 1 );
    }
    REQUIRE( capture.status().m_rateHz == Approx( 8000 ) );
    REQUIRE( capture.status().m_overwrites == 3 );
    REQUIRE( capture.samples().front().m_frame.m_counter == 4 );
    REQUIRE( capture.samples().back().m_frame.m_counter == 6 );
    REQUIRE( capture.drain( source ) == 0 );
    REQUIRE( capture.status().m_duplicates == 1 );
    for( std::uint64_t n = 7; n <= 13; ++n )
    {
        source.publish( n, n * 125000 );
    }
    REQUIRE( capture.drain( source ) == 3 );
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_dropped == 3 ); // Publications 7,8,9 were overwritten by the producer.
    REQUIRE( capture.samples().front().m_frame.m_counter == 11 );
    REQUIRE( capture.samples().back().m_frame.m_counter == 13 );
    REQUIRE( capture.status().m_source.m_depth == 4 );
    REQUIRE( capture.status().m_latest.m_segment > 1 );
    source.publish( 14, 4000000 ); // The producer skipped nominal ticks without skipping a published counter.
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_timingGaps == 1 );
    REQUIRE( capture.samples().size() == 1 );
}

/** \brief Invalid codes/times, unstable copies and silence cannot leave a usable retained segment. */
TEST_CASE( "ADC capture rejects corrupt metadata values and stale samples", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
    DDSPC::AdcCapture::tick;
#endif
    // clang-format on
    for( const auto codes : { std::array<std::uint16_t, 2>{ 0, 2048 }, { 4095, 2048 }, { 4096, 2048 } } )
    {
        DDSPC::AdcCapture capture( DDSPC::AdcConfig{} );
        capture.attached( 1, 1 );
        AdcFixture source;
        source.publish( 1, 1000000, codes );
        REQUIRE( capture.drain( source ) == 0 );
        REQUIRE_FALSE( capture.status().m_valid );
        REQUIRE( capture.status().m_invalid == 1 );
        REQUIRE( capture.samples().empty() );
    }
    for( const int fault : { 0, 1, 2, 3, 4, 5, 6, 7 } )
    {
        DDSPC::AdcCapture capture( DDSPC::AdcConfig{} );
        capture.attached( 1, 1 );
        AdcFixture source;
        source.publish( 1, 1000000 );
        if( fault == 0 )
        {
            source.m_header.m_write = true;
        }
        if( fault == 1 )
        {
            source.m_race = true;
        }
        if( fault == 2 )
        {
            source.m_frames[0].m_writtenRealNs = source.m_clock.m_realNs + 1;
        }
        if( fault == 3 )
        {
            source.m_header.m_uint16 = false;
        }
        if( fault == 4 )
        {
            source.m_frames[0].m_counter = 100;
        }
        if( fault == 5 )
        {
            source.m_frames[0].m_acquiredRealNs = -1;
        }
        if( fault == 6 )
        {
            source.m_frames[0].m_writtenRealNs = source.m_frames[0].m_acquiredRealNs - 1;
        }
        if( fault == 7 )
        {
            source.m_header.m_width = 1;
        }
        REQUIRE( capture.drain( source ) == 0 );
        REQUIRE_FALSE( capture.status().m_valid );
        REQUIRE( capture.samples().empty() );
    }
    DDSPC::AdcConfig config;
    config.m_maxAgeNs = 1000000;
    DDSPC::AdcCapture capture( config );
    capture.attached( 1, 1 );
    AdcFixture source;
    source.publish( 1, 1000000 );
    REQUIRE( capture.drain( source ) == 1 );
    capture.tick( 2000001 );
    REQUIRE( capture.status().m_state == DDSPC::AdcState::Stale );
    REQUIRE( capture.samples().empty() );
    source.publish( 2, 2500000 );
    REQUIRE( capture.drain( source ) == 1 );
    source.publish( 3, 2000000 ); // Reversed acquisition is not repaired by advancing publication counters.
    source.m_clock = { 10003000000LL, 3000000, 10 };
    REQUIRE( capture.drain( source ) == 0 );
    source.publish( 4, 2400000 );
    source.m_clock = { 10003100000LL, 3100000, 10 };
    REQUIRE( capture.drain( source ) == 0 ); // Acquisition high-water survives invalidation.
    source.publish( 5, 3200000 );
    REQUIRE( capture.drain( source ) == 1 );
}

/** \brief Clock faults latch, while source rollback/reconnection create explicit fresh generations. */
TEST_CASE( "ADC capture recovers source restarts without joining old timestamp histories", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::attached;
    DDSPC::AdcCapture::disconnected;
#endif
    // clang-format on
    DDSPC::AdcCapture capture( DDSPC::AdcConfig{} );
    capture.attached( 1, 2 );
    AdcFixture source;
    source.publish( 10, 1000000 );
    REQUIRE( capture.drain( source ) == 1 );
    source.publish( 1, 2000000 );
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_generation == 2 );
    REQUIRE( capture.status().m_restarts == 1 );
    REQUIRE( capture.samples().size() == 1 );
    source.publish( 2, 3000000 );
    source.m_clock.m_realNs += 2000000;
    REQUIRE( capture.drain( source ) == 0 );
    REQUIRE( capture.status().m_state == DDSPC::AdcState::ClockFault );
    source.publish( 3, 4000000 );
    REQUIRE( capture.drain( source ) == 0 ); // A new sample cannot silently certify a new clock mapping.
    capture.disconnected();
    REQUIRE_FALSE( capture.status().m_valid );
    REQUIRE( capture.samples().empty() );
    capture.attached( 1, 3 );
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_generation == 3 );
    REQUIRE( capture.status().m_inode == 3 );
    capture.tick( 1 );
    REQUIRE( capture.status().m_state == DDSPC::AdcState::ClockFault );
}

/** \brief Invalid metadata is rejected before payload access and corrected metadata permits recovery. */
TEST_CASE( "ADC rejects invalid layouts without reading a payload and recovers on a valid frame",
           "[loPredCtrlAcc][adc]" )
{
    for( int fault = 0; fault < 7; ++fault )
    {
        DDSPC::AdcCapture capture( DDSPC::AdcConfig{} );
        capture.attached( 1, 1 );
        AdcFixture source;
        source.publish( 1, 1000000 );
        const auto correct = source.m_header;
        if( fault == 0 )
        {
            source.m_header.m_axes = 1;
        }
        if( fault == 1 )
        {
            source.m_header.m_axes = 4;
        }
        if( fault == 2 )
        {
            source.m_header.m_height = 2;
        }
        if( fault == 3 )
        {
            source.m_header.m_depth = 0;
        }
        if( fault == 4 )
        {
            source.m_header.m_index = source.m_header.m_depth;
        }
        if( fault == 5 )
        {
            source.m_header.m_uint16 = false;
        }
        if( fault == 6 )
        {
            source.m_header.m_cpu = false;
        }
        REQUIRE( capture.drain( source ) == 0 );
        REQUIRE( source.m_reads == 0 );
        REQUIRE_FALSE( capture.status().m_valid );
        source.m_header = correct;
        REQUIRE( capture.drain( source ) == 1 );
        REQUIRE( capture.status().m_valid );
    }
}

/** \brief Configurable endpoint rejection and an invalid-to-valid sequence do not retain contaminated history. */
TEST_CASE( "ADC rail policy and invalid calibration are independent per channel", "[loPredCtrlAcc][adc]" )
{
    DDSPC::AdcConfig config;
    config.m_rejectRails = false;
    DDSPC::AdcCapture capture( config );
    capture.attached( 1, 1 );
    AdcFixture source;
    source.publish( 1, 1000000, { 0, 4095 } );
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_latest.m_acceleration[0] == Approx( -2048 * .0083799 / 1.02 ) );
    REQUIRE( capture.status().m_latest.m_acceleration[1] == Approx( 2047 * .0083799 / 1.02 ) );
    source.publish( 2, 1125000, { 5000, 2048 } );
    REQUIRE( capture.drain( source ) == 0 );
    REQUIRE( capture.samples().empty() );
    source.publish( 3, 1250000, { 2048, 2048 } );
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.samples().size() == 1 );
    REQUIRE( capture.samples()[0].m_acceleration == std::array<double, 2>{ 0, 0 } );
    for( int axis = 0; axis < 2; ++axis )
    {
        auto bad         = config;
        bad.m_zero[axis] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS( DDSPC::AdcCapture( bad ), std::invalid_argument );
        bad                       = config;
        bad.m_voltsPerCount[axis] = std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS( DDSPC::AdcCapture( bad ), std::invalid_argument );
        bad                  = config;
        bad.m_polarity[axis] = 0;
        REQUIRE_THROWS_AS( DDSPC::AdcCapture( bad ), std::invalid_argument );
    }
}

/** \brief Counter rollback is measured against stable publication evidence even while a bounded batch trails it. */
TEST_CASE( "ADC capture detects producer rollback while draining retained history", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::drain;
#endif
    // clang-format on
    DDSPC::AdcConfig config;
    config.m_capacity = 2;
    DDSPC::AdcCapture capture( config );
    capture.attached( 1, 2 );
    AdcFixture source;
    source.publish( 1, 125000 );
    REQUIRE( capture.drain( source ) == 1 );
    for( std::uint64_t n = 2; n <= 12; ++n )
    {
        source.publish( n, n * 125000 );
    }
    REQUIRE( capture.drain( source ) == 2 ); // Observed12, consumed through6, with7..12 still retained.
    REQUIRE( capture.status().m_source.m_counter == 12 );
    REQUIRE( capture.samples().back().m_frame.m_counter == 6 );
    source.publish( 7, 2000000 ); // New producer counter lies above the cursor but below the prior publication.
    REQUIRE( capture.drain( source ) == 1 );
    REQUIRE( capture.status().m_generation == 2 );
    REQUIRE( capture.status().m_restarts == 1 );
    REQUIRE( capture.samples().size() == 1 );
    REQUIRE( capture.samples().front().m_frame.m_counter == 7 );
}

/** \brief Cursor copies are bounded, chronological, non-consuming, and cannot consume future receipts. */
TEST_CASE( "ADC cursor forwards retained samples once with an explicit availability cutoff", "[loPredCtrlAcc][adc]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcCapture::copy_after;
#endif
    // clang-format on
    DDSPC::AdcCapture capture( DDSPC::AdcConfig{} );
    capture.attached( 7, 8 );
    AdcFixture                      source;
    std::array<DDSPC::AdcSample, 2> storage;
    auto                            batch = capture.copy_after( {}, storage, 2000000 );
    REQUIRE_FALSE( batch.m_next.m_bound );
    for( std::uint64_t n = 100; n < 104; ++n )
    {
        source.publish( n, 1000000 + ( n - 100 ) * 125000 );
        REQUIRE( capture.drain( source ) == 1 );
    }
    batch = capture.copy_after( batch.m_next, storage, 1009999 );
    REQUIRE( batch.m_count == 0 );
    REQUIRE( batch.m_missing == 0 ); // Initial attachment cannot measure older history.
    REQUIRE( batch.m_next.m_counter == 99 );
    batch = capture.copy_after( batch.m_next, storage, 1200000 );
    REQUIRE( batch.m_count == 2 );
    REQUIRE_FALSE( batch.m_more ); // Next retained sample exists, but was not yet available at this cutoff.
    REQUIRE( storage[0].m_frame.m_counter == 100 );
    REQUIRE( storage[1].m_frame.m_counter == 101 );
    REQUIRE( storage[0].m_acquiredNs == 1000000 );
    REQUIRE( storage[0].m_frame.m_acquiredRealNs == 10001000000LL );
    REQUIRE( storage[0].m_generation == batch.m_status.m_generation );
    REQUIRE( batch.m_status.m_device == 7 );
    REQUIRE( batch.m_status.m_inode == 8 );
    batch = capture.copy_after( batch.m_next, storage, 2000000 );
    REQUIRE( batch.m_count == 2 );
    REQUIRE( storage[0].m_frame.m_counter == 102 );
    REQUIRE( storage[1].m_frame.m_counter == 103 );
    REQUIRE( capture.copy_after( batch.m_next, storage, 2000000 ).m_count == 0 );
    REQUIRE( capture.samples().size() == 4 ); // A consumer cursor does not consume capture retention.
    REQUIRE( capture.copy_after( {}, storage, 2000000 ).m_more );
    REQUIRE_THROWS_AS( capture.copy_after( {}, {}, 2000000 ), std::invalid_argument );
    REQUIRE_THROWS_AS( capture.copy_after( {}, storage, -1 ), std::invalid_argument );
}

/** \brief Cursor loss is exact only within a known incarnation; invalid samples and restarts never join histories. */
TEST_CASE( "ADC cursor reports overwrite gaps invalidation and source identity changes", "[loPredCtrlAcc][adc]" )
{
    DDSPC::AdcConfig c;
    c.m_capacity = 3;
    DDSPC::AdcCapture capture( c );
    capture.attached( 1, 2 );
    AdcFixture                      source;
    std::array<DDSPC::AdcSample, 2> storage;
    source.publish( 10, 1000000 );
    REQUIRE( capture.drain( source ) == 1 );
    auto batch = capture.copy_after( {}, storage, 2000000 );
    for( std::uint64_t n = 11; n <= 15; ++n )
    {
        source.publish( n, 1000000 + ( n - 10 ) * 125000 );
        REQUIRE( capture.drain( source ) == 1 );
    }
    batch = capture.copy_after( batch.m_next, storage, 2000000 );
    REQUIRE( batch.m_missing == 2 ); // 11,12 evicted; 13,14 copied; 15 still pending.
    REQUIRE( batch.m_discontinuity );
    REQUIRE( batch.m_more );
    REQUIRE( storage[0].m_frame.m_counter == 13 );
    batch = capture.copy_after( batch.m_next, storage, 2000000 );
    REQUIRE( batch.m_count == 1 );
    REQUIRE( batch.m_missing == 0 );
    REQUIRE_FALSE( batch.m_discontinuity );
    source.publish( 16, 1750000, { 0, 2048 } );
    REQUIRE( capture.drain( source ) == 0 );
    batch = capture.copy_after( batch.m_next, storage, 2000000 );
    REQUIRE( batch.m_discontinuity );
    REQUIRE( batch.m_count == 0 );
    source.m_header.m_depth = 1;   // The producer retained no17; its older slot must not masquerade as valid history.
    source.publish( 18, 2000000 ); // Rejected16 and missing17 both unavailable; do not add to capture loss totals.
    REQUIRE( capture.drain( source ) == 1 );
    batch = capture.copy_after( batch.m_next, storage, 3000000 );
    REQUIRE( batch.m_missing == 2 );
    REQUIRE( batch.m_discontinuity );
    REQUIRE( storage[0].m_frame.m_counter == 18 );
    capture.disconnected();
    batch = capture.copy_after( batch.m_next, storage, 3000000 );
    REQUIRE( batch.m_discontinuity );
    capture.attached( 1, 9 );
    source.publish( 1, 3000000 );
    REQUIRE( capture.drain( source ) == 1 );
    batch = capture.copy_after( batch.m_next, storage, 4000000 );
    REQUIRE( batch.m_missing == 0 ); // New source has no comparable counter baseline.
    REQUIRE( batch.m_count == 1 );
    REQUIRE( batch.m_status.m_inode == 9 );
    REQUIRE( storage[0].m_generation == 2 );
    REQUIRE( storage[0].m_frame.m_counter == 1 );
}

} // namespace libXWCTest::loPredCtrlAccTest
