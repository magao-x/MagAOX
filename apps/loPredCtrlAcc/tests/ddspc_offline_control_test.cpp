/** \file ddspc_offline_control_test.cpp
 * \brief Deterministic offline window coverage, causality, fallback, hysteresis and command-bound tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "offline_control.hpp"
#include <cmath>
#include <limits>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond OFFLINE_TEST_SUPPORT
/// Construct owned test evidence in the same common-clock format returned by actual AdcCapture.
DDSPC::AdcSample window_sample( std::int64_t  acquired /**< [in] Physical common-clock acquisition. */,
                                std::uint64_t counter /**< [in] Publication identity. */,
                                double        value /**< [in] Decoded SI acceleration. */ )
{
    DDSPC::AdcSample s;
    s.m_acquiredNs      = acquired;
    s.m_publishedNs     = acquired + 1000;
    s.m_receivedNs      = acquired + 3000;
    s.m_frame.m_counter = counter;
    s.m_generation = s.m_segment = 1;
    s.m_acceleration             = { value, -value };
    return s;
}

/// Build a nominal four-sample physical window with known analytical mean and acquisition centroid.
std::vector<DDSPC::AdcSample> nominal_window()
{
    std::vector<DDSPC::AdcSample> samples;
    for( int i = 0; i < 4; ++i )
        samples.push_back( window_sample( 1000000000 + i * 125000, i + 1, i + 1 ) );
    return samples;
}

/// Supply fully qualified finite model proposals without any device or runtime implementation.
ControlEvidence qualified_evidence()
{
    ControlEvidence e;
    e.m_commonValid = e.m_featuresValid = e.m_standardReady = e.m_assistedReady = true;
    e.m_featureReason                                                           = FeatureReason::Ready;
    e.m_standard.setConstant( -.5 );
    e.m_assisted.setConstant( .5 );
    return e;
}
/// \endcond

/** \brief Timestamp integration has exact known weights, excludes future data, and checks effective sample age. */
TEST_CASE( "Offline acceleration windows use physical time and a frozen deadline", "[adc-ddspc][window]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::acceleration_window;
#endif
    // clang-format on

    auto          samples = nominal_window();
    FeatureConfig c;
    const auto    nominal = acceleration_window( samples, 1000500000, c );
    REQUIRE( nominal.m_reason == FeatureReason::Ready );
    REQUIRE( nominal.m_value( 0 ) == Approx( 2.5 ) );
    REQUIRE( nominal.m_value( 1 ) == Approx( -2.5 ) );
    REQUIRE( nominal.m_effectiveNs == 1000187500 );
    REQUIRE( nominal.m_latestNs == 1000375000 );
    REQUIRE( nominal.m_availableNs == 1000378000 );
    REQUIRE( nominal.m_samples == 4 );
    samples.push_back( window_sample( 1000500000, 5, 10000 ) );
    REQUIRE( acceleration_window( samples, 1000500000, c ).m_value == nominal.m_value );
    c.m_lagNs = 125000;
    REQUIRE( acceleration_window( samples, 1000625000, c ).m_effectiveNs == nominal.m_effectiveNs );
    c.m_maxAgeNs = 400000;
    REQUIRE( acceleration_window( samples, 1000625000, c ).m_reason == FeatureReason::Stale );
}

/** \brief Jitter changes integration weights and centroid, while gaps, late delivery and stale data fail closed. */
TEST_CASE( "Offline acceleration coverage rejects incomplete stale and late windows", "[adc-ddspc][window]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::acceleration_window;
#endif
    // clang-format on

    const std::int64_t            offsets[]{ -10000, 115000, 260000, 380000 };
    std::vector<DDSPC::AdcSample> jitter;
    for( int i = 0; i < 4; ++i )
        jitter.push_back( window_sample( 1000000000 + offsets[i], i + 1, i + 1 ) );
    const auto feature = acceleration_window( jitter, 1000500000, {} );
    REQUIRE( feature.m_reason == FeatureReason::Ready );
    REQUIRE( feature.m_value( 0 ) == Approx( 2.49 ) );
    REQUIRE( feature.m_effectiveNs == 1000184650 );
    auto samples = nominal_window();
    samples.erase( samples.begin() + 1 );
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_reason == FeatureReason::Gap );
    samples                      = nominal_window();
    samples.front().m_receivedNs = 1000500001;
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_reason == FeatureReason::Incomplete );
    samples                     = nominal_window();
    samples.back().m_receivedNs = 1000500001;
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_reason == FeatureReason::Gap );
    REQUIRE( acceleration_window( nominal_window(), 1001500000, {} ).m_reason == FeatureReason::Stale );
    samples              = nominal_window();
    samples[2].m_segment = 2;
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_reason == FeatureReason::Gap );
    samples                      = nominal_window();
    samples[1].m_acceleration[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_reason == FeatureReason::Invalid );
    REQUIRE( acceleration_window( samples, 1000500000, {} ).m_value.isZero( 0 ) );
    REQUIRE( acceleration_window( nominal_window(), 1000500000, {} ).m_reason == FeatureReason::Ready );
}

/** \brief Every failed prerequisite forces immediate fallback, and intermittent readiness cannot skip recovery dwell.
 */
TEST_CASE( "Offline requested and effective modes use qualified recovery hysteresis", "[adc-ddspc][policy]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowPolicy::decide;
#endif
    // clang-format on

    PolicyConfig c;
    c.m_recoveryFrames = 3;
    OfflinePolicy p( OfflineMode::Assisted, c );
    auto          e        = qualified_evidence();
    auto          previous = DDSPC::Matrix::Zero( 2, 1 ).eval();
    const auto    probe    = DDSPC::Matrix::Zero( 2, 1 ).eval();
    for( int i = 1; i <= 3; ++i )
    {
        const auto d = p.decide( e, previous, probe );
        REQUIRE( d.m_requested == OfflineMode::Assisted );
        REQUIRE( d.m_effective == ( i < 3 ? OfflineMode::Standard : OfflineMode::Assisted ) );
        REQUIRE( d.m_validStreak == unsigned( i ) );
        REQUIRE( d.m_step <= c.m_maxStep );
        previous = d.m_command;
    }
    e.m_featuresValid = false;
    e.m_featureReason = FeatureReason::Stale;
    auto d            = p.decide( e, previous, probe );
    REQUIRE( d.m_switched );
    REQUIRE( d.m_effective == OfflineMode::Standard );
    REQUIRE( std::string( d.m_reason ) == "adc_stale" );
    REQUIRE( d.m_validStreak == 0 );
    previous = d.m_command;
    e        = qualified_evidence();
    REQUIRE( p.decide( e, previous, probe ).m_validStreak == 1 );
    e.m_commonValid = false;
    d               = p.decide( e, previous, probe );
    REQUIRE( d.m_effective == OfflineMode::Safe );
    REQUIRE( d.m_validStreak == 0 );
    REQUIRE( std::string( d.m_reason ) == "common_input_invalid" );
    e = qualified_evidence();
    REQUIRE( p.decide( e, previous, probe ).m_validStreak == 1 );
    e.m_assistedReady = false;
    REQUIRE( p.decide( e, previous, probe ).m_validStreak == 0 );
    e.m_assistedReady = true;
    e.m_assisted( 0 ) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE( p.decide( e, previous, probe ).m_effective == OfflineMode::Standard );
    e = qualified_evidence();
    REQUIRE( p.decide( e, previous, probe ).m_effective == OfflineMode::Standard );
    REQUIRE( p.decide( e, previous, probe ).m_effective == OfflineMode::Standard );
    REQUIRE( p.decide( e, previous, probe ).m_effective == OfflineMode::Assisted );
}

/** \brief Unavailable models ramp to zero without probing, with bounds enforced beyond the first switch frame. */
TEST_CASE( "Offline safe commands and all model switches obey shared bounds", "[adc-ddspc][policy]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowPolicy::decide;
    DDSPC::bounded_command;
#endif
    // clang-format on

    PolicyConfig c;
    c.m_limit          = .3;
    c.m_maxStep        = .07;
    c.m_recoveryFrames = 1;
    OfflinePolicy p( OfflineMode::Assisted, c );
    auto          e        = qualified_evidence();
    auto          previous = DDSPC::Matrix::Zero( 2, 1 ).eval();
    const auto    probe    = DDSPC::Matrix::Constant( 2, 1, 10 ).eval();
    for( int i = 0; i < 50; ++i )
    {
        if( i == 10 )
            e.m_assistedReady = false;
        if( i == 20 )
            e.m_standardReady = false;
        const auto d = p.decide( e, previous, probe );
        REQUIRE( d.m_command.cwiseAbs().maxCoeff() <= c.m_limit );
        REQUIRE( d.m_step <= c.m_maxStep );
        if( i >= 20 )
        {
            REQUIRE( d.m_effective == OfflineMode::Safe );
            REQUIRE( d.m_command.cwiseAbs().maxCoeff() <= previous.cwiseAbs().maxCoeff() );
        }
        previous = d.m_command;
    }
    REQUIRE( previous.isZero( 0 ) );
    OfflinePolicy standard( OfflineMode::Standard, c );
    e                 = qualified_evidence();
    e.m_standardReady = false;
    REQUIRE( standard.decide( e, previous, probe ).m_effective == OfflineMode::Safe );
    c.m_maxStep = 0;
    REQUIRE_THROWS_AS( OfflinePolicy( OfflineMode::Assisted, c ), std::invalid_argument );
}
} // namespace libXWCTest::loPredCtrlAccTest
