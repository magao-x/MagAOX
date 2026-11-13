/** \file shadow_replay.cpp
 * \brief Offline virtual FIFO delivery with independent confirmed inputs and actual ADC decoding.
 */
#include "shadow_replay.hpp"
#ifdef DDSPC_REPLAY_ALLOCATION_PROBE
    #include "allocation_probe.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
constexpr std::int64_t origin = 1000000000, epoch = 1800000000000000000LL;

/// Immutable one-plane replay image; it implements only read-only ADC access and has no native API.
struct ReplayAdc : DDSPC::AdcSource
{
    DDSPC::AdcHeader      m_header;     ///< Original raw-pair publication metadata.
    DDSPC::AdcFrame       m_frame;      ///< Original raw ADC codes/timestamps.
    DDSPC::AdcClock       m_clock;      ///< Known common-clock receipt observation.
    std::array<double, 2> m_physical{}; ///< Latent physical forcing, unavailable through the input interface.
    bool                  m_observed{ true }, m_restart{ false }; ///< Consumer dropout and producer incarnation flags.

    /// Copy owned publication metadata.
    DDSPC::AdcHeader header() const override;

    /// Copy the sole raw sample pair.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Sole plane zero. */ ) const override;

    /// Return the fixed virtual delivery observation.
    DDSPC::AdcClock clock() const override;
};

DDSPC::AdcHeader ReplayAdc::header() const
{
    return m_header;
}

DDSPC::AdcFrame ReplayAdc::frame( std::size_t index ) const
{
    if( index )
        throw std::out_of_range( "Replay has one ADC plane" );
    return m_frame;
}

DDSPC::AdcClock ReplayAdc::clock() const
{
    return m_clock;
}

/// Construct one established source/clock/layout identity without relying on framework or hardware state.
DDSPC::InputStamp stamp( const DDSPC::InputConfig &c /**< [in] Synthetic immutable contract. */,
                         std::uint64_t             generation /**< [in] Source incarnation. */,
                         std::uint64_t             counter /**< [in] Publication identity. */,
                         std::int64_t              available /**< [in] Established virtual availability. */ )
{
    DDSPC::InputStamp s;
    s.m_generation    = generation;
    s.m_counter       = counter;
    s.m_clockEpoch    = c.m_clockEpoch;
    s.m_layout        = c.m_layout;
    s.m_availableNs   = available;
    s.m_receivedNs    = available + 10000;
    s.m_snapshotValid = true;
    return s;
}

/// Convert a callback duration to microseconds without pacing or changing its synthetic timing evidence.
double elapsed_us( std::chrono::steady_clock::time_point start /**< [in] Local callback start. */ )
{
    return std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count();
}
} // namespace

ShadowReplayConfig::ShadowReplayConfig()
{
    m_inputs.m_modes = m_inputs.m_accelChannels = 2;
    m_inputs.m_wfsGeneration                    = 11;
    m_inputs.m_commandGeneration                = 22;
    m_inputs.m_clockEpoch                       = 33;
    m_inputs.m_layout                           = 44;
    m_inputs.m_periodNs                         = 500000;
    m_inputs.m_exposureNs                       = 200000;
    m_inputs.m_responseLagNs                    = 50000;
    m_inputs.m_minCommandDelayNs                = 100000;
    m_inputs.m_maxAgeNs                         = 2000000;
    m_inputs.m_maxUncertaintyNs                 = 1000;
    m_inputs.m_evidence                         = DDSPC::InputEvidence::SyntheticApplied;
    m_inputs.m_timingVerified = m_inputs.m_mappingVerified = true;
    m_adc.m_calibration.m_capacity                         = 128;
}

ShadowReplayResult replay_shadow( ReplaySink &sink, const ShadowReplayConfig &c )
{
    if( c.m_frames < 2 || c.m_frames > 100000 || c.m_inputs.m_periodNs != 500000 || c.m_inputs.m_exposureNs != 200000 ||
        c.m_inputs.m_responseLagNs != 50000 || c.m_inputs.m_modes < 1 || c.m_inputs.m_modes > 128 ||
        c.m_inputs.m_evidence != DDSPC::InputEvidence::SyntheticApplied || c.m_skipWfsStart < 0 ||
        c.m_skipWfsCount < 0 || c.m_badAppliedStart < 0 || c.m_badAppliedCount < 0 )
        throw std::invalid_argument( "Unsupported or non-synthetic replay contract" );
    // All measurements/holds are immutable before the first application call; the recording never sees a proposal.
    SyntheticAdcSource     source( c.m_adc );
    std::vector<ReplayAdc> adc;
    const auto             lastDecision = origin + std::int64_t( c.m_frames - 1 ) * 500000 + 250000;
    std::int64_t           receipt      = 0;
    for( std::uint64_t tick = 0;; ++tick )
    {
        ReplayAdc event;
        event.m_observed = source.advance();
        event.m_header   = source.header();
        event.m_frame    = source.frame( 0 );
        event.m_clock    = source.clock();
        event.m_physical = source.physical();
        if( event.m_frame.m_acquiredRealNs - epoch >= lastDecision )
            break;
        event.m_restart = tick == c.m_restartTick;
        if( tick >= c.m_restartTick )
            event.m_header.m_counter = event.m_frame.m_counter = tick - c.m_restartTick + 1;
        event.m_clock.m_monoNs = std::max( receipt + 1, event.m_clock.m_monoNs );
        event.m_clock.m_realNs = epoch + event.m_clock.m_monoNs;
        receipt                = event.m_clock.m_monoNs;
        adc.push_back( event );
    }
    std::mt19937_64                        external( c.m_adc.m_seed + 100 ), wfsNoise( c.m_adc.m_seed + 101 );
    std::normal_distribution<double>       externalNormal, wfsNormal;
    std::vector<DDSPC::TimedWfsRecord>     measurements;
    std::vector<DDSPC::AppliedInputRecord> holds;
    DDSPC::Matrix                          previous    = DDSPC::Matrix::Zero( c.m_inputs.m_modes, 1 );
    DDSPC::Matrix                          disturbance = previous;
    std::size_t                            physical    = 0;
    std::array<double, 2>                  forcing{};
    for( std::size_t i = 0; i < c.m_frames; ++i )
    {
        const auto start = origin + std::int64_t( i ) * 500000;
        // A causal lumped disturbance driven only by physical ADC times before this exposure starts.
        while( physical < adc.size() && adc[physical].m_frame.m_acquiredRealNs - epoch < start )
        {
            forcing = adc[physical].m_physical;
            ++physical;
        }
        for( int axis = 0; axis < c.m_inputs.m_modes; ++axis )
            disturbance( axis ) = .8f * disturbance( axis ) + float( forcing[axis % 2] );
        DDSPC::TimedWfsRecord w;
        w.m_exposureStartNs = start;
        w.m_exposureEndNs   = start + 200000;
        w.m_decisionNs      = start + 250000;
        w.m_stamp           = stamp( c.m_inputs, c.m_inputs.m_wfsGeneration, 100 + i, start + 210000 );
        w.m_residual        = disturbance + previous;
        for( int axis = 0; axis < c.m_inputs.m_modes; ++axis )
            w.m_residual( axis ) += float( c.m_wfsNoise * wfsNormal( wfsNoise ) );
        measurements.push_back( w );
        DDSPC::AppliedInputRecord u;
        // These contiguous holds close before the following WFS decision. They cover its lag-aligned exposure.
        u.m_effectiveStartNs = start + 150000;
        u.m_effectiveEndNs   = start + 650000;
        u.m_requestedNs      = start + 50000;
        u.m_stamp            = stamp( c.m_inputs, c.m_inputs.m_commandGeneration, 900 + i, u.m_effectiveEndNs + 10000 );
        u.m_applied.resize( c.m_inputs.m_modes, 1 );
        for( int axis = 0; axis < c.m_inputs.m_modes; ++axis )
            u.m_applied( axis ) = float( std::clamp( .3 * externalNormal( external ), -.4, .4 ) );
        const bool unverified =
            c.m_rejectEveryCommand || ( std::int64_t( i ) >= c.m_badAppliedStart &&
                                        std::int64_t( i ) - c.m_badAppliedStart < c.m_badAppliedCount );
        u.m_evidence = unverified ? DDSPC::InputEvidence::DriverPrepared : DDSPC::InputEvidence::SyntheticApplied;
        holds.push_back( u );
        previous = u.m_applied;
    }
    DDSPC::AdcCapture capture( c.m_adc.m_calibration );
    capture.attached( 1, 1 );
    ShadowReplayResult result;
    result.m_adcUs.reserve( adc.size() );
    result.m_wfsUs.reserve( measurements.size() );
    result.m_commandUs.reserve( holds.size() );
    double frameUs = 0;
    result.m_frameUs.reserve( measurements.size() );
    std::size_t a = 0, w = 0, u = 0;
    while( w < measurements.size() )
    {
        const auto at  = a < adc.size() ? adc[a].m_clock.m_monoNs : std::numeric_limits<std::int64_t>::max();
        const auto ut  = u < holds.size() ? holds[u].m_stamp.m_receivedNs : std::numeric_limits<std::int64_t>::max();
        const auto wt  = measurements[w].m_decisionNs;
        const auto now = std::min( { at, ut, wt } );
        capture.tick( now );
        const auto begin = std::chrono::steady_clock::now();
#ifdef DDSPC_REPLAY_ALLOCATION_PROBE
        DDSPCBenchmark::enabled = true;
        const auto eigenBefore = DDSPCBenchmark::eigenRequests, cppBefore = DDSPCBenchmark::cppRequests;
#endif
        if( at == now )
        {
            const auto &event = adc[a++];
            if( event.m_restart )
                capture.attached( 1, 2 );
            if( event.m_observed && capture.drain( event ) == 1 )
            {
                DDSPC::ShadowAdcRecord record;
                record.m_sample        = capture.status().m_latest;
                record.m_clockEpoch    = c.m_inputs.m_clockEpoch + ( c.m_wrongAdcClock ? 1 : 0 );
                record.m_calibration   = 1;
                record.m_uncertaintyNs = 20;
                record.m_snapshotValid = true;
                result.m_adcAccepted += sink.adc( record, now );
            }
            result.m_adcUs.push_back( elapsed_us( begin ) );
            frameUs += result.m_adcUs.back();
        }
        else if( ut == now )
        {
            result.m_commandsAccepted += sink.applied( holds[u++], now );
            result.m_commandUs.push_back( elapsed_us( begin ) );
            frameUs += result.m_commandUs.back();
        }
        else
        {
            const bool skip =
                std::int64_t( w ) >= c.m_skipWfsStart && std::int64_t( w ) - c.m_skipWfsStart < c.m_skipWfsCount;
            if( !skip )
                result.m_wfsAccepted += sink.wfs( measurements[w], now + ( w == c.m_lateWfsIndex ? 1 : 0 ) );
            ++w;
            result.m_wfsUs.push_back( elapsed_us( begin ) );
            frameUs += result.m_wfsUs.back();
            result.m_frameUs.push_back( frameUs );
            frameUs = 0;
        }
#ifdef DDSPC_REPLAY_ALLOCATION_PROBE
        DDSPCBenchmark::enabled = false;
        result.m_eigenRequests += DDSPCBenchmark::eigenRequests - eigenBefore;
        result.m_cppRequests += DDSPCBenchmark::cppRequests - cppBefore;
#endif
    }
    result.m_capture = capture.status();
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
