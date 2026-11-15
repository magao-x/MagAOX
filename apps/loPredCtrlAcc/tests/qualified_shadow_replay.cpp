/** \file qualified_shadow_replay.cpp
 * \brief Qualified synthetic input replay, explicitly distinct from exposure/DM proof on the instrument.
 */
#include "qualified_shadow_replay.hpp"
#include "../synchronized_inputs.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

/// \cond QUALIFIED_REPLAY_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
constexpr std::int64_t origin = 1000000000, period = 500000, slot = 125000;

/// Bounded in-memory diagnostics; no I/O or allocation while a model emits scores.
class Scores : public DDSPC::ShadowPredictionSink
{
    QualifiedReplayResult &m_result; ///< Borrowed result outlives both learners.
  public:
    /// Allocate a strict upper bound once, outside model callbacks.
    Scores( QualifiedReplayResult &result /**< [out] Owned traces. */,
            std::size_t            maximum /**< [in] Maximum possible scalar forecasts. */ );

    /// Retain a fixed scalar or count a diagnostic loss, never alter training.
    void prediction( const DDSPC::ShadowPrediction &sample /**< [in] Original native score. */ ) noexcept override;
};
Scores::Scores( QualifiedReplayResult &result, std::size_t maximum ) : m_result( result )
{
    m_result.m_scores.reserve( maximum );
}
void Scores::prediction( const DDSPC::ShadowPrediction &sample ) noexcept
{
    if( m_result.m_scores.size() < m_result.m_scores.capacity() )
        m_result.m_scores.push_back( sample );
    else
        ++m_result.m_scoreDrops;
}

/// Original snapshot with explicit synthetic identity and separate acquisition/publication/receipt markers.
DDSPC::NativeObservation raw( const DDSPC::MultirateContract &contract /**< [in] Test-only source bindings. */,
                              DDSPC::ObservationSource        source /**< [in] Original source. */,
                              std::uint64_t                   counter /**< [in] Actual source count. */,
                              std::int64_t                    acquired /**< [in] Known synthetic acquisition. */,
                              std::int64_t                    received /**< [in] Known receipt. */ )
{
    DDSPC::NativeObservation r;
    const auto              &b = contract.m_sources[std::size_t( source )];
    r.m_source                 = source;
    r.m_device                 = b.m_device;
    r.m_fileInode              = b.m_inode;
    r.m_imageInode             = b.m_image;
    r.m_generation             = b.m_generation;
    r.m_segment                = 1;
    r.m_counter                = counter;
    r.m_payloadHash            = counter + 1000;
    r.m_acquiredMonoNs         = acquired;
    r.m_publishedMonoNs        = acquired + 1000;
    r.m_receivedNs = r.m_availableNs = received;
    return r;
}
} // namespace

QualifiedReplayResult replay_qualified( const QualifiedReplayConfig &c )
{
    if( c.m_frames < 100 || c.m_frames > 20000 || c.m_future < 1 || c.m_future > 8 )
        throw std::invalid_argument( "Qualified replay resource/horizon bounds" );
    QualifiedReplayResult result;
    result.m_computeUs.reserve( c.m_frames );
    result.m_numericalUs.reserve( c.m_frames );
    result.m_maxJobUs.reserve( c.m_frames );
    result.m_readiness.reserve( c.m_frames );
    DDSPC::OnlineShadowConfig config;
    auto                     &m = config.m_model;
    m.m_modes                   = 2;
    m.m_history                 = 2;
    m.m_future                  = c.m_future;
    m.m_accelHistory            = 1;
    m.m_prior                   = .001f;
    m.m_lambda                  = .1f;
    m.m_useQrd                  = c.m_qrd;
    m.m_standardCommandSlots    = 4; // Same command space: only ADC columns differ between models.
    DDSPC::MultirateContract contract;
    auto                    &i = contract.m_inputs;
    i.m_modes                  = 2;
    i.m_periodNs               = period;
    i.m_exposureNs             = 400000;
    i.m_jitterNs               = 2000;
    i.m_maxUncertaintyNs       = 20;
    i.m_maxAgeNs               = 5000000;
    i.m_wfsGeneration          = 1;
    i.m_commandGeneration      = 9;
    i.m_clockEpoch = i.m_layout = 1;
    i.m_evidence                = DDSPC::InputEvidence::SyntheticApplied;
    i.m_timingVerified = i.m_mappingVerified = true;
    contract.m_clockVerified = contract.m_associationVerified = contract.m_scheduleVerified = true;
    contract.m_exposureVerified                                                             = c.m_exposureEvidence;
    contract.m_provenance = "SYNTHETIC: known clock/exposure/consumption oracle, not instrument evidence";
    if( c.m_upstream )
    {
        contract.m_upstream.m_enabled = contract.m_upstream.m_combinedRequestModel = true;
        contract.m_upstream.m_epoch                                                = 77;
        contract.m_upstream.m_modalAnchor = contract.m_upstream.m_referenceAnchor = contract.m_upstream.m_adcAnchor = 1;
        contract.m_clockVerified = contract.m_exposureVerified = contract.m_associationVerified =
            contract.m_scheduleVerified                        = false;
        i.m_exposureNs                                         = 0;
        i.m_evidence                                           = DDSPC::InputEvidence::Requested;
        contract.m_provenance =
            "SYNTHETIC upstream semaphore/counter/command-sampler oracle; no physical exposure or clock claim";
    }
    for( unsigned s = 0; s < 4; ++s )
        contract.m_sources[s] = { 11, 100 + s, s ? 1000 + s : 0, 1 };
    if( c.m_noAdcAnchor )
    {
        contract.m_upstream.m_adcAnchor = 0;
        contract.m_sources[0]           = {};
    }
    Scores                 scores( result, 4 * c.m_future * c.m_frames );
    DDSPC::MultirateShadow engine(
        config, c.m_upstream ? DDSPC::InputEvidence::VerifiedApplied : DDSPC::InputEvidence::SyntheticApplied );
    engine.prediction_sink( &scores );
    engine.qualify( contract );
    engine.request( DDSPC::ShadowMode::AccelAssisted );

    SyntheticAdcConfig sensor;
    sensor.m_seed       = c.m_seed;
    sensor.m_forceNoise = .1; // Persistent excitation for software identification, not on-sky sensor modeling.
    SyntheticAdcSource source( sensor );
    SyntheticAdcConfig otherSensor = sensor;
    otherSensor.m_seed += 10000;
    otherSensor.m_lowAmplitude.fill( 0 );
    otherSensor.m_highAmplitude.fill( 0 );
    SyntheticAdcSource unrelated( otherSensor );
    DDSPC::AdcCapture  decoder( sensor.m_calibration );
    decoder.attached( 11, 100 );
    std::mt19937                           random( c.m_seed + 100 );
    std::normal_distribution<float>        command( 0, .25f ), noise( 0, .003f );
    std::array<float, 8>                   previousInputs{};
    std::array<double, 2>                  previousForce{}, y{};
    DDSPC::OnlineCommandFrame              pendingCommands;
    std::array<DDSPC::ProjectedRequest, 4> pendingProjection;
    bool                                   havePendingCommands = false;
    for( std::size_t frame = 0; frame < c.m_frames; ++frame )
    {
        const auto begin       = std::chrono::steady_clock::now();
        double     numericalUs = 0, maxJobUs = 0;
        auto       numerical = [&]( auto &&job )
        {
            const auto started = std::chrono::steady_clock::now();
            job();
            const double elapsed =
                std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - started ).count();
            numericalUs += elapsed;
            maxJobUs = std::max( maxJobUs, elapsed );
        };
        const auto                             t = origin + period * std::int64_t( frame + 1 );
        std::array<float, 8>                   inputs;
        std::array<DDSPC::ProjectedRequest, 4> projected;
        for( auto &u : inputs )
            u = std::clamp( command( random ), -.5f, .5f ); // Exogenous actual commands, never shadow plans.
        if( c.m_changeCommandAt && frame + 1 >= c.m_changeCommandAt )
            for( auto &u : inputs )
                u = std::clamp( u + .3f, -.5f, .5f );
        std::array<double, 2> force{};
        for( unsigned s = 0; s < 4; ++s )
        {
            source.advance();
            unrelated.advance();
            const auto physical = source.physical();
            for( unsigned a = 0; a < 2; ++a )
                force[a] += .25 * physical[a]; // Synthetic plant forcing only; predictor retains all four ADC pairs.
            decoder.drain( c.m_uncorrelated ? unrelated : source );
            const auto sample   = decoder.status().m_latest;
            auto       r        = raw( contract,
                                       DDSPC::ObservationSource::Adc,
                                       sample.m_frame.m_counter,
                                       sample.m_acquiredNs,
                                       sample.m_receivedNs );
            r.m_raw             = sample.m_frame.m_raw;
            r.m_values          = sample.m_acceleration;
            r.m_acquiredRawNs   = sample.m_frame.m_acquiredRealNs;
            r.m_publishedRawNs  = sample.m_frame.m_writtenRealNs;
            r.m_publishedMonoNs = sample.m_publishedNs;
            r.m_segment         = sample.m_segment;
            if( c.m_upstream )
                r.m_acquiredMonoNs = r.m_publishedMonoNs = -1;
            if( c.m_futureAdc )
                r.m_receivedNs = r.m_availableNs = t + 300000;
            if( c.m_duplicateChannels )
                r.m_values[1] = r.m_values[0];
            engine.observe( r );
        }
        for( unsigned a = 0; a < 2; ++a )
        {
            y[a] = .4 * y[a] + .8 * previousForce[a] + noise( random );
            for( unsigned s = 0; s < 4; ++s )
                y[a] += ( .1 + .05 * s ) * previousInputs[2 * s + a];
        }
        auto modal = raw( contract, DDSPC::ObservationSource::Modal, frame + 1, t, t + 100000 );
        for( unsigned a = 0; a < 2; ++a )
            modal.m_values[a] = float( y[a] + ( c.m_changeWfsAt && frame + 1 >= c.m_changeWfsAt ? 1.5 : 0 ) );
        engine.observe( modal );
        if( c.m_upstream && havePendingCommands )
        {
            DDSPC::OnlineCommandFrame complete;
            if( !DDSPC::synchronized_commands( contract,
                                               pendingCommands.m_wfsCounter,
                                               pendingCommands.m_stamp.m_counter,
                                               pendingProjection,
                                               pendingCommands.m_sampledNs,
                                               modal,
                                               complete ) )
                throw std::runtime_error( "Synthetic request adapter failed" );
            if( !c.m_appliedEvidence )
                complete.m_evidence = DDSPC::InputEvidence::Unverified;
            numerical( [&] { engine.commands( complete, t + 101000 ); } );
        }
        if( !c.m_upstream )
            engine.observe( raw( contract, DDSPC::ObservationSource::Camera, frame + 1, t, t + 30000 ) );
        auto observeRequest = [&]( unsigned index )
        {
            auto                       request = raw( contract,
                                                      DDSPC::ObservationSource::Dm,
                                                      4 * frame + index + 1,
                                                      t + index * slot,
                                                      t + index * slot + 1000 );
            const std::array<float, 2> payload{ inputs[2 * index], inputs[2 * index + 1] };
            request.m_elements    = 2;
            request.m_payloadHash = DDSPC::command_payload_hash( payload );
            engine.observe( request );
            if( c.m_upstream && !DDSPC::project_request(
                                    contract, request, payload, DDSPC::Matrix::Identity( 2, 2 ), projected[index] ) )
                throw std::runtime_error( "Synthetic projection adapter failed" );
        };
        for( unsigned s = 0; s < 2; ++s )
            observeRequest( s );
        DDSPC::OnlineOrigin o;
        o.m_stamp           = { 1, frame + 1, 1, 1, t + 50000, t + 100000, 20, true };
        o.m_cameraCounter   = frame + 1;
        o.m_residual        = { float( modal.m_values[0] ), float( modal.m_values[1] ) };
        o.m_exposureStartNs = t - 400000;
        o.m_exposureEndNs = o.m_gridNs = t;
        o.m_decisionNs                 = t + 200000;
        o.m_prefixSlots                = 2;
        o.m_prefix                     = inputs;
        o.m_prefixAvailableNs          = t + 150000;
        o.m_prefixVerified             = true;
        for( unsigned s = 0; s < 8; ++s )
            o.m_cutoffsNs[s] = t + s * slot - 20000;
        if( c.m_upstream )
        {
            o.m_cameraCounter = 0; // No camera parent/exposure is invented for synchronized modal measurements.
            const DDSPC::UpstreamStatus upstream{
                77,
                !c.m_timerMode,
                !( c.m_syncLossCount && frame >= c.m_syncLossStart && frame - c.m_syncLossStart < c.m_syncLossCount ),
                t + 150000 };
            if( !DDSPC::synchronized_origin( contract, upstream, modal, t + 200000, t + 700000, o ) )
                throw std::runtime_error( "Synthetic synchronized origin adapter failed" );
            if( c.m_wrongPhase )
                ++o.m_firstAdcCounter;
        }
        numerical(
            [&]
            {
                engine.origin( o, o.m_decisionNs );
                engine.finish_origin( o.m_decisionNs + 1000 );
            } ); // Virtual completion; CPU time is measured separately below.
        const auto status = engine.status();
        result.m_candidates += status.m_candidateValid;
        result.m_readiness.push_back( { status.m_standard.m_ready,
                                        status.m_assisted.m_ready,
                                        status.m_standard.m_rank,
                                        status.m_assisted.m_rank } );
        for( unsigned s = 0; s < 4; ++s )
        {
            if( s >= 2 )
                observeRequest( s );
            DDSPC::OnlineAppliedSlot u;
            u.m_startNs   = t + s * slot;
            u.m_endNs     = u.m_startNs + slot;
            u.m_stamp     = { 9, 4 * frame + s + 1, 1, 1, u.m_endNs + 1000, u.m_endNs + 2000, 20, true };
            u.m_dmCounter = u.m_stamp.m_counter;
            u.m_values    = { inputs[2 * s], inputs[2 * s + 1] };
            u.m_evidence  = c.m_appliedEvidence ? contract.m_inputs.m_evidence : DDSPC::InputEvidence::Unverified;
            if( !c.m_upstream )
                numerical( [&] { engine.applied( u, std::max( o.m_decisionNs + 1001, u.m_stamp.m_receivedNs ) ); } );
        }
        if( c.m_upstream )
        {
            DDSPC::OnlineCommandFrame u;
            u.m_wfsCounter = frame + 1;
            u.m_stamp      = { 9, frame + 1, 1, 1, t + period + 30000, t + period + 31000, 20, true };
            u.m_values     = inputs;
            u.m_complete   = true;
            u.m_evidence   = c.m_appliedEvidence ? contract.m_inputs.m_evidence : DDSPC::InputEvidence::Unverified;
            for( unsigned s = 0; s < 4; ++s )
            {
                u.m_counters[s]  = 4 * frame + s + 1;
                u.m_hashes[s]    = u.m_counters[s] + 1000;
                u.m_sampledNs[s] = t + s * slot + 1000;
            }
            pendingCommands     = u;
            pendingProjection   = projected;
            havePendingCommands = true;
        }
        previousInputs = inputs;
        previousForce  = force;
        result.m_numericalUs.push_back( numericalUs );
        result.m_maxJobUs.push_back( maxJobUs );
        result.m_computeUs.push_back(
            std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - begin ).count() );
    }
    result.m_status   = engine.status();
    result.m_adc      = decoder.status();
    result.m_standard = engine.coefficients( DDSPC::ShadowMode::Standard );
    result.m_assisted = engine.coefficients( DDSPC::ShadowMode::AccelAssisted );
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
