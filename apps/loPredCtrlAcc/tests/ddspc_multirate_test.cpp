/** \file ddspc_multirate_test.cpp
 * \brief Actual native multirate DDSPC identification, provenance, worker and non-actuation regression tests.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../multirate_shadow.hpp"
#include "../online_shadow.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond MULTIRATE_TEST_SUPPORT
namespace
{
/// Small, explicitly synthetic physical timing/provenance fixture; no stream, device or operational library.
class MultirateFixture
{
  public:
    DDSPC::OnlineShadowConfig m_config;   ///< Fixed bounded native numerical profile.
    DDSPC::MultirateContract  m_contract; ///< Synthetic certificates are accepted only by the explicit test instance.
    std::mt19937              m_random{ 73 };              ///< Exogenous inputs independent of every model proposal.
    std::uniform_real_distribution<float> m_draw{ -1, 1 }; ///< Finite independent persistent excitation.
    float                                 m_y{ 0 }, m_lastA{ 0 }; ///< Causal synthetic measured plant state.
    std::array<float, 4>                  m_previousU{};          ///< Actual prior inputs, never a shadow proposal.
    DDSPC::OnlineShadowStatus m_atOrigin;         ///< Snapshot before actual input confirmation and deadline expiry.
    DDSPC::Matrix m_standardPlan, m_assistedPlan; ///< Original independent diagnostic plans, not plant inputs.
    int           m_prefixSlots{ 2 };             ///< Explicit schedule commitment in this synthetic fixture.
    std::int64_t  m_adcJitterNs{ 0 }, m_adcDelayNs{ 0 }; ///< Acquisition jitter and delivery delay, kept distinct.
    bool m_prefixVerified{ true };    ///< Missing prefix evidence must inhibit prediction independently of training.
    std::int64_t m_computeNs{ 1000 }; ///< Synthetic completion time for deadline regression only.
    unsigned     m_interruptAdcAtOrigin{ 0 }; ///< Disconnect or reverse acquisition time after solving, before cutoff.

    /// Configure source/time evidence with known discrete dynamics and all four physical slot identities.
    MultirateFixture();

    /// Construct an original raw source snapshot with explicit source-host generation and local availability.
    DDSPC::NativeObservation raw( DDSPC::ObservationSource source /**< [in] Original source. */,
                                  std::uint64_t            counter /**< [in] Source sequence. */,
                                  std::int64_t available /**< [in] Established local availability. */ ) const;

    /// Feed one chronological physical frame and its independently confirmed actual input slots.
    bool frame( DDSPC::MultirateShadow &engine /**< [in,out] Actual numerical implementation. */,
                int                     index /**< [in] Chronological frame. */,
                bool                    adc  = true /**< [in] Deliver individual ADC samples. */,
                bool                    held = false /**< [in] Hold one actual command over all four slots. */ );
};

MultirateFixture::MultirateFixture()
{
    auto &m                  = m_config.m_model;
    m.m_modes                = 1;
    m.m_history              = 2;
    m.m_future               = 1;
    m.m_accelHistory         = 1;
    m.m_prior                = .001f;
    m.m_lambda               = .1f;
    m.m_useQrd               = true;
    m.m_standardCommandSlots = 4; // Matched command space for a paired multirate comparison; production defaults one.
    auto &c                  = m_contract.m_inputs;
    c.m_modes                = 1;
    c.m_periodNs             = 500000;
    c.m_exposureNs           = 400000;
    c.m_jitterNs             = 2000;
    c.m_maxUncertaintyNs     = 10;
    c.m_maxAgeNs             = 5000000;
    c.m_wfsGeneration        = 1;
    c.m_commandGeneration    = 9;
    c.m_clockEpoch = c.m_layout = 1;
    c.m_timingVerified = c.m_mappingVerified = true;
    c.m_evidence                             = DDSPC::InputEvidence::SyntheticApplied;
    m_contract.m_clockVerified = m_contract.m_exposureVerified = m_contract.m_associationVerified =
        m_contract.m_scheduleVerified                          = true;
    m_contract.m_provenance = "isolated synthetic timing/plant; not hardware evidence";
    for( int i = 0; i < 4; ++i )
        m_contract.m_sources[i] = { 11, std::uint64_t( 100 + i ), i ? std::uint64_t( 1000 + i ) : 0, 1 };
}
DDSPC::NativeObservation
MultirateFixture::raw( DDSPC::ObservationSource source, std::uint64_t counter, std::int64_t available ) const
{
    DDSPC::NativeObservation r;
    const auto              &binding = m_contract.m_sources[std::size_t( source )];
    r.m_source                       = source;
    r.m_device                       = binding.m_device;
    r.m_fileInode                    = binding.m_inode;
    r.m_imageInode                   = binding.m_image;
    r.m_generation                   = binding.m_generation;
    r.m_segment                      = 1;
    r.m_counter                      = counter;
    r.m_availableNs = r.m_receivedNs = available;
    r.m_acquiredMonoNs               = available - 2000;
    r.m_acquiredRawNs                = 1800000000000000000LL + r.m_acquiredMonoNs;
    r.m_publishedRawNs               = r.m_acquiredRawNs + 1000;
    return r;
}
bool MultirateFixture::frame( DDSPC::MultirateShadow &engine, int index, bool adc, bool held )
{
    const std::int64_t   t = 1000000000LL + std::int64_t( index ) * 500000;
    std::array<float, 4> commands;
    for( auto &u : commands )
        u = m_draw( m_random );
    if( held )
        commands.fill( commands[0] );
    float currentA = 0;
    for( int s = 0; s < 4; ++s )
    {
        auto r            = raw( DDSPC::ObservationSource::Adc, 4 * index + s + 1, t - 500000 + s * 125000 + 2000 );
        r.m_values        = { m_draw( m_random ), m_draw( m_random ) };
        const auto jitter = s % 2 ? m_adcJitterNs : -m_adcJitterNs;
        r.m_acquiredMonoNs += jitter;
        r.m_acquiredRawNs += jitter;
        r.m_publishedRawNs += jitter;
        r.m_receivedNs += m_adcDelayNs;
        r.m_availableNs += m_adcDelayNs;
        r.m_raw = { std::uint16_t( 2048 + 50 * r.m_values[0] ), std::uint16_t( 2048 + 50 * r.m_values[1] ) };
        if( s == 3 )
            currentA = r.m_values[0];
        if( adc )
            engine.observe( r );
    }
    m_y = .4f * m_y + .6f * m_lastA + .01f * m_draw( m_random );
    for( int s = 0; s < 4; ++s )
        m_y += ( .1f + .05f * s ) * m_previousU[s];
    auto modal     = raw( DDSPC::ObservationSource::Modal, index + 1, t + 100000 );
    modal.m_values = { m_y, 0 };
    engine.observe( modal );
    engine.observe( raw( DDSPC::ObservationSource::Camera, index + 1, t + 30000 ) );
    for( int s = 0; s < 2; ++s )
        engine.observe( raw( DDSPC::ObservationSource::Dm, 4 * index + s + 1, t + s * 125000 + 1000 ) );
    DDSPC::OnlineOrigin o;
    o.m_stamp             = { 1, std::uint64_t( index + 1 ), 1, 1, t + 50000, t + 100000, 10, true };
    o.m_cameraCounter     = index + 1;
    o.m_residual          = { m_y, 0 };
    o.m_gridNs            = t;
    o.m_exposureStartNs   = t - 400000;
    o.m_exposureEndNs     = t;
    o.m_decisionNs        = t + 200000;
    o.m_prefixSlots       = m_prefixSlots;
    o.m_prefix            = { commands[0], commands[1], commands[2], commands[3] };
    o.m_prefixAvailableNs = t + 150000;
    o.m_prefixVerified    = m_prefixVerified;
    for( int s = 0; s < 8; ++s )
        o.m_cutoffsNs[s] = t + s * 125000 - 20000;
    const bool accepted = engine.origin( o, o.m_decisionNs );
    engine.finish_origin( o.m_decisionNs + m_computeNs ); // Virtual time, never native deadline certification.
    if( m_interruptAdcAtOrigin )
    {
        auto loss    = raw( DDSPC::ObservationSource::Adc, 4 * index + 4, o.m_decisionNs + m_computeNs );
        loss.m_event = DDSPC::ObservationEvent::Disconnected;
        if( m_interruptAdcAtOrigin == 2 )
        {
            loss.m_event = DDSPC::ObservationEvent::Sample;
            ++loss.m_counter;
            loss.m_acquiredMonoNs -= 1000000000;
        }
        engine.observe( loss );
    }
    m_atOrigin = engine.status();
    m_standardPlan.resize( 0, 0 );
    m_assistedPlan.resize( 0, 0 );
    engine.candidate( DDSPC::ShadowMode::Standard, m_standardPlan );
    engine.candidate( DDSPC::ShadowMode::AccelAssisted, m_assistedPlan );
    for( int s = 0; s < 4; ++s )
    {
        if( s >= 2 )
            engine.observe( raw( DDSPC::ObservationSource::Dm, 4 * index + s + 1, t + s * 125000 + 1000 ) );
        DDSPC::OnlineAppliedSlot u;
        u.m_startNs   = t + s * 125000;
        u.m_endNs     = u.m_startNs + 125000;
        u.m_values    = { commands[s], 0 };
        u.m_dmCounter = 4 * index + s + 1;
        u.m_stamp     = { 9, u.m_dmCounter, 1, 1, u.m_endNs + 1000, u.m_endNs + 2000, 10, true };
        u.m_evidence  = DDSPC::InputEvidence::SyntheticApplied;
        if( !engine.applied( u, std::max( o.m_decisionNs + m_computeNs + 1, u.m_stamp.m_receivedNs ) ) )
            return false;
    }
    m_lastA     = currentA;
    m_previousU = commands;
    return accepted;
}
} // namespace
/// \endcond

/** \brief Four command slots form a rectangular future block while physical WFS targets remain at two kHz. */
TEST_CASE( "Native lifted DDSPC indexing is rectangular and retains every ADC regressor", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::get_training_example;
    DDSPC::PredictiveController::get_reference_controller;
#endif
    // clang-format on
    DDSPC::PredictiveController c( 2, 2, 2, 0, 1, .1f, 1, 8, 1, DDSPC::HistoryMode::Causal, 4 );
    c.use_qrd       = GENERATE( false, true );
    auto          w = c.make_workspace();
    DDSPC::Matrix y( 2, 1 ), u( 8, 1 ), a( 8, 1 );
    for( int i = 0; i < 5; ++i )
    {
        for( int m = 0; m < 2; ++m )
            y( m ) = 100 * i + m;
        for( int s = 0; s < 8; ++s )
        {
            u( s ) = 1000 * i + s;
            a( s ) = 10000 * i + s;
        }
        c.append_sample( i, y, u, a );
    }
    REQUIRE( c.prepare_training( *w ) );
    REQUIRE( w->target().rows() == 4 );
    REQUIRE( w->regressor().rows() == 36 );
    REQUIRE( w->regressor()( 0 ) == 3000 ); // Future newest frame i=3, eight distinct slot/mode values.
    REQUIRE( w->regressor()( 15 ) == 2007 );
    REQUIRE( w->regressor()( 16 ) == 1000 ); // Past input, eight columns rather than two.
    REQUIRE( w->regressor()( 24 ) == 20000 );
    REQUIRE( w->regressor()( 32 ) == 200 );
    REQUIRE( c.update_system( *w ) );
    const auto gain = c.get_reference_controller( *w, .1f );
    REQUIRE( gain.rows() == 16 );
    REQUIRE( gain.cols() == 20 );
    REQUIRE( w->theta().rows() == 4 );
    REQUIRE( w->theta().cols() == 36 );
    REQUIRE( gain.allFinite() );
}

/** \brief Both actual native models identify the same command path; only assisted state contains individual ADCs. */
TEST_CASE( "Qualified native DDSPC trains independent models and scores frozen causal origins", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture f;
    f.m_config.m_model.m_useQrd = GENERATE( false, true );
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    for( int i = 0; i < 600; ++i )
        REQUIRE( f.frame( engine, i ) );
    const auto s = engine.status();
    INFO( s.m_reason );
    REQUIRE( s.m_records == 600 );
    REQUIRE( s.m_standard.m_ready );
    REQUIRE( s.m_assisted.m_ready );
    REQUIRE( s.m_standard.m_rank == s.m_standard.m_features );
    REQUIRE( s.m_assisted.m_rank == s.m_assisted.m_features );
    REQUIRE( s.m_assisted.m_features == s.m_standard.m_features + 8 );
    REQUIRE( s.m_standard.m_updates > 500 );
    REQUIRE( s.m_assisted.m_updates > 500 );
    REQUIRE( s.m_assisted.m_predictions > 400 );
    REQUIRE( s.m_assisted.m_rmse < .15 * s.m_standard.m_rmse );
    REQUIRE( engine.coefficients( DDSPC::ShadowMode::Standard ).allFinite() );
    REQUIRE( engine.coefficients( DDSPC::ShadowMode::AccelAssisted ).allFinite() );
    const auto theta = engine.coefficients( DDSPC::ShadowMode::AccelAssisted );
    REQUIRE( theta( 0, 0 ) == Approx( .4 ).margin( .02 ) );
    REQUIRE( theta( 0, 6 ) == Approx( .6 ).margin( .02 ) );
    for( int s = 0; s < 4; ++s )
        REQUIRE( theta( 0, theta.cols() - 4 + s ) == Approx( .25 - .05 * s ).margin( .02 ) );
}

/** \brief Source clocks, exposure origins and physical inputs cannot be enabled by observing requests/SDK markers. */
TEST_CASE( "Native DDSPC retains timing and physical-command interlocks", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture       f;
    DDSPC::MultirateShadow engine( f.m_config );
    REQUIRE_FALSE(
        engine.qualify( f.m_contract ) ); // Explicit synthetic evidence cannot qualify a production instance.
    REQUIRE( engine.status().m_standard.m_updates == 0 );
    for( int s = 0; s < 4; ++s )
    {
        auto attach    = f.raw( DDSPC::ObservationSource( s ), 1, 1000000000 );
        attach.m_event = DDSPC::ObservationEvent::Attached;
        engine.observe( attach );
        REQUIRE( std::string( engine.status().m_reason ) ==
                 "applied_input_provenance_or_modal_projection_unqualified" );
    }
    f.m_contract.m_inputs.m_evidence = DDSPC::InputEvidence::VerifiedApplied;
    const int missing                = GENERATE( 0, 1, 2, 3 );
    if( missing == 0 )
        f.m_contract.m_clockVerified = false;
    if( missing == 1 )
        f.m_contract.m_exposureVerified = false;
    if( missing == 2 )
        f.m_contract.m_associationVerified = false;
    if( missing == 3 )
        f.m_contract.m_inputs.m_evidence = DDSPC::InputEvidence::DriverAccepted;
    REQUIRE_FALSE( engine.qualify( f.m_contract ) );
    REQUIRE_FALSE( engine.status().m_qualified );
    for( int i = 0; i < 10; ++i )
        REQUIRE_FALSE( f.frame( engine, i ) );
    REQUIRE( engine.status().m_standard.m_updates == 0 );
    REQUIRE( engine.status().m_assisted.m_updates == 0 );
}
/** \brief Sensor loss affects only assisted histories; standard weights and update counts are bit-identical. */
TEST_CASE( "Multirate model independence fallback and recovery retain the standard learner", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture       reference, experiment;
    DDSPC::MultirateShadow baseline( reference.m_config, DDSPC::InputEvidence::SyntheticApplied );
    DDSPC::MultirateShadow controller( experiment.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( baseline.qualify( reference.m_contract ) );
    REQUIRE( controller.qualify( experiment.m_contract ) );
    controller.request( DDSPC::ShadowMode::AccelAssisted );
    for( int i = 0; i < 350; ++i )
    {
        REQUIRE( reference.frame( baseline, i ) );
        REQUIRE( experiment.frame( controller, i, i < 200 || i >= 220 ) );
        if( i == 180 )
        {
            REQUIRE( experiment.m_atOrigin.m_candidateValid );
            REQUIRE( experiment.m_atOrigin.m_effective == DDSPC::ShadowMode::AccelAssisted );
            REQUIRE( experiment.m_assistedPlan.size() == 4 );
            REQUIRE( experiment.m_assistedPlan.bottomRows( 2 ) == experiment.m_standardPlan.bottomRows( 2 ) );
        }
        if( i == 210 )
        {
            REQUIRE( experiment.m_atOrigin.m_candidateValid );
            REQUIRE( experiment.m_atOrigin.m_effective == DDSPC::ShadowMode::Standard );
            REQUIRE_FALSE( controller.status().m_assisted.m_ready );
        }
        if( i == 300 )
        {
            REQUIRE( experiment.m_atOrigin.m_effective == DDSPC::ShadowMode::AccelAssisted );
            controller.request( DDSPC::ShadowMode::Standard );
        }
    }
    REQUIRE( baseline.coefficients( DDSPC::ShadowMode::Standard ) ==
             controller.coefficients( DDSPC::ShadowMode::Standard ) );
    REQUIRE( baseline.status().m_standard.m_updates == controller.status().m_standard.m_updates );
    REQUIRE( experiment.m_atOrigin.m_effective == DDSPC::ShadowMode::Standard );
    REQUIRE( controller.status().m_assisted.m_ready );
}

/** \brief ADC loss revokes a previously solved assisted plan immediately, preserving the independent standard plan. */
TEST_CASE( "Native ADC loss after prediction cannot retain an assisted candidate", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture       f;
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    engine.request( DDSPC::ShadowMode::AccelAssisted );
    for( int i = 0; i < 150; ++i )
        REQUIRE( f.frame( engine, i ) );
    REQUIRE( f.m_atOrigin.m_effective == DDSPC::ShadowMode::AccelAssisted );
    f.m_interruptAdcAtOrigin = GENERATE( 1, 2 );
    REQUIRE( f.frame( engine, 150 ) );
    REQUIRE( f.m_atOrigin.m_candidateValid );
    REQUIRE( f.m_atOrigin.m_effective == DDSPC::ShadowMode::Standard );
    REQUIRE( f.m_atOrigin.m_standard.m_ready );
    REQUIRE_FALSE( f.m_atOrigin.m_assisted.m_ready );
    REQUIRE( f.m_assistedPlan.size() == 0 );
}

/** \brief Four slots are selected by acquisition time with bounded jitter; late receipts never enter origin features.
 */
TEST_CASE( "Multirate timing rejects future deliveries and ambiguous phases", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture f;
    const int        scenario = GENERATE( 0, 1, 2 );
    if( scenario == 0 )
        f.m_adcJitterNs = 400;
    if( scenario == 1 )
        f.m_adcJitterNs = 10000;
    if( scenario == 2 )
        f.m_adcDelayNs = 1000000;
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    for( int i = 0; i < 150; ++i )
        REQUIRE( f.frame( engine, i ) );
    REQUIRE( engine.status().m_standard.m_ready );
    REQUIRE( engine.status().m_assisted.m_ready == ( scenario == 0 ) );
    if( scenario != 0 )
        REQUIRE( engine.status().m_assisted.m_updates == 0 );
}

/** \brief Computation deadlines and immutable-prefix evidence gate candidates, never the independent training truth. */
TEST_CASE( "Multirate prediction respects prefix evidence and actual completion time", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture f;
    const bool       late = GENERATE( false, true );
    f.m_prefixVerified    = late;
    if( late )
        f.m_computeNs = 40000; // First free slot cutoff is origin+30 us.
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    for( int i = 0; i < 100; ++i )
        REQUIRE( f.frame( engine, i ) );
    REQUIRE( engine.status().m_standard.m_ready );
    REQUIRE( engine.status().m_assisted.m_ready );
    REQUIRE_FALSE( f.m_atOrigin.m_candidateValid );
    REQUIRE( engine.status().m_assisted.m_predictions == 0 );
}

/** \brief Two-kHz held inputs qualify the original standard shape but cannot identify four independent input phases. */
TEST_CASE( "Single-rate standard training remains distinct from unexcited lifted control", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture f;
    f.m_config.m_model.m_standardCommandSlots = 1;
    f.m_config.m_model.m_future               = 2;
    f.m_prefixSlots                           = 4;
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    engine.request( DDSPC::ShadowMode::AccelAssisted );
    for( int i = 0; i < 200; ++i )
        REQUIRE( f.frame( engine, i, true, true ) );
    REQUIRE( engine.status().m_standard.m_ready );
    REQUIRE_FALSE( engine.status().m_assisted.m_ready );
    REQUIRE( engine.status().m_assisted.m_reason == DDSPC::ShadowReason::Unexcited );
    REQUIRE( f.m_atOrigin.m_candidateValid );
    REQUIRE( f.m_atOrigin.m_effective == DDSPC::ShadowMode::Standard );
    REQUIRE( f.m_standardPlan.size() == 2 );
}

/** \brief Raw stream observations/SDK success alone never train; the separate worker has bounded nonblocking queues. */
TEST_CASE( "Native worker remains inhibited without physical provenance and bounds overload", "[multirate][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::OnlineShadowWorker::start;
    DDSPC::OnlineShadowWorker::observe;
    DDSPC::OnlineShadowWorker::stop;
#endif
    // clang-format on
    MultirateFixture f;
    f.m_config.m_queueCapacity = 16;
    DDSPC::OnlineShadowWorker worker( f.m_config );
    for( unsigned i = 0; i < 80; ++i )
        worker.observe( f.raw( DDSPC::ObservationSource::Adc, i + 1, 1000000 + i * 125000 ) );
    REQUIRE_FALSE( worker.qualify( f.m_contract ) );
    worker.start();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( !worker.status().m_running && std::chrono::steady_clock::now() < until )
        std::this_thread::yield();
    REQUIRE( worker.status().m_running );
    REQUIRE( worker.status().m_drops >= 64 );
    REQUIRE( worker.status().m_rawQueued <= 16 );
    REQUIRE_FALSE( worker.status().m_qualified );
    REQUIRE( worker.status().m_standard.m_updates == 0 );
    REQUIRE( worker.status().m_assisted.m_updates == 0 );
    DDSPC::Matrix plan;
    REQUIRE_FALSE( worker.candidate( DDSPC::ShadowMode::Standard, plan ) );
    worker.stop();
    REQUIRE_FALSE( worker.status().m_running );
    f.m_config.m_model.m_history = 100;
    REQUIRE_THROWS_AS( DDSPC::OnlineShadowWorker( f.m_config ), std::invalid_argument );
    f.m_config.m_model.m_history = f.m_config.m_model.m_future = std::numeric_limits<int>::max();
    REQUIRE_THROWS_AS( DDSPC::OnlineShadowWorker( f.m_config ), std::invalid_argument );
}
/** \brief Actual input-clock restarts revoke qualification, while common input silence breaks readiness. */
TEST_CASE( "Native multirate clock epochs and source restarts cannot reuse ready models", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture       f;
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    for( int i = 0; i < 80; ++i )
        REQUIRE( f.frame( engine, i ) );
    REQUIRE( engine.status().m_standard.m_ready );
    const bool restart = GENERATE( false, true );
    if( restart )
    {
        auto r = f.raw( DDSPC::ObservationSource::Camera, 1, 2000000000 );
        ++r.m_generation;
        engine.observe( r );
        REQUIRE_FALSE( engine.status().m_qualified );
        REQUIRE_FALSE( engine.status().m_clock );
    }
    else
        engine.advance( 2000000000 );
    REQUIRE_FALSE( engine.status().m_standard.m_ready );
    REQUIRE_FALSE( engine.status().m_assisted.m_ready );
    DDSPC::Matrix proposal;
    REQUIRE_FALSE( engine.candidate( DDSPC::ShadowMode::Standard, proposal ) );
    if( !restart )
    {
        engine.advance( 1000000000 );
        REQUIRE_FALSE( engine.status().m_qualified );
    }
}

/** \brief Rank/law refresh and RLS cadence can be bounded without thinning WFS or raw ADC histories. */
TEST_CASE( "Multirate work stride retains every qualified record", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    MultirateFixture f;
    f.m_config.m_model.m_updateStride = 4;
    DDSPC::MultirateShadow engine( f.m_config, DDSPC::InputEvidence::SyntheticApplied );
    REQUIRE( engine.qualify( f.m_contract ) );
    for( int i = 0; i < 300; ++i )
        REQUIRE( f.frame( engine, i ) );
    const auto s = engine.status();
    REQUIRE( s.m_records == 300 );
    REQUIRE( s.m_raw == 300 * 10 ); // Four ADC + WFS + camera + four separate DM publication observations.
    REQUIRE( s.m_standard.m_samples == 300 );
    REQUIRE( s.m_assisted.m_samples == 300 );
    REQUIRE( s.m_assisted.m_updates > 60 );
    REQUIRE( s.m_assisted.m_updates < 80 );
    REQUIRE( s.m_assisted.m_ready );
}

/** \brief Local numerical throughput is reported separately from source provenance and 125-us deadlines. */
TEST_CASE( "Native DDSPC bounded default-dimension computation remains finite", "[multirate][performance]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateShadow::origin;
    DDSPC::MultirateShadow::applied;
    DDSPC::MultirateShadow::observe;
#endif
    // clang-format on
    DDSPC::ShadowConfig c;
    c.m_modes                = 2;
    c.m_history              = 5;
    c.m_future               = 3;
    c.m_accelChannels        = 8;
    c.m_accelHistory         = 5;
    c.m_standardCommandSlots = c.m_assistedCommandSlots = 4;
    c.m_externalOrigins                                 = true;
    c.m_updateStride                                    = 4;
    c.m_useQrd                                          = true;
    c.m_prior                                           = .001f;
    DDSPC::ShadowController models( c );
    models.training( DDSPC::ShadowMode::Standard, true );
    models.training( DDSPC::ShadowMode::AccelAssisted, true );
    std::mt19937                    random( 91 );
    std::normal_distribution<float> draw;
    DDSPC::ShadowRecord             r;
    r.m_wfsValid = r.m_appliedValid = r.m_assistedAppliedValid = r.m_timingValid = r.m_accelValid = true;
    r.m_residual.resize( 2, 1 );
    r.m_appliedCommand.resize( 8, 1 );
    r.m_acceleration.resize( 8, 1 );
    const auto begin   = std::chrono::steady_clock::now();
    double     maximum = 0;
    for( int i = 0; i < 1200; ++i )
    {
        for( auto *v : { &r.m_residual, &r.m_appliedCommand, &r.m_acceleration } )
            for( int j = 0; j < v->size(); ++j )
                ( *v )( j ) = draw( random );
        r.m_assistedAppliedCommand = r.m_appliedCommand;
        r.m_index                  = i;
        const auto start           = std::chrono::steady_clock::now();
        REQUIRE( models.process( r ) );
        const double elapsed =
            std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count();
        maximum = std::max( maximum, elapsed );
    }
    const double average =
        std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - begin ).count() / 1200;
    std::cout << "native_multirate_core average_us=" << average << " max_us=" << maximum
              << " wfs_records=1200 adc_regressors_per_frame=8 stride=4 deadline_qualified=false\n";
    REQUIRE( models.status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( models.status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( models.status( DDSPC::ShadowMode::AccelAssisted ).m_features == 106 );
    REQUIRE( models.prediction_matrix( DDSPC::ShadowMode::AccelAssisted ).allFinite() );
}
} // namespace libXWCTest::loPredCtrlAccTest
