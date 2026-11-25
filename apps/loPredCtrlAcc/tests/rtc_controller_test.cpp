/** \file rtc_controller_test.cpp
 * \brief Operational WFS DDSPC correction, independent ACC fitting, and original INDI workflow with private inputs.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc_operational.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <filesystem>
#include <fstream>
#include <functional>
#include <thread>
#include <unistd.h>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond RTC_TEST_SUPPORT
namespace
{
/// Real application numerical/control methods with a private calibrated four-pixel DM and explicit mock receipts.
class Controller : public MagAOX::app::loPredCtrlAcc_operational
{
  public:
    // The WFS monitor's members, named as the shmimMonitor base also shared with the modal frameGrabber.
    using MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc_operational>::m_height;
    using MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc_operational>::m_imageStream;
    using MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc_operational>::m_shmimName;
    using MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc_operational>::m_width;
    using loPredCtrlAcc_operational::allocate;
    using loPredCtrlAcc_operational::appliedSlots;
    using loPredCtrlAcc_operational::appLogic;
    using loPredCtrlAcc_operational::appShutdown;
    using loPredCtrlAcc_operational::appStartup;
    using loPredCtrlAcc_operational::configureAcquisition;
    using loPredCtrlAcc_operational::copied;
    using loPredCtrlAcc_operational::fastSample;
    using loPredCtrlAcc_operational::integrator;
    using loPredCtrlAcc_operational::loadCalibration;
    using loPredCtrlAcc_operational::loadImageIntoStream;
    using loPredCtrlAcc_operational::m_accelEnabled;
    using loPredCtrlAcc_operational::m_accFilled;
    using loPredCtrlAcc_operational::m_adcAnchor;
    using loPredCtrlAcc_operational::m_adcSegment;
    using loPredCtrlAcc_operational::m_assistedDiagnostic;
    using loPredCtrlAcc_operational::m_assistedValid;
    using loPredCtrlAcc_operational::m_calibration;
    using loPredCtrlAcc_operational::m_candidateValid;
    using loPredCtrlAcc_operational::m_commandKnown;
    using loPredCtrlAcc_operational::m_commandMutex;
    using loPredCtrlAcc_operational::m_decisionOrdinal;
    using loPredCtrlAcc_operational::m_exploration;
    using loPredCtrlAcc_operational::m_fastPrefix;
    using loPredCtrlAcc_operational::m_fastSamples;
    using loPredCtrlAcc_operational::m_fastStats;
    using loPredCtrlAcc_operational::m_fillHoldoff;
    using loPredCtrlAcc_operational::m_gain;
    using loPredCtrlAcc_operational::m_gainFront;
    using loPredCtrlAcc_operational::m_gainSets;
    using loPredCtrlAcc_operational::m_indiP_control;
    using loPredCtrlAcc_operational::m_indiP_exploration;
    using loPredCtrlAcc_operational::m_indiP_gain;
    using loPredCtrlAcc_operational::m_indiP_integrator;
    using loPredCtrlAcc_operational::m_indiP_learning;
    using loPredCtrlAcc_operational::m_indiP_learnStd;
    using loPredCtrlAcc_operational::m_indiP_learnStdExtra;
    using loPredCtrlAcc_operational::m_indiP_loopGain;
    using loPredCtrlAcc_operational::m_indiP_mode;
    using loPredCtrlAcc_operational::m_indiP_predicting;
    using loPredCtrlAcc_operational::m_indiP_qualification;
    using loPredCtrlAcc_operational::m_indiP_releaseTT;
    using loPredCtrlAcc_operational::m_indiP_reset;
    using loPredCtrlAcc_operational::m_indiP_sync;
    using loPredCtrlAcc_operational::m_indiP_zeroDm;
    using loPredCtrlAcc_operational::m_inputValid;
    using loPredCtrlAcc_operational::m_integrate;
    using loPredCtrlAcc_operational::m_integratorReady;
    using loPredCtrlAcc_operational::m_intState;
    using loPredCtrlAcc_operational::m_invalidReason;
    using loPredCtrlAcc_operational::m_invalidWindow;
    using loPredCtrlAcc_operational::m_learning;
    using loPredCtrlAcc_operational::m_learnStdExtra;
    using loPredCtrlAcc_operational::m_learnUs;
    using loPredCtrlAcc_operational::m_mainLoop;
    using loPredCtrlAcc_operational::m_maxFill;
    using loPredCtrlAcc_operational::m_modalEngaged;
    using loPredCtrlAcc_operational::m_modalFresh;
    using loPredCtrlAcc_operational::m_modalLock;
    using loPredCtrlAcc_operational::m_mode;
    using loPredCtrlAcc_operational::m_modeCube;
    using loPredCtrlAcc_operational::m_modelMutex;
    using loPredCtrlAcc_operational::m_models;
    using loPredCtrlAcc_operational::m_multirate;
    using loPredCtrlAcc_operational::m_output;
    using loPredCtrlAcc_operational::m_outputThread;
    using loPredCtrlAcc_operational::m_pending;
    using loPredCtrlAcc_operational::m_planBase;
    using loPredCtrlAcc_operational::m_predicting;
    using loPredCtrlAcc_operational::m_ratio;
    using loPredCtrlAcc_operational::m_renderLeadNs;
    using loPredCtrlAcc_operational::m_ripple;
    using loPredCtrlAcc_operational::m_settings;
    using loPredCtrlAcc_operational::m_slotPlan;
    using loPredCtrlAcc_operational::m_slots;
    using loPredCtrlAcc_operational::m_slotShifts;
    using loPredCtrlAcc_operational::m_syncLocked;
    using loPredCtrlAcc_operational::m_syncopationTiming;
    using loPredCtrlAcc_operational::m_unlockedTraining;
    using loPredCtrlAcc_operational::m_wasValid;
    using loPredCtrlAcc_operational::m_wfsCounter;
    using loPredCtrlAcc_operational::m_wfsFills;
    using loPredCtrlAcc_operational::m_wfsReceipt;
    using loPredCtrlAcc_operational::m_windowReason;
    using loPredCtrlAcc_operational::mainLoopInput;
    using loPredCtrlAcc_operational::modalZeroed;
    using loPredCtrlAcc_operational::outputCandidate;
    using loPredCtrlAcc_operational::outputValid;
    using loPredCtrlAcc_operational::parseExploration;
    using loPredCtrlAcc_operational::processImage;
    using loPredCtrlAcc_operational::resetModels;
    using loPredCtrlAcc_operational::sensorHistory;
    using loPredCtrlAcc_operational::slotInputs;
    using loPredCtrlAcc_operational::standardInput;
    using loPredCtrlAcc_operational::step;
    using loPredCtrlAcc_operational::validateModalOutput;

    /// Allocate private synthetic calibration and configure the actual existing producer state without starting it.
    explicit Controller( bool qr /**< [in] Select the real QR or covariance RLS implementation. */ );

    /// Next scheduler slot ordinal (cycle*4+step) rendered by advance().
    std::int64_t m_slot{ 8 };

    /// The application's default integrator setting, before the constructor turns it off.
    bool m_defaultIntegrate{ false };

    /// Advance one real decision, then render and copy the four following slots as the scheduler would.
    /** Returns the copied modal commands, modes by four, oldest slot first. */
    DDSPC::Matrix advance( const DDSPC::Matrix &y /**< [in] WFS residual. */,
                           const DDSPC::Matrix &a /**< [in] Individual ACC history. */,
                           bool                 valid /**< [in] Upstream ACC validity. */,
                           int missed = 0 /**< [in] WFS frames missed first: their slots are copied, no decision. */ );

    /// Render one scheduler slot through the actual candidate and report it copied unless skipped.
    DDSPC::Matrix slot( std::int64_t ordinal /**< [in] Slot ordinal, cycle*q+step. */,
                        std::int64_t epoch = 1 /**< [in] Scheduler epoch. */,
                        bool         copy  = true /**< [in] Report the scheduler copy. */ );

    /// Upstream ACC sample by counter, X/Y; it never depends on DM commands.
    std::function<std::array<double, 2>( std::uint64_t )> m_source;

    /// Next ACC counter to deliver.
    std::uint64_t m_nextSample{ 0 };

    /// ACC anchor of the latest assisted frame; -1 before the first.
    std::int64_t m_frameAnchor{ -1 };

    /// Deliver the r samples that arrive with the next WFS frame; return the frame's anchor.
    std::uint64_t nextFrame();

    /// Deliver ACC samples through the given counter to the DM worker's ring, as the upstream source would.
    void deliver( std::uint64_t last /**< [in] Newest counter now available. */ );

    /// Return the WFS-thread ACC base window ending at the newest delivered sample and set its anchor.
    DDSPC::Matrix window();

    /// Assisted frame: decision on the current window, then q slots, slot p after the samples of its phase.
    /** Returns the copied modal commands, modes by q, oldest slot first. Slots listed in skip are not copied. */
    DDSPC::Matrix advanceAssisted( const DDSPC::Matrix &y /**< [in] WFS residual. */,
                                   int                  skip   = -1 /**< [in] Slot index not copied, or -1. */,
                                   bool                 locked = true /**< [in] Upstream lock reported. */,
                                   int missed = 0 /**< [in] WFS frames missed first: ACC and slots continue. */ );

    /// Enable learning of the selected fit and original-format Gaussian excitation on the actual WFS path.
    void train( const char *sequence = "3000,0.15,0.1" /**< [in] Exploration triples. */ );

    /// Freeze both independent fits without stopping correction.
    void freeze();

    /// Qualify the own-channel output without a DM worker, so prediction requests are accepted.
    void qualifyOutput();

    /// Enable the integrator and deliver the main loop's parameters through the real cacaoInterface subscriptions and
    /// factor-stream monitors.
    void mainLoop( double                    gain /**< [in] holoop loop_gain. */,
                   double                    mult /**< [in] holoop loop_multcoeff. */,
                   double                    limit /**< [in] holoop loop_max_limit. */,
                   const std::vector<float> &gainFact  = { 1, 1, 1 } /**< [in] aol1_mgainfact by WFS mode. */,
                   const std::vector<float> &multFact  = { 1, 1, 1 } /**< [in] aol1_mmultfact by WFS mode. */,
                   const std::vector<float> &limitFact = { 1, 1, 1 } /**< [in] aol1_mlimitfact by WFS mode. */ );
};

Controller::Controller( bool qr )
{
    // The integrator is on by default; the absolute-position tests turn it off, and the integrator tests back on.
    m_defaultIntegrate = m_integrate;
    m_integrate        = false;

    m_settings.m_history       = 3;
    m_settings.m_future        = 2;
    m_settings.m_accelChannels = 8;
    m_settings.m_prior         = .001f;
    m_settings.m_lambda        = .1f;
    m_settings.m_useQrd        = qr;
    m_gain                     = 0;
    m_calibration.m_shapes     = DDSPC::Matrix::Zero( 4, 2 );
    m_calibration.m_shapes.topRows( 2 ).setIdentity();
    m_calibration.m_projection = m_calibration.m_shapes.transpose();
    m_copiedCommand            = DDSPC::Matrix::Zero( 2, 1 );
    m_ring                     = ringSize(); // As loadConfig(), so the default max_fill gap is retained.
    m_appliedSlots             = DDSPC::Matrix::Zero( 2, m_ring );
    m_commandKnown             = true;
    resetModels();
    setupINDI();
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = config.m_producerDefaults = true;
    m_output.configure( config, "private_aux", "private_timing", 50, 4 );
}

void Controller::train( const char *sequence )
{
    m_learning = m_predicting = true;
    m_exploration             = parseExploration( sequence );
    applyTraining();
}

void Controller::qualifyOutput()
{
    m_output.start(
        []
        {
            return true;
        },
        {},
        {},
        false );
}

void Controller::mainLoop( double                    gain,
                           double                    mult,
                           double                    limit,
                           const std::vector<float> &gainFact,
                           const std::vector<float> &multFact,
                           const std::vector<float> &limitFact )
{
    namespace app = MagAOX::app;
    m_integrate   = true;

    auto scalar = []( pcf::IndiProperty property, double value )
    {
        property.add( pcf::IndiElement( "current", value ) );
        return property;
    };

    REQUIRE( setCallBack_m_indiP_loopGain( scalar( m_indiP_loopGain, gain ) ) == 0 );
    REQUIRE( setCallBack_m_indiP_loopMult( scalar( m_indiP_loopMult, mult ) ) == 0 );
    REQUIRE( setCallBack_m_indiP_loopLimit( scalar( m_indiP_loopLimit, limit ) ) == 0 );

    using gainMonitor  = app::dev::shmimMonitor<app::loPredCtrlAcc_operational, app::ddspcGainFactShmimT>;
    using multMonitor  = app::dev::shmimMonitor<app::loPredCtrlAcc_operational, app::ddspcMultFactShmimT>;
    using limitMonitor = app::dev::shmimMonitor<app::loPredCtrlAcc_operational, app::ddspcLimitFactShmimT>;

    auto gains = gainFact, mults = multFact, limits = limitFact;
    gainMonitor::m_width   = gains.size();
    gainMonitor::m_height  = 1;
    multMonitor::m_width   = mults.size();
    multMonitor::m_height  = 1;
    limitMonitor::m_width  = limits.size();
    limitMonitor::m_height = 1;
    REQUIRE( processImage( gains.data(), app::ddspcGainFactShmimT{} ) == 0 );
    REQUIRE( processImage( mults.data(), app::ddspcMultFactShmimT{} ) == 0 );
    REQUIRE( processImage( limits.data(), app::ddspcLimitFactShmimT{} ) == 0 );
}

void Controller::deliver( std::uint64_t last )
{
    for( ; m_nextSample <= last; ++m_nextSample )
    {
        DDSPC::AdcSample sample;
        sample.m_frame.m_counter = m_nextSample;
        sample.m_generation = sample.m_segment = 1;
        sample.m_acceleration                  = m_source( m_nextSample );
        sample.m_acquiredNs                    = MagAOX::app::synco::monotonicNs();
        fastSample( sample );
    }
}

std::uint64_t Controller::nextFrame()
{
    m_frameAnchor = m_frameAnchor < 0 ? m_ratio - 1 : m_frameAnchor + m_ratio;
    deliver( m_frameAnchor );
    return m_frameAnchor;
}

DDSPC::Matrix Controller::window()
{
    const int length = m_settings.m_accelChannels / 2;
    if( m_nextSample < std::uint64_t( length ) )
        return {};
    m_adcAnchor  = m_nextSample - 1;
    m_adcSegment = ( std::uint64_t( 1 ) << 32 ) ^ 1;
    DDSPC::Matrix features( 2 * length, 1 );
    for( int j = 0; j < length; ++j )
        for( int axis = 0; axis < 2; ++axis )
            features( 2 * j + axis ) = m_source( m_adcAnchor - j )[axis];
    return features;
}

DDSPC::Matrix Controller::advanceAssisted( const DDSPC::Matrix &y, int skip, bool locked, int missed )
{
    for( int k = 0; k < missed; ++k ) // The missed frames' ACC samples arrive and their slots are copied.
    {
        const auto anchor = nextFrame();
        for( int p = 0; p < m_slots; ++p )
        {
            deliver( anchor + p * m_ratio / m_slots );
            slot( m_slot++ );
        }
    }

    const auto anchor   = nextFrame();
    const auto features = window();
    m_windowValid       = features.size() > 0; // As acceleration() reports a complete window.
    DDSPC::Matrix applied;
    std::int64_t  ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ordinal = appliedSlots( applied );
    }
    REQUIRE( step( y,
                   m_wfsCounter + 1 + missed,
                   features,
                   features.size() > 0 && locked,
                   applied,
                   ordinal,
                   MagAOX::app::synco::monotonicNs() ) == 0 );
    DDSPC::Matrix slots( 2, m_slots );
    for( int p = 0; p < m_slots; ++p )
    {
        deliver( anchor + p * m_ratio / m_slots );
        slots.col( p ) = slot( m_slot++, 1, p != skip );
    }
    return slots;
}

void Controller::freeze()
{
    auto property = m_indiP_learning;
    property["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( newCallBack_m_indiP_learning( property ) == 0 );
}

DDSPC::Matrix Controller::slot( std::int64_t ordinal, std::int64_t epoch, bool copy )
{
    namespace synco = MagAOX::app::synco;
    const synco::Request request{ epoch, 1, ordinal / m_slots, int( ordinal % m_slots ), 0, 0 };
    synco::Record        schedule{};
    schedule[synco::steps] = m_slots;
    std::array<float, 4> image{};
    REQUIRE( outputCandidate( request, schedule, image ) );
    if( !copy )
        return m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
    synco::CommitRecord commit{};
    commit[synco::cmdEpoch]      = epoch;
    commit[synco::cmdGeneration] = 1;
    commit[synco::cmdCycle]      = request.cycle;
    commit[synco::cmdStep]       = request.step;
    copied( commit, image );
    return m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
}

DDSPC::Matrix Controller::advance( const DDSPC::Matrix &y, const DDSPC::Matrix &a, bool valid, int missed )
{
    for( int k = 0; k < missed * m_slots; ++k ) // The missed frames' slots are still copied.
    {
        slot( m_slot++ );
    }

    DDSPC::Matrix applied;
    std::int64_t  ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ordinal = appliedSlots( applied );
    }
    REQUIRE( step( y, m_wfsCounter + 1 + missed, a, valid, applied, ordinal, MagAOX::app::synco::monotonicNs() ) == 0 );
    DDSPC::Matrix slots( 2, m_slots );
    for( int j = 0; j < m_slots; ++j )
        slots.col( j ) = slot( m_slot++ );
    return slots;
}

/// Synthetic WFS response to one frame of slot commands; earlier slots act longer before the next measurement.
DDSPC::Matrix response( const DDSPC::Matrix &slots /**< [in] Copied modal commands, oldest slot first. */ )
{
    return .35f * slots.col( 0 ) + .25f * slots.col( 1 ) + .15f * slots.col( 2 ) + .05f * slots.col( 3 );
}
} // namespace
/// \endcond

/** \brief The real WFS callback learns from acknowledged feedback with ACC disconnected and no explicit probes. */
TEST_CASE( "Standard WFS monitor trains from copied modal commands", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::allocate;
    MagAOX::app::loPredCtrlAcc_operational::processImage;
    MagAOX::app::loPredCtrlAcc_operational::copied;
#endif
    // clang-format on
    Controller           app( true );
    IMAGE_METADATA       md{};
    std::array<float, 4> wfs{}, surface{};
    app.m_imageStream.md      = &md;
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 3;
    md.size[0]                = 2;
    md.size[1]                = 1;
    md.size[2]                = 2;
    app.m_width               = 2;
    app.m_height              = 1;
    REQUIRE( app.allocate( MagAOX::app::dev::shmimT{} ) == 0 );
    app.m_gain = .2f;
    app.train();
    app.m_exploration.clear();
    app.m_commandKnown = false;
    for( unsigned n = 1; n <= 150; ++n )
    {
        md.cnt0     = n;
        md.cnt1     = n % 2;
        auto *frame = wfs.data() + 2 * md.cnt1;
        frame[0]    = std::sin( n * .13f );
        frame[1]    = std::cos( n * .07f );
        REQUIRE( app.processImage( wfs.data() + 2 * ( 1 - md.cnt1 ), MagAOX::app::dev::shmimT{} ) == 0 );
        REQUIRE( app.m_wfsCounter == n - 1 );
        REQUIRE( app.processImage( frame, MagAOX::app::dev::shmimT{} ) == 0 );
        REQUIRE( app.m_wfsCounter == n );
        if( n < 30 )
        {
            MagAOX::app::synco::Record schedule{};
            schedule[MagAOX::app::synco::steps] = 4;
            REQUIRE( app.outputCandidate( { 1, 1, 0, 1, 0, 0 }, schedule, surface ) );
            REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
            REQUIRE( surface[0] == Approx( .2f * frame[0] ) );
        }
        if( n >= 30 )
            for( int j = 0; j < 4; ++j )
                app.slot( app.m_slot++ );
    }
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates > 100 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
}

/** \brief Copied scheduler slots form the training lift; missed slots and idle frames hold the retained copy. */
TEST_CASE( "Operational slot history follows scheduler copies", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::appliedSlots;
    MagAOX::app::loPredCtrlAcc_operational::copied;
    MagAOX::app::loPredCtrlAcc_operational::outputCandidate;
#endif
    // clang-format on
    namespace synco = MagAOX::app::synco;
    Controller app( true );
    auto       copy = [&]( std::int64_t ordinal, float value, std::int64_t epoch = 1 )
    {
        synco::CommitRecord commit{};
        commit[synco::cmdEpoch]      = epoch;
        commit[synco::cmdGeneration] = 1;
        commit[synco::cmdCycle]      = ordinal / 4;
        commit[synco::cmdStep]       = ordinal % 4;
        const std::array<float, 4> image{ value, -value, 0, 0 };
        app.copied( commit, image );
    };
    auto lift = [&]( std::int64_t &ordinal )
    {
        DDSPC::Matrix               applied;
        std::lock_guard<std::mutex> lock( app.m_commandMutex );
        ordinal = app.appliedSlots( applied );
        return applied;
    };
    std::int64_t ordinal;
    copy( 12, 1 );
    REQUIRE( lift( ordinal ).cols() == 4 ); // First copy: no decision anchor yet, the retained copy is held.
    REQUIRE( ordinal == 12 );
    app.m_decisionOrdinal = 12;
    copy( 13, 2 );
    copy( 14, 3 );
    copy( 16, 5 ); // Ordinal 15 was missed: the destination retained 3.
    auto applied = lift( ordinal );
    REQUIRE( ordinal == 16 );
    REQUIRE( applied.row( 0 ) == ( DDSPC::Matrix( 1, 4 ) << 2, 3, 3, 5 ).finished() );
    REQUIRE( applied.row( 1 ) == -applied.row( 0 ) );
    app.m_decisionOrdinal = 16;
    copy( 17, 6 );
    copy( 18, 7 );
    applied = lift( ordinal ); // Two new copies: earlier positions retain the previous frame's last copy.
    REQUIRE( applied.row( 0 ) == ( DDSPC::Matrix( 1, 4 ) << 5, 5, 6, 7 ).finished() );
    app.m_decisionOrdinal = 18;
    applied               = lift( ordinal ); // Idle output: no copies, the retained command is held for the frame.
    REQUIRE( applied.row( 0 ) == ( DDSPC::Matrix( 1, 4 ) << 7, 7, 7, 7 ).finished() );
    copy( 3, 9, 2 ); // Scheduler reset: ordinals restart and the plan base is revoked.
    REQUIRE( app.m_decisionOrdinal == -1 );
    REQUIRE( app.m_planBase == -1 );
    REQUIRE( lift( ordinal ).row( 0 ) == ( DDSPC::Matrix( 1, 4 ) << 9, 9, 9, 9 ).finished() );

    // Each Request renders the plan column of its ordinal after the decision; later slots hold the last column.
    app.m_slotPlan = DDSPC::Matrix::Zero( 2, 8 );
    for( int c = 0; c < 8; ++c )
        app.m_slotPlan( 0, c ) = c + 1;
    app.m_planBase = 3;
    synco::Record schedule{};
    schedule[synco::steps] = 4;
    std::array<float, 4> image{};
    for( const auto &[cycle, step, column] :
         std::vector<std::array<int, 3>>{ { 1, 0, 0 }, { 1, 1, 1 }, { 1, 3, 3 }, { 2, 0, 4 }, { 9, 0, 7 } } )
    {
        REQUIRE( app.outputCandidate( { 2, 1, cycle, step, 0, 0 }, schedule, image ) );
        REQUIRE( image[0] == column + 1 );
    }
    REQUIRE( app.outputCandidate( { 1, 1, 1, 1, 0, 0 }, schedule, image ) ); // Other epoch: first column.
    REQUIRE( image[0] == 1 );

    // The reset above makes the next decision re-anchor; afterwards an interval with other than four new copies is
    // counted, not rejected.
    const DDSPC::Matrix y = DDSPC::Matrix::Zero( 2, 1 ), held = DDSPC::Matrix::Zero( 2, 4 );
    REQUIRE( app.step( y, 1, {}, false, held, 3, 0 ) == 0 );
    REQUIRE( app.m_decisionOrdinal == -1 );
    REQUIRE( app.step( y, 2, {}, false, held, 3, 0 ) == 0 );
    REQUIRE( app.step( y, 3, {}, false, held, 6, 0 ) == 0 );
    REQUIRE( app.m_slotShifts == 1 );
    REQUIRE( app.step( y, 4, {}, false, held, 10, 0 ) == 0 );
    REQUIRE( app.m_slotShifts == 1 );
    REQUIRE( app.m_planBase == 10 );
}

/** \brief The operational ACC vector retains four original upstream pairs in order and rejects incomplete recovery. */
TEST_CASE( "Operational WFS decision consumes individual upstream sensor samples", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::sensorHistory;
#endif
    // clang-format on
    Controller                      app( true );
    DDSPC::Matrix                   features;
    std::array<DDSPC::AdcSample, 4> rows{};
    DDSPC::AdcBatch                 batch;
    batch.m_status.m_valid = true;
    batch.m_count          = 4;
    for( unsigned i = 0; i < 4; ++i )
    {
        rows[i].m_frame.m_counter = 10 + i;
        rows[i].m_receivedNs = rows[i].m_acquiredNs = 1000000 + 125000 * i;
        rows[i].m_acceleration                      = { double( i ), -double( i ) };
        rows[i].m_generation = rows[i].m_segment = 1;
    }
    REQUIRE_FALSE( app.sensorHistory( batch, rows, features, 1500000 ) );
    app.m_syncLocked = true;
    REQUIRE( app.sensorHistory( batch, rows, features, 1500000 ) );
    for( unsigned i = 0; i < 4; ++i )
    {
        REQUIRE( features( 2 * i ) == 3 - i );
        REQUIRE( features( 2 * i + 1 ) == -double( 3 - i ) );
    }
    // Up to max_fill (5) consecutive missed ACC frames are filled, one frame on each, holding the newest sample; the
    // next one is not.
    REQUIRE( app.m_maxFill == 5 );
    batch.m_count = 0;
    for( unsigned k = 1; k <= 5; ++k )
    {
        REQUIRE( app.sensorHistory( batch, rows, features, 1500000 ) );
        REQUIRE( app.m_adcAnchor == 13 + 4 * k );
        REQUIRE( app.m_accFilled );
        for( unsigned i = 0; i < 4; ++i )
        {
            REQUIRE( features( 2 * i ) == 3 );
        }
    }
    REQUIRE_FALSE( app.sensorHistory( batch, rows, features, 1500000 ) );
    REQUIRE( std::string( app.m_windowReason.load() ) == "no_new_sample" );

    // One missing sample inside the window holds the sample before it: counters 20-23, then 24, 26, 27.
    for( unsigned i = 0; i < 4; ++i )
    {
        rows[i].m_frame.m_counter = 20 + i;
    }
    batch.m_count = 4;
    REQUIRE( app.sensorHistory( batch, rows, features, 1500000 ) );
    REQUIRE_FALSE( app.m_accFilled );
    for( unsigned i = 0; i < 3; ++i )
    {
        rows[i].m_frame.m_counter = i ? 25 + i : 24;
        rows[i].m_acceleration    = { 10. + i, 0 };
    }
    batch.m_count = 3;
    REQUIRE( app.sensorHistory( batch, rows, features, 1500000 ) );
    REQUIRE( app.m_accFilled );
    REQUIRE( app.m_adcAnchor == 27 );
    REQUIRE( features( 0 ) == 12 ); // 27
    REQUIRE( features( 2 ) == 11 ); // 26
    REQUIRE( features( 4 ) == 10 ); // 25, missing: holds 24
    REQUIRE( features( 6 ) == 10 ); // 24

    // Without filling, a missed ACC frame stops the window, as before.
    app.m_maxFill = 0;
    batch.m_count = 0;
    REQUIRE_FALSE( app.sensorHistory( batch, rows, features, 1500000 ) );
    app.m_maxFill = 1;

    batch.m_discontinuity = true;
    batch.m_count         = 1;
    REQUIRE_FALSE( app.sensorHistory( batch, rows, features, 1500000 ) );
}

/** \brief Read actual FITS planes and recover their signed coefficients through the operational projection. */
TEST_CASE( "Operational FITS calibration preserves modal sign and pixel order", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::loadCalibration;
#endif
    // clang-format on
    Controller app( true );
    const auto path = std::filesystem::temp_directory_path() / ( "rtc-ddspc-" + std::to_string( getpid() ) + ".fits" );
    mx::improc::eigenCube<float> cube( 3, 2, 2 );
    cube.image( 0 ) << 1, 2, -1, 0, 3, -4;
    cube.image( 1 ) << 4, 3, 0, -2, 2, 1;
    mx::fits::fitsFile<float> file;
    REQUIRE( file.write( path.string(), cube ) == mx::error_t::noerror );
    app.m_modeCube = path.string();
    app.loadCalibration();
    std::filesystem::remove( path );
    const auto &cal = app.m_calibration;
    for( int m = 0; m < 2; ++m )
        for( int j = 0; j < 6; ++j )
            REQUIRE( cal.m_shapes( j, m ) == cube.image( m ).data()[j] );
    REQUIRE( ( cal.m_projection * cal.m_shapes ).isApprox( DDSPC::Matrix::Identity( 2, 2 ), 1e-5f ) );
}

/** \brief Standard DDSPC trains and suppresses a real synthetic closed-loop residual with no ACC or sync input. */
TEST_CASE( "Operational standard DDSPC corrects without accelerometers", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    DDSPC::ShadowController::predict_next;
#endif
    // clang-format on
    const bool qr = GENERATE( false, true );
    Controller app( qr );
    app.train();
    DDSPC::Matrix command  = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty;
    double        residual = 0, uncontrolled = 0;
    std::uint64_t updates = 0;
    for( int n = 0; n < 6000; ++n )
    {
        DDSPC::Matrix disturbance( 2, 1 );
        disturbance << std::sin( .09 * n ) + .4 * std::cos( .17 * n ),
            .8 * std::cos( .13 * n ) + .3 * std::sin( .055 * n );
        y = disturbance + .8f * command;
        if( n == 3000 )
        {
            app.freeze();
            updates = app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates;
        }
        const auto slots = app.advance( y, empty, false );
        REQUIRE( slots.allFinite() );
        // Standard DDSPC is single-rate: one held command per WFS frame, as in loPredCtrl.
        for( int j = 1; j < 4; ++j )
            REQUIRE( slots.col( j ) == slots.col( 0 ) );
        command = response( slots ) / .8f;
        if( n >= 4000 )
        {
            residual += y.squaredNorm();
            uncontrolled += disturbance.squaredNorm();
        }
    }
    const auto standard = app.m_models->status( DDSPC::ShadowMode::Standard );
    const auto assisted = app.m_models->status( DDSPC::ShadowMode::AccelAssisted );
    INFO( "residual ratio " << std::sqrt( residual / uncontrolled ) );
    REQUIRE( updates > 2900 );
    REQUIRE( standard.m_updates == updates );
    REQUIRE( standard.m_ready );
    REQUIRE( assisted.m_updates == 0 );
    REQUIRE( app.m_multirate->status().m_updates == 0 ); // Only the selected mode learns.
    REQUIRE( residual < .25 * uncontrolled );
}

/** \brief The standard integrator applies the main loop's mfilt law, holds and clips its state, and restarts from zero.
 */
TEST_CASE( "Standard integrator follows the main loop's mfilt law", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    MagAOX::app::loPredCtrlAcc_operational::standardInput;
    MagAOX::app::loPredCtrlAcc_operational::combineIntegrator;
    MagAOX::app::loPredCtrlAcc_operational::setCallBack_m_indiP_loopGain;
    MagAOX::app::loPredCtrlAcc_operational::outputValid;
#endif
    // clang-format on
    Controller app( true );
    app.m_gain       = .2f;
    app.m_integrate  = true;
    app.m_predicting = true; // Untrained: each delta is gain*y alone.

    // Without the main loop's parameters the integrator has nothing to apply: output is not qualified.
    REQUIRE_FALSE( app.m_integratorReady.load() );
    DDSPC::Matrix y( 2, 1 ), empty;
    y << .4f, -.6f;
    REQUIRE( app.advance( y, empty, false ).isZero() );
    app.m_wasValid = true; // Record the first failing condition.
    REQUIRE_FALSE( app.outputValid() );
    REQUIRE( std::string( app.m_invalidReason.load() ) == "integrator_parameters" );

    // holoop gain .25, mc .9, limit .3; mode 0 gain factor 2, mode 1 limit factor 2; mode 2 is not a DDSPC mode.
    app.mainLoop( .25, .9, .3, { 2, 1, 7 }, { 1, 1, 7 }, { 1, 2, 7 } );
    REQUIRE( app.m_integratorReady.load() );
    const auto law = app.integrator();
    REQUIRE( law.m_gain[0] == Approx( .5 ) );
    REQUIRE( law.m_gain[1] == Approx( .25 ) );
    REQUIRE( law.m_mult[0] == Approx( .9 ) );
    REQUIRE( law.m_limit[0] == Approx( .3 ) );
    REQUIRE( law.m_limit[1] == Approx( .6 ) );

    const std::array<double, 2> gain{ .5, .25 }, limit{ .3, .6 };
    std::array<double, 2>       expected{};
    for( int n = 0; n < 12; ++n )
    {
        y << .4f + .1f * n, -.6f;
        const auto slots = app.advance( y, empty, false );
        for( int mode = 0; mode < 2; ++mode )
        {
            expected[mode] = std::clamp( .9 * expected[mode] - gain[mode] * .2 * y( mode ), -limit[mode], limit[mode] );
            for( int j = 0; j < 4; ++j )
            {
                REQUIRE( slots( mode, j ) == Approx( expected[mode] ).margin( 1e-6 ) );
            }
        }
    }
    REQUIRE( expected[0] == Approx( -.3 ) ); // The limit was reached.

    // An invalid update keeps the last value; a new factor takes effect at once.
    auto invalid = app.m_indiP_loopGain;
    invalid.add( pcf::IndiElement( "current", "nan" ) );
    REQUIRE( app.setCallBack_m_indiP_loopGain( invalid ) == 0 );
    REQUIRE( app.integrator().m_gain[0] == Approx( .5 ) );
    std::vector<float> factors{ 4, 1, 7 };
    REQUIRE( app.processImage( factors.data(), MagAOX::app::ddspcGainFactShmimT{} ) == 0 );
    REQUIRE( app.integrator().m_gain[0] == Approx( 1 ) );

    app.m_predicting = false;
    REQUIRE( app.advance( y, empty, false ).isZero() );
    REQUIRE( app.m_intState.isZero() );

    app.m_predicting = true;
    const auto restart = app.advance( y, empty, false );
    REQUIRE( restart( 0, 0 ) == Approx( -1 * .2 * y( 0 ) ) );
    REQUIRE( restart( 1, 0 ) == Approx( -.25 * .2 * y( 1 ) ) );

    // Training input: the delta that produced each copied position, unknown without a previous position.
    app.resetModels();
    DDSPC::Matrix applied( 2, 4 ), input;
    applied.row( 0 ).setConstant( .2f );
    applied.row( 1 ).setConstant( -.1f );
    REQUIRE_FALSE( app.standardInput( applied, input ) );
    applied.row( 0 ).setConstant( .1f );
    applied.row( 1 ).setConstant( .05f );
    REQUIRE( app.standardInput( applied, input ) );
    REQUIRE( input( 0 ) == Approx( -( .1 - .9 * .2 ) / 1 ) );
    REQUIRE( input( 1 ) == Approx( -( .05 + .9 * .1 ) / .25 ) );

    app.m_integrate = false;
    REQUIRE( app.standardInput( applied, input ) );
    REQUIRE( input.isApprox( applied.col( 0 ) ) );
}

/** \brief With the integrator, standard DDSPC learns from deltas and also rejects a constant disturbance. */
TEST_CASE( "Standard integrator corrects and removes a static offset", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
#endif
    // clang-format on
    const bool qr = GENERATE( false, true );
    Controller app( qr );
    app.mainLoop( 1, 1, 10 );
    app.m_gain = .2f;
    app.train();

    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty, mean = DDSPC::Matrix::Zero( 2, 1 );
    double        residual = 0, uncontrolled = 0;
    for( int n = 0; n < 6000; ++n )
    {
        DDSPC::Matrix disturbance( 2, 1 );
        disturbance << .5 + std::sin( .09 * n ) + .4 * std::cos( .17 * n ),
            -.3 + .8 * std::cos( .13 * n ) + .3 * std::sin( .055 * n );
        y = disturbance + .8f * command;

        if( n == 3000 )
        {
            app.freeze();
        }

        const auto slots = app.advance( y, empty, false );
        REQUIRE( slots.allFinite() );
        command = response( slots ) / .8f;

        if( n >= 4000 )
        {
            residual += y.squaredNorm();
            uncontrolled += disturbance.squaredNorm();
            mean += y / 2000;
        }
    }
    INFO( "residual ratio " << std::sqrt( residual / uncontrolled ) << ", mean residual " << mean.transpose() );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( residual < .25 * uncontrolled );
    REQUIRE( mean.cwiseAbs().maxCoeff() < .05 );
}

/// \cond RTC_TEST_SUPPORT
namespace
{
/// 8-kHz plant: narrowband random vibrations the accelerometers see, a slow drift they do not, a WFS integrating q
/// slots.
/** Each vibration is a resonant AR(2) process driven by white noise, so its phase wanders and recent measurements carry
 * new information. The residual of slot n is e_n = d_n + c_(n-1): the command calculated for slot n-1 acts during
 * slot n. The accelerometers measure the vibrations' acceleration, -omega^2 x, never the DM, so their samples do not
 * depend on commands. The true residual is used only for scoring, never by the controller.
 */
struct Plant
{
    /// One resonant vibration component.
    struct Line
    {
        int    m_axis;                             ///< Mode it disturbs.
        double m_hz;                               ///< Centre frequency.
        double m_rms;                              ///< Stationary RMS.
        double m_a1{ 0 }, m_a2{ 0 }, m_sigma{ 0 }; ///< AR(2) coefficients and drive.
        double m_x1{ 0 }, m_x2{ 0 };               ///< Last two values.
    };

    double       m_accNoise; ///< ACC noise; -1 gives independent noise (no information), -2 a disconnected (zero) ACC.
    std::mt19937 m_rng{ 17 };
    std::normal_distribution<double>   m_unit{ 0, 1 };
    std::vector<Line>                  m_lines{ { 0, 470, 1 }, { 0, 130, .5 }, { 1, 310, .8 } };
    std::vector<std::array<double, 2>> m_vibration, m_acc; ///< By slot/counter, generated in order.
    std::array<double, 2>              m_drift{};          ///< Slow WFS-only component.

    /// Select the accelerometer information level.
    explicit Plant( double accNoise /**< [in] Relative noise; negative for an independent ACC. */ )
        : m_accNoise( accNoise )
    {
        const double rho = .99;
        for( auto &line : m_lines )
        {
            const double w = 2 * M_PI * line.m_hz / 8000;
            line.m_a1      = 2 * rho * std::cos( w );
            line.m_a2      = -rho * rho;
            const double variance =
                ( 1 + rho * rho ) /
                ( ( 1 - rho * rho ) * ( std::pow( 1 + rho * rho, 2 ) - 4 * rho * rho * std::pow( std::cos( w ), 2 ) ) );
            line.m_sigma = line.m_rms / std::sqrt( variance );
        }
    }

    /// Generate vibrations and ACC samples through slot n.
    void generate( std::uint64_t n /**< [in] Slot/counter. */ )
    {
        const double reference = std::pow( 2 * M_PI * 300, 2 );
        while( m_vibration.size() <= n )
        {
            std::array<double, 2> x{}, a{};
            for( auto &line : m_lines )
            {
                const double value = line.m_a1 * line.m_x1 + line.m_a2 * line.m_x2 + line.m_sigma * m_unit( m_rng );
                line.m_x2          = line.m_x1;
                line.m_x1          = value;
                x[line.m_axis] += value;
                a[line.m_axis] -= std::pow( 2 * M_PI * line.m_hz, 2 ) / reference * value;
            }
            for( auto &axis : a )
                axis = m_accNoise < -1.5 ? 0 : m_accNoise < 0 ? m_unit( m_rng ) : axis + m_accNoise * m_unit( m_rng );
            m_vibration.push_back( x );
            m_acc.push_back( a );
        }
    }

    /// Vibration displacement at slot n.
    std::array<double, 2> vibration( std::uint64_t n /**< [in] Slot. */ )
    {
        generate( n );
        return m_vibration[n];
    }

    /// ACC sample n.
    std::array<double, 2> acc( std::uint64_t n /**< [in] Counter. */ )
    {
        generate( n );
        return m_acc[n];
    }

    /// Residual of slot n after the previous slot's command; advances the drift.
    std::array<double, 2> residual( std::uint64_t        n /**< [in] Slot. */,
                                    const DDSPC::Matrix &previous /**< [in] c_(n-1). */ )
    {
        const auto            vib = vibration( n );
        std::array<double, 2> e;
        for( int axis = 0; axis < 2; ++axis )
        {
            m_drift[axis] = .999 * m_drift[axis] + .01 * m_unit( m_rng );
            e[axis]       = vib[axis] + m_drift[axis] + previous( axis ) + .01 * m_unit( m_rng );
        }
        return e;
    }
};

/// Closed-loop result of one controller on one plant realization.
struct Result
{
    double              m_residual{ 0 }, m_wfs{ 0 }, m_open{ 0 }; ///< True 8-kHz, WFS and uncontrolled variances.
    double              m_rmse{ 0 };                              ///< Conditional future-WFS prediction RMSE.
    std::vector<double> m_phaseRmse;                              ///< Assisted per-phase RMSE.
    std::vector<float>  m_trace;                                  ///< Evaluation residuals, slot-major X/Y.
};

/// Train one mode on the plant, freeze it and score the evaluation frames.
Result closedLoop( bool   assisted /**< [in] Accelerometer-assisted mode. */,
                   double accNoise /**< [in] ACC information level. */,
                   int    train    = 12500 /**< [in] Training frames. */,
                   int    evaluate = 3000 /**< [in] Frozen evaluation frames. */ )
{
    Controller app( true );
    app.m_settings.m_history       = 8;
    app.m_settings.m_future        = 2;
    app.m_settings.m_accelChannels = 16;
    app.m_settings.m_lambda        = 100;
    app.m_settings.m_prior         = 1e-5f;
    app.m_mode.store( assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard );
    app.resetModels();
    app.m_syncLocked = true;
    Plant plant( accNoise );
    app.m_source = [&]( std::uint64_t n ) { return plant.acc( n ); };
    app.train( "4000,0.1,100,4000,0.1,3,4000,0.1,0.1" ); // loPredCtrl's sequence with four times the frames.
    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y = DDSPC::Matrix::Zero( 2, 1 ), empty;
    Result        result;
    std::uint64_t n = 0;
    for( int frame = 0; frame < train + evaluate; ++frame )
    {
        if( frame == train )
            app.freeze();
        const auto slots = assisted ? app.advanceAssisted( y ) : app.advance( y, empty, false );
        y.setZero();
        for( int p = 0; p < slots.cols(); ++p, ++n )
        {
            const auto e = plant.residual( n, command );
            command      = slots.col( p );
            for( int axis = 0; axis < 2; ++axis )
                y( axis ) += e[axis] / slots.cols();
            if( frame >= train )
            {
                const auto open = plant.vibration( n );
                result.m_residual += e[0] * e[0] + e[1] * e[1];
                result.m_open += open[0] * open[0] + open[1] * open[1];
                result.m_trace.insert( result.m_trace.end(), { float( e[0] ), float( e[1] ) } );
            }
        }
        if( frame >= train )
            result.m_wfs += y.squaredNorm();
    }
    const auto slots = double( evaluate ) * app.m_slots;
    result.m_residual /= slots;
    result.m_open /= slots;
    result.m_wfs /= evaluate;
    if( assisted )
    {
        result.m_phaseRmse = app.m_multirate->status().m_rmse;
        result.m_rmse      = result.m_phaseRmse.back();
        REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
    }
    else
    {
        const auto status = app.m_models->status( DDSPC::ShadowMode::Standard );
        REQUIRE( status.m_ready );
        result.m_rmse = status.m_rmse;
        REQUIRE( app.m_multirate->status().m_updates == 0 );
    }
    return result;
}
} // namespace
/// \endcond

/** \brief T9: with strong, weak, zero (independent noise) and disconnected ACC, 8-kHz assisted DDSPC is compared with
 * standard DDSPC trained and evaluated on the same plant realization.
 *
 * Writes the evaluation residuals to $DDSPC_MULTIRATE_TRACE/<case>_<mode>.f32 when set, for
 * tests/multirate_report.py (PSDs).
 */
TEST_CASE( "Assisted 8-kHz DDSPC against standard DDSPC by ACC information", "[rtc-controller][multirate-sim]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::stepAssisted;
    MagAOX::app::loPredCtrlAcc_operational::assistedCandidate;
#endif
    // clang-format on
    const auto [name, noise] = GENERATE( std::pair<const char *, double>{ "strong", .02 },
                                         std::pair<const char *, double>{ "weak", 1. },
                                         std::pair<const char *, double>{ "zero", -1. },
                                         std::pair<const char *, double>{ "disconnected", -2. } );
    const auto standard = closedLoop( false, noise ), assisted = closedLoop( true, noise );
    std::printf( "T9 %-6s: residual var standard %.4f assisted %.4f (open %.4f); WFS var %.4f / %.4f; "
                 "prediction RMSE standard %.4f assisted phases",
                 name,
                 standard.m_residual,
                 assisted.m_residual,
                 standard.m_open,
                 standard.m_wfs,
                 assisted.m_wfs,
                 standard.m_rmse );
    for( auto rmse : assisted.m_phaseRmse )
        std::printf( " %.4f", rmse );
    std::printf( "\n" );
    if( const char *directory = std::getenv( "DDSPC_MULTIRATE_TRACE" ) )
        for( const auto *result : { &standard, &assisted } )
            std::ofstream( std::string( directory ) + "/" + name + ( result == &standard ? "_standard" : "_assisted" ) +
                               ".f32",
                           std::ios::binary )
                .write( reinterpret_cast<const char *>( result->m_trace.data() ), result->m_trace.size() * 4 );
    REQUIRE( standard.m_residual < standard.m_open );
    if( noise >= 0 && noise < .1 )
    {
        REQUIRE( assisted.m_residual < .5 * standard.m_residual );
        REQUIRE( assisted.m_rmse < .5 * standard.m_rmse );
        REQUIRE( assisted.m_phaseRmse.back() < assisted.m_phaseRmse.front() ); // Later phases use newer samples.
    }
    else if( noise > 0 )
        REQUIRE( assisted.m_residual < .8 * standard.m_residual );
    else // No information: the extra regressors may cost estimation noise only.
        REQUIRE( assisted.m_residual < 1.2 * standard.m_residual );
}

/** \brief T3/T4/T6/T7/T8: one RLS update per WFS frame, one calculation per slot, the newest ACC sample acts on the
 * next slot, and executed (not planned) commands form the prefix.
 */
TEST_CASE( "Assisted DDSPC recalculates every DM slot from the newest ACC sample", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::assistedCandidate;
    MagAOX::app::loPredCtrlAcc_operational::fastSamples;
    DDSPC::MultirateDDSPC::command;
#endif
    // clang-format on
    Controller app( true );
    app.m_settings.m_history       = 6;
    app.m_settings.m_accelChannels = 16;
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.resetModels();
    app.m_syncLocked = true;
    Plant plant( .02 );
    app.m_source = [&]( std::uint64_t n ) { return plant.acc( n ); };
    app.train( "1500,0.1,0.1" );
    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y = DDSPC::Matrix::Zero( 2, 1 );
    std::uint64_t n     = 0;
    auto          frame = [&]( int skip = -1 )
    {
        const auto slots = app.advanceAssisted( y, skip );
        y.setZero();
        for( int p = 0; p < slots.cols(); ++p, ++n )
        {
            const auto e = plant.residual( n, command );
            command      = slots.col( p );
            for( int axis = 0; axis < 2; ++axis )
                y( axis ) += e[axis] / slots.cols();
        }
        return slots;
    };
    for( int i = 0; i < 1000; ++i )
        frame();
    // T3: one shared update per genuine WFS frame and one prediction per DM slot (four per frame).
    const auto updates = app.m_multirate->status().m_updates, predictions = app.m_fastStats.m_predictions;
    for( int i = 0; i < 100; ++i )
        frame();
    REQUIRE( app.m_multirate->status().m_ready );
    REQUIRE( app.m_multirate->status().m_updates - updates == 100 );
    REQUIRE( app.m_fastStats.m_predictions - predictions == 400 );
    REQUIRE( app.m_fastStats.m_phase[0] > 0 );
    REQUIRE( app.m_fastStats.m_phase[3] == app.m_fastStats.m_phase[0] ); // k = p on every slot.
    app.freeze();
    app.m_exploration.clear();
    for( int i = 0; i < 20; ++i )
    {
        const auto slots = frame();
        for( int p = 1; p < 4; ++p ) // Distinct calculations, not a held or replayed plan.
            REQUIRE( ( slots.col( p ) - slots.col( p - 1 ) ).norm() > 1e-6 );
    }

    // T4: a sample arriving between WFS frames changes the next slot by exactly its learned gain.
    namespace synco        = MagAOX::app::synco;
    const auto    anchor   = app.nextFrame();
    const auto    features = app.window();
    DDSPC::Matrix applied;
    std::int64_t  ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> lock( app.m_commandMutex );
        ordinal = app.appliedSlots( applied );
    }
    REQUIRE( app.step( y, app.m_wfsCounter + 1, features, true, applied, ordinal, 0 ) == 0 );
    const auto slot0 = app.slot( app.m_slot++ );
    app.deliver( anchor + 1 );
    const auto slot1 = app.slot( app.m_slot++ );
    app.deliver( anchor + 2 );
    const auto before = app.slot( app.m_slot, 1, false );
    auto      &sample = app.m_fastSamples[( anchor + 2 ) % app.m_fastSamples.size()];
    REQUIRE( sample.m_frame.m_counter == anchor + 2 );
    sample.m_acceleration[0] += .5;
    const auto  after = app.slot( app.m_slot, 1, false );
    const auto  s0    = app.m_multirate->config().state( 0 );
    const auto &gain  = app.m_gainSets[app.m_gainFront].m_slot[2 * 4 + 2];
    INFO( "slot 2 change " << ( after - before ).transpose() );
    for( int mode = 0; mode < 2; ++mode )
        REQUIRE( after( mode ) - before( mode ) == Approx( .5 * gain( mode, s0 + 2 ) ).margin( 1e-5 ) );
    REQUIRE( ( after - before ).norm() > 1e-4 );
    // Earlier slots of the frame were calculated before the sample existed (k = 0, 1): rendering is causal.
    REQUIRE( ( slot1 - slot0 ).norm() > 0 );

    // T6: slot 2 is not copied; slot 3's executed prefix holds the retained slot-1 copy, not the plan.
    sample.m_acceleration[0] -= .5;
    app.slot( app.m_slot++, 1, false ); // Rendered but never copied.
    app.deliver( anchor + 3 );
    app.slot( app.m_slot++ );
    const auto &prefix = app.m_fastPrefix; // Newest first: slot 2 (retained slot-1 copy), slot 1, slot 0.
    for( int mode = 0; mode < 2; ++mode )
    {
        REQUIRE( prefix[mode] == Approx( slot1( mode ) ).margin( 1e-6 ) );
        REQUIRE( prefix[2 + mode] == Approx( slot1( mode ) ).margin( 1e-6 ) );
        REQUIRE( prefix[4 + mode] == Approx( slot0( mode ) ).margin( 1e-6 ) );
    }

    // T7: a fifth slot before the next WFS frame holds the last command.
    const auto holds = app.m_fastStats.m_holds;
    const auto last  = app.slot( app.m_slot, 1, false );
    REQUIRE( app.m_fastStats.m_holds == holds + 1 );

    // T8: a late ACC sample (k < p) uses the matching gain and is counted.
    frame();
    const auto lagging = app.m_fastStats.m_lagging;
    app.nextFrame();
    const auto features2 = app.window();
    { //mutex scope
        std::lock_guard<std::mutex> lock( app.m_commandMutex );
        ordinal = app.appliedSlots( applied );
    }
    REQUIRE( app.step( y, app.m_wfsCounter + 1, features2, true, applied, ordinal, 0 ) == 0 );
    app.slot( app.m_slot++ );
    app.slot( app.m_slot++ ); // Sample anchor+1 has not arrived.
    REQUIRE( app.m_fastStats.m_lagging == lagging + 1 );
    REQUIRE( last.allFinite() );
}

/// \cond RTC_TEST_SUPPORT
namespace
{
/// Assisted controller with a short history, the test plant's ACC and upstream lock.
void assisted( Controller &app /**< [in,out] Application under test. */,
               Plant      &plant /**< [in] Source of the ACC samples. */ )
{
    app.m_settings.m_history       = 6;
    app.m_settings.m_accelChannels = 16;
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.resetModels();
    app.m_syncLocked = true;
    app.m_source     = [&]( std::uint64_t n )
    {
        return plant.acc( n );
    };
}
} // namespace
/// \endcond

/** \brief The assisted integrator applies the main loop's law on every DM slot, with gain/q, and trains on deltas. */
TEST_CASE( "Assisted integrator follows the per-slot mfilt law", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::assistedCandidate;
    MagAOX::app::loPredCtrlAcc_operational::slotInputs;
#endif
    // clang-format on
    Controller app( true );
    Plant      plant( .02 );
    assisted( app, plant );
    app.mainLoop( .5, .9, .3 );
    app.m_gain       = .2f;
    app.m_predicting = true; // Untrained: each slot's delta is gain*y alone.

    DDSPC::Matrix y = DDSPC::Matrix::Zero( 2, 1 );
    for( int i = 0; i < 20; ++i ) // Fill the history; a zero residual keeps the state at zero.
    {
        REQUIRE( app.advanceAssisted( y ).isZero() );
    }

    // gain/q per slot; the multiplier and limit are the main loop's, unchanged.
    const double  gain = .5 / 4, mult = .9;
    DDSPC::Matrix expected = DDSPC::Matrix::Zero( 2, 1 );
    for( int i = 0; i < 8; ++i )
    {
        y << .4f + .2f * i, -.6f;
        const auto slots = app.advanceAssisted( y );
        for( int p = 0; p < 4; ++p )
        {
            expected = ( mult * expected - gain * .2f * y ).cwiseMax( -.3f ).cwiseMin( .3f );
            REQUIRE( slots.col( p ).isApprox( expected, 1e-5f ) );
        }
    }
    REQUIRE( expected( 0 ) == Approx( -.3f ) ); // The limit was reached.

    // Training inputs: the per-slot deltas that produced the copied positions.
    app.resetModels();
    DDSPC::Matrix applied( 2, 4 );
    applied << .1f, .2f, .2f, .15f, 0, -.1f, -.05f, -.05f;
    REQUIRE( app.slotInputs( applied ).size() == 0 ); // No previous position yet.
    const DDSPC::Matrix previous = applied.col( 3 );
    applied << .3f, .1f, .1f, 0, .2f, .2f, .1f, 0;
    const auto deltas = app.slotInputs( applied );
    REQUIRE( deltas.cols() == 4 );
    for( int j = 0; j < 4; ++j )
    {
        const DDSPC::Matrix before = j ? DDSPC::Matrix( applied.col( j - 1 ) ) : previous;
        REQUIRE( deltas.col( j ).isApprox( -( applied.col( j ) - float( mult ) * before ) / float( gain ), 1e-5f ) );
    }
}

/** \brief With the integrator, assisted DDSPC learns from per-slot deltas and removes a constant offset. */
TEST_CASE( "Assisted integrator corrects and removes a static offset", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::assistedCandidate;
    MagAOX::app::loPredCtrlAcc_operational::stepAssisted;
#endif
    // clang-format on
    Controller app( true );
    Plant      plant( .02 );
    assisted( app, plant );
    app.mainLoop( 1, 1, 10 );
    app.m_gain = .2f;
    app.train( "1500,0.1,0.1" );

    const std::array<double, 2> offset{ .5, -.3 };
    DDSPC::Matrix               command = DDSPC::Matrix::Zero( 2, 1 ), y = DDSPC::Matrix::Zero( 2, 1 );
    std::uint64_t               n = 0;
    std::array<double, 2>       mean{}, variance{}, open{};
    for( int i = 0; i < 3000; ++i )
    {
        if( i == 2000 )
        {
            app.freeze();
            app.m_exploration.clear();
        }

        const auto slots = app.advanceAssisted( y );
        REQUIRE( slots.allFinite() );
        y.setZero();
        for( int p = 0; p < slots.cols(); ++p, ++n )
        {
            auto e = plant.residual( n, command );
            for( int axis = 0; axis < 2; ++axis )
            {
                e[axis] += offset[axis];
                y( axis ) += e[axis] / slots.cols();
                if( i >= 2500 )
                {
                    const double vibration = plant.vibration( n )[axis] + offset[axis];
                    mean[axis] += e[axis] / 2000;
                    variance[axis] += e[axis] * e[axis] / 2000;
                    open[axis] += vibration * vibration / 2000;
                }
            }
            command = slots.col( p );
        }
    }
    INFO( "mean residual " << mean[0] << ", " << mean[1] << "; residual/open power " << variance[0] / open[0] << ", "
                           << variance[1] / open[1] );
    REQUIRE( app.m_multirate->status().m_ready );
    for( int axis = 0; axis < 2; ++axis )
    {
        REQUIRE( std::abs( mean[axis] ) < .05 * std::abs( offset[axis] ) );
        REQUIRE( variance[axis] < .5 * open[axis] );
    }
}

/** \brief A missed WFS frame is filled: standard prediction continues without a fault, and learning pauses. */
TEST_CASE( "Standard prediction continues through a missed WFS frame", "[rtc-controller][fill]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    MagAOX::app::loPredCtrlAcc_operational::fillFrame;
#endif
    // clang-format on
    const bool integrate = GENERATE( false, true );
    Controller app( true );
    if( integrate )
    {
        app.mainLoop( 1, 1, 10 );
        app.m_gain = .2f;
    }
    app.train();

    DDSPC::Matrix command  = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty;
    double        residual = 0, uncontrolled = 0;
    auto          frame = [&]( int n, int missed )
    {
        DDSPC::Matrix disturbance( 2, 1 );
        disturbance << std::sin( .09 * n ) + .4 * std::cos( .17 * n ),
            .8 * std::cos( .13 * n ) + .3 * std::sin( .055 * n );
        y                = disturbance + .8f * command;
        const auto slots = app.advance( y, empty, false, missed );
        REQUIRE( slots.allFinite() );
        command = response( slots ) / .8f;
        if( n >= 3500 )
        {
            residual += y.squaredNorm();
            uncontrolled += disturbance.squaredNorm();
        }
    };

    int n = 0;
    for( ; n < 2500; ++n ) // Learning: a missed frame pauses it for history + future + 1 frames.
    {
        frame( n, 0 );
    }

    // The frame before the gap is still learned (its example is all real); then learning pauses.
    frame( n++, 1 );
    REQUIRE( app.m_wfsFills == 1 );
    const auto learned = app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates;
    for( int k = 0; k < 5; ++k, ++n )
    {
        frame( n, 0 );
        REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == learned );
    }
    for( int k = 0; k < 3; ++k, ++n )
    {
        frame( n, 0 );
    }
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates > learned );

    for( ; n < 3000; ++n )
    {
        frame( n, 0 );
    }

    // Frozen: gaps of up to max_fill frames keep the fit ready and the output qualified, with no slot shift.
    app.freeze();
    const auto shifts = app.m_slotShifts;
    for( ; n < 4500; ++n )
    {
        // Gaps of 1 to 5 frames.
        frame( n, n % 97 == 0 ? 1 + ( n / 97 ) % 5 : 0 );
        REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
        REQUIRE( app.m_candidateValid.load() );
    }
    REQUIRE( app.m_wfsFills > 10 );
    REQUIRE( app.m_slotShifts == shifts );
    INFO( "residual ratio with missed frames " << std::sqrt( residual / uncontrolled ) );
    REQUIRE( residual < .25 * uncontrolled );

    // Without filling a missed frame breaks the history, and the frozen fit stops qualifying.
    app.m_maxFill = 0;
    frame( n++, 1 );
    REQUIRE_FALSE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE_FALSE( app.m_candidateValid.load() );
}

/** \brief A missed WFS frame is filled: the assisted history is not broken and every slot is still predicted. */
TEST_CASE( "Assisted prediction continues through a missed WFS frame", "[rtc-controller][fill]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::stepAssisted;
    MagAOX::app::loPredCtrlAcc_operational::fillFrame;
#endif
    // clang-format on
    const bool integrate = GENERATE( false, true );
    Controller app( true );
    Plant      plant( .02 );
    assisted( app, plant );
    if( integrate )
    {
        app.mainLoop( 1, 1, 10 );
        app.m_gain = .2f;
    }
    app.train( "1500,0.1,0.1" );

    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y = DDSPC::Matrix::Zero( 2, 1 );
    std::uint64_t slotNumber = 0;
    auto          frame      = [&]( int missed )
    {
        slotNumber += 4 * missed; // The plant continues through the missed frames.
        const auto slots = app.advanceAssisted( y, -1, true, missed );
        REQUIRE( slots.allFinite() );
        y.setZero();
        for( int p = 0; p < slots.cols(); ++p, ++slotNumber )
        {
            const auto e = plant.residual( slotNumber, command );
            command      = slots.col( p );
            for( int axis = 0; axis < 2; ++axis )
            {
                y( axis ) += e[axis] / slots.cols();
            }
        }
    };

    for( int i = 0; i < 2000; ++i )
    {
        frame( 0 );
    }
    app.freeze();
    app.m_exploration.clear();
    frame( 0 );

    const auto breaks = app.m_multirate->status().m_breaks;
    for( int i = 0; i < 200; ++i )
    {
        const auto predictions = app.m_fastStats.m_predictions;
        // Gaps of 1 to 5 frames.
        frame( i % 23 == 0 ? 1 + ( i / 23 ) % 5 : 0 );
        REQUIRE( app.m_fastStats.m_predictions - predictions == 4 ); // The frame after a gap is predicted too.
        REQUIRE( app.m_candidateValid.load() );
    }
    REQUIRE( app.m_wfsFills > 5 );
    REQUIRE( app.m_multirate->status().m_breaks == breaks );
    REQUIRE( app.m_multirate->status().m_ready );
}

/** \brief assisted_qualification names every failing condition and the upstream values behind it. */
TEST_CASE( "Assisted qualification reports why assistance is blocked", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::setCallBack_m_indiP_sync;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on
    Controller app( true );
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.m_indiP_sync = pcf::IndiProperty( pcf::IndiProperty::Number );
    app.m_indiP_sync.setDevice( "acc000" );
    app.m_indiP_sync.setName( "timingDiag" );
    auto report = app.m_indiP_sync;
    for( const auto &[name, value] : std::vector<std::pair<const char *, double>>{
             { "mode_code", 1 }, { "delay_lock", 0 }, { "delay_phase_error_us", 0 }, { "read_latency_error_us", 52 } } )
        report.add( pcf::IndiElement( name, value ) );
    REQUIRE( app.setCallBack_m_indiP_sync( report ) == 0 );
    REQUIRE_FALSE( app.m_syncLocked );
    REQUIRE( app.appLogic() == 0 );
    const auto blockers = app.m_indiP_qualification["blockers"].get<std::string>();
    INFO( blockers );
    REQUIRE( blockers.find( "acc000.timingDiag.delay_lock=0" ) != std::string::npos );
    REQUIRE( blockers.find( "mode_code" ) == std::string::npos );
    REQUIRE( blockers.find( "adc_disabled" ) != std::string::npos );
    REQUIRE( blockers.find( "model_untrained_0_of_" ) != std::string::npos );
    REQUIRE( app.m_indiP_qualification["upstream_read_latency_error_us"].get<std::string>() == "52.0" );
    REQUIRE( app.m_indiP_qualification["diagnostic_mode"].get<std::string>() == "off" );
    REQUIRE( app.m_indiP_control["assisted_state"].get<std::string>() == blockers.substr( 0, blockers.find( ',' ) ) );
    report["delay_lock"].set( 1 );
    REQUIRE( app.setCallBack_m_indiP_sync( report ) == 0 );
    REQUIRE( app.m_syncLocked );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_indiP_qualification["blockers"].get<std::string>().find( "delay_lock" ) == std::string::npos );
}

/** \brief The default-off diagnostic mode evaluates and trains the assisted fit without upstream lock, never qualifies
 * output, and bars a fit trained without lock from prediction until reset.
 */
TEST_CASE( "Assisted diagnostic mode never qualifies output without upstream lock", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::stepAssisted;
    MagAOX::app::loPredCtrlAcc_operational::sensorHistory;
    MagAOX::app::loPredCtrlAcc_operational::outputValid;
#endif
    // clang-format on
    Controller app( true );
    app.m_settings.m_history       = 6;
    app.m_settings.m_accelChannels = 16;
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.resetModels();
    Plant plant( .02 );
    app.m_source          = [&]( std::uint64_t n ) { return plant.acc( n ); };
    DDSPC::Matrix y       = DDSPC::Matrix::Zero( 2, 1 );
    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 );
    std::uint64_t n       = 0;
    auto          frame   = [&]( bool locked )
    {
        const auto slots = app.advanceAssisted( y, -1, locked );
        y.setZero();
        for( int p = 0; p < slots.cols(); ++p, ++n )
        {
            const auto e = plant.residual( n, command );
            command      = slots.col( p );
            for( int axis = 0; axis < 2; ++axis )
                y( axis ) += e[axis] / slots.cols();
        }
    };
    // Without the configuration, unlocked frames never reach the assisted learner.
    app.train( "300,0.1,0.1" );
    for( int i = 0; i < 200; ++i )
        frame( false );
    REQUIRE( app.m_multirate->status().m_updates == 0 );
    REQUIRE_FALSE( app.m_unlockedTraining );

    // Evaluation of a locked-trained, frozen fit continues without lock and without contaminating it.
    app.m_syncLocked = true;
    app.train( "1000,0.1,0.1" );
    for( int i = 0; i < 1200; ++i )
        frame( true );
    REQUIRE( app.m_multirate->status().m_ready );
    app.freeze();
    app.m_syncLocked         = false;
    app.m_assistedDiagnostic = true;
    const auto scored        = app.m_multirate->status().m_scored.back();
    for( int i = 0; i < 50; ++i )
        frame( false );
    REQUIRE( app.m_multirate->status().m_scored.back() > scored );
    REQUIRE_FALSE( app.m_unlockedTraining );
    REQUIRE_FALSE( app.outputValid() ); // Unlocked assisted output is never qualified.

    // Learning without lock is allowed in the diagnostic mode, but the fit cannot then predict until reset.
    const auto updates = app.m_multirate->status().m_updates;
    auto       learn   = app.m_indiP_learning;
    learn["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learning( learn ) == 0 );
    for( int i = 0; i < 20; ++i )
        frame( false );
    REQUIRE( app.m_multirate->status().m_updates > updates );
    REQUIRE( app.m_unlockedTraining );
    app.m_syncLocked = true;
    frame( true );
    REQUIRE_FALSE( app.outputValid() );
    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    app.m_predicting = false;
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) < 0 );
    app.resetModels();
    REQUIRE_FALSE( app.m_unlockedTraining );
}

/** \brief The supplied RTC dimensions, prior, gain and calibration stages train each mode while it is selected. */
TEST_CASE( "Operational RTC settings produce finite standard and assisted fits", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::regularization;
    MagAOX::app::loPredCtrlAcc_operational::step;
#endif
    // clang-format on
    for( const bool assisted : { false, true } )
    {
        Controller app( true );
        app.m_settings.m_history       = 15;
        app.m_settings.m_future        = 3;
        app.m_settings.m_accelChannels = 40;
        app.m_settings.m_lambda        = 100;
        app.m_settings.m_prior         = 1e-5f;
        app.m_gain                     = .3f;
        app.m_mode.store( assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard );
        app.resetModels();
        app.m_syncLocked = true;
        Plant plant( .02 );
        app.m_source = [&]( std::uint64_t n ) { return plant.acc( n ); };
        app.train( "1000,0.1,100,1000,0.1,3,1000,0.1,0.1" );
        DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty;
        for( int n = 0; n < 3100; ++n )
        {
            y << std::sin( .09 * n ), std::cos( .13 * n );
            y += .8f * command;
            command = response( assisted ? app.advanceAssisted( y ) : app.advance( y, empty, false ) ) / .8f;
        }
        REQUIRE( app.m_exploration.empty() );
        if( assisted )
        {
            REQUIRE( app.m_multirate->status().m_ready );
            REQUIRE( app.m_gainSets[app.m_gainFront].m_ready );
        }
        else
            REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    }
}

/** \brief B1 (hidden benchmark): RTC dimensions, standard and assisted, learning and frozen; then the WFS thread
 * learning at 2 kHz while the DM worker calculates every 125-us slot at cutoff minus the render lead.
 */
TEST_CASE( "Operational RTC decision time", "[.][rtc-benchmark]" )
{
    using clock    = std::chrono::steady_clock;
    auto configure = []( Controller &app, bool assisted )
    {
        app.m_settings.m_history       = 15;
        app.m_settings.m_future        = 3;
        app.m_settings.m_accelChannels = 40;
        app.m_settings.m_lambda        = 100;
        app.m_settings.m_prior         = 1e-5f;
        app.m_gain                     = .3f;
        app.m_mode.store( assisted ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard );
        app.resetModels();
        app.m_syncLocked = true;
    };
    auto report = []( const char *name, std::vector<double> &times )
    {
        std::sort( times.begin(), times.end() );
        std::printf( "%-34s n %6zu mean %7.1f us p50 %7.1f p99 %7.1f max %7.1f\n",
                     name,
                     times.size(),
                     std::accumulate( times.begin(), times.end(), 0.0 ) / times.size(),
                     times[times.size() / 2],
                     times[times.size() * 99 / 100],
                     times.back() );
    };
    for( const bool assisted : { false, true } )
    {
        Controller app( true );
        configure( app, assisted );
        Plant plant( .02 );
        app.m_source = [&]( std::uint64_t n ) { return plant.acc( n ); };
        app.train( "2000,0.1,100" );
        DDSPC::Matrix       command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty;
        std::vector<double> learning, frozen, slots;
        for( int n = 0; n < 4000; ++n )
        {
            y << std::sin( .09 * n ), std::cos( .13 * n );
            y += .8f * command;
            if( n == 3000 )
                app.freeze();
            const auto before = app.m_learnUs.m_count;
            const auto start  = clock::now();
            const auto out    = assisted ? app.advanceAssisted( y ) : app.advance( y, empty, false );
            const auto us     = std::chrono::duration<double, std::micro>( clock::now() - start ).count();
            command           = response( out ) / .8f;
            if( n >= 500 )
                ( n < 3000 ? learning : frozen ).push_back( assisted ? app.m_learnUs.m_max : us );
            if( assisted && app.m_learnUs.m_count > before )
                app.m_learnUs = {};
        }
        report( assisted ? "assisted WFS frame, learning" : "standard decision+4 slots, learning", learning );
        report( assisted ? "assisted WFS frame, frozen" : "standard decision+4 slots, frozen", frozen );
        if( assisted )
        {
            const auto &compute = app.m_fastStats.m_compute;
            std::printf( "%-34s n %6llu mean %7.2f us p99 %7.1f max %7.1f\n",
                         "assisted slot calculation",
                         (unsigned long long)compute.m_count,
                         compute.mean(),
                         compute.quantile( .99 ),
                         compute.m_max );
        }
    }

    // Concurrent: WFS thread at 2 kHz with learning; DM worker on every 125-us slot with real cutoffs.
    namespace synco = MagAOX::app::synco;
    Controller app( true );
    configure( app, true );
    Plant plant( .02 );
    for( std::uint64_t n = 0; n < 40000; ++n )
        plant.acc( n ); // Pre-generated: both threads only read the samples.
    app.m_source = [&]( std::uint64_t n ) { return plant.m_acc[n]; };
    app.train( "6000,0.1,100" );
    const int frames     = 8000;
    auto      sleepUntil = []( std::int64_t ns )
    {
        if( const auto wait = ns - synco::monotonicNs(); wait > 100000 )
            std::this_thread::sleep_for( std::chrono::nanoseconds( wait - 100000 ) );
        while( synco::monotonicNs() < ns )
        {
        }
    };
    const std::int64_t  t0 = synco::monotonicNs() + 2000000;
    std::atomic<bool>   failed{ false };
    std::vector<double> learn, margin;
    std::thread         wfs(
        [&]
        {
            DDSPC::Matrix y( 2, 1 ), features( 40, 1 ), applied;
            for( int i = 1; i < frames; ++i )
            {
                sleepUntil( t0 + i * 500000 );
                const std::uint64_t anchor = 4 * i;
                for( int j = 0; j < 20; ++j )
                    for( int axis = 0; axis < 2; ++axis )
                        features( 2 * j + axis ) = plant.m_acc[anchor - j][axis];
                std::int64_t ordinal;
                { //mutex scope
                    std::lock_guard<std::mutex> lock( app.m_commandMutex );
                    ordinal = app.appliedSlots( applied );
                }
                y << std::sin( .09 * i ), std::cos( .13 * i );
                if( applied.cols() == 4 )
                    y += response( applied ); // The previous frame's copied slots act on this measurement.
                const auto start = synco::monotonicNs();
                {
                    std::lock_guard<std::mutex> lock( app.m_modelMutex ); // Serializes the anchor with step().
                    app.m_adcAnchor  = anchor;
                    app.m_adcSegment = ( std::uint64_t( 1 ) << 32 ) ^ 1;
                }
                if( app.step( y, i, features, true, applied, ordinal, start ) != 0 )
                    failed = true;
                learn.push_back( ( synco::monotonicNs() - start ) / 1000. );
            }
        } );
    std::uint64_t missed = 0;
    for( std::int64_t n = 4; n < 4 * frames; ++n )
    {
        const std::int64_t cutoff = t0 + n * 125000 + 55000;
        app.deliver( n ); // ACC sample n is available at the start of slot n.
        synco::Request request{ 1, 1, n / 4, int( n % 4 ), cutoff + 70000, cutoff };
        synco::Record  schedule{};
        schedule[synco::steps] = 4;
        std::array<float, 4> image{};
        app.outputCandidate( request, schedule, image );
        const auto done = synco::monotonicNs();
        margin.push_back( ( cutoff - done ) / 1000. );
        missed += done > cutoff;
        sleepUntil( cutoff );
        synco::CommitRecord commit{};
        commit[synco::cmdEpoch] = commit[synco::cmdGeneration] = 1;
        commit[synco::cmdCycle]                                = n / 4;
        commit[synco::cmdStep]                                 = n % 4;
        app.copied( commit, image );
    }
    wfs.join();
    REQUIRE_FALSE( failed );
    report( "concurrent WFS frame (learning)", learn );
    std::sort( margin.begin(), margin.end() );
    std::printf( "concurrent slots %zu: missed cutoffs %llu, margin min %.1f us p1 %.1f us median %.1f us\n",
                 margin.size(),
                 (unsigned long long)missed,
                 margin.front(),
                 margin[margin.size() / 100],
                 margin[margin.size() / 2] );
    const auto &stats = app.m_fastStats;
    std::printf( "concurrent predictions %llu holds %llu lagging %llu; slot calc mean %.2f us p99 %.1f max %.1f; "
                 "ACC age at calc mean %.1f us, at application mean %.1f us; learning updates %llu\n",
                 (unsigned long long)stats.m_predictions,
                 (unsigned long long)stats.m_holds,
                 (unsigned long long)stats.m_lagging,
                 stats.m_compute.mean(),
                 stats.m_compute.quantile( .99 ),
                 stats.m_compute.m_max,
                 stats.m_ageCalc.mean(),
                 stats.m_ageApply.mean(),
                 (unsigned long long)app.m_multirate->status().m_updates );
    std::printf( "concurrent history breaks %llu, slot shifts %llu, ACC anomalies %llu\n",
                 (unsigned long long)app.m_multirate->status().m_breaks,
                 (unsigned long long)app.m_slotShifts,
                 (unsigned long long)stats.m_anomalies );
    REQUIRE( app.m_multirate->status().m_ready );
}

/** \brief Startup runs the producer loop on the framework DM worker; shutdown joins it after the bounded removal wait.
 */
TEST_CASE( "Operational startup runs the DM worker through threadStart", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::appStartup;
    MagAOX::app::loPredCtrlAcc_operational::outputThreadStart;
    MagAOX::app::loPredCtrlAcc_operational::stopOutput;
#endif
    // clang-format on
    Controller app( true );
    app.m_accelEnabled = false;
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.m_outputThread.joinable() );
    const auto start = std::chrono::steady_clock::now();
    REQUIRE( app.appShutdown() == -1 ); // No auxiliary or consumer here, so removal cannot be confirmed.
    REQUIRE_FALSE( app.m_outputThread.joinable() );
    REQUIRE( std::chrono::steady_clock::now() - start < std::chrono::seconds( 2 ) );
}

/** \brief Original control names and exploration triples work without custom authorization or timing anchors. */
TEST_CASE( "Operational INDI controls retain the original workflow", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_gain;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_mode;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_reset;
    MagAOX::app::loPredCtrlAcc_operational::parseExploration;
#endif
    // clang-format on
    Controller app( true );
    auto       gain = app.m_indiP_gain;
    gain["target"].set( .07 );
    REQUIRE( app.newCallBack_m_indiP_gain( gain ) == 0 );
    REQUIRE( app.m_indiP_gain["current"].get<float>() == Approx( .07 ) );
    gain["target"].set( std::numeric_limits<double>::quiet_NaN() );
    REQUIRE( app.newCallBack_m_indiP_gain( gain ) < 0 );
    auto learning = app.m_indiP_learning;
    learning["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learning( learning ) == 0 );
    REQUIRE( app.m_learning );
    auto sequence = app.m_indiP_exploration;
    sequence["target"].set( "1000,0.1,100,1000,0.1,3,1000,0.1,0.1" );
    REQUIRE( app.newCallBack_m_indiP_exploration( sequence ) == 0 );
    REQUIRE( app.m_exploration.size() == 3 );
    for( const auto &bad : { "", "1,0.1", "1,0.1,0", "-1,0.1,3", "1,nan,3", "1,0.1,3," } )
        REQUIRE_THROWS( app.parseExploration( bad ) );
    auto mode = app.m_indiP_mode;
    mode["standard"].setSwitchState( pcf::IndiElement::Off );
    mode["accel_assisted"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_mode( mode ) == 0 );
    REQUIRE( app.m_mode.load() == DDSPC::ShadowMode::AccelAssisted );
    auto reset = app.m_indiP_reset;
    reset["request"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_reset( reset ) == 0 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
}

/// \cond RTC_TEST_SUPPORT
namespace
{
using ModalGrabber   = MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc_operational>;
using ModalTelemeter = MagAOX::app::dev::telemeter<MagAOX::app::loPredCtrlAcc_operational>;

/// One run over a fixed six-mode WFS frame sequence, through the WFS monitor or directly through step().
struct ModalRun
{
    std::vector<DDSPC::Matrix>      m_slots;     ///< Rendered DM slots after each frame, modes by four.
    std::vector<std::vector<float>> m_frames;    ///< Each incoming WFS frame.
    std::vector<std::vector<float>> m_published; ///< Each main-loop modal frame (empty without the monitor).
    DDSPC::ShadowStatus             m_standard;  ///< Final standard-fit status.
};

/// Train standard DDSPC with probes, as test "Standard WFS monitor trains...", and collect its output.
/** With monitor, frames enter through processImage(), which also builds the main-loop modal frame. Without it, the same
 * measurements go straight to step(), as Controller::advance() does: DDSPC with no modal output path at all.
 */
ModalRun runModal( bool monitor /**< [in] Use the WFS monitor callback. */ )
{
    constexpr int                modes = 6;
    Controller                   app( true );
    IMAGE_METADATA               md{};
    std::array<float, 2 * modes> wfs{};
    app.m_imageStream.md      = &md;
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 3;
    md.size[0]                = modes;
    md.size[1]                = 1;
    md.size[2]                = 2;
    app.m_width               = modes;
    app.m_height              = 1;
    REQUIRE( app.allocate( MagAOX::app::dev::shmimT{} ) == 0 );
    // Size the output as the frameGrabber thread does; the fixture stream is absent, so it would be created.
    char path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), "ddspc_modal_run_test" ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    static_cast<ModalGrabber &>( app ).m_shmimName = "ddspc_modal_run_test";
    REQUIRE( app.configureAcquisition() == 0 );
    app.m_gain = .2f;
    app.train();
    app.qualifyOutput();
    REQUIRE( app.m_output.request( true ) ); // Prediction requested: DDSPC owns tip/tilt.
    ModalRun run;
    for( unsigned n = 1; n <= 600; ++n )
    {
        md.cnt0     = n;
        md.cnt1     = n % 2;
        md.atime    = { time_t( n ), 7 };
        auto *frame = wfs.data() + modes * md.cnt1;
        for( int m = 0; m < modes; ++m )
            frame[m] = ( m + 1 ) * std::sin( n * .13f + m ) + 1e-7f * m; // Distinct, nonzero tip/tilt included.
        const auto input = wfs;
        if( monitor )
        {
            REQUIRE( app.processImage( frame, MagAOX::app::dev::shmimT{} ) == 0 );
            REQUIRE( wfs == input ); // The WFS input stream is never written.
            std::vector<float> out( modes, -1 );
            REQUIRE( app.m_modalFresh );
            REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
            REQUIRE( app.m_currImageTimestamp.tv_sec == time_t( n ) ); // The WFS frame's acquisition time.
            // A repeated counter publishes nothing.
            REQUIRE( app.processImage( frame, MagAOX::app::dev::shmimT{} ) == 0 );
            REQUIRE_FALSE( app.m_modalFresh );
            run.m_published.push_back( out );
        }
        else
        {
            DDSPC::Matrix y( 2, 1 ), empty, applied;
            y << frame[0], frame[1];
            std::int64_t ordinal;
            { //mutex scope
                std::lock_guard<std::mutex> lock( app.m_commandMutex );
                ordinal = app.appliedSlots( applied );
            }
            REQUIRE( app.step( y, n, empty, false, applied, ordinal, MagAOX::app::synco::monotonicNs() ) == 0 );
            REQUIRE_FALSE( app.m_modalFresh );
        }
        DDSPC::Matrix slots( 2, 4 );
        for( int j = 0; j < 4; ++j )
            slots.col( j ) = app.slot( app.m_slot++ );
        run.m_slots.push_back( slots );
        run.m_frames.emplace_back( frame, frame + modes );
    }
    run.m_standard = app.m_models->status( DDSPC::ShadowMode::Standard );
    return run;
}

/// Six-mode WFS input and a modal output named name.
void prepareModal( Controller        &app /**< [in,out] Application under test. */,
                   IMAGE_METADATA    &md /**< [out] WFS metadata kept alive by the caller. */,
                   const std::string &name /**< [in] Output stream name. */ )
{
    app.m_imageStream.md                           = &md;
    app.m_width                                    = 6;
    app.m_height                                   = 1;
    static_cast<ModalGrabber &>( app ).m_shmimName = name;
    REQUIRE( app.allocate( MagAOX::app::dev::shmimT{} ) == 0 );
}
} // namespace
/// \endcond

/** \brief While DDSPC owns tip/tilt, every WFS frame yields a main-loop modal frame with exactly modes 0 and 1 zeroed
 * and every higher mode copied bitwise; the WFS input is never written, and DDSPC learning and every rendered DM slot
 * are bitwise identical to the same measurements stepped without the modal path.
 */
TEST_CASE( "Modal output zeroes tip/tilt only and leaves DDSPC unchanged", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::processImage;
    MagAOX::app::loPredCtrlAcc_operational::publishModal;
    MagAOX::app::loPredCtrlAcc_operational::loadImageIntoStream;
    MagAOX::app::loPredCtrlAcc_operational::allocate;
#endif
    // clang-format on
    const auto direct = runModal( false ), monitor = runModal( true );
    REQUIRE( direct.m_published.empty() );
    REQUIRE( monitor.m_published.size() == monitor.m_frames.size() ); // One modal frame per WFS frame.
    for( std::size_t n = 0; n < monitor.m_frames.size(); ++n )
    {
        const auto &in = monitor.m_frames[n], &out = monitor.m_published[n];
        REQUIRE( in == direct.m_frames[n] );
        REQUIRE( in[0] != 0 );
        REQUIRE( in[1] != 0 );
        REQUIRE( out[0] == 0 );
        REQUIRE( out[1] == 0 );
        REQUIRE( std::equal( in.begin() + 2, in.end(), out.begin() + 2 ) ); // Exact, including sub-ulp offsets.
        REQUIRE( monitor.m_slots[n] == direct.m_slots[n] );                 // Bitwise identical DDSPC output.
    }
    INFO( "standard updates " << monitor.m_standard.m_updates );
    REQUIRE( monitor.m_standard.m_ready ); // Trained and predicting, so the comparison covers DDSPC predictions.
    REQUIRE( monitor.m_standard.m_updates == direct.m_standard.m_updates );
    REQUIRE( monitor.m_standard.m_originRmse == direct.m_standard.m_originRmse );
}

/** \brief A latched output fault keeps its reason after the zero is acknowledged, and outputValid() records the
 * condition that first ended a qualified period, so a qualification fault can be explained afterwards.
 */
TEST_CASE( "Output faults keep the condition that caused them", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::SyncoOutputState::fault;
    MagAOX::app::loPredCtrlAcc_operational::outputValid;
#endif
    // clang-format on
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = config.m_producerDefaults = true;
    MagAOX::app::SyncoOutputState output( config, 2, true );
    const std::vector<float>      zero( 2, 0 );
    REQUIRE( output.request( true, true ) );
    output.fault( "input_or_model_qualification_lost" );
    output.submitted( zero, true, 0 );
    output.acknowledge( true );
    CHECK( std::string( output.status().m_reason ) == "fault_zero_acknowledged" );
    CHECK( std::string( output.status().m_faultReason ) == "input_or_model_qualification_lost" );
    CHECK( output.status().m_faults == 1 );

    Controller app( true );
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.m_inputValid.store( true );
    app.m_candidateValid.store( true );
    app.m_assistedValid.store( true );
    app.m_syncLocked.store( true );
    app.m_wfsReceipt.store( MagAOX::app::synco::monotonicNs() );
    REQUIRE( app.outputValid() );

    // One frame without a complete ACC window ends qualification; the later loss of lock is a consequence.
    app.m_windowReason.store( "stale_sample" );
    app.m_assistedValid.store( false );
    CHECK_FALSE( app.outputValid() );
    app.m_windowReason.store( "ok" );
    app.m_assistedValid.store( true );
    app.m_syncLocked.store( false );
    CHECK_FALSE( app.outputValid() );
    CHECK( std::string( app.m_invalidReason.load() ) == "acc_window" );
    CHECK( std::string( app.m_invalidWindow.load() ) == "stale_sample" );

    // After qualifying again, a new failure is recorded.
    app.m_syncLocked.store( true );
    app.m_wfsReceipt.store( MagAOX::app::synco::monotonicNs() );
    REQUIRE( app.outputValid() );
    app.m_wfsReceipt.store( MagAOX::app::synco::monotonicNs() - std::int64_t( 1e9 ) );
    CHECK_FALSE( app.outputValid() );
    CHECK( std::string( app.m_invalidReason.load() ) == "wfs_stale" );
}

/** \brief The modal output requires DDSPC to control exactly modes 0 and 1 and a stream distinct from the WFS input and
 * the DM transport, so the zeroed vector can never be read back as DDSPC input.
 */
TEST_CASE( "Modal output configuration rejects feedback and other modes", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::validateModalOutput;
    MagAOX::app::loPredCtrlAcc_operational::loadConfigImpl;
#endif
    // clang-format on
    Controller app( true );
    auto      &grabber      = static_cast<ModalGrabber &>( app );
    app.m_shmimName         = "aol1_modevalWFS";
    app.m_syncopationTiming = "dmtweeter_synco_timing";
    grabber.m_shmimName     = "loPredCtrlAcc"; // The frameGrabber default: the application's name.
    REQUIRE_NOTHROW( app.validateModalOutput() );
    for( const auto *name : { "", "aol1_modevalWFS", "dm01disp05_aux", "dmtweeter_synco_timing" } )
    {
        grabber.m_shmimName = name;
        REQUIRE_THROWS( app.validateModalOutput() );
    }
    grabber.m_shmimName         = "loPredCtrlAcc";
    app.m_calibration.m_indices = { 0, 2 };
    REQUIRE_THROWS( app.validateModalOutput() );
    app.m_calibration.m_indices = { 0, 1 };
    app.m_settings.m_modes      = 1;
    REQUIRE_THROWS( app.validateModalOutput() );
}

/** \brief The publisher takes an exclusive lock, creates an absent stream, attaches an idle one of the same shape, and
 * refuses a stream of another shape, one not ready, or one another process is still publishing.
 */
TEST_CASE( "Modal output never shares or reshapes another publisher's stream", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::configureAcquisition;
    MagAOX::app::loPredCtrlAcc_operational::modalRefused;
#endif
    // clang-format on
    const std::string name = "ddspc_modal_output_test";
    char              path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    std::filesystem::remove( std::string( path ) + ".producer" );
    Controller     app( true ), other( true );
    IMAGE_METADATA md{}, otherMd{};
    prepareModal( app, md, name );
    prepareModal( other, otherMd, name );
    auto &grabber = static_cast<ModalGrabber &>( app );

    REQUIRE( app.configureAcquisition() == 0 ); // Absent: the frameGrabber creates it at the WFS size.
    REQUIRE( grabber.m_width == 6 );
    REQUIRE( grabber.m_height == 1 );
    REQUIRE( grabber.m_dataType == _DATATYPE_FLOAT );
    REQUIRE( grabber.m_imageStream == nullptr );
    REQUIRE( app.m_modalLock >= 0 );
    REQUIRE( other.configureAcquisition() == -1 ); // A second publisher cannot take the lock.
    REQUIRE( other.m_modalLock < 0 );

    std::ofstream( path ) << "stream";
    IMAGE_METADATA existing{};
    IMAGE          image{};
    existing.naxis    = 2;
    existing.size[0]  = 6;
    existing.size[1]  = 1;
    existing.datatype = _DATATYPE_FLOAT;
    image.md          = &existing;
    REQUIRE( app.configureAcquisition() == -1 ); // Present but not yet openable: wait, never recreate.
    grabber.m_existing = &image;
    REQUIRE( app.configureAcquisition() == 0 ); // Idle and the same shape: attach.
    REQUIRE( grabber.m_imageStream != nullptr );
    existing.size[0] = 5;
    REQUIRE( app.configureAcquisition() == -1 ); // Another shape: never resized.
    REQUIRE( grabber.m_imageStream == nullptr );
    existing.size[0] = 6;
    std::atomic<bool> stop{ false };
    std::thread       writer(
        [&]
        {
            while( !stop )
            {
                ++existing.cnt0; // Another process still publishing, e.g. the original loPredCtrl.
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
        } );
    const int active = app.configureAcquisition();
    stop             = true;
    writer.join();
    REQUIRE( active == -1 );
    REQUIRE( grabber.m_imageStream == nullptr );
    std::filesystem::remove( path );
}

/** \brief Main-loop tip/tilt passes through unchanged with prediction off and is zeroed from the accepted prediction
 * request until DDSPC's own-channel zero is confirmed, including after a fault; the startup zero never withholds it.
 */
TEST_CASE( "Modal output zeroes tip/tilt only while DDSPC owns it", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::modalZeroed;
    MagAOX::app::loPredCtrlAcc_operational::publishModal;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on

    // The ownership rule through the real own-channel interlock.
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = config.m_producerDefaults = true;
    MagAOX::app::SyncoOutputState output( config, 2, true );
    const std::vector<float>      zero( 2, 0 ), command{ .1f, -.2f };
    const auto                    owned = [&]( bool engaged )
    {
        return Controller::modalZeroed( output.status(), engaged );
    };

    REQUIRE( output.initialize() );
    CHECK_FALSE( owned( false ) ); // The startup zero, still unconfirmed, never withholds main-loop tip/tilt.
    output.submitted( zero, true, 0 );
    output.acknowledge( true );
    REQUIRE( output.status().m_zeroConfirmed );

    REQUIRE( output.request( true, true ) );
    CHECK( owned( false ) ); // Zeroed from the accepted request, before the application records it.
    output.submitted( command, false, 0 );
    output.acknowledge( true );
    CHECK( owned( true ) );

    REQUIRE( output.request( false, true ) );
    CHECK( owned( true ) ); // The DDSPC contribution is still on the DM.
    output.submitted( zero, true, 0 );
    CHECK( owned( true ) ); // Its removal is not yet confirmed.
    output.acknowledge( true );
    CHECK_FALSE( owned( true ) );

    REQUIRE( output.request( true, true ) );
    output.submitted( command, false, 0 );
    output.acknowledge( true );
    output.fault( "test_fault" );
    CHECK( owned( true ) ); // A fault keeps tip/tilt zeroed until the removal is confirmed.
    output.submitted( zero, true, 0 );
    output.acknowledge( true );
    CHECK_FALSE( owned( true ) );

    // The same rule on the WFS monitor path, driven by the operator's prediction requests.
    const std::string name = "ddspc_modal_owner_test";
    char              path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    Controller           app( true );
    IMAGE_METADATA       md{};
    std::array<float, 6> wfs{ 1, 2, 3, 4, 5, 6 };
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 2;
    md.size[0]                = 6;
    md.size[1]                = 1;
    prepareModal( app, md, name );
    REQUIRE( app.configureAcquisition() == 0 );
    app.qualifyOutput();
    app.train();
    app.m_predicting = false;

    std::uint64_t counter = 0;
    const auto    publish = [&]()
    {
        md.cnt0 = ++counter;
        REQUIRE( app.processImage( wfs.data(), MagAOX::app::dev::shmimT{} ) == 0 );
        std::vector<float> out( wfs.size(), -1 );
        REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
        return out;
    };
    const std::vector<float> input( wfs.begin(), wfs.end() ), zeroed{ 0, 0, 3, 4, 5, 6 };

    CHECK( publish() == input ); // Prediction off: unchanged, as if DDSPC were absent.
    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    REQUIRE( app.m_modalEngaged );
    CHECK( publish() == zeroed );
    CHECK( publish() == zeroed );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_modalEngaged ); // Still requested.
    CHECK( app.m_indiP_control["main_loop_tt"].get<std::string>() == "zeroed" );

    predict["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    CHECK( publish() == input ); // Nothing was written here, so the removal is already confirmed.
    REQUIRE( app.m_output.status().m_zeroConfirmed );
    REQUIRE( app.appLogic() == 0 );
    CHECK_FALSE( app.m_modalEngaged ); // Removed: the startup-zero case again.
    CHECK( app.m_indiP_control["main_loop_tt"].get<std::string>() == "passthrough" );
    std::filesystem::remove( path );
}

/** \brief zero_dm stops prediction and writes a fresh own-channel zero; release_tt also returns main-loop tip/tilt at
 * once, before that zero is confirmed.
 */
TEST_CASE( "Operator zero and tip/tilt release controls", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_zeroDm;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_releaseTT;
    MagAOX::app::loPredCtrlAcc_operational::stopPrediction;
#endif
    // clang-format on
    const std::string name = "ddspc_modal_release_test";
    char              path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    Controller           app( true );
    IMAGE_METADATA       md{};
    std::array<float, 6> wfs{ 1, 2, 3, 4, 5, 6 };
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 2;
    md.size[0]                = 6;
    md.size[1]                = 1;
    prepareModal( app, md, name );
    REQUIRE( app.configureAcquisition() == 0 );
    app.qualifyOutput();
    app.train();
    app.m_predicting = false;

    std::uint64_t counter = 0;
    const auto    publish = [&]()
    {
        md.cnt0 = ++counter;
        REQUIRE( app.processImage( wfs.data(), MagAOX::app::dev::shmimT{} ) == 0 );
        std::vector<float> out( wfs.size(), -1 );
        REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
        return out;
    };
    const std::vector<float> input( wfs.begin(), wfs.end() ), zeroed{ 0, 0, 3, 4, 5, 6 };
    auto                     predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );

    // zero_dm: prediction stops and a new zero is required even though nothing is outstanding.
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    auto zero = app.m_indiP_zeroDm;
    zero["request"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_zeroDm( zero ) == 0 );
    CHECK_FALSE( app.m_predicting );
    CHECK( app.m_indiP_predicting["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK_FALSE( app.m_output.status().m_requested );
    CHECK_FALSE( app.m_output.status().m_zeroConfirmed ); // The fresh zero is pending.
    CHECK( publish() == zeroed );                         // DDSPC was engaged: tip/tilt waits for the zero.

    // release_tt: the main loop takes tip/tilt back on the next frame, with the zero still unconfirmed.
    auto release = app.m_indiP_releaseTT;
    release["request"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_releaseTT( release ) == 0 );
    CHECK_FALSE( app.m_modalEngaged );
    CHECK_FALSE( app.m_output.status().m_zeroConfirmed );
    CHECK( publish() == input );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_control["main_loop_tt"].get<std::string>() == "passthrough" );

    // A zero with prediction already off never withholds main-loop tip/tilt.
    REQUIRE( app.newCallBack_m_indiP_zeroDm( zero ) == 0 );
    CHECK( publish() == input );
    std::filesystem::remove( path );
}

/// \cond RTC_TEST_SUPPORT
namespace
{
/// Select ddspc_mode element on through the real callback, with every other element off.
int selectMode( Controller &app /**< [in,out] Application under test. */, const char *on /**< [in] Element. */ )
{
    auto mode = app.m_indiP_mode;
    for( const char *element : { "standard", "accel_assisted", "main_loop", "mfilt_copy_8k", "mfilt_copy_2k" } )
    {
        mode[element].setSwitchState( std::string( element ) == on ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }

    return app.newCallBack_m_indiP_mode( mode );
}

/// Closed loop through the main loop's mfilt, DM = DM - gain*input, with the main-loop route selected.
/** Returns the residual over the last 2000 of 6000 frames relative to the disturbance. With learn, DDSPC learns with
 * probes for 3000 frames, then is frozen and keeps predicting; without it, the main loop corrects tip/tilt alone.
 */
double mainLoopRun( bool learn /**< [in] Train and predict with DDSPC; otherwise prediction stays off. */ )
{
    const std::string name = "ddspc_main_loop_run_test";
    char              path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    Controller           app( true );
    IMAGE_METADATA       md{};
    std::array<float, 6> wfs{};
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 2;
    md.size[0]                = 6;
    md.size[1]                = 1;
    prepareModal( app, md, name );
    REQUIRE( app.configureAcquisition() == 0 );
    REQUIRE( selectMode( app, "main_loop" ) == 0 );
    app.m_gain = 1; // Untrained, the delta is the measurement: the main loop's own integrator.
    if( learn )
    {
        app.train();
    }

    constexpr double gain = .5;
    double           dm[2]{}, residual = 0, uncontrolled = 0;
    for( int n = 0; n < 6000; ++n )
    {
        const double disturbance[2]{ std::sin( .09 * n ) + .4 * std::cos( .17 * n ),
                                     .8 * std::cos( .13 * n ) + .3 * std::sin( .055 * n ) };
        for( int m = 0; m < 2; ++m )
        {
            wfs[m] = float( disturbance[m] + dm[m] );
        }

        if( n >= 4000 )
        {
            residual += wfs[0] * wfs[0] + wfs[1] * wfs[1];
            uncontrolled += disturbance[0] * disturbance[0] + disturbance[1] * disturbance[1];
        }

        if( learn && n == 3000 )
        {
            app.freeze();
        }

        md.cnt0 = n + 1;
        REQUIRE( app.processImage( wfs.data(), MagAOX::app::dev::shmimT{} ) == 0 );
        std::vector<float> out( wfs.size() );
        REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
        for( int m = 0; m < 2; ++m )
        {
            dm[m] -= gain * out[m]; // mfilt, which acts on the next WFS frame.
        }
    }

    REQUIRE_FALSE( app.m_output.status().m_requested ); // The own DM channel is never used.
    REQUIRE( app.m_slotPlan.isZero() );
    if( learn )
    {
        REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    }

    std::filesystem::remove( path );
    return std::sqrt( residual / uncontrolled );
}
} // namespace
/// \endcond

/** \brief ddspc_mode main_loop, as loPredCtrl: each frame's standard DDSPC delta replaces modes 0 and 1 of the modal
 * output after the decision, for the main loop's mfilt to integrate. Prediction needs neither the own channel nor the
 * integrator parameters, the own channel is never requested, and prediction off passes tip/tilt through unchanged.
 */
TEST_CASE( "Main-loop route sends the DDSPC delta through the modal output", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_mode;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::publishModal;
    MagAOX::app::loPredCtrlAcc_operational::mainLoopInput;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on
    const std::string name = "ddspc_main_loop_test";
    char              path[4096];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) == 0 );
    std::filesystem::create_directories( std::filesystem::path( path ).parent_path() );
    std::filesystem::remove( path );
    Controller           app( true );
    IMAGE_METADATA       md{};
    std::array<float, 6> wfs{ 1, 2, 3, 4, 5, 6 };
    app.m_imageStream.array.F = wfs.data();
    md.naxis                  = 2;
    md.size[0]                = 6;
    md.size[1]                = 1;
    prepareModal( app, md, name );
    REQUIRE( app.configureAcquisition() == 0 );
    app.m_integrate = true; // The main loop's parameters are never received; this route does not need them.
    app.m_gain      = .5f;
    auto learning   = app.m_indiP_learning;
    learning["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learning( learning ) == 0 );

    std::uint64_t counter = 0;
    const auto    publish = [&]()
    {
        md.cnt0 = ++counter;
        REQUIRE( app.processImage( wfs.data(), MagAOX::app::dev::shmimT{} ) == 0 );
        std::vector<float> out( wfs.size(), -1 );
        REQUIRE( app.loadImageIntoStream( out.data() ) == 0 );
        return out;
    };
    const std::vector<float> input( wfs.begin(), wfs.end() ), delta{ .5f, 1, 3, 4, 5, 6 };

    REQUIRE( selectMode( app, "main_loop" ) == 0 );
    CHECK( app.m_mode.load() == DDSPC::ShadowMode::Standard );
    CHECK( app.m_indiP_mode["main_loop"].getSwitchState() == pcf::IndiElement::On );
    CHECK( app.m_indiP_mode["standard"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( publish() == input ); // Prediction off: unchanged.

    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    CHECK_FALSE( app.m_output.status().m_requested );
    CHECK_FALSE( app.m_modalEngaged );
    CHECK( publish() == delta ); // Untrained: gain*y in tip/tilt, every other mode copied.
    CHECK( publish() == delta );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_predicting ); // Not stopped for lack of an own-channel request.
    CHECK( app.m_indiP_control["requested_mode"].get<std::string>() == "main_loop" );
    CHECK( app.m_indiP_control["main_loop_tt"].get<std::string>() == "predicted" );
    CHECK( app.m_slotPlan.isZero() );

    // The fit's applied input is the published delta, not the own channel's copies.
    { //mutex scope
        std::lock_guard<std::mutex> lock( app.m_modelMutex );
        app.m_pending = {};
        app.mainLoopInput();
        REQUIRE( app.m_pending.m_appliedValid );
        CHECK( app.m_pending.m_appliedCommand( 0 ) == Approx( .5 ) );
        CHECK( app.m_pending.m_appliedCommand( 1 ) == Approx( 1 ) );
    }

    // A mode change while predicting stops prediction, logged, and tip/tilt passes through at once.
    REQUIRE( selectMode( app, "standard" ) == 0 );
    CHECK_FALSE( app.m_predicting );
    CHECK( app.m_textLogs.back().first == "DDSPC prediction stopped: DDSPC mode changed" );
    CHECK( app.m_indiP_mode["standard"].getSwitchState() == pcf::IndiElement::On );
    CHECK( app.m_indiP_mode["main_loop"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( publish() == input );
    std::filesystem::remove( path );
}

/** \brief Through the main loop's mfilt, the main-loop route learns from its published deltas and predicts a
 * disturbance the main loop's integrator alone corrects poorly.
 */
TEST_CASE( "Main-loop route trains and corrects through mfilt", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    MagAOX::app::loPredCtrlAcc_operational::processImage;
#endif
    // clang-format on
    const double alone = mainLoopRun( false ), ddspc = mainLoopRun( true );
    INFO( "residual ratio: main loop alone " << alone << ", DDSPC through the main loop " << ddspc );
    REQUIRE( ddspc < .5 * alone );
}

/** \brief ddspc_mode mfilt_copy_8k runs no DDSPC: with predict on, the main loop's tip/tilt integrator,
 * s = clamp(mult*s - gain*y, +/-limit) with its own parameters, is rendered on every own-channel slot even with
 * [integrator] disabled, main-loop tip/tilt is zeroed, nothing learns, and learn_std is refused.
 */
TEST_CASE( "mfilt copy applies the main loop's tip/tilt integrator on the own channel", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_mode;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStd;
    MagAOX::app::loPredCtrlAcc_operational::step;
    MagAOX::app::loPredCtrlAcc_operational::applyTraining;
#endif
    // clang-format on
    Controller app( true );
    REQUIRE( selectMode( app, "mfilt_copy_8k" ) == 0 );
    app.qualifyOutput();
    CHECK( app.m_mode.load() == DDSPC::ShadowMode::Standard );
    CHECK( app.m_indiP_mode["mfilt_copy_8k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( app.m_indiP_mode["mfilt_copy_2k"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_indiP_mode["standard"].getSwitchState() == pcf::IndiElement::Off );

    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == -1 ); // The main loop's parameters are still unknown.
    CHECK( app.m_textLogs.back().first ==
           "DDSPC prediction request refused: the main loop's integrator parameters have not been received" );

    // holoop gain .25, mc .9, limit .3; mode 0 gain factor 2, mode 1 limit factor 2.
    app.mainLoop( .25, .9, .3, { 2, 1, 7 }, { 1, 1, 7 }, { 1, 2, 7 } );
    app.m_integrate = false; // [integrator] enabled=false: the copy integrates regardless.
    app.m_gain      = .2f;   // Unused: the copy's input is the measurement itself.
    auto learning   = app.m_indiP_learning;
    learning["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learning( learning ) == 0 );
    auto learnStd = app.m_indiP_learnStd;
    learnStd["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learnStd( learnStd ) == -1 );
    CHECK( app.m_textLogs.back().first == "DDSPC standard learning sequence refused: mfilt copy runs no DDSPC model" );

    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 ); // No model needed.
    CHECK( app.m_output.status().m_requested );
    CHECK( app.m_modalEngaged ); // Main-loop tip/tilt is zeroed.

    const std::array<double, 2> gain{ .5, .25 }, limit{ .3, .6 };
    std::array<double, 2>       expected{};
    DDSPC::Matrix               y( 2, 1 ), empty;
    for( int n = 0; n < 12; ++n )
    {
        y << .1f + .05f * n, -.6f;
        const auto slots = app.advance( y, empty, false );
        for( int mode = 0; mode < 2; ++mode )
        {
            expected[mode] = std::clamp( .9 * expected[mode] - gain[mode] * y( mode ), -limit[mode], limit[mode] );
            for( int j = 0; j < 4; ++j )
            {
                REQUIRE( slots( mode, j ) == Approx( expected[mode] ).margin( 1e-6 ) );
            }
        }
    }
    REQUIRE( expected[0] == Approx( -.3 ) ); // The limit was reached.
    CHECK( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
    REQUIRE( app.outputValid() );

    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_predicting );
    CHECK( app.m_indiP_control["requested_mode"].get<std::string>() == "mfilt_copy_8k" );
    CHECK( app.m_indiP_control["main_loop_tt"].get<std::string>() == "zeroed" );

    predict["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    REQUIRE( app.advance( y, empty, false ).isZero() );
    REQUIRE( app.m_intState.isZero() ); // The next start is from zero, with the main loop's held state.
}

/** \brief ddspc_mode mfilt_copy_2k computes the same mfilt copy but offers each WFS decision's command to synco only
 * until it is copied once, so the DDSPC channel is written once per frame and synco holds it on the other slots. A
 * slot that is not copied (late) is retried on the next, and a late receipt for an earlier decision does not count as
 * the newer decision's copy.
 */
TEST_CASE( "mfilt copy 2k writes the DM channel once per WFS frame", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::outputCandidate;
    MagAOX::app::loPredCtrlAcc_operational::copied;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_mode;
#endif
    // clang-format on
    namespace synco = MagAOX::app::synco;
    using Image     = std::array<float, 4>;
    Controller app( true );
    REQUIRE( selectMode( app, "mfilt_copy_2k" ) == 0 );
    CHECK( app.m_indiP_mode["mfilt_copy_2k"].getSwitchState() == pcf::IndiElement::On );
    CHECK( app.m_indiP_mode["mfilt_copy_8k"].getSwitchState() == pcf::IndiElement::Off );
    app.qualifyOutput();
    app.mainLoop( .25, .9, .3, { 2, 1, 7 }, { 1, 1, 7 }, { 1, 2, 7 } );
    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );

    DDSPC::Matrix empty;
    const auto    decide = [&]( const DDSPC::Matrix &y )
    {
        DDSPC::Matrix applied;
        std::int64_t  ordinal;
        { //mutex scope
            std::lock_guard<std::mutex> lock( app.m_commandMutex );
            ordinal = app.appliedSlots( applied );
        }
        REQUIRE( app.step( y, app.m_wfsCounter + 1, empty, false, applied, ordinal, synco::monotonicNs() ) == 0 );
    };
    std::int64_t ordinal = 8;
    const auto   offer   = [&]( Image &image )
    {
        const synco::Request request{ 1, 1, ordinal / 4, int( ordinal % 4 ), 0, 0 };
        synco::Record        schedule{};
        schedule[synco::steps] = 4;
        image                  = {};
        return app.outputCandidate( request, schedule, image );
    };
    const auto copy = [&]( std::int64_t slot, const Image &image )
    {
        synco::CommitRecord commit{};
        commit[synco::cmdEpoch]      = 1;
        commit[synco::cmdGeneration] = 1;
        commit[synco::cmdCycle]      = slot / 4;
        commit[synco::cmdStep]       = slot % 4;
        app.copied( commit, image );
    };
    const auto modal = [&]( const Image &image ) -> DDSPC::Matrix
    {
        return app.m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
    };

    // Every frame: the first slot carries the reference mfilt value; the other three are not offered.
    const std::array<double, 2> gain{ .5, .25 }, limit{ .3, .6 };
    std::array<double, 2>       expected{};
    DDSPC::Matrix               y( 2, 1 );
    Image                       image;
    for( int n = 0; n < 12; ++n )
    {
        y << .1f + .05f * n, -.6f;
        decide( y );
        REQUIRE( offer( image ) );
        for( int mode = 0; mode < 2; ++mode )
        {
            expected[mode] = std::clamp( .9 * expected[mode] - gain[mode] * y( mode ), -limit[mode], limit[mode] );
            REQUIRE( modal( image )( mode ) == Approx( expected[mode] ).margin( 1e-6 ) );
        }
        copy( ordinal++, image );
        for( int j = 1; j < 4; ++j, ++ordinal )
        {
            REQUIRE_FALSE( offer( image ) );
        }
    }

    // A slot that was offered but never copied (late) is offered again on the next slot, then not after its copy.
    y << .2f, .1f;
    decide( y );
    REQUIRE( offer( image ) );
    ++ordinal; // Not copied.
    Image retry;
    REQUIRE( offer( retry ) );
    CHECK( retry == image );
    copy( ordinal++, retry );
    REQUIRE_FALSE( offer( image ) );
    ++ordinal;

    // A late receipt for the previous decision's image does not mark the newer decision as written.
    y << -.3f, .4f;
    decide( y );
    Image first, second;
    REQUIRE( offer( first ) );
    const auto firstSlot = ordinal++;
    y << .3f, -.2f;
    decide( y );
    REQUIRE( offer( second ) );
    REQUIRE( second != first );
    copy( firstSlot, first );
    REQUIRE( offer( second ) );
    copy( ordinal++, second );
    REQUIRE_FALSE( offer( image ) );

    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_control["requested_mode"].get<std::string>() == "mfilt_copy_2k" );
    CHECK( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
}

/** \brief A standard-path slot (here mfilt_copy_8k) is rendered at its cutoff minus dm.renderLeadUs, not when the
 * slot is selected, so a WFS decision completed while the slot waited reaches that slot rather than the next one.
 */
TEST_CASE( "Standard slots are rendered from the newest decision at the render lead", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::outputCandidate;
    MagAOX::app::loPredCtrlAcc_operational::renderWait;
#endif
    // clang-format on
    namespace synco = MagAOX::app::synco;
    Controller app( true );
    REQUIRE( selectMode( app, "mfilt_copy_8k" ) == 0 );
    app.qualifyOutput();
    app.mainLoop( .25, .9, .3, { 2, 1, 7 }, { 1, 1, 7 }, { 1, 2, 7 } );
    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );

    DDSPC::Matrix empty;
    const auto    decide = [&]( const DDSPC::Matrix &y )
    {
        DDSPC::Matrix applied;
        std::int64_t  ordinal;
        { //mutex scope
            std::lock_guard<std::mutex> lock( app.m_commandMutex );
            ordinal = app.appliedSlots( applied );
        }
        REQUIRE( app.step( y, app.m_wfsCounter + 1, empty, false, applied, ordinal, synco::monotonicNs() ) == 0 );
    };
    DDSPC::Matrix y( 2, 1 );
    y << .1f, -.2f;
    decide( y );

    // The slot is selected 5 ms before its cutoff; the next decision completes while it waits to render.
    const std::int64_t   cutoff = synco::monotonicNs() + 5000000;
    const synco::Request request{ 1, 1, 2, 0, cutoff + 70000, cutoff };
    synco::Record        schedule{};
    schedule[synco::steps] = 4;
    std::array<float, 4> image{};
    bool                 offered  = false;
    std::int64_t         rendered = 0;
    std::thread          worker(
        [&]
        {
            offered  = app.outputCandidate( request, schedule, image );
            rendered = synco::monotonicNs();
        } );
    y << .3f, .4f;
    decide( y );
    worker.join();

    REQUIRE( offered );
    CHECK( rendered >= cutoff - app.m_renderLeadNs );
    const DDSPC::Matrix modal = app.m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
    const std::array<double, 2> gain{ .5, .25 }, limit{ .3, .6 };
    for( int mode = 0; mode < 2; ++mode )
    {
        double expected = std::clamp( -gain[mode] * ( mode ? -.2 : .1 ), -limit[mode], limit[mode] );
        expected        = std::clamp( .9 * expected - gain[mode] * y( mode ), -limit[mode], limit[mode] );
        CHECK( modal( mode ) == Approx( expected ).margin( 1e-6 ) ); // The second decision, not the first.
    }
}

/** \brief A refused predict or learn_std request is answered at once with its unchanged state and INDI_ALERT, which
 * appLogic keeps until a request is accepted, so loPredCtrlAccGUI reverts the toggle and shows the refusal.
 */
TEST_CASE( "Refused prediction requests are answered with an alert", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStd;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on
    Controller app( true );
    auto       predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );

    // Output not qualified: learn_std is refused and keeps its Off state.
    auto learnStd = app.m_indiP_learnStd;
    learnStd["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_learnStd( learnStd ) < 0 );
    CHECK( app.m_indiP_learnStd["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_indiP_learnStd.getState() == pcf::IndiProperty::Alert );
    CHECK_FALSE( app.m_learning );

    // Nothing to apply yet: predict is refused, and the alert survives appLogic.
    app.qualifyOutput();
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) < 0 );
    CHECK_FALSE( app.m_predicting );
    CHECK( app.m_indiP_predicting["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_indiP_predicting.getState() == pcf::IndiProperty::Alert );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_predicting.getState() == pcf::IndiProperty::Alert );

    // An accepted request clears it.
    app.train();
    app.m_predicting = false;
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    CHECK( app.m_indiP_predicting["toggle"].getSwitchState() == pcf::IndiElement::On );
    CHECK( app.m_indiP_predicting.getState() == pcf::IndiProperty::Ok );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_predicting.getState() == pcf::IndiProperty::Ok );
}

/** \brief learn_std runs loPredCtrl's three stages and a fourth without probes, then stops learning while prediction
 * continues; learning turned on by hand is never stopped by an exploration sequence.
 */
TEST_CASE( "learn_std ends with a probe-free stage, then stops learning", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStd;
    MagAOX::app::loPredCtrlAcc_operational::learnStdFinished;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on
    Controller app( true );
    app.qualifyOutput();
    auto on = []( pcf::IndiProperty property )
    {
        property["toggle"].setSwitchState( pcf::IndiElement::On );
        return property;
    };
    REQUIRE( app.newCallBack_m_indiP_learnStd( on( app.m_indiP_learnStd ) ) == 0 );
    REQUIRE( app.m_learning );
    REQUIRE( app.m_predicting );

    // Four stages of 1000 frames; the last has no probes and keeps the final regularization.
    REQUIRE( app.m_exploration.size() == 4 );
    const std::array<float, 4> sigma{ .05f, .05f, .05f, 0 }, lambda{ 100, 3, .1f, .1f };
    for( int k = 0; k < 4; ++k )
    {
        CHECK( app.m_exploration[k].m_count == 1000 );
        CHECK( app.m_exploration[k].m_sigma == Approx( sigma[k] ) );
        CHECK( app.m_exploration[k].m_lambda == Approx( lambda[k] ) );
    }

    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), empty;
    auto          frame   = [&]( int n )
    {
        y << std::sin( .09 * n ), .8 * std::cos( .13 * n );
        y += .8f * command;
        command = response( app.advance( y, empty, false ) ) / .8f;
    };

    int n = 0;
    for( ; n < 3999; ++n )
    {
        frame( n );
    }
    REQUIRE( app.m_learning );
    REQUIRE( app.m_exploration.size() == 1 );

    // The last frame of the sequence stops learning at once; prediction continues with the frozen fit.
    frame( n++ );
    CHECK( app.m_exploration.empty() );
    CHECK_FALSE( app.m_learning );
    CHECK( app.m_predicting );
    const auto updates = app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates;
    for( int k = 0; k < 20; ++k, ++n )
    {
        frame( n );
    }
    CHECK( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == updates );
    CHECK( app.m_candidateValid.load() );

    // appLogic shows it: both switches off and a note in the log.
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_learning["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_indiP_learnStd["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_textLogs.back().first ==
           "DDSPC standard learning sequence finished: learning off, prediction continues" );

    // The assisted mode's sequence ends the same way.
    Controller assistedApp( true );
    Plant      plant( .02 );
    assisted( assistedApp, plant );
    assistedApp.qualifyOutput();
    REQUIRE( assistedApp.newCallBack_m_indiP_learnStd( on( assistedApp.m_indiP_learnStd ) ) == 0 );
    // Assisted stages count only frames with a valid ACC window, so the first frames while it fills are extra.
    for( int k = 0; k < 4100 && assistedApp.m_learning; ++k )
    {
        assistedApp.advanceAssisted( DDSPC::Matrix::Zero( 2, 1 ) );
    }
    CHECK( assistedApp.m_exploration.empty() );
    CHECK_FALSE( assistedApp.m_learning );
    CHECK( assistedApp.m_predicting );

    // Learning turned on by hand keeps learning after an exploration sequence ends.
    REQUIRE( app.newCallBack_m_indiP_learning( on( app.m_indiP_learning ) ) == 0 );
    app.m_exploration = app.parseExploration( "5,0.05,0.1" );
    for( int k = 0; k < 10; ++k, ++n )
    {
        frame( n );
    }
    CHECK( app.m_exploration.empty() );
    CHECK( app.m_learning );
}

/** \brief learn_std_extra (INDI, default 0) keeps learn_std learning for that many seconds after its sequence ends,
 * then stops learning while prediction continues; a change applies at once.
 */
TEST_CASE( "learn_std_extra keeps learning after the sequence", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStdExtra;
    MagAOX::app::loPredCtrlAcc_operational::learnStdEnded;
    MagAOX::app::loPredCtrlAcc_operational::step;
#endif
    // clang-format on
    Controller app( true );
    app.qualifyOutput();
    REQUIRE( app.m_learnStdExtra == 0 ); // The default: learning stops at the sequence's end.
    REQUIRE( app.m_indiP_learnStdExtra["current"].get<double>() == 0 );

    auto extra = [&]( double seconds )
    {
        auto property = app.m_indiP_learnStdExtra;
        property["target"].set( seconds );
        return app.newCallBack_m_indiP_learnStdExtra( property );
    };
    REQUIRE( extra( -1 ) < 0 );
    REQUIRE( extra( .05 ) == 0 );
    CHECK( app.m_learnStdExtra == Approx( .05 ) );
    CHECK( app.m_indiP_learnStdExtra["current"].get<double>() == Approx( .05 ) );

    DDSPC::Matrix y( 2, 1 ), empty;
    y << .1f, -.1f;
    auto sequence = [&]()
    {
        auto learnStd = app.m_indiP_learnStd;
        learnStd["toggle"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_learnStd( learnStd ) == 0 );
        app.m_exploration = app.parseExploration( "5,0.05,0.1" ); // A short sequence in place of the original.
        for( int n = 0; n < 5; ++n )
        {
            app.advance( y, empty, false );
        }
        REQUIRE( app.m_exploration.empty() );
    };

    // Learning continues after the sequence until learn_std_extra has passed.
    sequence();
    app.advance( y, empty, false );
    CHECK( app.m_learning );
    std::this_thread::sleep_for( std::chrono::milliseconds( 80 ) );
    app.advance( y, empty, false );
    CHECK_FALSE( app.m_learning );
    CHECK( app.m_predicting );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_learning["toggle"].getSwitchState() == pcf::IndiElement::Off );
    CHECK( app.m_indiP_learnStd["toggle"].getSwitchState() == pcf::IndiElement::Off );

    // A change applies at once, also while the extra learning runs.
    REQUIRE( extra( 600 ) == 0 );
    sequence();
    app.advance( y, empty, false );
    CHECK( app.m_learning );
    REQUIRE( extra( 0 ) == 0 );
    app.advance( y, empty, false );
    CHECK_FALSE( app.m_learning );
}

/** \brief Every prediction stop and every refused request is logged as a note with its reason. */
TEST_CASE( "Prediction stops and refusals are logged with their reason", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::stopPrediction;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStd;
#endif
    // clang-format on
    Controller app( true );
    auto       request = [&]( pcf::IndiProperty property, const char *element, bool on )
    {
        property[element].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
        return property;
    };
    auto last = [&]()
    {
        REQUIRE_FALSE( app.m_textLogs.empty() );
        CHECK( app.m_textLogs.back().second == MagAOX::app::logPrio::LOG_NOTICE );
        return app.m_textLogs.back().first;
    };

    // Refusals, with the condition that refused them.
    REQUIRE( app.newCallBack_m_indiP_learnStd( request( app.m_indiP_learnStd, "toggle", true ) ) < 0 );
    CHECK( last().find( "DDSPC standard learning sequence refused: the DDSPC output is not available" ) == 0 );
    app.qualifyOutput();
    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", true ) ) < 0 );
    CHECK( last() == "DDSPC prediction request refused: the selected model is not ready, and learning is off or has "
                     "neither gain nor exploration" );
    app.m_integrate = true;
    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", true ) ) < 0 );
    CHECK( last() == "DDSPC prediction request refused: the main loop's integrator parameters have not been received" );
    app.m_integrate = false;

    // Stops by request, with which request; turning off when already off logs nothing.
    app.train();
    app.m_predicting = false;
    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", true ) ) == 0 );
    const auto count = app.m_textLogs.size();
    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", false ) ) == 0 );
    CHECK( last() == "DDSPC prediction stopped: turned off by request" );
    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", false ) ) == 0 );
    CHECK( app.m_textLogs.size() == count + 1 );

    REQUIRE( app.newCallBack_m_indiP_predicting( request( app.m_indiP_predicting, "toggle", true ) ) == 0 );
    app.newCallBack_m_indiP_releaseTT( request( app.m_indiP_releaseTT, "request", true ) );
    CHECK( std::any_of( app.m_textLogs.begin() + count,
                        app.m_textLogs.end(),
                        []( const auto &entry )
                        {
                            return entry.first == "DDSPC prediction stopped: tip/tilt returned to the main loop";
                        } ) );
}

/** \brief While integrating, prediction is refused until the main loop's parameters are known, then accepted. */
TEST_CASE( "Integrator prediction waits for the main loop's parameters", "[rtc-controller][integrator]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_predicting;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learnStd;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
#endif
    // clang-format on
    Controller app( true );
    REQUIRE( app.m_defaultIntegrate ); // The integrator is on by default.
    app.qualifyOutput();
    app.train();
    app.m_predicting = false;
    app.m_integrate  = true;

    auto predict = app.m_indiP_predicting;
    predict["toggle"].setSwitchState( pcf::IndiElement::On );
    auto learnStd = app.m_indiP_learnStd;
    learnStd["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) < 0 );
    REQUIRE( app.newCallBack_m_indiP_learnStd( learnStd ) < 0 );
    CHECK_FALSE( app.m_predicting );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_integrator["ready"].get<double>() == 0 );
    CHECK( app.m_indiP_integrator.getState() == pcf::IndiProperty::Busy );

    app.mainLoop( .5, .99, .2, { 1, 2, 1 } );
    REQUIRE( app.newCallBack_m_indiP_predicting( predict ) == 0 );
    CHECK( app.m_predicting );
    REQUIRE( app.appLogic() == 0 );
    CHECK( app.m_indiP_integrator["ready"].get<double>() == 1 );
    CHECK( app.m_indiP_integrator["gain1"].get<double>() == Approx( 1 ) );
    CHECK( app.m_indiP_integrator["mult0"].get<double>() == Approx( .99 ) );
    CHECK( app.m_indiP_integrator["limit0"].get<double>() == Approx( .2 ) );
    CHECK( app.m_indiP_integrator.getState() == pcf::IndiProperty::Ok );
}

/** \brief The modal frameGrabber and its telemeter always start and stop with the application, as in loPredCtrl.
 */
TEST_CASE( "Modal output always starts with the application", "[rtc-controller][modal-output]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::appStartup;
    MagAOX::app::loPredCtrlAcc_operational::appLogic;
    MagAOX::app::loPredCtrlAcc_operational::appShutdown;
#endif
    // clang-format on
    Controller app( true );
    app.m_accelEnabled = false;
    REQUIRE( app.appStartup() == 0 );
    app.appShutdown();
    const auto &grabber = static_cast<ModalGrabber &>( app );
    const auto &tel     = static_cast<ModalTelemeter &>( app );
    REQUIRE( grabber.m_starts == 1 );
    REQUIRE( tel.m_starts == 1 );
    REQUIRE( grabber.m_stops == 1 );
}
} // namespace libXWCTest::loPredCtrlAccTest
