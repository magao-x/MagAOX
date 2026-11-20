/** \file rtc_controller_test.cpp
 * \brief Operational WFS DDSPC correction, independent ACC fitting, and original INDI workflow with private inputs.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc_operational.hpp"
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <filesystem>
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
    using loPredCtrlAcc_operational::allocate;
    using loPredCtrlAcc_operational::appliedSlots;
    using loPredCtrlAcc_operational::appShutdown;
    using loPredCtrlAcc_operational::appStartup;
    using loPredCtrlAcc_operational::c_slots;
    using loPredCtrlAcc_operational::copied;
    using loPredCtrlAcc_operational::loadCalibration;
    using loPredCtrlAcc_operational::m_accelEnabled;
    using loPredCtrlAcc_operational::m_calibration;
    using loPredCtrlAcc_operational::m_commandKnown;
    using loPredCtrlAcc_operational::m_commandMutex;
    using loPredCtrlAcc_operational::m_decisionOrdinal;
    using loPredCtrlAcc_operational::m_exploration;
    using loPredCtrlAcc_operational::m_gain;
    using loPredCtrlAcc_operational::m_indiP_exploration;
    using loPredCtrlAcc_operational::m_indiP_gain;
    using loPredCtrlAcc_operational::m_indiP_learning;
    using loPredCtrlAcc_operational::m_indiP_mode;
    using loPredCtrlAcc_operational::m_indiP_reset;
    using loPredCtrlAcc_operational::m_learning;
    using loPredCtrlAcc_operational::m_matchedTargets;
    using loPredCtrlAcc_operational::m_mode;
    using loPredCtrlAcc_operational::m_modeCube;
    using loPredCtrlAcc_operational::m_models;
    using loPredCtrlAcc_operational::m_outputThread;
    using loPredCtrlAcc_operational::m_planBase;
    using loPredCtrlAcc_operational::m_settings;
    using loPredCtrlAcc_operational::m_slotPlan;
    using loPredCtrlAcc_operational::m_slotShifts;
    using loPredCtrlAcc_operational::m_syncLocked;
    using loPredCtrlAcc_operational::m_wfsCounter;
    using loPredCtrlAcc_operational::outputCandidate;
    using loPredCtrlAcc_operational::parseExploration;
    using loPredCtrlAcc_operational::processImage;
    using loPredCtrlAcc_operational::resetModels;
    using loPredCtrlAcc_operational::sensorHistory;
    using loPredCtrlAcc_operational::step;

    /// Allocate private synthetic calibration and configure the actual existing producer state without starting it.
    explicit Controller( bool qr /**< [in] Select the real QR or covariance RLS implementation. */ );

    /// Next scheduler slot ordinal (cycle*4+step) rendered by advance().
    std::int64_t m_slot{ 8 };

    /// Advance one real decision, then render and copy the four following slots as the scheduler would.
    /** Returns the copied modal commands, modes by four, oldest slot first. */
    DDSPC::Matrix advance( const DDSPC::Matrix &y /**< [in] WFS residual. */,
                           const DDSPC::Matrix &a /**< [in] Individual ACC history. */,
                           bool                 valid /**< [in] Upstream ACC validity. */ );

    /// Render one scheduler slot through the actual candidate and report it copied.
    DDSPC::Matrix slot( std::int64_t ordinal /**< [in] Slot ordinal, cycle*4+step. */,
                        std::int64_t epoch = 1 /**< [in] Scheduler epoch. */ );

    /// Enable both fits and original-format Gaussian excitation on the actual WFS path.
    void train();

    /// Freeze both independent fits without stopping correction.
    void freeze();
};

Controller::Controller( bool qr )
{
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
    m_appliedSlots             = DDSPC::Matrix::Zero( 2, c_ring );
    m_commandKnown             = true;
    resetModels();
    setupINDI();
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = config.m_producerDefaults = true;
    m_output.configure( config, "private_aux", "private_timing", 50, 4 );
}

void Controller::train()
{
    m_learning = m_predicting = true;
    m_exploration             = parseExploration( "3000,0.15,0.1" );
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        m_models->training( mode, true );
}

void Controller::freeze()
{
    auto property = m_indiP_learning;
    property["toggle"].setSwitchState( pcf::IndiElement::Off );
    REQUIRE( newCallBack_m_indiP_learning( property ) == 0 );
}

DDSPC::Matrix Controller::slot( std::int64_t ordinal, std::int64_t epoch )
{
    namespace synco = MagAOX::app::synco;
    const synco::Request request{ epoch, 1, ordinal / 4, int( ordinal % 4 ), 0, 0 };
    synco::Record        schedule{};
    schedule[synco::steps] = 4;
    std::array<float, 4> image{};
    REQUIRE( outputCandidate( request, schedule, image ) );
    synco::CommitRecord commit{};
    commit[synco::cmdEpoch]      = epoch;
    commit[synco::cmdGeneration] = 1;
    commit[synco::cmdCycle]      = request.cycle;
    commit[synco::cmdStep]       = request.step;
    copied( commit, image );
    return m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
}

DDSPC::Matrix Controller::advance( const DDSPC::Matrix &y, const DDSPC::Matrix &a, bool valid )
{
    DDSPC::Matrix applied;
    std::int64_t  ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ordinal = appliedSlots( applied );
    }
    REQUIRE( step( y, m_wfsCounter + 1, a, valid, applied, ordinal, MagAOX::app::synco::monotonicNs() ) == 0 );
    DDSPC::Matrix slots( 2, c_slots );
    for( int j = 0; j < c_slots; ++j )
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
    REQUIRE( app.allocate( {} ) == 0 );
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
        REQUIRE( app.processImage( wfs.data() + 2 * ( 1 - md.cnt1 ), {} ) == 0 );
        REQUIRE( app.m_wfsCounter == n - 1 );
        REQUIRE( app.processImage( frame, {} ) == 0 );
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
    batch.m_count = 0;
    REQUIRE_FALSE( app.sensorHistory( batch, rows, features, 1500000 ) );
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
    REQUIRE( app.m_matchedTargets == 0 );
    REQUIRE( residual < .25 * uncontrolled );
}

/** \brief Both independent fits train on matched WFS/request records; assisted correction uses all eight raw features.
 */
TEST_CASE( "Operational assisted DDSPC corrects with upstream ACC features", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    DDSPC::ShadowController::training;
#endif
    // clang-format on
    const bool qr = GENERATE( false, true );
    Controller app( qr );
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.train();
    DDSPC::Matrix                   command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), a( 8, 1 );
    std::mt19937                    rng( 42 );
    std::normal_distribution<float> noise( 0, .03 );
    double                          residual = 0, uncontrolled = 0, spread = 0;
    for( int n = 0; n < 6000; ++n )
    {
        for( int sample = 0; sample < 4; ++sample )
        {
            a( 2 * sample )     = std::sin( .09 * ( n + 1 - sample * .25 ) );
            a( 2 * sample + 1 ) = std::cos( .13 * ( n + 1 - sample * .25 ) );
        }
        DDSPC::Matrix disturbance( 2, 1 );
        disturbance << std::sin( .09 * n ) + noise( rng ), .8 * std::cos( .13 * n ) + noise( rng );
        y = disturbance + .8f * command;
        if( n == 3000 )
            app.freeze();
        const auto slots = app.advance( y, a, true );
        REQUIRE( slots.allFinite() );
        command = response( slots ) / .8f;
        if( n >= 4000 )
        {
            residual += y.squaredNorm();
            uncontrolled += disturbance.squaredNorm();
            // Frozen, unprobed assisted plans still give each 8-kHz slot its own command.
            spread += ( slots.colwise() - slots.rowwise().mean() ).squaredNorm();
        }
    }
    INFO( "assisted residual ratio " << std::sqrt( residual / uncontrolled ) << " slot spread " << spread );
    REQUIRE( spread > 1e-3 * residual );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates > 2900 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_updates > 2900 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( app.m_matchedTargets > 5000 );
    REQUIRE( residual < .25 * uncontrolled );
    const auto before = app.m_models->prediction_matrix( DDSPC::ShadowMode::Standard );
    app.advance( y, {}, false );
    REQUIRE( ( before - app.m_models->prediction_matrix( DDSPC::ShadowMode::Standard ) ).norm() == 0 );
}

/** \brief Replay the supplied RTC dimensions, prior, gain and three calibration stages on both independent fits. */
TEST_CASE( "Operational RTC settings produce finite standard and assisted fits", "[rtc-controller]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::ShadowController::regularization;
    MagAOX::app::loPredCtrlAcc_operational::step;
#endif
    // clang-format on
    Controller app( true );
    app.m_settings.m_history       = 15;
    app.m_settings.m_future        = 3;
    app.m_settings.m_accelChannels = 40;
    app.m_settings.m_lambda        = 100;
    app.m_settings.m_prior         = 1e-5f;
    app.m_gain                     = .3f;
    app.resetModels();
    app.train();
    app.m_exploration     = app.parseExploration( "1000,0.1,100,1000,0.1,3,1000,0.1,0.1" );
    DDSPC::Matrix command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), a( 40, 1 );
    for( int n = 0; n < 3100; ++n )
    {
        for( int j = 0; j < 20; ++j )
        {
            a( 2 * j )     = std::sin( .09 * ( n + 1 - j * .25 ) );
            a( 2 * j + 1 ) = std::cos( .13 * ( n + 1 - j * .25 ) );
        }
        y << std::sin( .09 * n ), std::cos( .13 * n );
        y += .8f * command;
        command = response( app.advance( y, a, true ) ) / .8f;
    }
    REQUIRE( app.m_exploration.empty() );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( app.m_matchedTargets > 10000 );
}

/** \brief Time WFS decisions at the RTC dimensions while learning and frozen (hidden benchmark). */
TEST_CASE( "Operational RTC decision time", "[.][rtc-benchmark]" )
{
    Controller app( true );
    app.m_settings.m_history       = 15;
    app.m_settings.m_future        = 3;
    app.m_settings.m_accelChannels = 40;
    app.m_settings.m_lambda        = 100;
    app.m_settings.m_prior         = 1e-5f;
    app.m_gain                     = .3f;
    app.m_mode.store( DDSPC::ShadowMode::AccelAssisted );
    app.resetModels();
    app.train();
    app.m_exploration           = app.parseExploration( "2000,0.1,100" );
    DDSPC::Matrix       command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), a( 40, 1 );
    std::vector<double> learning, frozen;
    for( int n = 0; n < 4000; ++n )
    {
        for( int j = 0; j < 20; ++j )
        {
            a( 2 * j )     = std::sin( .09 * ( n + 1 - j * .25 ) );
            a( 2 * j + 1 ) = std::cos( .13 * ( n + 1 - j * .25 ) );
        }
        y << std::sin( .09 * n ), std::cos( .13 * n );
        y += .8f * command;
        if( n == 3000 )
            app.freeze();
        const auto start = std::chrono::steady_clock::now();
        const auto slots = app.advance( y, a, true );
        const auto us = std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count();
        command       = response( slots ) / .8f;
        if( n >= 500 )
            ( n < 3000 ? learning : frozen ).push_back( us );
    }
    for( auto *times : { &learning, &frozen } )
    {
        std::sort( times->begin(), times->end() );
        std::printf( "%s decision+4 slots: mean %.1f us p99 %.1f us max %.1f us\n",
                     times == &learning ? "learning" : "frozen  ",
                     std::accumulate( times->begin(), times->end(), 0.0 ) / times->size(),
                     ( *times )[times->size() * 99 / 100],
                     times->back() );
    }
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
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
} // namespace libXWCTest::loPredCtrlAccTest
