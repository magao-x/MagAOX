/** \file rtc_controller_test.cpp
 * \brief Operational WFS DDSPC correction, independent ACC fitting, and original INDI workflow with private inputs.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc_operational.hpp"
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
    using loPredCtrlAcc_operational::copied;
    using loPredCtrlAcc_operational::loadCalibration;
    using loPredCtrlAcc_operational::m_calibration;
    using loPredCtrlAcc_operational::m_commandKnown;
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
    using loPredCtrlAcc_operational::m_settings;
    using loPredCtrlAcc_operational::m_syncLocked;
    using loPredCtrlAcc_operational::m_wfsCounter;
    using loPredCtrlAcc_operational::outputCandidate;
    using loPredCtrlAcc_operational::parseExploration;
    using loPredCtrlAcc_operational::processImage;
    using loPredCtrlAcc_operational::resetModels;
    using loPredCtrlAcc_operational::sensorHistory;

    /// Allocate private synthetic calibration and configure the actual existing producer state without starting it.
    explicit Controller( bool qr /**< [in] Select the real QR or covariance RLS implementation. */ );

    /// Advance one real model decision and explicitly acknowledge its private output for this synthetic plant.
    DDSPC::Matrix advance( const DDSPC::Matrix &y /**< [in] WFS residual. */,
                           const DDSPC::Matrix &a /**< [in] Individual ACC history. */,
                           bool                 valid /**< [in] Upstream ACC validity. */ );

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
    m_pixels.assign( 4, 0 );
    m_copiedCommand = DDSPC::Matrix::Zero( 2, 1 );
    m_commandKnown  = true;
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

DDSPC::Matrix Controller::advance( const DDSPC::Matrix &y, const DDSPC::Matrix &a, bool valid )
{
    const auto counter = m_wfsCounter + 1;
    REQUIRE( step( y, counter, a, valid, m_copiedCommand, MagAOX::app::synco::monotonicNs() ) == 0 );
    std::array<float, 4> image{};
    REQUIRE( outputCandidate( image ) );
    copied( image );
    return m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
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
        REQUIRE( app.outputCandidate( surface ) );
        if( n < 30 )
        {
            REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates == 0 );
            REQUIRE( surface[0] == Approx( .2f * frame[0] ) );
        }
        if( n >= 30 )
            app.copied( surface );
    }
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_updates > 100 );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_updates == 0 );
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
        command = app.advance( y, empty, false );
        REQUIRE( command.allFinite() );
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
    double                          residual = 0, uncontrolled = 0;
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
        command = app.advance( y, a, true );
        REQUIRE( command.allFinite() );
        if( n >= 4000 )
        {
            residual += y.squaredNorm();
            uncontrolled += disturbance.squaredNorm();
        }
    }
    INFO( "assisted residual ratio " << std::sqrt( residual / uncontrolled ) );
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
        command = app.advance( y, a, true );
    }
    REQUIRE( app.m_exploration.empty() );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::AccelAssisted ).m_ready );
    REQUIRE( app.m_matchedTargets > 10000 );
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
