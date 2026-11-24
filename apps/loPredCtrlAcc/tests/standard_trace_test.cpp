/** \file standard_trace_test.cpp
 * \brief Record standard-mode DM output and decision timing for the T1 bit-for-bit equivalence check.
 * \ingroup loPredCtrlAcc_unit_test
 *
 * The same file is compiled against the baseline and the modified application by tests/standard_equivalence.py.
 * It uses only interfaces that exist in both versions.
 */
#include "../../../tests/testXWC.hpp"
#include "app_test_support.hpp"
#include "../loPredCtrlAcc_operational.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond STANDARD_TRACE_SUPPORT
namespace
{
/// Real application with a private four-pixel DM; every rendered slot is appended to an owned trace.
class TraceController : public MagAOX::app::loPredCtrlAcc_operational
{
  public:
    using loPredCtrlAcc_operational::m_indiP_exploration;
    using loPredCtrlAcc_operational::m_indiP_gain;
    using loPredCtrlAcc_operational::m_indiP_learning;
    using loPredCtrlAcc_operational::m_models;
    using loPredCtrlAcc_operational::m_predicting;
    using loPredCtrlAcc_operational::m_syncLocked;
    using loPredCtrlAcc_operational::m_wfsCounter;

    /// Configure the RTC or reduced dimensions, then allocate the models and controls.
    TraceController( bool qr /**< [in] QR-RLS or covariance RLS. */, bool rtc /**< [in] RTC dimensions. */ );

    /// Rendered pixel values of every slot, in order.
    std::vector<float> m_trace;

    /// Measured step() durations [us].
    std::vector<double> m_decisionUs;

    /// Turn off the integrator and missed-frame filling where they exist, so both versions behave as the original.
    /** Both are tested separately; T1 checks that nothing else changed. */
    template <class T>
    static void absolute( T &self /**< [in,out] Application under test. */ )
    {
        if constexpr( requires { self.m_integrate; } )
        {
            self.m_integrate = false;
        }

        if constexpr( requires { self.m_maxFill; } )
        {
            self.m_maxFill = 0;
        }
    }

    /// Next slot ordinal.
    std::int64_t m_slot{ 8 };

    /// Process one WFS frame and render the next four slots; slots listed in skip are not copied.
    DDSPC::Matrix advance( const DDSPC::Matrix &y /**< [in] WFS residual. */,
                           const DDSPC::Matrix &a /**< [in] ACC history. */,
                           bool                 valid /**< [in] ACC validity. */,
                           std::uint64_t        counter /**< [in] WFS counter. */,
                           int                  skip = -1 /**< [in] Slot index not copied, or -1. */ );
};

TraceController::TraceController( bool qr, bool rtc )
{
    absolute( *this );
    m_settings.m_history       = rtc ? 15 : 3;
    m_settings.m_future        = rtc ? 3 : 2;
    m_settings.m_accelChannels = rtc ? 40 : 8;
    m_settings.m_prior         = rtc ? 1e-5f : .001f;
    m_settings.m_lambda        = rtc ? 100 : .1f;
    m_settings.m_useQrd        = qr;
    m_gain                     = rtc ? .3f : .2f;
    m_calibration.m_shapes     = DDSPC::Matrix::Zero( 4, 2 );
    m_calibration.m_shapes.topRows( 2 ).setIdentity();
    m_calibration.m_shapes( 2, 0 ) = .5f;
    m_calibration.m_shapes( 3, 1 ) = -.25f;
    m_calibration.m_projection     = m_calibration.m_shapes.completeOrthogonalDecomposition().pseudoInverse();
    m_copiedCommand                = DDSPC::Matrix::Zero( 2, 1 );
    m_appliedSlots                 = DDSPC::Matrix::Zero( 2, 16 ); // The 16-slot ring of both versions.
    m_commandKnown                 = true;
    resetModels();
    setupINDI();
    MagAOX::app::SyncoOutputConfig config;
    config.m_authorized = config.m_routingVerified = config.m_producerDefaults = true;
    m_output.configure( config, "private_aux", "private_timing", 50, 4 );
}

DDSPC::Matrix
TraceController::advance( const DDSPC::Matrix &y, const DDSPC::Matrix &a, bool valid, std::uint64_t counter, int skip )
{
    namespace synco = MagAOX::app::synco;
    DDSPC::Matrix applied;
    std::int64_t  ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ordinal = appliedSlots( applied );
    }
    const auto start = std::chrono::steady_clock::now();
    REQUIRE( step( y, counter, a, valid, applied, ordinal, 1000 * std::int64_t( counter ) ) == 0 );
    m_decisionUs.push_back(
        std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count() );
    DDSPC::Matrix slots( 2, 4 );
    for( int j = 0; j < 4; ++j, ++m_slot )
    {
        const synco::Request request{ 1, 1, m_slot / 4, int( m_slot % 4 ), 0, 0 };
        synco::Record        schedule{};
        schedule[synco::steps] = 4;
        std::array<float, 4> image{};
        REQUIRE( outputCandidate( request, schedule, image ) );
        m_trace.insert( m_trace.end(), image.begin(), image.end() );
        if( j != skip )
        {
            synco::CommitRecord commit{};
            commit[synco::cmdEpoch]      = 1;
            commit[synco::cmdGeneration] = 1;
            commit[synco::cmdCycle]      = request.cycle;
            commit[synco::cmdStep]       = request.step;
            copied( commit, image );
        }
        slots.col( j ) = m_calibration.m_projection * Eigen::Map<const DDSPC::Matrix>( image.data(), 4, 1 );
    }
    return slots;
}

/// Set a switch through the real INDI callback.
template <class callbackT>
void toggle( pcf::IndiProperty property /**< [in] Registered property copy. */,
             bool              on /**< [in] Requested state. */,
             callbackT         callback /**< [in] Application callback. */ )
{
    property["toggle"].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
    REQUIRE( callback( property ) == 0 );
}

/// Append the summary statistics of a duration series to a text report.
void statistics( std::ostream        &out /**< [in,out] Report. */,
                 const char          *name /**< [in] Series name. */,
                 std::vector<double> &times /**< [in,out] Durations; sorted here. */ )
{
    std::sort( times.begin(), times.end() );
    out << name << ' ' << std::accumulate( times.begin(), times.end(), 0.0 ) / times.size() << ' '
        << times[times.size() / 2] << ' ' << times[times.size() * 99 / 100] << ' ' << times.back() << '\n';
}

/// Run one deterministic closed-loop standard-mode scenario and append its trace.
void scenario( std::vector<float> &trace /**< [in,out] Combined trace. */,
               std::ostream       &timing /**< [in,out] Timing report. */,
               const char         *name /**< [in] Scenario name. */,
               bool                qr /**< [in] QR-RLS. */,
               bool                rtc /**< [in] RTC dimensions. */,
               bool                accel /**< [in] Supply valid ACC features. */,
               bool                edges /**< [in] Missed slots, a WFS gap and a gain change. */ )
{
    TraceController app( qr, rtc );
    app.m_syncLocked = accel;
    // The real exploration and learning callbacks; prediction is enabled directly because no transport runs here.
    auto sequence = app.m_indiP_exploration;
    sequence["target"].set( edges ? "800,0.1,100,800,0.05,3" : "1000,0.1,100,1000,0.1,3,1000,0.1,0.1" );
    REQUIRE( app.newCallBack_m_indiP_exploration( sequence ) == 0 );
    toggle( app.m_indiP_learning,
            true,
            [&]( const pcf::IndiProperty &p ) { return app.newCallBack_m_indiP_learning( p ); } );
    app.m_predicting                         = true;
    const int                       channels = rtc ? 40 : 8, frames = 4000, freeze = 3200;
    DDSPC::Matrix                   command = DDSPC::Matrix::Zero( 2, 1 ), y( 2, 1 ), a( channels, 1 );
    std::mt19937                    rng( 7 );
    std::normal_distribution<float> noise( 0, .02f );
    std::uint64_t                   counter = 0;
    std::vector<double>             learning, frozen;
    for( int n = 0; n < frames; ++n )
    {
        for( int j = 0; j < channels / 2; ++j )
        {
            a( 2 * j )     = std::sin( .09 * ( n + 1 - j * .25 ) ) + noise( rng );
            a( 2 * j + 1 ) = std::cos( .13 * ( n + 1 - j * .25 ) ) + noise( rng );
        }
        y << std::sin( .09 * n ) + .4 * std::cos( .21 * n ) + noise( rng ),
            .8 * std::cos( .13 * n ) + .3 * std::sin( .055 * n ) + noise( rng );
        y += .8f * command;
        counter += edges && n == 1500 ? 2 : 1;
        if( n == freeze )
            toggle( app.m_indiP_learning,
                    false,
                    [&]( const pcf::IndiProperty &p ) { return app.newCallBack_m_indiP_learning( p ); } );
        if( edges && n == 2000 )
        {
            auto gain = app.m_indiP_gain;
            gain["target"].set( .1 );
            REQUIRE( app.newCallBack_m_indiP_gain( gain ) == 0 );
        }
        const auto slots = app.advance( y, a, accel, counter, edges && n % 37 == 5 ? n % 4 : -1 );
        command =
            ( .35f * slots.col( 0 ) + .25f * slots.col( 1 ) + .15f * slots.col( 2 ) + .05f * slots.col( 3 ) ) / .8f;
        if( n >= 500 )
            ( n < freeze ? learning : frozen ).push_back( app.m_decisionUs.back() );
    }
    REQUIRE( app.m_models->status( DDSPC::ShadowMode::Standard ).m_ready );
    trace.insert( trace.end(), app.m_trace.begin(), app.m_trace.end() );
    statistics( timing, ( std::string( name ) + "_learning" ).c_str(), learning );
    statistics( timing, ( std::string( name ) + "_frozen" ).c_str(), frozen );
}
} // namespace
/// \endcond

/** \brief Record standard-mode output for T1 (hidden; driven by tests/standard_equivalence.py).
 *
 * Writes the bit patterns of every rendered slot to $DDSPC_STANDARD_TRACE and decision timing statistics to
 * $DDSPC_STANDARD_TRACE.timing. Inputs, initial conditions and exploration seeds are fixed.
 */
TEST_CASE( "Standard-mode DM output trace", "[.][standard-trace]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc_operational::step;
    MagAOX::app::loPredCtrlAcc_operational::outputCandidate;
    MagAOX::app::loPredCtrlAcc_operational::newCallBack_m_indiP_learning;
#endif
    // clang-format on
    std::vector<float> trace;
    std::ostringstream timing;
    scenario( trace, timing, "rtc_acc", true, true, true, false );
    scenario( trace, timing, "rtc_edges", true, true, false, true );
    scenario( trace, timing, "small_rls", false, false, true, false );
    const char *path = std::getenv( "DDSPC_STANDARD_TRACE" );
    REQUIRE( path );
    std::ofstream( path, std::ios::binary )
        .write( reinterpret_cast<const char *>( trace.data() ), trace.size() * sizeof( float ) );
    std::ofstream( std::string( path ) + ".timing" ) << timing.str();
}
} // namespace libXWCTest::loPredCtrlAccTest
