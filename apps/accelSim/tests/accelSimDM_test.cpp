/** \file accelSimDM_test.cpp
 * \brief Calibrated physical DM synthesis and simple operator-control regressions.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimDM.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <unistd.h>

#ifndef ACCELSIM_MODEL_ONLY
    #ifdef ACCELSIM_APP_TEST
        #include "accelSimTestSupport.hpp"
    #endif
    #include "../accelSim.hpp"
#endif

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app::accelSimDetail;

/// \cond ACCELSIM_DM_FIXTURE
/// Synthetic rotated TT calibration with circular fitting support.
struct dmFixture
{
    std::vector<double> m_x, m_y, m_weights; ///< Native-order surface modes and pupil weights.

    /// Build finite column/row ramps without any stream or device access.
    dmFixture();
};

dmFixture::dmFixture() : m_x( 2500 ), m_y( 2500 ), m_weights( 2500 )
{
    for( size_t n = 0; n < 2500; ++n )
    {
        const double u = double( n % 50 ) - 24.5, v = double( n / 50 ) - 24.5;
        m_x[n]       = ( -.9 * u + .4 * v ) / 12;
        m_y[n]       = ( -.4 * u - .9 * v ) / 12;
        m_weights[n] = std::hypot( u, v ) <= 24;
    }
}
/// \endcond

/** \brief Amplitude scaling preserves the calibrated physical surface, including zero and large finite commands.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM physical mapping and amplitude scale", "[accelSim][dm-runtime]" )
{
    dmFixture   calibration;
    dmInjection dm;
    dm.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
    for( const auto angles : { std::array<double, 2>{ .001, 0 }, { 0, .001 }, { -.0029, .0029 }, { .5, -.8 } } )
    {
        sample value;
        value.m_pointing    = angles;
        value.m_wfsPointing = { NAN, NAN };
        const pointingDisturbance pointing( value );
        for( double scale : { 0., .125, .5, 1., 2.5, 100. } )
        {
            const auto &command = dm.synthesize( value, scale );
            for( size_t n = 0; n < command.size(); ++n )
            {
                const double desired =
                    pointing.surface( ( double( n % 50 ) - 24.5 ) * 6.5 / 48, ( double( n / 50 ) - 24.5 ) * 6.5 / 48 );
                REQUIRE( command[n] == Approx( desired * scale ).margin( 1e-8 ) );
            }
        }
    }
}

/** \brief Calibration and finite-command validation reject invalid input and recover on the next valid sample.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM numerical validation has no fault latch", "[accelSim][dm-runtime]" )
{
    dmFixture   calibration;
    dmInjection dm;
    sample      value;
    REQUIRE_FALSE( dm.ready() );
    REQUIRE_THROWS( dm.synthesize( value, 1 ) );
    REQUIRE_THROWS( dm.prepare( {}, {}, {}, 1e-6 ) );
    dm.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
    REQUIRE( dm.ready() );
    for( double bad : { double( NAN ), double( INFINITY ), std::numeric_limits<double>::max() } )
    {
        value.m_pointing = { bad, 0 };
        REQUIRE_THROWS( dm.synthesize( value, 1 ) );
        value.m_pointing = { .0029, 0 };
        REQUIRE_NOTHROW( dm.synthesize( value, .25 ) );
    }
    for( double good : { 1.1, 50., 100. } )
        REQUIRE_NOTHROW( dm.synthesize( value, good ) );
    for( double bad : { -1., 100.001, double( NAN ), double( INFINITY ) } )
        REQUIRE_THROWS( dm.synthesize( value, bad ) );
    calibration.m_x[0] = NAN;
    REQUIRE_THROWS( dm.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 ) );
    REQUIRE_FALSE( dm.ready() );
}

/** \brief Synthesis observes 8000 identical physical samples without changing model timing, ADC values or RNG.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM same sample and timing budget", "[accelSim][dm-runtime]" )
{
    dmFixture   calibration;
    dmInjection dm;
    dm.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
    parameters config;
    config.m_fps = 8000;
    model baseline, injected;
    baseline.configure( config, referenceModes( 123 ) );
    injected.configure( config, referenceModes( 123 ) );
    double worst = 0, total = 0;
    for( size_t n = 0; n < 8000; ++n )
    {
        const sample expected = baseline.step(), value = injected.step();
        const auto   start   = std::chrono::steady_clock::now();
        const auto  &command = dm.synthesize( value, .1 );
        const double us = std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count();
        total += us;
        worst = std::max( worst, us );
        REQUIRE( command.size() == 2500 );
        REQUIRE( injected.m_ticks == baseline.m_ticks );
        REQUIRE( value.m_adc == expected.m_adc );
        REQUIRE( value.m_pointing == expected.m_pointing );
        REQUIRE( value.m_acceleration == expected.m_acceleration );
    }
    std::cout << "DM synthesis: mean " << total / 8000 << " us, max " << worst
              << " us (not a native stream/driver benchmark)\n";
}

/// \cond ACCELSIM_DM_FIXTURE
/// Modes nonzero on the whole grid: the fixture's rotated tilt inside the 48-pitch pupil, and outside it slaved-edge
/// and corner values that are not on the tilt plane, as in a deployed DM mode cube.
dmFixture deployedLike()
{
    dmFixture modes;
    for( size_t n = 0; n < 2500; ++n )
    {
        const double u = double( n % 50 ) - 24.5, v = double( n / 50 ) - 24.5, r = std::hypot( u, v );
        if( r > 24 ) // Held at the edge value, then a small constant: nonzero but not tilt.
        {
            modes.m_x[n] = r <= 26 ? modes.m_x[n] * 24 / r : .013;
            modes.m_y[n] = r <= 26 ? modes.m_y[n] * 24 / r : -.007;
        }
    }
    return modes;
}

/// Native-order pupil coordinate of a pixel [m], as used by dmInjection::prepare().
double pupil( size_t n /**< [in] pixel */, bool row /**< [in] Y coordinate */ )
{
    return ( double( row ? n / 50 : n % 50 ) - 24.5 ) * 6.5 / 48;
}

/// Require every pixel of a command to be the full modes times K, with no masking, zeroing or renormalization.
void requireModes( const std::vector<float> &command /**< [in] synthesized command */,
                   const dmFixture          &modes /**< [in] mode planes */,
                   const Eigen::Matrix2d    &k /**< [in] K [coefficient/rad] */,
                   const sample             &value /**< [in] pointing sample */,
                   double                    scale /**< [in] amplitude scale */ )
{
    const pointingDisturbance pointing( value );
    const double              x = pointing.radians()[0], y = pointing.radians()[1];
    for( size_t n = 0; n < 2500; ++n )
    {
        const double expected = x * ( modes.m_x[n] * k( 0, 0 ) + modes.m_y[n] * k( 1, 0 ) ) +
                                y * ( modes.m_x[n] * k( 0, 1 ) + modes.m_y[n] * k( 1, 1 ) );
        REQUIRE( command[n] == Approx( expected * scale ).margin( 1e-9 ) );
        REQUIRE( command[n] != 0 );
    }
}
/// \endcond

/** \brief Deployed-like modes nonzero across the whole grid: the unmasked physical fit is rejected with its residuals,
 * because the modes are not a tilt plane over the whole grid, while explicit coefficients initialize and generate the
 * full-grid commands of the masked calibration bit for bit (same orientation, units and amplitude).
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM without a mask on full-grid modes", "[accelSim][dm-runtime]" )
{
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSimDetail::dmInjection::prepare;
    MagAOX::app::accelSimDetail::dmInjection::prepareCoefficients;
#endif
    // clang-format on
    const auto  modes = deployedLike();
    dmInjection uniform, masked, coefficients;
    REQUIRE_THROWS_WITH( uniform.prepare( modes.m_x, modes.m_y, {}, 1e-6 ),
                         Catch::Matchers::Contains( "within tolerance" ) &&
                             Catch::Matchers::Contains( "relative residuals X" ) );
    REQUIRE_FALSE( uniform.ready() );
    REQUIRE( uniform.method() == dmCalibrationMethod::none );
    masked.prepare( modes.m_x, modes.m_y, modes.m_weights, 1e-6 );
    REQUIRE( masked.method() == dmCalibrationMethod::fit );
    REQUIRE( masked.fitPixels() == size_t( std::count( modes.m_weights.begin(), modes.m_weights.end(), 1. ) ) );
    coefficients.prepareCoefficients( modes.m_x, modes.m_y, masked.mapping() );
    REQUIRE( coefficients.ready() );
    REQUIRE( coefficients.method() == dmCalibrationMethod::coefficients );
    REQUIRE( coefficients.fitPixels() == 0 );
    REQUIRE( std::isnan( coefficients.residual()[0] ) );
    for( const auto angles : { std::array<double, 2>{ .001, 0 }, { 0, .001 }, { -.0029, .0017 } } )
        for( double scale : { .25, 1., 37.5 } )
        {
            sample value;
            value.m_pointing   = angles;
            const auto command = coefficients.synthesize( value, scale );
            REQUIRE( command == masked.synthesize( value, scale ) );
            requireModes( command, modes, masked.mapping(), value, scale );
            const pointingDisturbance pointing( value );
            for( size_t n = 0; n < 2500; ++n ) // The physical tilt surface inside the pupil.
                if( modes.m_weights[n] )
                    REQUIRE( command[n] ==
                             Approx( pointing.surface( pupil( n, false ), pupil( n, true ) ) * scale ).margin( 1e-8 ) );
        }
}

/** \brief Without a mask the fit uses uniform weights 1.0 on all 2500 pixels; a cube that is the physical tilt over the
 * whole grid fits and generates the physical surface everywhere.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM uniform fitting weights", "[accelSim][dm-runtime]" )
{
    const dmFixture modes;
    dmInjection     dm;
    dm.prepare( modes.m_x, modes.m_y, {}, 1e-6 );
    REQUIRE( dm.fitPixels() == 2500 );
    REQUIRE( dm.residual()[0] < 1e-9 );
    REQUIRE( dm.residual()[1] < 1e-9 );
    sample value;
    value.m_pointing = { .0011, -.0023 };
    const pointingDisturbance pointing( value );
    const auto               &command = dm.synthesize( value, 1 );
    for( size_t n = 0; n < 2500; ++n )
        REQUIRE( command[n] == Approx( pointing.surface( pupil( n, false ), pupil( n, true ) ) ).margin( 1e-8 ) );
    dmInjection explicitOnes;
    explicitOnes.prepare( modes.m_x, modes.m_y, std::vector<double>( 2500, 1 ), 1e-6 );
    REQUIRE( explicitOnes.synthesize( value, 1 ) == dm.synthesize( value, 1 ) );
}

/** \brief An explicit mask is used exactly as before: the projection's weights and responses are unchanged, and it
 * still selects the fitting support when part of the plane is not tilt.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM explicit mask behavior is unchanged", "[accelSim][dm-runtime]" )
{
    const auto          modes = deployedLike();
    std::vector<double> x( 2500 ), y( 2500 );
    for( size_t n = 0; n < 2500; ++n )
    {
        x[n] = pupil( n, false );
        y[n] = pupil( n, true );
    }
    const tipTiltProjection projection( modes.m_x, modes.m_y, x, y, modes.m_weights, 1e-6 );
    dmInjection             dm;
    dm.prepare( modes.m_x, modes.m_y, modes.m_weights, 1e-6 );
    REQUIRE( dm.mapping() == projection.mapping() );
    REQUIRE( dm.residual() == projection.relativeResidual() );
    sample value;
    value.m_pointing = { -.0029, .0017 };
    const pointingDisturbance pointing( value );
    const auto               &command = dm.synthesize( value, 1 );
    for( size_t n = 0; n < 2500; ++n )
        REQUIRE( command[n] == static_cast<float>( pointing.radians()[0] * projection.response()[0][n] +
                                                   pointing.radians()[1] * projection.response()[1][n] ) );
    std::vector<double> left( 2500, 0 );
    for( size_t n = 0; n < 2500; ++n )
        left[n] = modes.m_weights[n] * ( double( n % 50 ) < 24.5 );
    dm.prepare( modes.m_x, modes.m_y, left, 1e-6 );
    REQUIRE( dm.fitPixels() == size_t( std::count( left.begin(), left.end(), 1. ) ) );
}

/** \brief Explicit coefficients still require finite, rank-two modes and K; a failure leaves DM output unprepared.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM explicit coefficient validation", "[accelSim][dm-runtime]" )
{
    const auto      modes = deployedLike();
    Eigen::Matrix2d k;
    k << -798911.751168, -315942.594932, 318826.690493, -792420.649088;
    dmInjection dm;
    dm.prepareCoefficients( modes.m_x, modes.m_y, k );
    REQUIRE( dm.ready() );
    Eigen::Matrix2d singular = k;
    singular.row( 1 )        = 2 * k.row( 0 );
    REQUIRE_THROWS( dm.prepareCoefficients( modes.m_x, modes.m_y, singular ) );
    REQUIRE_FALSE( dm.ready() );
    REQUIRE_THROWS( dm.prepareCoefficients( modes.m_x, modes.m_y, Eigen::Matrix2d::Constant( NAN ) ) );
    std::vector<double> dependent( modes.m_x.size() );
    for( size_t n = 0; n < dependent.size(); ++n )
        dependent[n] = 2 * modes.m_x[n];
    REQUIRE_THROWS( dm.prepareCoefficients( modes.m_x, dependent, k ) );
    REQUIRE_THROWS( dm.prepareCoefficients( std::vector<double>( 10 ), std::vector<double>( 10 ), k ) );
    auto bad   = modes;
    bad.m_x[7] = NAN;
    REQUIRE_THROWS( dm.prepareCoefficients( bad.m_x, bad.m_y, k ) );
    REQUIRE( dm.method() == dmCalibrationMethod::none );
}

#ifndef ACCELSIM_MODEL_ONLY
/// \cond ACCELSIM_DM_APP_FIXTURE
/// Expose actual app callbacks without starting framework threads.
struct dmAppTest : MagAOX::app::accelSim
{
    using accelSim::calibrationUnchanged;
    using accelSim::config;
    using accelSim::loadDMCalibration;
    using accelSim::m_configName;
    using accelSim::m_dmEnabled;
    using accelSim::m_dmInjection;
    using accelSim::m_dmMask;
    using accelSim::m_dmModes;
    using accelSim::m_dmReady;
    using accelSim::m_dmScale;
    using accelSim::m_dmZeroPending;
    using accelSim::m_indiP_dmCalibration;
    using accelSim::m_indiP_dmEnable;
    using accelSim::m_indiP_dmScale;
    using accelSim::m_indiP_dmStatus;
    using accelSim::prepareDMCalibration;
    using accelSim::setupProperties;
    using accelSim::updateDMINDI;
};
/// \endcond

/** \brief Operators set scale and toggle output without rate, WFS-lock, freshness or authorization gates.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM simple operator controls", "[accelSim][dm-controls]" )
{
    dmAppTest app;
    app.m_configName = "accelSim-dm-test";
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.setupProperties() == 0 );
    auto enable      = app.m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) < 0 );
    dmFixture calibration;
    app.m_dmInjection.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
    app.m_dmReady   = true;
    auto scale      = app.m_indiP_dmScale;
    scale["target"] = .25;
    REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) == 0 );
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    REQUIRE( app.m_dmEnabled );
    app.updateDMINDI();
    REQUIRE( app.m_indiP_dmStatus["ready"].get<int>() == 1 );
    REQUIRE( app.m_indiP_dmStatus["ready"].getValue() == "1" );
    REQUIRE( app.m_indiP_dmStatus["fault"].get<int>() == 0 );
    REQUIRE( app.m_indiP_dmStatus["attached"].get<int>() == 0 ); // Controls alone open no shared memory.
    for( double good : { 1.5, 100. } )
    {
        scale["target"] = good;
        REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) == 0 );
        REQUIRE( app.m_dmScale == good );
    }
    scale["target"] = .25;
    REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) == 0 );
    for( double bad : { -1., 100.001, double( NAN ), double( INFINITY ) } )
    {
        scale["target"] = bad;
        REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) < 0 );
        REQUIRE( app.m_dmScale == .25 );
    }
    REQUIRE( app.m_dmEnabled );
    enable["toggle"] = pcf::IndiElement::Off;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) == 0 );
    REQUIRE_FALSE( app.m_dmEnabled );
    REQUIRE( app.m_dmZeroPending ); // Removal is independent of a subsequent ADC sample.
    for( const auto &[text, valid] :
         { std::pair{ "100", true }, { "37.5", true }, { "100.001", false }, { "-1", false } } )
    {
        dmAppTest configured; // Startup dm.scale uses the same [0,100] range.
        configured.setupConfig();
        configured.config.m_values["dm.scale"] = text;
        REQUIRE( ( configured.loadConfigImpl( configured.config ) == 0 ) == valid );
        if( valid )
            REQUIRE( configured.m_dmScale == std::stod( text ) );
    }
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::setupProperties;
    MagAOX::app::accelSim::updateDMINDI;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmEnable;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmScale;
#endif
    // clang-format on
}
#endif

#ifndef ACCELSIM_MODEL_ONLY
/** \brief A full-grid mode cube without dm.mask: the default fit reports exactly why DM output is unavailable, and
 * dm.calibration=coefficients initializes from the cube alone, generating the masked calibration's commands.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM initialization from the mode cube alone", "[accelSim][dm-controls]" )
{
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadDMCalibration;
    MagAOX::app::accelSim::prepareDMCalibration;
    MagAOX::app::accelSim::calibrationUnchanged;
    MagAOX::app::accelSim::updateDMINDI;
#endif
    // clang-format on
    const auto directory = std::filesystem::temp_directory_path() / ( "accelSim-dm-" + std::to_string( getpid() ) );
    std::filesystem::create_directories( directory );
    const auto                    modes = deployedLike();
    mx::improc::eigenCube<double> cube( 50, 50, 2 ), mask( 50, 50, 1 );
    for( size_t n = 0; n < 2500; ++n )
    {
        cube.image( 0 ).data()[n] = modes.m_x[n];
        cube.image( 1 ).data()[n] = modes.m_y[n];
        mask.image( 0 ).data()[n] = modes.m_weights[n];
    }
    mx::fits::fitsFile<double> file;
    const auto cubePath = ( directory / "modes.fits" ).string(), maskPath = ( directory / "mask.fits" ).string();
    REQUIRE( file.write( cubePath, cube ) == mx::error_t::noerror );
    REQUIRE( file.write( maskPath, mask ) == mx::error_t::noerror );
    sample value;
    value.m_pointing = { .0011, -.0023 };
    auto make        = [&]( std::map<std::string, std::string> values )
    {
        auto app          = std::make_unique<dmAppTest>();
        app->m_configName = "accelSim-dm-test";
        app->setupConfig();
        values["dm.modeCube"] = cubePath;
        for( const auto &[key, text] : values )
            app->config.m_values[key] = text;
        return app;
    };

    // Default fit without a mask: uniform weights cannot represent these modes as a tilt plane; the reason is reported.
    auto unmasked = make( {} );
    REQUIRE( unmasked->loadConfigImpl( unmasked->config ) == 0 );
    REQUIRE( unmasked->setupProperties() == 0 );
    REQUIRE_FALSE( unmasked->prepareDMCalibration() );
    unmasked->updateDMINDI();
    const auto reason = unmasked->m_indiP_dmCalibration["status"].getValue();
    INFO( reason );
    REQUIRE( reason.find( "uniform weights 1.0 over 50x50" ) != std::string::npos );
    REQUIRE( reason.find( "relative residuals X" ) != std::string::npos );
    auto enable      = unmasked->m_indiP_dmEnable;
    enable["toggle"] = pcf::IndiElement::On;
    REQUIRE( unmasked->newCallBack_m_indiP_dmEnable( enable ) < 0 );

    // Explicit mask: unchanged fit; its K is the calibration to reuse.
    auto masked = make( { { "dm.mask", maskPath } } );
    REQUIRE( masked->loadConfigImpl( masked->config ) == 0 );
    REQUIRE( masked->prepareDMCalibration() );
    REQUIRE( masked->calibrationUnchanged() );
    const Eigen::Matrix2d k = masked->m_dmInjection.mapping();

    // Explicit coefficients: no mask and no fit; the cube alone generates the same full-grid commands.
    char text[160];
    std::snprintf( text, sizeof( text ), "%.17g,%.17g,%.17g,%.17g", k( 0, 0 ), k( 0, 1 ), k( 1, 0 ), k( 1, 1 ) );
    auto coefficients = make( { { "dm.calibration", "coefficients" }, { "dm.coefficients", text } } );
    REQUIRE( coefficients->loadConfigImpl( coefficients->config ) == 0 );
    REQUIRE( coefficients->setupProperties() == 0 );
    REQUIRE( coefficients->prepareDMCalibration() );
    REQUIRE( coefficients->m_dmInjection.mapping() == k );
    const auto command = coefficients->m_dmInjection.synthesize( value, 2.5 );
    REQUIRE( command == masked->m_dmInjection.synthesize( value, 2.5 ) );
    requireModes( command, modes, k, value, 2.5 );
    coefficients->m_dmReady = true; // As prepareDM() does before starting the worker.
    coefficients->updateDMINDI();
    REQUIRE( coefficients->m_indiP_dmCalibration["status"].getValue() == "ready" );
    REQUIRE( coefficients->m_indiP_dmCalibration["method"].getValue() == "coefficients" );
    REQUIRE( std::stod( coefficients->m_indiP_dmCalibration["k00"].getValue() ) == Approx( k( 0, 0 ) ) );
    REQUIRE( coefficients->newCallBack_m_indiP_dmEnable( enable ) == 0 );
    REQUIRE( coefficients->m_dmEnabled );

    // Invalid or ambiguous calibration configuration is rejected at load.
    for( const auto &values : std::vector<std::map<std::string, std::string>>{
             { { "dm.calibration", "coefficients" }, { "dm.coefficients", "1,2,3" } },
             { { "dm.calibration", "coefficients" }, { "dm.coefficients", "1,2,3,x" } },
             { { "dm.calibration", "coefficients" }, { "dm.coefficients", text }, { "dm.mask", maskPath } },
             { { "dm.calibration", "plane" } },
             { { "dm.coefficients", text } } } )
    {
        auto app = make( values );
        REQUIRE( app->loadConfigImpl( app->config ) < 0 );
    }
    std::filesystem::remove_all( directory );
}
#endif

} // namespace accelSimTest
} // namespace libXWCTest
