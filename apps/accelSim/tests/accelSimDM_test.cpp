/** \file accelSimDM_test.cpp
 * \brief Calibrated physical DM synthesis and simple operator-control regressions.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimDM.hpp"
#include <chrono>
#include <iostream>

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

#ifndef ACCELSIM_MODEL_ONLY
/// \cond ACCELSIM_DM_APP_FIXTURE
/// Expose actual app callbacks without starting framework threads.
struct dmAppTest : MagAOX::app::accelSim
{
    using accelSim::config;
    using accelSim::m_configName;
    using accelSim::m_dmEnabled;
    using accelSim::m_dmInjection;
    using accelSim::m_dmReady;
    using accelSim::m_dmScale;
    using accelSim::m_dmZeroPending;
    using accelSim::m_indiP_dmEnable;
    using accelSim::m_indiP_dmScale;
    using accelSim::m_indiP_dmStatus;
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

} // namespace accelSimTest
} // namespace libXWCTest
