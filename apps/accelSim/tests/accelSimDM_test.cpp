/** \file accelSimDM_test.cpp
 * \brief Same-sample DM synthesis, lifecycle, active publication, and timing regressions.
 * \ingroup accelSim_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelSimDM.hpp"
#include <chrono>
#include <iostream>

#ifndef ACCELSIM_MODEL_ONLY
    #include "../accelSim.hpp"
    #include "../accelSimDMStream.hpp"
    #include <filesystem>
    #include <unistd.h>
#endif

namespace libXWCTest
{
namespace accelSimTest
{
using namespace MagAOX::app::accelSimDetail;

/// \cond ACCELSIM_DM_FIXTURE
/// Preallocated stand-in for an active sink; captures the last entire frame without allocating.
struct dmFixture
{
    std::vector<double>     m_x, m_y, m_weights; ///< Synthetic TT planes and circular fitting support.
    std::array<float, 2500> m_last{};            ///< Last completed publication, independent of producer storage.
    size_t                  m_count{ 0 };        ///< Successful active updates.
    bool                    m_fail{ false };     ///< Inject a missing/broken stream.

    /// Build column/row ramps in surface microns per coefficient.
    dmFixture();

    /// Copy one complete command, or fail without changing the last published frame.
    bool operator()( const std::vector<float> &command /**< [in] complete actuator image */ );
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

bool dmFixture::operator()( const std::vector<float> &command )
{
    if( m_fail )
        return false;
    std::copy( command.begin(), command.end(), m_last.begin() );
    ++m_count;
    return true;
}
/// \endcond

/** \brief Disabled samples cause no publication; zero scale and control boundaries cannot replay commands.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM lifecycle and zero scale", "[accelSim][dm-runtime]" )
{
    dmFixture   sink;
    dmInjection dm;
    dm.prepare( sink.m_x, sink.m_y, sink.m_weights, 1e-6 );
    sample value;
    value.m_pointing = { .0029, -.001 };
    REQUIRE( dm.service( value, 0, 1, true, sink ) == dmState::disabled );
    REQUIRE( sink.m_count == 0 );
    REQUIRE( dm.zero( sink ) );
    REQUIRE( dm.service( value, 3, 0, true, sink ) == dmState::active );
    REQUIRE( std::all_of( sink.m_last.begin(), sink.m_last.end(), []( float v ) { return v == 0; } ) );
    REQUIRE( dm.service( value, 3, 1, true, sink ) == dmState::active );
    REQUIRE( sink.m_last[0] != 0 );
    REQUIRE( dm.service( value, 4, 1, true, sink ) == dmState::disabled );
    REQUIRE( std::all_of( sink.m_last.begin(), sink.m_last.end(), []( float v ) { return v == 0; } ) );
    const auto count = sink.m_count;
    for( int n = 0; n < 100; ++n )
        REQUIRE( dm.service( value, 4, 1, true, sink ) == dmState::disabled );
    REQUIRE( dm.service( value, 3, 1, true, sink ) == dmState::disabled );
    REQUIRE( sink.m_count == count );
    REQUIRE( dm.service( value, 7, 1, true, sink ) == dmState::active );
    REQUIRE( dm.zero( sink ) ); // Same operation used after acquisition joins on shutdown.
    REQUIRE( sink.m_last[0] == 0 );
}

/** \brief The requested amplitude scale preserves calibrated surfaces, including large finite commands.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM physical mapping and amplitude scale", "[accelSim][dm-runtime]" )
{
    dmFixture   sink;
    dmInjection dm;
    dm.prepare( sink.m_x, sink.m_y, sink.m_weights, 1e-6 );
    for( const auto angles : { std::array<double, 2>{ .001, 0 }, { 0, .001 }, { -.0029, .0029 }, { .5, -.8 } } )
    {
        sample value;
        value.m_pointing    = angles;
        value.m_wfsPointing = { NAN, NAN };
        const pointingDisturbance pointing( value );
        for( double scale : { .125, .5, 1. } )
        {
            REQUIRE( dm.service( value, 3, scale, true, sink ) == dmState::active );
            for( size_t n = 0; n < 2500; ++n )
            {
                const double desired =
                    pointing.surface( ( double( n % 50 ) - 24.5 ) * 6.5 / 48, ( double( n / 50 ) - 24.5 ) * 6.5 / 48 );
                REQUIRE( sink.m_last[n] == Approx( desired * scale ).margin( 1e-8 ) );
            }
        }
    }
    const auto extremes = std::minmax_element( sink.m_last.begin(), sink.m_last.end() );
    REQUIRE( std::max( std::abs( *extremes.first ), std::abs( *extremes.second ) ) > .05 );
    REQUIRE( double( *extremes.second ) - *extremes.first > .10 );
}

/** \brief Invalid samples, bad calibration, missing streams, and timing loss inhibit until explicit rearm.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM faults latch and zero", "[accelSim][dm-runtime]" )
{
    dmFixture   sink;
    dmInjection dm;
    sample      value;
    REQUIRE( dm.service( value, 3, 1, true, sink ) == dmState::calibrationFault );
    REQUIRE_THROWS( dm.prepare( {}, {}, {}, 1e-6 ) );
    dm.prepare( sink.m_x, sink.m_y, sink.m_weights, 1e-6 );
    uint64_t revision = 3;
    for( double bad : { double( NAN ), double( INFINITY ), std::numeric_limits<double>::max() } )
    {
        revision += 2;
        value.m_pointing = { bad, 0 };
        REQUIRE( dm.service( value, revision, 1, true, sink ) == dmState::numericalFault );
        REQUIRE( sink.m_last[0] == 0 );
        value.m_pointing = { .0029, 0 };
        REQUIRE( dm.service( value, revision, 1, true, sink ) == dmState::numericalFault );
        REQUIRE( dm.service( value, ++revision, 1, true, sink ) == dmState::disabled );
        revision += 3;
        REQUIRE( dm.service( value, revision, 1, true, sink ) == dmState::active );
    }
    REQUIRE( dm.service( value, revision, 1, false, sink ) == dmState::timingFault );
    REQUIRE( sink.m_last[0] == 0 );
    sink.m_fail = true;
    revision += 2;
    REQUIRE( dm.service( value, revision, 1, true, sink ) == dmState::streamFault );
    sink.m_fail = false;
    REQUIRE( dm.service( value, revision, 1, true, sink ) == dmState::streamFault );
    REQUIRE( sink.m_last[0] == 0 );
    REQUIRE( dm.service( value, ++revision, 1, true, sink ) == dmState::disabled );
}

/** \brief Injection observes exactly the existing model realization, with unchanged RNG and ADC evolution.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM same sample and timing budget", "[accelSim][dm-runtime]" )
{
    dmFixture   sink;
    dmInjection dm;
    dm.prepare( sink.m_x, sink.m_y, sink.m_weights, 1e-6 );
    parameters config;
    config.m_fps = 8000;
    model baseline, injected;
    baseline.configure( config, referenceModes( 123 ) );
    injected.configure( config, referenceModes( 123 ) );
    double worst = 0, total = 0;
    for( size_t n = 0; n < 8000; ++n )
    {
        const sample expected = baseline.step(), value = injected.step();
        const auto   start = std::chrono::steady_clock::now();
        const auto   state = dm.service( value, 3, .1, true, sink );
        const double us = std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count();
        total += us;
        worst = std::max( worst, us );
        REQUIRE( state == dmState::active );
        REQUIRE( injected.m_ticks == baseline.m_ticks );
        REQUIRE( value.m_adc == expected.m_adc );
        REQUIRE( value.m_pointing == expected.m_pointing );
        REQUIRE( value.m_acceleration == expected.m_acceleration ); // Also verifies the sensor-noise draws.
    }
    REQUIRE( sink.m_count == 8000 );
    std::cout << "DM synthesis + preallocated mock publication: mean " << total / 8000 << " us, max " << worst
              << " us (not a native stream/driver benchmark)\n";
}

#ifndef ACCELSIM_MODEL_ONLY
/// \cond ACCELSIM_DM_APP_FIXTURE
/// Expose app integration points without starting acquisition or connecting an INDI server.
struct dmAppTest : MagAOX::app::accelSim
{
    using accelSim::config;
    using accelSim::m_activeFps;
    using accelSim::m_configName;
    using accelSim::m_dmCalibrationValid;
    using accelSim::m_dmInjection;
    using accelSim::m_dmReady;
    using accelSim::m_dmRequest;
    using accelSim::m_dmScale;
    using accelSim::m_dmStream;
    using accelSim::m_fresh;
    using accelSim::m_model;
    using accelSim::setupProperties;
};
/// \endcond

/** \brief Unconfigured apps remain disabled; enable is rejected and scale is validated without stream access.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM controls default disabled", "[accelSim][dm-controls]" )
{
    dmAppTest app;
    app.m_configName = "accelSim-dm-test";
    app.setupConfig();
    REQUIRE( app.loadConfigImpl( app.config ) == 0 );
    REQUIRE( app.setupProperties() == 0 );
    REQUIRE_FALSE( app.m_dmReady );
    REQUIRE( app.m_dmRequest == 0 );
    REQUIRE( app.m_dmScale == 0 );
    pcf::IndiProperty enable( pcf::IndiProperty::Switch );
    enable.setDevice( "accelSim-dm-test" );
    enable.setName( "dm_enable" );
    enable.add( pcf::IndiElement( "toggle" ) );
    enable["toggle"] = pcf::IndiElement::On;
    REQUIRE( app.newCallBack_m_indiP_dmEnable( enable ) < 0 );
    pcf::IndiProperty scale( pcf::IndiProperty::Number );
    scale.setDevice( "accelSim-dm-test" );
    scale.setName( "dm_scale" );
    scale.add( pcf::IndiElement( "target" ) );
    scale["target"] = .1;
    REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) == 0 );
    scale["target"] = -1;
    REQUIRE( app.newCallBack_m_indiP_dmScale( scale ) < 0 );
    REQUIRE( app.m_dmScale == .1 );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::setupProperties;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmEnable;
    MagAOX::app::accelSim::newCallBack_m_indiP_dmScale;
#endif
    // clang-format on
}

/** \brief Native active updates are tested only in a private namespace with synthetic images, never a physical DM.
 * \ingroup accelSim_unit_test
 */
TEST_CASE( "accelSim DM private native publication", "[.][accelSim-dm-streams]" )
{
    char path[] = "/tmp/accelSim-dm-private-XXXXXX";
    REQUIRE( mkdtemp( path ) != nullptr );
    REQUIRE( setenv( "MILK_SHM_DIR", path, 1 ) == 0 );
    char resolved[1024];
    ImageStreamIO_filename( resolved, sizeof( resolved ), "accelsim-private-dm" );
    REQUIRE( std::string( resolved ).starts_with( std::string( path ) + "/" ) );
    // Run in a separate test process: ImageStreamIO can cache its namespace.
    IMAGE    image{};
    uint32_t size[2] = { 50, 50 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &image, "accelsim-private-dm", 2, size, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) == 0 );
    dmStream stream;
    REQUIRE( stream.open( "accelsim-private-dm" ) );
    std::vector<float> command( 2500, .01f );
    const auto         count = image.md->cnt0;
    stream.acquisitionTime( timespec{ 123, 456 } );
    REQUIRE( stream( command ) );
    REQUIRE( image.md->cnt0 == count + 1 );
    REQUIRE( image.md->write == 0 );
    REQUIRE( image.md->atime.tv_sec == 123 );
    REQUIRE( image.md->atime.tv_nsec == 456 );
    REQUIRE( image.md->writetime.tv_sec > 0 );
    REQUIRE( image.array.F[0] == .01f );
    int semValue = 0;
    REQUIRE( sem_getvalue( image.semptr[0], &semValue ) == 0 );
    REQUIRE( semValue > 0 );
    image.md->write = 1;
    REQUIRE_FALSE( stream( command ) ); // No wait on a competing/stale writer.
    image.md->write   = 0;
    image.md->size[0] = 49;
    REQUIRE_FALSE( stream( command ) );
    image.md->size[0] = 50;
    stream.close();
    {
        dmAppTest app;
        app.m_configName = "accelSim-dm-test";
        app.setupConfig();
        REQUIRE( app.loadConfigImpl( app.config ) == 0 );
        app.m_model.setRate( 8000 );
        app.m_activeFps = 8000;
        dmFixture calibration;
        app.m_dmInjection.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
        REQUIRE( app.m_dmStream.open( "accelsim-private-dm" ) );
        app.m_dmReady            = true;
        app.m_dmCalibrationValid = true;
        app.m_dmRequest          = 3;
        app.m_dmScale            = 1;
        app.m_model.step();
        const auto sample = app.m_model.m_sample;
        const auto ticks  = app.m_model.m_ticks;
        uint16_t   adc[2];
        app.m_fresh = true;
        REQUIRE( app.loadImageIntoStream( adc ) == 0 );
        const auto before = image.md->cnt0;
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
        REQUIRE( image.md->cnt0 == before + 1 );
        REQUIRE( app.m_model.m_ticks == ticks );
        REQUIRE( adc[0] == sample.m_adc[0] );
        REQUIRE( adc[1] == sample.m_adc[1] );
        dmInjection reference;
        reference.prepare( calibration.m_x, calibration.m_y, calibration.m_weights, 1e-6 );
        REQUIRE( reference.service( sample, 3, 1, true, calibration ) == dmState::active );
        for( size_t n = 0; n < 2500; ++n )
            REQUIRE( image.array.F[n] == calibration.m_last[n] );
        app.m_dmRequest = 4;
        REQUIRE( app.frameGrabberPostPublish( nullptr ) == 0 );
        REQUIRE( image.array.F[0] == 0 );
        app.m_dmStream.close();
    }
    ImageStreamIO_destroyIm( &image );
    size[0] = 49;
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &image, "accelsim-private-bad", 2, size, _DATATYPE_FLOAT, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) == 0 );
    REQUIRE_FALSE( stream.open( "accelsim-private-bad" ) );
    ImageStreamIO_destroyIm( &image );
    size[0] = 50;
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &image, "accelsim-private-type", 2, size, _DATATYPE_UINT16, -1, 1, IMAGE_NB_SEMAPHORE, 0, 0, 0 ) ==
             0 );
    REQUIRE_FALSE( stream.open( "accelsim-private-type" ) );
    ImageStreamIO_destroyIm( &image );
    REQUIRE_FALSE( stream.open( "accelsim-private-absent" ) );
    REQUIRE_FALSE( std::filesystem::exists( std::string( path ) + "/accelsim-private-absent.im.shm" ) );
    std::filesystem::remove_all( path );
    // clang-format off
#ifdef ACCELSIM_TEST_DOXYGEN_REF
    MagAOX::app::accelSim::loadImageIntoStream;
    MagAOX::app::accelSim::frameGrabberPostPublish;
#endif
    // clang-format on
}
#endif

} // namespace accelSimTest
} // namespace libXWCTest
