// #define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "dm_test.hpp"


/** \defgroup dm_tests libXWC::app::dev::dm Unit Tests
 * \ingroup app_dev_unit_tests
 */

/// Test dm Configuration
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm Configuration", "[dev::dm]" )
{
    SECTION( "a config file with no [dm] section, loading defaults" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/dm_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        dm_tests::dmTest pdt( "xx", false );

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/dm_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        REQUIRE( pdt.calibPath() == "/tmp/dmtest_calibs/dmtest" );

        // There will be no shmimName set
        REQUIRE( pdt.shmimName() == "" );
    }

    /// \todo finish implementing this
    SECTION( "a config file with a [dm] section changing everything" )
    {

        std::vector<std::string> s, k, v;

        s.push_back( "dm" );
        k.push_back( "calibPath" );
        v.push_back( "/tmp/dmtest_calibs2/dmtest2" );

        mx::app::writeConfigFile( "/tmp/dm_test.conf", s, k, v );

        mx::app::appConfigurator config;

        dm_tests::dmTest pdt( "xx", false );

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/dm_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        REQUIRE( pdt.calibPath() == "/tmp/dmtest_calibs2/dmtest2" );
    }

#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::calibPath();
#endif
}

/// Test dmcomb detection, configuration, and manipulation
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dmcomb detection, configuration, and manipulation", "[dev::dm]" )
{
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    char ppath[1024];
    snprintf( ppath, sizeof( ppath ), "%s=/tmp/dmtest/shm", "MILK_SHM_DIR" );
    putenv( ppath );

    mx::improc::milkImage<float> ch0, ch1, ch2, ch3, ch4, chT;
    try
    {

        ch0.create( "dmtest00", 50, 50 );
        ch0().setConstant(1);

        ch1.create( "dmtest01", 50, 50 );
        ch1().setConstant(2);

        ch2.create( "dmtest02", 50, 50 );
        ch2().setConstant(3);

        ch3.create( "dmtest03", 50, 50 );
        ch3().setConstant(4);

        ch4.create( "dmtest04", 50, 50 );
        ch4().setConstant(5);

        chT.create( "dmtest", 50, 50 );
        chT() = ch0() + ch1() + ch2() + ch3() + ch4();

    }
    catch( const std::exception &e )
    {
        std::cerr << "dm_test: Exception creating dm channels: " << e.what() << '\n';
    }

    std::vector<std::string> s, k, v;

    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmtest" );

    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "50" );

    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "50" );

    s.push_back( "dm" );
    k.push_back( "deltaChannels" );
    v.push_back( "dmtest02,dmtest03" );

    mx::app::writeConfigFile( "/tmp/dm_test.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt( "xx", false );

    int rv;
    rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    REQUIRE( pdt.shmimName() == "dmtest" );

    pdt.setSize( 50, 50, IMAGESTRUCT_FLOAT );

    rv = pdt.allocate( MagAOX::app::dev::shmimT() );

    REQUIRE( rv == 0 );

    int rows = pdt.instSatMap().rows();
    REQUIRE( rows == 50 );

    int cols = pdt.instSatMap().cols();
    REQUIRE( cols == 50 );

    rows = pdt.accumSatMap().rows();
    REQUIRE( rows == 50 );

    cols = pdt.accumSatMap().cols();
    REQUIRE( cols == 50 );

    rows = pdt.satPercMap().rows();
    REQUIRE( rows == 50 );

    cols = pdt.satPercMap().cols();
    REQUIRE( cols == 50 );

    int nc = pdt.numChannels();
    REQUIRE( nc == 5 );

    size_t nd = pdt.deltaChannels().size();
    REQUIRE( nd == 2);

    const std::vector<size_t> &notDeltas = pdt.notDeltas();
    REQUIRE(notDeltas.size() == 3);
    REQUIRE(notDeltas[0] == 0);
    REQUIRE(notDeltas[1] == 1);
    REQUIRE(notDeltas[2] == 4);

    mx::improc::milkImage<float> outputShape;
    bool pass = false;
    try
    {
        outputShape.open("dmtest_shape");
        pass = true;
    }
    catch(...)
    {}

    REQUIRE(pass == true);
    REQUIRE(outputShape.rows() == 50);
    REQUIRE(outputShape.cols() == 50);
    REQUIRE(outputShape().sum() == 0);

    outputShape = chT;
    float sum = outputShape().sum();
    REQUIRE(sum == (50*50)*(1+2+3+4+5));

    mx::improc::milkImage<float> outputDelta;
    pass = false;
    try
    {
        outputDelta.open("dmtest_delta");
        pass = true;
    }
    catch(...)
    {}

    REQUIRE(pass == true);
    REQUIRE(outputDelta.rows() == 50);
    REQUIRE(outputDelta.cols() == 50);
    REQUIRE(outputDelta().sum() == 0);

    rows = pdt.totalFlat().rows();
    REQUIRE( rows == 50 );

    cols = pdt.totalFlat().cols();
    REQUIRE( cols == 50 );

    sum = pdt.totalFlat().sum();
    REQUIRE(sum == 0);

    pdt.makeDelta();

    sum = pdt.totalFlat().sum();
    REQUIRE(sum == (50*50)*(1+2+5));

    //Check that it's still the same
    sum = outputShape().sum();
    REQUIRE(sum == (50*50)*(1+2+3+4+5));

    sum = outputDelta().sum();
    REQUIRE(sum == (50*50)*(3+4));

}

#if 0
/// Test dm app logic
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm app logic", "[dev::dm]" )
{
    SECTION( "no errors" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/dm_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        dm_tests::dmTest pdt( "xx", false );

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/dm_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        pdt.m_tel.logPath( "/tmp/telems" );

        rv = pdt.appStartup();
        REQUIRE( rv == 0 );

        rv = pdt.appLogic();
        REQUIRE( rv == 0 );

        rv = pdt.appShutdown();
        REQUIRE( rv == 0 );
    }

    SECTION( "log thread shutsdown" )
    {
        // Just a dummy config setting
        mx::app::writeConfigFile( "/tmp/dm_test.conf", { "none" }, { "nada" }, { "0" } );

        mx::app::appConfigurator config;

        dm_tests::dmTest pdt( "xx", false );

        int rv;
        rv = pdt.setupConfig( config );
        REQUIRE( rv == 0 );

        config.readConfig( "/tmp/dm_test.conf" );

        rv = pdt.loadConfig( config );
        REQUIRE( rv == 0 );

        pdt.m_tel.logPath( "/tmp/telems" );

        rv = pdt.appStartup();
        REQUIRE( rv == 0 );

        rv = pdt.appLogic();
        REQUIRE( rv == 0 );

        pdt.m_tel.logShutdown( true );
        sleep( 1 );

        rv = pdt.appLogic();
        REQUIRE( rv == -1 );

        rv = pdt.appShutdown();
        REQUIRE( rv == 0 );
    }

    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::appStartup();
    MagAOX::app::dev::dm::appLogic();
    MagAOX::app::dev::dm::appShutdown();
    #endif
}

/// Test dm telem-logger fails to start
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm telem-logger fails to start", "[dev::dm]" )
{
    // Just a dummy config setting
    mx::app::writeConfigFile( "/tmp/dm_test.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;

    dm_tests::XWCTEST_TELEMETER_LOGSTART_ns::dmTest pdt( "xx", false );

    int rv;
    rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    pdt.m_tel.logPath( "/tmp/telems" );

    rv = pdt.appStartup();
    REQUIRE( rv == -1 );

    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::appStartup();
    #endif
}

#endif
