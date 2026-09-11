// #define CATCH_CONFIG_MAIN
/** \file dm_test.cpp
  * \brief Catch2 tests for the MagAOX::app::dev::dm device mixin.
  *
  * The component under test is the CRTP mixin in libMagAOX/app/dev/dm.hpp. It manages the
  * deformable mirror channels, flats, test patterns, and saturation monitoring.
  *
  * The tests drive the real mixin through the dmTest harness declared in dm_test.hpp. The
  * harness is a MagAOXApp<false> with a real shmimMonitor base. The hardware hooks initDM(),
  * zeroDM(), releaseDM(), and commandDM() are stubs whose return values a test can set. Real
  * milk shared memory image streams are created under /tmp/dmtest/shm by setting
  * MILK_SHM_DIR. Real FITS flat and test files are written under /tmp/dmtest_calibs. INDI
  * callbacks are driven directly with hand-built pcf::IndiProperty objects. No INDI server is
  * needed. Some tests install a FIFO-less INDI driver so the property update code runs. Tests
  * that need the INDI properties or the saturation semaphore run the real appStartup(), which
  * also starts the saturation thread. The harness destructor stops that thread.
  *
  * A few tests force real operating system failures by making /tmp/dmtest/shm unwritable.
  * milk caches the shm directory the first time it is resolved in a process, so every test in
  * this file must use /tmp/dmtest/shm.
  *
  * \ingroup dm_tests
  */
#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>
#include <filesystem>
#include <fstream>

#include "dm_test.hpp"


/** \defgroup dm_tests libXWC::app::dev::dm Unit Tests
 * \ingroup app_dev_unit_tests
 */

namespace
{

/// Short names for the switch states used by the hand-built properties below.
constexpr pcf::IndiElement::SwitchStateType swOn  = pcf::IndiElement::On;
constexpr pcf::IndiElement::SwitchStateType swOff = pcf::IndiElement::Off;

/// Hand-build a switch property addressed to one of the dm<> INDI callbacks. Each entry in
/// elements names a switch element and gives its state. The callbacks accept a property that
/// carries only some of the elements of the registered one, or none at all.
pcf::IndiProperty switchProperty( const std::string &device,
                                  const std::string &name,
                                  const std::vector<std::pair<std::string, pcf::IndiElement::SwitchStateType>> &elements = {} )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( device );
    ip.setName( name );
    for( const auto &e : elements )
    {
        ip.add( pcf::IndiElement( e.first, e.second ) );
    }
    return ip;
}

} // namespace

/// Verify that MagAOXApp<false>::setupBasicConfig() requests a critical shutdown when power
/// management is enabled without INDI. dmTest is a convenient MagAOXApp<false> harness for this.
/// MagAOXApp_test.cpp only instantiates MagAOXApp<true>, so the branch is unreachable there.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> setupBasicConfig rejects power mgt without INDI", "[dev::dm]" )
{
    dm_tests::dmTest pdt;
    pdt.m_powerMgtEnabled = true;

    REQUIRE( pdt.shutdown() == 0 );
    pdt.setupBasicConfig();
    REQUIRE( pdt.shutdown() != 0 );
}

/// Verify that MagAOXApp<false>::startINDI() trivially succeeds when _useINDI is false.
/// This branch is unreachable from MagAOXApp_test.cpp, which only instantiates MagAOXApp<true>.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> startINDI trivially succeeds", "[dev::dm]" )
{
    dm_tests::dmTest pdt;
    REQUIRE( pdt.startINDI() == 0 );
}

/// Verify that MagAOXApp<false>::createINDIFIFOS() trivially succeeds when INDI is compiled out.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> createINDIFIFOS trivially succeeds", "[dev::dm]" )
{
    dm_tests::dmTest pdt;
    REQUIRE( pdt.callCreateINDIFIFOS() == 0 );
}

/// Verify that the MagAOXApp<false> callbacks handleGetProperties(), handleNewProperty(), and
/// handleSetProperty() return immediately when m_useINDI is false. These constexpr early-return
/// branches are also unreachable from MagAOXApp_test.cpp.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> INDI handle* callbacks are no-ops", "[dev::dm]" )
{
    dm_tests::dmTest  pdt;
    pcf::IndiProperty ip;
    ip.setDevice( "xx" ); // any device name, the callbacks ignore it
    ip.setName( "someprop" );

    // With _useINDI == false these all return immediately, regardless of m_indiDriver.
    pdt.handleGetProperties( ip );
    pdt.handleNewProperty( ip );
    pdt.handleSetProperty( ip );
    REQUIRE( true );
}

/// Verify the MagAOXApp<false>::registerIndiPropertyNew() overloads that build the property from
/// a type, a permission, and a state. One overload takes an explicit switch rule and one does
/// not. Both are also unreachable from MagAOXApp_test.cpp.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> registerIndiPropertyNew type/perm/state overloads", "[dev::dm]" )
{
    dm_tests::dmTest  pdt;
    pcf::IndiProperty prop1, prop2;

    REQUIRE( pdt.registerIndiPropertyNew( prop1,
                                          "someprop",
                                          pcf::IndiProperty::Number,
                                          pcf::IndiProperty::ReadWrite,
                                          pcf::IndiProperty::Idle,
                                          nullptr ) == 0 );
    REQUIRE( pdt.registerIndiPropertyNew( prop2,
                                          "otherprop",
                                          pcf::IndiProperty::Switch,
                                          pcf::IndiProperty::ReadWrite,
                                          pcf::IndiProperty::Idle,
                                          pcf::IndiProperty::OneOfMany,
                                          nullptr ) == 0 );
}

/// Verify that MagAOXApp<false>::sendNewStandardIndiToggle() trivially succeeds when INDI is compiled out.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test MagAOXApp<false> sendNewStandardIndiToggle trivially succeeds", "[dev::dm]" )
{
    dm_tests::dmTest pdt;
    REQUIRE( pdt.callSendNewStandardIndiToggle( "somedev", "someprop", true ) == 0 );
}

/// Test dm Configuration
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm Configuration", "[dev::dm]" )
{
    SECTION( "a config file with no [dm] section, loading defaults" )
    {
        mx::app::writeConfigFile( "/tmp/dm_test.conf", { "none" }, { "nada" }, { "0" } ); // the placeholder entry every test uses so the config file is not empty

        mx::app::appConfigurator config;

        dm_tests::dmTest pdt;

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

        dm_tests::dmTest pdt;

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

    dm_tests::dmTest pdt;

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

    // Exercise the makeDelta() call inside processImage(). This is a different call site from
    // the direct makeDelta() call above. It runs because m_deltaChannels is not empty here.
    // processImage() posts the saturation semaphore, which appStartup() creates. appStartup()
    // also starts the saturation thread. The harness destructor stops it.
    REQUIRE( pdt.appStartup() == 0 );
    std::vector<float> srcbuf( 50 * 50, 0.0f );
    REQUIRE( pdt.processImage( srcbuf.data(), MagAOX::app::dev::shmimT() ) == 0 );

    //Check that it's still the same
    sum = outputShape().sum();
    REQUIRE(sum == (50*50)*(1+2+3+4+5));

    sum = outputDelta().sum();
    REQUIRE(sum == (50*50)*(3+4));

}

/// Verify that makeDelta() trivially succeeds when every channel is configured as a delta
/// channel. In that case m_notDeltas is empty.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm makeDelta with no non-delta channels", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    mx::improc::milkImage<float> ch0, ch1;
    ch0.create( "dmalldelta00", 4, 4 );
    ch1.create( "dmalldelta01", 4, 4 );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmalldelta" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "deltaChannels" );
    v.push_back( "dmalldelta00,dmalldelta01" );

    mx::app::writeConfigFile( "/tmp/dm_test_alldelta.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_alldelta.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    pdt.setSize( 4, 4, IMAGESTRUCT_FLOAT );
    REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == 0 );
    REQUIRE( pdt.notDeltas().size() == 0 );

    REQUIRE( pdt.makeDelta() == 0 );
}

/// Verify that appStartup() rejects an unsupported ImageStreamIO data type. The dmTestBadType
/// harness uses int as the DM data type, which has no ImageStreamIO type code.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm appStartup rejects unsupported data type", "[dev::dm]" )
{
    mx::app::writeConfigFile( "/tmp/dm_test_badtype.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;

    dm_tests::dmTestBadType pdt;

    int rv;
    rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test_badtype.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    rv = pdt.appStartup();
    REQUIRE( rv == -1 );
}

/// Verify that allocate() fails when the incoming stream width, height, or data type does not
/// match the configured DM.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm allocate size and type mismatches", "[dev::dm]" )
{
    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "20" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "20" );

    mx::app::writeConfigFile( "/tmp/dm_test_alloc.conf", s, k, v );

    SECTION( "width mismatch" )
    {
        dm_tests::dmTest pdt;
        mx::app::appConfigurator config;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_alloc.conf" );
        REQUIRE( pdt.loadConfig( config ) == 0 );
        pdt.setSize( 21, 20, IMAGESTRUCT_FLOAT );
        REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == -1 );
    }

    SECTION( "height mismatch" )
    {
        dm_tests::dmTest pdt;
        mx::app::appConfigurator config;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_alloc.conf" );
        REQUIRE( pdt.loadConfig( config ) == 0 );
        pdt.setSize( 20, 21, IMAGESTRUCT_FLOAT );
        REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == -1 );
    }

    SECTION( "data type mismatch" )
    {
        dm_tests::dmTest pdt;
        mx::app::appConfigurator config;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_alloc.conf" );
        REQUIRE( pdt.loadConfig( config ) == 0 );
        pdt.setSize( 20, 20, IMAGESTRUCT_DOUBLE );
        REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == -1 );
    }
}

/// Verify that findDMChannels() fails when no channels exist for the configured shmimName.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm findDMChannels with no channels found", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmnochan" );

    mx::app::writeConfigFile( "/tmp/dm_test_nochan.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    int rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test_nochan.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    REQUIRE( pdt.findDMChannels() == -1 );
}

/// Verify that findDMChannels() falls back to the hardcoded default "/milk/shm" when
/// MILK_SHM_DIR is unset.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm findDMChannels falls back to the default shm dir when MILK_SHM_DIR is unset", "[dev::dm]" )
{
    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmnoenvvar" );

    mx::app::writeConfigFile( "/tmp/dm_test_noenvvar.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_noenvvar.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    unsetenv( "MILK_SHM_DIR" );
    // This finds no channels in the real default "/milk/shm", or fails outright if that
    // directory does not exist. A test runner is not expected to have this test's shmim there.
    // Either way the fallback lookup itself runs for real, without mocking.
    REQUIRE( pdt.findDMChannels() == -1 );
}

/// Verify the catch block in findDMChannels() that logs an exception while opening a channel.
/// The exception is forced with a real corrupt file. Its name matches the channel naming
/// pattern so getFileNames() finds it, but its content cannot be parsed by ImageStreamIO. This
/// is the same real fault injection technique used by the corrupt shmim file test in
/// frameGrabber_test.cpp.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm findDMChannels logs an exception opening a corrupt channel file", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmcorruptchan" );

    mx::app::writeConfigFile( "/tmp/dm_test_corruptchan.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_corruptchan.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    char path[1024];
    ImageStreamIO_filename( path, sizeof( path ), "dmcorruptchan00" );
    FILE *f = fopen( path, "w" );
    REQUIRE( f != nullptr );
    const char junk[16] = { 0 };
    fwrite( junk, 1, sizeof( junk ), f );
    fclose( f );

    // findDMChannels() logs the exception and continues. It leaves that channel's pointer
    // null rather than failing outright, because m_numChannels is still greater than zero.
    REQUIRE( pdt.findDMChannels() == 0 );

    unlink( path );
}

/// Verify flat file discovery, loading, setting, and zeroing, including the error branches.
/// Real FITS flat files are written under /tmp/dmtest_calibs/flattest and a real milk channel
/// is created for setFlat() to write into.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm flat file management", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    // Remove any leftovers from a previous run first. This test creates more flat files as it
    // goes, and checkFlats() counts whatever is on disk.
    std::filesystem::remove_all( "/tmp/dmtest_calibs/flattest" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/flattest/flats" );

    // Write some flat fits files, one of which is size-mismatched.
    mx::improc::eigenImage<float> flatA( 6, 6 ), flatB( 6, 6 ), flatBad( 3, 3 );
    flatA.setConstant( 2 );
    flatB.setConstant( 5 );
    flatBad.setConstant( 1 );

    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/flattest/flats/flatA.fits", flatA );
    ff.write( "/tmp/dmtest_calibs/flattest/flats/flatB.fits", flatB );
    ff.write( "/tmp/dmtest_calibs/flattest/flats/flatBad.fits", flatBad );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/flattest" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmflat" );
    // Configuring dm.flatDefault makes m_flatCurrent start out as "default". That exercises
    // the branch in checkFlats() that turns the "default" element On.
    s.push_back( "dm" );
    k.push_back( "flatDefault" );
    v.push_back( "flatA" );

    mx::app::writeConfigFile( "/tmp/dm_test_flat.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    int rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test_flat.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    // Install a real INDI driver without FIFOs. This makes checkFlats() take its
    // m_indiDriver != nullptr branch, which calls sendDelProperty() and erase() when the flat
    // list has changed.
    pdt.setupRealDriver();

    // appStartup() creates the INDI properties that the callbacks below are checked against.
    // It also starts the saturation thread, which idles because nothing is allocated. The
    // harness destructor stops it.
    REQUIRE( pdt.appStartup() == 0 );

    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatCommands.size() == 3 );

    // Actuator mask mismatch. flatBad is 3x3 but the actuator mask is 6x6.
    REQUIRE( pdt.loadFlat( "flatBad" ) == -1 );

    // Not found.
    REQUIRE( pdt.loadFlat( "doesNotExist" ) == -1 );

    // Loading "default" resolves to flatA.fits through dm.flatDefault. It sets m_flatCurrent
    // to the literal string "default", which exercises the branch that turns
    // m_indiP_flats["default"] On.
    REQUIRE( pdt.loadFlat( "default" ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "default" );

    // Successful load.
    REQUIRE( pdt.loadFlat( "flatA" ) == 0 );
    REQUIRE( pdt.m_flatLoaded == true );
    REQUIRE( pdt.m_flatCurrent == "flatA" );

    // m_flatCurrent is now "flatA" rather than "default". checkFlats() only rebuilds the INDI
    // property when something changed on disk, and only then re-evaluates the "default"
    // element. Adding one more flat file forces the rebuild. This exercises the branch that
    // turns the "default" element Off, the mirror image of the On branch exercised above.
    ff.write( "/tmp/dmtest_calibs/flattest/flats/flatC.fits", flatA );
    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatCommands.size() == 4 );

    // Not READY or OPERATING yet.
    REQUIRE( pdt.setFlat() == -1 );

    pdt.state( MagAOX::app::stateCodes::READY );

    // The channel does not exist yet. Remove any stale shmim left from a previous test run.
    std::remove( "/tmp/dmtest/shm/dmflat00.im.shm" );
    REQUIRE( pdt.setFlat() == -1 );

    mx::improc::milkImage<float> flatChan;
    flatChan.create( "dmflat00", 6, 6 );

    REQUIRE( pdt.setFlat() == 0 );
    REQUIRE( pdt.m_flatSet == true );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE( flatChan().sum() == 6 * 6 * 2 );

    // The update == true path.
    REQUIRE( pdt.setFlat( true ) == 0 );

    // Switching flats while the flat is set re-applies it automatically.
    REQUIRE( pdt.loadFlat( "flatB" ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "flatB" );
    REQUIRE( pdt.m_flatSet == true );
    REQUIRE( flatChan().sum() == 6 * 6 * 5 );

    REQUIRE( pdt.zeroFlat() == 0 );
    REQUIRE( pdt.m_flatSet == false );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::READY );
    REQUIRE( flatChan().sum() == 0 );

    // Exercise the newCallBack_setFlat() and newCallBack_flats() INDI paths with hand-built
    // properties. The device is the config name. The property names are the ones appStartup()
    // and checkFlats() register.
    const std::string dev = pdt.configName();

    REQUIRE( pdt.newCallBack_setFlat( switchProperty( dev, "flat_set", { { "toggle", swOn } } ) ) == 0 );
    REQUIRE( pdt.m_flatSet == true );

    REQUIRE( pdt.newCallBack_setFlat( switchProperty( dev, "flat_set", { { "toggle", swOff } } ) ) == 0 );
    REQUIRE( pdt.m_flatSet == false );

    // A real client sends the full new selection state, so the other elements are sent Off.
    // Otherwise the previous selection, flatB, would still be On too.
    pcf::IndiProperty ipFlats =
        switchProperty( dev, "flat", { { "flatA", swOn }, { "flatB", swOff }, { "flatBad", swOff } } );
    REQUIRE( pdt.newCallBack_flats( ipFlats ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "flatA" );

    // Selecting "default" resolves to loadFlat("default").
    pcf::IndiProperty ipFlatsDefault = switchProperty(
        dev, "flat", { { "default", swOn }, { "flatA", swOff }, { "flatB", swOff }, { "flatBad", swOff } } );
    REQUIRE( pdt.newCallBack_flats( ipFlatsDefault ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "default" );

    // Selecting more than one flat at once is an error.
    pcf::IndiProperty ipFlatsTwo = switchProperty(
        dev, "flat", { { "default", swOff }, { "flatA", swOn }, { "flatB", swOn }, { "flatBad", swOff } } );
    REQUIRE( pdt.newCallBack_flats( ipFlatsTwo ) == -1 );

    // Selecting none at all is a trivial no-op.
    pcf::IndiProperty ipFlatsNone = switchProperty(
        dev, "flat", { { "default", swOff }, { "flatA", swOff }, { "flatB", swOff }, { "flatBad", swOff } } );
    REQUIRE( pdt.newCallBack_flats( ipFlatsNone ) == 0 );

    // A property that only reports a subset of the known flat elements, rather than the full
    // current state, exercises the per-element skip branch for elements that are not present.
    REQUIRE( pdt.newCallBack_flats( switchProperty( dev, "flat", { { "flatA", swOn } } ) ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "flatA" );

    // Wrong device and property name.
    REQUIRE( pdt.newCallBack_setFlat( switchProperty( "somethingelse", "notflat" ) ) == -1 );
}

/// Verify three parts of checkFlats(). It excludes the default file from the trim. It keeps
/// only the 5 most recent non-default files. Repeated calls exercise the bookkeeping of the
/// tracked command map, which is the per-call reset, the already-tracked case, and stale
/// removal.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm checkFlats file-count trimming and stale-removal", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/flattrim" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/flattrim/flats" );

    mx::improc::eigenImage<float> flat( 4, 4 );
    flat.setConstant( 1 );
    mx::fits::fitsFile<float> ff;

    // A "default" file must be excluded from the timestamp-based trim below.
    ff.write( "/tmp/dmtest_calibs/flattrim/flats/default.fits", flat );

    // Write 6 timestamped files. That is more than the hardcoded limit of 5 most recent
    // files, so the single oldest file, f1, must be trimmed on the first call.
    std::vector<std::string> names = { "f1", "f2", "f3", "f4", "f5", "f6" };
    for( size_t n = 0; n < names.size(); ++n )
    {
        std::string path = "/tmp/dmtest_calibs/flattrim/flats/" + names[n] + ".fits";
        ff.write( path, flat );
        auto t = std::filesystem::last_write_time( path ) -
                 std::chrono::seconds( static_cast<long>( ( names.size() - n ) * 10 ) );
        std::filesystem::last_write_time( path, t );
    }

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/flattrim" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmflattrim" );

    mx::app::writeConfigFile( "/tmp/dm_test_flattrim.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_flattrim.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // First call. This exercises the branch that removes the default file and the branch that
    // trims to 5 files. It keeps the 5 most recent of f1 through f6. f1 is the oldest and is
    // dropped.
    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatCommands.size() == 5 );

    // Second call with the same files still on disk. This exercises the per-call reset loop
    // and the else branch of the insert loop for files that are already tracked.
    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatCommands.size() == 5 );

    // Remove one previously tracked file, f6, then call again. This exercises the stale entry
    // removal branch. f1 was already excluded by the trim above but is still on disk until
    // now, and it must also be removed here. Otherwise dropping to 5 files on disk re-admits
    // f1 into the trim window, and f1 taking the place of f6 would mask the intended stale
    // removal.
    std::filesystem::remove( "/tmp/dmtest_calibs/flattrim/flats/f6.fits" );
    std::filesystem::remove( "/tmp/dmtest_calibs/flattrim/flats/f1.fits" );
    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatCommands.size() == 4 );
}

/// Verify the width and height mismatch branches of setFlat() against the real channel.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm setFlat channel size mismatch", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/flattest2" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/flattest2/flats" );

    mx::improc::eigenImage<float> flatOK( 6, 6 );
    flatOK.setConstant( 1 );
    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/flattest2/flats/flatOK.fits", flatOK );

    std::vector<std::string> s2, k2, v2;
    s2.push_back( "dm" );
    k2.push_back( "calibPath" );
    v2.push_back( "/tmp/dmtest_calibs/flattest2" );
    s2.push_back( "dm" );
    k2.push_back( "width" );
    v2.push_back( "6" );
    s2.push_back( "dm" );
    k2.push_back( "height" );
    v2.push_back( "6" );
    s2.push_back( "dm" );
    k2.push_back( "shmimName" );
    v2.push_back( "dmflatmm" );
    mx::app::writeConfigFile( "/tmp/dm_test_flat2.conf", s2, k2, v2 );

    dm_tests::dmTest         pdt2;
    mx::app::appConfigurator config2;
    REQUIRE( pdt2.setupConfig( config2 ) == 0 );
    config2.readConfig( "/tmp/dm_test_flat2.conf" );
    REQUIRE( pdt2.loadConfig( config2 ) == 0 );
    pdt2.state( MagAOX::app::stateCodes::READY );

    // The channel is 3x3 but the configured DM is 6x6. The width mismatch is checked first.
    mx::improc::milkImage<float> flatChanBad;
    flatChanBad.create( "dmflatmm00", 3, 3 );

    REQUIRE( pdt2.checkFlats() == 0 );
    REQUIRE( pdt2.loadFlat( "flatOK" ) == 0 );

    REQUIRE( pdt2.setFlat() == -1 );

    // The channel width now matches at 6 but the height is 3. This exercises the separate
    // height mismatch branch, which the width mismatch above never reaches.
    flatChanBad.create( "dmflatmm00", 6, 3 );
    REQUIRE( pdt2.setFlat() == -1 );
}

/// Verify test pattern file discovery, loading, setting, and zeroing. Real FITS files are
/// written under /tmp/dmtest_calibs/testtest and a real milk channel is created for setTest()
/// to write into.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm test pattern file management", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    // Remove any leftovers from a previous run first. This test creates more test pattern
    // files as it goes, and checkTests() counts whatever is on disk.
    std::filesystem::remove_all( "/tmp/dmtest_calibs/testtest" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/testtest/tests" );

    mx::improc::eigenImage<float> testA( 6, 6 ), testB( 6, 6 );
    testA.setConstant( 3 );
    testB.setConstant( 7 );

    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testA.fits", testA );
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testB.fits", testB );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/testtest" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmtst" );
    // Configuring dm.testDefault makes m_testCurrent start out as "default". That exercises
    // the branches in checkTests() and loadTest() that turn the "default" element On.
    s.push_back( "dm" );
    k.push_back( "testDefault" );
    v.push_back( "testA" );

    mx::app::writeConfigFile( "/tmp/dm_test_test.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    int rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test_test.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    // Install a real INDI driver without FIFOs. This makes checkTests() take its
    // m_indiDriver != nullptr branch, which calls sendDelProperty() and erase() when the test
    // list has changed.
    pdt.setupRealDriver();

    // appStartup() creates the INDI properties that the callbacks below are checked against.
    // It also starts the saturation thread, which idles because nothing is allocated. The
    // harness destructor stops it.
    REQUIRE( pdt.appStartup() == 0 );

    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.m_testCommands.size() == 2 );

    // Second call with the same files still on disk. This exercises the per-call reset loop
    // and the else branch of the insert loop for files that are already tracked.
    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.m_testCommands.size() == 2 );

    // Remove one previously tracked file, then call again. This exercises the stale entry
    // removal branch.
    std::filesystem::remove( "/tmp/dmtest_calibs/testtest/tests/testB.fits" );
    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.m_testCommands.size() == 1 );

    // Re-create testB.fits. It is used again below by loadTest("testB").
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testB.fits", testB );
    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.m_testCommands.size() == 2 );

    REQUIRE( pdt.loadTest( "doesNotExist" ) == -1 );

    // Write a real file that cannot be parsed. checkTests() tracks it by file name alone, but
    // the real FITS read in loadTest() then fails on its content. Nothing is mocked.
    {
        std::ofstream corrupt( "/tmp/dmtest_calibs/testtest/tests/corrupt.fits" );
        corrupt << "not a fits file";
        corrupt.close();
        REQUIRE( pdt.checkTests() == 0 );
        REQUIRE( pdt.loadTest( "corrupt" ) == -1 );
        std::filesystem::remove( "/tmp/dmtest_calibs/testtest/tests/corrupt.fits" );
        REQUIRE( pdt.checkTests() == 0 );
    }

    // Loading "default" resolves to testA.fits through dm.testDefault. It sets m_testCurrent
    // to the literal string "default", which exercises the branch that turns
    // m_indiP_tests["default"] On.
    REQUIRE( pdt.loadTest( "default" ) == 0 );
    REQUIRE( pdt.m_testCurrent == "default" );

    REQUIRE( pdt.loadTest( "testA" ) == 0 );
    REQUIRE( pdt.m_testLoaded == true );
    REQUIRE( pdt.m_testCurrent == "testA" );

    // m_testCurrent is now "testA" rather than "default". Adding a new file forces
    // checkTests() to rebuild the INDI property again. This exercises the branch that turns
    // the "default" element Off, the mirror image of the On branch exercised by the first
    // checkTests() call above, when m_testCurrent was still "default".
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testC.fits", testA );
    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.m_testCommands.size() == 3 );

    // The channel does not exist yet. Remove any stale shmim left from a previous test run.
    std::remove( "/tmp/dmtest/shm/dmtst02.im.shm" );
    REQUIRE( pdt.setTest() == -1 );

    mx::improc::milkImage<float> testChan;
    testChan.create( "dmtst02", 6, 6 );

    REQUIRE( pdt.setTest() == 0 );
    REQUIRE( pdt.m_testSet == true );
    REQUIRE( testChan().sum() == 6 * 6 * 3 );

    REQUIRE( pdt.loadTest( "testB" ) == 0 );
    REQUIRE( testChan().sum() == 6 * 6 * 7 );

    // Unlike loadFlat(), loadTest() does not validate the size of the loaded file. A
    // mismatched test file therefore loads successfully. setTest() must catch the mismatch
    // against the real channel and the configured DM size.
    mx::improc::eigenImage<float> testBadW( 3, 6 ), testBadH( 6, 3 );
    testBadW.setConstant( 9 );
    testBadH.setConstant( 9 );
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testBadW.fits", testBadW );
    ff.write( "/tmp/dmtest_calibs/testtest/tests/testBadH.fits", testBadH );
    REQUIRE( pdt.checkTests() == 0 );

    REQUIRE( pdt.loadTest( "testBadW" ) == 0 );
    REQUIRE( pdt.setTest() == -1 );

    REQUIRE( pdt.loadTest( "testBadH" ) == 0 );
    REQUIRE( pdt.setTest() == -1 );

    // Restore a valid current test before the zeroTest() and INDI checks below.
    REQUIRE( pdt.loadTest( "testB" ) == 0 );
    REQUIRE( pdt.setTest() == 0 );

    REQUIRE( pdt.zeroTest() == 0 );
    REQUIRE( pdt.m_testSet == false );
    REQUIRE( testChan().sum() == 0 );

    // Exercise the newCallBack_setTest() and newCallBack_tests() INDI paths with hand-built
    // properties. See the flats case above.
    const std::string dev = pdt.configName();

    REQUIRE( pdt.newCallBack_setTest( switchProperty( dev, "test_set", { { "toggle", swOn } } ) ) == 0 );
    REQUIRE( pdt.m_testSet == true );

    REQUIRE( pdt.newCallBack_setTest( switchProperty( dev, "test_set", { { "toggle", swOff } } ) ) == 0 );
    REQUIRE( pdt.m_testSet == false );

    // The other selection is sent Off along with the target.
    REQUIRE( pdt.newCallBack_tests( switchProperty( dev, "test", { { "testA", swOn }, { "testB", swOff } } ) ) == 0 );
    REQUIRE( pdt.m_testCurrent == "testA" );

    // Selecting "default" resolves to loadTest("default").
    pcf::IndiProperty ipTestsDefault =
        switchProperty( dev, "test", { { "default", swOn }, { "testA", swOff }, { "testB", swOff } } );
    REQUIRE( pdt.newCallBack_tests( ipTestsDefault ) == 0 );
    REQUIRE( pdt.m_testCurrent == "default" );

    // Selecting more than one test at once is an error.
    pcf::IndiProperty ipTestsTwo =
        switchProperty( dev, "test", { { "default", swOff }, { "testA", swOn }, { "testB", swOn } } );
    REQUIRE( pdt.newCallBack_tests( ipTestsTwo ) == -1 );

    // Selecting none at all is a trivial no-op.
    pcf::IndiProperty ipTestsNone =
        switchProperty( dev, "test", { { "default", swOff }, { "testA", swOff }, { "testB", swOff } } );
    REQUIRE( pdt.newCallBack_tests( ipTestsNone ) == 0 );

    // A property that only reports a subset of the known test elements, rather than the full
    // current state, exercises the per-element skip branch for elements that are not present.
    REQUIRE( pdt.newCallBack_tests( switchProperty( dev, "test", { { "testA", swOn } } ) ) == 0 );
    REQUIRE( pdt.m_testCurrent == "testA" );

    // Wrong device and property name.
    REQUIRE( pdt.newCallBack_setTest( switchProperty( "somethingelse", "nottest" ) ) == -1 );
}

/// Verify that setTest() attempts an internal auto-load when called before any explicit
/// loadTest(), and reports the failure. The dm<> is fresh with no test target configured. The
/// internal load fails because the empty target is not found, and setTest() then reports that
/// no test is loaded.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm setTest internal auto-load failure", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmtstauto" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_testauto.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_testauto.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    mx::improc::milkImage<float> testChan;
    testChan.create( "dmtstauto02", 4, 4 );

    // m_testLoaded is false and m_testCurrent is empty. setTest() tries to load the test
    // internally. That fails because there is no such target, and setTest() reports no test
    // loaded.
    REQUIRE( pdt.m_testLoaded == false );
    REQUIRE( pdt.setTest() == -1 );
}

/// Verify the width and height mismatch branches of setTest() and zeroTest() against the real
/// channel. A mismatch in the in-memory test command is covered above.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm setTest and zeroTest channel size mismatch", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/testmm" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/testmm/tests" );

    mx::improc::eigenImage<float> testOK( 6, 6 );
    testOK.setConstant( 1 );
    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/testmm/tests/testOK.fits", testOK );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/testmm" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "6" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmtstmm" );
    mx::app::writeConfigFile( "/tmp/dm_test_testmm.conf", s, k, v );

    dm_tests::dmTest         pdt;
    mx::app::appConfigurator config;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_testmm.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );
    pdt.state( MagAOX::app::stateCodes::READY );

    REQUIRE( pdt.checkTests() == 0 );
    REQUIRE( pdt.loadTest( "testOK" ) == 0 );

    // The channel does not exist yet. Remove any stale shmim left from a previous test run.
    std::remove( "/tmp/dmtest/shm/dmtstmm02.im.shm" );
    REQUIRE( pdt.setTest() == -1 );
    REQUIRE( pdt.zeroTest() == -1 );

    // Width mismatch. It is checked first.
    mx::improc::milkImage<float> testChanBad;
    testChanBad.create( "dmtstmm02", 3, 6 );
    REQUIRE( pdt.setTest() == -1 );
    REQUIRE( pdt.zeroTest() == -1 );

    // The width now matches at 6 but the height is 3.
    testChanBad.create( "dmtstmm02", 6, 3 );
    REQUIRE( pdt.setTest() == -1 );
    REQUIRE( pdt.zeroTest() == -1 );
}

/// Verify zeroAll() and clearSat() using real dmcomb-style shmim channels.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm zeroAll and clearSat", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    mx::improc::milkImage<float> ch0, ch1;
    ch0.create( "dmza00", 4, 4 );
    ch0().setConstant( 3 );
    ch1.create( "dmza01", 4, 4 );
    ch1().setConstant( 4 );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmza" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_zeroall.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    int rv = pdt.setupConfig( config );
    REQUIRE( rv == 0 );

    config.readConfig( "/tmp/dm_test_zeroall.conf" );

    rv = pdt.loadConfig( config );
    REQUIRE( rv == 0 );

    // appStartup() creates the zeroAll INDI property that clearSat() and the callback below
    // use. It also starts the saturation thread. Stop that thread right away. It would
    // otherwise create the saturation streams that this test manages by hand.
    REQUIRE( pdt.appStartup() == 0 );
    pdt.requestShutdown();
    REQUIRE( pdt.appShutdown() == 0 );

    // clearSat() with no shmimSat configured trivially succeeds.
    REQUIRE( pdt.clearSat() == 0 );

    pdt.setSize( 4, 4, IMAGESTRUCT_FLOAT );
    REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == 0 );
    REQUIRE( pdt.numChannels() == 2 );

    // The saturation shmims do not exist yet, so clearSat() logs a warning and returns 0.
    // Remove any stale shmim left from a previous test run first.
    std::remove( ( std::string( "/tmp/dmtest/shm/" ) + pdt.shmimSat() + ".im.shm" ).c_str() );
    std::remove( ( std::string( "/tmp/dmtest/shm/" ) + pdt.shmimSatPerc() + ".im.shm" ).c_str() );
    REQUIRE( pdt.clearSat() == 0 );

    mx::improc::milkImage<uint8_t> satChan;
    satChan.create( pdt.shmimSat(), 4, 4 );
    mx::improc::milkImage<float> satPercChan;
    satPercChan.create( pdt.shmimSatPerc(), 4, 4 );

    // Fill both saturation maps so clearSat() has something to clear.
    pdt.m_accumSatMap.setConstant( 3 );
    pdt.m_instSatMap.setConstant( 1 );
    REQUIRE( pdt.clearSat() == 0 );
    REQUIRE( pdt.accumSatMap().sum() == 0 );
    REQUIRE( pdt.instSatMap().sum() == 0 );

    pdt.state( MagAOX::app::stateCodes::READY );
    REQUIRE( pdt.zeroAll() == 0 );
    REQUIRE( ch0().sum() == 0 );
    REQUIRE( ch1().sum() == 0 );

    // zeroAll() while OPERATING transitions back to READY.
    pdt.state( MagAOX::app::stateCodes::OPERATING );
    REQUIRE( pdt.zeroAll() == 0 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::READY );

    // A missing channel that cannot be opened is skipped. It is logged but not fatal.
    std::remove( "/tmp/dmtest/shm/dmza01.im.shm" );
    REQUIRE( pdt.zeroAll() == 0 );
    ch1.create( "dmza01", 4, 4 ); // Recreate it for the mismatch checks below.

    // zeroAll() propagates a clearSat() failure. Here the failure is a real size mismatch on a
    // saturation channel.
    mx::improc::milkImage<uint8_t> satChanBad;
    satChanBad.create( pdt.shmimSat(), 3, 3 );
    REQUIRE( pdt.zeroAll() == -1 );

    // The width now matches at 4 but the height is 3. This exercises the separate height
    // mismatch branch in clearSat().
    satChanBad.create( pdt.shmimSat(), 4, 3 );
    REQUIRE( pdt.zeroAll() == -1 );

    satChanBad.create( pdt.shmimSat(), 4, 4 ); // Restore it for the checks below.

    // The zeroAll() INDI callback path.
    pdt.state( MagAOX::app::stateCodes::READY );
    REQUIRE( pdt.newCallBack_zeroAll( switchProperty( pdt.configName(), "zeroAll", { { "request", swOn } } ) ) == 0 );

    // zeroAll() propagates a width mismatch between a real channel and the configured DM.
    ch1.create( "dmza01", 3, 4 );
    REQUIRE( pdt.zeroAll() == -1 );

    // zeroAll() propagates a height mismatch between a real channel and the configured DM.
    ch1.create( "dmza01", 4, 5 );
    REQUIRE( pdt.zeroAll() == -1 );

    // baseReleaseDM() propagates that same zeroAll() failure.
    pdt.state( MagAOX::app::stateCodes::READY );
    REQUIRE( pdt.baseReleaseDM() == -1 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::ERROR );
}

/// Verify that zeroAll() trivially succeeds when no shmimName is configured.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm zeroAll with no shmimName configured", "[dev::dm]" )
{
    dm_tests::dmTest pdtEmpty;
    mx::app::writeConfigFile( "/tmp/dm_test_zeroall_empty.conf", { "none" }, { "nada" }, { "0" } );
    mx::app::appConfigurator configEmpty;
    REQUIRE( pdtEmpty.setupConfig( configEmpty ) == 0 );
    configEmpty.readConfig( "/tmp/dm_test_zeroall_empty.conf" );
    REQUIRE( pdtEmpty.loadConfig( configEmpty ) == 0 );
    REQUIRE( pdtEmpty.zeroAll() == 0 );

    // m_shmimFlat and m_shmimTest are also empty when no shmimName is configured. setFlat()
    // and setTest() both trivially succeed without touching any real channel.
    REQUIRE( pdtEmpty.setFlat() == 0 );
    REQUIRE( pdtEmpty.setTest() == 0 );
    REQUIRE( pdtEmpty.zeroTest() == 0 );

    // clearSat() has its own trivial success branch for an empty name or zero size. This is
    // distinct from the early return in zeroAll() above, which never reaches clearSat() at all
    // when m_shmimName is empty.
    REQUIRE( pdtEmpty.clearSat() == 0 );
}

/// Verify the baseInitDM() and baseReleaseDM() state machine transitions and error paths. The
/// harness forces initDM() and releaseDM() to fail through m_initDMRV and m_releaseDMRV.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm baseInitDM and baseReleaseDM", "[dev::dm]" )
{
    mx::app::writeConfigFile( "/tmp/dm_test_baseinit.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_baseinit.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // appStartup() creates the INDI properties that the callbacks below are checked against.
    // It also starts the saturation thread, which idles because nothing is allocated. The
    // harness destructor stops it.
    REQUIRE( pdt.appStartup() == 0 );

    // Wrong state. The app is UNINITIALIZED, not NOTHOMED.
    REQUIRE( pdt.baseInitDM() == -1 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::ERROR );

    pdt.state( MagAOX::app::stateCodes::NOTHOMED );
    REQUIRE( pdt.baseInitDM() == 0 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::HOMING );

    // baseInitDM() propagates an initDM() failure.
    pdt.state( MagAOX::app::stateCodes::NOTHOMED );
    pdt.m_initDMRV = -1;
    REQUIRE( pdt.baseInitDM() == -1 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::ERROR );
    pdt.m_initDMRV = 0;

    // baseReleaseDM() propagates a releaseDM() failure.
    pdt.state( MagAOX::app::stateCodes::READY );
    pdt.m_releaseDMRV = -1;
    REQUIRE( pdt.baseReleaseDM() == -1 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::ERROR );
    pdt.m_releaseDMRV = 0;

    // baseReleaseDM() from a state other than POWEROFF moves to NOTHOMED, then calls
    // releaseDM() and zeroAll().
    pdt.state( MagAOX::app::stateCodes::READY );
    REQUIRE( pdt.baseReleaseDM() == 0 );

    // INDI callbacks for init, zero, and release, driven with hand-built properties.
    const std::string dev = pdt.configName();
    REQUIRE( pdt.newCallBack_init( switchProperty( dev, "initDM", { { "request", swOff } } ) ) == 0 );

    pcf::IndiProperty ipInitOn = switchProperty( dev, "initDM", { { "request", swOn } } );
    pdt.state( MagAOX::app::stateCodes::NOTHOMED );
    REQUIRE( pdt.newCallBack_init( ipInitOn ) == 0 );
    REQUIRE( pdt.state() == MagAOX::app::stateCodes::HOMING );

    // newCallBack_init() propagates an initDM() failure through baseInitDM().
    pdt.state( MagAOX::app::stateCodes::NOTHOMED );
    pdt.m_initDMRV = -1;
    REQUIRE( pdt.newCallBack_init( ipInitOn ) == -1 );
    pdt.m_initDMRV = 0;

    REQUIRE( pdt.newCallBack_zero( switchProperty( dev, "zeroDM", { { "request", swOn } } ) ) == 0 );

    REQUIRE( pdt.newCallBack_release( switchProperty( dev, "releaseDM", { { "request", swOn } } ) ) == 0 );

    // Wrong device and property name errors.
    pcf::IndiProperty ipWrong = switchProperty( "somethingelse", "notinit" );
    REQUIRE( pdt.newCallBack_init( ipWrong ) == -1 );
    REQUIRE( pdt.newCallBack_zero( ipWrong ) == -1 );
    REQUIRE( pdt.newCallBack_release( ipWrong ) == -1 );
    REQUIRE( pdt.newCallBack_zeroAll( ipWrong ) == -1 );
}

/// Verify that onPowerOff() and whilePowerOff() succeed on a configured harness.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm onPowerOff and whilePowerOff", "[dev::dm]" )
{
    mx::app::writeConfigFile( "/tmp/dm_test_poweroff.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_poweroff.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    REQUIRE( pdt.onPowerOff() == 0 );
    REQUIRE( pdt.whilePowerOff() == 0 );
}

/// Verify the full appStartup(), appLogic(), and appShutdown() lifecycle. This includes the
/// real saturation processing thread, processImage(), intervalSatTrip(), and updateINDI().
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm full lifecycle with saturation thread", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    mx::improc::milkImage<float> ch0;
    ch0.create( "dmlc00", 4, 4 );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/lifecycle" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmlc" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "satAvgInt" );
    v.push_back( "0" );
    s.push_back( "dm" );
    k.push_back( "percThreshold" );
    v.push_back( "0" );
    s.push_back( "dm" );
    k.push_back( "intervalSatThreshold" );
    v.push_back( "0" );
    s.push_back( "dm" );
    k.push_back( "intervalSatCountThreshold" );
    v.push_back( "1" );
    s.push_back( "dm" );
    k.push_back( "satTriggerDevice" );
    v.push_back( "someDevice" );
    s.push_back( "dm" );
    k.push_back( "satTriggerProperty" );
    v.push_back( "someProperty" );
    // Exercise the loadFlat("default") and loadTest("default") calls in appStartup(). The
    // targets do not need to exist. loadFlat() and loadTest() safely log and return -1 on a
    // missing file, and appStartup() does not check their return values.
    s.push_back( "dm" );
    k.push_back( "flatDefault" );
    v.push_back( "nosuchflat" );
    s.push_back( "dm" );
    k.push_back( "testDefault" );
    v.push_back( "nosuchtest" );

    mx::app::writeConfigFile( "/tmp/dm_test_lifecycle.conf", s, k, v );

    mx::app::appConfigurator config;

    dm_tests::dmTest pdt;

    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_lifecycle.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    pdt.setSize( 4, 4, IMAGESTRUCT_FLOAT );
    REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == 0 );

    REQUIRE( pdt.appStartup() == 0 );

    // Let the saturation thread reach steady state.
    mx::sys::milliSleep( 300 );

    // First without a driver, which is the m_indiDriver == nullptr branch.
    REQUIRE( pdt.updateINDI() == 0 );

    // Then with a real driver without FIFOs, so the m_indiDriver != nullptr branch of
    // updateINDI() is exercised too. Both branches trivially return 0.
    pdt.setupRealDriver();
    REQUIRE( pdt.updateINDI() == 0 );
    REQUIRE( pdt.appLogic() == 0 );

    // Drive processImage() and commandDM() to saturate every actuator and trip the interval
    // trigger.
    pdt.m_testSatValue = 1;
    float srcbuf[16]   = { 0 };
    for( int i = 0; i < 10; ++i )
    {
        REQUIRE( pdt.processImage( srcbuf, MagAOX::app::dev::shmimT() ) == 0 );
        mx::sys::milliSleep( 50 );
    }

    // Give the saturation thread time to compute statistics and set m_intervalSatTrip.
    mx::sys::milliSleep( 500 );

    // This should now call intervalSatTrip(). It tries to send an INDI property to
    // "someDevice" and safely fails. The failure is caught internally.
    REQUIRE( pdt.appLogic() == 0 );

    // Verify that the saturation shmims were created and updated. Use CHECK rather than
    // REQUIRE so that appShutdown() below always runs and cleanly joins the saturation
    // thread, even if this timing-sensitive check should ever fail.
    mx::improc::milkImage<uint8_t> satChan;
    bool                           opened = false;
    try
    {
        satChan.open( pdt.shmimSat() );
        opened = true;
    }
    catch( ... )
    {
    }
    CHECK( opened == true );

    // Set the shutdown flag first, as execute() has done by the time it calls appShutdown().
    // The thread loop then exits whether or not the SIGUSR1 interrupt lands inside its wait.
    pdt.requestShutdown();
    REQUIRE( pdt.appShutdown() == 0 );
}

/// Verify two branches of satThreadExec(). The first keeps accumulating during the averaging
/// interval. A nonzero dm.satAvgInt means several semaphore posts arrive before the interval
/// elapses. The second resets the exceeds counter, because the accumulated saturation stays
/// below dm.percThreshold. A real background thread is used, the same way the full lifecycle
/// test above does.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm satThreadExec averaging interval and reset branches", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    mx::improc::milkImage<float> ch0;
    ch0.create( "dmsat200", 2, 2 );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmsat2" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "2" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "2" );
    // The interval is wide enough that several posts below, spaced about 50 ms apart, arrive
    // before it elapses.
    s.push_back( "dm" );
    k.push_back( "satAvgInt" );
    v.push_back( "300" );
    // The threshold is above the always-zero instSatMap value used below, so the accumulated
    // saturation never counts as over threshold.
    s.push_back( "dm" );
    k.push_back( "percThreshold" );
    v.push_back( "1" );
    s.push_back( "dm" );
    k.push_back( "intervalSatThreshold" );
    v.push_back( "0" );
    s.push_back( "dm" );
    k.push_back( "intervalSatCountThreshold" );
    v.push_back( "1" );

    mx::app::writeConfigFile( "/tmp/dm_test_satavg.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_satavg.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    pdt.setSize( 2, 2, IMAGESTRUCT_FLOAT );
    REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == 0 );

    REQUIRE( pdt.appStartup() == 0 );
    // Wait comfortably past the worst-case startup handshake in threadStart(), which is about
    // 1 second. See dmPokeWFS_test.cpp for the same rationale. This guarantees that the
    // saturation thread is already waiting on its semaphore before posting starts.
    mx::sys::milliSleep( 1200 );

    // m_testSatValue defaults to 0. Every post below accumulates an all-zero instSatMap, so
    // the accumulated saturation never exceeds dm.percThreshold.
    float srcbuf[4] = { 0 };
    for( int i = 0; i < 8; ++i )
    {
        REQUIRE( pdt.processImage( srcbuf, MagAOX::app::dev::shmimT() ) == 0 );
        mx::sys::milliSleep( 50 );
    }

    // Give the thread time to complete at least one full averaging interval and reset.
    mx::sys::milliSleep( 300 );

    pdt.requestShutdown();
    REQUIRE( pdt.appShutdown() == 0 );
}

/// Verify the INDI static callback trampolines, the early-return branches for a missing
/// element, and the wrong-key branches of newCallBack_flats() and newCallBack_tests().
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm INDI static callback trampolines and edge branches", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/xc" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/xc/flats" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/xc/tests" );

    // One flat and one test file, so checkFlats() and checkTests() both register their
    // selection properties.
    mx::improc::eigenImage<float> flatX( 4, 4 );
    flatX.setConstant( 1 );
    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/xc/flats/flatX.fits", flatX );
    ff.write( "/tmp/dmtest_calibs/xc/tests/testX.fits", flatX );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/xc" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmxc" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_xc.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_xc.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // appStartup() creates the INDI properties and runs checkFlats() and checkTests(). It
    // also starts the saturation thread, which idles because nothing is allocated. The
    // harness destructor stops it.
    REQUIRE( pdt.appStartup() == 0 );

    // Put the DM in a state where zeroFlat() and zeroTest() will succeed, with real channels
    // to zero. They are invoked below through the toggle == Off path.
    pdt.state( MagAOX::app::stateCodes::READY );
    mx::improc::milkImage<float> flatChanXc, testChanXc;
    flatChanXc.create( "dmxc00", 4, 4 );
    testChanXc.create( "dmxc02", 4, 4 );

    typedef MAPPNS::dm<dm_tests::dmTest, float> dmT;

    const std::string dev = pdt.configName();

    // The static trampolines forward to the member callbacks.
    REQUIRE( dmT::st_newCallBack_init( &pdt, switchProperty( dev, "initDM", { { "request", swOff } } ) ) == 0 );
    REQUIRE( dmT::st_newCallBack_zero( &pdt, switchProperty( dev, "zeroDM", { { "request", swOff } } ) ) == 0 );
    REQUIRE( dmT::st_newCallBack_release( &pdt, switchProperty( dev, "releaseDM", { { "request", swOff } } ) ) == 0 );
    REQUIRE( dmT::st_newCallBack_zeroAll( &pdt, switchProperty( dev, "zeroAll", { { "request", swOff } } ) ) == 0 );

    pdt.state( MagAOX::app::stateCodes::READY );
    REQUIRE( dmT::st_newCallBack_setFlat( &pdt, switchProperty( dev, "flat_set", { { "toggle", swOff } } ) ) == 0 );
    REQUIRE( dmT::st_newCallBack_setTest( &pdt, switchProperty( dev, "test_set", { { "toggle", swOff } } ) ) == 0 );

    // checkFlats() and checkTests() selected the only file of each kind as current. Selecting
    // it again through the callback loads it.
    REQUIRE( dmT::st_newCallBack_flats( &pdt, switchProperty( dev, "flat", { { "flatX", swOn } } ) ) == 0 );
    REQUIRE( dmT::st_newCallBack_tests( &pdt, switchProperty( dev, "test", { { "testX", swOn } } ) ) == 0 );

    // Early-return branches for a missing element. The device and name match, but no elements
    // are added.
    REQUIRE( pdt.newCallBack_init( switchProperty( dev, "initDM" ) ) == 0 );
    REQUIRE( pdt.newCallBack_zero( switchProperty( dev, "zeroDM" ) ) == 0 );
    REQUIRE( pdt.newCallBack_release( switchProperty( dev, "releaseDM" ) ) == 0 );
    REQUIRE( pdt.newCallBack_zeroAll( switchProperty( dev, "zeroAll" ) ) == 0 );
    REQUIRE( pdt.newCallBack_setFlat( switchProperty( dev, "flat_set" ) ) == 0 );
    REQUIRE( pdt.newCallBack_setTest( switchProperty( dev, "test_set" ) ) == 0 );

    // Wrong-key branches of newCallBack_flats() and newCallBack_tests().
    pcf::IndiProperty ipWrong = switchProperty( "somethingelse", "notflats" );
    REQUIRE( pdt.newCallBack_flats( ipWrong ) == -1 );
    REQUIRE( pdt.newCallBack_tests( ipWrong ) == -1 );
}

/// Verify that processImage() propagates a commandDM() failure. The harness forces the failure
/// through m_commandDMFail.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm processImage error branches", "[dev::dm]" )
{
    mx::app::writeConfigFile( "/tmp/dm_test_procimg.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_procimg.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    float srcbuf[4] = { 0 };

    // commandDM() failure.
    pdt.m_commandDMFail = true;
    REQUIRE( pdt.processImage( srcbuf, MagAOX::app::dev::shmimT() ) == -1 );
    pdt.m_commandDMFail = false;
}

/// Verify that allocate() propagates a findDMChannels() failure when the size and type checks
/// pass but no channels exist for the configured shmimName.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm allocate propagates findDMChannels failure", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmxcnochan" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_xcnochan.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_xcnochan.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // loadConfig() unconditionally creates a "dmxcnochan_actmask" shmim because width and
    // height are set. The loose prefix search in findDMChannels() would otherwise pick that up
    // as a channel, along with the shape, delta, and diff streams created later. Remove every
    // shmim with that prefix so the branch for no channels found can be reached.
    for( const auto &entry : std::filesystem::directory_iterator( "/tmp/dmtest/shm" ) )
    {
        if( entry.path().filename().string().rfind( "dmxcnochan", 0 ) == 0 )
        {
            std::filesystem::remove( entry.path() );
        }
    }

    pdt.setSize( 4, 4, IMAGESTRUCT_FLOAT );
    REQUIRE( pdt.allocate( MagAOX::app::dev::shmimT() ) == -1 );
}

/// Verify that allocate() propagates an exception from creating the output shape shmim.
/// milkImage::create() throws a real exception when it cannot open the shm file for writing.
/// Nothing is mocked. The failure is forced by making the target directory unwritable, the
/// same way runCommand_test.cpp forces real operating system resource failures.
///
/// NOTE: this test must target "/tmp/dmtest/shm" specifically. The milk function
/// ImageStreamIO_filename() resolves its shm directory from MILK_SHM_DIR and caches it in a
/// function-local static the first time it is called in this process. An earlier test in this
/// file has already triggered that resolution against "/tmp/dmtest/shm". A later setenv() to
/// any other directory has no effect on where milkImage::create() writes. It silently still
/// uses "/tmp/dmtest/shm". This test previously set its own "/tmp/dmtest_shapefail/shm"
/// directory. Changing the permissions of that unrelated directory blocked nothing, so the
/// test passed for the wrong reason. findDMChannels() failed to find the channel that was
/// created in "/tmp/dmtest/shm", which is not the intended shmim creation exception.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm allocate propagates an output shape shmim creation exception", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    // Start from a clean set of files every run. A stale shmim with the same name left over
    // from a previous run would still be writable. create() would reuse it instead of needing
    // a fresh open() with O_CREAT, which is what the directory permission blocks. That would
    // silently defeat the fault injection below.
    std::filesystem::remove( "/tmp/dmtest/shm/dmshapefail00.im.shm" );
    std::filesystem::remove( "/tmp/dmtest/shm/dmshapefail_actmask.im.shm" );
    std::filesystem::remove( "/tmp/dmtest/shm/dmshapefail_shape.im.shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmshapefail" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_shapefail.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_shapefail.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // A real channel must exist for findDMChannels() to succeed before allocate() reaches
    // the output shmim creation this test targets.
    mx::improc::milkImage<float> chan0;
    chan0.create( "dmshapefail00", 4, 4 );

    // Now make the shm directory itself unwritable. milkImage::create() then fails with a real
    // EACCES from the operating system the next time it tries to create a new shmim.
    REQUIRE( chmod( "/tmp/dmtest/shm", 0555 ) == 0 );

    pdt.setSize( 4, 4, IMAGESTRUCT_FLOAT );
    int rv = pdt.allocate( MagAOX::app::dev::shmimT() );

    chmod( "/tmp/dmtest/shm", 0755 );

    REQUIRE( rv == -1 );
}

/// Verify the branch in appLogic() that detects that the saturation thread has exited. The
/// real thread is started with appStartup() and then told to shut down. It returns on its own
/// once it sees the flag, and appLogic() then reaps it with pthread_tryjoin_np().
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm appLogic saturation thread exited", "[dev::dm]" )
{
    mx::app::writeConfigFile( "/tmp/dm_test_threadexit.conf", { "none" }, { "nada" }, { "0" } );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_threadexit.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    REQUIRE( pdt.appStartup() == 0 );
    pdt.requestShutdown();

    // Nothing is allocated, so the thread sleeps in one second steps while it waits for
    // allocation. It checks the shutdown flag after each step and then returns.
    mx::sys::milliSleep( 2000 );

    REQUIRE( pdt.appLogic() == -1 );

    // appLogic() reaped the thread, but the std::thread object still looks joinable. Drop it
    // so the harness destructor does not try to join it again.
    pdt.abandonSatThread();
}

/// Verify the internal auto-load path of setFlat() when it is called before any explicit
/// loadFlat(). This test covers the success case. The failure case is the next test.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm setFlat internal auto-load", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/autoload" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/autoload/flats" );

    mx::improc::eigenImage<float> flatY( 5, 5 );
    flatY.setConstant( 4 );
    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/autoload/flats/flatY.fits", flatY );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/autoload" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmauto" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "5" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "5" );

    mx::app::writeConfigFile( "/tmp/dm_test_autoload.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_autoload.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // checkFlats() auto-selects the only flat as current, but does not load it into memory.
    REQUIRE( pdt.checkFlats() == 0 );
    REQUIRE( pdt.m_flatLoaded == false );
    REQUIRE( pdt.m_flatCurrent == "flatY" );

    std::remove( "/tmp/dmtest/shm/dmauto00.im.shm" );
    mx::improc::milkImage<float> flatChan;
    flatChan.create( "dmauto00", 5, 5 );

    pdt.state( MagAOX::app::stateCodes::READY );

    // setFlat() internally loads the current flat because it is not loaded yet.
    REQUIRE( pdt.setFlat() == 0 );
    REQUIRE( pdt.m_flatLoaded == true );
    REQUIRE( flatChan().sum() == 5 * 5 * 4 );
}

/// Verify the internal auto-load failure path of setFlat(). The on-demand loadFlat() call
/// inside setFlat() fails because the configured default flat file does not exist.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm setFlat internal auto-load failure", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    std::filesystem::remove_all( "/tmp/dmtest_calibs/autoloadfail" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/autoloadfail/flats" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "calibPath" );
    v.push_back( "/tmp/dmtest_calibs/autoloadfail" );
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmautofail" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "5" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "5" );
    s.push_back( "dm" );
    k.push_back( "flatDefault" );
    v.push_back( "missingdefault" );

    mx::app::writeConfigFile( "/tmp/dm_test_autoloadfail.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_autoloadfail.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );
    REQUIRE( pdt.m_flatCurrent == "default" );

    std::remove( "/tmp/dmtest/shm/dmautofail00.im.shm" );
    mx::improc::milkImage<float> flatChan;
    flatChan.create( "dmautofail00", 5, 5 );

    pdt.state( MagAOX::app::stateCodes::READY );

    // missingdefault.fits does not exist, so the internal loadFlat("default") call fails.
    REQUIRE( pdt.setFlat() == -1 );
    REQUIRE( pdt.m_flatLoaded == false );
}

/// Verify the zeroFlat() error branches for a wrong state, a missing channel, and a size
/// mismatch.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm zeroFlat error branches", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    // An empty m_shmimFlat, from no configured shmimName, trivially succeeds.
    {
        mx::app::writeConfigFile( "/tmp/dm_test_zf_empty.conf", { "none" }, { "nada" }, { "0" } );
        mx::app::appConfigurator config;
        dm_tests::dmTest         pdtEmpty;
        REQUIRE( pdtEmpty.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_zf_empty.conf" );
        REQUIRE( pdtEmpty.loadConfig( config ) == 0 );
        REQUIRE( pdtEmpty.zeroFlat() == 0 );
    }

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmzf" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "5" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "5" );

    mx::app::writeConfigFile( "/tmp/dm_test_zf.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_zf.conf" );
    REQUIRE( pdt.loadConfig( config ) == 0 );

    // Wrong state.
    REQUIRE( pdt.zeroFlat() == -1 );

    pdt.state( MagAOX::app::stateCodes::READY );

    // The channel does not exist.
    std::remove( "/tmp/dmtest/shm/dmzf00.im.shm" );
    REQUIRE( pdt.zeroFlat() == -1 );

    // Width mismatch. It is checked first.
    mx::improc::milkImage<float> flatChanBad;
    flatChanBad.create( "dmzf00", 3, 3 );
    REQUIRE( pdt.zeroFlat() == -1 );

    // The width now matches at 5 but the height is 3. This exercises the separate height
    // mismatch branch.
    flatChanBad.create( "dmzf00", 5, 3 );
    REQUIRE( pdt.zeroFlat() == -1 );

    // With a matching-size channel zeroFlat() proceeds. Force zeroDM() and clearSat()
    // failures. Both are logged but not fatal, so zeroFlat() itself still returns 0.
    flatChanBad.create( "dmzf00", 5, 5 );
    pdt.m_zeroDMRV = -1;
    REQUIRE( pdt.zeroFlat() == 0 );
    pdt.m_zeroDMRV = 0;

    // clearSat() fails on a real size mismatch between a sat channel and the DM.
    mx::improc::milkImage<uint8_t> satChanBad;
    satChanBad.create( pdt.shmimSat(), 3, 3 );
    REQUIRE( pdt.zeroFlat() == 0 );
}

/// Verify that loadConfig() propagates a real exception from creating the actuator mask
/// shmim. Nothing is mocked. The exception is forced by making the shm directory unwritable.
///
/// NOTE: the milk function ImageStreamIO_filename() resolves its shm directory from
/// MILK_SHM_DIR and caches it in a function-local static the first time it is called in a
/// process. Every later call reuses that first directory, no matter what MILK_SHM_DIR is set
/// to afterwards. Some earlier test in this same binary has already triggered that resolution
/// against "/tmp/dmtest/shm". This test must therefore use that same cached directory rather
/// than setting its own. The same applies to any other test in this file that needs a real
/// milkImage::create() to target a specific directory. Otherwise the fault-injecting chmod()
/// below would silently target an unrelated, still writable directory, and the real shmim
/// would be created in "/tmp/dmtest/shm" regardless.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm loadConfig propagates an actMask shmim creation exception", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    // Remove any stray file from a previous run of this test. Otherwise create() could reuse
    // a still writable file instead of needing a fresh open() with O_CREAT, which is what the
    // chmod below blocks.
    std::filesystem::remove( "/tmp/dmtest/shm/dmactmaskfail_actmask.im.shm" );

    std::vector<std::string> s, k, v;
    s.push_back( "dm" );
    k.push_back( "shmimName" );
    v.push_back( "dmactmaskfail" );
    s.push_back( "dm" );
    k.push_back( "width" );
    v.push_back( "4" );
    s.push_back( "dm" );
    k.push_back( "height" );
    v.push_back( "4" );

    mx::app::writeConfigFile( "/tmp/dm_test_actmaskfail.conf", s, k, v );

    mx::app::appConfigurator config;
    dm_tests::dmTest         pdt;
    REQUIRE( pdt.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/dm_test_actmaskfail.conf" );

    REQUIRE( chmod( "/tmp/dmtest/shm", 0555 ) == 0 );
    int rv = pdt.loadConfig( config );
    chmod( "/tmp/dmtest/shm", 0755 );

    REQUIRE( rv == -1 );
}

/// Verify how loadConfig() handles dm.testDefault and dm.actMaskPath. The default test name
/// has its path and extension stripped. A configured actuator mask file is loaded and
/// validated against dm.width and dm.height. Both the matching and the mismatched cases are
/// covered.
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dm testDefault and actMaskPath config handling", "[dev::dm]" )
{
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    mx::ioutils::createDirectories( "/tmp/dmtest_calibs/actmasktest" );

    mx::improc::eigenImage<float> maskGood( 5, 5 ), maskBad( 3, 3 );
    maskGood.setConstant( 1 );
    maskBad.setConstant( 1 );

    mx::fits::fitsFile<float> ff;
    ff.write( "/tmp/dmtest_calibs/actmasktest/maskGood.fits", maskGood );
    ff.write( "/tmp/dmtest_calibs/actmasktest/maskBad.fits", maskBad );

    SECTION( "testDefault is stripped to a bare name, and a matching actMaskPath loads" )
    {
        std::vector<std::string> s, k, v;
        s.push_back( "dm" );
        k.push_back( "calibPath" );
        v.push_back( "/tmp/dmtest_calibs/actmasktest" );
        s.push_back( "dm" );
        k.push_back( "width" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "height" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "shmimName" );
        v.push_back( "dmactmaskgood" );
        s.push_back( "dm" );
        k.push_back( "testDefault" );
        v.push_back( "/some/path/mytest.fits" );
        s.push_back( "dm" );
        k.push_back( "actMaskPath" );
        v.push_back( "maskGood.fits" );

        mx::app::writeConfigFile( "/tmp/dm_test_actmask_good.conf", s, k, v );

        mx::app::appConfigurator config;
        dm_tests::dmTest         pdt;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_actmask_good.conf" );
        REQUIRE( pdt.loadConfig( config ) == 0 );

        REQUIRE( pdt.testDefault() == "mytest" );
        REQUIRE( pdt.m_testCurrent == "default" );
    }

    SECTION( "a size-mismatched actMaskPath fails loadConfig" )
    {
        std::vector<std::string> s, k, v;
        s.push_back( "dm" );
        k.push_back( "calibPath" );
        v.push_back( "/tmp/dmtest_calibs/actmasktest" );
        s.push_back( "dm" );
        k.push_back( "width" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "height" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "shmimName" );
        v.push_back( "dmactmaskbad" );
        s.push_back( "dm" );
        k.push_back( "actMaskPath" );
        v.push_back( "maskBad.fits" );

        mx::app::writeConfigFile( "/tmp/dm_test_actmask_bad.conf", s, k, v );

        mx::app::appConfigurator config;
        dm_tests::dmTest         pdt;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_actmask_bad.conf" );
        REQUIRE( pdt.loadConfig( config ) == -1 );
    }

    SECTION( "a non-existent actMaskPath fails loadConfig" )
    {
        std::vector<std::string> s, k, v;
        s.push_back( "dm" );
        k.push_back( "calibPath" );
        v.push_back( "/tmp/dmtest_calibs/actmasktest" );
        s.push_back( "dm" );
        k.push_back( "width" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "height" );
        v.push_back( "5" );
        s.push_back( "dm" );
        k.push_back( "shmimName" );
        v.push_back( "dmactmasknone" );
        s.push_back( "dm" );
        k.push_back( "actMaskPath" );
        v.push_back( "doesNotExist.fits" );

        mx::app::writeConfigFile( "/tmp/dm_test_actmask_none.conf", s, k, v );

        mx::app::appConfigurator config;
        dm_tests::dmTest         pdt;
        REQUIRE( pdt.setupConfig( config ) == 0 );
        config.readConfig( "/tmp/dm_test_actmask_none.conf" );
        REQUIRE( pdt.loadConfig( config ) == -1 );
    }
}

