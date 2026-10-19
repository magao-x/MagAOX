/** \file dm_test.cpp
 * \brief Unit tests for generic deformable mirror configuration and channel handling.
 */
#include "../../../../tests/testXWC.hpp"

#include <mx/sys/timeUtils.hpp>

#include "dm_test.hpp"

/// \cond DM_TEST_HARNESS
namespace
{

/// Own local shared-memory streams used by one channel-clearing test.
struct dmClearStreams
{
    std::string m_prefix; ///< Unique main-channel prefix for this fixture.

    std::vector<IMAGE> m_images; ///< Owned streams removed when the fixture leaves scope.

    /// Create a local stream namespace, reusing ImageStreamIO's process-wide shared-memory directory.
    dmClearStreams();

    /// Remove every stream created by the fixture.
    ~dmClearStreams();

    /// Create a numbered main or auxiliary stream with nonzero pixels.
    IMAGE &add( const std::string &suffix /**< [in] Suffix appended to the fixture prefix. */,
                uint32_t           width  = 4 /**< [in] Stream width. */,
                uint32_t           height = 3 /**< [in] Stream height. */,
                uint8_t            type   = _DATATYPE_FLOAT /**< [in] ImageStreamIO data type. */,
                uint32_t           depth  = 0 /**< [in] Nonzero depth creates a three-axis image. */ );
};

dmClearStreams::dmClearStreams()
{
    static unsigned fixture = 0;
    m_prefix                = "dmclear" + std::to_string( getpid() ) + "x" + std::to_string( ++fixture ) + "disp";
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );
    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );
    m_images.reserve( 8 );
}

dmClearStreams::~dmClearStreams()
{
    for( auto &image : m_images )
    {
        if( image.md != nullptr )
        {
            ImageStreamIO_destroyIm( &image );
        }
    }
}

IMAGE &dmClearStreams::add( const std::string &suffix, uint32_t width, uint32_t height, uint8_t type, uint32_t depth )
{
    m_images.emplace_back();
    IMAGE   &image   = m_images.back();
    uint32_t size[3] = { width, height, depth };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &image, ( m_prefix + suffix ).c_str(), depth == 0 ? 2 : 3, size, type, -1, 1, 10, 0, 0, 0 ) ==
             IMAGESTREAMIO_SUCCESS );
    memset( image.array.raw, 1, image.md->nelement * ImageStreamIO_typesize( type ) );
    ImageStreamIO_semflush( &image, -1 );
    return image;
}

} // namespace
/// \endcond

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

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::calibPath();
#endif
    // clang-format on
}

/// Test dmcomb detection, configuration, and manipulation
/**
 * \ingroup dm_tests
 */
TEST_CASE( "Test dmcomb detection, configuration, and manipulation", "[dev::dm]" )
{
    mx::ioutils::createDirectories( "/tmp/dmtest/shm" );

    setenv( "MILK_SHM_DIR", "/tmp/dmtest/shm", 1 );

    mx::improc::milkImage<float> ch0, ch1, ch2, ch3, ch4, chT;
    try
    {

        ch0.create( "dmtest00", 50, 50 );
        ch0().setConstant( 1 );

        ch1.create( "dmtest01", 50, 50 );
        ch1().setConstant( 2 );

        ch2.create( "dmtest02", 50, 50 );
        ch2().setConstant( 3 );

        ch3.create( "dmtest03", 50, 50 );
        ch3().setConstant( 4 );

        ch4.create( "dmtest04", 50, 50 );
        ch4().setConstant( 5 );

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
    REQUIRE( nd == 2 );

    const std::vector<size_t> &notDeltas = pdt.notDeltas();
    REQUIRE( notDeltas.size() == 3 );
    REQUIRE( notDeltas[0] == 0 );
    REQUIRE( notDeltas[1] == 1 );
    REQUIRE( notDeltas[2] == 4 );

    mx::improc::milkImage<float> outputShape;
    bool                         pass = false;
    try
    {
        outputShape.open( "dmtest_shape" );
        pass = true;
    }
    catch( ... )
    {
    }

    REQUIRE( pass == true );
    REQUIRE( outputShape.rows() == 50 );
    REQUIRE( outputShape.cols() == 50 );
    REQUIRE( outputShape().sum() == 0 );

    outputShape = chT;
    float sum   = outputShape().sum();
    REQUIRE( sum == ( 50 * 50 ) * ( 1 + 2 + 3 + 4 + 5 ) );

    mx::improc::milkImage<float> outputDelta;
    pass = false;
    try
    {
        outputDelta.open( "dmtest_delta" );
        pass = true;
    }
    catch( ... )
    {
    }

    REQUIRE( pass == true );
    REQUIRE( outputDelta.rows() == 50 );
    REQUIRE( outputDelta.cols() == 50 );
    REQUIRE( outputDelta().sum() == 0 );

    rows = pdt.totalFlat().rows();
    REQUIRE( rows == 50 );

    cols = pdt.totalFlat().cols();
    REQUIRE( cols == 50 );

    sum = pdt.totalFlat().sum();
    REQUIRE( sum == 0 );

    pdt.makeDelta();

    sum = pdt.totalFlat().sum();
    REQUIRE( sum == ( 50 * 50 ) * ( 1 + 2 + 5 ) );

    // Check that it's still the same
    sum = outputShape().sum();
    REQUIRE( sum == ( 50 * 50 ) * ( 1 + 2 + 3 + 4 + 5 ) );

    sum = outputDelta().sum();
    REQUIRE( sum == ( 50 * 50 ) * ( 3 + 4 ) );

    // clang-format off
    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::findDMChannels();
    MagAOX::app::dev::dm::allocate();
    MagAOX::app::dev::dm::makeDelta();
    #endif
    // clang-format on
}

/// Verify optional auxiliary clearing without changing numbered-channel discovery or notifications.
/** \ingroup dm_tests
 */
TEST_CASE( "Zero DM main channels and optional auxiliary sources", "[dev::dm]" )
{
    dmClearStreams streams;
    auto          &first           = streams.add( "00" );
    auto          &last            = streams.add( "01" );
    auto          &aux             = streams.add( "00_aux" );
    auto          &unmapped        = streams.add( "02_aux" );
    uint64_t       firstCounter    = first.md->cnt0;
    uint64_t       lastCounter     = last.md->cnt0;
    uint64_t       auxCounter      = aux.md->cnt0;
    uint64_t       unmappedCounter = unmapped.md->cnt0;

    dm_tests::dmTest controller( "xx", false );
    controller.setChannels( streams.m_prefix, 4, 3, 0 );
    REQUIRE( controller.findDMChannels() == 0 );
    REQUIRE( controller.numChannels() == 2 );

    bool suppress = GENERATE( false, true );
    REQUIRE( controller.zeroAll( suppress ) == 0 );
    for( size_t n = 0; n < 12; ++n )
    {
        REQUIRE( first.array.F[n] == 0 );
        REQUIRE( last.array.F[n] == 0 );
        REQUIRE( aux.array.F[n] == 0 );
        REQUIRE( unmapped.array.F[n] != 0 );
    }
    REQUIRE( first.md->cnt0 == firstCounter + 1 );
    REQUIRE( last.md->cnt0 == lastCounter + 1 );
    REQUIRE( aux.md->cnt0 == auxCounter + 1 );
    REQUIRE( unmapped.md->cnt0 == unmappedCounter );
    REQUIRE( aux.md->write == 0 );
    REQUIRE( aux.md->writetime.tv_sec != 0 );
    REQUIRE( ImageStreamIO_semvalue( &first, 0 ) == 0 );
    REQUIRE( ImageStreamIO_semvalue( &aux, 0 ) == 0 );
    REQUIRE( ImageStreamIO_semvalue( &last, 0 ) == ( suppress ? 0 : 1 ) );

    // clang-format off
    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::findDMChannels();
    MagAOX::app::dev::dm::zeroAll();
    #endif
    // clang-format on
}

/// Reject incompatible auxiliary sources before modifying their data or counters.
/** \ingroup dm_tests
 */
TEST_CASE( "Reject incompatible DM auxiliary sources during clear", "[dev::dm]" )
{
    dmClearStreams streams;
    auto          &main  = streams.add( "00" );
    uint32_t       width = 4, height = 3, depth = 0;
    uint8_t        type = _DATATYPE_FLOAT;
    SECTION( "wrong width" )
    {
        width = 2;
    }
    SECTION( "wrong height" )
    {
        height = 2;
    }
    SECTION( "wrong data type" )
    {
        type = _DATATYPE_UINT8;
    }
    SECTION( "cube instead of a two-dimensional source" )
    {
        depth = 2;
    }
    auto             &aux   = streams.add( "00_aux", width, height, type, depth );
    size_t            bytes = aux.md->nelement * ImageStreamIO_typesize( type );
    std::vector<char> original( bytes );
    memcpy( original.data(), aux.array.raw, bytes );
    uint64_t counter = aux.md->cnt0;

    dm_tests::dmTest controller( "xx", false );
    controller.setChannels( streams.m_prefix, 4, 3, 1 );
    REQUIRE( controller.zeroAll() < 0 );
    REQUIRE( memcmp( original.data(), aux.array.raw, bytes ) == 0 );
    REQUIRE( aux.md->cnt0 == counter );
    REQUIRE( ImageStreamIO_semvalue( &aux, 0 ) == 0 );
    for( size_t n = 0; n < 12; ++n )
    {
        REQUIRE( main.array.F[n] == 0 );
    }
    REQUIRE( ImageStreamIO_semvalue( &main, 0 ) == 1 );

    // clang-format off
    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::zeroAll();
    #endif
    // clang-format on
}

/// Verify that release clears sources added after main-channel discovery without posting a DM trigger.
/** \ingroup dm_tests
 */
TEST_CASE( "DM release clears late-created auxiliary sources", "[dev::dm]" )
{
    dmClearStreams   streams;
    auto            &main = streams.add( "00" );
    dm_tests::dmTest controller( "xx", false );
    controller.setChannels( streams.m_prefix, 4, 3, 0 );
    REQUIRE( controller.findDMChannels() == 0 );
    auto &aux = streams.add( "00_aux" );
    controller.state( MagAOX::app::stateCodes::OPERATING );

    REQUIRE( controller.baseReleaseDM() == 0 );
    REQUIRE( controller.m_releaseCount == 1 );
    REQUIRE( controller.state() == MagAOX::app::stateCodes::NOTHOMED );
    for( size_t n = 0; n < 12; ++n )
    {
        REQUIRE( main.array.F[n] == 0 );
        REQUIRE( aux.array.F[n] == 0 );
    }
    REQUIRE( ImageStreamIO_semvalue( &main, 0 ) == 0 );
    REQUIRE( ImageStreamIO_semvalue( &aux, 0 ) == 0 );

    // clang-format off
    #ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::baseReleaseDM();
    MagAOX::app::dev::dm::zeroAll();
    #endif
    // clang-format on
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

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::appStartup();
    MagAOX::app::dev::dm::appLogic();
    MagAOX::app::dev::dm::appShutdown();
    #endif
    // clang-format on
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

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::dm::setupConfig();
    MagAOX::app::dev::dm::loadConfig();
    MagAOX::app::dev::dm::appStartup();
    #endif
    // clang-format on
}

#endif
