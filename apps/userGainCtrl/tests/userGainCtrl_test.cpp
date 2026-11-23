/** \file userGainCtrl_test.cpp
 * \brief Catch2 tests for the userGainCtrl app.
 *
 * \ingroup userGainCtrl_files
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../userGainCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup userGainCtrl_unit_test userGainCtrl Unit Tests
 * \brief Unit tests for the userGainCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `userGainCtrl` unit tests.
/** \ingroup userGainCtrl_unit_test
 */
namespace userGainCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class userGainCtrl_test : public userGainCtrl
{
    IMAGE_METADATA     m_metadata{}; ///< Counter and stream metadata for offline gain writes.
    std::vector<float> m_gains;      ///< In-memory stand-in for the gain stream pixels.

  public:
    /// Configure an offline instance and its INDI callback names.
    userGainCtrl_test( const std::string &device /**< [in] INDI device and instance name. */ );

    /// Attach a local buffer and prepare the per-instance sys directory.
    void setupGainStream( const std::filesystem::path &sysDir, /**< [in] Root of the test sys directory. */
                          const std::vector<float>    &gains /**< [in] Initial per-mode gains. */ );

    /// Inspect the gains written into the local stream buffer.
    const std::vector<float> &gains() const;

    /// Inspect the local stream publication count.
    uint64_t writes() const;

    using userGainCtrl::loadSavedGainVector;
    using userGainCtrl::savedGainVectorPath;
    using userGainCtrl::saveGainVector;
    using userGainCtrl::setSingleGain;
    using userGainCtrl::shmimMonitorTiming;
};

userGainCtrl_test::userGainCtrl_test( const std::string &device )
{
    m_configName = device;

    XWCTEST_SETUP_INDI_NEW_PROP( zeroAll );
    XWCTEST_SETUP_INDI_NEW_PROP( singleModeNo );
    XWCTEST_SETUP_INDI_NEW_PROP( singleGain );
    XWCTEST_SETUP_INDI_NEW_PROP( singleMC );
    XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_saveVector, save_vector );
    XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_loadSaved, load_saved );
}

void userGainCtrl_test::setupGainStream( const std::filesystem::path &sysDir, const std::vector<float> &gains )
{
    m_sysPath = sysDir.string();
    std::filesystem::create_directories( sysDir / m_configName );

    m_gains                                = gains;
    shmimMonitorT::m_width                 = gains.size();
    shmimMonitorT::m_height                = 1;
    shmimMonitorT::m_dataType              = _DATATYPE_FLOAT;
    shmimMonitorT::m_typeSize              = sizeof( float );
    shmimMonitorT::m_imageStream.md        = &m_metadata;
    shmimMonitorT::m_imageStream.array.raw = m_gains.data();
    m_gainsCurrent.resize( gains.size(), 1 );
    m_gainsTarget.resize( gains.size(), 1 );
    m_modeBlockStart = { 0 };
    m_modeBlockN     = { static_cast<uint16_t>( gains.size() ) };
    m_modeBlockGains.resize( 1 );
    m_modeBlockGainsConstant.resize( 1 );
    pixget = getPixPointer<realT>( shmimMonitorT::m_dataType );
    processImage( m_gains.data(), gainShmimT{} );
}

const std::vector<float> &userGainCtrl_test::gains() const
{
    return m_gains;
}

uint64_t userGainCtrl_test::writes() const
{
    return m_metadata.cnt0;
}

/// A temporary sys directory shared by consecutive offline instances.
class TestSysDirectory
{
  public:
    std::filesystem::path m_path; ///< Temporary sys directory root.

    /// Create a unique sys directory for this test.
    TestSysDirectory();

    /// Remove this test's sys directory and saved vectors.
    ~TestSysDirectory();
};

TestSysDirectory::TestSysDirectory()
{
    char path[] = "/tmp/userGainCtrl-test-XXXXXX";
    if( !::mkdtemp( path ) )
    {
        throw std::runtime_error( "cannot create temporary userGainCtrl sys directory" );
    }

    m_path = path;
}

TestSysDirectory::~TestSysDirectory()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}
/// \endcond

/// Verify the userGainCtrl callback validators accept only the expected properties.
/**
 * \ingroup userGainCtrl_unit_test
 */
SCENARIO( "INDI Callbacks", "[userGainCtrl]" )
{
    // clang-format off
    #ifdef USERGAINCTRL_TEST_DOXYGEN_REF
    userGainCtrl::newCallBack_m_indiP_zeroAll( pcf::IndiProperty() );
    userGainCtrl::newCallBack_m_indiP_singleModeNo( pcf::IndiProperty() );
    userGainCtrl::newCallBack_m_indiP_singleGain( pcf::IndiProperty() );
    userGainCtrl::newCallBack_m_indiP_singleMC( pcf::IndiProperty() );
    userGainCtrl::newCallBack_m_indiP_saveVector( pcf::IndiProperty() );
    userGainCtrl::newCallBack_m_indiP_loadSaved( pcf::IndiProperty() );
    userGainCtrl::newCallBack_blockGains( pcf::IndiProperty() );
    userGainCtrl::newCallBack_blockMCs( pcf::IndiProperty() );
    userGainCtrl::newCallBack_blockLimits( pcf::IndiProperty() );
    blockModes( *(std::vector<uint16_t> *)nullptr, *(std::vector<std::string> *)nullptr, 0, 0, false );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, zeroAll );
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleModeNo );
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleGain );
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleMC );
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_m_indiP_saveVector, save_vector );
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_m_indiP_loadSaved, load_saved );
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockGains, block00_gain );
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockMCs, block70_multcoeff );
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockLimits, block32_limit );
}

/// Save the full modal vector and load the most recent save from a new controller instance.
/** \ingroup userGainCtrl_unit_test */
TEST_CASE( "Saved gain vector survives restart", "[userGainCtrl]" )
{
    // clang-format off
    #ifdef USERGAINCTRL_TEST_DOXYGEN_REF
    userGainCtrl::saveGainVector();
    userGainCtrl::loadSavedGainVector();
    #endif
    // clang-format on

    TestSysDirectory         sys;
    const std::vector<float> original{ 0.025f, 0.34123457f, 1.125f };
    std::vector<float>       saved = original;
    saved[0]                       = 0.32751f;

    {
        userGainCtrl_test app( "testgainctrl" );
        app.setupGainStream( sys.m_path, original );
        REQUIRE( app.saveGainVector() == 0 );
        REQUIRE( app.writes() == 0 );
        REQUIRE( app.setSingleGain( saved[0] ) == 0 );
        REQUIRE( app.saveGainVector() == 0 );
        REQUIRE( app.writes() == 1 );
        REQUIRE( app.gains() == saved );
        REQUIRE( std::distance( std::filesystem::directory_iterator( sys.m_path / "testgainctrl" ),
                                std::filesystem::directory_iterator{} ) == 1 );

        REQUIRE( app.setSingleGain( std::numeric_limits<float>::infinity() ) == 0 );
        CHECK( app.saveGainVector() < 0 );
    }

    userGainCtrl_test app( "testgainctrl" );
    app.setupGainStream( sys.m_path, original );
    REQUIRE( app.gains() == original );
    REQUIRE( app.loadSavedGainVector() == 0 );
    CHECK( app.gains() == saved );
    CHECK( app.writes() == 1 );
}

/// Leave the active gain vector intact when the saved file is missing, malformed, or the wrong size.
/** \ingroup userGainCtrl_unit_test */
TEST_CASE( "Invalid saved gain vectors leave current gains intact", "[userGainCtrl]" )
{
    // clang-format off
    #ifdef USERGAINCTRL_TEST_DOXYGEN_REF
    userGainCtrl::loadSavedGainVector();
    userGainCtrl::shmimMonitorTiming( gainShmimT(), dev::shmimMonitorEvent::disconnected );
    #endif
    // clang-format on

    TestSysDirectory         sys;
    const std::vector<float> original{ 0.025f, 0.34123457f, 1.125f };
    userGainCtrl_test        app( "testgainctrl" );
    app.setupGainStream( sys.m_path, original );

    CHECK( app.loadSavedGainVector() < 0 );
    for( const char *contents :
         { "2 1\n0.1\n0.2\n", "3 1\n0.1\n0.2\n", "3 1\n0.1\nnan\n0.2\n", "3 1\n0.1\n0.2\n0.3\n0.4\n" } )
    {
        std::ofstream( app.savedGainVectorPath() ) << contents;
        CHECK( app.loadSavedGainVector() < 0 );
        CHECK( app.gains() == original );
        CHECK( app.writes() == 0 );
    }

    REQUIRE( app.saveGainVector() == 0 );
    app.shmimMonitorTiming( gainShmimT(), dev::shmimMonitorEvent::disconnected );
    CHECK( app.saveGainVector() < 0 );
    CHECK( app.loadSavedGainVector() < 0 );
    CHECK( app.gains() == original );
    CHECK( app.writes() == 0 );
}

/// Verify `blockModes()` partitions the requested modes into controller-sized blocks.
/**
 * \ingroup userGainCtrl_unit_test
 */
SCENARIO( "Calculating Blocks", "[userGainCtrl]" )
{
    GIVEN( "No Zernikes" )
    {
        int                      rv;
        std::vector<uint16_t>    blocks;
        std::vector<std::string> names;
        // A full block means that all the modes in the last block are present
        WHEN( "Full blocks" )
        {
            rv = blockModes( blocks, names, 24, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 1 );
            REQUIRE( blocks[0] == 24 );

            rv = blockModes( blocks, names, 80, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 2 );
            REQUIRE( blocks[0] == 24 );
            REQUIRE( blocks[1] == 56 );

            rv = blockModes( blocks, names, 168, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 3 );
            REQUIRE( blocks[0] == 24 );
            REQUIRE( blocks[1] == 56 );
            REQUIRE( blocks[2] == 88 );
        }

        // A partial block means that not all modes in last block are present
        WHEN( "Partial blocks" )
        {
            rv = blockModes( blocks, names, 25, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 2 );
            REQUIRE( blocks[0] == 24 );
            REQUIRE( blocks[1] == 1 );

            rv = blockModes( blocks, names, 85, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 3 );
            REQUIRE( blocks[0] == 24 );
            REQUIRE( blocks[1] == 56 );
            REQUIRE( blocks[2] == 5 );

            rv = blockModes( blocks, names, 287, 0, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 24 );
            REQUIRE( blocks[1] == 56 );
            REQUIRE( blocks[2] == 88 );
            REQUIRE( blocks[3] == 119 );
        }

        // Test when Zernikes cover various numbers of blocks
        WHEN( "Full blocks, Zernikes in 1 block, no split" )
        {
            rv = blockModes( blocks, names, 24, 10, false );
            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 7 );
            REQUIRE( blocks[3] == 14 );

            rv = blockModes( blocks, names, 80, 5, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 2 );
            REQUIRE( blocks[3] == 19 );
            REQUIRE( blocks[4] == 56 );

            rv = blockModes( blocks, names, 168, 23, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 20 );
            REQUIRE( blocks[3] == 1 );
            REQUIRE( blocks[4] == 56 );
            REQUIRE( blocks[5] == 88 );
        }

        WHEN( "Full blocks, Zernikes in 2 blocks, no split" )
        {
            rv = blockModes( blocks, names, 24, 25, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 3 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 22 );

            rv = blockModes( blocks, names, 80, 25, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 22 );
            REQUIRE( blocks[3] == 55 );

            rv = blockModes( blocks, names, 168, 79, false );
            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 76 );
            REQUIRE( blocks[3] == 1 );
            REQUIRE( blocks[4] == 88 );
        }

        WHEN( "Partial blocks, Zernikes in 1 block, no split" )
        {
            rv = blockModes( blocks, names, 25, 10, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 7 );
            REQUIRE( blocks[3] == 14 );
            REQUIRE( blocks[4] == 1 );

            rv = blockModes( blocks, names, 85, 5, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 2 );
            REQUIRE( blocks[3] == 19 );
            REQUIRE( blocks[4] == 56 );
            REQUIRE( blocks[5] == 5 );

            rv = blockModes( blocks, names, 287, 23, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 7 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 20 );
            REQUIRE( blocks[3] == 1 );
            REQUIRE( blocks[4] == 56 );
            REQUIRE( blocks[5] == 88 );
            REQUIRE( blocks[6] == 119 );
        }

        WHEN( "Partial blocks, Zernikes in 2 blocks, no split" )
        {
            rv = blockModes( blocks, names, 26, 25, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 22 );
            REQUIRE( blocks[3] == 1 );

            rv = blockModes( blocks, names, 85, 25, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 22 );
            REQUIRE( blocks[3] == 55 );
            REQUIRE( blocks[4] == 5 );

            rv = blockModes( blocks, names, 287, 79, false );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 2 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 76 );
            REQUIRE( blocks[3] == 1 );  // b = 1, 80
            REQUIRE( blocks[4] == 88 ); // b = 2, 168
            REQUIRE( blocks[5] == 119 );
        }

        WHEN( "Full blocks, Zernikes in 1 block, with split" )
        {
            rv = blockModes( blocks, names, 24, 10, true );
            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 7 );
            REQUIRE( blocks[4] == 14 );

            rv = blockModes( blocks, names, 80, 5, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 2 );
            REQUIRE( blocks[4] == 19 );
            REQUIRE( blocks[5] == 56 );

            rv = blockModes( blocks, names, 168, 23, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 7 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 20 );
            REQUIRE( blocks[4] == 1 );
            REQUIRE( blocks[5] == 56 );
            REQUIRE( blocks[6] == 88 );
        }

        WHEN( "Full blocks, Zernikes in 2 blocks, with split" )
        {
            rv = blockModes( blocks, names, 24, 25, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 22 );

            rv = blockModes( blocks, names, 80, 25, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 22 );
            REQUIRE( blocks[4] == 55 );

            rv = blockModes( blocks, names, 168, 79, true );
            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 76 );
            REQUIRE( blocks[4] == 1 );
            REQUIRE( blocks[5] == 88 );
        }

        WHEN( "Partial blocks, Zernikes in 1 block, with split" )
        {
            rv = blockModes( blocks, names, 25, 10, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 7 );
            REQUIRE( blocks[4] == 14 );
            REQUIRE( blocks[5] == 1 );

            rv = blockModes( blocks, names, 85, 5, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 7 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 2 );
            REQUIRE( blocks[4] == 19 );
            REQUIRE( blocks[5] == 56 );
            REQUIRE( blocks[6] == 5 );

            rv = blockModes( blocks, names, 287, 23, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 8 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 20 );
            REQUIRE( blocks[4] == 1 );
            REQUIRE( blocks[5] == 56 );
            REQUIRE( blocks[6] == 88 );
            REQUIRE( blocks[7] == 119 );
        }

        WHEN( "Partial blocks, Zernikes in 2 blocks, with split" )
        {
            rv = blockModes( blocks, names, 26, 25, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 22 );
            REQUIRE( blocks[4] == 1 );

            rv = blockModes( blocks, names, 85, 25, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 22 );
            REQUIRE( blocks[4] == 55 );
            REQUIRE( blocks[5] == 5 );

            rv = blockModes( blocks, names, 287, 79, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 7 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 76 );
            REQUIRE( blocks[4] == 1 );  // b = 1, 80
            REQUIRE( blocks[5] == 88 ); // b = 2, 168
            REQUIRE( blocks[6] == 119 );
        }

        // Test if it all generalizes to even more zernikes
        WHEN( "Partial blocks, Zernikes in 3 blocks, with split" )
        {
            rv = blockModes( blocks, names, 26, 81, true ); // this should give 81 modes and quit

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 4 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 78 );

            rv = blockModes( blocks, names, 85, 81, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 5 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 78 );
            REQUIRE( blocks[4] == 4 );

            rv = blockModes( blocks, names, 287, 100, true );

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 6 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 97 );
            REQUIRE( blocks[4] == 68 ); // b = 2, 168
            REQUIRE( blocks[5] == 119 );
        }

        WHEN( "The Full MagAO-X" )
        {
            rv = blockModes( blocks, names, 2400, 10, true ); // this should give 81 modes and quite

            REQUIRE( rv == 0 );
            REQUIRE( blocks.size() == 16 );
            REQUIRE( blocks[0] == 1 );
            REQUIRE( blocks[1] == 1 );
            REQUIRE( blocks[2] == 1 );
            REQUIRE( blocks[3] == 7 );
            REQUIRE( blocks[4] == 14 ); // b=0 remainder
            REQUIRE( blocks[5] == 56 ); // b=1
            REQUIRE( blocks[6] == 88 );
            REQUIRE( blocks[7] == 120 );
            REQUIRE( blocks[8] == 152 );
            REQUIRE( blocks[9] == 184 );
            REQUIRE( blocks[10] == 216 );
            REQUIRE( blocks[11] == 248 );
            REQUIRE( blocks[12] == 280 );
            REQUIRE( blocks[13] == 312 );
            REQUIRE( blocks[14] == 344 );
            REQUIRE( blocks[15] == 376 );
        }
    }
}

} // namespace userGainCtrlTest

} // namespace libXWCTest
