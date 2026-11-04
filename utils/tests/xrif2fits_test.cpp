/** \file xrif2fits_test.cpp
 * \brief Catch2 tests for the xrif2fits utility.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xrif2fits_files
 */

#include "../../tests/testXWC.hpp"

#include "../xrif2fits/xrif2fits.hpp"

#include <filesystem>
#include <string>
#include <unistd.h>

namespace libXWCTest
{

/** \defgroup xrif2fits_unit_test xrif2fits Unit Tests
 * \brief Unit tests for the xrif2fits utility.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xrif2fits` unit tests.
/** \ingroup xrif2fits_unit_test
 */
namespace xrif2fitsTest
{

/// Test harness exposing selected protected helpers.
/** \cond xrif2fits_test_harness
 */
class xrif2fitsHarness : public xrif2fits
{
  public:
    using xrif2fits::hasTelemetry;
    using xrif2fits::validateMetaSourceDir;

    /// Load telemetry maps for a synthetic archive interval.
    mx::error_t loadTelemetryMaps( const std::vector<std::string> &dirs /**< [in] telemetry source directories */ )
    {
        MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> firstFile(
            "cam1/2024_11_19/cam1_20241119030000000000000.xrif" );
        MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> lastFile(
            "cam1/2024_11_19/cam1_20241119040000000000000.xrif" );

        return loadMetaFileMaps( m_tels, dirs, "cam1", ".bintel", firstFile, lastFile, "telemetry" );
    }
};
/** \endcond
 */

/// Remove a temporary directory tree when a test section exits.
/** \cond xrif2fits_test_harness
 */
struct tempTree
{
    std::filesystem::path m_path; ///< Temporary directory path owned by this fixture.

    /// Construct a unique temporary directory path for this test process.
    tempTree()
    {
        m_path = std::filesystem::temp_directory_path() /
                 ( "xrif2fits_test_" + std::to_string( static_cast<long long>( getpid() ) ) );
        std::filesystem::remove_all( m_path );
    }

    /// Remove the temporary directory tree.
    ~tempTree()
    {
        std::filesystem::remove_all( m_path );
    }
};
/** \endcond
 */

/// Verify metadata source validation distinguishes missing directories from empty date ranges.
/**
 * \ingroup xrif2fits_unit_test
 */
TEST_CASE( "xrif2fits metadata source validation handles missing and empty directories", "[xrif2fits]" )
{
    // clang-format off
    #ifdef XRIF2FITS_TEST_DOXYGEN_REF
    xrif2fits::validateMetaSourceDir();
    xrif2fits::loadMetaFileMaps();
    #endif
    // clang-format on

    tempTree         tmp;
    xrif2fitsHarness app;

    SECTION( "missing configured source directory is fatal" )
    {
        mx::error_t errc = app.validateMetaSourceDir( ( tmp.m_path / "missing" ).string(), "cam1", "telemetry" );

        REQUIRE( errc == mx::error_t::dirnotfound );
    }

    SECTION( "missing app subdirectory is fatal" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetry" );

        mx::error_t errc = app.validateMetaSourceDir( ( tmp.m_path / "telemetry" ).string(), "cam1", "telemetry" );

        REQUIRE( errc == mx::error_t::dirnotfound );
    }

    SECTION( "missing app subdirectory across all source directories is fatal" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetryLocal" );
        std::filesystem::create_directories( tmp.m_path / "telemetryNfs" );

        mx::error_t errc = app.loadTelemetryMaps(
            { ( tmp.m_path / "telemetryLocal" ).string(), ( tmp.m_path / "telemetryNfs" ).string() } );

        REQUIRE( errc == mx::error_t::dirnotfound );
    }

    SECTION( "app subdirectory only needs to exist in one source directory" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetryLocal" );
        std::filesystem::create_directories( tmp.m_path / "telemetryNfs" / "cam1" );

        mx::error_t errc = app.loadTelemetryMaps(
            { ( tmp.m_path / "telemetryLocal" ).string(), ( tmp.m_path / "telemetryNfs" ).string() } );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( app.hasTelemetry( "cam1" ) == false );
    }

    SECTION( "empty source entries are ignored when explicit source directories are present" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetryNfs" / "cam1" );

        mx::error_t errc = app.loadTelemetryMaps( { "", ( tmp.m_path / "telemetryNfs" ).string() } );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( app.hasTelemetry( "cam1" ) == false );
    }

    SECTION( "default dot entries are ignored when explicit source directories are present" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetryNfs" / "cam1" );

        mx::error_t errc = app.loadTelemetryMaps( { ".", ( tmp.m_path / "telemetryNfs" ).string() } );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( app.hasTelemetry( "cam1" ) == false );
    }

    SECTION( "existing app directory with no matching files is non-fatal" )
    {
        std::filesystem::create_directories( tmp.m_path / "telemetry" / "cam1" );

        mx::error_t errc = app.loadTelemetryMaps( { ( tmp.m_path / "telemetry" ).string() } );

        REQUIRE( errc == mx::error_t::noerror );
        REQUIRE( app.hasTelemetry( "cam1" ) == false );
    }
}

/// Verify unavailable log metadata is represented with the requested FITS-card value.
/**
 * \ingroup xrif2fits_unit_test
 */
TEST_CASE( "xrif2fits unavailable metadata cards use NOT AVAILABLE", "[xrif2fits]" )
{
    // clang-format off
    #ifdef XRIF2FITS_TEST_DOXYGEN_REF
    MagAOX::logger::logMeta::unavailableCard();
    MagAOX::logger::logMeta::unavailableValue();
    #endif
    // clang-format on

    MagAOX::logger::logMeta exptimeMeta(
        MagAOX::logger::logMetaSpec( "cam1", MagAOX::logger::telem_stdcam::eventCode, "exptime" ) );

    mx::fits::fitsHeaderCard<MagAOX::logger::logMeta::verboseT> card = exptimeMeta.unavailableCard();
    mx::error_t                                                 errc;

    REQUIRE( MagAOX::logger::logMeta::unavailableValue() == "NOT AVAILABLE" );
    REQUIRE( card.keyword() == "cam1 EXPTIME" );
    REQUIRE( card.value<std::string>( &errc ) == MagAOX::logger::logMeta::unavailableValue() );
    REQUIRE( errc == mx::error_t::noerror );
}

/// Verify empty loaded-log buffers fail lookup without dereferencing invalid memory.
/**
 * \ingroup xrif2fits_unit_test
 */
TEST_CASE( "xrif2fits log metadata lookup handles empty loaded buffers", "[xrif2fits]" )
{
    // clang-format off
    #ifdef XRIF2FITS_TEST_DOXYGEN_REF
    MagAOX::logger::logMap<>::getPriorLog();
    #endif
    // clang-format on

    MagAOX::logger::logMap<>                         logMap;
    MagAOX::file::stdFileName<XWC_DEFAULT_VERBOSITY> logFile( "cam1/1970_01_01/cam1_19700101000000000000000.bintel" );

    logMap.m_appToFileMap["cam1"].insert( logFile );

    char                *prior = nullptr;
    flatlogs::timespecX  ts{ 10, 0 };
    flatlogs::eventCodeT ev = MagAOX::logger::telem_stdcam::eventCode;
    int                  rv = logMap.getPriorLog( prior, "cam1", ev, ts );

    REQUIRE( rv == -1 );
    REQUIRE( prior == nullptr );
}

} // namespace xrif2fitsTest

} // namespace libXWCTest
