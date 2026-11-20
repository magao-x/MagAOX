/** \file autoJayCalibration_test.cpp
 * \brief Offline calibration persistence and validation tests for autoJay.
 * \ingroup autoJay_files
 */

#include "../../../tests/testXWC.hpp"
#include "../autoJayCalibration.hpp"

namespace libXWCTest
{
namespace autoJayTest
{

using namespace MagAOX::app::autoJayDetail;

/// \cond AUTOJAY_CALIBRATION_TEST_HELPERS
/// Own one unique scratch directory and remove it when the test exits.
class calibrationDirectory
{
  public:
    std::filesystem::path m_path; ///< Directory owned by this fixture.

    /// Allocate a directory under /tmp without touching operational calibration files.
    calibrationDirectory();

    /// Remove this fixture's files without throwing during assertion unwinding.
    ~calibrationDirectory();
};

/// Create a finite two-DM calibration with a common binned reference ROI.
calibration sampleCalibration();

/// Replace one exact manifest fragment, requiring that it existed.
void changeManifest( const std::filesystem::path &path /**< [in] manifest filename */,
                     const std::string           &from /**< [in] original fragment */,
                     const std::string           &to /**< [in] replacement fragment */ );

calibrationDirectory::calibrationDirectory()
{
    char directory[] = "/tmp/autoJay-calibration-XXXXXX";
    auto created     = ::mkdtemp( directory );
    REQUIRE( created != nullptr );
    m_path = created;
}

calibrationDirectory::~calibrationDirectory()
{
    std::error_code ignored;
    std::filesystem::remove_all( m_path, ignored );
}

calibration sampleCalibration()
{
    calibration value;
    value.profile                    = "flowfs-pupil";
    value.camera                     = "camflowfs";
    value.stream                     = "camflowfs";
    value.cameraGeometry.region      = { 30.5F, 40.5F, 12, 8, 2, 2 };
    value.cameraGeometry.units       = MagAOX::roi::units::sensor;
    value.cameraGeometry.orientation = "flipNone";
    value.baseline                   = imageT::Constant( 6, 4, 50 );
    value.dm[0].name                 = "dmncpc";
    value.dm[0].channel              = "dm02disp02";
    value.dm[0].trigger              = "dm02disp";
    value.dm[1].name                 = "dmtweeter";
    value.dm[1].channel              = "dm01disp02";
    value.dm[1].trigger              = "dm01disp";
    for( size_t i = 0; i < value.dm.size(); ++i )
    {
        value.dm[i].pattern               = imageT::Zero( 4 + i, 3 );
        value.dm[i].pattern( 1, 1 )       = 1;
        value.dm[i].pattern( 2, 2 )       = -0.5F;
        value.dm[i].reference             = imageT::Zero( 6, 4 );
        value.dm[i].reference( 1 + i, 2 ) = 7 + i;
        value.dm[i].reference( 4, 1 )     = -2;
        value.dm[i].amplitude             = i ? -0.125F : 0.25F;
    }
    value.frames       = 7;
    value.settleFrames = 3;
    value.dmSleep      = 0.0123;
    value.phaseTimeout = 4.5;
    value.setup        = { { "stageflowfs", "position", "current", "32.0", 0, 0.025 },
                           { "stagelosel", "fsm", "state", "READY", 1, 0 },
                           { "aoloop", "loop_state", "toggle", "0", 2, 0 } };
    return value;
}

void changeManifest( const std::filesystem::path &path, const std::string &from, const std::string &to )
{
    std::ifstream input( path );
    std::string   contents( ( std::istreambuf_iterator<char>( input ) ), std::istreambuf_iterator<char>() );
    auto          offset = contents.find( from );
    REQUIRE( offset != std::string::npos );
    contents.replace( offset, from.size(), to );
    input.close();
    std::ofstream output( path );
    output << contents;
    output.close();
    REQUIRE( output.good() );
}
/// \endcond

/// Captured patterns, references, geometry, and settings survive one immutable bundle round trip.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay calibration round trip preserves the complete pair", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    auto                 original = sampleCalibration();
    const auto           unmonitored = GENERATE( 0, 1, 2 );
    for( int i = 0; i < unmonitored; ++i )
    {
        original.dm[i].trigger.clear();
    }

    std::string          error;
    REQUIRE( validateCalibration( original, error ) );
    REQUIRE( saveCalibration( original, directory.m_path, error ) );
    REQUIRE( original.id.starts_with( "calibration__T" ) );
    calibration loaded;
    REQUIRE( loadCalibration( loaded, directory.m_path / original.profile / original.id, error ) );
    REQUIRE( loaded.id == original.id );
    REQUIRE( loaded.profile == original.profile );
    REQUIRE( loaded.camera == original.camera );
    REQUIRE( loaded.stream == original.stream );
    REQUIRE( MagAOX::roi::same( loaded.cameraGeometry.region, original.cameraGeometry.region ) );
    REQUIRE( loaded.cameraGeometry.units == original.cameraGeometry.units );
    REQUIRE( loaded.cameraGeometry.orientation == original.cameraGeometry.orientation );
    REQUIRE( loaded.frames == original.frames );
    REQUIRE( loaded.settleFrames == original.settleFrames );
    REQUIRE( loaded.dmSleep == original.dmSleep );
    REQUIRE( loaded.phaseTimeout == original.phaseTimeout );
    REQUIRE( ( loaded.baseline == original.baseline ).all() );
    for( size_t i = 0; i < original.dm.size(); ++i )
    {
        REQUIRE( loaded.dm[i].name == original.dm[i].name );
        REQUIRE( loaded.dm[i].channel == original.dm[i].channel );
        REQUIRE( loaded.dm[i].trigger == original.dm[i].trigger );
        REQUIRE( loaded.dm[i].amplitude == original.dm[i].amplitude );
        REQUIRE( ( loaded.dm[i].pattern == original.dm[i].pattern ).all() );
        REQUIRE( ( loaded.dm[i].reference == original.dm[i].reference ).all() );
    }
    REQUIRE( loaded.setup.size() == original.setup.size() );
    for( size_t i = 0; i < original.setup.size(); ++i )
    {
        REQUIRE( loaded.setup[i].device == original.setup[i].device );
        REQUIRE( loaded.setup[i].property == original.setup[i].property );
        REQUIRE( loaded.setup[i].element == original.setup[i].element );
        REQUIRE( loaded.setup[i].value == original.setup[i].value );
        REQUIRE( loaded.setup[i].type == original.setup[i].type );
        REQUIRE( loaded.setup[i].tolerance == original.setup[i].tolerance );
    }
    original.dm[0].pattern.setZero();
    REQUIRE( loaded.dm[0].pattern.abs().sum() > 0 );
}

/// Reject invalid geometry, signals, channel identity, settings, and malformed setup values.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay validates complete calibration invariants", "[autoJay][calibration]" )
{
    auto        value = sampleCalibration();
    std::string error;
    SECTION( "mismatched pair geometry" )
    {
        value.dm[1].reference.resize( 3, 2 );
    }
    SECTION( "missing baseline" )
    {
        value.baseline.resize( 0, 0 );
    }
    SECTION( "fractional image extent" )
    {
        value.cameraGeometry.region.w = 11;
    }
    SECTION( "unknown coordinate units" )
    {
        value.cameraGeometry.units = MagAOX::roi::units::unsupported;
    }
    SECTION( "zero binning" )
    {
        value.cameraGeometry.region.bin_x = 0;
    }
    SECTION( "nonfinite ROI" )
    {
        value.cameraGeometry.region.x = std::numeric_limits<float>::infinity();
    }
    SECTION( "missing orientation" )
    {
        value.cameraGeometry.orientation.clear();
    }
    SECTION( "empty pattern" )
    {
        value.dm[0].pattern.resize( 0, 0 );
    }
    SECTION( "zero pattern" )
    {
        value.dm[0].pattern.setZero();
    }
    SECTION( "nonfinite pattern" )
    {
        value.dm[0].pattern( 0, 0 ) = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "uniform reference" )
    {
        value.dm[1].reference.setConstant( 4 );
    }
    SECTION( "nonfinite reference" )
    {
        value.dm[1].reference( 0, 0 ) = std::numeric_limits<float>::infinity();
    }
    SECTION( "nonfinite baseline" )
    {
        value.baseline( 0, 0 ) = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "zero amplitude" )
    {
        value.dm[0].amplitude = 0;
    }
    SECTION( "nonfinite amplitude" )
    {
        value.dm[0].amplitude = std::numeric_limits<float>::infinity();
    }
    SECTION( "scaled command overflow" )
    {
        value.dm[0].amplitude       = std::numeric_limits<float>::max();
        value.dm[0].pattern( 1, 1 ) = 2;
    }
    SECTION( "shared channel" )
    {
        value.dm[1].channel = value.dm[0].channel;
    }
    SECTION( "combined channel" )
    {
        value.dm[0].channel = value.dm[1].trigger;
    }
    SECTION( "variance unavailable" )
    {
        value.frames = 1;
    }
    SECTION( "unbounded averaging" )
    {
        value.frames = 1000001;
    }
    SECTION( "nonfinite timeout" )
    {
        value.phaseTimeout = std::numeric_limits<double>::quiet_NaN();
    }
    SECTION( "negative delay" )
    {
        value.dmSleep = -1;
    }
    SECTION( "zero deadline" )
    {
        value.phaseTimeout = 0;
    }
    SECTION( "deadline before settling" )
    {
        value.phaseTimeout = value.dmSleep;
    }
    SECTION( "missing setup" )
    {
        value.setup.clear();
    }
    SECTION( "duplicate setup" )
    {
        value.setup.push_back( value.setup[0] );
    }
    SECTION( "trailing numeric data" )
    {
        value.setup[0].value = "32.0junk";
    }
    SECTION( "negative tolerance" )
    {
        value.setup[0].tolerance = -1;
    }
    SECTION( "invalid switch" )
    {
        value.setup[2].value = "On";
    }
    SECTION( "path traversal" )
    {
        value.profile = "../other";
    }
    SECTION( "manifest injection" )
    {
        value.setup[1].value = "READY\nframes=1";
    }
    REQUIRE_FALSE( validateCalibration( value, error ) );
    REQUIRE_FALSE( error.empty() );
}

/// Both native sensor and native binned coordinates describe the same stored image sizes.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay accepts supported binned geometry and signed patterns", "[autoJay][calibration]" )
{
    auto        value = sampleCalibration();
    std::string error;
    value.cameraGeometry.units    = MagAOX::roi::units::binned;
    value.cameraGeometry.region.w = 6;
    value.cameraGeometry.region.h = 4;
    value.dmSleep                 = 0;
    value.settleFrames            = 0;
    value.dm[0].pattern.setConstant( 1 );
    REQUIRE( validateCalibration( value, error ) );
}

/// Recall uses the selected profile and reports a corrupt newest bundle while ignoring staging files.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay recalls newest complete calibration in its profile", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    std::string          error;
    auto                 first  = sampleCalibration();
    auto                 second = first;
    REQUIRE( saveCalibration( first, directory.m_path, error ) );
    second.dm[0].amplitude = 0.75F;
    REQUIRE( saveCalibration( second, directory.m_path, error ) );
    REQUIRE( second.id > first.id );
    auto other    = second;
    other.profile = "science-pupil";
    REQUIRE( saveCalibration( other, directory.m_path, error ) );
    std::filesystem::create_directory( directory.m_path / first.profile / ".calibration__T99999999999999999999999" );
    calibration              loaded;
    std::vector<std::string> skipped;
    REQUIRE( latestCalibration( loaded, directory.m_path, first.profile, error, &skipped ) );
    REQUIRE( loaded.id == second.id );
    REQUIRE( skipped.empty() );
    std::filesystem::resize_file( directory.m_path / second.profile / second.id / "ncpc_reference.fits", 40 );
    REQUIRE( latestCalibration( loaded, directory.m_path, first.profile, error, &skipped ) );
    REQUIRE( loaded.id == first.id );
    REQUIRE( skipped.size() == 1 );
    REQUIRE( skipped.front().find( second.id ) != std::string::npos );
    REQUIRE_FALSE( latestCalibration( loaded, directory.m_path, "missing", error ) );
    REQUIRE( loaded.id == first.id );
    REQUIRE_FALSE( latestCalibration( loaded, directory.m_path, "../escape", error ) );
}

/// Missing, truncated, altered, and redirected components never partially replace the active calibration.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects damaged calibration components", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    auto                 value = sampleCalibration();
    std::string          error;
    REQUIRE( saveCalibration( value, directory.m_path, error ) );
    auto path          = directory.m_path / value.profile / value.id;
    auto active        = value;
    auto originalData  = active.dm[0].reference.data();
    auto index         = GENERATE( 0, 1, 2, 3, 4 );
    auto componentPath = path / calibrationIO::fileNames[index];
    SECTION( "missing" )
    {
        std::filesystem::remove( componentPath );
    }
    SECTION( "truncated" )
    {
        std::filesystem::resize_file( componentPath, 20 );
    }
    SECTION( "same size corruption" )
    {
        std::fstream output( componentPath, std::ios::in | std::ios::out | std::ios::binary );
        output.put( '!' );
    }
    SECTION( "symlink" )
    {
        auto backup = path / "backup.fits";
        std::filesystem::rename( componentPath, backup );
        std::filesystem::create_symlink( backup, componentPath );
    }
    REQUIRE_FALSE( loadCalibration( active, path, error ) );
    REQUIRE( active.id == value.id );
    REQUIRE( active.dm[0].reference.data() == originalData );
    REQUIRE_FALSE( error.empty() );
}

/// Strict metadata rejects unknown schema, duplicate fields, trailing numeric data, and identity mismatch.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects malformed calibration manifests", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    auto                 value = sampleCalibration();
    std::string          error;
    REQUIRE( saveCalibration( value, directory.m_path, error ) );
    auto path     = directory.m_path / value.profile / value.id;
    auto manifest = path / "calibration.conf";
    SECTION( "new schema" )
    {
        changeManifest( manifest, "schema=1", "schema=2" );
    }
    SECTION( "duplicate schema" )
    {
        changeManifest( manifest, "schema=1", "schema=1\nschema=1" );
    }
    SECTION( "new processing convention" )
    {
        changeManifest( manifest, "processing=1", "processing=2" );
    }
    SECTION( "trailing scalar junk" )
    {
        changeManifest( manifest, "frames=7", "frames=7junk" );
    }
    SECTION( "unbounded rule allocation" )
    {
        changeManifest( manifest, "setupCount=3", "setupCount=99999999" );
    }
    SECTION( "missing metadata" )
    {
        changeManifest( manifest, "frames=7\n", "" );
    }
    SECTION( "unexpected metadata" )
    {
        changeManifest( manifest, "frames=7", "frames=7\nunknown=1" );
    }
    SECTION( "profile mismatch" )
    {
        changeManifest( manifest, "profile=flowfs-pupil", "profile=other" );
    }
    SECTION( "identifier mismatch" )
    {
        changeManifest( manifest, "id=" + value.id, "id=different" );
    }
    SECTION( "uncommitted directory" )
    {
        auto hidden = path.parent_path() / ( "." + value.id );
        std::filesystem::rename( path, hidden );
        path = hidden;
    }
    calibration active = value;
    REQUIRE_FALSE( loadCalibration( active, path, error ) );
    REQUIRE( active.id == value.id );
}

/// Each failed image write and final commit leaves no selectable partial bundle or changed caller ID.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay calibration save is atomic across component failures", "[autoJay][calibration]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJayDetail::saveCalibration;
    #endif
    // clang-format on
    calibrationDirectory directory;
    auto                 first = sampleCalibration();
    std::string          error;
    REQUIRE( saveCalibration( first, directory.m_path, error ) );
    auto next    = first;
    auto failure = GENERATE( 0, 1, 2, 3, 4, 5, 6, 7 );
    int  writes  = 0;
    auto writer  = [&]( const std::filesystem::path &path, const imageT &image, std::string &description )
    {
        if( failure == 7 )
            throw std::runtime_error( "injected writer exception" );
        if( writes++ == failure )
            return calibrationIO::fail( description, "injected component write failure" );
        if( failure == 6 )
        {
            std::ofstream invalid( path );
            invalid << "invalid FITS";
            return true;
        }
        return calibrationIO::writeImage( path, image, description );
    };
    auto committer = [&]( const std::filesystem::path &from, const std::filesystem::path &to, std::string &description )
    {
        if( failure == 5 )
            return calibrationIO::fail( description, "injected final rename failure" );
        return calibrationIO::commit( from, to, description );
    };
    REQUIRE_FALSE( calibrationIO::save( next, directory.m_path, error, writer, committer ) );
    REQUIRE_FALSE( error.empty() );
    REQUIRE( next.id == first.id );
    calibration loaded;
    REQUIRE( latestCalibration( loaded, directory.m_path, first.profile, error ) );
    REQUIRE( loaded.id == first.id );
    size_t remaining = 0;
    for( const auto &entry : std::filesystem::directory_iterator( directory.m_path / first.profile ) )
    {
        REQUIRE( entry.path().filename() == first.id );
        ++remaining;
    }
    REQUIRE( remaining == 1 );
}

/// Filesystem failures are reported without changing the caller's valid calibration object.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay reports calibration directory failures", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    auto                 value = sampleCalibration();
    value.id                   = "previous";
    std::ofstream( directory.m_path / "blocked" ) << "file";
    std::string error;
    REQUIRE_FALSE( saveCalibration( value, directory.m_path / "blocked", error ) );
    REQUIRE( value.id == "previous" );
    REQUIRE_FALSE( error.empty() );
}

/// Bootstrap patterns are accepted without custom checksums, while truncated and invalid files are rejected.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay strictly loads supplied pattern FITS", "[autoJay][calibration]" )
{
    calibrationDirectory directory;
    auto                 pattern = sampleCalibration().dm[0].pattern;
    auto                 path    = directory.m_path / "supplied.fits";
    std::string          error;
    REQUIRE( calibrationIO::writeImage( path, pattern, error ) );
    imageT loaded;
    REQUIRE( loadPattern( loaded, path, error ) );
    REQUIRE( ( loaded == pattern ).all() );
    auto storage = loaded.data();
    SECTION( "short header" )
    {
        std::filesystem::resize_file( path, 30 );
    }
    SECTION( "short data block" )
    {
        std::filesystem::resize_file( path, 2880 + 4 );
    }
    SECTION( "missing padding" )
    {
        std::filesystem::resize_file( path, std::filesystem::file_size( path ) - 1 );
    }
    SECTION( "no response" )
    {
        pattern.setZero();
        REQUIRE( calibrationIO::writeImage( path, pattern, error ) );
    }
    SECTION( "nonfinite pixels" )
    {
        pattern( 1, 1 ) = std::numeric_limits<float>::quiet_NaN();
        REQUIRE( calibrationIO::writeImage( path, pattern, error ) );
    }
    SECTION( "three dimensional input" )
    {
        mx::fits::fitsFile<float> file;
        std::vector<float>        pixels( 24, 1 );
        REQUIRE( file.write( path, pixels.data(), 3, 4, 2 ) == mx::error_t::noerror );
    }
    REQUIRE_FALSE( loadPattern( loaded, path, error ) );
    REQUIRE( loaded.data() == storage );
    REQUIRE_FALSE( error.empty() );
}

} // namespace autoJayTest
} // namespace libXWCTest
