/** \file autoJayCalibration.hpp
 * \brief Immutable calibration bundles for the dual-DM alignment sensor.
 * \ingroup autoJay_files
 */
#ifndef autoJayCalibration_hpp
#define autoJayCalibration_hpp

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <type_traits>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mx/app/ini.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "autoJayTypes.hpp"

namespace MagAOX
{
namespace app
{
namespace autoJayDetail
{

/// Validate a complete captured calibration without consulting devices or files.
inline bool validateCalibration( const calibration &value /**< [in] candidate bundle */,
                                 std::string       &error /**< [out] failure description */ );

/// Load a supplied full-DM pattern with bounded allocation and strict FITS integrity checks.
inline bool loadPattern( imageT            &image /**< [out] finite nonzero command map */,
                         const std::string &filename /**< [in] supplied FITS file */,
                         std::string       &error /**< [out] failure description */ );

/// Write and atomically publish a new bundle; update its ID only after success.
inline bool saveCalibration( calibration       &value /**< [in,out] captured bundle and committed ID */,
                             const std::string &rootDir /**< [in] app calibration directory */,
                             std::string       &error /**< [out] failure description */ );

/// Load one complete bundle, preserving the destination on every failure.
inline bool loadCalibration( calibration       &value /**< [out] complete validated bundle */,
                             const std::string &directory /**< [in] committed bundle directory */,
                             std::string       &error /**< [out] failure description */ );

/// Select the newest valid bundle in the requested profile, reporting rejected candidates.
inline bool latestCalibration( calibration              &value /**< [out] complete validated bundle */,
                               const std::string        &rootDir /**< [in] app calibration directory */,
                               const std::string        &profile /**< [in] explicitly selected profile */,
                               std::string              &error /**< [out] failure description */,
                               std::vector<std::string> *skipped = nullptr /**< [out] optional rejected candidates */ );

/// Local serialization helpers; no helper connects to instrument resources.
namespace calibrationIO
{
/// Limit allocations from calibration files to sixteen million pixels per image.
inline constexpr size_t maxPixels = 16 * 1024 * 1024;

/// Fixed component names prevent a manifest from redirecting image reads.
inline const std::array<std::string, 5> fileNames = {
    "ncpc_pattern.fits", "tweeter_pattern.fits", "ncpc_reference.fits", "tweeter_reference.fits", "unpoked.fits" };

/// Record an error and return false.
inline bool fail( std::string       &error /**< [out] failure description */,
                  const std::string &message /**< [in] explanation */ );

/// Check one filename component without accepting traversal or hidden names.
inline bool component( const std::string &value /**< [in] component */ );

/// Check that an INI value survives the parser without truncation or trimming.
inline bool textValue( const std::string &value /**< [in] manifest string */ );

/// Parse a complete bounded scalar, rejecting trailing data and nonfinite values.
template <typename T>
inline bool number( T &value /**< [out] parsed scalar */, const std::string &text /**< [in] numeric string */ );

/// Check finite pixels and optionally require a spatially varying signal.
inline bool imageValid( const imageT &image /**< [in] image to inspect */,
                        bool          signal /**< [in] require nonconstant data */ );

/// Require a finite nonzero DM command map without restricting its spatial structure.
inline bool patternValid( const imageT &image /**< [in] command map */ );

/// Return mutable image pointers in the fixed on-disk component order.
inline std::array<imageT *, 5> images( calibration &value /**< [in,out] bundle owning the components */ );

/// Hash a regular file and record its size for accidental-corruption detection.
inline bool fingerprint( const std::filesystem::path &path /**< [in] file to inspect */,
                         uint64_t                    &bytes /**< [out] complete file size */,
                         uint64_t                    &hash /**< [out] FNV-1a checksum, not a cryptographic signature */,
                         std::string                 &error /**< [out] failure description */ );

/// Flush an already closed file or directory before publishing a bundle.
inline bool syncPath( const std::filesystem::path &path /**< [in] file or directory */,
                      std::string                 &error /**< [out] failure description */ );

/// Write one FITS image and check its close and filesystem flush.
inline bool writeImage( const std::filesystem::path &path /**< [in] staging filename */,
                        const imageT                &image /**< [in] pixels */,
                        std::string                 &error /**< [out] failure description */ );

/// Close a CFITSIO handle while preserving the original read error.
inline void closeFits( fitsfile *file /**< [in] owned handle */ );

/// Read exactly one bounded image, rejecting short files and FITS null pixels.
inline bool readImage( imageT                      &image /**< [out] complete finite pixels */,
                       const std::filesystem::path &path /**< [in] regular FITS file */,
                       std::string                 &error /**< [out] failure description */ );

/// Atomically move one complete directory into the committed namespace.
inline bool commit( const std::filesystem::path &from /**< [in] staging directory */,
                    const std::filesystem::path &to /**< [in] unused final path */,
                    std::string                 &error /**< [out] failure description */ );

/// INI callback rejecting repeated keys, including multiline continuations.
inline int parseEntry( void       *context /**< [in,out] parsed key map */,
                       const char *section /**< [in] section name */,
                       const char *name /**< [in] key name */,
                       const char *value /**< [in] raw value */ );

/// Serialize the fixed schema and integrity records into a staging manifest.
inline bool writeManifest( const calibration           &value /**< [in] captured bundle */,
                           const std::filesystem::path &directory /**< [in] staging directory */,
                           std::string                 &error /**< [out] failure description */ );

/// Read all components, including an unpublished staging bundle during save validation.
inline bool read( calibration       &value /**< [out] complete validated bundle */,
                  const std::string &directory /**< [in] bundle directory */,
                  std::string       &error /**< [out] failure description */ );

/// Save implementation with narrow write/rename seams for deterministic offline failure tests.
template <typename Writer, typename Committer>
inline bool save( calibration       &value /**< [in,out] captured bundle and committed ID */,
                  const std::string &rootDir /**< [in] app calibration directory */,
                  std::string       &error /**< [out] failure description */,
                  Writer             writer /**< [in] FITS writer */,
                  Committer          committer /**< [in] final directory rename */ );

inline bool fail( std::string &error, const std::string &message )
{
    error = message;
    return false;
}

inline bool component( const std::string &value )
{
    if( value.empty() || value.size() > 128 || value.front() == '.' )
        return false;
    for( unsigned char c : value )
        if( !( ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ) || ( c >= '0' && c <= '9' ) || c == '_' ||
               c == '-' || c == '.' ) )
            return false;
    return true;
}

inline bool textValue( const std::string &value )
{
    return !value.empty() && value.size() <= 512 && value.find_first_of( "\r\n;#" ) == std::string::npos &&
           value.front() != ' ' && value.back() != ' ' && value.find( '\0' ) == std::string::npos &&
           value.find( '\t' ) == std::string::npos;
}

template <typename T>
inline bool number( T &value, const std::string &text )
{
    T    candidate{};
    auto result = std::from_chars( text.data(), text.data() + text.size(), candidate );
    if( result.ec != std::errc{} || result.ptr != text.data() + text.size() )
        return false;
    if constexpr( std::is_floating_point_v<T> )
        if( !mx::math::isFinite( candidate ) )
            return false;
    value = candidate;
    return true;
}

inline bool imageValid( const imageT &image, bool signal )
{
    if( image.rows() <= 0 || image.cols() <= 0 || size_t( image.size() ) > maxPixels )
        return false;
    bool varying = false;
    for( Eigen::Index n = 0; n < image.size(); ++n )
    {
        if( !mx::math::isFinite( image.data()[n] ) )
            return false;
        varying |= image.data()[n] != image.data()[0];
    }
    return !signal || varying;
}

inline bool patternValid( const imageT &image )
{
    if( !imageValid( image, false ) )
        return false;
    for( Eigen::Index n = 0; n < image.size(); ++n )
        if( image.data()[n] != 0 )
            return true;
    return false;
}

inline std::array<imageT *, 5> images( calibration &value )
{
    return {
        &value.dm[0].pattern, &value.dm[1].pattern, &value.dm[0].reference, &value.dm[1].reference, &value.baseline };
}

inline bool fingerprint( const std::filesystem::path &path, uint64_t &bytes, uint64_t &hash, std::string &error )
{
    if( std::filesystem::symlink_status( path ).type() != std::filesystem::file_type::regular )
        return fail( error, "not a regular calibration file: " + path.string() );
    bytes = std::filesystem::file_size( path );
    if( bytes == 0 || bytes > maxPixels * sizeof( float ) + 1024 * 1024 )
        return fail( error, "calibration file size outside bounds: " + path.string() );
    std::ifstream input( path, std::ios::binary );
    hash          = UINT64_C( 14695981039346656037 );
    uint64_t read = 0;
    char     buffer[8192];
    while( input.read( buffer, sizeof( buffer ) ) || input.gcount() )
    {
        read += input.gcount();
        for( std::streamsize i = 0; i < input.gcount(); ++i )
        {
            hash ^= static_cast<unsigned char>( buffer[i] );
            hash *= UINT64_C( 1099511628211 );
        }
    }
    return ( input.eof() && read == bytes ) || fail( error, "cannot read " + path.string() );
}

inline bool syncPath( const std::filesystem::path &path, std::string &error )
{
    int fd = ::open( path.c_str(), O_RDONLY | O_CLOEXEC );
    if( fd < 0 )
        return fail( error, "cannot open for sync: " + path.string() );
    bool ok = ::fsync( fd ) == 0;
    ok      = ( ::close( fd ) == 0 ) && ok;
    return ok || fail( error, "cannot sync: " + path.string() );
}

inline bool writeImage( const std::filesystem::path &path, const imageT &image, std::string &error )
{
    mx::fits::fitsFile<float> file;
    if( file.write( path.string(), image ) != mx::error_t::noerror )
        return fail( error, "cannot write FITS: " + path.string() );
    return syncPath( path, error );
}

inline void closeFits( fitsfile *file )
{
    int status = 0;
    fits_close_file( file, &status );
}

inline bool readImage( imageT &image, const std::filesystem::path &path, std::string &error )
{
    if( std::filesystem::symlink_status( path ).type() != std::filesystem::file_type::regular )
        return fail( error, "not a regular FITS file: " + path.string() );
    auto bytes = std::filesystem::file_size( path );
    if( bytes == 0 || bytes > maxPixels * sizeof( double ) + 1024 * 1024 )
        return fail( error, "FITS file size outside bounds: " + path.string() );
    fitsfile *raw    = nullptr;
    int       status = 0;
    fits_open_file( &raw, path.c_str(), READONLY, &status );
    std::unique_ptr<fitsfile, decltype( &closeFits )> file( raw, closeFits );
    if( status != 0 )
        return fail( error, "cannot open FITS: " + path.string() );
    int  dimensions = 0;
    long axes[2]    = {};
    fits_get_img_dim( raw, &dimensions, &status );
    fits_get_img_size( raw, 2, axes, &status );
    if( status != 0 || dimensions != 2 || axes[0] <= 0 || axes[1] <= 0 || uint64_t( axes[0] ) > maxPixels ||
        uint64_t( axes[1] ) > maxPixels / axes[0] )
        return fail( error, "FITS image must have two bounded positive dimensions" );
    LONGLONG header, data, end;
    fits_get_hduaddrll( raw, &header, &data, &end, &status );
    if( status != 0 || header < 0 || data < header || end < data || uint64_t( end ) > bytes )
        return fail( error, "truncated FITS image: " + path.string() );
    imageT candidate( axes[0], axes[1] );
    long   first[2]  = { 1, 1 };
    float  nullValue = std::numeric_limits<float>::quiet_NaN();
    int    anyNull   = 0;
    fits_read_pix( raw, TFLOAT, first, candidate.size(), &nullValue, candidate.data(), &anyNull, &status );
    if( status != 0 || anyNull || !imageValid( candidate, false ) )
        return fail( error, "incomplete or nonfinite FITS image: " + path.string() );
    fits_close_file( file.release(), &status );
    if( status != 0 )
        return fail( error, "cannot close FITS image: " + path.string() );
    image = std::move( candidate );
    return true;
}

inline bool commit( const std::filesystem::path &from, const std::filesystem::path &to, std::string &error )
{
    if( std::filesystem::exists( to ) || ::rename( from.c_str(), to.c_str() ) != 0 )
        return fail( error, "cannot commit calibration: " + to.string() );
    return true;
}

inline int parseEntry( void *context, const char *section, const char *name, const char *value )
{
    auto &entries = *static_cast<std::map<std::string, std::string> *>( context );
    if( !section || !name || !value || entries.size() > 1024 )
        return 0;
    return entries.emplace( std::string( section ) + "." + name, value ).second;
}

inline bool writeManifest( const calibration &value, const std::filesystem::path &directory, std::string &error )
{
    std::ofstream out( directory / "calibration.conf" );
    out << std::setprecision( std::numeric_limits<double>::max_digits10 );
    out << "[calibration]\nschema=1\nprocessing=1\nid=" << value.id << "\nprofile=" << value.profile
        << "\ncamera=" << value.camera << "\nstream=" << value.stream << "\nframes=" << value.frames
        << "\nsettleFrames=" << value.settleFrames << "\ndmSleep=" << value.dmSleep
        << "\nphaseTimeout=" << value.phaseTimeout << "\nsetupCount=" << value.setup.size() << '\n';
    const auto &g = value.cameraGeometry;
    out << "[geometry]\nx=" << g.region.x << "\ny=" << g.region.y << "\nw=" << g.region.w << "\nh=" << g.region.h
        << "\nbin_x=" << g.region.bin_x << "\nbin_y=" << g.region.bin_y << "\nunits=" << static_cast<int>( g.units )
        << "\norientation=" << g.orientation << '\n';
    for( size_t i = 0; i < value.dm.size(); ++i )
    {
        const auto &dm = value.dm[i];
        out << "[dm" << i << "]\nname=" << dm.name << "\nchannel=" << dm.channel << "\ntrigger=" << dm.trigger
            << "\namplitude=" << dm.amplitude << '\n';
    }
    for( size_t i = 0; i < value.setup.size(); ++i )
    {
        const auto &rule = value.setup[i];
        out << "[setup" << i << "]\ndevice=" << rule.device << "\nproperty=" << rule.property
            << "\nelement=" << rule.element << "\nvalue=" << rule.value << "\ntype=" << rule.type
            << "\ntolerance=" << rule.tolerance << '\n';
    }
    for( size_t i = 0; i < fileNames.size(); ++i )
    {
        uint64_t bytes, hash;
        if( !fingerprint( directory / fileNames[i], bytes, hash, error ) )
            return false;
        out << "[file" << i << "]\nbytes=" << bytes << "\nfnv1a=" << hash << '\n';
    }
    out.close();
    return ( bool( out ) || fail( error, "cannot write calibration manifest" ) ) &&
           syncPath( directory / "calibration.conf", error );
}

} // namespace calibrationIO

inline bool validateCalibration( const calibration &value, std::string &error )
{
    using namespace calibrationIO;
    error.clear();
    if( !component( value.profile ) || !component( value.camera ) || !component( value.stream ) ||
        ( !value.id.empty() && !component( value.id ) ) )
        return fail( error, "invalid calibration identity" );
    const auto &g = value.cameraGeometry;
    if( !roi::valid( g.region ) || ( g.units != roi::units::sensor && g.units != roi::units::binned ) ||
        !textValue( g.orientation ) )
        return fail( error, "invalid reference geometry or orientation" );
    double width  = roi::imageSize( g.region.w, g.region.bin_x, g.units );
    double height = roi::imageSize( g.region.h, g.region.bin_y, g.units );
    if( width != value.baseline.rows() || height != value.baseline.cols() || !imageValid( value.baseline, false ) )
        return fail( error, "baseline does not match reference geometry" );
    if( value.frames < 2 || value.frames > 10000 || value.settleFrames > 10000 ||
        !mx::math::isFinite( value.dmSleep ) || value.dmSleep < 0 || value.dmSleep > 3600 ||
        ( ( value.dm[0].trigger.empty() || value.dm[1].trigger.empty() ) && value.dmSleep == 0 ) ||
        !mx::math::isFinite( value.phaseTimeout ) || value.phaseTimeout <= value.dmSleep || value.phaseTimeout > 3600 )
    {
        return fail( error, "invalid calibration acquisition parameters" );
    }
    for( const auto &dm : value.dm )
    {
        if( !component( dm.name ) || !component( dm.channel ) || ( !dm.trigger.empty() && !component( dm.trigger ) ) ||
            !mx::math::isFinite( dm.amplitude ) || dm.amplitude == 0 || !patternValid( dm.pattern ) ||
            !imageValid( dm.reference, true ) || dm.reference.rows() != value.baseline.rows() ||
            dm.reference.cols() != value.baseline.cols() )
        {
            return fail( error, "invalid DM identity, pattern, amplitude, or reference" );
        }
        for( Eigen::Index i = 0; i < dm.pattern.size(); ++i )
            if( !mx::math::isFinite( dm.pattern.data()[i] * dm.amplitude ) )
                return fail( error, "scaled DM pattern overflows" );
        for( const auto &other : value.dm )
            if( dm.channel == other.trigger )
                return fail( error, "poke channel is a combined trigger stream" );
    }
    if( value.dm[0].channel == value.dm[1].channel || value.dm[0].name == value.dm[1].name )
        return fail( error, "DM identities and owned channels must differ" );
    if( value.setup.empty() || value.setup.size() > 128 )
        return fail( error, "calibration requires a bounded nonempty setup profile" );
    std::set<std::string> rules;
    for( const auto &rule : value.setup )
    {
        double numeric;
        if( !component( rule.device ) || !component( rule.property ) || !component( rule.element ) ||
            !textValue( rule.value ) || rule.type < 0 || rule.type > 2 || !mx::math::isFinite( rule.tolerance ) ||
            rule.tolerance < 0 || ( rule.type == 0 && !number( numeric, rule.value ) ) ||
            ( rule.type == 2 && rule.value != "0" && rule.value != "1" ) ||
            !rules.insert( rule.device + "/" + rule.property + "/" + rule.element ).second )
            return fail( error, "invalid or duplicate setup requirement" );
    }
    return true;
}

inline bool calibrationIO::read( calibration &value, const std::string &directory, std::string &error )
{
    using namespace calibrationIO;
    error.clear();
    try
    {
        std::filesystem::path path( directory );
        if( std::filesystem::symlink_status( path ).type() != std::filesystem::file_type::directory ||
            std::filesystem::symlink_status( path / "calibration.conf" ).type() !=
                std::filesystem::file_type::regular ||
            std::filesystem::file_size( path / "calibration.conf" ) > 128 * 1024 )
            return fail( error, "invalid calibration directory or manifest" );
        std::map<std::string, std::string> fields;
        if( ini_parse( ( path / "calibration.conf" ).c_str(), parseEntry, &fields ) != 0 )
            return fail( error, "malformed or duplicate calibration manifest fields" );
        auto take = [&]( const std::string &key, auto &destination, bool allowEmpty = false )
        {
            auto found = fields.find( key );
            if( found == fields.end() )
                return fail( error, "missing manifest field: " + key );
            using T = std::decay_t<decltype( destination )>;
            bool ok;
            if constexpr( std::is_same_v<T, std::string> )
            {
                destination = found->second;
                ok          = ( allowEmpty && destination.empty() ) || textValue( destination );
            }
            else
                ok = number( destination, found->second );
            fields.erase( found );
            return ok || fail( error, "invalid manifest field: " + key );
        };
        calibration candidate;
        int         schema, processing, units;
        unsigned    setupCount;
        if( !take( "calibration.schema", schema ) || !take( "calibration.processing", processing ) )
            return false;
        if( schema != 1 || processing != 1 )
            return fail( error, "unsupported calibration schema or processing version" );
        if( !take( "calibration.id", candidate.id ) || !take( "calibration.profile", candidate.profile ) ||
            !take( "calibration.camera", candidate.camera ) || !take( "calibration.stream", candidate.stream ) ||
            !take( "calibration.frames", candidate.frames ) ||
            !take( "calibration.settleFrames", candidate.settleFrames ) ||
            !take( "calibration.dmSleep", candidate.dmSleep ) ||
            !take( "calibration.phaseTimeout", candidate.phaseTimeout ) ||
            !take( "calibration.setupCount", setupCount ) )
            return false;
        if( setupCount == 0 || setupCount > 128 )
            return fail( error, "invalid setup count" );
        auto &g = candidate.cameraGeometry;
        if( !take( "geometry.x", g.region.x ) || !take( "geometry.y", g.region.y ) ||
            !take( "geometry.w", g.region.w ) || !take( "geometry.h", g.region.h ) ||
            !take( "geometry.bin_x", g.region.bin_x ) || !take( "geometry.bin_y", g.region.bin_y ) ||
            !take( "geometry.units", units ) || !take( "geometry.orientation", g.orientation ) )
            return false;
        g.units = static_cast<roi::units>( units );
        for( size_t i = 0; i < candidate.dm.size(); ++i )
        {
            auto  prefix = "dm" + std::to_string( i ) + ".";
            auto &dm     = candidate.dm[i];
            if( !take( prefix + "name", dm.name ) || !take( prefix + "channel", dm.channel ) ||
                !take( prefix + "trigger", dm.trigger, true ) || !take( prefix + "amplitude", dm.amplitude ) )
            {
                return false;
            }
        }
        candidate.setup.resize( setupCount );
        for( size_t i = 0; i < candidate.setup.size(); ++i )
        {
            auto  prefix = "setup" + std::to_string( i ) + ".";
            auto &rule   = candidate.setup[i];
            if( !take( prefix + "device", rule.device ) || !take( prefix + "property", rule.property ) ||
                !take( prefix + "element", rule.element ) || !take( prefix + "value", rule.value ) ||
                !take( prefix + "type", rule.type ) || !take( prefix + "tolerance", rule.tolerance ) )
                return false;
        }
        auto components = images( candidate );
        for( size_t i = 0; i < fileNames.size(); ++i )
        {
            uint64_t expectedBytes, expectedHash, bytes, hash;
            auto     prefix = "file" + std::to_string( i ) + ".";
            if( !take( prefix + "bytes", expectedBytes ) || !take( prefix + "fnv1a", expectedHash ) ||
                !fingerprint( path / fileNames[i], bytes, hash, error ) )
                return false;
            if( bytes != expectedBytes || hash != expectedHash )
                return fail( error, "calibration component integrity mismatch: " + fileNames[i] );
            if( !readImage( *components[i], path / fileNames[i], error ) )
                return false;
        }
        if( !fields.empty() )
            return fail( error, "unexpected calibration manifest field: " + fields.begin()->first );
        if( !validateCalibration( candidate, error ) )
            return false;
        value = std::move( candidate );
        return true;
    }
    catch( const std::exception &exception )
    {
        return fail( error, exception.what() );
    }
}

namespace calibrationIO
{
template <typename Writer, typename Committer>
inline bool
save( calibration &value, const std::string &rootDir, std::string &error, Writer writer, Committer committer )
{
    if( !validateCalibration( value, error ) )
        return false;
    std::filesystem::path staging;
    try
    {
        auto profile = std::filesystem::path( rootDir ) / value.profile;
        std::filesystem::create_directories( profile );
        auto        now       = std::chrono::system_clock::now();
        auto        seconds   = std::chrono::time_point_cast<std::chrono::seconds>( now );
        auto        nanos     = std::chrono::duration_cast<std::chrono::nanoseconds>( now - seconds ).count();
        std::time_t timestamp = std::chrono::system_clock::to_time_t( seconds );
        std::tm     utc;
        if( !gmtime_r( &timestamp, &utc ) )
            return fail( error, "cannot generate calibration timestamp" );
        std::ostringstream id;
        id << "calibration__T" << std::put_time( &utc, "%Y%m%d%H%M%S" ) << std::setw( 9 ) << std::setfill( '0' )
           << nanos;
        calibration candidate = value;
        candidate.id          = id.str();
        std::string temporary = ( profile / ( "." + candidate.id + ".XXXXXX" ) ).string();
        char       *made      = ::mkdtemp( temporary.data() );
        if( !made )
            return fail( error, "cannot create calibration staging directory" );
        staging      = made;
        auto discard = [&]()
        {
            std::error_code ignored;
            std::filesystem::remove_all( staging, ignored );
            return false;
        };
        auto components = images( candidate );
        for( size_t i = 0; i < fileNames.size(); ++i )
            if( !writer( staging / fileNames[i], *components[i], error ) )
                return discard();
        calibration verified;
        if( !writeManifest( candidate, staging, error ) || !read( verified, staging.string(), error ) )
            return discard();
        if( ::chmod( staging.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH ) != 0 )
        {
            fail( error, "cannot make committed calibration readable" );
            return discard();
        }
        if( !syncPath( staging, error ) )
            return discard();
        auto finalPath = profile / candidate.id;
        if( !committer( staging, finalPath, error ) )
            return discard();
        staging.clear();
        if( !syncPath( profile, error ) )
        {
            // A failed durable commit must not leave a candidate eligible for recall.
            std::filesystem::remove_all( finalPath );
            return false;
        }
        value.id = candidate.id;
        return true;
    }
    catch( const std::exception &exception )
    {
        if( !staging.empty() )
        {
            std::error_code ignored;
            std::filesystem::remove_all( staging, ignored );
        }
        return fail( error, exception.what() );
    }
}
} // namespace calibrationIO

inline bool saveCalibration( calibration &value, const std::string &rootDir, std::string &error )
{
    return calibrationIO::save( value, rootDir, error, calibrationIO::writeImage, calibrationIO::commit );
}

inline bool loadPattern( imageT &image, const std::string &filename, std::string &error )
{
    error.clear();
    try
    {
        imageT candidate;
        if( !calibrationIO::readImage( candidate, filename, error ) )
            return false;
        if( !calibrationIO::patternValid( candidate ) )
            return calibrationIO::fail( error, "pattern contains no nonzero commands" );
        image = std::move( candidate );
        return true;
    }
    catch( const std::exception &exception )
    {
        return calibrationIO::fail( error, exception.what() );
    }
}

inline bool loadCalibration( calibration &value, const std::string &directory, std::string &error )
{
    auto path = std::filesystem::path( directory );
    auto name = path.filename().string();
    if( name.size() != 37 || !name.starts_with( "calibration__T" ) ||
        name.find_first_not_of( "0123456789", 14 ) != std::string::npos )
        return calibrationIO::fail( error, "not a committed calibration directory" );
    calibration candidate;
    if( !calibrationIO::read( candidate, directory, error ) )
        return false;
    if( candidate.id != name || candidate.profile != path.parent_path().filename() )
        return calibrationIO::fail( error, "calibration profile or identifier mismatch" );
    value = std::move( candidate );
    return true;
}

inline bool latestCalibration( calibration              &value,
                               const std::string        &rootDir,
                               const std::string        &profile,
                               std::string              &error,
                               std::vector<std::string> *skipped )
{
    using namespace calibrationIO;
    error.clear();
    if( skipped )
        skipped->clear();
    if( !component( profile ) )
        return fail( error, "invalid calibration profile" );
    try
    {
        auto                               directory = std::filesystem::path( rootDir ) / profile;
        std::vector<std::filesystem::path> candidates;
        if( !std::filesystem::exists( directory ) )
            return fail( error, "no calibration for profile " + profile );
        for( const auto &entry : std::filesystem::directory_iterator( directory ) )
        {
            auto name = entry.path().filename().string();
            if( name.size() == 37 && name.starts_with( "calibration__T" ) &&
                name.find_first_not_of( "0123456789", 14 ) == std::string::npos )
                candidates.push_back( entry.path() );
        }
        std::sort( candidates.rbegin(), candidates.rend() );
        for( const auto &candidatePath : candidates )
        {
            calibration candidate;
            std::string reason;
            if( loadCalibration( candidate, candidatePath.string(), reason ) && candidate.profile == profile &&
                candidate.id == candidatePath.filename() )
            {
                value = std::move( candidate );
                return true;
            }
            if( skipped )
                skipped->push_back( candidatePath.filename().string() + ": " +
                                    ( reason.empty() ? "profile or identifier mismatch" : reason ) );
        }
        return fail( error, "no valid calibration for profile " + profile );
    }
    catch( const std::exception &exception )
    {
        return fail( error, exception.what() );
    }
}

} // namespace autoJayDetail
} // namespace app
} // namespace MagAOX
#endif // autoJayCalibration_hpp
