/** \file dlCollection.hpp
 * \brief Hardware-independent coefficient validation, capture, and bounded file work.
 * \ingroup dlDataCollection_files
 */
#ifndef dlCollection_hpp
#define dlCollection_hpp

#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <fcntl.h>
#include <unistd.h>
#include <openssl/evp.h>

namespace MagAOX
{
namespace app
{
/// Implementation helpers shared by the collector and its CPU tests.
namespace dlCollection
{
/// Return a checked allocation or file-size product.
inline size_t product( size_t a /**< [in] First factor. */, size_t b /**< [in] Second factor. */ );

/// Check IEEE float bits; this app and its tests are compiled without fast-math.
inline bool finite( float value /**< [in] Value to validate. */ );

/// Validate a timeout's IEEE double bits without floating-point arithmetic.
inline bool finite( double value /**< [in] Value to validate. */ );

/// Return a SHA-256 digest of a memory buffer.
inline std::string digest( const void *data /**< [in] Bytes to hash. */, size_t bytes /**< [in] Byte count. */ );

/// Read an exact float32 file, rejecting unsupported host byte order and trailing bytes.
inline std::vector<float> readFloats( const std::filesystem::path &path /**< [in] Input file. */,
                                      size_t                       count /**< [in] Required number of floats. */ );

/// Write, flush and finalize a file without replacing any previous result.
inline void writeFile( const std::filesystem::path &path /**< [in] Final filename. */,
                       const void                  *data /**< [in] Bytes to save. */,
                       size_t                       bytes /**< [in] Byte count. */ );

/// Write a checked text file with the same no-replace behavior as binary data.
inline void writeText( const std::filesystem::path &path /**< [in] Final filename. */,
                       const std::string           &text /**< [in] Text to save. */ );

/// Immutable identity and dimensions of an offline-prepared command bank.
struct Bank
{
    /// Directory containing the bank manifest and payloads.
    std::filesystem::path m_path;

    /// Number of 50x50 commands in each set.
    size_t m_samples{ 0 };

    /// Number of available sets.
    size_t m_sets{ 0 };

    /// SHA-256 identity of the manifest used for this run.
    std::string m_manifestHash;

    /// SHA-256 identity of the recorded actuator mask.
    std::string m_maskHash;

    /// Identity of generation/legacy-conversion provenance.
    std::string m_generationHash;

    /// Binary controllable-actuator mask in the command stream's memory order.
    std::vector<float> m_mask;

    /// Expected SHA-256 digest of each float32 command payload.
    std::vector<std::string> m_hashes;
};

/// Read and validate a bank manifest and its explicit actuator mask.
inline Bank readBank( const std::filesystem::path &path /**< [in] Bank directory. */ );

/// Validate every command in one set before exposing it to acquisition.
inline std::vector<float> readSet( const Bank &bank /**< [in] Validated bank identity. */,
                                   size_t      index /**< [in] Zero-based set index. */ );

/// Immutable settings for one acquisition; old config key meanings are retained.
struct Settings
{
    /// Offline-prepared bank directory.
    std::filesystem::path m_bank;

    /// Parent directory for an exclusively created run directory.
    std::filesystem::path m_output;

    /// Expected samples per set; zero selects the bank's value.
    size_t m_samples{ 0 };

    /// Requested prefix of the bank; zero selects all sets.
    size_t m_sets{ 0 };

    /// Minimum source-counter advance after publication; zero still requires a fresh frame.
    uint64_t m_skip{ 8 };

    /// Padded square pupil size; two pixels on each edge are zero.
    size_t m_pupil{ 64 };

    /// Input image rows, preserving the old Eigen map's memory order.
    size_t m_height{ 0 };

    /// Input image columns.
    size_t m_width{ 0 };

    /// Horizontal starts of the left and right pupils.
    std::array<size_t, 2> m_x{ 0, 60 };

    /// Vertical starts of the top and bottom pupils.
    std::array<size_t, 2> m_y{ 0, 60 };

    /// Image scaling, with no command scaling or clipping.
    float m_imageNorm{ 1 };

    /// Maximum time without a fresh source frame during acquisition.
    double m_timeout{ 1 };

    /// Application version and configured source/channel identity for the run record.
    std::string m_context;
};

/// Validate dimensions, crop bounds, scaling and timeout before capture.
inline void validate( const Settings &settings /**< [in] Proposed run settings. */ );

/// Frame identity and command timing for one saved image.
struct Frame
{
    /// Producer's unique frame counter.
    uint64_t m_counter{ 0 };

    /// Producer timestamp in nanoseconds, or zero when not supplied.
    uint64_t m_sourceTime{ 0 };

    /// Counter observed immediately after command publication.
    uint64_t m_issueCounter{ 0 };

    /// Host system-clock timestamp immediately after command publication.
    uint64_t m_issueTime{ 0 };
};

/// Progress values copied under the session mutex for INDI publication.
struct Progress
{
    /// Idle, preparing, collecting, waiting_io, draining, complete, cancelled, or failed.
    std::string m_phase{ "idle" };

    /// Exclusively created run directory, populated before acquisition begins.
    std::string m_directory;

    /// Last failure, never a silent skip.
    std::string m_error;

    /// Accepted images, including partial sets.
    size_t m_saved{ 0 };

    /// Successfully finalized complete sets.
    size_t m_written{ 0 };

    /// Rejected duplicate semaphore callbacks.
    size_t m_duplicates{ 0 };

    /// Rejected snapshots changed during the copy.
    size_t m_torn{ 0 };
};

/// One bounded acquisition; the app owns the ordinary-priority I/O thread.
class Session
{
  public:
    /// Publish a command and return the source counter observed after publication.
    /** A null command releases only this collector's owned channel. */
    using Publisher = std::function<uint64_t( const float * )>;

    /// Checked file output adapter; tests can inject a stalled or failed filesystem.
    using FileWriter = std::function<void( const std::filesystem::path &, const void *, size_t )>;

    /// Construct an inert session with an instrument adapter or a test publisher.
    explicit Session( Publisher  publish /**< [in] Checked command publication adapter. */,
                      FileWriter writer /**< [in] Checked file writer, or an injected test fault. */ = writeFile );

    /// Queue a validated-settings run; no coefficient or file I/O occurs here.
    bool start( const Settings &settings /**< [in] Immutable run settings. */ );

    /// Report whether a new run can start without discarding unsaved failure buffers.
    bool canStart();

    /// Abort capture, release the owned command and queue any partial set.
    void abort( const std::string &reason /**< [in] Empty for operator Stop; otherwise a failure. */ = "" );

    /// Request worker exit after preserving any captured partial set.
    void shutdown();

    /// Perform preflight, preloading and saving on an ordinary-priority thread.
    void ioLoop();

    /// Consume one frame notification; crop only an eligible stable snapshot.
    void
    onFrame( uint64_t                     counter /**< [in] Producer counter before copying. */,
             uint64_t                     sourceTime /**< [in] Producer timestamp. */,
             const float                 *image /**< [in] Producer's float32 image. */,
             const std::function<bool()> &stable /**< [in] Recheck producer counter and write flag after copying. */ );

    /// Abort an acquisition that no longer receives fresh source frames.
    void checkTimeout();

    /// Copy user-visible progress and errors without exposing mutable buffers.
    Progress progress();

  private:
    /// Ownership state of a buffer, protected by m_mutex.
    enum class SlotState
    {
        empty,
        loading,
        ready,
        active,
        complete,
        writing
    };

    /// A set buffer used by exactly one of capture, preload, or writing.
    struct Slot
    {
        /// Current buffer owner.
        SlotState m_state{ SlotState::empty };

        /// Bank set represented by these commands and images.
        size_t m_index{ 0 };

        /// Exact float32 commands; saved labels are the accepted prefix.
        std::vector<float> m_commands;

        /// Padded four-pupil images in the existing binary layout.
        std::vector<float> m_images;

        /// Valid image count, independent of allocated capacity.
        size_t m_count{ 0 };

        /// Timing and frame identity for that same valid prefix.
        std::vector<Frame> m_frames;
    };

    /// Publish the next command or pause at a zeroed set boundary; caller holds m_mutex.
    void issueLocked();

    /// Stop without joining any thread; caller holds m_mutex.
    void abortLocked( const std::string &reason /**< [in] Operator cancellation or failure description. */ );

    /// Check idle/terminal state and retained data; caller holds m_mutex.
    bool canStartLocked() const;

    /// Preflight and run one worker job, with all file work outside m_mutex.
    void run( const Settings &settings /**< [in] Immutable requested settings. */ );

    /// Save a stable complete or partial slot owned by the I/O thread.
    void save( const Slot                  &slot /**< [in] Captured set. */,
               const Bank                  &bank /**< [in] Recorded input bank. */,
               const Settings              &settings /**< [in] Validated settings. */,
               const std::filesystem::path &directory /**< [in] Exclusive run directory. */ );

    /// Shared capture/buffer ownership and progress lock; never held during file I/O.
    std::mutex m_mutex;

    /// Wake the worker when a run, buffer, cancellation or shutdown becomes available.
    std::condition_variable m_condition;

    /// Instrument adapter, called only while capture ownership is serialized.
    Publisher m_publish;

    /// File writer, used exclusively by the ordinary-priority worker.
    FileWriter m_write;

    /// Two buffers bound resident set data independently of run length.
    std::array<Slot, 2> m_slots;

    /// Settings copied at Start and never changed during a run.
    Settings m_settings;

    /// User-visible state.
    Progress m_progress;

    /// Pending run request for the worker.
    bool m_requested{ false };

    /// Worker exit requested.
    bool m_stop{ false };

    /// Capture ended; queued complete and partial sets still need draining.
    bool m_ended{ true };

    /// Operator cancelled this run.
    bool m_cancelled{ false };

    /// Input/capture failure; the worker still attempts to preserve partial data.
    bool m_failed{ false };

    /// Active buffer, or -1 when paused at a set boundary.
    int m_active{ -1 };

    /// Next set to activate, ensuring ordered capture.
    size_t m_next{ 0 };

    /// A command has been issued and is awaiting its image.
    bool m_issued{ false };

    /// Producer counter sampled after the current command publication.
    uint64_t m_issueCounter{ 0 };

    /// Host timestamp for that command.
    uint64_t m_issueTime{ 0 };

    /// Most recently processed unique source counter.
    uint64_t m_lastCounter{ 0 };

    /// Whether m_lastCounter belongs to this run.
    bool m_haveCounter{ false };

    /// Receipt time of the most recent fresh frame for the watchdog.
    std::chrono::steady_clock::time_point m_lastFrame;
};

inline size_t product( size_t a, size_t b )
{
    if( b && a > std::numeric_limits<size_t>::max() / b )
        throw std::invalid_argument( "size overflow" );
    return a * b;
}

inline bool finite( float value )
{
    uint32_t bits;
    std::memcpy( &bits, &value, sizeof( bits ) );
    return ( bits & 0x7f800000U ) != 0x7f800000U;
}

inline bool finite( double value )
{
    uint64_t bits;
    std::memcpy( &bits, &value, sizeof( bits ) );
    return ( bits & 0x7ff0000000000000ULL ) != 0x7ff0000000000000ULL;
}

inline std::string digest( const void *data, size_t bytes )
{
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned      length{ 0 };
    if( EVP_Digest( data, bytes, hash, &length, EVP_sha256(), nullptr ) != 1 )
        throw std::runtime_error( "SHA-256 calculation failed" );
    std::ostringstream text;
    for( unsigned i = 0; i < length; ++i )
        text << std::hex << std::setw( 2 ) << std::setfill( '0' ) << unsigned( hash[i] );
    return text.str();
}

inline std::vector<float> readFloats( const std::filesystem::path &path, size_t count )
{
    const uint32_t endian{ 1 };
    if( *reinterpret_cast<const unsigned char *>( &endian ) != 1 || sizeof( float ) != 4 ||
        !std::numeric_limits<float>::is_iec559 )
        throw std::runtime_error( "banks require an IEEE float32 little-endian host" );
    const size_t bytes = product( count, sizeof( float ) );
    if( bytes > size_t( std::numeric_limits<std::streamsize>::max() ) || std::filesystem::file_size( path ) != bytes )
        throw std::runtime_error( "wrong float32 payload size: " + path.string() );
    std::ifstream      file( path, std::ios::binary );
    std::vector<float> result( count );
    file.read( reinterpret_cast<char *>( result.data() ), bytes );
    if( !file || file.peek() != std::char_traits<char>::eof() )
        throw std::runtime_error( "incomplete or changed payload: " + path.string() );
    return result;
}

inline void writeFile( const std::filesystem::path &path, const void *data, size_t bytes )
{
    const auto temporary = path.string() + ".part";
    int        fd        = ::open( temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0640 );
    if( fd < 0 )
        throw std::runtime_error( "cannot create " + temporary + ": " + std::strerror( errno ) );
    try
    {
        const char *position = static_cast<const char *>( data );
        size_t      left     = bytes;
        while( left )
        {
            const ssize_t n = ::write( fd, position, std::min( left, size_t( 1 << 20 ) ) );
            if( n < 0 && errno == EINTR )
                continue;
            if( n <= 0 )
                throw std::runtime_error( "write failed: " + temporary );
            position += n;
            left -= n;
        }
        if( ::fsync( fd ) != 0 )
            throw std::runtime_error( "flush failed: " + temporary );
        const int result = ::close( fd );
        fd               = -1;
        if( result != 0 )
            throw std::runtime_error( "close failed: " + temporary );
        if( ::link( temporary.c_str(), path.c_str() ) != 0 )
            throw std::runtime_error( "cannot finalize without replacement: " + path.string() );
        if( ::unlink( temporary.c_str() ) != 0 )
            throw std::runtime_error( "cannot remove finalized temporary: " + temporary );
    }
    catch( ... )
    {
        if( fd >= 0 )
            ::close( fd );
        // Keep incomplete files for diagnosis; never replace them on a later attempt.
        throw;
    }
}

inline void writeText( const std::filesystem::path &path, const std::string &text )
{
    writeFile( path, text.data(), text.size() );
}

inline Bank readBank( const std::filesystem::path &path )
{
    Bank bank;
    bank.m_path = std::filesystem::absolute( path );
    std::ifstream input( bank.m_path / "manifest.conf" );
    if( !input )
        throw std::runtime_error( "missing offline bank manifest" );
    const std::string text( ( std::istreambuf_iterator<char>( input ) ), {} );
    if( input.bad() )
        throw std::runtime_error( "bank manifest read failed" );
    bank.m_manifestHash = digest( text.data(), text.size() );
    std::map<std::string, std::string> fields;
    std::istringstream                 lines( text );
    std::string                        line;
    while( std::getline( lines, line ) )
    {
        if( line.empty() )
            continue;
        const auto eq = line.find( '=' );
        if( eq == std::string::npos || !fields.emplace( line.substr( 0, eq ), line.substr( eq + 1 ) ).second )
            throw std::runtime_error( "malformed or duplicated bank manifest field" );
    }
    if( fields.at( "format" ) != "dlDataCollection-1" || fields.at( "dtype" ) != "float32-le" ||
        fields.at( "units" ) != "um_surface" || fields.at( "modes" ) != "2500" )
        throw std::runtime_error( "unsupported bank layout or command units" );
    auto count = [&]( const std::string &name )
    {
        const auto &value = fields.at( name );
        if( value.empty() || value.find_first_not_of( "0123456789" ) != std::string::npos )
            throw std::runtime_error( "invalid bank count: " + name );
        const auto n = std::stoull( value );
        if( !n || n > std::numeric_limits<size_t>::max() )
            throw std::runtime_error( "invalid bank count" );
        return size_t( n );
    };
    bank.m_samples = count( "samples" );
    bank.m_sets    = count( "sets" );
    product( bank.m_samples, 2500 );
    bank.m_maskHash = fields.at( "mask_sha256" );
    std::ifstream provenance( bank.m_path / "generation.json" );
    if( !provenance )
        throw std::runtime_error( "missing bank generation provenance" );
    const std::string generation( ( std::istreambuf_iterator<char>( provenance ) ), {} );
    bank.m_generationHash = fields.at( "generation_sha256" );
    if( provenance.bad() || digest( generation.data(), generation.size() ) != bank.m_generationHash )
        throw std::runtime_error( "bank generation provenance checksum mismatch" );
    bank.m_mask = readFloats( bank.m_path / "actuator_mask.bin", 2500 );
    if( digest( bank.m_mask.data(), 10000 ) != bank.m_maskHash )
        throw std::runtime_error( "actuator mask checksum mismatch" );
    bool any = false;
    for( float value : bank.m_mask )
    {
        if( !finite( value ) || ( value != 0.F && value != 1.F ) )
            throw std::runtime_error( "actuator mask must be binary" );
        any |= value == 1.F;
    }
    if( !any )
        throw std::runtime_error( "empty controllable actuator mask" );
    for( size_t i = 0; i < bank.m_sets; ++i )
    {
        const auto hash = fields.at( "set_" + std::to_string( i ) + "_sha256" );
        if( hash.size() != 64 || hash.find_first_not_of( "0123456789abcdef" ) != std::string::npos )
            throw std::runtime_error( "invalid coefficient checksum" );
        bank.m_hashes.push_back( hash );
    }
    if( fields.size() != 8 + bank.m_sets )
        throw std::runtime_error( "unknown bank manifest field" );
    return bank;
}

inline std::vector<float> readSet( const Bank &bank, size_t index )
{
    if( index >= bank.m_sets )
        throw std::invalid_argument( "set index outside bank" );
    auto values = readFloats( bank.m_path / ( "modeval_dataset_" + std::to_string( index ) + ".bin" ),
                              product( bank.m_samples, 2500 ) );
    if( digest( values.data(), product( values.size(), sizeof( float ) ) ) != bank.m_hashes.at( index ) )
        throw std::runtime_error( "coefficient checksum mismatch in set " + std::to_string( index ) );
    for( size_t i = 0; i < values.size(); ++i )
        if( !finite( values[i] ) || ( bank.m_mask[i % 2500] == 0.F && values[i] != 0.F ) )
            throw std::runtime_error( "non-finite or excluded-actuator command in set " + std::to_string( index ) );
    return values;
}

inline void validate( const Settings &s )
{
    if( s.m_bank.empty() || s.m_output.empty() || s.m_pupil <= 4 || !s.m_width || !s.m_height ||
        !finite( s.m_imageNorm ) || s.m_imageNorm <= 0.F || !finite( s.m_timeout ) || !( s.m_timeout > 0 ) ||
        s.m_timeout > 3600 )
        throw std::invalid_argument( "invalid collection dimensions, scaling, paths or timeout" );
    const size_t crop = s.m_pupil - 4;
    for( size_t p = 0; p < 2; ++p )
        if( crop > s.m_width || crop > s.m_height || s.m_x[p] > s.m_width - crop || s.m_y[p] > s.m_height - crop )
            throw std::invalid_argument( "pupil crop is outside the source image" );
    product( product( 4, product( s.m_pupil, s.m_pupil ) ), std::max( size_t( 1 ), s.m_samples ) );
}

inline Session::Session( Publisher publish, FileWriter writer )
    : m_publish( std::move( publish ) ), m_write( std::move( writer ) )
{
}

inline bool Session::start( const Settings &settings )
{
    validate( settings );
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        if( !canStartLocked() )
            return false;
        m_settings         = settings;
        m_progress         = Progress{};
        m_progress.m_phase = "preparing";
        m_requested        = true;
        m_ended            = false;
        m_cancelled        = false;
        m_failed           = false;
        m_active           = -1;
        m_next             = 0;
        m_issued           = false;
        m_haveCounter      = false;
    }
    m_condition.notify_all();
    return true;
}

inline bool Session::canStartLocked() const
{
    if( m_stop || m_requested )
        return false;
    if( m_progress.m_phase == "idle" || m_progress.m_phase == "complete" || m_progress.m_phase == "cancelled" )
        return true;
    if( m_progress.m_phase != "failed" )
        return false;
    for( const auto &slot : m_slots )
        if( slot.m_count && ( slot.m_state == SlotState::complete || slot.m_state == SlotState::writing ||
                              slot.m_state == SlotState::active ) )
            return false;
    return true;
}

inline bool Session::canStart()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return canStartLocked();
}

inline void Session::abortLocked( const std::string &reason )
{
    if( m_ended )
        return;
    m_failed |= !reason.empty();
    m_cancelled |= reason.empty();
    if( !reason.empty() )
        m_progress.m_error = reason;
    if( m_active >= 0 )
    {
        auto &slot   = m_slots[m_active];
        slot.m_state = slot.m_count ? SlotState::complete : SlotState::empty;
        m_active     = -1;
    }
    try
    {
        m_publish( nullptr );
    }
    catch( const std::exception &e )
    {
        m_failed = true;
        m_progress.m_error += std::string( "; channel release: " ) + e.what();
    }
    m_issued           = false;
    m_ended            = true;
    m_progress.m_phase = "draining";
    m_condition.notify_all();
}

inline void Session::abort( const std::string &reason )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    abortLocked( reason );
}

inline void Session::shutdown()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        abortLocked( "" );
        m_stop = true;
    }
    m_condition.notify_all();
}

inline Progress Session::progress()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_progress;
}

inline void Session::issueLocked()
{
    if( m_active < 0 )
    {
        for( size_t i = 0; i < m_slots.size(); ++i )
            if( m_slots[i].m_state == SlotState::ready && m_slots[i].m_index == m_next )
            {
                m_active           = int( i );
                m_slots[i].m_state = SlotState::active;
                ++m_next;
                break;
            }
        if( m_active < 0 )
        {
            m_progress.m_phase = "waiting_io";
            return;
        }
    }
    auto &slot     = m_slots[m_active];
    m_issueCounter = m_publish( slot.m_commands.data() + 2500 * slot.m_count );
    m_issueTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
            .count();
    m_issued           = true;
    m_progress.m_phase = "collecting";
}

inline void
Session::onFrame( uint64_t counter, uint64_t sourceTime, const float *image, const std::function<bool()> &stable )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_ended || ( m_progress.m_phase != "collecting" && m_progress.m_phase != "waiting_io" ) )
        return;
    try
    {
        if( !image )
            throw std::runtime_error( "source frame pointer is null" );
        if( m_haveCounter && counter == m_lastCounter )
        {
            ++m_progress.m_duplicates;
            return;
        }
        if( m_haveCounter && counter < m_lastCounter )
            throw std::runtime_error( "source frame counter reset" );
        m_lastCounter = counter;
        m_haveCounter = true;
        m_lastFrame   = std::chrono::steady_clock::now();
        if( !m_issued )
        {
            issueLocked();
            return;
        }
        if( counter < m_issueCounter )
            throw std::runtime_error( "source counter reset after publication" );
        const uint64_t advance = counter - m_issueCounter;
        if( advance == 0 || advance < m_settings.m_skip )
            return;
        auto        &slot = m_slots[m_active];
        const size_t p = m_settings.m_pupil, area = product( p, p ), stride = 4 * area;
        float       *dest = slot.m_images.data() + stride * slot.m_count;
        std::fill( dest, dest + stride, 0.F );
        for( size_t col = 0; col < p - 4; ++col )
            for( size_t row = 0; row < p - 4; ++row )
                for( size_t pupil = 0; pupil < 4; ++pupil )
                {
                    const float value =
                        m_settings.m_imageNorm * image[m_settings.m_y[pupil / 2] + row +
                                                       ( m_settings.m_x[pupil % 2] + col ) * m_settings.m_height];
                    if( !finite( value ) )
                        throw std::runtime_error( "non-finite preprocessed WFS image" );
                    dest[pupil * area + ( col + 2 ) * p + row + 2] = value;
                }
        if( !stable() )
        {
            ++m_progress.m_torn;
            return;
        }
        slot.m_frames.push_back( Frame{ counter, sourceTime, m_issueCounter, m_issueTime } );
        ++slot.m_count;
        ++m_progress.m_saved;
        m_issued = false;
        if( slot.m_count == m_settings.m_samples )
        {
            slot.m_state = SlotState::complete;
            m_active     = -1;
            m_publish( nullptr );
            if( m_next == m_settings.m_sets )
            {
                m_ended            = true;
                m_progress.m_phase = "draining";
                m_condition.notify_all();
                return;
            }
            m_condition.notify_all();
        }
        issueLocked();
    }
    catch( const std::exception &e )
    {
        abortLocked( e.what() );
    }
}

inline void Session::checkTimeout()
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( !m_ended && ( m_progress.m_phase == "collecting" || m_progress.m_phase == "waiting_io" ) &&
        std::chrono::duration<double>( std::chrono::steady_clock::now() - m_lastFrame ).count() > m_settings.m_timeout )
        abortLocked( "source frame timeout" );
}

inline void
Session::save( const Slot &slot, const Bank &bank, const Settings &s, const std::filesystem::path &directory )
{
    const std::string stem =
        ( slot.m_count == s.m_samples ? "dataset_" : "partial_dataset_" ) + std::to_string( slot.m_index );
    std::ostringstream frames;
    frames << "sample,source_counter,source_time_ns,issue_counter,issue_time_ns\n";
    for( size_t i = 0; i < slot.m_count; ++i )
    {
        const auto &f = slot.m_frames[i];
        frames << i << ',' << f.m_counter << ',' << f.m_sourceTime << ',' << f.m_issueCounter << ',' << f.m_issueTime
               << '\n';
    }
    // Save identities/counts first, so an interrupted payload still has an explicit accepted prefix.
    const auto frameText = frames.str();
    m_write( directory / ( "frames_" + stem + ".csv" ), frameText.data(), frameText.size() );
    const size_t imageBytes   = product( product( slot.m_count, 4 * s.m_pupil * s.m_pupil ), sizeof( float ) );
    const size_t commandBytes = product( product( slot.m_count, 2500 ), sizeof( float ) );
    m_write( directory / ( "images_" + stem + ".bin" ), slot.m_images.data(), imageBytes );
    m_write( directory / ( "commands_" + stem + ".bin" ), slot.m_commands.data(), commandBytes );
    std::ostringstream marker;
    marker << "samples=" << slot.m_count << "\nimage_shape=4," << s.m_pupil << ',' << s.m_pupil
           << "\ncommand_shape=2500\nunits=um_surface\nbank_set_sha256=" << bank.m_hashes.at( slot.m_index )
           << "\nimages_sha256=" << digest( slot.m_images.data(), imageBytes )
           << "\ncommands_sha256=" << digest( slot.m_commands.data(), commandBytes ) << '\n';
    const auto markerText = marker.str();
    m_write( directory / ( ( slot.m_count == s.m_samples ? "complete_" : "" ) + stem + ".conf" ),
             markerText.data(),
             markerText.size() );
}

inline void Session::run( const Settings &request )
{
    Settings settings = request;
    Bank     bank     = readBank( settings.m_bank );
    if( settings.m_samples && settings.m_samples != bank.m_samples )
        throw std::runtime_error( "Nperset disagrees with bank" );
    settings.m_samples = bank.m_samples;
    if( !settings.m_sets )
        settings.m_sets = bank.m_sets;
    if( settings.m_sets > bank.m_sets )
        throw std::runtime_error( "Nset exceeds bank" );
    validate( settings );
    for( size_t i = 0; i < settings.m_sets; ++i )
    {
        readSet( bank, i );
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            if( m_ended )
                return;
        }
    }
    std::filesystem::create_directories( settings.m_output );
    const auto stamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::system_clock::now().time_since_epoch() )
            .count();
    auto directory = settings.m_output / ( "run_" + std::to_string( stamp ) + "_" + std::to_string( ::getpid() ) );
    if( !std::filesystem::create_directory( directory ) )
        throw std::runtime_error( "run directory collision" );
    std::ostringstream manifest;
    manifest << "format=dlDataCollection-run-1\nbank=" << bank.m_path.string()
             << "\nbank_manifest_sha256=" << bank.m_manifestHash << "\nmask_sha256=" << bank.m_maskHash
             << "\ngeneration_sha256=" << bank.m_generationHash << "\nsamples_per_set=" << settings.m_samples
             << "\nsets=" << settings.m_sets << "\nskip=" << settings.m_skip << "\nimageNorm=" << settings.m_imageNorm
             << "\npupil=" << settings.m_pupil << "\nsource_shape=" << settings.m_height << ',' << settings.m_width
             << "\npupil_x=" << settings.m_x[0] << ',' << settings.m_x[1] << "\npupil_y=" << settings.m_y[0] << ','
             << settings.m_y[1] << '\n'
             << settings.m_context;
    manifest << "source_time_clock=producer_defined\nissue_time_clock=unix_system_clock\n";
    writeText( directory / "run.conf", manifest.str() );
    for( auto &slot : m_slots )
    {
        slot.m_images.resize(
            product( settings.m_samples, product( 4, product( settings.m_pupil, settings.m_pupil ) ) ) );
        slot.m_frames.clear();
        slot.m_frames.reserve( settings.m_samples );
        slot.m_state = SlotState::empty;
        slot.m_count = 0;
    }
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_settings             = settings;
        m_progress.m_directory = directory.string();
    }
    try
    {
        size_t loaded = 0;
        while( true )
        {
            int  writing = -1, loading = -1;
            bool finished = false;
            { //mutex scope
                std::unique_lock<std::mutex> lock( m_mutex );
                for( size_t i = 0; i < m_slots.size(); ++i )
                    if( m_slots[i].m_state == SlotState::complete )
                    {
                        writing            = int( i );
                        m_slots[i].m_state = SlotState::writing;
                        break;
                    }
                if( writing < 0 && !m_ended && loaded < settings.m_sets )
                    for( size_t i = 0; i < m_slots.size(); ++i )
                        if( m_slots[i].m_state == SlotState::empty )
                        {
                            loading            = int( i );
                            m_slots[i].m_state = SlotState::loading;
                            break;
                        }
                finished = m_ended && writing < 0;
                if( !finished && writing < 0 && loading < 0 )
                {
                    m_condition.wait_for( lock, std::chrono::milliseconds( 50 ) );
                    continue;
                }
            }
            if( finished )
                break;
            if( writing >= 0 )
            {
                auto &slot = m_slots[writing];
                save( slot, bank, settings, directory );
                { //mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    if( slot.m_count == settings.m_samples )
                        ++m_progress.m_written;
                    slot.m_state = SlotState::empty;
                }
            }
            else
            {
                auto  values    = readSet( bank, loaded );
                auto &slot      = m_slots[loading];
                slot.m_commands = std::move( values );
                slot.m_frames.clear();
                slot.m_count = 0;
                slot.m_index = loaded++;
                { //mutex scope
                    std::lock_guard<std::mutex> lock( m_mutex );
                    slot.m_state = SlotState::ready;
                    if( !m_ended && m_progress.m_phase == "preparing" )
                    {
                        m_progress.m_phase = "collecting";
                        m_lastFrame        = std::chrono::steady_clock::now();
                    }
                }
            }
        }
    }
    catch( const std::exception &e )
    {
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            if( m_ended )
            {
                m_failed           = true;
                m_progress.m_error = e.what();
            }
            else
                abortLocked( e.what() );
        }
        // Preserve a different active partial set even when one output payload failed.
        for( auto &slot : m_slots )
            if( slot.m_state == SlotState::complete && slot.m_count )
            {
                try
                {
                    save( slot, bank, settings, directory );
                    std::lock_guard<std::mutex> lock( m_mutex );
                    slot.m_state = SlotState::empty;
                }
                catch( const std::exception & )
                {
                } // Retain owned buffers and any .part file on further I/O failure.
            }
    }
    const auto status = progress();
    writeText( directory / "finished.conf",
               std::string( "status=" ) +
                   ( m_failed      ? "failed"
                     : m_cancelled ? "cancelled"
                                   : "complete" ) +
                   "\nsaved=" + std::to_string( status.m_saved ) + "\nerror=" + status.m_error + "\n" );
}

inline void Session::ioLoop()
{
    while( true )
    {
        Settings settings;
        { //mutex scope
            std::unique_lock<std::mutex> lock( m_mutex );
            m_condition.wait( lock, [this] { return m_stop || m_requested; } );
            if( m_stop && !m_requested )
                return;
            settings    = m_settings;
            m_requested = false;
        }
        try
        {
            run( settings );
        }
        catch( const std::exception &e )
        {
            { //mutex scope
                std::lock_guard<std::mutex> lock( m_mutex );
                if( m_ended )
                {
                    m_failed           = true;
                    m_progress.m_error = e.what();
                }
                else
                    abortLocked( e.what() );
            }
            // Completed or active buffers remain owned on a write failure; Start is disabled.
        }
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );
            m_ended            = true;
            m_progress.m_phase = m_failed ? "failed" : m_cancelled ? "cancelled" : "complete";
            if( m_stop )
                return;
        }
    }
}

} // namespace dlCollection
} // namespace app
} // namespace MagAOX
#endif // dlCollection_hpp
