/** \file flipperCtrl_test.cpp
 * \brief Behavioral and failure-contract tests for flipper configuration, parking, FSM, INDI, and telemetry.
 * \author Jared R. Males (jaredmales@gmail.com)
 * \ingroup flipperCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"

#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <poll.h>
#include <sys/socket.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace flipperHarness
{
/// Captured log severity and formatted message.
struct LogEntry
{
    /// Priority assigned by the real app call site.
    flatlogs::logPrioT m_priority;

    /// Message formatted by the real log type.
    std::string m_message;
};

/// Deterministic transport and filesystem failures for the production controller.
struct Faults
{
    /// Property operation whose startup result should fail.
    enum class PropertyFailure
    {
        none,        ///< Allow all property operations.
        selection,   ///< Fail selection creation.
        newProperty, ///< Fail callback registration.
        readOnly     ///< Fail parked-property registration.
    };

    /// Ordered dependency calls made by production configuration and lifecycle methods.
    std::vector<std::string> m_calls;

    /// Discovery results; an empty queue reports an absent device.
    std::deque<int> m_discoveryResults;

    /// Connection results; an empty queue succeeds with a harmless descriptor.
    std::deque<int> m_connectResults;

    /// Result returned after loading real USB configuration.
    int m_usbLoadResult{ TTY_E_DEVNOTFOUND };

    /// Result returned after loading real I/O configuration.
    int m_ioLoadResult{ 0 };

    /// Result returned by telemetry configuration setup.
    int m_telemSetupResult{ 0 };

    /// Result returned by telemetry configuration loading.
    int m_telemLoadResult{ 0 };

    /// Result returned by threadless telemetry startup.
    int m_telemStartupResult{ 0 };

    /// Result returned by telemetry scheduling before checking record times.
    int m_telemLogicResult{ 0 };

    /// Result returned by telemetry shutdown.
    int m_telemShutdownResult{ 0 };

    /// Selected fatal property operation.
    PropertyFailure m_propertyFailure{ PropertyFailure::none };

    /// Whether temporary-file creation should fail with ENOSPC.
    bool m_failedTemporary{ false };

    /// Whether setting the temporary file's mode should fail.
    bool m_failedMode{ false };

    /// Whether persistence writes should report zero bytes.
    bool m_zeroWrite{ false };

    /// Synchronous event injected after a descriptor has actually closed.
    std::function<void( unsigned )> m_afterClose;

    /// Synchronous power transition injected during a status read.
    std::function<void()> m_beforeRead;

    /// Commands written to the serial transport.
    std::vector<std::string> m_commands;

    /// Complete or partial receive chunks supplied by the transport.
    std::deque<std::string> m_replies;

    /// Hook observing the installed state before a serial write.
    std::function<void( const std::string & )> m_beforeWrite;

    /// Message ID whose serial write should fail; -1 allows every write.
    int m_failedCommand{ -1 };

    /// Number of serial read calls.
    unsigned m_reads{ 0 };

    /// Number of file synchronization calls.
    unsigned m_syncs{ 0 };

    /// Synchronization call to fail; zero disables the fault.
    unsigned m_failedSync{ 0 };

    /// Whether atomic rename should fail.
    bool m_failedRename{ false };

    /// Whether file writes should fail.
    bool m_failedFileWrite{ false };

    /// Whether one interrupted file write should be injected.
    bool m_interruptWrite{ false };

    /// Whether file writes should return one byte at a time.
    bool m_shortWrites{ false };

    /// Whether serial I/O should use the real tty utilities over a local socket.
    bool m_nativeSerial{ false };

    /// Number of file/directory close calls.
    unsigned m_closes{ 0 };

    /// Close call whose result should report failure after releasing the descriptor.
    unsigned m_failedClose{ 0 };
};

/// Current transport and filesystem injection state.
Faults g_faults;

/// App logs captured without starting a process logger.
std::vector<LogEntry> g_logs;

/// Telemetry payloads emitted by the production recordStage implementation.
std::vector<std::vector<uint8_t>> g_telemetry;

/// Reset capture and failure state between independent tests.
void reset();

/// Form a documented APT status packet.
std::string status( uint32_t bits /**< [in] little-endian status flags */ );

/// Read a state file's exact contents.
std::string contents( const std::filesystem::path &path /**< [in] state file path */ );

/// Count power-on mismatch warnings in the captured logs.
size_t mismatchWarnings();

/// Count a dependency operation in the captured call sequence.
size_t calls( const std::string &name /**< [in] operation label */ );

/// Count captured logs of a particular severity containing a diagnostic.
size_t logs( flatlogs::logPrioT priority /**< [in] required severity */,
             const std::string &message /**< [in] diagnostic substring */ );

/// Own a unique temporary test directory.
class Directory
{
  public:
    /// Create a temporary directory beneath /tmp.
    Directory();

    /// Remove the test directory without throwing.
    ~Directory();

    /// Root of this test's isolated app state.
    std::filesystem::path m_path;
};
} // namespace flipperHarness

namespace MagAOX
{
namespace app
{
/// App base capturing calls from the controller while retaining the real configuration, FSM, and INDI support.
template <bool useINDI>
class flipperTestApp : public MagAOXApp<useINDI>
{
  public:
    /// Let the real telemetry configuration helper read the test app's configuration name.
    using MagAOXApp<useINDI>::m_configName;

    /// Preserve the shared base's other registration overloads.
    using MagAOXApp<useINDI>::registerIndiPropertyNew;
    using MagAOXApp<useINDI>::registerIndiPropertyReadOnly;

    /// INDI registration callback type, preserving the real driver interface.
    using Callback = int ( * )( void *, const pcf::IndiProperty & );

    /// Construct the real app base after suppressing its process logger.
    flipperTestApp( const std::string &sha /**< [in] repository revision */,
                    bool               modified /**< [in] working-tree flag */ );

    /// Suppress shared-library logs before the real base constructor runs.
    static const std::string &quiet( const std::string &sha /**< [in] revision passed through to the base */ );

    /// Capture an application log using its real message format.
    template <typename logT, int retval = 0>
    static int log( const typename logT::messageT &msg /**< [in] log payload */,
                    logPrioT                       level = logPrio::LOG_DEFAULT /**< [in] requested severity */ );

    /// Capture a default-constructed application log.
    template <typename logT, int retval = 0>
    static int log( logPrioT level = logPrio::LOG_DEFAULT /**< [in] requested severity */ );

    /// Create a real selection unless its specific startup failure is requested.
    int createStandardIndiSelectionSw( pcf::IndiProperty              &property /**< [out] selection to initialize */,
                                       const std::string              &name /**< [in] property name */,
                                       const std::vector<std::string> &elements /**< [in] selection names */ );

    /// Register a real callback unless its specific startup failure is requested.
    int registerIndiPropertyNew( pcf::IndiProperty &property /**< [in/out] initialized property */,
                                 Callback           callback /**< [in] real production callback */ );

    /// Register a real read-only property unless its startup failure is requested.
    int registerIndiPropertyReadOnly( pcf::IndiProperty &property /**< [in/out] initialized parked property */ );
};

template <bool useINDI>
flipperTestApp<useINDI>::flipperTestApp( const std::string &sha, bool modified )
    : MagAOXApp<useINDI>( quiet( sha ), modified )
{
}

template <bool useINDI>
const std::string &flipperTestApp<useINDI>::quiet( const std::string &sha )
{
    MagAOXApp<useINDI>::m_log.m_logLevel = logPrio::LOG_EMERGENCY;
    return sha;
}

template <bool useINDI>
template <typename logT, int retval>
int flipperTestApp<useINDI>::log( const typename logT::messageT &msg, logPrioT level )
{
    if( level == logPrio::LOG_DEFAULT )
        level = logT::defaultLevel;
    flipperHarness::g_logs.push_back(
        { level, logT::msgString( msg.builder.GetBufferPointer(), msg.builder.GetSize() ) } );
    return retval;
}

template <bool useINDI>
template <typename logT, int retval>
int flipperTestApp<useINDI>::log( logPrioT level )
{
    return log<logT, retval>( typename logT::messageT(), level );
}

template <bool useINDI>
int flipperTestApp<useINDI>::createStandardIndiSelectionSw( pcf::IndiProperty              &property,
                                                            const std::string              &name,
                                                            const std::vector<std::string> &elements )
{
    flipperHarness::g_faults.m_calls.push_back( "selection" );
    if( flipperHarness::g_faults.m_propertyFailure == flipperHarness::Faults::PropertyFailure::selection )
        return -1;
    return MagAOXApp<useINDI>::createStandardIndiSelectionSw( property, name, elements );
}

template <bool useINDI>
int flipperTestApp<useINDI>::registerIndiPropertyNew( pcf::IndiProperty &property, Callback callback )
{
    flipperHarness::g_faults.m_calls.push_back( "register-new" );
    if( flipperHarness::g_faults.m_propertyFailure == flipperHarness::Faults::PropertyFailure::newProperty )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertyNew( property, callback );
}

template <bool useINDI>
int flipperTestApp<useINDI>::registerIndiPropertyReadOnly( pcf::IndiProperty &property )
{
    flipperHarness::g_faults.m_calls.push_back( "register-read-only" );
    if( flipperHarness::g_faults.m_propertyFailure == flipperHarness::Faults::PropertyFailure::readOnly )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertyReadOnly( property );
}

namespace dev
{
/// Telemetry sink that records actual FlatBuffer payloads and supplies controllable scheduled deadlines.
template <class derivedT>
class flipperTestTelemeter : public telemeter<derivedT>
{
  public:
    /// Number of times the app invokes telemetry scheduling.
    unsigned m_schedules{ 0 };

    /// Whether the next schedule check should force a telemetry record.
    bool m_due{ false };

    /// Delegate real telemetry configuration setup while allowing a selected failure.
    int setupConfig( mx::app::appConfigurator &config /**< [in/out] app configurator */ );

    /// Delegate real telemetry configuration loading while allowing a selected failure.
    int loadConfig( mx::app::appConfigurator &config /**< [in] app configurator */ );

    /// Start the test telemetry sink without a background thread.
    int appStartup();

    /// Exercise the controller's real scheduling dispatch.
    int appLogic();

    /// Stop the test telemetry sink.
    int appShutdown();

    /// Apply the next injected telemetry deadline.
    int checkRecordTimes( const telem_stage &type /**< [in] stage telemetry type selector */ );

    /// Capture a serialized telemetry payload.
    template <typename telT>
    int telem( const typename telT::messageT &msg /**< [in] real telemetry payload */ );
};

template <class derivedT>
int flipperTestTelemeter<derivedT>::setupConfig( mx::app::appConfigurator &config )
{
    flipperHarness::g_faults.m_calls.push_back( "telem-setup" );
    if( flipperHarness::g_faults.m_telemSetupResult < 0 )
        return flipperHarness::g_faults.m_telemSetupResult;
    return telemeter<derivedT>::setupConfig( config );
}

template <class derivedT>
int flipperTestTelemeter<derivedT>::loadConfig( mx::app::appConfigurator &config )
{
    flipperHarness::g_faults.m_calls.push_back( "telem-load" );
    if( flipperHarness::g_faults.m_telemLoadResult < 0 )
        return flipperHarness::g_faults.m_telemLoadResult;
    return telemeter<derivedT>::loadConfig( config );
}

template <class derivedT>
int flipperTestTelemeter<derivedT>::appStartup()
{
    flipperHarness::g_faults.m_calls.push_back( "telem-startup" );
    return flipperHarness::g_faults.m_telemStartupResult;
}

template <class derivedT>
int flipperTestTelemeter<derivedT>::appShutdown()
{
    flipperHarness::g_faults.m_calls.push_back( "telem-shutdown" );
    return flipperHarness::g_faults.m_telemShutdownResult;
}

template <class derivedT>
int flipperTestTelemeter<derivedT>::appLogic()
{
    ++m_schedules;
    flipperHarness::g_faults.m_calls.push_back( "telem-logic" );
    if( flipperHarness::g_faults.m_telemLogicResult < 0 )
        return flipperHarness::g_faults.m_telemLogicResult;
    return static_cast<derivedT *>( this )->checkRecordTimes();
}

template <class derivedT>
int flipperTestTelemeter<derivedT>::checkRecordTimes( const telem_stage &type )
{
    if( !m_due )
        return 0;
    m_due = false;
    return static_cast<derivedT *>( this )->recordTelem( &type );
}

template <class derivedT>
template <typename telT>
int flipperTestTelemeter<derivedT>::telem( const typename telT::messageT &msg )
{
    auto *begin = msg.builder.GetBufferPointer();
    flipperHarness::g_telemetry.emplace_back( begin, begin + msg.builder.GetSize() );
    return 0;
}

/// I/O base retaining real configuration while injecting its error contract.
class flipperTestIODevice : public ioDevice
{
  public:
    /// Load real timeout values before applying a selected failure.
    int loadConfig( mx::app::appConfigurator &config /**< [in] app configurator */ );
};

int flipperTestIODevice::loadConfig( mx::app::appConfigurator &config )
{
    flipperHarness::g_faults.m_calls.push_back( "io-load" );
    int rv = ioDevice::loadConfig( config );
    return rv < 0 ? rv : flipperHarness::g_faults.m_ioLoadResult;
}
} // namespace dev
} // namespace app

namespace tty
{
/// USB base retaining real configuration and supplying deterministic discovery/connection results.
class flipperTestUSBDevice : public usbDevice
{
  public:
    /// Load real USB fields before returning the selected discovery result.
    int loadConfig( mx::app::appConfigurator &config /**< [in] app configurator */ );

    /// Consume a discovery result without depending on attached hardware.
    int getDeviceName();

    /// Consume a connection result and own a harmless descriptor on success.
    int connect();
};

int flipperTestUSBDevice::loadConfig( mx::app::appConfigurator &config )
{
    flipperHarness::g_faults.m_calls.push_back( "usb-load" );
    usbDevice::loadConfig( config );
    return flipperHarness::g_faults.m_usbLoadResult;
}

int flipperTestUSBDevice::getDeviceName()
{
    auto &faults = flipperHarness::g_faults;
    faults.m_calls.push_back( "discover" );
    int rv = TTY_E_DEVNOTFOUND;
    if( !faults.m_discoveryResults.empty() )
    {
        rv = faults.m_discoveryResults.front();
        faults.m_discoveryResults.pop_front();
    }
    if( rv == 0 )
        m_deviceName = "/dev/flipper-test";
    return rv;
}

int flipperTestUSBDevice::connect()
{
    auto &faults = flipperHarness::g_faults;
    faults.m_calls.push_back( "connect" );
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    int rv        = 0;
    if( !faults.m_connectResults.empty() )
    {
        rv = faults.m_connectResults.front();
        faults.m_connectResults.pop_front();
    }
    if( rv == 0 )
    {
        m_fileDescrip = ::open( "/dev/null", O_RDWR | O_CLOEXEC );
        if( m_fileDescrip < 0 )
            return TTY_E_ERRORONWRITE;
    }
    return rv;
}

/// Supply a controlled serial write result while observing the real command bytes.
int flipperTestWrite( const std::string &command /**< [in] bytes sent by the controller */,
                      int                fd /**< [in] descriptor used only by the native transport test */,
                      int                timeout /**< [in] write timeout used only by the native transport test */ );

/// Supply complete or fragmented device replies to the production packet reader.
int flipperTestRead( std::string &response /**< [out] injected receive chunk */,
                     int          bytes /**< [in] requested receive length */,
                     int          fd /**< [in] descriptor used only by the native transport test */,
                     int          timeout /**< [in] read timeout used only by the native transport test */ );

int flipperTestWrite( const std::string &command, int fd, int timeout )
{
    auto &faults = flipperHarness::g_faults;
    if( faults.m_beforeWrite )
        faults.m_beforeWrite( command );
    faults.m_commands.push_back( command );
    int id = static_cast<unsigned char>( command[0] ) | ( static_cast<unsigned char>( command[1] ) << 8 );
    if( id == faults.m_failedCommand )
        return TTY_E_ERRORONWRITE;
    return faults.m_nativeSerial ? ttyWrite( command, fd, timeout ) : 0;
}

int flipperTestRead( std::string &response, int bytes, int fd, int timeout )
{
    auto &faults = flipperHarness::g_faults;
    ++faults.m_reads;
    if( faults.m_beforeRead )
        faults.m_beforeRead();
    if( faults.m_nativeSerial )
        return ttyRead( response, bytes, fd, timeout );
    if( bytes <= 0 || faults.m_replies.empty() )
        return TTY_E_TIMEOUTONREAD;
    response = faults.m_replies.front();
    faults.m_replies.pop_front();
    return 0;
}
} // namespace tty
} // namespace MagAOX

/// Fail a selected synchronization call while otherwise syncing the real temporary file/directory.
int flipperTestSync( int fd /**< [in] file or directory descriptor */ );

/// Fail atomic replacement without discarding the existing state record.
int flipperTestRename( const char *oldPath /**< [in] temporary file */,
                       const char *newPath /**< [in] installed record */ );

/// Inject short, interrupted, and failed writes into the real persistence code.
ssize_t flipperTestFileWrite( int         fd /**< [in] file descriptor */,
                              const void *data /**< [in] record bytes */,
                              size_t      size /**< [in] byte count */ );

/// Release a descriptor while injecting a selected close error.
int flipperTestClose( int fd /**< [in] descriptor to release */ );

/// Fail temporary-file creation without leaking an opened descriptor.
int flipperTestTemporary( char *path /**< [in/out] mkstemp template */ );

/// Fail mode setting while leaving the temporary descriptor available for cleanup.
int flipperTestMode( int fd /**< [in] temporary descriptor */, mode_t mode /**< [in] requested mode */ );

int flipperTestTemporary( char *path )
{
    if( flipperHarness::g_faults.m_failedTemporary )
    {
        errno = ENOSPC;
        return -1;
    }
    return ::mkstemp( path );
}

int flipperTestMode( int fd, mode_t mode )
{
    if( flipperHarness::g_faults.m_failedMode )
    {
        errno = EPERM;
        return -1;
    }
    return ::fchmod( fd, mode );
}

int flipperTestClose( int fd )
{
    int   result = ::close( fd );
    auto &faults = flipperHarness::g_faults;
    if( ++faults.m_closes == faults.m_failedClose )
    {
        errno = EIO;
        return -1;
    }
    if( faults.m_afterClose )
        faults.m_afterClose( faults.m_closes );
    return result;
}

int flipperTestSync( int fd )
{
    auto &faults = flipperHarness::g_faults;
    if( ++faults.m_syncs == faults.m_failedSync )
    {
        errno = EIO;
        return -1;
    }
    return ::fsync( fd );
}

int flipperTestRename( const char *oldPath, const char *newPath )
{
    if( flipperHarness::g_faults.m_failedRename )
    {
        errno = EIO;
        return -1;
    }
    return ::rename( oldPath, newPath );
}

ssize_t flipperTestFileWrite( int fd, const void *data, size_t size )
{
    auto &faults = flipperHarness::g_faults;
    if( faults.m_zeroWrite )
        return 0;
    if( faults.m_failedFileWrite || faults.m_interruptWrite )
    {
        errno                   = faults.m_interruptWrite ? EINTR : EIO;
        faults.m_interruptWrite = false;
        return -1;
    }
    return ::write( fd, data, faults.m_shortWrites ? std::min( size, size_t( 1 ) ) : size );
}

// Substitute only the application header, leaving the shared library's real declarations intact.
#define MagAOXApp flipperTestApp
#define telemeter flipperTestTelemeter
#define usbDevice flipperTestUSBDevice
#define ioDevice flipperTestIODevice
#define ttyWrite flipperTestWrite
#define ttyRead flipperTestRead
#define fsync flipperTestSync
#define rename flipperTestRename
#define write flipperTestFileWrite
#define close flipperTestClose
#define mkstemp flipperTestTemporary
#define fchmod flipperTestMode
#include "../flipperCtrl.hpp"
#undef fchmod
#undef mkstemp
#undef close
#undef write
#undef rename
#undef fsync
#undef ttyRead
#undef ttyWrite
#undef telemeter
#undef ioDevice
#undef usbDevice
#undef MagAOXApp

namespace flipperHarness
{
void reset()
{
    g_faults = Faults();
    g_logs.clear();
    g_telemetry.clear();
}

std::string status( uint32_t bits )
{
    std::string reply( "\x81\x04\x0e\x00\x81\x50\x01\x00", 8 );
    reply.resize( 20, '\0' );
    for( unsigned i = 0; i < 4; ++i )
        reply[16 + i] = static_cast<char>( bits >> ( 8 * i ) );
    return reply;
}

std::string contents( const std::filesystem::path &path )
{
    std::ifstream file( path );
    return std::string( std::istreambuf_iterator<char>( file ), std::istreambuf_iterator<char>() );
}

size_t mismatchWarnings()
{
    return std::count_if( g_logs.begin(),
                          g_logs.end(),
                          []( const LogEntry &entry )
                          {
                              return entry.m_priority == flatlogs::logPrio::LOG_WARNING &&
                                     entry.m_message.find( "differs from inferred parked position" ) !=
                                         std::string::npos;
                          } );
}

size_t calls( const std::string &name )
{
    return std::count( g_faults.m_calls.begin(), g_faults.m_calls.end(), name );
}

size_t logs( flatlogs::logPrioT priority, const std::string &message )
{
    return std::count_if(
        g_logs.begin(),
        g_logs.end(),
        [&]( const LogEntry &entry )
        { return entry.m_priority == priority && entry.m_message.find( message ) != std::string::npos; } );
}

Directory::Directory()
{
    std::string name = "/tmp/flipperCtrl-test-XXXXXX";
    auto       *path = ::mkdtemp( name.data() );
    if( !path )
        throw std::runtime_error( "cannot create test directory" );
    m_path = path;
}

Directory::~Directory()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}
} // namespace flipperHarness
/// \endcond

using namespace MagAOX::app;
using namespace flipperHarness;

namespace libXWCTest
{
/** \defgroup flipperCtrl_unit_test flipperCtrl Unit Tests
 * \brief Unit tests for the flipperCtrl application.
 * \ingroup application_unit_test
 */

/// Namespace for flipperCtrl application unit tests.
/** \ingroup flipperCtrl_unit_test */
namespace flipperCtrlTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Real driver whose unactivated input descriptor is released explicitly on destruction.
class LocalDriver : public indiDriver<MagAOXApp<true>>
{
  public:
    /// Open the controller's private FIFOs without activating a processing thread.
    LocalDriver( MagAOXApp<true> *parent /**< [in] controller owning this driver */ );

    /// Release the input descriptor; the shared destructor releases output.
    ~LocalDriver();

  private:
    /// Input FIFO descriptor opened by the real driver, which has no descriptor getter.
    int m_inputFd{ -1 };
};

LocalDriver::LocalDriver( MagAOXApp<true> *parent )
    : indiDriver<MagAOXApp<true>>( parent, "flipper-local-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( ::stat( parent->driverInName().c_str(), &input ) == 0 );
    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        int         descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};
        if( ::fstat( descriptor, &candidate ) == 0 && candidate.st_dev == input.st_dev &&
            candidate.st_ino == input.st_ino )
        {
            REQUIRE( m_inputFd == -1 );
            m_inputFd = descriptor;
        }
    }
    REQUIRE( m_inputFd >= 0 );
}

LocalDriver::~LocalDriver()
{
    if( m_inputFd >= 0 )
        ::close( m_inputFd );
    setInputFd( -1 );
}

/// Expose protected app state while running the real app lifecycle and helper implementations.
class Controller : public flipperCtrl
{
  public:
    /// Configure isolated state storage and initial power state.
    Controller( const std::filesystem::path &root /**< [in] test directory root */,
                const std::string           &name = "flipper" /**< [in] app configuration name */ );

    /// Close any descriptor left by the test.
    ~Controller();

    using flipperCtrl::decodePosition;
    using flipperCtrl::publishPosition;
    using flipperCtrl::readStateFile;
    using flipperCtrl::reportedPosition;
    using flipperCtrl::saveState;
    using flipperCtrl::writeStateFile;

    /// Read and load a temporary configuration through the real app methods.
    int configure( const std::string &settings /**< [in] configuration text */ );

    /// Return the registered configuration so tests can inspect real option definitions.
    const mx::app::appConfigurator &configuration() const;

    /// Return the configured physical endpoint for logical in.
    int inPosition() const;

    /// Return the configured telemetry maximum interval.
    double telemetryInterval() const;

    /// Return the current serial descriptor without transferring ownership.
    int descriptor() const;

    /// Install a real local INDI driver and its bounded output reader.
    void localDriver();

    /// Drain complete outgoing INDI XML messages after a synchronous publication.
    std::vector<pcf::IndiProperty> messages();

    /// Attach a harmless real descriptor and select the connected FSM state.
    void connected();

    /// Replace the serial descriptor with a real socket endpoint.
    void attach( int fd /**< [in] descriptor whose ownership transfers to the controller */ );

    /// Set observed and target power states for lifecycle/guard tests.
    void power( int observed /**< [in] actual power */, int target /**< [in] requested power */ );

    /// Enable reversed logical endpoint mapping.
    void reverse();

    /// Permit an immediate retry without waiting in a test.
    void retryNow();

    /// Return the stored target endpoint.
    int target() const;

    /// Return whether a move is pending.
    bool pending() const;

    /// Return the published parked flag.
    int parked() const;

    /// Return the backing-record path.
    std::filesystem::path path() const;

    /// Send a real callback request for a selected logical endpoint.
    int request( bool in /**< [in] select in */, bool out /**< [in] select out */ );

  private:
    /// Descriptor reading this controller's private outgoing FIFO; -1 means unattached.
    int m_outputReader{ -1 };
};

Controller::Controller( const std::filesystem::path &root, const std::string &name )
{
    m_configName = name;
    m_basePath   = root.string();
    m_sysPath    = ( root / "sys" ).string();
    std::filesystem::create_directories( std::filesystem::path( m_sysPath ) / name );
    m_powerState = m_powerTargetState = 0;
    m_log.m_logLevel = logPrio::LOG_EMERGENCY; // Suppress shared-library logs; app calls use the capture base.
    state( stateCodes::POWEROFF );
}

Controller::~Controller()
{
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = 0;
    delete m_indiDriver;
    m_indiDriver = nullptr;
    if( m_outputReader >= 0 )
        ::close( m_outputReader );
}

int Controller::configure( const std::string &settings )
{
    setupConfig();
    const auto file = std::filesystem::path( m_basePath ) / "flipper-test.conf";
    std::ofstream( file ) << settings;
    config.readConfig( file.string() );
    return loadConfigImpl( config );
}

const mx::app::appConfigurator &Controller::configuration() const
{
    return config;
}

int Controller::inPosition() const
{
    return m_inPos;
}

double Controller::telemetryInterval() const
{
    return m_maxInterval;
}

int Controller::descriptor() const
{
    return m_fileDescrip;
}

void Controller::localDriver()
{
    REQUIRE(
        registerIndiPropertyNew(
            m_indiP_state, "fsm", pcf::IndiProperty::Text, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle, 0 ) ==
        0 );
    m_indiP_state.add( pcf::IndiElement( "state" ) );
    auto root = std::filesystem::path( m_basePath ) / "indi";
    std::filesystem::create_directories( root );
    m_driverInName   = ( root / "input" ).string();
    m_driverOutName  = ( root / "output" ).string();
    m_driverCtrlName = ( root / "control" ).string();
    for( const auto &path : { m_driverInName, m_driverOutName, m_driverCtrlName } )
    {
        REQUIRE( ::mkfifo( path.c_str(), 0600 ) == 0 );
    }
    m_outputReader = ::open( m_driverOutName.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC );
    REQUIRE( m_outputReader >= 0 );
    m_indiDriver = new LocalDriver( this );
    REQUIRE( m_indiDriver->good() );
}

std::vector<pcf::IndiProperty> Controller::messages()
{
    REQUIRE( m_outputReader >= 0 );
    std::string xml;
    char        buffer[4096];
    pollfd      event{ m_outputReader, POLLIN, 0 };
    while( ::poll( &event, 1, 0 ) > 0 && ( event.revents & POLLIN ) )
    {
        ssize_t size = ::read( m_outputReader, buffer, sizeof( buffer ) );
        REQUIRE( size > 0 );
        xml.append( buffer, static_cast<size_t>( size ) );
        REQUIRE( xml.size() < 65536 );
    }
    std::vector<pcf::IndiProperty> properties;
    pcf::IndiXmlParser             parser( "1.7" );
    std::string                    error, tail;
    for( char byte : xml )
    {
        tail += byte;
        parser.parseXml( &byte, 1, error );
        if( !error.empty() )
            break;
        if( parser.getState() == pcf::IndiXmlParser::CompleteState )
        {
            auto message = parser.createIndiMessage();
            REQUIRE( message.getType() == pcf::IndiMessage::SetProperty );
            properties.push_back( message.getProperty() );
            parser.clear();
            tail.clear();
        }
    }
    REQUIRE( error.empty() );
    REQUIRE( tail.find_first_not_of( " \r\n\t" ) == std::string::npos );
    return properties;
}

void Controller::connected()
{
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = ::open( "/dev/null", O_RDWR );
    REQUIRE( m_fileDescrip > 0 );
    m_powerState = m_powerTargetState = 1;
    state( stateCodes::CONNECTED );
}

void Controller::attach( int fd )
{
    if( m_fileDescrip > 0 )
        ::close( m_fileDescrip );
    m_fileDescrip = fd;
}

void Controller::power( int observed, int target )
{
    m_powerState       = observed;
    m_powerTargetState = target;
    if( observed == 0 )
        state( stateCodes::POWEROFF );
}

void Controller::reverse()
{
    m_inPos  = 2;
    m_outPos = 1;
}

void Controller::retryNow()
{
    m_nextSave = std::chrono::steady_clock::time_point::min();
}

int Controller::target() const
{
    return m_tgt;
}

bool Controller::pending() const
{
    return m_movePending;
}

int Controller::parked() const
{
    return m_indiP_parked["current"].get<int>();
}

std::filesystem::path Controller::path() const
{
    return std::filesystem::path( m_sysPath ) / m_configName / "position";
}

int Controller::request( bool in, bool out )
{
    pcf::IndiProperty request = m_indiP_position;
    request["in"].setSwitchState( in ? pcf::IndiElement::On : pcf::IndiElement::Off );
    request["out"].setSwitchState( out ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_position( request );
}
/// \endcond

/// Validate endpoint/motion masks and reject malformed or contradictory status packets.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper status replies distinguish settled endpoints and motion", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::decodePosition(std::string(), int(), bool());
    #endif
    // clang-format on
    int  pos    = -1;
    bool moving = false;
    for( uint32_t bits : { 1u, 2u, 0x80000501u, 0x80000502u } )
    {
        REQUIRE( Controller::decodePosition( status( bits ), pos, moving ) == 0 );
        REQUIRE( pos == static_cast<int>( bits & 3 ) );
        REQUIRE_FALSE( moving );
    }
    for( uint32_t bits : { 0u, 0x10u, 0x21u, 0x42u, 0x81u, 0x200u } )
    {
        REQUIRE( Controller::decodePosition( status( bits ), pos, moving ) == 0 );
        REQUIRE( pos == 0 );
        REQUIRE( moving );
    }
    REQUIRE( Controller::decodePosition( status( 3 ), pos, moving ) == -1 );
    for( unsigned byte : { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u } )
    {
        auto reply  = status( 1 );
        reply[byte] = 0x7f;
        REQUIRE( Controller::decodePosition( reply, pos, moving ) == -1 );
    }
    for( size_t length = 0; length < 20; ++length )
    {
        REQUIRE( Controller::decodePosition( status( 1 ).substr( 0, length ), pos, moving ) == -1 );
    }
    REQUIRE( Controller::decodePosition( status( 1 ) + "extra", pos, moving ) == -1 );
}

/// Recover only valid parked records, including reversed mapping and app-name isolation.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper backing records recover position while off", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appStartup();
    flipperCtrl::readStateFile();
    flipperCtrl::writeStateFile(1, true);
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory directory;
    int       endpoint = GENERATE( 1, 2 );
    bool      reversed = GENERATE( false, true );
    auto      writer   = std::make_unique<Controller>( directory.m_path );
    REQUIRE( writer->writeStateFile( endpoint, true ) == 0 );
    REQUIRE( contents( writer->path() ) == std::to_string( endpoint ) + "\n1\n" );
    struct stat info;
    REQUIRE( ::stat( writer->path().c_str(), &info ) == 0 );
    REQUIRE( ( info.st_mode & 0777 ) == 0644 );
    writer.reset();
    auto reader = std::make_unique<Controller>( directory.m_path );
    if( reversed )
        reader->reverse();
    REQUIRE( reader->appStartup() == 0 );
    REQUIRE( reader->onPowerOff() == 0 );
    REQUIRE( reader->state() == stateCodes::POWEROFF );
    REQUIRE( reader->reportedPosition() == endpoint );
    REQUIRE( reader->parked() == 1 );
    bool in = endpoint == ( reversed ? 2 : 1 );
    REQUIRE( reader->m_indiP_position["in"].getSwitchState() == ( in ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    REQUIRE( reader->m_indiP_position["out"].getSwitchState() ==
             ( in ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    REQUIRE( reader->m_indiP_position.getState() == INDI_IDLE );
    REQUIRE( telem_stage::moving( g_telemetry.back().data() ) == -2 );
    REQUIRE( telem_stage::preset( g_telemetry.back().data() ) == endpoint );
    REQUIRE( telem_stage::presetName( g_telemetry.back().data() ) == ( in ? "in" : "out" ) );
    reader.reset();
    Controller other( directory.m_path, "other" );
    REQUIRE( other.appStartup() == 0 );
    REQUIRE( other.reportedPosition() == 0 );
}

/// Invalid, missing, and explicitly unparked records must not invent a retained position.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper invalid snapshots remain unknown", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::readStateFile();
    flipperCtrl::appStartup();
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory   directory;
    Controller  app( directory.m_path );
    std::string record =
        GENERATE( "missing", "", "1", "garbage", "1 2", "-1 1", "3 1", "0 1", "2 1 extra", "1 0", "0 0" );
    if( record != "missing" )
        std::ofstream( app.path() ) << record;
    std::ofstream( app.path().string() + ".tmp.abandoned" ) << "1\n1\n";
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.onPowerOff() == 0 );
    REQUIRE( app.parked() == 0 );
    REQUIRE( app.reportedPosition() == 0 );
    REQUIRE( app.target() == 0 );
    REQUIRE( app.m_indiP_position["in"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_position["out"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( app.m_indiP_position.getState() == INDI_ALERT );
    REQUIRE( telem_stage::moving( g_telemetry.back().data() ) == -2 );
    REQUIRE( telem_stage::preset( g_telemetry.back().data() ) == 0 );
    REQUIRE( telem_stage::presetName( g_telemetry.back().data() ).empty() );
    auto syncs   = g_faults.m_syncs;
    auto records = g_telemetry.size();
    REQUIRE( app.whilePowerOff() == 0 );
    REQUIRE( app.whilePowerOff() == 0 );
    REQUIRE( g_telemetry.size() == records );
    app.m_due = true;
    REQUIRE( app.whilePowerOff() == 0 );
    REQUIRE( g_telemetry.size() == records + 1 );
    REQUIRE( app.m_schedules == 3 );
    REQUIRE( g_faults.m_syncs == syncs );
    REQUIRE( g_faults.m_commands.empty() );
    REQUIRE( g_faults.m_reads == 0 );
}

/// Persist invalidation before moving and restore unknown after power-off interrupts a command.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper moves invalidate parking before hardware IO", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::appLogic();
    flipperCtrl::onPowerOff();
    flipperCtrl::newCallBack_m_indiP_position(pcf::IndiProperty());
    #endif
    // clang-format on
    reset();
    Directory   directory;
    auto        active = std::make_unique<Controller>( directory.m_path );
    Controller &app    = *active;
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( contents( app.path() ) == "1\n1\n" );
    g_faults.m_beforeWrite = [&]( const std::string &command )
    {
        if( static_cast<unsigned char>( command[0] ) == 0x6a )
            REQUIRE( contents( app.path() ) == "1\n0\n" );
    };
    REQUIRE( app.request( false, true ) == 0 );
    REQUIRE( app.state() == stateCodes::OPERATING );
    REQUIRE( app.pending() );
    REQUIRE( app.parked() == 0 );
    REQUIRE( app.target() == 2 );
    REQUIRE( g_faults.m_commands.back() == std::string( "\x6a\x04\x00\x02\x50\x01", 6 ) );
    REQUIRE( telem_stage::moving( g_telemetry.back().data() ) == 1 );
    REQUIRE( app.m_indiP_position.getState() == INDI_BUSY );
    g_faults.m_replies.push_back( status( 1 ) ); // Old endpoint is still active briefly.
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.pending() );
    REQUIRE( contents( app.path() ) == "1\n0\n" );
    SECTION( "interrupted motion recovers unknown" )
    {
        app.power( 0, 0 );
        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.target() == 0 );
        REQUIRE( app.state() == stateCodes::POWEROFF );
        REQUIRE( app.m_indiP_position.getState() == INDI_ALERT );
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart( directory.m_path );
        REQUIRE( restart.appStartup() == 0 );
        REQUIRE( restart.onPowerOff() == 0 );
        REQUIRE( restart.reportedPosition() == 0 );
        REQUIRE( restart.parked() == 0 );
    }
    SECTION( "confirmed completion recovers the new endpoint" )
    {
        g_faults.m_replies.push_back( status( 0x20 ) );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.pending() );
        g_faults.m_replies.push_back( status( 2 ) );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::READY );
        REQUIRE_FALSE( app.pending() );
        REQUIRE( app.parked() == 1 );
        REQUIRE( contents( app.path() ) == "2\n1\n" );
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart( directory.m_path );
        REQUIRE( restart.appStartup() == 0 );
        REQUIRE( restart.onPowerOff() == 0 );
        REQUIRE( restart.reportedPosition() == 2 );
    }
    SECTION( "replacement target keeps parking invalid until confirmed" )
    {
        REQUIRE( app.moveTo( 1 ) == 0 );
        REQUIRE( app.pending() );
        g_faults.m_replies.push_back( status( 1 ) );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.parked() == 1 );
        REQUIRE( app.target() == 1 );
    }
}

/// Log one WARNING on power-on mismatch and let the first confirmed live endpoint replace the inference.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper power-on mismatch warns once and trusts live position", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::onPowerOff();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.writeStateFile( 1, true ) == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.onPowerOff() == 0 );
    app.connected();
    bool differs  = GENERATE( false, true );
    int  endpoint = differs ? 2 : 1;
    g_faults.m_replies.push_back( status( 0 ) );
    REQUIRE( app.appLogic() == 0 ); // Transitional status must not consume the comparison.
    REQUIRE( mismatchWarnings() == 0 );
    g_faults.m_replies.push_back( status( endpoint ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( mismatchWarnings() == ( differs ? 1 : 0 ) );
    REQUIRE( app.reportedPosition() == endpoint );
    REQUIRE( contents( app.path() ) == std::to_string( endpoint ) + "\n1\n" );
    g_faults.m_replies.push_back( status( endpoint ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( mismatchWarnings() == ( differs ? 1 : 0 ) );
    // A later power cycle gets a fresh comparison against its own retained endpoint.
    app.power( 0, 0 );
    REQUIRE( app.onPowerOff() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 3 - endpoint ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( mismatchWarnings() == ( differs ? 2 : 1 ) );
}

/// Query failures and malformed packets never create a false READY or parked endpoint.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper position query handles transport and framing errors", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::appLogic();
    flipperCtrl::decodePosition(std::string(), int(), bool());
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    SECTION( "request write fails" )
    {
        g_faults.m_failedCommand = 0x0480;
    }
    SECTION( "read times out" )
    {
    }
    SECTION( "truncated reply" )
    {
        g_faults.m_replies.push_back( status( 1 ).substr( 0, 17 ) );
    }
    SECTION( "contradictory endpoint switches" )
    {
        g_faults.m_replies.push_back( status( 3 ) );
    }
    SECTION( "unexpected reply" )
    {
        auto reply = status( 1 );
        reply[0]   = 0x91;
        g_faults.m_replies.push_back( reply );
    }
    SECTION( "bad length" )
    {
        auto reply = status( 1 );
        reply[3]   = 0x7f;
        g_faults.m_replies.push_back( reply );
    }
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    REQUIRE( app.parked() == 0 );
    REQUIRE( app.reportedPosition() == 0 );
    REQUIRE( contents( app.path() ) == "0\n0\n" );
    REQUIRE( app.m_indiP_position.getState() == INDI_ALERT );
}

/// Accept fragmented/coalesced status packets while ignoring unsolicited completion snapshots.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper reader assembles packets and ignores completion notifications", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    auto completion = status( 1 );
    completion[0]   = 0x64;
    SECTION( "coalesced" )
    {
        g_faults.m_replies.push_back( completion + status( 2 ) );
    }
    SECTION( "fragmented" )
    {
        g_faults.m_replies.push_back( completion.substr( 0, 9 ) );
        g_faults.m_replies.push_back( completion.substr( 9 ) + status( 2 ).substr( 0, 6 ) );
        g_faults.m_replies.push_back( status( 2 ).substr( 6 ) );
    }
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.reportedPosition() == 2 );
    REQUIRE( contents( app.path() ) == "2\n1\n" );
}

/// Reject unpowered, disconnected, ambiguous, and invalid commands without changing retained state.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper move guards preserve state and prevent hardware IO", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::newCallBack_m_indiP_position(pcf::IndiProperty());
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.writeStateFile( 1, true ) == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.onPowerOff() == 0 );
    SECTION( "off" )
    {
    }
    SECTION( "disconnected" )
    {
        app.power( 1, 1 );
        app.state( stateCodes::NOTCONNECTED );
    }
    SECTION( "power-off target" )
    {
        app.connected();
        g_faults.m_replies.push_back( status( 1 ) );
        REQUIRE( app.appLogic() == 0 );
        app.power( 1, 0 );
        g_faults.m_commands.clear();
    }
    REQUIRE( app.moveTo( 2 ) == -1 );
    REQUIRE( app.request( false, true ) == -1 );
    REQUIRE( app.request( true, true ) == -1 );
    REQUIRE( app.moveTo( 7 ) == -1 );
    pcf::IndiProperty wrong = app.m_indiP_position;
    wrong.setDevice( "wrong" );
    REQUIRE( app.newCallBack_m_indiP_position( wrong ) == -1 );
    wrong = app.m_indiP_position;
    wrong.setName( "wrong" );
    REQUIRE( app.newCallBack_m_indiP_position( wrong ) == -1 );
    REQUIRE( app.target() == 1 );
    REQUIRE( contents( app.path() ) == "1\n1\n" );
    REQUIRE( g_faults.m_commands.empty() );
}

/// Filesystem failures prevent movement; partial serial writes leave the installed record unparked.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper failed invalidation or command cannot recover stale parking", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::writeStateFile(1, false);
    #endif
    // clang-format on
    reset();
    Directory   directory;
    auto        active    = std::make_unique<Controller>( directory.m_path );
    Controller &app       = *active;
    const auto  statePath = app.path();
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    g_faults.m_commands.clear();
    SECTION( "data write failure" )
    {
        g_faults.m_failedFileWrite = true;
    }
    SECTION( "file sync failure" )
    {
        g_faults.m_failedSync = g_faults.m_syncs + 1;
    }
    SECTION( "file close failure" )
    {
        g_faults.m_failedClose = g_faults.m_closes + 1;
    }
    SECTION( "directory sync failure" )
    {
        g_faults.m_failedSync = g_faults.m_syncs + 2;
    }
    SECTION( "rename failure" )
    {
        g_faults.m_failedRename = true;
    }
    SECTION( "serial command failure" )
    {
        g_faults.m_failedCommand = 0x046a;
    }
    REQUIRE( app.moveTo( 2 ) == -1 );
    bool commandFailed = g_faults.m_failedCommand == 0x046a;
    REQUIRE( g_faults.m_commands.size() == ( commandFailed ? 1 : 0 ) );
    if( commandFailed )
    {
        REQUIRE( contents( app.path() ) == "1\n0\n" );
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart( directory.m_path );
        REQUIRE( restart.appStartup() == 0 );
        REQUIRE( restart.reportedPosition() == 0 );
    }
    else
    {
        REQUIRE( app.target() == 1 );
        REQUIRE_FALSE( app.pending() );
    }
    for( const auto &entry : std::filesystem::directory_iterator( statePath.parent_path() ) )
    {
        REQUIRE( entry.path().filename() == "position" );
    }
}

/// Failed completion saves retry without blocking live reporting or writing every FSM loop.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper completion save retries are bounded", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::saveState();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.moveTo( 2 ) == 0 );
    g_faults.m_failedRename = true;
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.reportedPosition() == 2 );
    REQUIRE( app.parked() == 1 );
    REQUIRE( contents( app.path() ) == "1\n0\n" );
    auto syncs = g_faults.m_syncs;
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_syncs == syncs );
    g_faults.m_failedRename = false;
    app.retryNow();
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( contents( app.path() ) == "2\n1\n" );
    syncs = g_faults.m_syncs;
    REQUIRE( app.moveTo( 2 ) == 0 ); // Settled same-endpoint requests are a no-op.
    REQUIRE( g_faults.m_syncs == syncs );
}

/// Handle short/interrupted writes and invalidate uncommanded changes without warning during normal operation.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper state writes handle interruptions and external changes", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::writeStateFile(1, true);
    flipperCtrl::saveState();
    flipperCtrl::getPos();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    g_faults.m_interruptWrite = g_faults.m_shortWrites = true;
    REQUIRE( app.writeStateFile( 1, true ) == 0 );
    REQUIRE( contents( app.path() ) == "1\n1\n" );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    auto syncs = g_faults.m_syncs;
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( g_faults.m_syncs == syncs + 4 ); // Invalidation and confirmation each sync file and directory.
    REQUIRE( contents( app.path() ) == "2\n1\n" );
    REQUIRE( mismatchWarnings() == 0 );
    REQUIRE( app.appShutdown() == 0 );
}

/// Exercise the real tty byte-count reader with coalesced notifications and a status payload.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper status queries use the real tty transport", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    int sockets[2];
    REQUIRE( ::socketpair( AF_UNIX, SOCK_STREAM, 0, sockets ) == 0 );
    app.attach( sockets[0] );
    g_faults.m_nativeSerial = true;
    auto completion         = status( 1 );
    completion[0]           = 0x64;
    auto reply              = completion + status( 2 );
    REQUIRE( ::write( sockets[1], reply.data(), reply.size() ) == static_cast<ssize_t>( reply.size() ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.reportedPosition() == 2 );
    std::string query;
    REQUIRE( MagAOX::tty::ttyRead( query, 6, sockets[1], 100 ) == 0 );
    REQUIRE( query == std::string( "\x80\x04\x00\x00\x50\x01", 6 ) );
    REQUIRE( ::close( sockets[1] ) == 0 );
}

/// Keep the retained comparison across an initial failed read and reject queries while off.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper retained inference survives initial query failure", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.writeStateFile( 1, true ) == 0 );
    REQUIRE( app.appStartup() == 0 );
    REQUIRE( app.getPos() == -1 );
    REQUIRE( g_faults.m_commands.empty() );
    app.connected();
    REQUIRE( app.appLogic() == 0 ); // An initial timeout must not consume the inferred endpoint.
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    REQUIRE( app.parked() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( mismatchWarnings() == 1 );
    REQUIRE( app.reportedPosition() == 2 );
}

/// Load real defaults and overrides, including USB, I/O, reversal, and telemetry configuration.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper configuration registers and loads real helper options", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::setupConfig();
    flipperCtrl::loadConfigImpl(mx::app::appConfigurator());
    flipperCtrl::loadConfig();
    #endif
    // clang-format on
    reset();
    Directory   directory;
    Controller  app( directory.m_path );
    int         mode = GENERATE( 0, 1, 2 );
    std::string settings;
    if( mode )
    {
        settings = "[usb]\nidVendor=ffff\nidProduct=fffe\nserial=coverage-only\nbaud=9600\n"
                   "[device]\nreadTimeout=41\nwriteTimeout=73\n[telemeter]\nmaxInterval=3.5\n"
                   "[flipper]\nreverse=" +
                   std::string( mode == 2 ? "true\n" : "false\n" );
    }
    REQUIRE( app.configure( settings ) == 0 );
    for( const auto &name : { "usb.idVendor",
                              "usb.idProduct",
                              "usb.serial",
                              "usb.baud",
                              "device.readTimeout",
                              "device.writeTimeout",
                              "flipper.reverse",
                              "telemeter.maxInterval" } )
    {
        REQUIRE( app.configuration().m_targets.count( name ) == 1 );
    }
    REQUIRE( app.inPosition() == ( mode == 2 ? 2 : 1 ) );
    REQUIRE( app.m_baudRate == ( mode ? B9600 : B115200 ) );
    REQUIRE( app.m_readTimeout == ( mode ? 41 : 1000 ) );
    REQUIRE( app.m_writeTimeout == ( mode ? 73 : 1000 ) );
    REQUIRE( app.telemetryInterval() == ( mode ? 3.5 : 10.0 ) );
    REQUIRE( app.m_idVendor == ( mode ? "ffff" : "" ) );
    REQUIRE( app.m_idProduct == ( mode ? "fffe" : "" ) );
    REQUIRE( app.m_serial == ( mode ? "coverage-only" : "" ) );
    app.loadConfig();
    REQUIRE( app.shutdown() == 0 );
    REQUIRE( calls( "usb-load" ) == 2 );
    REQUIRE( calls( "io-load" ) == 2 );
    REQUIRE( calls( "telem-load" ) == 2 );
}

/// Distinguish tolerated USB discovery results from logged errors and fatal configuration failures.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper configuration preserves helper failure contracts", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::setupConfig();
    flipperCtrl::loadConfigImpl(mx::app::appConfigurator());
    flipperCtrl::loadConfig();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    SECTION( "USB errors are recoverable" )
    {
        int rv                   = GENERATE( 0, TTY_E_DEVNOTFOUND, TTY_E_NODEVNAMES, TTY_E_BADBAUDRATE );
        g_faults.m_usbLoadResult = rv;
        REQUIRE( app.configure( "" ) == 0 );
        REQUIRE( logs( software_error::defaultLevel, "" ) == ( rv == TTY_E_BADBAUDRATE ? 1 : 0 ) );
        REQUIRE( calls( "io-load" ) == 1 );
        REQUIRE( calls( "telem-load" ) == 1 );
        REQUIRE( app.shutdown() == 0 );
    }
    SECTION( "I/O loading stops before telemetry and requests shutdown" )
    {
        g_faults.m_ioLoadResult = -1;
        REQUIRE( app.configure( "" ) == -1 );
        REQUIRE( calls( "telem-load" ) == 0 );
        app.loadConfig();
        REQUIRE( app.shutdown() == 1 );
        REQUIRE( logs( software_critical::defaultLevel, "" ) == 1 );
    }
    SECTION( "telemetry loading errors propagate to shutdown" )
    {
        g_faults.m_telemLoadResult = -1;
        REQUIRE( app.configure( "" ) == -1 );
        app.loadConfig();
        REQUIRE( app.shutdown() == 1 );
        REQUIRE( logs( software_error::defaultLevel, "telemeterT::loadConfig" ) == 2 );
        REQUIRE( logs( software_critical::defaultLevel, "" ) == 1 );
    }
    SECTION( "telemetry setup requests shutdown without loading configuration" )
    {
        g_faults.m_telemSetupResult = -1;
        app.setupConfig();
        REQUIRE( app.shutdown() == 1 );
        REQUIRE( calls( "usb-load" ) == 0 );
        REQUIRE( logs( software_error::defaultLevel, "telemeterT::setupConfig" ) == 1 );
    }
}

/// Stop startup at its exact failed property or telemetry operation.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper startup rejects fatal registration and telemetry failures", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appStartup();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    int        failure = GENERATE( 0, 1, 2, 3 );
    if( failure < 3 )
    {
        g_faults.m_propertyFailure = static_cast<Faults::PropertyFailure>( failure + 1 );
    }
    else
        g_faults.m_telemStartupResult = -1;
    REQUIRE( app.appStartup() == -1 );
    REQUIRE( calls( "selection" ) == 1 );
    REQUIRE( calls( "register-new" ) == ( failure >= 1 ? 1 : 0 ) );
    REQUIRE( calls( "register-read-only" ) == ( failure >= 2 ? 1 : 0 ) );
    REQUIRE( calls( "telem-startup" ) == ( failure == 3 ? 1 : 0 ) );
    REQUIRE( logs( software_error::defaultLevel, "" ) == 1 );
    REQUIRE( g_faults.m_commands.empty() );
    REQUIRE_FALSE( std::filesystem::exists( app.path() ) );
}

/// Exercise powered-on and OFF telemetry errors, including the non-propagating shutdown macro.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper lifecycle honors telemetry error returns", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    flipperCtrl::whilePowerOff();
    flipperCtrl::appShutdown();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    SECTION( "powered-on scheduling fails after live publication" )
    {
        app.connected();
        g_faults.m_replies.push_back( status( 1 ) );
        g_faults.m_telemLogicResult = -1;
        REQUIRE( app.appLogic() == -1 );
        REQUIRE( app.reportedPosition() == 1 );
        REQUIRE( app.state() == stateCodes::READY );
    }
    SECTION( "OFF scheduling fails without any hardware or disk I/O" )
    {
        g_faults.m_telemLogicResult = -1;
        REQUIRE( app.whilePowerOff() == -1 );
        REQUIRE( g_faults.m_commands.empty() );
        REQUIRE( g_faults.m_syncs == 0 );
        REQUIRE( app.state() == stateCodes::POWEROFF );
    }
    SECTION( "shutdown logs an error but returns success and releases serial ownership" )
    {
        app.connected();
        int fd                         = app.descriptor();
        g_faults.m_telemShutdownResult = -1;
        REQUIRE( app.appShutdown() == 0 );
        REQUIRE( app.descriptor() == 0 );
        REQUIRE( ::fcntl( fd, F_GETFD ) == -1 );
        REQUIRE( errno == EBADF );
        REQUIRE( logs( software_error::defaultLevel, "telemeterT::appShutdown" ) == 1 );
    }
    REQUIRE( logs( software_error::defaultLevel, "" ) == 1 );
}

/// Guard every phase of discovery while observed or requested power is unavailable.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper FSM power guards avoid dependency side effects", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.state( stateCodes::CONNECTED );
    int mode = GENERATE( 0, 1, 2, 3 );
    app.power( mode == 0 ? 0 : ( mode == 1 ? -1 : 1 ), mode < 2 ? 1 : ( mode == 2 ? 0 : -1 ) );
    auto previous = app.state();
    g_faults.m_calls.clear();
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == previous );
    REQUIRE( g_faults.m_calls.empty() );
    REQUIRE( g_faults.m_commands.empty() );
    REQUIRE( g_faults.m_reads == 0 );
    REQUIRE( g_faults.m_syncs == 0 );
}

/// Discover, connect, and freshly confirm endpoints through the production POWERON FSM.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper discovers absent devices and recovers without false readiness", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.power( 1, 1 );
    app.state( stateCodes::POWERON );
    int missing = GENERATE( TTY_E_DEVNOTFOUND, TTY_E_NODEVNAMES );
    g_faults.m_calls.clear();
    g_faults.m_discoveryResults = { missing, missing, 0 };
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::NODEVICE );
    REQUIRE( calls( "connect" ) == 0 );
    REQUIRE( g_faults.m_commands.empty() );
    REQUIRE( logs( text_log::defaultLevel, "not found in udev" ) == 1 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( logs( text_log::defaultLevel, "not found in udev" ) == 1 );
    bool moving = GENERATE( false, true );
    g_faults.m_replies.push_back( status( moving ? 0 : 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == ( moving ? stateCodes::OPERATING : stateCodes::READY ) );
    REQUIRE( app.reportedPosition() == ( moving ? 0 : 2 ) );
    REQUIRE( app.descriptor() > 0 );
    REQUIRE( calls( "discover" ) == 3 );
    REQUIRE( calls( "connect" ) == 1 );
    REQUIRE( g_faults.m_commands.size() == 1 );
    REQUIRE( logs( text_log::defaultLevel, "found in udev as /dev/flipper-test" ) == 1 );
}

/// Distinguish disappearance, fatal rediscovery, and retryable connection failure.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper connection errors rediscover and preserve FSM contracts", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.power( 1, 1 );
    app.state( stateCodes::POWERON );
    g_faults.m_calls.clear();
    SECTION( "initial discovery fails critically" )
    {
        g_faults.m_discoveryResults = { TTY_E_BADBAUDRATE };
        REQUIRE( app.appLogic() == -1 );
        REQUIRE( app.state() == stateCodes::FAILURE );
        REQUIRE( calls( "connect" ) == 0 );
        REQUIRE( logs( software_critical::defaultLevel, "" ) == 1 );
    }
    SECTION( "failed connection triggers rediscovery before retry" )
    {
        int rv                      = GENERATE( 0, TTY_E_DEVNOTFOUND, TTY_E_NODEVNAMES, TTY_E_BADBAUDRATE );
        g_faults.m_discoveryResults = { 0, rv };
        g_faults.m_connectResults   = { TTY_E_ERRORONWRITE };
        REQUIRE( app.appLogic() == ( rv == TTY_E_BADBAUDRATE ? -1 : 0 ) );
        REQUIRE( g_faults.m_calls == std::vector<std::string>{ "discover", "connect", "discover" } );
        REQUIRE( app.descriptor() == 0 );
        REQUIRE( app.state() == ( rv == TTY_E_BADBAUDRATE
                                      ? stateCodes::FAILURE
                                      : ( rv == 0 ? stateCodes::NOTCONNECTED : stateCodes::NODEVICE ) ) );
        REQUIRE( logs( software_critical::defaultLevel, "" ) == ( rv == TTY_E_BADBAUDRATE ? 1 : 0 ) );
        if( rv != 0 && rv != TTY_E_BADBAUDRATE )
        {
            REQUIRE( logs( text_log::defaultLevel, "no longer found in udev" ) == 1 );
        }
        if( rv != TTY_E_BADBAUDRATE )
        {
            g_faults.m_discoveryResults = { 0 };
            g_faults.m_replies.push_back( status( 1 ) );
            REQUIRE( app.appLogic() == 0 );
            REQUIRE( app.state() == stateCodes::READY );
            REQUIRE( app.reportedPosition() == 1 );
        }
    }
}

/// Recover from a real FSM query error by reconnecting and replacing the lost descriptor.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper reconnects after a failed initial status query", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.power( 1, 1 );
    app.state( stateCodes::POWERON );
    g_faults.m_discoveryResults = { 0 };
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::NOTCONNECTED );
    REQUIRE( app.reportedPosition() == 0 );
    REQUIRE( app.parked() == 0 );
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.state() == stateCodes::READY );
    REQUIRE( app.reportedPosition() == 2 );
    REQUIRE( calls( "discover" ) == 1 );
    REQUIRE( calls( "connect" ) == 2 );
    REQUIRE( g_faults.m_commands.size() == 2 );
    REQUIRE( contents( app.path() ) == "2\n1\n" );
}

/// Verify actual timestamped INDI messages for unknown, retained, moving, completed, and OFF states.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper publication sends coherent properties through real private FIFOs", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::publishPosition();
    flipperCtrl::appLogic();
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    bool       reversed = GENERATE( false, true );
    if( reversed )
        app.reverse();
    app.localDriver();
    REQUIRE( app.appStartup() == 0 );
    auto initial = app.messages();
    REQUIRE( initial.size() == 1 );
    REQUIRE( initial[0].getName() == "presetName" );
    REQUIRE( initial[0].getState() == INDI_ALERT );
    REQUIRE( initial[0]["in"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( initial[0]["out"].getSwitchState() == pcf::IndiElement::Off );
    app.connected();
    auto connected = app.messages();
    REQUIRE( connected.size() == 1 );
    REQUIRE( connected[0].getName() == "fsm" );
    REQUIRE( connected[0]["state"].get<std::string>() == "CONNECTED" );
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    auto parked = app.messages();
    REQUIRE( parked.size() == 2 );
    REQUIRE( parked[0].getName() == "presetName" );
    REQUIRE( parked[0].getState() == INDI_IDLE );
    REQUIRE( parked[0]["in"].getSwitchState() == ( reversed ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    REQUIRE( parked[0]["out"].getSwitchState() == ( reversed ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    REQUIRE( parked[1].getName() == "parked" );
    REQUIRE( parked[1]["current"].get<int>() == 1 );
    REQUIRE( parked[0].getTimeStamp().getTimeValSecs() > 0 );
    REQUIRE( app.publishPosition() == 0 );
    REQUIRE( app.messages().empty() );
    REQUIRE( app.moveTo( 2 ) == 0 );
    auto busy = app.messages();
    REQUIRE( busy.size() == 2 );
    REQUIRE( busy[0].getState() == INDI_BUSY );
    REQUIRE( busy[0]["in"].getSwitchState() == parked[0]["in"].getSwitchState() );
    REQUIRE( busy[0]["out"].getSwitchState() == parked[0]["out"].getSwitchState() );
    REQUIRE( busy[1]["current"].get<int>() == 0 );
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    auto settled = app.messages();
    REQUIRE( settled.size() == 2 );
    REQUIRE( settled[0].getState() == INDI_IDLE );
    REQUIRE( settled[0]["in"].getSwitchState() == ( reversed ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    REQUIRE( settled[0]["out"].getSwitchState() == ( reversed ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    REQUIRE( settled[1]["current"].get<int>() == 1 );
    app.power( 0, 0 );
    REQUIRE( app.onPowerOff() == 0 );
    auto off = app.messages();
    REQUIRE( off.size() == 1 );
    REQUIRE( off[0].getName() == "fsm" );
    REQUIRE( off[0]["state"].get<std::string>() == "POWEROFF" );
    REQUIRE( std::none_of( off.begin(),
                           off.end(),
                           []( const pcf::IndiProperty &property )
                           { return property.getName() == "presetName" || property.getName() == "parked"; } ) );
    REQUIRE( app.reportedPosition() == 2 );
    REQUIRE( app.publishPosition() == 0 );
    REQUIRE( app.messages().empty() );
}

/// Handle directory, temporary-file, permissions, zero-write, and directory-close failures safely.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper persistence cleans up every remaining filesystem failure", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::readStateFile();
    flipperCtrl::writeStateFile(1, true);
    flipperCtrl::moveTo(2);
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    SECTION( "non-missing open error is reported and remains unknown" )
    {
        std::filesystem::create_symlink( "position", app.path() );
        REQUIRE( app.readStateFile() == -1 );
        REQUIRE( app.reportedPosition() == 0 );
        REQUIRE( logs( software_error::defaultLevel, "cannot read flipper state file" ) == 1 );
    }
    SECTION( "missing state directory rejects the durable write" )
    {
        std::filesystem::remove( app.path().parent_path() );
        REQUIRE( app.writeStateFile( 1, true ) == -1 );
        REQUIRE( g_faults.m_closes == 0 );
        REQUIRE( logs( software_error::defaultLevel, "cannot open flipper state directory" ) == 1 );
    }
    SECTION( "failed temporary-file creation closes the directory" )
    {
        g_faults.m_failedTemporary = true;
        REQUIRE( app.writeStateFile( 1, true ) == -1 );
        REQUIRE( g_faults.m_closes == 1 );
        REQUIRE( std::filesystem::is_empty( app.path().parent_path() ) );
        REQUIRE( logs( software_error::defaultLevel, "cannot create flipper state file" ) == 1 );
    }
    SECTION( "mode-setting and zero-byte failures remove temporary files" )
    {
        bool mode             = GENERATE( false, true );
        g_faults.m_failedMode = mode;
        g_faults.m_zeroWrite  = !mode;
        REQUIRE( app.writeStateFile( 1, true ) == -1 );
        REQUIRE( g_faults.m_closes == 2 );
        REQUIRE( std::filesystem::is_empty( app.path().parent_path() ) );
        REQUIRE( logs( software_error::defaultLevel, "cannot durably store flipper position" ) == 1 );
    }
    SECTION( "failed directory close rejects movement even after replacement" )
    {
        REQUIRE( app.appStartup() == 0 );
        app.connected();
        g_faults.m_replies.push_back( status( 1 ) );
        REQUIRE( app.appLogic() == 0 );
        g_faults.m_commands.clear();
        g_faults.m_failedClose = g_faults.m_closes + 2;
        REQUIRE( app.moveTo( 2 ) == -1 );
        REQUIRE( g_faults.m_commands.empty() );
        REQUIRE( contents( app.path() ) == "1\n0\n" );
        REQUIRE( app.target() == 1 );
        REQUIRE_FALSE( app.pending() );
    }
}

/// Reject failed external-change invalidation and exercise immediate forced retry semantics.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper external invalidation failure retries conservatively", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::saveState(true);
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    g_faults.m_failedRename = true;
    g_faults.m_replies.push_back( status( 2 ) );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.reportedPosition() == 2 );
    REQUIRE( contents( app.path() ) == "1\n1\n" );
    REQUIRE( logs( software_error::defaultLevel, "cannot durably store flipper position" ) == 1 );
    auto syncs = g_faults.m_syncs;
    REQUIRE( app.saveState() == -1 );
    REQUIRE( g_faults.m_syncs == syncs );
    g_faults.m_failedRename = false;
    REQUIRE( app.saveState( true ) == 0 );
    REQUIRE( contents( app.path() ) == "2\n1\n" );
    REQUIRE( g_faults.m_syncs == syncs + 4 );
    syncs = g_faults.m_syncs;
    REQUIRE( app.saveState( true ) == 0 );
    REQUIRE( g_faults.m_syncs == syncs + 2 );
    REQUIRE( app.saveState() == 0 );
    REQUIRE( g_faults.m_syncs == syncs + 2 );
    REQUIRE( mismatchWarnings() == 0 );
}

/// Bound memory and read attempts without accepting an incomplete position frame.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper query rejects missing descriptors and exhausted framing", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    SECTION( "descriptor is missing before request" )
    {
        app.attach( 0 );
        REQUIRE( app.getPos() == -1 );
        REQUIRE( g_faults.m_commands.empty() );
        REQUIRE( logs( software_error::defaultLevel, "without a connection" ) == 1 );
    }
    SECTION( "oversized reply is rejected before decoding" )
    {
        g_faults.m_replies.push_back( std::string( 4097, 'x' ) );
        REQUIRE( app.getPos() == -1 );
        REQUIRE( logs( software_error::defaultLevel, "oversized flipper response" ) == 1 );
    }
    SECTION( "eight valid completion messages do not manufacture a status" )
    {
        auto completion = status( 1 );
        completion[0]   = 0x66;
        for( unsigned i = 0; i < 8; ++i )
            g_faults.m_replies.push_back( completion );
        REQUIRE( app.getPos() == -1 );
        REQUIRE( g_faults.m_reads == 8 );
        REQUIRE( logs( software_error::defaultLevel, "incomplete flipper position response" ) == 1 );
    }
    SECTION( "eight tiny fragments are still incomplete" )
    {
        for( char byte : status( 1 ) )
            g_faults.m_replies.emplace_back( 1, byte );
        REQUIRE( app.getPos() == -1 );
        REQUIRE( g_faults.m_reads == 8 );
        REQUIRE( logs( software_error::defaultLevel, "incomplete flipper position response" ) == 1 );
    }
    SECTION( "completion from the wrong source is rejected" )
    {
        auto completion = status( 1 );
        completion[0]   = 0x64;
        completion[5]   = 0x51;
        g_faults.m_replies.push_back( completion );
        REQUIRE( app.getPos() == -1 );
        REQUIRE( logs( software_error::defaultLevel, "unexpected flipper response" ) == 1 );
    }
    SECTION( "deadline is already expired" )
    {
        app.m_readTimeout = 0;
        REQUIRE( app.getPos() == -1 );
        REQUIRE( g_faults.m_reads == 0 );
    }
    REQUIRE( app.reportedPosition() == 0 );
    REQUIRE( contents( app.path() ) == "0\n0\n" );
}

/// Inject synchronous power events at the guarded query and pre-command persistence boundaries.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper power transitions cannot bypass post-I/O guards", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appLogic();
    flipperCtrl::moveTo(2);
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    g_faults.m_commands.clear();
    SECTION( "power changes while the status reply is received" )
    {
        bool readFails        = GENERATE( false, true );
        g_faults.m_beforeRead = [&] { app.power( 0, 0 ); };
        if( !readFails )
            g_faults.m_replies.push_back( status( 1 ) );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.state() == stateCodes::POWEROFF );
        REQUIRE( contents( app.path() ) == "1\n1\n" );
        REQUIRE( g_faults.m_commands.size() == 1 );
        REQUIRE( app.reportedPosition() == 1 );
    }
    SECTION( "observed or requested power changes after durable invalidation" )
    {
        bool     requestedPowerChanges = GENERATE( false, true );
        unsigned afterDirectoryClose   = g_faults.m_closes + 2;
        g_faults.m_afterClose          = [&]( unsigned closed )
        {
            if( closed == afterDirectoryClose )
                app.power( requestedPowerChanges ? 1 : 0, requestedPowerChanges ? 0 : 1 );
        };
        REQUIRE( app.moveTo( 2 ) == -1 );
        REQUIRE( g_faults.m_commands.empty() );
        REQUIRE( contents( app.path() ) == "1\n0\n" );
        app.power( 0, 0 );
        REQUIRE( app.onPowerOff() == 0 );
        REQUIRE( app.reportedPosition() == 0 );
        REQUIRE( app.target() == 0 );
        REQUIRE_FALSE( app.pending() );
    }
    g_faults.m_beforeRead = {};
    g_faults.m_afterClose = {};
}

/// Execute the real static callback dispatcher and both no-selection representations.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE( "flipper callback dispatch handles complete and empty selections", "[flipperCtrl]" )
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::st_newCallBack_m_indiP_position(nullptr, pcf::IndiProperty());
    flipperCtrl::newCallBack_m_indiP_position(pcf::IndiProperty());
    #endif
    // clang-format on
    reset();
    Directory  directory;
    Controller app( directory.m_path );
    bool       reversed = GENERATE( false, true );
    if( reversed )
        app.reverse();
    REQUIRE( app.appStartup() == 0 );
    app.connected();
    g_faults.m_replies.push_back( status( 1 ) );
    REQUIRE( app.appLogic() == 0 );
    g_faults.m_commands.clear();
    pcf::IndiProperty request = app.m_indiP_position;
    SECTION( "both switches are off" )
    {
        request["in"].setSwitchState( pcf::IndiElement::Off );
        request["out"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE( flipperCtrl::st_newCallBack_m_indiP_position( &app, request ) == 0 );
        REQUIRE( g_faults.m_commands.empty() );
        REQUIRE( app.target() == 1 );
    }
    SECTION( "no selection elements are supplied" )
    {
        pcf::IndiProperty empty( pcf::IndiProperty::Switch );
        empty.setDevice( request.getDevice() );
        empty.setName( request.getName() );
        REQUIRE( flipperCtrl::st_newCallBack_m_indiP_position( &app, empty ) == 0 );
        REQUIRE( g_faults.m_commands.empty() );
    }
    SECTION( "valid selection dispatches the mapped raw endpoint" )
    {
        request["in"].setSwitchState( reversed ? pcf::IndiElement::On : pcf::IndiElement::Off );
        request["out"].setSwitchState( reversed ? pcf::IndiElement::Off : pcf::IndiElement::On );
        REQUIRE( flipperCtrl::st_newCallBack_m_indiP_position( &app, request ) == 0 );
        REQUIRE( app.target() == 2 );
        REQUIRE( g_faults.m_commands == std::vector<std::string>{ std::string( "\x6a\x04\x00\x02\x50\x01", 6 ) } );
    }
    SECTION( "wrong device or property name is rejected by the dispatcher" )
    {
        bool wrongName = GENERATE( false, true );
        if( wrongName )
            request.setName( "wrong" );
        else
            request.setDevice( "wrong" );
        REQUIRE( flipperCtrl::st_newCallBack_m_indiP_position( &app, request ) == -1 );
        REQUIRE( g_faults.m_commands.empty() );
    }
}

} // namespace flipperCtrlTest
} // namespace libXWCTest
