/** \file shmimMonitor_timing_test.cpp
 * \brief Offline semaphore and lifecycle coverage for optional monitor timing hooks.
 */
#include "../../../../tests/testXWC.hpp"
#include "../../MagAOXApp.hpp"
#include "../shmimMonitor.hpp"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>

/// \cond SHMIM_TIMING_TEST_HARNESS
namespace
{
using namespace MagAOX::app;
using dev::shmimMonitorEvent;

/// Keep ImageStreamIO's cached shared-memory directory alive for the complete test process.
struct sharedDirectory
{
    std::string m_path; ///< Process-wide private stream directory.

    std::string m_previous; ///< Previous environment path, if one existed.

    bool m_hadPrevious{ false }; ///< Previous environment defined MILK_SHM_DIR.

    /// Install the private path before the first ImageStreamIO operation.
    sharedDirectory();

    /// Remove the directory after every fixture has stopped and restore the environment.
    ~sharedDirectory();
};

sharedDirectory::sharedDirectory()
{
    char directory[] = "/tmp/shmim-timing-test-XXXXXX";
    auto result      = mkdtemp( directory );
    REQUIRE( result != nullptr );
    m_path               = result;
    const char *previous = getenv( "MILK_SHM_DIR" );
    m_hadPrevious        = previous != nullptr;
    if( previous )
        m_previous = previous;
    REQUIRE( setenv( "MILK_SHM_DIR", m_path.c_str(), 1 ) == 0 );
}

sharedDirectory::~sharedDirectory()
{
    if( m_hadPrevious )
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    else
        unsetenv( "MILK_SHM_DIR" );
    std::filesystem::remove_all( m_path );
}

/// Get process-lifetime storage required by ImageStreamIO's cached directory lookup.
sharedDirectory &streamDirectory();

sharedDirectory &streamDirectory()
{
    static sharedDirectory directory;
    return directory;
}

/// Minimal offline monitor owner with an optionally absent timing observer.
template <bool withHook>
class monitorHarness : public dev::shmimMonitor<monitorHarness<withHook>>
{
    friend class dev::shmimMonitor<monitorHarness<withHook>>;

    using monitorT = dev::shmimMonitor<monitorHarness<withHook>>;

  protected:
    std::atomic<stateCodes::stateCodeT> m_state{ stateCodes::OPERATING }; ///< Worker-visible application state.

    mutable std::mutex m_mutex; ///< Protect fixture observations, not the stream under test.

    std::condition_variable m_changed; ///< Bounded waits for real monitor events.

    std::vector<int> m_observations; ///< Hook IDs and callback marker in execution order.

    bool m_pendingReceipt{ false }; ///< Worker-local receipt precedes the next callback.

  public:
    std::atomic<bool> m_shutdown{ false }; ///< Normal monitor exit condition, without signals.

    std::atomic<bool> m_restartOnReceipt{ false }; ///< Request one worker-local restart from its observation hook.

    std::atomic<bool> m_rejectStateOnReceipt{ false }; ///< Request one state rejection after the next successful wait.

    std::atomic<unsigned> m_callbacks{ 0 }; ///< Number of actual processImage invocations.

    std::atomic<unsigned> m_receiptCallbacks{ 0 }; ///< Callbacks that observed a preceding real receipt.

    /// Return the ordinary monitor shutdown condition.
    bool shutdown() const;

    /// Return the ordinary monitor application state.
    stateCodes::stateCodeT state() const;

    /// Suppress logging while exercising the real monitor loop on a private stream.
    template <class logT>
    static int log( const typename logT::messageT &message /**< [in] Unused test log payload. */,
                    flatlogs::logPrioT priority = flatlogs::logPrio::LOG_DEFAULT /**< [in] Unused priority. */ );

    /// Accept the locally imported stream descriptor without allocating controller resources.
    int allocate( const dev::shmimT &specific /**< [in] Default monitor tag. */ );

    /// Record one callback after the real monitor's metadata/state checks.
    int processImage( void              *pixels /**< [in] Unused private stream pixels. */,
                      const dev::shmimT &specific /**< [in] Default monitor tag. */ );

    /// Optional hook; substitution removes this interface completely for the fallback fixture.
    template <bool enabled = withHook, std::enable_if_t<enabled, int> = 0>
    void shmimMonitorTiming( const dev::shmimT &specific /**< [in] Default monitor tag. */,
                             shmimMonitorEvent  event /**< [in] Actual monitor event. */ );

    /// Start only the real monitor loop, without INDI, priorities or instrument access.
    void start( const std::string &name /**< [in] Existing private workstation stream. */,
                bool               initial /**< [in] Process existing pixels before waiting. */ );

    /// Request exit and join after a caller posts the private stream semaphore.
    void join();

    /// Wait at most three seconds for a specified observation count.
    bool wait( int    observation /**< [in] Hook event ID, or 100 for a callback. */,
               size_t count /**< [in] Required count. */ );

    /// Copy observations while the worker is running.
    std::vector<int> observations() const;
};

template <bool withHook>
bool monitorHarness<withHook>::shutdown() const
{
    return m_shutdown.load();
}

template <bool withHook>
stateCodes::stateCodeT monitorHarness<withHook>::state() const
{
    return m_state.load();
}

template <bool withHook>
template <class logT>
int monitorHarness<withHook>::log( const typename logT::messageT &message, flatlogs::logPrioT priority )
{
    static_cast<void>( message );
    static_cast<void>( priority );
    return 0;
}

template <bool withHook>
int monitorHarness<withHook>::allocate( const dev::shmimT &specific )
{
    static_cast<void>( specific );
    return 0;
}

template <bool withHook>
int monitorHarness<withHook>::processImage( void *pixels, const dev::shmimT &specific )
{
    static_cast<void>( pixels );
    static_cast<void>( specific );
    m_receiptCallbacks.fetch_add( m_pendingReceipt );
    m_pendingReceipt = false;
    m_callbacks.fetch_add( 1 );
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_observations.push_back( 100 );
    }
    m_changed.notify_all();
    return 0;
}

template <bool withHook>
template <bool enabled, std::enable_if_t<enabled, int>>
void monitorHarness<withHook>::shmimMonitorTiming( const dev::shmimT &specific, shmimMonitorEvent event )
{
    static_cast<void>( specific );
    if( event == shmimMonitorEvent::receipt )
    {
        m_pendingReceipt = true;
        if( m_restartOnReceipt.exchange( false ) )
            monitorT::m_restart = true;
        if( m_rejectStateOnReceipt.exchange( false ) )
            m_state.store( stateCodes::READY );
    }
    else
        m_pendingReceipt = false;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_mutex );
        m_observations.push_back( static_cast<int>( event ) );
    }
    m_changed.notify_all();
}

template <bool withHook>
void monitorHarness<withHook>::start( const std::string &name, bool initial )
{
    monitorT::m_shmimName        = name;
    monitorT::m_getExistingFirst = initial;
    monitorT::m_smThreadInit     = false;
    monitorT::m_smThread         = std::thread( [this]() { monitorT::smThreadExec(); } );
}

template <bool withHook>
void monitorHarness<withHook>::join()
{
    if( monitorT::m_smThread.joinable() )
        monitorT::m_smThread.join();
}

template <bool withHook>
bool monitorHarness<withHook>::wait( int observation, size_t count )
{
    std::unique_lock<std::mutex> lock( m_mutex );
    return m_changed.wait_for( lock,
                               std::chrono::seconds( 3 ),
                               [&]
                               {
                                   return static_cast<size_t>( std::count(
                                              m_observations.begin(), m_observations.end(), observation ) ) >= count;
                               } );
}

template <bool withHook>
std::vector<int> monitorHarness<withHook>::observations() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_observations;
}

/// Own one private workstation stream and stop its monitor before unmapping it.
template <bool withHook>
struct streamFixture
{
    IMAGE m_image{}; ///< Producer mapping owned exclusively by this fixture.

    std::string m_name; ///< Unique stream filename within the private test directory.

    monitorHarness<withHook> m_monitor; ///< Real monitor worker, without an app or instrument process.

    /// Create a private four-by-four float stream and start its monitor.
    explicit streamFixture( bool initial /**< [in] Enable the initial existing-image callback. */ );

    /// Join the worker before destroying and unmapping its input stream.
    ~streamFixture();

    /// Publish exactly one token to the real monitor semaphore.
    void publish();
};

template <bool withHook>
streamFixture<withHook>::streamFixture( bool initial )
{
    static_cast<void>( streamDirectory() );
    static unsigned sequence = 0;
    m_name                   = "timing" + std::to_string( getpid() ) + "x" + std::to_string( ++sequence );
    uint32_t size[3]         = { 4, 4, 0 };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image, m_name.c_str(), 2, size, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) ==
             0 );
    m_monitor.start( m_name, initial );
}

template <bool withHook>
streamFixture<withHook>::~streamFixture()
{
    m_monitor.m_shutdown.store( true );
    ImageStreamIO_sempost( &m_image, -1 );
    m_monitor.join();
    ImageStreamIO_destroyIm( &m_image );
}

template <bool withHook>
void streamFixture<withHook>::publish()
{
    ++m_image.md[0].cnt0;
    REQUIRE( ImageStreamIO_sempost( &m_image, -1 ) == 0 );
}

} // namespace
/// \endcond

/** \defgroup shmimMonitor_timing_tests Optional monitor timing hook tests
 * \ingroup app_dev_unit_tests
 */

/** \brief Initial images have no fake receipt and successful waits are observed before callbacks.
 * \ingroup shmimMonitor_timing_tests
 */
TEST_CASE( "Monitor timing distinguishes existing pixels from semaphore receipts", "[dev::shmimMonitor][timing]" )
{
    streamFixture<true> fixture( true );
    REQUIRE( fixture.m_monitor.wait( 100, 1 ) );
    auto observations = fixture.m_monitor.observations();
    REQUIRE( observations.size() == 3 );
    CHECK( observations[0] == static_cast<int>( shmimMonitorEvent::connected ) );
    CHECK( observations[1] == static_cast<int>( shmimMonitorEvent::initial ) );
    CHECK( observations[2] == 100 );
    CHECK( fixture.m_monitor.m_receiptCallbacks.load() == 0 );
    fixture.publish();
    REQUIRE( fixture.m_monitor.wait( 100, 2 ) );
    observations = fixture.m_monitor.observations();
    REQUIRE( observations.size() == 5 );
    CHECK( observations[3] == static_cast<int>( shmimMonitorEvent::receipt ) );
    CHECK( observations[4] == 100 );
    CHECK( fixture.m_monitor.m_receiptCallbacks.load() == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::shmimMonitor::smThreadExec;
    MagAOX::app::dev::shmimMonitor::timingHook;
#endif
    // clang-format on
}

/** \brief A successful receipt discarded during restart is finalized before a fresh attachment.
 * \ingroup shmimMonitor_timing_tests
 */
TEST_CASE( "Monitor timing reports discarded receipts and reattachment", "[dev::shmimMonitor][timing]" )
{
    streamFixture<true> fixture( true );
    REQUIRE( fixture.m_monitor.wait( 100, 1 ) );
    fixture.m_monitor.m_restartOnReceipt.store( true );
    fixture.publish();
    REQUIRE( fixture.m_monitor.wait( static_cast<int>( shmimMonitorEvent::discarded ), 1 ) );
    REQUIRE( fixture.m_monitor.wait( 100, 2 ) );
    auto                   observations = fixture.m_monitor.observations();
    const std::vector<int> expected{ static_cast<int>( shmimMonitorEvent::connected ),
                                     static_cast<int>( shmimMonitorEvent::initial ),
                                     100,
                                     static_cast<int>( shmimMonitorEvent::receipt ),
                                     static_cast<int>( shmimMonitorEvent::discarded ),
                                     static_cast<int>( shmimMonitorEvent::connected ),
                                     static_cast<int>( shmimMonitorEvent::initial ),
                                     100 };
    CHECK( observations == expected );
    CHECK( fixture.m_monitor.m_receiptCallbacks.load() == 0 );
    fixture.publish();
    REQUIRE( fixture.m_monitor.wait( 100, 3 ) );
    CHECK( fixture.m_monitor.m_receiptCallbacks.load() == 1 );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::shmimMonitor::smThreadExec;
    MagAOX::app::dev::shmimMonitor::timingHook;
#endif
    // clang-format on
}

/** \brief A state change after receipt rejects the notification without forwarding pixels.
 * \ingroup shmimMonitor_timing_tests
 */
TEST_CASE( "Monitor timing records state-rejected notifications", "[dev::shmimMonitor][timing]" )
{
    streamFixture<true> fixture( false );
    REQUIRE( fixture.m_monitor.wait( static_cast<int>( shmimMonitorEvent::connected ), 1 ) );
    fixture.m_monitor.m_rejectStateOnReceipt.store( true );
    fixture.publish();
    REQUIRE( fixture.m_monitor.wait( static_cast<int>( shmimMonitorEvent::discarded ), 1 ) );
    CHECK( fixture.m_monitor.m_callbacks.load() == 0 );
    auto                   observations = fixture.m_monitor.observations();
    const std::vector<int> expected{ static_cast<int>( shmimMonitorEvent::connected ),
                                     static_cast<int>( shmimMonitorEvent::receipt ),
                                     static_cast<int>( shmimMonitorEvent::discarded ) };
    CHECK( observations == expected );

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::shmimMonitor::smThreadExec;
    MagAOX::app::dev::shmimMonitor::timingHook;
#endif
    // clang-format on
}

/** \brief A timed-out wait emits no receipt and owners lacking the optional observer remain supported.
 * \ingroup shmimMonitor_timing_tests
 */
TEST_CASE( "Monitor timing ignores timeouts and supports owners without hooks", "[dev::shmimMonitor][timing]" )
{
    SECTION( "Timeout is not a receipt" )
    {
        streamFixture<true> fixture( false );
        REQUIRE( fixture.m_monitor.wait( static_cast<int>( shmimMonitorEvent::connected ), 1 ) );
        std::this_thread::sleep_for( std::chrono::milliseconds( 1150 ) );
        const std::vector<int> expected{ static_cast<int>( shmimMonitorEvent::connected ) };
        CHECK( fixture.m_monitor.observations() == expected );
        fixture.publish();
        REQUIRE( fixture.m_monitor.wait( 100, 1 ) );
        CHECK( fixture.m_monitor.m_receiptCallbacks.load() == 1 );
    }
    SECTION( "No hook interface" )
    {
        streamFixture<false> fixture( true );
        REQUIRE( fixture.m_monitor.wait( 100, 1 ) );
        fixture.publish();
        REQUIRE( fixture.m_monitor.wait( 100, 2 ) );
        const std::vector<int> expected{ 100, 100 };
        CHECK( fixture.m_monitor.observations() == expected );
    }

    // clang-format off
#ifdef XWCTEST_DOX_REF
    MagAOX::app::dev::shmimMonitor::smThreadExec;
    MagAOX::app::dev::shmimMonitor::timingHook;
#endif
    // clang-format on
}
