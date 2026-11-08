/** \file po4ao_unit_test.cpp
 * \brief Exercise the real monitor loop and POSIX semaphores with an ImageStreamIO mapping double.
 * \ingroup po4ao_unit_test
 *
 * This is not an installed ImageStreamIO ABI or Linux performance test. It measures
 * actual wait-call admission and reader cleanup, on the normal attached READY path.
 */
#define CATCH_CONFIG_MAIN
#include "../../../tests/testXWC.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <semaphore.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

/// \cond MONITOR_BOUNDARIES
#define ImageStruct_hpp
#ifdef __APPLE__
    #undef SYS_gettid
    #define SYS_gettid SYS_getpid
/// Unused platform boundary; the fixture joins its actual C++ thread directly.
inline int pthread_tryjoin_np( pthread_t, void ** )
{
    return EBUSY;
}
#endif
namespace mx::app
{
struct appConfigurator
{
    template <class... T>
    void add( T &&... )
    {
    }
    template <class... T>
    void operator()( T &&... )
    {
    }
};
} // namespace mx::app
namespace mx::sys
{
inline void sleep( int s )
{
    std::this_thread::sleep_for( std::chrono::seconds( s ) );
}
} // namespace mx::sys
namespace argType
{
constexpr int Required = 0;
}
namespace pcf
{
struct IndiElement
{
    explicit IndiElement( const char * )
    {
    }
};
struct IndiProperty
{
    enum Type
    {
        Text,
        Number,
        ReadOnly,
        Idle
    };
    IndiProperty( Type = Text )
    {
    }
    void setDevice( const std::string & )
    {
    }
    void setName( const std::string & )
    {
    }
    void setPerm( Type )
    {
    }
    void setState( Type )
    {
    }
    void add( IndiElement )
    {
    }
    IndiProperty &operator[]( const char * )
    {
        return *this;
    }
    template <class T>
    IndiProperty &operator=( const T & )
    {
        return *this;
    }
};
} // namespace pcf
namespace indi
{
template <class... T>
void updateIfChanged( T &&... )
{
}
} // namespace indi
namespace MagAOX::app
{
namespace stateCodes
{
using stateCodeT    = int;
constexpr int READY = 0, OPERATING = 1;
} // namespace stateCodes
struct software_error
{
};
struct software_critical
{
};
struct text_log
{
};
namespace logPrio
{
constexpr int LOG_NOTICE = 0, LOG_INFO = 0;
}
inline void sigUsr1Handler( int, siginfo_t *, void * )
{
}
} // namespace MagAOX::app
struct Message
{
    template <class... T>
    Message( T &&... )
    {
    }
};
struct Metadata
{
    int      sem{ 10 }, naxis{ 2 };
    uint8_t  datatype{ 9 };
    uint32_t size[3]{ 2, 2, 0 };
    uint64_t cnt0{ 0 }, cnt1{ 0 }, imagetype{ 0 };
    timespec atime{}, writetime{};
};
struct IMAGE
{
    timespec *atimearray{ nullptr }, *writetimearray{ nullptr };
    uint64_t *cntarray{ nullptr };
    Metadata *md{ nullptr };
    sem_t   **semptr{ nullptr };
    pid_t    *semReadPID{ nullptr };
    struct
    {
        void *raw{ nullptr };
    } array;
};
constexpr int SEMAPHORE_MAXVAL = 10, IMAGE_NB_SEMAPHORE = 10, IMAGESTREAMIO_SUCCESS = 0, CIRCULAR_BUFFER = 1,
              ZAXIS_TEMPORAL = 2;
namespace boundary
{
inline std::atomic<unsigned> waits{ 0 }, opens{ 0 }, closes{ 0 };
inline Metadata              md;
inline sem_t                *semaphores[10]{};
inline pid_t                 readers[10]{};
inline float                 pixels[4]{};
inline std::string           path;
inline std::function<void()> afterOpen;
inline bool                  noReader = false;
} // namespace boundary
inline int ImageStreamIO_filename( char *out, size_t count, const char * )
{
    std::snprintf( out, count, "%s", boundary::path.c_str() );
    return 0;
}
inline int ImageStreamIO_openIm( IMAGE *image, const char * )
{
    image->md         = &boundary::md;
    image->semptr     = boundary::semaphores;
    image->semReadPID = boundary::readers;
    image->array.raw  = boundary::pixels;
    ++boundary::opens;
    if( boundary::afterOpen )
        boundary::afterOpen();
    return 0;
}
inline int ImageStreamIO_closeIm( IMAGE *image )
{
    ++boundary::closes;
    image->md = nullptr;
    return 0;
}
inline int ImageStreamIO_getsemwaitindex( IMAGE *, int slot )
{
    if( boundary::noReader )
        return -1;
    boundary::readers[slot] = getpid();
    return slot;
}
inline int ImageStreamIO_semflush( IMAGE *, int slot )
{
    while( sem_trywait( boundary::semaphores[slot] ) == 0 )
    {
    }
    return 0;
}
inline size_t ImageStreamIO_typesize( uint8_t )
{
    return sizeof( float );
}
inline int ImageStreamIO_destroyIm( IMAGE * )
{
    return 0;
}
inline int ImageStreamIO_createIm_gpu( IMAGE *, const char *, int, uint32_t *, uint8_t, int, int, int, int, int, int )
{
    return 0;
}
/// Instrument every actual monitor wait; Darwin needs a bounded POSIX polling fallback.
inline int countedWait( sem_t *sem, const timespec *until )
{
    ++boundary::waits;
#ifdef __APPLE__
    for( ;; )
    {
        if( sem_trywait( sem ) == 0 )
            return 0;
        if( errno != EAGAIN )
            return -1;
        timespec now{};
        clock_gettime( CLOCK_REALTIME, &now );
        if( now.tv_sec > until->tv_sec || ( now.tv_sec == until->tv_sec && now.tv_nsec >= until->tv_nsec ) )
        {
            errno = ETIMEDOUT;
            return -1;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
#else
    return sem_timedwait( sem, until );
#endif
}
#define sem_timedwait countedWait
#include "../../../libMagAOX/app/dev/shmimMonitor.hpp"
#undef sem_timedwait

namespace libXWCTest::po4aoTest
{
/// Fixture instantiating the production monitor loop, not a mirrored lifecycle.
class Monitor : public MagAOX::app::dev::shmimMonitor<Monitor>
{
  public:
    std::atomic<bool>    m_shutdown{ false };                             ///< Signal the production monitor to exit.
    std::atomic<int>     m_state{ 0 }, m_frames{ 0 }, m_allocations{ 0 }; ///< FSM and observed callback counts.
    std::function<int()> m_allocate;           ///< Optional allocation failure/cancellation injection.
    std::thread          m_thread, m_producer; ///< Owned monitor and synthetic producer threads.
    std::atomic<bool>    m_producing{ true };  ///< Producer exit request.
    std::string          m_semName;            ///< Unique process-local named semaphore.
    /// Set up an isolated producer with a unique named POSIX semaphore.
    Monitor();
    /// Stop both threads before destroying the test mapping/semaphore.
    ~Monitor();
    /// Return atomic fixture FSM state.
    int state()
    {
        return m_state.load();
    }
    /// Apply the FSM state used by the actual monitor.
    void state( int state /**< [in] Desired FSM state. */ )
    {
        m_state = state;
    }
    /// Read the fixture shutdown flag.
    int shutdown()
    {
        return m_shutdown.load();
    }
    /// Observe allocation and optionally inject cancellation/failure.
    int allocate( const MagAOX::app::dev::shmimT & /**< [in] Primary monitor tag. */ )
    {
        ++m_allocations;
        return m_allocate ? m_allocate() : 0;
    }
    /// Count admitted frames.
    int processImage( void * /**< [in] Synthetic frame. */, const MagAOX::app::dev::shmimT & /**< [in] Monitor tag. */ )
    {
        ++m_frames;
        return 0;
    }
    /// Discard test logs without starting the application logging system.
    template <class T, int value = 0>
    static int log( Message, int = 0 )
    {
        return value;
    }
    /// Start the actual monitor implementation.
    void run()
    {
        m_thread = std::thread( [this] { smThreadExec(); } );
    }
};
Monitor::Monitor()
{
    boundary::waits = boundary::opens = boundary::closes = 0;
    boundary::afterOpen                                  = {};
    boundary::noReader                                   = false;
    std::fill( std::begin( boundary::readers ), std::end( boundary::readers ), 0 );
    m_semName = "/smtest-" + std::to_string( getpid() );
    sem_unlink( m_semName.c_str() );
    auto *sem = sem_open( m_semName.c_str(), O_CREAT | O_EXCL, 0600, 0 );
    if( sem == SEM_FAILED )
        throw std::runtime_error( "Cannot create private test semaphore" );
    for( auto &p : boundary::semaphores )
        p = sem;
    boundary::path = "/tmp/smtest-" + std::to_string( getpid() ) + ".im.shm";
    std::ofstream( boundary::path ) << "mapping boundary";
    m_shmimName    = "private-monitor-test";
    m_smThreadInit = false;
    m_producer     = std::thread(
        [this]
        {
            while( m_producing )
            {
                sem_post( boundary::semaphores[5] );
                std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
            }
        } );
}
Monitor::~Monitor()
{
    m_shutdown  = true;
    m_producing = false;
    if( m_thread.joinable() )
        m_thread.join();
    if( m_producer.joinable() )
        m_producer.join();
    sem_close( boundary::semaphores[5] );
    sem_unlink( m_semName.c_str() );
    std::filesystem::remove( boundary::path );
}
/// Wait boundedly for a test observation.
bool await( const std::function<bool()> &ready /**< [in] Completion predicate. */ )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );
    while( !ready() && std::chrono::steady_clock::now() < end )
        std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    return ready();
}
/// \endcond

/** \brief Continued producer notifications do not cause waits after READY cleanup; resume reacquires the reader. */
TEST_CASE( "Monitor READY releases the reader and stops wait calls", "[shmimMonitor][lifecycle]" )
{
    // clang-format off
#ifdef SHMIMMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::dev::shmimMonitor<Monitor>::smThreadExec;
#endif
    // clang-format on
    Monitor app;
    app.run();
    REQUIRE( boundary::waits == 0 );
    for( int cycle = 0; cycle < 3; ++cycle )
    {
        const auto frames = app.m_frames.load();
        app.state( MagAOX::app::stateCodes::OPERATING );
        REQUIRE( await( [&] { return app.m_frames > frames + 2; } ) );
        app.state( MagAOX::app::stateCodes::READY );
        REQUIRE( await( [&] { return boundary::closes == static_cast<unsigned>( cycle + 1 ); } ) );
        const auto waits        = boundary::waits.load();
        const auto parkedFrames = app.m_frames.load();
        REQUIRE( boundary::readers[5] == 0 );
        REQUIRE( boundary::opens == boundary::closes );
        std::this_thread::sleep_for( std::chrono::milliseconds( 80 ) );
        REQUIRE( boundary::waits == waits );
        REQUIRE( app.m_frames == parkedFrames );
    }
}

} // namespace libXWCTest::po4aoTest
