/** \file shmimMonitor_test.hpp
 * \brief Test harness for the MagAOX::app::dev::shmimMonitor device mixin.
 *
 * One harness class, smTest, drives the real shmimMonitor code against real ImageStreamIO
 * shared memory streams. It records every allocate() and processImage() call and can make
 * either one fail or mutate the stream metadata to provoke a reconnect. It hides threadStart()
 * so a thread start failure can be injected, and it exposes the protected thread and state
 * members for direct control. ThreadGuard makes sure the monitor thread never outlives the
 * harness when a REQUIRE fails.
 *
 * The common parts of every dev:: harness, such as the FIFO-less indiDriver and the
 * registration fault injection, come from appHarnessBase in testHarnessCommon.hpp.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>

#include "../../MagAOXApp.hpp"
#include "../shmimMonitor.hpp"
#include "testHarnessCommon.hpp"

#ifndef XWCTEST_NAMESPACE
#define MAPPNS MagAOX::app::dev
#else
#define MAPPNS MagAOX::app::dev::XWCTEST_NAMESPACE
#endif

// LCOV_EXCL_START

namespace shmimMonitor_tests
{

/// Test harness for dev::shmimMonitor.
/** It forwards the MagAOXApp lifecycle calls to the mixin. It records every
 * allocate() and processImage() call and can make either one fail or mutate the
 * stream metadata to provoke a reconnect. It name-hides threadStart() so an
 * appStartup() failure can be injected, and it exposes the protected thread and
 * state members for direct control. INDI registration failures are injected through
 * m_regFailAt from appHarnessBase.
 * \ingroup shmimMonitor_tests
 */
struct smTest : public MagAOX::app::dev::testHarness::appHarnessBase, public MAPPNS::shmimMonitor<smTest>
{
    friend class MAPPNS::shmimMonitor<smTest>;

    typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;
    typedef MAPPNS::shmimMonitor<smTest> shmimMonitorT;

    // ---- allocate()/processImage() instrumentation -------------------------------
    int m_allocateCount{ 0 };
    bool m_failAllocate{ false };
    bool m_mutateOnAllocate{ false }; ///< If true, mutate size[0] right after the first allocate() call only.

    std::atomic<int> m_processImageCount{ 0 };
    bool m_failProcessImage{ false };
    int m_mutateAtProcessCount{ -1 }; ///< When m_processImageCount reaches this value, mutate size[0] and repost.

    std::vector<char> m_lastFrame;
    std::mutex m_frameMutex;

    // ---- appStartup() failure injection --------------------------------------------
    bool m_failThreadStart{ false };

    smTest() : baseT( "shmimMonitorTest" )
    {
    }

    ~smTest() noexcept override
    {
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return shmimMonitorT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return shmimMonitorT::loadConfig( config );
    }

    int appStartup()
    {
        return shmimMonitorT::appStartup();
    }

    int appLogic()
    {
        return shmimMonitorT::appLogic();
    }

    int appShutdown()
    {
        return shmimMonitorT::appShutdown();
    }

    int allocate( const MAPPNS::shmimT & )
    {
        ++m_allocateCount;

        {
            std::lock_guard<std::mutex> lk( m_frameMutex );
            m_lastFrame.assign( (size_t)m_width * m_height * m_typeSize, 0 );
        }

        if( m_mutateOnAllocate && m_allocateCount == 1 )
        {
            m_imageStream.md[0].size[0] = m_width + 1000;
        }

        if( m_failAllocate )
            return -1;

        return 0;
    }

    int processImage( void *curr_src, const MAPPNS::shmimT & )
    {
        int n = ++m_processImageCount;

        if( n == m_mutateAtProcessCount )
        {
            m_imageStream.md[0].size[0] = m_width + 1000;
            ImageStreamIO_sempost( &m_imageStream, m_semaphoreNumber );
        }

        {
            std::lock_guard<std::mutex> lk( m_frameMutex );
            if( !m_lastFrame.empty() )
                memcpy( m_lastFrame.data(), curr_src, m_lastFrame.size() );
        }

        if( m_failProcessImage )
            return -1;

        return 0;
    }

    // ---- exposing protected shmimMonitor members and methods for testing ---------

    int doCreate( uint32_t w, uint32_t h, uint32_t d, uint8_t dt, void *initData = nullptr )
    {
        return shmimMonitorT::create( w, h, d, dt, initData );
    }

    void runSmThreadExec()
    {
        shmimMonitorT::smThreadExec();
    }

    /// Run smThreadExec() on a background thread without going through threadStart().
    void startMonitorThread()
    {
        m_smThreadInit = false; // Skip the thread-priority-setup synchronizer wait.
        m_smThread = std::thread( &smTest::runSmThreadExec, this );
    }

    bool smThreadJoinable()
    {
        return m_smThread.joinable();
    }

    void joinMonitorThread()
    {
        if( m_smThread.joinable() )
            m_smThread.join();
    }

    /// Send SIGUSR1 to the monitor thread to interrupt a blocking semaphore wait.
    void killMonitorThread()
    {
        if( m_smThread.joinable() )
            pthread_kill( m_smThread.native_handle(), SIGUSR1 );
    }

    /// Safely abandon the m_smThread bookkeeping after its underlying OS thread has
    /// already been reaped by a raw pthread_tryjoin_np() call. appLogic() does that
    /// when it detects the thread has exited. std::thread has no public API for this.
    /// Both join() and detach() require the OS-level thread to still be valid, and the
    /// destructor terminates the process if joinable() is wrongly still true. swap()
    /// has no such check. So the stale id is swapped into a throwaway local, and that
    /// local is reinitialized with placement-new instead of ever running its destructor
    /// while it holds the stale id.
    void abandonSmThread()
    {
        std::thread tmp;
        tmp.swap( m_smThread );
        new( &tmp ) std::thread();
    }

    void setSmThread( std::thread &&t )
    {
        m_smThread = std::move( t );
    }

    void setRestart( bool r )
    {
        m_restart = r;
    }

    bool getRestart()
    {
        return m_restart;
    }

    MAPPNS::shmimMonitorState smState()
    {
        return m_smState;
    }

    void setShutdownFlag( int v )
    {
        m_shutdown = v;
    }

    void setShmimName( const std::string &name )
    {
        m_shmimName = name;
    }

    /// Zero the semaphore count in the stream metadata to simulate a source that cleaned up.
    void corruptSemCount()
    {
        m_imageStream.md[0].sem = 0;
    }

    void setGetExistingFirst( bool b )
    {
        m_getExistingFirst = b;
    }

    // ---- appStartup() failure injection hook. This name-hides the MagAOXApp base. ----

    template <class thisPtr, class Function>
    int threadStart( std::thread &thrd,
                      bool &thrdInit,
                      pid_t &tpid,
                      pcf::IndiProperty &thProp,
                      int thrdPrio,
                      const std::string &cpuset,
                      const std::string &thrdName,
                      thisPtr *thrdThis,
                      Function &&thrdStart )
    {
        if( m_failThreadStart )
            return -1;

        return baseT::appT::threadStart(
            thrd, thrdInit, tpid, thProp, thrdPrio, cpuset, thrdName, thrdThis, std::forward<Function>( thrdStart ) );
    }
};

/// RAII guard to make sure the background monitor thread of a test never outlives the
/// harness object, even if a REQUIRE fails partway through a test. Catch2 unwinds the
/// stack on a failed REQUIRE, so this destructor still runs. Without this, a stray
/// joinable std::thread member at harness-destruction time would call std::terminate()
/// and abort the whole test binary, which would also lose all coverage data.
struct ThreadGuard
{
    smTest &m_app;

    explicit ThreadGuard( smTest &app ) : m_app( app )
    {
    }

    ~ThreadGuard()
    {
        try
        {
            if( m_app.smThreadJoinable() )
            {
                m_app.setShutdownFlag( 1 );
                m_app.killMonitorThread();
                m_app.joinMonitorThread();
            }
        }
        catch( ... )
        {
        }
    }
};

} // namespace shmimMonitor_tests

// LCOV_EXCL_STOP
