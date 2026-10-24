/** \file MagAOXApp_threadStart_test.cpp
 * \brief Inject positive pthread scheduler errors and verify thread-start diagnostics preserve the returned code.
 * \ingroup app_files
 */
#include <atomic>

#include "../../../tests/testXWC.hpp"
#include "../MagAOXApp.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace threadStartTestDetail
{
inline std::atomic<bool> workerReady{ false };    ///< Release/acquire handoff for the test worker's PID.
inline int               schedulerResult{ 0 };    ///< Injected pthread return value, independently of errno.
inline int               schedulerCalls{ 0 };     ///< Number of intercepted calls during the current trial.
inline int               schedulerPolicy{ -1 };   ///< Policy requested by the actual production path.
inline int               schedulerPriority{ -1 }; ///< Priority requested by the actual production path.

/// Capture the production request while injecting a POSIX positive error without changing the real scheduler.
int setScheduler( pthread_t          thread /**< [in] worker handle, unused */,
                  int                policy /**< [in] requested scheduler policy */,
                  const sched_param *parameters /**< [in] requested scheduler parameters */ )
{
    static_cast<void>( thread );

    while( !workerReady.load( std::memory_order_acquire ) )
    {
        std::this_thread::yield();
    }

    ++schedulerCalls;
    schedulerPolicy   = policy;
    schedulerPriority = parameters->sched_priority;
    errno = EBUSY; // Deliberately differs from the returned error and may be changed by privilege restoration.
    return schedulerResult;
}
} // namespace threadStartTestDetail

#undef app_MagAOXApp_hpp
#define XWCTEST_NAMESPACE XWCTEST_THREAD_SCHEDULER_ns
#define pthread_setschedparam ::threadStartTestDetail::setScheduler
#include "../MagAOXApp.hpp"
#undef pthread_setschedparam
#undef XWCTEST_NAMESPACE
/// \endcond

namespace libXWCTest
{
namespace appTest
{
namespace MagAOXAppTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
using SchedulerBase = MagAOX::app::XWCTEST_THREAD_SCHEDULER_ns::MagAOXApp<false>;

/// Minimal worker-owning application; no hardware, INDI driver or logger thread is started.
class SchedulerApp : public SchedulerBase
{
  public:
    pid_t pid{ 0 }; ///< PID populated by the worker before the injected scheduler call completes.

    /// Construct an isolated application instance with its own injected base specialization.
    SchedulerApp();

    /// Satisfy the application startup interface without external actions.
    int appStartup() override;

    /// Satisfy the ordinary application interface without external actions.
    int appLogic() override;

    /// Satisfy the application shutdown interface without external actions.
    int appShutdown() override;

    /// Publish the real local worker TID through a release/acquire test handoff.
    static void worker( SchedulerApp *app /**< [in/out] worker-owning test application */ );
};

SchedulerApp::SchedulerApp() : SchedulerBase( "test", false )
{
}

int SchedulerApp::appStartup()
{
    return 0;
}

int SchedulerApp::appLogic()
{
    return 0;
}

int SchedulerApp::appShutdown()
{
    return 0;
}

void SchedulerApp::worker( SchedulerApp *app )
{
    app->pid = static_cast<pid_t>( syscall( SYS_gettid ) );
    threadStartTestDetail::workerReady.store( true, std::memory_order_release );
}

/// Access protected queue members through legal member pointers without replacing production logging.
struct LoggerQueueView : SchedulerBase::logManagerT
{
    /// Drain this specialization's real queued logs without starting its file-writing thread.
    static std::vector<flatlogs::bufferPtrT>
    drain( SchedulerBase::logManagerT &logger /**< [in/out] isolated base specialization's logger */ );
};

std::vector<flatlogs::bufferPtrT> LoggerQueueView::drain( SchedulerBase::logManagerT &logger )
{
    auto                             &mutex = logger.*&LoggerQueueView::m_qMutex;
    auto                             &queue = logger.*&LoggerQueueView::m_logQueue;
    std::lock_guard<std::mutex>       lock( mutex );
    std::vector<flatlogs::bufferPtrT> result( queue.begin(), queue.end() );
    queue.clear();
    return result;
}
/// \endcond

/// Preserve positive pthread errors in software logs and never report scheduler success for a failed request.
/** \ingroup MagAOXApp_unit_test */
TEST_CASE( "MagAOXApp threadStart reports returned scheduler errors", "[app::MagAOXApp][threadStart]" )
{
    for( const int result : { EPERM, EINVAL, 0 } )
    {
        for( const int priority : { 30, 0 } )
        {
            INFO( "injected scheduler result " << result << " requested priority " << priority );
            SchedulerApp app;
            LoggerQueueView::drain( app.m_log );
            threadStartTestDetail::workerReady.store( false, std::memory_order_relaxed );
            threadStartTestDetail::schedulerResult = result;
            threadStartTestDetail::schedulerCalls  = 0;
            std::thread       worker;
            bool              initializing = false;
            pcf::IndiProperty property;
            const int         started = app.threadStart(
                worker, initializing, app.pid, property, priority, "", "testWorker", &app, &SchedulerApp::worker );
            worker.join();
            const auto logs = LoggerQueueView::drain( app.m_log );
            REQUIRE( started == 0 );
            REQUIRE_FALSE( initializing );
            REQUIRE( app.pid > 0 );
            REQUIRE( threadStartTestDetail::schedulerCalls == 1 );
            REQUIRE( threadStartTestDetail::schedulerPolicy == ( priority ? MAGAOX_RT_SCHED_POLICY : SCHED_OTHER ) );
            REQUIRE( threadStartTestDetail::schedulerPriority == priority );
            size_t schedulerErrors  = 0;
            size_t schedulerSuccess = 0;

            for( const auto &buffer : logs )
            {
                const auto code    = flatlogs::logHeader::eventCode( buffer.get() );
                auto      *payload = flatlogs::logHeader::messageBuffer( buffer.get() );

                if( code == MagAOX::logger::software_error::eventCode )
                {
                    const auto *message = MagAOX::logger::GetSoftware_log_fb( payload );

                    if( message->explanation() &&
                        message->explanation()->str().find( "scheduler priority" ) != std::string::npos )
                    {
                        ++schedulerErrors;
                        REQUIRE( message->errnoCode() == result );
                        REQUIRE( message->errnoCode() != EBUSY );
                    }
                }
                else if( code == MagAOX::logger::text_log::eventCode )
                {
                    const auto message =
                        MagAOX::logger::text_log::msgString( payload, flatlogs::logHeader::msgLen( buffer.get() ) );

                    if( message.find( "scheduler priority set to" ) != std::string::npos )
                    {
                        ++schedulerSuccess;
                    }
                }
            }

            REQUIRE( schedulerErrors == ( result ? 1 : 0 ) );
            REQUIRE( schedulerSuccess == ( result ? 0 : 1 ) );
        }
    }

    // clang-format off
#ifdef MAGAOXAPP_TEST_DOXYGEN_REF
    MagAOX::app::MagAOXApp<false>::threadStart;
#endif
    // clang-format on
}

} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest
