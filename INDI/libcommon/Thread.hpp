/// Thread.hpp
///
/// @author Paul Grenz
///
/// The "Thread" class provides the functionality of a recurring
/// event or task. The way this is implemented is by having a separate thread
/// carry out the waiting in between events, and when the interval has elapsed,
/// the "execute" function is called. When the thread is started, the
/// "beforeExecute" function is called, and this can be used to set up any
/// variables or objects which the thread uses. In a symmetrical fashon,
/// "afterExecute" is called after the thread has been told to stop, either by
/// calling "stop" or the by the object being destroyed. This function can be
/// used to "clean-up" after the thread is done. To start the thread, call
/// "start". When stopping, a sigquit is sent to thread, ensuring that it will
/// not be stuck in a system call, but will return and process the new
/// 'stopped' condition.
///
/// Thread also supports "pausing". This is accomplished by calling "pause"
/// and conversly "resume". When the thread is paused, it will still respond
/// to "stop" being called. While paused, it will be 'sleeping' and only wake
/// up in response to a 'resume' or 'stop'.
///
/// After calling stop, call "join" to synchronize the main application thread
/// with this thread. This will ensure that the application does not perform an
/// operation (such as exiting!) until the thread has cleaned up its worker
/// thread.
///
/// Another concept this class supports is the idea of a 'trigger'. This can
/// be used to run the 'execute' method only when a certain event happens, not
/// only after 'sleeping' for a fixed time interval. With a socket set as a
/// trigger, for example, (calling 'setTrigger' with a &socket argument) the
/// 'beforeExecute' method will be run as before, but the 'execute' method
/// will only run if data is ready to be read on the socket. All other times,
/// this thread will be asleep, and only wake up if a signal is received to
/// stop it running. If the socket is closed for any reason, the trigger will
/// be ignored, and the execute method will be run as if there was no trigger
/// set.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_THREAD_HPP
#define PCF_THREAD_HPP

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdexcept>
#include "MutexLock.hpp"
#include "SystemSocket.hpp"
#ifdef WIN32
#include <windows.h>
#endif

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Thread
{
  // Constants.
  public:
    enum Error
    {
      ErrNone =                     0, // This value must stay zero.
      ErrThreadUnjoinable =        -ENODEV,
      ErrBadThreadId =             -ESRCH,
      ErrDeadlock =                -EDEADLK,
      ErrCouldNotCreateThread =    -EAGAIN,
      ErrInvalidParameter  =       -EINVAL,
      ErrWrongPermission =         -EPERM,
      ErrAlreadyRunning =          -EALREADY,
      ErrInterrupted =             -EINTR,
      ErrCopy =                    -EFAULT,
      ErrTimedOut =                -ETIMEDOUT,
      ErrNotRunning =              -EHOSTDOWN,
      ErrCpuNumberOutOfRange =     -ENOENT,
      ErrUnknown =                 -9999
    };

    enum State
    {
      Idle =          0,
      BeforeExecute = 1,
      Execute =       2,
      AfterExecute =  3
    };

    enum ScheduleType
    {
      Inherit    = 0,
      Normal     = 1,
      Turbo      = 2,
    };

  // Constructor/destructor/operators.
  public:
    Thread();
    virtual ~Thread();
    Thread( const Thread &copy );
    const Thread &operator =( const Thread &copy );

  // Methods.
  public:
    /// Return the message concerning the error.
    static std::string getErrorMsg( const int &nErr );
    /// Returns the number of milliseconds between the 'execute' firing.
    unsigned int getInterval() const;
    /// Returns the current state of the thread.
    State getState() const;
    /// Returns true if the 'stop' flag has been set to true. The thread
    /// may still be running its last iteration, though. (see 'isRunning').
    bool isStopping() const;
    /// Is the thread "paused"?
    bool isPaused() const;
    /// Is the thread "running"?
    bool isRunning() const;
    /// Join the current thread back to the parent thread.
    int join();
    /// Pause the execution of the thread. "stop" will still end it.
    void pause();
    /// Resumes the Thread if it is paused. Calling 'resume' twice or more
    /// will have no effect.
    void resume();
    /// Resumes the Thread if it is paused, but only for one iteration. Calling
    /// 'resumeOnce' twice or more will have no effect.
    void resumeOnce();
    /// This is the interval between the "execute" functions firing.
    void setInterval( const unsigned int &nMSecs );
    /// should this thread only execute once?
    void setOneShot( const bool &oOneShot );
    /// Sets the socket file descriptor which will be used to cause the 'execute'
    /// method to run. If the socket descriptor is valid, the 'runLoop' will
    /// check to see if there is data waiting to be read on the socket and will
    /// not call 'execute' until there is.
    void setTrigger( pcf::SystemSocket *psocTrigger );
    /// Starts the thread running.
    int start( const int &iCpuAffinity = -1,
               const ScheduleType &tSchedule = Inherit );
    /// Stops the 'execute' function from firing by stopping the thread.
    void stop();
    /// Waits for the 'ready' mutex to be unlocked. This indicates that the
    /// 'execute' function is about to be run.
    void waitForReady();

   // Sleep and thread id functions.
  public:
    /// Put the calling thread to sleep for uiMillis milliseconds.
    static int msleep( const unsigned int &uiMillis );
    /// Put the calling thread to sleep for a timespec amount of time.
    static int nanosleep( const unsigned int &uiSeconds,
                          const unsigned int &uiNanos );
    /// Put the calling thread to sleep for uiNanos nanoseconds.
    static int nsleep( const unsigned int &uiNanos );
    /// Put the calling thread to sleep for uiSeconds seconds.
    static int sleep( const unsigned int &uiSeconds );
    /// Put the calling thread to sleep for uiMicros microseconds.
    static int usleep( const unsigned int &uiMicros );
    /// This will handle the signal which forces any system calls to return.
    static void processSignalHandler( int nSignal );

  // Overridable methods - these give the thread its personality.
  public:
    virtual void afterExecute();
    virtual void beforeExecute();
    virtual void execute();

  // Helper functions.
  protected:
#ifdef WIN32
    static DWORD WINAPI threadFuncWin( void *pUnknown );
#else
    static void *pthreadFunc( void *pUnknown );
#endif
    void runLoop();
    // Create and open the pipe for the pause/resume functionality.
    void setupPipe();

  // Variables.
  private:
    /// id or handle of the thread created.
#ifdef WIN32
    HANDLE m_idThis;
#else
    pthread_t m_idThis;
#endif
    /// the interval between calls of "execute".
    unsigned int m_uiInterval;
    /// The state the thread is currently in.
    State m_tState;
    /// Should we stop the thread?.
    bool m_oStop;
    /// Are we only performing this task ('execute' function) once?
    bool m_oOneShot;
    /// Is the thread performing the 'execute' function?
    bool m_oIsRunning;
    /// Are we pausing the thread for some amount of time?
    bool m_oIsPaused;
    /// The signal to send to make sure the thread returns from system calls.
    static int sm_nStopSignal;
    /// A structure to hold information about the signal action.
    struct sigaction m_saStop;
    /// This socket will be used to trigger the next iteration of the
    /// 'execute' loop if it is valid and it has data to read.
    pcf::SystemSocket *m_psocTrigger;
    /// This pipe is used as a way to resume the thread after it has
    /// been paused.
    int m_pfdPipe[2];
    /// A mutex that stays locked until this class is running its 'execute'
    /// method. It is locked on creation, and unlocked when 'execute' is called.
    mutable pcf::MutexLock m_mutReady;

}; // Class Thread
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_THREAD_HPP
