/// Thread.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "Thread.hpp"
#ifdef WIN32
#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>
#endif

using std::string;
using pcf::Thread;
using pcf::SystemSocket;

////////////////////////////////////////////////////////////////////////////////
/// pcf::Thread::sm_nStopSignal is the signal sent to ensure that
/// system calls exit with a 'EINTR' error.

int pcf::Thread::sm_nStopSignal = SIGQUIT;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

Thread::Thread() : m_idThis(0), m_uiInterval(1000), m_tState(Idle), m_oStop(true), 
                    m_oOneShot(false), m_oIsRunning(false), m_oIsPaused(false), m_psocTrigger(NULL)
{ 
  // We are not running in 'execute' yet.
  m_mutReady.lock();

  setupPipe();
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor. Will stop any thread automatically.

Thread::~Thread()
{
  stop();
  join();

  ::close( m_pfdPipe[0] );
  ::close( m_pfdPipe[1] );
  m_pfdPipe[0] = -1;
  m_pfdPipe[1] = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.
/// @param copy An existing Thread that is being copied. This one will be
/// stopped if it is currently running, and once assigned will not initially
/// be running.
/// @return A reference to this Thread object.

const Thread &Thread::operator=( const Thread &copy )
{
  if ( &copy != this )
  {
    stop();
    join();

    m_uiInterval = copy.m_uiInterval;
    m_oOneShot = copy.m_oOneShot;
    m_idThis = 0;
    m_oStop = true;
    m_oIsRunning = false;
    m_oIsPaused = false;
    m_psocTrigger = NULL;
    m_tState = Idle;

    // We are not running in 'execute' yet.
    m_mutReady.lock();
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.
/// @param copy An existing Thread that will be used to initialize this one. The
/// parameters will be copied, but if the original is running, this one will not
/// be.

Thread::Thread( const Thread &copy ) : m_idThis(0), m_uiInterval(copy.m_uiInterval), m_tState(Idle), m_oStop(true), 
                                      m_oOneShot(copy.m_oOneShot), m_oIsRunning(false), m_oIsPaused(false), m_psocTrigger(NULL)
{
  // We are not running in 'execute' yet.
  m_mutReady.lock();

  setupPipe();
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the pipe - this is used to kick the thread out of 'select';
/// (the infamous 'self-pipe' trick)

void Thread::setupPipe()
{
  // Set the file descriptors to an invald value.
  // If we cannot initialize the fd's correctly, we will fail when we try
  // to pause.
  m_pfdPipe[0] = -1;
  m_pfdPipe[1] = -1;

  if ( ::pipe( m_pfdPipe ) == 0 )
  {
    // Set both the read and write ends non-blocking.
    for ( uint64_t ii = 0; ii < 2; ii++ )
    {
      int iFlags = 0;
      if ( ( iFlags = ::fcntl( m_pfdPipe[ii], F_GETFL ) ) == -1 )
      {
        ::close( m_pfdPipe[0] );
        ::close( m_pfdPipe[1] );
        m_pfdPipe[0] = -1;
        m_pfdPipe[1] = -1;
        break;
      }
      iFlags |= O_NONBLOCK;
      if ( ::fcntl( m_pfdPipe[ii], F_SETFL, iFlags ) == -1 )
      {
        ::close( m_pfdPipe[0] );
        ::close( m_pfdPipe[1] );
        m_pfdPipe[0] = -1;
        m_pfdPipe[1] = -1;
        break;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// @return The interval to wait in between 'execute' calls.

unsigned int Thread::getInterval() const
{
  return m_uiInterval;
}

////////////////////////////////////////////////////////////////////////////////
/// @return The current state the running thread is in.

Thread::State Thread::getState() const
{
  return m_tState;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the socket file descriptor which will be used to cause the 'execute'
/// method to run. If the socket descriptor is valid, the 'runLoop' will
/// check to see if there is data waiting to be read on the socket and will
/// not call 'execute' until there is.
///
void Thread::setTrigger( pcf::SystemSocket *psocTrigger )
{
  m_psocTrigger = psocTrigger;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the interval to wait in between 'execute' calls.

void Thread::setInterval( const unsigned int &uiInterval )
{
  m_uiInterval = uiInterval;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if the 'stop' flag has been set to true. The thread
/// may still be running its last iteration, though. (see 'isRunning').

bool Thread::isStopping() const
{
  return m_oStop;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether or not the Thread is running. "isRunning" will still return
/// true, even if the Thread is paused.
/// @return True if the Thread is running, false if it is not.

bool Thread::isRunning() const
{
  return m_oIsRunning;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether or not the Thread is paused. "isRunning" will still return
/// true, even if the Thread is paused.
/// @return True if the Thread is paused, false if it is not.

bool Thread::isPaused() const
{
  return m_oIsPaused;
}

////////////////////////////////////////////////////////////////////////////////
/// Pauses the Thread. "stop" will still end it. Calling 'pause' twice or more
/// will have no effect.

void Thread::pause()
{
  m_oIsPaused = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Resumes the Thread if it is paused. Calling 'resume' twice or more will have
/// no effect.

void Thread::resume()
{
  m_oIsPaused = false;

  // Write a char to the pipe to kick the thread out of the 'select' call.
  int rv = ::write( m_pfdPipe[1], "A", 1 );
  if(rv < 0) std::cerr << __FILE__ << " " << __LINE__ << " " << strerror(errno) << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Resumes the thread if it is paused, but only for one iteration. Calling
/// 'resumeOnce' twice or more will have no effect.

void Thread::resumeOnce()
{
  // Write a char to the pipe to kick the thread out of the 'select' call.
  int rv = ::write( m_pfdPipe[1], "A", 1 );
  if(rv < 0) std::cerr << __FILE__ << " " << __LINE__ << " " << strerror(errno) << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the flag to stop the Thread. This will cause 'afterExecute' to be
/// called.

void Thread::stop()
{
  m_oStop = true;

  // Send a signal to ensure that any system calls return and enable
  // the checking of the stop flag.
  if ( m_idThis != 0 )
    ::pthread_kill( m_idThis, Thread::sm_nStopSignal );
}

////////////////////////////////////////////////////////////////////////////////
/// Puts the calling thread to sleep for uiSeconds seconds.
/// @param uiSeconds The number of seconds to go to sleep.
/// @return An error code. See 'nanosleep' for more details.

int Thread::sleep( const unsigned int &uiSeconds )
{
  return nanosleep( uiSeconds, 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Puts the calling thread to sleep for uiMillis milliseconds.
/// @param uiMillis The number of milliseconds to go to sleep.
/// @return An error code. See 'nanosleep' for more details.

int Thread::msleep( const unsigned int &uiMillis )
{
  return nanosleep( uiMillis / 1000, uiMillis % 1000 * 1000000 );
}

////////////////////////////////////////////////////////////////////////////////
/// Puts the calling thread to sleep for uiMicros microseconds.
/// @param uiMicros The number of microseconds to go to sleep.
/// @return An error code. See 'nanosleep' for more details.

int Thread::usleep( const unsigned int &uiMicros )
{
  return nanosleep( uiMicros / 1000000, uiMicros % 1000000 * 1000 );
}

////////////////////////////////////////////////////////////////////////////////
/// Puts the calling thread to sleep for uiNanos nanoseconds.
/// @param uiNanos The number of nanoseconds to go to sleep.
/// @return An error code. See 'nanosleep' for more details.

int Thread::nsleep( const unsigned int &uiNanos )
{
  return nanosleep( uiNanos / 1000000000, uiNanos % 1000000000 );
}

////////////////////////////////////////////////////////////////////////////////
/// Puts the calling thread to sleep for a specific amount of time. Under
/// windows, the best we can do is a millisecond resolution without a third
/// party library, so the other calls will be estimated. This will always
/// return ErrNone under windows, as no error information is available.
/// @param uiSeconds The number of seconds to go to sleep.
/// @param uiNanos An additional number of nanoseconds to sleep.
/// @return ErrNone The sleep occurred successfully.
/// @return ErrInterrupted The sleep was interupted by a system call (signal).
/// @return ErrInvalidParameter Invalid parameter to function call.
/// @return ErrCopy

int Thread::nanosleep( const unsigned int &uiSeconds,
                       const unsigned int &uiNanos )
{
  int nErr = ErrNone;

#ifdef WIN32
  DWORD uiMillis = uiSeconds * 1000 + uiNanos / 1000000;
  Sleep( uiMillis );
#else
  timespec ts;
  ts.tv_sec = uiSeconds;
  ts.tv_nsec = uiNanos;

  // Minus sign to convert to our error code.
  nErr = - ::nanosleep( &ts, NULL );

#endif

  return nErr;
}

////////////////////////////////////////////////////////////////////////////////
/// Suspends the calling thread until the thread managed by this object
/// finishes.
/// @return ErrNone
/// @return ErrThreadUnjoinable
/// @return ErrBadThreadId
/// @return ErrDeadlock

int Thread::join()
{
  // The pthread_join() function shall suspend execution of the calling thread
  // until the target thread terminates, unless  the  target  thread
  // has  already  terminated.  On  return from a successful pthread_join()
  // call  with  a  non-NULL  value_ptr  argument,  the  value  passed   to
  // pthread_exit()  by  the  terminating thread shall be made available in
  // the location referenced by value_ptr. When  a  pthread_join()  returns
  // successfully,  the  target  thread has been terminated. The results of
  // multiple simultaneous calls to pthread_join() specifying the same target
  // thread are undefined. If the thread calling pthread_join() is canceled,
  // then the target thread shall not be detached.
  // It is unspecified  whether  a  thread  that  has  exited  but  remains
  // unjoined counts against {PTHREAD_THREADS_MAX}.

  int nErr = ErrNone;

  if ( m_oIsRunning == false )
  {
    nErr = -EHOSTDOWN;
  }
#ifdef WIN32
  else if ( m_idThis != NULL )
  {
    //  throw away the result of the thread.
    nErr = WaitForSingleObject( m_idThis, INFINITE );

    switch ( nErr )
    {
      case WAIT_ABANDONED:
        nErr = -EINTR;
        break;
      case WAIT_OBJECT_0:
        nErr = ErrNone;
        break;
      case WAIT_TIMEOUT:
        nErr = -ETIMEDOUT;
        break;
      case WAIT_FAILED:
        nErr = -EINVAL;
        break;
      default:
        nErr = ErrUnknown;
        break;
    }
    Thread::msleep( 1 );
    // Do not set the thread id back to 0, as this will cause a mem leak!
    //m_idThis = 0;
  }
#else
  else if ( m_idThis != 0 )
  {
    // Throw away the result of the thread.
    void *pResult;
    // Minus sign to convert to our error code.
    nErr = - ::pthread_join( m_idThis, &pResult );
    // Do not set the thread id back to 0, as this will cause a mem leak!
    //m_idThis = 0;
    // Convert to our error code as necessary.
    nErr = ( nErr == -EINVAL ) ? ( -ENODEV ) : ( nErr );
  }
#endif

  return nErr;
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something before the thread has been told to
/// start, like allocate memory.

void Thread::beforeExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override this function to do something after the thread has been told to
/// stop, like clean up allocated memory.

void Thread::afterExecute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Override in derived class, place the code to do something here.

void Thread::execute()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Waits for the 'ready' mutex to be unlocked. This indicates that the
/// 'execute' function is about to be run.

void Thread::waitForReady()
{
  // We will wait here until we can get the lock.
  m_mutReady.lock();
  // unlock it immediately, we can can continue now.
  m_mutReady.unlock();
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes the Thread by calling 'beforeExecute', then loops, calling
/// 'execute' until 'm_oStop' is true. Then calls 'afterExecute' and ends.

void Thread::runLoop()
{
  // Make sure we can handle the signal which will come when we want to stop.
  // This is signal will ensure that we are not stuck waiting on I/O.
  // Any system calls in progress should return with an "EINTR" error.
  ::memset( &m_saStop, 0, sizeof( struct sigaction ) );
  m_saStop.sa_handler = Thread::processSignalHandler;
  sigemptyset( &m_saStop.sa_mask );
  ::sigaction( Thread::sm_nStopSignal, &m_saStop, 0 );

  // Ensure the loop starts.
  m_oStop = false;
  m_oIsRunning = true;

  // Call function to allow user to set things up before the
  // main loop starts.
  m_tState = BeforeExecute;
  beforeExecute();

  // Once we are about to enter the 'execute' method, we are ready to do
  // whatever this thread needs to do!
  m_mutReady.unlock();

  // The main loop.
  m_tState = Execute;
  while ( m_oStop == false )
  {
    // This is a loop that is entered pausing the polling
    // and processing. quitting will end it as well.
    if ( m_oIsPaused == true )
    {
      // Create and clear out the FD set.
      fd_set fdsRead;
      FD_ZERO( &fdsRead );
      FD_SET( m_pfdPipe[0], &fdsRead);
      int nHighestNumberedFd = m_pfdPipe[0];
      // Wait for it....
      ::select( nHighestNumberedFd+1, &fdsRead, NULL, NULL, NULL );
      // Did we get knocked out of 'select' by something on the pipe?
      if ( FD_ISSET( m_pfdPipe[0], &fdsRead ) )
      {
        // Read the single char sent.
        char ch;
        int rv = ::read( m_pfdPipe[0], &ch, 1 );
        if(rv < 0) std::cerr << __FILE__ << " " << __LINE__ << " " << strerror(errno) << "\n";
      }
    }

    // Check to see if we should stop looping, since the flag may have
    // toggled while we were paused, above.
    if ( m_oStop == true )
      break;

    // Do we have a trigger? In this case, we have a socket which
    // we will use to signal the "execute" method to run. This will
    // happen if we have data available to be read on said socket.
    if ( m_psocTrigger != NULL && m_psocTrigger->isValid() == true )
    {
      // Create and clear out the FD set.
      fd_set fdsRead;
      FD_ZERO( &fdsRead );
      // Watch the to see when we get some input.
      FD_SET( m_psocTrigger->getFd(), &fdsRead );
      int nHighestNumberedFd = m_psocTrigger->getFd();
      // Wait for it....
      ::select( nHighestNumberedFd+1, &fdsRead, NULL, NULL, NULL );
    }

    // Check to see if we should stop looping, since the flag may have
    // toggled while we were waiting for a trigger, above.
    if ( m_oStop == true )
      break;

    // Wait for a bit....
    // If we have set a trigger, above, we may not want to wait here.
    if ( m_uiInterval > 0 )
      msleep( m_uiInterval );

    // Check to see if we should stop looping, since the flag may have
    // toggled while we were sleeping, above.
    if ( m_oStop == true )
      break;

    // Call the function to actually do something.
    execute();

    // If this is a one-shot Thread, 'stop' will become true here.
    m_oStop = ( m_oOneShot || m_oStop );
  }

  // Call function to allow user to clean things up before the
  // thread exits. There must be nothing done after this.
  m_tState = AfterExecute;
  afterExecute();

  m_oIsRunning = false;

  m_tState = Idle;
}

////////////////////////////////////////////////////////////////////////////////
/// Code to get a new Thread running.
/// We need a static function for the 'pthread_create' or 'CreateThread'
/// call - see below. We can also pass in a CPU number to set the affinity
/// of this thread to a particular CPU.
/// @param iCpuAffinity The CPU to run this thread on.
/// @return ErrNone
/// @return ErrAlreadyRunning
/// @return ErrCouldNotCreateThread
/// @return ErrInvalidParameter
/// @return ErrWrongPermission

int Thread::start( const int &iCpuAffinity,
                   const ScheduleType &tSchedule )
{
  // The   pthread_create()  function  shall  create  a  new  thread,  with
  // attributes specified by attr, within a process. If attr is  NULL,  the
  // default  attributes shall be used. If the attributes specified by attr
  // are modified later, the thread’s attributes  shall  not  be  affected.
  // Upon successful completion, pthread_create() shall store the ID of the
  // created thread in the location referenced by thread.
  //
  // The thread is created executing start_routine with  arg  as  its  sole
  // argument.  If  the  start_routine  returns,  the effect shall be as if
  // there was an implicit call to pthread_exit() using the return value of
  // start_routine as the exit status. Note that the thread in which main()
  // was originally invoked differs from this. When it returns from main(),
  // the  effect  shall be as if there was an implicit call to exit() using
  // the return value of main() as the exit status.

  int nErr = ErrNone;

  if ( m_oIsRunning != false )
  {
    nErr = -EALREADY;
  }
  else
  {
#ifdef WIN32
    DWORD idDummy;
    m_idThis  = CreateThread( NULL, 0, Thread::threadFuncWin,
                              this, 0, &idDummy );
    nErr = ( m_idThis == 0 ) ? ( -EAGAIN ) : ( ErrNone );
#else
#ifndef __APPLE__

    // We need an attributes object to modify the behavior of the new thread.
    sched_param param;
    pthread_attr_t attr;
    ::pthread_attr_init( &attr );

    // Should we set the affinity of this thread to a particular CPU?
    if ( iCpuAffinity > -1 )
    {
      // Get the number of CPU's on this system.
      int iCpuCount = ::sysconf( _SC_NPROCESSORS_ONLN );

      // Make sure we have chosen a CPU that exists. If it doesn't,
      // we will fail silently and run on whatever CPU the scheduler decrees.
      if ( iCpuAffinity < iCpuCount )
      {
        // Create a CPU set containing the current cpu index.
        cpu_set_t setCpu;
        CPU_ZERO( &setCpu );
        CPU_SET( iCpuAffinity, &setCpu );
        ::pthread_attr_setaffinity_np( &attr, sizeof( cpu_set_t ), &setCpu );
      }
    }

    // Set the scheduler attributes for the thread we are about to create.
    switch ( tSchedule )
    {
      // Nothing special for this thread; just use whatever is set here.
      case Inherit:
        ::pthread_attr_setinheritsched( &attr, PTHREAD_INHERIT_SCHED );
        break;
      // Specifically ask for normal scheduling and priority.
      case Normal:
        ::pthread_attr_setinheritsched( &attr, PTHREAD_EXPLICIT_SCHED );
        param.sched_priority = 0;
        ::pthread_attr_setschedparam( &attr, &param);
        ::pthread_attr_setschedpolicy( &attr, SCHED_OTHER );
        break;
      // Ask for better scheduling and priority.
      case Turbo:
        ::pthread_attr_setinheritsched( &attr, PTHREAD_EXPLICIT_SCHED );
        param.sched_priority = ::sched_get_priority_max( SCHED_FIFO );
        ::pthread_attr_setschedparam( &attr, &param);
        ::pthread_attr_setschedpolicy( &attr, SCHED_FIFO );
    }

    // Minus sign to convert to our error code.
    nErr = - ::pthread_create( &m_idThis, &attr, Thread::pthreadFunc, this );

#endif
#endif
  }
  return nErr;
}

#ifdef WIN32
////////////////////////////////////////////////////////////////////////////////
/// 'CreateThread' needs a static function to get the
/// thread going. Passing a pointer back to the Thread object allows us to
/// call the 'runLoop' function from within the new thread.
/// @param A pointer to this class that the thread function can use.

DWORD WINAPI Thread::threadFuncWin( void *pUnknown )
{
  //  do a static cast to get the pointer back to a "Thread" object.
  Thread *pThis = static_cast<Thread *>( pUnknown );

  //  we are now within the new thread and up and running.
  pThis->runLoop();

  ///  no return is necessary, since it is not examined.
  return 0;
}

#else
////////////////////////////////////////////////////////////////////////////////
/// 'pthread_create' needs a static function to get the
/// thread going. Passing a pointer back to the Thread object allows us to
/// call the 'runLoop' function from within the new thread.
/// @param A pointer to this class that the thread function can use.

void *Thread::pthreadFunc( void *pUnknown )
{
  //  do a static cast to get the pointer back to a "Thread" object.
  Thread *pThis = static_cast<Thread *>( pUnknown );

  //  we are now within the new thread and up and running.
  try
  {
    pThis->runLoop();
  }
  catch ( const std::exception &excep )
  {
    std::cerr << "Thread exited: " << excep.what() << std::endl;
  }
  catch ( ... )
  {
    std::cerr << "An exception was thrown, exiting the thread." << std::endl;
  }

  ///  no return is necessary, since it is not examined.
  return NULL;
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// This variable and function are used to trap a signal to tell the thread
/// to return from any system calls gracefully.

void Thread::processSignalHandler( int nSignal )
{
  static_cast<void>(nSignal);
  // The signal handler is not re-installed, so it will not work again.
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a message associated with an error code.
/// @param nErr The error code.
/// @return A message associated with the error.

string Thread::getErrorMsg( const int &nErr )
{
  switch ( nErr )
  {
    case ErrNone:
      return string( "No Error." );
      break;
    case ErrThreadUnjoinable:
      return string( "Thread unjoinable." );
      break;
    case ErrBadThreadId:
      return string( "Bad thread Id." );
      break;
    case ErrDeadlock:
      return string( "Operation would deadlock." );
      break;
    case ErrCouldNotCreateThread:
      return string( "Could not create thread." );
      break;
    case ErrInvalidParameter:
      return string( "Invalid parameter to function call." );
      break;
    case ErrWrongPermission:
      return string( "Wrong permission." );
      break;
    case ErrAlreadyRunning:
      return string( "Thread already running." );
      break;
    case ErrInterrupted:
      return string( "Sleep was interrupted before it could complete." );
      break;
    case ErrCopy:
      return string( "Information could not be copied from user space." );
      break;
    case ErrTimedOut:
      return string( "Timed out waiting for thread." );
      break;
    case ErrNotRunning:
      return string( "Thread is not running." );
      break;
    case ErrUnknown:
      return string( "Unknown error." );
      break;
  }
  return string( "" );
}

////////////////////////////////////////////////////////////////////////////////
