/** \file semUtils_test.cpp
  * \brief Catch2 tests for the XWC_SEM_ semaphore macros in semUtils.hpp and semUtilsDerived.hpp.
  *
  * The macros are function-like and expect a log<>() template in the enclosing class. Each macro
  * is wrapped in a small member function of a test struct that provides a stub log<>(). The
  * DERIVED variants are wrapped in a CRTP mixin so that derivedT::log<>() resolves as it would
  * in a real device mixin. The tests use real unnamed POSIX semaphores and the real system clock.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include <semaphore.h>

#include <mx/sys/timeUtils.hpp>

#include "../../logger/logManager.hpp"

using namespace MagAOX::logger;

#include "../semUtils.hpp"
#include "../dev/semUtilsDerived.hpp"

namespace semUtils_test
{

/// Stub logger that satisfies the log<>() call the macros expand to.
/// It discards the message and returns the retval template argument.
struct testLogger
{
    template <typename logT, int retval = 0>
    static int log( const typename logT::messageT & )
    {
        return retval;
    }
};

/// Wraps each plain XWC_SEM_ macro in a member function so it can be called from a test.
struct semUtilsTest : public testLogger
{
    void waitTsRetVoid( timespec &ts, int sec, int nsec )
    {
        XWC_SEM_WAIT_TS_RETVOID( ts, sec, nsec );
    }

    int waitTs( timespec &ts, int sec, int nsec )
    {
        XWC_SEM_WAIT_TS( ts, sec, nsec );
        return 0;
    }

    /// Run the timed wait macro inside a bounded loop.
    /// Returns the iteration count on which the wait succeeded, or -1 if the loop exited
    /// without a successful wait. The macro uses continue on a timeout and break on any
    /// other error, so both paths end at the -1 return once maxIters is reached.
    int timedwaitLoop( sem_t &sem, timespec &ts, int maxIters )
    {
        int iters = 0;
        while( iters < maxIters )
        {
            ++iters;
            XWC_SEM_TIMEDWAIT_LOOP( sem, ts )
            return iters;
        }
        return -1;
    }

    int flush( sem_t &sem )
    {
        XWC_SEM_FLUSH( sem );
        return 0;
    }
};

/// CRTP mixin that wraps each XWC_SEM_ DERIVED macro in a member function.
/// The macros call derivedT::template log<>(), so derivedT must supply that template.
template <class derivedT>
struct semUtilsDerivedMixin
{
    void waitTsRetVoidDerived( timespec &ts, int sec, int nsec )
    {
        XWC_SEM_WAIT_TS_RETVOID_DERIVED( ts, sec, nsec );
    }

    int waitTsDerived( timespec &ts, int sec, int nsec )
    {
        XWC_SEM_WAIT_TS_DERIVED( ts, sec, nsec );
        return 0;
    }

    /// Same contract as semUtilsTest::timedwaitLoop() but using the DERIVED macro.
    int timedwaitLoopDerived( sem_t &sem, timespec &ts, int maxIters )
    {
        int iters = 0;
        while( iters < maxIters )
        {
            ++iters;
            XWC_SEM_TIMEDWAIT_LOOP_DERIVED( sem, ts )
            return iters;
        }
        return -1;
    }

    int flushDerived( sem_t &sem )
    {
        XWC_SEM_FLUSH_DERIVED( sem );
        return 0;
    }
};

/// Concrete derived type for the mixin. It gets its log<>() from testLogger.
struct semUtilsDerivedTest : public testLogger, public semUtilsDerivedMixin<semUtilsDerivedTest>
{
};

} // namespace semUtils_test

using namespace semUtils_test;

/// Verify that XWC_SEM_WAIT_TS_RETVOID and XWC_SEM_WAIT_TS fill a timespec with the current
/// time plus the requested offset, and that the nanosecond field stays normalized.
SCENARIO( "Adding wait time to a timespec", "[semUtils]" )
{
    GIVEN( "a semUtilsTest object" )
    {
        semUtilsTest t;

        WHEN( "using XWC_SEM_WAIT_TS_RETVOID with a working clock" )
        {
            timespec before;
            clock_gettime( CLOCK_REALTIME, &before );

            timespec ts;
            t.waitTsRetVoid( ts, 5, 500 );

            REQUIRE( ts.tv_sec >= before.tv_sec + 5 );
            REQUIRE( ts.tv_nsec >= 0 );
            REQUIRE( ts.tv_nsec < 1000000000 ); // one second in ns
        }

        WHEN( "using XWC_SEM_WAIT_TS with a working clock" )
        {
            timespec before;
            clock_gettime( CLOCK_REALTIME, &before );

            timespec ts;
            int rv = t.waitTs( ts, 3, 250 );

            REQUIRE( rv == 0 );
            REQUIRE( ts.tv_sec >= before.tv_sec + 3 );
            REQUIRE( ts.tv_nsec >= 0 );
            REQUIRE( ts.tv_nsec < 1000000000 ); // one second in ns
        }
    }
}

/// Verify XWC_SEM_TIMEDWAIT_LOOP on a real semaphore. A posted semaphore must return on the
/// first iteration. A timeout must continue the loop and leave errno as ETIMEDOUT. An invalid
/// timespec must break the loop and leave errno as EINVAL.
SCENARIO( "Waiting on a semaphore in a standard loop", "[semUtils]" )
{
    GIVEN( "a semUtilsTest object and a semaphore" )
    {
        semUtilsTest t;
        sem_t sem;
        sem_init( &sem, 0, 0 );

        WHEN( "the semaphore is already posted" )
        {
            sem_post( &sem );

            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );
            ts.tv_sec += 1;

            int rv = t.timedwaitLoop( sem, ts, 1 );
            REQUIRE( rv == 1 );
        }

        WHEN( "the wait times out" )
        {
            // The deadline is now, so the unposted semaphore times out at once.
            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );

            errno = 0;
            int rv = t.timedwaitLoop( sem, ts, 1 );
            REQUIRE( rv == -1 );
            REQUIRE( errno == ETIMEDOUT );
        }

        WHEN( "sem_timedwait fails for a reason other than a timeout" )
        {
            // A nanosecond field of two seconds is out of range, so sem_timedwait fails with EINVAL.
            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );
            ts.tv_nsec = 2000000000; // two seconds in ns, not a valid tv_nsec

            errno = 0;
            int rv = t.timedwaitLoop( sem, ts, 1 );
            REQUIRE( rv == -1 );
            REQUIRE( errno == EINVAL );
        }

        sem_destroy( &sem );
    }
}

/// Verify that XWC_SEM_FLUSH drains every pending post from a semaphore and leaves its value
/// at zero, and that it is harmless on a semaphore with no pending posts.
SCENARIO( "Flushing a semaphore", "[semUtils]" )
{
    GIVEN( "a semUtilsTest object and a semaphore" )
    {
        semUtilsTest t;
        sem_t sem;
        sem_init( &sem, 0, 0 );

        WHEN( "the semaphore has several pending posts" )
        {
            sem_post( &sem );
            sem_post( &sem );
            sem_post( &sem );

            int rv = t.flush( sem );
            REQUIRE( rv == 0 );

            int val = -1;
            sem_getvalue( &sem, &val );
            REQUIRE( val == 0 );
        }

        WHEN( "the semaphore has no pending posts" )
        {
            int rv = t.flush( sem );
            REQUIRE( rv == 0 );

            int val = -1;
            sem_getvalue( &sem, &val );
            REQUIRE( val == 0 );
        }

        sem_destroy( &sem );
    }
}

/// Verify the DERIVED variants of the wait time macros through the CRTP mixin.
/// The checks are the same as for the plain macros.
SCENARIO( "Adding wait time to a timespec, derived", "[semUtilsDerived]" )
{
    GIVEN( "a semUtilsDerivedTest object" )
    {
        semUtilsDerivedTest t;

        WHEN( "using XWC_SEM_WAIT_TS_RETVOID_DERIVED with a working clock" )
        {
            timespec before;
            clock_gettime( CLOCK_REALTIME, &before );

            timespec ts;
            t.waitTsRetVoidDerived( ts, 5, 500 );

            REQUIRE( ts.tv_sec >= before.tv_sec + 5 );
            REQUIRE( ts.tv_nsec >= 0 );
            REQUIRE( ts.tv_nsec < 1000000000 ); // one second in ns
        }

        WHEN( "using XWC_SEM_WAIT_TS_DERIVED with a working clock" )
        {
            timespec before;
            clock_gettime( CLOCK_REALTIME, &before );

            timespec ts;
            int rv = t.waitTsDerived( ts, 3, 250 );

            REQUIRE( rv == 0 );
            REQUIRE( ts.tv_sec >= before.tv_sec + 3 );
            REQUIRE( ts.tv_nsec >= 0 );
            REQUIRE( ts.tv_nsec < 1000000000 ); // one second in ns
        }
    }
}

/// Verify XWC_SEM_TIMEDWAIT_LOOP_DERIVED through the CRTP mixin.
/// The posted, timed out, and invalid timespec cases mirror the plain macro scenario.
SCENARIO( "Waiting on a semaphore in a standard loop, derived", "[semUtilsDerived]" )
{
    GIVEN( "a semUtilsDerivedTest object and a semaphore" )
    {
        semUtilsDerivedTest t;
        sem_t sem;
        sem_init( &sem, 0, 0 );

        WHEN( "the semaphore is already posted" )
        {
            sem_post( &sem );

            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );
            ts.tv_sec += 1;

            int rv = t.timedwaitLoopDerived( sem, ts, 1 );
            REQUIRE( rv == 1 );
        }

        WHEN( "the wait times out" )
        {
            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );

            errno = 0;
            int rv = t.timedwaitLoopDerived( sem, ts, 1 );
            REQUIRE( rv == -1 );
            REQUIRE( errno == ETIMEDOUT );
        }

        WHEN( "sem_timedwait fails for a reason other than a timeout" )
        {
            timespec ts;
            clock_gettime( CLOCK_REALTIME, &ts );
            ts.tv_nsec = 2000000000; // two seconds in ns, not a valid tv_nsec

            errno = 0;
            int rv = t.timedwaitLoopDerived( sem, ts, 1 );
            REQUIRE( rv == -1 );
            REQUIRE( errno == EINVAL );
        }

        sem_destroy( &sem );
    }
}

/// Verify XWC_SEM_FLUSH_DERIVED through the CRTP mixin. The checks are the same as for the
/// plain flush macro.
SCENARIO( "Flushing a semaphore, derived", "[semUtilsDerived]" )
{
    GIVEN( "a semUtilsDerivedTest object and a semaphore" )
    {
        semUtilsDerivedTest t;
        sem_t sem;
        sem_init( &sem, 0, 0 );

        WHEN( "the semaphore has several pending posts" )
        {
            sem_post( &sem );
            sem_post( &sem );
            sem_post( &sem );

            int rv = t.flushDerived( sem );
            REQUIRE( rv == 0 );

            int val = -1;
            sem_getvalue( &sem, &val );
            REQUIRE( val == 0 );
        }

        WHEN( "the semaphore has no pending posts" )
        {
            int rv = t.flushDerived( sem );
            REQUIRE( rv == 0 );

            int val = -1;
            sem_getvalue( &sem, &val );
            REQUIRE( val == 0 );
        }

        sem_destroy( &sem );
    }
}
