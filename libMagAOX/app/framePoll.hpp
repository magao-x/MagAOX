/** \file framePoll.hpp
 * \brief Wait for a new ImageStreamIO frame by polling the stream metadata instead of a semaphore
 *
 * \ingroup app_files
 */

#ifndef app_framePoll_hpp
#define app_framePoll_hpp

#include <cerrno>
#include <cstdint>
#include <ctime>

#include <ImageStreamIO/ImageStruct.h>

#if defined( __x86_64__ ) || defined( __i386__ )
    #include <immintrin.h>
#endif

namespace MagAOX
{
namespace app
{

/// Wait for a new frame by polling a stream's frame counter (`md->cnt0`) instead of waiting on its semaphore
/** The writer then has no waiter to wake for this reader.  A frame is new when `md->cnt0` differs from `lastCnt0` and
 * `md->write` is 0, i.e. the writer has finished publishing it, including the per-frame arrays.
 *
 * With `sleepUs` 0 this polls continuously with a pause instruction, which keeps the core busy, so it needs a dedicated
 * cpuset: a SCHED_FIFO thread of the same or lower priority on the same core would never run.  Otherwise it sleeps
 * `sleepUs` between polls, adding up to that much latency.
 *
 * \returns 0 when a new frame is available, and `lastCnt0` is updated
 * \returns -1 with errno set to ETIMEDOUT at the deadline, or to EINTR if `stop()` returns true
 *
 * \tparam stopT a callable returning true when waiting should stop, e.g. on shutdown
 *
 * \ingroup app
 */
template <typename stopT>
int pollFrame( IMAGE          &image,    /**< [in] the attached stream */
               uint64_t       &lastCnt0, /**< [in/out] the frame counter last seen */
               const timespec &deadline, /**< [in] the CLOCK_REALTIME time at which to stop waiting */
               int             sleepUs,  /**< [in] microseconds to sleep between polls, 0 to poll continuously */
               stopT         &&stop      /**< [in] checked periodically, waiting stops when it returns true */
);

template <typename stopT>
int pollFrame( IMAGE &image, uint64_t &lastCnt0, const timespec &deadline, int sleepUs, stopT &&stop )
{
    timespec sleep{ sleepUs / 1000000, ( sleepUs % 1000000 ) * 1000L };

    for( uint64_t n = 1;; ++n )
    {
        // The writer sets write before cnt0, and clears it after the frame and its per-frame entries
        uint64_t cnt0 = __atomic_load_n( &image.md->cnt0, __ATOMIC_ACQUIRE );

        if( cnt0 != lastCnt0 && __atomic_load_n( &image.md->write, __ATOMIC_ACQUIRE ) == 0 )
        {
            lastCnt0 = cnt0;
            return 0;
        }

        if( sleepUs > 0 || n % 1024 == 0 )
        {
            if( stop() )
            {
                errno = EINTR;
                return -1;
            }

            timespec now;
            clock_gettime( CLOCK_REALTIME, &now );

            if( now.tv_sec > deadline.tv_sec || ( now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec ) )
            {
                errno = ETIMEDOUT;
                return -1;
            }
        }

        if( sleepUs > 0 )
        {
            clock_nanosleep( CLOCK_MONOTONIC, 0, &sleep, nullptr );
        }
        else
        {
#if defined( __x86_64__ ) || defined( __i386__ )
            _mm_pause();
#elif defined( __aarch64__ )
            asm volatile( "yield" );
#endif
        }
    }
}

} // namespace app
} // namespace MagAOX

#endif // app_framePoll_hpp
