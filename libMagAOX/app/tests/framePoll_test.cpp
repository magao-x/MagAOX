/** \file framePoll_test.cpp
 * \brief Catch2 tests for waiting on ImageStreamIO frames by polling.
 */

#include "../../../tests/testXWC.hpp"

#include "../framePoll.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace libXWCTest
{
namespace framePollTest
{

/// \cond FRAMEPOLL_TEST_HELPERS
/// A CLOCK_REALTIME deadline the given number of milliseconds from now.
timespec deadlineIn( long ms /**< [in] milliseconds from now */ )
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += ( ms % 1000 ) * 1000000L;

    if( ts.tv_nsec >= 1000000000L )
    {
        ts.tv_nsec -= 1000000000L;
        ++ts.tv_sec;
    }

    return ts;
}
/// \endcond

/// A frame is seen once its counter changes and the writer has cleared write, by spinning and by sleeping.
/**
 * \ingroup app_unit_test
 */
TEST_CASE( "pollFrame sees a frame only when it is fully published", "[framePoll]" )
{
    // clang-format off
    #ifdef FRAMEPOLL_TEST_DOXYGEN_REF
    MagAOX::app::pollFrame;
    #endif
    // clang-format on

    for( int sleepUs : { 0, 100 } )
    {
        IMAGE_METADATA md{};
        IMAGE          image{};
        image.md = &md;

        uint64_t lastCnt0 = 0;

        // being written: cnt0 has changed but write is still set
        md.write = 1;
        md.cnt0  = 1;

        std::thread writer(
            [&md]()
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
                __atomic_store_n( &md.write, 0, __ATOMIC_RELEASE );
            } );

        auto start = std::chrono::steady_clock::now();
        int  rv    = MagAOX::app::pollFrame( image,
                                         lastCnt0,
                                         deadlineIn( 2000 ),
                                         sleepUs,
                                         []()
                                         {
                                             return false;
                                         } );
        auto took  = std::chrono::steady_clock::now() - start;
        writer.join();

        REQUIRE( rv == 0 );
        REQUIRE( lastCnt0 == 1 );
        REQUIRE( took >= std::chrono::milliseconds( 15 ) );

        // nothing new: times out at the deadline
        errno = 0;
        rv    = MagAOX::app::pollFrame( image,
                                     lastCnt0,
                                     deadlineIn( 50 ),
                                     sleepUs,
                                     []()
                                     {
                                         return false;
                                     } );
        REQUIRE( rv == -1 );
        REQUIRE( errno == ETIMEDOUT );
        REQUIRE( lastCnt0 == 1 );

        // a stop request interrupts the wait
        std::atomic<bool> stop{ false };
        std::thread       stopper(
            [&stop]()
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
                stop = true;
            } );

        errno = 0;
        rv    = MagAOX::app::pollFrame( image,
                                     lastCnt0,
                                     deadlineIn( 2000 ),
                                     sleepUs,
                                     [&stop]()
                                     {
                                         return stop.load();
                                     } );
        stopper.join();

        REQUIRE( rv == -1 );
        REQUIRE( errno == EINTR );
    }
}

} // namespace framePollTest
} // namespace libXWCTest
